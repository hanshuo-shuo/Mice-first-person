"""Mouse-centred visual observations for the 2-D Cellworld environments.

The simulation itself is deliberately left in two dimensions.  This module
extrudes the exact arena/occlusion polygons into a small 2.5-D scene and casts
one ray per image column from the prey pose.  Consequently, what the camera
occludes is driven by the same geometry used by physics and line-of-sight.

``FirstPersonVisionWrapper`` follows the Gymnasium wrapper contract.  In its
recommended mode it returns two ``uint8`` HWC eye images, normalized
proprioception, and the previous action.  It also converts body-frame velocity
commands into the base environment's point-mass acceleration without taking
over physics, reward, termination, or ``info``.
"""

from __future__ import annotations

import copy
import math
import typing

import gymnasium as gym
import numpy as np
from gymnasium import spaces


RGBFrame = np.ndarray


# Keep the billboard dimensions in one place so rendering and the camera
# visibility label use the same projected predator silhouette.
PREDATOR_OBJECT_WIDTH = 0.070
PREDATOR_OBJECT_HEIGHT = 0.12
PREDATOR_DEPTH_TOLERANCE = 0.026
_PROJECT_DEFAULT_FORWARD_CLIP = object()


def _state_body_heading(state) -> float:
    """Read the canonical body heading with legacy-state compatibility."""

    if hasattr(state, "body_heading"):
        return float(state.body_heading)
    return float(getattr(state, "direction", 0.0))


def _set_state_body_heading(state, heading: float) -> None:
    """Write body heading on canonical and legacy-compatible state objects."""

    if hasattr(state, "body_heading"):
        state.body_heading = float(heading)
    elif hasattr(state, "direction"):
        state.direction = float(heading)
    else:
        raise AttributeError("Agent state must expose body_heading")


def _polygon_vertices(polygon: typing.Any) -> np.ndarray:
    """Return vertices from either Cellworld's Torch or Shapely polygon."""

    vertices = getattr(polygon, "vertices", None)
    if vertices is None and hasattr(polygon, "exterior"):
        vertices = polygon.exterior.coords
    if callable(vertices):
        vertices = vertices()
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError(f"Expected polygon vertices shaped (N, 2), got {array.shape}")
    # Shapely rings repeat their first vertex at the end.
    if len(array) > 2 and np.allclose(array[0], array[-1]):
        array = array[:-1]
    return array


def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


class FirstPersonRenderer:
    """CPU-only first-person renderer tied to a Cellworld model.

    Rendering ``rgb_array`` never initializes a window, so it works on
    headless training machines.  A pygame window is created lazily only when
    :meth:`show` is called.
    """

    def __init__(
        self,
        model: typing.Any,
        width: int = 256,
        height: int = 256,
        horizontal_fov: float = 120.0,
        camera_height: float = 0.025,
        occlusion_height: float = 0.16,
        arena_wall_height: float = 0.22,
        far_clip: float = 2.0,
        render_fps: int = 30,
    ) -> None:
        if width < 16 or height < 16:
            raise ValueError("First-person image width and height must both be >= 16")
        if not 10.0 <= horizontal_fov < 180.0:
            raise ValueError("horizontal_fov must be in [10, 180) degrees")
        if camera_height <= 0 or occlusion_height <= camera_height:
            raise ValueError("occlusion_height must be greater than camera_height")
        if arena_wall_height <= camera_height:
            raise ValueError("arena_wall_height must be greater than camera_height")
        if far_clip <= 0:
            raise ValueError("far_clip must be positive")

        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.horizontal_fov = float(horizontal_fov)
        self.camera_height = float(camera_height)
        self.occlusion_height = float(occlusion_height)
        self.arena_wall_height = float(arena_wall_height)
        self.far_clip = float(far_clip)
        self.render_fps = int(render_fps)

        self._focal_length = (self.width / 2.0) / math.tan(
            math.radians(self.horizontal_fov) / 2.0,
        )
        self._horizon = int(round(self.height * 0.46))
        self._ray_offsets = np.linspace(
            self.horizontal_fov / 2.0,
            -self.horizontal_fov / 2.0,
            self.width,
            dtype=np.float64,
        )
        self._ray_offset_cos = np.cos(np.radians(self._ray_offsets))

        self._segment_starts: np.ndarray
        self._segment_vectors: np.ndarray
        self._segment_heights: np.ndarray
        self._segment_colors: np.ndarray
        self._segment_light: np.ndarray
        self._build_wall_segments()

        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        nx = (xx - (self.width - 1) / 2.0) / max(self.width / 2.0, 1.0)
        ny = (yy - (self.height - 1) / 2.0) / max(self.height / 2.0, 1.0)
        radius = np.sqrt(nx * nx + ny * ny)
        self._vignette = np.clip(1.04 - 0.16 * radius * radius, 0.78, 1.0)[..., None]

        self._window = None
        self._clock = None
        self._last_frame: typing.Optional[RGBFrame] = None
        self._last_depth: typing.Optional[np.ndarray] = None

    @property
    def image_shape(self) -> typing.Tuple[int, int, int]:
        return self.height, self.width, 3

    def _build_wall_segments(self) -> None:
        starts: typing.List[np.ndarray] = []
        vectors: typing.List[np.ndarray] = []
        heights: typing.List[float] = []
        colors: typing.List[typing.Tuple[int, int, int]] = []

        def append_polygon(
            polygon: typing.Any,
            wall_height: float,
            color: typing.Tuple[int, int, int],
        ) -> None:
            vertices = _polygon_vertices(polygon)
            ends = np.roll(vertices, -1, axis=0)
            for start, end in zip(vertices, ends):
                starts.append(start)
                vectors.append(end - start)
                heights.append(wall_height)
                colors.append(color)

        # Arena boundary and occlusions intentionally use different materials,
        # making the navigable boundary distinguishable without adding a HUD.
        append_polygon(self.model.arena, self.arena_wall_height, (147, 145, 137))
        for index, occlusion in enumerate(self.model.occlusions):
            variation = (index * 17) % 14 - 7
            base = np.array((82, 86, 91), dtype=np.int16) + variation
            append_polygon(
                occlusion,
                self.occlusion_height,
                tuple(np.clip(base, 0, 255).astype(int)),
            )

        self._segment_starts = np.asarray(starts, dtype=np.float64)
        self._segment_vectors = np.asarray(vectors, dtype=np.float64)
        self._segment_heights = np.asarray(heights, dtype=np.float64)
        self._segment_colors = np.asarray(colors, dtype=np.float64)

        lengths = np.linalg.norm(self._segment_vectors, axis=1)
        normals = np.stack(
            (-self._segment_vectors[:, 1], self._segment_vectors[:, 0]),
            axis=1,
        ) / np.maximum(lengths[:, None], 1e-9)
        light_direction = np.asarray((-0.45, 0.89), dtype=np.float64)
        self._segment_light = 0.72 + 0.28 * np.abs(normals @ light_direction)

    def _camera_pose(self) -> typing.Tuple[np.ndarray, float]:
        prey = getattr(self.model, "prey", None)
        if prey is None:
            raise AttributeError("The model must expose a prey agent for first-person rendering")
        origin = np.asarray(prey.state.location, dtype=np.float64)
        if origin.shape != (2,):
            raise ValueError(f"Expected prey location shaped (2,), got {origin.shape}")
        return origin, _state_body_heading(prey.state)

    def _cast_walls(
        self,
        origin: np.ndarray,
        direction_degrees: float,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = np.radians(direction_degrees + self._ray_offsets)
        rays = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        start_delta = self._segment_starts - origin

        denominator = _cross_2d(rays[:, None, :], self._segment_vectors[None, :, :])
        safe_denominator = np.where(np.abs(denominator) > 1e-10, denominator, 1.0)
        ray_distance = _cross_2d(
            start_delta[None, :, :],
            self._segment_vectors[None, :, :],
        ) / safe_denominator
        segment_fraction = _cross_2d(
            start_delta[None, :, :],
            rays[:, None, :],
        ) / safe_denominator

        valid = (
            (np.abs(denominator) > 1e-10)
            & (ray_distance > 1e-6)
            & (segment_fraction >= -1e-8)
            & (segment_fraction <= 1.0 + 1e-8)
        )
        candidates = np.where(valid, ray_distance, np.inf)
        segment_ids = np.argmin(candidates, axis=1)
        raw_depth = candidates[np.arange(self.width), segment_ids]
        missing = ~np.isfinite(raw_depth)
        raw_depth[missing] = self.far_clip
        segment_ids[missing] = 0

        # Perpendicular depth removes the classic fish-eye distortion while
        # preserving exact ray/segment occlusion tests.
        perpendicular_depth = np.maximum(raw_depth * self._ray_offset_cos, 1e-5)
        hit_points = origin + rays * raw_depth[:, None]
        return perpendicular_depth, segment_ids, hit_points

    def _background(self) -> np.ndarray:
        frame = np.empty(self.image_shape, dtype=np.float64)

        sky_top = np.asarray((75, 121, 159), dtype=np.float64)
        sky_horizon = np.asarray((188, 207, 215), dtype=np.float64)
        sky_rows = max(self._horizon, 1)
        sky_mix = np.linspace(0.0, 1.0, sky_rows, dtype=np.float64)[:, None]
        frame[:sky_rows] = (sky_top * (1.0 - sky_mix) + sky_horizon * sky_mix)[:, None, :]

        floor_near = np.asarray((40, 49, 39), dtype=np.float64)
        floor_far = np.asarray((112, 119, 103), dtype=np.float64)
        floor_rows = self.height - sky_rows
        floor_mix = np.linspace(0.0, 1.0, max(floor_rows, 1), dtype=np.float64)[:, None]
        floor = floor_far * (1.0 - floor_mix) + floor_near * floor_mix
        frame[sky_rows:] = floor[:floor_rows, None, :]

        return frame

    def _draw_walls(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        segment_ids: np.ndarray,
        hit_points: np.ndarray,
    ) -> None:
        fog_color = np.asarray((155, 169, 170), dtype=np.float64)
        for x in range(self.width):
            wall_depth = float(depth[x])
            segment_id = int(segment_ids[x])
            wall_height = self._segment_heights[segment_id]
            top = int(round(self._horizon - (wall_height - self.camera_height) * self._focal_length / wall_depth))
            bottom = int(round(self._horizon + self.camera_height * self._focal_length / wall_depth))
            if bottom < 0 or top >= self.height:
                continue
            top = max(0, top)
            bottom = min(self.height - 1, bottom)
            if bottom < top:
                continue

            base = self._segment_colors[segment_id] * self._segment_light[segment_id]
            fog = np.clip(wall_depth / self.far_clip, 0.0, 1.0) * 0.48
            base = base * (1.0 - fog) + fog_color * fog

            count = bottom - top + 1
            vertical_light = np.linspace(1.08, 0.76, count, dtype=np.float64)[:, None]
            column = base[None, :] * vertical_light

            # Stable low-frequency mottling gives depth/texture without the
            # artificial range rings and brick seams used by the old demo.
            texture = 0.96 + 0.04 * math.sin(
                hit_points[x, 0] * 71.0 + hit_points[x, 1] * 47.0,
            )
            column *= texture
            frame[top : bottom + 1, x] = column

    def _project_object(
        self,
        location: typing.Sequence[float],
        object_width: float,
        object_height: float,
        direction_degrees: float,
        origin: np.ndarray,
        max_forward_distance: typing.Any = _PROJECT_DEFAULT_FORWARD_CLIP,
    ) -> typing.Optional[typing.Tuple[float, int, int, int, int]]:
        if max_forward_distance is _PROJECT_DEFAULT_FORWARD_CLIP:
            max_forward_distance = self.far_clip
        angle = math.radians(direction_degrees)
        forward_axis = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
        right_axis = np.asarray((math.sin(angle), -math.cos(angle)), dtype=np.float64)
        delta = np.asarray(location, dtype=np.float64) - origin
        forward = float(delta @ forward_axis)
        if forward <= 0.006:
            return None
        if max_forward_distance is not None and forward > max_forward_distance:
            return None
        lateral = float(delta @ right_axis)
        center_x = self.width / 2.0 + self._focal_length * lateral / forward
        half_width = max(1.0, self._focal_length * object_width / (2.0 * forward))
        left = int(math.floor(center_x - half_width))
        right = int(math.ceil(center_x + half_width))
        bottom = int(round(self._horizon + self.camera_height * self._focal_length / forward))
        top = int(round(self._horizon - (object_height - self.camera_height) * self._focal_length / forward))
        if right < 0 or left >= self.width or bottom < 0 or top >= self.height:
            return None
        return forward, left, right, top, bottom

    @staticmethod
    def _goal_patch(height: int, width: int, active: bool) -> typing.Tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[0:height, 0:width]
        xn = (xx + 0.5) / max(width, 1) * 2.0 - 1.0
        yn = (yy + 0.5) / max(height, 1)
        orb = xn * xn + ((yn - 0.20) / 0.22) ** 2 <= 0.82
        stem = (np.abs(xn) <= 0.16) & (yn >= 0.30) & (yn <= 0.86)
        base = (xn * xn + ((yn - 0.88) / 0.10) ** 2 <= 0.92) & (yn >= 0.79)
        mask = orb | stem | base

        color = np.zeros((height, width, 3), dtype=np.float64)
        primary = np.asarray((45, 242, 92) if active else (194, 67, 62), dtype=np.float64)
        color[mask] = primary
        highlight = orb & (xn < -0.16) & (yn < 0.18)
        color[highlight] = np.minimum(primary + 65, 255)
        alpha = np.zeros((height, width), dtype=np.float64)
        alpha[mask] = 0.94 if active else 0.72
        return color, alpha

    @staticmethod
    def _predator_patch(height: int, width: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[0:height, 0:width]
        xn = (xx + 0.5) / max(width, 1) * 2.0 - 1.0
        yn = (yy + 0.5) / max(height, 1)

        head = xn * xn + ((yn - 0.24) / 0.24) ** 2 <= 0.72
        body = (np.abs(xn) <= 0.64) & (yn >= 0.29) & (yn <= 0.88)
        shoulder = (np.abs(xn) <= 0.82) & (yn >= 0.42) & (yn <= 0.70)
        wheel = xn * xn + ((yn - 0.88) / 0.12) ** 2 <= 0.84
        mask = head | body | shoulder | wheel

        color = np.zeros((height, width, 3), dtype=np.float64)
        color[mask] = (54, 57, 63)
        color[shoulder] = (73, 76, 82)
        color[wheel] = (30, 32, 35)
        warning = (np.abs(xn) <= 0.17) & (yn >= 0.34) & (yn <= 0.66)
        color[warning] = (235, 186, 27)
        left_eye = ((xn + 0.30) / 0.13) ** 2 + ((yn - 0.20) / 0.08) ** 2 <= 1.0
        right_eye = ((xn - 0.30) / 0.13) ** 2 + ((yn - 0.20) / 0.08) ** 2 <= 1.0
        color[left_eye | right_eye] = (239, 53, 43)
        mask |= left_eye | right_eye

        alpha = np.zeros((height, width), dtype=np.float64)
        alpha[mask] = 1.0
        return color, alpha

    @staticmethod
    def _blend_patch(
        frame: np.ndarray,
        depth_buffer: np.ndarray,
        projection: typing.Tuple[float, int, int, int, int],
        patch_builder: typing.Callable[[int, int], typing.Tuple[np.ndarray, np.ndarray]],
        depth_tolerance: float,
    ) -> None:
        object_depth, left, right, top, bottom = projection
        unclipped_width = max(right - left + 1, 1)
        unclipped_height = max(bottom - top + 1, 1)
        patch, alpha = patch_builder(unclipped_height, unclipped_width)

        clipped_left = max(left, 0)
        clipped_right = min(right, frame.shape[1] - 1)
        clipped_top = max(top, 0)
        clipped_bottom = min(bottom, frame.shape[0] - 1)
        if clipped_right < clipped_left or clipped_bottom < clipped_top:
            return

        source_x0 = clipped_left - left
        source_x1 = source_x0 + clipped_right - clipped_left + 1
        source_y0 = clipped_top - top
        source_y1 = source_y0 + clipped_bottom - clipped_top + 1
        patch = patch[source_y0:source_y1, source_x0:source_x1]
        alpha = alpha[source_y0:source_y1, source_x0:source_x1]

        visible_columns = object_depth - depth_tolerance <= depth_buffer[
            clipped_left : clipped_right + 1
        ]
        alpha *= visible_columns[None, :]
        target = frame[clipped_top : clipped_bottom + 1, clipped_left : clipped_right + 1]
        target[:] = target * (1.0 - alpha[..., None]) + patch * alpha[..., None]

    def _object_has_visible_pixels(
        self,
        depth_buffer: np.ndarray,
        projection: typing.Tuple[float, int, int, int, int],
        patch_builder: typing.Callable[[int, int], typing.Tuple[np.ndarray, np.ndarray]],
        depth_tolerance: float,
    ) -> bool:
        """Return whether a projected object contributes any camera pixels.

        This deliberately follows the same billboard mask and wall-depth test
        as :meth:`_blend_patch`.  It is therefore a renderer/camera label, not
        the simulator's 360-degree line-of-sight result.
        """

        object_depth, left, right, top, bottom = projection
        unclipped_width = max(right - left + 1, 1)
        unclipped_height = max(bottom - top + 1, 1)
        _, alpha = patch_builder(unclipped_height, unclipped_width)

        clipped_left = max(left, 0)
        clipped_right = min(right, self.width - 1)
        clipped_top = max(top, 0)
        clipped_bottom = min(bottom, self.height - 1)
        if clipped_right < clipped_left or clipped_bottom < clipped_top:
            return False

        source_x0 = clipped_left - left
        source_x1 = source_x0 + clipped_right - clipped_left + 1
        source_y0 = clipped_top - top
        source_y1 = source_y0 + clipped_bottom - clipped_top + 1
        alpha = alpha[source_y0:source_y1, source_x0:source_x1]
        visible_columns = object_depth - depth_tolerance <= depth_buffer[
            clipped_left : clipped_right + 1
        ]
        return bool(np.any((alpha > 0.0) & visible_columns[None, :]))

    def object_visibility(
        self,
        location: typing.Sequence[float],
        origin: typing.Sequence[float],
        direction_degrees: float,
        object_width: float,
        object_height: float,
        depth_tolerance: float,
    ) -> typing.Dict[str, bool]:
        """Classify one billboard in this eye's frustum and rendered pixels.

        ``in_frustum`` ignores ``far_clip`` so it answers the angular question
        independently. ``pixels_visible`` uses the actual renderer clip and
        wall depth buffer, which is the signal suitable for visual labels.
        """

        camera_origin = np.asarray(origin, dtype=np.float64)
        if camera_origin.shape != (2,):
            raise ValueError(f"Expected camera origin shaped (2,), got {camera_origin.shape}")

        frustum_projection = self._project_object(
            location,
            object_width=object_width,
            object_height=object_height,
            direction_degrees=direction_degrees,
            origin=camera_origin,
            max_forward_distance=None,
        )
        render_projection = self._project_object(
            location,
            object_width=object_width,
            object_height=object_height,
            direction_degrees=direction_degrees,
            origin=camera_origin,
            max_forward_distance=self.far_clip,
        )
        pixels_visible = False
        if render_projection is not None:
            depth_buffer, _, _ = self._cast_walls(
                camera_origin,
                float(direction_degrees),
            )
            pixels_visible = self._object_has_visible_pixels(
                depth_buffer,
                render_projection,
                self._predator_patch,
                PREDATOR_DEPTH_TOLERANCE,
            )
        return {
            "in_frustum": frustum_projection is not None,
            "pixels_visible": pixels_visible,
        }

    def _draw_objects(
        self,
        frame: np.ndarray,
        depth_buffer: np.ndarray,
        origin: np.ndarray,
        direction_degrees: float,
    ) -> None:
        objects: typing.List[
            typing.Tuple[
                float,
                typing.Tuple[float, int, int, int, int],
                typing.Callable[[int, int], typing.Tuple[np.ndarray, np.ndarray]],
                float,
            ]
        ] = []

        current_goal = getattr(self.model, "goal_location", None)
        goal_locations = list(getattr(self.model, "goal_locations", []) or [])
        if current_goal is not None and not any(np.allclose(current_goal, goal) for goal in goal_locations):
            goal_locations.append(current_goal)
        for goal in goal_locations:
            active = current_goal is not None and np.allclose(goal, current_goal)
            projection = self._project_object(
                goal,
                object_width=0.052 if active else 0.035,
                object_height=0.13 if active else 0.08,
                direction_degrees=direction_degrees,
                origin=origin,
                max_forward_distance=self.far_clip,
            )
            if projection is not None:
                builder = lambda h, w, active=active: self._goal_patch(h, w, active)
                objects.append((projection[0], projection, builder, 0.030))

        if getattr(self.model, "use_predator", False) and hasattr(self.model, "predator"):
            projection = self._project_object(
                self.model.predator.state.location,
                object_width=PREDATOR_OBJECT_WIDTH,
                object_height=PREDATOR_OBJECT_HEIGHT,
                direction_degrees=direction_degrees,
                origin=origin,
                max_forward_distance=self.far_clip,
            )
            if projection is not None:
                objects.append((projection[0], projection, self._predator_patch, PREDATOR_DEPTH_TOLERANCE))

        # Painter's algorithm handles object-object overlap; the wall depth
        # buffer independently clips every billboard column.
        for _, projection, builder, tolerance in sorted(objects, reverse=True, key=lambda item: item[0]):
            self._blend_patch(frame, depth_buffer, projection, builder, tolerance)

    def render_rgb(
        self,
        origin: typing.Optional[typing.Sequence[float]] = None,
        direction_degrees: typing.Optional[float] = None,
    ) -> RGBFrame:
        """Render and return one ``uint8`` image shaped ``(H, W, 3)``."""

        default_origin, default_direction = self._camera_pose()
        if origin is None:
            camera_origin = default_origin
        else:
            camera_origin = np.asarray(origin, dtype=np.float64)
            if camera_origin.shape != (2,):
                raise ValueError(
                    f"Expected camera origin shaped (2,), got {camera_origin.shape}",
                )
        if direction_degrees is None:
            camera_direction = default_direction
        else:
            camera_direction = float(direction_degrees)
        depth, segment_ids, hit_points = self._cast_walls(camera_origin, camera_direction)
        frame = self._background()
        self._draw_walls(frame, depth, segment_ids, hit_points)
        self._draw_objects(frame, depth, camera_origin, camera_direction)
        frame *= self._vignette
        result = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
        self._last_frame = result
        self._last_depth = np.ascontiguousarray(depth.astype(np.float32))
        return result

    def show(self, frame: typing.Optional[RGBFrame] = None) -> None:
        """Display a frame in a lazily-created pygame window."""

        import pygame

        if frame is None:
            frame = self.render_rgb()
        window_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self._window is None or self._window.get_size() != window_size:
            pygame.init()
            self._window = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Cellworld - prey first-person camera")
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT and getattr(self.model, "running", False):
                self.model.stop()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(surface, (0, 0))
        pygame.display.flip()
        if self._clock is not None and self.render_fps > 0:
            self._clock.tick(self.render_fps)

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None
            self._clock = None


class FirstPersonVisionWrapper(gym.Wrapper):
    """Expose mouse-centred vision and egocentric controls.

    ``observation_mode="mouse"`` returns separate left/right eye images plus
    compact proprioception and the previous policy action.  This is the
    recommended contract for a VLA.  ``observation_mode="single_rgb"`` and
    ``action_mode="passthrough"`` preserve the original image-only/world-frame
    interface for ablations and old controllers.

    Egocentric velocity actions are translated into the wrapped environment's
    existing world-frame point-mass acceleration.  Collision, reward,
    termination, and all task state therefore remain owned by the base env.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    OBSERVATION_MODES = ("mouse", "single_rgb")
    ACTION_MODES = (
        "passthrough",
        "egocentric_velocity",
        "egocentric_velocity_head",
    )

    def __init__(
        self,
        env: gym.Env,
        width: int = 256,
        height: int = 256,
        horizontal_fov: float = 120.0,
        render_mode: str = "rgb_array",
        camera_height: float = 0.025,
        occlusion_height: float = 0.16,
        arena_wall_height: float = 0.22,
        far_clip: float = 2.0,
        detection_range: typing.Optional[float] = None,
        observation_mode: str = "mouse",
        action_mode: str = "egocentric_velocity",
        eye_yaw_degrees: float = 40.0,
        eye_separation: float = 0.016,
        eye_forward_offset: float = 0.012,
        max_body_turn_rate: float = 180.0,
        max_head_turn_rate: float = 240.0,
        head_yaw_limit: float = 60.0,
        head_recenter_rate: float = 90.0,
        passive_gaze_mode: str = "center",
        fixed_head_yaw_degrees: float = 0.0,
        passive_scan_targets_degrees: typing.Sequence[float] = (
            -60.0,
            -30.0,
            0.0,
            30.0,
            60.0,
            30.0,
            0.0,
            -30.0,
        ),
        passive_scan_dwell_steps: int = 2,
        velocity_gain: float = 5.0,
    ) -> None:
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}",
            )
        if observation_mode not in self.OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {self.OBSERVATION_MODES}, "
                f"got {observation_mode!r}",
            )
        if action_mode not in self.ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {self.ACTION_MODES}, got {action_mode!r}",
            )
        if not 0.0 <= eye_yaw_degrees < 90.0:
            raise ValueError("eye_yaw_degrees must be in [0, 90)")
        if eye_separation < 0 or eye_forward_offset < 0:
            raise ValueError("eye offsets must be non-negative")
        if max_body_turn_rate <= 0 or max_head_turn_rate <= 0:
            raise ValueError("body/head turn rates must be positive")
        if head_yaw_limit <= 0 or head_recenter_rate < 0:
            raise ValueError("head_yaw_limit must be positive and recenter rate non-negative")
        if passive_gaze_mode not in ("center", "fixed", "scan"):
            raise ValueError("passive_gaze_mode must be center, fixed, or scan")
        if action_mode == "egocentric_velocity_head" and passive_gaze_mode != "center":
            raise ValueError("Policy-controlled head yaw cannot also use passive gaze")
        if action_mode != "egocentric_velocity" and passive_gaze_mode != "center":
            raise ValueError("Passive gaze requires action_mode=egocentric_velocity")
        if abs(float(fixed_head_yaw_degrees)) > float(head_yaw_limit):
            raise ValueError("fixed_head_yaw_degrees exceeds head_yaw_limit")
        if not passive_scan_targets_degrees or any(
            abs(float(value)) > float(head_yaw_limit)
            for value in passive_scan_targets_degrees
        ):
            raise ValueError("Passive scan targets must lie within the head-yaw limit")
        if int(passive_scan_dwell_steps) <= 0:
            raise ValueError("passive_scan_dwell_steps must be positive")
        if velocity_gain <= 0:
            raise ValueError("velocity_gain must be positive")
        if detection_range is not None and detection_range <= 0:
            raise ValueError("detection_range must be positive when provided")

        super().__init__(env)
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            raise TypeError("FirstPersonVisionWrapper requires env.unwrapped.model")
        prey = getattr(model, "prey", None)
        if prey is None:
            raise TypeError("FirstPersonVisionWrapper requires env.unwrapped.model.prey")

        self._render_mode = render_mode
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.proprio_names = ("forward_speed", "body_yaw_rate", "head_yaw")
        self.eye_yaw_degrees = float(eye_yaw_degrees)
        self.eye_separation = float(eye_separation)
        self.eye_forward_offset = float(eye_forward_offset)
        self.detection_range = float(far_clip if detection_range is None else detection_range)
        self.max_body_turn_rate = float(max_body_turn_rate)
        self.max_head_turn_rate = float(max_head_turn_rate)
        self.head_yaw_limit = float(head_yaw_limit)
        self.head_recenter_rate = float(head_recenter_rate)
        self.passive_gaze_mode = str(passive_gaze_mode)
        self.fixed_head_yaw_degrees = float(fixed_head_yaw_degrees)
        self.passive_scan_targets_degrees = tuple(
            float(value) for value in passive_scan_targets_degrees
        )
        self.passive_scan_dwell_steps = int(passive_scan_dwell_steps)
        self.velocity_gain = float(velocity_gain)
        self._control_dt = float(getattr(env.unwrapped, "time_step", 0.1))
        if self._control_dt <= 0:
            raise ValueError("The wrapped environment time_step must be positive")

        self.renderer = FirstPersonRenderer(
            model=model,
            width=width,
            height=height,
            horizontal_fov=horizontal_fov,
            camera_height=camera_height,
            occlusion_height=occlusion_height,
            arena_wall_height=arena_wall_height,
            far_clip=far_clip,
            render_fps=self.metadata["render_fps"],
        )

        base_action_space = env.action_space
        if self.action_mode == "passthrough":
            self.action_space = base_action_space
            self.action_names = ("base_action",)
        else:
            if not isinstance(base_action_space, spaces.Box) or base_action_space.shape != (2,):
                raise TypeError(
                    "Egocentric first-person control requires a continuous base "
                    "action space shaped (2,)",
                )
            action_dimensions = 3 if self.action_mode.endswith("_head") else 2
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dimensions,),
                dtype=np.float32,
            )
            self.action_names = (
                ("forward_velocity", "body_yaw_rate", "head_yaw_rate")
                if action_dimensions == 3
                else ("forward_velocity", "body_yaw_rate")
            )

        image_space = spaces.Box(
            low=0,
            high=255,
            shape=self.renderer.image_shape,
            dtype=np.uint8,
        )
        if self.observation_mode == "single_rgb":
            self.observation_space = image_space
        else:
            if not isinstance(self.action_space, spaces.Box):
                raise TypeError("Mouse VLA observations require a continuous action space")
            previous_action_space = spaces.Box(
                low=np.asarray(self.action_space.low, dtype=np.float32),
                high=np.asarray(self.action_space.high, dtype=np.float32),
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict(
                {
                    "image_left": image_space,
                    "image_right": image_space,
                    # Normalized [forward speed, body yaw rate, head yaw].
                    "proprio": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
                    "previous_action": previous_action_space,
                },
            )

        self.state_observation: typing.Optional[np.ndarray] = None
        self._last_frame: typing.Optional[RGBFrame] = None
        self._head_yaw_degrees = 0.0
        self._passive_gaze_step = 0
        self._last_body_turn_command = 0.0
        self._previous_action = self._zero_policy_action()
        self._predator_visibility: typing.Dict[str, bool] = self._empty_predator_visibility()

    @staticmethod
    def _empty_predator_visibility() -> typing.Dict[str, bool]:
        return {
            "predator_geometric_los": False,
            "predator_in_left_frustum": False,
            "predator_in_right_frustum": False,
            "predator_pixels_visible": False,
            "predator_within_detection_range": False,
            "predator_believed_visible": False,
        }

    @property
    def predator_visibility(self) -> typing.Dict[str, bool]:
        """Latest camera-derived predator visibility fields.

        The canonical visual training target is
        ``predator_pixels_visible``.  ``predator_geometric_los`` is retained
        as a privileged diagnostic and is never used as that target.
        """

        return dict(self._predator_visibility)

    def get_predator_visibility(self) -> typing.Dict[str, bool]:
        """Return a fresh camera visibility snapshot for diagnostics."""

        self._predator_visibility = self._compute_predator_visibility()
        return self.predator_visibility

    @property
    def render_mode(self) -> str:
        return self._render_mode

    @property
    def body_heading_degrees(self) -> float:
        return _state_body_heading(self.env.unwrapped.model.prey.state)

    @property
    def head_yaw_degrees(self) -> float:
        return self._head_yaw_degrees

    @staticmethod
    def _wrap_degrees(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    def _zero_policy_action(self) -> np.ndarray:
        if isinstance(self.action_space, spaces.Box):
            return np.zeros(self.action_space.shape, dtype=np.float32)
        return np.zeros((1,), dtype=np.float32)

    def _reset_embodiment_state(self) -> None:
        self._head_yaw_degrees = (
            self.fixed_head_yaw_degrees
            if self.passive_gaze_mode == "fixed"
            else 0.0
        )
        self._passive_gaze_step = 0
        self._last_body_turn_command = 0.0
        self._previous_action = self._zero_policy_action()

    def _validated_policy_action(self, action) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != self.action_space.shape:
            raise ValueError(
                f"Expected policy action shaped {self.action_space.shape}, got {array.shape}",
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Policy action must contain only finite values")
        return np.clip(array, self.action_space.low, self.action_space.high).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _move_toward(value: float, target: float, maximum_delta: float) -> float:
        delta = target - value
        if abs(delta) <= maximum_delta:
            return target
        return value + math.copysign(maximum_delta, delta)

    def _advance_gaze(self, action: np.ndarray) -> None:
        self._last_body_turn_command = float(action[1])
        body_heading = self._wrap_degrees(
            self.body_heading_degrees
            + self._last_body_turn_command * self.max_body_turn_rate * self._control_dt,
        )
        _set_state_body_heading(self.env.unwrapped.model.prey.state, body_heading)

        if self.action_mode == "egocentric_velocity_head":
            head_command = float(action[2])
            if abs(head_command) > 0.05:
                self._head_yaw_degrees += (
                    head_command * self.max_head_turn_rate * self._control_dt
                )
            else:
                self._head_yaw_degrees = self._move_toward(
                    self._head_yaw_degrees,
                    0.0,
                    self.head_recenter_rate * self._control_dt,
                )
            self._head_yaw_degrees = float(
                np.clip(
                    self._head_yaw_degrees,
                    -self.head_yaw_limit,
                    self.head_yaw_limit,
                ),
            )
        elif self.passive_gaze_mode == "fixed":
            self._head_yaw_degrees = self.fixed_head_yaw_degrees
        elif self.passive_gaze_mode == "scan":
            target_index = (
                self._passive_gaze_step // self.passive_scan_dwell_steps
            ) % len(self.passive_scan_targets_degrees)
            target = self.passive_scan_targets_degrees[target_index]
            self._head_yaw_degrees = self._move_toward(
                self._head_yaw_degrees,
                target,
                self.max_head_turn_rate * self._control_dt,
            )
            self._passive_gaze_step += 1
        else:
            self._head_yaw_degrees = 0.0

    def _egocentric_to_world_acceleration(self, action: np.ndarray) -> np.ndarray:
        self._advance_gaze(action)
        prey = self.env.unwrapped.model.prey
        dynamics = getattr(prey, "dynamics", None)
        required = ("v_max", "accel_scale", "damping")
        if dynamics is None or any(not hasattr(dynamics, name) for name in required):
            raise TypeError(
                "Egocentric velocity control requires prey point-mass dynamics "
                "with v_max, accel_scale, and damping",
            )

        heading = math.radians(self.body_heading_degrees)
        forward_axis = np.asarray((math.cos(heading), math.sin(heading)), dtype=np.float64)
        desired_velocity = forward_axis * float(action[0]) * float(dynamics.v_max)
        current_velocity = np.asarray(prey.state.velocity, dtype=np.float64)

        # Invert dv/dt = accel_scale * u - damping * v, then add velocity
        # error feedback.  The wrapped environment still performs integration
        # and collision handling.
        physical_acceleration = (
            self.velocity_gain * (desired_velocity - current_velocity)
            + float(dynamics.damping) * current_velocity
        )
        world_action = physical_acceleration / float(dynamics.accel_scale)
        return np.clip(world_action, -1.0, 1.0).astype(np.float32)

    def _camera_centre_pose(self) -> typing.Tuple[np.ndarray, float]:
        prey = self.env.unwrapped.model.prey
        location = np.asarray(prey.state.location, dtype=np.float64)
        direction = self._wrap_degrees(
            self.body_heading_degrees + self._head_yaw_degrees,
        )
        return location, direction

    def _eye_pose(self, side: float) -> typing.Tuple[np.ndarray, float]:
        location, head_direction = self._camera_centre_pose()
        angle = math.radians(head_direction)
        forward_axis = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
        left_axis = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
        eye_origin = (
            location
            + forward_axis * self.eye_forward_offset
            + left_axis * side * self.eye_separation / 2.0
        )
        eye_direction = self._wrap_degrees(
            head_direction + side * self.eye_yaw_degrees,
        )
        return eye_origin, eye_direction

    def _compute_predator_visibility(self) -> typing.Dict[str, bool]:
        """Compute visibility from the two rendered camera frusta.

        The simulator's directional LOS is intentionally not used here.  The
        geometric field is an unbounded 360-degree occlusion diagnostic;
        angular frusta, rendered pixels, and the camera detection range are
        classified independently.
        """

        visibility = self._empty_predator_visibility()
        model = self.env.unwrapped.model
        if not getattr(model, "use_predator", False) or not hasattr(model, "predator"):
            return visibility

        prey_location = np.asarray(model.prey.state.location, dtype=np.float64)
        predator_location = np.asarray(model.predator.state.location, dtype=np.float64)
        distance = float(np.linalg.norm(predator_location - prey_location))
        visibility["predator_within_detection_range"] = distance <= self.detection_range

        model_visibility = getattr(model, "visibility", None)
        if model_visibility is not None and hasattr(model_visibility, "line_of_sight"):
            visibility["predator_geometric_los"] = bool(
                model_visibility.line_of_sight(prey_location, predator_location),
            )
        else:
            # Small test/fallback environments may expose only the cached LOS.
            line_of_sight = getattr(model, "line_of_sight", None)
            if line_of_sight is not None:
                try:
                    visibility["predator_geometric_los"] = bool(
                        line_of_sight["prey", "predator"],
                    )
                except (KeyError, TypeError, IndexError):
                    visibility["predator_geometric_los"] = False

        left_origin, left_direction = self._eye_pose(+1.0)
        right_origin, right_direction = self._eye_pose(-1.0)
        left = self.renderer.object_visibility(
            predator_location,
            origin=left_origin,
            direction_degrees=left_direction,
            object_width=PREDATOR_OBJECT_WIDTH,
            object_height=PREDATOR_OBJECT_HEIGHT,
            depth_tolerance=PREDATOR_DEPTH_TOLERANCE,
        )
        right = self.renderer.object_visibility(
            predator_location,
            origin=right_origin,
            direction_degrees=right_direction,
            object_width=PREDATOR_OBJECT_WIDTH,
            object_height=PREDATOR_OBJECT_HEIGHT,
            depth_tolerance=PREDATOR_DEPTH_TOLERANCE,
        )
        visibility["predator_in_left_frustum"] = bool(left["in_frustum"])
        visibility["predator_in_right_frustum"] = bool(right["in_frustum"])
        visibility["predator_pixels_visible"] = bool(
            left["pixels_visible"] or right["pixels_visible"],
        )
        visibility["predator_believed_visible"] = bool(
            visibility["predator_pixels_visible"]
            and visibility["predator_within_detection_range"],
        )
        return visibility

    def _augment_camera_info(self, info: typing.Optional[dict]) -> dict:
        """Attach canonical camera labels and update legacy aliases."""

        augmented = dict(info or {})
        model = self.env.unwrapped.model
        # Preserve the ordinary Gym wrapper contract for environments without
        # a predator (including lightweight callers that use the wrapper only
        # to test image observations).
        if not getattr(model, "use_predator", False) or not hasattr(model, "predator"):
            return augmented
        if "transition_events" not in augmented and not hasattr(model, "visibility"):
            return augmented
        visibility = self.predator_visibility
        events = dict(augmented.get("transition_events", {}))
        events.update(visibility)
        # Compatibility aliases.  New consumers should use the six canonical
        # fields above; the old names now follow camera semantics in this
        # first-person wrapper instead of simulator 360-degree LOS.
        events["predator_visible_camera"] = bool(visibility["predator_pixels_visible"])
        events["predator_visible_geometric"] = bool(visibility["predator_geometric_los"])
        events["prey_sees_predator"] = bool(visibility["predator_believed_visible"])
        augmented["transition_events"] = events
        for name, value in visibility.items():
            augmented[name] = bool(value)
        augmented["predator_visible_camera"] = bool(visibility["predator_pixels_visible"])
        augmented["predator_visible_geometric"] = bool(visibility["predator_geometric_los"])
        augmented["prey_sees_predator"] = bool(visibility["predator_believed_visible"])
        augmented["predator_visible_last_step"] = int(
            visibility["predator_believed_visible"],
        )
        return augmented

    def _render_single_eye(self) -> RGBFrame:
        origin, direction = self._camera_centre_pose()
        return self.renderer.render_rgb(origin=origin, direction_degrees=direction)

    def _render_binocular(self) -> typing.Tuple[RGBFrame, RGBFrame, RGBFrame]:
        left_origin, left_direction = self._eye_pose(+1.0)
        right_origin, right_direction = self._eye_pose(-1.0)
        left = self.renderer.render_rgb(
            origin=left_origin,
            direction_degrees=left_direction,
        )
        right = self.renderer.render_rgb(
            origin=right_origin,
            direction_degrees=right_direction,
        )
        divider = np.full((left.shape[0], 4, 3), (12, 15, 18), dtype=np.uint8)
        preview = np.ascontiguousarray(np.concatenate((left, divider, right), axis=1))
        return left, right, preview

    def _proprioception(self) -> np.ndarray:
        prey = self.env.unwrapped.model.prey
        velocity = np.asarray(
            getattr(prey.state, "velocity", (0.0, 0.0)),
            dtype=np.float64,
        )
        heading = math.radians(self.body_heading_degrees)
        forward_axis = np.asarray((math.cos(heading), math.sin(heading)), dtype=np.float64)
        dynamics = getattr(prey, "dynamics", None)
        v_max = max(float(getattr(dynamics, "v_max", 1.0)), 1e-8)
        forward_speed = float(np.clip(velocity @ forward_axis / v_max, -1.0, 1.0))
        normalized_head_yaw = float(
            np.clip(self._head_yaw_degrees / self.head_yaw_limit, -1.0, 1.0),
        )
        return np.asarray(
            (forward_speed, self._last_body_turn_command, normalized_head_yaw),
            dtype=np.float32,
        )

    def _vision_observation(self):
        if self.observation_mode == "single_rgb":
            frame = self._render_single_eye()
            self._last_frame = frame
            observation = frame
        else:
            left, right, preview = self._render_binocular()
            self._last_frame = preview
            observation = {
                "image_left": left,
                "image_right": right,
                "proprio": self._proprioception(),
                "previous_action": self._previous_action.copy(),
            }
        self._predator_visibility = self._compute_predator_visibility()
        if self.render_mode == "human":
            self.renderer.show(self._last_frame)
        return observation

    def get_state_dict(self) -> dict:
        """Snapshot the wrapped simulation and first-person embodiment state."""

        get_base_state = getattr(self.env, "get_state_dict", None)
        if get_base_state is None:
            raise AttributeError("The wrapped environment does not support state snapshots")
        state = get_base_state()
        state["first_person"] = {
            # Retained as a compatibility/diagnostic field.  The physical
            # model state is the only source of truth for this value.
            "body_heading_degrees": float(self.body_heading_degrees),
            "head_yaw_degrees": float(self._head_yaw_degrees),
            "last_body_turn_command": float(self._last_body_turn_command),
            "previous_action": self._previous_action.copy(),
            "state_observation": copy.deepcopy(self.state_observation),
            "last_frame": copy.deepcopy(self._last_frame),
            "predator_visibility": copy.deepcopy(self._predator_visibility),
        }
        if self.passive_gaze_mode == "scan":
            state["first_person"]["passive_gaze_step"] = int(
                self._passive_gaze_step,
            )
        return state

    def set_state_dict(self, state: dict) -> None:
        """Restore a snapshot produced by :meth:`get_state_dict`."""

        set_base_state = getattr(self.env, "set_state_dict", None)
        if set_base_state is None:
            raise AttributeError("The wrapped environment does not support state snapshots")
        set_base_state(state)
        embodiment_state = state.get("first_person", {})
        if "head_yaw_degrees" in embodiment_state:
            self._head_yaw_degrees = float(embodiment_state["head_yaw_degrees"])
        if "passive_gaze_step" in embodiment_state:
            self._passive_gaze_step = int(embodiment_state["passive_gaze_step"])
        if "last_body_turn_command" in embodiment_state:
            self._last_body_turn_command = float(embodiment_state["last_body_turn_command"])
        if "previous_action" in embodiment_state:
            self._previous_action = np.array(
                embodiment_state["previous_action"],
                dtype=np.float32,
                copy=True,
            )
        if "state_observation" in embodiment_state:
            self.state_observation = copy.deepcopy(embodiment_state["state_observation"])
        if "last_frame" in embodiment_state:
            self._last_frame = copy.deepcopy(embodiment_state["last_frame"])
        if "predator_visibility" in embodiment_state:
            restored_visibility = self._empty_predator_visibility()
            restored_visibility.update(
                {
                    key: bool(value)
                    for key, value in embodiment_state["predator_visibility"].items()
                    if key in restored_visibility
                },
            )
            self._predator_visibility = restored_visibility

    def reset(self, **kwargs):
        self.state_observation, info = self.env.reset(**kwargs)
        self._reset_embodiment_state()
        observation = self._vision_observation()
        return observation, self._augment_camera_info(info)

    def step(self, action):
        if self.action_mode == "passthrough":
            base_action = action
            if isinstance(self.action_space, spaces.Box):
                policy_action = self._validated_policy_action(action)
                base_action = policy_action
                self._previous_action = policy_action.copy()
        else:
            policy_action = self._validated_policy_action(action)
            self._previous_action = policy_action.copy()
            base_action = self._egocentric_to_world_acceleration(policy_action)

        self.state_observation, reward, terminated, truncated, info = self.env.step(
            base_action,
        )
        if self.action_mode == "passthrough":
            self._head_yaw_degrees = 0.0
            self._last_body_turn_command = 0.0
        observation = self._vision_observation()
        return observation, reward, terminated, truncated, self._augment_camera_info(info)

    def render_first_person(self, mode: typing.Optional[str] = None):
        """Render the current prey camera without advancing simulation state."""

        selected_mode = self.render_mode if mode is None else mode
        if selected_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported first-person render mode: {selected_mode!r}")
        if self.observation_mode == "mouse":
            _, _, frame = self._render_binocular()
        else:
            frame = self._render_single_eye()
        self._predator_visibility = self._compute_predator_visibility()
        self._last_frame = frame
        if selected_mode == "human":
            self.renderer.show(frame)
            return None
        return frame

    def render(self):
        """Gymnasium render API; binocular mode returns a side-by-side preview."""

        return self.render_first_person(self.render_mode)

    def render_top_down(self, normalized: bool = False) -> np.ndarray:
        """Return the legacy top-down view when the base env has ``render=True``."""

        view = getattr(self.env.unwrapped.model, "view", None)
        if view is None:
            raise RuntimeError("Top-down rendering requires render=True on the base environment")
        view.render()
        return view.get_screen(normalized=normalized)

    def close(self) -> None:
        self.renderer.close()
        self.env.close()
