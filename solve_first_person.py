"""Solve BotEvade with the real point-mass dynamics and record prey-eye GIF.

The controller is intentionally transparent and does not alter simulation
state.  It follows Cellworld's precomputed free-cell path with a velocity
feedback controller, then adds a short-range predator repulsion term.  Its
output uses legacy world-frame ``(ax, ay)`` passthrough control so it remains a
transparent physics/vision baseline; learned first-person policies use the
egocentric interface.

Example:
    conda run -n Mice-BotEvade python -B solve_first_person.py \
        --output results/botevade_first_person_solution.gif
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="21_05")
    parser.add_argument(
        "--output",
        default="results/botevade_first_person_solution.gif",
        help="Output split-screen GIF path",
    )
    parser.add_argument(
        "--trajectory-output",
        help="Final top-down trajectory PNG (default: derived from --output)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--time-step", type=float, default=0.10)
    parser.add_argument("--width", type=int, default=192, help="Width of each eye image")
    parser.add_argument("--height", type=int, default=128, help="Height of each eye image")
    parser.add_argument("--fov", type=float, default=120.0, help="Horizontal FOV per eye")
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--predator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predator-ratio", type=float, default=0.15)
    parser.add_argument("--avoid-radius", type=float, default=0.24)
    parser.add_argument("--human", action="store_true", help="Also show the live pygame window")
    return parser.parse_args()


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _vertices(polygon) -> np.ndarray:
    vertices = polygon.vertices
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    return np.asarray(vertices, dtype=np.float64)


class TopDownTrajectoryRenderer:
    """Draw the exact map and progressively accumulated agent trajectories."""

    def __init__(self, model, width: int, height: int, margin: int = 12) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        arena_vertices = _vertices(model.arena)
        minimum = arena_vertices.min(axis=0)
        maximum = arena_vertices.max(axis=0)
        world_size = maximum - minimum
        self.scale = min(
            (self.width - 2 * self.margin) / world_size[0],
            (self.height - 2 * self.margin) / world_size[1],
        )
        drawing_size = world_size * self.scale
        self.offset_x = (self.width - drawing_size[0]) / 2.0 - minimum[0] * self.scale
        self.offset_y = (self.height - drawing_size[1]) / 2.0 + maximum[1] * self.scale

    def _screen_point(self, point: Sequence[float]) -> Tuple[int, int]:
        x = self.offset_x + float(point[0]) * self.scale
        y = self.offset_y - float(point[1]) * self.scale
        return int(round(x)), int(round(y))

    def _screen_polygon(self, polygon) -> List[Tuple[int, int]]:
        return [self._screen_point(vertex) for vertex in _vertices(polygon)]

    def render(
        self,
        prey_trajectory: Sequence[Point],
        predator_trajectory: Sequence[Point],
    ) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (24, 29, 34))
        draw = ImageDraw.Draw(image)
        draw.polygon(
            self._screen_polygon(self.model.arena),
            fill=(205, 207, 197),
            outline=(235, 236, 226),
            width=2,
        )
        for occlusion in self.model.occlusions:
            draw.polygon(
                self._screen_polygon(occlusion),
                fill=(65, 69, 74),
                outline=(91, 95, 100),
            )

        if len(predator_trajectory) >= 2:
            draw.line(
                [self._screen_point(point) for point in predator_trajectory],
                fill=(225, 70, 60),
                width=2,
                joint="curve",
            )
        if len(prey_trajectory) >= 2:
            draw.line(
                [self._screen_point(point) for point in prey_trajectory],
                fill=(24, 191, 224),
                width=3,
                joint="curve",
            )

        goal = self._screen_point(self.model.goal_location)
        goal_radius = 7
        draw.ellipse(
            (
                goal[0] - goal_radius,
                goal[1] - goal_radius,
                goal[0] + goal_radius,
                goal[1] + goal_radius,
            ),
            fill=(45, 229, 91),
            outline=(224, 255, 229),
            width=2,
        )

        if prey_trajectory:
            start = self._screen_point(prey_trajectory[0])
            draw.ellipse(
                (start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4),
                outline=(236, 240, 243),
                width=2,
            )
            prey = self._screen_point(prey_trajectory[-1])
            draw.ellipse(
                (prey[0] - 5, prey[1] - 5, prey[0] + 5, prey[1] + 5),
                fill=(24, 191, 224),
                outline=(239, 252, 255),
                width=2,
            )
            heading = math.radians(float(self.model.prey.state.body_heading))
            heading_tip = self._screen_point(
                (
                    prey_trajectory[-1][0] + 0.045 * math.cos(heading),
                    prey_trajectory[-1][1] + 0.045 * math.sin(heading),
                ),
            )
            draw.line((prey, heading_tip), fill=(255, 218, 74), width=2)

        if predator_trajectory:
            predator = self._screen_point(predator_trajectory[-1])
            draw.ellipse(
                (
                    predator[0] - 5,
                    predator[1] - 5,
                    predator[0] + 5,
                    predator[1] + 5,
                ),
                fill=(225, 70, 60),
                outline=(255, 225, 221),
                width=2,
            )
        return np.asarray(image, dtype=np.uint8)


def combine_views(first_person: np.ndarray, top_down: np.ndarray) -> np.ndarray:
    if first_person.shape[0] != top_down.shape[0]:
        raise ValueError("First-person and top-down frames must have equal heights")
    divider = np.full((first_person.shape[0], 4, 3), (12, 15, 18), dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((first_person, divider, top_down), axis=1))


def binocular_preview(observation) -> np.ndarray:
    """Convert either first-person observation contract to one display frame."""

    if not isinstance(observation, dict):
        return observation
    left = observation["image_left"]
    right = observation["image_right"]
    divider = np.full((left.shape[0], 4, 3), (12, 15, 18), dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((left, divider, right), axis=1))


class CellPathController:
    """Free-cell path follower that emits only legal point-mass actions."""

    def __init__(
        self,
        env: FirstPersonBotEvadeEnv,
        avoid_radius: float = 0.24,
        waypoint_tolerance: float = 0.035,
        replan_interval: int = 12,
    ) -> None:
        self.env = env
        self.base_env: BotEvadeEnv = env.unwrapped
        self.model = self.base_env.model
        self.navigation = self.base_env.loader.navigation
        self.locations = self.base_env.loader.locations
        self.avoid_radius = float(avoid_radius)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.replan_interval = int(replan_interval)
        self.waypoints: List[Point] = []
        self.steps = 0

    def reset(self) -> None:
        self.steps = 0
        self._replan()

    def _replan(self) -> None:
        """Expand Cellworld's next-hop table into collision-safe cell centers."""

        start = self.model.prey.state.location
        goal = self.model.goal_location
        current = self.navigation.closest_location(start)
        destination = self.navigation.closest_location(goal)
        path: List[Point] = []
        visited = set()

        while current is not None and current != destination:
            if current in visited:
                raise RuntimeError("Cellworld navigation table contains a cycle")
            visited.add(current)
            next_cell = self.navigation.paths[current][destination]
            if next_cell is None:
                break
            # Cellworld uses a self next-hop when the destination can be
            # reached directly from the current visibility region.
            if next_cell == current:
                destination_location = self.locations[destination]
                if destination_location is not None:
                    path.append(tuple(destination_location))
                current = destination
                break
            location = self.locations[next_cell]
            if location is not None:
                path.append(tuple(location))
            current = next_cell

        if current != destination:
            raise RuntimeError(f"No navigation path from {start} to {goal}")
        if not path:
            path.append(tuple(self.locations[destination]))
        self.waypoints = path

    def _drop_reached_waypoints(self, position: np.ndarray) -> None:
        while self.waypoints and _distance(position, self.waypoints[0]) <= self.waypoint_tolerance:
            self.waypoints.pop(0)

    def _target(self, position: np.ndarray) -> np.ndarray:
        self._drop_reached_waypoints(position)
        if not self.waypoints:
            return np.asarray(self.model.goal_location, dtype=np.float64)

        # A small look-ahead smooths the dense hex-cell route while retaining
        # clearance around corners.  Never skip more than three cell centers.
        target_index = 0
        for index, waypoint in enumerate(self.waypoints[:3]):
            if _distance(position, waypoint) <= 0.115:
                target_index = index
        return np.asarray(self.waypoints[target_index], dtype=np.float64)

    def action(self) -> np.ndarray:
        if self.steps and self.steps % self.replan_interval == 0:
            self._replan()
        self.steps += 1

        prey = self.model.prey
        dynamics = prey.dynamics
        position = np.asarray(prey.state.location, dtype=np.float64)
        velocity = np.asarray(prey.state.velocity, dtype=np.float64)
        target_delta = self._target(position) - position
        target_distance = float(np.linalg.norm(target_delta))

        if target_distance > 1e-8:
            target_direction = target_delta / target_distance
        else:
            target_direction = np.zeros(2, dtype=np.float64)

        goal_distance = _distance(position, self.model.goal_location)
        desired_speed = min(dynamics.v_max * 0.82, max(0.10, 2.8 * goal_distance))
        desired_velocity = target_direction * desired_speed

        if self.model.use_predator:
            predator_delta = position - np.asarray(
                self.model.predator.state.location,
                dtype=np.float64,
            )
            predator_distance = float(np.linalg.norm(predator_delta))
            if 1e-8 < predator_distance < self.avoid_radius:
                danger = (self.avoid_radius - predator_distance) / self.avoid_radius
                desired_velocity += (
                    predator_delta / predator_distance
                ) * dynamics.v_max * 1.35 * danger

        desired_norm = float(np.linalg.norm(desired_velocity))
        if desired_norm > dynamics.v_max:
            desired_velocity *= dynamics.v_max / desired_norm

        # Invert dv/dt = accel_scale * action - damping * velocity, with an
        # additional velocity-error response term.  This is feedback control,
        # not a bypass of Model._point_step.
        acceleration = 5.0 * (desired_velocity - velocity) + dynamics.damping * velocity
        action = acceleration / dynamics.accel_scale
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def save_gif(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("Cannot save an empty GIF")
    if fps <= 0:
        raise ValueError("GIF fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(1, int(round(1000.0 / fps)))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def make_env(args: argparse.Namespace) -> FirstPersonBotEvadeEnv:
    return FirstPersonBotEvadeEnv(
        world_name=args.world,
        use_lppos=False,
        use_predator=args.predator,
        max_step=args.max_steps,
        time_step=args.time_step,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.CONTINUOUS,
        frame_stack_k=1,
        predator_prey_forward_speed_ratio=args.predator_ratio,
        vision_width=args.width,
        vision_height=args.height,
        vision_fov=args.fov,
        observation_mode="mouse",
        # The transparent path solver still emits the base env's world-frame
        # accelerations.  Learned first-person policies use the new egocentric
        # default instead.
        action_mode="passthrough",
        render_mode="human" if args.human else "rgb_array",
    )


def run_attempt(
    env: FirstPersonBotEvadeEnv,
    controller: CellPathController,
    top_down_renderer: TopDownTrajectoryRenderer,
    seed: int,
    max_steps: int,
) -> Tuple[bool, List[np.ndarray], List[Point], List[Point], dict]:
    random.seed(seed)
    np.random.seed(seed)
    observation, _ = env.reset(seed=seed)
    controller.reset()
    model = env.unwrapped.model
    prey_trajectory = [tuple(model.prey.state.location)]
    predator_trajectory = (
        [tuple(model.predator.state.location)] if model.use_predator else []
    )
    top_down = top_down_renderer.render(prey_trajectory, predator_trajectory)
    frames = [combine_views(binocular_preview(observation), top_down)]
    final_info = {}

    for _ in range(max_steps):
        action = controller.action()
        observation, _, terminated, truncated, final_info = env.step(action)
        prey_trajectory.append(tuple(model.prey.state.location))
        if model.use_predator:
            predator_trajectory.append(tuple(model.predator.state.location))
        top_down = top_down_renderer.render(prey_trajectory, predator_trajectory)
        frames.append(combine_views(binocular_preview(observation), top_down))
        if terminated or truncated:
            success = bool(
                final_info.get("transition_events", {}).get("goal_event", False)
            )
            return success, frames, prey_trajectory, predator_trajectory, final_info
    return False, frames, prey_trajectory, predator_trajectory, final_info


def main() -> None:
    args = parse_args()
    env = make_env(args)
    controller = CellPathController(env, avoid_radius=args.avoid_radius)
    top_down_renderer = TopDownTrajectoryRenderer(
        env.unwrapped.model,
        width=args.width,
        height=args.height,
    )
    best_frames: List[np.ndarray] = []
    best_prey_trajectory: List[Point] = []
    best_predator_trajectory: List[Point] = []
    best_goal_distance = math.inf
    best_captures = math.inf
    success = False
    chosen_attempt = 0
    final_info = {}

    try:
        for attempt in range(args.attempts):
            attempt_seed = args.seed + attempt
            solved, frames, prey_trajectory, predator_trajectory, info = run_attempt(
                env,
                controller,
                top_down_renderer,
                seed=attempt_seed,
                max_steps=args.max_steps,
            )
            reward_terms = info.get("reward_terms", env.unwrapped.reward_terms)
            episode_metrics = info.get("episode_metrics", {})
            goal_distance = float(reward_terms["goal_distance"])
            captures = int(
                episode_metrics.get(
                    "capture_count",
                    env.unwrapped.capture_event_count,
                )
            )
            clean_success = solved and captures == 0
            print(
                f"attempt={attempt + 1}/{args.attempts} seed={attempt_seed} "
                f"steps={len(frames) - 1} goal_reached={solved} "
                f"clean_success={clean_success} "
                f"goal_distance={goal_distance:.4f} "
                f"captures={captures}",
            )
            if (captures, goal_distance) < (best_captures, best_goal_distance):
                best_captures = captures
                best_goal_distance = goal_distance
                best_frames = frames
                best_prey_trajectory = prey_trajectory
                best_predator_trajectory = predator_trajectory
                final_info = info
                chosen_attempt = attempt + 1
            if clean_success:
                success = True
                best_frames = frames
                best_prey_trajectory = prey_trajectory
                best_predator_trajectory = predator_trajectory
                final_info = info
                chosen_attempt = attempt + 1
                break
    finally:
        env.close()

    if not best_frames:
        raise RuntimeError("Solver produced no frames")

    # Hold the result briefly at the end so success is readable in a player.
    best_frames.extend([best_frames[-1].copy()] * max(1, int(round(args.gif_fps))))
    output_path = Path(args.output).expanduser().resolve()
    save_gif(output_path, best_frames, args.gif_fps)
    if args.trajectory_output:
        trajectory_path = Path(args.trajectory_output).expanduser().resolve()
    else:
        trajectory_path = output_path.with_name(f"{output_path.stem}_trajectory.png")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    final_top_down = top_down_renderer.render(
        best_prey_trajectory,
        best_predator_trajectory,
    )
    Image.fromarray(final_top_down).save(trajectory_path)
    print(
        f"saved={output_path} frames={len(best_frames)} "
        f"trajectory={trajectory_path} "
        f"chosen_attempt={chosen_attempt} success={success} info={final_info}",
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
