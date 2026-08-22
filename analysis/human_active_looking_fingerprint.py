"""EXP-02 human active-looking fingerprint analysis.

The module is intentionally descriptive.  It reads recorded human
demonstrations, keeps privileged simulator fields on the analysis side, and
writes episode/session/participant summaries that describe active-looking
structure rather than only success rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmarks.peekbench.artifacts import write_csv, write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "human_demos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "human_active_looking_fingerprint"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cellworld_cache"

LEFT_SIGN = 1.0
RIGHT_SIGN = -1.0
RISK_BINS = ("low", "medium", "high")


@dataclass(frozen=True)
class Exp02Issue:
    severity: str
    session: str
    episode: str | None
    message: str


@dataclass(frozen=True)
class AnalysisParameters:
    split_seed: int = 23
    look_threshold: float = 0.10
    body_threshold: float = 0.10
    deceleration_threshold: float = 0.10
    lead_window_seconds: float = 1.50
    approach_window_seconds: float = 1.50
    baseline_window_seconds: float = 1.50
    reconfirm_window_seconds: float = 5.00
    route_pre_window_seconds: float = 1.00
    route_post_window_seconds: float = 2.00
    route_change_degrees: float = 25.0
    risk_high_distance: float = 0.18
    risk_medium_distance: float = 0.35
    info_value_angle_degrees: float = 12.0
    info_value_distance: float = 0.55
    junction_occlusion_distance: float = 0.12


@dataclass(frozen=True)
class WorldGeometry:
    world_name: str
    occlusion_polygons: tuple[np.ndarray, ...]

    def distance_to_occlusions(self, positions: np.ndarray) -> np.ndarray:
        distances = np.full((len(positions),), np.nan, dtype=np.float64)
        if not self.occlusion_polygons:
            return distances
        for index, point in enumerate(np.asarray(positions, dtype=np.float64)):
            if point.shape != (2,) or not np.isfinite(point).all():
                continue
            distances[index] = min(
                _point_to_polygon_distance(point, polygon)
                for polygon in self.occlusion_polygons
            )
        return distances


def _safe_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _sum_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.sum(finite)) if finite else None


def _rate(mask: np.ndarray, denominator_mask: np.ndarray | None = None) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if denominator_mask is None:
        return float(mask.mean()) if len(mask) else None
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    total = int(denominator_mask.sum())
    if total == 0:
        return None
    return float((mask & denominator_mask).sum() / total)


def _split_for_group(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def _episode_files(session_dir: Path, metadata: Mapping[str, Any]) -> list[Path]:
    listed = [
        session_dir / str(record["file"])
        for record in metadata.get("episodes", [])
        if isinstance(record, Mapping) and record.get("file")
    ]
    return listed if listed else sorted(session_dir.glob("episode_*.npz"))


def _regular_polygon_vertices(
    center: Sequence[float],
    *,
    diameter: float,
    rotation_degrees: float,
    sides: int,
) -> np.ndarray:
    radius = float(diameter) / 2.0
    angles = np.radians(float(rotation_degrees) + np.arange(int(sides)) * 360.0 / int(sides))
    center_array = np.asarray(center, dtype=np.float64)
    return center_array + radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = float(vertices[i, 0]), float(vertices[i, 1])
        xj, yj = float(vertices[j, 0]), float(vertices[j, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            denominator = yj - yi
            if abs(denominator) <= 1e-12:
                denominator = math.copysign(1e-12, denominator or 1.0)
            x_intersect = (xj - xi) * (y - yi) / denominator + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared <= 1e-18:
        return float(np.linalg.norm(point - start))
    projection = float(np.clip(((point - start) @ segment) / length_squared, 0.0, 1.0))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def _point_to_polygon_distance(point: np.ndarray, vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    if _point_in_polygon(point, vertices):
        return 0.0
    ends = np.roll(vertices, -1, axis=0)
    return min(
        _point_segment_distance(point, start, end)
        for start, end in zip(vertices, ends)
    )


def load_world_geometry(
    world_name: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> WorldGeometry | None:
    cache_root = Path(cache_root)
    implementation_path = cache_root / "world_implementation" / "hexagonal.canonical"
    configuration_path = cache_root / "world_configuration" / "hexagonal"
    occlusion_path = cache_root / "cell_group" / f"hexagonal.{world_name}.occlusions"
    if not (
        implementation_path.exists()
        and configuration_path.exists()
        and occlusion_path.exists()
    ):
        return None
    implementation = _safe_json(implementation_path)
    configuration = _safe_json(configuration_path)
    occlusion_ids = json.loads(occlusion_path.read_text(encoding="utf-8"))
    if not isinstance(occlusion_ids, list):
        raise TypeError(f"Occlusion group is not a JSON list: {occlusion_path}")

    sides = int(configuration.get("cell_shape", {}).get("sides", 6))
    locations = list(implementation.get("cell_locations", []))
    cell_transform = implementation.get("cell_transformation", {})
    space_transform = implementation.get("space", {}).get("transformation", {})
    cell_size = float(cell_transform.get("size", 0.0))
    cell_rotation = float(space_transform.get("rotation", 0.0)) + float(
        cell_transform.get("rotation", 0.0),
    )
    polygons = []
    for cell_id in occlusion_ids:
        cell_index = int(cell_id)
        if cell_index < 0 or cell_index >= len(locations):
            continue
        location = locations[cell_index]
        polygons.append(
            _regular_polygon_vertices(
                (float(location["x"]), float(location["y"])),
                diameter=cell_size,
                rotation_degrees=cell_rotation,
                sides=sides,
            ),
        )
    return WorldGeometry(str(world_name), tuple(polygons))


def _wrap_degrees(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _circular_mean_degrees(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    radians = np.radians(values)
    vector = complex(float(np.cos(radians).mean()), float(np.sin(radians).mean()))
    if abs(vector) <= 1e-12:
        return None
    return float(math.degrees(math.atan2(vector.imag, vector.real)))


def _first_after(mask: np.ndarray, start: int, stop: int | None = None) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    end = len(mask) if stop is None else min(int(stop), len(mask))
    if start >= end:
        return None
    indices = np.flatnonzero(mask[int(start) : end])
    return int(start + indices[0]) if len(indices) else None


def _rising_edges(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def _falling_edges(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.flatnonzero(~mask & np.r_[False, mask[:-1]])


def _frames_for_seconds(sim_time: np.ndarray, seconds: float, control_hz: float | None) -> int:
    if len(sim_time) > 1:
        deltas = np.diff(np.asarray(sim_time, dtype=np.float64))
        positive = deltas[np.isfinite(deltas) & (deltas > 1e-9)]
        if len(positive):
            return max(1, int(round(float(seconds) / float(np.median(positive)))))
    if control_hz and control_hz > 0:
        return max(1, int(round(float(seconds) * float(control_hz))))
    return max(1, int(round(float(seconds) * 10.0)))


def _state_column(
    privileged: np.ndarray,
    names: Sequence[str],
    name: str,
    *,
    default: float = math.nan,
) -> np.ndarray:
    if name in names and privileged.ndim == 2 and names.index(name) < privileged.shape[1]:
        return privileged[:, names.index(name)].astype(np.float64)
    return np.full((len(privileged),), float(default), dtype=np.float64)


def _positions_from_state(privileged: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.stack(
        (
            _state_column(privileged, names, "prey_x"),
            _state_column(privileged, names, "prey_y"),
        ),
        axis=1,
    )


def _trajectory_metrics(
    positions: np.ndarray,
    metadata: Mapping[str, Any],
    success: bool,
) -> Mapping[str, float | None]:
    finite = np.isfinite(positions).all(axis=1)
    positions = np.asarray(positions, dtype=np.float64)[finite]
    if len(positions) < 2:
        return {
            "path_length": None,
            "path_efficiency": None,
            "progress_efficiency": None,
        }
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    goal = np.asarray(metadata.get("goal_location", (1.0, 0.5)), dtype=np.float64)
    start_distance = float(np.linalg.norm(goal - positions[0]))
    final_distance = float(np.linalg.norm(goal - positions[-1]))
    path_efficiency = None
    if success and path_length > 1e-8:
        path_efficiency = float(np.clip(start_distance / path_length, 0.0, 1.0))
    progress_efficiency = None
    if path_length > 1e-8:
        progress_efficiency = float(
            np.clip((start_distance - final_distance) / path_length, -1.0, 1.0),
        )
    return {
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "progress_efficiency": progress_efficiency,
    }


def _minimum_distance(
    data: Mapping[str, np.ndarray],
    positions: np.ndarray,
    predator_positions: np.ndarray,
) -> np.ndarray:
    if "minimum_distance" in data:
        return np.asarray(data["minimum_distance"], dtype=np.float64)
    if "prey_predator_distance" in data:
        return np.asarray(data["prey_predator_distance"], dtype=np.float64)
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(predator_positions).all(axis=1)
    distance = np.full((len(positions),), np.nan, dtype=np.float64)
    distance[finite] = np.linalg.norm(predator_positions[finite] - positions[finite], axis=1)
    return distance


def _risk_context(
    *,
    visible: np.ndarray,
    geometric: np.ndarray,
    within_detection: np.ndarray,
    minimum_distance: np.ndarray,
    parameters: AnalysisParameters,
) -> np.ndarray:
    near = np.isfinite(minimum_distance) & (minimum_distance <= parameters.risk_medium_distance)
    return np.asarray(visible | geometric | within_detection | near, dtype=bool)


def _risk_bins(
    *,
    visible: np.ndarray,
    geometric: np.ndarray,
    minimum_distance: np.ndarray,
    parameters: AnalysisParameters,
) -> np.ndarray:
    bins = np.full((len(visible),), "low", dtype=object)
    medium = geometric | (
        np.isfinite(minimum_distance)
        & (minimum_distance <= parameters.risk_medium_distance)
    )
    high = visible | (
        np.isfinite(minimum_distance)
        & (minimum_distance <= parameters.risk_high_distance)
    )
    bins[medium] = "medium"
    bins[high] = "high"
    return bins


def _danger_deceleration_metrics(
    *,
    risk_mask: np.ndarray,
    occlusion_distance: np.ndarray,
    forward_command: np.ndarray,
    physical_speed: np.ndarray,
    sim_time: np.ndarray,
    control_hz: float | None,
    parameters: AnalysisParameters,
) -> Mapping[str, Any]:
    approach_frames = _frames_for_seconds(sim_time, parameters.approach_window_seconds, control_hz)
    baseline_frames = _frames_for_seconds(sim_time, parameters.baseline_window_seconds, control_hz)
    onsets = _rising_edges(risk_mask)
    deltas = []
    speed_deltas = []
    evaluated = 0
    for onset in onsets:
        if math.isfinite(float(occlusion_distance[onset])) and (
            occlusion_distance[onset] > parameters.junction_occlusion_distance
        ):
            continue
        approach_start = max(0, int(onset) - approach_frames)
        baseline_start = max(0, approach_start - baseline_frames)
        if approach_start >= int(onset) or baseline_start >= approach_start:
            continue
        baseline = forward_command[baseline_start:approach_start]
        approach = forward_command[approach_start:int(onset)]
        baseline_speed = physical_speed[baseline_start:approach_start]
        approach_speed = physical_speed[approach_start:int(onset)]
        if len(baseline) == 0 or len(approach) == 0:
            continue
        evaluated += 1
        deltas.append(float(np.nanmean(baseline) - np.nanmean(approach)))
        if np.isfinite(baseline_speed).any() and np.isfinite(approach_speed).any():
            speed_deltas.append(float(np.nanmean(baseline_speed) - np.nanmean(approach_speed)))
    return {
        "danger_junction_events": int(evaluated),
        "pre_danger_deceleration": _mean_or_none(deltas),
        "pre_danger_physical_speed_drop": _mean_or_none(speed_deltas),
        "decelerated_before_danger_fraction": (
            _mean_or_none([delta > parameters.deceleration_threshold for delta in deltas])
            if deltas
            else None
        ),
    }


def _head_lead_metrics(
    *,
    head_command: np.ndarray,
    body_command: np.ndarray,
    sim_time: np.ndarray,
    control_hz: float | None,
    parameters: AnalysisParameters,
) -> Mapping[str, Any]:
    lead_frames = _frames_for_seconds(sim_time, parameters.lead_window_seconds, control_hz)
    head_onsets = _rising_edges(np.abs(head_command) > parameters.look_threshold)
    body_onsets = _rising_edges(np.abs(body_command) > parameters.body_threshold)
    leads = []
    simultaneous = 0
    for body_index in body_onsets:
        body_sign = math.copysign(1.0, float(body_command[body_index]))
        if abs(float(head_command[body_index])) > parameters.look_threshold and (
            math.copysign(1.0, float(head_command[body_index])) == body_sign
        ):
            simultaneous += 1
        candidates = [
            int(index)
            for index in head_onsets
            if int(body_index) - lead_frames <= int(index) < int(body_index)
            and math.copysign(1.0, float(head_command[index])) == body_sign
        ]
        if candidates:
            head_index = candidates[-1]
            leads.append(float(sim_time[body_index] - sim_time[head_index]))
    body_count = int(len(body_onsets))
    return {
        "body_turn_onsets": body_count,
        "head_turn_before_body_fraction": (
            float(len(leads) / body_count) if body_count else None
        ),
        "mean_head_lead_seconds": _mean_or_none(leads),
        "head_body_simultaneous_fraction": (
            float(simultaneous / body_count) if body_count else None
        ),
    }


def _reconfirmation_metrics(
    *,
    visible: np.ndarray,
    look_mask: np.ndarray,
    sim_time: np.ndarray,
    control_hz: float | None,
    parameters: AnalysisParameters,
) -> Mapping[str, Any]:
    falling = _falling_edges(visible)
    horizon = _frames_for_seconds(sim_time, parameters.reconfirm_window_seconds, control_hz)
    action_latencies = []
    pixel_latencies = []
    for index in falling:
        next_look = _first_after(look_mask, int(index) + 1, int(index) + 1 + horizon)
        if next_look is not None:
            action_latencies.append(float(sim_time[next_look] - sim_time[index]))
        next_visible = _first_after(visible, int(index) + 1, int(index) + 1 + horizon)
        if next_visible is not None:
            pixel_latencies.append(float(sim_time[next_visible] - sim_time[index]))
    return {
        "predator_loss_events": int(len(falling)),
        "reconfirm_action_latency_after_loss": _mean_or_none(action_latencies),
        "reconfirm_pixels_latency_after_loss": _mean_or_none(pixel_latencies),
        "reconfirm_action_events": int(len(action_latencies)),
        "reconfirm_pixels_events": int(len(pixel_latencies)),
    }


def _information_value_metrics(
    *,
    positions: np.ndarray,
    predator_positions: np.ndarray,
    body_heading: np.ndarray,
    left_frustum: np.ndarray,
    right_frustum: np.ndarray,
    risk_mask: np.ndarray,
    minimum_distance: np.ndarray,
    head_command: np.ndarray,
    parameters: AnalysisParameters,
) -> Mapping[str, Any]:
    look_direction = np.zeros((len(head_command),), dtype=np.float64)
    look_direction[head_command > parameters.look_threshold] = LEFT_SIGN
    look_direction[head_command < -parameters.look_threshold] = RIGHT_SIGN

    value_side = np.zeros((len(head_command),), dtype=np.float64)
    finite_position = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(predator_positions).all(axis=1)
        & np.isfinite(body_heading)
    )
    if finite_position.any():
        delta = predator_positions - positions
        bearing = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        relative = _wrap_degrees(bearing - body_heading)
        valuable = (
            finite_position
            & (np.abs(relative) >= parameters.info_value_angle_degrees)
            & (
                risk_mask
                | (
                    np.isfinite(minimum_distance)
                    & (minimum_distance <= parameters.info_value_distance)
                )
            )
        )
        value_side[valuable & (relative > 0.0)] = LEFT_SIGN
        value_side[valuable & (relative < 0.0)] = RIGHT_SIGN

    frustum_side = np.zeros((len(head_command),), dtype=np.float64)
    frustum_side[left_frustum & ~right_frustum] = LEFT_SIGN
    frustum_side[right_frustum & ~left_frustum] = RIGHT_SIGN
    value_side[value_side == 0.0] = frustum_side[value_side == 0.0]

    evaluable = (look_direction != 0.0) & (value_side != 0.0)
    left_value = value_side == LEFT_SIGN
    right_value = value_side == RIGHT_SIGN
    return {
        "information_value_frames": int((value_side != 0.0).sum()),
        "look_information_value_agreement": (
            float((look_direction[evaluable] == value_side[evaluable]).mean())
            if evaluable.any()
            else None
        ),
        "left_value_left_look_rate": _rate(look_direction == LEFT_SIGN, left_value),
        "right_value_right_look_rate": _rate(look_direction == RIGHT_SIGN, right_value),
        "opposite_value_look_rate": (
            float((look_direction[evaluable] == -value_side[evaluable]).mean())
            if evaluable.any()
            else None
        ),
    }


def _route_change_metrics(
    *,
    look_mask: np.ndarray,
    body_heading: np.ndarray,
    body_command: np.ndarray,
    sim_time: np.ndarray,
    control_hz: float | None,
    parameters: AnalysisParameters,
) -> Mapping[str, Any]:
    pre_frames = _frames_for_seconds(sim_time, parameters.route_pre_window_seconds, control_hz)
    post_frames = _frames_for_seconds(sim_time, parameters.route_post_window_seconds, control_hz)
    look_onsets = _rising_edges(look_mask)
    changes = []
    heading_changes = []
    for onset in look_onsets:
        pre_start = max(0, int(onset) - pre_frames)
        pre_end = int(onset)
        post_start = int(onset) + 1
        post_end = min(len(look_mask), post_start + post_frames)
        if pre_start >= pre_end or post_start >= post_end:
            continue
        pre_heading = _circular_mean_degrees(body_heading[pre_start:pre_end])
        post_heading = _circular_mean_degrees(body_heading[post_start:post_end])
        if pre_heading is not None and post_heading is not None:
            change = abs(float(_wrap_degrees(post_heading - pre_heading)))
            heading_changes.append(change)
            changes.append(change >= parameters.route_change_degrees)
            continue
        pre_body = float(np.nanmean(body_command[pre_start:pre_end]))
        post_body = float(np.nanmean(body_command[post_start:post_end]))
        changes.append(abs(post_body - pre_body) > parameters.body_threshold)
    return {
        "active_look_bouts": int(len(look_onsets)),
        "route_change_probability_after_look": _mean_or_none(changes),
        "mean_route_heading_change_after_look_degrees": _mean_or_none(heading_changes),
    }


def _risk_bin_rows(
    *,
    bins: np.ndarray,
    look_mask: np.ndarray,
    forward_command: np.ndarray,
    physical_speed: np.ndarray,
    head_command: np.ndarray,
    minimum_distance: np.ndarray,
    capture_event: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for risk_bin in RISK_BINS:
        mask = bins == risk_bin
        if not mask.any():
            continue
        rows.append(
            {
                "risk_bin": risk_bin,
                "frames": int(mask.sum()),
                "look_rate": _rate(look_mask, mask),
                "mean_forward_command": _mean_or_none(forward_command[mask]),
                "mean_physical_speed": _mean_or_none(physical_speed[mask]),
                "mean_absolute_head_command": _mean_or_none(np.abs(head_command[mask])),
                "mean_minimum_distance": _mean_or_none(minimum_distance[mask]),
                "capture_event_rate": _rate(capture_event, mask),
            },
        )
    return rows


def _episode_metrics(
    data: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    episode_metadata: Mapping[str, Any],
    geometry: WorldGeometry | None,
    parameters: AnalysisParameters,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    action = np.asarray(data["action"], dtype=np.float64)
    length = int(len(action))
    sim_time = np.asarray(
        data.get("sim_time", np.arange(length, dtype=np.float32) / float(metadata.get("control_hz", 10.0))),
        dtype=np.float64,
    )
    if action.ndim != 2 or action.shape[1] < 2:
        raise ValueError("action must have shape (T, 2) or (T, 3)")
    if len(sim_time) != length:
        raise ValueError("sim_time is not transition-aligned")

    control_hz = float(metadata.get("control_hz", 0.0) or 0.0)
    forward_command = action[:, 0]
    body_command = action[:, 1]
    head_command = action[:, 2] if action.shape[1] >= 3 else np.zeros((length,), dtype=np.float64)
    look_mask = np.abs(head_command) > parameters.look_threshold

    privileged = np.asarray(data.get("privileged_state", np.zeros((length, 0), dtype=np.float32)))
    names = list(metadata.get("privileged_state_names", []))
    positions = _positions_from_state(privileged, names)
    predator_positions = np.stack(
        (
            _state_column(privileged, names, "predator_x"),
            _state_column(privileged, names, "predator_y"),
        ),
        axis=1,
    )
    body_heading = _state_column(privileged, names, "body_heading_degrees")
    if not np.isfinite(body_heading).any():
        body_heading = np.full((length,), math.nan, dtype=np.float64)
    physical_speed = np.linalg.norm(
        np.stack(
            (
                _state_column(privileged, names, "prey_vx", default=math.nan),
                _state_column(privileged, names, "prey_vy", default=math.nan),
            ),
            axis=1,
        ),
        axis=1,
    )
    minimum_distance = _minimum_distance(data, positions, predator_positions)

    visible = np.asarray(data.get("predator_pixels_visible", np.zeros((length,), dtype=bool)), dtype=bool)
    geometric = np.asarray(data.get("predator_geometric_los", np.zeros((length,), dtype=bool)), dtype=bool)
    within_detection = np.asarray(
        data.get("predator_within_detection_range", np.zeros((length,), dtype=bool)),
        dtype=bool,
    )
    left_frustum = np.asarray(data.get("predator_in_left_frustum", np.zeros((length,), dtype=bool)), dtype=bool)
    right_frustum = np.asarray(data.get("predator_in_right_frustum", np.zeros((length,), dtype=bool)), dtype=bool)
    capture_event = np.asarray(data.get("capture_event", np.zeros((length,), dtype=bool)), dtype=bool)
    capture_count_array = np.asarray(data.get("capture_count", np.zeros((length,), dtype=np.int32)))
    capture_count = int(capture_count_array[-1]) if len(capture_count_array) else int(capture_event.sum())
    success = bool(episode_metadata.get("is_success", False))

    if geometry is not None:
        occlusion_distance = geometry.distance_to_occlusions(positions)
    else:
        occlusion_distance = np.full((length,), math.nan, dtype=np.float64)
    risk_mask = _risk_context(
        visible=visible,
        geometric=geometric,
        within_detection=within_detection,
        minimum_distance=minimum_distance,
        parameters=parameters,
    )
    bins = _risk_bins(
        visible=visible,
        geometric=geometric,
        minimum_distance=minimum_distance,
        parameters=parameters,
    )
    risk_rows = _risk_bin_rows(
        bins=bins,
        look_mask=look_mask,
        forward_command=forward_command,
        physical_speed=physical_speed,
        head_command=head_command,
        minimum_distance=minimum_distance,
        capture_event=capture_event,
    )
    risk_by_bin = {row["risk_bin"]: row for row in risk_rows}
    first_look = int(np.flatnonzero(look_mask)[0]) if look_mask.any() else None
    trajectory = _trajectory_metrics(positions, metadata, success)
    path_efficiency = trajectory["path_efficiency"]
    safety_efficiency = (
        float(path_efficiency) * float(capture_count == 0)
        if path_efficiency is not None
        else None
    )
    no_predator_context = ~risk_mask

    metrics: dict[str, Any] = {
        "steps": length,
        "duration_seconds": float(sim_time[-1] - sim_time[0]) if length > 1 else 0.0,
        "return": float(np.asarray(data.get("reward", np.zeros((length,), dtype=np.float32))).sum()),
        "success": success,
        "capture_count": capture_count,
        "capture_events": int(capture_event.sum()),
        "look_frequency": _rate(look_mask),
        "risk_context_look_rate": _rate(look_mask, risk_mask),
        "no_predator_context_look_rate": _rate(look_mask, no_predator_context),
        "unnecessary_look_rate": (
            float((look_mask & no_predator_context).sum() / max(int(look_mask.sum()), 1))
            if length
            else None
        ),
        "look_suppression_when_no_predator": (
            None
            if _rate(look_mask, risk_mask) is None or _rate(look_mask, no_predator_context) is None
            else float(_rate(look_mask, risk_mask) - _rate(look_mask, no_predator_context))
        ),
        "first_active_look_time_seconds": (
            float(sim_time[first_look] - sim_time[0]) if first_look is not None else None
        ),
        "first_active_look_distance_to_occlusion": (
            float(occlusion_distance[first_look])
            if first_look is not None and math.isfinite(float(occlusion_distance[first_look]))
            else None
        ),
        "first_active_look_distance_to_predator": (
            float(minimum_distance[first_look])
            if first_look is not None and math.isfinite(float(minimum_distance[first_look]))
            else None
        ),
        **_danger_deceleration_metrics(
            risk_mask=risk_mask,
            occlusion_distance=occlusion_distance,
            forward_command=forward_command,
            physical_speed=physical_speed,
            sim_time=sim_time,
            control_hz=control_hz,
            parameters=parameters,
        ),
        **_head_lead_metrics(
            head_command=head_command,
            body_command=body_command,
            sim_time=sim_time,
            control_hz=control_hz,
            parameters=parameters,
        ),
        **_reconfirmation_metrics(
            visible=visible,
            look_mask=look_mask,
            sim_time=sim_time,
            control_hz=control_hz,
            parameters=parameters,
        ),
        **_information_value_metrics(
            positions=positions,
            predator_positions=predator_positions,
            body_heading=body_heading,
            left_frustum=left_frustum,
            right_frustum=right_frustum,
            risk_mask=risk_mask,
            minimum_distance=minimum_distance,
            head_command=head_command,
            parameters=parameters,
        ),
        **_route_change_metrics(
            look_mask=look_mask,
            body_heading=body_heading,
            body_command=body_command,
            sim_time=sim_time,
            control_hz=control_hz,
            parameters=parameters,
        ),
        **trajectory,
        "safety_efficiency": safety_efficiency,
        "low_risk_look_rate": risk_by_bin.get("low", {}).get("look_rate"),
        "medium_risk_look_rate": risk_by_bin.get("medium", {}).get("look_rate"),
        "high_risk_look_rate": risk_by_bin.get("high", {}).get("look_rate"),
        "low_risk_mean_forward_command": risk_by_bin.get("low", {}).get("mean_forward_command"),
        "high_risk_mean_forward_command": risk_by_bin.get("high", {}).get("mean_forward_command"),
        "high_risk_forward_suppression": (
            None
            if risk_by_bin.get("low", {}).get("mean_forward_command") is None
            or risk_by_bin.get("high", {}).get("mean_forward_command") is None
            else float(
                risk_by_bin["low"]["mean_forward_command"]
                - risk_by_bin["high"]["mean_forward_command"],
            )
        ),
        "high_risk_look_increase": (
            None
            if risk_by_bin.get("low", {}).get("look_rate") is None
            or risk_by_bin.get("high", {}).get("look_rate") is None
            else float(risk_by_bin["high"]["look_rate"] - risk_by_bin["low"]["look_rate"])
        ),
        "high_risk_mean_minimum_distance": risk_by_bin.get("high", {}).get("mean_minimum_distance"),
    }
    return metrics, risk_rows


def _validate_minimum_arrays(
    data: Mapping[str, np.ndarray],
    *,
    session: str,
    episode: str,
) -> tuple[int, list[Exp02Issue]]:
    issues = []
    for name in ("action", "privileged_state"):
        if name not in data:
            issues.append(Exp02Issue("error", session, episode, f"Missing array: {name}"))
    if issues:
        return 0, issues
    length = int(len(data["action"]))
    if length <= 0:
        issues.append(Exp02Issue("error", session, episode, "Episode has zero transitions"))
    for name, array in data.items():
        if name in {"image_left", "image_right"}:
            continue
        if hasattr(array, "__len__") and len(array) not in (length,):
            if name not in {"state_offsets", "action_offsets"}:
                issues.append(
                    Exp02Issue(
                        "warning",
                        session,
                        episode,
                        f"{name} length {len(array)} != action length {length}",
                    ),
                )
    return length, issues


def _merge_episode_metadata(
    session_dir: Path,
    episode_path: Path,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    by_file = {
        str(item.get("file")): item
        for item in metadata.get("episodes", [])
        if isinstance(item, Mapping)
    }
    episode_metadata: dict[str, Any] = dict(by_file.get(episode_path.name, {}))
    sidecar_path = episode_path.with_suffix(".json")
    if sidecar_path.exists():
        episode_metadata.update(_safe_json(sidecar_path))
    return episode_metadata


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        aggregate = {field: value for field, value in zip(group_fields, key)}
        aggregate.update(
            {
                "episodes": len(subset),
                "success_rate": _mean_or_none(row["success"] for row in subset),
                "capture_episode_rate": _mean_or_none(row["capture_count"] > 0 for row in subset),
                "mean_look_frequency": _mean_or_none(row.get("look_frequency") for row in subset),
                "mean_first_active_look_distance_to_occlusion": _mean_or_none(
                    row.get("first_active_look_distance_to_occlusion") for row in subset
                ),
                "mean_pre_danger_deceleration": _mean_or_none(
                    row.get("pre_danger_deceleration") for row in subset
                ),
                "mean_head_turn_before_body_fraction": _mean_or_none(
                    row.get("head_turn_before_body_fraction") for row in subset
                ),
                "mean_reconfirm_action_latency_after_loss": _mean_or_none(
                    row.get("reconfirm_action_latency_after_loss") for row in subset
                ),
                "mean_look_information_value_agreement": _mean_or_none(
                    row.get("look_information_value_agreement") for row in subset
                ),
                "mean_route_change_probability_after_look": _mean_or_none(
                    row.get("route_change_probability_after_look") for row in subset
                ),
                "mean_no_predator_context_look_rate": _mean_or_none(
                    row.get("no_predator_context_look_rate") for row in subset
                ),
                "mean_high_risk_forward_suppression": _mean_or_none(
                    row.get("high_risk_forward_suppression") for row in subset
                ),
                "mean_safety_efficiency": _mean_or_none(row.get("safety_efficiency") for row in subset),
                "total_steps": int(sum(int(row.get("steps", 0)) for row in subset)),
            },
        )
        output.append(aggregate)
    return output


def _metric_definitions() -> Mapping[str, str]:
    return {
        "first_active_look_distance_to_occlusion": (
            "Distance from prey position to nearest Cellworld occlusion polygon "
            "at the first frame where abs(head_yaw_rate) exceeds the look threshold."
        ),
        "pre_danger_deceleration": (
            "Mean forward command in the baseline window minus the approach "
            "window before a rising risk-context event near an occlusion."
        ),
        "head_turn_before_body_fraction": (
            "Fraction of body-yaw onsets preceded by a same-direction head-yaw "
            "onset inside the configured lead window."
        ),
        "reconfirm_action_latency_after_loss": (
            "Time from predator_pixels_visible falling edge to the next active "
            "head-yaw command inside the reconfirmation window."
        ),
        "look_information_value_agreement": (
            "Among active left/right look frames with a directional value proxy, "
            "fraction where look sign matches the side of the predator bearing "
            "or, if bearing is unavailable, asymmetric left/right frustum label."
        ),
        "route_change_probability_after_look": (
            "Probability that a look bout is followed by a body-heading change "
            "larger than the route-change threshold in the post-look window."
        ),
        "no_predator_context_look_rate": (
            "Active look rate when no pixel/geometric/range/distance risk proxy "
            "is present; this is an unnecessary-observation proxy."
        ),
        "high_risk_forward_suppression": (
            "Low-risk mean forward command minus high-risk mean forward command; "
            "positive values indicate slowing under higher risk."
        ),
        "safety_efficiency": (
            "Successful-path efficiency multiplied by an indicator of zero captures; "
            "a descriptive composite, not a validated endpoint."
        ),
    }


def _write_ethics_note(path: Path) -> None:
    text = """# EXP-02 Human Data Governance Note

This file is a project reminder, not an ethics approval.

Before using multi-participant human active-looking data in a paper or public
claim, follow the institution's process for informed consent, data management,
privacy protection, and any required IRB or ethics review.  Generated audits,
unit tests, smoke runs, or descriptive fingerprints do not establish that the
dataset is approved for publication.

The analysis split manifest is grouped by participant/session/world and never
by adjacent video frame.  Policy-training or confirmatory modeling should use
the manifest, keep privileged simulator fields out of model inputs, and report
any departures from the pre-registered split policy.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def run_exp02_human_active_looking(
    data_root: Path,
    output_dir: Path,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    parameters: AnalysisParameters = AnalysisParameters(),
) -> Mapping[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_files = sorted(data_root.glob("**/session.json")) if data_root.exists() else []
    if not session_files:
        summary = {
            "status": "no_data",
            "message": f"No human demonstration sessions found under {data_root}",
            "sessions": 0,
            "episodes": 0,
            "errors": 0,
            "warnings": 0,
            "split_unit": "participant/session/world; never frame",
        }
        write_json(output_dir / "exp02_summary.json", summary)
        write_csv(output_dir / "episode_fingerprint.csv", [])
        write_csv(output_dir / "session_summary.csv", [])
        write_csv(output_dir / "participant_summary.csv", [])
        write_csv(output_dir / "risk_bin_summary.csv", [])
        write_csv(output_dir / "split_manifest.csv", [])
        write_json(output_dir / "metric_definitions.json", _metric_definitions())
        write_jsonl(output_dir / "exp02_issues.jsonl", [])
        _write_ethics_note(output_dir / "ETHICS_NOTE.md")
        print(summary["message"])
        return summary

    issues: list[Exp02Issue] = []
    episode_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []
    geometry_cache: dict[str, WorldGeometry | None] = {}

    for session_path in session_files:
        session_dir = session_path.parent
        session_name = session_dir.name
        try:
            metadata = _safe_json(session_path)
        except Exception as error:
            issues.append(Exp02Issue("error", session_name, None, f"Invalid session.json: {error}"))
            continue
        participant = str(metadata.get("participant_id", metadata.get("participant", "unknown")))
        world_name = str(metadata.get("world_name", "unknown_world"))
        if participant == "unknown":
            issues.append(
                Exp02Issue(
                    "warning",
                    session_name,
                    None,
                    "No participant_id; split key falls back to unknown/session/world.",
                ),
            )
        group_key = f"{participant}/{session_name}/{world_name}"
        split = _split_for_group(group_key, parameters.split_seed)
        split_manifest.append(
            {
                "participant_id": participant,
                "session": session_name,
                "world_name": world_name,
                "group_key": group_key,
                "split": split,
                "split_unit": "participant/session/world; never frame",
            },
        )
        if world_name not in geometry_cache:
            try:
                geometry_cache[world_name] = load_world_geometry(world_name, cache_root=cache_root)
            except Exception as error:
                geometry_cache[world_name] = None
                issues.append(
                    Exp02Issue(
                        "warning",
                        session_name,
                        None,
                        f"Could not load world geometry for {world_name}: {error}",
                    ),
                )
        geometry = geometry_cache[world_name]
        if geometry is None:
            issues.append(
                Exp02Issue(
                    "warning",
                    session_name,
                    None,
                    f"No occlusion geometry available for world {world_name}; occlusion-distance metrics are blank.",
                ),
            )

        for episode_path in _episode_files(session_dir, metadata):
            episode_name = episode_path.name
            if not episode_path.exists():
                issues.append(Exp02Issue("error", session_name, episode_name, "Listed NPZ does not exist"))
                continue
            try:
                with np.load(episode_path, allow_pickle=False) as archive:
                    data = {name: np.array(archive[name], copy=True) for name in archive.files}
            except Exception as error:
                issues.append(Exp02Issue("error", session_name, episode_name, f"Cannot read NPZ safely: {error}"))
                continue
            length, validation_issues = _validate_minimum_arrays(
                data,
                session=session_name,
                episode=episode_name,
            )
            issues.extend(validation_issues)
            if length <= 0 or any(issue.severity == "error" for issue in validation_issues):
                continue
            try:
                episode_metadata = _merge_episode_metadata(session_dir, episode_path, metadata)
                metrics, episode_risk_rows = _episode_metrics(
                    data,
                    metadata=metadata,
                    episode_metadata=episode_metadata,
                    geometry=geometry,
                    parameters=parameters,
                )
            except Exception as error:
                issues.append(Exp02Issue("error", session_name, episode_name, f"Metric extraction failed: {error}"))
                continue
            row = {
                "participant_id": participant,
                "session": session_name,
                "world_name": world_name,
                "episode": episode_name,
                "split": split,
                **metrics,
            }
            episode_rows.append(row)
            for risk_row in episode_risk_rows:
                risk_rows.append(
                    {
                        "participant_id": participant,
                        "session": session_name,
                        "world_name": world_name,
                        "episode": episode_name,
                        "split": split,
                        **risk_row,
                    },
                )

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    split_counts = {
        split: sum(row["split"] == split for row in split_manifest)
        for split in ("train", "validation", "test")
    }
    summary = {
        "status": "ok" if error_count == 0 else "validation_errors",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "sessions": len(session_files),
        "episodes": len(episode_rows),
        "participants": sorted({row["participant_id"] for row in split_manifest}),
        "worlds": sorted({row["world_name"] for row in split_manifest}),
        "split_counts": split_counts,
        "split_unit": "participant/session/world; never frame",
        "parameters": asdict(parameters),
        "errors": error_count,
        "warnings": warning_count,
        "aggregate": {
            "success_rate": _mean_or_none(row.get("success") for row in episode_rows),
            "capture_episode_rate": _mean_or_none(row.get("capture_count", 0) > 0 for row in episode_rows),
            "look_frequency": _mean_or_none(row.get("look_frequency") for row in episode_rows),
            "first_active_look_distance_to_occlusion": _mean_or_none(
                row.get("first_active_look_distance_to_occlusion") for row in episode_rows
            ),
            "pre_danger_deceleration": _mean_or_none(row.get("pre_danger_deceleration") for row in episode_rows),
            "head_turn_before_body_fraction": _mean_or_none(
                row.get("head_turn_before_body_fraction") for row in episode_rows
            ),
            "reconfirm_action_latency_after_loss": _mean_or_none(
                row.get("reconfirm_action_latency_after_loss") for row in episode_rows
            ),
            "look_information_value_agreement": _mean_or_none(
                row.get("look_information_value_agreement") for row in episode_rows
            ),
            "route_change_probability_after_look": _mean_or_none(
                row.get("route_change_probability_after_look") for row in episode_rows
            ),
            "no_predator_context_look_rate": _mean_or_none(
                row.get("no_predator_context_look_rate") for row in episode_rows
            ),
            "high_risk_forward_suppression": _mean_or_none(
                row.get("high_risk_forward_suppression") for row in episode_rows
            ),
            "safety_efficiency": _mean_or_none(row.get("safety_efficiency") for row in episode_rows),
        },
        "ethics_note": (
            "Before using multi-participant data for a paper, follow the "
            "institutional process for informed consent, data management, and "
            "any required ethics or IRB review."
        ),
        "metric_definitions_path": "metric_definitions.json",
    }
    write_csv(output_dir / "episode_fingerprint.csv", episode_rows)
    write_csv(output_dir / "risk_bin_summary.csv", risk_rows)
    write_csv(
        output_dir / "session_summary.csv",
        _aggregate_rows(episode_rows, group_fields=("participant_id", "session", "world_name", "split")),
    )
    write_csv(
        output_dir / "participant_summary.csv",
        _aggregate_rows(episode_rows, group_fields=("participant_id", "split")),
    )
    write_csv(output_dir / "split_manifest.csv", split_manifest)
    write_jsonl(output_dir / "exp02_issues.jsonl", [asdict(issue) for issue in issues])
    write_json(output_dir / "metric_definitions.json", _metric_definitions())
    write_json(output_dir / "exp02_summary.json", summary)
    _write_ethics_note(output_dir / "ETHICS_NOTE.md")
    print(
        f"EXP-02 human active-looking fingerprint: status={summary['status']} "
        f"sessions={summary['sessions']} episodes={summary['episodes']} "
        f"errors={error_count} warnings={warning_count}",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AnalysisParameters()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--split-seed", type=int, default=defaults.split_seed)
    parser.add_argument("--look-threshold", type=float, default=defaults.look_threshold)
    parser.add_argument("--body-threshold", type=float, default=defaults.body_threshold)
    parser.add_argument("--risk-high-distance", type=float, default=defaults.risk_high_distance)
    parser.add_argument("--risk-medium-distance", type=float, default=defaults.risk_medium_distance)
    parser.add_argument(
        "--junction-occlusion-distance",
        type=float,
        default=defaults.junction_occlusion_distance,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    parameters = AnalysisParameters(
        split_seed=args.split_seed,
        look_threshold=args.look_threshold,
        body_threshold=args.body_threshold,
        risk_high_distance=args.risk_high_distance,
        risk_medium_distance=args.risk_medium_distance,
        junction_occlusion_distance=args.junction_occlusion_distance,
    )
    run_exp02_human_active_looking(
        args.data_root,
        args.output_dir,
        cache_root=args.cache_root,
        parameters=parameters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
