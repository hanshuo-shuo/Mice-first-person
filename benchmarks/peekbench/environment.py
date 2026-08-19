"""BotEvade environment and exact-state helpers for PeekBench."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import cellworld.util as cellworld_util

from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv
from policies.base import hash_array
from reward import custom_reward


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_env(config: Mapping[str, Any], *, use_predator: bool) -> FirstPersonBotEvadeEnv:
    # Cellworld reads CELLWORLD_CACHE only on its first import.  Other callers
    # and tests may temporarily change the module global, so reassert the
    # repository's offline cache at this construction boundary.
    cellworld_util.cellworld_cache_folder = str(PROJECT_ROOT / "cellworld_cache")
    values = config["environment"]
    return FirstPersonBotEvadeEnv(
        world_name=str(values["world_name"]),
        use_lppos=False,
        use_predator=bool(use_predator),
        max_step=int(values["max_step"]),
        reward_function=custom_reward,
        time_step=float(values["time_step"]),
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.CONTINUOUS,
        frame_stack_k=1,
        predator_prey_forward_speed_ratio=float(
            values["predator_prey_forward_speed_ratio"],
        ),
        vision_width=int(values["vision_width"]),
        vision_height=int(values["vision_height"]),
        vision_fov=float(values["vision_fov"]),
        vision_far_clip=float(values["vision_far_clip"]),
        vision_detection_range=float(values["vision_detection_range"]),
        observation_mode="mouse",
        action_mode="egocentric_velocity_head",
        render_mode="rgb_array",
    )


def observe_current(env: FirstPersonBotEvadeEnv) -> dict[str, np.ndarray]:
    """Render the current public observation without advancing simulation."""

    observation = env._vision_observation()
    if not isinstance(observation, dict) or not env.observation_space.contains(observation):
        raise RuntimeError("Current first-person observation violates its Gym space")
    return {name: np.array(value, copy=True) for name, value in observation.items()}


def observation_hashes(observation: Mapping[str, np.ndarray]) -> Mapping[str, str]:
    return {name: hash_array(np.asarray(value)) for name, value in observation.items()}


def state_with_gaze(state: Mapping[str, Any], gaze_degrees: float) -> dict[str, Any]:
    branch = copy.deepcopy(dict(state))
    first_person = branch.get("first_person")
    if not isinstance(first_person, dict):
        raise KeyError("Snapshot has no first_person embodiment state")
    first_person["head_yaw_degrees"] = float(gaze_degrees)
    first_person["last_body_turn_command"] = 0.0
    return branch


def restore_and_observe(
    env: FirstPersonBotEvadeEnv,
    state: Mapping[str, Any],
    *,
    gaze_degrees: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    branch = copy.deepcopy(dict(state))
    if gaze_degrees is not None:
        branch = state_with_gaze(branch, gaze_degrees)
    env.set_state_dict(branch)
    observation = observe_current(env)
    return observation, env.get_state_dict()


def privileged_label(
    env: FirstPersonBotEvadeEnv,
    *,
    recent_visibility: Sequence[bool] = (),
) -> dict[str, Any]:
    model = env.unwrapped.model
    prey = model.prey
    prey_velocity = tuple(float(value) for value in prey.state.velocity)
    visibility = env.get_predator_visibility()
    label: dict[str, Any] = {
        "use_predator": bool(model.use_predator),
        "prey_location": [float(value) for value in prey.state.location],
        "prey_velocity": list(prey_velocity),
        "body_heading_degrees": float(env.body_heading_degrees),
        "head_yaw_degrees": float(env.head_yaw_degrees),
        "recent_visibility": [bool(value) for value in recent_visibility],
        **{key: bool(value) for key, value in visibility.items()},
    }
    if model.use_predator:
        predator = model.predator
        predator_location = np.asarray(predator.state.location, dtype=np.float64)
        prey_location = np.asarray(prey.state.location, dtype=np.float64)
        label.update(
            {
                "predator_location": [float(value) for value in predator_location],
                "predator_velocity": [float(value) for value in predator.state.velocity],
                "prey_predator_distance": float(
                    np.linalg.norm(predator_location - prey_location),
                ),
            },
        )
    else:
        label.update(
            {
                "predator_location": None,
                "predator_velocity": None,
                "prey_predator_distance": None,
            },
        )
    return label


def classify_state(
    label: Mapping[str, Any],
    *,
    recent_visibility_horizon: int,
) -> str:
    if not bool(label["use_predator"]):
        return "no_predator_control"
    visible = bool(label["predator_pixels_visible"])
    history = list(label.get("recent_visibility", []))[-recent_visibility_horizon:]
    if not visible and any(bool(value) for value in history):
        return "recently_visible_hidden"
    if visible:
        return "predator_visible"
    in_frustum = bool(
        label["predator_in_left_frustum"] or label["predator_in_right_frustum"],
    )
    if bool(label["predator_geometric_los"]) and not in_frustum:
        return "geometric_outside_frustum"
    if in_frustum and not visible:
        return "frustum_pixel_occluded"
    return "unclassified_hidden"


def _deduplicate_anchors(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for record in records:
        key = (record["source"], int(record["cell_id"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(record))
    return result


def sampling_anchors(env: FirstPersonBotEvadeEnv) -> Mapping[str, list[dict[str, Any]]]:
    base_env = env.unwrapped
    locations = base_env.loader.locations
    open_cells = [
        (cell_id, np.asarray(location, dtype=np.float64))
        for cell_id, location in enumerate(locations)
        if location is not None
    ]

    near_occlusion = [
        {
            "source": "near_occlusion",
            "cell_id": int(cell_id),
            "location": [float(value) for value in locations[int(cell_id)]],
        }
        for cell_id in sorted(int(value) for value in base_env.cell_ids_near_occlusion)
        if 0 <= int(cell_id) < len(locations) and locations[int(cell_id)] is not None
    ]

    point_array = np.stack([location for _, location in open_cells])
    pairwise = np.linalg.norm(point_array[:, None, :] - point_array[None, :, :], axis=-1)
    positive = pairwise[pairwise > 1e-8]
    neighbor_radius = float(positive.min() * 1.05)
    junction = []
    for index, (cell_id, location) in enumerate(open_cells):
        degree = int(np.count_nonzero((pairwise[index] > 1e-8) & (pairwise[index] <= neighbor_radius)))
        if degree >= 3:
            junction.append(
                {
                    "source": "junction",
                    "cell_id": int(cell_id),
                    "location": [float(value) for value in location],
                    "neighbor_degree": degree,
                },
            )

    peek_records = []
    peek_path = PROJECT_ROOT / "data" / "2105_peek_locations.csv"
    with peek_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            point = np.asarray((float(row["x"]), float(row["y"])), dtype=np.float64)
            distances = np.linalg.norm(point_array - point[None, :], axis=1)
            nearest = int(np.argmin(distances))
            cell_id, location = open_cells[nearest]
            peek_records.append(
                {
                    "source": "peek_location",
                    "cell_id": int(cell_id),
                    "location": [float(value) for value in location],
                    "source_point": [float(value) for value in point],
                    "source_distance": float(distances[nearest]),
                },
            )

    return {
        "near_occlusion": _deduplicate_anchors(near_occlusion),
        "junction": _deduplicate_anchors(junction),
        "peek_location": _deduplicate_anchors(peek_records),
    }


def bearing_degrees(origin: Sequence[float], target: Sequence[float]) -> float:
    delta_x = float(target[0]) - float(origin[0])
    delta_y = float(target[1]) - float(origin[1])
    return math.degrees(math.atan2(delta_y, delta_x))
