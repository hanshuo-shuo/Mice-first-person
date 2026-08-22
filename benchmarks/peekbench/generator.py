"""Deterministic snapshot generation for PeekBench P0."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import (
    canonical_typed_bytes,
    prepare_experiment,
    save_observation,
    save_state,
    state_digest,
    write_csv,
    write_json,
    write_jsonl,
)
from .environment import (
    PROJECT_ROOT,
    bearing_degrees,
    classify_state,
    make_env,
    observation_hashes,
    observe_current,
    privileged_label,
    restore_and_observe,
    sampling_anchors,
)


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _constructed_state(
    base_state: Mapping[str, Any],
    *,
    prey_location: Sequence[float],
    body_heading_degrees: float,
    predator_location: Sequence[float] | None,
) -> dict[str, Any]:
    state = copy.deepcopy(dict(base_state))
    prey = state["model"]["agents"]["prey"]
    prey["state"]["location"] = tuple(float(value) for value in prey_location)
    prey["state"]["velocity"] = (0.0, 0.0)
    prey["state"]["body_heading"] = _wrap_degrees(body_heading_degrees)
    prey["collision"] = False
    first_person = state["first_person"]
    first_person["body_heading_degrees"] = _wrap_degrees(body_heading_degrees)
    first_person["head_yaw_degrees"] = 0.0
    first_person["last_body_turn_command"] = 0.0
    first_person["previous_action"] = np.zeros((3,), dtype=np.float32)
    environment = state["environment"]
    environment["previous_action"] = np.zeros((2,), dtype=np.float32)
    environment["prey_visible_last_step"] = 0
    environment["predator_visible_last_step"] = 0
    if predator_location is not None:
        predator = state["model"]["agents"]["predator"]
        predator["state"]["location"] = tuple(float(value) for value in predator_location)
        predator["state"]["velocity"] = (0.0, 0.0)
        predator["state"]["body_heading"] = _wrap_degrees(
            bearing_degrees(predator_location, prey_location),
        )
        predator["dynamics"]["forward_speed"] = 0
        predator["dynamics"]["turn_speed"] = 0
        predator["new_destination"] = None
        predator["destination"] = None
        predator["path"] = []
        predator["destination_wait"] = 0
        predator["navigation_plan_update_wait"] = 0
        predator["collision"] = False
    return state


def _settle_constructed_state(
    env,
    state: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    """Advance one legal zero-motion transition to refresh task bookkeeping."""

    env.set_state_dict(copy.deepcopy(dict(state)))
    zero_action = np.zeros((3,), dtype=np.float32)
    observation, _, terminated, truncated, _ = env.step(zero_action)
    if not isinstance(observation, dict):
        raise RuntimeError("PeekBench requires the public mouse observation")
    return (
        {name: np.array(value, copy=True) for name, value in observation.items()},
        env.get_state_dict(),
        bool(terminated or truncated),
    )


def _candidate_predator_locations(
    env,
    anchor: Sequence[float],
    sampling_config: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray, float, bool]]:
    minimum = float(sampling_config["minimum_predator_distance"])
    maximum = float(sampling_config["maximum_predator_distance"])
    candidates = []
    visibility = env.unwrapped.model.visibility
    anchor_array = np.asarray(anchor, dtype=np.float64)
    for cell_id, location in enumerate(env.unwrapped.loader.locations):
        if location is None:
            continue
        location_array = np.asarray(location, dtype=np.float64)
        distance = float(np.linalg.norm(location_array - anchor_array))
        if not minimum <= distance <= maximum:
            continue
        geometric_los = bool(visibility.line_of_sight(anchor_array, location_array))
        candidates.append((cell_id, location_array, distance, geometric_los))
    permutation = rng.permutation(len(candidates)) if candidates else []
    randomized = [candidates[int(index)] for index in permutation]
    randomized.sort(key=lambda item: (abs(item[2] - 0.34), item[0]))
    return randomized[: int(sampling_config["candidate_search_limit"])]


def _construct_no_predator(
    env,
    *,
    seed: int,
    anchor: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    base_state = env.get_state_dict()
    body_heading = float((seed * 37) % 360)
    state = _constructed_state(
        base_state,
        prey_location=anchor["location"],
        body_heading_degrees=body_heading,
        predator_location=None,
    )
    observation, state, ended = _settle_constructed_state(env, state)
    if ended:
        raise RuntimeError("No-predator control ended during state construction")
    label = privileged_label(env)
    category = classify_state(
        label,
        recent_visibility_horizon=int(
            config["sampling"]["recent_visibility_horizon"],
        ),
    )
    return {
        "state": state,
        "observation": observation,
        "history": [],
        "privileged_label": label,
        "category": category,
        "construction_success": category == "no_predator_control",
        "construction_failure": None if category == "no_predator_control" else category,
    }


def _construct_predator_category(
    env,
    *,
    seed: int,
    anchor: Mapping[str, Any],
    target_category: str,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    base_state = env.get_state_dict()
    candidates = _candidate_predator_locations(
        env,
        anchor["location"],
        config["sampling"],
        rng,
    )
    desired_geometric = target_category != "frustum_pixel_occluded"
    if target_category == "frustum_pixel_occluded":
        ordered = [candidate for candidate in candidates if not candidate[3]]
    else:
        ordered = [candidate for candidate in candidates if candidate[3] == desired_geometric]
    if not ordered:
        ordered = candidates

    first_result: Mapping[str, Any] | None = None
    for _, predator_location, _, _ in ordered:
        toward = bearing_degrees(anchor["location"], predator_location)
        body_heading = (
            toward - 135.0
            if target_category == "geometric_outside_frustum"
            else toward
        )
        state = _constructed_state(
            base_state,
            prey_location=anchor["location"],
            body_heading_degrees=body_heading,
            predator_location=predator_location,
        )
        observation, settled_state, ended = _settle_constructed_state(env, state)
        if ended:
            continue
        history: list[Mapping[str, np.ndarray]] = []
        recent_visibility: list[bool] = []

        if target_category == "recently_visible_hidden":
            visible_label = privileged_label(env)
            if not visible_label["predator_pixels_visible"]:
                continue
            history = [observation]
            hidden_state = copy.deepcopy(settled_state)
            hidden_heading = _wrap_degrees(env.body_heading_degrees - 135.0)
            hidden_state["model"]["agents"]["prey"]["state"][
                "body_heading"
            ] = hidden_heading
            hidden_state["first_person"]["body_heading_degrees"] = hidden_heading
            env.set_state_dict(hidden_state)
            observation = observe_current(env)
            settled_state = env.get_state_dict()
            recent_visibility = [True]

        label = privileged_label(env, recent_visibility=recent_visibility)
        category = classify_state(
            label,
            recent_visibility_horizon=int(
                config["sampling"]["recent_visibility_horizon"],
            ),
        )
        result = {
            "state": settled_state,
            "observation": observation,
            "history": history,
            "privileged_label": label,
            "category": category,
            "construction_success": category == target_category,
            "construction_failure": None if category == target_category else category,
        }
        if first_result is None:
            first_result = result
        if category == target_category:
            return result

    if first_result is not None:
        return first_result
    raise RuntimeError(
        f"Could not construct any valid predator state near anchor {anchor['cell_id']}",
    )


def _transition_signature(env, state: Mapping[str, Any]) -> bytes:
    env.set_state_dict(copy.deepcopy(dict(state)))
    action = np.asarray((0.15, -0.10, 0.20), dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    signature = {
        "observation_hashes": observation_hashes(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "state_digest": state_digest(env.get_state_dict()),
    }
    return canonical_typed_bytes(signature)


def verify_replay_determinism(env, state: Mapping[str, Any]) -> bool:
    source_digest = state_digest(state)
    first = _transition_signature(env, state)
    second = _transition_signature(env, state)
    if state_digest(state) != source_digest:
        raise AssertionError("Determinism check mutated the source snapshot")
    env.set_state_dict(copy.deepcopy(dict(state)))
    return first == second


def _snapshot_id(
    *,
    config_hash: str,
    seed: int,
    index: int,
    state_hash: str,
) -> str:
    identity = {
        "config_hash": config_hash,
        "seed": int(seed),
        "index": int(index),
        "state_hash": state_hash,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()[:16]
    return f"pb-{index:05d}-{digest}"


def generate_snapshots(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Generate and save deterministic PeekBench snapshot artifacts."""

    experiment_dir = prepare_experiment(config, project_root=project_root)
    predator_env = make_env(config, use_predator=True)
    control_env = make_env(config, use_predator=False)
    records: list[dict[str, Any]] = []
    try:
        predator_env.reset(seed=int(config["seed"]))
        anchors = sampling_anchors(predator_env)
        source_names = list(config["sampling"]["sources"])
        categories = list(config["sampling"]["categories"])
        source_offsets = {source: 0 for source in source_names}
        rng = np.random.default_rng(int(config["seed"]))
        for source in source_names:
            order = rng.permutation(len(anchors[source]))
            anchors[source] = [anchors[source][int(index)] for index in order]

        for index in range(int(config["num_snapshots"])):
            source = source_names[index % len(source_names)]
            source_records = anchors[source]
            if not source_records:
                raise RuntimeError(f"No sampling anchors available for {source}")
            target_category = categories[index % len(categories)]
            sample_seed = int(config["seed"]) + index * 1009
            require_success = bool(
                config["sampling"].get("require_construction_success", False),
            )
            retry_limit = int(
                config["sampling"].get("anchor_retry_limit", 1),
            )
            construction_errors = []
            for _ in range(retry_limit if require_success else 1):
                anchor_index = source_offsets[source] % len(source_records)
                source_offsets[source] += 1
                anchor = source_records[anchor_index]
                sample_rng = np.random.default_rng(sample_seed)
                try:
                    if target_category == "no_predator_control":
                        constructed = _construct_no_predator(
                            control_env,
                            seed=sample_seed,
                            anchor=anchor,
                            config=config,
                        )
                        env = control_env
                    else:
                        constructed = _construct_predator_category(
                            predator_env,
                            seed=sample_seed,
                            anchor=anchor,
                            target_category=target_category,
                            config=config,
                            rng=sample_rng,
                        )
                        env = predator_env
                except RuntimeError as error:
                    if not require_success:
                        raise
                    construction_errors.append(
                        f"cell {anchor['cell_id']}: {error}",
                    )
                    continue
                if require_success and not constructed["construction_success"]:
                    construction_errors.append(
                        f"cell {anchor['cell_id']}: got {constructed['category']}",
                    )
                    continue
                break
            else:
                details = "; ".join(construction_errors[-3:])
                raise RuntimeError(
                    f"Could not construct required category {target_category} "
                    f"after {retry_limit} deterministic anchor attempts for "
                    f"snapshot {index}. Last failures: {details}",
                )

            state = constructed["state"]
            observation = constructed["observation"]
            state_hash = state_digest(state)
            snapshot_id = _snapshot_id(
                config_hash=str(config["config_hash"]),
                seed=int(config["seed"]),
                index=index,
                state_hash=state_hash,
            )
            snapshot_dir = experiment_dir / "snapshots" / snapshot_id
            state_path = snapshot_dir / "state.json.gz"
            observation_path = snapshot_dir / "observation.npz"
            save_state(state_path, state)
            save_observation(observation_path, observation)
            history_paths = []
            for history_index, history_observation in enumerate(constructed["history"]):
                history_path = snapshot_dir / f"history_{history_index:03d}.npz"
                save_observation(history_path, history_observation)
                history_paths.append(str(history_path.relative_to(experiment_dir)))

            replay_deterministic = verify_replay_determinism(env, state)
            record = {
                "snapshot_id": snapshot_id,
                "index": index,
                "seed": sample_seed,
                "config_hash": config["config_hash"],
                "source": source,
                "source_cell_id": int(anchor["cell_id"]),
                "source_location": list(anchor["location"]),
                "target_category": target_category,
                "category": constructed["category"],
                "construction_success": bool(constructed["construction_success"]),
                "construction_failure": constructed["construction_failure"],
                "use_predator": bool(
                    constructed["privileged_label"]["use_predator"],
                ),
                "gaze_candidates_degrees": list(config["gaze_candidates_degrees"]),
                "state_hash": state_hash,
                "state_hash_semantics": (
                    "all state fields except nonsemantic model.last_step wall clock"
                ),
                "observation_hashes": observation_hashes(observation),
                "replay_deterministic": bool(replay_deterministic),
                "state_path": str(state_path.relative_to(experiment_dir)),
                "observation_path": str(observation_path.relative_to(experiment_dir)),
                "history_paths": history_paths,
                "privileged_label": constructed["privileged_label"],
            }
            write_json(snapshot_dir / "metadata.json", record)
            records.append(record)

        write_jsonl(experiment_dir / "snapshots.jsonl", records)
        summary = [
            {
                "snapshot_id": record["snapshot_id"],
                "index": record["index"],
                "seed": record["seed"],
                "source": record["source"],
                "source_cell_id": record["source_cell_id"],
                "target_category": record["target_category"],
                "category": record["category"],
                "construction_success": record["construction_success"],
                "construction_failure": record["construction_failure"],
                "use_predator": record["use_predator"],
                "replay_deterministic": record["replay_deterministic"],
                "state_hash": record["state_hash"],
            }
            for record in records
        ]
        write_csv(experiment_dir / "snapshots.csv", summary)
        return records
    finally:
        predator_env.close()
        control_env.close()
