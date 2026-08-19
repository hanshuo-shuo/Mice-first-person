"""Open-loop and exact-state branch evaluation for PeekBench."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policies.base import PolicyDecision, PolicyInput, PublicHistoryFrame, VisionPolicy
from policies.openrouter_vlm import build_policy

from .artifacts import (
    load_observation,
    load_state,
    prepare_experiment,
    read_jsonl,
    state_digest,
    write_csv,
    write_jsonl,
)
from .environment import PROJECT_ROOT, make_env, restore_and_observe
from .generator import generate_snapshots


def _experiment_dir(config: Mapping[str, Any], project_root: Path) -> Path:
    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root / str(config["experiment_id"])


def _snapshot_records(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    experiment_dir = prepare_experiment(config, project_root=project_root)
    path = experiment_dir / "snapshots.jsonl"
    if not path.exists():
        generate_snapshots(config, project_root=project_root)
    records = read_jsonl(path)
    if not records:
        raise RuntimeError("PeekBench snapshot manifest is empty")
    return experiment_dir, records


def _history_frames(
    experiment_dir: Path,
    record: Mapping[str, Any],
) -> tuple[PublicHistoryFrame, ...]:
    frames = []
    for relative_path in record.get("history_paths", []):
        observation = load_observation(experiment_dir / relative_path)
        frames.append(
            PublicHistoryFrame(
                image_left=observation["image_left"],
                image_right=observation["image_right"],
            ),
        )
    return tuple(frames)


def run_open_loop_evaluation(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    policy: VisionPolicy | None = None,
) -> list[dict[str, Any]]:
    experiment_dir, snapshots = _snapshot_records(config, project_root=project_root)
    selected_policy = policy or build_policy(
        config["policy"],
        experiment_dir=experiment_dir,
    )
    records = []
    for snapshot in snapshots:
        observation = load_observation(experiment_dir / snapshot["observation_path"])
        policy_input = PolicyInput.from_observation(
            observation,
            history=_history_frames(experiment_dir, snapshot),
        )
        result = selected_policy.decide(policy_input)
        records.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "source": snapshot["source"],
                "category": snapshot["category"],
                "decision": result.decision.to_dict(),
                "telemetry": result.telemetry.to_dict(),
            },
        )
    write_jsonl(experiment_dir / "open_loop.jsonl", records)
    summary = [
        {
            "snapshot_id": record["snapshot_id"],
            "source": record["source"],
            "category": record["category"],
            **record["decision"],
            "backend": record["telemetry"]["backend"],
            "model": record["telemetry"]["model"],
            "latency_ms": record["telemetry"]["latency_ms"],
            "cost": record["telemetry"]["cost"],
            "parse_success": record["telemetry"]["parse_success"],
            "cache_hit": record["telemetry"]["cache_hit"],
        }
        for record in records
    ]
    write_csv(experiment_dir / "open_loop.csv", summary)
    return records


LOOK_COMMAND = {
    "far_left": 1.0,
    "left": 0.5,
    "center": 0.0,
    "right": -0.5,
    "far_right": -1.0,
    "hold": 0.0,
}

MOTION_COMMAND = {
    "stop": (0.0, 0.0),
    "forward": (0.50, 0.0),
    "backward": (-0.40, 0.0),
    "turn_left": (0.10, 0.70),
    "turn_right": (0.10, -0.70),
    "evade_left": (-0.10, 0.85),
    "evade_right": (-0.10, -0.85),
}


def decision_action(decision: PolicyDecision) -> np.ndarray:
    forward, body_turn = MOTION_COMMAND[decision.recommended_motion]
    head_turn = LOOK_COMMAND[decision.recommended_look]
    return np.asarray((forward, body_turn, head_turn), dtype=np.float32)


def _distance(env) -> float | None:
    model = env.unwrapped.model
    if not model.use_predator:
        return None
    return float(
        np.linalg.norm(
            np.asarray(model.prey.state.location, dtype=np.float64)
            - np.asarray(model.predator.state.location, dtype=np.float64),
        ),
    )


def _outcome_template(env) -> dict[str, Any]:
    initial_distance = _distance(env)
    return {
        "steps": 0,
        "captured": False,
        "goal_event": False,
        "terminated": False,
        "truncated": False,
        "total_reward": 0.0,
        "minimum_predator_distance": initial_distance,
        "final_goal_distance": float(
            env.unwrapped.reward_terms.get("goal_distance", math.nan),
        ),
    }


def _update_outcome(
    outcome: dict[str, Any],
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    env,
) -> None:
    events = info.get("transition_events", {})
    outcome["steps"] += 1
    outcome["captured"] = bool(outcome["captured"] or events.get("capture_event", False))
    outcome["goal_event"] = bool(outcome["goal_event"] or events.get("goal_event", False))
    outcome["terminated"] = bool(terminated)
    outcome["truncated"] = bool(truncated)
    outcome["total_reward"] += float(reward)
    distance = _distance(env)
    if distance is not None:
        current = outcome["minimum_predator_distance"]
        outcome["minimum_predator_distance"] = (
            distance if current is None else min(float(current), distance)
        )
    outcome["final_goal_distance"] = float(
        info.get("reward_terms", env.unwrapped.reward_terms).get(
            "goal_distance",
            math.nan,
        ),
    )


def evaluate_policy_branch(
    env,
    source_state: Mapping[str, Any],
    *,
    gaze_degrees: float,
    policy: VisionPolicy,
    history: Sequence[PublicHistoryFrame],
    horizon_steps: int,
) -> dict[str, Any]:
    source_hash = state_digest(source_state)
    observation, branch_state = restore_and_observe(
        env,
        source_state,
        gaze_degrees=gaze_degrees,
    )
    # Evaluation-only label.  It is captured separately and never added to
    # PolicyInput or provider logs.
    camera_visibility = env.get_predator_visibility()
    result = policy.decide(PolicyInput.from_observation(observation, history=history))
    action = decision_action(result.decision)
    env.set_state_dict(branch_state)
    outcome = _outcome_template(env)
    for _ in range(horizon_steps):
        _, reward, terminated, truncated, info = env.step(action)
        _update_outcome(
            outcome,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env=env,
        )
        if terminated or truncated:
            break
    if state_digest(source_state) != source_hash:
        raise AssertionError("Policy branch mutated its source snapshot")
    return {
        "gaze_degrees": float(gaze_degrees),
        "camera_visibility": camera_visibility,
        "decision": result.decision.to_dict(),
        "telemetry": result.telemetry.to_dict(),
        "action": [float(value) for value in action],
        "outcome": outcome,
    }


def _privileged_safe_action(env) -> np.ndarray:
    model = env.unwrapped.model
    if not model.use_predator:
        return np.asarray((0.50, 0.0, 0.0), dtype=np.float32)
    prey = np.asarray(model.prey.state.location, dtype=np.float64)
    predator = np.asarray(model.predator.state.location, dtype=np.float64)
    delta = predator - prey
    distance = float(np.linalg.norm(delta))
    predator_bearing = math.degrees(math.atan2(delta[1], delta[0]))
    relative = (predator_bearing - env.body_heading_degrees + 180.0) % 360.0 - 180.0
    turn_away = -1.0 if relative >= 0.0 else 1.0
    if distance < 0.34:
        return np.asarray((-0.55, turn_away, 0.0), dtype=np.float32)
    return np.asarray((0.25, 0.45 * turn_away, 0.0), dtype=np.float32)


def evaluate_privileged_branch(
    env,
    source_state: Mapping[str, Any],
    *,
    horizon_steps: int,
) -> dict[str, Any]:
    source_hash = state_digest(source_state)
    env.set_state_dict(copy.deepcopy(dict(source_state)))
    outcome = _outcome_template(env)
    actions = []
    for _ in range(horizon_steps):
        action = _privileged_safe_action(env)
        actions.append([float(value) for value in action])
        _, reward, terminated, truncated, info = env.step(action)
        _update_outcome(
            outcome,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env=env,
        )
        if terminated or truncated:
            break
    if state_digest(source_state) != source_hash:
        raise AssertionError("Privileged branch mutated its source snapshot")
    return {
        "baseline": "privileged_safe",
        "actions": actions,
        "outcome": outcome,
    }


def _candidate_score(branch: Mapping[str, Any]) -> tuple[Any, ...]:
    outcome = branch["outcome"]
    distance = outcome["minimum_predator_distance"]
    safe_distance = float(distance) if distance is not None else math.inf
    return (
        bool(outcome["captured"]),
        -safe_distance,
        -float(outcome["total_reward"]),
        float(outcome["final_goal_distance"]),
    )


def _criterion(
    name: str,
    passed: bool,
    observed: Any,
    failure_reason: str,
) -> Mapping[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "failure_reason": None if passed else failure_reason,
    }


def avoidable_by_looking_filter(
    fixed: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    privileged: Mapping[str, Any],
    *,
    source_unchanged: bool,
    risk_distance: float,
    minimum_distance_improvement: float,
) -> Mapping[str, Any]:
    best = min(candidates, key=_candidate_score)
    fixed_outcome = fixed["outcome"]
    best_outcome = best["outcome"]
    oracle_outcome = privileged["outcome"]
    fixed_distance = fixed_outcome["minimum_predator_distance"]
    best_distance = best_outcome["minimum_predator_distance"]
    fixed_adverse = bool(
        fixed_outcome["captured"]
        or (fixed_distance is not None and float(fixed_distance) < risk_distance)
    )
    fixed_pixels_visible = bool(
        fixed["camera_visibility"]["predator_pixels_visible"],
    )
    candidate_pixels_visible = [
        bool(branch["camera_visibility"]["predator_pixels_visible"])
        for branch in candidates
    ]
    reveals = bool(not fixed_pixels_visible and any(candidate_pixels_visible))
    improves = bool(
        fixed_outcome["captured"] and not best_outcome["captured"]
        or (
            fixed_distance is not None
            and best_distance is not None
            and float(best_distance) - float(fixed_distance)
            >= minimum_distance_improvement
        )
    )
    oracle_safe = bool(
        not oracle_outcome["captured"]
        and (
            oracle_outcome["minimum_predator_distance"] is None
            or float(oracle_outcome["minimum_predator_distance"]) >= risk_distance
        )
    )
    criteria = [
        _criterion(
            "source_snapshot_unchanged",
            source_unchanged,
            source_unchanged,
            "A branch mutated the stored source state.",
        ),
        _criterion(
            "fixed_gaze_has_no_predator_pixels",
            not fixed_pixels_visible,
            fixed_pixels_visible,
            "The predator already contributes pixels under fixed gaze.",
        ),
        _criterion(
            "candidate_gaze_reveals_threat",
            reveals,
            candidate_pixels_visible,
            "No candidate gaze makes predator pixels visible.",
        ),
        _criterion(
            "fixed_branch_has_registered_adverse_risk",
            fixed_adverse,
            fixed_outcome,
            "Fixed branch neither captured nor crossed the risk-distance threshold.",
        ),
        _criterion(
            "best_candidate_improves_registered_outcome",
            improves,
            {
                "fixed_minimum_distance": fixed_distance,
                "best_minimum_distance": best_distance,
                "fixed_captured": fixed_outcome["captured"],
                "best_captured": best_outcome["captured"],
            },
            "Best candidate did not meet the registered distance/capture improvement.",
        ),
        _criterion(
            "privileged_safe_baseline_avoids_registered_risk",
            oracle_safe,
            oracle_outcome,
            "Privileged baseline did not remain outside the registered risk condition.",
        ),
    ]
    return {
        "avoidable_by_looking_candidate": all(item["passed"] for item in criteria),
        "paper_definition_satisfied": False,
        "paper_definition_note": (
            "Engineering candidate only; legal gaze duration, held-out sampling, "
            "policy independence, and preregistered paper criteria remain unverified."
        ),
        "best_candidate_gaze_degrees": best["gaze_degrees"],
        "criteria": criteria,
    }


def run_branch_evaluation(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    policy: VisionPolicy | None = None,
) -> list[dict[str, Any]]:
    experiment_dir, snapshots = _snapshot_records(config, project_root=project_root)
    selected_policy = policy or build_policy(
        config["policy"],
        experiment_dir=experiment_dir,
    )
    environments = {
        True: make_env(config, use_predator=True),
        False: make_env(config, use_predator=False),
    }
    records = []
    try:
        for snapshot in snapshots:
            env = environments[bool(snapshot["use_predator"])]
            state = load_state(experiment_dir / snapshot["state_path"])
            source_hash = state_digest(state)
            history = _history_frames(experiment_dir, snapshot)
            fixed_gaze = float(state["first_person"]["head_yaw_degrees"])
            fixed = evaluate_policy_branch(
                env,
                state,
                gaze_degrees=fixed_gaze,
                policy=selected_policy,
                history=history,
                horizon_steps=int(config["branch"]["horizon_steps"]),
            )
            candidates = [
                evaluate_policy_branch(
                    env,
                    state,
                    gaze_degrees=float(gaze),
                    policy=selected_policy,
                    history=history,
                    horizon_steps=int(config["branch"]["horizon_steps"]),
                )
                for gaze in snapshot["gaze_candidates_degrees"]
            ]
            privileged = evaluate_privileged_branch(
                env,
                state,
                horizon_steps=int(config["branch"]["horizon_steps"]),
            )
            source_unchanged = state_digest(state) == source_hash
            screening = avoidable_by_looking_filter(
                fixed,
                candidates,
                privileged,
                source_unchanged=source_unchanged,
                risk_distance=float(config["branch"]["risk_distance"]),
                minimum_distance_improvement=float(
                    config["branch"]["minimum_distance_improvement"],
                ),
            )
            records.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "source": snapshot["source"],
                    "category": snapshot["category"],
                    "source_state_hash": source_hash,
                    "source_snapshot_unchanged": source_unchanged,
                    "fixed_gaze": fixed,
                    "candidate_gazes": candidates,
                    "privileged_safe_baseline": privileged,
                    "screening": screening,
                },
            )
        write_jsonl(experiment_dir / "branches.jsonl", records)
        summary = [
            {
                "snapshot_id": record["snapshot_id"],
                "source": record["source"],
                "category": record["category"],
                "fixed_captured": record["fixed_gaze"]["outcome"]["captured"],
                "fixed_minimum_distance": record["fixed_gaze"]["outcome"][
                    "minimum_predator_distance"
                ],
                "best_candidate_gaze_degrees": record["screening"][
                    "best_candidate_gaze_degrees"
                ],
                "avoidable_by_looking_candidate": record["screening"][
                    "avoidable_by_looking_candidate"
                ],
                "source_snapshot_unchanged": record["source_snapshot_unchanged"],
                "failed_criteria": [
                    criterion["name"]
                    for criterion in record["screening"]["criteria"]
                    if not criterion["passed"]
                ],
            }
            for record in records
        ]
        write_csv(experiment_dir / "branches.csv", summary)
        return records
    finally:
        for env in environments.values():
            env.close()
