"""EXP-00: legal-duration active-gaze headroom evaluation.

This module intentionally does not train or call a remote model.  Fixed,
random, coverage-scan, and privileged-best-gaze branches share the same
deterministic public-observation motion policy.  The evaluator alone may read
future outcomes to select the best legal gaze branch.  A separate privileged
safe controller is reported as a task-level upper reference.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policies.base import MockVisionPolicy, PolicyInput, PublicHistoryFrame, VisionPolicy

from .artifacts import (
    canonical_typed_bytes,
    load_observation,
    load_state,
    prepare_experiment,
    read_jsonl,
    state_digest,
    write_csv,
    write_json,
    write_jsonl,
)
from .environment import PROJECT_ROOT, make_env, observe_current
from .evaluation import decision_action
from .generator import generate_snapshots


METHOD_ORDER = (
    "fixed_head",
    "random_head",
    "coverage_scan",
    "privileged_best_gaze",
    "privileged_safe_controller",
)

METHOD_LABELS = {
    "fixed_head": "Fixed head",
    "random_head": "Random head",
    "coverage_scan": "Coverage scan",
    "privileged_best_gaze": "Privileged best gaze",
    "privileged_safe_controller": "Privileged safe controller",
}


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
    manifest_path = experiment_dir / "snapshots.jsonl"
    if not manifest_path.exists():
        generate_snapshots(config, project_root=project_root)
    records = read_jsonl(manifest_path)
    if not records:
        raise RuntimeError("EXP-00 snapshot manifest is empty")
    mismatched = [
        record["snapshot_id"]
        for record in records
        if record.get("config_hash") != config["config_hash"]
    ]
    if mismatched:
        raise RuntimeError(
            "Existing snapshots were generated from a different config; use a "
            "new experiment_id. Mismatched IDs: " + ", ".join(mismatched[:3]),
        )
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


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _target_command(
    env,
    target_degrees: float,
    *,
    tolerance_degrees: float,
) -> float:
    """Track a target using only legal normalized head-yaw-rate commands."""

    target = float(np.clip(target_degrees, -env.head_yaw_limit, env.head_yaw_limit))
    error = target - float(env.head_yaw_degrees)
    maximum_delta = float(env.max_head_turn_rate * env._control_dt)
    if abs(target) <= tolerance_degrees and abs(error) <= tolerance_degrees:
        return 0.0

    raw_command = error / max(maximum_delta, 1e-9)
    if abs(error) <= tolerance_degrees:
        # A zero command invokes automatic recentering.  Small alternating
        # legal commands hold a non-zero target without editing wrapper state.
        direction = error if abs(error) > 0.1 else target
        raw_command = math.copysign(0.051, direction)
    elif abs(raw_command) <= 0.05:
        raw_command = math.copysign(0.051, error)
    return float(np.clip(raw_command, -1.0, 1.0))


@dataclass
class _TargetGaze:
    target_degrees: float
    tolerance_degrees: float

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "kind": "legal_target_tracking",
            "target_degrees": float(self.target_degrees),
        }

    def command(self, env, step: int) -> float:
        del step
        return _target_command(
            env,
            self.target_degrees,
            tolerance_degrees=self.tolerance_degrees,
        )


class _CoverageScan:
    def __init__(
        self,
        *,
        head_yaw_limit: float,
        dwell_steps: int,
        tolerance_degrees: float,
    ) -> None:
        self.targets = (-float(head_yaw_limit), float(head_yaw_limit))
        self.dwell_steps = int(dwell_steps)
        self.tolerance_degrees = float(tolerance_degrees)
        self._target_index = 0
        self._dwell_count = 0

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "kind": "legal_coverage_scan",
            "targets_degrees": list(self.targets),
            "dwell_steps": self.dwell_steps,
        }

    def command(self, env, step: int) -> float:
        del step
        target = self.targets[self._target_index]
        if abs(float(env.head_yaw_degrees) - target) <= self.tolerance_degrees:
            if self._dwell_count >= self.dwell_steps:
                self._target_index = (self._target_index + 1) % len(self.targets)
                self._dwell_count = 0
                target = self.targets[self._target_index]
            else:
                self._dwell_count += 1
        return _target_command(
            env,
            target,
            tolerance_degrees=self.tolerance_degrees,
        )


class _RandomGaze:
    def __init__(
        self,
        *,
        candidates_degrees: Sequence[float],
        horizon_steps: int,
        hold_steps: int,
        tolerance_degrees: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(int(seed))
        target_count = int(math.ceil(horizon_steps / hold_steps))
        self.targets = tuple(
            float(value)
            for value in rng.choice(
                np.asarray(candidates_degrees, dtype=np.float64),
                size=target_count,
                replace=True,
            )
        )
        self.hold_steps = int(hold_steps)
        self.tolerance_degrees = float(tolerance_degrees)
        self.seed = int(seed)

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "kind": "legal_random_targets",
            "seed": self.seed,
            "hold_steps": self.hold_steps,
            "targets_degrees": list(self.targets),
        }

    def command(self, env, step: int) -> float:
        target_index = min(step // self.hold_steps, len(self.targets) - 1)
        return _target_command(
            env,
            self.targets[target_index],
            tolerance_degrees=self.tolerance_degrees,
        )


class _PublicActionBuilder:
    """Combine a public-only motion policy with an independently fixed gaze rule."""

    def __init__(
        self,
        *,
        policy: VisionPolicy,
        gaze_controller,
        history: Sequence[PublicHistoryFrame],
        max_history_frames: int,
    ) -> None:
        self.policy = policy
        self.gaze_controller = gaze_controller
        self.history = list(history)[-max_history_frames:]
        self.max_history_frames = int(max_history_frames)

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "information": "public_observation_only",
            "motion_policy": "mock/vision-v1",
            "gaze": dict(self.gaze_controller.description),
        }

    def __call__(self, env, observation: Mapping[str, np.ndarray], step: int):
        result = self.policy.decide(
            PolicyInput.from_observation(observation, history=tuple(self.history)),
        )
        motion = decision_action(result.decision)[:2]
        head_command = self.gaze_controller.command(env, step)
        action = np.asarray(
            (float(motion[0]), float(motion[1]), float(head_command)),
            dtype=np.float32,
        )
        self.history.append(
            PublicHistoryFrame(
                image_left=observation["image_left"],
                image_right=observation["image_right"],
            ),
        )
        self.history = self.history[-self.max_history_frames :]
        return action, result.decision.to_dict()


class _PrivilegedSafeActionBuilder:
    def __init__(self, *, danger_distance: float) -> None:
        self.danger_distance = float(danger_distance)

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "information": "privileged_exact_state",
            "motion_policy": "goal_attraction_with_predator_repulsion",
            "gaze": {"kind": "fixed_center"},
        }

    def __call__(self, env, observation: Mapping[str, np.ndarray], step: int):
        del observation, step
        model = env.unwrapped.model
        prey = np.asarray(model.prey.state.location, dtype=np.float64)
        goal_location = getattr(model, "goal_location", None)
        if goal_location is None:
            heading = math.radians(float(env.body_heading_degrees))
            desired = np.asarray((math.cos(heading), math.sin(heading)))
        else:
            desired = np.asarray(goal_location, dtype=np.float64) - prey

        if model.use_predator:
            predator = np.asarray(model.predator.state.location, dtype=np.float64)
            away = prey - predator
            distance = float(np.linalg.norm(away))
            away_norm = float(np.linalg.norm(away))
            if away_norm > 1e-9:
                away = away / away_norm
            if distance <= self.danger_distance * 1.5:
                desired = away
            else:
                goal_norm = float(np.linalg.norm(desired))
                if goal_norm > 1e-9:
                    desired = desired / goal_norm
                repulsion = max(
                    0.0,
                    (self.danger_distance * 2.0 - distance)
                    / max(self.danger_distance * 2.0, 1e-9),
                )
                desired = desired + 1.5 * repulsion * away

        desired_norm = float(np.linalg.norm(desired))
        if desired_norm <= 1e-9:
            desired = np.asarray((1.0, 0.0), dtype=np.float64)
        desired_bearing = math.degrees(math.atan2(desired[1], desired[0]))
        body_heading = float(env.body_heading_degrees)
        forward_error = _wrap_degrees(desired_bearing - body_heading)
        backward_error = _wrap_degrees(desired_bearing - (body_heading + 180.0))
        if abs(forward_error) <= abs(backward_error):
            forward_command = 1.0
            heading_error = forward_error
        else:
            forward_command = -1.0
            heading_error = backward_error
        body_delta = float(env.max_body_turn_rate * env._control_dt)
        body_command = float(np.clip(heading_error / max(body_delta, 1e-9), -1.0, 1.0))
        if abs(heading_error) > 60.0:
            forward_command *= 0.35
        action = np.asarray(
            (forward_command, body_command, 0.0),
            dtype=np.float32,
        )
        return action, {
            "controller": "privileged_safe",
            "used_predator_state": bool(model.use_predator),
        }


def _predator_distance(env) -> float | None:
    model = env.unwrapped.model
    if not model.use_predator:
        return None
    return float(
        np.linalg.norm(
            np.asarray(model.prey.state.location, dtype=np.float64)
            - np.asarray(model.predator.state.location, dtype=np.float64),
        ),
    )


def _goal_distance(env) -> float:
    return float(env.unwrapped.model.prey_data.prey_goal_distance)


def _run_branch(
    env,
    source_state: Mapping[str, Any],
    *,
    method: str,
    action_builder,
    horizon_steps: int,
    risk_distance: float,
) -> dict[str, Any]:
    source_hash = state_digest(source_state)
    source_bytes = canonical_typed_bytes(source_state)
    env.set_state_dict(copy.deepcopy(dict(source_state)))
    observation = observe_current(env)
    initial_visibility = env.get_predator_visibility()
    initial_location = np.asarray(env.unwrapped.model.prey.state.location, dtype=np.float64)
    previous_location = np.array(initial_location, copy=True)
    initial_goal_distance = _goal_distance(env)
    minimum_predator_distance = _predator_distance(env)
    previous_head_yaw = float(env.head_yaw_degrees)

    action_trace: list[list[float]] = []
    head_yaw_trace = [previous_head_yaw]
    visibility_trace = [bool(initial_visibility["predator_pixels_visible"])]
    decision_trace: list[Mapping[str, Any]] = []
    total_reward = 0.0
    path_cost = 0.0
    gaze_travel_degrees = 0.0
    active_look_steps = 0
    captured = False
    goal_event = False
    terminated = False
    truncated = False
    legal_gaze = True

    for step in range(int(horizon_steps)):
        action, decision = action_builder(env, observation, step)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError(f"{method} produced an invalid action: {action!r}")
        if np.any(action < -1.0) or np.any(action > 1.0):
            legal_gaze = False
            raise ValueError(f"{method} produced an out-of-range action: {action!r}")
        action_trace.append([float(value) for value in action])
        decision_trace.append(dict(decision))
        if abs(float(action[2])) > 0.05:
            active_look_steps += 1

        observation, reward, terminated, truncated, info = env.step(action)
        current_location = np.asarray(
            env.unwrapped.model.prey.state.location,
            dtype=np.float64,
        )
        path_cost += float(np.linalg.norm(current_location - previous_location))
        previous_location = current_location
        current_head_yaw = float(env.head_yaw_degrees)
        gaze_travel_degrees += abs(current_head_yaw - previous_head_yaw)
        previous_head_yaw = current_head_yaw
        head_yaw_trace.append(current_head_yaw)
        if abs(current_head_yaw) > float(env.head_yaw_limit) + 1e-8:
            legal_gaze = False

        events = info.get("transition_events", {})
        captured = bool(captured or events.get("capture_event", False))
        goal_event = bool(goal_event or events.get("goal_event", False))
        total_reward += float(reward)
        if env.unwrapped.model.use_predator:
            transition_distance = events.get("minimum_distance")
            if transition_distance is None:
                transition_distance = _predator_distance(env)
            if transition_distance is not None:
                minimum_predator_distance = min(
                    float(minimum_predator_distance),
                    float(transition_distance),
                )
        visibility_trace.append(bool(events.get("predator_pixels_visible", False)))
        if terminated or truncated:
            break

    final_goal_distance = _goal_distance(env)
    safe_success = bool(
        not captured
        and (
            minimum_predator_distance is None
            or float(minimum_predator_distance) >= float(risk_distance)
        )
    )
    source_unchanged = canonical_typed_bytes(source_state) == source_bytes
    if not source_unchanged:
        raise AssertionError(f"{method} mutated its source snapshot")

    return {
        "method": method,
        "controller": dict(action_builder.description),
        "source_snapshot_unchanged": source_unchanged,
        "legal_gaze": bool(legal_gaze),
        "actions": action_trace,
        "head_yaw_degrees": head_yaw_trace,
        "predator_pixels_visible": visibility_trace,
        "decision_trace": decision_trace,
        "outcome": {
            "steps": len(action_trace),
            "safe_success": safe_success,
            "captured": captured,
            "goal_event": goal_event,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": total_reward,
            "minimum_predator_distance": minimum_predator_distance,
            "initial_goal_distance": initial_goal_distance,
            "final_goal_distance": final_goal_distance,
            "goal_progress": initial_goal_distance - final_goal_distance,
            "path_cost": path_cost,
            "gaze_travel_degrees": gaze_travel_degrees,
            "active_look_steps": active_look_steps,
            "initial_predator_pixels_visible": bool(
                initial_visibility["predator_pixels_visible"],
            ),
            "ever_predator_pixels_visible": any(visibility_trace),
            "predator_revealed_after_start": bool(
                not visibility_trace[0] and any(visibility_trace[1:]),
            ),
        },
    }


def _stable_random_seed(seed: int, snapshot_id: str, replicate: int) -> int:
    material = f"{int(seed)}:{snapshot_id}:{int(replicate)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _oracle_score(branch: Mapping[str, Any]) -> tuple[Any, ...]:
    outcome = branch["outcome"]
    minimum_distance = outcome["minimum_predator_distance"]
    safe_distance = 1e9 if minimum_distance is None else float(minimum_distance)
    target = float(branch["controller"]["gaze"].get("target_degrees", 0.0))
    return (
        not bool(outcome["safe_success"]),
        bool(outcome["captured"]),
        -safe_distance,
        not bool(outcome["goal_event"]),
        -float(outcome["goal_progress"]),
        float(outcome["path_cost"]),
        float(outcome["gaze_travel_degrees"]),
        abs(target),
        target,
    )


def _annotate_against_fixed(
    branch: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    active_look_degrees_threshold: float,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(branch))
    fixed_outcome = fixed["outcome"]
    outcome = result["outcome"]
    looked = bool(
        float(outcome["gaze_travel_degrees"])
        >= float(active_look_degrees_threshold)
    )
    recovered_safety = bool(
        not fixed_outcome["safe_success"] and outcome["safe_success"],
    )
    result["comparison_to_fixed"] = {
        "looked": looked,
        "recovered_safety": recovered_safety,
        "unnecessary_look": bool(looked and not recovered_safety),
        "safe_success_delta": int(bool(outcome["safe_success"]))
        - int(bool(fixed_outcome["safe_success"])),
        "capture_delta": int(bool(outcome["captured"]))
        - int(bool(fixed_outcome["captured"])),
        "path_cost_delta": float(outcome["path_cost"])
        - float(fixed_outcome["path_cost"]),
    }
    return result


def _method_branches(record: Mapping[str, Any], method: str) -> list[Mapping[str, Any]]:
    value = record["methods"][method]
    return list(value) if isinstance(value, list) else [value]


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_methods(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_ORDER:
        per_snapshot = []
        run_count = 0
        for record in records:
            branches = _method_branches(record, method)
            run_count += len(branches)
            per_snapshot.append(
                {
                    "safe_success": float(
                        np.mean(
                            [bool(branch["outcome"]["safe_success"]) for branch in branches],
                        ),
                    ),
                    "captured": float(
                        np.mean(
                            [bool(branch["outcome"]["captured"]) for branch in branches],
                        ),
                    ),
                    "goal_event": float(
                        np.mean(
                            [bool(branch["outcome"]["goal_event"]) for branch in branches],
                        ),
                    ),
                    "unnecessary_look": float(
                        np.mean(
                            [
                                bool(branch["comparison_to_fixed"]["unnecessary_look"])
                                for branch in branches
                            ],
                        ),
                    ),
                    "path_cost": float(
                        np.mean(
                            [float(branch["outcome"]["path_cost"]) for branch in branches],
                        ),
                    ),
                    "path_cost_delta": float(
                        np.mean(
                            [
                                float(branch["comparison_to_fixed"]["path_cost_delta"])
                                for branch in branches
                            ],
                        ),
                    ),
                    "gaze_travel_degrees": float(
                        np.mean(
                            [
                                float(branch["outcome"]["gaze_travel_degrees"])
                                for branch in branches
                            ],
                        ),
                    ),
                    "minimum_predator_distance": _mean(
                        [
                            float(branch["outcome"]["minimum_predator_distance"])
                            for branch in branches
                            if branch["outcome"]["minimum_predator_distance"] is not None
                        ],
                    ),
                    "control": not bool(record["use_predator"]),
                },
            )
        control_rows = [row for row in per_snapshot if row["control"]]
        distance_values = [
            float(row["minimum_predator_distance"])
            for row in per_snapshot
            if row["minimum_predator_distance"] is not None
        ]
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "snapshots": len(per_snapshot),
                "runs": run_count,
                "safe_success_rate": _mean(
                    [float(row["safe_success"]) for row in per_snapshot],
                ),
                "capture_rate": _mean(
                    [float(row["captured"]) for row in per_snapshot],
                ),
                "goal_success_rate": _mean(
                    [float(row["goal_event"]) for row in per_snapshot],
                ),
                "unnecessary_look_rate": _mean(
                    [float(row["unnecessary_look"]) for row in per_snapshot],
                ),
                "control_unnecessary_look_rate": _mean(
                    [float(row["unnecessary_look"]) for row in control_rows],
                ),
                "mean_path_cost": _mean(
                    [float(row["path_cost"]) for row in per_snapshot],
                ),
                "mean_path_cost_delta_vs_fixed": _mean(
                    [float(row["path_cost_delta"]) for row in per_snapshot],
                ),
                "mean_gaze_travel_degrees": _mean(
                    [float(row["gaze_travel_degrees"]) for row in per_snapshot],
                ),
                "mean_minimum_predator_distance": _mean(distance_values),
            },
        )
    return rows


def _criterion(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
    }


def evaluate_go_condition(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    go_config = config["headroom"]["go"]
    threat_records = [record for record in records if bool(record["use_predator"])]
    fixed_failures = [
        record
        for record in threat_records
        if not bool(record["methods"]["fixed_head"]["outcome"]["safe_success"])
    ]
    oracle_recoveries = [
        record
        for record in fixed_failures
        if bool(
            record["methods"]["privileged_best_gaze"]["outcome"]["safe_success"],
        )
    ]
    minimum_safe_candidates = int(
        go_config["minimum_safe_nonzero_gaze_candidates"],
    )
    stable_recoveries = []
    for record in oracle_recoveries:
        safe_nonzero_candidates = sum(
            bool(candidate["outcome"]["safe_success"])
            and abs(
                float(
                    candidate["controller"]["gaze"].get("target_degrees", 0.0),
                ),
            )
            > 1e-9
            for candidate in record["legal_gaze_candidates"]
        )
        if safe_nonzero_candidates >= minimum_safe_candidates:
            stable_recoveries.append(record)

    threat_count = len(threat_records)
    fixed_failure_count = len(fixed_failures)
    fixed_failure_fraction = fixed_failure_count / threat_count if threat_count else 0.0
    stable_headroom_fraction = (
        len(stable_recoveries) / threat_count if threat_count else 0.0
    )
    recovery_fraction = (
        len(stable_recoveries) / fixed_failure_count if fixed_failure_count else 0.0
    )
    all_legal = all(
        bool(branch["legal_gaze"])
        and bool(branch["source_snapshot_unchanged"])
        for record in records
        for method in METHOD_ORDER
        for branch in _method_branches(record, method)
    ) and all(
        bool(candidate["legal_gaze"])
        and bool(candidate["source_snapshot_unchanged"])
        for record in records
        for candidate in record["legal_gaze_candidates"]
    )
    all_constructed_as_requested = all(
        bool(record.get("construction_success", True)) for record in records
    )

    conditions = [
        _criterion(
            "minimum_predator_snapshots",
            threat_count >= int(go_config["minimum_predator_snapshots"]),
            threat_count,
            int(go_config["minimum_predator_snapshots"]),
        ),
        _criterion(
            "fixed_failure_fraction",
            fixed_failure_fraction
            >= float(go_config["minimum_fixed_failure_fraction"]),
            fixed_failure_fraction,
            float(go_config["minimum_fixed_failure_fraction"]),
        ),
        _criterion(
            "stable_headroom_fraction",
            stable_headroom_fraction
            >= float(go_config["minimum_stable_headroom_fraction"]),
            stable_headroom_fraction,
            float(go_config["minimum_stable_headroom_fraction"]),
        ),
        _criterion(
            "stable_recovery_fraction_of_fixed_failures",
            recovery_fraction
            >= float(go_config["minimum_recovery_fraction_of_fixed_failures"]),
            recovery_fraction,
            float(go_config["minimum_recovery_fraction_of_fixed_failures"]),
        ),
        _criterion(
            "minimum_stable_recoveries",
            len(stable_recoveries)
            >= int(go_config["minimum_stable_recoveries"]),
            len(stable_recoveries),
            int(go_config["minimum_stable_recoveries"]),
        ),
        _criterion(
            "all_gaze_branches_legal_and_source_immutable",
            all_legal,
            all_legal,
            True,
        ),
        _criterion(
            "all_snapshots_constructed_as_requested",
            all_constructed_as_requested,
            all_constructed_as_requested,
            True,
        ),
    ]
    passed = all(condition["passed"] for condition in conditions)
    return {
        "verdict": "GO" if passed else "NO_GO",
        "go_condition_met": passed,
        "engineering_screen_only": True,
        "research_hypothesis_verified": False,
        "definition": (
            "A stable recovery is a fixed-head safety failure for which the "
            "privileged best legal-duration gaze succeeds and at least the "
            "registered number of distinct non-zero legal gaze targets also succeed."
        ),
        "predator_snapshots": threat_count,
        "fixed_failure_count": fixed_failure_count,
        "fixed_failure_fraction": fixed_failure_fraction,
        "oracle_recovery_count": len(oracle_recoveries),
        "stable_recovery_count": len(stable_recoveries),
        "stable_headroom_fraction": stable_headroom_fraction,
        "stable_recovery_fraction_of_fixed_failures": recovery_fraction,
        "stable_recovery_snapshot_ids": [
            record["snapshot_id"] for record in stable_recoveries
        ],
        "conditions": conditions,
    }


def _flatten_runs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for method in METHOD_ORDER:
            branches = _method_branches(record, method)
            for replicate, branch in enumerate(branches):
                outcome = branch["outcome"]
                rows.append(
                    {
                        "snapshot_id": record["snapshot_id"],
                        "source": record["source"],
                        "category": record["category"],
                        "use_predator": record["use_predator"],
                        "method": method,
                        "replicate": replicate,
                        "safe_success": outcome["safe_success"],
                        "captured": outcome["captured"],
                        "goal_event": outcome["goal_event"],
                        "minimum_predator_distance": outcome[
                            "minimum_predator_distance"
                        ],
                        "path_cost": outcome["path_cost"],
                        "path_cost_delta_vs_fixed": branch["comparison_to_fixed"][
                            "path_cost_delta"
                        ],
                        "gaze_travel_degrees": outcome["gaze_travel_degrees"],
                        "unnecessary_look": branch["comparison_to_fixed"][
                            "unnecessary_look"
                        ],
                        "legal_gaze": branch["legal_gaze"],
                        "source_snapshot_unchanged": branch[
                            "source_snapshot_unchanged"
                        ],
                    },
                )
    return rows


def _stratified_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stratum_type, key in (("source", "source"), ("category", "category")):
        values = sorted({str(record[key]) for record in records})
        for value in values:
            subset = [record for record in records if str(record[key]) == value]
            for summary in summarize_methods(subset):
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum": value,
                        **summary,
                    },
                )
    return rows


def run_headroom_evaluation(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run EXP-00 on a fixed exact-state snapshot batch and save all artifacts."""

    experiment_dir, snapshots = _snapshot_records(config, project_root=project_root)
    values = config["headroom"]
    horizon_steps = int(values["horizon_steps"])
    risk_distance = float(values["risk_distance"])
    tolerance = float(values["target_tolerance_degrees"])
    active_threshold = float(values["active_look_degrees_threshold"])
    max_history_frames = int(config["policy"]["max_history_frames"])
    policy = MockVisionPolicy()
    environments = {
        True: make_env(config, use_predator=True),
        False: make_env(config, use_predator=False),
    }
    records: list[dict[str, Any]] = []
    try:
        for snapshot in snapshots:
            env = environments[bool(snapshot["use_predator"])]
            state = load_state(experiment_dir / snapshot["state_path"])
            source_hash = state_digest(state)
            source_bytes = canonical_typed_bytes(state)
            history = _history_frames(experiment_dir, snapshot)

            fixed = _run_branch(
                env,
                state,
                method="fixed_head",
                action_builder=_PublicActionBuilder(
                    policy=policy,
                    gaze_controller=_TargetGaze(0.0, tolerance),
                    history=history,
                    max_history_frames=max_history_frames,
                ),
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
            )
            fixed = _annotate_against_fixed(
                fixed,
                fixed,
                active_look_degrees_threshold=active_threshold,
            )

            random_branches = []
            for replicate in range(int(values["random_replicates"])):
                random_seed = _stable_random_seed(
                    int(config["seed"]),
                    str(snapshot["snapshot_id"]),
                    replicate,
                )
                branch = _run_branch(
                    env,
                    state,
                    method="random_head",
                    action_builder=_PublicActionBuilder(
                        policy=policy,
                        gaze_controller=_RandomGaze(
                            candidates_degrees=config["gaze_candidates_degrees"],
                            horizon_steps=horizon_steps,
                            hold_steps=int(values["random_target_hold_steps"]),
                            tolerance_degrees=tolerance,
                            seed=random_seed,
                        ),
                        history=history,
                        max_history_frames=max_history_frames,
                    ),
                    horizon_steps=horizon_steps,
                    risk_distance=risk_distance,
                )
                random_branches.append(
                    _annotate_against_fixed(
                        branch,
                        fixed,
                        active_look_degrees_threshold=active_threshold,
                    ),
                )

            coverage = _run_branch(
                env,
                state,
                method="coverage_scan",
                action_builder=_PublicActionBuilder(
                    policy=policy,
                    gaze_controller=_CoverageScan(
                        head_yaw_limit=float(env.head_yaw_limit),
                        dwell_steps=int(values["scan_dwell_steps"]),
                        tolerance_degrees=tolerance,
                    ),
                    history=history,
                    max_history_frames=max_history_frames,
                ),
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
            )
            coverage = _annotate_against_fixed(
                coverage,
                fixed,
                active_look_degrees_threshold=active_threshold,
            )

            legal_candidates = []
            for gaze_degrees in config["gaze_candidates_degrees"]:
                branch = _run_branch(
                    env,
                    state,
                    method="legal_gaze_candidate",
                    action_builder=_PublicActionBuilder(
                        policy=policy,
                        gaze_controller=_TargetGaze(float(gaze_degrees), tolerance),
                        history=history,
                        max_history_frames=max_history_frames,
                    ),
                    horizon_steps=horizon_steps,
                    risk_distance=risk_distance,
                )
                legal_candidates.append(
                    _annotate_against_fixed(
                        branch,
                        fixed,
                        active_look_degrees_threshold=active_threshold,
                    ),
                )
            selected_candidate = min(legal_candidates, key=_oracle_score)
            privileged_best = copy.deepcopy(selected_candidate)
            privileged_best["method"] = "privileged_best_gaze"
            privileged_best["oracle_selection"] = {
                "uses_future_privileged_outcomes": True,
                "candidate_count": len(legal_candidates),
                "selected_target_degrees": float(
                    privileged_best["controller"]["gaze"]["target_degrees"],
                ),
                "ranking": [
                    "safe_success",
                    "capture",
                    "minimum_predator_distance",
                    "goal_success",
                    "goal_progress",
                    "path_cost",
                    "gaze_travel",
                ],
            }

            privileged_safe = _run_branch(
                env,
                state,
                method="privileged_safe_controller",
                action_builder=_PrivilegedSafeActionBuilder(
                    danger_distance=float(values["privileged_danger_distance"]),
                ),
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
            )
            privileged_safe = _annotate_against_fixed(
                privileged_safe,
                fixed,
                active_look_degrees_threshold=active_threshold,
            )

            if canonical_typed_bytes(state) != source_bytes:
                raise AssertionError("EXP-00 branches mutated the stored source state")
            records.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "source": snapshot["source"],
                    "target_category": snapshot["target_category"],
                    "category": snapshot["category"],
                    "construction_success": bool(snapshot["construction_success"]),
                    "construction_failure": snapshot["construction_failure"],
                    "use_predator": bool(snapshot["use_predator"]),
                    "source_state_hash": source_hash,
                    "legal_gaze_candidates": legal_candidates,
                    "methods": {
                        "fixed_head": fixed,
                        "random_head": random_branches,
                        "coverage_scan": coverage,
                        "privileged_best_gaze": privileged_best,
                        "privileged_safe_controller": privileged_safe,
                    },
                },
            )
    finally:
        for env in environments.values():
            env.close()

    method_summary = summarize_methods(records)
    go_summary = evaluate_go_condition(records, config)
    summary = {
        "experiment": "EXP-00 Gaze Oracle Headroom",
        "experiment_id": config["experiment_id"],
        "config_hash": config["config_hash"],
        "seed": int(config["seed"]),
        "snapshots": len(records),
        "construction_success_count": sum(
            bool(record["construction_success"]) for record in records
        ),
        "horizon_steps": horizon_steps,
        "risk_distance": risk_distance,
        "motion_policy_information_boundary": "public_observation_only",
        "remote_model_calls": 0,
        "methods": method_summary,
        "go": go_summary,
    }
    write_jsonl(experiment_dir / "headroom.jsonl", records)
    write_csv(experiment_dir / "headroom_runs.csv", _flatten_runs(records))
    write_csv(experiment_dir / "headroom_methods.csv", method_summary)
    write_csv(experiment_dir / "headroom_strata.csv", _stratified_rows(records))
    write_json(experiment_dir / "headroom_summary.json", summary)
    return {"records": records, "summary": summary}
