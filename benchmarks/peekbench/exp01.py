"""EXP-01: paired VLM perception--action gap measurement.

The evaluator keeps public policy inputs structurally separate from privileged
labels and future outcomes.  Every look probe, semantic macro action, and
closed-loop controller is restored from the same exact source state.  This is
an exploratory measurement pipeline, not a paper-level hypothesis test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policies.base import (
    MOTION_ACTIONS,
    PolicyDecision,
    PolicyInput,
    PolicyResult,
    PublicHistoryFrame,
    VisionPolicy,
)
from policies.openrouter_vlm import build_policy

from .artifacts import (
    canonical_typed_bytes,
    load_observation,
    load_state,
    state_digest,
    write_csv,
    write_json,
    write_jsonl,
)
from .environment import PROJECT_ROOT, make_env
from .evaluation import MOTION_COMMAND
from .headroom import (
    _PrivilegedSafeActionBuilder,
    _experiment_dir,
    _history_frames,
    _oracle_score,
    _run_branch,
    _snapshot_records,
    _target_command,
)


CONDITIONS = ("current_only", "public_history")
CLOSED_LOOP_METHODS = (
    "fixed_continue",
    "vlm_open_loop_public_history",
    "vlm_closed_loop_current_only",
    "vlm_closed_loop_public_history",
    "privileged_safe_controller",
)
LOOK_TARGET_DEGREES: Mapping[str, float | None] = {
    "far_left": 60.0,
    "left": 30.0,
    "center": 0.0,
    "right": -30.0,
    "far_right": -60.0,
    "hold": None,
}


def _telemetry_summary(result: PolicyResult) -> dict[str, Any]:
    telemetry = result.telemetry
    return {
        "backend": telemetry.backend,
        "model": telemetry.model,
        "provider": dict(telemetry.provider),
        "prompt_hash": telemetry.prompt_hash,
        "image_hashes": dict(telemetry.image_hashes),
        "latency_ms": float(telemetry.latency_ms),
        "token_usage": dict(telemetry.token_usage),
        "cost": telemetry.cost,
        "parse_success": bool(telemetry.parse_success),
        "cache_hit": bool(telemetry.cache_hit),
    }


def _history_for_condition(
    condition: str,
    history: Sequence[PublicHistoryFrame],
) -> tuple[PublicHistoryFrame, ...]:
    if condition == "current_only":
        return ()
    if condition == "public_history":
        return tuple(history)
    raise ValueError(f"Unknown EXP-01 condition: {condition}")


class _FixedSemanticActionBuilder:
    """Execute one registered semantic macro through legal normalized actions."""

    def __init__(
        self,
        *,
        motion: str,
        look: str,
        tolerance_degrees: float,
    ) -> None:
        if motion not in MOTION_COMMAND:
            raise ValueError(f"Unknown semantic motion: {motion}")
        if look not in LOOK_TARGET_DEGREES:
            raise ValueError(f"Unknown semantic look: {look}")
        self.motion = motion
        self.look = look
        self.tolerance_degrees = float(tolerance_degrees)
        self._resolved_target: float | None = None

    @property
    def description(self) -> Mapping[str, Any]:
        target = LOOK_TARGET_DEGREES[self.look]
        if target is None and self._resolved_target is not None:
            target = self._resolved_target
        return {
            "information": "registered_semantic_action_only",
            "motion": self.motion,
            "look": self.look,
            "gaze": {
                "kind": "legal_target_tracking",
                "target_degrees": target,
            },
        }

    def __call__(self, env, observation: Mapping[str, np.ndarray], step: int):
        del observation
        if self._resolved_target is None:
            configured = LOOK_TARGET_DEGREES[self.look]
            self._resolved_target = (
                float(env.head_yaw_degrees)
                if configured is None
                else float(configured)
            )
        forward, body_turn = MOTION_COMMAND[self.motion]
        head_turn = _target_command(
            env,
            self._resolved_target,
            tolerance_degrees=self.tolerance_degrees,
        )
        action = np.asarray((forward, body_turn, head_turn), dtype=np.float32)
        return action, {
            "controller": "fixed_semantic_macro",
            "recommended_motion": self.motion,
            "recommended_look": self.look,
            "decision_update": step == 0,
        }


class _ClosedLoopPolicyActionBuilder:
    """Re-query a public-only policy at a registered macro interval."""

    def __init__(
        self,
        *,
        policy: VisionPolicy,
        condition: str,
        initial_history: Sequence[PublicHistoryFrame],
        max_history_frames: int,
        decision_interval_steps: int,
        tolerance_degrees: float,
        initial_result: PolicyResult,
    ) -> None:
        self.policy = policy
        self.condition = condition
        self.max_history_frames = (
            0 if condition == "current_only" else int(max_history_frames)
        )
        self.history = (
            list(initial_history)[-self.max_history_frames :]
            if self.max_history_frames
            else []
        )
        self.decision_interval_steps = int(decision_interval_steps)
        self.tolerance_degrees = float(tolerance_degrees)
        self.initial_result = initial_result
        self.current_decision: PolicyDecision | None = None
        self.current_target: float | None = None

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "information": "public_observation_only",
            "motion_policy": "configured_vlm",
            "history_condition": self.condition,
            "decision_interval_steps": self.decision_interval_steps,
            "max_history_frames": self.max_history_frames,
            "gaze": {"kind": "vlm_semantic_legal_target_tracking"},
        }

    def _decide(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        step: int,
    ) -> tuple[PolicyResult, str]:
        if step == 0:
            return self.initial_result, "precomputed_static"
        result = self.policy.decide(
            PolicyInput.from_observation(
                observation,
                history=tuple(self.history),
            ),
        )
        return result, "live_closed_loop"

    def __call__(self, env, observation: Mapping[str, np.ndarray], step: int):
        decision_update = bool(
            self.current_decision is None
            or step % self.decision_interval_steps == 0
        )
        telemetry: Mapping[str, Any] | None = None
        origin: str | None = None
        history_frames_supplied: int | None = None
        if decision_update:
            history_frames_supplied = len(self.history)
            result, origin = self._decide(observation, step=step)
            self.current_decision = result.decision
            telemetry = _telemetry_summary(result)
            configured_target = LOOK_TARGET_DEGREES[
                self.current_decision.recommended_look
            ]
            self.current_target = (
                float(env.head_yaw_degrees)
                if configured_target is None
                else float(configured_target)
            )

        assert self.current_decision is not None
        assert self.current_target is not None
        forward, body_turn = MOTION_COMMAND[
            self.current_decision.recommended_motion
        ]
        head_turn = _target_command(
            env,
            self.current_target,
            tolerance_degrees=self.tolerance_degrees,
        )
        action = np.asarray((forward, body_turn, head_turn), dtype=np.float32)
        record = {
            **self.current_decision.to_dict(),
            "decision_update": decision_update,
            "decision_origin": origin,
            "history_frames_supplied": history_frames_supplied,
            "telemetry": telemetry,
        }
        if self.max_history_frames:
            self.history.append(
                PublicHistoryFrame(
                    image_left=observation["image_left"],
                    image_right=observation["image_right"],
                ),
            )
            self.history = self.history[-self.max_history_frames :]
        return action, record


def _run_fixed_semantic_branch(
    env,
    state: Mapping[str, Any],
    *,
    method: str,
    motion: str,
    look: str,
    horizon_steps: int,
    risk_distance: float,
    tolerance_degrees: float,
) -> dict[str, Any]:
    return _run_branch(
        env,
        state,
        method=method,
        action_builder=_FixedSemanticActionBuilder(
            motion=motion,
            look=look,
            tolerance_degrees=tolerance_degrees,
        ),
        horizon_steps=horizon_steps,
        risk_distance=risk_distance,
    )


def _static_decisions(
    policy: VisionPolicy,
    observation: Mapping[str, np.ndarray],
    history: Sequence[PublicHistoryFrame],
) -> tuple[dict[str, Any], dict[str, PolicyResult]]:
    records: dict[str, Any] = {}
    results: dict[str, PolicyResult] = {}
    for condition in CONDITIONS:
        selected_history = _history_for_condition(condition, history)
        result = policy.decide(
            PolicyInput.from_observation(
                observation,
                history=selected_history,
            ),
        )
        results[condition] = result
        records[condition] = {
            "history_frames_supplied": len(selected_history),
            "decision": result.decision.to_dict(),
            "telemetry": _telemetry_summary(result),
        }
    return records, results


def _look_probes(
    env,
    state: Mapping[str, Any],
    *,
    horizon_steps: int,
    risk_distance: float,
    tolerance_degrees: float,
) -> dict[str, dict[str, Any]]:
    probes = {}
    for look in LOOK_TARGET_DEGREES:
        probes[look] = _run_fixed_semantic_branch(
            env,
            state,
            method=f"look_probe_{look}",
            motion="stop",
            look=look,
            horizon_steps=horizon_steps,
            risk_distance=risk_distance,
            tolerance_degrees=tolerance_degrees,
        )
    return probes


def _macro_candidates(
    env,
    state: Mapping[str, Any],
    *,
    horizon_steps: int,
    risk_distance: float,
    tolerance_degrees: float,
) -> dict[str, dict[str, Any]]:
    candidates = {}
    for motion in MOTION_ACTIONS:
        for look in LOOK_TARGET_DEGREES:
            key = f"{motion}|{look}"
            candidates[key] = _run_fixed_semantic_branch(
                env,
                state,
                method="semantic_macro_candidate",
                motion=motion,
                look=look,
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
                tolerance_degrees=tolerance_degrees,
            )
    return candidates


def _decision_updates(branch: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        value
        for value in branch["decision_trace"]
        if bool(value.get("decision_update", False))
    ]


def _closed_loop_diagnostics(
    branch: Mapping[str, Any],
    *,
    risk_threshold: float,
) -> dict[str, Any]:
    updates = _decision_updates(branch)
    motions = [str(value["recommended_motion"]) for value in updates]
    looks = [str(value["recommended_look"]) for value in updates]
    motion_switches = sum(
        left != right for left, right in zip(motions, motions[1:])
    )
    look_switches = sum(left != right for left, right in zip(looks, looks[1:]))
    high_risk_forward = sum(
        float(value["risk_next_horizon"]) >= risk_threshold
        and value["recommended_motion"] == "forward"
        for value in updates
    )
    conservative = sum(motion != "forward" for motion in motions)
    denominator = max(len(updates) - 1, 1)
    return {
        "decision_updates": len(updates),
        "motion_switches": motion_switches,
        "look_switches": look_switches,
        "motion_switch_rate": motion_switches / denominator,
        "look_switch_rate": look_switches / denominator,
        "high_risk_forward_count": high_risk_forward,
        "conservative_motion_rate": conservative / len(updates) if updates else None,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(values: Sequence[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("Binary labels and predictions must have equal length")
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    return {
        "n": len(labels),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def _summarize_static(
    records: Sequence[Mapping[str, Any]],
    *,
    risk_threshold: float,
) -> dict[str, Any]:
    summary = {}
    for condition in CONDITIONS:
        labels_visible = [
            bool(record["labels"]["predator_pixels_visible"])
            for record in records
        ]
        predictions_visible = [
            bool(record["static"][condition]["decision"]["threat_visible"])
            for record in records
        ]
        labels_danger = [bool(record["labels"]["danger_next_horizon"]) for record in records]
        risks = [
            float(record["static"][condition]["decision"]["risk_next_horizon"])
            for record in records
        ]
        predictions_danger = [risk >= risk_threshold for risk in risks]
        eligible = [
            record for record in records if record["labels"]["look_direction_eligible"]
        ]
        look_correct = [
            bool(record["measurements"][condition]["look_direction_correct"])
            for record in eligible
        ]
        summary[condition] = {
            "predator_pixel_detection": _binary_metrics(
                labels_visible,
                predictions_visible,
            ),
            "danger_classification": _binary_metrics(
                labels_danger,
                predictions_danger,
            ),
            "danger_brier_score": _mean(
                [
                    (risk - float(label)) ** 2
                    for risk, label in zip(risks, labels_danger)
                ],
            ),
            "look_direction": {
                "eligible_n": len(eligible),
                "correct_n": sum(look_correct),
                "accuracy": _rate(look_correct),
            },
        }
    return summary


def _summarize_methods(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in CLOSED_LOOP_METHODS:
        method_records = [record["closed_loop"][method] for record in records]
        predator_records = [
            record["closed_loop"][method]
            for record in records
            if bool(record["use_predator"])
        ]
        controls = [
            record["closed_loop"][method]
            for record in records
            if not bool(record["use_predator"])
        ]
        diagnostics = [
            branch.get("diagnostics", {})
            for branch in method_records
            if branch.get("diagnostics")
        ]
        distances = [
            float(branch["outcome"]["minimum_predator_distance"])
            for branch in predator_records
            if branch["outcome"]["minimum_predator_distance"] is not None
        ]
        rows.append(
            {
                "method": method,
                "snapshots": len(method_records),
                "predator_snapshots": len(predator_records),
                "predator_capture_rate": _rate(
                    [bool(branch["outcome"]["captured"]) for branch in predator_records],
                ),
                "predator_safe_success_rate": _rate(
                    [bool(branch["outcome"]["safe_success"]) for branch in predator_records],
                ),
                "control_capture_rate": _rate(
                    [bool(branch["outcome"]["captured"]) for branch in controls],
                ),
                "mean_minimum_predator_distance": _mean(distances),
                "mean_goal_progress": _mean(
                    [float(branch["outcome"]["goal_progress"]) for branch in method_records],
                ),
                "mean_gaze_travel_degrees": _mean(
                    [
                        float(branch["outcome"]["gaze_travel_degrees"])
                        for branch in method_records
                    ],
                ),
                "mean_motion_switch_rate": _mean(
                    [
                        float(value["motion_switch_rate"])
                        for value in diagnostics
                        if value.get("motion_switch_rate") is not None
                    ],
                ),
                "mean_look_switch_rate": _mean(
                    [
                        float(value["look_switch_rate"])
                        for value in diagnostics
                        if value.get("look_switch_rate") is not None
                    ],
                ),
                "control_conservative_motion_rate": _mean(
                    [
                        float(branch["diagnostics"]["conservative_motion_rate"])
                        for branch in controls
                        if branch.get("diagnostics", {}).get(
                            "conservative_motion_rate",
                        )
                        is not None
                    ],
                ),
            },
        )
    return rows


def _paired_capture(
    records: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    candidate: str,
) -> dict[str, Any]:
    predator_records = [record for record in records if bool(record["use_predator"])]
    reference_values = [
        bool(record["closed_loop"][reference]["outcome"]["captured"])
        for record in predator_records
    ]
    candidate_values = [
        bool(record["closed_loop"][candidate]["outcome"]["captured"])
        for record in predator_records
    ]
    avoided = sum(
        reference_value and not candidate_value
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    )
    introduced = sum(
        not reference_value and candidate_value
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "n": len(predator_records),
        "reference_capture_rate": _rate(reference_values),
        "candidate_capture_rate": _rate(candidate_values),
        "capture_rate_delta_candidate_minus_reference": (
            _rate(candidate_values) - _rate(reference_values)
            if predator_records
            else None
        ),
        "captures_avoided": avoided,
        "captures_introduced": introduced,
        "net_captures_avoided": avoided - introduced,
    }


def _summarize_gap(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actionable = [
        record
        for record in records
        if bool(record["labels"]["danger_next_horizon"])
        and bool(record["labels"]["safe_macro_exists"])
    ]
    correct_danger = [
        record
        for record in actionable
        if bool(
            record["measurements"]["public_history"]["danger_correct"],
        )
    ]
    macro_failures = [
        record
        for record in correct_danger
        if not bool(record["measurements"]["public_history"]["macro_action_safe"])
    ]
    closed_captures = [
        record
        for record in correct_danger
        if bool(
            record["closed_loop"]["vlm_closed_loop_public_history"]["outcome"][
                "captured"
            ],
        )
    ]
    return {
        "definition": (
            "Actionable states are registered-danger states with at least one "
            "safe semantic macro candidate. A gap remains when the public-history "
            "risk classification is correct but the selected macro is unsafe or "
            "the public-history closed loop is captured."
        ),
        "actionable_states": len(actionable),
        "correct_danger_judgments": len(correct_danger),
        "correct_danger_but_unsafe_macro_count": len(macro_failures),
        "correct_danger_but_unsafe_macro_rate": (
            len(macro_failures) / len(correct_danger) if correct_danger else None
        ),
        "correct_danger_but_closed_loop_capture_count": len(closed_captures),
        "correct_danger_but_closed_loop_capture_rate": (
            len(closed_captures) / len(correct_danger) if correct_danger else None
        ),
        "unsafe_macro_snapshot_ids": [record["snapshot_id"] for record in macro_failures],
        "closed_loop_capture_snapshot_ids": [
            record["snapshot_id"] for record in closed_captures
        ],
    }


def _summarize_memory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hidden = [record for record in records if record["category"] == "recently_visible_hidden"]
    current_risks = [
        float(record["static"]["current_only"]["decision"]["risk_next_horizon"])
        for record in hidden
    ]
    history_risks = [
        float(record["static"]["public_history"]["decision"]["risk_next_horizon"])
        for record in hidden
    ]
    return {
        "recently_visible_hidden_n": len(hidden),
        "mean_risk_current_only": _mean(current_risks),
        "mean_risk_public_history": _mean(history_risks),
        "mean_risk_delta_history_minus_current": _mean(
            [history - current for current, history in zip(current_risks, history_risks)],
        ),
        "closed_loop_captures_current_only": sum(
            bool(
                record["closed_loop"]["vlm_closed_loop_current_only"]["outcome"][
                    "captured"
                ],
            )
            for record in hidden
        ),
        "closed_loop_captures_public_history": sum(
            bool(
                record["closed_loop"]["vlm_closed_loop_public_history"]["outcome"][
                    "captured"
                ],
            )
            for record in hidden
        ),
    }


def _flatten_measurements(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for condition in CONDITIONS:
            decision = record["static"][condition]["decision"]
            measurement = record["measurements"][condition]
            rows.append(
                {
                    "snapshot_id": record["snapshot_id"],
                    "source": record["source"],
                    "category": record["category"],
                    "condition": condition,
                    "use_predator": record["use_predator"],
                    "predator_pixels_visible": record["labels"][
                        "predator_pixels_visible"
                    ],
                    "danger_next_horizon": record["labels"]["danger_next_horizon"],
                    "look_direction_eligible": record["labels"][
                        "look_direction_eligible"
                    ],
                    **decision,
                    **measurement,
                },
            )
    return rows


def _flatten_closed_loop(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for method in CLOSED_LOOP_METHODS:
            branch = record["closed_loop"][method]
            rows.append(
                {
                    "snapshot_id": record["snapshot_id"],
                    "source": record["source"],
                    "category": record["category"],
                    "use_predator": record["use_predator"],
                    "method": method,
                    **branch["outcome"],
                    **branch.get("diagnostics", {}),
                    "legal_gaze": branch["legal_gaze"],
                    "source_snapshot_unchanged": branch[
                        "source_snapshot_unchanged"
                    ],
                },
            )
    return rows


def run_exp01_evaluation(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    policy: VisionPolicy | None = None,
) -> dict[str, Any]:
    """Run all five EXP-01 measurements on one exact-state snapshot batch."""

    values = config["exp01"]
    policy_experiment_dir = _experiment_dir(config, project_root)
    policy_config = dict(config["policy"])
    time_step = float(config["environment"]["time_step"])
    policy_config["risk_horizon_seconds"] = (
        float(values["horizon_steps"]) * time_step
    )
    policy_config["macro_duration_seconds"] = (
        float(values["decision_interval_steps"]) * time_step
    )
    selected_policy = policy or build_policy(
        policy_config,
        experiment_dir=policy_experiment_dir,
    )
    backend = getattr(selected_policy, "backend", None)
    if bool(values["require_remote_policy"]) and backend != "openrouter":
        raise RuntimeError(
            "EXP-01 requires a remote VLM for this config, but "
            "OPENROUTER_API_KEY is absent or the supplied policy is not OpenRouter.",
        )
    experiment_dir, snapshots = _snapshot_records(config, project_root=project_root)

    horizon_steps = int(values["horizon_steps"])
    macro_horizon_steps = int(values["macro_horizon_steps"])
    look_probe_steps = int(values["look_probe_steps"])
    risk_distance = float(values["risk_distance"])
    risk_threshold = float(values["risk_threshold"])
    tolerance = float(values["target_tolerance_degrees"])
    decision_interval = int(values["decision_interval_steps"])
    max_history_frames = int(config["policy"]["max_history_frames"])
    environments = {
        True: make_env(config, use_predator=True),
        False: make_env(config, use_predator=False),
    }
    records: list[dict[str, Any]] = []
    live_closed_loop_telemetry: list[Mapping[str, Any]] = []
    try:
        for snapshot in snapshots:
            env = environments[bool(snapshot["use_predator"])]
            state = load_state(experiment_dir / snapshot["state_path"])
            source_bytes = canonical_typed_bytes(state)
            source_hash = state_digest(state)
            history = _history_frames(experiment_dir, snapshot)
            history = (
                history[-max_history_frames:] if max_history_frames else ()
            )
            observation = load_observation(
                experiment_dir / snapshot["observation_path"],
            )
            static, static_results = _static_decisions(
                selected_policy,
                observation,
                history,
            )

            look_probes = _look_probes(
                env,
                state,
                horizon_steps=look_probe_steps,
                risk_distance=risk_distance,
                tolerance_degrees=tolerance,
            )
            macro_candidates = _macro_candidates(
                env,
                state,
                horizon_steps=macro_horizon_steps,
                risk_distance=risk_distance,
                tolerance_degrees=tolerance,
            )
            fixed_continue = _run_fixed_semantic_branch(
                env,
                state,
                method="fixed_continue",
                motion="forward",
                look="hold",
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
                tolerance_degrees=tolerance,
            )
            oracle_macro_key, oracle_macro = min(
                macro_candidates.items(),
                key=lambda item: _oracle_score(item[1]),
            )
            safe_macro_keys = sorted(
                key
                for key, branch in macro_candidates.items()
                if bool(branch["outcome"]["safe_success"])
            )
            initial_visible = bool(
                fixed_continue["outcome"]["initial_predator_pixels_visible"],
            )
            reveal_looks = sorted(
                look
                for look, branch in look_probes.items()
                if bool(branch["outcome"]["ever_predator_pixels_visible"])
            )
            look_eligible = bool(not initial_visible and reveal_looks)

            measurements = {}
            for condition in CONDITIONS:
                decision = static_results[condition].decision
                selected_key = (
                    f"{decision.recommended_motion}|{decision.recommended_look}"
                )
                selected_macro = macro_candidates[selected_key]
                predicted_danger = bool(decision.risk_next_horizon >= risk_threshold)
                danger_label = bool(not fixed_continue["outcome"]["safe_success"])
                measurements[condition] = {
                    "threat_visible_correct": bool(
                        decision.threat_visible == initial_visible,
                    ),
                    "danger_predicted": predicted_danger,
                    "danger_correct": bool(predicted_danger == danger_label),
                    "recommended_look_reveals_predator": bool(
                        decision.recommended_look in reveal_looks
                    ),
                    "look_direction_correct": (
                        bool(decision.recommended_look in reveal_looks)
                        if look_eligible
                        else None
                    ),
                    "selected_macro_key": selected_key,
                    "macro_action_safe": bool(
                        selected_macro["outcome"]["safe_success"],
                    ),
                    "macro_action_correct_when_safe_exists": (
                        bool(selected_macro["outcome"]["safe_success"])
                        if safe_macro_keys
                        else None
                    ),
                    "macro_action_matches_oracle": selected_key == oracle_macro_key,
                    "macro_capture_regret": int(
                        bool(selected_macro["outcome"]["captured"]),
                    )
                    - int(bool(oracle_macro["outcome"]["captured"])),
                    "macro_minimum_distance_regret": (
                        None
                        if selected_macro["outcome"]["minimum_predator_distance"] is None
                        or oracle_macro["outcome"]["minimum_predator_distance"] is None
                        else float(
                            oracle_macro["outcome"]["minimum_predator_distance"],
                        )
                        - float(
                            selected_macro["outcome"]["minimum_predator_distance"],
                        )
                    ),
                }

            selected_open_decision = static_results["public_history"].decision
            open_loop = _run_fixed_semantic_branch(
                env,
                state,
                method="vlm_open_loop_public_history",
                motion=selected_open_decision.recommended_motion,
                look=selected_open_decision.recommended_look,
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
                tolerance_degrees=tolerance,
            )
            closed_loop: dict[str, dict[str, Any]] = {
                "fixed_continue": fixed_continue,
                "vlm_open_loop_public_history": open_loop,
            }
            for condition in CONDITIONS:
                method = f"vlm_closed_loop_{condition}"
                branch = _run_branch(
                    env,
                    state,
                    method=method,
                    action_builder=_ClosedLoopPolicyActionBuilder(
                        policy=selected_policy,
                        condition=condition,
                        initial_history=_history_for_condition(condition, history),
                        max_history_frames=max_history_frames,
                        decision_interval_steps=decision_interval,
                        tolerance_degrees=tolerance,
                        initial_result=static_results[condition],
                    ),
                    horizon_steps=horizon_steps,
                    risk_distance=risk_distance,
                )
                branch["diagnostics"] = _closed_loop_diagnostics(
                    branch,
                    risk_threshold=risk_threshold,
                )
                closed_loop[method] = branch
                live_closed_loop_telemetry.extend(
                    value["telemetry"]
                    for value in _decision_updates(branch)
                    if value.get("decision_origin") == "live_closed_loop"
                    and value.get("telemetry") is not None
                )

            privileged = _run_branch(
                env,
                state,
                method="privileged_safe_controller",
                action_builder=_PrivilegedSafeActionBuilder(
                    danger_distance=float(values["privileged_danger_distance"]),
                ),
                horizon_steps=horizon_steps,
                risk_distance=risk_distance,
            )
            closed_loop["privileged_safe_controller"] = privileged

            if canonical_typed_bytes(state) != source_bytes:
                raise AssertionError("EXP-01 branches mutated the source snapshot")
            records.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "source": snapshot["source"],
                    "target_category": snapshot["target_category"],
                    "category": snapshot["category"],
                    "construction_success": bool(snapshot["construction_success"]),
                    "use_predator": bool(snapshot["use_predator"]),
                    "source_state_hash": source_hash,
                    "static": static,
                    "labels": {
                        "predator_pixels_visible": initial_visible,
                        "danger_next_horizon": bool(
                            not fixed_continue["outcome"]["safe_success"],
                        ),
                        "danger_definition": (
                            "capture or minimum distance below risk_distance under "
                            "registered forward|hold reference macro"
                        ),
                        "look_direction_eligible": look_eligible,
                        "revealing_look_actions": reveal_looks,
                        "safe_macro_exists": bool(safe_macro_keys),
                        "safe_macro_keys": safe_macro_keys,
                        "oracle_macro_key": oracle_macro_key,
                    },
                    "measurements": measurements,
                    "look_probes": look_probes,
                    "macro_candidates": macro_candidates,
                    "closed_loop": closed_loop,
                    "source_snapshot_unchanged": (
                        canonical_typed_bytes(state) == source_bytes
                    ),
                },
            )
    finally:
        for env in environments.values():
            env.close()

    static_telemetry = [
        record["static"][condition]["telemetry"]
        for record in records
        for condition in CONDITIONS
    ]
    all_telemetry = static_telemetry + live_closed_loop_telemetry
    backends = sorted({str(value["backend"]) for value in all_telemetry})
    remote_uncached = sum(
        value["backend"] == "openrouter" and not bool(value["cache_hit"])
        for value in all_telemetry
    )
    summary = {
        "experiment": "EXP-01 VLM Perception--Action Gap",
        "experiment_id": config["experiment_id"],
        "config_hash": config["config_hash"],
        "seed": int(config["seed"]),
        "snapshots": len(records),
        "closed_loop_horizon_steps": horizon_steps,
        "macro_horizon_steps": macro_horizon_steps,
        "look_probe_steps": look_probe_steps,
        "decision_interval_steps": decision_interval,
        "risk_distance": risk_distance,
        "risk_threshold": risk_threshold,
        "categories": {
            category: sum(record["category"] == category for record in records)
            for category in sorted({record["category"] for record in records})
        },
        "backends": backends,
        "model_decisions": len(all_telemetry),
        "remote_uncached_model_calls": remote_uncached,
        "cache_hits": sum(bool(value["cache_hit"]) for value in all_telemetry),
        "total_reported_cost": sum(
            float(value["cost"]) for value in all_telemetry if value["cost"] is not None
        ),
        "all_parse_success": all(bool(value["parse_success"]) for value in all_telemetry),
        "all_source_snapshots_unchanged": all(
            bool(record["source_snapshot_unchanged"]) for record in records
        ),
        "all_actions_legal": all(
            bool(branch["legal_gaze"])
            for record in records
            for branch in (
                list(record["look_probes"].values())
                + list(record["macro_candidates"].values())
                + [record["closed_loop"][method] for method in CLOSED_LOOP_METHODS]
            )
        ),
        "static_measurements": _summarize_static(
            records,
            risk_threshold=risk_threshold,
        ),
        "macro_action": {
            "safe_macro_available_rate": _rate(
                [bool(record["labels"]["safe_macro_exists"]) for record in records],
            ),
            "public_history_safe_selection_rate_when_available": _rate(
                [
                    bool(
                        record["measurements"]["public_history"][
                            "macro_action_correct_when_safe_exists"
                        ],
                    )
                    for record in records
                    if record["measurements"]["public_history"][
                        "macro_action_correct_when_safe_exists"
                    ]
                    is not None
                ],
            ),
            "public_history_oracle_match_rate": _rate(
                [
                    bool(
                        record["measurements"]["public_history"][
                            "macro_action_matches_oracle"
                        ],
                    )
                    for record in records
                ],
            ),
        },
        "closed_loop_methods": _summarize_methods(records),
        "paired_capture_comparisons": [
            _paired_capture(
                records,
                reference="fixed_continue",
                candidate="vlm_closed_loop_public_history",
            ),
            _paired_capture(
                records,
                reference="vlm_open_loop_public_history",
                candidate="vlm_closed_loop_public_history",
            ),
            _paired_capture(
                records,
                reference="vlm_closed_loop_current_only",
                candidate="vlm_closed_loop_public_history",
            ),
        ],
        "memory": _summarize_memory(records),
        "perception_action_gap": _summarize_gap(records),
        "evidence_level": (
            "exploratory_remote_vlm_pilot"
            if backends == ["openrouter"]
            else "engineering_mock_only"
        ),
        "research_hypothesis_verified": False,
        "paper_claim_allowed": False,
        "limitations": [
            "Constructed states are not an on-policy encounter distribution.",
            "This exploratory batch has no inferential confidence claim.",
            "Danger is operationalized by one registered forward|hold reference branch.",
            "The semantic macro oracle is an evaluator using privileged future outcomes.",
        ],
    }
    write_jsonl(experiment_dir / "exp01.jsonl", records)
    write_csv(
        experiment_dir / "exp01_measurements.csv",
        _flatten_measurements(records),
    )
    write_csv(
        experiment_dir / "exp01_closed_loop.csv",
        _flatten_closed_loop(records),
    )
    write_csv(
        experiment_dir / "exp01_methods.csv",
        summary["closed_loop_methods"],
    )
    write_json(experiment_dir / "exp01_summary.json", summary)
    return {"records": records, "summary": summary}
