"""EXP-04: Active Gaze Without Free Compute.

Every gaze method receives exactly one binocular observation, one call to the
same public visual encoder, and one decision update per simulator step.  Gaze
selection may use only the encoded public history; candidate views are never
rendered speculatively.  The run is rejected if any registered budget differs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import load_state, write_csv, write_json, write_jsonl
from .controlled_memory import ENCODER_ID, PublicVisualEncoder, VisualFeatures, feature_dict
from .environment import PROJECT_ROOT, make_env
from .headroom import _run_branch, _snapshot_records, _target_command


METHOD_ORDER = (
    "random_gaze",
    "fixed_scan",
    "entropy_maximization",
    "visual_saliency",
    "decision_centric_gaze",
)


class _EqualBudgetController:
    """Common motion/encoding loop with interchangeable public gaze selection."""

    def __init__(
        self,
        *,
        method: str,
        seed: int,
        head_yaw_limit: float,
        candidates: Sequence[float],
        tolerance_degrees: float,
        scan_dwell_steps: int,
    ) -> None:
        if method not in METHOD_ORDER:
            raise ValueError(f"Unknown EXP-04 gaze method: {method}")
        self.method = method
        self.seed = int(seed)
        self.head_yaw_limit = float(head_yaw_limit)
        self.candidates = tuple(float(value) for value in candidates)
        self.tolerance_degrees = float(tolerance_degrees)
        self.scan_dwell_steps = int(scan_dwell_steps)
        self.encoder = PublicVisualEncoder()
        self.decision_calls = 0
        self.binocular_observations = 0
        self.target_trace: list[float] = []
        self.feature_trace: list[VisualFeatures] = []
        self.last_threat_target: float | None = None
        self.sector_visits = {candidate: 0 for candidate in self.candidates}

    @property
    def description(self) -> Mapping[str, Any]:
        return {
            "information": "public_observation_and_public_history_only",
            "motion_policy": "shared_persistent_threat_rule_v1",
            "gaze": {"kind": self.method},
            "budget": {
                "binocular_observations": self.binocular_observations,
                "image_frames": self.binocular_observations * 2,
                "encoder_calls": self.encoder.calls,
                "model_calls": self.decision_calls,
                "encoder_id": ENCODER_ID,
            },
        }

    def _random_target(self, step: int) -> float:
        payload = f"{self.seed}:{step}:{self.method}".encode("ascii")
        index = int(hashlib.sha256(payload).hexdigest()[:8], 16) % len(self.candidates)
        return self.candidates[index]

    def _fixed_scan_target(self, step: int) -> float:
        dwell = max(self.scan_dwell_steps, 1)
        sweep = (*self.candidates, *reversed(self.candidates[1:-1]))
        return float(sweep[(step // dwell) % len(sweep)])

    def _entropy_target(self, current_yaw: float) -> float:
        # Coverage entropy proxy: select the least-observed sector.  Tie-break
        # by shortest legal travel, then angle.  No candidate image is rendered.
        target = min(
            self.candidates,
            key=lambda value: (self.sector_visits[value], abs(value - current_yaw), value),
        )
        return float(target)

    def _saliency_target(self, features: VisualFeatures, current_yaw: float) -> float:
        if features.threat_score > 0.0:
            return float(
                np.clip(
                    current_yaw - features.bearing * self.head_yaw_limit,
                    -self.head_yaw_limit,
                    self.head_yaw_limit,
                )
            )
        # Deterministic public-image saliency fallback.  It does not query a
        # second view and therefore consumes no free observation.
        direction = -1.0 if features.edge_energy * 1000.0 % 2.0 < 1.0 else 1.0
        return float(direction * self.head_yaw_limit)

    def _decision_target(self, features: VisualFeatures, current_yaw: float) -> float:
        if features.threat_score > 0.0:
            self.last_threat_target = float(
                np.clip(
                    current_yaw - features.bearing * self.head_yaw_limit,
                    -self.head_yaw_limit,
                    self.head_yaw_limit,
                )
            )
        if self.last_threat_target is not None:
            return self.last_threat_target
        return self._entropy_target(current_yaw)

    def _motion(self, features: VisualFeatures) -> tuple[float, float]:
        recent_threat = any(value.threat_score > 0.0 for value in self.feature_trace[-8:])
        if features.threat_score > 0.0 or recent_threat:
            turn = -1.0 if features.bearing > 0.0 else 1.0
            return -0.35, turn
        return 0.55, 0.0

    def __call__(self, env, observation: Mapping[str, np.ndarray], step: int):
        self.binocular_observations += 1
        features = self.encoder.encode_observation(observation)
        self.feature_trace.append(features)
        self.decision_calls += 1
        current_yaw = float(env.head_yaw_degrees)
        if self.method == "random_gaze":
            target = self._random_target(step)
        elif self.method == "fixed_scan":
            target = self._fixed_scan_target(step)
        elif self.method == "entropy_maximization":
            target = self._entropy_target(current_yaw)
        elif self.method == "visual_saliency":
            target = self._saliency_target(features, current_yaw)
        else:
            target = self._decision_target(features, current_yaw)
        nearest = min(self.candidates, key=lambda value: abs(value - target))
        self.sector_visits[nearest] += 1
        self.target_trace.append(float(target))
        forward, body_turn = self._motion(features)
        head_turn = _target_command(
            env, target, tolerance_degrees=self.tolerance_degrees
        )
        action = np.asarray((forward, body_turn, head_turn), dtype=np.float32)
        return action, {
            "decision_update": True,
            "method": self.method,
            "target_degrees": float(target),
            "features": feature_dict(features),
            "encoder_call": self.encoder.calls,
            "model_call": self.decision_calls,
        }


def _registered_budget(
    branch: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    controller_budget = branch["controller"]["budget"]
    steps = len(branch["actions"])
    return {
        **controller_budget,
        "simulator_steps": steps,
        "observation_time_seconds": float(steps * config["environment"]["time_step"]),
        "image_width": int(config["environment"]["vision_width"]),
        "image_height": int(config["environment"]["vision_height"]),
        "eyes_per_observation": 2,
    }


def _assert_equal_budgets(methods: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    budgets = {method: value["budget"] for method, value in methods.items()}
    canonical = next(iter(budgets.values()))
    mismatches = {
        method: budget for method, budget in budgets.items() if budget != canonical
    }
    if mismatches:
        raise RuntimeError(f"EXP-04 unequal compute/observation budgets: {mismatches}")
    return dict(canonical)


def run_exp04_evaluation(
    config: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> Mapping[str, Any]:
    experiment_dir, snapshots = _snapshot_records(config, project_root=project_root)
    settings = config["exp04"]
    env = make_env(config, use_predator=True)
    records: list[dict[str, Any]] = []
    try:
        for snapshot_index, snapshot in enumerate(snapshots):
            if not snapshot["use_predator"]:
                continue
            state = load_state(experiment_dir / snapshot["state_path"])
            methods = {}
            for method_index, method in enumerate(METHOD_ORDER):
                controller = _EqualBudgetController(
                    method=method,
                    seed=int(config["seed"]) + snapshot_index * 1009 + method_index,
                    head_yaw_limit=60.0,
                    candidates=config["gaze_candidates_degrees"],
                    tolerance_degrees=float(settings["target_tolerance_degrees"]),
                    scan_dwell_steps=int(settings["scan_dwell_steps"]),
                )
                branch = _run_branch(
                    env,
                    state,
                    method=method,
                    action_builder=controller,
                    horizon_steps=int(settings["horizon_steps"]),
                    risk_distance=float(settings["risk_distance"]),
                )
                branch["budget"] = _registered_budget(branch, config=config)
                methods[method] = branch
            common_budget = _assert_equal_budgets(methods)
            records.append({
                "snapshot_id": snapshot["snapshot_id"],
                "category": snapshot["category"],
                "methods": methods,
                "budget_equal": True,
                "common_budget": common_budget,
            })
    finally:
        env.close()
    if not records:
        raise RuntimeError("EXP-04 requires at least one predator snapshot")

    rows = []
    for record in records:
        for method, branch in record["methods"].items():
            outcome = branch["outcome"]
            rows.append({
                "snapshot_id": record["snapshot_id"],
                "method": method,
                "safe_success": outcome["safe_success"],
                "captured": outcome["captured"],
                "goal_event": outcome["goal_event"],
                "minimum_predator_distance": outcome["minimum_predator_distance"],
                "total_reward": outcome["total_reward"],
                "budget_equal": record["budget_equal"],
            })
    summaries = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "n": len(selected),
            "safe_success_rate": float(np.mean([row["safe_success"] for row in selected])),
            "capture_rate": float(np.mean([row["captured"] for row in selected])),
            "mean_total_reward": float(np.mean([row["total_reward"] for row in selected])),
        }
    summary = {
        "experiment": "EXP-04 Active Gaze Without Free Compute",
        "snapshots": len(records),
        "all_budgets_equal": all(record["budget_equal"] for record in records),
        "controlled_fields": [
            "image_frames",
            "model_calls",
            "encoder_calls",
            "observation_time_seconds",
            "image_width",
            "image_height",
            "encoder_id",
        ],
        "methods": summaries,
        "evidence_level": "controlled_engineering_probe",
        "research_hypothesis_verified": False,
        "paper_claim_allowed": False,
    }
    write_jsonl(experiment_dir / "exp04.jsonl", records)
    write_csv(experiment_dir / "exp04_outcomes.csv", rows)
    write_json(experiment_dir / "exp04_summary.json", summary)
    return {"records": records, "summary": summary}
