"""EXP-03: Memory Is Not More Frames.

Pairs have byte-identical current public observations.  Their legally generated
public histories differ: one contains a predator that has since left the camera
frustum, while the control never contains a predator.  The diagnostic methods
therefore cannot solve the pair through current-frame object detection.

This is a controlled benchmark generator and engineering evaluation.  It does
not itself establish a paper-level empirical claim.
"""

from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policies.base import PublicHistoryFrame

from .artifacts import (
    prepare_experiment,
    save_observation,
    save_state,
    state_digest,
    write_csv,
    write_json,
    write_jsonl,
)
from .controlled_memory import (
    ENCODER_ID,
    PublicVisualEncoder,
    VisualFeatures,
)
from .environment import PROJECT_ROOT, bearing_degrees, make_env, observe_current, sampling_anchors
from .generator import _candidate_predator_locations, _constructed_state


METHOD_ORDER = (
    "single_frame_reactive",
    "frame_stacking",
    "gru_belief",
    "transformer_history",
    "privileged_belief",
    "vlm_textual_memory",
)


def _experiment_dir(config: Mapping[str, Any], project_root: Path) -> Path:
    root = Path(str(config["output_root"]))
    if not root.is_absolute():
        root = project_root / root
    return root / str(config["experiment_id"])


def _same_public_observation(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key])) for key in left
    )


def _pair_id(config_hash: str, index: int, current: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(config_hash.encode("ascii"))
    digest.update(str(index).encode("ascii"))
    for key in sorted(current):
        digest.update(np.ascontiguousarray(current[key]).tobytes())
    return f"exp03-{index:05d}-{digest.hexdigest()[:16]}"


def _history_frame(observation: Mapping[str, np.ndarray]) -> PublicHistoryFrame:
    return PublicHistoryFrame(
        image_left=observation["image_left"],
        image_right=observation["image_right"],
    )


def _visible_bearing(features: Sequence[VisualFeatures]) -> float | None:
    visible = [feature for feature in features if feature.threat_score > 0.0]
    return float(visible[-1].bearing) if visible else None


def _gru_belief(features: Sequence[VisualFeatures], decay: float) -> tuple[float, float]:
    """Fixed-gate GRU diagnostic: persistent scalar threat and last bearing."""

    belief = 0.0
    bearing = 0.0
    for feature in features:
        evidence = 1.0 if feature.threat_score > 0.0 else 0.0
        update_gate = 1.0 if evidence else float(decay)
        candidate = evidence
        belief = (1.0 - update_gate) * belief + update_gate * candidate
        if evidence:
            bearing = float(feature.bearing)
    return float(belief), bearing


def _transformer_belief(features: Sequence[VisualFeatures]) -> tuple[float, float]:
    """Fixed attention diagnostic over the complete registered history."""

    if not features:
        return 0.0, 0.0
    scores = np.asarray([feature.threat_score for feature in features], dtype=np.float64)
    logits = 40.0 * scores + np.linspace(-0.25, 0.0, len(features))
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    belief = float(np.max(scores) > 0.0)
    bearing = float(sum(weight * feature.bearing for weight, feature in zip(weights, features)))
    return belief, bearing


def _prediction(has_belief: bool, bearing: float) -> str:
    del bearing
    # Construction faces the predator/goal at history start, then makes a
    # legal 180-degree body turn.  At the identical endpoint, forward moves
    # away from a persistent rear threat; backward moves toward the goal in
    # the no-predator control.
    return "forward" if has_belief else "backward"


def _method_predictions(
    current: Mapping[str, np.ndarray],
    history: Sequence[PublicHistoryFrame],
    *,
    correct_action: str,
    frame_stack_k: int,
    gru_decay: float,
    privileged_threat: bool,
) -> dict[str, Mapping[str, Any]]:
    encoder = PublicVisualEncoder()
    history_features = encoder.encode_history(history)
    current_features = encoder.encode_observation(current)
    full = [*history_features, current_features]

    reactive = _prediction(current_features.threat_score > 0.0, current_features.bearing)
    stack = full[-int(frame_stack_k) :]
    stack_bearing = _visible_bearing(stack)
    stack_prediction = _prediction(stack_bearing is not None, stack_bearing or 0.0)
    gru_value, gru_bearing = _gru_belief(full, gru_decay)
    transformer_value, transformer_bearing = _transformer_belief(full)
    text_bearing = _visible_bearing(history_features)
    textual_memory = (
        "threat_last_seen_left"
        if text_bearing is not None and text_bearing < 0
        else "threat_last_seen_right"
        if text_bearing is not None
        else "no_threat_seen"
    )
    predictions = {
        "single_frame_reactive": (reactive, "current_rgb_only"),
        "frame_stacking": (stack_prediction, f"last_{int(frame_stack_k)}_frames"),
        "gru_belief": (_prediction(gru_value >= 0.5, gru_bearing), "fixed_gate_recurrent_probe"),
        "transformer_history": (
            _prediction(transformer_value >= 0.5, transformer_bearing),
            "full_history_attention_probe",
        ),
        "privileged_belief": (
            _prediction(privileged_threat, text_bearing or 0.0),
            "privileged_threat_presence_upper_reference",
        ),
        "vlm_textual_memory": (
            _prediction(text_bearing is not None, text_bearing or 0.0),
            "public_visual_detection_to_registered_text_memory",
        ),
    }
    return {
        method: {
            "prediction": prediction,
            "correct": prediction == correct_action,
            "information": information,
            "encoder_id": ENCODER_ID if method != "privileged_belief" else None,
            "text_memory": textual_memory if method == "vlm_textual_memory" else None,
        }
        for method, (prediction, information) in predictions.items()
    }


def _build_pair(
    predator_env,
    control_env,
    *,
    config: Mapping[str, Any],
    anchor: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    values = config["exp03"]
    random.seed(seed)
    np.random.seed(seed)
    predator_env.reset(seed=seed)
    control_env.reset(seed=seed)
    predator_base = predator_env.get_state_dict()
    control_base = control_env.get_state_dict()
    rng = np.random.default_rng(seed)
    candidates = _candidate_predator_locations(
        predator_env, anchor["location"], config["sampling"], rng
    )
    goal_location = predator_env.unwrapped.model.goal_location
    goal_bearing = bearing_degrees(anchor["location"], goal_location)
    for _, predator_location, distance, geometric_los in candidates:
        if not geometric_los:
            continue
        heading = bearing_degrees(anchor["location"], predator_location)
        alignment = abs((heading - goal_bearing + 180.0) % 360.0 - 180.0)
        if alignment > float(values["goal_alignment_degrees"]):
            continue
        threat_state = _constructed_state(
            predator_base,
            prey_location=anchor["location"],
            body_heading_degrees=heading,
            predator_location=predator_location,
        )
        control_state = _constructed_state(
            control_base,
            prey_location=anchor["location"],
            body_heading_degrees=heading,
            predator_location=None,
        )
        predator_env.set_state_dict(copy.deepcopy(threat_state))
        control_env.set_state_dict(copy.deepcopy(control_state))
        # Reject geometrically plausible but raster-invisible candidates before
        # paying for a complete paired legal trajectory.
        initial_observation = observe_current(predator_env)
        initial_features = PublicVisualEncoder().encode_observation(initial_observation)
        if initial_features.threat_score <= 0.0:
            continue
        threat_history: list[Mapping[str, np.ndarray]] = []
        control_history: list[Mapping[str, np.ndarray]] = []
        was_visible = True
        ended = False
        turn = float(values["turn_command"])
        action = np.asarray((0.0, turn, 0.0), dtype=np.float32)
        for _ in range(int(values["history_steps"])):
            threat_observation = observe_current(predator_env)
            control_observation = observe_current(control_env)
            threat_history.append(threat_observation)
            control_history.append(control_observation)
            was_visible = was_visible or bool(
                predator_env.get_predator_visibility()["predator_pixels_visible"]
            )
            _, _, t1, x1, _ = predator_env.step(action)
            _, _, t2, x2, _ = control_env.step(action)
            ended = ended or t1 or x1 or t2 or x2
            if ended:
                break
        if ended or not was_visible:
            continue
        threat_current = observe_current(predator_env)
        control_current = observe_current(control_env)
        if predator_env.get_predator_visibility()["predator_pixels_visible"]:
            continue
        if not _same_public_observation(threat_current, control_current):
            continue
        if any(
            PublicVisualEncoder().encode_observation(value).threat_score > 0.0
            for value in threat_history[-int(values["frame_stack_k"]) :]
        ):
            continue
        public_frames = [_history_frame(value) for value in threat_history]
        bearing = _visible_bearing(PublicVisualEncoder().encode_history(public_frames))
        if bearing is None:
            continue
        return {
            "threat": {
                "state": predator_env.get_state_dict(),
                "current": threat_current,
                "history": threat_history,
                "correct_action": "forward",
                "privileged": {
                    "threat_present": True,
                    "predator_distance": float(distance),
                    "predator_goal_bearing_difference_degrees": float(alignment),
                },
            },
            "control": {
                "state": control_env.get_state_dict(),
                "current": control_current,
                "history": control_history,
                "correct_action": "backward",
                "privileged": {"threat_present": False, "predator_distance": None},
            },
        }
    return None


def run_exp03_evaluation(
    config: Mapping[str, Any], *, project_root: Path = PROJECT_ROOT
) -> Mapping[str, Any]:
    experiment_dir = prepare_experiment(config, project_root=project_root)
    predator_env = make_env(config, use_predator=True)
    control_env = make_env(config, use_predator=False)
    records: list[dict[str, Any]] = []
    try:
        predator_env.reset(seed=int(config["seed"]))
        anchors_by_source = sampling_anchors(predator_env)
        anchors = [
            anchor
            for source in config["sampling"]["sources"]
            for anchor in anchors_by_source[source]
        ]
        attempts = 0
        anchor_index = 0
        while len(records) < int(config["num_snapshots"]):
            if attempts >= int(config["exp03"]["candidate_search_limit"]):
                raise RuntimeError(
                    f"EXP-03 found only {len(records)} valid identical-current pairs "
                    f"after {attempts} attempts"
                )
            anchor = anchors[anchor_index % len(anchors)]
            anchor_index += 1
            seed = int(config["seed"]) + attempts * 1009
            attempts += 1
            pair = _build_pair(
                predator_env,
                control_env,
                config=config,
                anchor=anchor,
                seed=seed,
            )
            if pair is None:
                continue
            pair_id = _pair_id(config["config_hash"], len(records), pair["threat"]["current"])
            pair_dir = experiment_dir / "exp03_pairs" / pair_id
            conditions = {}
            for condition in ("threat", "control"):
                value = pair[condition]
                state_path = pair_dir / f"{condition}_state.json.gz"
                current_path = pair_dir / f"{condition}_current.npz"
                save_state(state_path, value["state"])
                save_observation(current_path, value["current"])
                history_paths = []
                for index, observation in enumerate(value["history"]):
                    path = pair_dir / f"{condition}_history_{index:03d}.npz"
                    save_observation(path, observation)
                    history_paths.append(str(path.relative_to(experiment_dir)))
                frames = [_history_frame(observation) for observation in value["history"]]
                conditions[condition] = {
                    "correct_action": value["correct_action"],
                    "state_hash": state_digest(value["state"]),
                    "state_path": str(state_path.relative_to(experiment_dir)),
                    "current_path": str(current_path.relative_to(experiment_dir)),
                    "history_paths": history_paths,
                    "privileged": value["privileged"],
                    "methods": _method_predictions(
                        value["current"],
                        frames,
                        correct_action=value["correct_action"],
                        frame_stack_k=int(config["exp03"]["frame_stack_k"]),
                        gru_decay=float(config["exp03"]["gru_decay"]),
                        privileged_threat=bool(value["privileged"]["threat_present"]),
                    ),
                }
            record = {
                "pair_id": pair_id,
                "seed": seed,
                "source": anchor["source"],
                "source_cell_id": int(anchor["cell_id"]),
                "current_observation_byte_identical": True,
                "correct_actions_differ": (
                    conditions["threat"]["correct_action"]
                    != conditions["control"]["correct_action"]
                ),
                "history_steps": int(config["exp03"]["history_steps"]),
                "frame_stack_k": int(config["exp03"]["frame_stack_k"]),
                "conditions": conditions,
            }
            records.append(record)
    finally:
        predator_env.close()
        control_env.close()

    rows = []
    for record in records:
        for condition, value in record["conditions"].items():
            for method, result in value["methods"].items():
                rows.append({
                    "pair_id": record["pair_id"],
                    "condition": condition,
                    "method": method,
                    "prediction": result["prediction"],
                    "correct_action": value["correct_action"],
                    "correct": result["correct"],
                })
    method_summary = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        pair_success = [
            all(row["correct"] for row in selected if row["pair_id"] == record["pair_id"])
            for record in records
        ]
        method_summary[method] = {
            "condition_accuracy": float(np.mean([row["correct"] for row in selected])),
            "paired_both_correct_rate": float(np.mean(pair_success)),
            "n_conditions": len(selected),
        }
    summary = {
        "experiment": "EXP-03 Memory Is Not More Frames",
        "pairs": len(records),
        "all_current_observations_byte_identical": all(
            record["current_observation_byte_identical"] for record in records
        ),
        "all_correct_actions_differ_within_pair": all(
            record["correct_actions_differ"] for record in records
        ),
        "methods": method_summary,
        "evidence_level": "controlled_engineering_probe",
        "research_hypothesis_verified": False,
        "paper_claim_allowed": False,
    }
    write_jsonl(experiment_dir / "exp03.jsonl", records)
    write_csv(experiment_dir / "exp03_predictions.csv", rows)
    write_json(experiment_dir / "exp03_summary.json", summary)
    return {"records": records, "summary": summary}
