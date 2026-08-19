"""Validate and summarize Mouse First-Person Lab human demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.peekbench.artifacts import write_csv, write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "human_demos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "human_demo_audit"

REQUIRED_ARRAYS = {
    "image_left": (np.dtype(np.uint8), 4),
    "image_right": (np.dtype(np.uint8), 4),
    "proprio": (np.dtype(np.float32), 2),
    "previous_action": (np.dtype(np.float32), 2),
    "action": (np.dtype(np.float32), 2),
    "reward": (np.dtype(np.float32), 1),
    "terminated": (np.dtype(np.bool_), 1),
    "truncated": (np.dtype(np.bool_), 1),
    "sim_time": (np.dtype(np.float32), 1),
    "privileged_state": (np.dtype(np.float32), 2),
}

EVENT_DTYPES = {
    "puffed": np.dtype(np.bool_),
    "capture_event": np.dtype(np.bool_),
    "goal_event": np.dtype(np.bool_),
    "capture_count": np.dtype(np.int32),
    "predator_sees_prey": np.dtype(np.bool_),
    "prey_sees_predator": np.dtype(np.bool_),
    "goal_achieved": np.dtype(np.bool_),
    "prey_predator_distance": np.dtype(np.float32),
    "predator_geometric_los": np.dtype(np.bool_),
    "predator_in_left_frustum": np.dtype(np.bool_),
    "predator_in_right_frustum": np.dtype(np.bool_),
    "predator_pixels_visible": np.dtype(np.bool_),
    "predator_within_detection_range": np.dtype(np.bool_),
    "predator_believed_visible": np.dtype(np.bool_),
    "minimum_distance": np.dtype(np.float32),
}


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    session: str
    episode: str | None
    message: str


def _safe_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _mean_or_none(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _first_after(mask: np.ndarray, start: int) -> int | None:
    indices = np.flatnonzero(mask[start:])
    return int(start + indices[0]) if len(indices) else None


def _event_latencies(
    event_mask: np.ndarray,
    response_mask: np.ndarray,
    sim_time: np.ndarray,
) -> list[float]:
    rising = np.flatnonzero(event_mask & ~np.r_[False, event_mask[:-1]])
    latencies = []
    for index in rising:
        response = _first_after(response_mask, int(index))
        if response is not None:
            latencies.append(float(sim_time[response] - sim_time[index]))
    return latencies


def _reconfirmation_intervals(
    visible: np.ndarray,
    sim_time: np.ndarray,
) -> list[float]:
    falling = np.flatnonzero(~visible & np.r_[False, visible[:-1]])
    intervals = []
    for index in falling:
        reconfirm = _first_after(visible, int(index) + 1)
        if reconfirm is not None:
            intervals.append(float(sim_time[reconfirm] - sim_time[index]))
    return intervals


def _unnecessary_look_mask(
    look: np.ndarray,
    visible: np.ndarray,
    geometric_los: np.ndarray,
    context_frames: int,
) -> np.ndarray:
    threat_context = visible | geometric_los
    if context_frames > 0 and len(threat_context):
        kernel = np.ones(context_frames * 2 + 1, dtype=np.int32)
        threat_context = np.convolve(
            threat_context.astype(np.int32),
            kernel,
            mode="same",
        ) > 0
    return look & ~threat_context


def _trajectory_metrics(
    data: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    success: bool,
) -> Mapping[str, float | None]:
    names = list(metadata.get("privileged_state_names", []))
    if not names or "prey_x" not in names or "prey_y" not in names:
        return {
            "path_length": None,
            "path_efficiency": None,
            "progress_efficiency": None,
        }
    privileged = data["privileged_state"]
    x_index = names.index("prey_x")
    y_index = names.index("prey_y")
    positions = privileged[:, [x_index, y_index]].astype(np.float64)
    if len(positions) < 2 or not np.isfinite(positions).all():
        return {
            "path_length": 0.0 if len(positions) else None,
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


def _split_for_group(group: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def _episode_files(
    session_dir: Path,
    metadata: Mapping[str, Any],
) -> list[Path]:
    listed = [
        session_dir / str(record["file"])
        for record in metadata.get("episodes", [])
        if isinstance(record, Mapping) and record.get("file")
    ]
    return listed if listed else sorted(session_dir.glob("episode_*.npz"))


def _validate_arrays(
    data: Mapping[str, np.ndarray],
    *,
    session_name: str,
    episode_name: str,
) -> tuple[int, list[AuditIssue]]:
    issues = []
    missing = sorted(set(REQUIRED_ARRAYS).difference(data))
    for name in missing:
        issues.append(AuditIssue("error", session_name, episode_name, f"Missing array: {name}"))
    if missing:
        return 0, issues
    length = int(len(data["action"]))
    if length <= 0:
        issues.append(AuditIssue("error", session_name, episode_name, "Episode has zero transitions"))
    for name, (dtype, ndim) in REQUIRED_ARRAYS.items():
        array = data[name]
        if len(array) != length:
            issues.append(
                AuditIssue(
                    "error",
                    session_name,
                    episode_name,
                    f"{name} length {len(array)} != action length {length}",
                ),
            )
        if array.dtype != dtype:
            issues.append(
                AuditIssue(
                    "error",
                    session_name,
                    episode_name,
                    f"{name} dtype {array.dtype} != {dtype}",
                ),
            )
        if array.ndim != ndim:
            issues.append(
                AuditIssue(
                    "error",
                    session_name,
                    episode_name,
                    f"{name} ndim {array.ndim} != {ndim}",
                ),
            )
    for image_name in ("image_left", "image_right"):
        if data[image_name].ndim == 4 and data[image_name].shape[-1] != 3:
            issues.append(
                AuditIssue("error", session_name, episode_name, f"{image_name} is not HWC RGB"),
            )
    if data["proprio"].ndim == 2 and data["proprio"].shape[1] != 3:
        issues.append(AuditIssue("error", session_name, episode_name, "proprio must have shape (T, 3)"))
    if data["action"].ndim == 2 and data["action"].shape[1] not in (2, 3):
        issues.append(AuditIssue("error", session_name, episode_name, "action must have 2 or 3 columns"))
    if data["previous_action"].shape != data["action"].shape:
        issues.append(
            AuditIssue("error", session_name, episode_name, "previous_action shape differs from action"),
        )
    elif length > 1 and not np.allclose(
        data["previous_action"][1:],
        data["action"][:-1],
        atol=1e-6,
    ):
        issues.append(
            AuditIssue(
                "error",
                session_name,
                episode_name,
                "observation_t previous_action is not aligned with action_(t-1)",
            ),
        )
    done = data["terminated"] | data["truncated"]
    if length > 1 and bool(done[:-1].any()):
        issues.append(AuditIssue("error", session_name, episode_name, "done occurs before final row"))
    for name, dtype in EVENT_DTYPES.items():
        if name not in data:
            issues.append(AuditIssue("warning", session_name, episode_name, f"Missing event array: {name}"))
            continue
        if len(data[name]) != length:
            issues.append(AuditIssue("error", session_name, episode_name, f"{name} is not transition-aligned"))
        if data[name].dtype != dtype:
            issues.append(AuditIssue("error", session_name, episode_name, f"{name} dtype {data[name].dtype} != {dtype}"))
    return length, issues


def _episode_metrics(
    data: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    episode_metadata: Mapping[str, Any],
    look_threshold: float,
    body_threshold: float,
    context_frames: int,
) -> Mapping[str, Any]:
    length = len(data["action"])
    action = data["action"]
    sim_time = data["sim_time"].astype(np.float64)
    look = np.abs(action[:, 2]) > look_threshold if action.shape[1] >= 3 else np.zeros(length, dtype=bool)
    body = np.abs(action[:, 1]) > body_threshold
    backward = action[:, 0] < -body_threshold
    response = look | body | backward
    visible = np.asarray(
        data.get("predator_pixels_visible", np.zeros(length, dtype=bool)),
        dtype=bool,
    )
    geometric = np.asarray(
        data.get("predator_geometric_los", np.zeros(length, dtype=bool)),
        dtype=bool,
    )
    first_look = int(np.flatnonzero(look)[0]) if look.any() else None
    appearance_latencies = _event_latencies(visible, response, sim_time)
    reconfirmation = _reconfirmation_intervals(visible, sim_time)
    unnecessary = _unnecessary_look_mask(look, visible, geometric, context_frames)
    capture_events = int(np.asarray(data.get("capture_event", np.zeros(length))).sum())
    capture_count = int(
        np.asarray(data.get("capture_count", np.zeros(length, dtype=np.int32)))[-1]
        if length
        else 0
    )
    success = bool(episode_metadata.get("is_success", False))
    trajectory = _trajectory_metrics(data, metadata, success)
    safety_efficiency = None
    if trajectory["path_efficiency"] is not None:
        safety_efficiency = float(trajectory["path_efficiency"]) * float(capture_count == 0)
    duration = float(sim_time[-1] - sim_time[0]) if length > 1 else 0.0
    return {
        "steps": int(length),
        "duration_seconds": duration,
        "return": float(data["reward"].sum()),
        "success": success,
        "capture_events": capture_events,
        "capture_count": capture_count,
        "look_frequency": float(look.mean()) if length else 0.0,
        "body_head_decoupling": float((look & ~body).mean()) if length else 0.0,
        "time_to_first_look": (
            float(sim_time[first_look] - sim_time[0]) if first_look is not None else None
        ),
        "reaction_time_after_predator_appearance": _mean_or_none(appearance_latencies),
        "predator_appearance_events": int(
            np.count_nonzero(visible & ~np.r_[False, visible[:-1]]),
        ),
        "occlusion_reconfirmation_interval": _mean_or_none(reconfirmation),
        "reconfirmation_events": len(reconfirmation),
        "unnecessary_look_rate": (
            float(unnecessary.sum() / max(int(look.sum()), 1)) if length else 0.0
        ),
        **trajectory,
        "safety_efficiency": safety_efficiency,
    }


def audit_human_demos(
    data_root: Path,
    output_dir: Path,
    *,
    split_seed: int = 23,
    look_threshold: float = 0.10,
    body_threshold: float = 0.10,
    unnecessary_context_frames: int = 10,
) -> Mapping[str, Any]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_files = sorted(data_root.glob("**/session.json")) if data_root.exists() else []
    if not session_files:
        result = {
            "status": "no_data",
            "message": f"No human demonstration sessions found under {data_root}",
            "sessions": 0,
            "episodes": 0,
            "errors": 0,
            "warnings": 0,
        }
        write_json(output_dir / "audit_summary.json", result)
        write_csv(output_dir / "episode_summary.csv", [])
        write_csv(output_dir / "session_summary.csv", [])
        write_csv(output_dir / "split_manifest.csv", [])
        write_jsonl(output_dir / "audit_issues.jsonl", [])
        print(result["message"])
        return result

    issues: list[AuditIssue] = []
    episodes: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []
    for session_path in session_files:
        session_dir = session_path.parent
        session_name = session_dir.name
        try:
            metadata = _safe_json(session_path)
        except Exception as error:
            issues.append(AuditIssue("error", session_name, None, f"Invalid session.json: {error}"))
            continue
        if metadata.get("transition_convention") != "observation_t, action_t, reward_t, done_t":
            issues.append(
                AuditIssue(
                    "error",
                    session_name,
                    None,
                    "Unexpected or missing transition_convention",
                ),
            )
        try:
            format_version = int(metadata.get("format_version", -1))
        except (TypeError, ValueError):
            format_version = -1
        if format_version < 3:
            issues.append(
                AuditIssue(
                    "error",
                    session_name,
                    None,
                    f"Unsupported or missing format_version: {format_version}",
                ),
            )
        participant = str(
            metadata.get("participant_id", metadata.get("participant", "unknown")),
        )
        if participant == "unknown":
            issues.append(
                AuditIssue(
                    "warning",
                    session_name,
                    None,
                    "No participant_id; all unknown-participant sessions stay in one split group.",
                ),
            )
        group = participant if participant != "unknown" else "unknown_participant"
        split = _split_for_group(group, split_seed)
        split_manifest.append(
            {
                "participant_id": participant,
                "session": session_name,
                "group_key": group,
                "split": split,
                "split_unit": "participant_then_session; never frame",
            },
        )
        episode_metadata_by_file = {
            str(item.get("file")): item
            for item in metadata.get("episodes", [])
            if isinstance(item, Mapping)
        }
        for episode_path in _episode_files(session_dir, metadata):
            episode_name = episode_path.name
            if not episode_path.exists():
                issues.append(AuditIssue("error", session_name, episode_name, "Listed NPZ does not exist"))
                continue
            try:
                with np.load(episode_path, allow_pickle=False) as archive:
                    data = {name: np.array(archive[name], copy=True) for name in archive.files}
            except Exception as error:
                issues.append(AuditIssue("error", session_name, episode_name, f"Cannot read NPZ safely: {error}"))
                continue
            length, array_issues = _validate_arrays(
                data,
                session_name=session_name,
                episode_name=episode_name,
            )
            issues.extend(array_issues)
            if length <= 0 or any(issue.severity == "error" for issue in array_issues):
                continue
            expected_eye_shape = metadata.get("eye_shape")
            if expected_eye_shape is not None and list(data["image_left"].shape[1:]) != list(expected_eye_shape):
                issues.append(
                    AuditIssue(
                        "error",
                        session_name,
                        episode_name,
                        "session eye_shape differs from NPZ image shape",
                    ),
                )
            action_names = list(metadata.get("action_names", []))
            if action_names and len(action_names) != data["action"].shape[1]:
                issues.append(
                    AuditIssue(
                        "error",
                        session_name,
                        episode_name,
                        "session action_names length differs from action columns",
                    ),
                )
            proprio_names = list(metadata.get("proprio_names", []))
            if proprio_names and len(proprio_names) != data["proprio"].shape[1]:
                issues.append(
                    AuditIssue(
                        "error",
                        session_name,
                        episode_name,
                        "session proprio_names length differs from proprio columns",
                    ),
                )
            sidecar_path = episode_path.with_suffix(".json")
            episode_metadata = episode_metadata_by_file.get(episode_name, {})
            if sidecar_path.exists():
                try:
                    episode_metadata = {**episode_metadata, **_safe_json(sidecar_path)}
                except Exception as error:
                    issues.append(AuditIssue("error", session_name, episode_name, f"Invalid sidecar JSON: {error}"))
            if episode_metadata.get("steps") is not None and int(episode_metadata["steps"]) != length:
                issues.append(AuditIssue("error", session_name, episode_name, "Sidecar steps differs from NPZ length"))
            metrics = _episode_metrics(
                data,
                metadata=metadata,
                episode_metadata=episode_metadata,
                look_threshold=look_threshold,
                body_threshold=body_threshold,
                context_frames=unnecessary_context_frames,
            )
            episodes.append(
                {
                    "participant_id": participant,
                    "session": session_name,
                    "episode": episode_name,
                    "split": split,
                    **metrics,
                },
            )

    session_rows = []
    for session_name in sorted({episode["session"] for episode in episodes}):
        rows = [episode for episode in episodes if episode["session"] == session_name]
        session_rows.append(
            {
                "participant_id": rows[0]["participant_id"],
                "session": session_name,
                "split": rows[0]["split"],
                "episodes": len(rows),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "capture_episode_rate": float(np.mean([row["capture_count"] > 0 for row in rows])),
                "mean_look_frequency": _mean_or_none([row["look_frequency"] for row in rows]),
                "mean_body_head_decoupling": _mean_or_none([row["body_head_decoupling"] for row in rows]),
                "mean_path_efficiency": _mean_or_none([
                    row["path_efficiency"]
                    for row in rows
                    if row["path_efficiency"] is not None
                ]),
                "mean_safety_efficiency": _mean_or_none([
                    row["safety_efficiency"]
                    for row in rows
                    if row["safety_efficiency"] is not None
                ]),
            },
        )

    issue_records = [asdict(issue) for issue in issues]
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    summary = {
        "status": "ok" if error_count == 0 else "validation_errors",
        "data_root": str(data_root),
        "sessions": len(session_files),
        "episodes": len(episodes),
        "participants": sorted({row["participant_id"] for row in split_manifest}),
        "errors": error_count,
        "warnings": warning_count,
        "aggregate": {
            "success_rate": _mean_or_none([episode["success"] for episode in episodes]),
            "capture_episode_rate": _mean_or_none([
                episode["capture_count"] > 0 for episode in episodes
            ]),
            "look_frequency": _mean_or_none([episode["look_frequency"] for episode in episodes]),
            "body_head_decoupling": _mean_or_none([
                episode["body_head_decoupling"] for episode in episodes
            ]),
            "time_to_first_look": _mean_or_none([
                episode["time_to_first_look"]
                for episode in episodes
                if episode["time_to_first_look"] is not None
            ]),
            "reaction_time_after_predator_appearance": _mean_or_none([
                episode["reaction_time_after_predator_appearance"]
                for episode in episodes
                if episode["reaction_time_after_predator_appearance"] is not None
            ]),
            "occlusion_reconfirmation_interval": _mean_or_none([
                episode["occlusion_reconfirmation_interval"]
                for episode in episodes
                if episode["occlusion_reconfirmation_interval"] is not None
            ]),
            "unnecessary_look_rate": _mean_or_none([
                episode["unnecessary_look_rate"] for episode in episodes
            ]),
            "path_efficiency": _mean_or_none([
                episode["path_efficiency"]
                for episode in episodes
                if episode["path_efficiency"] is not None
            ]),
            "safety_efficiency": _mean_or_none([
                episode["safety_efficiency"]
                for episode in episodes
                if episode["safety_efficiency"] is not None
            ]),
        },
        "metric_notes": {
            "unnecessary_look_rate": (
                "Fraction of look-command frames outside a +/-context window of "
                "privileged pixel visibility or geometric LOS; descriptive proxy only."
            ),
            "path_efficiency": (
                "Straight start-to-goal distance divided by recorded path length, "
                "reported only for successful episodes."
            ),
            "safety_efficiency": (
                "path_efficiency multiplied by an indicator of zero captures; "
                "not a validated composite endpoint."
            ),
        },
    }
    write_csv(output_dir / "episode_summary.csv", episodes)
    write_csv(output_dir / "session_summary.csv", session_rows)
    write_csv(output_dir / "split_manifest.csv", split_manifest)
    write_jsonl(output_dir / "audit_issues.jsonl", issue_records)
    write_json(output_dir / "audit_summary.json", summary)
    print(
        f"Human demo audit: status={summary['status']} sessions={summary['sessions']} "
        f"episodes={summary['episodes']} errors={error_count} warnings={warning_count}",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--look-threshold", type=float, default=0.10)
    parser.add_argument("--body-threshold", type=float, default=0.10)
    parser.add_argument("--unnecessary-context-frames", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_human_demos(
        args.data_root,
        args.output_dir,
        split_seed=args.split_seed,
        look_threshold=args.look_threshold,
        body_threshold=args.body_threshold,
        unnecessary_context_frames=args.unnecessary_context_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
