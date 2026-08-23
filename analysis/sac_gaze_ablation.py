"""EXP-05: active-gaze versus fixed-camera ablations for a frozen SAC policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from stable_baselines3 import SAC

# This import must precede benchmark.environment: botevade_gym selects the
# repository's offline Cellworld cache before cellworld.util is imported.
from training.first_person_sac import PROJECT_ROOT, load_sac_config, make_first_person_env
from benchmarks.peekbench.artifacts import (
    environment_metadata,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from benchmarks.peekbench.environment import observe_current, state_with_gaze
from benchmarks.peekbench.headroom import _target_command


METHODS = (
    "sac_active_gaze",
    "sac_fixed_0",
    "sac_fixed_p60",
    "sac_fixed_m60",
    "sac_fixed_p30",
    "sac_fixed_scan",
)
METHOD_LABELS = {
    "sac_active_gaze": "Learned active gaze",
    "sac_fixed_0": "Fixed 0 degrees",
    "sac_fixed_p60": "Fixed +60 degrees",
    "sac_fixed_m60": "Fixed -60 degrees",
    "sac_fixed_p30": "Fixed +30 degrees",
    "sac_fixed_scan": "Fixed scan",
}
FIXED_HEAD_YAWS = {
    "sac_fixed_0": 0.0,
    "sac_fixed_p60": 60.0,
    "sac_fixed_m60": -60.0,
    "sac_fixed_p30": 30.0,
}
PRIMARY_COMPARATOR = "sac_fixed_p60"
SCAN_TARGETS_DEGREES = (-60.0, -30.0, 0.0, 30.0, 60.0, 30.0, 0.0, -30.0)
SCAN_DWELL_STEPS = 2
SCAN_TOLERANCE_DEGREES = 2.0
PAIRED_FIELDS = (
    "clean_success",
    "capture_episode",
    "goal_reached",
    "steps",
    "minimum_predator_distance",
    "path_cost",
    "predator_pixels_visible_fraction",
    "mean_head_yaw_degrees",
    "gaze_travel_degrees",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_scan_target(step: int) -> float:
    index = (int(step) // SCAN_DWELL_STEPS) % len(SCAN_TARGETS_DEGREES)
    return SCAN_TARGETS_DEGREES[index]


def _initialize_fixed_gaze(env, method: str) -> Mapping[str, np.ndarray]:
    target = FIXED_HEAD_YAWS[method]
    env.head_recenter_rate = 0.0
    state = state_with_gaze(env.get_state_dict(), target)
    env.set_state_dict(state)
    return observe_current(env)


def _policy_action(
    env,
    model: SAC,
    observation: Mapping[str, np.ndarray],
    *,
    method: str,
    step: int,
) -> np.ndarray:
    action, _ = model.predict(observation, deterministic=True)
    action = np.asarray(action, dtype=np.float32)
    if action.shape != env.action_space.shape:
        raise ValueError(f"{method} emitted {action.shape}, expected {env.action_space.shape}")
    action = np.array(action, copy=True)
    if method in FIXED_HEAD_YAWS:
        action[2] = 0.0
    elif method == "sac_fixed_scan":
        action[2] = _target_command(
            env,
            fixed_scan_target(step),
            tolerance_degrees=SCAN_TOLERANCE_DEGREES,
        )
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def run_episode(env, model: SAC, *, method: str, seed: int) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown EXP-05 method: {method}")
    observation, _ = env.reset(seed=int(seed))
    if method in FIXED_HEAD_YAWS:
        observation = _initialize_fixed_gaze(env, method)

    model_state = env.unwrapped.model
    previous_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
    previous_head_yaw = float(env.head_yaw_degrees)
    minimum_distance = float(
        np.linalg.norm(
            previous_location
            - np.asarray(model_state.predator.state.location, dtype=np.float64),
        ),
    )
    total_reward = 0.0
    path_cost = 0.0
    gaze_travel = 0.0
    capture_count = 0
    goal_reached = False
    visible_steps = 0
    absolute_head_action = 0.0
    head_yaw_sum = 0.0
    minimum_head_yaw = previous_head_yaw
    maximum_head_yaw = previous_head_yaw
    maximum_fixed_yaw_error = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = _policy_action(env, model, observation, method=method, step=steps)
        absolute_head_action += abs(float(action[2]))
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        events = info["transition_events"]
        captured = bool(events["capture_event"])
        capture_count += int(captured)
        goal_reached = bool(goal_reached or events["goal_event"])
        visible_steps += int(events["predator_pixels_visible"])
        minimum_distance = min(minimum_distance, float(events["minimum_distance"]))

        current_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
        path_cost += float(np.linalg.norm(current_location - previous_location))
        previous_location = current_location
        current_head_yaw = float(env.head_yaw_degrees)
        gaze_travel += abs(current_head_yaw - previous_head_yaw)
        previous_head_yaw = current_head_yaw
        head_yaw_sum += current_head_yaw
        minimum_head_yaw = min(minimum_head_yaw, current_head_yaw)
        maximum_head_yaw = max(maximum_head_yaw, current_head_yaw)
        if method in FIXED_HEAD_YAWS:
            maximum_fixed_yaw_error = max(
                maximum_fixed_yaw_error,
                abs(current_head_yaw - FIXED_HEAD_YAWS[method]),
            )

    return {
        "method": method,
        "seed": int(seed),
        "steps": steps,
        "return": total_reward,
        "goal_reached": goal_reached,
        "clean_success": bool(goal_reached and capture_count == 0),
        "capture_count": capture_count,
        "capture_episode": bool(capture_count > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "minimum_predator_distance": minimum_distance,
        "final_goal_distance": float(env.unwrapped.reward_terms["goal_distance"]),
        "path_cost": path_cost,
        "gaze_travel_degrees": gaze_travel,
        "mean_absolute_head_action": absolute_head_action / steps if steps else 0.0,
        "predator_pixels_visible_fraction": visible_steps / steps if steps else 0.0,
        "initial_head_yaw_degrees": (
            FIXED_HEAD_YAWS[method] if method in FIXED_HEAD_YAWS else 0.0
        ),
        "mean_head_yaw_degrees": head_yaw_sum / steps if steps else previous_head_yaw,
        "minimum_head_yaw_degrees": minimum_head_yaw,
        "maximum_head_yaw_degrees": maximum_head_yaw,
        "maximum_fixed_yaw_error_degrees": (
            maximum_fixed_yaw_error if method in FIXED_HEAD_YAWS else None
        ),
    }


def run_method(
    config: Mapping[str, Any],
    *,
    model_path: Path,
    method: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    model = SAC.load(model_path, device="cpu")
    env = make_first_person_env(config)
    try:
        return [run_episode(env, model, method=method, seed=int(seed)) for seed in seeds]
    finally:
        env.close()


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total),
    ) / denominator
    return centre - margin, centre + margin


def summarize_methods(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        if not subset:
            continue
        success_count = sum(bool(record["clean_success"]) for record in subset)
        capture_count = sum(bool(record["capture_episode"]) for record in subset)
        success_interval = wilson_interval(success_count, len(subset))
        capture_interval = wilson_interval(capture_count, len(subset))

        def mean(field: str) -> float:
            return float(np.mean([float(record[field]) for record in subset]))

        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "episodes": len(subset),
                "clean_success_rate": success_count / len(subset),
                "clean_success_ci_low": success_interval[0],
                "clean_success_ci_high": success_interval[1],
                "capture_episode_rate": capture_count / len(subset),
                "capture_episode_ci_low": capture_interval[0],
                "capture_episode_ci_high": capture_interval[1],
                "goal_reach_rate": mean("goal_reached"),
                "mean_steps": mean("steps"),
                "mean_minimum_predator_distance": mean("minimum_predator_distance"),
                "mean_path_cost": mean("path_cost"),
                "mean_predator_pixels_visible_fraction": mean(
                    "predator_pixels_visible_fraction",
                ),
                "mean_head_yaw_degrees": mean("mean_head_yaw_degrees"),
                "mean_gaze_travel_degrees": mean("gaze_travel_degrees"),
            },
        )
    return rows


def exact_mcnemar_p(active_only: int, comparator_only: int) -> float:
    discordant = int(active_only) + int(comparator_only)
    if discordant == 0:
        return 1.0
    lower = min(int(active_only), int(comparator_only))
    lower_tail = math.ldexp(sum(math.comb(discordant, k) for k in range(lower + 1)), -discordant)
    return min(1.0, 2.0 * lower_tail)


def paired_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    by_method = {
        method: {
            int(record["seed"]): record
            for record in records
            if record["method"] == method
        }
        for method in METHODS
    }
    rows = []
    for comparator_index, comparator in enumerate(METHODS[1:]):
        seeds = sorted(set(by_method["sac_active_gaze"]).intersection(by_method[comparator]))
        for field_index, field in enumerate(PAIRED_FIELDS):
            differences = np.asarray(
                [
                    float(by_method["sac_active_gaze"][seed][field])
                    - float(by_method[comparator][seed][field])
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            rng = np.random.default_rng(20260823 + comparator_index * 100 + field_index)
            bootstrap_means = np.empty((int(bootstrap_samples),), dtype=np.float64)
            for sample_index in range(int(bootstrap_samples)):
                indices = rng.integers(0, len(differences), size=len(differences))
                bootstrap_means[sample_index] = differences[indices].mean()
            low, high = np.percentile(bootstrap_means, (2.5, 97.5))
            row = {
                "left_method": "sac_active_gaze",
                "right_method": comparator,
                "field": field,
                "paired_episodes": len(seeds),
                "mean_delta": float(differences.mean()),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "active_only_successes": None,
                "comparator_only_successes": None,
                "mcnemar_exact_p": None,
            }
            if field == "clean_success":
                active_only = sum(
                    bool(by_method["sac_active_gaze"][seed][field])
                    and not bool(by_method[comparator][seed][field])
                    for seed in seeds
                )
                comparator_only = sum(
                    bool(by_method[comparator][seed][field])
                    and not bool(by_method["sac_active_gaze"][seed][field])
                    for seed in seeds
                )
                row.update(
                    {
                        "active_only_successes": active_only,
                        "comparator_only_successes": comparator_only,
                        "mcnemar_exact_p": exact_mcnemar_p(active_only, comparator_only),
                    },
                )
            rows.append(row)
    return rows


def classify_primary_case(
    active_success_rate: float,
    fixed_p60_success_rate: float,
) -> str:
    delta = float(active_success_rate) - float(fixed_p60_success_rate)
    if delta <= 0.02 + 1e-12:
        return "case_a_like_camera_pose_sufficient"
    if delta >= 0.10 - 1e-12:
        return "case_b_like_dynamic_gaze_large_gain"
    return "intermediate"


def _report_markdown(summary: Mapping[str, Any]) -> str:
    metadata = summary["metadata"]
    methods = {row["method"]: row for row in summary["methods"]}
    primary = summary["primary_comparison"]
    lines = [
        "# EXP-05 — Is It Active Gaze or Just Camera Placement?",
        "",
        f"Frozen checkpoint evaluated on {metadata['episodes_per_method']:,} paired seeds "
        f"(`{metadata['seed_start']}`--`{metadata['seed_end']}`).",
        "",
        "| Method | Clean success (95% CI) | Capture episode | Steps | "
        "Mean head yaw | Gaze travel |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = methods[method]
        lines.append(
            f"| {row['label']} | {row['clean_success_rate']:.1%} "
            f"[{row['clean_success_ci_low']:.1%}, {row['clean_success_ci_high']:.1%}] "
            f"| {row['capture_episode_rate']:.1%} | {row['mean_steps']:.2f} "
            f"| {row['mean_head_yaw_degrees']:.1f} deg "
            f"| {row['mean_gaze_travel_degrees']:.1f} deg |",
        )
    lines.extend(
        (
            "",
            "## Primary comparison: Active vs Fixed +60 degrees",
            "",
            f"Paired clean-success delta (Active - Fixed +60): "
            f"**{primary['mean_delta']:+.1%}**; bootstrap 95% CI "
            f"`[{primary['bootstrap_95_low']:+.1%}, {primary['bootstrap_95_high']:+.1%}]`. "
            f"Discordant seeds: {primary['active_only_successes']} Active-only and "
            f"{primary['comparator_only_successes']} Fixed-only; exact McNemar "
            f"`p={primary['mcnemar_exact_p']:.6g}`.",
            "",
            f"Registered descriptive outcome: `{summary['case_classification']}`.",
            "",
            "Fixed placements are exact evaluation-time camera interventions from the "
            "first observation onward. SAC still controls forward velocity and body yaw; "
            "its head-rate output is replaced by zero. Fixed scan uses the EXP-04 sweep "
            "with the normal head turn-rate limit.",
        ),
    )
    return "\n".join(lines) + "\n"


def finalize(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    model_path: Path,
    config_path: Path,
    records: Sequence[Mapping[str, Any]],
    episodes: int,
    seed_start: int,
    bootstrap_samples: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    method_rows = summarize_methods(records)
    if [row["method"] for row in method_rows] != list(METHODS):
        raise RuntimeError("EXP-05 finalization requires all methods in registered order")
    paired_rows = paired_differences(records, bootstrap_samples=int(bootstrap_samples))
    primary = next(
        row
        for row in paired_rows
        if row["right_method"] == PRIMARY_COMPARATOR and row["field"] == "clean_success"
    )
    methods_by_name = {row["method"]: row for row in method_rows}
    fixed_errors = [
        float(record["maximum_fixed_yaw_error_degrees"])
        for record in records
        if record["method"] in FIXED_HEAD_YAWS
    ]
    if fixed_errors and max(fixed_errors) > 1e-8:
        raise RuntimeError("A registered fixed-camera rollout changed its head yaw")
    metadata = {
        **environment_metadata(PROJECT_ROOT),
        "experiment": "EXP-05 Is It Active Gaze or Just Camera Placement?",
        "source_experiment_id": str(config["experiment_id"]),
        "source_model": str(model_path.resolve()),
        "source_model_sha256": _file_sha256(model_path.resolve()),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": _file_sha256(config_path.resolve()),
        "episodes_per_method": int(episodes),
        "seed_start": int(seed_start),
        "seed_end": int(seed_start) + int(episodes) - 1,
        "elapsed_seconds": float(elapsed_seconds),
        "methods": list(METHODS),
        "fixed_head_yaws_degrees": dict(FIXED_HEAD_YAWS),
        "fixed_scan_targets_degrees": list(SCAN_TARGETS_DEGREES),
        "fixed_scan_dwell_steps": SCAN_DWELL_STEPS,
        "fixed_scan_uses_legal_rate_commands": True,
        "policy_deterministic": True,
        "policy_input_fields": [
            "image_left",
            "image_right",
            "proprio",
            "previous_action",
        ],
        "case_a_like_threshold": "Active - Fixed +60 <= 0.02",
        "case_b_like_threshold": "Active - Fixed +60 >= 0.10",
    }
    summary = {
        "metadata": metadata,
        "methods": method_rows,
        "paired_differences": paired_rows,
        "primary_comparison": primary,
        "case_classification": classify_primary_case(
            methods_by_name["sac_active_gaze"]["clean_success_rate"],
            methods_by_name[PRIMARY_COMPARATOR]["clean_success_rate"],
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episodes.jsonl", records)
    write_csv(output_dir / "episodes.csv", records)
    write_csv(output_dir / "methods.csv", method_rows)
    write_csv(output_dir / "paired_differences.csv", paired_rows)
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report_markdown(summary), encoding="utf-8")
    return summary


def shard_bounds(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if total <= 0 or shard_count <= 0:
        raise ValueError("total and shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index is outside shard_count")
    return total * shard_index // shard_count, total * (shard_index + 1) // shard_count


def shard_stem(method: str, shard_index: int, shard_count: int) -> str:
    return f"{method}_shard_{shard_index:03d}_of_{shard_count:03d}"


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    config = load_sac_config(args.config)
    start, end = shard_bounds(args.episodes, args.shard_index, args.shard_count)
    seeds = list(range(args.seed_start + start, args.seed_start + end))
    started_epoch = time.time()
    records = run_method(
        config,
        model_path=args.model.resolve(),
        method=args.method,
        seeds=seeds,
    )
    completed_epoch = time.time()
    shard_dir = args.output_dir.resolve() / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_stem(args.method, args.shard_index, args.shard_count)
    write_jsonl(shard_dir / f"{stem}.jsonl", records)
    manifest = {
        "method": args.method,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "episodes_total": args.episodes,
        "episodes_in_shard": len(seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "started_epoch": started_epoch,
        "completed_epoch": completed_epoch,
        "elapsed_seconds": completed_epoch - started_epoch,
        "clean_success_rate": float(np.mean([record["clean_success"] for record in records])),
    }
    write_json(shard_dir / f"{stem}.json", manifest)
    return manifest


def aggregate_shards(args: argparse.Namespace) -> dict[str, Any]:
    config = load_sac_config(args.config)
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    expected_seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    all_records = []
    started_epochs = []
    completed_epochs = []
    for method in METHODS:
        method_records = []
        for shard_index in range(args.shard_count):
            stem = shard_stem(method, shard_index, args.shard_count)
            manifest = json.loads((shard_dir / f"{stem}.json").read_text(encoding="utf-8"))
            records = read_jsonl(shard_dir / f"{stem}.jsonl")
            if manifest["method"] != method or manifest["shard_index"] != shard_index:
                raise RuntimeError(f"Shard identity mismatch: {stem}")
            if len(records) != manifest["episodes_in_shard"]:
                raise RuntimeError(f"Shard record count mismatch: {stem}")
            method_records.extend(records)
            started_epochs.append(float(manifest["started_epoch"]))
            completed_epochs.append(float(manifest["completed_epoch"]))
        if [int(record["seed"]) for record in method_records] != expected_seeds:
            raise RuntimeError(f"Shard seeds are incomplete or out of order for {method}")
        if any(record["method"] != method for record in method_records):
            raise RuntimeError(f"Shard method mismatch for {method}")
        write_jsonl(output_dir / f"episodes_{method}.jsonl", method_records)
        all_records.extend(method_records)
    return finalize(
        config,
        output_dir=output_dir,
        model_path=args.model,
        config_path=args.config,
        records=all_records,
        episodes=args.episodes,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        elapsed_seconds=max(completed_epochs) - min(started_epochs),
    )


def run_local(args: argparse.Namespace) -> dict[str, Any]:
    config = load_sac_config(args.config)
    output_dir = args.output_dir.resolve()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    all_records = []
    started = time.perf_counter()
    for method in METHODS:
        records = run_method(config, model_path=args.model.resolve(), method=method, seeds=seeds)
        write_jsonl(output_dir / f"episodes_{method}.jsonl", records)
        all_records.extend(records)
    return finalize(
        config,
        output_dir=output_dir,
        model_path=args.model,
        config_path=args.config,
        records=all_records,
        episodes=args.episodes,
        seed_start=args.seed_start,
        bootstrap_samples=args.bootstrap_samples,
        elapsed_seconds=time.perf_counter() - started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "rollout", "aggregate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--model", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--episodes", type=int, default=1000)
        child.add_argument("--seed-start", type=int, default=1_000_000)
        child.add_argument("--bootstrap-samples", type=int, default=5000)
    rollout = subparsers.choices["rollout"]
    rollout.add_argument("--method", choices=METHODS, required=True)
    rollout.add_argument("--shard-index", type=int, required=True)
    rollout.add_argument("--shard-count", type=int, default=10)
    aggregate = subparsers.choices["aggregate"]
    aggregate.add_argument("--shard-count", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.episodes <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("episodes and bootstrap-samples must be positive")
    if args.command == "rollout":
        result = run_shard(args)
    elif args.command == "aggregate":
        result = aggregate_shards(args)
    else:
        result = run_local(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
