"""Evaluate one matched-training SAC checkpoint on an exact task manifest."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from stable_baselines3 import SAC

from benchmarks.peekbench.artifacts import environment_metadata, write_csv, write_json, write_jsonl
from task_distribution import load_task_records, manifest_sha256
from training.first_person_sac import PROJECT_ROOT, load_sac_config, make_first_person_env


def _manifest_path(config: Mapping[str, Any], split: str) -> Path:
    root = Path(str(config["task_distribution"]["manifest_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / f"{split}.jsonl"


def run_episode(
    env,
    model: SAC,
    *,
    task_index: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    observation, reset_info = env.reset(
        seed=int(evaluation_seed) + int(task_index),
        options={"task_index": int(task_index)},
    )
    task = reset_info["task"]
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
    head_yaw_sum = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != env.action_space.shape:
            raise ValueError(
                f"Checkpoint emitted action {action.shape}; environment expects "
                f"{env.action_space.shape}",
            )
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        events = info["transition_events"]
        capture_count += int(events["capture_event"])
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

    return {
        "matched_condition": getattr(env, "matched_condition", None),
        "task_index": int(task_index),
        "task_id": str(task["task_id"]),
        "task_split": str(task["split"]),
        "start_region": str(task["start_region"]),
        "goal_region": str(task["goal_region"]),
        "predator_speed_ratio": float(task["predator_speed_ratio"]),
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
        "mean_head_yaw_degrees": head_yaw_sum / steps if steps else previous_head_yaw,
        "predator_pixels_visible_fraction": visible_steps / steps if steps else 0.0,
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total),
    ) / denominator
    return centre - margin, centre + margin


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(record["clean_success"]) for record in records)
    interval = wilson_interval(successes, len(records))

    def mean(field: str) -> float:
        return float(np.mean([float(record[field]) for record in records]))

    return {
        "episodes": len(records),
        "clean_success_rate": successes / len(records),
        "clean_success_ci_low": interval[0],
        "clean_success_ci_high": interval[1],
        "capture_episode_rate": mean("capture_episode"),
        "goal_reach_rate": mean("goal_reached"),
        "mean_steps": mean("steps"),
        "mean_minimum_predator_distance": mean("minimum_predator_distance"),
        "mean_path_cost": mean("path_cost"),
        "mean_gaze_travel_degrees": mean("gaze_travel_degrees"),
        "mean_predator_pixels_visible_fraction": mean(
            "predator_pixels_visible_fraction",
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_sac_config(args.config)
    manifest_path = _manifest_path(config, args.split)
    tasks = load_task_records(manifest_path)
    end_index = len(tasks) if args.count is None else args.start_index + args.count
    if not 0 <= args.start_index < end_index <= len(tasks):
        raise ValueError("Requested task range is outside the selected manifest")
    model = SAC.load(args.model, device=args.device)
    env = make_first_person_env(
        config,
        task_split=args.split,
        task_selection_mode="sequential",
    )
    env.matched_condition = str(config["matched_condition"])
    started = time.perf_counter()
    try:
        records = [
            run_episode(
                env,
                model,
                task_index=task_index,
                evaluation_seed=int(args.evaluation_seed),
            )
            for task_index in range(int(args.start_index), int(end_index))
        ]
    finally:
        env.close()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.name)
    method_summary = summarize(records)
    summary = {
        "metadata": {
            **environment_metadata(PROJECT_ROOT),
            "matched_condition": str(config["matched_condition"]),
            "training_seed": int(config["seed"]),
            "split": str(args.split),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256(manifest_path),
            "task_start_index": int(args.start_index),
            "task_end_index": int(end_index),
            "evaluation_seed": int(args.evaluation_seed),
            "model": str(args.model.resolve()),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": method_summary,
    }
    write_jsonl(output_dir / f"{stem}.jsonl", records)
    write_csv(output_dir / f"{stem}.csv", records)
    write_json(output_dir / f"{stem}_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--name", default="matched_evaluation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--evaluation-seed", type=int, default=8_000_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
