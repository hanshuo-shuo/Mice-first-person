"""Aggregate matched SAC test tasks with training seed as the inference unit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.peekbench.artifacts import read_jsonl, write_csv, write_json, write_jsonl
from evaluate_matched_sac import summarize


CONDITIONS = ("active", "fixed_center", "fixed_p60", "fixed_scan")
TRAINING_SEEDS = (2026082401, 2026082402, 2026082403, 2026082404, 2026082405)
T_CRITICAL_95_DF4 = 2.7764451051977987


def _seed_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) <= 1:
        return math.nan, math.nan
    margin = T_CRITICAL_95_DF4 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean - margin, mean + margin


def _load_run_records(
    result_root: Path,
    *,
    condition: str,
    training_seed: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    shard_dir = result_root / f"{condition}_seed{training_seed}" / "shards"
    records = []
    for shard_index in range(int(shard_count)):
        records.extend(read_jsonl(shard_dir / f"test_shard_{shard_index}.jsonl"))
    if len(records) != 1000:
        raise RuntimeError(
            f"Expected 1000 test tasks for {condition} seed {training_seed}, got {len(records)}",
        )
    if [int(record["task_index"]) for record in records] != list(range(1000)):
        raise RuntimeError(f"Test task indices are incomplete for {condition} seed {training_seed}")
    return records


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    result_root = args.result_root.resolve()
    output_dir = result_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    run_rows = []
    canonical_task_ids = None
    for condition in CONDITIONS:
        for training_seed in TRAINING_SEEDS:
            records = _load_run_records(
                result_root,
                condition=condition,
                training_seed=training_seed,
                shard_count=int(args.shard_count),
            )
            task_ids = [str(record["task_id"]) for record in records]
            if canonical_task_ids is None:
                canonical_task_ids = task_ids
            elif task_ids != canonical_task_ids:
                raise RuntimeError("Matched runs do not contain identical ordered test tasks")
            records_by_run[(condition, training_seed)] = records
            run_rows.append(
                {
                    "condition": condition,
                    "training_seed": training_seed,
                    **summarize(records),
                },
            )

    condition_rows = []
    for condition in CONDITIONS:
        selected = [row for row in run_rows if row["condition"] == condition]
        success_rates = [float(row["clean_success_rate"]) for row in selected]
        interval = _seed_interval(success_rates)
        condition_rows.append(
            {
                "condition": condition,
                "training_seeds": len(selected),
                "mean_clean_success_rate": float(np.mean(success_rates)),
                "training_seed_sd": float(np.std(success_rates, ddof=1)),
                "training_seed_min": float(np.min(success_rates)),
                "training_seed_max": float(np.max(success_rates)),
                "training_seed_t95_low": interval[0],
                "training_seed_t95_high": interval[1],
                "mean_capture_episode_rate": float(
                    np.mean([row["capture_episode_rate"] for row in selected]),
                ),
                "mean_steps": float(np.mean([row["mean_steps"] for row in selected])),
                "mean_minimum_predator_distance": float(
                    np.mean([row["mean_minimum_predator_distance"] for row in selected]),
                ),
                "mean_path_cost": float(
                    np.mean([row["mean_path_cost"] for row in selected]),
                ),
            },
        )

    paired_rows = []
    for comparator in CONDITIONS[1:]:
        seed_deltas = []
        for training_seed in TRAINING_SEEDS:
            active = records_by_run[("active", training_seed)]
            fixed = records_by_run[(comparator, training_seed)]
            differences = np.asarray(
                [
                    float(left["clean_success"]) - float(right["clean_success"])
                    for left, right in zip(active, fixed)
                ],
                dtype=np.float64,
            )
            delta = float(differences.mean())
            seed_deltas.append(delta)
            paired_rows.append(
                {
                    "left_condition": "active",
                    "right_condition": comparator,
                    "training_seed": training_seed,
                    "paired_test_tasks": len(differences),
                    "clean_success_delta": delta,
                    "active_only_successes": int(np.count_nonzero(differences == 1.0)),
                    "comparator_only_successes": int(np.count_nonzero(differences == -1.0)),
                },
            )
        interval = _seed_interval(seed_deltas)
        paired_rows.append(
            {
                "left_condition": "active",
                "right_condition": comparator,
                "training_seed": "mean",
                "paired_test_tasks": 1000,
                "clean_success_delta": float(np.mean(seed_deltas)),
                "training_seed_sd": float(np.std(seed_deltas, ddof=1)),
                "training_seed_t95_low": interval[0],
                "training_seed_t95_high": interval[1],
            },
        )

    methods = {row["condition"]: row for row in condition_rows}
    summary = {
        "experiment": "Matched training with randomized task generalization",
        "training_seeds_per_condition": len(TRAINING_SEEDS),
        "test_tasks_per_checkpoint": 1000,
        "conditions": condition_rows,
        "paired_training_seed_differences": paired_rows,
        "primary": {
            "active": methods["active"],
            "fixed_p60": methods["fixed_p60"],
        },
        "training_procedure_stability_estimated": True,
    }
    all_records = [
        {"condition": condition, "training_seed": seed, **record}
        for (condition, seed), records in records_by_run.items()
        for record in records
    ]
    write_jsonl(output_dir / "episodes.jsonl", all_records)
    write_csv(output_dir / "runs.csv", run_rows)
    write_csv(output_dir / "conditions.csv", condition_rows)
    write_csv(output_dir / "paired_training_seed_differences.csv", paired_rows)
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Matched SAC training-seed evaluation",
        "",
        "Training seed is the inference unit: five independently trained checkpoints "
        "per condition, each evaluated on the same 1,000 held-out test tasks.",
        "",
        "| Condition | Mean clean success | Seed SD | Seed range | t 95% interval |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in condition_rows:
        report.append(
            f"| {row['condition']} | {row['mean_clean_success_rate']:.1%} "
            f"| {row['training_seed_sd']:.1%} "
            f"| {row['training_seed_min']:.1%}--{row['training_seed_max']:.1%} "
            f"| [{row['training_seed_t95_low']:.1%}, {row['training_seed_t95_high']:.1%}] |",
        )
    report.extend(("", "## Active paired training-seed effects", ""))
    for comparator in CONDITIONS[1:]:
        row = next(
            value
            for value in paired_rows
            if value["right_condition"] == comparator
            and value["training_seed"] == "mean"
        )
        report.append(
            f"- Versus `{comparator}`: {row['clean_success_delta']:+.1%}; "
            f"training-seed t interval "
            f"[{row['training_seed_t95_low']:+.1%}, {row['training_seed_t95_high']:+.1%}].",
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = aggregate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
