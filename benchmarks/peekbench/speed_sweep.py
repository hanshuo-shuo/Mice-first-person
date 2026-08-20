"""Finite predator-speed sweep for EXP-00 headroom evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import write_csv, write_json
from .config import config_hash, load_config
from .environment import PROJECT_ROOT
from .headroom import run_headroom_evaluation


def ratio_label(ratio: float) -> str:
    text = f"{float(ratio):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_speed_config(
    base_config_path: str | Path,
    *,
    ratio: float,
    experiment_id: str,
    num_snapshots: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(
        base_config_path,
        experiment_id=experiment_id,
        num_snapshots=num_snapshots,
        output_root=output_root,
    )
    config["environment"]["predator_prey_forward_speed_ratio"] = float(ratio)
    config["config_hash"] = config_hash(config)
    return config


def _method_by_name(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["method"]): row for row in summary["methods"]}


def _sweep_row(ratio: float, result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    go = summary["go"]
    methods = _method_by_name(summary)
    fixed = methods["fixed_head"]
    random_head = methods["random_head"]
    oracle = methods["privileged_best_gaze"]
    safe_controller = methods["privileged_safe_controller"]
    return {
        "ratio": float(ratio),
        "experiment_id": summary["experiment_id"],
        "verdict": go["verdict"],
        "go_condition_met": go["go_condition_met"],
        "predator_snapshots": go["predator_snapshots"],
        "fixed_failure_count": go["fixed_failure_count"],
        "fixed_failure_fraction": go["fixed_failure_fraction"],
        "stable_recovery_count": go["stable_recovery_count"],
        "stable_headroom_fraction": go["stable_headroom_fraction"],
        "stable_recovery_fraction_of_fixed_failures": go[
            "stable_recovery_fraction_of_fixed_failures"
        ],
        "fixed_safe_success_rate": fixed["safe_success_rate"],
        "fixed_capture_rate": fixed["capture_rate"],
        "random_safe_success_rate": random_head["safe_success_rate"],
        "oracle_safe_success_rate": oracle["safe_success_rate"],
        "oracle_capture_rate": oracle["capture_rate"],
        "privileged_safe_controller_safe_success_rate": safe_controller[
            "safe_success_rate"
        ],
        "privileged_safe_controller_capture_rate": safe_controller["capture_rate"],
    }


def run_speed_sweep(
    base_config_path: str | Path,
    *,
    ratios: Sequence[float],
    experiment_prefix: str,
    num_snapshots: int | None = None,
    output_root: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    rows = []
    for ratio in ratios:
        experiment_id = f"{experiment_prefix}_r{ratio_label(float(ratio))}"
        config = build_speed_config(
            base_config_path,
            ratio=float(ratio),
            experiment_id=experiment_id,
            num_snapshots=num_snapshots,
            output_root=output_root,
        )
        result = run_headroom_evaluation(config, project_root=project_root)
        rows.append(_sweep_row(float(ratio), result))

    output_base = Path(str(output_root or "results/peekbench"))
    if not output_base.is_absolute():
        output_base = project_root / output_base
    summary_path = output_base / f"{experiment_prefix}_speed_sweep_summary.json"
    csv_path = output_base / f"{experiment_prefix}_speed_sweep_summary.csv"
    payload = {
        "experiment_prefix": experiment_prefix,
        "base_config": str(base_config_path),
        "ratios": [float(value) for value in ratios],
        "rows": rows,
    }
    write_json(summary_path, payload)
    write_csv(csv_path, rows)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--ratios", type=float, nargs="+", required=True)
    parser.add_argument("--num-snapshots", type=int)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_speed_sweep(
        args.config,
        ratios=args.ratios,
        experiment_prefix=args.experiment_prefix,
        num_snapshots=args.num_snapshots,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
