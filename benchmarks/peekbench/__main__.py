"""Command-line entry point for PeekBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .environment import PROJECT_ROOT
from .evaluation import run_branch_evaluation, run_open_loop_evaluation
from .exp01 import run_exp01_evaluation
from .exp03 import run_exp03_evaluation
from .exp04 import run_exp04_evaluation
from .generator import generate_snapshots
from .headroom import run_headroom_evaluation


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--num-snapshots", type=int)
    parser.add_argument("--output-root", type=Path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("generate", "Generate deterministic state snapshots"),
        ("open-loop", "Run vision-only semantic decisions"),
        ("branches", "Run exact-state H-step branches"),
        ("headroom", "Run EXP-00 legal-duration gaze headroom evaluation"),
        ("exp01", "Run EXP-01 paired perception--action gap evaluation"),
        ("exp03", "Run EXP-03 identical-current temporal-memory evaluation"),
        ("exp04", "Run EXP-04 equal-budget active-gaze evaluation"),
        ("all", "Generate snapshots and run both evaluations"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_common(child)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config,
        experiment_id=args.experiment_id,
        num_snapshots=args.num_snapshots,
        output_root=args.output_root,
    )
    summary = {
        "command": args.command,
        "experiment_id": config["experiment_id"],
        "config_hash": config["config_hash"],
        "seed": config["seed"],
    }
    if args.command in ("generate", "all"):
        snapshots = generate_snapshots(config, project_root=PROJECT_ROOT)
        summary["snapshots"] = len(snapshots)
        summary["snapshot_ids"] = [record["snapshot_id"] for record in snapshots]
    if args.command in ("open-loop", "all"):
        open_loop = run_open_loop_evaluation(config, project_root=PROJECT_ROOT)
        summary["open_loop_records"] = len(open_loop)
        summary["backend"] = sorted(
            {record["telemetry"]["backend"] for record in open_loop},
        )
    if args.command in ("branches", "all"):
        branches = run_branch_evaluation(config, project_root=PROJECT_ROOT)
        summary["branch_records"] = len(branches)
        summary["avoidable_by_looking_candidates"] = sum(
            bool(record["screening"]["avoidable_by_looking_candidate"])
            for record in branches
        )
    if args.command == "headroom":
        headroom = run_headroom_evaluation(config, project_root=PROJECT_ROOT)
        summary["headroom_records"] = len(headroom["records"])
        summary["go_verdict"] = headroom["summary"]["go"]["verdict"]
        summary["go_condition_met"] = headroom["summary"]["go"][
            "go_condition_met"
        ]
    if args.command == "exp01":
        exp01 = run_exp01_evaluation(config, project_root=PROJECT_ROOT)
        summary["exp01_records"] = len(exp01["records"])
        summary["backends"] = exp01["summary"]["backends"]
        summary["evidence_level"] = exp01["summary"]["evidence_level"]
        summary["remote_uncached_model_calls"] = exp01["summary"][
            "remote_uncached_model_calls"
        ]
    if args.command == "exp03":
        exp03 = run_exp03_evaluation(config, project_root=PROJECT_ROOT)
        summary["exp03_pairs"] = len(exp03["records"])
        summary["current_observations_identical"] = exp03["summary"][
            "all_current_observations_byte_identical"
        ]
    if args.command == "exp04":
        exp04 = run_exp04_evaluation(config, project_root=PROJECT_ROOT)
        summary["exp04_snapshots"] = len(exp04["records"])
        summary["all_budgets_equal"] = exp04["summary"]["all_budgets_equal"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
