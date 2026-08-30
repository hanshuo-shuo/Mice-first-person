"""Command-line entry point for the bounded phase-1 autoresearch loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autoresearch.runner import AutoresearchRunner, RunnerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch",
        description=(
            "Bounded engineering search over legal-rate gaze controllers. "
            "Normal experiments cannot access confirmation seeds."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository/worktree root (place before the subcommand)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "shared results/autoresearch root for experiment worktrees "
            "(place before the subcommand)"
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    setup = subcommands.add_parser("setup", help="freeze a new run manifest")
    setup.add_argument("--config", type=Path, required=True)
    setup.add_argument("--run-tag", required=True)

    baseline = subcommands.add_parser(
        "baseline", help="record the registered legal-rate incumbent"
    )
    baseline.add_argument("--run-tag", required=True)
    baseline_external = baseline.add_mutually_exclusive_group()
    baseline_external.add_argument(
        "--prepare-external",
        action="store_true",
        help="preregister, run smoke, and emit Quest worker identities",
    )
    baseline_external.add_argument(
        "--finalize-external",
        "--evaluation-result",
        type=Path,
        default=None,
        dest="finalize_external",
        help="finalize a preregistered baseline from aggregate.manifest.json",
    )
    baseline.add_argument("--experiment-id", default=None)

    experiment = subcommands.add_parser(
        "experiment", help="run one smoke+development candidate decision"
    )
    experiment.add_argument("--run-tag", required=True)
    experiment.add_argument("--hypothesis-file", type=Path, default=None)
    experiment.add_argument(
        "--predicted-effect",
        default="Improve paired development clean success.",
    )
    experiment_external = experiment.add_mutually_exclusive_group()
    experiment_external.add_argument(
        "--prepare-external",
        action="store_true",
        help="preregister, run smoke, archive sources, and emit worker identities",
    )
    experiment_external.add_argument(
        "--finalize-external",
        "--evaluation-result",
        type=Path,
        default=None,
        dest="finalize_external",
        help="finalize one preregistered E#### from aggregate.manifest.json",
    )
    experiment.add_argument("--experiment-id", default=None)

    status = subcommands.add_parser("status", help="show compact durable state")
    status.add_argument("--run-tag", required=True)
    status.add_argument("--json", action="store_true", dest="as_json")

    confirm = subcommands.add_parser(
        "confirm", help="spend the one-time registered confirmation set"
    )
    confirm.add_argument("--run-tag", required=True)
    confirm.add_argument(
        "--authorize-confirmation",
        "--authorize",
        "--yes",
        action="store_true",
        dest="authorized",
        help="record explicit authorization to spend the confirmation set once",
    )

    abort_external = subcommands.add_parser(
        "abort-external",
        help="explicitly crash a waiting Quest lifecycle while preserving artifacts",
    )
    abort_external.add_argument("--run-tag", required=True)
    abort_external.add_argument("--experiment-id", required=True)
    abort_external.add_argument("--reason", required=True)

    # Quest rollout/aggregation is implemented by its dedicated command.  This
    # read-only hook makes the frozen worker identities available without
    # weakening the local evaluator or exposing arbitrary seed overrides.
    worker = subcommands.add_parser(
        "worker-context",
        help="emit verified read-only identities for an independent worker",
    )
    worker.add_argument("--run-tag", required=True)
    worker.add_argument("--candidate-path", type=Path, default=None)
    worker.add_argument("--candidate-commit", default=None)
    worker.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _compact_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "run_tag",
            "experiment_id",
            "status",
            "primary_delta",
            "decision_reason",
            "candidate_sha256",
            "candidate_commit",
            "confirmation_passed",
        )
        if key in result
    }


def _print_human_status(status: Mapping[str, Any]) -> None:
    incumbent = status.get("incumbent") or {}
    last = status.get("last_experiment") or {}
    budget = status.get("budget") or {}
    confirmation = status.get("confirmation") or {}
    lines = (
        f"run: {status.get('run_tag')}",
        f"state: {status.get('state')}",
        f"incumbent: {incumbent.get('experiment_id') or '-'}",
        f"last experiment: {last.get('experiment_id') or '-'} "
        f"({last.get('status') or '-'})",
        "budget: "
        f"{budget.get('experiments_used', 0)}/"
        f"{(budget.get('limits') or {}).get('max_experiments', '?')}",
        f"confirmation: {confirmation.get('state', 'unknown')}",
        f"next: {status.get('next_action')}",
    )
    print("\n".join(lines))


def _terminal_exit_code(result: Mapping[str, Any]) -> int:
    status = result.get("status")
    if status == "contract_failure":
        return 2
    if status == "crash":
        return 3
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: AutoresearchRunner | None = None,
    runner_factory: Callable[..., AutoresearchRunner] = AutoresearchRunner,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    selected_runner = runner or runner_factory(
        repo_root=arguments.repo_root,
        results_root=arguments.results_root,
    )
    try:
        if arguments.command == "setup":
            result = selected_runner.setup(
                config_path=arguments.config,
                run_tag=arguments.run_tag,
            )
            output = {
                "run_tag": result["run_tag"],
                "status": "ready",
                "run_manifest_sha256": __import__("hashlib")
                .sha256(
                    (
                        selected_runner.results_root
                        / arguments.run_tag
                        / "run.json"
                    ).read_bytes()
                )
                .hexdigest(),
                "source_ready": all(
                    item["verified"]
                    for item in result["source_artifacts_at_setup"].values()
                ),
            }
            print(json.dumps(output, sort_keys=True, separators=(",", ":")))
            return 0
        if arguments.command == "baseline":
            if arguments.prepare_external:
                if arguments.experiment_id is not None:
                    raise RunnerError(
                        "--experiment-id is valid only with --finalize-external"
                    )
                result = selected_runner.prepare_external_baseline(
                    run_tag=arguments.run_tag
                )
            elif arguments.finalize_external is not None:
                if not arguments.experiment_id:
                    raise RunnerError(
                        "--finalize-external requires --experiment-id"
                    )
                result = selected_runner.finalize_external_baseline(
                    run_tag=arguments.run_tag,
                    experiment_id=arguments.experiment_id,
                    aggregate_manifest_path=arguments.finalize_external,
                )
            else:
                if arguments.experiment_id is not None:
                    raise RunnerError(
                        "--experiment-id is valid only with --finalize-external"
                    )
                result = selected_runner.baseline(run_tag=arguments.run_tag)
            print(
                json.dumps(
                    result if result.get("status") == "running" else _compact_result(result),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return _terminal_exit_code(result)
        if arguments.command == "experiment":
            if arguments.prepare_external:
                if arguments.hypothesis_file is None:
                    raise RunnerError(
                        "--prepare-external requires --hypothesis-file"
                    )
                if arguments.experiment_id is not None:
                    raise RunnerError(
                        "--experiment-id is valid only with --finalize-external"
                    )
                result = selected_runner.prepare_external_experiment(
                    run_tag=arguments.run_tag,
                    hypothesis_file=arguments.hypothesis_file,
                    predicted_effect=arguments.predicted_effect,
                )
            elif arguments.finalize_external is not None:
                if not arguments.experiment_id:
                    raise RunnerError(
                        "--finalize-external requires --experiment-id"
                    )
                if arguments.hypothesis_file is not None:
                    raise RunnerError(
                        "--hypothesis-file belongs to prepare, not finalize"
                    )
                result = selected_runner.finalize_external_experiment(
                    run_tag=arguments.run_tag,
                    experiment_id=arguments.experiment_id,
                    aggregate_manifest_path=arguments.finalize_external,
                )
            else:
                if arguments.hypothesis_file is None:
                    raise RunnerError("experiment requires --hypothesis-file")
                if arguments.experiment_id is not None:
                    raise RunnerError(
                        "--experiment-id is valid only with --finalize-external"
                    )
                result = selected_runner.experiment(
                    run_tag=arguments.run_tag,
                    hypothesis_file=arguments.hypothesis_file,
                    predicted_effect=arguments.predicted_effect,
                )
            print(
                json.dumps(
                    result if result.get("status") == "running" else _compact_result(result),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return _terminal_exit_code(result)
        if arguments.command == "status":
            result = selected_runner.status(run_tag=arguments.run_tag)
            if arguments.as_json:
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            else:
                _print_human_status(result)
            return 0
        if arguments.command == "confirm":
            result = selected_runner.confirm(
                run_tag=arguments.run_tag,
                authorized=arguments.authorized,
            )
            print(
                json.dumps(
                    _compact_result(result), sort_keys=True, separators=(",", ":")
                )
            )
            return _terminal_exit_code(result)
        if arguments.command == "abort-external":
            result = selected_runner.abort_external(
                run_tag=arguments.run_tag,
                experiment_id=arguments.experiment_id,
                reason=arguments.reason,
            )
            print(
                json.dumps(
                    _compact_result(result), sort_keys=True, separators=(",", ":")
                )
            )
            return _terminal_exit_code(result)
        if arguments.command == "worker-context":
            result = selected_runner.worker_context(
                run_tag=arguments.run_tag,
                candidate_path=arguments.candidate_path,
                candidate_commit=arguments.candidate_commit,
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        parser.error(f"unknown command: {arguments.command}")
    except RunnerError as exc:
        print(f"autoresearch: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("autoresearch: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        # CLI output stays bounded; full rollout diagnostics belong in the
        # experiment artifact directory or Quest job logs.
        detail = " ".join(str(exc).split())[:600]
        print(
            f"autoresearch: internal {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        return 3
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through ``-m``.
    raise SystemExit(main())
