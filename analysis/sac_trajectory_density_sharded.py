"""Slurm-friendly single-process shards for the SAC trajectory-density audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.peekbench.artifacts import read_jsonl, write_json, write_jsonl
from training.first_person_sac import load_sac_config

from .sac_trajectory_density import (
    METHODS,
    finalize_audit,
    pack_traces,
    run_method,
    save_trace_store,
    unpack_state_traces,
)


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
    start_offset, end_offset = shard_bounds(
        int(args.episodes),
        int(args.shard_index),
        int(args.shard_count),
    )
    seeds = list(
        range(
            int(args.seed_start) + start_offset,
            int(args.seed_start) + end_offset,
        ),
    )
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_stem(args.method, int(args.shard_index), int(args.shard_count))
    started_epoch = time.time()
    records, traces = run_method(
        config,
        model_path=args.model.resolve(),
        method=args.method,
        seeds=seeds,
        workers=1,
    )
    completed_epoch = time.time()
    write_jsonl(shard_dir / f"{stem}.jsonl", records)
    save_trace_store(shard_dir / f"{stem}.npz", pack_traces(seeds, traces))
    manifest = {
        "method": args.method,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "episodes_total": int(args.episodes),
        "episodes_in_shard": len(seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "started_epoch": started_epoch,
        "completed_epoch": completed_epoch,
        "elapsed_seconds": completed_epoch - started_epoch,
        "clean_success_rate": float(
            np.mean([record["clean_success"] for record in records]),
        ),
        "capture_episode_rate": float(
            np.mean([record["capture_episode"] for record in records]),
        ),
    }
    write_json(shard_dir / f"{stem}.json", manifest)
    return manifest


def _load_packed(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def aggregate_shards(args: argparse.Namespace) -> dict[str, Any]:
    config = load_sac_config(args.config)
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    expected_seeds = list(
        range(int(args.seed_start), int(args.seed_start) + int(args.episodes)),
    )
    all_records: list[dict[str, Any]] = []
    traces_by_method = {}
    started_epochs = []
    completed_epochs = []

    for method in METHODS:
        method_records = []
        method_traces = []
        method_seeds = []
        for shard_index in range(int(args.shard_count)):
            stem = shard_stem(method, shard_index, int(args.shard_count))
            manifest_path = shard_dir / f"{stem}.json"
            record_path = shard_dir / f"{stem}.jsonl"
            trace_path = shard_dir / f"{stem}.npz"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["method"] != method or manifest["shard_index"] != shard_index:
                raise RuntimeError(f"Shard manifest identity mismatch: {manifest_path}")
            records = read_jsonl(record_path)
            packed = _load_packed(trace_path)
            traces = unpack_state_traces(packed)
            if len(records) != len(traces) or len(records) != manifest["episodes_in_shard"]:
                raise RuntimeError(f"Shard record/trace count mismatch: {stem}")
            method_records.extend(records)
            method_traces.extend(traces)
            method_seeds.extend(int(value) for value in packed["seeds"])
            started_epochs.append(float(manifest["started_epoch"]))
            completed_epochs.append(float(manifest["completed_epoch"]))

        if method_seeds != expected_seeds:
            raise RuntimeError(f"Shard seeds are incomplete or out of order for {method}")
        if len(method_records) != int(args.episodes):
            raise RuntimeError(f"Expected {args.episodes} records for {method}")
        all_records.extend(method_records)
        traces_by_method[method] = method_traces
        write_jsonl(output_dir / f"episodes_{method}.jsonl", method_records)
        save_trace_store(
            output_dir / f"traces_{method}.npz",
            pack_traces(expected_seeds, method_traces),
        )

    elapsed_seconds = max(completed_epochs) - min(started_epochs)
    return finalize_audit(
        config,
        output_dir=output_dir,
        model_path=args.model,
        config_path=args.config,
        all_records=all_records,
        traces_by_method=traces_by_method,
        episodes=int(args.episodes),
        seed_start=int(args.seed_start),
        workers=1,
        bootstrap_samples=int(args.bootstrap_samples),
        bins=int(args.bins),
        trajectory_sample=int(args.trajectory_sample),
        elapsed_seconds=elapsed_seconds,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rollout", "aggregate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--model", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--episodes", type=int, default=1000)
        child.add_argument("--seed-start", type=int, default=1_000_000)
        child.add_argument("--shard-count", type=int, default=10)
    rollout = subparsers.choices["rollout"]
    rollout.add_argument("--method", choices=METHODS, required=True)
    rollout.add_argument("--shard-index", type=int, required=True)
    aggregate = subparsers.choices["aggregate"]
    aggregate.add_argument("--bins", type=int, default=64)
    aggregate.add_argument("--trajectory-sample", type=int, default=100)
    aggregate.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_shard(args) if args.command == "rollout" else aggregate_shards(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
