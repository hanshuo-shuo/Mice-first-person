"""Authorized, one-time sharded confirmation for the frozen phase-1 run.

This driver is intentionally separate from the immutable development runner.
It never selects or promotes a candidate.  ``prepare`` writes a durable C0001
record marked spent before any rollout; Quest workers require the resulting
content-addressed authorization marker.  ``finalize`` independently rebuilds
all paired statistics before publishing report-only evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import hmac
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from autoresearch import sharding as shardlib
from autoresearch.evaluator import (
    EVALUATION_METHODS,
    REFERENCE_METHOD,
    RealExp05EpisodeFactory,
    canonical_sha256,
    confirmation_statistics,
    ordered_seed_sha256,
    records_sha256,
    seed_set_from_config,
    summarize_records,
    validate_episode_records,
    verify_exp05_artifacts,
)
from autoresearch.guard import (
    assert_no_leaks,
    sha256_bytes,
    sha256_file,
    validate_candidate_source,
)
from autoresearch.ledger import ExperimentLedger, TERMINAL_STATUSES
from autoresearch.runner import AutoresearchRunner, RunContractError
from autoresearch.worker import IsolatedCandidateController


SCHEMA_VERSION = 1
EXPERIMENT_ID = "C0001"
SEED_SET_NAME = "confirmation"
SHARD_ARTIFACT_TYPE = "autoresearch_rollout_shard"
AGGREGATE_ARTIFACT_TYPE = "autoresearch_confirmation_aggregate"
AUTHORIZATION_ARTIFACT_TYPE = "autoresearch_confirmation_authorization"
MAX_JSON_BYTES = 64 * 1024 * 1024


class ConfirmationDriverError(RuntimeError):
    """The authorized confirmation evidence violated its frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def _driver_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _atomic_write(path: Path, payload: bytes) -> None:
    shardlib._atomic_write_bytes(path, payload)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(dict(value))
    assert_no_leaks(payload, source=path.name)
    _atomic_write(path, payload)


def _write_new_or_same(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ConfirmationDriverError(f"conflicting confirmation artifact: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short confirmation artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationDriverError(f"required JSON artifact is missing: {path}")
    payload = path.read_bytes()
    if not payload or len(payload) > maximum_bytes:
        raise ConfirmationDriverError(f"JSON artifact has invalid size: {path}")
    assert_no_leaks(payload, source=path.name)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfirmationDriverError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ConfirmationDriverError(f"JSON artifact is not an object: {path}")
    return value


def _run_manifest_sha256(run_dir: Path) -> str:
    payload = (run_dir / "run.json").read_bytes()
    expected = (run_dir / "run.sha256").read_text(encoding="ascii").strip()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ConfirmationDriverError("run manifest digest mismatch")
    return actual


def _confirmation_partition(
    config: Mapping[str, Any],
    *,
    shard_count: int,
) -> tuple[shardlib.RegisteredShard, ...]:
    seed_set = seed_set_from_config(config, SEED_SET_NAME)
    if not seed_set.one_time or not seed_set.requires_explicit_authorization:
        raise ConfirmationDriverError("confirmation seed set is not one-time authorized")
    count = int(shard_count)
    if count <= 0 or count > len(seed_set.seeds):
        raise ConfirmationDriverError("invalid confirmation shard count")
    shards: list[shardlib.RegisteredShard] = []
    quotient, remainder = divmod(len(seed_set.seeds), count)
    start = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        seeds = tuple(seed_set.seeds[start : start + size])
        start += size
        shards.append(
            shardlib.RegisteredShard(
                seed_set_name=seed_set.name,
                seed_set_id=seed_set.seed_set_id,
                all_seeds=tuple(seed_set.seeds),
                shard_index=index,
                shard_count=count,
                seeds=seeds,
                one_time=True,
                requires_explicit_authorization=True,
            ),
        )
    if tuple(seed for shard in shards for seed in shard.seeds) != tuple(seed_set.seeds):
        raise ConfirmationDriverError("confirmation shards do not exactly cover seeds")
    return tuple(shards)


def _authorization(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationDriverError("authorization marker is missing or a symlink")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(actual, expected_sha256):
        raise ConfirmationDriverError("authorization marker SHA-256 mismatch")
    assert_no_leaks(payload, source=path.name)
    try:
        marker = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfirmationDriverError("authorization marker is invalid JSON") from exc
    if not isinstance(marker, Mapping):
        raise ConfirmationDriverError("authorization marker is not an object")
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("artifact_type") != AUTHORIZATION_ARTIFACT_TYPE
        or marker.get("experiment_id") != EXPERIMENT_ID
        or marker.get("confirmation_set_spent") is not True
        or marker.get("explicit_user_authorization") is not True
    ):
        raise ConfirmationDriverError("authorization marker is not valid and spent")
    if not hmac.compare_digest(str(marker.get("driver_sha256", "")), _driver_sha256()):
        raise ConfirmationDriverError("authorization marker names another driver")
    return {**dict(marker), "authorization_sha256": actual}


def _load_context(
    *,
    project_root: Path,
    results_root: Path,
    run_tag: str,
    config_path: Path,
    candidate_source: Path,
    incumbent_source: Path,
    authorization_path: Path,
    authorization_sha256: str,
    shard_count: int,
) -> tuple[
    Mapping[str, Any],
    shardlib.LoadedConfig,
    tuple[shardlib.RegisteredShard, ...],
    shardlib.LoadedController,
    shardlib.LoadedController,
    Mapping[str, Any],
    Any,
    int,
]:
    runner = AutoresearchRunner(repo_root=project_root, results_root=results_root)
    manifest = runner._load_run(run_tag, verify_sources=False)
    run_dir = results_root / run_tag
    run_digest = _run_manifest_sha256(run_dir)
    authorization = _authorization(
        authorization_path,
        expected_sha256=authorization_sha256,
    )
    if (
        authorization.get("run_tag") != run_tag
        or authorization.get("run_manifest_sha256") != run_digest
    ):
        raise ConfirmationDriverError("authorization marker names another frozen run")
    loaded = shardlib.load_registered_config(config_path, project_root=project_root)
    if not hmac.compare_digest(loaded.sha256, str(manifest["config_sha256"])):
        raise ConfirmationDriverError("confirmation config differs from run manifest")
    partition = _confirmation_partition(loaded.payload, shard_count=shard_count)
    if (
        authorization.get("seed_set_id") != partition[0].seed_set_id
        or authorization.get("ordered_seed_sha256")
        != ordered_seed_sha256(partition[0].all_seeds)
    ):
        raise ConfirmationDriverError("authorization marker names another seed set")
    candidate = shardlib.load_guarded_controller(candidate_source, project_root=project_root)
    incumbent = shardlib.load_guarded_controller(incumbent_source, project_root=project_root)
    if (
        authorization.get("candidate_sha256") != candidate.sha256
        or authorization.get("incumbent_sha256") != incumbent.sha256
    ):
        raise ConfirmationDriverError("controller snapshots differ from authorization")
    bundle, horizon = shardlib._verified_aggregate_artifacts_and_horizon(
        loaded.payload,
        project_root=project_root,
    )
    environment_digest = shardlib.environment_contract_sha256(project_root)
    if not hmac.compare_digest(
        environment_digest,
        str(manifest["environment_contract_sha256"]),
    ):
        raise ConfirmationDriverError("environment contract changed after development")
    base = shardlib._base_identity(
        loaded_config=loaded,
        shard=partition[0],
        candidate=candidate,
        incumbent=incumbent,
        max_horizon=horizon,
        environment_digest=environment_digest,
        artifact_bundle=bundle,
    )
    base.pop("run_identity_sha256", None)
    base.update(
        {
            "authorization_sha256": authorization_sha256,
            "confirmation_driver_sha256": _driver_sha256(),
            "confirmation_experiment_id": EXPERIMENT_ID,
            "run_manifest_sha256": run_digest,
        },
    )
    base["run_identity_sha256"] = canonical_sha256(base)
    return (
        manifest,
        loaded,
        partition,
        candidate,
        incumbent,
        base,
        bundle,
        horizon,
    )


def prepare_confirmation(
    *,
    repo_root: Path,
    results_root: Path,
    run_tag: str,
    authorized: bool,
) -> Mapping[str, Any]:
    if authorized is not True:
        raise ConfirmationDriverError("explicit confirmation authorization is required")
    runner = AutoresearchRunner(repo_root=repo_root, results_root=results_root)
    manifest = runner._load_run(run_tag, verify_sources=True)
    run_dir = results_root / run_tag
    with runner._run_lock(run_dir):
        ledger = ExperimentLedger(run_dir)
        seed_set = seed_set_from_config(manifest["config"], SEED_SET_NAME)
        resumable: Mapping[str, Any] | None = None
        spent = [
            record
            for record in ledger.read_records()
            if record.get("seed_set_id") == seed_set.seed_set_id
        ]
        if spent:
            latest = ledger.latest_records().get(EXPERIMENT_ID)
            if (
                latest is None
                or latest.get("status") not in {"planned", "running"}
                or latest.get("confirmation_authorized") is not True
                or latest.get("confirmation_set_spent") is not True
            ):
                raise ConfirmationDriverError(
                    f"confirmation already spent by {spent[-1]['experiment_id']}",
                )
            staging = ledger.artifact_staging_path(EXPERIMENT_ID)
            marker_path = staging / "authorization.json"
            if latest.get("status") == "running" and marker_path.is_file():
                marker = _authorization(marker_path)
                return {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "running",
                    "authorization_path": str(marker_path),
                    "authorization_sha256": marker["authorization_sha256"],
                    "candidate_commit": marker["candidate_commit"],
                    "candidate_sha256": marker["candidate_sha256"],
                    "incumbent_commit": marker["incumbent_commit"],
                    "incumbent_sha256": marker["incumbent_sha256"],
                }
            resumable = latest
        incumbent_info = runner._incumbent_record(ledger)
        if incumbent_info is None:
            raise ConfirmationDriverError("selected incumbent is missing")
        candidate_record, candidate_path = incumbent_info
        parent_id = str(candidate_record.get("parent_incumbent_id") or "")
        if not parent_id:
            raise ConfirmationDriverError("selected candidate has no parent comparator")
        parent_record = runner._latest_by_id(ledger, parent_id)
        incumbent_path = ledger.artifact_path(parent_id) / "candidate.py"
        if parent_record.get("status") != "keep" or not incumbent_path.is_file():
            raise ConfirmationDriverError("selected parent comparator is invalid")
        candidate_sha = sha256_file(candidate_path)
        incumbent_sha = sha256_file(incumbent_path)
        if (
            candidate_sha != candidate_record.get("candidate_sha256")
            or incumbent_sha != parent_record.get("candidate_sha256")
        ):
            raise ConfirmationDriverError("archived controller hash mismatch")
        validate_candidate_source(candidate_path)
        validate_candidate_source(incumbent_path)
        driver_commit = runner._current_commit()
        driver_sha = _driver_sha256()
        committed_driver = runner._committed_file(
            driver_commit,
            "autoresearch/confirmation.py",
        )
        if not hmac.compare_digest(sha256_bytes(committed_driver), driver_sha):
            raise ConfirmationDriverError("driver working bytes differ from commit")
        plan = {
            **runner._base_plan_fields(
                manifest=manifest,
                parent_incumbent_id=parent_id,
                candidate_commit=str(candidate_record.get("candidate_commit") or ""),
                candidate_sha256=candidate_sha,
                changed_paths=(),
                hypothesis="One-time held-out confirmation of selected E0003.",
                predicted_effect=(
                    "Paired clean-success interval excludes zero favorably without "
                    "worse capture."
                ),
                seed_set_name=SEED_SET_NAME,
            ),
            "confirmation_authorized": True,
            "confirmation_set_spent": True,
            "explicit_user_authorization": True,
            "external_prepared": True,
            "external_mode": "confirmation",
            "external_stage": "awaiting_confirmation_aggregate",
            "confirmation_driver_commit": driver_commit,
            "confirmation_driver_sha256": driver_sha,
        }
        if resumable is None:
            ledger.plan_experiment({"experiment_id": EXPERIMENT_ID, **plan})
            running = ledger.start_experiment(EXPERIMENT_ID)
        elif resumable["status"] == "planned":
            for key, value in plan.items():
                if resumable.get(key) != value:
                    raise ConfirmationDriverError(
                        "interrupted confirmation plan differs from current driver",
                    )
            running = ledger.start_experiment(EXPERIMENT_ID)
        else:
            if resumable.get("confirmation_driver_sha256") != driver_sha:
                raise ConfirmationDriverError(
                    "interrupted confirmation uses another driver identity",
                )
            running = ledger.resume_experiment(
                EXPERIMENT_ID,
                evaluation_is_idempotent=True,
            )
        staging_path = ledger.artifact_staging_path(EXPERIMENT_ID)
        staging = ledger.begin_artifacts(
            EXPERIMENT_ID,
            resume=staging_path.exists(),
        )
        try:
            _write_new_or_same(staging / "candidate.py", candidate_path.read_bytes())
            _write_new_or_same(staging / "incumbent.py", incumbent_path.read_bytes())
            marker = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": AUTHORIZATION_ARTIFACT_TYPE,
                "authorized_at": _utc_now(),
                "explicit_user_authorization": True,
                "confirmation_set_spent": True,
                "experiment_id": EXPERIMENT_ID,
                "run_tag": run_tag,
                "run_manifest_sha256": _run_manifest_sha256(run_dir),
                "ledger_running_record_sha256": running["record_sha256"],
                "driver_commit": driver_commit,
                "driver_sha256": driver_sha,
                "candidate_experiment_id": candidate_record["experiment_id"],
                "candidate_commit": candidate_record["candidate_commit"],
                "candidate_sha256": candidate_sha,
                "incumbent_experiment_id": parent_id,
                "incumbent_commit": parent_record["candidate_commit"],
                "incumbent_sha256": incumbent_sha,
                "seed_set_id": seed_set.seed_set_id,
                "ordered_seed_sha256": ordered_seed_sha256(seed_set.seeds),
                "seed_start": seed_set.seeds[0],
                "seed_end": seed_set.seeds[-1],
                "episodes": len(seed_set.seeds),
            }
            marker_payload = _json_bytes(marker)
            assert_no_leaks(marker_payload, source="authorization.json")
            _write_new_or_same(staging / "authorization.json", marker_payload)
            marker_sha = hashlib.sha256(marker_payload).hexdigest()
            _write_new_or_same(
                staging / "authorization.sha256",
                (marker_sha + "\n").encode("ascii"),
            )
            return {
                "experiment_id": EXPERIMENT_ID,
                "status": "running",
                "confirmation_set_spent": True,
                "authorization_path": str(staging / "authorization.json"),
                "authorization_sha256": marker_sha,
                "driver_commit": driver_commit,
                "driver_sha256": driver_sha,
                "candidate_commit": marker["candidate_commit"],
                "candidate_sha256": candidate_sha,
                "incumbent_commit": marker["incumbent_commit"],
                "incumbent_sha256": incumbent_sha,
            }
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": "confirmation prepare failed"}
            _write_new_or_same(staging / "failure.json", _json_bytes(failure))
            ledger.finalize_experiment(
                EXPERIMENT_ID,
                status="crash",
                fields={
                    "checks": {"confirmation_prepare": False},
                    "decision_reason": "confirmation prepare failed after spending set",
                    "confirmation_set_spent": True,
                },
                artifact_staging=staging,
            )
            raise


def rollout_confirmation_shard(
    *,
    project_root: Path,
    results_root: Path,
    run_tag: str,
    config_path: Path,
    candidate_source: Path,
    incumbent_source: Path,
    authorization_path: Path,
    authorization_sha256: str,
    output_dir: Path,
    method: str,
    shard_index: int,
    shard_count: int,
) -> Mapping[str, Any]:
    if method not in EVALUATION_METHODS:
        raise ConfirmationDriverError(f"unknown confirmation method: {method}")
    (
        _manifest,
        loaded,
        partition,
        candidate,
        incumbent,
        base_identity,
        _bundle,
        horizon,
    ) = _load_context(
        project_root=project_root,
        results_root=results_root,
        run_tag=run_tag,
        config_path=config_path,
        candidate_source=candidate_source,
        incumbent_source=incumbent_source,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        shard_count=shard_count,
    )
    index = int(shard_index)
    if index < 0 or index >= len(partition):
        raise ConfirmationDriverError("confirmation shard index is out of range")
    shard = partition[index]
    factory = RealExp05EpisodeFactory.from_config(loaded.payload, project_root=project_root)
    shard_identity = shardlib._shard_identity(base_identity, shard=shard)
    controller_context: Any
    if method == "candidate":
        controller_context = IsolatedCandidateController.from_source(candidate.path)
    elif method == "incumbent":
        controller_context = IsolatedCandidateController.from_source(incumbent.path)
    else:
        controller_context = contextlib.nullcontext(None)
    records: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    with controller_context as controller:
        for seed in shard.seeds:
            first = shardlib._run_one_episode(
                factory,
                controller=controller,
                method=method,
                seed=seed,
                max_horizon=horizon,
                public_history_limit=int(base_identity["public_history_limit"]),
            )
            second = shardlib._run_one_episode(
                factory,
                controller=controller,
                method=method,
                seed=seed,
                max_horizon=horizon,
                public_history_limit=int(base_identity["public_history_limit"]),
            )
            shardlib.assert_deterministic_records((first,), (second,))
            records.append(first)
            enriched.append(
                {
                    **first,
                    "determinism_repeat": 2,
                    "identity": dict(shard_identity),
                    "seed_set_id": shard.seed_set_id,
                    "shard_count": shard.shard_count,
                    "shard_index": shard.shard_index,
                },
            )
    validate_episode_records(records, seeds=shard.seeds, methods=(method,))
    paths = shardlib.shard_paths(
        output_dir,
        method=method,
        shard_index=index,
        shard_count=shard_count,
    )
    record_payload = shardlib._jsonl_bytes(enriched)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SHARD_ARTIFACT_TYPE,
        "authorization_sha256": authorization_sha256,
        "determinism_verified": True,
        "experiment_id": EXPERIMENT_ID,
        "identity": shard_identity,
        "method": method,
        "records": {
            "count": len(enriched),
            "file": paths.records.name,
            "records_sha256": records_sha256(enriched),
            "sha256": sha256_bytes(record_payload),
        },
        "repeat": 2,
        "seed_set": {
            "id": shard.seed_set_id,
            "name": shard.seed_set_name,
            "one_time": True,
            "requires_explicit_authorization": True,
        },
        "shard": {
            "count": shard.shard_count,
            "index": shard.shard_index,
            "seeds": list(shard.seeds),
        },
        "sources": {
            "candidate_path": str(candidate.path),
            "candidate_sha256": candidate.sha256,
            "incumbent_path": str(incumbent.path),
            "incumbent_sha256": incumbent.sha256,
        },
    }
    assert_no_leaks(record_payload, source=paths.records.name)
    _atomic_write(paths.records, record_payload)
    _write_json(paths.manifest, manifest)
    return {
        "method": method,
        "shard_index": index,
        "seed_count": len(shard.seeds),
        "manifest_path": str(paths.manifest),
        "records_sha256": manifest["records"]["records_sha256"],
    }


def _confirmation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    checks: Mapping[str, bool],
    bootstrap_samples: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    statistics = confirmation_statistics(
        records,
        seeds=seeds,
        bootstrap_samples=int(bootstrap_samples),
    )
    candidate_captures = sum(
        bool(record["capture_episode"])
        for record in records
        if record["method"] == "candidate"
    )
    incumbent_captures = sum(
        bool(record["capture_episode"])
        for record in records
        if record["method"] == "incumbent"
    )
    capture_ok = candidate_captures <= incumbent_captures
    interval_ok = float(statistics["bootstrap_95_low"]) > 0.0
    passed = bool(checks) and all(checks.values()) and capture_ok and interval_ok
    gate = {
        "decision": "confirmed" if passed else "rejected",
        "confirmation_passed": passed,
        "capture_nonworsening": capture_ok,
        "favorable_interval": interval_ok,
        "candidate_capture_episodes": candidate_captures,
        "incumbent_capture_episodes": incumbent_captures,
        "decision_reason": (
            "held-out interval excludes zero favorably and capture did not worsen"
            if passed
            else "held-out confirmation failed at least one registered gate"
        ),
    }
    return gate, statistics


def aggregate_confirmation(
    *,
    project_root: Path,
    results_root: Path,
    run_tag: str,
    config_path: Path,
    candidate_source: Path,
    incumbent_source: Path,
    authorization_path: Path,
    authorization_sha256: str,
    output_dir: Path,
    shard_count: int,
) -> Mapping[str, Any]:
    (
        _manifest,
        loaded,
        partition,
        candidate,
        incumbent,
        base_identity,
        _bundle,
        _horizon,
    ) = _load_context(
        project_root=project_root,
        results_root=results_root,
        run_tag=run_tag,
        config_path=config_path,
        candidate_source=candidate_source,
        incumbent_source=incumbent_source,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        shard_count=shard_count,
    )
    shards_root = output_dir / "shards"
    actual_methods = {
        path.name
        for path in shards_root.iterdir()
        if path.is_dir() and any(path.glob("shard-*"))
    } if shards_root.is_dir() else set()
    if actual_methods != set(EVALUATION_METHODS):
        raise ConfirmationDriverError("confirmation method coverage is not exact")
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    shard_summaries: list[dict[str, Any]] = []
    for method in EVALUATION_METHODS:
        plain, deterministic, enriched = shardlib._load_method_shards(
            output_dir=output_dir,
            method=method,
            partition=partition,
            base_identity=base_identity,
            candidate=candidate,
            incumbent=incumbent,
        )
        if not deterministic:
            raise ConfirmationDriverError("confirmation shard determinism is incomplete")
        records_by_method[method] = plain
        shard_summaries.append(
            {
                "method": method,
                "records": len(enriched),
                "records_sha256": records_sha256(enriched),
            },
        )
    seeds = partition[0].all_seeds
    indices = {
        method: {int(record["seed"]): record for record in records_by_method[method]}
        for method in EVALUATION_METHODS
    }
    combined = [
        indices[method][seed]
        for seed in seeds
        for method in EVALUATION_METHODS
    ]
    validate_episode_records(combined, seeds=seeds, methods=EVALUATION_METHODS)
    checks = {
        "authorization_spent": True,
        "confirmation_driver_identity": True,
        "determinism": True,
        "identity_hashes": True,
        "records_complete": True,
        "shard_coverage": True,
        "source_guard": True,
    }
    gate, statistics = _confirmation_gate(
        combined,
        seeds=seeds,
        checks=checks,
        bootstrap_samples=int(loaded.payload["evaluation"]["bootstrap_samples"]),
    )
    summary = summarize_records(combined)
    enriched_records = [
        {**dict(record), "identity": dict(base_identity), "seed_set_id": partition[0].seed_set_id}
        for record in combined
    ]
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    statistics_path = output_dir / "statistics.json"
    gate_path = output_dir / "gate.json"
    manifest_path = output_dir / "aggregate.manifest.json"
    record_payload = shardlib._jsonl_bytes(enriched_records)
    assert_no_leaks(record_payload, source="records.jsonl")
    _atomic_write(records_path, record_payload)
    _write_json(summary_path, summary)
    _write_json(statistics_path, statistics)
    _write_json(gate_path, gate)
    aggregate_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": AGGREGATE_ARTIFACT_TYPE,
        "experiment_id": EXPERIMENT_ID,
        "authorization_sha256": authorization_sha256,
        "identity": base_identity,
        "checks": checks,
        "seed_set": {
            "id": partition[0].seed_set_id,
            "name": SEED_SET_NAME,
            "episodes": len(seeds),
            "one_time": True,
            "spent": True,
        },
        "shard_count": len(partition),
        "artifacts": {
            "records": records_path.name,
            "records_file_sha256": sha256_file(records_path),
            "records_sha256": records_sha256(combined),
            "summary": summary_path.name,
            "summary_sha256": sha256_file(summary_path),
            "statistics": statistics_path.name,
            "statistics_sha256": sha256_file(statistics_path),
            "gate": gate_path.name,
            "gate_sha256": sha256_file(gate_path),
        },
        "shards": shard_summaries,
    }
    _write_json(manifest_path, aggregate_manifest)
    return {
        "aggregate_manifest_path": str(manifest_path),
        "decision": gate["decision"],
        "confirmation_passed": gate["confirmation_passed"],
        "records_sha256": records_sha256(combined),
    }


def _load_aggregate_records(
    manifest_path: Path,
    *,
    expected_identity: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    aggregate = _read_json(manifest_path)
    if (
        aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("artifact_type") != AGGREGATE_ARTIFACT_TYPE
        or aggregate.get("experiment_id") != EXPERIMENT_ID
        or aggregate.get("identity") != expected_identity
    ):
        raise ConfirmationDriverError("confirmation aggregate identity mismatch")
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ConfirmationDriverError("confirmation aggregate artifacts missing")
    records_name = str(artifacts.get("records", ""))
    if not records_name or Path(records_name).name != records_name:
        raise ConfirmationDriverError("confirmation records path is not portable")
    records_path = manifest_path.parent / records_name
    if records_path.is_symlink() or not records_path.is_file():
        raise ConfirmationDriverError("confirmation records file is missing or unsafe")
    payload = records_path.read_bytes()
    assert_no_leaks(payload, source=records_path.name)
    if not hmac.compare_digest(
        hashlib.sha256(payload).hexdigest(),
        str(artifacts.get("records_file_sha256", "")),
    ):
        raise ConfirmationDriverError("confirmation records file hash mismatch")
    enriched: list[dict[str, Any]] = []
    for line in payload.splitlines():
        value = json.loads(line)
        if (
            not isinstance(value, Mapping)
            or value.get("identity") != expected_identity
            or value.get("seed_set_id")
            != expected_identity.get("seed_set_id")
        ):
            raise ConfirmationDriverError("confirmation record identity mismatch")
        enriched.append(dict(value))
    plain = [shardlib._plain_record(record) for record in enriched]
    validate_episode_records(plain, seeds=seeds, methods=EVALUATION_METHODS)
    if not hmac.compare_digest(
        records_sha256(plain),
        str(artifacts.get("records_sha256", "")),
    ):
        raise ConfirmationDriverError("confirmation record content hash mismatch")
    return aggregate, plain


def finalize_confirmation(
    *,
    repo_root: Path,
    results_root: Path,
    run_tag: str,
    aggregate_manifest_path: Path,
) -> Mapping[str, Any]:
    runner = AutoresearchRunner(repo_root=repo_root, results_root=results_root)
    manifest = runner._load_run(run_tag, verify_sources=True)
    run_dir = results_root / run_tag
    with runner._run_lock(run_dir):
        ledger = ExperimentLedger(run_dir)
        latest = ledger.latest_records().get(EXPERIMENT_ID)
        if latest is None:
            raise ConfirmationDriverError("confirmation was not prepared")
        if latest["status"] in TERMINAL_STATUSES:
            return latest
        if (
            latest.get("status") != "running"
            or latest.get("confirmation_set_spent") is not True
            or latest.get("external_mode") != "confirmation"
        ):
            raise ConfirmationDriverError("confirmation ledger state is not running/spent")
        staging = ledger.begin_artifacts(EXPERIMENT_ID, resume=True)
        ledger.resume_experiment(EXPERIMENT_ID, evaluation_is_idempotent=True)
        authorization_path = staging / "authorization.json"
        authorization_sha = (staging / "authorization.sha256").read_text(
            encoding="ascii",
        ).strip()
        candidate_path = staging / "candidate.py"
        incumbent_path = staging / "incumbent.py"
        config_path = repo_root / str(manifest["config_path"])
        aggregate_header = _read_json(aggregate_manifest_path)
        shard_count = int(aggregate_header.get("shard_count", 0))
        if shard_count <= 0:
            raise ConfirmationDriverError("confirmation aggregate shard count is invalid")
        (
            _manifest,
            loaded,
            partition,
            _candidate,
            _incumbent,
            expected_identity,
            _bundle,
            _horizon,
        ) = _load_context(
            project_root=repo_root,
            results_root=results_root,
            run_tag=run_tag,
            config_path=config_path,
            candidate_source=candidate_path,
            incumbent_source=incumbent_path,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha,
            shard_count=shard_count,
        )
        # The aggregate-recorded partition is reconstructed against the
        # registered seed set; exact coverage is checked again below.
        seeds = partition[0].all_seeds
        aggregate, records = _load_aggregate_records(
            aggregate_manifest_path,
            expected_identity=expected_identity,
            seeds=seeds,
        )
        checks = aggregate.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(
            value is True for value in checks.values()
        ):
            raise ConfirmationDriverError("confirmation aggregate hard checks failed")
        summary = summarize_records(records)
        gate, statistics = _confirmation_gate(
            records,
            seeds=seeds,
            checks={str(key): bool(value) for key, value in checks.items()},
            bootstrap_samples=int(loaded.payload["evaluation"]["bootstrap_samples"]),
        )
        artifacts = aggregate["artifacts"]
        comparisons = {
            "summary": summary,
            "statistics": statistics,
            "gate": gate,
        }
        for name, recomputed in comparisons.items():
            artifact_name = str(artifacts[name])
            if not artifact_name or Path(artifact_name).name != artifact_name:
                raise ConfirmationDriverError(f"confirmation {name} path is unsafe")
            path = aggregate_manifest_path.parent / artifact_name
            if path.is_symlink() or not path.is_file():
                raise ConfirmationDriverError(f"confirmation {name} file is unsafe")
            if not hmac.compare_digest(sha256_file(path), str(artifacts[f"{name}_sha256"])):
                raise ConfirmationDriverError(f"confirmation {name} file hash mismatch")
            if _read_json(path) != recomputed:
                raise ConfirmationDriverError(f"confirmation {name} differs on local rebuild")
        envelope = {
            "confirmation_passed": bool(gate["confirmation_passed"]),
            "capture_nonworsening": bool(gate["capture_nonworsening"]),
            "favorable_interval": bool(gate["favorable_interval"]),
            "statistics": statistics,
            "summary": summary,
            "checks": dict(checks),
            "records_sha256": records_sha256(records),
            "external_aggregate_sha256": sha256_file(aggregate_manifest_path),
        }
        _write_new_or_same(staging / "confirmation.json", _json_bytes(envelope))
        _write_new_or_same(
            staging / "external-aggregate.manifest.json",
            aggregate_manifest_path.read_bytes(),
        )
        reason = (
            "one-time held-out confirmation passed the engineering gate"
            if gate["confirmation_passed"]
            else "one-time held-out confirmation did not pass the engineering gate"
        )
        return ledger.finalize_experiment(
            EXPERIMENT_ID,
            status="discard",
            fields={
                "primary_delta": float(statistics["mean_delta"]),
                "paired_counts": {
                    "candidate_only_successes": int(statistics["candidate_only_successes"]),
                    "incumbent_only_successes": int(statistics["incumbent_only_successes"]),
                },
                "secondary_metrics": {
                    "confirmation_statistics": dict(statistics),
                    "summary": summary,
                },
                "checks": dict(checks),
                "decision_reason": reason,
                "confirmation_passed": bool(gate["confirmation_passed"]),
                "confirmation_set_spent": True,
                "external_evaluation": True,
                "external_stage": "finalized",
            },
            artifact_staging=staging,
        )


def abort_confirmation(
    *,
    repo_root: Path,
    results_root: Path,
    run_tag: str,
    reason: str,
) -> Mapping[str, Any]:
    normalized = " ".join(reason.split())
    if not normalized or len(normalized.encode("utf-8")) > 4096:
        raise ConfirmationDriverError("confirmation abort reason is invalid")
    assert_no_leaks(normalized, source="confirmation_abort_reason")
    runner = AutoresearchRunner(repo_root=repo_root, results_root=results_root)
    runner._load_run(run_tag)
    run_dir = results_root / run_tag
    with runner._run_lock(run_dir):
        ledger = ExperimentLedger(run_dir)
        latest = ledger.latest_records().get(EXPERIMENT_ID)
        if latest is None or latest.get("status") != "running":
            raise ConfirmationDriverError("confirmation abort target is not running")
        staging = ledger.begin_artifacts(EXPERIMENT_ID, resume=True)
        _write_new_or_same(
            staging / "confirmation-abort.json",
            _json_bytes({"experiment_id": EXPERIMENT_ID, "reason": normalized}),
        )
        return ledger.finalize_experiment(
            EXPERIMENT_ID,
            status="crash",
            fields={
                "checks": {"confirmation_abort_recorded": True},
                "decision_reason": normalized,
                "confirmation_set_spent": True,
                "external_stage": "aborted",
            },
            artifact_staging=staging,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-tag", required=True)
    prepare.add_argument("--authorize-confirmation", action="store_true")
    for command in ("rollout", "aggregate"):
        child = sub.add_parser(command)
        child.add_argument("--run-tag", required=True)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--candidate-source", type=Path, required=True)
        child.add_argument("--incumbent-source", type=Path, required=True)
        child.add_argument("--authorization", type=Path, required=True)
        child.add_argument("--authorization-sha256", required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--shard-count", type=int, required=True)
    rollout = sub.choices["rollout"]
    rollout.add_argument("--method", choices=EVALUATION_METHODS, required=True)
    rollout.add_argument("--shard-index", type=int, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-tag", required=True)
    finalize.add_argument("--aggregate-manifest", type=Path, required=True)
    abort = sub.add_parser("abort")
    abort.add_argument("--run-tag", required=True)
    abort.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else repo_root / "results" / "autoresearch"
    )
    try:
        if args.command == "prepare":
            result = prepare_confirmation(
                repo_root=repo_root,
                results_root=results_root,
                run_tag=args.run_tag,
                authorized=bool(args.authorize_confirmation),
            )
        elif args.command == "rollout":
            result = rollout_confirmation_shard(
                project_root=repo_root,
                results_root=results_root,
                run_tag=args.run_tag,
                config_path=args.config,
                candidate_source=args.candidate_source,
                incumbent_source=args.incumbent_source,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
                output_dir=args.output_dir,
                method=args.method,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        elif args.command == "aggregate":
            result = aggregate_confirmation(
                project_root=repo_root,
                results_root=results_root,
                run_tag=args.run_tag,
                config_path=args.config,
                candidate_source=args.candidate_source,
                incumbent_source=args.incumbent_source,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
                output_dir=args.output_dir,
                shard_count=args.shard_count,
            )
        elif args.command == "finalize":
            result = finalize_confirmation(
                repo_root=repo_root,
                results_root=results_root,
                run_tag=args.run_tag,
                aggregate_manifest_path=args.aggregate_manifest.expanduser().resolve(),
            )
        else:
            result = abort_confirmation(
                repo_root=repo_root,
                results_root=results_root,
                run_tag=args.run_tag,
                reason=args.reason,
            )
        print(_canonical_json(result))
        return 0 if result.get("status") != "crash" else 3
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        message = " ".join(str(exc).split())[:800]
        print(f"confirmation failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
