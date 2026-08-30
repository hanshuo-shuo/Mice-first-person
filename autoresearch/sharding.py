"""Strict sharded execution for the frozen autoresearch gaze evaluator.

The normal phase-1 runner is responsible for experiment bookkeeping and the
one-time local confirmation flow.  Independent sharded workers categorically
refuse confirmation.  This module owns only deterministic development/smoke
rollout shards, comparator-cache construction, and strict aggregation.  It has
no seed, horizon, checkpoint, or resolved-config override: those values are
resolved from the registered autoresearch config or the verified real factory.

Quest entry points::

    python -B -m autoresearch.sharding rollout ...
    python -B -m autoresearch.sharding aggregate ...

Each rollout writes one JSONL file followed by one atomic manifest.  Aggregate
accepts a shard only when its exact registered seed slice, source identities,
code identities, record order, record hash, and per-record identities match.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hmac
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from autoresearch import contract as contract_module
from autoresearch import evaluator as evaluator_module
from autoresearch import guard as guard_module
from autoresearch import worker as worker_module
from autoresearch.evaluator import (
    COMPARATOR_METHODS,
    EVALUATION_METHODS,
    REFERENCE_METHOD,
    ComparatorCacheIdentity,
    DeterminismError,
    EpisodeAdapter,
    EvaluationError,
    RealExp05EpisodeFactory,
    assert_deterministic_records,
    canonical_sha256,
    load_comparator_cache,
    mechanical_keep_or_discard,
    ordered_seed_sha256,
    records_sha256,
    run_frozen_episode,
    seed_set_from_config,
    summarize_records,
    validate_episode_records,
    verify_exp05_artifacts,
    write_comparator_cache,
)
from autoresearch.guard import (
    GuardError,
    assert_no_leaks,
    build_hash_manifest,
    manifest_sha256,
    sha256_bytes,
    sha256_file,
    validate_candidate_source,
)
from autoresearch.worker import IsolatedCandidateController


SHARD_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
ALLOWED_REPEATS = (1, 2)
BASELINE_MODE = "baseline"
EXPERIMENT_MODE = "experiment"
AGGREGATE_MODES = (BASELINE_MODE, EXPERIMENT_MODE)

# These files jointly own the environment, renderer, real factory, and exact
# state helper used by the EXP-05 adapter.  Vendored Cellworld source and its
# renderer assets are added recursively by ``environment_contract_manifest``.
ENVIRONMENT_CONTRACT_FILES = (
    "benchmarks/peekbench/environment.py",
    "botevade_gym.py",
    "first_person.py",
    "policies/binocular_sac.py",
    "reward.py",
    "training/first_person_sac.py",
    "util.py",
)
ENVIRONMENT_CONTRACT_TREE = "cellworld_game-main/cellworld_game"
ENVIRONMENT_CONTRACT_SUFFIXES = frozenset({".py", ".png"})
ENVIRONMENT_ASSET_TREE = "cellworld_cache"


class ShardingError(EvaluationError):
    """A shard or aggregate violated the registered evaluation design."""


@dataclasses.dataclass(frozen=True)
class LoadedConfig:
    path: Path | None
    payload: Mapping[str, Any]
    sha256: str


@dataclasses.dataclass(frozen=True)
class LoadedController:
    path: Path
    sha256: str


@dataclasses.dataclass(frozen=True)
class RegisteredShard:
    seed_set_name: str
    seed_set_id: str
    all_seeds: tuple[int, ...]
    shard_index: int
    shard_count: int
    seeds: tuple[int, ...]
    one_time: bool
    requires_explicit_authorization: bool


@dataclasses.dataclass(frozen=True)
class ShardPaths:
    records: Path
    manifest: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (_canonical_json(dict(payload)) + "\n").encode("utf-8")
    assert_no_leaks(encoded, source=path.name)
    _atomic_write_bytes(path, encoded)


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(dict(record)) + "\n").encode("utf-8")
        for record in records
    )


def _resolve_path(path: str | os.PathLike[str], *, project_root: Path) -> Path:
    raw = Path(path).expanduser()
    return (raw if raw.is_absolute() else project_root / raw).resolve()


def load_registered_config(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
) -> LoadedConfig:
    root = Path(project_root).expanduser().resolve()
    resolved = _resolve_path(path, project_root=root)
    if not resolved.is_file() or resolved.is_symlink():
        raise ShardingError(f"Registered config is missing or is a symlink: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ShardingError(f"Cannot read registered config: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ShardingError("Registered autoresearch config must be a mapping")
    if int(payload.get("schema_version", -1)) != 1:
        raise ShardingError("Unsupported autoresearch config schema_version")
    return LoadedConfig(
        path=resolved,
        payload=payload,
        sha256=sha256_file(resolved),
    )


def config_from_mapping(config: Mapping[str, Any]) -> LoadedConfig:
    """Create the same immutable config identity for fake-backend tests."""

    if not isinstance(config, Mapping):
        raise ShardingError("Autoresearch config must be a mapping")
    plain = dict(config)
    if int(plain.get("schema_version", -1)) != 1:
        raise ShardingError("Unsupported autoresearch config schema_version")
    return LoadedConfig(path=None, payload=plain, sha256=canonical_sha256(plain))


def _reject_confirmation(seed_set: Any) -> None:
    if seed_set.requires_explicit_authorization:
        raise ShardingError(
            "CONFIRMATION REFUSED: independent sharded workers cannot authorize "
            "or spend the confirmation set; use the runner-owned local confirm flow",
        )


def registered_shard(
    config: Mapping[str, Any],
    *,
    seed_set_name: str,
    shard_index: int,
    shard_count: int,
) -> RegisteredShard:
    """Resolve one balanced contiguous slice of a registered ordered seed set."""

    seed_set = seed_set_from_config(config, seed_set_name)
    _reject_confirmation(seed_set)
    count = int(shard_count)
    index = int(shard_index)
    if count <= 0:
        raise ShardingError("shard_count must be positive")
    if count > len(seed_set.seeds):
        raise ShardingError("shard_count may not exceed registered episode count")
    if index < 0 or index >= count:
        raise ShardingError("shard_index lies outside the registered shard range")
    quotient, remainder = divmod(len(seed_set.seeds), count)
    start = index * quotient + min(index, remainder)
    stop = start + quotient + int(index < remainder)
    seeds = tuple(seed_set.seeds[start:stop])
    if not seeds:
        raise ShardingError("registered shard unexpectedly has no seeds")
    return RegisteredShard(
        seed_set_name=seed_set.name,
        seed_set_id=seed_set.seed_set_id,
        all_seeds=tuple(seed_set.seeds),
        shard_index=index,
        shard_count=count,
        seeds=seeds,
        one_time=seed_set.one_time,
        requires_explicit_authorization=seed_set.requires_explicit_authorization,
    )


def validate_registered_partition(
    config: Mapping[str, Any],
    *,
    seed_set_name: str,
    shard_count: int,
) -> tuple[RegisteredShard, ...]:
    shards = tuple(
        registered_shard(
            config,
            seed_set_name=seed_set_name,
            shard_index=index,
            shard_count=shard_count,
        )
        for index in range(int(shard_count))
    )
    flattened = tuple(seed for shard in shards for seed in shard.seeds)
    expected = shards[0].all_seeds if shards else ()
    if flattened != expected or len(set(flattened)) != len(flattened):
        raise ShardingError("registered shards do not form one exact ordered partition")
    return shards


def shard_paths(
    output_dir: str | os.PathLike[str],
    *,
    method: str,
    shard_index: int,
    shard_count: int,
) -> ShardPaths:
    if method not in EVALUATION_METHODS:
        raise ShardingError(f"Unknown evaluation method: {method!r}")
    stem = f"shard-{int(shard_index):04d}-of-{int(shard_count):04d}"
    method_dir = Path(output_dir) / "shards" / method
    return ShardPaths(
        records=method_dir / f"{stem}.jsonl",
        manifest=method_dir / f"{stem}.manifest.json",
    )


def load_guarded_controller(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = ".",
) -> LoadedController:
    """Hash and statically validate an explicitly named controller source."""

    root = Path(project_root).expanduser().resolve()
    raw = Path(path).expanduser()
    unresolved = raw if raw.is_absolute() else root / raw
    if unresolved.is_symlink():
        raise ShardingError(f"Controller source is missing or is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ShardingError(f"Controller source is missing or is a symlink: {resolved}")
    digest = sha256_file(resolved)
    validate_candidate_source(resolved)
    # Hash again after static validation to reject a source changed during read.
    after = sha256_file(resolved)
    if not hmac.compare_digest(digest, after):
        raise ShardingError(f"Controller source changed while loading: {resolved}")
    return LoadedController(path=resolved, sha256=digest)


def environment_contract_manifest(
    project_root: str | os.PathLike[str] = ".",
) -> dict[str, str]:
    """Hash the trusted first-person environment and vendored simulator source."""

    root = Path(project_root).expanduser().resolve()
    relative_paths = list(ENVIRONMENT_CONTRACT_FILES)
    tree = root / ENVIRONMENT_CONTRACT_TREE
    if not tree.is_dir():
        raise ShardingError(f"Environment contract tree is missing: {tree}")
    for path in sorted(tree.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in ENVIRONMENT_CONTRACT_SUFFIXES
            and "__pycache__" not in path.parts
        ):
            relative_paths.append(path.relative_to(root).as_posix())
    asset_tree = root / ENVIRONMENT_ASSET_TREE
    if not asset_tree.is_dir():
        raise ShardingError(f"Environment asset tree is missing: {asset_tree}")
    for path in sorted(asset_tree.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative_paths.append(path.relative_to(root).as_posix())
    try:
        return build_hash_manifest(root, relative_paths)
    except GuardError as exc:
        raise ShardingError("Cannot build the frozen environment contract hash") from exc


def environment_contract_sha256(
    project_root: str | os.PathLike[str] = ".",
) -> str:
    return manifest_sha256(environment_contract_manifest(project_root))


def _runtime_code_hashes() -> dict[str, str]:
    paths = {
        "contract_sha256": Path(contract_module.__file__).resolve(),
        "evaluator_sha256": Path(evaluator_module.__file__).resolve(),
        "guard_sha256": Path(guard_module.__file__).resolve(),
        "sharding_sha256": Path(__file__).resolve(),
        "worker_sha256": Path(worker_module.__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _factory_horizon(episode_factory: Callable[[], EpisodeAdapter]) -> int:
    value = int(getattr(episode_factory, "max_horizon", 0))
    if value <= 0:
        value = int(getattr(episode_factory, "horizon", 0))
    if value <= 0:
        raise ShardingError("Episode factory must expose a positive frozen max_horizon")
    return value


def _base_identity(
    *,
    loaded_config: LoadedConfig,
    shard: RegisteredShard,
    candidate: LoadedController,
    incumbent: LoadedController,
    max_horizon: int,
    environment_digest: str,
    artifact_bundle: Any | None,
) -> dict[str, Any]:
    config = loaded_config.payload
    source = config.get("source")
    evaluation = config.get("evaluation")
    if not isinstance(source, Mapping) or not isinstance(evaluation, Mapping):
        raise ShardingError("Config requires source and evaluation mappings")
    registered_checkpoint = str(source.get("checkpoint_sha256", "")).strip()
    registered_resolved = str(source.get("resolved_config_sha256", "")).strip()
    if not registered_checkpoint or not registered_resolved:
        raise ShardingError("Config must register checkpoint and resolved-config hashes")
    if artifact_bundle is not None:
        if not hmac.compare_digest(
            str(artifact_bundle.checkpoint_sha256),
            registered_checkpoint,
        ) or not hmac.compare_digest(
            str(artifact_bundle.resolved_config_sha256),
            registered_resolved,
        ):
            raise ShardingError("Verified real artifacts differ from the registered config")
    history_limit = int(evaluation.get("maximum_history_length", -1))
    if history_limit < 0:
        raise ShardingError("Config maximum_history_length must be non-negative")
    identity: dict[str, Any] = {
        "candidate_sha256": candidate.sha256,
        "checkpoint_sha256": registered_checkpoint,
        "config_sha256": loaded_config.sha256,
        "environment_contract_sha256": str(environment_digest),
        "incumbent_sha256": incumbent.sha256,
        "max_horizon": int(max_horizon),
        "ordered_seed_sha256": ordered_seed_sha256(shard.all_seeds),
        "public_history_limit": history_limit,
        "resolved_config_sha256": registered_resolved,
        "seed_set_id": shard.seed_set_id,
        **_runtime_code_hashes(),
    }
    if any(
        not str(value).strip()
        for name, value in identity.items()
        if name.endswith("_sha256")
    ):
        raise ShardingError("One or more required identity hashes are empty")
    identity["run_identity_sha256"] = canonical_sha256(identity)
    return identity


def _shard_identity(
    base_identity: Mapping[str, Any],
    *,
    shard: RegisteredShard,
) -> dict[str, Any]:
    identity = {
        **dict(base_identity),
        "shard_seed_sha256": ordered_seed_sha256(shard.seeds),
    }
    identity["shard_identity_sha256"] = canonical_sha256(identity)
    return identity


def _run_one_episode(
    episode_factory: Callable[[], EpisodeAdapter],
    *,
    controller: Any | None,
    method: str,
    seed: int,
    max_horizon: int,
    public_history_limit: int,
) -> dict[str, Any]:
    episode = episode_factory()
    try:
        return run_frozen_episode(
            episode,
            controller=controller,
            method=method,
            seed=seed,
            max_horizon=max_horizon,
            public_history_limit=public_history_limit,
        )
    finally:
        episode.close()


def rollout_shard(
    *,
    config: Mapping[str, Any] | LoadedConfig,
    seed_set_name: str,
    method: str,
    shard_index: int,
    shard_count: int,
    output_dir: str | os.PathLike[str],
    candidate_source: str | os.PathLike[str],
    incumbent_source: str | os.PathLike[str],
    project_root: str | os.PathLike[str] = ".",
    repeat: int = 2,
    episode_factory: Callable[[], EpisodeAdapter] | None = None,
    environment_digest: str | None = None,
) -> dict[str, Any]:
    """Run and atomically persist one exact registered method/seed shard.

    ``episode_factory`` and ``environment_digest`` are injection points for
    cheap tests.  The CLI never exposes them and always constructs the verified
    real EXP-05 factory from the config.
    """

    if method not in EVALUATION_METHODS:
        raise ShardingError(f"Unknown evaluation method: {method!r}")
    repetitions = int(repeat)
    if repetitions not in ALLOWED_REPEATS:
        raise ShardingError("repeat must be 1 or 2; use 2 for determinism evidence")
    root = Path(project_root).expanduser().resolve()
    loaded = config if isinstance(config, LoadedConfig) else config_from_mapping(config)
    shard = registered_shard(
        loaded.payload,
        seed_set_name=seed_set_name,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    candidate = load_guarded_controller(candidate_source, project_root=root)
    incumbent = load_guarded_controller(incumbent_source, project_root=root)

    selected_factory = episode_factory
    if selected_factory is None:
        selected_factory = RealExp05EpisodeFactory.from_config(
            loaded.payload,
            project_root=root,
        )
    max_horizon = _factory_horizon(selected_factory)
    artifact_bundle = getattr(selected_factory, "artifacts", None)
    selected_environment_digest = str(
        environment_digest or environment_contract_sha256(root),
    )
    base_identity = _base_identity(
        loaded_config=loaded,
        shard=shard,
        candidate=candidate,
        incumbent=incumbent,
        max_horizon=max_horizon,
        environment_digest=selected_environment_digest,
        artifact_bundle=artifact_bundle,
    )
    identity = _shard_identity(base_identity, shard=shard)
    public_history_limit = int(base_identity["public_history_limit"])
    controller_context: Any
    if method == "candidate":
        controller_context = IsolatedCandidateController.from_source(candidate.path)
    elif method == "incumbent":
        controller_context = IsolatedCandidateController.from_source(incumbent.path)
    else:
        controller_context = contextlib.nullcontext(None)

    plain_records: list[dict[str, Any]] = []
    enriched_records: list[dict[str, Any]] = []
    with controller_context as controller:
        for seed in shard.seeds:
            repeated = [
                _run_one_episode(
                    selected_factory,
                    controller=controller,
                    method=method,
                    seed=seed,
                    max_horizon=max_horizon,
                    public_history_limit=public_history_limit,
                )
                for _ in range(repetitions)
            ]
            if repetitions == 2:
                assert_deterministic_records(repeated[:1], repeated[1:])
            record = dict(repeated[0])
            plain_records.append(record)
            enriched_records.append(
                {
                    **record,
                    "determinism_repeat": repetitions,
                    "identity": dict(identity),
                    "seed_set_id": shard.seed_set_id,
                    "shard_count": shard.shard_count,
                    "shard_index": shard.shard_index,
                },
            )
    validate_episode_records(plain_records, seeds=shard.seeds, methods=(method,))

    paths = shard_paths(
        output_dir,
        method=method,
        shard_index=shard.shard_index,
        shard_count=shard.shard_count,
    )
    records_payload = _jsonl_bytes(enriched_records)
    assert_no_leaks(records_payload, source=paths.records.name)
    manifest = {
        "artifact_type": "autoresearch_rollout_shard",
        "determinism_verified": repetitions == 2,
        "experiment_id": str(loaded.payload.get("experiment_id", "")),
        "identity": identity,
        "method": method,
        "records": {
            "count": len(enriched_records),
            "file": paths.records.name,
            "records_sha256": records_sha256(enriched_records),
            "sha256": sha256_bytes(records_payload),
        },
        "repeat": repetitions,
        "schema_version": SHARD_SCHEMA_VERSION,
        "seed_set": {
            "id": shard.seed_set_id,
            "name": shard.seed_set_name,
            "one_time": shard.one_time,
            "requires_explicit_authorization": (
                shard.requires_explicit_authorization
            ),
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
    # The manifest is the commit marker: an interrupted records write cannot
    # create a shard that aggregate will mistake for complete.
    _atomic_write_bytes(paths.records, records_payload)
    _atomic_write_json(paths.manifest, manifest)
    return {
        "manifest_path": str(paths.manifest),
        "method": method,
        "records_path": str(paths.records),
        "records_sha256": manifest["records"]["records_sha256"],
        "run_identity_sha256": base_identity["run_identity_sha256"],
        "seed_count": len(shard.seeds),
        "shard_count": shard.shard_count,
        "shard_index": shard.shard_index,
    }


def _read_json_mapping(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise ShardingError(f"Manifest has invalid size: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShardingError(f"Cannot read shard manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ShardingError(f"Shard manifest must be a mapping: {path}")
    return value


def _read_jsonl(path: Path, *, expected_count: int) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ShardingError(f"Cannot read shard records: {path}") from exc
    if not payload or len(payload) > 64 * 1024 * 1024:
        raise ShardingError(f"Shard records have invalid size: {path}")
    assert_no_leaks(payload, source=path.name)
    lines = payload.splitlines()
    if len(lines) != int(expected_count):
        raise ShardingError(f"Shard JSONL record count mismatch: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShardingError(
                f"Invalid JSONL record at {path.name}:{line_number}",
            ) from exc
        if not isinstance(value, Mapping):
            raise ShardingError(
                f"JSONL record must be a mapping at {path.name}:{line_number}",
            )
        records.append(dict(value))
    return records, payload


_SHARD_METADATA_FIELDS = frozenset(
    {
        "determinism_repeat",
        "identity",
        "seed_set_id",
        "shard_count",
        "shard_index",
    },
)


def _plain_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if key not in _SHARD_METADATA_FIELDS
    }


def _load_method_shards(
    *,
    output_dir: Path,
    method: str,
    partition: Sequence[RegisteredShard],
    base_identity: Mapping[str, Any],
    candidate: LoadedController,
    incumbent: LoadedController,
) -> tuple[list[dict[str, Any]], bool, list[dict[str, Any]]]:
    all_plain: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    deterministic = True
    expected_paths = {
        shard_paths(
            output_dir,
            method=method,
            shard_index=shard.shard_index,
            shard_count=shard.shard_count,
        )
        for shard in partition
    }
    method_dir = output_dir / "shards" / method
    actual_manifests = set(method_dir.glob("shard-*.manifest.json"))
    expected_manifests = {paths.manifest for paths in expected_paths}
    if actual_manifests != expected_manifests:
        missing = sorted(expected_manifests.difference(actual_manifests))
        unexpected = sorted(actual_manifests.difference(expected_manifests))
        if missing:
            raise ShardingError(f"Required shard manifest is missing: {missing[0]}")
        raise ShardingError(f"Unexpected/duplicate shard manifest: {unexpected[0]}")
    actual_records = set(method_dir.glob("shard-*.jsonl"))
    expected_records = {paths.records for paths in expected_paths}
    if actual_records != expected_records:
        missing = sorted(expected_records.difference(actual_records))
        unexpected = sorted(actual_records.difference(expected_records))
        if missing:
            raise ShardingError(f"Required shard records are missing: {missing[0]}")
        raise ShardingError(f"Unexpected/duplicate shard records: {unexpected[0]}")
    for shard in partition:
        paths = shard_paths(
            output_dir,
            method=method,
            shard_index=shard.shard_index,
            shard_count=shard.shard_count,
        )
        if not paths.manifest.is_file():
            raise ShardingError(f"Required shard manifest is missing: {paths.manifest}")
        manifest = _read_json_mapping(paths.manifest)
        expected_identity = _shard_identity(base_identity, shard=shard)
        if int(manifest.get("schema_version", -1)) != SHARD_SCHEMA_VERSION:
            raise ShardingError(f"Shard schema mismatch: {paths.manifest}")
        if manifest.get("artifact_type") != "autoresearch_rollout_shard":
            raise ShardingError(f"Unexpected shard artifact type: {paths.manifest}")
        if str(manifest.get("method", "")) != method:
            raise ShardingError(f"Shard method mismatch: {paths.manifest}")
        if manifest.get("identity") != expected_identity:
            raise ShardingError(f"Shard identity mismatch: {paths.manifest}")
        expected_shard = {
            "count": shard.shard_count,
            "index": shard.shard_index,
            "seeds": list(shard.seeds),
        }
        if manifest.get("shard") != expected_shard:
            raise ShardingError(f"Shard seed coverage/order mismatch: {paths.manifest}")
        expected_seed_set = {
            "id": shard.seed_set_id,
            "name": shard.seed_set_name,
            "one_time": shard.one_time,
            "requires_explicit_authorization": (
                shard.requires_explicit_authorization
            ),
        }
        if manifest.get("seed_set") != expected_seed_set:
            raise ShardingError(f"Shard seed-set identity mismatch: {paths.manifest}")
        sources = manifest.get("sources")
        if not isinstance(sources, Mapping) or sources != {
            "candidate_path": str(candidate.path),
            "candidate_sha256": candidate.sha256,
            "incumbent_path": str(incumbent.path),
            "incumbent_sha256": incumbent.sha256,
        }:
            raise ShardingError(f"Shard controller source mismatch: {paths.manifest}")
        records_info = manifest.get("records")
        if not isinstance(records_info, Mapping):
            raise ShardingError(f"Shard records manifest is missing: {paths.manifest}")
        if str(records_info.get("file", "")) != paths.records.name:
            raise ShardingError(f"Shard record filename mismatch: {paths.manifest}")
        expected_count = len(shard.seeds)
        if int(records_info.get("count", -1)) != expected_count:
            raise ShardingError(f"Shard record count mismatch: {paths.manifest}")
        records, payload = _read_jsonl(paths.records, expected_count=expected_count)
        if not hmac.compare_digest(
            str(records_info.get("sha256", "")),
            sha256_bytes(payload),
        ) or not hmac.compare_digest(
            str(records_info.get("records_sha256", "")),
            records_sha256(records),
        ):
            raise ShardingError(f"Shard record content hash mismatch: {paths.records}")
        repeat = int(manifest.get("repeat", 0))
        if repeat not in ALLOWED_REPEATS:
            raise ShardingError(f"Shard repeat value is invalid: {paths.manifest}")
        shard_deterministic = manifest.get("determinism_verified") is True and repeat == 2
        deterministic = deterministic and shard_deterministic
        for record, expected_seed in zip(records, shard.seeds, strict=True):
            if record.get("identity") != expected_identity:
                raise ShardingError(f"Per-record identity mismatch: {paths.records}")
            if (
                str(record.get("seed_set_id", "")) != shard.seed_set_id
                or int(record.get("shard_count", -1)) != shard.shard_count
                or int(record.get("shard_index", -1)) != shard.shard_index
                or int(record.get("determinism_repeat", -1)) != repeat
                or int(record.get("seed", -1)) != expected_seed
                or str(record.get("method", "")) != method
            ):
                raise ShardingError(f"Per-record shard metadata mismatch: {paths.records}")
        plain = [_plain_record(record) for record in records]
        validate_episode_records(plain, seeds=shard.seeds, methods=(method,))
        all_plain.extend(plain)
        all_enriched.extend(records)

    expected_seeds = partition[0].all_seeds if partition else ()
    validate_episode_records(all_plain, seeds=expected_seeds, methods=(method,))
    actual_keys = [(int(record["seed"]), str(record["method"])) for record in all_plain]
    if len(actual_keys) != len(set(actual_keys)):
        raise ShardingError(f"Duplicate records found across {method} shards")
    return all_plain, deterministic, all_enriched


def comparator_identity(
    *,
    base_identity: Mapping[str, Any],
    seed_set_id: str,
    seeds: Sequence[int],
) -> ComparatorCacheIdentity:
    """Build the evaluator cache key from every required comparator identity."""

    return ComparatorCacheIdentity.from_seeds(
        checkpoint_sha256=str(base_identity["checkpoint_sha256"]),
        resolved_config_sha256=str(base_identity["resolved_config_sha256"]),
        evaluator_sha256=str(base_identity["evaluator_sha256"]),
        seed_set_id=seed_set_id,
        seeds=seeds,
        environment_contract_sha256=str(
            base_identity["environment_contract_sha256"],
        ),
        incumbent_sha256=str(base_identity["incumbent_sha256"]),
        max_horizon=int(base_identity["max_horizon"]),
        public_history_limit=int(base_identity["public_history_limit"]),
    )


def _assert_safe_cache_file(path: Path) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise ShardingError(f"Comparator cache is missing or unsafe: {path}")
        size = path.stat().st_size
        if size <= 0 or size > 64 * 1024 * 1024:
            raise ShardingError(f"Comparator cache has invalid size: {path}")
        payload = path.read_bytes()
    except OSError as exc:
        raise ShardingError(f"Cannot read comparator cache: {path}") from exc
    assert_no_leaks(payload, source=path.name)


def aggregate_shards(
    *,
    config: Mapping[str, Any] | LoadedConfig,
    seed_set_name: str,
    mode: str,
    shard_count: int,
    output_dir: str | os.PathLike[str],
    candidate_source: str | os.PathLike[str],
    incumbent_source: str | os.PathLike[str],
    comparator_cache_path: str | os.PathLike[str],
    max_horizon: int,
    project_root: str | os.PathLike[str] = ".",
    environment_digest: str | None = None,
    artifact_bundle: Any | None = None,
) -> dict[str, Any]:
    """Strictly aggregate a comparator baseline or candidate-only experiment."""

    if mode not in AGGREGATE_MODES:
        raise ShardingError(f"Unknown aggregate mode: {mode!r}")
    root = Path(project_root).expanduser().resolve()
    loaded = config if isinstance(config, LoadedConfig) else config_from_mapping(config)
    partition = validate_registered_partition(
        loaded.payload,
        seed_set_name=seed_set_name,
        shard_count=shard_count,
    )
    shard_zero = partition[0]
    candidate = load_guarded_controller(candidate_source, project_root=root)
    incumbent = load_guarded_controller(incumbent_source, project_root=root)
    if mode == BASELINE_MODE and not hmac.compare_digest(
        candidate.sha256,
        incumbent.sha256,
    ):
        raise ShardingError(
            "Baseline candidate source must be byte-identical to its initial incumbent",
        )
    selected_environment_digest = str(
        environment_digest or environment_contract_sha256(root),
    )
    base_identity = _base_identity(
        loaded_config=loaded,
        shard=shard_zero,
        candidate=candidate,
        incumbent=incumbent,
        max_horizon=int(max_horizon),
        environment_digest=selected_environment_digest,
        artifact_bundle=artifact_bundle,
    )
    output = Path(output_dir)
    expected_methods = COMPARATOR_METHODS if mode == BASELINE_MODE else ("candidate",)
    shards_root = output / "shards"
    actual_method_dirs = {
        path.name
        for path in shards_root.iterdir()
        if path.is_dir() and any(path.glob("shard-*"))
    } if shards_root.is_dir() else set()
    if actual_method_dirs != set(expected_methods):
        missing = sorted(set(expected_methods).difference(actual_method_dirs))
        unexpected = sorted(actual_method_dirs.difference(expected_methods))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ShardingError(
            "Aggregate method coverage is not exact" + (
                ": " + " ".join(details) if details else ""
            ),
        )
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    all_deterministic = True
    shard_artifacts: list[dict[str, Any]] = []
    for method in expected_methods:
        plain, deterministic, enriched = _load_method_shards(
            output_dir=output,
            method=method,
            partition=partition,
            base_identity=base_identity,
            candidate=candidate,
            incumbent=incumbent,
        )
        records_by_method[method] = plain
        all_deterministic = all_deterministic and deterministic
        shard_artifacts.append(
            {
                "method": method,
                "records": len(enriched),
                "records_sha256": records_sha256(enriched),
            },
        )

    seeds = shard_zero.all_seeds
    cache_identity = comparator_identity(
        base_identity=base_identity,
        seed_set_id=shard_zero.seed_set_id,
        seeds=seeds,
    )
    cache_path = Path(comparator_cache_path)
    verified_comparator_records: list[dict[str, Any]]
    if mode == BASELINE_MODE:
        if not all_deterministic:
            raise ShardingError(
                "Comparator baseline requires repeat=2 deterministic shards before caching",
            )
        index = {
            (int(record["seed"]), str(record["method"])): record
            for method in COMPARATOR_METHODS
            for record in records_by_method[method]
        }
        combined = [
            index[(seed, method)]
            for seed in seeds
            for method in COMPARATOR_METHODS
        ]
        validate_episode_records(combined, seeds=seeds, methods=COMPARATOR_METHODS)
        write_comparator_cache(
            cache_path,
            identity=cache_identity,
            seeds=seeds,
            records=combined,
            determinism_verified=True,
        )
        _assert_safe_cache_file(cache_path)
        verified_comparator_records = [dict(record) for record in combined]
        checks = {
            "comparator_cache_identity": True,
            "determinism": True,
            "identity_hashes": True,
            "records_complete": True,
            "shard_coverage": True,
            "source_guard": True,
        }
        gate: dict[str, Any] = {
            "decision": "baseline_cached",
            "decision_reason": (
                "all incumbent and fixed_p60 comparator shards passed strict "
                "coverage, identity, and repeat=2 determinism checks"
            ),
            "keep": False,
        }
    else:
        _assert_safe_cache_file(cache_path)
        cached = load_comparator_cache(
            cache_path,
            identity=cache_identity,
            seeds=seeds,
        )
        if cached is None:
            raise ShardingError(
                "Comparator cache is missing, corrupt, or stale for checkpoint/config/"
                "evaluator/environment/seed/incumbent/horizon/history identity",
            )
        verified_comparator_records = [dict(record) for record in cached]
        candidate_index = {
            int(record["seed"]): record
            for record in records_by_method["candidate"]
        }
        comparator_index = {
            (int(record["seed"]), str(record["method"])): record
            for record in cached
        }
        combined = [
            (
                candidate_index[seed]
                if method == "candidate"
                else comparator_index[(seed, method)]
            )
            for seed in seeds
            for method in EVALUATION_METHODS
        ]
        validate_episode_records(combined, seeds=seeds, methods=EVALUATION_METHODS)
        checks = {
            "comparator_cache_identity": True,
            "determinism": all_deterministic,
            "identity_hashes": True,
            "records_complete": True,
            "shard_coverage": True,
            "source_guard": True,
        }
        decision = loaded.payload.get("decision")
        if not isinstance(decision, Mapping):
            raise ShardingError("Config decision mapping is missing")
        minimum = int(decision.get("minimum_paired_episode_improvement", 0))
        if minimum <= 0:
            raise ShardingError("Registered minimum improvement must be positive")
        gate = mechanical_keep_or_discard(
            combined,
            seeds=seeds,
            checks=checks,
            minimum_improvement_episodes=minimum,
        ).to_dict()

    summary = summarize_records(combined)
    aggregate_records = [
        {
            **dict(record),
            "identity": dict(base_identity),
            "seed_set_id": shard_zero.seed_set_id,
        }
        for record in combined
    ]
    aggregate_records_payload = _jsonl_bytes(aggregate_records)
    aggregate_records_path = output / "records.jsonl"
    exported_cache_path = output / "comparator_cache.json"
    summary_path = output / "summary.json"
    gate_path = output / "gate.json"
    aggregate_manifest_path = output / "aggregate.manifest.json"
    # Export a self-contained, revalidated comparator cache beside the
    # manifest.  Quest result directories can then be copied without retaining
    # any absolute cluster path.  Re-serialize through the evaluator instead
    # of copying bytes from the shared cache.
    write_comparator_cache(
        exported_cache_path,
        identity=cache_identity,
        seeds=seeds,
        records=verified_comparator_records,
        determinism_verified=True,
    )
    _assert_safe_cache_file(exported_cache_path)
    _atomic_write_bytes(aggregate_records_path, aggregate_records_payload)
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(gate_path, gate)
    aggregate_manifest = {
        "artifact_type": "autoresearch_shard_aggregate",
        "artifacts": {
            "comparator_cache": exported_cache_path.name,
            "comparator_cache_sha256": sha256_file(exported_cache_path),
            "gate": gate_path.name,
            "gate_sha256": sha256_file(gate_path),
            "records": aggregate_records_path.name,
            "records_file_sha256": sha256_file(aggregate_records_path),
            "records_sha256": records_sha256(combined),
            "summary": summary_path.name,
            "summary_sha256": sha256_file(summary_path),
        },
        "checks": checks,
        "identity": base_identity,
        "mode": mode,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "seed_set": {
            "id": shard_zero.seed_set_id,
            "name": shard_zero.seed_set_name,
            "one_time": shard_zero.one_time,
            "requires_explicit_authorization": (
                shard_zero.requires_explicit_authorization
            ),
            "spent_state_owner": "autoresearch runner",
            "spent_state_was_mutated": False,
        },
        "shards": shard_artifacts,
    }
    _atomic_write_json(aggregate_manifest_path, aggregate_manifest)
    return {
        "aggregate_manifest_path": str(aggregate_manifest_path),
        "comparator_cache_key": cache_identity.key,
        "decision": str(gate["decision"]),
        "gate_path": str(gate_path),
        "mode": mode,
        "records_sha256": records_sha256(combined),
        "run_identity_sha256": base_identity["run_identity_sha256"],
        "summary_path": str(summary_path),
    }


def _source_path_from_config(
    config: Mapping[str, Any],
    name: str,
    *,
    project_root: Path,
) -> Path:
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ShardingError("Config source mapping is missing")
    raw = str(source.get(name, "")).strip()
    if not raw:
        raise ShardingError(f"Config source.{name} is required")
    return _resolve_path(raw, project_root=project_root)


def _verified_aggregate_artifacts_and_horizon(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Any, int]:
    """Verify real bytes for aggregate without deserializing the SAC model."""

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ShardingError("Config source mapping is missing")
    checkpoint = _source_path_from_config(
        config,
        "checkpoint_path",
        project_root=project_root,
    )
    resolved_config = _source_path_from_config(
        config,
        "resolved_config_path",
        project_root=project_root,
    )
    bundle = verify_exp05_artifacts(
        checkpoint,
        resolved_config,
        expected_checkpoint_sha256=str(source.get("checkpoint_sha256", "")),
        expected_resolved_config_sha256=str(
            source.get("resolved_config_sha256", ""),
        ),
    )
    try:
        resolved = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        environment = resolved["environment"]
        horizon = int(environment["max_step"])
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise ShardingError("Verified resolved config has no valid environment.max_step") from exc
    if not isinstance(environment, Mapping) or horizon <= 0:
        raise ShardingError("Verified resolved config has no positive environment.max_step")
    if (
        environment.get("observation_mode") != "mouse"
        or environment.get("action_mode") != "egocentric_velocity_head"
    ):
        raise ShardingError("Verified resolved config violates the EXP-05 public task contract")
    return bundle, horizon


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-set", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--incumbent-source", required=True)
    parser.add_argument("--project-root", default=".")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -B -m autoresearch.sharding",
        description="Strict registered-seed Quest sharding for autoresearch gaze evaluation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    rollout = subparsers.add_parser("rollout", help="run one method/shard")
    _common_arguments(rollout)
    rollout.add_argument("--method", choices=EVALUATION_METHODS, required=True)
    rollout.add_argument("--shard-index", required=True, type=int)
    rollout.add_argument("--repeat", choices=ALLOWED_REPEATS, default=2, type=int)

    aggregate = subparsers.add_parser("aggregate", help="strictly aggregate shards")
    _common_arguments(aggregate)
    aggregate.add_argument("--mode", choices=AGGREGATE_MODES, required=True)
    aggregate.add_argument("--comparator-cache", required=True)
    return parser


def _bounded_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 800:
        message = message[:797] + "..."
    return f"{type(exc).__name__}: {message}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).expanduser().resolve()
        loaded = load_registered_config(args.config, project_root=root)
        common = {
            "config": loaded,
            "seed_set_name": args.seed_set,
            "shard_count": args.shard_count,
            "output_dir": _resolve_path(args.output_dir, project_root=root),
            "candidate_source": args.candidate_source,
            "incumbent_source": args.incumbent_source,
            "project_root": root,
        }
        if args.command == "rollout":
            result = rollout_shard(
                **common,
                method=args.method,
                shard_index=args.shard_index,
                repeat=args.repeat,
            )
        else:
            artifact_bundle, horizon = _verified_aggregate_artifacts_and_horizon(
                loaded.payload,
                project_root=root,
            )
            result = aggregate_shards(
                **common,
                mode=args.mode,
                comparator_cache_path=_resolve_path(
                    args.comparator_cache,
                    project_root=root,
                ),
                max_horizon=horizon,
                artifact_bundle=artifact_bundle,
            )
        print(_canonical_json(result))
        return 0
    except KeyboardInterrupt:
        print("autoresearch sharding interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # bounded CLI: full diagnostics belong in Slurm logs
        print(f"autoresearch sharding failed: {_bounded_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_MODES",
    "BASELINE_MODE",
    "EXPERIMENT_MODE",
    "LoadedConfig",
    "RegisteredShard",
    "SHARD_SCHEMA_VERSION",
    "ShardPaths",
    "ShardingError",
    "aggregate_shards",
    "build_parser",
    "comparator_identity",
    "config_from_mapping",
    "environment_contract_manifest",
    "environment_contract_sha256",
    "load_guarded_controller",
    "load_registered_config",
    "main",
    "registered_shard",
    "rollout_shard",
    "shard_paths",
    "validate_registered_partition",
]
