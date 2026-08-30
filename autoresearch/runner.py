"""Safe orchestration for the bounded phase-1 autoresearch loop.

The runner owns durable run setup and lifecycle decisions.  Candidate code is
loaded through :mod:`autoresearch.guard`; rollout mechanics and the mechanical
gate remain in :mod:`autoresearch.evaluator`; append-only state remains in
:mod:`autoresearch.ledger`.

Normal experiments have no seed-set argument.  They always run the registered
``smoke`` contract set followed by the registered ``development`` set.  The
one-time ``confirmation`` set is reachable only through :meth:`confirm` after
an explicit authorization flag has been recorded.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import yaml

from autoresearch.contract import CandidateContractError
from autoresearch.evaluator import (
    ArtifactVerificationError,
    COMPARATOR_METHODS,
    EVALUATION_METHODS,
    EPISODE_RESULT_FIELDS,
    EpisodeContractError,
    EvaluationError,
    RealExp05EpisodeFactory,
    ComparatorCacheIdentity,
    canonical_sha256,
    confirmation_statistics,
    evaluator_sha256,
    evaluate_paired_from_config,
    mechanical_keep_or_discard,
    ordered_seed_sha256,
    records_sha256,
    seed_set_from_config,
    summarize_records,
    validate_episode_records,
)
from autoresearch.guard import (
    CandidateSourceError,
    ChangedPathError,
    GuardError,
    HashManifestError,
    LeakError,
    assert_hash_manifest,
    assert_no_leaks,
    build_hash_manifest,
    manifest_sha256,
    sha256_bytes,
    sha256_file,
    validate_changed_paths,
)
from autoresearch.ledger import ExperimentLedger, LedgerError, TERMINAL_STATUSES
from autoresearch.worker import IsolatedCandidateController


SCHEMA_VERSION = 1
RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
EXPERIMENT_PATTERN = re.compile(r"E([0-9]{4,})\Z")
MAX_HYPOTHESIS_BYTES = 32 * 1024
MAX_ABORT_REASON_BYTES = 4 * 1024

# These files define orchestration, the public boundary, evaluator semantics,
# and the real EXP-05 environment adapter.  The mutable candidate is
# deliberately absent.  The selected config is added separately at setup.
DEFAULT_IMMUTABLE_PATHS = (
    "autoresearch/__init__.py",
    "autoresearch/__main__.py",
    "autoresearch/contract.py",
    "autoresearch/evaluator.py",
    "autoresearch/guard.py",
    "autoresearch/ledger.py",
    "autoresearch/program.md",
    "autoresearch/runner.py",
    "autoresearch/sharding.py",
    "autoresearch/worker.py",
    "analysis/sac_gaze_ablation.py",
    "benchmarks/peekbench/environment.py",
    "botevade_gym.py",
    "first_person.py",
    "training/first_person_sac.py",
)


class RunnerError(RuntimeError):
    """Base class for a refused or failed runner operation."""


class SetupError(RunnerError):
    """The frozen run manifest could not be created or validated."""


class RunContractError(RunnerError):
    """The run or candidate violated a frozen phase-1 contract."""


class BudgetExhausted(RunnerError):
    """Raised only by callers that request exception-style budget handling."""


class ConfirmationError(RunnerError):
    """The one-time confirmation gate was unavailable or unauthorized."""


class SourceArtifactError(RunContractError):
    """A registered checkpoint or resolved config is absent or has wrong bytes."""


_CONTRACT_EXCEPTIONS = (
    ArtifactVerificationError,
    CandidateContractError,
    CandidateSourceError,
    ChangedPathError,
    EpisodeContractError,
    EvaluationError,
    GuardError,
    HashManifestError,
    LeakError,
    SourceArtifactError,
    RunContractError,
    ValueError,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("runner timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RunContractError("run manifest timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunContractError("run manifest timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RunContractError("run manifest timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"runner evidence is not canonical JSON: {exc}") from exc


def _write_private_file(path: Path, payload: bytes) -> None:
    """Write a new file completely; callers only use private staging dirs."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while creating runner evidence")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_relative_path(raw: Any, *, label: str) -> str:
    text = str(raw).replace("\\", "/")
    if not text or text.startswith("/") or "\x00" in text:
        raise SetupError(f"{label} must be a non-empty repository-relative path")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise SetupError(f"{label} contains an unsafe path segment")
    return PurePosixPath(*parts).as_posix()


def _sha256_shape(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exception_summary(exc: BaseException) -> Mapping[str, str]:
    # Exception text may contain arbitrary provider output in future adapters.
    # The phase-1 real adapter is local, but keep failure records bounded now.
    message = " ".join(str(exc).split())[:1000]
    return {"type": type(exc).__name__, "message": message or type(exc).__name__}


class AutoresearchRunner:
    """One-process coordinator for a single frozen phase-1 run.

    ``episode_factory_builder`` and ``paired_evaluator`` are dependency
    injection seams for tiny deterministic tests.  Production defaults are
    the verified EXP-05 adapter and frozen evaluator.  ``immutable_paths`` may
    be narrowed only by tests constructing an isolated miniature repository;
    the command-line interface never overrides it.
    """

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike[str] = ".",
        results_root: str | os.PathLike[str] | None = None,
        immutable_paths: Sequence[str] = DEFAULT_IMMUTABLE_PATHS,
        episode_factory_builder: Callable[..., Any] | None = None,
        paired_evaluator: Callable[..., Mapping[str, Any]] | None = None,
        confirmation_statistics_fn: Callable[..., Mapping[str, Any]] | None = None,
        environment_contract_provider: Callable[[Path], str] | None = None,
        now: Callable[[], datetime] = _utc_now,
        commit_provider: Callable[[], str] | None = None,
        changed_paths_provider: Callable[[str, str], Sequence[str]] | None = None,
        committed_file_provider: Callable[[str, str], bytes] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.results_root = (
            Path(results_root).expanduser().resolve()
            if results_root is not None
            else self.repo_root / "results" / "autoresearch"
        )
        self.immutable_paths = tuple(str(path) for path in immutable_paths)
        self.episode_factory_builder = (
            episode_factory_builder or RealExp05EpisodeFactory.from_config
        )
        self.paired_evaluator = paired_evaluator or evaluate_paired_from_config
        self.confirmation_statistics_fn = (
            confirmation_statistics_fn or confirmation_statistics
        )
        self.environment_contract_provider = environment_contract_provider
        self.now = now
        self.commit_provider = commit_provider or self._current_commit
        self.changed_paths_provider = changed_paths_provider or self._changed_paths
        self.committed_file_provider = (
            committed_file_provider or self._committed_file
        )

    def _run_dir(self, run_tag: str) -> Path:
        if not isinstance(run_tag, str) or not RUN_TAG_PATTERN.fullmatch(run_tag):
            raise RunnerError(
                "run tag must contain only letters, digits, '.', '_' or '-'"
            )
        return self.results_root / run_tag

    def _git(self, arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=self.repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=not binary,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            summary = " ".join(str(detail or exc).split())[:400]
            raise RunContractError(f"git identity check failed: {summary}") from exc
        return completed.stdout

    def _current_commit(self) -> str:
        commit = str(self._git(("rev-parse", "HEAD"))).strip()
        if not COMMIT_PATTERN.fullmatch(commit):
            raise RunContractError("git HEAD is not a full hexadecimal commit")
        return commit

    def _changed_paths(self, parent_commit: str, candidate_commit: str) -> Sequence[str]:
        for label, value in (
            ("parent", parent_commit),
            ("candidate", candidate_commit),
        ):
            if not COMMIT_PATTERN.fullmatch(str(value)):
                raise RunContractError(f"{label} commit is not a full hexadecimal commit")
        output = str(
            self._git(
                (
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "--diff-filter=ACDMRTUXB",
                    f"{parent_commit}..{candidate_commit}",
                    "--",
                )
            )
        )
        return tuple(line for line in output.splitlines() if line)

    def _committed_file(self, commit: str, relative_path: str) -> bytes:
        if not COMMIT_PATTERN.fullmatch(str(commit)):
            raise RunContractError("candidate commit is not a full hexadecimal commit")
        return bytes(self._git(("show", f"{commit}:{relative_path}"), binary=True))

    @contextmanager
    def _run_lock(self, run_dir: Path) -> Iterator[None]:
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = run_dir / "run.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunnerError(
                    f"another autoresearch command is active for {run_dir.name}"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _resolve_repo_path(self, raw: Any, *, label: str) -> tuple[str, Path]:
        relative = _safe_relative_path(raw, label=label)
        path = self.repo_root.joinpath(*PurePosixPath(relative).parts)
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise SetupError(f"{label} escapes the repository root") from exc
        if path.is_symlink():
            raise SetupError(f"{label} may not be a symlink")
        return relative, path

    def _source_state(
        self, config: Mapping[str, Any], *, require: bool
    ) -> Mapping[str, Mapping[str, Any]]:
        source = config.get("source")
        if not isinstance(source, Mapping):
            raise SetupError("config.source must be a mapping")
        result: dict[str, Mapping[str, Any]] = {}
        for kind, path_key, hash_key in (
            ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
            ("resolved_config", "resolved_config_path", "resolved_config_sha256"),
        ):
            relative, path = self._resolve_repo_path(
                source.get(path_key, ""), label=f"source.{path_key}"
            )
            expected = source.get(hash_key)
            if not _sha256_shape(expected):
                raise SetupError(f"source.{hash_key} must be a lowercase SHA-256")
            entry: dict[str, Any] = {
                "path": relative,
                "expected_sha256": expected,
                "present": path.is_file(),
                "verified": False,
            }
            if path.is_file():
                actual = sha256_file(path)
                entry["actual_sha256"] = actual
                entry["verified"] = hmac.compare_digest(actual, str(expected))
                if not entry["verified"] and require:
                    raise SourceArtifactError(
                        f"registered {kind} SHA-256 mismatch at {relative}"
                    )
            elif require:
                raise SourceArtifactError(
                    f"registered {kind} is missing at {relative}"
                )
            result[kind] = entry
        return result

    def _validate_config(
        self, config: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Any]]:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise SetupError("unsupported autoresearch config schema_version")
        for key in (
            "source",
            "candidate",
            "seed_sets",
            "evaluation",
            "decision",
            "budget",
        ):
            if not isinstance(config.get(key), Mapping):
                raise SetupError(f"config.{key} must be a mapping")

        candidate = config["candidate"]
        candidate_path, _ = self._resolve_repo_path(
            candidate.get("path", ""), label="candidate.path"
        )
        mutable = tuple(
            _safe_relative_path(path, label="candidate.mutable_paths entry")
            for path in candidate.get("mutable_paths", ())
        )
        if candidate_path != "autoresearch/candidate.py" or mutable != (
            candidate_path,
        ):
            raise SetupError(
                "phase-1 permits exactly autoresearch/candidate.py as mutable source"
            )
        if str(candidate.get("class_name", "")) != "CandidateGazeController":
            raise SetupError("candidate.class_name violates the phase-1 interface")
        if not _sha256_shape(candidate.get("initial_incumbent_sha256")):
            raise SetupError(
                "candidate.initial_incumbent_sha256 must freeze the legal scan source"
            )
        initial_incumbent_id = candidate.get("initial_incumbent_id")
        if not isinstance(initial_incumbent_id, str) or not RUN_TAG_PATTERN.fullmatch(
            initial_incumbent_id
        ):
            raise SetupError("candidate.initial_incumbent_id is not filesystem-safe")
        public_fields = tuple(
            str(value) for value in candidate.get("public_observation_fields", ())
        )
        if public_fields != (
            "image_left",
            "image_right",
            "proprio",
            "previous_action",
        ):
            raise SetupError("candidate public observation fields are not frozen")

        evaluation = config["evaluation"]
        if evaluation.get("deterministic_policy") is not True:
            raise SetupError("evaluation must use deterministic policy inference")
        if int(evaluation.get("maximum_history_length", -1)) != int(
            candidate.get("public_history_length", -2)
        ):
            raise SetupError("candidate and evaluator history lengths differ")
        if tuple(evaluation.get("methods", ())) != (
            "candidate",
            "search_incumbent",
            "fixed_p60_research_reference",
        ):
            raise SetupError("evaluation methods violate the frozen paired design")

        decision = config["decision"]
        if str(decision.get("primary_metric", "")) != "paired_clean_success_delta":
            raise SetupError("phase-1 primary metric is not frozen")
        if int(decision.get("minimum_paired_episode_improvement", 0)) < 2:
            raise SetupError("development improvement gate must be at least two episodes")
        if decision.get("capture_rate_must_not_exceed_incumbent") is not True:
            raise SetupError("capture non-worsening gate must remain enabled")
        if decision.get("ties_keep_incumbent") is not True:
            raise SetupError("ties must keep the incumbent")

        budget = config["budget"]
        for name in (
            "max_experiments",
            "max_wall_seconds",
            "max_consecutive_crashes",
        ):
            value = budget.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SetupError(f"budget.{name} must be a finite positive integer")
        if budget.get("on_exhausted") != "stop_successfully_and_report":
            raise SetupError("budget exhaustion policy must stop successfully")

        raw_seed_sets = config["seed_sets"]
        required = {"smoke", "development", "confirmation"}
        if not required.issubset(raw_seed_sets):
            raise SetupError("smoke, development, and confirmation seed sets are required")
        summaries: dict[str, Mapping[str, Any]] = {}
        occupied: dict[int, str] = {}
        ids: set[str] = set()
        for name in sorted(raw_seed_sets):
            # ``rationale`` is registered prose alongside the three named
            # mappings in gaze_dev.yaml, not an executable seed set.
            if not isinstance(raw_seed_sets[name], Mapping):
                if name != "rationale" or not isinstance(
                    raw_seed_sets[name], str
                ):
                    raise SetupError(
                        f"seed_sets.{name} must be a registered seed mapping"
                    )
                continue
            try:
                frozen = seed_set_from_config(config, str(name))
            except (EvaluationError, TypeError, ValueError) as exc:
                raise SetupError(f"invalid seed set {name!r}: {exc}") from exc
            if frozen.seed_set_id in ids:
                raise SetupError(f"duplicate seed-set id: {frozen.seed_set_id}")
            ids.add(frozen.seed_set_id)
            for seed in frozen.seeds:
                previous = occupied.get(seed)
                if previous is not None:
                    raise SetupError(
                        f"seed sets {previous!r} and {name!r} overlap at seed {seed}"
                    )
                occupied[seed] = str(name)
            summaries[str(name)] = {
                "id": frozen.seed_set_id,
                "episodes": len(frozen.seeds),
                "seed_start": frozen.seeds[0],
                "seed_end": frozen.seeds[-1],
                "ordered_seed_sha256": ordered_seed_sha256(frozen.seeds),
                "purpose": frozen.purpose,
                "one_time": frozen.one_time,
                "requires_explicit_authorization": (
                    frozen.requires_explicit_authorization
                ),
            }

        try:
            confirmation = seed_set_from_config(config, "confirmation")
        except (EvaluationError, TypeError, ValueError) as exc:
            raise SetupError(f"invalid confirmation seed set: {exc}") from exc
        if not confirmation.one_time or not confirmation.requires_explicit_authorization:
            raise SetupError(
                "confirmation must be one-time and require explicit authorization"
            )
        source = config["source"]
        historical_start = int(source.get("historical_exp05_seed_start", -1))
        historical_episodes = int(source.get("historical_exp05_episodes", 0))
        if historical_start < 0 or historical_episodes <= 0:
            raise SetupError("historical EXP-05 seed range is incomplete")
        historical = range(historical_start, historical_start + historical_episodes)
        collision = next((seed for seed in historical if seed in occupied), None)
        if collision is not None:
            raise SetupError(
                f"registered seed sets reuse historical EXP-05 seed {collision}"
            )
        return summaries

    def _configured_results_root(self, config: Mapping[str, Any]) -> Path:
        raw = config.get("results_root", "results/autoresearch")
        relative = _safe_relative_path(raw, label="results_root")
        configured = self.repo_root.joinpath(*PurePosixPath(relative).parts).resolve(
            strict=False
        )
        try:
            configured.relative_to(self.repo_root)
        except ValueError as exc:
            raise SetupError("results_root escapes the repository") from exc
        return configured

    def _config_path_identity(self, config_path: Path) -> tuple[str, str]:
        resolved = config_path.expanduser().resolve()
        try:
            relative = resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            relative = str(resolved)
        return relative, sha256_file(resolved)

    def _environment_contract_digest(
        self, content_manifest: Mapping[str, str]
    ) -> str:
        if self.environment_contract_provider is not None:
            digest = str(self.environment_contract_provider(self.repo_root))
        elif "autoresearch/sharding.py" in content_manifest:
            # Import lazily to avoid a package cycle when the sharding module
            # itself starts through ``python -m``.
            from autoresearch.sharding import environment_contract_sha256

            try:
                digest = environment_contract_sha256(self.repo_root)
            except (OSError, GuardError, EvaluationError) as exc:
                raise RunContractError(
                    f"cannot hash the frozen environment contract: {exc}"
                ) from exc
        else:
            digest = manifest_sha256(content_manifest)
        if not _sha256_shape(digest):
            raise RunContractError(
                "environment contract provider did not return a SHA-256"
            )
        return digest

    def setup(
        self,
        *,
        config_path: str | os.PathLike[str],
        run_tag: str,
    ) -> Mapping[str, Any]:
        """Create ``run.json`` once with config, provenance, and source hashes."""

        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.repo_root / config_file
        try:
            raw_config = config_file.read_bytes()
        except OSError as exc:
            raise SetupError(f"cannot read autoresearch config: {config_file}") from exc
        try:
            loaded = yaml.safe_load(raw_config.decode("utf-8"))
        except (UnicodeError, yaml.YAMLError) as exc:
            raise SetupError(f"cannot parse autoresearch config: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise SetupError("autoresearch config must contain a mapping")
        config = json.loads(json.dumps(dict(loaded), allow_nan=False))
        seed_summary = self._validate_config(config)
        if self._configured_results_root(config) != self.results_root:
            raise SetupError(
                "configured results_root differs from the runner results root"
            )
        source_state = self._source_state(config, require=False)
        for kind, state in source_state.items():
            if state["present"] and not state["verified"]:
                raise SetupError(
                    f"registered {kind} exists but its SHA-256 does not match"
                )

        run_dir = self._run_dir(run_tag)
        config_identity, config_digest = self._config_path_identity(config_file)
        if run_dir.exists():
            manifest = self._load_run(run_tag, verify_sources=False)
            if manifest.get("config_sha256") == config_digest:
                return manifest
            raise SetupError(f"run tag {run_tag!r} already exists with another config")

        immutable = list(self.immutable_paths)
        try:
            config_relative = config_file.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            raise SetupError(
                "autoresearch config must be a versioned file inside the repository"
            )
        immutable.append(config_relative)
        candidate_path = str(config["candidate"]["path"])
        if candidate_path in immutable:
            raise SetupError("mutable candidate may not appear in immutable manifest")
        try:
            content_manifest = build_hash_manifest(self.repo_root, immutable)
        except (OSError, GuardError) as exc:
            raise SetupError(f"cannot freeze immutable source manifest: {exc}") from exc
        if not hmac.compare_digest(
            str(content_manifest.get(config_relative, "")), config_digest
        ):
            raise SetupError("autoresearch config changed while setup was freezing it")

        commit = str(self.commit_provider()).strip()
        if not COMMIT_PATTERN.fullmatch(commit):
            raise SetupError("setup commit is not a full hexadecimal git identity")
        for relative_path, expected_digest in content_manifest.items():
            try:
                committed_source = self.committed_file_provider(
                    commit, relative_path
                )
            except (OSError, RunContractError) as exc:
                raise SetupError(
                    f"setup commit does not contain immutable source: {relative_path}"
                ) from exc
            if not hmac.compare_digest(
                sha256_bytes(committed_source), str(expected_digest)
            ):
                raise SetupError(
                    f"working immutable source differs from setup commit: {relative_path}"
                )
        candidate_relative = str(config["candidate"]["path"])
        candidate_file = self.repo_root.joinpath(
            *PurePosixPath(candidate_relative).parts
        )
        expected_incumbent = str(
            config["candidate"]["initial_incumbent_sha256"]
        )
        try:
            working_candidate = candidate_file.read_bytes()
        except OSError as exc:
            raise SetupError(
                f"initial incumbent source is missing: {candidate_relative}"
            ) from exc
        if not hmac.compare_digest(
            sha256_bytes(working_candidate), expected_incumbent
        ):
            raise SetupError(
                "working candidate bytes do not match initial_incumbent_sha256"
            )
        try:
            committed_candidate = self.committed_file_provider(
                commit, candidate_relative
            )
        except (OSError, RunContractError) as exc:
            raise SetupError(
                "setup commit does not contain the registered initial incumbent"
            ) from exc
        if not hmac.compare_digest(
            sha256_bytes(committed_candidate), expected_incumbent
        ):
            raise SetupError(
                "setup commit candidate bytes do not match initial_incumbent_sha256"
            )
        evaluator_digest = content_manifest.get(
            "autoresearch/evaluator.py", evaluator_sha256()
        )
        try:
            environment_digest = self._environment_contract_digest(content_manifest)
        except RunContractError as exc:
            raise SetupError(str(exc)) from exc
        now = self.now()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_tag": run_tag,
            "created_at": _timestamp(now),
            "setup_commit": commit,
            "config_path": config_identity,
            "config_sha256": config_digest,
            "config": config,
            "immutable_manifest": content_manifest,
            "immutable_manifest_sha256": manifest_sha256(content_manifest),
            "environment_contract_sha256": environment_digest,
            "evaluator_sha256": evaluator_digest,
            "mutable_paths": [candidate_path],
            "seed_sets": seed_summary,
            "source_artifacts_at_setup": source_state,
            "claims": {
                "scope": "engineering_selection",
                "scientific_active_gaze_verification": False,
            },
        }
        payload = _json_bytes(manifest)
        digest = hashlib.sha256(payload).hexdigest()

        self.results_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{run_tag}.setup-", dir=self.results_root)
        )
        try:
            _write_private_file(staging / "run.json", payload)
            _write_private_file(staging / "run.sha256", (digest + "\n").encode("ascii"))
            os.chmod(staging / "run.json", 0o444)
            os.chmod(staging / "run.sha256", 0o444)
            _fsync_directory(staging)
            os.replace(staging, run_dir)
            _fsync_directory(self.results_root)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        ExperimentLedger(run_dir).regenerate_results_tsv()
        return manifest

    def _load_run(
        self, run_tag: str, *, verify_sources: bool = False
    ) -> Mapping[str, Any]:
        run_dir = self._run_dir(run_tag)
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise RunnerError(f"autoresearch run does not exist: {run_tag}")
        run_path = run_dir / "run.json"
        digest_path = run_dir / "run.sha256"
        try:
            payload = run_path.read_bytes()
            expected_digest = digest_path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise RunContractError("run manifest or its digest is missing") from exc
        actual_digest = hashlib.sha256(payload).hexdigest()
        if not _sha256_shape(expected_digest) or not hmac.compare_digest(
            actual_digest, expected_digest
        ):
            raise RunContractError("immutable run.json digest mismatch")
        try:
            manifest = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RunContractError("run.json is not valid JSON") from exc
        if not isinstance(manifest, Mapping):
            raise RunContractError("run.json must contain an object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RunContractError("unsupported run manifest schema")
        if manifest.get("run_tag") != run_tag:
            raise RunContractError("run.json tag does not match its directory")
        immutable = manifest.get("immutable_manifest")
        if not isinstance(immutable, Mapping):
            raise RunContractError("run.json immutable manifest is missing")
        if not hmac.compare_digest(
            manifest_sha256(immutable),
            str(manifest.get("immutable_manifest_sha256", "")),
        ):
            raise RunContractError("run.json immutable manifest identity mismatch")
        try:
            assert_hash_manifest(self.repo_root, immutable)
        except HashManifestError as exc:
            raise RunContractError(str(exc)) from exc
        current_environment = self._environment_contract_digest(immutable)
        if not hmac.compare_digest(
            current_environment,
            str(manifest.get("environment_contract_sha256", "")),
        ):
            raise RunContractError("frozen environment contract SHA-256 mismatch")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise RunContractError("run.json frozen config is missing")
        try:
            self._validate_config(config)
        except (SetupError, EvaluationError, TypeError, ValueError) as exc:
            raise RunContractError(f"frozen config contract failed: {exc}") from exc
        if verify_sources:
            self._source_state(config, require=True)
        return manifest

    def _recover_startup(self, ledger: ExperimentLedger) -> None:
        # The per-run process lock proves no live command owns these records.
        # A prepared external lifecycle is intentionally waiting for Quest and
        # must be finalized explicitly, never auto-crashed by another command.
        prepared = [
            record["experiment_id"]
            for record in ledger.latest_records().values()
            if record["status"] == "running"
            and record.get("external_prepared") is True
        ]
        if prepared:
            raise RunContractError(
                "external evaluation is awaiting explicit finalize: "
                + ", ".join(sorted(str(item) for item in prepared))
            )
        planned = [
            str(record["experiment_id"])
            for record in ledger.latest_records().values()
            if record["status"] == "planned"
        ]
        for experiment_id in sorted(planned):
            ledger.finalize_experiment(
                experiment_id,
                status="crash",
                fields={
                    "decision_reason": (
                        "startup recovered an interrupted planned lifecycle"
                    ),
                    "recovery_action": "mark_planned_crash",
                },
            )
        # Every other remaining running lifecycle is stale immediately.
        ledger.recover_stale_running(0.0)

    def _candidate_path(self, manifest: Mapping[str, Any]) -> tuple[str, Path]:
        relative = str(manifest["config"]["candidate"]["path"])
        return relative, self.repo_root.joinpath(*PurePosixPath(relative).parts)

    def _verify_initial_incumbent_binding(
        self, manifest: Mapping[str, Any], candidate_path: Path
    ) -> str:
        relative = str(manifest["config"]["candidate"]["path"])
        expected = str(
            manifest["config"]["candidate"]["initial_incumbent_sha256"]
        )
        try:
            working = candidate_path.read_bytes()
        except OSError as exc:
            raise RunContractError("registered initial incumbent source is missing") from exc
        if not hmac.compare_digest(sha256_bytes(working), expected):
            raise RunContractError(
                "working candidate is not the registered legal-scan incumbent"
            )
        try:
            committed = self.committed_file_provider(
                str(manifest["setup_commit"]), relative
            )
        except (OSError, RunContractError) as exc:
            raise RunContractError(
                "cannot recover initial incumbent bytes from setup commit"
            ) from exc
        if not hmac.compare_digest(sha256_bytes(committed), expected):
            raise RunContractError(
                "setup commit no longer identifies the registered legal-scan incumbent"
            )
        return expected

    def _validate_and_load_candidate(self, path: Path) -> tuple[Any, str, bytes]:
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise RunContractError(f"candidate source is missing at {path}") from exc
        try:
            decoded = source.decode("utf-8")
        except UnicodeError as exc:
            raise RunContractError("candidate source is not UTF-8") from exc
        assert_no_leaks(decoded, source="candidate.py")
        controller = IsolatedCandidateController.from_source(
            source,
            filename=str(path),
        )
        return controller, sha256_bytes(source), source

    @staticmethod
    def _close_controllers(*controllers: Any) -> None:
        for controller in controllers:
            close = getattr(controller, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Evidence/failure from the evaluator remains primary;
                    # worker.close already performs bounded forceful cleanup.
                    pass

    @staticmethod
    def _latest_by_id(ledger: ExperimentLedger, experiment_id: str) -> Mapping[str, Any]:
        latest = ledger.latest_records().get(experiment_id)
        if latest is None:
            raise RunContractError(f"ledger has no experiment {experiment_id}")
        return latest

    def _incumbent_record(
        self, ledger: ExperimentLedger
    ) -> tuple[Mapping[str, Any], Path] | None:
        incumbent = ledger.recover_incumbent()
        if incumbent is None:
            return None
        experiment_id = str(incumbent["experiment_id"])
        record = self._latest_by_id(ledger, experiment_id)
        source_path = ledger.artifact_path(experiment_id) / "candidate.py"
        if not source_path.is_file():
            raise RunContractError("incumbent candidate source artifact is missing")
        actual = sha256_file(source_path)
        expected = str(record.get("candidate_sha256", ""))
        if not _sha256_shape(expected) or not hmac.compare_digest(actual, expected):
            raise RunContractError("incumbent candidate source hash mismatch")
        return record, source_path

    def _budget_status(
        self,
        manifest: Mapping[str, Any],
        ledger: ExperimentLedger,
    ) -> Mapping[str, Any]:
        budget = manifest["config"]["budget"]
        latest = sorted(
            ledger.latest_records().values(),
            key=lambda record: int(record["ledger_sequence"]),
        )
        normal = [
            record
            for record in latest
            if EXPERIMENT_PATTERN.fullmatch(str(record["experiment_id"]))
        ]
        consecutive_crashes = 0
        for record in reversed(normal):
            if record["status"] == "crash":
                consecutive_crashes += 1
            else:
                break
        elapsed = max(
            0.0,
            (self.now() - _parse_timestamp(manifest["created_at"])).total_seconds(),
        )
        reasons: list[str] = []
        if len(normal) >= int(budget["max_experiments"]):
            reasons.append("max_experiments")
        if elapsed >= float(budget["max_wall_seconds"]):
            reasons.append("max_wall_seconds")
        if consecutive_crashes >= int(budget["max_consecutive_crashes"]):
            reasons.append("max_consecutive_crashes")
        return {
            "exhausted": bool(reasons),
            "reasons": reasons,
            "experiments_used": len(normal),
            "experiments_remaining": max(
                0, int(budget["max_experiments"]) - len(normal)
            ),
            "wall_seconds_used": round(elapsed, 6),
            "wall_seconds_remaining": max(
                0.0, round(float(budget["max_wall_seconds"]) - elapsed, 6)
            ),
            "consecutive_crashes": consecutive_crashes,
            "limits": {
                "max_experiments": int(budget["max_experiments"]),
                "max_wall_seconds": int(budget["max_wall_seconds"]),
                "max_consecutive_crashes": int(
                    budget["max_consecutive_crashes"]
                ),
            },
        }

    @staticmethod
    def _next_experiment_id(ledger: ExperimentLedger) -> str:
        numbers = [
            int(match.group(1))
            for experiment_id in ledger.latest_records()
            if (match := EXPERIMENT_PATTERN.fullmatch(str(experiment_id)))
        ]
        return f"E{max(numbers, default=0) + 1:04d}"

    @staticmethod
    def _write_artifact(staging: Path, name: str, payload: bytes) -> Path:
        if "/" in name or name in {"", ".", ".."}:
            raise RunnerError("artifact name is unsafe")
        path = staging / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RunContractError(
                    f"existing artifact is not a regular file: {name}"
                )
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise RunContractError(f"cannot verify existing artifact: {name}") from exc
            if existing != payload:
                raise RunContractError(
                    f"existing artifact conflicts with retry evidence: {name}"
                )
            return path
        _write_private_file(path, payload)
        return path

    @staticmethod
    def _result_checks(result: Mapping[str, Any], *, phase: str) -> Mapping[str, bool]:
        raw = result.get("checks")
        if not isinstance(raw, Mapping) or not raw:
            raise RunContractError(f"{phase} evaluator returned no hard checks")
        checks = {str(name): value is True for name, value in raw.items()}
        failures = sorted(name for name, passed in checks.items() if not passed)
        if failures:
            raise RunContractError(
                f"{phase} evaluator hard checks failed: {', '.join(failures)}"
            )
        return checks

    def _evaluate_phase(
        self,
        *,
        manifest: Mapping[str, Any],
        factory: Any,
        seed_set_name: str,
        candidate: Any,
        incumbent: Any,
        incumbent_sha256: str,
        checks: Mapping[str, bool],
        cache_path: Path,
        allow_confirmation: bool = False,
    ) -> Mapping[str, Any]:
        if seed_set_name == "confirmation" and not allow_confirmation:
            raise RunContractError(
                "normal evaluation cannot access confirmation seeds"
            )
        if allow_confirmation and seed_set_name != "confirmation":
            raise RunContractError(
                "confirmation authorization cannot be applied to another seed set"
            )
        result = self.paired_evaluator(
            manifest["config"],
            seed_set_name=seed_set_name,
            episode_factory=factory,
            candidate=candidate,
            incumbent=incumbent,
            checks=checks,
            environment_contract_sha256=str(
                manifest["environment_contract_sha256"]
            ),
            incumbent_sha256=incumbent_sha256,
            cache_path=cache_path,
            evaluator_digest=str(manifest["evaluator_sha256"]),
            allow_confirmation=allow_confirmation,
        )
        if not isinstance(result, Mapping):
            raise RunContractError(f"{seed_set_name} evaluator result is not a mapping")
        self._result_checks(result, phase=seed_set_name)
        return result

    def _base_plan_fields(
        self,
        *,
        manifest: Mapping[str, Any],
        parent_incumbent_id: str | None,
        candidate_commit: str,
        candidate_sha256: str,
        changed_paths: Sequence[str],
        hypothesis: str,
        predicted_effect: str,
        seed_set_name: str,
    ) -> Mapping[str, Any]:
        source = manifest["config"]["source"]
        seed_set = seed_set_from_config(manifest["config"], seed_set_name)
        return {
            "parent_incumbent_id": parent_incumbent_id,
            "candidate_commit": candidate_commit,
            "candidate_sha256": candidate_sha256,
            "hypothesis": hypothesis,
            "predicted_effect": predicted_effect,
            "changed_paths": list(changed_paths),
            "source_model_sha256": str(source["checkpoint_sha256"]),
            "resolved_config_sha256": str(source["resolved_config_sha256"]),
            "evaluator_sha256": str(manifest["evaluator_sha256"]),
            "contract_sha256": str(
                manifest["immutable_manifest"].get(
                    "autoresearch/contract.py", ""
                )
            ),
            "guard_sha256": str(
                manifest["immutable_manifest"].get("autoresearch/guard.py", "")
            ),
            "sharding_sha256": str(
                manifest["immutable_manifest"].get(
                    "autoresearch/sharding.py", ""
                )
            ),
            "worker_sha256": str(
                manifest["immutable_manifest"].get("autoresearch/worker.py", "")
            ),
            "seed_set_id": seed_set.seed_set_id,
            "run_manifest_sha256": hashlib.sha256(
                (self._run_dir(str(manifest["run_tag"])) / "run.json").read_bytes()
            ).hexdigest(),
            "environment_contract_sha256": str(
                manifest["environment_contract_sha256"]
            ),
        }

    def _finalize_failure(
        self,
        *,
        ledger: ExperimentLedger,
        experiment_id: str,
        staging: Path,
        status: str,
        exc: BaseException,
    ) -> Mapping[str, Any]:
        failure = _exception_summary(exc)
        payload = _json_bytes(failure)
        # If even the exception text contains a forbidden token, retain only
        # its type.  Never copy a secret-like value into durable evidence.
        try:
            assert_no_leaks(payload, source="failure.json")
        except LeakError:
            failure = {"type": type(exc).__name__, "message": "redacted"}
            payload = _json_bytes(failure)
        self._write_artifact(staging, "failure.json", payload)
        return ledger.finalize_experiment(
            experiment_id,
            status=status,
            fields={
                "checks": {"contract": status != "contract_failure"},
                "decision_reason": failure["message"],
                "failure": failure,
            },
            artifact_staging=staging,
        )

    @staticmethod
    def _read_external_bytes(path: Path, *, maximum_bytes: int) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise RunContractError(f"external aggregate artifact is missing: {path}")
        try:
            size = path.stat().st_size
            if size > maximum_bytes:
                raise RunContractError(
                    f"external aggregate artifact is too large: {path.name}"
                )
            return path.read_bytes()
        except OSError as exc:
            raise RunContractError(
                f"cannot read external aggregate artifact: {path}"
            ) from exc

    @staticmethod
    def _external_artifact_path(raw: Any, *, manifest_path: Path) -> Path:
        value = Path(str(raw)).expanduser()
        return (
            value.resolve()
            if value.is_absolute()
            else (manifest_path.parent / value).resolve()
        )

    def validate_external_evaluation(
        self,
        *,
        run_tag: str,
        aggregate_manifest_path: str | os.PathLike[str],
        mode: str,
        seed_set_name: str,
        candidate_sha256: str,
        incumbent_sha256: str,
    ) -> Mapping[str, Any]:
        """Revalidate a Quest aggregate instead of trusting its JSON gate.

        Every frozen identity is compared with ``run.json`` and current source
        bytes.  Records, order, completeness, hashes, summaries, comparator
        cache, and the mechanical decision are recomputed locally.  This hook
        is intentionally incapable of authorizing confirmation.
        """

        if mode not in {"baseline", "experiment"}:
            raise RunContractError("external aggregate mode must be baseline or experiment")
        if seed_set_name not in {"smoke", "development"}:
            raise RunContractError(
                "normal external evaluation cannot access confirmation seeds"
            )
        manifest = self._load_run(run_tag, verify_sources=True)
        aggregate_path = Path(aggregate_manifest_path).expanduser().resolve()
        aggregate_payload = self._read_external_bytes(
            aggregate_path, maximum_bytes=8 * 1024 * 1024
        )
        assert_no_leaks(aggregate_payload, source="aggregate.manifest.json")
        try:
            aggregate = json.loads(aggregate_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RunContractError("external aggregate manifest is invalid JSON") from exc
        if not isinstance(aggregate, Mapping):
            raise RunContractError("external aggregate manifest must be an object")
        if aggregate.get("schema_version") != 1 or aggregate.get(
            "artifact_type"
        ) != "autoresearch_shard_aggregate":
            raise RunContractError("external aggregate schema/type is not registered")
        if aggregate.get("mode") != mode:
            raise RunContractError("external aggregate mode mismatch")
        seed_set = seed_set_from_config(manifest["config"], seed_set_name)
        aggregate_seed = aggregate.get("seed_set")
        if not isinstance(aggregate_seed, Mapping):
            raise RunContractError("external aggregate seed identity is missing")
        if aggregate_seed.get("name") != seed_set_name or aggregate_seed.get(
            "id"
        ) != seed_set.seed_set_id:
            raise RunContractError("external aggregate used another registered seed set")
        if aggregate_seed.get("spent_state_was_mutated") is not False:
            raise RunContractError("external worker mutated runner-owned spent state")
        if (
            aggregate_seed.get("requires_explicit_authorization") is not False
            or aggregate_seed.get("one_time") is not False
            or aggregate_seed.get("spent_state_owner") != "autoresearch runner"
        ):
            raise RunContractError("confirmation aggregate cannot enter normal selection")
        if aggregate.get("confirmation") is not None:
            raise RunContractError("normal aggregate contains confirmation evidence")

        checks = aggregate.get("checks")
        expected_check_names = {
            "comparator_cache_identity",
            "determinism",
            "identity_hashes",
            "records_complete",
            "shard_coverage",
            "source_guard",
        }
        if not isinstance(checks, Mapping) or set(checks) != expected_check_names:
            raise RunContractError("external aggregate hard-check set is incomplete")
        if not all(value is True for value in checks.values()):
            failed = sorted(name for name, value in checks.items() if value is not True)
            raise RunContractError(
                "external aggregate hard checks failed: " + ", ".join(failed)
            )

        source = manifest["config"]["source"]
        resolved_path = self.repo_root / str(source["resolved_config_path"])
        try:
            resolved_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
            max_horizon = int(resolved_payload["environment"]["max_step"])
        except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
            raise RunContractError(
                "verified resolved config has no positive environment.max_step"
            ) from exc
        if max_horizon <= 0:
            raise RunContractError(
                "verified resolved config has no positive environment.max_step"
            )
        immutable = manifest["immutable_manifest"]
        required_code = {
            "contract_sha256": "autoresearch/contract.py",
            "evaluator_sha256": "autoresearch/evaluator.py",
            "guard_sha256": "autoresearch/guard.py",
            "sharding_sha256": "autoresearch/sharding.py",
            "worker_sha256": "autoresearch/worker.py",
        }
        missing_code = [path for path in required_code.values() if path not in immutable]
        if missing_code:
            raise RunContractError(
                "run manifest predates Quest sharding identities: "
                + ", ".join(missing_code)
            )
        expected_identity: dict[str, Any] = {
            "candidate_sha256": candidate_sha256,
            "checkpoint_sha256": str(source["checkpoint_sha256"]),
            "config_sha256": str(manifest["config_sha256"]),
            "environment_contract_sha256": str(
                manifest["environment_contract_sha256"]
            ),
            "incumbent_sha256": incumbent_sha256,
            "max_horizon": max_horizon,
            "ordered_seed_sha256": ordered_seed_sha256(seed_set.seeds),
            "public_history_limit": int(
                manifest["config"]["evaluation"]["maximum_history_length"]
            ),
            "resolved_config_sha256": str(source["resolved_config_sha256"]),
            "seed_set_id": seed_set.seed_set_id,
            **{
                name: str(immutable[path])
                for name, path in required_code.items()
            },
        }
        expected_identity["run_identity_sha256"] = canonical_sha256(
            expected_identity
        )
        identity = aggregate.get("identity")
        if identity != expected_identity:
            raise RunContractError("external aggregate frozen identity mismatch")

        artifacts = aggregate.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RunContractError("external aggregate artifact manifest is missing")
        paths = {
            name: self._external_artifact_path(artifacts.get(name, ""), manifest_path=aggregate_path)
            for name in ("records", "summary", "gate", "comparator_cache")
        }
        records_payload = self._read_external_bytes(
            paths["records"], maximum_bytes=64 * 1024 * 1024
        )
        if not hmac.compare_digest(
            hashlib.sha256(records_payload).hexdigest(),
            str(artifacts.get("records_file_sha256", "")),
        ):
            raise RunContractError("external records file SHA-256 mismatch")
        assert_no_leaks(records_payload, source="records.jsonl")
        enriched: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(records_payload.splitlines(), 1):
            if not line.strip():
                raise RunContractError(
                    f"external records contain a blank line at {line_number}"
                )
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RunContractError(
                    f"external record {line_number} is invalid JSON"
                ) from exc
            if not isinstance(record, Mapping):
                raise RunContractError(
                    f"external record {line_number} is not an object"
                )
            if record.get("identity") != expected_identity or record.get(
                "seed_set_id"
            ) != seed_set.seed_set_id:
                raise RunContractError(
                    f"external record {line_number} identity mismatch"
                )
            enriched.append(record)
        plain = [
            {
                str(key): value
                for key, value in record.items()
                if key not in {"identity", "seed_set_id"}
            }
            for record in enriched
        ]
        expected_record_fields = {
            "method",
            "seed",
            "action_trace_sha256",
            *EPISODE_RESULT_FIELDS,
        }
        def assert_strict_records(
            records_to_check: Sequence[Mapping[str, Any]], *, label: str
        ) -> None:
            for index, record in enumerate(records_to_check):
                if set(record) != expected_record_fields:
                    raise RunContractError(
                        f"{label} record {index} contains non-evaluator fields"
                    )
                if (
                    isinstance(record["seed"], bool)
                    or not isinstance(record["seed"], int)
                    or not isinstance(record["method"], str)
                    or not _sha256_shape(record["action_trace_sha256"])
                ):
                    raise RunContractError(
                        f"{label} record {index} has malformed identity fields"
                    )
                if any(
                    type(record[field]) is not bool
                    for field in ("clean_success", "capture_episode", "goal_reached")
                ):
                    raise RunContractError(
                        f"{label} record {index} has non-boolean outcomes"
                    )
                steps = record["steps"]
                if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                    raise RunContractError(
                        f"{label} record {index} has invalid step count"
                    )
                for field in (
                    "minimum_predator_distance",
                    "path_cost",
                    "gaze_travel_degrees",
                    "predator_pixels_visible_fraction",
                ):
                    value = record[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise RunContractError(
                            f"{label} record {index} has invalid {field}"
                        )
                    if not math.isfinite(float(value)):
                        raise RunContractError(
                            f"{label} record {index} has non-finite {field}"
                        )
                visible = float(record["predator_pixels_visible_fraction"])
                if not 0.0 <= visible <= 1.0:
                    raise RunContractError(
                        f"{label} record {index} has invalid visible fraction"
                    )

        assert_strict_records(plain, label="external")
        methods = COMPARATOR_METHODS if mode == "baseline" else EVALUATION_METHODS
        validate_episode_records(plain, seeds=seed_set.seeds, methods=methods)
        actual_records_sha = records_sha256(plain)
        if not hmac.compare_digest(
            actual_records_sha, str(artifacts.get("records_sha256", ""))
        ):
            raise RunContractError("external plain-record identity mismatch")

        summary_payload = self._read_external_bytes(
            paths["summary"], maximum_bytes=4 * 1024 * 1024
        )
        gate_payload = self._read_external_bytes(
            paths["gate"], maximum_bytes=4 * 1024 * 1024
        )
        for name, payload in (("summary", summary_payload), ("gate", gate_payload)):
            if not hmac.compare_digest(
                hashlib.sha256(payload).hexdigest(),
                str(artifacts.get(f"{name}_sha256", "")),
            ):
                raise RunContractError(f"external {name} SHA-256 mismatch")
            assert_no_leaks(payload, source=f"{name}.json")
        try:
            supplied_summary = json.loads(summary_payload)
            supplied_gate = json.loads(gate_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RunContractError("external summary/gate JSON is invalid") from exc
        recomputed_summary = summarize_records(plain)
        if supplied_summary != recomputed_summary:
            raise RunContractError("external summary differs from recomputed records")

        cache_identity = ComparatorCacheIdentity.from_seeds(
            checkpoint_sha256=str(source["checkpoint_sha256"]),
            resolved_config_sha256=str(source["resolved_config_sha256"]),
            evaluator_sha256=str(manifest["evaluator_sha256"]),
            seed_set_id=seed_set.seed_set_id,
            seeds=seed_set.seeds,
            environment_contract_sha256=str(
                manifest["environment_contract_sha256"]
            ),
            incumbent_sha256=incumbent_sha256,
            max_horizon=max_horizon,
            public_history_limit=int(
                manifest["config"]["evaluation"]["maximum_history_length"]
            ),
        )
        comparator_payload = self._read_external_bytes(
            paths["comparator_cache"], maximum_bytes=64 * 1024 * 1024
        )
        assert_no_leaks(comparator_payload, source="comparator_cache.json")
        registered_cache_digest = artifacts.get("comparator_cache_sha256")
        if registered_cache_digest is not None and not hmac.compare_digest(
            hashlib.sha256(comparator_payload).hexdigest(),
            str(registered_cache_digest),
        ):
            raise RunContractError("external comparator cache SHA-256 mismatch")
        try:
            cache_envelope = json.loads(comparator_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RunContractError("external comparator cache is invalid JSON") from exc
        if not isinstance(cache_envelope, Mapping):
            raise RunContractError("external comparator cache is not an object")
        cached_raw = cache_envelope.get("records")
        if (
            cache_envelope.get("cache_schema_version") != 1
            or cache_envelope.get("cache_key") != cache_identity.key
            or cache_envelope.get("identity") != cache_identity.payload()
            or cache_envelope.get("determinism_verified") is not True
            or not isinstance(cached_raw, list)
            or cache_envelope.get("records_sha256")
            != canonical_sha256(cached_raw)
        ):
            raise RunContractError("external comparator cache is missing or stale")
        cached = [dict(record) for record in cached_raw]
        assert_strict_records(cached, label="comparator cache")
        validate_episode_records(
            cached, seeds=seed_set.seeds, methods=COMPARATOR_METHODS
        )
        if mode == "baseline":
            if records_sha256(cached) != actual_records_sha:
                raise RunContractError(
                    "external baseline records differ from comparator cache"
                )
            expected_gate = {
                "decision": "baseline_cached",
                "decision_reason": (
                    "all incumbent and fixed_p60 comparator shards passed strict "
                    "coverage, identity, and repeat=2 determinism checks"
                ),
                "keep": False,
            }
            if supplied_gate != expected_gate:
                raise RunContractError("external baseline gate is invalid")
        else:
            comparator_from_records = [
                record for record in plain if record["method"] in COMPARATOR_METHODS
            ]
            if records_sha256(comparator_from_records) != records_sha256(cached):
                raise RunContractError(
                    "external experiment comparator records differ from cache"
                )
            recomputed_gate = mechanical_keep_or_discard(
                plain,
                seeds=seed_set.seeds,
                checks=checks,
                minimum_improvement_episodes=int(
                    manifest["config"]["decision"][
                        "minimum_paired_episode_improvement"
                    ]
                ),
            ).to_dict()
            if supplied_gate != recomputed_gate:
                raise RunContractError(
                    "external gate differs from the locally recomputed decision"
                )

        shards = aggregate.get("shards")
        expected_records = len(seed_set.seeds) * (
            len(COMPARATOR_METHODS) if mode == "baseline" else 1
        )
        if not isinstance(shards, list) or sum(
            int(item.get("records", -1))
            for item in shards
            if isinstance(item, Mapping)
        ) != expected_records:
            raise RunContractError("external shard coverage metadata is incomplete")
        return {
            "records": plain,
            "records_sha256": actual_records_sha,
            "summary": recomputed_summary,
            "checks": dict(checks),
            "cache_hit": mode == "experiment",
            "decision": dict(supplied_gate),
            "identity": dict(expected_identity),
            "seed_set": asdict(seed_set),
            "external_aggregate": {
                "path": str(aggregate_path),
                "sha256": hashlib.sha256(aggregate_payload).hexdigest(),
            },
        }

    def baseline(
        self,
        *,
        run_tag: str,
        evaluation_result: str | os.PathLike[str] | None = None,
    ) -> Mapping[str, Any]:
        """Evaluate and register the initial legal-rate scan incumbent."""

        if evaluation_result is not None:
            raise RunContractError(
                "external baseline requires prepare_external_baseline followed by "
                "finalize_external_baseline"
            )

        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            self._recover_startup(ledger)
            baseline_id = str(
                manifest["config"]["candidate"].get(
                    "initial_incumbent_id", "legal_fixed_scan_v1"
                )
            )
            existing = ledger.latest_records().get(baseline_id)
            if existing is not None:
                if existing["status"] == "keep":
                    return existing
                raise RunContractError(
                    f"initial baseline {baseline_id} is already {existing['status']}"
                )
            if self._incumbent_record(ledger) is not None:
                raise RunContractError("a different incumbent is already registered")

            relative, candidate_path = self._candidate_path(manifest)
            candidate_sha = self._verify_initial_incumbent_binding(
                manifest, candidate_path
            )
            plan_fields = self._base_plan_fields(
                manifest=manifest,
                parent_incumbent_id=None,
                candidate_commit=str(manifest["setup_commit"]),
                candidate_sha256=candidate_sha,
                changed_paths=(),
                hypothesis="Register the frozen legal-rate scan as search incumbent.",
                predicted_effect="Establish paired engineering reference evidence.",
                seed_set_name="development",
            )
            ledger.plan_experiment({"experiment_id": baseline_id, **plan_fields})
            ledger.start_experiment(baseline_id)
            staging = ledger.begin_artifacts(baseline_id)
            candidate = None
            incumbent = None
            try:
                self._load_run(run_tag, verify_sources=True)
                candidate, candidate_sha, source = self._validate_and_load_candidate(
                    candidate_path
                )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    candidate_path
                )
                if candidate_sha != plan_fields["candidate_sha256"]:
                    raise RunContractError("candidate changed after baseline was planned")
                if candidate_sha != str(
                    manifest["config"]["candidate"]["initial_incumbent_sha256"]
                ):
                    raise RunContractError(
                        "loaded baseline is not the registered legal-scan incumbent"
                    )
                checks = {
                    "immutable_hashes": True,
                    "candidate_source": True,
                    "changed_paths": True,
                    "source_artifacts": True,
                    "seed_separation": True,
                    "leak_scan": True,
                }
                factory = self.episode_factory_builder(
                    manifest["config"], project_root=self.repo_root
                )
                smoke = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="smoke",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir / "comparator-cache" / "smoke.json",
                )
                development = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="development",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir
                    / "comparator-cache"
                    / "development.json",
                )
                smoke_payload = _json_bytes(smoke)
                development_payload = _json_bytes(development)
                assert_no_leaks(smoke_payload, source="smoke.json")
                assert_no_leaks(development_payload, source="development.json")
                self._write_artifact(staging, "candidate.py", source)
                self._write_artifact(staging, "smoke.json", smoke_payload)
                self._write_artifact(
                    staging, "development.json", development_payload
                )
                decision = development.get("decision", {})
                return ledger.finalize_experiment(
                    baseline_id,
                    status="keep",
                    fields={
                        "candidate_sha256": candidate_sha,
                        "primary_delta": 0.0,
                        "paired_counts": dict(decision.get("paired_counts", {})),
                        "secondary_metrics": dict(
                            development.get("summary", {})
                        ),
                        "checks": dict(development["checks"]),
                        "decision_reason": (
                            "registered frozen legal-rate search incumbent; "
                            "this is not a candidate-selection claim"
                        ),
                        "baseline": True,
                        "external_evaluation": False,
                    },
                    artifact_staging=staging,
                )
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=baseline_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=baseline_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    @staticmethod
    def _read_hypothesis(path: str | os.PathLike[str]) -> str:
        hypothesis_path = Path(path)
        if hypothesis_path.is_symlink() or not hypothesis_path.is_file():
            raise RunContractError(
                f"hypothesis must be a regular non-symlink file: {hypothesis_path}"
            )
        try:
            payload = hypothesis_path.read_bytes()
            if len(payload) > MAX_HYPOTHESIS_BYTES:
                raise RunContractError("hypothesis file exceeds 32 KiB")
            text = payload.decode("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise RunContractError(f"cannot read hypothesis file: {hypothesis_path}") from exc
        if not text:
            raise RunContractError("hypothesis file is empty")
        assert_no_leaks(text, source="hypothesis")
        return text

    def experiment(
        self,
        *,
        run_tag: str,
        hypothesis_file: str | os.PathLike[str],
        predicted_effect: str = "Improve paired development clean success.",
        candidate_commit: str | None = None,
        changed_paths: Sequence[str] | None = None,
        evaluation_result: str | os.PathLike[str] | None = None,
    ) -> Mapping[str, Any]:
        """Run exactly one smoke+development candidate selection.

        ``candidate_commit`` and ``changed_paths`` are injectable for unit-test
        repositories.  The production CLI does not expose either override;
        it resolves both from Git and verifies the committed candidate bytes.
        """

        if evaluation_result is not None:
            raise RunContractError(
                "external experiment requires prepare_external_experiment followed "
                "by finalize_external_experiment"
            )

        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            self._recover_startup(ledger)
            budget = self._budget_status(manifest, ledger)
            if budget["exhausted"]:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "run_tag": run_tag,
                    "status": "budget_exhausted",
                    "decision_reason": "finite budget exhausted: "
                    + ", ".join(budget["reasons"]),
                    "budget": budget,
                }
            incumbent_info = self._incumbent_record(ledger)
            if incumbent_info is None:
                raise RunContractError("baseline must be registered before experiments")
            incumbent_record, incumbent_path = incumbent_info
            hypothesis = self._read_hypothesis(hypothesis_file)
            predicted_effect = " ".join(str(predicted_effect).split())
            if not predicted_effect:
                raise RunContractError("predicted effect must be non-empty")
            assert_no_leaks(predicted_effect, source="predicted_effect")

            relative, candidate_path = self._candidate_path(manifest)
            candidate_sha = sha256_file(candidate_path)
            commit_overridden = candidate_commit is not None
            paths_overridden = changed_paths is not None
            selected_commit = str(
                candidate_commit if candidate_commit is not None else self.commit_provider()
            ).strip()
            if not COMMIT_PATTERN.fullmatch(selected_commit):
                raise RunContractError(
                    "candidate commit must be a full lowercase hexadecimal commit"
                )
            parent_commit = str(incumbent_record.get("candidate_commit") or "")
            selected_paths = (
                tuple(changed_paths)
                if changed_paths is not None
                else tuple(self.changed_paths_provider(parent_commit, selected_commit))
            )

            experiment_id = self._next_experiment_id(ledger)
            plan_fields = self._base_plan_fields(
                manifest=manifest,
                parent_incumbent_id=str(incumbent_record["experiment_id"]),
                candidate_commit=selected_commit,
                candidate_sha256=candidate_sha,
                changed_paths=selected_paths,
                hypothesis=hypothesis,
                predicted_effect=predicted_effect,
                seed_set_name="development",
            )
            ledger.plan_experiment({"experiment_id": experiment_id, **plan_fields})
            ledger.start_experiment(experiment_id)
            staging = ledger.begin_artifacts(experiment_id)
            candidate = None
            incumbent = None
            try:
                self._load_run(run_tag, verify_sources=True)
                normalized_paths = validate_changed_paths(
                    selected_paths, allowed_paths=manifest["mutable_paths"]
                )
                if normalized_paths != (relative,):
                    raise ChangedPathError(
                        "each experiment must change exactly autoresearch/candidate.py"
                    )
                candidate, loaded_sha, source = self._validate_and_load_candidate(
                    candidate_path
                )
                if loaded_sha != candidate_sha:
                    raise RunContractError("candidate changed after experiment was planned")
                if not commit_overridden and not paths_overridden:
                    committed = self.committed_file_provider(selected_commit, relative)
                    if not hmac.compare_digest(sha256_bytes(committed), candidate_sha):
                        raise RunContractError(
                            "working candidate bytes differ from candidate commit"
                        )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    incumbent_path
                )
                if incumbent_sha != str(incumbent_record["candidate_sha256"]):
                    raise RunContractError("incumbent identity changed before evaluation")
                checks = {
                    "immutable_hashes": True,
                    "candidate_source": True,
                    "changed_paths": True,
                    "source_artifacts": True,
                    "seed_separation": True,
                    "leak_scan": True,
                    "commit_snapshot": True,
                }
                factory = self.episode_factory_builder(
                    manifest["config"], project_root=self.repo_root
                )
                smoke = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="smoke",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir / "comparator-cache" / "smoke.json",
                )
                development = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="development",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir
                    / "comparator-cache"
                    / "development.json",
                )
                decision = development.get("decision")
                if not isinstance(decision, Mapping) or decision.get("decision") not in {
                    "keep",
                    "discard",
                }:
                    raise RunContractError(
                        "development evaluator returned no mechanical decision"
                    )
                smoke_payload = _json_bytes(smoke)
                development_payload = _json_bytes(development)
                assert_no_leaks(smoke_payload, source="smoke.json")
                assert_no_leaks(development_payload, source="development.json")
                self._write_artifact(staging, "candidate.py", source)
                self._write_artifact(
                    staging, "hypothesis.md", (hypothesis + "\n").encode("utf-8")
                )
                self._write_artifact(staging, "smoke.json", smoke_payload)
                self._write_artifact(
                    staging, "development.json", development_payload
                )
                return ledger.finalize_experiment(
                    experiment_id,
                    status=str(decision["decision"]),
                    fields={
                        "changed_paths": list(normalized_paths),
                        "primary_delta": float(decision["primary_delta"]),
                        "paired_counts": dict(decision.get("paired_counts", {})),
                        "secondary_metrics": dict(
                            development.get("summary", {})
                        ),
                        "checks": dict(development["checks"]),
                        "decision_reason": str(decision["decision_reason"]),
                        "records_sha256": str(
                            development.get("records_sha256", "")
                        ),
                        "selection_scope": "engineering_selection",
                        "external_evaluation": False,
                    },
                    artifact_staging=staging,
                )
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    def prepare_external_baseline(self, *, run_tag: str) -> Mapping[str, Any]:
        """Preregister baseline evidence and run smoke before Quest rollout."""

        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        baseline_id = str(
            manifest["config"]["candidate"].get(
                "initial_incumbent_id", "legal_fixed_scan_v1"
            )
        )
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            existing = ledger.latest_records().get(baseline_id)
            if existing is not None:
                if existing["status"] == "running" and existing.get(
                    "external_prepared"
                ) is True:
                    staging = ledger.artifact_staging_path(baseline_id)
                    context = self.worker_context(
                        run_tag=run_tag,
                        candidate_path=staging / "candidate.py",
                        candidate_commit=str(existing.get("candidate_commit") or ""),
                        incumbent_path=staging / "candidate.py",
                    )
                    return {
                        "schema_version": SCHEMA_VERSION,
                        "run_tag": run_tag,
                        "experiment_id": baseline_id,
                        "status": "running",
                        "external_stage": "awaiting_development_aggregate",
                        "worker_context": context,
                    }
                raise RunContractError(
                    f"initial baseline {baseline_id} is already {existing['status']}"
                )
            self._recover_startup(ledger)
            if self._incumbent_record(ledger) is not None:
                raise RunContractError("a different incumbent is already registered")
            _, candidate_path = self._candidate_path(manifest)
            candidate_sha = self._verify_initial_incumbent_binding(
                manifest, candidate_path
            )
            plan_fields = {
                **self._base_plan_fields(
                    manifest=manifest,
                    parent_incumbent_id=None,
                    candidate_commit=str(manifest["setup_commit"]),
                    candidate_sha256=candidate_sha,
                    changed_paths=(),
                    hypothesis="Register the frozen legal-rate scan as search incumbent.",
                    predicted_effect="Establish paired engineering reference evidence.",
                    seed_set_name="development",
                ),
                "external_prepared": True,
                "external_mode": "baseline",
                "external_stage": "awaiting_development_aggregate",
            }
            ledger.plan_experiment({"experiment_id": baseline_id, **plan_fields})
            ledger.start_experiment(baseline_id)
            staging = ledger.begin_artifacts(baseline_id)
            candidate = None
            incumbent = None
            try:
                self._load_run(run_tag, verify_sources=True)
                candidate, loaded_sha, source = self._validate_and_load_candidate(
                    candidate_path
                )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    candidate_path
                )
                if loaded_sha != candidate_sha or incumbent_sha != candidate_sha:
                    raise RunContractError("baseline candidate changed after planning")
                if loaded_sha != str(
                    manifest["config"]["candidate"]["initial_incumbent_sha256"]
                ):
                    raise RunContractError(
                        "loaded baseline is not the registered legal-scan incumbent"
                    )
                checks = {
                    "immutable_hashes": True,
                    "candidate_source": True,
                    "changed_paths": True,
                    "source_artifacts": True,
                    "seed_separation": True,
                    "leak_scan": True,
                }
                factory = self.episode_factory_builder(
                    manifest["config"], project_root=self.repo_root
                )
                smoke = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="smoke",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir / "comparator-cache" / "smoke.json",
                )
                smoke_payload = _json_bytes(smoke)
                assert_no_leaks(smoke_payload, source="smoke.json")
                self._write_artifact(staging, "candidate.py", source)
                self._write_artifact(staging, "smoke.json", smoke_payload)
                context = self.worker_context(
                    run_tag=run_tag,
                    candidate_path=staging / "candidate.py",
                    candidate_commit=str(manifest["setup_commit"]),
                    incumbent_path=staging / "candidate.py",
                )
                preparation = {
                    "experiment_id": baseline_id,
                    "external_mode": "baseline",
                    "smoke_records_sha256": smoke.get("records_sha256"),
                    "worker_context": context,
                }
                self._write_artifact(
                    staging, "external-preparation.json", _json_bytes(preparation)
                )
                return {
                    "schema_version": SCHEMA_VERSION,
                    "run_tag": run_tag,
                    "experiment_id": baseline_id,
                    "status": "running",
                    "external_stage": "awaiting_development_aggregate",
                    "worker_context": context,
                }
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=baseline_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=baseline_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    def finalize_external_baseline(
        self,
        *,
        run_tag: str,
        experiment_id: str,
        aggregate_manifest_path: str | os.PathLike[str],
    ) -> Mapping[str, Any]:
        """Validate Quest baseline evidence and finalize its preregistered record."""

        manifest = self._load_run(run_tag)
        baseline_id = str(
            manifest["config"]["candidate"].get(
                "initial_incumbent_id", "legal_fixed_scan_v1"
            )
        )
        if experiment_id != baseline_id:
            raise RunContractError("external baseline experiment ID mismatch")
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            latest = ledger.latest_records().get(experiment_id)
            if latest is None:
                raise RunContractError("external baseline was not prepared")
            if latest["status"] in TERMINAL_STATUSES:
                supplied_digest = hashlib.sha256(
                    self._read_external_bytes(
                        Path(aggregate_manifest_path).expanduser().resolve(),
                        maximum_bytes=8 * 1024 * 1024,
                    )
                ).hexdigest()
                if latest.get("external_aggregate_sha256") != supplied_digest:
                    raise RunContractError(
                        "external baseline is terminal with different evidence"
                    )
                return latest
            if latest["status"] != "running" or latest.get(
                "external_prepared"
            ) is not True or latest.get("external_mode") != "baseline":
                raise RunContractError("external baseline is not in prepared running state")
            staging = ledger.begin_artifacts(experiment_id, resume=True)
            candidate = None
            try:
                ledger.resume_experiment(
                    experiment_id, evaluation_is_idempotent=True
                )
                self._load_run(run_tag, verify_sources=True)
                candidate_path = staging / "candidate.py"
                candidate, candidate_sha, _ = self._validate_and_load_candidate(
                    candidate_path
                )
                if candidate_sha != str(latest["candidate_sha256"]):
                    raise RunContractError("prepared baseline candidate hash mismatch")
                development = self.validate_external_evaluation(
                    run_tag=run_tag,
                    aggregate_manifest_path=aggregate_manifest_path,
                    mode="baseline",
                    seed_set_name="development",
                    candidate_sha256=candidate_sha,
                    incumbent_sha256=candidate_sha,
                )
                payload = _json_bytes(development)
                assert_no_leaks(payload, source="development.json")
                self._write_artifact(staging, "development.json", payload)
                decision = development.get("decision", {})
                checks = {
                    **dict(development["checks"]),
                    "registered_smoke": True,
                    "external_preregistered": True,
                }
                return ledger.finalize_experiment(
                    experiment_id,
                    status="keep",
                    fields={
                        "primary_delta": 0.0,
                        "paired_counts": dict(decision.get("paired_counts", {})),
                        "secondary_metrics": dict(development.get("summary", {})),
                        "checks": checks,
                        "decision_reason": (
                            "registered frozen legal-rate search incumbent from "
                            "strictly revalidated Quest evidence; no selection claim"
                        ),
                        "baseline": True,
                        "external_evaluation": True,
                        "external_stage": "finalized",
                        "external_aggregate_sha256": development[
                            "external_aggregate"
                        ]["sha256"],
                    },
                    artifact_staging=staging,
                )
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate)

    def prepare_external_experiment(
        self,
        *,
        run_tag: str,
        hypothesis_file: str | os.PathLike[str],
        predicted_effect: str = "Improve paired development clean success.",
        candidate_commit: str | None = None,
        changed_paths: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        """Preregister one candidate, validate it, and run the smoke gate."""

        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            prepared = [
                record
                for record in ledger.latest_records().values()
                if record["status"] == "running"
                and record.get("external_prepared") is True
                and record.get("external_mode") == "experiment"
                and record.get("external_stage")
                == "awaiting_development_aggregate"
            ]
            if prepared:
                if len(prepared) != 1:
                    raise RunContractError(
                        "multiple external experiments are awaiting finalize"
                    )
                existing = prepared[0]
                retry_hypothesis = self._read_hypothesis(hypothesis_file)
                retry_effect = " ".join(str(predicted_effect).split())
                _, retry_candidate_path = self._candidate_path(manifest)
                retry_sha = sha256_file(retry_candidate_path)
                retry_commit = str(
                    candidate_commit
                    if candidate_commit is not None
                    else self.commit_provider()
                ).strip()
                parent_id = str(existing.get("parent_incumbent_id") or "")
                parent_record = self._latest_by_id(ledger, parent_id)
                retry_paths = (
                    tuple(changed_paths)
                    if changed_paths is not None
                    else tuple(
                        self.changed_paths_provider(
                            str(parent_record.get("candidate_commit") or ""),
                            retry_commit,
                        )
                    )
                )
                if (
                    existing.get("hypothesis") != retry_hypothesis
                    or existing.get("predicted_effect") != retry_effect
                    or existing.get("candidate_sha256") != retry_sha
                    or existing.get("candidate_commit") != retry_commit
                    or tuple(existing.get("changed_paths", ())) != retry_paths
                ):
                    raise RunContractError(
                        "external prepare retry conflicts with the running experiment"
                    )
                staging = ledger.artifact_staging_path(
                    str(existing["experiment_id"])
                )
                context = self.worker_context(
                    run_tag=run_tag,
                    candidate_path=staging / "candidate.py",
                    candidate_commit=retry_commit,
                )
                return {
                    "schema_version": SCHEMA_VERSION,
                    "run_tag": run_tag,
                    "experiment_id": existing["experiment_id"],
                    "status": "running",
                    "external_stage": "awaiting_development_aggregate",
                    "worker_context": context,
                }
            self._recover_startup(ledger)
            budget = self._budget_status(manifest, ledger)
            if budget["exhausted"]:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "run_tag": run_tag,
                    "status": "budget_exhausted",
                    "decision_reason": "finite budget exhausted: "
                    + ", ".join(budget["reasons"]),
                    "budget": budget,
                }
            incumbent_info = self._incumbent_record(ledger)
            if incumbent_info is None:
                raise RunContractError("baseline must be registered before experiments")
            incumbent_record, incumbent_path = incumbent_info
            hypothesis = self._read_hypothesis(hypothesis_file)
            predicted_effect = " ".join(str(predicted_effect).split())
            if not predicted_effect:
                raise RunContractError("predicted effect must be non-empty")
            assert_no_leaks(predicted_effect, source="predicted_effect")
            relative, candidate_path = self._candidate_path(manifest)
            try:
                candidate_sha = sha256_file(candidate_path)
            except OSError as exc:
                raise RunContractError("candidate source is missing") from exc
            commit_overridden = candidate_commit is not None
            paths_overridden = changed_paths is not None
            selected_commit = str(
                candidate_commit if candidate_commit is not None else self.commit_provider()
            ).strip()
            if not COMMIT_PATTERN.fullmatch(selected_commit):
                raise RunContractError(
                    "candidate commit must be a full lowercase hexadecimal commit"
                )
            parent_commit = str(incumbent_record.get("candidate_commit") or "")
            selected_paths = (
                tuple(changed_paths)
                if changed_paths is not None
                else tuple(self.changed_paths_provider(parent_commit, selected_commit))
            )
            experiment_id = self._next_experiment_id(ledger)
            plan_fields = {
                **self._base_plan_fields(
                    manifest=manifest,
                    parent_incumbent_id=str(incumbent_record["experiment_id"]),
                    candidate_commit=selected_commit,
                    candidate_sha256=candidate_sha,
                    changed_paths=selected_paths,
                    hypothesis=hypothesis,
                    predicted_effect=predicted_effect,
                    seed_set_name="development",
                ),
                "external_prepared": True,
                "external_mode": "experiment",
                "external_stage": "awaiting_development_aggregate",
            }
            ledger.plan_experiment({"experiment_id": experiment_id, **plan_fields})
            ledger.start_experiment(experiment_id)
            staging = ledger.begin_artifacts(experiment_id)
            candidate = None
            incumbent = None
            try:
                self._load_run(run_tag, verify_sources=True)
                normalized_paths = validate_changed_paths(
                    selected_paths, allowed_paths=manifest["mutable_paths"]
                )
                if normalized_paths != (relative,):
                    raise ChangedPathError(
                        "each experiment must change exactly autoresearch/candidate.py"
                    )
                candidate, loaded_sha, source = self._validate_and_load_candidate(
                    candidate_path
                )
                if loaded_sha != candidate_sha:
                    raise RunContractError("candidate changed after experiment was planned")
                if not commit_overridden and not paths_overridden:
                    committed = self.committed_file_provider(selected_commit, relative)
                    if not hmac.compare_digest(sha256_bytes(committed), candidate_sha):
                        raise RunContractError(
                            "working candidate bytes differ from candidate commit"
                        )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    incumbent_path
                )
                if incumbent_sha != str(incumbent_record["candidate_sha256"]):
                    raise RunContractError("incumbent identity changed before smoke")
                checks = {
                    "immutable_hashes": True,
                    "candidate_source": True,
                    "changed_paths": True,
                    "source_artifacts": True,
                    "seed_separation": True,
                    "leak_scan": True,
                    "commit_snapshot": True,
                }
                factory = self.episode_factory_builder(
                    manifest["config"], project_root=self.repo_root
                )
                smoke = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="smoke",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir / "comparator-cache" / "smoke.json",
                )
                smoke_payload = _json_bytes(smoke)
                assert_no_leaks(smoke_payload, source="smoke.json")
                self._write_artifact(staging, "candidate.py", source)
                self._write_artifact(
                    staging, "hypothesis.md", (hypothesis + "\n").encode("utf-8")
                )
                self._write_artifact(staging, "smoke.json", smoke_payload)
                context = self.worker_context(
                    run_tag=run_tag,
                    candidate_path=staging / "candidate.py",
                    candidate_commit=selected_commit,
                )
                preparation = {
                    "experiment_id": experiment_id,
                    "external_mode": "experiment",
                    "smoke_records_sha256": smoke.get("records_sha256"),
                    "worker_context": context,
                }
                self._write_artifact(
                    staging, "external-preparation.json", _json_bytes(preparation)
                )
                return {
                    "schema_version": SCHEMA_VERSION,
                    "run_tag": run_tag,
                    "experiment_id": experiment_id,
                    "status": "running",
                    "external_stage": "awaiting_development_aggregate",
                    "worker_context": context,
                }
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    def finalize_external_experiment(
        self,
        *,
        run_tag: str,
        experiment_id: str,
        aggregate_manifest_path: str | os.PathLike[str],
    ) -> Mapping[str, Any]:
        """Recompute and finalize a preregistered Quest candidate evaluation."""

        manifest = self._load_run(run_tag)
        if not EXPERIMENT_PATTERN.fullmatch(experiment_id):
            raise RunContractError("external experiment ID must have form E####")
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            latest = ledger.latest_records().get(experiment_id)
            if latest is None:
                raise RunContractError("external experiment was not prepared")
            if latest["status"] in TERMINAL_STATUSES:
                supplied_digest = hashlib.sha256(
                    self._read_external_bytes(
                        Path(aggregate_manifest_path).expanduser().resolve(),
                        maximum_bytes=8 * 1024 * 1024,
                    )
                ).hexdigest()
                if latest.get("external_aggregate_sha256") != supplied_digest:
                    raise RunContractError(
                        "external experiment is terminal with different evidence"
                    )
                return latest
            if latest["status"] != "running" or latest.get(
                "external_prepared"
            ) is not True or latest.get("external_mode") != "experiment":
                raise RunContractError("external experiment is not prepared and running")
            parent_id = str(latest.get("parent_incumbent_id") or "")
            parent_record = self._latest_by_id(ledger, parent_id)
            incumbent_path = ledger.artifact_path(parent_id) / "candidate.py"
            staging = ledger.begin_artifacts(experiment_id, resume=True)
            candidate = None
            incumbent = None
            try:
                ledger.resume_experiment(
                    experiment_id, evaluation_is_idempotent=True
                )
                self._load_run(run_tag, verify_sources=True)
                candidate_path = staging / "candidate.py"
                candidate, candidate_sha, _ = self._validate_and_load_candidate(
                    candidate_path
                )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    incumbent_path
                )
                if candidate_sha != str(latest["candidate_sha256"]):
                    raise RunContractError("prepared candidate archive hash mismatch")
                if incumbent_sha != str(parent_record["candidate_sha256"]):
                    raise RunContractError("prepared incumbent archive hash mismatch")
                development = self.validate_external_evaluation(
                    run_tag=run_tag,
                    aggregate_manifest_path=aggregate_manifest_path,
                    mode="experiment",
                    seed_set_name="development",
                    candidate_sha256=candidate_sha,
                    incumbent_sha256=incumbent_sha,
                )
                decision = development.get("decision")
                if not isinstance(decision, Mapping) or decision.get("decision") not in {
                    "keep",
                    "discard",
                }:
                    raise RunContractError(
                        "external development aggregate has no mechanical decision"
                    )
                payload = _json_bytes(development)
                assert_no_leaks(payload, source="development.json")
                self._write_artifact(staging, "development.json", payload)
                checks = {
                    **dict(development["checks"]),
                    "registered_smoke": True,
                    "external_preregistered": True,
                }
                return ledger.finalize_experiment(
                    experiment_id,
                    status=str(decision["decision"]),
                    fields={
                        "primary_delta": float(decision["primary_delta"]),
                        "paired_counts": dict(decision.get("paired_counts", {})),
                        "secondary_metrics": dict(development.get("summary", {})),
                        "checks": checks,
                        "decision_reason": str(decision["decision_reason"]),
                        "records_sha256": str(development["records_sha256"]),
                        "selection_scope": "engineering_selection",
                        "external_evaluation": True,
                        "external_stage": "finalized",
                        "external_aggregate_sha256": development[
                            "external_aggregate"
                        ]["sha256"],
                    },
                    artifact_staging=staging,
                )
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    def _confirmation_spend(
        self, manifest: Mapping[str, Any], ledger: ExperimentLedger
    ) -> Mapping[str, Any] | None:
        seed_id = seed_set_from_config(
            manifest["config"], "confirmation"
        ).seed_set_id
        for record in reversed(ledger.read_records()):
            if record.get("seed_set_id") == seed_id:
                return record
        return None

    def abort_external(
        self,
        *,
        run_tag: str,
        experiment_id: str,
        reason: str,
    ) -> Mapping[str, Any]:
        """Explicitly crash one waiting external lifecycle without deletion."""

        normalized_reason = " ".join(str(reason).split())
        if not normalized_reason:
            raise RunContractError("external abort reason must be non-empty")
        if len(normalized_reason.encode("utf-8")) > MAX_ABORT_REASON_BYTES:
            raise RunContractError("external abort reason exceeds 4 KiB")
        try:
            assert_no_leaks(normalized_reason, source="external_abort_reason")
        except LeakError as exc:
            raise RunContractError("external abort reason failed leak scan") from exc

        self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            latest = ledger.latest_records().get(experiment_id)
            if latest is None:
                raise RunContractError("external abort target does not exist")
            if latest["status"] != "running":
                raise RunContractError(
                    "external abort accepts only a running lifecycle"
                )
            if (
                latest.get("external_prepared") is not True
                or latest.get("external_stage")
                != "awaiting_development_aggregate"
                or latest.get("external_mode") not in {"baseline", "experiment"}
                or str(experiment_id).startswith("C")
                or latest.get("confirmation_authorized") is True
            ):
                raise RunContractError(
                    "abort target is not a waiting baseline/experiment aggregate"
                )
            staging = ledger.begin_artifacts(experiment_id, resume=True)
            abort_evidence = {
                "experiment_id": experiment_id,
                "reason": normalized_reason,
                "external_mode": latest["external_mode"],
                "external_stage": "aborted",
            }
            self._write_artifact(
                staging, "external-abort.json", _json_bytes(abort_evidence)
            )
            return ledger.finalize_experiment(
                experiment_id,
                status="crash",
                fields={
                    "checks": {"external_abort_recorded": True},
                    "decision_reason": normalized_reason,
                    "external_abort_reason": normalized_reason,
                    "external_stage": "aborted",
                    "recovery_action": "explicit_external_abort",
                },
                artifact_staging=staging,
            )

    def confirm(
        self,
        *,
        run_tag: str,
        authorized: bool,
    ) -> Mapping[str, Any]:
        """Spend the registered confirmation set once, without promotion."""

        if authorized is not True:
            raise ConfirmationError(
                "confirmation requires explicit --authorize-confirmation"
            )
        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        with self._run_lock(run_dir):
            ledger = ExperimentLedger(run_dir)
            self._recover_startup(ledger)
            spent = self._confirmation_spend(manifest, ledger)
            if spent is not None:
                raise ConfirmationError(
                    "confirmation seed set is already spent by "
                    f"{spent['experiment_id']} ({spent['status']})"
                )
            incumbent_info = self._incumbent_record(ledger)
            if incumbent_info is None:
                raise ConfirmationError("confirmation requires an accepted candidate")
            candidate_record, candidate_path = incumbent_info
            parent_id = candidate_record.get("parent_incumbent_id")
            if not parent_id:
                raise ConfirmationError(
                    "confirmation requires a selected candidate beyond the baseline"
                )
            parent_record = self._latest_by_id(ledger, str(parent_id))
            incumbent_path = ledger.artifact_path(str(parent_id)) / "candidate.py"
            if parent_record["status"] != "keep" or not incumbent_path.is_file():
                raise ConfirmationError("selected candidate parent is not a valid incumbent")

            experiment_id = "C0001"
            plan_fields = self._base_plan_fields(
                manifest=manifest,
                parent_incumbent_id=str(parent_id),
                candidate_commit=str(candidate_record.get("candidate_commit") or ""),
                candidate_sha256=str(candidate_record["candidate_sha256"]),
                changed_paths=(),
                hypothesis=(
                    "One-time confirmation of the selected engineering candidate."
                ),
                predicted_effect=(
                    "Paired clean-success interval excludes zero favorably without "
                    "worse capture."
                ),
                seed_set_name="confirmation",
            )
            ledger.plan_experiment(
                {
                    "experiment_id": experiment_id,
                    **plan_fields,
                    "confirmation_authorized": True,
                    "confirmation_set_spent": True,
                }
            )
            # Planning happens before any rollout, so even a crash spends the
            # registered set and prevents selective retries.
            ledger.start_experiment(experiment_id)
            staging = ledger.begin_artifacts(experiment_id)
            candidate = None
            incumbent = None
            try:
                self._load_run(run_tag, verify_sources=True)
                candidate, candidate_sha, source = self._validate_and_load_candidate(
                    candidate_path
                )
                incumbent, incumbent_sha, _ = self._validate_and_load_candidate(
                    incumbent_path
                )
                if candidate_sha != str(candidate_record["candidate_sha256"]):
                    raise RunContractError("selected candidate archive hash mismatch")
                if incumbent_sha != str(parent_record["candidate_sha256"]):
                    raise RunContractError("confirmation comparator archive hash mismatch")
                checks = {
                    "immutable_hashes": True,
                    "candidate_source": True,
                    "source_artifacts": True,
                    "seed_separation": True,
                    "leak_scan": True,
                    "explicit_confirmation_authorization": True,
                    "confirmation_unspent_before_plan": True,
                }
                factory = self.episode_factory_builder(
                    manifest["config"], project_root=self.repo_root
                )
                result = self._evaluate_phase(
                    manifest=manifest,
                    factory=factory,
                    seed_set_name="confirmation",
                    candidate=candidate,
                    incumbent=incumbent,
                    incumbent_sha256=incumbent_sha,
                    checks=checks,
                    cache_path=run_dir
                    / "comparator-cache"
                    / "confirmation.json",
                    allow_confirmation=True,
                )
                records = result.get("records")
                if not isinstance(records, Sequence):
                    raise RunContractError("confirmation evaluator returned no records")
                frozen = seed_set_from_config(manifest["config"], "confirmation")
                statistics = self.confirmation_statistics_fn(
                    records,
                    seeds=frozen.seeds,
                    bootstrap_samples=int(
                        manifest["config"]["evaluation"].get(
                            "bootstrap_samples", 5000
                        )
                    ),
                )
                decision = result.get("decision")
                if not isinstance(decision, Mapping):
                    raise RunContractError("confirmation evaluator returned no gate counts")
                capture_ok = int(decision["candidate_capture_episodes"]) <= int(
                    decision["incumbent_capture_episodes"]
                )
                interval_ok = float(statistics["bootstrap_95_low"]) > 0.0
                confirmation_passed = bool(
                    interval_ok
                    and capture_ok
                    and all(value is True for value in result["checks"].values())
                )
                envelope = {
                    "evaluation": result,
                    "statistics": statistics,
                    "confirmation_passed": confirmation_passed,
                    "capture_nonworsening": capture_ok,
                    "favorable_interval": interval_ok,
                }
                payload = _json_bytes(envelope)
                assert_no_leaks(payload, source="confirmation.json")
                self._write_artifact(staging, "candidate.py", source)
                self._write_artifact(staging, "confirmation.json", payload)
                reason = (
                    "confirmation passed engineering gate; no scientific claim"
                    if confirmation_passed
                    else "confirmation did not pass the engineering gate"
                )
                # Confirmation is report-only.  Canonical ledger status
                # ``discard`` deliberately prevents it from becoming a new
                # incumbent projection.
                return ledger.finalize_experiment(
                    experiment_id,
                    status="discard",
                    fields={
                        "primary_delta": float(statistics["mean_delta"]),
                        "paired_counts": {
                            "candidate_only_successes": int(
                                statistics["candidate_only_successes"]
                            ),
                            "incumbent_only_successes": int(
                                statistics["incumbent_only_successes"]
                            ),
                        },
                        "secondary_metrics": {
                            "confirmation_statistics": dict(statistics),
                            "summary": dict(result.get("summary", {})),
                        },
                        "checks": dict(result["checks"]),
                        "decision_reason": reason,
                        "confirmation_passed": confirmation_passed,
                        "confirmation_set_spent": True,
                    },
                    artifact_staging=staging,
                )
            except _CONTRACT_EXCEPTIONS as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="contract_failure",
                    exc=exc,
                )
            except Exception as exc:
                return self._finalize_failure(
                    ledger=ledger,
                    experiment_id=experiment_id,
                    staging=staging,
                    status="crash",
                    exc=exc,
                )
            finally:
                self._close_controllers(candidate, incumbent)

    def worker_context(
        self,
        *,
        run_tag: str,
        candidate_path: str | os.PathLike[str] | None = None,
        candidate_commit: str | None = None,
        incumbent_path: str | os.PathLike[str] | None = None,
    ) -> Mapping[str, Any]:
        """Return frozen identities required by an independent Quest worker.

        This is deliberately read-only and contains no seeds or privileged
        simulator state.  The sharding/aggregation command owns registered
        seed allocation and feeds only complete evidence back to the runner.
        """

        manifest = self._load_run(run_tag)
        ledger = ExperimentLedger(self._run_dir(run_tag))
        incumbent_info = self._incumbent_record(ledger)
        _, default_candidate_path = self._candidate_path(manifest)
        selected_candidate = (
            Path(candidate_path).expanduser().resolve()
            if candidate_path is not None
            else default_candidate_path.resolve()
        )
        candidate_digest = sha256_file(selected_candidate)
        if incumbent_path is not None:
            selected_incumbent = Path(incumbent_path).expanduser().resolve()
            incumbent_record = incumbent_info[0] if incumbent_info is not None else None
            incumbent_path = selected_incumbent
            incumbent_digest = sha256_file(selected_incumbent)
        elif incumbent_info is None:
            incumbent_record = None
            incumbent_path = default_candidate_path.resolve()
            incumbent_digest = sha256_file(incumbent_path)
        else:
            incumbent_record, incumbent_source = incumbent_info
            incumbent_path = incumbent_source.resolve()
            incumbent_digest = str(incumbent_record["candidate_sha256"])
        source = manifest["config"]["source"]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_tag": run_tag,
            "run_dir": str(self._run_dir(run_tag)),
            "run_manifest_sha256": hashlib.sha256(
                (self._run_dir(run_tag) / "run.json").read_bytes()
            ).hexdigest(),
            "config_path": str(
                (self.repo_root / str(manifest["config_path"])).resolve()
                if not Path(str(manifest["config_path"])).is_absolute()
                else Path(str(manifest["config_path"])).resolve()
            ),
            "config_repository_path": str(manifest["config_path"]),
            "config_sha256": str(manifest["config_sha256"]),
            "candidate": {
                "path": str(selected_candidate),
                "repository_path": str(
                    manifest["config"]["candidate"]["path"]
                ),
                "sha256": candidate_digest,
                "commit": candidate_commit,
            },
            "incumbent": {
                "experiment_id": (
                    str(incumbent_record["experiment_id"])
                    if incumbent_record is not None
                    else None
                ),
                "path": str(incumbent_path),
                "repository_path": str(
                    manifest["config"]["candidate"]["path"]
                ),
                "sha256": incumbent_digest,
                "commit": (
                    incumbent_record.get("candidate_commit")
                    if incumbent_record is not None
                    else manifest["setup_commit"]
                ),
            },
            "environment_contract_sha256": str(
                manifest["environment_contract_sha256"]
            ),
            "evaluator_sha256": str(manifest["evaluator_sha256"]),
            "source": {
                "checkpoint_path": str(
                    (self.repo_root / str(source["checkpoint_path"])).resolve()
                ),
                "checkpoint_repository_path": str(source["checkpoint_path"]),
                "checkpoint_sha256": str(source["checkpoint_sha256"]),
                "resolved_config_path": str(
                    (self.repo_root / str(source["resolved_config_path"])).resolve()
                ),
                "resolved_config_repository_path": str(
                    source["resolved_config_path"]
                ),
                "resolved_config_sha256": str(
                    source["resolved_config_sha256"]
                ),
            },
        }

    def status(self, *, run_tag: str) -> Mapping[str, Any]:
        """Return a bounded state summary sufficient for the next loop action."""

        manifest = self._load_run(run_tag)
        run_dir = self._run_dir(run_tag)
        ledger = ExperimentLedger(run_dir)
        inspection = ledger.inspect()
        latest = sorted(
            ledger.latest_records().values(),
            key=lambda record: int(record["ledger_sequence"]),
        )
        incumbent_info = self._incumbent_record(ledger)
        incumbent_record = incumbent_info[0] if incumbent_info is not None else None
        budget = self._budget_status(manifest, ledger)
        confirmation_spend = self._confirmation_spend(manifest, ledger)
        source_state = self._source_state(manifest["config"], require=False)
        active = [
            str(record["experiment_id"])
            for record in latest
            if record["status"] in {"planned", "running"}
        ]
        last_normal = next(
            (
                record
                for record in reversed(latest)
                if EXPERIMENT_PATTERN.fullmatch(str(record["experiment_id"]))
            ),
            None,
        )
        baseline_id = str(
            manifest["config"]["candidate"].get(
                "initial_incumbent_id", "legal_fixed_scan_v1"
            )
        )
        baseline = ledger.latest_records().get(baseline_id)
        prepared_external = [
            record
            for record in latest
            if record["status"] == "running"
            and record.get("external_prepared") is True
        ]
        if prepared_external:
            state = "awaiting_external_aggregate"
            next_action = (
                "finalize_external:"
                + str(prepared_external[-1]["experiment_id"])
            )
        elif active:
            state = "recovery_needed"
            next_action = "rerun a command to recover stale evidence"
        elif incumbent_record is None:
            state = "needs_baseline"
            next_action = "baseline"
        elif budget["exhausted"]:
            state = "budget_exhausted"
            next_action = "stop_and_report"
        else:
            state = "ready"
            next_action = "experiment"
        if not all(entry["verified"] for entry in source_state.values()):
            state = "source_blocked"
            next_action = "restore registered checkpoint/config bytes"

        def compact(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
            if record is None:
                return None
            return {
                "experiment_id": record["experiment_id"],
                "status": record["status"],
                "primary_delta": record.get("primary_delta"),
                "decision_reason": record.get("decision_reason"),
                "candidate_sha256": record.get("candidate_sha256"),
                "candidate_commit": record.get("candidate_commit"),
            }

        counts: dict[str, int] = {}
        for record in latest:
            counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "run_tag": run_tag,
            "state": state,
            "next_action": next_action,
            "scope": "engineering_selection",
            "incumbent": compact(incumbent_record),
            "baseline": compact(baseline),
            "last_experiment": compact(last_normal),
            "budget": budget,
            "confirmation": {
                "state": "spent" if confirmation_spend is not None else "unspent",
                "experiment_id": (
                    confirmation_spend.get("experiment_id")
                    if confirmation_spend is not None
                    else None
                ),
                "status": (
                    confirmation_spend.get("status")
                    if confirmation_spend is not None
                    else None
                ),
            },
            "source_ready": all(
                entry["verified"] for entry in source_state.values()
            ),
            "active_experiments": active,
            "ledger": {
                "records": len(inspection.records),
                "issues": len(inspection.issues),
                "terminal_counts": counts,
            },
        }


Runner = AutoresearchRunner


__all__ = [
    "AutoresearchRunner",
    "BudgetExhausted",
    "ConfirmationError",
    "DEFAULT_IMMUTABLE_PATHS",
    "MAX_ABORT_REASON_BYTES",
    "MAX_HYPOTHESIS_BYTES",
    "RunContractError",
    "Runner",
    "RunnerError",
    "SCHEMA_VERSION",
    "SetupError",
    "SourceArtifactError",
]
