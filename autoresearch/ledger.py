"""Durable, append-only state for the autoresearch experiment loop.

The JSONL ledger is the source of truth.  ``incumbent.json`` and
``results.tsv`` are deliberately rebuildable projections of valid ledger
records.  Experiment artifacts are made visible with a same-filesystem atomic
rename before a terminal record is appended.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
ALL_STATUSES = frozenset(
    {"planned", "running", "keep", "discard", "crash", "contract_failure"}
)
TERMINAL_STATUSES = frozenset(
    {"keep", "discard", "crash", "contract_failure"}
)

_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SYSTEM_FIELDS = frozenset(
    {"schema_version", "ledger_sequence", "recorded_at", "record_sha256"}
)
_RECORD_DEFAULTS: Mapping[str, Any] = {
    "parent_incumbent_id": None,
    "candidate_commit": None,
    "candidate_sha256": None,
    "hypothesis": None,
    "predicted_effect": None,
    "changed_paths": [],
    "source_model_sha256": None,
    "resolved_config_sha256": None,
    "evaluator_sha256": None,
    "seed_set_id": None,
    "planned_at": None,
    "started_at": None,
    "completed_at": None,
    "primary_delta": None,
    "paired_counts": {},
    "secondary_metrics": {},
    "checks": {},
    "artifacts": {"directory": None, "files": []},
    "decision_reason": None,
}
_REQUIRED_RECORD_FIELDS = frozenset(
    {"experiment_id", "status", *_RECORD_DEFAULTS.keys(), *_SYSTEM_FIELDS}
)
_TSV_FIELDS = (
    "experiment_id",
    "status",
    "parent_incumbent_id",
    "candidate_commit",
    "candidate_sha256",
    "seed_set_id",
    "primary_delta",
    "paired_counts",
    "secondary_metrics",
    "checks",
    "started_at",
    "completed_at",
    "decision_reason",
    "artifact_directory",
    "record_sha256",
)


class LedgerError(RuntimeError):
    """Base class for durable-ledger failures."""


class InvalidRecordError(LedgerError):
    """A record or requested lifecycle transition is invalid."""


class DuplicateExperimentError(LedgerError):
    """An experiment ID was reused with conflicting evidence."""


class ArtifactError(LedgerError):
    """An artifact directory is unsafe, missing, or conflicts with evidence."""


@dataclass(frozen=True)
class LedgerIssue:
    """A ledger line that was ignored instead of poisoning later evidence."""

    line_number: int
    reason: str


@dataclass(frozen=True)
class LedgerReadResult:
    """Validated records plus quarantined malformed/invalid line reports."""

    records: tuple[Mapping[str, Any], ...]
    issues: tuple[LedgerIssue, ...]


@dataclass(frozen=True)
class RecoveryReport:
    """Actions taken during stale-running startup recovery."""

    crashed: tuple[str, ...]
    resumed: tuple[str, ...]
    active: tuple[str, ...]
    incumbent: Mapping[str, Any] | None


def _utc_timestamp(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidRecordError(f"record is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def _record_digest(record: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write while appending the experiment ledger")
        offset += written


def _discard_interrupted_tail(fd: int) -> None:
    """Drop bytes after the last durable JSONL newline.

    A complete-looking JSON object without its trailing newline is still an
    interrupted append.  Adding a newline later would incorrectly commit that
    record and can duplicate the next ledger sequence.
    """

    end = os.lseek(fd, 0, os.SEEK_END)
    if end == 0:
        return
    os.lseek(fd, end - 1, os.SEEK_SET)
    if os.read(fd, 1) == b"\n":
        return

    position = end
    cutoff = 0
    block_size = 64 * 1024
    while position > 0:
        start = max(0, position - block_size)
        os.lseek(fd, start, os.SEEK_SET)
        block = os.read(fd, position - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            cutoff = start + newline + 1
            break
        position = start
    os.ftruncate(fd, cutoff)
    os.fsync(fd)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        # A failed replace leaves the last complete projection untouched.  The
        # temp file is intentionally best-effort cleanup, never source state.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _without_system_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _SYSTEM_FIELDS}


def _safe_experiment_id(experiment_id: Any) -> str:
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID.fullmatch(
        experiment_id
    ):
        raise InvalidRecordError(
            "experiment_id must contain only letters, digits, '.', '_' or '-'"
        )
    return experiment_id


def _shape_record(record: Mapping[str, Any]) -> dict[str, Any]:
    shaped: dict[str, Any] = {}
    for key, default in _RECORD_DEFAULTS.items():
        # Round-trip mutable defaults to ensure records do not share state.
        shaped[key] = json.loads(json.dumps(default))
    shaped.update(record)
    return shaped


def _requested_fields_match(
    existing: Mapping[str, Any], requested: Mapping[str, Any]
) -> bool:
    for key, value in requested.items():
        if key in _SYSTEM_FIELDS or key == "status":
            continue
        if existing.get(key) != value:
            return False
    return True


class ExperimentLedger:
    """Manage one autoresearch run's append-only experiment evidence.

    All mutation methods lock ``experiments.jsonl``.  The lock is held while
    terminal projections are rebuilt, so concurrent finalizations cannot move
    ``incumbent.json`` backwards.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self.ledger_path = self.run_dir / "experiments.jsonl"
        self.results_path = self.run_dir / "results.tsv"
        self.incumbent_path = self.run_dir / "incumbent.json"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.touch(exist_ok=True)

    @contextmanager
    def _locked_ledger(self, *, exclusive: bool) -> Iterator[int]:
        fd = os.open(self.ledger_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _transition_error(
        previous: Mapping[str, Any] | None, current: Mapping[str, Any]
    ) -> str | None:
        status = current.get("status")
        if previous is None:
            if status != "planned":
                return "the first record for an experiment must be planned"
            return None

        previous_status = previous["status"]
        if previous_status in TERMINAL_STATUSES:
            return f"terminal status {previous_status!r} cannot transition"
        if previous_status == "planned":
            if status not in {"running", "crash", "contract_failure"}:
                return f"planned cannot transition to {status!r}"
            return None
        if previous_status == "running":
            if status in TERMINAL_STATUSES:
                return None
            if status == "running":
                if current.get("idempotent_resume") is not True:
                    return "running -> running requires an idempotent resume"
                if current.get("resume_of_record_sha256") != previous.get(
                    "record_sha256"
                ):
                    return "idempotent resume does not reference latest running record"
                return None
            return f"running cannot transition to {status!r}"
        return f"unknown prior status {previous_status!r}"

    @classmethod
    def _validate_stored_record(cls, record: Any) -> str | None:
        if not isinstance(record, dict):
            return "JSON value is not an object"
        missing = _REQUIRED_RECORD_FIELDS.difference(record)
        if missing:
            return f"missing required fields: {', '.join(sorted(missing))}"
        try:
            _safe_experiment_id(record.get("experiment_id"))
        except InvalidRecordError as exc:
            return str(exc)
        if record.get("schema_version") != SCHEMA_VERSION:
            return "unsupported schema_version"
        if record.get("status") not in ALL_STATUSES:
            return "unknown status"
        sequence = record.get("ledger_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            return "ledger_sequence must be a positive integer"
        try:
            _parse_timestamp(record.get("recorded_at"))
        except (TypeError, ValueError):
            return "invalid recorded_at timestamp"
        supplied_digest = record.get("record_sha256")
        try:
            expected_digest = _record_digest(record)
        except InvalidRecordError:
            return "record contains a non-canonical JSON value"
        if not isinstance(supplied_digest, str) or not hmac.compare_digest(
            supplied_digest, expected_digest
        ):
            return "record_sha256 mismatch"
        return None

    @classmethod
    def _read_fd(cls, fd: int) -> LedgerReadResult:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)

        records: list[Mapping[str, Any]] = []
        issues: list[LedgerIssue] = []
        latest: dict[str, Mapping[str, Any]] = {}
        last_sequence = 0
        seen_digests: set[str] = set()
        for line_number, raw_line in enumerate(payload.splitlines(keepends=True), 1):
            if not raw_line.endswith(b"\n"):
                issues.append(LedgerIssue(line_number, "interrupted JSONL tail"))
                continue
            stripped = raw_line[:-1].strip()
            if not stripped:
                continue
            try:
                decoded = stripped.decode("utf-8")
                record = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                issues.append(LedgerIssue(line_number, "invalid JSON"))
                continue
            error = cls._validate_stored_record(record)
            if error:
                issues.append(LedgerIssue(line_number, error))
                continue
            sequence = record["ledger_sequence"]
            if sequence <= last_sequence:
                issues.append(
                    LedgerIssue(line_number, "ledger_sequence is not strictly increasing")
                )
                continue
            digest = record["record_sha256"]
            if digest in seen_digests:
                issues.append(LedgerIssue(line_number, "duplicate record_sha256"))
                continue
            previous = latest.get(record["experiment_id"])
            error = cls._transition_error(previous, record)
            if error:
                issues.append(LedgerIssue(line_number, error))
                continue
            records.append(record)
            latest[record["experiment_id"]] = record
            last_sequence = sequence
            seen_digests.add(digest)
        return LedgerReadResult(tuple(records), tuple(issues))

    def inspect(self) -> LedgerReadResult:
        """Return validated records and non-fatal invalid-line diagnostics."""

        with self._locked_ledger(exclusive=False) as fd:
            return self._read_fd(fd)

    def read_records(self) -> list[Mapping[str, Any]]:
        """Return only valid records, in append order."""

        return list(self.inspect().records)

    def latest_records(self) -> dict[str, Mapping[str, Any]]:
        """Return the latest valid lifecycle record for every experiment."""

        latest: dict[str, Mapping[str, Any]] = {}
        for record in self.read_records():
            latest[record["experiment_id"]] = record
        return latest

    @staticmethod
    def _latest(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        latest: dict[str, Mapping[str, Any]] = {}
        for record in records:
            latest[record["experiment_id"]] = record
        return latest

    def _append_locked(
        self,
        fd: int,
        record: Mapping[str, Any],
        current_records: Sequence[Mapping[str, Any]],
        *,
        recorded_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        candidate = _shape_record(_without_system_fields(record))
        candidate["experiment_id"] = _safe_experiment_id(candidate.get("experiment_id"))
        status = candidate.get("status")
        if status not in ALL_STATUSES:
            raise InvalidRecordError(f"unknown experiment status: {status!r}")

        previous = self._latest(current_records).get(candidate["experiment_id"])
        transition_error = self._transition_error(previous, candidate)
        if transition_error:
            raise InvalidRecordError(transition_error)

        max_sequence = max(
            (int(existing["ledger_sequence"]) for existing in current_records),
            default=0,
        )
        candidate["schema_version"] = SCHEMA_VERSION
        candidate["ledger_sequence"] = max_sequence + 1
        candidate["recorded_at"] = _utc_timestamp(recorded_at)
        candidate["record_sha256"] = _record_digest(candidate)
        encoded = _canonical_json_bytes(candidate) + b"\n"

        _discard_interrupted_tail(fd)
        os.lseek(fd, 0, os.SEEK_END)
        _write_all(fd, encoded)
        os.fsync(fd)
        _fsync_directory(self.run_dir)
        return candidate

    @staticmethod
    def _payload(
        experiment: str | Mapping[str, Any], fields: Mapping[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(experiment, Mapping):
            payload = dict(experiment)
            if fields:
                payload.update(fields)
            experiment_id = _safe_experiment_id(payload.get("experiment_id"))
        else:
            experiment_id = _safe_experiment_id(experiment)
            payload = dict(fields or {})
            payload["experiment_id"] = experiment_id
        forbidden = _SYSTEM_FIELDS.intersection(payload)
        if forbidden:
            raise InvalidRecordError(
                f"caller cannot set ledger fields: {', '.join(sorted(forbidden))}"
            )
        return experiment_id, payload

    def plan_experiment(
        self,
        experiment: str | Mapping[str, Any],
        *,
        fields: Mapping[str, Any] | None = None,
        planned_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Append a plan, or return the identical existing plan on retry."""

        experiment_id, requested = self._payload(experiment, fields)
        if requested.get("status") not in {None, "planned"}:
            raise InvalidRecordError("plan_experiment status must be 'planned'")
        requested["status"] = "planned"
        requested.setdefault("planned_at", _utc_timestamp(planned_at))
        with self._locked_ledger(exclusive=True) as fd:
            read = self._read_fd(fd)
            matching = [
                record
                for record in read.records
                if record["experiment_id"] == experiment_id
                and record["status"] == "planned"
            ]
            if matching:
                existing = matching[0]
                comparison = {
                    key: value
                    for key, value in requested.items()
                    if key != "planned_at" or planned_at is not None
                }
                if _requested_fields_match(existing, comparison):
                    return existing
                raise DuplicateExperimentError(
                    f"experiment {experiment_id} already has a conflicting plan"
                )
            return self._append_locked(
                fd, requested, read.records, recorded_at=planned_at
            )

    def start_experiment(
        self,
        experiment_id: str,
        *,
        fields: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Transition a planned experiment to running, idempotently."""

        experiment_id = _safe_experiment_id(experiment_id)
        requested = dict(fields or {})
        if _SYSTEM_FIELDS.intersection(requested):
            raise InvalidRecordError("caller cannot set ledger-managed fields")
        if requested.get("status") not in {None, "running"}:
            raise InvalidRecordError("start_experiment status must be 'running'")
        with self._locked_ledger(exclusive=True) as fd:
            read = self._read_fd(fd)
            latest = self._latest(read.records).get(experiment_id)
            if latest is None:
                raise InvalidRecordError(f"experiment {experiment_id} is not planned")
            if latest["status"] == "running":
                comparison = dict(requested)
                if started_at is not None:
                    comparison["started_at"] = _utc_timestamp(started_at)
                if _requested_fields_match(latest, comparison):
                    return latest
                raise DuplicateExperimentError(
                    f"experiment {experiment_id} is already running with other fields"
                )
            if latest["status"] in TERMINAL_STATUSES:
                raise InvalidRecordError(f"experiment {experiment_id} is already terminal")
            payload = _without_system_fields(latest)
            payload.update(requested)
            payload["status"] = "running"
            payload["started_at"] = _utc_timestamp(started_at)
            return self._append_locked(
                fd, payload, read.records, recorded_at=started_at
            )

    def artifact_staging_path(self, experiment_id: str) -> Path:
        experiment_id = _safe_experiment_id(experiment_id)
        return self.artifacts_dir / f".{experiment_id}.tmp"

    def artifact_path(self, experiment_id: str) -> Path:
        experiment_id = _safe_experiment_id(experiment_id)
        return self.artifacts_dir / experiment_id

    def begin_artifacts(self, experiment_id: str, *, resume: bool = False) -> Path:
        """Create (or explicitly reopen) the private artifact staging directory."""

        experiment_id = _safe_experiment_id(experiment_id)
        staging = self.artifact_staging_path(experiment_id)
        final = self.artifact_path(experiment_id)
        with self._locked_ledger(exclusive=True) as fd:
            latest = self._latest(self._read_fd(fd).records).get(experiment_id)
            if latest is None or latest["status"] not in {"planned", "running"}:
                raise ArtifactError("artifacts require a planned or running experiment")
            if final.exists():
                raise ArtifactError(
                    f"final artifacts already exist for {experiment_id}; finalize the "
                    "idempotent ledger retry without reopening them"
                )
            if staging.exists():
                if not staging.is_dir() or staging.is_symlink():
                    raise ArtifactError(f"unsafe staging path for {experiment_id}")
                if not resume:
                    raise ArtifactError(
                        f"staging artifacts already exist for {experiment_id}; use resume=True"
                    )
                return staging
            staging.mkdir(mode=0o755)
            _fsync_directory(self.artifacts_dir)
            return staging

    @staticmethod
    def _fsync_artifact_tree(root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise ArtifactError(f"artifact staging path is not a real directory: {root}")
        directories: list[Path] = [root]
        for directory, names, filenames in os.walk(root):
            directory_path = Path(directory)
            for name in names:
                child = directory_path / name
                if child.is_symlink() or not child.is_dir():
                    raise ArtifactError(f"unsafe artifact directory: {child}")
                directories.append(child)
            for name in filenames:
                child = directory_path / name
                child_stat = child.lstat()
                if not stat.S_ISREG(child_stat.st_mode):
                    raise ArtifactError(f"artifact is not a regular file: {child}")
                descriptor = os.open(child, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _fsync_directory(directory)

    def _artifact_manifest(self, experiment_id: str, root: Path) -> Mapping[str, Any]:
        files: list[Mapping[str, Any]] = []
        for path in sorted(root.rglob("*")):
            path_stat = path.lstat()
            if stat.S_ISDIR(path_stat.st_mode):
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                raise ArtifactError(f"artifact is not a regular file: {path}")
            digest, size = _sha256_file(path)
            files.append(
                {
                    "path": path.relative_to(self.run_dir).as_posix(),
                    "sha256": digest,
                    "size": size,
                }
            )
        return {
            "directory": self.artifact_path(experiment_id)
            .relative_to(self.run_dir)
            .as_posix(),
            "files": files,
        }

    def _finalize_artifacts_locked(
        self, experiment_id: str, artifact_staging: str | os.PathLike[str] | None
    ) -> Mapping[str, Any]:
        expected_staging = self.artifact_staging_path(experiment_id)
        final = self.artifact_path(experiment_id)
        if artifact_staging is not None:
            supplied = Path(artifact_staging)
            try:
                matches = supplied.resolve(strict=False) == expected_staging.resolve(
                    strict=False
                )
            except OSError as exc:
                raise ArtifactError(f"cannot resolve artifact staging path: {exc}") from exc
            if not matches:
                raise ArtifactError(
                    f"artifact staging must be {expected_staging}, got {supplied}"
                )

        if final.exists():
            if final.is_symlink() or not final.is_dir():
                raise ArtifactError(f"unsafe final artifact path: {final}")
            if expected_staging.exists():
                raise ArtifactError(
                    f"both final and staging artifacts exist for {experiment_id}"
                )
            self._fsync_artifact_tree(final)
            return self._artifact_manifest(experiment_id, final)

        staging = expected_staging
        if not staging.exists():
            staging.mkdir(mode=0o755)
            _fsync_directory(self.artifacts_dir)
        self._fsync_artifact_tree(staging)
        os.replace(staging, final)
        _fsync_directory(self.artifacts_dir)
        return self._artifact_manifest(experiment_id, final)

    def _verify_artifacts(self, record: Mapping[str, Any]) -> bool:
        experiment_id = record["experiment_id"]
        expected_directory = self.artifact_path(experiment_id)
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return False
        if artifacts.get("directory") != expected_directory.relative_to(
            self.run_dir
        ).as_posix():
            return False
        if not expected_directory.is_dir() or expected_directory.is_symlink():
            return False
        try:
            actual = self._artifact_manifest(experiment_id, expected_directory)
        except (ArtifactError, OSError):
            return False
        return actual == artifacts

    @staticmethod
    def _incumbent_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "incumbent_id": record["experiment_id"],
            "experiment_id": record["experiment_id"],
            "candidate_commit": record.get("candidate_commit"),
            "candidate_sha256": record.get("candidate_sha256"),
            "parent_incumbent_id": record.get("parent_incumbent_id"),
            "updated_at": record.get("completed_at") or record["recorded_at"],
            "source_record_sha256": record["record_sha256"],
        }

    def _last_valid_keep(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        for record in reversed(records):
            if record["status"] == "keep" and self._verify_artifacts(record):
                return record
        return None

    def _recover_incumbent_locked(
        self, records: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        keep = self._last_valid_keep(records)
        if keep is None:
            # ``incumbent.json`` is only a projection.  Leaving an older copy
            # in place after every keep artifact becomes invalid would make a
            # caller act on evidence the ledger can no longer verify.
            try:
                self.incumbent_path.unlink()
            except FileNotFoundError:
                pass
            else:
                _fsync_directory(self.run_dir)
            return None
        expected = self._incumbent_payload(keep)
        try:
            current = json.loads(self.incumbent_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            current = None
        if current != expected:
            _atomic_write(
                self.incumbent_path,
                _canonical_json_bytes(expected) + b"\n",
            )
        return expected

    def recover_incumbent(self) -> Mapping[str, Any] | None:
        """Atomically rebuild the incumbent from the last valid keep record."""

        with self._locked_ledger(exclusive=True) as fd:
            return self._recover_incumbent_locked(self._read_fd(fd).records)

    @staticmethod
    def _tsv_cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        return str(value)

    def _regenerate_results_locked(
        self, records: Sequence[Mapping[str, Any]]
    ) -> None:
        latest = sorted(
            self._latest(records).values(), key=lambda record: record["ledger_sequence"]
        )
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=_TSV_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in latest:
            row = {field: record.get(field) for field in _TSV_FIELDS}
            row["artifact_directory"] = (
                record.get("artifacts", {}).get("directory")
                if isinstance(record.get("artifacts"), Mapping)
                else None
            )
            writer.writerow({key: self._tsv_cell(value) for key, value in row.items()})
        _atomic_write(self.results_path, output.getvalue().encode("utf-8"))

    def regenerate_results_tsv(self) -> Path:
        """Atomically rebuild the human view solely from valid JSONL records."""

        with self._locked_ledger(exclusive=True) as fd:
            self._regenerate_results_locked(self._read_fd(fd).records)
        return self.results_path

    def finalize_experiment(
        self,
        experiment_id: str,
        *,
        status: str,
        fields: Mapping[str, Any] | None = None,
        artifact_staging: str | os.PathLike[str] | None = None,
        completed_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Publish artifacts, append one terminal record, then repair projections.

        Retrying the same call after interruption is safe whether the
        interruption happened before the artifact rename, after the rename, or
        after the ledger append.  Conflicting retries are rejected.
        """

        experiment_id = _safe_experiment_id(experiment_id)
        if status == "pass":
            raise InvalidRecordError("use canonical terminal status 'keep', not 'pass'")
        if status not in TERMINAL_STATUSES:
            raise InvalidRecordError(f"not a terminal status: {status!r}")
        requested = dict(fields or {})
        if _SYSTEM_FIELDS.intersection(requested):
            raise InvalidRecordError("caller cannot set ledger-managed fields")
        if "artifacts" in requested:
            raise InvalidRecordError("artifact evidence is generated by the ledger")
        if requested.get("status") not in {None, status}:
            raise InvalidRecordError("conflicting terminal status in fields")

        with self._locked_ledger(exclusive=True) as fd:
            read = self._read_fd(fd)
            records = list(read.records)
            latest = self._latest(records).get(experiment_id)
            if latest is None:
                raise InvalidRecordError(f"experiment {experiment_id} is not planned")
            if latest["status"] in TERMINAL_STATUSES:
                comparison = dict(requested)
                if completed_at is not None:
                    comparison["completed_at"] = _utc_timestamp(completed_at)
                if latest["status"] != status or not _requested_fields_match(
                    latest, comparison
                ):
                    raise DuplicateExperimentError(
                        f"experiment {experiment_id} already has conflicting terminal evidence"
                    )
                if not self._verify_artifacts(latest):
                    raise ArtifactError(
                        f"terminal artifacts no longer match {experiment_id} evidence"
                    )
                incumbent = self._recover_incumbent_locked(records)
                self._regenerate_results_locked(records)
                del incumbent
                return latest

            payload = _without_system_fields(latest)
            payload.update(requested)
            payload["experiment_id"] = experiment_id
            payload["status"] = status
            payload["completed_at"] = _utc_timestamp(completed_at)
            transition_error = self._transition_error(latest, payload)
            if transition_error:
                raise InvalidRecordError(transition_error)
            payload["artifacts"] = self._finalize_artifacts_locked(
                experiment_id, artifact_staging
            )
            terminal = self._append_locked(
                fd, payload, records, recorded_at=completed_at
            )
            records.append(terminal)
            self._recover_incumbent_locked(records)
            self._regenerate_results_locked(records)
            return terminal

    def resume_experiment(
        self,
        experiment_id: str,
        *,
        evaluation_is_idempotent: bool,
        resumed_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Record an explicit retry of a running idempotent evaluation."""

        if evaluation_is_idempotent is not True:
            raise InvalidRecordError("only an explicitly idempotent evaluation may resume")
        experiment_id = _safe_experiment_id(experiment_id)
        with self._locked_ledger(exclusive=True) as fd:
            read = self._read_fd(fd)
            records = list(read.records)
            latest = self._latest(records).get(experiment_id)
            if latest is None or latest["status"] != "running":
                raise InvalidRecordError(f"experiment {experiment_id} is not running")
            payload = _without_system_fields(latest)
            payload.update(
                {
                    "status": "running",
                    "idempotent_resume": True,
                    "resume_count": int(latest.get("resume_count", 0)) + 1,
                    "resume_of_record_sha256": latest["record_sha256"],
                    "last_resumed_at": _utc_timestamp(resumed_at),
                }
            )
            resumed = self._append_locked(
                fd, payload, records, recorded_at=resumed_at
            )
            records.append(resumed)
            self._regenerate_results_locked(records)
            return resumed

    def recover_stale_running(
        self,
        stale_after_seconds: float,
        *,
        idempotent_resume_ids: Iterable[str] = (),
        now: datetime | None = None,
    ) -> RecoveryReport:
        """Resume explicitly idempotent stale runs and crash every other stale run.

        Staged logs are retained: crash recovery fsyncs and publishes their
        directory before appending the crash record.  Calling this method again
        with the same clock does not add duplicate evidence.
        """

        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be non-negative")
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        current_time = current_time.astimezone(timezone.utc)
        resume_ids = {_safe_experiment_id(item) for item in idempotent_resume_ids}
        crashed: list[str] = []
        resumed: list[str] = []
        active: list[str] = []

        with self._locked_ledger(exclusive=True) as fd:
            read = self._read_fd(fd)
            records = list(read.records)
            snapshot = sorted(
                self._latest(records).values(),
                key=lambda record: record["ledger_sequence"],
            )
            for latest in snapshot:
                if latest["status"] != "running":
                    continue
                try:
                    last_activity = _parse_timestamp(latest["recorded_at"])
                except ValueError:
                    last_activity = _parse_timestamp(latest["started_at"])
                age = (current_time - last_activity).total_seconds()
                experiment_id = latest["experiment_id"]
                if age < stale_after_seconds:
                    active.append(experiment_id)
                    continue
                if experiment_id in resume_ids:
                    payload = _without_system_fields(latest)
                    payload.update(
                        {
                            "status": "running",
                            "idempotent_resume": True,
                            "resume_count": int(latest.get("resume_count", 0)) + 1,
                            "resume_of_record_sha256": latest["record_sha256"],
                            "last_resumed_at": _utc_timestamp(current_time),
                            "recovery_action": "idempotent_resume",
                        }
                    )
                    event = self._append_locked(
                        fd, payload, records, recorded_at=current_time
                    )
                    records.append(event)
                    resumed.append(experiment_id)
                    continue

                payload = _without_system_fields(latest)
                payload.update(
                    {
                        "status": "crash",
                        "completed_at": _utc_timestamp(current_time),
                        "decision_reason": (
                            "stale running record recovered as crash after "
                            f"{age:.3f} seconds"
                        ),
                        "recovery_action": "mark_crash",
                        "artifacts": self._finalize_artifacts_locked(
                            experiment_id, None
                        ),
                    }
                )
                event = self._append_locked(
                    fd, payload, records, recorded_at=current_time
                )
                records.append(event)
                crashed.append(experiment_id)

            incumbent = self._recover_incumbent_locked(records)
            self._regenerate_results_locked(records)
        return RecoveryReport(
            crashed=tuple(crashed),
            resumed=tuple(resumed),
            active=tuple(active),
            incumbent=incumbent,
        )

    # The startup spelling makes runner intent explicit while keeping one
    # implementation for tests and command-line callers.
    recover_on_startup = recover_stale_running


__all__ = [
    "ALL_STATUSES",
    "TERMINAL_STATUSES",
    "ArtifactError",
    "DuplicateExperimentError",
    "ExperimentLedger",
    "InvalidRecordError",
    "LedgerError",
    "LedgerIssue",
    "LedgerReadResult",
    "RecoveryReport",
]
