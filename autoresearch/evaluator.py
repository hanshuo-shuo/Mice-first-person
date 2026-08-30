"""Frozen paired evaluator for the phase-1 autoresearch gaze search.

This module is trusted evaluation code.  Candidate controllers can replace
only the third component of a frozen SAC action and receive only defensive
copies of the public first-person observation contract.  The simulator,
checkpoint, locomotion action, reward, and termination semantics remain owned
by the ordinary environment and frozen model.

The generic evaluator is intentionally expressed in terms of a tiny episode
adapter.  :class:`FakeEpisodeFactory` makes contract and orchestration tests
cheap; :class:`RealExp05EpisodeFactory` is the production adapter and refuses
to load a missing or unverified EXP-05 checkpoint.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import hmac
import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


EXPECTED_EXP05_CHECKPOINT_SHA256 = (
    "7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec"
)
DEFAULT_EXP05_ROOT = Path("results/sac/sac_cnn_active_gaze_9903898")
DEFAULT_EXP05_CHECKPOINT = DEFAULT_EXP05_ROOT / "checkpoints/final_model.zip"
DEFAULT_EXP05_RESOLVED_CONFIG = DEFAULT_EXP05_ROOT / "resolved_config.yaml"

PUBLIC_OBSERVATION_FIELDS = (
    "image_left",
    "image_right",
    "proprio",
    "previous_action",
)
RATE_METHODS = ("candidate", "incumbent")
REFERENCE_METHOD = "fixed_p60"
EVALUATION_METHODS = (*RATE_METHODS, REFERENCE_METHOD)
COMPARATOR_METHODS = ("incumbent", REFERENCE_METHOD)
CACHE_SCHEMA_VERSION = 1
CONFIG_METHOD_ALIASES = {
    "candidate": "candidate",
    "incumbent": "incumbent",
    "search_incumbent": "incumbent",
    "fixed_p60": REFERENCE_METHOD,
    "fixed_p60_research_reference": REFERENCE_METHOD,
}

# Only these aggregate/episode values may leave the trusted adapter.  In
# particular, coordinates, exact state, geometric LOS, and transition info are
# never copied into records returned to the runner.
EPISODE_RESULT_FIELDS = (
    "clean_success",
    "capture_episode",
    "goal_reached",
    "steps",
    "minimum_predator_distance",
    "path_cost",
    "gaze_travel_degrees",
    "predator_pixels_visible_fraction",
)


class EvaluationError(RuntimeError):
    """Base class for trusted-evaluator failures."""


class ArtifactVerificationError(EvaluationError):
    """A frozen source artifact is missing or differs from its registered hash."""


class EpisodeContractError(EvaluationError):
    """An episode adapter, base model, or candidate violated the frozen contract."""


class DeterminismError(EvaluationError):
    """Repeated evaluation produced different public records or action traces."""


class CacheValidationError(EvaluationError):
    """Comparator records are incomplete, duplicated, reordered, or malformed."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one regular file without following directory input."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ArtifactVerificationError(f"Required artifact is missing: {resolved}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactVerificationError(f"Cannot read required artifact: {resolved}") from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def ordered_seed_sha256(seeds: Sequence[int]) -> str:
    return canonical_sha256([int(seed) for seed in seeds])


def evaluator_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def _validate_nonempty_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise ValueError("The frozen evaluator requires at least one episode seed")
    if any(seed < 0 for seed in values):
        raise ValueError("Episode seeds must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError("Ordered episode seeds must not contain duplicates")
    return values


@dataclasses.dataclass(frozen=True)
class ArtifactBundle:
    """Verified real source artifacts and their content identities."""

    checkpoint_path: Path
    resolved_config_path: Path
    checkpoint_sha256: str
    resolved_config_sha256: str


@dataclasses.dataclass(frozen=True)
class FrozenSeedSet:
    """One registered, ordered seed set from ``gaze_dev.yaml``."""

    name: str
    seed_set_id: str
    seeds: tuple[int, ...]
    purpose: str
    one_time: bool
    requires_explicit_authorization: bool


def seed_set_from_config(
    config: Mapping[str, Any],
    name: str,
) -> FrozenSeedSet:
    """Resolve a registered contiguous seed range without accepting overrides."""

    try:
        raw = config["seed_sets"][str(name)]
    except (KeyError, TypeError) as exc:
        raise EvaluationError(f"Unknown frozen seed set: {name!r}") from exc
    if not isinstance(raw, Mapping):
        raise EvaluationError(f"Seed set {name!r} must be a mapping")
    seed_start = int(raw.get("seed_start", -1))
    episodes = int(raw.get("episodes", 0))
    seed_set_id = str(raw.get("id", "")).strip()
    if seed_start < 0 or episodes <= 0 or not seed_set_id:
        raise EvaluationError(f"Seed set {name!r} is incomplete")
    seeds = _validate_nonempty_seeds(range(seed_start, seed_start + episodes))
    return FrozenSeedSet(
        name=str(name),
        seed_set_id=seed_set_id,
        seeds=seeds,
        purpose=str(raw.get("purpose", "")),
        one_time=bool(raw.get("one_time", False)),
        requires_explicit_authorization=bool(
            raw.get("requires_explicit_authorization", False),
        ),
    )


def _validate_autoresearch_config_contract(config: Mapping[str, Any]) -> None:
    candidate = config.get("candidate")
    evaluation = config.get("evaluation")
    source = config.get("source")
    decision = config.get("decision")
    if not all(isinstance(value, Mapping) for value in (candidate, evaluation, source, decision)):
        raise EvaluationError(
            "Autoresearch config requires candidate, evaluation, source, and decision mappings",
        )
    fields = tuple(str(field) for field in candidate.get("public_observation_fields", ()))
    if fields != PUBLIC_OBSERVATION_FIELDS:
        raise EvaluationError("Configured public observation fields violate the frozen contract")
    configured_methods = tuple(
        CONFIG_METHOD_ALIASES.get(str(method), "")
        for method in evaluation.get("methods", ())
    )
    if configured_methods != EVALUATION_METHODS:
        raise EvaluationError("Configured evaluation methods violate the frozen paired design")
    history_limit = int(evaluation.get("maximum_history_length", -1))
    candidate_history_limit = int(candidate.get("public_history_length", -1))
    if history_limit < 0 or candidate_history_limit != history_limit:
        raise EvaluationError("Candidate and evaluator public-history limits must match")
    if evaluation.get("deterministic_policy") is not True:
        raise EvaluationError("The phase-1 evaluator requires deterministic SAC inference")
    if str(decision.get("primary_metric", "")) != "paired_clean_success_delta":
        raise EvaluationError("The phase-1 primary metric must remain paired clean success")
    if decision.get("capture_rate_must_not_exceed_incumbent") is not True:
        raise EvaluationError("The phase-1 capture-rate hard gate must remain enabled")
    if decision.get("ties_keep_incumbent") is not True:
        raise EvaluationError("Phase-1 ties must keep the incumbent")


def verify_exp05_artifacts(
    checkpoint_path: str | Path = DEFAULT_EXP05_CHECKPOINT,
    resolved_config_path: str | Path = DEFAULT_EXP05_RESOLVED_CONFIG,
    *,
    expected_checkpoint_sha256: str = EXPECTED_EXP05_CHECKPOINT_SHA256,
    expected_resolved_config_sha256: str | None = None,
) -> ArtifactBundle:
    """Verify frozen artifacts before importing or asking SAC to deserialize them.

    The model digest is fixed by the registered EXP-05 report.  A run should
    also pass the resolved-config digest frozen in ``run.json``; the optional
    argument exists so setup can compute and freeze that digest on first use.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    config = Path(resolved_config_path).expanduser().resolve()
    checkpoint_digest = file_sha256(checkpoint)
    if not hmac.compare_digest(checkpoint_digest, str(expected_checkpoint_sha256)):
        raise ArtifactVerificationError(
            "EXP-05 checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_digest}",
        )
    config_digest = file_sha256(config)
    if expected_resolved_config_sha256 is not None and not hmac.compare_digest(
        config_digest,
        str(expected_resolved_config_sha256),
    ):
        raise ArtifactVerificationError(
            "EXP-05 resolved config SHA-256 mismatch: "
            f"expected {expected_resolved_config_sha256}, got {config_digest}",
        )
    return ArtifactBundle(
        checkpoint_path=checkpoint,
        resolved_config_path=config,
        checkpoint_sha256=checkpoint_digest,
        resolved_config_sha256=config_digest,
    )


@dataclasses.dataclass(frozen=True)
class ComparatorCacheIdentity:
    """Every source identity that can change frozen comparator outcomes."""

    checkpoint_sha256: str
    resolved_config_sha256: str
    evaluator_sha256: str
    seed_set_id: str
    ordered_seed_sha256: str
    environment_contract_sha256: str
    incumbent_sha256: str
    max_horizon: int
    public_history_limit: int

    @classmethod
    def from_seeds(
        cls,
        *,
        checkpoint_sha256: str,
        resolved_config_sha256: str,
        evaluator_sha256: str,
        seed_set_id: str,
        seeds: Sequence[int],
        environment_contract_sha256: str,
        incumbent_sha256: str,
        max_horizon: int,
        public_history_limit: int,
    ) -> "ComparatorCacheIdentity":
        frozen_seeds = _validate_nonempty_seeds(seeds)
        if int(max_horizon) <= 0:
            raise ValueError("max_horizon must be positive")
        if int(public_history_limit) < 0:
            raise ValueError("public_history_limit must be non-negative")
        values = {
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "evaluator_sha256": evaluator_sha256,
            "seed_set_id": seed_set_id,
            "environment_contract_sha256": environment_contract_sha256,
            "incumbent_sha256": incumbent_sha256,
        }
        if any(not str(value).strip() for value in values.values()):
            raise ValueError("Comparator cache identity fields must be non-empty")
        return cls(
            checkpoint_sha256=str(checkpoint_sha256),
            resolved_config_sha256=str(resolved_config_sha256),
            evaluator_sha256=str(evaluator_sha256),
            seed_set_id=str(seed_set_id),
            ordered_seed_sha256=ordered_seed_sha256(frozen_seeds),
            environment_contract_sha256=str(environment_contract_sha256),
            incumbent_sha256=str(incumbent_sha256),
            max_horizon=int(max_horizon),
            public_history_limit=int(public_history_limit),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            **dataclasses.asdict(self),
        }

    @property
    def key(self) -> str:
        return canonical_sha256(self.payload())


def comparator_cache_key(**kwargs: Any) -> str:
    """Convenience wrapper used by setup code and cache-invalidation tests."""

    return ComparatorCacheIdentity.from_seeds(**kwargs).key


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_outcome(
    outcome: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    action_trace_sha256: str,
) -> dict[str, Any]:
    if not isinstance(outcome, Mapping):
        raise EpisodeContractError("Episode finalization must return a mapping")
    missing = [field for field in EPISODE_RESULT_FIELDS if field not in outcome]
    if missing:
        raise EpisodeContractError(f"Episode result is missing fields: {missing}")

    record = {
        "method": str(method),
        "seed": int(seed),
        **{field: _json_scalar(outcome[field]) for field in EPISODE_RESULT_FIELDS},
        "action_trace_sha256": str(action_trace_sha256),
    }
    for field in ("clean_success", "capture_episode", "goal_reached"):
        record[field] = bool(record[field])
    record["steps"] = int(record["steps"])
    if record["steps"] < 0:
        raise EpisodeContractError("Episode steps must be non-negative")
    for field in (
        "minimum_predator_distance",
        "path_cost",
        "gaze_travel_degrees",
        "predator_pixels_visible_fraction",
    ):
        value = float(record[field])
        if not math.isfinite(value):
            raise EpisodeContractError(f"Episode result {field} must be finite")
        record[field] = value
    if not 0.0 <= record["predator_pixels_visible_fraction"] <= 1.0:
        raise EpisodeContractError(
            "predator_pixels_visible_fraction must lie in [0, 1]",
        )
    return record


def _expected_record_order(
    seeds: Sequence[int],
    methods: Sequence[str],
) -> list[tuple[int, str]]:
    return [(int(seed), str(method)) for seed in seeds for method in methods]


def validate_episode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    methods: Sequence[str],
) -> None:
    """Reject missing, duplicated, reordered, partial, and non-finite records."""

    frozen_seeds = _validate_nonempty_seeds(seeds)
    expected = _expected_record_order(frozen_seeds, methods)
    actual: list[tuple[int, str]] = []
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise CacheValidationError("Every episode record must be a mapping")
        required = {
            "method",
            "seed",
            "action_trace_sha256",
            *EPISODE_RESULT_FIELDS,
        }
        missing = required.difference(raw_record)
        if missing:
            raise CacheValidationError(
                f"Episode record is partial; missing {sorted(missing)}",
            )
        actual.append((int(raw_record["seed"]), str(raw_record["method"])))
        if int(raw_record["steps"]) < 0:
            raise CacheValidationError("Episode steps must be non-negative")
        for field in (
            "minimum_predator_distance",
            "path_cost",
            "gaze_travel_degrees",
            "predator_pixels_visible_fraction",
        ):
            if not math.isfinite(float(raw_record[field])):
                raise CacheValidationError(f"Episode field {field} must be finite")
    if actual != expected:
        raise CacheValidationError(
            "Episode records are incomplete, duplicated, or outside frozen order",
        )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(payload))
            stream.write("\n")
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


def write_comparator_cache(
    path: str | Path,
    *,
    identity: ComparatorCacheIdentity,
    seeds: Sequence[int],
    records: Sequence[Mapping[str, Any]],
    determinism_verified: bool,
) -> None:
    if not determinism_verified:
        raise CacheValidationError("Unverified comparator records must not be cached")
    validate_episode_records(records, seeds=seeds, methods=COMPARATOR_METHODS)
    plain_records = [dict(record) for record in records]
    envelope = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": identity.key,
        "identity": identity.payload(),
        "determinism_verified": True,
        "records_sha256": canonical_sha256(plain_records),
        "records": plain_records,
    }
    _atomic_write_json(Path(path), envelope)


def load_comparator_cache(
    path: str | Path,
    *,
    identity: ComparatorCacheIdentity,
    seeds: Sequence[int],
) -> list[dict[str, Any]] | None:
    """Return a verified cache hit or ``None`` for any stale/corrupt input."""

    cache_path = Path(path)
    try:
        envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(envelope, Mapping):
            return None
        if envelope.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if not hmac.compare_digest(str(envelope.get("cache_key", "")), identity.key):
            return None
        if envelope.get("identity") != identity.payload():
            return None
        if envelope.get("determinism_verified") is not True:
            return None
        records = envelope.get("records")
        if not isinstance(records, list):
            return None
        if not hmac.compare_digest(
            str(envelope.get("records_sha256", "")),
            canonical_sha256(records),
        ):
            return None
        validate_episode_records(records, seeds=seeds, methods=COMPARATOR_METHODS)
        return [dict(record) for record in records]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        CacheValidationError,
    ):
        return None


class EpisodeAdapter(Protocol):
    """Minimal trusted surface controlled by :func:`run_frozen_episode`."""

    max_horizon: int

    def reset(
        self,
        *,
        seed: int,
        fixed_head_yaw_degrees: float | None,
    ) -> Mapping[str, np.ndarray]: ...

    def predict_base_action(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> np.ndarray: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Mapping[str, np.ndarray], bool, bool]: ...

    def finalize_outcome(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _contract_functions() -> tuple[Callable[..., Any], type, type[Exception]]:
    # Imported lazily so artifact verification and cache inspection do not
    # import candidate code or simulator dependencies.
    from autoresearch.contract import (
        CandidateBoundary,
        ContractViolation,
        copy_public_observation,
    )

    return copy_public_observation, CandidateBoundary, ContractViolation


def _copy_public_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    copy_observation, _, _ = _contract_functions()
    copied = copy_observation(observation)
    if tuple(copied.keys()) != PUBLIC_OBSERVATION_FIELDS and set(copied) != set(
        PUBLIC_OBSERVATION_FIELDS,
    ):
        raise EpisodeContractError(
            "Episode adapter did not provide exactly the public observation fields",
        )
    # Canonicalize order as well as content; no caller receives the adapter's
    # original arrays.
    return {
        field: np.array(copied[field], copy=True)
        for field in PUBLIC_OBSERVATION_FIELDS
    }


def _validate_full_action(action: Any, *, source: str) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (3,):
        raise EpisodeContractError(f"{source} emitted action shape {array.shape}, expected (3,)")
    if not np.all(np.isfinite(array)):
        raise EpisodeContractError(f"{source} emitted a non-finite action")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise EpisodeContractError(f"{source} emitted an action outside [-1, 1]")
    return np.array(array, dtype=np.float32, copy=True)


def _call_head_controller(
    boundary: Any,
    *,
    observation: Mapping[str, np.ndarray],
    public_history: Sequence[Mapping[str, np.ndarray]],
    base_head_action: float,
    step_index: int,
) -> float:
    _, _, contract_violation = _contract_functions()
    from autoresearch.guard import CandidateRuntimeError
    try:
        value = boundary.head_action(
            observation=observation,
            public_history=public_history,
            base_head_action=float(base_head_action),
            step_index=int(step_index),
        )
    except Exception as exc:
        # Preserve the contract module's typed violation while making unknown
        # candidate failures unambiguously evaluator failures.
        if isinstance(exc, (contract_violation, CandidateRuntimeError)):
            raise
        raise EpisodeContractError("Candidate head_action call failed") from exc
    return float(value)


def _action_trace_sha256(actions: Sequence[np.ndarray]) -> str:
    return canonical_sha256(
        [np.asarray(action, dtype=np.float32).tolist() for action in actions],
    )


def run_frozen_episode(
    episode: EpisodeAdapter,
    *,
    controller: Any | None,
    method: str,
    seed: int,
    max_horizon: int,
    public_history_limit: int,
) -> dict[str, Any]:
    """Run one branch while keeping every candidate call outside trusted state."""

    if method not in EVALUATION_METHODS:
        raise ValueError(f"Unknown autoresearch evaluation method: {method!r}")
    if method in RATE_METHODS and controller is None:
        raise ValueError(f"{method} requires a rate-controller instance")
    if method == REFERENCE_METHOD and controller is not None:
        raise ValueError("The fixed +60 research reference does not use a controller")
    if int(max_horizon) <= 0:
        raise ValueError("max_horizon must be positive")
    if int(public_history_limit) < 0:
        raise ValueError("public_history_limit must be non-negative")
    if int(getattr(episode, "max_horizon")) != int(max_horizon):
        raise EpisodeContractError(
            "Episode adapter horizon differs from the frozen evaluator horizon",
        )

    boundary = None
    if controller is not None:
        _, boundary_type, contract_violation = _contract_functions()
        from autoresearch.guard import CandidateRuntimeError

        boundary = boundary_type(controller)
        try:
            boundary.reset(episode_seed=int(seed))
        except Exception as exc:
            if isinstance(exc, (contract_violation, CandidateRuntimeError)):
                raise
            raise EpisodeContractError("Candidate reset failed") from exc

    fixed_yaw = 60.0 if method == REFERENCE_METHOD else None
    observation = _copy_public_observation(
        episode.reset(seed=int(seed), fixed_head_yaw_degrees=fixed_yaw),
    )
    history: deque[dict[str, np.ndarray]] = deque(maxlen=int(public_history_limit) or None)
    actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        if steps >= int(max_horizon):
            raise EpisodeContractError(
                "Episode exceeded the frozen horizon without normal termination/truncation",
            )
        # The frozen locomotion policy and the controller get independent
        # defensive copies.  The base policy is called exactly once here.
        base_observation = _copy_public_observation(observation)
        base_action = _validate_full_action(
            episode.predict_base_action(base_observation),
            source="Frozen SAC policy",
        )
        action = np.array(base_action, copy=True)
        if method in RATE_METHODS:
            candidate_observation = _copy_public_observation(observation)
            candidate_history = tuple(
                _copy_public_observation(item) for item in history
            )
            action[2] = _call_head_controller(
                boundary,
                observation=candidate_observation,
                public_history=candidate_history,
                base_head_action=float(base_action[2]),
                step_index=steps,
            )
        else:
            # The fixed placement is an explicitly non-rate-controlled
            # research reference; holding zero here keeps its teleported pose.
            action[2] = 0.0
        action = _validate_full_action(action, source=method)

        # Update history from our pristine public copy, never from anything the
        # candidate could have mutated.
        if int(public_history_limit) > 0:
            history.append(_copy_public_observation(observation))
        next_observation, terminated, truncated = episode.step(action)
        actions.append(np.array(action, copy=True))
        steps += 1
        observation = _copy_public_observation(next_observation)

    outcome = _normalize_outcome(
        episode.finalize_outcome(),
        method=method,
        seed=int(seed),
        action_trace_sha256=_action_trace_sha256(actions),
    )
    if int(outcome["steps"]) != steps:
        raise EpisodeContractError(
            "Episode adapter reported a step count different from executed env.step calls",
        )
    return outcome


def _run_branch(
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
            seed=int(seed),
            max_horizon=int(max_horizon),
            public_history_limit=int(public_history_limit),
        )
    finally:
        episode.close()


def _run_methods(
    episode_factory: Callable[[], EpisodeAdapter],
    *,
    controllers: Mapping[str, Any | None],
    methods: Sequence[str],
    seeds: Sequence[int],
    max_horizon: int,
    public_history_limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            records.append(
                _run_branch(
                    episode_factory,
                    controller=controllers[method],
                    method=method,
                    seed=int(seed),
                    max_horizon=max_horizon,
                    public_history_limit=public_history_limit,
                ),
            )
    validate_episode_records(records, seeds=seeds, methods=methods)
    return records


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(record) for record in records])


def assert_deterministic_records(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> None:
    if not hmac.compare_digest(records_sha256(first), records_sha256(second)):
        raise DeterminismError(
            "Repeated evaluation changed episode outcomes or public action traces",
        )


@dataclasses.dataclass(frozen=True)
class GateDecision:
    decision: str
    decision_reason: str
    primary_delta: float
    clean_success_episode_delta: int
    candidate_clean_successes: int
    incumbent_clean_successes: int
    candidate_capture_episodes: int
    incumbent_capture_episodes: int
    paired_counts: Mapping[str, int]

    @property
    def keep(self) -> bool:
        return self.decision == "keep"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def mechanical_keep_or_discard(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    checks: Mapping[str, bool],
    minimum_improvement_episodes: int = 2,
) -> GateDecision:
    """Apply the phase-1 decision rule with no composite score or tie-break."""

    frozen_seeds = _validate_nonempty_seeds(seeds)
    validate_episode_records(records, seeds=frozen_seeds, methods=EVALUATION_METHODS)
    if int(minimum_improvement_episodes) <= 0:
        raise ValueError("minimum_improvement_episodes must be positive")

    by_method = {
        method: {
            int(record["seed"]): record
            for record in records
            if record["method"] == method
        }
        for method in EVALUATION_METHODS
    }
    candidate_successes = sum(
        bool(by_method["candidate"][seed]["clean_success"])
        for seed in frozen_seeds
    )
    incumbent_successes = sum(
        bool(by_method["incumbent"][seed]["clean_success"])
        for seed in frozen_seeds
    )
    candidate_captures = sum(
        bool(by_method["candidate"][seed]["capture_episode"])
        for seed in frozen_seeds
    )
    incumbent_captures = sum(
        bool(by_method["incumbent"][seed]["capture_episode"])
        for seed in frozen_seeds
    )
    candidate_only = sum(
        bool(by_method["candidate"][seed]["clean_success"])
        and not bool(by_method["incumbent"][seed]["clean_success"])
        for seed in frozen_seeds
    )
    incumbent_only = sum(
        bool(by_method["incumbent"][seed]["clean_success"])
        and not bool(by_method["candidate"][seed]["clean_success"])
        for seed in frozen_seeds
    )
    both_success = sum(
        bool(by_method["candidate"][seed]["clean_success"])
        and bool(by_method["incumbent"][seed]["clean_success"])
        for seed in frozen_seeds
    )
    both_failure = len(frozen_seeds) - candidate_only - incumbent_only - both_success
    success_delta = candidate_successes - incumbent_successes
    primary_delta = success_delta / len(frozen_seeds)

    failed_checks = sorted(str(name) for name, passed in checks.items() if passed is not True)
    if not checks:
        decision = "discard"
        reason = "no hard-gate checks were supplied"
    elif failed_checks:
        decision = "discard"
        reason = "hard checks failed: " + ", ".join(failed_checks)
    elif candidate_captures > incumbent_captures:
        decision = "discard"
        reason = (
            "candidate capture-episode rate exceeds the incumbent "
            f"({candidate_captures}/{len(frozen_seeds)} > "
            f"{incumbent_captures}/{len(frozen_seeds)})"
        )
    elif success_delta < int(minimum_improvement_episodes):
        decision = "discard"
        reason = (
            "candidate clean success improves by fewer than "
            f"{minimum_improvement_episodes} paired episodes "
            f"({success_delta:+d})"
        )
    else:
        decision = "keep"
        reason = (
            "all hard gates passed; clean success improved by "
            f"{success_delta} paired episodes without worse capture rate"
        )

    return GateDecision(
        decision=decision,
        decision_reason=reason,
        primary_delta=primary_delta,
        clean_success_episode_delta=success_delta,
        candidate_clean_successes=candidate_successes,
        incumbent_clean_successes=incumbent_successes,
        candidate_capture_episodes=candidate_captures,
        incumbent_capture_episodes=incumbent_captures,
        paired_counts={
            "both_success": both_success,
            "candidate_only_success": candidate_only,
            "incumbent_only_success": incumbent_only,
            "both_failure": both_failure,
        },
    )


def _mean(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(record[field]) for record in records]))


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in EVALUATION_METHODS:
        subset = [record for record in records if record["method"] == method]
        if not subset:
            continue
        result[method] = {
            "episodes": len(subset),
            "clean_success_rate": _mean(subset, "clean_success"),
            "capture_episode_rate": _mean(subset, "capture_episode"),
            "goal_reach_rate": _mean(subset, "goal_reached"),
            "mean_steps": _mean(subset, "steps"),
            "mean_minimum_predator_distance": _mean(
                subset,
                "minimum_predator_distance",
            ),
            "mean_path_cost": _mean(subset, "path_cost"),
            "mean_gaze_travel_degrees": _mean(subset, "gaze_travel_degrees"),
            "mean_predator_pixels_visible_fraction": _mean(
                subset,
                "predator_pixels_visible_fraction",
            ),
        }
    return result


def evaluate_paired(
    *,
    episode_factory: Callable[[], EpisodeAdapter],
    candidate: Any,
    incumbent: Any,
    seeds: Sequence[int],
    max_horizon: int,
    public_history_limit: int,
    checks: Mapping[str, bool],
    cache_path: str | Path | None = None,
    cache_identity: ComparatorCacheIdentity | None = None,
    verify_determinism: bool = True,
    minimum_improvement_episodes: int = 2,
) -> dict[str, Any]:
    """Evaluate candidate, legal-rate incumbent, and fixed +60 on paired seeds."""

    frozen_seeds = _validate_nonempty_seeds(seeds)
    if (cache_path is None) != (cache_identity is None):
        raise ValueError("cache_path and cache_identity must be supplied together")
    if cache_identity is not None:
        if cache_identity.ordered_seed_sha256 != ordered_seed_sha256(frozen_seeds):
            raise EvaluationError("Cache identity does not match the ordered seed set")
        if cache_identity.max_horizon != int(max_horizon):
            raise EvaluationError("Cache identity does not match max_horizon")
        if cache_identity.public_history_limit != int(public_history_limit):
            raise EvaluationError("Cache identity does not match public_history_limit")

    comparator_records = None
    cache_hit = False
    if cache_path is not None and cache_identity is not None:
        comparator_records = load_comparator_cache(
            cache_path,
            identity=cache_identity,
            seeds=frozen_seeds,
        )
        cache_hit = comparator_records is not None

    controllers = {
        "candidate": candidate,
        "incumbent": incumbent,
        REFERENCE_METHOD: None,
    }
    if comparator_records is None:
        comparator_records = _run_methods(
            episode_factory,
            controllers=controllers,
            methods=COMPARATOR_METHODS,
            seeds=frozen_seeds,
            max_horizon=int(max_horizon),
            public_history_limit=int(public_history_limit),
        )
        comparator_deterministic = False
        if verify_determinism:
            repeated_comparators = _run_methods(
                episode_factory,
                controllers=controllers,
                methods=COMPARATOR_METHODS,
                seeds=frozen_seeds,
                max_horizon=int(max_horizon),
                public_history_limit=int(public_history_limit),
            )
            assert_deterministic_records(comparator_records, repeated_comparators)
            comparator_deterministic = True
        if cache_path is not None and cache_identity is not None:
            write_comparator_cache(
                cache_path,
                identity=cache_identity,
                seeds=frozen_seeds,
                records=comparator_records,
                determinism_verified=comparator_deterministic,
            )

    candidate_records = _run_methods(
        episode_factory,
        controllers=controllers,
        methods=("candidate",),
        seeds=frozen_seeds,
        max_horizon=int(max_horizon),
        public_history_limit=int(public_history_limit),
    )
    candidate_deterministic = False
    if verify_determinism:
        repeated_candidate = _run_methods(
            episode_factory,
            controllers=controllers,
            methods=("candidate",),
            seeds=frozen_seeds,
            max_horizon=int(max_horizon),
            public_history_limit=int(public_history_limit),
        )
        assert_deterministic_records(candidate_records, repeated_candidate)
        candidate_deterministic = True

    by_seed_method = {
        (int(record["seed"]), str(record["method"])): dict(record)
        for record in (*candidate_records, *comparator_records)
    }
    records = [
        by_seed_method[(seed, method)]
        for seed in frozen_seeds
        for method in EVALUATION_METHODS
    ]
    validate_episode_records(records, seeds=frozen_seeds, methods=EVALUATION_METHODS)
    intrinsic_checks = {
        str(name): passed is True
        for name, passed in checks.items()
    }
    intrinsic_checks["external_checks_supplied"] = bool(checks)
    intrinsic_checks["actions_legal"] = bool(
        intrinsic_checks.get("actions_legal", True),
    )
    intrinsic_checks["records_complete"] = bool(
        intrinsic_checks.get("records_complete", True),
    )
    intrinsic_checks["determinism"] = bool(
        intrinsic_checks.get("determinism", True)
        and candidate_deterministic
        and (cache_hit or comparator_deterministic)
    )
    decision = mechanical_keep_or_discard(
        records,
        seeds=frozen_seeds,
        checks=intrinsic_checks,
        minimum_improvement_episodes=int(minimum_improvement_episodes),
    )
    return {
        "records": records,
        "records_sha256": records_sha256(records),
        "summary": summarize_records(records),
        "checks": intrinsic_checks,
        "cache_hit": cache_hit,
        "decision": decision.to_dict(),
    }


def comparator_identity_from_config(
    config: Mapping[str, Any],
    *,
    seed_set_name: str,
    evaluator_digest: str,
    environment_contract_sha256: str,
    incumbent_sha256: str,
    max_horizon: int,
) -> ComparatorCacheIdentity:
    """Build a comparator identity directly from the frozen config mapping."""

    _validate_autoresearch_config_contract(config)
    seed_set = seed_set_from_config(config, seed_set_name)
    source = config["source"]
    history_limit = int(config["evaluation"]["maximum_history_length"])
    return ComparatorCacheIdentity.from_seeds(
        checkpoint_sha256=str(source.get("checkpoint_sha256", "")),
        resolved_config_sha256=str(source.get("resolved_config_sha256", "")),
        evaluator_sha256=str(evaluator_digest),
        seed_set_id=seed_set.seed_set_id,
        seeds=seed_set.seeds,
        environment_contract_sha256=str(environment_contract_sha256),
        incumbent_sha256=str(incumbent_sha256),
        max_horizon=int(max_horizon),
        public_history_limit=history_limit,
    )


def evaluate_paired_from_config(
    config: Mapping[str, Any],
    *,
    seed_set_name: str,
    episode_factory: Callable[[], EpisodeAdapter],
    candidate: Any,
    incumbent: Any,
    checks: Mapping[str, bool],
    environment_contract_sha256: str,
    incumbent_sha256: str,
    cache_path: str | Path | None = None,
    evaluator_digest: str | None = None,
    allow_confirmation: bool = False,
) -> dict[str, Any]:
    """Config-shaped entry point for runner/CLI integration.

    ``confirmation`` is deliberately unavailable unless the dedicated confirm
    path passes ``allow_confirmation=True`` after recording user authorization.
    No caller can override seeds, horizon, history length, source hashes, or
    the two-episode development threshold through this function.
    """

    _validate_autoresearch_config_contract(config)
    seed_set = seed_set_from_config(config, seed_set_name)
    if seed_set.requires_explicit_authorization and not allow_confirmation:
        raise EvaluationError(
            "The confirmation seed set requires the dedicated authorized confirm command",
        )
    if allow_confirmation and not seed_set.requires_explicit_authorization:
        raise EvaluationError(
            "allow_confirmation may be used only with the registered confirmation set",
        )
    max_horizon = int(getattr(episode_factory, "max_horizon", 0))
    if max_horizon <= 0:
        # Lightweight factories expose ``horizon`` while production exposes
        # ``max_horizon``.  Both values remain factory-owned, not caller-owned.
        max_horizon = int(getattr(episode_factory, "horizon", 0))
    if max_horizon <= 0:
        raise EvaluationError("Episode factory must expose its frozen max_horizon")
    source = config["source"]
    artifacts = getattr(episode_factory, "artifacts", None)
    if artifacts is not None:
        if not hmac.compare_digest(
            str(artifacts.checkpoint_sha256),
            str(source.get("checkpoint_sha256", "")),
        ) or not hmac.compare_digest(
            str(artifacts.resolved_config_sha256),
            str(source.get("resolved_config_sha256", "")),
        ):
            raise EvaluationError("Verified real artifacts differ from the frozen run config")

    selected_evaluator_digest = str(evaluator_digest or evaluator_sha256())
    cache_identity = comparator_identity_from_config(
        config,
        seed_set_name=seed_set.name,
        evaluator_digest=selected_evaluator_digest,
        environment_contract_sha256=environment_contract_sha256,
        incumbent_sha256=incumbent_sha256,
        max_horizon=max_horizon,
    )
    decision = config["decision"]
    minimum_improvement = int(decision.get("minimum_paired_episode_improvement", 0))
    if minimum_improvement <= 0:
        raise EvaluationError("Configured paired-episode improvement gate must be positive")
    history_limit = int(config["evaluation"]["maximum_history_length"])
    result = evaluate_paired(
        episode_factory=episode_factory,
        candidate=candidate,
        incumbent=incumbent,
        seeds=seed_set.seeds,
        max_horizon=max_horizon,
        public_history_limit=history_limit,
        checks=checks,
        cache_path=cache_path,
        cache_identity=cache_identity if cache_path is not None else None,
        verify_determinism=True,
        minimum_improvement_episodes=minimum_improvement,
    )
    result["identity"] = {
        **cache_identity.payload(),
        "comparator_cache_key": cache_identity.key,
    }
    result["seed_set"] = dataclasses.asdict(seed_set)
    return result


# Stable runner-facing aliases.  Keep the descriptive implementation names for
# tests while allowing the CLI to speak in terms of a registered evaluation.
resolve_seed_set = seed_set_from_config
evaluate_registered = evaluate_paired_from_config


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    lower = min(int(left_only), int(right_only))
    lower_tail = math.ldexp(
        sum(math.comb(discordant, index) for index in range(lower + 1)),
        -discordant,
    )
    return min(1.0, 2.0 * lower_tail)


def confirmation_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20_260_830,
) -> dict[str, Any]:
    """Deterministic paired uncertainty for an explicitly authorized confirm run."""

    frozen_seeds = _validate_nonempty_seeds(seeds)
    validate_episode_records(records, seeds=frozen_seeds, methods=EVALUATION_METHODS)
    by_method = {
        method: {
            int(record["seed"]): record
            for record in records
            if record["method"] == method
        }
        for method in RATE_METHODS
    }
    differences = np.asarray(
        [
            float(by_method["candidate"][seed]["clean_success"])
            - float(by_method["incumbent"][seed]["clean_success"])
            for seed in frozen_seeds
        ],
        dtype=np.float64,
    )
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        len(differences),
        size=(int(bootstrap_samples), len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    candidate_only = int(np.sum(differences == 1.0))
    incumbent_only = int(np.sum(differences == -1.0))
    return {
        "paired_episodes": len(frozen_seeds),
        "mean_delta": float(differences.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "candidate_only_successes": candidate_only,
        "incumbent_only_successes": incumbent_only,
        "mcnemar_exact_p": exact_mcnemar_p(candidate_only, incumbent_only),
    }


class FakeEpisodeAdapter:
    """Small deterministic episode used to test the trusted orchestration."""

    def __init__(
        self,
        *,
        horizon: int,
        outcome_fn: Callable[[int, tuple[np.ndarray, ...], float], Mapping[str, Any]]
        | None = None,
        base_action_fn: Callable[[int, int, Mapping[str, np.ndarray]], Any] | None = None,
    ) -> None:
        if int(horizon) <= 0:
            raise ValueError("Fake horizon must be positive")
        self.max_horizon = int(horizon)
        self._outcome_fn = outcome_fn
        self._base_action_fn = base_action_fn
        self.seed: int | None = None
        self.step_index = 0
        self.initial_head_yaw_degrees = 0.0
        self.head_yaw_degrees = 0.0
        self.fixed_reference = False
        self.actions: list[np.ndarray] = []
        self.predict_calls = 0
        self.step_calls = 0
        self.closed = False
        self.privileged_sentinel = "must-never-cross-boundary"

    def _observation(self, previous_action: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if self.seed is None:
            raise RuntimeError("Fake episode must be reset before observation")
        pixel = np.uint8((self.seed + self.step_index) % 251)
        previous = (
            np.zeros((3,), dtype=np.float32)
            if previous_action is None
            else np.asarray(previous_action, dtype=np.float32).copy()
        )
        return {
            "image_left": np.full((4, 5, 3), pixel, dtype=np.uint8),
            "image_right": np.full((4, 5, 3), np.uint8((int(pixel) + 1) % 251), dtype=np.uint8),
            "proprio": np.asarray(
                (0.0, 0.0, self.head_yaw_degrees / 60.0),
                dtype=np.float32,
            ),
            "previous_action": previous,
        }

    def reset(
        self,
        *,
        seed: int,
        fixed_head_yaw_degrees: float | None,
    ) -> Mapping[str, np.ndarray]:
        self.seed = int(seed)
        self.step_index = 0
        self.actions = []
        self.predict_calls = 0
        self.step_calls = 0
        self.closed = False
        self.initial_head_yaw_degrees = float(fixed_head_yaw_degrees or 0.0)
        self.head_yaw_degrees = self.initial_head_yaw_degrees
        self.fixed_reference = fixed_head_yaw_degrees is not None
        return self._observation()

    def predict_base_action(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if self.seed is None:
            raise RuntimeError("Fake episode must be reset before prediction")
        self.predict_calls += 1
        if self._base_action_fn is None:
            return np.zeros((3,), dtype=np.float32)
        return np.asarray(
            self._base_action_fn(self.seed, self.step_index, observation),
            dtype=np.float32,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Mapping[str, np.ndarray], bool, bool]:
        self.step_calls += 1
        applied = np.asarray(action, dtype=np.float32).copy()
        self.actions.append(applied)
        head_command = float(applied[2])
        if not self.fixed_reference:
            if abs(head_command) > 0.05:
                self.head_yaw_degrees += head_command * 24.0
            else:
                recenter_delta = min(abs(self.head_yaw_degrees), 9.0)
                self.head_yaw_degrees -= math.copysign(
                    recenter_delta,
                    self.head_yaw_degrees,
                )
            self.head_yaw_degrees = float(
                np.clip(self.head_yaw_degrees, -60.0, 60.0),
            )
        self.step_index += 1
        terminated = self.step_index >= self.max_horizon
        return self._observation(applied), terminated, False

    def finalize_outcome(self) -> Mapping[str, Any]:
        if self.seed is None:
            raise RuntimeError("Fake episode was not reset")
        frozen_actions = tuple(np.array(action, copy=True) for action in self.actions)
        if self._outcome_fn is None:
            clean_success = bool(sum(float(action[2]) for action in frozen_actions) > 0.5)
            supplied: Mapping[str, Any] = {
                "clean_success": clean_success,
                "capture_episode": not clean_success,
                "goal_reached": clean_success,
            }
        else:
            supplied = self._outcome_fn(
                self.seed,
                frozen_actions,
                self.initial_head_yaw_degrees,
            )
        defaults = {
            "clean_success": False,
            "capture_episode": False,
            "goal_reached": False,
            "steps": self.step_calls,
            "minimum_predator_distance": 0.25,
            "path_cost": float(self.step_calls),
            "gaze_travel_degrees": float(
                sum(abs(float(action[2])) * 24.0 for action in frozen_actions)
            ),
            "predator_pixels_visible_fraction": 0.5,
        }
        defaults.update(dict(supplied))
        return defaults

    def close(self) -> None:
        self.closed = True


class FakeEpisodeFactory:
    """Callable factory that records every fresh fake branch for assertions."""

    def __init__(
        self,
        *,
        horizon: int = 3,
        outcome_fn: Callable[[int, tuple[np.ndarray, ...], float], Mapping[str, Any]]
        | None = None,
        base_action_fn: Callable[[int, int, Mapping[str, np.ndarray]], Any] | None = None,
    ) -> None:
        self.horizon = int(horizon)
        self.outcome_fn = outcome_fn
        self.base_action_fn = base_action_fn
        self.episodes: list[FakeEpisodeAdapter] = []

    def __call__(self) -> FakeEpisodeAdapter:
        episode = FakeEpisodeAdapter(
            horizon=self.horizon,
            outcome_fn=self.outcome_fn,
            base_action_fn=self.base_action_fn,
        )
        self.episodes.append(episode)
        return episode


class Exp05EpisodeAdapter:
    """One fresh real simulator branch driven by the shared frozen SAC model."""

    def __init__(self, *, env: Any, model: Any, max_horizon: int) -> None:
        self.env = env
        self.model = model
        self.max_horizon = int(max_horizon)
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self._seed: int | None = None
        self._steps = 0
        self._total_reward = 0.0
        self._capture_count = 0
        self._goal_reached = False
        self._visible_steps = 0
        self._minimum_distance = math.inf
        self._path_cost = 0.0
        self._gaze_travel = 0.0
        self._previous_location: np.ndarray | None = None
        self._previous_head_yaw = 0.0
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: int,
        fixed_head_yaw_degrees: float | None,
    ) -> Mapping[str, np.ndarray]:
        from benchmarks.peekbench.environment import observe_current, state_with_gaze

        self._reset_metrics()
        self._seed = int(seed)
        observation, _ = self.env.reset(seed=int(seed))
        if fixed_head_yaw_degrees is not None:
            if not math.isclose(float(fixed_head_yaw_degrees), 60.0, abs_tol=0.0):
                raise EpisodeContractError("EXP-05 reference supports only fixed +60 degrees")
            self.env.head_recenter_rate = 0.0
            state = state_with_gaze(
                self.env.get_state_dict(),
                float(fixed_head_yaw_degrees),
            )
            self.env.set_state_dict(state)
            observation = observe_current(self.env)

        model_state = self.env.unwrapped.model
        self._previous_location = np.asarray(
            model_state.prey.state.location,
            dtype=np.float64,
        )
        self._previous_head_yaw = float(self.env.head_yaw_degrees)
        self._minimum_distance = float(
            np.linalg.norm(
                self._previous_location
                - np.asarray(model_state.predator.state.location, dtype=np.float64),
            ),
        )
        return observation

    def predict_base_action(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[Mapping[str, np.ndarray], bool, bool]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._total_reward += float(reward)
        events = info["transition_events"]
        self._capture_count += int(bool(events["capture_event"]))
        self._goal_reached = bool(self._goal_reached or events["goal_event"])
        self._visible_steps += int(bool(events["predator_pixels_visible"]))
        self._minimum_distance = min(
            self._minimum_distance,
            float(events["minimum_distance"]),
        )

        model_state = self.env.unwrapped.model
        current_location = np.asarray(
            model_state.prey.state.location,
            dtype=np.float64,
        )
        if self._previous_location is None:
            raise RuntimeError("EXP-05 episode was not reset")
        self._path_cost += float(np.linalg.norm(current_location - self._previous_location))
        self._previous_location = current_location
        current_head_yaw = float(self.env.head_yaw_degrees)
        self._gaze_travel += abs(current_head_yaw - self._previous_head_yaw)
        self._previous_head_yaw = current_head_yaw
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        return observation, self._terminated, self._truncated

    def finalize_outcome(self) -> Mapping[str, Any]:
        if not (self._terminated or self._truncated):
            raise EpisodeContractError("Cannot finalize an unfinished EXP-05 episode")
        return {
            "clean_success": bool(self._goal_reached and self._capture_count == 0),
            "capture_episode": bool(self._capture_count > 0),
            "goal_reached": bool(self._goal_reached),
            "steps": self._steps,
            "minimum_predator_distance": self._minimum_distance,
            "path_cost": self._path_cost,
            "gaze_travel_degrees": self._gaze_travel,
            "predator_pixels_visible_fraction": (
                self._visible_steps / self._steps if self._steps else 0.0
            ),
        }

    def close(self) -> None:
        self.env.close()


class RealExp05EpisodeFactory:
    """Verified production factory for the registered frozen EXP-05 policy."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path = DEFAULT_EXP05_CHECKPOINT,
        resolved_config_path: str | Path = DEFAULT_EXP05_RESOLVED_CONFIG,
        expected_checkpoint_sha256: str = EXPECTED_EXP05_CHECKPOINT_SHA256,
        expected_resolved_config_sha256: str | None = None,
    ) -> None:
        # Verification deliberately precedes deserialization/import-heavy
        # environment setup.  Wrong bytes can never reach SAC.load.
        self.artifacts = verify_exp05_artifacts(
            checkpoint_path,
            resolved_config_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_resolved_config_sha256=expected_resolved_config_sha256,
        )
        from stable_baselines3 import SAC
        from training.first_person_sac import load_sac_config

        self.config = load_sac_config(self.artifacts.resolved_config_path)
        environment = self.config["environment"]
        if environment.get("observation_mode") != "mouse":
            raise ArtifactVerificationError(
                "Frozen EXP-05 config does not use the public mouse observation",
            )
        if environment.get("action_mode") != "egocentric_velocity_head":
            raise ArtifactVerificationError(
                "Frozen EXP-05 config does not expose the three-component active-gaze action",
            )
        self.max_horizon = int(environment["max_step"])
        self.model = SAC.load(self.artifacts.checkpoint_path, device="cpu")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
    ) -> "RealExp05EpisodeFactory":
        """Construct from the registered ``gaze_dev.yaml`` mapping."""

        _validate_autoresearch_config_contract(config)
        source = config["source"]
        root = Path(project_root).expanduser().resolve()

        def source_path(name: str) -> Path:
            raw = Path(str(source.get(name, "")))
            if not str(raw):
                raise EvaluationError(f"Autoresearch source.{name} is required")
            return raw if raw.is_absolute() else root / raw

        return cls(
            checkpoint_path=source_path("checkpoint_path"),
            resolved_config_path=source_path("resolved_config_path"),
            expected_checkpoint_sha256=str(source.get("checkpoint_sha256", "")),
            expected_resolved_config_sha256=str(
                source.get("resolved_config_sha256", ""),
            ),
        )

    def __call__(self) -> Exp05EpisodeAdapter:
        from training.first_person_sac import make_first_person_env

        env = make_first_person_env(self.config)
        if tuple(env.action_space.shape) != (3,):
            env.close()
            raise EpisodeContractError("Frozen EXP-05 environment action shape is not (3,)")
        return Exp05EpisodeAdapter(
            env=env,
            model=self.model,
            max_horizon=self.max_horizon,
        )


__all__ = [
    "ArtifactBundle",
    "ArtifactVerificationError",
    "CACHE_SCHEMA_VERSION",
    "COMPARATOR_METHODS",
    "ComparatorCacheIdentity",
    "DEFAULT_EXP05_CHECKPOINT",
    "DEFAULT_EXP05_RESOLVED_CONFIG",
    "DeterminismError",
    "EPISODE_RESULT_FIELDS",
    "EVALUATION_METHODS",
    "EXPECTED_EXP05_CHECKPOINT_SHA256",
    "EpisodeContractError",
    "EvaluationError",
    "Exp05EpisodeAdapter",
    "FakeEpisodeAdapter",
    "FakeEpisodeFactory",
    "FrozenSeedSet",
    "GateDecision",
    "PUBLIC_OBSERVATION_FIELDS",
    "RealExp05EpisodeFactory",
    "assert_deterministic_records",
    "canonical_sha256",
    "comparator_cache_key",
    "comparator_identity_from_config",
    "confirmation_statistics",
    "evaluator_sha256",
    "evaluate_paired",
    "evaluate_paired_from_config",
    "evaluate_registered",
    "exact_mcnemar_p",
    "file_sha256",
    "load_comparator_cache",
    "mechanical_keep_or_discard",
    "ordered_seed_sha256",
    "records_sha256",
    "run_frozen_episode",
    "resolve_seed_set",
    "seed_set_from_config",
    "summarize_records",
    "validate_episode_records",
    "verify_exp05_artifacts",
    "write_comparator_cache",
]
