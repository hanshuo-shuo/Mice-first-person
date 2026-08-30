import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from autoresearch.contract import CandidateContractError
from autoresearch.evaluator import (
    EVALUATION_METHODS,
    ArtifactVerificationError,
    CacheValidationError,
    ComparatorCacheIdentity,
    DeterminismError,
    EpisodeContractError,
    EvaluationError,
    FakeEpisodeAdapter,
    FakeEpisodeFactory,
    assert_deterministic_records,
    canonical_sha256,
    confirmation_statistics,
    evaluate_paired,
    evaluate_paired_from_config,
    load_comparator_cache,
    mechanical_keep_or_discard,
    records_sha256,
    run_frozen_episode,
    seed_set_from_config,
    validate_episode_records,
    verify_exp05_artifacts,
)


class ConstantController:
    def __init__(self, value):
        self.value = value
        self.reset_seeds = []
        self.calls = []

    def reset(self, *, episode_seed):
        self.reset_seeds.append(int(episode_seed))

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        self.calls.append(
            {
                "fields": tuple(observation),
                "history_fields": [tuple(item) for item in public_history],
                "history_length": len(public_history),
                "base_head_action": float(base_head_action),
                "step_index": int(step_index),
            },
        )
        return self.value


class MutatingController(ConstantController):
    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        result = super().head_action(
            observation=observation,
            public_history=public_history,
            base_head_action=base_head_action,
            step_index=step_index,
        )
        for value in observation.values():
            value[...] = 0
        for frame in public_history:
            for value in frame.values():
                value[...] = 0
        return result


class StatefulNondeterministicController(ConstantController):
    def __init__(self):
        super().__init__(0.0)
        self.toggle = False

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, base_head_action, step_index
        self.toggle = not self.toggle
        return 1.0 if self.toggle else -1.0


def _identity(seeds=(11, 12), **overrides):
    values = {
        "checkpoint_sha256": "checkpoint-a",
        "resolved_config_sha256": "config-a",
        "evaluator_sha256": "evaluator-a",
        "seed_set_id": "development-a",
        "seeds": seeds,
        "environment_contract_sha256": "environment-a",
        "incumbent_sha256": "incumbent-a",
        "max_horizon": 2,
        "public_history_limit": 2,
    }
    values.update(overrides)
    return ComparatorCacheIdentity.from_seeds(**values)


def _record(seed, method, *, success=False, capture=False):
    return {
        "method": method,
        "seed": int(seed),
        "clean_success": bool(success),
        "capture_episode": bool(capture),
        "goal_reached": bool(success),
        "steps": 2,
        "minimum_predator_distance": 0.25,
        "path_cost": 1.0,
        "gaze_travel_degrees": 0.0,
        "predator_pixels_visible_fraction": 0.5,
        "action_trace_sha256": canonical_sha256([method, int(seed)]),
    }


def _paired_records(seeds, candidate_successes, incumbent_successes):
    records = []
    for seed in seeds:
        records.extend(
            (
                _record(
                    seed,
                    "candidate",
                    success=seed in candidate_successes,
                ),
                _record(
                    seed,
                    "incumbent",
                    success=seed in incumbent_successes,
                ),
                _record(seed, "fixed_p60", success=False),
            ),
        )
    return records


def test_fake_paired_evaluator_has_public_defensive_boundary_and_fresh_branches():
    candidate = MutatingController(1.0)
    incumbent = ConstantController(0.0)
    factory = FakeEpisodeFactory(horizon=2)

    result = evaluate_paired(
        episode_factory=factory,
        candidate=candidate,
        incumbent=incumbent,
        seeds=(101, 102),
        max_horizon=2,
        public_history_limit=1,
        checks={"contract": True, "hashes": True, "leaks": True},
    )

    assert result["decision"]["decision"] == "keep"
    assert result["decision"]["clean_success_episode_delta"] == 2
    assert len(factory.episodes) == 12  # three branches x two seeds x two repeats
    assert all(episode.closed for episode in factory.episodes)
    assert all(episode.predict_calls == episode.step_calls == 2 for episode in factory.episodes)
    assert all(call["fields"] == (
        "image_left",
        "image_right",
        "proprio",
        "previous_action",
    ) for call in candidate.calls)
    assert max(call["history_length"] for call in candidate.calls) == 1
    serialized = json.dumps(result, sort_keys=True)
    assert "must-never-cross-boundary" not in serialized
    assert "privileged" not in serialized

    # Candidate mutation did not alter any applied locomotion component or the
    # deterministic observations used by the repeated branch.
    candidate_episodes = [
        episode
        for episode in factory.episodes
        if episode.actions and float(episode.actions[0][2]) == 1.0
    ]
    assert candidate_episodes
    assert all(
        np.array_equal(action[:2], np.zeros((2,), dtype=np.float32))
        for episode in candidate_episodes
        for action in episode.actions
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0001, 1.0001, [0.0]])
def test_invalid_candidate_action_fails_before_episode_step(bad_value):
    episode = FakeEpisodeAdapter(horizon=2)
    with pytest.raises(CandidateContractError):
        run_frozen_episode(
            episode,
            controller=ConstantController(bad_value),
            method="candidate",
            seed=7,
            max_horizon=2,
            public_history_limit=1,
        )
    assert episode.step_calls == 0


@pytest.mark.parametrize(
    "bad_base_action",
    [
        np.asarray((0.0, 0.0), dtype=np.float32),
        np.asarray((0.0, np.nan, 0.0), dtype=np.float32),
        np.asarray((0.0, 1.01, 0.0), dtype=np.float32),
    ],
)
def test_invalid_frozen_policy_action_fails_before_episode_step(bad_base_action):
    episode = FakeEpisodeAdapter(
        horizon=2,
        base_action_fn=lambda seed, step, observation: bad_base_action,
    )
    with pytest.raises(EpisodeContractError):
        run_frozen_episode(
            episode,
            controller=ConstantController(0.0),
            method="candidate",
            seed=7,
            max_horizon=2,
            public_history_limit=1,
        )
    assert episode.step_calls == 0


def test_repeated_fake_evaluation_is_byte_stable_and_detects_stateful_controller():
    kwargs = {
        "episode_factory": FakeEpisodeFactory(horizon=2),
        "candidate": ConstantController(0.75),
        "incumbent": ConstantController(0.0),
        "seeds": (1, 2),
        "max_horizon": 2,
        "public_history_limit": 2,
        "checks": {"all_contracts": True},
    }
    first = evaluate_paired(**kwargs)
    second = evaluate_paired(**{**kwargs, "episode_factory": FakeEpisodeFactory(horizon=2)})
    assert first["records_sha256"] == second["records_sha256"]
    assert first["records"] == second["records"]

    with pytest.raises(DeterminismError):
        evaluate_paired(
            episode_factory=FakeEpisodeFactory(horizon=3),
            candidate=StatefulNondeterministicController(),
            incumbent=ConstantController(0.0),
            seeds=(3,),
            max_horizon=3,
            public_history_limit=0,
            checks={"all_contracts": True},
        )


def test_comparator_cache_hits_and_every_frozen_identity_change_invalidates(tmp_path):
    seeds = (11, 12)
    identity = _identity(seeds)
    cache_path = tmp_path / "comparators.json"
    first_factory = FakeEpisodeFactory(horizon=2)
    first = evaluate_paired(
        episode_factory=first_factory,
        candidate=ConstantController(1.0),
        incumbent=ConstantController(0.0),
        seeds=seeds,
        max_horizon=2,
        public_history_limit=2,
        checks={"all_contracts": True},
        cache_path=cache_path,
        cache_identity=identity,
    )
    assert first["cache_hit"] is False
    assert cache_path.is_file()

    second_factory = FakeEpisodeFactory(horizon=2)
    second = evaluate_paired(
        episode_factory=second_factory,
        candidate=ConstantController(1.0),
        incumbent=ConstantController(0.0),
        seeds=seeds,
        max_horizon=2,
        public_history_limit=2,
        checks={"all_contracts": True},
        cache_path=cache_path,
        cache_identity=identity,
    )
    assert second["cache_hit"] is True
    assert len(second_factory.episodes) == 4  # candidate only, twice per seed
    assert first["records_sha256"] == second["records_sha256"]

    for field in (
        "checkpoint_sha256",
        "resolved_config_sha256",
        "evaluator_sha256",
        "seed_set_id",
        "environment_contract_sha256",
        "incumbent_sha256",
        "max_horizon",
        "public_history_limit",
    ):
        current = getattr(identity, field)
        replacement = current + 1 if isinstance(current, int) else current + "-changed"
        changed = dataclasses.replace(identity, **{field: replacement})
        assert changed.key != identity.key
        assert load_comparator_cache(
            cache_path,
            identity=changed,
            seeds=seeds,
        ) is None

    changed_seeds = _identity((11, 13))
    assert changed_seeds.key != identity.key
    assert load_comparator_cache(
        cache_path,
        identity=changed_seeds,
        seeds=(11, 13),
    ) is None

    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    envelope["records"] = envelope["records"][:-1]
    envelope["records_sha256"] = records_sha256(envelope["records"])
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert load_comparator_cache(
        cache_path,
        identity=identity,
        seeds=seeds,
    ) is None


def test_mechanical_gate_keeps_only_two_episode_gain_without_worse_capture():
    seeds = (1, 2, 3, 4)
    checks = {"contract": True, "hashes": True, "determinism": True}
    records = _paired_records(seeds, candidate_successes={1, 2}, incumbent_successes=set())
    kept = mechanical_keep_or_discard(records, seeds=seeds, checks=checks)
    assert kept.keep
    assert kept.primary_delta == pytest.approx(0.5)
    assert kept.paired_counts["candidate_only_success"] == 2

    tied = mechanical_keep_or_discard(
        _paired_records(seeds, candidate_successes={1}, incumbent_successes={1}),
        seeds=seeds,
        checks=checks,
    )
    assert not tied.keep
    assert "fewer than 2" in tied.decision_reason

    worse_capture_records = _paired_records(
        seeds,
        candidate_successes={1, 2},
        incumbent_successes=set(),
    )
    worse_capture_records[0]["capture_episode"] = True
    worse_capture = mechanical_keep_or_discard(
        worse_capture_records,
        seeds=seeds,
        checks=checks,
    )
    assert not worse_capture.keep
    assert "capture-episode rate exceeds" in worse_capture.decision_reason

    failed_check = mechanical_keep_or_discard(
        records,
        seeds=seeds,
        checks={**checks, "whitelist": False},
    )
    assert not failed_check.keep
    assert "whitelist" in failed_check.decision_reason


def test_record_validator_rejects_missing_duplicate_and_reordered_records():
    records = _paired_records((1, 2), candidate_successes={1}, incumbent_successes=set())
    validate_episode_records(records, seeds=(1, 2), methods=EVALUATION_METHODS)
    with pytest.raises(CacheValidationError):
        validate_episode_records(records[:-1], seeds=(1, 2), methods=EVALUATION_METHODS)
    with pytest.raises(CacheValidationError):
        validate_episode_records(
            [records[1], records[0], *records[2:]],
            seeds=(1, 2),
            methods=EVALUATION_METHODS,
        )
    duplicate = copy.deepcopy(records)
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    with pytest.raises(CacheValidationError):
        validate_episode_records(duplicate, seeds=(1, 2), methods=EVALUATION_METHODS)


def test_artifact_verification_refuses_missing_and_wrong_checkpoint(tmp_path):
    config = tmp_path / "resolved_config.yaml"
    config.write_text("experiment_id: fake\n", encoding="utf-8")
    missing = tmp_path / "missing.zip"
    with pytest.raises(ArtifactVerificationError, match="missing"):
        verify_exp05_artifacts(missing, config)

    checkpoint = tmp_path / "final_model.zip"
    checkpoint.write_bytes(b"not the registered model")
    with pytest.raises(ArtifactVerificationError, match="checkpoint SHA-256 mismatch"):
        verify_exp05_artifacts(checkpoint, config)

    verified = verify_exp05_artifacts(
        checkpoint,
        config,
        expected_checkpoint_sha256=hashlib.sha256(
            b"not the registered model",
        ).hexdigest(),
        expected_resolved_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert verified.checkpoint_path == checkpoint.resolve()

    with pytest.raises(ArtifactVerificationError, match="resolved config SHA-256 mismatch"):
        verify_exp05_artifacts(
            checkpoint,
            config,
            expected_checkpoint_sha256=verified.checkpoint_sha256,
            expected_resolved_config_sha256="0" * 64,
        )


def test_registered_config_locks_seeds_source_history_gate_and_confirmation(tmp_path):
    config = yaml.safe_load(Path("configs/autoresearch/gaze_dev.yaml").read_text())
    smoke = seed_set_from_config(config, "smoke")
    assert smoke.seeds == tuple(range(1_100_000, 1_100_004))
    factory = FakeEpisodeFactory(horizon=2)
    result = evaluate_paired_from_config(
        config,
        seed_set_name="smoke",
        episode_factory=factory,
        candidate=ConstantController(1.0),
        incumbent=ConstantController(0.0),
        checks={"immutable_hashes": True, "whitelist": True, "leaks": True},
        environment_contract_sha256="environment-contract-test",
        incumbent_sha256="incumbent-test",
        cache_path=tmp_path / "smoke-comparators.json",
        evaluator_digest="evaluator-test",
    )
    assert result["seed_set"]["seed_set_id"] == "gaze-smoke-v1-1100000-4"
    assert result["identity"]["checkpoint_sha256"] == config["source"][
        "checkpoint_sha256"
    ]
    assert result["identity"]["resolved_config_sha256"] == config["source"][
        "resolved_config_sha256"
    ]
    assert result["identity"]["public_history_limit"] == 4
    assert result["decision"]["decision"] == "keep"

    with pytest.raises(EvaluationError, match="authorized confirm"):
        evaluate_paired_from_config(
            config,
            seed_set_name="confirmation",
            episode_factory=factory,
            candidate=ConstantController(1.0),
            incumbent=ConstantController(0.0),
            checks={"all_contracts": True},
            environment_contract_sha256="environment-contract-test",
            incumbent_sha256="incumbent-test",
            evaluator_digest="evaluator-test",
        )

    changed = copy.deepcopy(config)
    changed["candidate"]["public_observation_fields"].append("predator_coordinates")
    with pytest.raises(EvaluationError, match="public observation"):
        seed_set_from_config(changed, "smoke")  # seed resolution alone is harmless
        evaluate_paired_from_config(
            changed,
            seed_set_name="smoke",
            episode_factory=factory,
            candidate=ConstantController(1.0),
            incumbent=ConstantController(0.0),
            checks={"all_contracts": True},
            environment_contract_sha256="environment-contract-test",
            incumbent_sha256="incumbent-test",
            evaluator_digest="evaluator-test",
        )


def test_confirmation_statistics_are_paired_and_deterministic():
    seeds = (1, 2, 3, 4)
    records = _paired_records(
        seeds,
        candidate_successes={1, 2, 3},
        incumbent_successes={1},
    )
    first = confirmation_statistics(records, seeds=seeds, bootstrap_samples=100)
    second = confirmation_statistics(records, seeds=seeds, bootstrap_samples=100)
    assert first == second
    assert first["mean_delta"] == pytest.approx(0.5)
    assert first["candidate_only_successes"] == 2
    assert first["incumbent_only_successes"] == 0
    assert first["mcnemar_exact_p"] == pytest.approx(0.5)


def test_action_trace_is_part_of_determinism_identity():
    first = [_record(1, "candidate", success=True)]
    second = copy.deepcopy(first)
    second[0]["action_trace_sha256"] = "different"
    assert records_sha256(first) != records_sha256(second)
    with pytest.raises(DeterminismError):
        assert_deterministic_records(first, second)
