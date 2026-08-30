import json
import subprocess
from pathlib import Path

import pytest

from autoresearch.evaluator import DeterminismError, FakeEpisodeFactory
from autoresearch.guard import CandidateRuntimeError
from autoresearch.sharding import (
    BASELINE_MODE,
    EXPERIMENT_MODE,
    ShardingError,
    aggregate_shards,
    build_parser,
    config_from_mapping,
    environment_contract_sha256,
    registered_shard,
    rollout_shard,
    shard_paths,
    validate_registered_partition,
)
from autoresearch.worker import IsolatedCandidateController


INCUMBENT_SOURCE = '''
"""Fake guarded incumbent."""


class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, base_head_action, step_index
        return 0.0
'''


CANDIDATE_SOURCE = '''
"""Fake guarded candidate."""


class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, base_head_action, step_index
        return 1.0
'''


INFINITE_CANDIDATE_SOURCE = '''
class CandidateGazeController:
    def __init__(self):
        while True:
            pass

    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, base_head_action, step_index
        return 0.0
'''


def _config():
    return {
        "schema_version": 1,
        "experiment_id": "sharding-test",
        "source": {
            "checkpoint_path": "missing-for-fake/checkpoint.zip",
            "checkpoint_sha256": "checkpoint-test",
            "resolved_config_path": "missing-for-fake/config.yaml",
            "resolved_config_sha256": "resolved-config-test",
        },
        "candidate": {
            "path": "autoresearch/candidate.py",
            "class_name": "CandidateGazeController",
            "public_observation_fields": [
                "image_left",
                "image_right",
                "proprio",
                "previous_action",
            ],
            "public_history_length": 1,
        },
        "seed_sets": {
            "development": {
                "id": "development-test-20-4",
                "seed_start": 20,
                "episodes": 4,
                "purpose": "fake",
            },
            "confirmation": {
                "id": "confirmation-test-40-4",
                "seed_start": 40,
                "episodes": 4,
                "purpose": "fake-one-time",
                "one_time": True,
                "requires_explicit_authorization": True,
            },
        },
        "evaluation": {
            "deterministic_policy": True,
            "maximum_history_length": 1,
            "bootstrap_samples": 100,
            "methods": [
                "candidate",
                "search_incumbent",
                "fixed_p60_research_reference",
            ],
        },
        "decision": {
            "primary_metric": "paired_clean_success_delta",
            "minimum_paired_episode_improvement": 2,
            "capture_rate_must_not_exceed_incumbent": True,
            "ties_keep_incumbent": True,
        },
    }


def _write_sources(tmp_path):
    incumbent = tmp_path / "incumbent.py"
    candidate = tmp_path / "candidate.py"
    incumbent.write_text(INCUMBENT_SOURCE, encoding="utf-8")
    candidate.write_text(CANDIDATE_SOURCE, encoding="utf-8")
    return candidate, incumbent


def _rollout_methods(
    *,
    config,
    output_dir,
    methods,
    candidate,
    incumbent,
    shard_count=2,
    factory_fn=None,
):
    for method in methods:
        for shard_index in range(shard_count):
            rollout_shard(
                config=config,
                seed_set_name="development",
                method=method,
                shard_index=shard_index,
                shard_count=shard_count,
                output_dir=output_dir,
                candidate_source=candidate,
                incumbent_source=incumbent,
                project_root=Path.cwd(),
                repeat=2,
                episode_factory=(factory_fn or (lambda: FakeEpisodeFactory(horizon=2)))(),
                environment_digest="environment-test",
            )


def _build_baseline_and_cache(tmp_path, *, config, incumbent, shard_count=2):
    baseline_dir = tmp_path / "baseline"
    cache = tmp_path / "comparator-cache.json"
    _rollout_methods(
        config=config,
        output_dir=baseline_dir,
        methods=("incumbent", "fixed_p60"),
        candidate=incumbent,
        incumbent=incumbent,
        shard_count=shard_count,
    )
    result = aggregate_shards(
        config=config,
        seed_set_name="development",
        mode=BASELINE_MODE,
        shard_count=shard_count,
        output_dir=baseline_dir,
        candidate_source=incumbent,
        incumbent_source=incumbent,
        comparator_cache_path=cache,
        max_horizon=2,
        project_root=Path.cwd(),
        environment_digest="environment-test",
    )
    return baseline_dir, cache, result


def test_registered_partition_uses_only_config_seeds_with_exact_order_and_coverage():
    config = _config()
    shards = validate_registered_partition(
        config,
        seed_set_name="development",
        shard_count=3,
    )

    assert [shard.seeds for shard in shards] == [(20, 21), (22,), (23,)]
    assert tuple(seed for shard in shards for seed in shard.seeds) == (20, 21, 22, 23)
    assert len({seed for shard in shards for seed in shard.seeds}) == 4
    with pytest.raises(ShardingError, match="may not exceed"):
        validate_registered_partition(
            config,
            seed_set_name="development",
            shard_count=5,
        )
    with pytest.raises(ShardingError, match="outside"):
        registered_shard(
            config,
            seed_set_name="development",
            shard_index=2,
            shard_count=2,
        )


def test_environment_identity_covers_reward_policy_util_and_cellworld_assets(tmp_path):
    required = (
        "benchmarks/peekbench/environment.py",
        "botevade_gym.py",
        "first_person.py",
        "policies/binocular_sac.py",
        "reward.py",
        "training/first_person_sac.py",
        "util.py",
        "cellworld_game-main/cellworld_game/model.py",
        "cellworld_cache/world_configuration/hexagonal",
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))

    first = environment_contract_sha256(tmp_path)
    (tmp_path / "reward.py").write_text("changed reward\n", encoding="utf-8")
    second = environment_contract_sha256(tmp_path)
    (tmp_path / "cellworld_cache/world_configuration/hexagonal").write_text(
        "changed world asset\n",
        encoding="utf-8",
    )
    third = environment_contract_sha256(tmp_path)
    assert len({first, second, third}) == 3


def test_confirmation_is_unconditionally_refused_by_independent_sharding():
    config = _config()
    with pytest.raises(ShardingError, match="CONFIRMATION REFUSED"):
        registered_shard(
            config,
            seed_set_name="confirmation",
            shard_index=0,
            shard_count=2,
        )

def test_repeat_two_checks_each_record_and_writes_atomic_jsonl_manifest(tmp_path):
    config = config_from_mapping(_config())
    candidate, incumbent = _write_sources(tmp_path)
    output = tmp_path / "rollout"

    result = rollout_shard(
        config=config,
        seed_set_name="development",
        method="candidate",
        shard_index=0,
        shard_count=2,
        output_dir=output,
        candidate_source=candidate,
        incumbent_source=incumbent,
        project_root=Path.cwd(),
        repeat=2,
        episode_factory=FakeEpisodeFactory(horizon=2),
        environment_digest="environment-test",
    )

    paths = shard_paths(output, method="candidate", shard_index=0, shard_count=2)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in paths.records.read_text().splitlines()]
    assert result["seed_count"] == 2
    assert manifest["determinism_verified"] is True
    assert manifest["repeat"] == 2
    assert [record["seed"] for record in records] == [20, 21]
    assert all(record["determinism_repeat"] == 2 for record in records)
    assert all(record["identity"] == manifest["identity"] for record in records)
    required_hashes = {
        "candidate_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "contract_sha256",
        "environment_contract_sha256",
        "evaluator_sha256",
        "guard_sha256",
        "incumbent_sha256",
        "ordered_seed_sha256",
        "resolved_config_sha256",
        "run_identity_sha256",
        "shard_identity_sha256",
        "shard_seed_sha256",
        "sharding_sha256",
        "worker_sha256",
    }
    assert required_hashes <= set(manifest["identity"])
    assert not list(paths.records.parent.glob(".*.tmp"))


def test_repeat_two_detects_per_seed_nondeterminism(tmp_path):
    config = _config()
    candidate, incumbent = _write_sources(tmp_path)
    calls = {"count": 0}

    def changing_outcome(seed, actions, initial_yaw):
        del seed, actions, initial_yaw
        calls["count"] += 1
        success = calls["count"] % 2 == 1
        return {
            "clean_success": success,
            "capture_episode": not success,
            "goal_reached": success,
        }

    with pytest.raises(DeterminismError):
        rollout_shard(
            config=config,
            seed_set_name="development",
            method="candidate",
            shard_index=0,
            shard_count=2,
            output_dir=tmp_path / "nondeterministic",
            candidate_source=candidate,
            incumbent_source=incumbent,
            project_root=Path.cwd(),
            repeat=2,
            episode_factory=FakeEpisodeFactory(
                horizon=2,
                outcome_fn=changing_outcome,
            ),
            environment_digest="environment-test",
        )


def test_rollout_isolates_and_times_out_infinite_candidate_constructor(
    tmp_path,
    monkeypatch,
):
    config = _config()
    _, incumbent = _write_sources(tmp_path)
    infinite = tmp_path / "infinite.py"
    infinite.write_text(INFINITE_CANDIDATE_SOURCE, encoding="utf-8")

    real_from_source = IsolatedCandidateController.from_source

    def fast_startup_timeout(source):
        return real_from_source(source, startup_timeout_seconds=0.15)

    monkeypatch.setattr(
        IsolatedCandidateController,
        "from_source",
        fast_startup_timeout,
    )

    with pytest.raises(CandidateRuntimeError, match="timed out"):
        rollout_shard(
            config=config,
            seed_set_name="development",
            method="candidate",
            shard_index=0,
            shard_count=2,
            output_dir=tmp_path / "timeout",
            candidate_source=infinite,
            incumbent_source=incumbent,
            project_root=Path.cwd(),
            repeat=2,
            episode_factory=FakeEpisodeFactory(horizon=2),
            environment_digest="environment-test",
        )
    paths = shard_paths(
        tmp_path / "timeout",
        method="candidate",
        shard_index=0,
        shard_count=2,
    )
    assert not paths.manifest.exists()


def test_aggregate_uses_static_guard_without_spawning_candidate(tmp_path, monkeypatch):
    config = _config()
    _, incumbent = _write_sources(tmp_path)
    baseline_dir = tmp_path / "static-aggregate"
    cache = tmp_path / "static-cache.json"
    _rollout_methods(
        config=config,
        output_dir=baseline_dir,
        methods=("incumbent", "fixed_p60"),
        candidate=incumbent,
        incumbent=incumbent,
    )

    def forbidden_spawn(*args, **kwargs):
        del args, kwargs
        raise AssertionError("aggregate must not spawn editable candidate code")

    monkeypatch.setattr(IsolatedCandidateController, "from_source", forbidden_spawn)
    result = aggregate_shards(
        config=config,
        seed_set_name="development",
        mode=BASELINE_MODE,
        shard_count=2,
        output_dir=baseline_dir,
        candidate_source=incumbent,
        incumbent_source=incumbent,
        comparator_cache_path=cache,
        max_horizon=2,
        project_root=Path.cwd(),
        environment_digest="environment-test",
    )
    assert result["decision"] == "baseline_cached"


def test_baseline_cache_then_candidate_only_experiment_produces_strict_gate(tmp_path):
    config = _config()
    candidate, incumbent = _write_sources(tmp_path)
    baseline_dir, cache, baseline = _build_baseline_and_cache(
        tmp_path,
        config=config,
        incumbent=incumbent,
    )
    assert baseline["decision"] == "baseline_cached"
    assert cache.is_file()
    assert (baseline_dir / "summary.json").is_file()

    experiment_dir = tmp_path / "experiment"
    _rollout_methods(
        config=config,
        output_dir=experiment_dir,
        methods=("candidate",),
        candidate=candidate,
        incumbent=incumbent,
    )
    result = aggregate_shards(
        config=config,
        seed_set_name="development",
        mode=EXPERIMENT_MODE,
        shard_count=2,
        output_dir=experiment_dir,
        candidate_source=candidate,
        incumbent_source=incumbent,
        comparator_cache_path=cache,
        max_horizon=2,
        project_root=Path.cwd(),
        environment_digest="environment-test",
    )

    gate = json.loads((experiment_dir / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (experiment_dir / "aggregate.manifest.json").read_text(encoding="utf-8"),
    )
    records = [
        json.loads(line)
        for line in (experiment_dir / "records.jsonl").read_text().splitlines()
    ]
    assert result["decision"] == gate["decision"] == "keep"
    assert gate["clean_success_episode_delta"] == 4
    assert [(record["seed"], record["method"]) for record in records] == [
        (seed, method)
        for seed in range(20, 24)
        for method in ("candidate", "incumbent", "fixed_p60")
    ]
    assert manifest["checks"] == {
        "comparator_cache_identity": True,
        "determinism": True,
        "identity_hashes": True,
        "records_complete": True,
        "shard_coverage": True,
        "source_guard": True,
    }
    assert manifest["seed_set"]["spent_state_was_mutated"] is False
    for name in ("records", "summary", "gate", "comparator_cache"):
        artifact = Path(manifest["artifacts"][name])
        assert not artifact.is_absolute()
        assert (experiment_dir / artifact).is_file()


def test_aggregate_rejects_missing_shard_and_tampered_seed_order(tmp_path):
    config = _config()
    candidate, incumbent = _write_sources(tmp_path)
    output = tmp_path / "baseline"
    _rollout_methods(
        config=config,
        output_dir=output,
        methods=("incumbent", "fixed_p60"),
        candidate=incumbent,
        incumbent=incumbent,
    )
    missing = shard_paths(output, method="fixed_p60", shard_index=1, shard_count=2)
    missing.manifest.unlink()
    with pytest.raises(ShardingError, match="manifest is missing"):
        aggregate_shards(
            config=config,
            seed_set_name="development",
            mode=BASELINE_MODE,
            shard_count=2,
            output_dir=output,
            candidate_source=incumbent,
            incumbent_source=incumbent,
            comparator_cache_path=tmp_path / "cache.json",
            max_horizon=2,
            project_root=Path.cwd(),
            environment_digest="environment-test",
        )

    # Regenerate, then change only the manifest's registered order.  Aggregate
    # rejects it before consuming any outcomes.
    rollout_shard(
        config=config,
        seed_set_name="development",
        method="fixed_p60",
        shard_index=1,
        shard_count=2,
        output_dir=output,
        candidate_source=incumbent,
        incumbent_source=incumbent,
        project_root=Path.cwd(),
        repeat=2,
        episode_factory=FakeEpisodeFactory(horizon=2),
        environment_digest="environment-test",
    )
    manifest = json.loads(missing.manifest.read_text(encoding="utf-8"))
    manifest["shard"]["seeds"].reverse()
    missing.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShardingError, match="coverage/order"):
        aggregate_shards(
            config=config,
            seed_set_name="development",
            mode=BASELINE_MODE,
            shard_count=2,
            output_dir=output,
            candidate_source=incumbent,
            incumbent_source=incumbent,
            comparator_cache_path=tmp_path / "cache.json",
            max_horizon=2,
            project_root=Path.cwd(),
            environment_digest="environment-test",
        )


def test_comparator_cache_invalidates_when_incumbent_source_changes(tmp_path):
    config = _config()
    candidate, incumbent = _write_sources(tmp_path)
    _, cache, _ = _build_baseline_and_cache(
        tmp_path,
        config=config,
        incumbent=incumbent,
    )

    incumbent.write_text(INCUMBENT_SOURCE + "\n", encoding="utf-8")
    experiment_dir = tmp_path / "changed-incumbent"
    _rollout_methods(
        config=config,
        output_dir=experiment_dir,
        methods=("candidate",),
        candidate=candidate,
        incumbent=incumbent,
    )
    with pytest.raises(ShardingError, match="missing, corrupt, or stale"):
        aggregate_shards(
            config=config,
            seed_set_name="development",
            mode=EXPERIMENT_MODE,
            shard_count=2,
            output_dir=experiment_dir,
            candidate_source=candidate,
            incumbent_source=incumbent,
            comparator_cache_path=cache,
            max_horizon=2,
            project_root=Path.cwd(),
            environment_digest="environment-test",
        )


def test_cli_has_no_seed_horizon_or_artifact_override_and_scripts_are_safe():
    parser = build_parser()
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["rollout"]._actions
        for option in action.option_strings
    }
    assert "--seed-start" not in option_strings
    assert "--episodes" not in option_strings
    assert "--max-horizon" not in option_strings
    assert "--checkpoint" not in option_strings
    assert "--resolved-config" not in option_strings
    assert "--authorize-one-time-confirmation" not in option_strings

    root = Path(__file__).resolve().parents[1]
    shard_script = (root / "setup/autoresearch_gaze_shards.sbatch").read_text()
    aggregate_script = (root / "setup/autoresearch_gaze_aggregate.sbatch").read_text()
    submit_script = (root / "setup/submit_autoresearch_gaze.sh").read_text()
    for script in (shard_script, aggregate_script):
        assert "#SBATCH --ntasks=1" in script
        assert "#SBATCH --cpus-per-task=1" in script
        assert "#SBATCH --mem=8G" in script
        assert "unset OPENROUTER_API_KEY" in script
        assert '--seed-set "${seed_set}"' in script
        assert 'seed_set="development"' in script
        assert "AUTORESEARCH_CANDIDATE_SHA256" in script
        assert "AUTORESEARCH_INCUMBENT_SHA256" in script
        assert "expected_candidate_source" in script
        assert "expected_incumbent_source" in script
        assert "${incumbent_sha256}" in script
        assert "authorize-one-time-confirmation" not in script
    assert "git status --porcelain" in submit_script
    assert "git diff --quiet" in submit_script
    assert "git diff --cached --quiet" in submit_script
    assert "Quest main checkout has tracked modifications" in submit_script
    assert "origin/${current_branch}" in submit_script
    assert 'git worktree add --quiet --detach "${project_dir}" "${submitted_commit}"' in submit_script
    assert "git switch" not in submit_script
    assert "Mice-autoresearch-worktrees" in submit_script
    assert 'ensure_results_link "${project_dir}/results/autoresearch"' in submit_script
    assert 'ensure_results_link "${project_dir}/results/sac/sac_cnn_active_gaze_9903898"' in submit_script
    assert "ssh -O check" in submit_script
    assert "Submitted autoresearch ${mode} shards" in submit_script
    assert "Submitted autoresearch ${mode} aggregate" in submit_script
    assert "authorize-one-time-confirmation" not in submit_script
    assert "--candidate-source" not in submit_script
    assert "--incumbent-source" not in submit_script
    for option in (
        "--candidate-commit",
        "--candidate-sha256",
        "--incumbent-commit",
        "--incumbent-sha256",
    ):
        assert option in submit_script
    assert 'git show "${source_commit}:autoresearch/candidate.py"' in submit_script
    assert 'seed_set="development"' in submit_script
    assert 'incumbents/${incumbent_sha256}.py' in submit_script
    assert 'comparator_caches/${incumbent_sha256}.json' in submit_script
    assert 'submitted_branch="$4"' in submit_script
    assert 'candidate_commit="$5"' in submit_script
    assert 'shard_count="$9"' in submit_script
    assert 'remote_project="${10}"' in submit_script
    assert 'submitted_commit="${candidate_commit}"' in submit_script
    assert 'remote_project="${11}"' not in submit_script

    usage = subprocess.run(
        ["bash", str(root / "setup/submit_autoresearch_gaze.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert usage.returncode == 2
    assert "--candidate-commit" in usage.stderr
    assert "--candidate-sha256" in usage.stderr
    assert "--incumbent-commit" in usage.stderr
    assert "--incumbent-sha256" in usage.stderr
    assert "--candidate-source" not in usage.stderr
    assert "--incumbent-source" not in usage.stderr
