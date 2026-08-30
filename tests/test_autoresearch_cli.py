from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
import numpy as np

import autoresearch.runner as runner_module
from autoresearch.__main__ import build_parser, main
from autoresearch.evaluator import FakeEpisodeFactory, evaluate_paired_from_config
from autoresearch.runner import (
    AutoresearchRunner,
    ConfirmationError,
    RunContractError,
    SetupError,
)
from autoresearch.ledger import ExperimentLedger
from autoresearch.sharding import (
    aggregate_shards,
    load_registered_config,
    rollout_shard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ZERO_CANDIDATE = '''\
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

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

BETTER_CANDIDATE = '''\
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

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

WORSE_CANDIDATE = '''\
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, base_head_action, step_index
        return -1.0
'''

INVALID_CANDIDATE = '''\
import os

class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        return float(os.getpid())
'''

INFINITE_CANDIDATE = '''\
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        while True:
            pass
'''


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(repo: Path, *, max_experiments: int = 2) -> Path:
    checkpoint = repo / "artifacts" / "checkpoint.zip"
    resolved = repo / "artifacts" / "resolved.yaml"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake-frozen-checkpoint")
    resolved.write_text("environment:\n  max_step: 2\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "experiment_id": "test-phase1",
        "results_root": "results/autoresearch",
        "source": {
            "resolved_config_path": "artifacts/resolved.yaml",
            "resolved_config_sha256": _digest(resolved),
            "checkpoint_path": "artifacts/checkpoint.zip",
            "checkpoint_sha256": _digest(checkpoint),
            "historical_exp05_seed_start": 1000,
            "historical_exp05_episodes": 10,
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
            "public_history_length": 2,
            "initial_incumbent_id": "legal_fixed_scan_v1",
            "initial_incumbent_sha256": _digest(
                repo / "autoresearch" / "candidate.py"
            ),
            "mutable_paths": ["autoresearch/candidate.py"],
        },
        "seed_sets": {
            "smoke": {
                "id": "smoke-v1",
                "seed_start": 2000,
                "episodes": 4,
                "purpose": "contract_and_determinism_only",
            },
            "development": {
                "id": "development-v1",
                "seed_start": 3000,
                "episodes": 4,
                "purpose": "reusable_engineering_selection",
            },
            "confirmation": {
                "id": "confirmation-v1",
                "seed_start": 4000,
                "episodes": 6,
                "purpose": "one_time_confirmatory_gate",
                "one_time": True,
                "requires_explicit_authorization": True,
            },
            "rationale": "frozen before outcomes",
        },
        "evaluation": {
            "deterministic_policy": True,
            "maximum_history_length": 2,
            "bootstrap_samples": 100,
            "methods": [
                "candidate",
                "search_incumbent",
                "fixed_p60_research_reference",
            ],
            "fixed_p60_is_prepositioned_reference": True,
        },
        "decision": {
            "primary_metric": "paired_clean_success_delta",
            "minimum_paired_episode_improvement": 2,
            "capture_rate_must_not_exceed_incumbent": True,
            "ties_keep_incumbent": True,
            "confirmation_interval_must_exclude_zero": True,
            "confirmation_capture_must_not_worsen": True,
        },
        "budget": {
            "max_experiments": max_experiments,
            "max_wall_seconds": 14400,
            "max_consecutive_crashes": 3,
            "on_exhausted": "stop_successfully_and_report",
        },
        "claims": {
            "phase1_scope": "engineering_selection",
            "scientific_active_gaze_verification": False,
        },
    }
    config_path = repo / "configs" / "autoresearch" / "gaze_dev.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def harness(tmp_path):
    repo = tmp_path / "repo"
    candidate = repo / "autoresearch" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    config_path = _write_config(repo)
    phases: list[tuple[str, bool]] = []

    def recording_evaluator(config, **kwargs):
        phases.append(
            (kwargs["seed_set_name"], bool(kwargs["allow_confirmation"]))
        )
        return evaluate_paired_from_config(config, **kwargs)

    runner = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=(),
        episode_factory_builder=lambda config, project_root: FakeEpisodeFactory(
            horizon=2
        ),
        paired_evaluator=recording_evaluator,
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (repo / path).read_bytes(),
    )
    return repo, candidate, config_path, runner, phases


def _setup_and_baseline(harness, run_tag="unit"):
    repo, candidate, config_path, runner, phases = harness
    runner.setup(config_path=config_path, run_tag=run_tag)
    baseline = runner.baseline(run_tag=run_tag)
    assert baseline["status"] == "keep"
    return repo, candidate, runner, phases


def test_parser_has_separate_confirmation_gate_and_no_experiment_seed_override():
    parser = build_parser()
    experiment = parser.parse_args(
        [
            "experiment",
            "--run-tag",
            "run",
            "--hypothesis-file",
            "hypothesis.md",
        ]
    )
    assert experiment.command == "experiment"
    assert not hasattr(experiment, "seed_set")
    confirmation = parser.parse_args(["confirm", "--run-tag", "run"])
    assert confirmation.authorized is False
    confirmation = parser.parse_args(
        ["confirm", "--run-tag", "run", "--authorize-confirmation"]
    )
    assert confirmation.authorized is True
    abort = parser.parse_args(
        [
            "abort-external",
            "--run-tag",
            "run",
            "--experiment-id",
            "E0001",
            "--reason",
            "Slurm dependency failed",
        ]
    )
    assert abort.experiment_id == "E0001"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "experiment",
                "--run-tag",
                "run",
                "--prepare-external",
                "--finalize-external",
                "aggregate.manifest.json",
            ]
        )


def test_setup_freezes_manifest_and_status_json_is_compact(harness, capsys):
    repo, _, config_path, runner, _ = harness
    manifest = runner.setup(config_path=config_path, run_tag="manifest")
    run_dir = repo / "results" / "autoresearch" / "manifest"
    assert manifest["claims"]["scope"] == "engineering_selection"
    assert manifest["seed_sets"]["smoke"]["seed_end"] < manifest["seed_sets"][
        "development"
    ]["seed_start"]
    assert (run_dir / "run.sha256").read_text().strip() == hashlib.sha256(
        (run_dir / "run.json").read_bytes()
    ).hexdigest()

    assert main(["status", "--run-tag", "manifest", "--json"], runner=runner) == 0
    output = capsys.readouterr().out.strip()
    status = json.loads(output)
    assert status["state"] == "needs_baseline"
    assert status["next_action"] == "baseline"
    assert len(output) < 3000


def test_seed_sets_must_be_disjoint_and_confirmation_must_be_one_time(tmp_path):
    repo = tmp_path / "repo"
    (repo / "autoresearch").mkdir(parents=True)
    (repo / "autoresearch" / "candidate.py").write_text(
        ZERO_CANDIDATE, encoding="utf-8"
    )
    config_path = _write_config(repo)
    config = yaml.safe_load(config_path.read_text())
    config["seed_sets"]["confirmation"]["seed_start"] = 3002
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    runner = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=(),
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (repo / path).read_bytes(),
    )
    with pytest.raises(SetupError, match="overlap"):
        runner.setup(config_path=config_path, run_tag="overlap")


def test_setup_binds_initial_incumbent_to_worktree_and_setup_commit(tmp_path):
    repo = tmp_path / "repo"
    candidate = repo / "autoresearch" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    config_path = _write_config(repo)
    wrong_commit = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=(),
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (
            BETTER_CANDIDATE.encode()
            if path == "autoresearch/candidate.py"
            else (repo / path).read_bytes()
        ),
    )
    with pytest.raises(SetupError, match="setup commit candidate bytes"):
        wrong_commit.setup(config_path=config_path, run_tag="wrong-commit")

    runner = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=(),
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (
            ZERO_CANDIDATE.encode()
            if path == "autoresearch/candidate.py"
            else (repo / path).read_bytes()
        ),
        episode_factory_builder=lambda config, project_root: FakeEpisodeFactory(
            horizon=2
        ),
    )
    runner.setup(config_path=config_path, run_tag="bound")
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    with pytest.raises(RunContractError, match="registered legal-scan"):
        runner.baseline(run_tag="bound")


def test_candidate_loader_executes_the_single_hashed_source_read(
    tmp_path, monkeypatch
):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    original_loader = runner_module.IsolatedCandidateController.from_source

    class MutatingIsolatedController:
        @classmethod
        def from_source(cls, source, **kwargs):
            candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
            return original_loader(source, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "IsolatedCandidateController",
        MutatingIsolatedController,
    )
    runner = AutoresearchRunner(repo_root=tmp_path, immutable_paths=())
    controller, digest, source = runner._validate_and_load_candidate(candidate)
    try:
        controller.reset(episode_seed=1)
        observation = {
            "image_left": np.zeros((2, 2, 3), dtype=np.uint8),
            "image_right": np.zeros((2, 2, 3), dtype=np.uint8),
            "proprio": np.zeros((3,), dtype=np.float32),
            "previous_action": np.zeros((3,), dtype=np.float32),
        }
        action = controller.head_action(
            observation=observation,
            public_history=(),
            base_head_action=0.0,
            step_index=0,
        )
        assert action == 0.0
        assert digest == hashlib.sha256(source).hexdigest()
        assert candidate.read_text() == BETTER_CANDIDATE
    finally:
        controller.close()


def test_environment_contract_is_rehashed_before_status_or_cache_use(tmp_path):
    repo = tmp_path / "repo"
    candidate = repo / "autoresearch" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    config_path = _write_config(repo)
    environment = {"digest": "d" * 64}
    runner = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=(),
        environment_contract_provider=lambda root: environment["digest"],
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (repo / path).read_bytes(),
    )
    runner.setup(config_path=config_path, run_tag="environment")
    environment["digest"] = "e" * 64
    with pytest.raises(RunContractError, match="environment contract"):
        runner.status(run_tag="environment")


def test_one_good_and_one_bad_candidate_make_mechanical_decisions_and_stop_budget(
    harness,
):
    repo, candidate, runner, phases = _setup_and_baseline(harness)
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("Holding positive gaze should improve clean success.\n")

    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    better = runner.experiment(
        run_tag="unit",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert better["experiment_id"] == "E0001"
    assert better["status"] == "keep"
    assert better["primary_delta"] == 1.0

    candidate.write_text(WORSE_CANDIDATE, encoding="utf-8")
    worse = runner.experiment(
        run_tag="unit",
        hypothesis_file=hypothesis,
        candidate_commit="c" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert worse["experiment_id"] == "E0002"
    assert worse["status"] == "discard"
    assert runner.status(run_tag="unit")["incumbent"]["experiment_id"] == "E0001"

    stopped = runner.experiment(
        run_tag="unit",
        hypothesis_file=hypothesis,
        candidate_commit="d" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert stopped["status"] == "budget_exhausted"
    assert phases == [
        ("smoke", False),
        ("development", False),
        ("smoke", False),
        ("development", False),
        ("smoke", False),
        ("development", False),
    ]


def test_contract_failure_and_changed_path_gate_never_call_confirmation(harness):
    repo, candidate, runner, phases = _setup_and_baseline(harness, "contract")
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("This deliberately invalid candidate must be rejected.\n")
    calls_before = len(phases)
    candidate.write_text(INVALID_CANDIDATE, encoding="utf-8")
    failed = runner.experiment(
        run_tag="contract",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert failed["status"] == "contract_failure"
    assert len(phases) == calls_before
    assert all(name != "confirmation" for name, _ in phases)

    # A fresh run isolates the changed-path failure from the first terminal ID.
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    second_config = _write_config(repo)
    runner.setup(config_path=second_config, run_tag="paths")
    assert runner.baseline(run_tag="paths")["status"] == "keep"
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    failed = runner.experiment(
        run_tag="paths",
        hypothesis_file=hypothesis,
        candidate_commit="c" * 40,
        changed_paths=["autoresearch/evaluator.py"],
    )
    assert failed["status"] == "contract_failure"
    assert "outside the whitelist" in failed["decision_reason"]


def test_git_changed_path_discovery_includes_deletions_and_rejects_extra_path(
    harness, monkeypatch
):
    repo, candidate, runner, _ = _setup_and_baseline(harness, "deleted")
    captured = {}

    def fake_git(arguments, *, binary=False):
        captured["arguments"] = tuple(arguments)
        return "autoresearch/candidate.py\ndocs/deleted.md\n"

    monkeypatch.setattr(runner, "_git", fake_git)
    assert tuple(runner._changed_paths("a" * 40, "b" * 40)) == (
        "autoresearch/candidate.py",
        "docs/deleted.md",
    )
    assert "--diff-filter=ACDMRTUXB" in captured["arguments"]
    assert "--no-renames" in captured["arguments"]

    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("A candidate plus deleted source must be rejected.\n")
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    runner.changed_paths_provider = lambda parent, commit: (
        "autoresearch/candidate.py",
        "docs/deleted.md",
    )
    failed = runner.experiment(
        run_tag="deleted",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
    )
    assert failed["status"] == "contract_failure"
    assert "outside the whitelist" in failed["decision_reason"]


def test_real_source_path_and_hash_failures_are_clear_before_evaluation(harness):
    repo, _, config_path, runner, phases = harness
    runner.setup(config_path=config_path, run_tag="source")
    (repo / "artifacts" / "checkpoint.zip").unlink()
    baseline = runner.baseline(run_tag="source")
    assert baseline["status"] == "contract_failure"
    assert "checkpoint is missing" in baseline["decision_reason"]
    assert phases == []


def test_runner_turns_infinite_candidate_into_bounded_contract_failure(harness):
    import time

    repo, candidate, runner, _ = _setup_and_baseline(harness, "timeout")
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("An infinite candidate must time out safely.\n")
    candidate.write_text(INFINITE_CANDIDATE, encoding="utf-8")
    started = time.monotonic()
    result = runner.experiment(
        run_tag="timeout",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert time.monotonic() - started < 4.0
    assert result["status"] == "contract_failure"
    assert "timed out" in result["decision_reason"]


def test_confirmation_is_explicit_one_time_report_only_and_preserves_incumbent(harness):
    repo, candidate, runner, phases = _setup_and_baseline(harness, "confirm")
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("Positive gaze should improve paired clean success.\n")
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    selected = runner.experiment(
        run_tag="confirm",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert selected["status"] == "keep"

    with pytest.raises(ConfirmationError, match="explicit"):
        runner.confirm(run_tag="confirm", authorized=False)
    result = runner.confirm(run_tag="confirm", authorized=True)
    assert result["status"] == "discard"  # report-only; never promotes C0001
    assert result["confirmation_passed"] is True
    assert runner.status(run_tag="confirm")["incumbent"]["experiment_id"] == "E0001"
    confirmation_status = runner.status(run_tag="confirm")["confirmation"]
    assert confirmation_status["state"] == "spent"
    assert confirmation_status["status"] == "discard"
    assert phases[-1] == ("confirmation", True)
    with pytest.raises(ConfirmationError, match="already spent"):
        runner.confirm(run_tag="confirm", authorized=True)


def test_worker_context_exposes_archived_incumbent_and_frozen_identities(harness):
    repo, candidate, runner, _ = _setup_and_baseline(harness, "worker")
    context = runner.worker_context(
        run_tag="worker", candidate_path=candidate, candidate_commit="b" * 40
    )
    assert context["config_path"].endswith("configs/autoresearch/gaze_dev.yaml")
    assert context["candidate"]["sha256"] == _digest(candidate)
    assert context["incumbent"]["experiment_id"] == "legal_fixed_scan_v1"
    assert Path(context["incumbent"]["path"]).is_file()
    assert len(context["environment_contract_sha256"]) == 64
    assert len(context["evaluator_sha256"]) == 64


def test_explicit_external_abort_preserves_artifacts_and_counts_as_crash(harness):
    repo, candidate, runner, _ = _setup_and_baseline(harness, "abort")
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("Prepare a candidate whose Slurm job will fail.\n")
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    prepared = runner.prepare_external_experiment(
        run_tag="abort",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert prepared["status"] == "running"
    with pytest.raises(RunContractError, match="leak scan"):
        runner.abort_external(
            run_tag="abort",
            experiment_id=prepared["experiment_id"],
            reason="exact_state must not enter evidence",
        )

    aborted = runner.abort_external(
        run_tag="abort",
        experiment_id=prepared["experiment_id"],
        reason="Slurm afterok dependency produced no aggregate",
    )
    assert aborted["status"] == "crash"
    assert aborted["recovery_action"] == "explicit_external_abort"
    artifact_dir = (
        repo
        / "results"
        / "autoresearch"
        / "abort"
        / "artifacts"
        / prepared["experiment_id"]
    )
    assert (artifact_dir / "candidate.py").is_file()
    assert (artifact_dir / "smoke.json").is_file()
    assert (artifact_dir / "external-abort.json").is_file()
    status = runner.status(run_tag="abort")
    assert status["incumbent"]["experiment_id"] == "legal_fixed_scan_v1"
    assert status["last_experiment"]["status"] == "crash"
    assert status["budget"]["consecutive_crashes"] == 1
    with pytest.raises(RunContractError, match="only a running"):
        runner.abort_external(
            run_tag="abort",
            experiment_id=prepared["experiment_id"],
            reason="duplicate abort",
        )


def test_external_quest_aggregate_is_recomputed_before_ledger_promotion(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    candidate = repo / "autoresearch" / "candidate.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(ZERO_CANDIDATE, encoding="utf-8")
    immutable = (
        "autoresearch/contract.py",
        "autoresearch/evaluator.py",
        "autoresearch/guard.py",
        "autoresearch/sharding.py",
        "autoresearch/worker.py",
    )
    for relative in immutable:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    config_path = _write_config(repo)
    environment_digest = "d" * 64
    runner = AutoresearchRunner(
        repo_root=repo,
        immutable_paths=immutable,
        episode_factory_builder=lambda config, project_root: FakeEpisodeFactory(
            horizon=2
        ),
        environment_contract_provider=lambda root: environment_digest,
        commit_provider=lambda: "a" * 40,
        committed_file_provider=lambda commit, path: (repo / path).read_bytes(),
    )
    runner.setup(config_path=config_path, run_tag="quest")
    loaded = load_registered_config(config_path, project_root=repo)
    prepared_baseline = runner.prepare_external_baseline(run_tag="quest")
    assert prepared_baseline["status"] == "running"
    assert runner.status(run_tag="quest")["state"] == "awaiting_external_aggregate"
    baseline_context = prepared_baseline["worker_context"]
    output = repo / "quest-baseline"
    comparator_cache = repo / "results" / "autoresearch" / "quest" / "comparator-cache" / "development.json"
    for method in ("incumbent", "fixed_p60"):
        rollout_shard(
            config=loaded,
            seed_set_name="development",
            method=method,
            shard_index=0,
            shard_count=1,
            output_dir=output,
            candidate_source=baseline_context["candidate"]["path"],
            incumbent_source=baseline_context["incumbent"]["path"],
            project_root=repo,
            repeat=2,
            episode_factory=FakeEpisodeFactory(horizon=2),
            environment_digest=environment_digest,
        )
    baseline_aggregate = aggregate_shards(
        config=loaded,
        seed_set_name="development",
        mode="baseline",
        shard_count=1,
        output_dir=output,
        candidate_source=baseline_context["candidate"]["path"],
        incumbent_source=baseline_context["incumbent"]["path"],
        comparator_cache_path=comparator_cache,
        max_horizon=2,
        project_root=repo,
        environment_digest=environment_digest,
    )
    baseline = runner.finalize_external_baseline(
        run_tag="quest",
        experiment_id=prepared_baseline["experiment_id"],
        aggregate_manifest_path=baseline_aggregate["aggregate_manifest_path"],
    )
    assert baseline["status"] == "keep"
    assert baseline["external_evaluation"] is True

    incumbent_source = (
        repo
        / "results"
        / "autoresearch"
        / "quest"
        / "artifacts"
        / "legal_fixed_scan_v1"
        / "candidate.py"
    )
    candidate.write_text(BETTER_CANDIDATE, encoding="utf-8")
    hypothesis = repo / "hypothesis.md"
    hypothesis.write_text("Positive gaze should improve paired success.\n")
    prepared_experiment = runner.prepare_external_experiment(
        run_tag="quest",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert prepared_experiment["status"] == "running"
    ledger_path = (
        repo
        / "results"
        / "autoresearch"
        / "quest"
        / "experiments.jsonl"
    )
    record_count = len(ledger_path.read_text().splitlines())
    repeated_prepare = runner.prepare_external_experiment(
        run_tag="quest",
        hypothesis_file=hypothesis,
        candidate_commit="b" * 40,
        changed_paths=["autoresearch/candidate.py"],
    )
    assert repeated_prepare["experiment_id"] == prepared_experiment["experiment_id"]
    assert len(ledger_path.read_text().splitlines()) == record_count
    prepared_record = json.loads(
        ledger_path.read_text().splitlines()[-1]
    )
    assert prepared_record["hypothesis"].startswith("Positive gaze")
    assert prepared_record["external_stage"] == "awaiting_development_aggregate"
    experiment_context = prepared_experiment["worker_context"]
    experiment_output = repo / "quest-experiment"
    rollout_shard(
        config=loaded,
        seed_set_name="development",
        method="candidate",
        shard_index=0,
        shard_count=1,
        output_dir=experiment_output,
        candidate_source=experiment_context["candidate"]["path"],
        incumbent_source=experiment_context["incumbent"]["path"],
        project_root=repo,
        repeat=2,
        episode_factory=FakeEpisodeFactory(horizon=2),
        environment_digest=environment_digest,
    )
    experiment_aggregate = aggregate_shards(
        config=loaded,
        seed_set_name="development",
        mode="experiment",
        shard_count=1,
        output_dir=experiment_output,
        candidate_source=experiment_context["candidate"]["path"],
        incumbent_source=experiment_context["incumbent"]["path"],
        comparator_cache_path=comparator_cache,
        max_horizon=2,
        project_root=repo,
        environment_digest=environment_digest,
    )
    original_finalize = ExperimentLedger.finalize_experiment

    def interrupt_before_ledger_append(self, selected_id, **kwargs):
        if selected_id == prepared_experiment["experiment_id"]:
            raise KeyboardInterrupt("simulated interruption after artifact write")
        return original_finalize(self, selected_id, **kwargs)

    monkeypatch.setattr(
        ExperimentLedger, "finalize_experiment", interrupt_before_ledger_append
    )
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        runner.finalize_external_experiment(
            run_tag="quest",
            experiment_id=prepared_experiment["experiment_id"],
            aggregate_manifest_path=experiment_aggregate[
                "aggregate_manifest_path"
            ],
        )
    monkeypatch.setattr(ExperimentLedger, "finalize_experiment", original_finalize)
    selected = runner.finalize_external_experiment(
        run_tag="quest",
        experiment_id=prepared_experiment["experiment_id"],
        aggregate_manifest_path=experiment_aggregate["aggregate_manifest_path"],
    )
    assert selected["status"] == "keep"
    assert selected["external_evaluation"] is True
    repeated_final = runner.finalize_external_experiment(
        run_tag="quest",
        experiment_id=prepared_experiment["experiment_id"],
        aggregate_manifest_path=experiment_aggregate["aggregate_manifest_path"],
    )
    assert repeated_final["record_sha256"] == selected["record_sha256"]

    # The aggregate gate is never trusted: changing it without updating all
    # identities/hashes is rejected before another ledger decision.
    gate_path = Path(experiment_aggregate["gate_path"])
    gate = json.loads(gate_path.read_text())
    gate["decision"] = "discard"
    gate_path.write_text(json.dumps(gate) + "\n")
    with pytest.raises(RunContractError, match="gate SHA-256 mismatch"):
        runner.validate_external_evaluation(
            run_tag="quest",
            aggregate_manifest_path=experiment_aggregate[
                "aggregate_manifest_path"
            ],
            mode="experiment",
            seed_set_name="development",
            candidate_sha256=_digest(candidate),
            incumbent_sha256=_digest(incumbent_source),
        )
