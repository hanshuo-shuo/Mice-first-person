import copy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.peekbench.artifacts import (
    canonical_typed_bytes,
    git_commit,
    load_state,
    state_digest,
)
from benchmarks.peekbench.config import load_config, validate_config
from benchmarks.peekbench.environment import classify_state, make_env
from benchmarks.peekbench.evaluation import evaluate_policy_branch
from benchmarks.peekbench.generator import generate_snapshots
from benchmarks.peekbench.headroom import (
    METHOD_ORDER,
    _oracle_score,
    evaluate_go_condition,
    run_headroom_evaluation,
)
from policies.base import MockVisionPolicy


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"predator_pixels_visible": True}, "predator_visible"),
        (
            {"predator_geometric_los": True},
            "geometric_outside_frustum",
        ),
        (
            {"predator_in_left_frustum": True},
            "frustum_pixel_occluded",
        ),
        (
            {"recent_visibility": [True]},
            "recently_visible_hidden",
        ),
        ({"use_predator": False}, "no_predator_control"),
    ],
)
def test_state_category_definitions(overrides, expected):
    label = {
        "use_predator": True,
        "predator_pixels_visible": False,
        "predator_geometric_los": False,
        "predator_in_left_frustum": False,
        "predator_in_right_frustum": False,
        "recent_visibility": [],
        **overrides,
    }
    assert classify_state(label, recent_visibility_horizon=4) == expected


@pytest.fixture(scope="module")
def generated_benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("peekbench_primary")
    config = load_config(
        "configs/peekbench/smoke.yaml",
        experiment_id="pytest_primary",
        num_snapshots=2,
        output_root=root,
    )
    records = generate_snapshots(config)
    experiment_dir = root / "pytest_primary"
    return config, experiment_dir, records


def test_state_restore_determinism(generated_benchmark):
    config, experiment_dir, records = generated_benchmark
    record = records[0]
    state = load_state(experiment_dir / record["state_path"])
    env = make_env(config, use_predator=record["use_predator"])
    action = np.asarray((0.2, -0.1, 0.3), dtype=np.float32)
    traces = []
    try:
        for _ in range(2):
            env.set_state_dict(copy.deepcopy(state))
            observation, reward, terminated, truncated, info = env.step(action)
            traces.append(
                {
                    "observation": {
                        key: np.array(value, copy=True)
                        for key, value in observation.items()
                    },
                    "outcome": canonical_typed_bytes(
                        {
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        },
                    ),
                    "state_hash": state_digest(env.get_state_dict()),
                },
            )
    finally:
        env.close()
    for key in traces[0]["observation"]:
        np.testing.assert_array_equal(
            traces[0]["observation"][key],
            traces[1]["observation"][key],
        )
    assert traces[0]["outcome"] == traces[1]["outcome"]
    assert traces[0]["state_hash"] == traces[1]["state_hash"]
    assert record["replay_deterministic"] is True


def test_gaze_branch_does_not_mutate_source_snapshot(generated_benchmark):
    config, experiment_dir, records = generated_benchmark
    record = records[0]
    state = load_state(experiment_dir / record["state_path"])
    before = canonical_typed_bytes(state)
    env = make_env(config, use_predator=record["use_predator"])
    try:
        evaluate_policy_branch(
            env,
            state,
            gaze_degrees=60.0,
            policy=MockVisionPolicy(),
            history=(),
            horizon_steps=2,
        )
    finally:
        env.close()
    assert canonical_typed_bytes(state) == before


def test_identical_config_and_seed_reproduce_snapshot_ids(
    generated_benchmark,
    tmp_path,
):
    config, _, first_records = generated_benchmark
    second_config = load_config(
        "configs/peekbench/smoke.yaml",
        experiment_id="pytest_second",
        num_snapshots=2,
        output_root=tmp_path,
    )
    second_records = generate_snapshots(second_config)
    assert config["config_hash"] == second_config["config_hash"]
    assert [record["snapshot_id"] for record in first_records] == [
        record["snapshot_id"] for record in second_records
    ]
    assert [record["state_hash"] for record in first_records] == [
        record["state_hash"] for record in second_records
    ]


def test_exp00_uses_legal_paired_gaze_actions(generated_benchmark):
    config, experiment_dir, snapshot_records = generated_benchmark
    before = {
        record["snapshot_id"]: (experiment_dir / record["state_path"]).read_bytes()
        for record in snapshot_records
    }
    result = run_headroom_evaluation(config)

    assert result["summary"]["remote_model_calls"] == 0
    assert len(result["records"]) == len(snapshot_records)
    for record in result["records"]:
        assert tuple(record["methods"]) == METHOD_ORDER
        fixed = record["methods"]["fixed_head"]
        zero_candidate = next(
            branch
            for branch in record["legal_gaze_candidates"]
            if branch["controller"]["gaze"]["target_degrees"] == 0.0
        )
        assert fixed["actions"] == zero_candidate["actions"]
        assert fixed["outcome"] == zero_candidate["outcome"]
        oracle = record["methods"]["privileged_best_gaze"]
        assert _oracle_score(oracle) == min(
            _oracle_score(branch) for branch in record["legal_gaze_candidates"]
        )
        for method in METHOD_ORDER:
            value = record["methods"][method]
            branches = value if isinstance(value, list) else [value]
            for branch in branches:
                assert branch["legal_gaze"] is True
                assert branch["source_snapshot_unchanged"] is True
                actions = np.asarray(branch["actions"], dtype=np.float64)
                assert np.all(actions >= -1.0)
                assert np.all(actions <= 1.0)
                assert np.max(np.abs(branch["head_yaw_degrees"])) <= 60.0

    after = {
        record["snapshot_id"]: (experiment_dir / record["state_path"]).read_bytes()
        for record in snapshot_records
    }
    assert before == after


def test_exp00_repeated_run_is_deterministic(generated_benchmark):
    config, _, _ = generated_benchmark
    first = run_headroom_evaluation(config)["records"]
    second = run_headroom_evaluation(config)["records"]
    assert canonical_typed_bytes(first) == canonical_typed_bytes(second)


def _synthetic_headroom_branch(*, safe_success, target=0.0):
    return {
        "legal_gaze": True,
        "source_snapshot_unchanged": True,
        "controller": {"gaze": {"target_degrees": float(target)}},
        "outcome": {"safe_success": bool(safe_success)},
    }


def test_exp00_go_requires_stable_nonzero_legal_recovery(generated_benchmark):
    config = copy.deepcopy(generated_benchmark[0])
    config["headroom"]["go"] = {
        "minimum_predator_snapshots": 2,
        "minimum_fixed_failure_fraction": 0.5,
        "minimum_stable_headroom_fraction": 0.5,
        "minimum_recovery_fraction_of_fixed_failures": 1.0,
        "minimum_stable_recoveries": 1,
        "minimum_safe_nonzero_gaze_candidates": 2,
    }

    def record(snapshot_id, fixed_safe, candidate_safety):
        fixed = _synthetic_headroom_branch(safe_success=fixed_safe)
        oracle = _synthetic_headroom_branch(
            safe_success=any(candidate_safety),
            target=30.0,
        )
        methods = {
            "fixed_head": fixed,
            "random_head": [_synthetic_headroom_branch(safe_success=fixed_safe)],
            "coverage_scan": _synthetic_headroom_branch(safe_success=fixed_safe),
            "privileged_best_gaze": oracle,
            "privileged_safe_controller": _synthetic_headroom_branch(
                safe_success=True,
            ),
        }
        candidates = [
            _synthetic_headroom_branch(
                safe_success=safe,
                target=target,
            )
            for safe, target in zip(candidate_safety, (-60.0, -30.0, 30.0, 60.0))
        ]
        return {
            "snapshot_id": snapshot_id,
            "use_predator": True,
            "methods": methods,
            "legal_gaze_candidates": candidates,
        }

    records = [
        record("recoverable", False, (True, True, False, False)),
        record("already-safe", True, (True, True, True, True)),
    ]
    decision = evaluate_go_condition(records, config)
    assert decision["verdict"] == "GO"
    assert decision["stable_recovery_snapshot_ids"] == ["recoverable"]

    records[0]["legal_gaze_candidates"][1]["outcome"]["safe_success"] = False
    decision = evaluate_go_condition(records, config)
    assert decision["verdict"] == "NO_GO"


def test_exp00_config_rejects_impossible_stability_requirement(generated_benchmark):
    config = copy.deepcopy(generated_benchmark[0])
    config["headroom"]["go"]["minimum_safe_nonzero_gaze_candidates"] = 5
    with pytest.raises(ValueError, match="minimum_safe_nonzero_gaze_candidates"):
        validate_config(config)


def test_quest_git_commit_override_is_validated(monkeypatch, tmp_path):
    commit = "8d9ac79a9f539738c92a0fdc9347e34fd5b620c4"
    monkeypatch.setenv("QUEST_GIT_COMMIT", commit.upper())
    assert git_commit(tmp_path) == commit

    monkeypatch.setenv("QUEST_GIT_COMMIT", "not-a-commit")
    with pytest.raises(ValueError, match="QUEST_GIT_COMMIT"):
        git_commit(tmp_path)
