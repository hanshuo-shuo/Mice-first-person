import copy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.peekbench.artifacts import (
    canonical_typed_bytes,
    load_state,
    state_digest,
)
from benchmarks.peekbench.config import load_config
from benchmarks.peekbench.environment import classify_state, make_env
from benchmarks.peekbench.evaluation import evaluate_policy_branch
from benchmarks.peekbench.generator import generate_snapshots
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
