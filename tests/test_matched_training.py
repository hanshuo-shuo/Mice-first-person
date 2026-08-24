from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analysis.matched_training_aggregate as matched_aggregate
from analysis.matched_training_aggregate import _seed_interval
from task_distribution import TaskBank, load_task_records, validate_split_contract
from training.first_person_sac import (
    MATCHED_CONDITIONS,
    apply_matched_condition,
    force_cellworld_cpu,
    load_sac_config,
    make_first_person_env,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = PROJECT_ROOT / "configs" / "tasksets" / "matched_v1"


def test_matched_task_manifests_enforce_registered_generalization_split():
    manifests = {
        split: load_task_records(TASK_ROOT / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    contract = validate_split_contract(
        manifests,
        heldout_region="southeast",
        validation_region_pair=("northwest", "northeast"),
        heldout_occlusion_cell_ids={
            int(cell_id)
            for cell_id in np.load(
                PROJECT_ROOT / "data" / "cell_ids_near_occlusion_21_05.npy",
            )
            if int(cell_id) in {
                int(value)
                for record in manifests["test"]
                for value in record["path_cell_ids"]
            }
        },
    )
    assert contract["task_counts"] == {
        "train": 4096,
        "validation": 256,
        "test": 1000,
    }


def test_task_bank_random_and_sequential_selection_are_reproducible():
    records = load_task_records(TASK_ROOT / "validation.jsonl")
    sequential = TaskBank(records, mode="sequential")
    rng = np.random.default_rng(7)
    assert sequential.select(rng)["task_id"] == records[0]["task_id"]
    state = sequential.get_state_dict()
    second = sequential.select(rng)["task_id"]
    sequential.set_state_dict(state)
    assert sequential.select(rng)["task_id"] == second

    first_bank = TaskBank(records, mode="random")
    second_bank = TaskBank(records, mode="random")
    assert first_bank.select(np.random.default_rng(11))["task_id"] == second_bank.select(
        np.random.default_rng(11),
    )["task_id"]


def test_four_conditions_share_task_but_only_active_controls_head():
    base = load_sac_config(PROJECT_ROOT / "configs" / "sac_matched_generalization.yaml")
    assert base["matched_training"] == {
        "conditions": list(MATCHED_CONDITIONS),
        "seeds": [2026082401, 2026082402, 2026082403, 2026082404, 2026082405],
    }
    expected_shapes = {
        "active": (3,),
        "fixed_center": (2,),
        "fixed_p60": (2,),
        "fixed_scan": (2,),
    }
    task_ids = set()
    physical_tasks = set()
    for condition in MATCHED_CONDITIONS:
        config = apply_matched_condition(base, condition)
        env = make_first_person_env(
            config,
            task_split="validation",
            task_selection_mode="sequential",
        )
        try:
            observation, info = env.reset(seed=9, options={"task_index": 0})
            task = info["task"]
            task_ids.add(task["task_id"])
            physical_tasks.add(
                (
                    tuple(env.unwrapped.model.prey.state.location),
                    tuple(env.unwrapped.model.goal_location),
                    tuple(env.unwrapped.model.predator.state.location),
                    env.unwrapped.model.predator.max_forward_speed,
                ),
            )
            assert env.action_space.shape == expected_shapes[condition]
            assert observation["previous_action"].shape == expected_shapes[condition]
            expected_yaw = 60.0 if condition == "fixed_p60" else 0.0
            assert env.head_yaw_degrees == expected_yaw
        finally:
            env.close()
    assert len(task_ids) == 1
    assert len(physical_tasks) == 1


def test_passive_scan_is_rate_limited_and_snapshot_deterministic():
    base = load_sac_config(PROJECT_ROOT / "configs" / "sac_matched_generalization.yaml")
    config = apply_matched_condition(base, "fixed_scan")
    env = make_first_person_env(
        config,
        task_split="validation",
        task_selection_mode="sequential",
    )
    try:
        env.reset(seed=3, options={"task_index": 1})
        action = np.zeros((2,), dtype=np.float32)
        before = env.get_state_dict()
        first, _, _, _, _ = env.step(action)
        assert env.head_yaw_degrees == -24.0
        env.set_state_dict(before)
        second, _, _, _, _ = env.step(action)
        assert env.head_yaw_degrees == -24.0
        for field in ("image_left", "image_right", "proprio", "previous_action"):
            np.testing.assert_array_equal(first[field], second[field])
    finally:
        env.close()


def test_training_seed_interval_uses_five_seed_replication():
    values = [0.70, 0.75, 0.80, 0.85, 0.90]
    low, high = _seed_interval(values)
    assert low < np.mean(values) < high
    assert _seed_interval([0.8] * 5) == (0.8, 0.8)


def test_cellworld_simulator_tensors_are_forced_off_training_gpu():
    modules = [
        __import__(name, fromlist=["default_device"])
        for name in (
            "cellworld_game.torch.device",
            "cellworld_game.torch.points",
            "cellworld_game.torch.polygon",
            "cellworld_game.torch.visibility",
            "cellworld_game.torch.geometry",
        )
    ]
    for module in modules:
        module.default_device = __import__("torch").device("cuda")
    force_cellworld_cpu()
    assert all(str(module.default_device) == "cpu" for module in modules)


def test_matched_aggregate_uses_training_seed_as_replication_unit(
    tmp_path,
    monkeypatch,
):
    success_cutoffs = {
        "active": 900,
        "fixed_center": 500,
        "fixed_p60": 850,
        "fixed_scan": 875,
    }

    def fake_records(result_root, *, condition, training_seed, shard_count):
        del result_root, shard_count
        seed_offset = int(training_seed) - 2026082401
        cutoff = success_cutoffs[condition] + seed_offset
        return [
            {
                "task_index": task_index,
                "task_id": f"task-{task_index:04d}",
                "clean_success": task_index < cutoff,
                "capture_episode": task_index >= cutoff,
                "goal_reached": True,
                "steps": 20,
                "minimum_predator_distance": 0.2,
                "path_cost": 1.0,
                "gaze_travel_degrees": 0.0,
                "predator_pixels_visible_fraction": 0.5,
            }
            for task_index in range(1000)
        ]

    monkeypatch.setattr(matched_aggregate, "_load_run_records", fake_records)
    monkeypatch.setattr(matched_aggregate, "write_jsonl", lambda path, rows: None)
    summary = matched_aggregate.aggregate(
        SimpleNamespace(result_root=tmp_path, shard_count=5),
    )
    conditions = {row["condition"]: row for row in summary["conditions"]}
    assert conditions["active"]["training_seeds"] == 5
    assert np.isclose(conditions["active"]["mean_clean_success_rate"], 0.902)
    primary = next(
        row
        for row in summary["paired_training_seed_differences"]
        if row["right_condition"] == "fixed_p60"
        and row["training_seed"] == "mean"
    )
    assert np.isclose(primary["clean_success_delta"], 0.05)
