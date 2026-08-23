import numpy as np
import pytest

from analysis.sac_gaze_ablation import (
    METHODS,
    classify_primary_case,
    exact_mcnemar_p,
    finalize,
    fixed_scan_target,
    run_episode,
)
from training.first_person_sac import load_sac_config, make_first_person_env


class _ZeroPolicy:
    def predict(self, observation, deterministic):
        del observation
        assert deterministic
        return np.zeros((3,), dtype=np.float32), None


def test_exp05_method_order_keeps_fixed_plus_60_as_primary_camera_pose():
    assert METHODS == (
        "sac_active_gaze",
        "sac_fixed_0",
        "sac_fixed_p60",
        "sac_fixed_m60",
        "sac_fixed_p30",
        "sac_fixed_scan",
    )


def test_fixed_scan_reuses_exp04_symmetric_sweep():
    assert [fixed_scan_target(step) for step in range(16)] == [
        -60.0,
        -60.0,
        -30.0,
        -30.0,
        0.0,
        0.0,
        30.0,
        30.0,
        60.0,
        60.0,
        30.0,
        30.0,
        0.0,
        0.0,
        -30.0,
        -30.0,
    ]


def test_fixed_camera_is_exact_from_first_observation_through_episode():
    config = load_sac_config("configs/sac_cnn_active_gaze.yaml")
    config["environment"]["max_step"] = 2
    env = make_first_person_env(config)
    try:
        record = run_episode(
            env,
            _ZeroPolicy(),
            method="sac_fixed_p30",
            seed=1_000_000,
        )
    finally:
        env.close()
    assert record["initial_head_yaw_degrees"] == 30.0
    assert record["minimum_head_yaw_degrees"] == pytest.approx(30.0)
    assert record["maximum_head_yaw_degrees"] == pytest.approx(30.0)
    assert record["maximum_fixed_yaw_error_degrees"] == pytest.approx(0.0)
    assert record["mean_absolute_head_action"] == pytest.approx(0.0)


def test_exact_mcnemar_and_registered_case_thresholds():
    assert exact_mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert classify_primary_case(0.945, 0.930) == "case_a_like_camera_pose_sufficient"
    assert classify_primary_case(0.945, 0.750) == "case_b_like_dynamic_gaze_large_gain"
    assert classify_primary_case(0.945, 0.900) == "intermediate"


def test_finalize_writes_primary_paired_result(tmp_path):
    records = []
    for method in METHODS:
        records.append(
            {
                "method": method,
                "seed": 1_000_000,
                "steps": 10,
                "return": 1.0,
                "goal_reached": True,
                "clean_success": method != "sac_fixed_p60",
                "capture_count": int(method == "sac_fixed_p60"),
                "capture_episode": method == "sac_fixed_p60",
                "terminated": True,
                "truncated": False,
                "minimum_predator_distance": 0.2,
                "final_goal_distance": 0.0,
                "path_cost": 1.0,
                "gaze_travel_degrees": 0.0,
                "mean_absolute_head_action": 0.0,
                "predator_pixels_visible_fraction": 0.5,
                "initial_head_yaw_degrees": 0.0,
                "mean_head_yaw_degrees": 0.0,
                "minimum_head_yaw_degrees": 0.0,
                "maximum_head_yaw_degrees": 0.0,
                "maximum_fixed_yaw_error_degrees": (
                    0.0 if method.startswith("sac_fixed_") and method != "sac_fixed_scan" else None
                ),
            },
        )
    model_path = tmp_path / "model.zip"
    config_path = tmp_path / "config.yaml"
    model_path.write_bytes(b"frozen checkpoint")
    config_path.write_text("experiment_id: exp05-test\n", encoding="utf-8")
    summary = finalize(
        {"experiment_id": "exp05-test"},
        output_dir=tmp_path / "output",
        model_path=model_path,
        config_path=config_path,
        records=records,
        episodes=1,
        seed_start=1_000_000,
        bootstrap_samples=20,
        elapsed_seconds=1.0,
    )
    assert summary["case_classification"] == "case_b_like_dynamic_gaze_large_gain"
    assert summary["primary_comparison"]["active_only_successes"] == 1
    assert summary["primary_comparison"]["comparator_only_successes"] == 0
    assert (tmp_path / "output" / "REPORT.md").is_file()
