import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.human_active_looking_fingerprint import (
    AnalysisParameters,
    run_exp02_human_active_looking,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_unit_world_cache(cache_root: Path):
    _write_json(
        cache_root / "world_configuration" / "hexagonal",
        {"cell_shape": {"sides": 4}},
    )
    _write_json(
        cache_root / "world_implementation" / "hexagonal.canonical",
        {
            "cell_locations": [{"x": 0.30, "y": 0.50}],
            "space": {"transformation": {"rotation": 0.0}},
            "cell_transformation": {"size": 0.10, "rotation": 0.0},
        },
    )
    _write_json(
        cache_root / "cell_group" / "hexagonal.unit_world.occlusions",
        [0],
    )


def _write_structured_session(data_root: Path):
    session = data_root / "session_alpha"
    session.mkdir(parents=True)
    steps = 14
    sim_time = np.arange(steps, dtype=np.float32) / 10.0

    action = np.zeros((steps, 3), dtype=np.float32)
    action[:, 0] = np.asarray(
        [0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7],
        dtype=np.float32,
    )
    action[5, 1] = 0.7
    action[3, 2] = 0.8
    action[9, 2] = 0.8

    prey_x = np.asarray(
        [0.08, 0.11, 0.14, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.30, 0.34, 0.38, 0.42],
        dtype=np.float32,
    )
    prey_y = np.full((steps,), 0.50, dtype=np.float32)
    body_heading = np.zeros((steps,), dtype=np.float32)
    body_heading[4:] = 45.0
    privileged_state = np.stack(
        (
            prey_x,
            prey_y,
            action[:, 0] * 0.04,
            np.zeros((steps,), dtype=np.float32),
            body_heading,
            np.zeros((steps,), dtype=np.float32),
            prey_x,
            prey_y + 0.20,
            np.zeros((steps,), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)

    visible = np.zeros((steps,), dtype=np.bool_)
    visible[6:8] = True
    geometric = np.zeros((steps,), dtype=np.bool_)
    geometric[6:8] = True
    minimum_distance = np.full((steps,), 0.60, dtype=np.float32)
    minimum_distance[3] = 0.50
    minimum_distance[6:8] = 0.15
    minimum_distance[9] = 0.50

    np.savez_compressed(
        session / "episode_00000.npz",
        action=action,
        previous_action=np.vstack((np.zeros((1, 3), dtype=np.float32), action[:-1])),
        proprio=np.zeros((steps, 3), dtype=np.float32),
        reward=np.zeros((steps,), dtype=np.float32),
        terminated=np.r_[np.zeros((steps - 1,), dtype=np.bool_), True],
        truncated=np.zeros((steps,), dtype=np.bool_),
        sim_time=sim_time,
        privileged_state=privileged_state,
        predator_pixels_visible=visible,
        predator_geometric_los=geometric,
        predator_in_left_frustum=visible,
        predator_in_right_frustum=np.zeros((steps,), dtype=np.bool_),
        predator_within_detection_range=visible,
        minimum_distance=minimum_distance,
        capture_event=np.zeros((steps,), dtype=np.bool_),
        capture_count=np.zeros((steps,), dtype=np.int32),
    )
    session_metadata = {
        "format_version": 3,
        "transition_convention": "observation_t, action_t, reward_t, done_t",
        "participant_id": "p1",
        "world_name": "unit_world",
        "control_hz": 10.0,
        "action_names": ["forward_velocity", "body_yaw_rate", "head_yaw_rate"],
        "privileged_state_names": [
            "prey_x",
            "prey_y",
            "prey_vx",
            "prey_vy",
            "body_heading_degrees",
            "head_yaw_degrees",
            "predator_x",
            "predator_y",
            "predator_pixels_visible",
        ],
        "episodes": [{"file": "episode_00000.npz", "is_success": True}],
    }
    _write_json(session / "session.json", session_metadata)
    _write_json(
        session / "episode_00000.json",
        {"file": "episode_00000.npz", "steps": steps, "is_success": True},
    )


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_no_human_active_looking_data_is_clear_non_error(tmp_path, capsys):
    summary = run_exp02_human_active_looking(
        tmp_path / "missing",
        tmp_path / "out",
    )
    assert summary["status"] == "no_data"
    assert summary["split_unit"] == "participant/session/world; never frame"
    assert (tmp_path / "out" / "ETHICS_NOTE.md").exists()
    assert "No human demonstration sessions found" in capsys.readouterr().out


def test_exp02_extracts_active_looking_structure_and_group_split(tmp_path):
    data_root = tmp_path / "demos"
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "out"
    _write_unit_world_cache(cache_root)
    _write_structured_session(data_root)

    parameters = AnalysisParameters(
        lead_window_seconds=0.5,
        approach_window_seconds=0.2,
        baseline_window_seconds=0.2,
        reconfirm_window_seconds=0.5,
        route_pre_window_seconds=0.2,
        route_post_window_seconds=0.4,
        junction_occlusion_distance=0.20,
    )
    summary = run_exp02_human_active_looking(
        data_root,
        output_dir,
        cache_root=cache_root,
        parameters=parameters,
    )

    assert summary["status"] == "ok"
    assert summary["episodes"] == 1

    episode = _read_csv(output_dir / "episode_fingerprint.csv")[0]
    assert float(episode["first_active_look_distance_to_occlusion"]) > 0.0
    assert float(episode["pre_danger_deceleration"]) == pytest.approx(0.6, abs=1e-6)
    assert float(episode["head_turn_before_body_fraction"]) == pytest.approx(1.0)
    assert float(episode["reconfirm_action_latency_after_loss"]) == pytest.approx(0.1, abs=1e-6)
    assert float(episode["look_information_value_agreement"]) == pytest.approx(1.0)
    assert float(episode["route_change_probability_after_look"]) == pytest.approx(0.5)
    assert float(episode["high_risk_forward_suppression"]) > 0.0

    manifest = _read_csv(output_dir / "split_manifest.csv")[0]
    assert manifest["group_key"] == "p1/session_alpha/unit_world"
    assert manifest["split_unit"] == "participant/session/world; never frame"

    risk_bins = {row["risk_bin"]: row for row in _read_csv(output_dir / "risk_bin_summary.csv")}
    assert float(risk_bins["high"]["mean_forward_command"]) < float(
        risk_bins["low"]["mean_forward_command"],
    )
