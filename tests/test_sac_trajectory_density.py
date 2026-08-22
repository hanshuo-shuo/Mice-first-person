import numpy as np
import pytest

from analysis.sac_trajectory_density import (
    capture_density,
    episode_normalized_density,
    pack_traces,
    unpack_state_traces,
    wilson_interval,
)


def trace(points, captures):
    points = np.asarray(points, dtype=np.float32)
    state_count = len(points)
    return {
        "prey_xy": points,
        "predator_xy": np.zeros_like(points),
        "head_yaw_degrees": np.arange(state_count, dtype=np.float32),
        "predator_pixels_visible": np.zeros((state_count,), dtype=np.bool_),
        "capture_event": np.asarray(captures, dtype=np.bool_),
        "actions": np.zeros((state_count - 1, 3), dtype=np.float32),
    }


def test_episode_normalized_density_does_not_overweight_long_episodes():
    long_trace = trace([[0.1, 0.1]] * 10, [False] * 10)
    short_trace = trace([[0.9, 0.9]], [False])
    density = episode_normalized_density(
        [long_trace, short_trace],
        bins=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    assert density.sum() == pytest.approx(1.0)
    assert density[0, 0] == pytest.approx(0.5)
    assert density[1, 1] == pytest.approx(0.5)


def test_capture_density_is_events_per_episode():
    first = trace(
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [False, True, True],
    )
    second = trace([[0.9, 0.9]], [False])
    density = capture_density(
        [first, second],
        bins=2,
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    assert density.sum() == pytest.approx(1.0)
    assert density[0, 0] == pytest.approx(1.0)


def test_pickle_free_trace_offsets_round_trip():
    traces = [
        trace([[0.1, 0.2], [0.2, 0.3]], [False, True]),
        trace([[0.8, 0.7]], [False]),
    ]
    packed = pack_traces([11, 12], traces)
    restored = unpack_state_traces(packed)
    np.testing.assert_array_equal(packed["state_offsets"], [0, 2, 3])
    np.testing.assert_array_equal(packed["action_offsets"], [0, 1, 1])
    for expected, actual in zip(traces, restored):
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])


def test_wilson_interval_retains_uncertainty_at_observed_ceiling():
    low, high = wilson_interval(40, 40)
    assert low == pytest.approx(0.9124, abs=1e-3)
    assert high == pytest.approx(1.0)
