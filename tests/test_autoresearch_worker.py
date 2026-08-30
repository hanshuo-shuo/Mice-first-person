import os
import signal
import time

import numpy as np
import pytest

from autoresearch.contract import CandidateBoundary
from autoresearch.guard import CandidateRuntimeError, CandidateSourceError
from autoresearch.worker import IsolatedCandidateController


STATEFUL_SOURCE = """
class CandidateGazeController:
    def __init__(self):
        self._seed = 0
        self._rng_state = 0

    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)
        self._rng_state = 0

    def head_action(self, *, observation, public_history, base_head_action, step_index):
        self._rng_state += 1
        value = ((self._seed % 13) + self._rng_state + int(step_index)) / 32.0
        return max(-1.0, min(1.0, value))
"""


INFINITE_SOURCE = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(self, *, observation, public_history, base_head_action, step_index):
        while True:
            pass
"""


EXCEPTION_SOURCE = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(self, *, observation, public_history, base_head_action, step_index):
        raise ValueError("candidate-private-exception-payload")
"""


MEMORY_SOURCE = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)

    def head_action(self, *, observation, public_history, base_head_action, step_index):
        oversized = [0] * (256 * 1024 * 1024)
        del oversized
        return 0.0
"""


def public_observation():
    return {
        "image_left": np.zeros((3, 4, 3), dtype=np.uint8),
        "image_right": np.ones((3, 4, 3), dtype=np.uint8),
        "proprio": np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
        "previous_action": np.asarray((0.4, 0.0, -0.5), dtype=np.float32),
    }


def head_action(worker, *, step_index=0):
    return worker.head_action(
        observation=public_observation(),
        public_history=(public_observation(),),
        base_head_action=0.0,
        step_index=step_index,
    )


def test_spawned_worker_preserves_state_and_seeded_replay_through_boundary():
    with IsolatedCandidateController.from_source(
        STATEFUL_SOURCE.encode("utf-8"),
        timeout_seconds=0.75,
    ) as worker:
        boundary = CandidateBoundary(worker)
        boundary.reset(episode_seed=17)
        first = boundary.head_action(
            observation=public_observation(),
            public_history=(),
            base_head_action=0.0,
            step_index=0,
        )
        second = boundary.head_action(
            observation=public_observation(),
            public_history=(),
            base_head_action=0.0,
            step_index=1,
        )
        assert second != first

        boundary.reset(episode_seed=17)
        replay = boundary.head_action(
            observation=public_observation(),
            public_history=(),
            base_head_action=0.0,
            step_index=0,
        )
        assert replay == first
        assert worker.is_alive


def test_from_source_accepts_a_path_and_close_is_idempotent(tmp_path):
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(STATEFUL_SOURCE, encoding="utf-8")
    worker = IsolatedCandidateController.from_source(candidate_path)
    assert worker.pid is not None
    assert worker.is_alive
    worker.reset(episode_seed=3)
    assert -1.0 <= head_action(worker) <= 1.0

    worker.close()
    worker.close()
    assert worker.closed
    assert not worker.is_alive
    with pytest.raises(CandidateRuntimeError, match="closed"):
        worker.reset(episode_seed=3)


def test_infinite_candidate_has_a_hard_wall_timeout_and_is_reaped():
    worker = IsolatedCandidateController.from_source(
        INFINITE_SOURCE,
        timeout_seconds=0.15,
    )
    try:
        worker.reset(episode_seed=9)
        started = time.monotonic()
        with pytest.raises(CandidateRuntimeError, match="timed out"):
            head_action(worker)
        assert time.monotonic() - started < 2.0
        assert not worker.is_alive
    finally:
        worker.close()


def test_candidate_exception_is_generic_and_makes_worker_terminal():
    worker = IsolatedCandidateController.from_source(EXCEPTION_SOURCE)
    try:
        worker.reset(episode_seed=9)
        with pytest.raises(CandidateRuntimeError) as caught:
            head_action(worker)
        assert "candidate-private-exception-payload" not in str(caught.value)
        assert not worker.is_alive
        with pytest.raises(CandidateRuntimeError, match="closed"):
            worker.reset(episode_seed=9)
    finally:
        worker.close()


def test_memory_budget_fails_closed_without_constraining_parent_numpy():
    parent_before = np.ones((1024,), dtype=np.float64).sum()
    worker = IsolatedCandidateController.from_source(
        MEMORY_SOURCE,
        timeout_seconds=1.0,
        memory_limit_bytes=32 * 1024 * 1024,
    )
    try:
        worker.reset(episode_seed=1)
        started = time.monotonic()
        with pytest.raises(CandidateRuntimeError):
            head_action(worker)
        assert time.monotonic() - started < 2.0
        assert not worker.is_alive
    finally:
        worker.close()
    assert np.ones((1024,), dtype=np.float64).sum() == parent_before


def test_unexpected_worker_death_is_detected_and_reaped():
    worker = IsolatedCandidateController.from_source(STATEFUL_SOURCE)
    try:
        assert worker.pid is not None
        os.kill(worker.pid, signal.SIGKILL)
        deadline = time.monotonic() + 1.0
        while worker.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)
        with pytest.raises(CandidateRuntimeError, match="exited unexpectedly"):
            worker.reset(episode_seed=1)
        assert not worker.is_alive
    finally:
        worker.close()


def test_invalid_source_fails_before_spawning_a_worker():
    source = STATEFUL_SOURCE.replace(
        "self._seed = int(episode_seed)",
        "self._seed = open('/tmp/forbidden').read()",
        1,
    )
    with pytest.raises(CandidateSourceError):
        IsolatedCandidateController.from_source(source)
