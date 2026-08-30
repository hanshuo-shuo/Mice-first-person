import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import pytest

from autoresearch.contract import (
    PUBLIC_OBSERVATION_FIELDS,
    CandidateBoundary,
    CandidateContractError,
    call_candidate,
    reset_candidate,
    step_with_candidate,
    validate_head_action,
)
from autoresearch.guard import (
    CandidateRuntimeError,
    CandidateSourceError,
    ChangedPathError,
    LeakError,
    assert_hash_manifest,
    assert_no_leaks,
    build_hash_manifest,
    load_candidate_controller,
    load_candidate_controller_from_source,
    manifest_sha256,
    scan_for_leaks,
    validate_candidate_source,
    validate_candidate_source_text,
    validate_changed_paths,
    verify_hash_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = PROJECT_ROOT / "autoresearch" / "candidate.py"


def public_observation(**extra):
    return {
        "image_left": np.zeros((4, 5, 3), dtype=np.uint8),
        "image_right": np.ones((4, 5, 3), dtype=np.uint8),
        "proprio": np.asarray((0.1, -0.2, 0.3), dtype=np.float32),
        "previous_action": np.asarray((0.4, 0.0, -0.5), dtype=np.float32),
        **extra,
    }


class MutatingSpyCandidate:
    def __init__(self):
        self.observation_keys = None
        self.history_keys = None

    def reset(self, *, episode_seed):
        self.seed = episode_seed

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        self.observation_keys = tuple(observation)
        self.history_keys = tuple(tuple(frame) for frame in public_history)
        observation["image_left"].fill(255)
        observation["proprio"][0] = 0.99
        public_history[0]["image_right"].fill(17)
        return base_head_action + 0.1 * step_index


def test_candidate_receives_exact_public_fields_and_defensive_copies():
    sentinel = "PRIVILEGED_SENTINEL_c7d0"
    observation = public_observation(
        privileged_state=sentinel,
        predator_location=np.asarray((0.2, 0.4)),
    )
    history_frame = public_observation(info={"sentinel": sentinel})
    original_left = observation["image_left"].copy()
    original_proprio = observation["proprio"].copy()
    original_history_right = history_frame["image_right"].copy()
    candidate = MutatingSpyCandidate()

    reset_candidate(candidate, episode_seed=19)
    result = call_candidate(
        candidate,
        observation=observation,
        public_history=(history_frame,),
        base_head_action=-0.2,
        step_index=1,
    )

    assert result == pytest.approx(-0.1)
    assert candidate.seed == 19
    assert candidate.observation_keys == PUBLIC_OBSERVATION_FIELDS
    assert candidate.history_keys == (PUBLIC_OBSERVATION_FIELDS,)
    np.testing.assert_array_equal(observation["image_left"], original_left)
    np.testing.assert_array_equal(observation["proprio"], original_proprio)
    np.testing.assert_array_equal(history_frame["image_right"], original_history_right)
    assert sentinel not in repr(candidate.__dict__)


@pytest.mark.parametrize(
    "bad_action",
    [
        np.nan,
        np.inf,
        -np.inf,
        1.000001,
        -1.000001,
        np.asarray([0.0]),
        True,
        0.2 + 0.0j,
        "0.2",
    ],
)
def test_invalid_action_is_rejected_before_env_step(bad_action):
    class InvalidCandidate:
        def head_action(self, **kwargs):
            del kwargs
            return bad_action

    class StepSpy:
        calls = 0

        def step(self, action):
            del action
            self.calls += 1

    env = StepSpy()
    with pytest.raises(CandidateContractError):
        step_with_candidate(
            env,
            InvalidCandidate(),
            observation=public_observation(),
            public_history=(),
            base_action=np.zeros(3, dtype=np.float32),
            step_index=0,
        )
    assert env.calls == 0


def test_scalar_boundary_accepts_endpoints_and_numpy_zero_dimensional_values():
    assert validate_head_action(-1) == -1.0
    assert validate_head_action(np.asarray(1.0, dtype=np.float32)) == 1.0


@pytest.mark.parametrize(
    "statement",
    [
        "import os",
        "from pathlib import Path",
        "import socket",
        "import subprocess",
        "import sys",
        "import importlib",
        "import requests",
        "open('secret')",
        "thing.__class__",
        "np.fromfile('secret')",
    ],
)
def test_candidate_static_guard_rejects_io_process_network_and_reflection(statement):
    source = f"""
class CandidateGazeController:
    def reset(self, *, episode_seed):
        {statement}
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return 0.0
"""
    with pytest.raises(CandidateSourceError):
        validate_candidate_source_text(source)


def test_candidate_static_guard_requires_seeded_private_rng():
    source = """
import numpy as np
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._rng = np.random.default_rng()
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return 0.0
"""
    with pytest.raises(CandidateSourceError, match="explicit seed"):
        validate_candidate_source_text(source)


def test_candidate_annotations_cannot_execute_io_during_module_load():
    source = """
import numpy as np
LEAK = []
class CandidateGazeController:
    def reset(self, *, episode_seed: LEAK.append(np.genfromtxt('/etc/hosts', dtype='U256'))):
        self._seed = int(episode_seed)
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return float(bool(LEAK))
"""
    with pytest.raises(CandidateSourceError, match="annotations"):
        validate_candidate_source_text(source)
    with pytest.raises(CandidateSourceError, match="annotations"):
        load_candidate_controller_from_source(source)


@pytest.mark.parametrize(
    "expression",
    [
        "np.datetime64('now', 'D')",
        "np.asarray('now', dtype='datetime64[s]').astype('int64')",
        "np.array('today', dtype='M8[D]').astype('int64')",
        "np.random.Generator(np.random.PCG64())",
        "np.random.default_rng(None)",
        "np.random.default_rng(*())",
    ],
)
def test_candidate_rejects_clock_and_unseeded_rng_construction(expression):
    source = f"""
import numpy as np
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._value = {expression}
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return 0.0
"""
    with pytest.raises(CandidateSourceError):
        validate_candidate_source_text(source)


def test_candidate_cannot_retain_observation_frames_beyond_public_history():
    retaining = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)
        self._frames = []
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        self._frames.append(observation["image_left"])
        return 0.0
"""
    with pytest.raises(CandidateSourceError, match="collections may not grow"):
        load_candidate_controller_from_source(retaining)

    collection_state = retaining.replace(
        'self._frames.append(observation["image_left"])',
        "del observation, public_history, base_head_action, step_index",
    )
    candidate = load_candidate_controller_from_source(collection_state)
    with pytest.raises(CandidateContractError, match="collections are not permitted"):
        reset_candidate(candidate, episode_seed=3)

    indexed_history = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._past = [0.0] * 300
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        self._past[step_index] = float(observation["image_left"].mean())
        return 0.0
"""
    with pytest.raises(CandidateSourceError, match="subscript assignment"):
        load_candidate_controller_from_source(indexed_history)


def test_explicitly_seeded_private_rng_remains_available():
    source = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._rng_state = int(episode_seed) & 0xffffffff
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        self._rng_state = (1664525 * self._rng_state + 1013904223) & 0xffffffff
        return float(self._rng_state / 2147483648.0 - 1.0)
"""
    candidate = load_candidate_controller_from_source(source)
    reset_candidate(candidate, episode_seed=7)
    first = call_candidate(
        candidate,
        observation=public_observation(),
        public_history=(),
        base_head_action=0.0,
        step_index=0,
    )
    reset_candidate(candidate, episode_seed=7)
    replay = call_candidate(
        candidate,
        observation=public_observation(),
        public_history=(),
        base_head_action=0.0,
        step_index=0,
    )
    assert replay == first


def test_numpy_module_import_is_not_available_to_candidate_code():
    source = """
import numpy as np
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._seed = int(episode_seed)
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return float(np.mean(observation["image_left"]))
"""
    with pytest.raises(CandidateSourceError, match="not allowlisted"):
        load_candidate_controller_from_source(source)


def test_physical_head_state_cannot_encode_visual_history():
    source = """
class CandidateGazeController:
    def reset(self, *, episode_seed):
        self._head_yaw_degrees = 0.0
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        self._head_yaw_degrees = (
            self._head_yaw_degrees * 0.5 + float(observation["image_left"].mean())
        )
        return 0.0
"""
    candidate = load_candidate_controller_from_source(source)
    reset_candidate(candidate, episode_seed=3)
    observation = public_observation()
    observation["image_left"].fill(1)
    with pytest.raises(CandidateContractError, match="diverged"):
        call_candidate(
            candidate,
            observation=observation,
            public_history=(),
            base_head_action=0.0,
            step_index=0,
        )


@pytest.mark.parametrize(
    "binding, call",
    [
        ("make_rng = np.random.default_rng", "make_rng()"),
        ("funcs = (np.random.default_rng,)", "funcs[0]()"),
        ("random_module = np.random", "random_module.random()"),
        ("random_value = np.random.random", "random_value()"),
    ],
)
def test_default_rng_cannot_be_aliased_to_bypass_seed_check(binding, call):
    source = f"""
import numpy as np
class CandidateGazeController:
    def reset(self, *, episode_seed):
        {binding}
        self._rng = {call}
    def head_action(self, *, observation, public_history, base_head_action, step_index):
        return 0.0
"""
    with pytest.raises(CandidateSourceError, match="np.random"):
        load_candidate_controller_from_source(source)


def test_registered_candidate_passes_static_guard_and_public_boundary():
    validate_candidate_source(CANDIDATE_PATH)
    candidate = load_candidate_controller(CANDIDATE_PATH)
    boundary = CandidateBoundary(candidate)
    boundary.reset(episode_seed=3)
    command = boundary.head_action(
        observation=public_observation(),
        public_history=(),
        base_head_action=0.25,
        step_index=0,
    )
    assert -1.0 <= command <= 1.0
    assert command == -1.0


def test_registered_scan_replays_public_head_dynamics_without_proprio_roundoff():
    candidate = load_candidate_controller(CANDIDATE_PATH)
    reset_candidate(candidate, episode_seed=1_100_000)
    observation = public_observation()
    observation["proprio"][2] = np.float32(0.9876543)  # deliberately misleading
    previous = np.float32(0.0)
    expected_yaw = 0.0
    targets = (-60.0, -30.0, 0.0, 30.0, 60.0, 30.0, 0.0, -30.0)

    for step in range(32):
        observation["previous_action"][2] = previous
        if step:
            if abs(float(previous)) > 0.05:
                expected_yaw += float(previous) * 24.0
            elif abs(expected_yaw) <= 9.0:
                expected_yaw = 0.0
            else:
                expected_yaw += -9.0 if expected_yaw > 0.0 else 9.0
            expected_yaw = max(-60.0, min(60.0, expected_yaw))

        target = targets[(step // 2) % len(targets)]
        error = target - expected_yaw
        if abs(target) <= 2.0 and abs(error) <= 2.0:
            expected = 0.0
        else:
            expected = error / 24.0
            if abs(error) <= 2.0:
                direction = error if abs(error) > 0.1 else target
                expected = 0.051 if direction >= 0.0 else -0.051
            elif abs(expected) <= 0.05:
                expected = 0.051 if error >= 0.0 else -0.051
            expected = max(-1.0, min(1.0, expected))

        actual = call_candidate(
            candidate,
            observation=observation,
            public_history=(),
            base_head_action=0.0,
            step_index=step,
        )
        assert actual == expected
        previous = np.float32(actual)


@pytest.mark.parametrize(
    "operation",
    ["file", "environment", "environment_bytes", "network", "process"],
)
def test_runtime_guard_blocks_common_external_access(operation, tmp_path):
    target = tmp_path / "secret.txt"
    target.write_text("do-not-read", encoding="utf-8")

    class RuntimeEscapeCandidate:
        def head_action(self, **kwargs):
            del kwargs
            if operation == "file":
                return float(bool(open(target, encoding="utf-8").read()))
            if operation == "environment":
                return float(bool(os.getenv("PATH")))
            if operation == "environment_bytes":
                return float(bool(os.environb.get(b"PATH")))
            if operation == "network":
                return float(bool(socket.socket()))
            subprocess.run(["true"], check=True)
            return 0.0

    with pytest.raises(CandidateRuntimeError):
        call_candidate(
            RuntimeEscapeCandidate(),
            observation=public_observation(),
            public_history=(),
            base_head_action=0.0,
            step_index=0,
        )


def test_changed_path_whitelist_is_exact_and_rejects_path_tricks():
    assert validate_changed_paths(["autoresearch/candidate.py"]) == (
        "autoresearch/candidate.py",
    )
    for path in (
        "autoresearch/contract.py",
        "autoresearch/candidate.py.bak",
        "../autoresearch/candidate.py",
        "/tmp/candidate.py",
    ):
        with pytest.raises((ChangedPathError, ValueError)):
            validate_changed_paths([path])


def test_content_hash_manifest_detects_change_and_has_stable_identity(tmp_path):
    (tmp_path / "trusted.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_hash_manifest(tmp_path, ["trusted.py"])
    assert verify_hash_manifest(tmp_path, manifest) == ()
    assert manifest_sha256(manifest) == manifest_sha256(dict(reversed(manifest.items())))

    (tmp_path / "trusted.py").write_text("VALUE = 2\n", encoding="utf-8")
    mismatch = verify_hash_manifest(tmp_path, manifest)
    assert len(mismatch) == 1
    assert mismatch[0].reason == "changed"
    with pytest.raises(Exception, match="manifest mismatch"):
        assert_hash_manifest(tmp_path, manifest)


def test_leak_scan_finds_privileged_names_and_secrets_without_echoing_secret():
    secret = "sk-test-secret-cd962"
    payload = json.dumps(
        {
            "privileged_state": "hidden",
            "Authorization": f"Bearer {secret}",
        },
    )
    findings = scan_for_leaks(payload, source="calls.jsonl", secret_values=(secret,))
    assert findings
    assert secret not in repr(findings)
    assert all(finding.source == "calls.jsonl" for finding in findings)
    with pytest.raises(LeakError):
        assert_no_leaks(payload, secret_values=(secret,))

    # This is a required aggregate metric, not a candidate/prompt leak.
    assert_no_leaks('{"mean_predator_pixels_visible_fraction": 0.2}')
