"""Public candidate boundary for the phase-1 autoresearch loop.

The helpers in this module are the sole path from trusted evaluation code to
an editable gaze controller.  They select and copy public observations before
the call and validate the returned scalar before an environment can consume
it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PUBLIC_OBSERVATION_FIELDS = (
    "image_left",
    "image_right",
    "proprio",
    "previous_action",
)
PUBLIC_OBSERVATION_FIELD_SET = frozenset(PUBLIC_OBSERVATION_FIELDS)


class CandidateContractError(ValueError):
    """Raised before simulation advances when a candidate violates its API."""


# Short alias for callers that prefer the generic contract terminology.
ContractViolation = CandidateContractError


def _copy_public_array(value: Any, *, field: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise CandidateContractError(
            f"public field {field!r} is not array-like",
        ) from exc
    if array.dtype.kind not in "buifc":
        raise CandidateContractError(
            f"public field {field!r} must have a numeric dtype",
        )
    return np.array(array, copy=True, order="K")


def copy_public_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a fresh mapping containing exactly the four public fields.

    Extra evaluator fields are intentionally dropped.  Every array owns a
    defensive copy, so a candidate may neither mutate evaluator state nor use
    aliases to reach another frame.
    """

    if not isinstance(observation, Mapping):
        raise CandidateContractError("observation must be a mapping")
    missing = [name for name in PUBLIC_OBSERVATION_FIELDS if name not in observation]
    if missing:
        raise CandidateContractError(
            "observation is missing public fields: " + ", ".join(missing),
        )
    return {
        name: _copy_public_array(observation[name], field=name)
        for name in PUBLIC_OBSERVATION_FIELDS
    }


def copy_public_history(
    public_history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], ...]:
    """Copy every public history frame and make the outer sequence immutable."""

    if isinstance(public_history, (str, bytes, bytearray)) or not isinstance(
        public_history,
        Sequence,
    ):
        raise CandidateContractError("public_history must be a sequence of mappings")
    return tuple(copy_public_observation(frame) for frame in public_history)


def _finite_normalized_scalar(value: Any, *, label: str) -> float:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise CandidateContractError(f"{label} must be one numeric scalar") from exc
    if array.shape != () or array.dtype.kind not in "iuf":
        raise CandidateContractError(f"{label} must be one real numeric scalar")
    result = float(array.item())
    if not np.isfinite(result):
        raise CandidateContractError(f"{label} must be finite")
    if result < -1.0 or result > 1.0:
        raise CandidateContractError(f"{label} must lie in [-1, 1]")
    return result


def validate_head_action(value: Any) -> float:
    """Validate and normalize a candidate's scalar head-yaw-rate command."""

    return _finite_normalized_scalar(value, label="candidate head action")


def _validate_episode_seed(episode_seed: Any) -> int:
    if isinstance(episode_seed, (bool, np.bool_)) or not isinstance(
        episode_seed,
        (int, np.integer),
    ):
        raise CandidateContractError("episode_seed must be an integer")
    return int(episode_seed)


def _validate_step_index(step_index: Any) -> int:
    if isinstance(step_index, (bool, np.bool_)) or not isinstance(
        step_index,
        (int, np.integer),
    ):
        raise CandidateContractError("step_index must be a non-negative integer")
    result = int(step_index)
    if result < 0:
        raise CandidateContractError("step_index must be a non-negative integer")
    return result


def reset_candidate(controller: Any, *, episode_seed: int) -> None:
    """Reset a controller inside the runtime sandbox."""

    reset = getattr(controller, "reset", None)
    if not callable(reset):
        raise CandidateContractError("candidate must provide reset()")
    from .guard import restricted_candidate_runtime

    with restricted_candidate_runtime():
        reset(episode_seed=_validate_episode_seed(episode_seed))
    validate_candidate_private_state(controller)
    if (
        type(controller).__module__ == "autoresearch_candidate"
        and "_head_yaw_degrees" in controller.__dict__
        and float(controller.__dict__["_head_yaw_degrees"]).hex() != (0.0).hex()
    ):
        raise CandidateContractError(
            "candidate physical head-yaw state must initialize at zero",
        )


def _expected_physical_head_yaw(
    controller: Any,
    *,
    observation: Mapping[str, np.ndarray],
    step_index: int,
) -> tuple[bool, float | None]:
    if type(controller).__module__ != "autoresearch_candidate":
        return False, None
    state = controller.__dict__
    if "_head_yaw_degrees" not in state:
        return False, None
    before = float(state["_head_yaw_degrees"])
    if step_index == 0:
        return True, 0.0
    previous_action = np.asarray(observation["previous_action"])
    if previous_action.shape != (3,):
        raise CandidateContractError(
            "previous_action must be length 3 for physical head-state replay",
        )
    previous_command = float(previous_action[2])
    if not math.isfinite(previous_command) or not -1.0 <= previous_command <= 1.0:
        raise CandidateContractError("previous head action is not finite and normalized")
    if abs(previous_command) > 0.05:
        expected = before + previous_command * 24.0
    elif abs(before) <= 9.0:
        expected = 0.0
    else:
        expected = before - 9.0 if before > 0.0 else before + 9.0
    return True, max(-60.0, min(60.0, expected))


def _validate_private_value(
    value: Any,
    *,
    path: str,
    seen: set[int],
    budget: list[int],
    allow_collections: bool,
) -> None:
    budget[0] += 1
    if budget[0] > 4096:
        raise CandidateContractError("candidate private state exceeds 4096 values")
    if value is None or isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, np.integer)):
        if int(value).bit_length() > 4096:
            raise CandidateContractError(f"candidate private integer is too large at {path}")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise CandidateContractError(f"candidate private scalar is non-finite at {path}")
        return
    if isinstance(value, str):
        if len(value) > 1024:
            raise CandidateContractError(f"candidate private text is too large at {path}")
        return
    if isinstance(value, bytes):
        if len(value) > 1024:
            raise CandidateContractError(f"candidate private bytes are too large at {path}")
        return
    if isinstance(value, np.ndarray):
        raise CandidateContractError(
            f"candidate may not retain observation arrays in private state at {path}",
        )

    identity = id(value)
    if identity in seen:
        raise CandidateContractError(f"candidate private state contains a cycle at {path}")
    if isinstance(value, Mapping):
        if not allow_collections:
            raise CandidateContractError(
                f"candidate instance collections are not permitted at {path}",
            )
        if len(value) > 1024:
            raise CandidateContractError(f"candidate private mapping is too large at {path}")
        seen.add(identity)
        for key, item in value.items():
            _validate_private_value(
                key,
                path=f"{path}.key",
                seen=seen,
                budget=budget,
                allow_collections=allow_collections,
            )
            _validate_private_value(
                item,
                path=f"{path}[value]",
                seen=seen,
                budget=budget,
                allow_collections=allow_collections,
            )
        seen.remove(identity)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        if not allow_collections:
            raise CandidateContractError(
                f"candidate instance collections are not permitted at {path}",
            )
        if len(value) > 1024:
            raise CandidateContractError(f"candidate private collection is too large at {path}")
        seen.add(identity)
        for index, item in enumerate(value):
            _validate_private_value(
                item,
                path=f"{path}[{index}]",
                seen=seen,
                budget=budget,
                allow_collections=allow_collections,
            )
        seen.remove(identity)
        return
    raise CandidateContractError(
        f"candidate private state contains a forbidden object at {path}",
    )


def validate_candidate_private_state(controller: Any) -> None:
    """Reject retained frames, opaque objects, cycles, and oversized state."""

    # The trusted proxy owns pipes/process handles in the parent, while its
    # child invokes this same validator on the actual editable controller.
    from .worker import IsolatedCandidateController

    if type(controller) is IsolatedCandidateController:
        return
    if type(controller).__module__ != "autoresearch_candidate":
        # Test doubles are trusted evaluator fixtures.  Production candidate
        # source is always loaded into the fixed module above (also in the
        # isolated child) before crossing this boundary.
        return
    state = getattr(controller, "__dict__", None)
    if not isinstance(state, dict):
        raise CandidateContractError("candidate must expose ordinary private state")
    seen: set[int] = set()
    budget = [0]
    for name, value in state.items():
        _validate_private_value(
            value,
            path=f"self.{name}",
            seen=seen,
            budget=budget,
            allow_collections=False,
        )
    for name, value in vars(type(controller)).items():
        if name.startswith("__") or callable(value):
            continue
        _validate_private_value(
            value,
            path=f"class.{name}",
            seen=seen,
            budget=budget,
            allow_collections=True,
        )


def call_candidate(
    controller: Any,
    *,
    observation: Mapping[str, Any],
    public_history: Sequence[Mapping[str, Any]],
    base_head_action: Any,
    step_index: int,
) -> float:
    """Call a candidate with copied public inputs, then validate its output."""

    head_action = getattr(controller, "head_action", None)
    if not callable(head_action):
        raise CandidateContractError("candidate must provide head_action()")

    safe_observation = copy_public_observation(observation)
    safe_history = copy_public_history(public_history)
    safe_base_head_action = _finite_normalized_scalar(
        base_head_action,
        label="base head action",
    )
    safe_step_index = _validate_step_index(step_index)
    has_physical_state, expected_head_yaw = _expected_physical_head_yaw(
        controller,
        observation=safe_observation,
        step_index=safe_step_index,
    )

    from .guard import restricted_candidate_runtime

    with restricted_candidate_runtime():
        result = head_action(
            observation=safe_observation,
            public_history=safe_history,
            base_head_action=safe_base_head_action,
            step_index=safe_step_index,
        )
    validate_candidate_private_state(controller)
    if has_physical_state:
        actual = controller.__dict__.get("_head_yaw_degrees")
        if actual is None or float(actual).hex() != float(expected_head_yaw).hex():
            raise CandidateContractError(
                "candidate physical head-yaw state diverged from public legal dynamics",
            )
    return validate_head_action(result)


def compose_candidate_action(
    controller: Any,
    *,
    observation: Mapping[str, Any],
    public_history: Sequence[Mapping[str, Any]],
    base_action: Any,
    step_index: int,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Copy a legal SAC action and replace only its third component."""

    try:
        action = np.asarray(base_action)
    except Exception as exc:
        raise CandidateContractError("base action must be a numeric length-3 vector") from exc
    if action.shape != (3,) or action.dtype.kind not in "iuf":
        raise CandidateContractError("base action must be a real numeric length-3 vector")
    copied = np.array(action, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(copied)) or np.any(copied < -1.0) or np.any(copied > 1.0):
        raise CandidateContractError("base action must be finite and lie in [-1, 1]")
    copied[2] = call_candidate(
        controller,
        observation=observation,
        public_history=public_history,
        base_head_action=float(copied[2]),
        step_index=step_index,
    )
    return copied.astype(dtype, copy=False)


def step_with_candidate(
    env: Any,
    controller: Any,
    *,
    observation: Mapping[str, Any],
    public_history: Sequence[Mapping[str, Any]],
    base_action: Any,
    step_index: int,
) -> Any:
    """Validate the complete action before the trusted evaluator calls ``step``."""

    action = compose_candidate_action(
        controller,
        observation=observation,
        public_history=public_history,
        base_action=base_action,
        step_index=step_index,
    )
    return env.step(action)


@dataclass
class CandidateBoundary:
    """Small state-free facade convenient for evaluators and test doubles."""

    controller: Any

    def reset(self, *, episode_seed: int) -> None:
        reset_candidate(self.controller, episode_seed=episode_seed)

    def head_action(
        self,
        *,
        observation: Mapping[str, Any],
        public_history: Sequence[Mapping[str, Any]],
        base_head_action: Any,
        step_index: int,
    ) -> float:
        return call_candidate(
            self.controller,
            observation=observation,
            public_history=public_history,
            base_head_action=base_head_action,
            step_index=step_index,
        )
