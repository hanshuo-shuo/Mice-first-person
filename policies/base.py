"""Strict vision-only policy contracts used by PeekBench.

The input type intentionally has no ``info``, simulator state, predator
coordinates, visibility labels, rewards, or future outcomes.  Evaluators keep
those privileged values in separate records.
"""

from __future__ import annotations

import abc
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import jsonschema
import numpy as np


THREAT_BEARINGS = ("left", "center", "right", "behind", "unknown")
LOOK_ACTIONS = ("far_left", "left", "center", "right", "far_right", "hold")
MOTION_ACTIONS = (
    "stop",
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "evade_left",
    "evade_right",
)

DECISION_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "threat_visible",
        "threat_bearing",
        "risk_next_horizon",
        "uncertainty",
        "recommended_look",
        "recommended_motion",
    ],
    "properties": {
        "threat_visible": {"type": "boolean"},
        "threat_bearing": {"type": "string", "enum": list(THREAT_BEARINGS)},
        "risk_next_horizon": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "uncertainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "recommended_look": {"type": "string", "enum": list(LOOK_ACTIONS)},
        "recommended_motion": {"type": "string", "enum": list(MOTION_ACTIONS)},
    },
}


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype,
    shape: Tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected array shaped {shape}, got {array.shape}")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB image, got {array.shape}")
    return _readonly_array(array, dtype=np.uint8)


def hash_array(value: np.ndarray) -> str:
    """Hash dtype, shape, and bytes so equal public inputs share cache keys."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicHistoryFrame:
    """One historical pair of public eye images."""

    image_left: np.ndarray
    image_right: np.ndarray

    def __post_init__(self) -> None:
        left = _validate_image(self.image_left, "history.image_left")
        right = _validate_image(self.image_right, "history.image_right")
        if left.shape != right.shape:
            raise ValueError("Historical left/right eye shapes must match")
        object.__setattr__(self, "image_left", left)
        object.__setattr__(self, "image_right", right)


@dataclass(frozen=True)
class PolicyInput:
    """The complete and exclusive set of fields visible to a policy."""

    image_left: np.ndarray
    image_right: np.ndarray
    proprio: np.ndarray
    previous_action: np.ndarray
    history: Tuple[PublicHistoryFrame, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        left = _validate_image(self.image_left, "image_left")
        right = _validate_image(self.image_right, "image_right")
        if left.shape != right.shape:
            raise ValueError("Current left/right eye shapes must match")
        proprio = _readonly_array(self.proprio, dtype=np.float32, shape=(3,))
        previous_action = np.asarray(self.previous_action, dtype=np.float32)
        if previous_action.ndim != 1 or previous_action.shape[0] not in (2, 3):
            raise ValueError(
                "previous_action must be a one-dimensional 2- or 3-value action",
            )
        previous_action = _readonly_array(previous_action, dtype=np.float32)
        history = tuple(self.history)
        if not all(isinstance(frame, PublicHistoryFrame) for frame in history):
            raise TypeError("history must contain only PublicHistoryFrame objects")
        object.__setattr__(self, "image_left", left)
        object.__setattr__(self, "image_right", right)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "previous_action", previous_action)
        object.__setattr__(self, "history", history)

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, np.ndarray],
        history: Sequence[PublicHistoryFrame] = (),
    ) -> "PolicyInput":
        required = {"image_left", "image_right", "proprio", "previous_action"}
        missing = required.difference(observation)
        if missing:
            raise KeyError(f"Public observation is missing fields: {sorted(missing)}")
        return cls(
            image_left=observation["image_left"],
            image_right=observation["image_right"],
            proprio=observation["proprio"],
            previous_action=observation["previous_action"],
            history=tuple(history),
        )

    def image_hashes(self) -> Mapping[str, Any]:
        return {
            "image_left": hash_array(self.image_left),
            "image_right": hash_array(self.image_right),
            "history": [
                {
                    "image_left": hash_array(frame.image_left),
                    "image_right": hash_array(frame.image_right),
                }
                for frame in self.history
            ],
        }

    def public_sensor_values(self) -> Mapping[str, Any]:
        return {
            "proprio": [float(value) for value in self.proprio],
            "previous_action": [float(value) for value in self.previous_action],
        }


def validate_decision(value: Mapping[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` unless output is strictly valid."""

    jsonschema.Draft202012Validator(DECISION_JSON_SCHEMA).validate(dict(value))


@dataclass(frozen=True)
class PolicyDecision:
    threat_visible: bool
    threat_bearing: str
    risk_next_horizon: float
    uncertainty: float
    recommended_look: str
    recommended_motion: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyDecision":
        validate_decision(value)
        return cls(
            threat_visible=bool(value["threat_visible"]),
            threat_bearing=str(value["threat_bearing"]),
            risk_next_horizon=float(value["risk_next_horizon"]),
            uncertainty=float(value["uncertainty"]),
            recommended_look=str(value["recommended_look"]),
            recommended_motion=str(value["recommended_motion"]),
        )

    def to_dict(self) -> Mapping[str, Any]:
        result = asdict(self)
        validate_decision(result)
        return result


@dataclass(frozen=True)
class PolicyTelemetry:
    backend: str
    model: str
    provider: Mapping[str, Any]
    prompt_hash: str
    image_hashes: Mapping[str, Any]
    latency_ms: float
    token_usage: Mapping[str, Any]
    cost: float | None
    parse_success: bool
    cache_hit: bool
    raw_response: Any

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    telemetry: PolicyTelemetry


class VisionPolicy(abc.ABC):
    @abc.abstractmethod
    def decide(self, policy_input: PolicyInput) -> PolicyResult:
        """Return one semantic decision without mutating an environment."""


class MockVisionPolicy(VisionPolicy):
    """Deterministic image-only backend for offline and test execution."""

    model = "mock/vision-v1"

    @staticmethod
    def _threat_mask(image: np.ndarray) -> np.ndarray:
        red = (
            (image[..., 0] > 180)
            & (image[..., 1] < 115)
            & (image[..., 2] < 110)
        )
        yellow = (
            (image[..., 0] > 175)
            & (image[..., 1] > 125)
            & (image[..., 2] < 95)
        )
        return red | yellow

    def decide(self, policy_input: PolicyInput) -> PolicyResult:
        started = time.perf_counter()
        combined = np.concatenate(
            (policy_input.image_left, policy_input.image_right),
            axis=1,
        )
        mask = self._threat_mask(combined)
        visible = bool(mask.any())
        history_visible = any(
            self._threat_mask(frame.image_left).any()
            or self._threat_mask(frame.image_right).any()
            for frame in policy_input.history
        )
        if visible:
            _, x_coordinates = np.nonzero(mask)
            normalized_x = float(x_coordinates.mean() / max(combined.shape[1] - 1, 1))
            if normalized_x < 0.40:
                bearing = "left"
                motion = "evade_right"
            elif normalized_x > 0.60:
                bearing = "right"
                motion = "evade_left"
            else:
                bearing = "center"
                motion = "backward"
            risk = float(np.clip(0.35 + mask.mean() * 240.0, 0.0, 1.0))
            uncertainty = 0.15
            recommended_look = "hold"
        else:
            bearing = "unknown"
            motion = "stop" if history_visible else "forward"
            risk = 0.55 if history_visible else 0.12
            uncertainty = 0.35 if history_visible else 0.65
            head_yaw = float(policy_input.proprio[2])
            if head_yaw > 0.25:
                recommended_look = "right"
            elif head_yaw < -0.25:
                recommended_look = "left"
            else:
                parity = int(hash_array(combined)[-1], 16) % 2
                recommended_look = "left" if parity == 0 else "right"

        decision = PolicyDecision(
            threat_visible=visible,
            threat_bearing=bearing,
            risk_next_horizon=risk,
            uncertainty=uncertainty,
            recommended_look=recommended_look,
            recommended_motion=motion,
        )
        image_hashes = policy_input.image_hashes()
        prompt_material = {
            "backend": self.model,
            "images": image_hashes,
            "sensors": policy_input.public_sensor_values(),
        }
        prompt_hash = hashlib.sha256(
            json.dumps(prompt_material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8",
            ),
        ).hexdigest()
        raw_response = json.dumps(
            decision.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        telemetry = PolicyTelemetry(
            backend="mock",
            model=self.model,
            provider={"name": "local_deterministic_mock"},
            prompt_hash=prompt_hash,
            image_hashes=image_hashes,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            cost=0.0,
            parse_success=True,
            cache_hit=False,
            raw_response=raw_response,
        )
        return PolicyResult(decision=decision, telemetry=telemetry)
