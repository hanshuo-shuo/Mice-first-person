"""Shared public-only visual features for controlled EXP-03/04 baselines.

The encoder is deliberately small and deterministic.  It is an engineering
probe, not a claimed biological or learned visual representation.  Keeping it
identical across methods isolates temporal state and gaze selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from policies.base import MockVisionPolicy, PublicHistoryFrame


ENCODER_ID = "public_rgb_threat_features_v1"


@dataclass(frozen=True)
class VisualFeatures:
    threat_score: float
    bearing: float
    edge_energy: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (self.threat_score, self.bearing, self.edge_energy),
            dtype=np.float64,
        )


class PublicVisualEncoder:
    """Encode one public binocular observation, with no privileged inputs."""

    encoder_id = ENCODER_ID

    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, left: np.ndarray, right: np.ndarray) -> VisualFeatures:
        self.calls += 1
        combined = np.concatenate((np.asarray(left), np.asarray(right)), axis=1)
        mask = MockVisionPolicy._threat_mask(combined)
        score = float(mask.mean())
        if mask.any():
            _, xs = np.nonzero(mask)
            bearing = float(2.0 * xs.mean() / max(combined.shape[1] - 1, 1) - 1.0)
        else:
            bearing = 0.0
        gray = combined.astype(np.float64).mean(axis=2) / 255.0
        dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
        dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
        return VisualFeatures(score, bearing, float(dx + dy))

    def encode_observation(self, observation: Mapping[str, np.ndarray]) -> VisualFeatures:
        return self.encode_images(observation["image_left"], observation["image_right"])

    def encode_history(
        self,
        history: Sequence[PublicHistoryFrame],
    ) -> list[VisualFeatures]:
        return [self.encode_images(frame.image_left, frame.image_right) for frame in history]


def action_away_from_bearing(bearing: float) -> str:
    """Map image bearing (-left, +right) to a registered evasive action."""

    return "evade_right" if float(bearing) < 0.0 else "evade_left"


def feature_dict(features: VisualFeatures) -> Mapping[str, Any]:
    return {
        "threat_score": float(features.threat_score),
        "bearing": float(features.bearing),
        "edge_energy": float(features.edge_energy),
    }
