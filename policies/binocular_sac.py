"""Shared binocular CNN feature extractor for first-person SB3 policies."""

from __future__ import annotations

import math
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


PUBLIC_OBSERVATION_FIELDS = (
    "image_left",
    "image_right",
    "proprio",
    "previous_action",
)


class BinocularCombinedExtractor(BaseFeaturesExtractor):
    """Encode both eyes with shared weights and fuse public vector sensors.

    Stable-Baselines3 normally transposes image spaces to CHW before creating
    the policy.  The extractor also accepts HWC spaces directly so its contract
    can be tested without relying on an implicit wrapper.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_features_dim: int = 128,
        vector_features_dim: int = 32,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("BinocularCombinedExtractor requires a Dict observation")
        if set(observation_space.spaces) != set(PUBLIC_OBSERVATION_FIELDS):
            raise ValueError(
                "First-person SAC observation fields must be exactly "
                f"{PUBLIC_OBSERVATION_FIELDS}, got {tuple(observation_space.spaces)}",
            )
        if image_features_dim <= 0 or vector_features_dim <= 0:
            raise ValueError("Feature dimensions must be positive")
        super().__init__(observation_space, features_dim=1)

        left_space = observation_space["image_left"]
        right_space = observation_space["image_right"]
        if left_space.shape != right_space.shape:
            raise ValueError("Left and right image spaces must have equal shapes")
        if len(left_space.shape) != 3 or left_space.dtype != np.uint8:
            raise TypeError("Eye observations must be three-dimensional uint8 images")

        channels_first = left_space.shape[0] in (1, 3, 4)
        channels_last = left_space.shape[-1] in (1, 3, 4)
        if not channels_first and not channels_last:
            raise ValueError(f"Cannot infer image channels from {left_space.shape}")
        self.channels_last = bool(not channels_first and channels_last)
        input_channels = int(
            left_space.shape[-1] if self.channels_last else left_space.shape[0],
        )

        # Small-stride convolutions preserve information in the intentionally
        # low-resolution 64x48 eye images and also support the 32x24 smoke size.
        self.shared_eye_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(left_space.sample()[None]).float()
            sample = self._channels_first(sample)
            flattened_size = int(self.shared_eye_cnn(sample).shape[1])
        self.eye_projection = nn.Sequential(
            nn.Linear(flattened_size, int(image_features_dim)),
            nn.ReLU(),
        )

        vector_input_dim = int(
            math.prod(observation_space["proprio"].shape)
            + math.prod(observation_space["previous_action"].shape),
        )
        self.vector_projection = nn.Sequential(
            nn.Linear(vector_input_dim, int(vector_features_dim)),
            nn.ReLU(),
        )
        features_dim = int(image_features_dim) * 2 + int(vector_features_dim)
        self._features_dim = features_dim

    def _channels_first(self, image: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return image.permute(0, 3, 1, 2).contiguous()
        return image

    def _encode_eye(self, image: torch.Tensor) -> torch.Tensor:
        image = self._channels_first(image)
        return self.eye_projection(self.shared_eye_cnn(image))

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        left = self._encode_eye(observations["image_left"])
        right = self._encode_eye(observations["image_right"])
        vector = torch.cat(
            (
                observations["proprio"].flatten(start_dim=1),
                observations["previous_action"].flatten(start_dim=1),
            ),
            dim=1,
        )
        vector_features = self.vector_projection(vector)
        return torch.cat((left, right, vector_features), dim=1)
