import copy

import numpy as np
import torch

from evaluate_first_person_sac import run_episode
from policies.binocular_sac import BinocularCombinedExtractor, PUBLIC_OBSERVATION_FIELDS
from training.first_person_sac import (
    apply_smoke_overrides,
    load_sac_config,
    make_first_person_env,
    policy_kwargs,
)


def smoke_config():
    return apply_smoke_overrides(load_sac_config("configs/sac_cnn_active_gaze.yaml"))


def test_binocular_extractor_consumes_only_public_contract():
    config = smoke_config()
    env = make_first_person_env(config)
    try:
        observation, _ = env.reset(seed=7)
        assert set(observation) == set(PUBLIC_OBSERVATION_FIELDS)
        assert env.action_space.shape == (3,)
        extractor = BinocularCombinedExtractor(
            env.observation_space,
            image_features_dim=16,
            vector_features_dim=8,
        )
        batch = {}
        for name, value in observation.items():
            tensor = torch.as_tensor(value[None]).float()
            if name.startswith("image_"):
                tensor = tensor / 255.0
            batch[name] = tensor
        features = extractor(batch)
        assert features.shape == (1, 40)
        assert torch.isfinite(features).all()
    finally:
        env.close()


def test_sac_policy_kwargs_register_binocular_extractor():
    values = policy_kwargs(smoke_config())
    assert values["features_extractor_class"] is BinocularCombinedExtractor
    assert values["normalize_images"] is True
    assert values["share_features_extractor"] is False


class PublicObservationSpyPolicy:
    def __init__(self):
        self.calls = 0

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert set(observation) == set(PUBLIC_OBSERVATION_FIELDS)
        self.calls += 1
        return np.zeros((3,), dtype=np.float32), None


def test_evaluator_does_not_pass_privileged_state_to_policy():
    config = smoke_config()
    config = copy.deepcopy(config)
    config["environment"]["max_step"] = 3
    env = make_first_person_env(config)
    policy = PublicObservationSpyPolicy()
    try:
        result = run_episode(
            env,
            policy,
            method="sac_active_gaze",
            seed=11,
        )
    finally:
        env.close()
    assert policy.calls == result["steps"]
    assert result["steps"] == 3
    assert result["truncated"] is True
