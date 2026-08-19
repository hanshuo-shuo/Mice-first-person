"""Shared configuration and environment helpers for binocular SAC."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from benchmarks.peekbench.artifacts import environment_metadata, write_json
from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv
from policies.binocular_sac import BinocularCombinedExtractor, PUBLIC_OBSERVATION_FIELDS
from reward import custom_reward, first_person_sac_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REWARD_FUNCTIONS = {
    "custom_reward": custom_reward,
    "first_person_sac_reward": first_person_sac_reward,
}


def load_sac_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("First-person SAC config must contain a YAML mapping")
    config = copy.deepcopy(dict(value))
    validate_sac_config(config)
    return config


def validate_sac_config(config: Mapping[str, Any]) -> None:
    if not str(config.get("experiment_id", "")).strip():
        raise ValueError("experiment_id must be non-empty")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("seed must be non-negative")

    environment = config.get("environment", {})
    if environment.get("world_name") != "21_05":
        raise ValueError("First-person SAC currently supports only world 21_05")
    if environment.get("observation_mode") != "mouse":
        raise ValueError("First-person SAC requires observation_mode=mouse")
    if environment.get("action_mode") not in (
        "egocentric_velocity",
        "egocentric_velocity_head",
    ):
        raise ValueError("First-person SAC requires an egocentric action mode")
    if int(environment.get("vision_width", 0)) < 24:
        raise ValueError("environment.vision_width must be at least 24")
    if int(environment.get("vision_height", 0)) < 24:
        raise ValueError("environment.vision_height must be at least 24")
    if int(environment.get("max_step", 0)) <= 0:
        raise ValueError("environment.max_step must be positive")
    if float(environment.get("time_step", 0.0)) <= 0.0:
        raise ValueError("environment.time_step must be positive")

    reward_name = str(config.get("reward_function", ""))
    if reward_name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward_function: {reward_name!r}")

    training = config.get("training", {})
    for key in (
        "total_timesteps",
        "num_envs",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "checkpoint_freq",
        "eval_freq",
        "eval_episodes",
    ):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["buffer_size"]) <= int(training["batch_size"]):
        raise ValueError("training.buffer_size must exceed training.batch_size")
    if int(training["learning_starts"]) >= int(training["total_timesteps"]):
        raise ValueError("training.learning_starts must be below total_timesteps")
    if str(training.get("device", "")) not in ("auto", "cpu", "cuda"):
        raise ValueError("training.device must be auto, cpu, or cuda")

    policy = config.get("policy", {})
    if int(policy.get("image_features_dim", 0)) <= 0:
        raise ValueError("policy.image_features_dim must be positive")
    if int(policy.get("vector_features_dim", 0)) <= 0:
        raise ValueError("policy.vector_features_dim must be positive")
    net_arch = policy.get("net_arch", [])
    if not net_arch or any(int(value) <= 0 for value in net_arch):
        raise ValueError("policy.net_arch must contain positive layer sizes")


def config_hash(config: Mapping[str, Any]) -> str:
    identity = copy.deepcopy(dict(config))
    identity.pop("experiment_id", None)
    identity.pop("output_root", None)
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_dir(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root / str(config["experiment_id"])


def prepare_sac_experiment(
    config: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    output_dir = experiment_dir(config, project_root=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(dict(config))
    resolved["config_hash"] = config_hash(config)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        **environment_metadata(project_root),
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "config_hash": resolved["config_hash"],
        "policy_input_fields": list(PUBLIC_OBSERVATION_FIELDS),
        "reward_function": config["reward_function"],
    }
    write_json(output_dir / "run_metadata.json", metadata)
    return output_dir


def make_first_person_env(config: Mapping[str, Any]) -> FirstPersonBotEvadeEnv:
    values = config["environment"]
    reward_function = REWARD_FUNCTIONS[str(config["reward_function"])]
    env = FirstPersonBotEvadeEnv(
        world_name=str(values["world_name"]),
        use_lppos=False,
        use_predator=bool(values["use_predator"]),
        reward_function=reward_function,
        max_step=int(values["max_step"]),
        time_step=float(values["time_step"]),
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.CONTINUOUS,
        frame_stack_k=1,
        predator_prey_forward_speed_ratio=float(
            values["predator_prey_forward_speed_ratio"],
        ),
        vision_width=int(values["vision_width"]),
        vision_height=int(values["vision_height"]),
        vision_fov=float(values["vision_fov"]),
        vision_far_clip=float(values["vision_far_clip"]),
        vision_detection_range=float(values["vision_detection_range"]),
        observation_mode=str(values["observation_mode"]),
        action_mode=str(values["action_mode"]),
        render_mode="rgb_array",
    )
    if set(env.observation_space.spaces) != set(PUBLIC_OBSERVATION_FIELDS):
        raise RuntimeError("Training environment violates the public observation contract")
    return env


def _environment_factory(
    config: Mapping[str, Any],
    *,
    rank: int,
    monitor_dir: Path,
):
    frozen_config = copy.deepcopy(dict(config))

    def _make():
        env = make_first_person_env(frozen_config)
        return Monitor(
            env,
            filename=str(monitor_dir / f"env_{rank}"),
            info_keywords=("is_success", "captures"),
        )

    return _make


def make_vec_env(
    config: Mapping[str, Any],
    *,
    num_envs: int,
    monitor_dir: Path,
    seed_offset: int = 0,
):
    monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        _environment_factory(config, rank=rank, monitor_dir=monitor_dir)
        for rank in range(int(num_envs))
    ]
    if int(num_envs) == 1:
        vec_env = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories, start_method="forkserver")
    vec_env.seed(int(config["seed"]) + int(seed_offset))
    return VecTransposeImage(vec_env)


def policy_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    values = config["policy"]
    return {
        "features_extractor_class": BinocularCombinedExtractor,
        "features_extractor_kwargs": {
            "image_features_dim": int(values["image_features_dim"]),
            "vector_features_dim": int(values["vector_features_dim"]),
        },
        "net_arch": [int(value) for value in values["net_arch"]],
        "normalize_images": True,
        "share_features_extractor": bool(values["share_features_extractor"]),
    }


def apply_smoke_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    smoke = copy.deepcopy(dict(config))
    smoke["training"].update(
        {
            "total_timesteps": 24,
            "num_envs": 1,
            "buffer_size": 128,
            "learning_starts": 8,
            "batch_size": 8,
            "train_freq": 1,
            "gradient_steps": 1,
            "checkpoint_freq": 12,
            "eval_freq": 12,
            "eval_episodes": 1,
            "device": "cpu",
            "require_cuda": False,
        },
    )
    smoke["environment"]["max_step"] = 12
    validate_sac_config(smoke)
    return smoke
