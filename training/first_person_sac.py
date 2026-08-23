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
# Import botevade first so it selects the repository cache before cellworld.util.
from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv
import cellworld.util as cellworld_util
from policies.binocular_sac import BinocularCombinedExtractor, PUBLIC_OBSERVATION_FIELDS
from reward import custom_reward, first_person_sac_reward
from task_distribution import manifest_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REWARD_FUNCTIONS = {
    "custom_reward": custom_reward,
    "first_person_sac_reward": first_person_sac_reward,
}
MATCHED_CONDITIONS = {
    "active": {
        "action_mode": "egocentric_velocity_head",
        "passive_gaze_mode": "center",
        "fixed_head_yaw_degrees": 0.0,
    },
    "fixed_center": {
        "action_mode": "egocentric_velocity",
        "passive_gaze_mode": "center",
        "fixed_head_yaw_degrees": 0.0,
    },
    "fixed_p60": {
        "action_mode": "egocentric_velocity",
        "passive_gaze_mode": "fixed",
        "fixed_head_yaw_degrees": 60.0,
    },
    "fixed_scan": {
        "action_mode": "egocentric_velocity",
        "passive_gaze_mode": "scan",
        "fixed_head_yaw_degrees": 0.0,
    },
}


def apply_matched_condition(
    config: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    if condition not in MATCHED_CONDITIONS:
        raise ValueError(f"Unknown matched condition: {condition!r}")
    resolved = copy.deepcopy(dict(config))
    resolved["matched_condition"] = str(condition)
    resolved["environment"].update(MATCHED_CONDITIONS[condition])
    validate_sac_config(resolved)
    return resolved


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
    condition = config.get("matched_condition")
    if condition is not None:
        if condition not in MATCHED_CONDITIONS:
            raise ValueError(f"Unknown matched_condition: {condition!r}")
        expected = MATCHED_CONDITIONS[str(condition)]
        for key, value in expected.items():
            if environment.get(key, 0.0 if key == "fixed_head_yaw_degrees" else None) != value:
                raise ValueError(
                    f"matched_condition={condition} requires environment.{key}={value!r}",
                )
    if int(environment.get("vision_width", 0)) < 24:
        raise ValueError("environment.vision_width must be at least 24")
    if int(environment.get("vision_height", 0)) < 24:
        raise ValueError("environment.vision_height must be at least 24")
    if int(environment.get("max_step", 0)) <= 0:
        raise ValueError("environment.max_step must be positive")
    if float(environment.get("time_step", 0.0)) <= 0.0:
        raise ValueError("environment.time_step must be positive")

    task_distribution = config.get("task_distribution")
    if task_distribution is not None:
        if not str(task_distribution.get("manifest_root", "")).strip():
            raise ValueError("task_distribution.manifest_root must be non-empty")
        if task_distribution.get("train_split") != "train":
            raise ValueError("task_distribution.train_split must be train")
        if task_distribution.get("validation_split") != "validation":
            raise ValueError("task_distribution.validation_split must be validation")
        if task_distribution.get("test_split") != "test":
            raise ValueError("task_distribution.test_split must be test")

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
    if config.get("task_distribution") is not None:
        metadata["task_manifests"] = {
            split: {
                "path": str(_task_manifest_path(config, split)),
                "sha256": manifest_sha256(_task_manifest_path(config, split)),
            }
            for split in ("train", "validation", "test")
        }
    write_json(output_dir / "run_metadata.json", metadata)
    return output_dir


def _task_manifest_path(config: Mapping[str, Any], split: str) -> Path:
    settings = config.get("task_distribution")
    if settings is None:
        raise ValueError("Config has no task_distribution")
    root = Path(str(settings["manifest_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / f"{split}.jsonl"


def make_first_person_env(
    config: Mapping[str, Any],
    *,
    task_split: str | None = None,
    task_selection_mode: str | None = None,
) -> FirstPersonBotEvadeEnv:
    cellworld_util.cellworld_cache_folder = str(PROJECT_ROOT / "cellworld_cache")
    values = config["environment"]
    reward_function = REWARD_FUNCTIONS[str(config["reward_function"])]
    task_settings = config.get("task_distribution")
    task_manifest_path = None
    if task_settings is not None:
        selected_split = str(task_split or task_settings["train_split"])
        task_manifest_path = str(_task_manifest_path(config, selected_split))
        if task_selection_mode is None:
            task_selection_mode = (
                "random" if selected_split == task_settings["train_split"] else "sequential"
            )
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
        passive_gaze_mode=str(values.get("passive_gaze_mode", "center")),
        fixed_head_yaw_degrees=float(values.get("fixed_head_yaw_degrees", 0.0)),
        passive_scan_targets_degrees=tuple(
            float(value)
            for value in values.get(
                "passive_scan_targets_degrees",
                (-60.0, -30.0, 0.0, 30.0, 60.0, 30.0, 0.0, -30.0),
            )
        ),
        passive_scan_dwell_steps=int(values.get("passive_scan_dwell_steps", 2)),
        task_manifest_path=task_manifest_path,
        task_selection_mode=str(task_selection_mode or "random"),
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
    task_split: str | None,
    task_selection_mode: str | None,
):
    frozen_config = copy.deepcopy(dict(config))

    def _make():
        env = make_first_person_env(
            frozen_config,
            task_split=task_split,
            task_selection_mode=task_selection_mode,
        )
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
    task_split: str | None = None,
    task_selection_mode: str | None = None,
):
    monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        _environment_factory(
            config,
            rank=rank,
            monitor_dir=monitor_dir,
            task_split=task_split,
            task_selection_mode=task_selection_mode,
        )
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
