"""Train SAC from the public binocular first-person observation.

The Gym observation remains exactly image_left, image_right, proprio, and
previous_action.  The policy never receives simulator coordinates, visibility
labels, rewards, or exact-state dictionaries.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Sequence

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from benchmarks.peekbench.artifacts import write_json
from training.first_person_sac import (
    PROJECT_ROOT,
    apply_smoke_overrides,
    load_sac_config,
    make_vec_env,
    policy_kwargs,
    prepare_sac_experiment,
    validate_sac_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sac_cnn_active_gaze.yaml"),
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU gradient/save/load smoke instead of training",
    )
    return parser.parse_args(argv)


def resolved_config(args: argparse.Namespace) -> dict:
    config = load_sac_config(args.config)
    if args.smoke:
        config = apply_smoke_overrides(config)
    else:
        config = copy.deepcopy(config)
    if args.experiment_id:
        config["experiment_id"] = str(args.experiment_id)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = int(args.total_timesteps)
    if args.num_envs is not None:
        config["training"]["num_envs"] = int(args.num_envs)
    if args.device:
        config["training"]["device"] = args.device
        if args.device != "cuda":
            config["training"]["require_cuda"] = False
    validate_sac_config(config)
    return config


def train(config: dict) -> dict:
    output_dir = prepare_sac_experiment(config, project_root=PROJECT_ROOT)
    training = config["training"]
    requested_device = str(training["device"])
    cuda_available = bool(torch.cuda.is_available())
    if bool(training.get("require_cuda", False)) and not cuda_available:
        raise RuntimeError("CUDA is required by this run but torch.cuda.is_available() is false")
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("training.device=cuda but CUDA is unavailable")

    num_envs = int(training["num_envs"])
    train_env = make_vec_env(
        config,
        num_envs=num_envs,
        monitor_dir=output_dir / "monitor" / "train",
    )
    eval_env = make_vec_env(
        config,
        num_envs=1,
        monitor_dir=output_dir / "monitor" / "eval",
        seed_offset=1_000_000,
    )
    checkpoints_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    evaluations_dir = output_dir / "evaluations"
    tensorboard_dir = output_dir / "tensorboard"
    for directory in (checkpoints_dir, best_dir, evaluations_dir, tensorboard_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(training["checkpoint_freq"]) // num_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="sac_binocular",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(evaluations_dir),
        eval_freq=max(int(training["eval_freq"]) // num_envs, 1),
        n_eval_episodes=int(training["eval_episodes"]),
        deterministic=True,
        render=False,
        verbose=1,
        warn=True,
    )

    model = None
    started = time.perf_counter()
    try:
        model = SAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=float(training["learning_rate"]),
            buffer_size=int(training["buffer_size"]),
            learning_starts=int(training["learning_starts"]),
            batch_size=int(training["batch_size"]),
            tau=float(training["tau"]),
            gamma=float(training["gamma"]),
            train_freq=int(training["train_freq"]),
            gradient_steps=int(training["gradient_steps"]),
            ent_coef=str(training["ent_coef"]),
            target_update_interval=int(training["target_update_interval"]),
            policy_kwargs=policy_kwargs(config),
            tensorboard_log=str(tensorboard_dir),
            seed=int(config["seed"]),
            device=requested_device,
            verbose=1,
        )
        model.learn(
            total_timesteps=int(training["total_timesteps"]),
            callback=CallbackList([checkpoint_callback, eval_callback]),
            log_interval=int(training["log_interval"]),
            progress_bar=False,
        )
        final_model_path = checkpoints_dir / "final_model"
        model.save(final_model_path)
        elapsed_seconds = time.perf_counter() - started
        summary = {
            "experiment_id": config["experiment_id"],
            "total_timesteps": int(model.num_timesteps),
            "elapsed_seconds": elapsed_seconds,
            "transitions_per_second": float(model.num_timesteps / elapsed_seconds),
            "requested_device": requested_device,
            "resolved_device": str(model.device),
            "cuda_available": cuda_available,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if cuda_available else None
            ),
            "num_envs": num_envs,
            "final_model": str((final_model_path.with_suffix(".zip")).relative_to(output_dir)),
            "best_model": (
                "best/best_model.zip"
                if (best_dir / "best_model.zip").exists()
                else None
            ),
        }
        write_json(output_dir / "training_summary.json", summary)
        return summary
    finally:
        if model is not None:
            del model
        eval_env.close()
        train_env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolved_config(args)
    summary = train(config)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
