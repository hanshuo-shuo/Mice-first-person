"""Paired task evaluation and rendering for a binocular SAC checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from stable_baselines3 import SAC

from benchmarks.peekbench.artifacts import write_csv, write_json, write_jsonl
from solve_first_person import (
    TopDownTrajectoryRenderer,
    binocular_preview,
    combine_views,
    save_gif,
)
from training.first_person_sac import load_sac_config, make_first_person_env


METHODS = ("sac_active_gaze", "sac_head_clamped", "random_action")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render-count", type=int, default=3)
    return parser.parse_args(argv)


def _method_action(
    method: str,
    *,
    model: SAC,
    observation: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    action_space,
) -> np.ndarray:
    if method == "random_action":
        action = rng.uniform(action_space.low, action_space.high).astype(np.float32)
    else:
        action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if method == "sac_head_clamped" and action.shape[0] == 3:
            action = np.array(action, copy=True)
            action[2] = 0.0
    if action.shape != action_space.shape:
        raise ValueError(f"{method} emitted {action.shape}, expected {action_space.shape}")
    return np.clip(action, action_space.low, action_space.high).astype(np.float32)


def _scaled_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return np.ascontiguousarray(frame)
    image = Image.fromarray(frame)
    image = image.resize(
        (image.width * int(scale), image.height * int(scale)),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(image, dtype=np.uint8)


def run_episode(
    env,
    model: SAC,
    *,
    method: str,
    seed: int,
    gif_path: Path | None = None,
    gif_fps: float = 10.0,
    render_scale: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 17_000_003)
    observation, _ = env.reset(seed=int(seed))
    model_state = env.unwrapped.model
    previous_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
    initial_distance = (
        float(
            np.linalg.norm(
                previous_location
                - np.asarray(model_state.predator.state.location, dtype=np.float64),
            ),
        )
        if model_state.use_predator
        else None
    )
    minimum_distance = initial_distance
    previous_head_yaw = float(env.head_yaw_degrees)
    path_cost = 0.0
    gaze_travel = 0.0
    total_reward = 0.0
    capture_count = 0
    goal_reached = False
    visible_steps = 0
    active_look_steps = 0
    look_without_current_pixels_steps = 0
    absolute_head_action = 0.0
    steps = 0
    terminated = False
    truncated = False

    frames: list[np.ndarray] = []
    prey_trajectory = [tuple(model_state.prey.state.location)]
    predator_trajectory = (
        [tuple(model_state.predator.state.location)] if model_state.use_predator else []
    )
    top_down_renderer = None
    if gif_path is not None:
        height = int(observation["image_left"].shape[0])
        width = int(observation["image_left"].shape[1])
        top_down_renderer = TopDownTrajectoryRenderer(
            model_state,
            width=width,
            height=height,
        )
        frame = combine_views(
            binocular_preview(observation),
            top_down_renderer.render(prey_trajectory, predator_trajectory),
        )
        frames.append(_scaled_frame(frame, render_scale))

    while not (terminated or truncated):
        current_visibility = env.get_predator_visibility()
        action = _method_action(
            method,
            model=model,
            observation=observation,
            rng=rng,
            action_space=env.action_space,
        )
        if action.shape[0] == 3:
            head_action = float(action[2])
            absolute_head_action += abs(head_action)
            if abs(head_action) > 0.05:
                active_look_steps += 1
                if not bool(current_visibility["predator_pixels_visible"]):
                    look_without_current_pixels_steps += 1

        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        events = info["transition_events"]
        capture_count += int(events["capture_event"])
        goal_reached = bool(goal_reached or events["goal_event"])
        visible_steps += int(events["predator_pixels_visible"])
        if model_state.use_predator:
            minimum_distance = min(
                float(minimum_distance),
                float(events["minimum_distance"]),
            )

        current_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
        path_cost += float(np.linalg.norm(current_location - previous_location))
        previous_location = current_location
        current_head_yaw = float(env.head_yaw_degrees)
        gaze_travel += abs(current_head_yaw - previous_head_yaw)
        previous_head_yaw = current_head_yaw
        prey_trajectory.append(tuple(model_state.prey.state.location))
        if model_state.use_predator:
            predator_trajectory.append(tuple(model_state.predator.state.location))

        if top_down_renderer is not None:
            frame = combine_views(
                binocular_preview(observation),
                top_down_renderer.render(prey_trajectory, predator_trajectory),
            )
            frames.append(_scaled_frame(frame, render_scale))

    final_goal_distance = float(env.unwrapped.reward_terms["goal_distance"])
    clean_success = bool(goal_reached and capture_count == 0)
    result = {
        "method": method,
        "seed": int(seed),
        "steps": steps,
        "return": total_reward,
        "goal_reached": goal_reached,
        "clean_success": clean_success,
        "capture_count": capture_count,
        "capture_episode": bool(capture_count > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "minimum_predator_distance": minimum_distance,
        "final_goal_distance": final_goal_distance,
        "path_cost": path_cost,
        "gaze_travel_degrees": gaze_travel,
        "active_look_fraction": active_look_steps / steps if steps else 0.0,
        "look_without_current_pixels_fraction": (
            look_without_current_pixels_steps / active_look_steps
            if active_look_steps
            else 0.0
        ),
        "mean_absolute_head_action": absolute_head_action / steps if steps else 0.0,
        "predator_pixels_visible_fraction": visible_steps / steps if steps else 0.0,
    }
    if gif_path is not None:
        if not frames:
            raise RuntimeError("Rendering produced no frames")
        frames.extend([frames[-1].copy()] * max(1, int(round(gif_fps))))
        save_gif(gif_path, frames, gif_fps)
        trajectory_path = gif_path.with_name(f"{gif_path.stem}_trajectory.png")
        final_top_down = top_down_renderer.render(prey_trajectory, predator_trajectory)
        Image.fromarray(_scaled_frame(final_top_down, render_scale)).save(trajectory_path)
        result["gif_path"] = str(gif_path)
        result["trajectory_path"] = str(trajectory_path)
    return result


def _mean(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(record[field]) for record in records]))


def summarize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        distances = [
            float(record["minimum_predator_distance"])
            for record in subset
            if record["minimum_predator_distance"] is not None
        ]
        rows.append(
            {
                "method": method,
                "episodes": len(subset),
                "clean_success_rate": _mean(subset, "clean_success"),
                "goal_reach_rate": _mean(subset, "goal_reached"),
                "capture_episode_rate": _mean(subset, "capture_episode"),
                "mean_capture_count": _mean(subset, "capture_count"),
                "mean_return": _mean(subset, "return"),
                "mean_steps": _mean(subset, "steps"),
                "mean_final_goal_distance": _mean(subset, "final_goal_distance"),
                "mean_minimum_predator_distance": (
                    float(np.mean(distances)) if distances else None
                ),
                "mean_path_cost": _mean(subset, "path_cost"),
                "mean_gaze_travel_degrees": _mean(subset, "gaze_travel_degrees"),
                "mean_active_look_fraction": _mean(subset, "active_look_fraction"),
                "mean_look_without_current_pixels_fraction": _mean(
                    subset,
                    "look_without_current_pixels_fraction",
                ),
                "mean_predator_pixels_visible_fraction": _mean(
                    subset,
                    "predator_pixels_visible_fraction",
                ),
            },
        )
    return rows


def _paired_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    left_method: str,
    right_method: str,
    field: str,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    left = {
        int(record["seed"]): float(record[field])
        for record in records
        if record["method"] == left_method
    }
    right = {
        int(record["seed"]): float(record[field])
        for record in records
        if record["method"] == right_method
    }
    seeds = sorted(set(left).intersection(right))
    differences = np.asarray([left[seed] - right[seed] for seed in seeds])
    rng = np.random.default_rng(7_301_991)
    if len(differences):
        samples = rng.choice(
            differences,
            size=(int(bootstrap_samples), len(differences)),
            replace=True,
        ).mean(axis=1)
        interval = [float(value) for value in np.percentile(samples, (2.5, 97.5))]
        mean_delta = float(differences.mean())
    else:
        interval = [math.nan, math.nan]
        mean_delta = math.nan
    return {
        "left_method": left_method,
        "right_method": right_method,
        "field": field,
        "paired_episodes": len(seeds),
        "mean_delta": mean_delta,
        "bootstrap_95_interval": interval,
    }


def _render_selection(active_records: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    ordered = sorted(
        active_records,
        key=lambda record: (
            not bool(record["clean_success"]),
            int(record["capture_count"]),
            float(record["final_goal_distance"]),
            int(record["steps"]),
        ),
    )
    candidates = (
        ("best", ordered[0]),
        ("representative", ordered[len(ordered) // 2]),
        ("worst", ordered[-1]),
    )
    result = []
    seen = set()
    for label, record in candidates:
        seed = int(record["seed"])
        if seed not in seen:
            result.append((label, seed))
            seen.add(seed)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0 or args.render_count < 0:
        raise ValueError("episodes must be positive and render-count non-negative")
    config = load_sac_config(args.config)
    evaluation_config = config.get("evaluation", {})
    seed_start = int(
        args.seed_start
        if args.seed_start is not None
        else evaluation_config.get("seed_start", 900_000)
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SAC.load(args.model, device=args.device)
    env = make_first_person_env(config)
    records = []
    try:
        for method in METHODS:
            for episode in range(int(args.episodes)):
                records.append(
                    run_episode(
                        env,
                        model,
                        method=method,
                        seed=seed_start + episode,
                    ),
                )

        method_summary = summarize(records)
        paired = [
            _paired_delta(
                records,
                left_method="sac_active_gaze",
                right_method=comparator,
                field=field,
            )
            for comparator in ("sac_head_clamped", "random_action")
            for field in ("clean_success", "capture_episode", "path_cost")
        ]
        render_manifest = []
        active_records = [
            record for record in records if record["method"] == "sac_active_gaze"
        ]
        if args.render_count:
            selections = _render_selection(active_records)[: int(args.render_count)]
            render_dir = output_dir / f"{args.name}_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            for label, seed in selections:
                gif_path = render_dir / f"{label}_seed_{seed}.gif"
                rendered = run_episode(
                    env,
                    model,
                    method="sac_active_gaze",
                    seed=seed,
                    gif_path=gif_path,
                    gif_fps=float(evaluation_config.get("gif_fps", 10.0)),
                    render_scale=int(evaluation_config.get("render_scale", 3)),
                )
                render_manifest.append({"selection": label, **rendered})
    finally:
        env.close()

    summary = {
        "model": str(args.model.resolve()),
        "config": str(args.config.resolve()),
        "episodes_per_method": int(args.episodes),
        "seed_start": seed_start,
        "methods": method_summary,
        "paired_differences": paired,
        "render_selection_rule": (
            "Best, median-ranked representative, and worst active-gaze episodes "
            "under (clean failure, captures, final goal distance, steps)."
        ),
        "renders": render_manifest,
    }
    write_jsonl(output_dir / f"{args.name}.jsonl", records)
    write_csv(output_dir / f"{args.name}.csv", records)
    write_csv(output_dir / f"{args.name}_methods.csv", method_summary)
    write_json(output_dir / f"{args.name}_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate(args)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
