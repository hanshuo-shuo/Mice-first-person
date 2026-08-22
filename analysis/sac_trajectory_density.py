"""Large paired trajectory and occupancy audit for first-person SAC.

The evaluator keeps simulator coordinates strictly on the analysis side.  The
policy receives only the public binocular observation.  Outputs include flat,
pickle-free NPZ trajectory stores, episode-level records, paired statistics,
static figures, and a concise Markdown report.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
import numpy as np
import torch
from stable_baselines3 import SAC

from benchmarks.peekbench.artifacts import (
    environment_metadata,
    write_csv,
    write_json,
    write_jsonl,
)
from training.first_person_sac import (
    PROJECT_ROOT,
    load_sac_config,
    make_first_person_env,
)


METHODS = ("sac_active_gaze", "sac_head_clamped", "random_action")
METHOD_LABELS = {
    "sac_active_gaze": "SAC active gaze",
    "sac_head_clamped": "SAC head clamped",
    "random_action": "Random action",
}
METHOD_COLORS = {
    "sac_active_gaze": "#0072B2",
    "sac_head_clamped": "#D55E00",
    "random_action": "#777777",
}

_WORKER_ENV = None
_WORKER_MODEL = None
_WORKER_METHOD = ""


def _vertices(polygon) -> np.ndarray:
    vertices = polygon.vertices
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    return np.asarray(vertices, dtype=np.float64)


def _worker_close() -> None:
    global _WORKER_ENV
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()
        _WORKER_ENV = None


def _worker_initialize(
    config: Mapping[str, Any],
    model_path: str,
    method: str,
) -> None:
    global _WORKER_ENV, _WORKER_MODEL, _WORKER_METHOD
    torch.set_num_threads(1)
    _WORKER_ENV = make_first_person_env(copy.deepcopy(dict(config)))
    _WORKER_METHOD = str(method)
    _WORKER_MODEL = (
        None if method == "random_action" else SAC.load(model_path, device="cpu")
    )
    atexit.register(_worker_close)


def _action(
    observation: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    env = _WORKER_ENV
    if _WORKER_METHOD == "random_action":
        action = rng.uniform(env.action_space.low, env.action_space.high)
    else:
        action, _ = _WORKER_MODEL.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if _WORKER_METHOD == "sac_head_clamped":
            action = np.array(action, copy=True)
            action[2] = 0.0
    return np.clip(action, env.action_space.low, env.action_space.high).astype(
        np.float32,
    )


def _rollout_seed(seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = _WORKER_ENV
    method = _WORKER_METHOD
    rng = np.random.default_rng(int(seed) + 17_000_003)
    observation, _ = env.reset(seed=int(seed))
    model_state = env.unwrapped.model

    prey_locations = [np.asarray(model_state.prey.state.location, dtype=np.float32)]
    predator_locations = [
        np.asarray(model_state.predator.state.location, dtype=np.float32)
    ]
    head_yaws = [float(env.head_yaw_degrees)]
    pixels_visible = [
        bool(env.get_predator_visibility()["predator_pixels_visible"]),
    ]
    capture_events = [False]
    actions: list[np.ndarray] = []

    minimum_distance = float(
        np.linalg.norm(
            np.asarray(model_state.prey.state.location, dtype=np.float64)
            - np.asarray(model_state.predator.state.location, dtype=np.float64),
        ),
    )
    total_reward = 0.0
    path_cost = 0.0
    gaze_travel = 0.0
    capture_count = 0
    goal_reached = False
    active_look_steps = 0
    look_without_pixels_steps = 0
    absolute_head_action = 0.0
    visible_steps = 0
    terminated = False
    truncated = False
    previous_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
    previous_head_yaw = float(env.head_yaw_degrees)

    while not (terminated or truncated):
        current_visibility = env.get_predator_visibility()
        action = _action(observation, rng=rng)
        actions.append(action)
        head_action = float(action[2])
        absolute_head_action += abs(head_action)
        if abs(head_action) > 0.05:
            active_look_steps += 1
            if not bool(current_visibility["predator_pixels_visible"]):
                look_without_pixels_steps += 1

        observation, reward, terminated, truncated, info = env.step(action)
        events = info["transition_events"]
        total_reward += float(reward)
        captured = bool(events["capture_event"])
        capture_count += int(captured)
        goal_reached = bool(goal_reached or events["goal_event"])
        visible_steps += int(events["predator_pixels_visible"])
        minimum_distance = min(minimum_distance, float(events["minimum_distance"]))

        current_location = np.asarray(model_state.prey.state.location, dtype=np.float64)
        path_cost += float(np.linalg.norm(current_location - previous_location))
        previous_location = current_location
        current_head_yaw = float(env.head_yaw_degrees)
        gaze_travel += abs(current_head_yaw - previous_head_yaw)
        previous_head_yaw = current_head_yaw

        prey_locations.append(np.asarray(model_state.prey.state.location, dtype=np.float32))
        predator_locations.append(
            np.asarray(model_state.predator.state.location, dtype=np.float32),
        )
        head_yaws.append(current_head_yaw)
        pixels_visible.append(bool(events["predator_pixels_visible"]))
        capture_events.append(captured)

    steps = len(actions)
    record = {
        "method": method,
        "seed": int(seed),
        "steps": steps,
        "return": total_reward,
        "goal_reached": goal_reached,
        "clean_success": bool(goal_reached and capture_count == 0),
        "capture_count": capture_count,
        "capture_episode": bool(capture_count > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "minimum_predator_distance": minimum_distance,
        "final_goal_distance": float(env.unwrapped.reward_terms["goal_distance"]),
        "path_cost": path_cost,
        "gaze_travel_degrees": gaze_travel,
        "active_look_fraction": active_look_steps / steps if steps else 0.0,
        "look_without_current_pixels_fraction": (
            look_without_pixels_steps / active_look_steps
            if active_look_steps
            else 0.0
        ),
        "mean_absolute_head_action": absolute_head_action / steps if steps else 0.0,
        "predator_pixels_visible_fraction": visible_steps / steps if steps else 0.0,
    }
    trace = {
        "prey_xy": np.asarray(prey_locations, dtype=np.float32),
        "predator_xy": np.asarray(predator_locations, dtype=np.float32),
        "head_yaw_degrees": np.asarray(head_yaws, dtype=np.float32),
        "predator_pixels_visible": np.asarray(pixels_visible, dtype=np.bool_),
        "capture_event": np.asarray(capture_events, dtype=np.bool_),
        "actions": np.asarray(actions, dtype=np.float32),
    }
    return record, trace


def run_method(
    config: Mapping[str, Any],
    *,
    model_path: Path,
    method: str,
    seeds: Sequence[int],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    if workers == 1:
        _worker_initialize(config, str(model_path), method)
        try:
            results = [_rollout_seed(int(seed)) for seed in seeds]
        finally:
            _worker_close()
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=context,
            initializer=_worker_initialize,
            initargs=(dict(config), str(model_path), method),
        ) as executor:
            results = list(executor.map(_rollout_seed, seeds, chunksize=4))
    records = [result[0] for result in results]
    traces = [result[1] for result in results]
    return records, traces


def pack_traces(
    seeds: Sequence[int],
    traces: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    state_lengths = np.asarray([len(trace["prey_xy"]) for trace in traces], dtype=np.int64)
    action_lengths = np.asarray([len(trace["actions"]) for trace in traces], dtype=np.int64)
    state_offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(state_lengths)),
    )
    action_offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(action_lengths)),
    )
    return {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "state_offsets": state_offsets,
        "action_offsets": action_offsets,
        "prey_xy": np.concatenate([trace["prey_xy"] for trace in traces], axis=0),
        "predator_xy": np.concatenate(
            [trace["predator_xy"] for trace in traces],
            axis=0,
        ),
        "head_yaw_degrees": np.concatenate(
            [trace["head_yaw_degrees"] for trace in traces],
        ),
        "predator_pixels_visible": np.concatenate(
            [trace["predator_pixels_visible"] for trace in traces],
        ),
        "capture_event": np.concatenate(
            [trace["capture_event"] for trace in traces],
        ),
        "actions": np.concatenate([trace["actions"] for trace in traces], axis=0),
    }


def save_trace_store(path: Path, packed: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **packed)
    temporary.replace(path)


def unpack_state_traces(packed: Mapping[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    traces = []
    offsets = np.asarray(packed["state_offsets"], dtype=np.int64)
    action_offsets = np.asarray(packed["action_offsets"], dtype=np.int64)
    for index in range(len(offsets) - 1):
        start, end = int(offsets[index]), int(offsets[index + 1])
        action_start = int(action_offsets[index])
        action_end = int(action_offsets[index + 1])
        traces.append(
            {
                "prey_xy": np.asarray(packed["prey_xy"])[start:end],
                "predator_xy": np.asarray(packed["predator_xy"])[start:end],
                "head_yaw_degrees": np.asarray(packed["head_yaw_degrees"])[start:end],
                "predator_pixels_visible": np.asarray(
                    packed["predator_pixels_visible"],
                )[start:end],
                "capture_event": np.asarray(packed["capture_event"])[start:end],
                "actions": np.asarray(packed["actions"])[action_start:action_end],
            },
        )
    return traces


def episode_normalized_density(
    traces: Sequence[Mapping[str, np.ndarray]],
    *,
    bins: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    density = np.zeros((bins, bins), dtype=np.float64)
    for trace in traces:
        points = np.asarray(trace["prey_xy"], dtype=np.float64)
        histogram, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=bins,
            range=bounds,
        )
        mass = float(histogram.sum())
        if mass:
            density += histogram / mass
    if traces:
        density /= len(traces)
    return density


def capture_density(
    traces: Sequence[Mapping[str, np.ndarray]],
    *,
    bins: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    density = np.zeros((bins, bins), dtype=np.float64)
    for trace in traces:
        points = np.asarray(trace["prey_xy"], dtype=np.float64)
        mask = np.asarray(trace["capture_event"], dtype=np.bool_)
        capture_points = points[mask]
        if len(capture_points):
            histogram, _, _ = np.histogram2d(
                capture_points[:, 0],
                capture_points[:, 1],
                bins=bins,
                range=bounds,
            )
            density += histogram
    if traces:
        density /= len(traces)
    return density


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total),
    ) / denominator
    return centre - margin, centre + margin


def summarize_methods(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        if not subset:
            continue
        success_count = sum(bool(record["clean_success"]) for record in subset)
        capture_count = sum(bool(record["capture_episode"]) for record in subset)
        success_interval = wilson_interval(success_count, len(subset))
        capture_interval = wilson_interval(capture_count, len(subset))

        def mean(field: str) -> float:
            return float(np.mean([float(record[field]) for record in subset]))

        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "episodes": len(subset),
                "clean_success_rate": success_count / len(subset),
                "clean_success_ci_low": success_interval[0],
                "clean_success_ci_high": success_interval[1],
                "capture_episode_rate": capture_count / len(subset),
                "capture_episode_ci_low": capture_interval[0],
                "capture_episode_ci_high": capture_interval[1],
                "goal_reach_rate": mean("goal_reached"),
                "mean_capture_count": mean("capture_count"),
                "mean_return": mean("return"),
                "mean_steps": mean("steps"),
                "mean_minimum_predator_distance": mean("minimum_predator_distance"),
                "mean_path_cost": mean("path_cost"),
                "mean_gaze_travel_degrees": mean("gaze_travel_degrees"),
                "mean_active_look_fraction": mean("active_look_fraction"),
                "mean_look_without_current_pixels_fraction": mean(
                    "look_without_current_pixels_fraction",
                ),
                "mean_predator_pixels_visible_fraction": mean(
                    "predator_pixels_visible_fraction",
                ),
            },
        )
    return rows


def paired_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    by_method = {
        method: {
            int(record["seed"]): record
            for record in records
            if record["method"] == method
        }
        for method in METHODS
    }
    fields = (
        "clean_success",
        "capture_episode",
        "return",
        "steps",
        "minimum_predator_distance",
        "path_cost",
        "gaze_travel_degrees",
    )
    rows = []
    for comparator_index, comparator in enumerate(("sac_head_clamped", "random_action")):
        seeds = sorted(
            set(by_method["sac_active_gaze"]).intersection(by_method[comparator]),
        )
        for field_index, field in enumerate(fields):
            differences = np.asarray(
                [
                    float(by_method["sac_active_gaze"][seed][field])
                    - float(by_method[comparator][seed][field])
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            rng = np.random.default_rng(20260821 + comparator_index * 100 + field_index)
            bootstrap_means = np.empty((bootstrap_samples,), dtype=np.float64)
            for sample_index in range(bootstrap_samples):
                indices = rng.integers(0, len(differences), size=len(differences))
                bootstrap_means[sample_index] = differences[indices].mean()
            low, high = np.percentile(bootstrap_means, (2.5, 97.5))
            rows.append(
                {
                    "left_method": "sac_active_gaze",
                    "right_method": comparator,
                    "field": field,
                    "paired_episodes": len(seeds),
                    "mean_delta": float(differences.mean()),
                    "bootstrap_95_low": float(low),
                    "bootstrap_95_high": float(high),
                },
            )
    return rows


def arena_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    env = make_first_person_env(config)
    try:
        env.reset(seed=int(config["seed"]))
        model = env.unwrapped.model
        return {
            "arena": _vertices(model.arena),
            "occlusions": [_vertices(value) for value in model.occlusions],
            "goal": np.asarray(model.goal_location, dtype=np.float64),
            "start": np.asarray(model.prey.state.location, dtype=np.float64),
        }
    finally:
        env.close()


def _draw_arena(ax, spec: Mapping[str, Any], *, fill: bool = True) -> None:
    arena = np.asarray(spec["arena"])
    ax.add_patch(
        Polygon(
            arena,
            closed=True,
            facecolor="#F0F0EA" if fill else "none",
            edgecolor="#30343B",
            linewidth=1.0,
            zorder=0,
        ),
    )
    for occlusion in spec["occlusions"]:
        ax.add_patch(
            Polygon(
                np.asarray(occlusion),
                closed=True,
                facecolor="#50545B",
                edgecolor="#30343B",
                linewidth=0.5,
                zorder=4,
            ),
        )
    ax.scatter(*spec["start"], marker="o", s=28, color="#00A6D6", zorder=7, label="start")
    ax.scatter(*spec["goal"], marker="*", s=80, color="#009E73", zorder=7, label="goal")
    minimum = arena.min(axis=0)
    maximum = arena.max(axis=0)
    padding = (maximum - minimum) * 0.03
    ax.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    ax.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    ax.set_aspect("equal")
    ax.set_xlabel("World x")
    ax.set_ylabel("World y")


def plot_trajectory_overview(
    path: Path,
    *,
    traces_by_method: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    summaries: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
    sample_count: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        traces = traces_by_method[method]
        indices = np.linspace(
            0,
            len(traces) - 1,
            num=min(sample_count, len(traces)),
            dtype=int,
        )
        _draw_arena(ax, spec)
        for index in indices:
            trace = traces[int(index)]
            points = trace["prey_xy"]
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=METHOD_COLORS[method],
                alpha=0.10,
                linewidth=0.8,
                zorder=2,
            )
        capture_points = [
            trace["prey_xy"][trace["capture_event"]]
            for trace in traces
            if np.any(trace["capture_event"])
        ]
        if capture_points:
            values = np.concatenate(capture_points, axis=0)
            ax.scatter(
                values[:, 0],
                values[:, 1],
                marker="x",
                s=8,
                color="#CC3311",
                alpha=0.35,
                zorder=6,
            )
        summary = summaries[method]
        ax.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"success {summary['clean_success_rate']:.1%}, "
            f"capture {summary['capture_episode_rate']:.1%}",
        )
    figure.suptitle(f"Paired held-out trajectories ({sample_count} displayed per method)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _positive_limits(densities: Sequence[np.ndarray]) -> tuple[float, float] | None:
    positive = np.concatenate([density[density > 0] for density in densities])
    if not len(positive):
        return None
    return max(float(np.percentile(positive, 2.0)), 1e-8), float(positive.max())


def plot_density_panels(
    path: Path,
    *,
    densities: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    title: str,
    colorbar_label: str,
) -> None:
    arena = np.asarray(spec["arena"])
    minimum = arena.min(axis=0)
    maximum = arena.max(axis=0)
    values = [densities[method] for method in METHODS]
    limits = _positive_limits(values)
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    image = None
    for ax, method in zip(axes, METHODS):
        density = np.ma.masked_less_equal(densities[method].T, 0.0)
        if limits is not None and density.count():
            image = ax.imshow(
                density,
                origin="lower",
                extent=(minimum[0], maximum[0], minimum[1], maximum[1]),
                cmap="viridis",
                norm=LogNorm(vmin=limits[0], vmax=limits[1]),
                interpolation="nearest",
                zorder=1,
            )
        _draw_arena(ax, spec, fill=False)
        if not density.count():
            ax.text(
                0.5,
                0.5,
                "No events",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        ax.set_title(METHOD_LABELS[method])
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    probabilities = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    return ordered, probabilities


def plot_outcome_distributions(
    path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    x = np.arange(len(METHODS), dtype=np.float64)
    width = 0.36
    success = [summaries[method]["clean_success_rate"] for method in METHODS]
    capture = [summaries[method]["capture_episode_rate"] for method in METHODS]
    success_errors = np.asarray(
        [
            [
                summaries[method]["clean_success_rate"]
                - summaries[method]["clean_success_ci_low"],
                summaries[method]["clean_success_ci_high"]
                - summaries[method]["clean_success_rate"],
            ]
            for method in METHODS
        ],
    ).T
    capture_errors = np.asarray(
        [
            [
                summaries[method]["capture_episode_rate"]
                - summaries[method]["capture_episode_ci_low"],
                summaries[method]["capture_episode_ci_high"]
                - summaries[method]["capture_episode_rate"],
            ]
            for method in METHODS
        ],
    ).T
    axes[0, 0].bar(x - width / 2, success, width, yerr=success_errors, label="Clean success")
    axes[0, 0].bar(x + width / 2, capture, width, yerr=capture_errors, label="Capture episode")
    axes[0, 0].set_xticks(x, ["Active", "Clamped", "Random"])
    axes[0, 0].set_ylim(0.0, 1.08)
    axes[0, 0].set_ylabel("Episode rate")
    axes[0, 0].set_title("Task outcomes with Wilson 95% intervals")
    axes[0, 0].legend(frameon=False)

    panels = (
        (axes[0, 1], "minimum_predator_distance", "Minimum predator distance", "Distance"),
        (axes[1, 0], "steps", "Episode length", "Steps"),
        (axes[1, 1], "path_cost", "Physical path cost", "World distance"),
    )
    for ax, field, title, xlabel in panels:
        for method in METHODS:
            values = [
                float(record[field])
                for record in records
                if record["method"] == method
            ]
            ordered, probability = _ecdf(values)
            ax.plot(
                ordered,
                probability,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linewidth=1.8,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative episode fraction")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[1, 1].legend(frameon=False, loc="lower right")
    figure.suptitle("Safety and efficiency distributions across 1,000 paired seeds")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_gaze_density(
    path: Path,
    *,
    active_traces: Sequence[Mapping[str, np.ndarray]],
) -> None:
    bins_progress = 50
    bins_yaw = 60
    density = np.zeros((bins_progress, bins_yaw), dtype=np.float64)
    for trace in active_traces:
        yaw = np.asarray(trace["head_yaw_degrees"], dtype=np.float64)
        progress = np.linspace(0.0, 1.0, num=len(yaw))
        histogram, _, _ = np.histogram2d(
            progress,
            yaw,
            bins=(bins_progress, bins_yaw),
            range=((0.0, 1.0), (-60.0, 60.0)),
        )
        density += histogram / max(float(histogram.sum()), 1.0)
    density /= max(len(active_traces), 1)
    figure, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    image = ax.imshow(
        density.T,
        origin="lower",
        extent=(0.0, 1.0, -60.0, 60.0),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("Normalized episode progress")
    ax.set_ylabel("Head yaw (degrees)")
    ax.set_title("Active-gaze head-yaw occupancy")
    figure.colorbar(image, ax=ax, label="Mean per-episode occupancy mass / bin")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report_markdown(
    *,
    methods: Mapping[str, Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    episodes: int,
    seed_start: int,
    elapsed_seconds: float,
) -> str:
    lines = [
        "# SAC trajectory-density audit",
        "",
        f"This audit uses {episodes:,} paired held-out seeds "
        f"(`{seed_start}`--`{seed_start + episodes - 1}`) per method.",
        f"Evaluation wall time: {elapsed_seconds / 3600.0:.2f} hours.",
        "",
        "## Method summary",
        "",
        "| Method | Clean success (95% CI) | Capture episode | Goal reach | Steps | Min distance | Path cost | Gaze travel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = methods[method]
        lines.append(
            f"| {row['label']} | {row['clean_success_rate']:.1%} "
            f"[{row['clean_success_ci_low']:.1%}, {row['clean_success_ci_high']:.1%}] "
            f"| {row['capture_episode_rate']:.1%} | {row['goal_reach_rate']:.1%} "
            f"| {row['mean_steps']:.2f} | {row['mean_minimum_predator_distance']:.3f} "
            f"| {row['mean_path_cost']:.3f} | {row['mean_gaze_travel_degrees']:.1f} |",
        )
    lines.extend(
        (
            "",
            "## Paired active-gaze effects",
            "",
        ),
    )
    for comparator in ("sac_head_clamped", "random_action"):
        lines.append(f"Versus **{METHOD_LABELS[comparator]}**:")
        lines.append("")
        for field in ("clean_success", "capture_episode", "steps", "minimum_predator_distance", "path_cost"):
            row = next(
                value
                for value in paired
                if value["right_method"] == comparator and value["field"] == field
            )
            lines.append(
                f"- `{field}` delta: {row['mean_delta']:.4f}; paired bootstrap 95% "
                f"CI `[{row['bootstrap_95_low']:.4f}, {row['bootstrap_95_high']:.4f}]`.",
            )
        lines.append("")
    active = methods["sac_active_gaze"]
    lines.extend(
        (
            "## Gaze and evidence boundary",
            "",
            f"The active policy moves its head on {active['mean_active_look_fraction']:.1%} "
            f"of steps and travels {active['mean_gaze_travel_degrees']:.1f} degrees per "
            "episode on average. This measures effective but potentially expensive "
            "continuous scanning, not sparse peeking.",
            "",
            "Head clamping is an inference-time distribution shift, not an independently "
            "trained fixed-gaze policy. These paired results establish dependence of this "
            "trained policy on head control; a causal active-versus-fixed claim still "
            "requires matched independent training seeds.",
            "",
            "## Figures",
            "",
            "- `trajectory_overview.png`: same-seed route geometry and capture locations.",
            "- `occupancy_density.png`: episode-normalized prey occupancy; long episodes do not receive extra total mass.",
            "- `capture_density.png`: expected capture-event locations per episode.",
            "- `outcome_distributions.png`: outcome rates and safety/efficiency ECDFs.",
            "- `gaze_density.png`: head yaw over normalized episode progress.",
        ),
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--trajectory-sample", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0 or args.workers <= 0 or args.bins <= 1:
        raise ValueError("episodes/workers must be positive and bins must exceed one")
    if tuple(args.methods) != METHODS:
        raise ValueError(f"Project audit requires methods in order: {METHODS}")
    config = load_sac_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.episodes)))
    all_records: list[dict[str, Any]] = []
    traces_by_method: dict[str, list[dict[str, np.ndarray]]] = {}
    started = time.perf_counter()

    for method in METHODS:
        print(f"trajectory-audit method={method} episodes={len(seeds)} workers={args.workers}")
        records, traces = run_method(
            config,
            model_path=args.model.resolve(),
            method=method,
            seeds=seeds,
            workers=min(int(args.workers), len(seeds)),
        )
        all_records.extend(records)
        traces_by_method[method] = traces
        packed = pack_traces(seeds, traces)
        save_trace_store(output_dir / f"traces_{method}.npz", packed)
        write_jsonl(output_dir / f"episodes_{method}.jsonl", records)
        print(
            f"trajectory-audit complete method={method} "
            f"success={np.mean([r['clean_success'] for r in records]):.4f} "
            f"capture={np.mean([r['capture_episode'] for r in records]):.4f}",
        )

    method_rows = summarize_methods(all_records)
    methods_by_name = {row["method"]: row for row in method_rows}
    paired_rows = paired_differences(
        all_records,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    elapsed_seconds = time.perf_counter() - started
    spec = arena_spec(config)
    arena = np.asarray(spec["arena"])
    minimum = arena.min(axis=0)
    maximum = arena.max(axis=0)
    bounds = ((float(minimum[0]), float(maximum[0])), (float(minimum[1]), float(maximum[1])))
    occupancy = {
        method: episode_normalized_density(
            traces_by_method[method],
            bins=int(args.bins),
            bounds=bounds,
        )
        for method in METHODS
    }
    captures = {
        method: capture_density(
            traces_by_method[method],
            bins=int(args.bins),
            bounds=bounds,
        )
        for method in METHODS
    }

    plot_trajectory_overview(
        output_dir / "trajectory_overview.png",
        traces_by_method=traces_by_method,
        summaries=methods_by_name,
        spec=spec,
        sample_count=int(args.trajectory_sample),
    )
    plot_density_panels(
        output_dir / "occupancy_density.png",
        densities=occupancy,
        spec=spec,
        title="Episode-normalized prey occupancy",
        colorbar_label="Mean occupancy mass per episode / bin",
    )
    plot_density_panels(
        output_dir / "capture_density.png",
        densities=captures,
        spec=spec,
        title="Capture-event density",
        colorbar_label="Capture events per episode / bin",
    )
    plot_outcome_distributions(
        output_dir / "outcome_distributions.png",
        records=all_records,
        summaries=methods_by_name,
    )
    plot_gaze_density(
        output_dir / "gaze_density.png",
        active_traces=traces_by_method["sac_active_gaze"],
    )

    write_jsonl(output_dir / "episodes.jsonl", all_records)
    write_csv(output_dir / "episodes.csv", all_records)
    write_csv(output_dir / "methods.csv", method_rows)
    write_csv(output_dir / "paired_differences.csv", paired_rows)
    metadata = {
        **environment_metadata(PROJECT_ROOT),
        "source_model": str(args.model.resolve()),
        "source_config": str(args.config.resolve()),
        "episodes_per_method": int(args.episodes),
        "seed_start": int(args.seed_start),
        "seed_end": int(args.seed_start) + int(args.episodes) - 1,
        "workers": int(args.workers),
        "elapsed_seconds": elapsed_seconds,
        "density_semantics": "each episode normalized to unit mass before averaging",
    }
    summary = {
        "metadata": metadata,
        "methods": method_rows,
        "paired_differences": paired_rows,
        "artifacts": [
            "trajectory_overview.png",
            "occupancy_density.png",
            "capture_density.png",
            "outcome_distributions.png",
            "gaze_density.png",
        ],
    }
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        report_markdown(
            methods=methods_by_name,
            paired=paired_rows,
            episodes=int(args.episodes),
            seed_start=int(args.seed_start),
            elapsed_seconds=elapsed_seconds,
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
