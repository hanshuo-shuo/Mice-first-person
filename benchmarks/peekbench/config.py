"""Configuration loading and validation for PeekBench."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


REQUIRED_GAZE_CANDIDATES = [-60.0, -30.0, 0.0, 30.0, 60.0]
STATE_CATEGORIES = [
    "predator_visible",
    "geometric_outside_frustum",
    "frustum_pixel_occluded",
    "recently_visible_hidden",
    "no_predator_control",
]

DEFAULT_CONFIG: Mapping[str, Any] = {
    "experiment_id": "peekbench_p0",
    "seed": 23,
    "num_snapshots": 25,
    "output_root": "results/peekbench",
    "environment": {
        "world_name": "21_05",
        "max_step": 400,
        "time_step": 0.10,
        "predator_prey_forward_speed_ratio": 0.15,
        "vision_width": 64,
        "vision_height": 48,
        "vision_fov": 120.0,
        "vision_far_clip": 2.0,
        "vision_detection_range": 2.0,
    },
    "sampling": {
        "sources": ["near_occlusion", "junction", "peek_location"],
        "categories": STATE_CATEGORIES,
        "minimum_predator_distance": 0.16,
        "maximum_predator_distance": 0.80,
        "recent_visibility_horizon": 8,
        "candidate_search_limit": 64,
    },
    "gaze_candidates_degrees": REQUIRED_GAZE_CANDIDATES,
    "branch": {
        "horizon_steps": 8,
        "risk_distance": 0.18,
        "minimum_distance_improvement": 0.03,
    },
    "policy": {
        "model": "openai/gpt-4.1-mini",
        "provider": {
            "order": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        },
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "retry_backoff_seconds": 0.5,
        "max_history_frames": 4,
    },
}


def _deep_merge(
    target: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Mapping[str, Any]) -> str:
    identity_config = copy.deepcopy(dict(config))
    identity_config.pop("output_root", None)
    identity_config.pop("experiment_id", None)
    return hashlib.sha256(canonical_json(identity_config).encode("utf-8")).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if not str(config.get("experiment_id", "")).strip():
        raise ValueError("experiment_id must be non-empty")
    if int(config.get("num_snapshots", 0)) <= 0:
        raise ValueError("num_snapshots must be positive")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("seed must be non-negative")

    environment = config.get("environment", {})
    if environment.get("world_name") != "21_05":
        raise ValueError("PeekBench P0 supports only the offline-ready 21_05 world")
    if int(environment.get("vision_width", 0)) < 16:
        raise ValueError("vision_width must be >= 16")
    if int(environment.get("vision_height", 0)) < 16:
        raise ValueError("vision_height must be >= 16")

    candidates = [float(value) for value in config.get("gaze_candidates_degrees", [])]
    if candidates != REQUIRED_GAZE_CANDIDATES:
        raise ValueError(
            "PeekBench P0 gaze candidates must be exactly "
            f"{REQUIRED_GAZE_CANDIDATES}",
        )

    sampling = config.get("sampling", {})
    categories = list(sampling.get("categories", []))
    unknown_categories = sorted(set(categories).difference(STATE_CATEGORIES))
    if not categories or unknown_categories:
        raise ValueError(f"Invalid sampling categories: {unknown_categories}")
    allowed_sources = {"near_occlusion", "junction", "peek_location"}
    sources = list(sampling.get("sources", []))
    if not sources or set(sources).difference(allowed_sources):
        raise ValueError(f"Invalid sampling sources: {sources}")
    minimum = float(sampling.get("minimum_predator_distance", 0.0))
    maximum = float(sampling.get("maximum_predator_distance", 0.0))
    if not 0 < minimum < maximum:
        raise ValueError("Predator distance bounds must satisfy 0 < min < max")

    branch = config.get("branch", {})
    if int(branch.get("horizon_steps", 0)) <= 0:
        raise ValueError("branch.horizon_steps must be positive")


def load_config(
    path: str | Path,
    *,
    experiment_id: str | None = None,
    num_snapshots: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("PeekBench config must contain a YAML mapping")
    resolved = _deep_merge(copy.deepcopy(dict(DEFAULT_CONFIG)), loaded)
    if experiment_id is not None:
        resolved["experiment_id"] = experiment_id
    if num_snapshots is not None:
        resolved["num_snapshots"] = int(num_snapshots)
    if output_root is not None:
        resolved["output_root"] = str(output_root)
    resolved["seed"] = int(resolved["seed"])
    resolved["num_snapshots"] = int(resolved["num_snapshots"])
    resolved["gaze_candidates_degrees"] = [
        float(value) for value in resolved["gaze_candidates_degrees"]
    ]
    validate_config(resolved)
    resolved["config_hash"] = config_hash(resolved)
    return resolved
