"""Type-preserving snapshot and result artifacts for PeekBench."""

from __future__ import annotations

import base64
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


TYPE_KEY = "__peekbench_type__"


def encode_typed(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            TYPE_KEY: "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {TYPE_KEY: "tuple", "items": [encode_typed(item) for item in value]}
    if isinstance(value, list):
        return [encode_typed(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): encode_typed(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported snapshot value type: {type(value).__name__}")


def decode_typed(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_typed(item) for item in value]
    if isinstance(value, Mapping):
        value_type = value.get(TYPE_KEY)
        if value_type == "tuple":
            return tuple(decode_typed(item) for item in value["items"])
        if value_type == "ndarray":
            raw = base64.b64decode(value["data"])
            array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
            return np.array(array.reshape(tuple(value["shape"])), copy=True)
        return {key: decode_typed(item) for key, item in value.items()}
    return value


def canonical_typed_bytes(value: Any) -> bytes:
    return json.dumps(
        encode_typed(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def semantic_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the one non-semantic wall-clock field before comparisons.

    The vendored model writes ``model.last_step`` from the host wall clock on
    every step even when ``real_time=False``.  It is preserved in the saved
    full state, but cannot be part of reproducible IDs or transition equality.
    No physics, task, agent, observation, or RNG field is normalized.
    """

    normalized = copy.deepcopy(dict(state))
    model = normalized.get("model")
    if isinstance(model, dict) and "last_step" in model:
        model["last_step"] = "<nonsemantic-wall-clock>"
    return normalized


def state_digest(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_typed_bytes(semantic_state(state))).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    payload = canonical_typed_bytes(state)
    _atomic_bytes(path, gzip.compress(payload, compresslevel=6, mtime=0))


def load_state(path: Path) -> dict[str, Any]:
    encoded = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    decoded = decode_typed(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Decoded state artifact is not a dictionary")
    return decoded


def save_observation(path: Path, observation: Mapping[str, np.ndarray]) -> None:
    required = ("image_left", "image_right", "proprio", "previous_action")
    missing = [name for name in required if name not in observation]
    if missing:
        raise KeyError(f"Observation is missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.asarray(observation[name]) for name in required},
        )
    os.replace(temporary, path)


def load_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_bytes(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for record in records for key in record})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty(project_root: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(project_root: Path) -> Mapping[str, Any]:
    packages = {}
    for name in (
        "cellworld",
        "gymnasium",
        "jsonschema",
        "numpy",
        "pygame",
        "stable-baselines3",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def prepare_experiment(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    output_root = Path(str(config["output_root"]))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    experiment_dir = output_root / str(config["experiment_id"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_payload = yaml.safe_dump(
        dict(config),
        sort_keys=True,
        allow_unicode=True,
    ).encode("utf-8")
    _atomic_bytes(experiment_dir / "config.yaml", config_payload)
    metadata = {
        **environment_metadata(project_root),
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "config_hash": config["config_hash"],
    }
    write_json(experiment_dir / "run_metadata.json", metadata)
    return experiment_dir
