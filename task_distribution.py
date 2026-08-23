"""Versioned task banks for matched first-person training and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_FIELDS = (
    "task_id",
    "split",
    "world_name",
    "start_cell_id",
    "goal_cell_id",
    "predator_cell_id",
    "start_location",
    "goal_location",
    "predator_location",
    "start_region",
    "goal_region",
    "path_cell_ids",
    "path_regions",
    "prey_body_heading_degrees",
    "predator_body_heading_degrees",
    "predator_speed_ratio",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_digest(record: Mapping[str, Any]) -> str:
    identity = {name: record[name] for name in TASK_FIELDS if name != "task_id"}
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def load_task_records(path: str | Path) -> list[dict[str, Any]]:
    selected = Path(path)
    records = [
        json.loads(line)
        for line in selected.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Task manifest is empty: {selected}")
    for index, record in enumerate(records):
        missing = [name for name in TASK_FIELDS if name not in record]
        if missing:
            raise KeyError(f"Task {index} in {selected} is missing fields: {missing}")
        if task_digest(record)[:16] not in str(record["task_id"]):
            raise ValueError(f"Task identity digest mismatch at {selected}:{index + 1}")
    task_ids = [str(record["task_id"]) for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Task IDs are not unique in {selected}")
    return records


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TaskBank:
    """Select exact task records randomly for training or cyclically for eval."""

    MODES = ("random", "sequential")

    def __init__(self, records: Sequence[Mapping[str, Any]], *, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Task selection mode must be one of {self.MODES}")
        if not records:
            raise ValueError("TaskBank requires at least one task")
        self.records = tuple(copy.deepcopy(dict(record)) for record in records)
        self.mode = str(mode)
        self.cursor = 0
        self.by_id = {str(record["task_id"]): record for record in self.records}

    @classmethod
    def from_path(cls, path: str | Path, *, mode: str) -> "TaskBank":
        return cls(load_task_records(path), mode=mode)

    def select(
        self,
        rng: np.random.Generator,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = {} if options is None else dict(options)
        requested_id = options.get("task_id")
        if requested_id is not None:
            task = self.by_id.get(str(requested_id))
            if task is None:
                raise KeyError(f"Unknown task_id: {requested_id}")
            return copy.deepcopy(dict(task))
        if "task_index" in options:
            index = int(options["task_index"])
            if not 0 <= index < len(self.records):
                raise IndexError(f"task_index {index} is outside the task bank")
        elif self.mode == "random":
            index = int(rng.integers(0, len(self.records)))
        else:
            index = self.cursor % len(self.records)
            self.cursor += 1
        return copy.deepcopy(dict(self.records[index]))

    def get_state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "cursor": int(self.cursor)}

    def set_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("mode", self.mode) != self.mode:
            raise ValueError("TaskBank state mode does not match this bank")
        self.cursor = int(state.get("cursor", 0))


def validate_split_contract(
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    heldout_region: str,
    validation_region_pair: tuple[str, str],
    heldout_occlusion_cell_ids: set[int] | None = None,
) -> dict[str, Any]:
    required = {"train", "validation", "test"}
    if set(manifests) != required:
        raise ValueError(f"Task manifests must contain exactly {sorted(required)}")
    task_ids = {
        split: {str(record["task_id"]) for record in records}
        for split, records in manifests.items()
    }
    pair_sets = {
        split: {
            (int(record["start_cell_id"]), int(record["goal_cell_id"]))
            for record in records
        }
        for split, records in manifests.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if task_ids[left].intersection(task_ids[right]):
            raise ValueError(f"Task IDs overlap between {left} and {right}")
        if pair_sets[left].intersection(pair_sets[right]):
            raise ValueError(f"Exact start-goal pairs overlap between {left} and {right}")

    train_pairs = {
        (str(record["start_region"]), str(record["goal_region"]))
        for record in manifests["train"]
    }
    if validation_region_pair in train_pairs:
        raise ValueError("Validation region pair appears in training")
    if {
        (str(record["start_region"]), str(record["goal_region"]))
        for record in manifests["validation"]
    } != {validation_region_pair}:
        raise ValueError("Validation tasks do not use exactly the registered held-out pair")
    if any(heldout_region in record["path_regions"] for record in manifests["train"]):
        raise ValueError("A training path enters the held-out test region")
    if any(heldout_region in record["path_regions"] for record in manifests["validation"]):
        raise ValueError("A validation path enters the held-out test region")
    if any(heldout_region not in record["path_regions"] for record in manifests["test"]):
        raise ValueError("A test path does not enter the held-out test region")
    if heldout_occlusion_cell_ids is not None and any(
        not heldout_occlusion_cell_ids.intersection(
            int(cell_id) for cell_id in record["path_cell_ids"]
        )
        for record in manifests["test"]
    ):
        raise ValueError("A test path misses the held-out occlusion-region cells")
    return {
        "task_counts": {split: len(records) for split, records in manifests.items()},
        "exact_pair_overlap": False,
        "validation_region_pair_held_out": True,
        "test_region_held_out": True,
        "test_paths_cross_heldout_occlusion_cells": (
            heldout_occlusion_cell_ids is not None
        ),
    }
