"""Generate deterministic matched-training task manifests for world 21_05."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks.peekbench.artifacts import write_json, write_jsonl
from botevade_gym import cwgame
from task_distribution import task_digest, validate_split_contract


REGIONS = ("northwest", "northeast", "southwest", "southeast")
TRAIN_REGIONS = ("northwest", "northeast", "southwest")
HELDOUT_TEST_REGION = "southeast"
VALIDATION_REGION_PAIR = ("northwest", "northeast")
TRAIN_REGION_PAIRS = tuple(
    (start, goal)
    for start in TRAIN_REGIONS
    for goal in TRAIN_REGIONS
    if start != goal and (start, goal) != VALIDATION_REGION_PAIR
)
PREDATOR_SPEED_RATIOS = (0.10, 0.15, 0.20, 0.25)
HEADINGS_DEGREES = tuple(float(value) for value in range(-180, 180, 30))


def spatial_region(location: Sequence[float]) -> str:
    x, y = float(location[0]), float(location[1])
    if x < 0.5:
        return "northwest" if y >= 0.5 else "southwest"
    return "northeast" if y >= 0.5 else "southeast"


def _cell_partition(seed: int, cell_id: int) -> str:
    payload = f"{int(seed)}:{int(cell_id)}".encode("ascii")
    bucket = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 3
    return "evaluation" if bucket == 0 else "training"


def _path_cell_ids(loader, start_cell: int, goal_cell: int) -> list[int]:
    current = int(start_cell)
    path = [current]
    for _ in range(len(loader.locations) + 1):
        if current == int(goal_cell):
            return path
        next_cell = loader.paths[current][int(goal_cell)]
        if next_cell is None or int(next_cell) == current:
            break
        current = int(next_cell)
        path.append(current)
    raise RuntimeError(f"No path from cell {start_cell} to {goal_cell}")


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _task_record(
    *,
    split: str,
    world_name: str,
    start_cell: int,
    goal_cell: int,
    predator_cell: int,
    path_cells: Sequence[int],
    locations: Mapping[int, Sequence[float]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    start = locations[int(start_cell)]
    goal = locations[int(goal_cell)]
    predator = locations[int(predator_cell)]
    path_regions = []
    for cell_id in path_cells:
        region = spatial_region(locations[int(cell_id)])
        if not path_regions or path_regions[-1] != region:
            path_regions.append(region)
    record = {
        "split": str(split),
        "world_name": str(world_name),
        "start_cell_id": int(start_cell),
        "goal_cell_id": int(goal_cell),
        "predator_cell_id": int(predator_cell),
        "start_location": [float(value) for value in start],
        "goal_location": [float(value) for value in goal],
        "predator_location": [float(value) for value in predator],
        "start_region": spatial_region(start),
        "goal_region": spatial_region(goal),
        "path_cell_ids": [int(cell_id) for cell_id in path_cells],
        "path_regions": path_regions,
        "prey_body_heading_degrees": float(rng.choice(HEADINGS_DEGREES)),
        "predator_body_heading_degrees": float(rng.choice(HEADINGS_DEGREES)),
        "predator_speed_ratio": float(rng.choice(PREDATOR_SPEED_RATIOS)),
    }
    record["task_id"] = f"matched-{split}-{task_digest(record)[:16]}"
    return record


def generate_split(
    loader,
    *,
    split: str,
    count: int,
    seed: int,
    cell_pools: Mapping[str, Mapping[str, Sequence[int]]],
    locations: Mapping[int, Sequence[float]],
    heldout_occlusion_cell_ids: set[int],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    records = []
    used_pairs: set[tuple[int, int]] = set()
    attempts = 0
    maximum_attempts = int(count) * 1000
    while len(records) < int(count) and attempts < maximum_attempts:
        attempts += 1
        if split == "train":
            start_region, goal_region = TRAIN_REGION_PAIRS[
                int(rng.integers(0, len(TRAIN_REGION_PAIRS)))
            ]
            pool_name = "training"
        elif split == "validation":
            start_region, goal_region = VALIDATION_REGION_PAIR
            pool_name = "evaluation"
        else:
            other_region = TRAIN_REGIONS[int(rng.integers(0, len(TRAIN_REGIONS)))]
            if bool(rng.integers(0, 2)):
                start_region, goal_region = HELDOUT_TEST_REGION, other_region
            else:
                start_region, goal_region = other_region, HELDOUT_TEST_REGION
            pool_name = "evaluation"
        start_cell = int(rng.choice(cell_pools[start_region][pool_name]))
        goal_cell = int(rng.choice(cell_pools[goal_region][pool_name]))
        pair = (start_cell, goal_cell)
        if pair in used_pairs:
            continue
        try:
            path_cells = _path_cell_ids(loader, start_cell, goal_cell)
        except RuntimeError:
            continue
        if not 8 <= len(path_cells) <= 36:
            continue
        path_regions = {spatial_region(locations[cell_id]) for cell_id in path_cells}
        if split in ("train", "validation") and HELDOUT_TEST_REGION in path_regions:
            continue
        if split == "test" and HELDOUT_TEST_REGION not in path_regions:
            continue
        if split == "test" and not heldout_occlusion_cell_ids.intersection(path_cells):
            continue
        predator_regions = TRAIN_REGIONS if split in ("train", "validation") else REGIONS
        predator_candidates = [
            cell_id
            for region in predator_regions
            for cell_id in cell_pools[region][pool_name]
            if _distance(locations[cell_id], locations[start_cell]) >= 0.20
            and _distance(locations[cell_id], locations[goal_cell]) >= 0.15
        ]
        if not predator_candidates:
            continue
        predator_cell = int(rng.choice(predator_candidates))
        record = _task_record(
            split=split,
            world_name=loader.world_name,
            start_cell=start_cell,
            goal_cell=goal_cell,
            predator_cell=predator_cell,
            path_cells=path_cells,
            locations=locations,
            rng=rng,
        )
        records.append(record)
        used_pairs.add(pair)
    if len(records) != int(count):
        raise RuntimeError(
            f"Generated only {len(records)}/{count} {split} tasks after {attempts} attempts",
        )
    return records


def generate_manifests(
    *,
    output_dir: Path,
    world_name: str,
    seed: int,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    loader = cwgame.CellWorldLoader(world_name=world_name)
    project_root = Path(__file__).resolve().parents[1]
    heldout_occlusion_cell_ids = {
        int(cell_id)
        for cell_id in np.load(
            project_root / "data" / "cell_ids_near_occlusion_21_05.npy",
        )
        if loader.locations[int(cell_id)] is not None
        and spatial_region(loader.locations[int(cell_id)]) == HELDOUT_TEST_REGION
    }
    locations = {
        cell_id: tuple(location)
        for cell_id, location in enumerate(loader.locations)
        if location is not None
    }
    cell_pools = {
        region: {
            partition: [
                cell_id
                for cell_id, location in locations.items()
                if spatial_region(location) == region
                and _cell_partition(seed, cell_id) == partition
            ]
            for partition in ("training", "evaluation")
        }
        for region in REGIONS
    }
    manifests = {
        split: generate_split(
            loader,
            split=split,
            count=int(counts[split]),
            seed=int(seed) + index * 1_000_003,
            cell_pools=cell_pools,
            locations=locations,
            heldout_occlusion_cell_ids=heldout_occlusion_cell_ids,
        )
        for index, split in enumerate(("train", "validation", "test"))
    }
    contract = validate_split_contract(
        manifests,
        heldout_region=HELDOUT_TEST_REGION,
        validation_region_pair=VALIDATION_REGION_PAIR,
        heldout_occlusion_cell_ids=heldout_occlusion_cell_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in manifests.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    metadata = {
        "version": 1,
        "world_name": world_name,
        "seed": int(seed),
        "regions": list(REGIONS),
        "training_regions": list(TRAIN_REGIONS),
        "heldout_test_region": HELDOUT_TEST_REGION,
        "heldout_occlusion_cell_count": len(heldout_occlusion_cell_ids),
        "train_region_pairs": [list(pair) for pair in TRAIN_REGION_PAIRS],
        "validation_region_pair": list(VALIDATION_REGION_PAIR),
        "predator_speed_ratios": list(PREDATOR_SPEED_RATIOS),
        "counts": {split: len(records) for split, records in manifests.items()},
        "cell_pool_counts": {
            region: {name: len(values) for name, values in pools.items()}
            for region, pools in cell_pools.items()
        },
        "contract": contract,
    }
    write_json(output_dir / "manifest.json", metadata)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-name", default="21_05")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_manifests(
        output_dir=args.output_dir.resolve(),
        world_name=str(args.world_name),
        seed=int(args.seed),
        counts={
            "train": int(args.train_count),
            "validation": int(args.validation_count),
            "test": int(args.test_count),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
