import json
from pathlib import Path
import subprocess

import pytest
import yaml

from autoresearch.confirmation import (
    ConfirmationDriverError,
    _authorization,
    _confirmation_gate,
    _confirmation_partition,
    _driver_sha256,
    build_parser,
)
from autoresearch.evaluator import EVALUATION_METHODS, canonical_sha256


def _record(seed, method, *, success=False, capture=False):
    return {
        "method": method,
        "seed": int(seed),
        "clean_success": bool(success),
        "capture_episode": bool(capture),
        "goal_reached": bool(success),
        "steps": 2,
        "minimum_predator_distance": 0.25,
        "path_cost": 1.0,
        "gaze_travel_degrees": 0.0,
        "predator_pixels_visible_fraction": 0.5,
        "action_trace_sha256": canonical_sha256([method, int(seed)]),
    }


def _records(seeds, *, candidate_success, incumbent_success, candidate_capture=()):
    rows = []
    for seed in seeds:
        rows.extend(
            (
                _record(
                    seed,
                    "candidate",
                    success=seed in candidate_success,
                    capture=seed in candidate_capture,
                ),
                _record(seed, "incumbent", success=seed in incumbent_success),
                _record(seed, "fixed_p60", success=False),
            ),
        )
    return rows


def test_confirmation_partition_is_exact_registered_one_time_set():
    config = yaml.safe_load(Path("configs/autoresearch/gaze_dev.yaml").read_text())
    shards = _confirmation_partition(config, shard_count=50)
    assert len(shards) == 50
    assert all(len(shard.seeds) == 20 for shard in shards)
    flattened = tuple(seed for shard in shards for seed in shard.seeds)
    assert flattened == tuple(range(1_200_000, 1_201_000))
    assert len(set(flattened)) == 1000
    with pytest.raises(ConfirmationDriverError):
        _confirmation_partition(config, shard_count=1001)


def test_confirmation_gate_requires_favorable_interval_and_capture_nonworsening():
    seeds = tuple(range(100))
    records = _records(
        seeds,
        candidate_success=set(seeds),
        incumbent_success=set(),
    )
    gate, statistics = _confirmation_gate(
        records,
        seeds=seeds,
        checks={"all_contracts": True},
        bootstrap_samples=200,
    )
    assert gate["confirmation_passed"] is True
    assert statistics["bootstrap_95_low"] > 0.0

    worse_capture = _records(
        seeds,
        candidate_success=set(seeds),
        incumbent_success=set(),
        candidate_capture={0},
    )
    gate, _ = _confirmation_gate(
        worse_capture,
        seeds=seeds,
        checks={"all_contracts": True},
        bootstrap_samples=200,
    )
    assert gate["confirmation_passed"] is False
    assert gate["capture_nonworsening"] is False


def test_authorization_marker_must_be_explicit_spent_and_driver_bound(tmp_path):
    marker = {
        "schema_version": 1,
        "artifact_type": "autoresearch_confirmation_authorization",
        "experiment_id": "C0001",
        "confirmation_set_spent": True,
        "explicit_user_authorization": True,
        "driver_sha256": _driver_sha256(),
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    loaded = _authorization(path, expected_sha256=digest)
    assert loaded["authorization_sha256"] == digest

    marker["confirmation_set_spent"] = False
    path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(ConfirmationDriverError, match="valid and spent"):
        _authorization(path)


def test_confirmation_cli_exposes_no_seed_or_horizon_override():
    parser = build_parser()
    rollout = parser._subparsers._group_actions[0].choices["rollout"]
    options = {
        option
        for action in rollout._actions
        for option in action.option_strings
    }
    assert "--seed-start" not in options
    assert "--episodes" not in options
    assert "--max-horizon" not in options
    prepare = parser.parse_args(["prepare", "--run-tag", "run"])
    assert prepare.authorize_confirmation is False
    assert set(EVALUATION_METHODS) == {"candidate", "incumbent", "fixed_p60"}


def test_confirmation_slurm_scripts_are_authorized_cpu_only_and_key_free():
    root = Path(__file__).resolve().parents[1]
    shard = root / "setup/autoresearch_confirmation_shards.sbatch"
    aggregate = root / "setup/autoresearch_confirmation_aggregate.sbatch"
    remote = root / "setup/remote_submit_autoresearch_confirmation.sh"
    subprocess.run(
        ["bash", "-n", str(shard), str(aggregate), str(remote)],
        check=True,
    )
    shard_text = shard.read_text()
    aggregate_text = aggregate.read_text()
    remote_text = remote.read_text()
    assert "#SBATCH --mem=12G" in shard_text
    assert "#SBATCH --mem=8G" in aggregate_text
    for text in (shard_text, aggregate_text):
        assert "unset OPENROUTER_API_KEY" in text
        assert "--authorization-sha256" in text
        assert "--seed-start" not in text
        assert "--episodes" not in text
    assert "3 * shard_count - 1" in remote_text
    assert "afterok:${array_job_id}" in remote_text
    assert "authorization.json" in remote_text
