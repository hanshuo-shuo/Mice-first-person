from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import autoresearch.ledger as ledger_module
from autoresearch.ledger import (
    ArtifactError,
    DuplicateExperimentError,
    ExperimentLedger,
    InvalidRecordError,
)


UTC = timezone.utc
OLD = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)


def _metadata(experiment_id: str, *, parent: str | None = None) -> dict:
    return {
        "experiment_id": experiment_id,
        "parent_incumbent_id": parent,
        "candidate_commit": f"commit-{experiment_id}",
        "candidate_sha256": (experiment_id.encode().hex() + "0" * 64)[:64],
        "hypothesis": f"candidate {experiment_id} improves clean success",
        "predicted_effect": "+2 paired clean-success episodes",
        "changed_paths": ["autoresearch/candidate.py"],
        "source_model_sha256": "1" * 64,
        "resolved_config_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "seed_set_id": "dev-v1",
    }


def _start(
    ledger: ExperimentLedger,
    experiment_id: str,
    *,
    parent: str | None = None,
    when: datetime | None = None,
) -> None:
    ledger.plan_experiment(_metadata(experiment_id, parent=parent), planned_at=when)
    ledger.start_experiment(experiment_id, started_at=when)


def _terminal_fields(reason: str, *, delta: int | None = 2) -> dict:
    return {
        "primary_delta": delta,
        "paired_counts": {
            "candidate_clean_success": 12 if delta is not None else 0,
            "incumbent_clean_success": 10 if delta is not None else 0,
            "candidate_only_success": 3 if delta is not None else 0,
            "incumbent_only_success": 1 if delta is not None else 0,
        },
        "secondary_metrics": {"candidate_capture_rate": 0.1},
        "checks": {"contract": "pass", "leak": "pass"},
        "decision_reason": reason,
    }


def _write_artifact(
    ledger: ExperimentLedger, experiment_id: str, text: str = "evidence\n"
) -> Path:
    staging = ledger.begin_artifacts(experiment_id)
    nested = staging / "logs"
    nested.mkdir()
    output = nested / "stdout.txt"
    output.write_text(text, encoding="utf-8")
    return staging


def _read_incumbent(ledger: ExperimentLedger) -> dict:
    return json.loads(ledger.incumbent_path.read_text(encoding="utf-8"))


def test_pass_path_uses_keep_and_is_idempotent(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    metadata = _metadata("E0001")

    planned = ledger.plan_experiment(metadata, planned_at=OLD)
    assert ledger.plan_experiment(metadata, planned_at=OLD) == planned
    running = ledger.start_experiment("E0001", started_at=OLD + timedelta(seconds=1))
    assert (
        ledger.start_experiment("E0001", started_at=OLD + timedelta(seconds=1))
        == running
    )

    staging = _write_artifact(ledger, "E0001")
    fields = _terminal_fields("all gates passed")
    kept = ledger.finalize_experiment(
        "E0001",
        status="keep",
        fields=fields,
        artifact_staging=staging,
        completed_at=OLD + timedelta(seconds=2),
    )

    assert kept["status"] == "keep"
    assert kept["checks"] == {"contract": "pass", "leak": "pass"}
    assert not staging.exists()
    assert (ledger.artifact_path("E0001") / "logs" / "stdout.txt").is_file()
    artifact = kept["artifacts"]["files"][0]
    assert artifact["path"] == "artifacts/E0001/logs/stdout.txt"
    assert len(artifact["sha256"]) == 64
    assert _read_incumbent(ledger)["incumbent_id"] == "E0001"

    # A retry after the ledger append repairs derived files without appending.
    ledger.results_path.unlink()
    retried = ledger.finalize_experiment(
        "E0001",
        status="keep",
        fields=fields,
        completed_at=OLD + timedelta(seconds=2),
    )
    assert retried == kept
    assert len(ledger.read_records()) == 3
    assert ledger.results_path.is_file()
    with ledger.results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [(row["experiment_id"], row["status"]) for row in rows] == [
        ("E0001", "keep")
    ]


def test_duplicate_and_conflicting_terminal_evidence_is_rejected(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    planned = ledger.plan_experiment(_metadata("E0001"))
    assert ledger.plan_experiment(_metadata("E0001"))["record_sha256"] == planned[
        "record_sha256"
    ]

    conflicting = _metadata("E0001")
    conflicting["hypothesis"] = "a different hypothesis"
    with pytest.raises(DuplicateExperimentError, match="conflicting plan"):
        ledger.plan_experiment(conflicting)

    ledger.start_experiment("E0001")
    _write_artifact(ledger, "E0001")
    fields = _terminal_fields("no improvement", delta=0)
    discarded = ledger.finalize_experiment(
        "E0001", status="discard", fields=fields
    )
    assert discarded["status"] == "discard"
    with pytest.raises(DuplicateExperimentError, match="terminal evidence"):
        ledger.finalize_experiment("E0001", status="crash", fields=fields)
    with pytest.raises(InvalidRecordError, match="canonical terminal status"):
        ledger.finalize_experiment("E0001", status="pass", fields=fields)


def test_discard_and_crash_never_promote_over_last_keep(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001", when=OLD)
    _write_artifact(ledger, "E0001", "kept")
    ledger.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("passed")
    )

    _start(ledger, "E0002", parent="E0001")
    _write_artifact(ledger, "E0002", "discarded")
    ledger.finalize_experiment(
        "E0002",
        status="discard",
        fields=_terminal_fields("capture gate failed", delta=-1),
    )

    _start(ledger, "E0003", parent="E0001")
    _write_artifact(ledger, "E0003", "log before crash")
    ledger.finalize_experiment(
        "E0003",
        status="crash",
        fields=_terminal_fields("worker exited", delta=None),
    )

    assert _read_incumbent(ledger)["incumbent_id"] == "E0001"
    assert (
        ledger.artifact_path("E0003") / "logs" / "stdout.txt"
    ).read_text(encoding="utf-8") == "log before crash"
    assert {
        experiment_id: record["status"]
        for experiment_id, record in ledger.latest_records().items()
    } == {"E0001": "keep", "E0002": "discard", "E0003": "crash"}


def test_interruption_after_artifact_rename_can_finalize_without_duplication(
    tmp_path, monkeypatch
):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    staging = _write_artifact(ledger, "E0001", "durable before ledger")
    original_append = ledger._append_locked

    def interrupted_append(fd, record, current_records, *, recorded_at=None):
        if record["status"] == "keep":
            raise OSError("simulated power loss after rename")
        return original_append(
            fd, record, current_records, recorded_at=recorded_at
        )

    monkeypatch.setattr(ledger, "_append_locked", interrupted_append)
    with pytest.raises(OSError, match="power loss"):
        ledger.finalize_experiment(
            "E0001", status="keep", fields=_terminal_fields("passed")
        )

    assert not staging.exists()
    assert (ledger.artifact_path("E0001") / "logs" / "stdout.txt").is_file()
    assert ledger.latest_records()["E0001"]["status"] == "running"
    assert not ledger.incumbent_path.exists()

    recovered = ExperimentLedger(ledger.run_dir)
    final = recovered.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("passed")
    )
    assert final["status"] == "keep"
    assert len(recovered.read_records()) == 3
    assert _read_incumbent(recovered)["incumbent_id"] == "E0001"


def test_interrupted_jsonl_tail_is_quarantined_before_later_append(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    _write_artifact(ledger, "E0001")

    with ledger.ledger_path.open("ab", buffering=0) as stream:
        stream.write(b'{"experiment_id":"E0001"')
        os.fsync(stream.fileno())
    before = ledger.inspect()
    assert before.records[-1]["status"] == "running"
    assert before.issues[-1].reason == "interrupted JSONL tail"

    final = ledger.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("passed")
    )
    after = ledger.inspect()
    assert after.records[-1] == final
    assert after.issues == ()
    assert _read_incumbent(ledger)["incumbent_id"] == "E0001"


def test_complete_json_without_newline_is_discarded_not_revived(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    _write_artifact(ledger, "E0001")
    running = ledger.latest_records()["E0001"]

    interrupted = {
        **running,
        "status": "discard",
        "completed_at": "2026-08-30T00:00:01.000000Z",
        "ledger_sequence": int(running["ledger_sequence"]) + 1,
        "recorded_at": "2026-08-30T00:00:01.000000Z",
    }
    interrupted["record_sha256"] = ledger_module._record_digest(interrupted)
    with ledger.ledger_path.open("ab", buffering=0) as stream:
        stream.write(ledger_module._canonical_json_bytes(interrupted))
        os.fsync(stream.fileno())

    assert ledger.inspect().issues[-1].reason == "interrupted JSONL tail"
    final = ledger.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("durable keep")
    )
    inspection = ledger.inspect()
    assert inspection.issues == ()
    assert ledger.latest_records()["E0001"] == final
    assert final["status"] == "keep"
    assert [record["ledger_sequence"] for record in inspection.records] == [1, 2, 3]


def test_incumbent_write_interruption_is_repaired_from_last_valid_keep(
    tmp_path, monkeypatch
):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    _write_artifact(ledger, "E0001")
    ledger.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("first pass")
    )

    _start(ledger, "E0002", parent="E0001")
    _write_artifact(ledger, "E0002")
    real_replace = ledger_module.os.replace

    def fail_incumbent_only(source, destination):
        if Path(destination) == ledger.incumbent_path:
            raise OSError("simulated incumbent replace interruption")
        return real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "replace", fail_incumbent_only)
        with pytest.raises(OSError, match="incumbent replace"):
            ledger.finalize_experiment(
                "E0002", status="keep", fields=_terminal_fields("second pass")
            )

    # The keep is already durable, while atomic replacement kept E0001 intact.
    assert ledger.latest_records()["E0002"]["status"] == "keep"
    assert _read_incumbent(ledger)["incumbent_id"] == "E0001"

    repaired = ledger.finalize_experiment(
        "E0002", status="keep", fields=_terminal_fields("second pass")
    )
    assert repaired["status"] == "keep"
    assert _read_incumbent(ledger)["incumbent_id"] == "E0002"
    assert len(
        [record for record in ledger.read_records() if record["experiment_id"] == "E0002"]
    ) == 3


def test_recovery_falls_back_when_latest_keep_artifact_is_invalid(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    for experiment_id, parent in (("E0001", None), ("E0002", "E0001")):
        _start(ledger, experiment_id, parent=parent)
        _write_artifact(ledger, experiment_id, experiment_id)
        ledger.finalize_experiment(
            experiment_id,
            status="keep",
            fields=_terminal_fields(f"kept {experiment_id}"),
        )
    assert _read_incumbent(ledger)["incumbent_id"] == "E0002"

    (ledger.artifact_path("E0002") / "logs" / "stdout.txt").write_text(
        "tampered", encoding="utf-8"
    )
    ledger.incumbent_path.write_text("{interrupted", encoding="utf-8")
    recovered = ledger.recover_incumbent()

    assert recovered is not None
    assert recovered["incumbent_id"] == "E0001"
    assert _read_incumbent(ledger)["incumbent_id"] == "E0001"


def test_recovery_removes_stale_incumbent_when_no_keep_artifact_is_valid(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    _write_artifact(ledger, "E0001")
    ledger.finalize_experiment(
        "E0001", status="keep", fields=_terminal_fields("passed")
    )
    assert ledger.incumbent_path.is_file()

    (ledger.artifact_path("E0001") / "logs" / "stdout.txt").write_text(
        "tampered", encoding="utf-8"
    )
    assert ledger.recover_incumbent() is None
    assert not ledger.incumbent_path.exists()


def test_stale_running_recovery_crashes_or_explicitly_resumes_idempotent_work(
    tmp_path,
):
    ledger = ExperimentLedger(tmp_path / "run")
    now = OLD + timedelta(hours=1)

    _start(ledger, "Ecrash", when=OLD)
    _write_artifact(ledger, "Ecrash", "preserve this crash log")
    _start(ledger, "Eresume", when=OLD)
    _write_artifact(ledger, "Eresume", "partial idempotent output")
    _start(ledger, "Eactive", when=now - timedelta(seconds=30))

    report = ledger.recover_stale_running(
        60, idempotent_resume_ids={"Eresume"}, now=now
    )

    assert report.crashed == ("Ecrash",)
    assert report.resumed == ("Eresume",)
    assert report.active == ("Eactive",)
    latest = ledger.latest_records()
    assert latest["Ecrash"]["status"] == "crash"
    assert latest["Ecrash"]["recovery_action"] == "mark_crash"
    assert latest["Eresume"]["status"] == "running"
    assert latest["Eresume"]["idempotent_resume"] is True
    assert latest["Eresume"]["resume_count"] == 1
    assert ledger.artifact_staging_path("Eresume").is_dir()
    assert (
        ledger.artifact_path("Ecrash") / "logs" / "stdout.txt"
    ).read_text(encoding="utf-8") == "preserve this crash log"

    record_count = len(ledger.read_records())
    repeated = ledger.recover_on_startup(
        60, idempotent_resume_ids={"Eresume"}, now=now
    )
    assert repeated.crashed == ()
    assert repeated.resumed == ()
    assert set(repeated.active) == {"Eresume", "Eactive"}
    assert len(ledger.read_records()) == record_count

    resumed_staging = ledger.begin_artifacts("Eresume", resume=True)
    (resumed_staging / "completed.txt").write_text("done", encoding="utf-8")
    ledger.finalize_experiment(
        "Eresume",
        status="discard",
        fields=_terminal_fields("idempotent rerun completed", delta=0),
    )
    assert ledger.latest_records()["Eresume"]["status"] == "discard"


def test_resume_requires_explicit_idempotence_and_artifact_paths_are_confined(
    tmp_path,
):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    with pytest.raises(InvalidRecordError, match="explicitly idempotent"):
        ledger.resume_experiment("E0001", evaluation_is_idempotent=False)
    with pytest.raises(InvalidRecordError, match="experiment_id"):
        ledger.plan_experiment("../escape")

    staging = ledger.begin_artifacts("E0001")
    with pytest.raises(ArtifactError, match="use resume=True"):
        ledger.begin_artifacts("E0001")
    assert ledger.begin_artifacts("E0001", resume=True) == staging


def test_results_tsv_is_fully_derived_from_jsonl(tmp_path):
    ledger = ExperimentLedger(tmp_path / "run")
    _start(ledger, "E0001")
    ledger.results_path.write_text("not source of truth\n", encoding="utf-8")

    rebuilt = ledger.regenerate_results_tsv()

    assert rebuilt == ledger.results_path
    with rebuilt.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "E0001"
    assert rows[0]["status"] == "running"
