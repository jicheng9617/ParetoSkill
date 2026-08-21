from __future__ import annotations

import json
from pathlib import Path

from paretoskill.cli import main
from paretoskill.config import load_manifest
from paretoskill.runner import run_synthetic_dry_run


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


def test_synthetic_dry_run_writes_required_artifacts_and_resumes(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    first = run_synthetic_dry_run(manifest, output_root=tmp_path)
    assert first.external_calls == 0
    assert first.unique_task_executions > 0
    assert first.archive_size >= 1

    expected = {
        "resolved_manifest.yaml",
        "run_metadata.json",
        "task_outcomes.jsonl",
        "candidates.jsonl",
        "archive.json",
        "lineage.jsonl",
        "metrics.json",
        "token_accounting.json",
        "checkpoint.json",
    }
    assert expected <= {path.name for path in first.output_directory.iterdir()}
    line_count = len(
        (first.output_directory / "task_outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    )

    second = run_synthetic_dry_run(manifest, output_root=tmp_path)
    assert second.unique_task_executions == 0
    assert second.reused_task_executions == first.unique_task_executions
    assert len(
        (first.output_directory / "task_outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ) == line_count
    metadata = json.loads((first.output_directory / "run_metadata.json").read_text())
    assert metadata["external_calls"] == 0


def test_cli_validates_and_runs_offline(tmp_path, capsys) -> None:
    assert main(["validate-config", str(CONFIG)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["valid"] is True
    assert validation["offline"] is True

    assert main(["dry-run", str(CONFIG), "--output-root", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["synthetic_only"] is True
    assert result["external_calls"] == 0


def test_resume_reconciles_checkpoint_and_cache(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    first = run_synthetic_dry_run(manifest, output_root=tmp_path)
    checkpoint_path = first.output_directory / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    missing_key = checkpoint["completed_cache_keys"][0]
    cache_file = first.output_directory / "cache" / missing_key[:2] / f"{missing_key}.json"
    cache_file.unlink()

    resumed = run_synthetic_dry_run(manifest, output_root=tmp_path)
    assert resumed.unique_task_executions == 1
    assert resumed.reused_task_executions == first.unique_task_executions - 1


def test_resume_rebuilds_a_partial_task_ledger_from_cache(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    first = run_synthetic_dry_run(manifest, output_root=tmp_path)
    ledger = first.output_directory / "task_outcomes.jsonl"
    ledger.write_text('{"partial":', encoding="utf-8")

    resumed = run_synthetic_dry_run(manifest, output_root=tmp_path)

    assert resumed.unique_task_executions == 0
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == first.unique_task_executions
