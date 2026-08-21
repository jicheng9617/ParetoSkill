from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from paretoskill.config import load_manifest
from paretoskill.task_manifests import TaskManifestError, validate_declared_splits


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


def write_ids(path: Path, prefix: str, count: int) -> None:
    path.write_text(json.dumps([f"{prefix}-{index}" for index in range(count)]), encoding="utf-8")


def test_primary_split_manifests_validate_counts_and_disjointness(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    data = copy.deepcopy(dict(manifest.data))
    selected = ("evolution_trace", "id_validation", "heldout_verified")
    for split_id in selected:
        count = int(data["splits"][split_id]["expected_count"])
        path = tmp_path / f"{split_id}.json"
        write_ids(path, split_id, count)
        data["splits"][split_id]["manifest"] = str(path)
        data["splits"][split_id]["manifest_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    loaded = validate_declared_splits(data, split_ids=selected)
    assert {key: len(value) for key, value in loaded.items()} == {
        "evolution_trace": 160,
        "id_validation": 40,
        "heldout_verified": 200,
    }


def test_overlap_is_rejected(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    data = copy.deepcopy(dict(manifest.data))
    first = tmp_path / "evolution.json"
    second = tmp_path / "validation.json"
    write_ids(first, "task", 160)
    ids = ["task-0", *[f"validation-{index}" for index in range(39)]]
    second.write_text(json.dumps(ids), encoding="utf-8")
    data["splits"]["evolution_trace"]["manifest"] = str(first)
    data["splits"]["id_validation"]["manifest"] = str(second)
    data["splits"]["evolution_trace"]["manifest_sha256"] = hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    data["splits"]["id_validation"]["manifest_sha256"] = hashlib.sha256(
        second.read_bytes()
    ).hexdigest()
    with pytest.raises(TaskManifestError, match="overlap"):
        validate_declared_splits(data, split_ids=("evolution_trace", "id_validation"))


def test_manifest_content_digest_mismatch_is_rejected(tmp_path) -> None:
    manifest = load_manifest(CONFIG, environment={})
    data = copy.deepcopy(dict(manifest.data))
    path = tmp_path / "ids.json"
    write_ids(path, "task", 160)
    data["splits"]["evolution_trace"]["manifest"] = str(path)
    data["splits"]["evolution_trace"]["manifest_sha256"] = "0" * 64

    with pytest.raises(TaskManifestError, match="digest mismatch"):
        validate_declared_splits(data, split_ids=("evolution_trace",))
