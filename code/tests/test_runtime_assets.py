import json

import pytest

from paretoskill.runtime_assets import (
    RuntimeAssetError,
    load_base_skill,
    load_patch_pool,
    load_trace_evidence,
)


def _evidence_row(evidence_id: str = "e-1") -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "task_id": "task-1",
        "seed": 17,
        "target_id": "target-1",
        "outcome": False,
        "verifier_summary": "fixture",
        "tags": ["id_accuracy"],
        "metadata": {},
    }


def test_load_runtime_assets_and_bind_base_placeholder(tmp_path):
    skill_root = tmp_path / "skill"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text("# Local skill\n", encoding="utf-8")
    base = load_base_skill(skill_root)

    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(json.dumps(_evidence_row()) + "\n", encoding="utf-8")
    evidence = load_trace_evidence(evidence_path)

    patch_path = tmp_path / "patches.json"
    patch_path.write_text(
        json.dumps(
            {
                "patches": [
                    {
                        "patch_id": "p-1",
                        "operation": "add",
                        "target_path": "SKILL.md",
                        "parent_version_id": "$BASE",
                        "evidence_ids": ["e-1"],
                        "content": "Use a verified local procedure.",
                        "sequence": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    patches = load_patch_pool(
        patch_path,
        base_version_id=base.lineage.version_id,
        evidence=evidence,
    )

    assert patches[0].parent_version_id == base.lineage.version_id
    assert patches[0].evidence_ids == ("e-1",)


def test_runtime_assets_reject_unknown_evidence_and_symlinks(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps([_evidence_row()]), encoding="utf-8")
    evidence = load_trace_evidence(evidence_path)
    patch_path = tmp_path / "patches.jsonl"
    patch_path.write_text(
        json.dumps(
            {
                "patch_id": "p-1",
                "operation": "add",
                "target_path": "SKILL.md",
                "parent_version_id": "$BASE",
                "evidence_ids": ["missing"],
                "content": "text",
                "sequence": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeAssetError, match="unknown evidence"):
        load_patch_pool(patch_path, base_version_id="base-1", evidence=evidence)


def test_runtime_assets_reject_duplicate_ids(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps([_evidence_row(), _evidence_row()]), encoding="utf-8"
    )
    with pytest.raises(RuntimeAssetError, match="duplicate evidence_id"):
        load_trace_evidence(evidence_path)
