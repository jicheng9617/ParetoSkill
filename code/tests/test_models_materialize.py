from __future__ import annotations

import json

import pytest

from paretoskill.materialize import MaterializationError, MaterializationStore, Materializer
from paretoskill.models import (
    Patch,
    PatchOperation,
    Skill,
    TraceEvidence,
    make_base_version,
)


def evidence() -> dict[str, TraceEvidence]:
    item = TraceEvidence(
        evidence_id="trace-1",
        task_id="task-1",
        seed=7,
        target_id="mock-target",
        outcome=False,
        verifier_summary="synthetic failure fixture",
    )
    return {item.evidence_id: item}


def test_all_patch_operations_are_deterministic_and_preserve_lineage() -> None:
    base = make_base_version(Skill(name="demo", files={"SKILL.md": "alpha beta gamma"}))
    patches = [
        Patch(
            "add",
            PatchOperation.ADD,
            "SKILL.md",
            base.lineage.version_id,
            ("trace-1",),
            content="delta",
            sequence=3,
        ),
        Patch(
            "drop",
            PatchOperation.DROP,
            "SKILL.md",
            base.lineage.version_id,
            ("trace-1",),
            match_text="alpha ",
            sequence=0,
        ),
        Patch(
            "rewrite",
            PatchOperation.REWRITE,
            "SKILL.md",
            base.lineage.version_id,
            ("trace-1",),
            match_text="beta",
            content="B",
            sequence=1,
        ),
        Patch(
            "compress",
            PatchOperation.COMPRESS,
            "SKILL.md",
            base.lineage.version_id,
            ("trace-1",),
            match_text="gamma",
            content="g",
            sequence=2,
        ),
    ]
    materializer = Materializer()
    left = materializer.materialize(base, patches, evidence=evidence())
    right = materializer.materialize(base, reversed(patches), evidence=evidence())

    assert left == right
    assert left.skill.files["SKILL.md"] == "B g\n\ndelta"
    assert left.lineage.patch_ids == ("drop", "rewrite", "compress", "add")
    assert left.lineage.evidence_ids == ("trace-1",)


def test_materializer_rejects_ambiguous_and_ungrounded_edits() -> None:
    base = make_base_version(Skill(name="demo", files={"SKILL.md": "same same"}))
    ambiguous = Patch(
        "rewrite",
        PatchOperation.REWRITE,
        "SKILL.md",
        base.lineage.version_id,
        ("trace-1",),
        match_text="same",
        content="new",
    )
    with pytest.raises(MaterializationError, match="ambiguous"):
        Materializer().materialize(base, [ambiguous], evidence=evidence())

    unknown = Patch(
        "add",
        PatchOperation.ADD,
        "SKILL.md",
        base.lineage.version_id,
        ("missing",),
        content="new",
    )
    with pytest.raises(MaterializationError, match="unknown evidence"):
        Materializer().materialize(base, [unknown], evidence=evidence())


def test_content_deduplication_keeps_distinct_lineages(tmp_path) -> None:
    base = make_base_version(Skill(name="demo", files={"SKILL.md": "base"}))
    first_patch = Patch(
        "path-a",
        PatchOperation.ADD,
        "SKILL.md",
        base.lineage.version_id,
        ("trace-1",),
        content="shared",
    )
    second_patch = Patch(
        "path-b",
        PatchOperation.ADD,
        "SKILL.md",
        base.lineage.version_id,
        ("trace-1",),
        content="shared",
    )
    materializer = Materializer()
    first = materializer.materialize(base, [first_patch], evidence=evidence())
    second = materializer.materialize(base, [second_patch], evidence=evidence())
    assert first.skill.content_hash == second.skill.content_hash
    assert first.lineage.version_id != second.lineage.version_id

    store = MaterializationStore()
    store.add(base)
    store.add(first)
    store.add(second)
    assert len(store.artifacts) == 2
    assert len(store.lineages) == 3

    path = tmp_path / "materializations.json"
    store.save(path)
    restored = MaterializationStore.load(path)
    assert restored.to_dict() == store.to_dict()
    json.dumps(restored.to_dict())


def test_skill_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        Skill(name="bad", files={"SKILL.md": "ok", "../secret": "no"})


def test_add_dedup_uses_full_block_boundaries() -> None:
    base = make_base_version(Skill(name="demo", files={"SKILL.md": "foobar"}))
    patch = Patch(
        "substring",
        PatchOperation.ADD,
        "SKILL.md",
        base.lineage.version_id,
        ("trace-1",),
        content="bar",
    )
    result = Materializer().materialize(base, [patch], evidence=evidence())
    assert result.skill.files["SKILL.md"] == "foobar\n\nbar"


def test_materialization_restore_rejects_lineage_alias(tmp_path) -> None:
    base = make_base_version(Skill(name="demo", files={"SKILL.md": "base"}))
    store = MaterializationStore()
    store.add(base)
    payload = store.to_dict()
    payload["lineages"] = {"alias": base.lineage.to_dict()}  # type: ignore[index]
    with pytest.raises(MaterializationError, match="map key"):
        MaterializationStore.from_dict(payload)
