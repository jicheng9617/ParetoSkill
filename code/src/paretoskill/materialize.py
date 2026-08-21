"""Deterministic patch composition, content-addressed reuse, and lineage persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .models import (
    Patch,
    PatchOperation,
    Skill,
    SkillVersion,
    TraceEvidence,
    VersionLineage,
    canonical_json,
    stable_hash,
)

MATERIALIZER_VERSION = "paretoskill-materializer/v1"


class MaterializationError(ValueError):
    """Raised before evaluation when an edit cannot be applied unambiguously."""


def _replace_exact(text: str, before: str, after: str, *, allow_multiple: bool) -> str:
    count = text.count(before)
    if count == 0:
        raise MaterializationError("patch match_text was not found")
    if count > 1 and not allow_multiple:
        raise MaterializationError(f"patch match_text is ambiguous ({count} occurrences)")
    return text.replace(before, after) if allow_multiple else text.replace(before, after, 1)


def _contains_exact_block(text: str, block: str) -> bool:
    return (
        text == block
        or text.startswith(block + "\n\n")
        or text.endswith("\n\n" + block)
        or f"\n\n{block}\n\n" in text
    )


@dataclass(slots=True)
class Materializer:
    version: str = MATERIALIZER_VERSION

    def materialize(
        self,
        parent: SkillVersion,
        patches: Iterable[Patch],
        *,
        evidence: Mapping[str, TraceEvidence] | None = None,
        label: str = "skill",
    ) -> SkillVersion:
        ordered = sorted(patches, key=lambda patch: (patch.sequence, patch.patch_id))
        if len({patch.patch_id for patch in ordered}) != len(ordered):
            raise MaterializationError("patch ids must be unique within one materialization")
        if ordered and evidence is None:
            raise MaterializationError(
                "materialization requires the TraceEvidence mapping cited by every patch"
            )
        if evidence is not None:
            mismatched = sorted(
                key for key, item in evidence.items() if key != item.evidence_id
            )
            if mismatched:
                raise MaterializationError(
                    "evidence mapping keys must equal TraceEvidence.evidence_id: "
                    f"{mismatched}"
                )
        for patch in ordered:
            if patch.parent_version_id != parent.lineage.version_id:
                raise MaterializationError(
                    f"patch {patch.patch_id!r} targets {patch.parent_version_id!r}, "
                    f"not parent {parent.lineage.version_id!r}"
                )
            if evidence is not None:
                missing = set(patch.evidence_ids) - set(evidence)
                if missing:
                    raise MaterializationError(
                        f"patch {patch.patch_id!r} cites unknown evidence: {sorted(missing)}"
                    )

        files = dict(parent.skill.files)
        for patch in ordered:
            self._apply(files, patch)

        skill = Skill(
            name=parent.skill.name,
            files=files,
            metadata={
                **dict(parent.skill.metadata),
                "materializer_version": self.version,
            },
        )
        patch_ids = tuple(patch.patch_id for patch in ordered)
        patch_fingerprints = tuple(patch.fingerprint for patch in ordered)
        evidence_ids = tuple(
            sorted({evidence_id for patch in ordered for evidence_id in patch.evidence_ids})
        )
        version_payload = {
            "parent_version_id": parent.lineage.version_id,
            "content_hash": skill.content_hash,
            "patch_fingerprints": patch_fingerprints,
            "materializer_version": self.version,
        }
        version_id = f"{label}-{stable_hash(version_payload)[:16]}"
        lineage = VersionLineage(
            version_id=version_id,
            parent_version_id=parent.lineage.version_id,
            content_hash=skill.content_hash,
            patch_ids=patch_ids,
            patch_fingerprints=patch_fingerprints,
            evidence_ids=evidence_ids,
            materializer_version=self.version,
        )
        return SkillVersion(skill=skill, lineage=lineage)

    @staticmethod
    def _apply(files: dict[str, str], patch: Patch) -> None:
        current = files.get(patch.target_path)
        allow_multiple = patch.metadata.get("allow_multiple", False)
        if not isinstance(allow_multiple, bool):
            raise MaterializationError("patch metadata allow_multiple must be boolean")

        if patch.operation is PatchOperation.ADD:
            if current is None:
                files[patch.target_path] = patch.content
            elif _contains_exact_block(current, patch.content):
                # Exact duplicate guidance is a deterministic no-op.
                return
            else:
                separator = "" if current.endswith("\n\n") else "\n\n"
                files[patch.target_path] = f"{current}{separator}{patch.content}"
            return

        if current is None:
            raise MaterializationError(
                f"{patch.operation.value} target does not exist: {patch.target_path}"
            )

        if patch.operation is PatchOperation.DROP:
            assert patch.match_text is not None
            files[patch.target_path] = _replace_exact(
                current, patch.match_text, "", allow_multiple=allow_multiple
            )
        elif patch.operation is PatchOperation.REWRITE:
            assert patch.match_text is not None
            files[patch.target_path] = _replace_exact(
                current, patch.match_text, patch.content, allow_multiple=allow_multiple
            )
        elif patch.operation is PatchOperation.COMPRESS:
            before = patch.match_text if patch.match_text is not None else current
            expands = len(patch.content) > len(before)
            allow_expansion = patch.metadata.get("allow_expansion", False)
            if not isinstance(allow_expansion, bool):
                raise MaterializationError("patch metadata allow_expansion must be boolean")
            if expands and not allow_expansion:
                raise MaterializationError(
                    "compress patch expands its selected content; set allow_expansion explicitly "
                    "only for a declared ablation"
                )
            files[patch.target_path] = _replace_exact(
                current, before, patch.content, allow_multiple=allow_multiple
            )
        else:  # pragma: no cover - Enum construction prevents this.
            raise MaterializationError(f"unsupported patch operation: {patch.operation}")

        if patch.target_path == "SKILL.md" and not files[patch.target_path].strip():
            raise MaterializationError("a patch may not empty the required root SKILL.md")


@dataclass(slots=True)
class MaterializationStore:
    """Deduplicate artifacts by content while retaining every derivation path."""

    artifacts: dict[str, Skill] = field(default_factory=dict)
    lineages: dict[str, VersionLineage] = field(default_factory=dict)

    def add(self, version: SkillVersion) -> SkillVersion:
        parent_id = version.lineage.parent_version_id
        if parent_id is not None and parent_id not in self.lineages:
            raise MaterializationError(f"lineage parent is not present in store: {parent_id}")
        existing_lineage = self.lineages.get(version.lineage.version_id)
        if existing_lineage is not None and existing_lineage != version.lineage:
            raise MaterializationError(f"version id collision: {version.lineage.version_id}")
        artifact = self.artifacts.setdefault(version.skill.content_hash, version.skill)
        if artifact.files != version.skill.files:
            raise MaterializationError("content hash collision between different artifacts")
        self.lineages[version.lineage.version_id] = version.lineage
        return SkillVersion(skill=artifact, lineage=version.lineage)

    def get(self, version_id: str) -> SkillVersion:
        lineage = self.lineages[version_id]
        return SkillVersion(skill=self.artifacts[lineage.content_hash], lineage=lineage)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifacts": {
                content_hash: skill.to_dict()
                for content_hash, skill in sorted(self.artifacts.items())
            },
            "lineages": {
                version_id: lineage.to_dict()
                for version_id, lineage in sorted(self.lineages.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MaterializationStore:
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise MaterializationError("unsupported materialization store schema")
        raw_artifacts = value.get("artifacts", {})
        raw_lineages = value.get("lineages", {})
        if not isinstance(raw_artifacts, Mapping) or not isinstance(raw_lineages, Mapping):
            raise MaterializationError("malformed materialization store")
        if any(
            not isinstance(content_hash, str) or not isinstance(skill, Mapping)
            for content_hash, skill in raw_artifacts.items()
        ):
            raise MaterializationError("artifact entries must map string hashes to objects")
        if any(
            not isinstance(version_id, str) or not isinstance(lineage, Mapping)
            for version_id, lineage in raw_lineages.items()
        ):
            raise MaterializationError("lineage entries must map string ids to objects")
        store = cls(
            artifacts={
                content_hash: Skill.from_dict(skill)  # type: ignore[arg-type]
                for content_hash, skill in raw_artifacts.items()
            },
            lineages={
                version_id: VersionLineage.from_dict(lineage)  # type: ignore[arg-type]
                for version_id, lineage in raw_lineages.items()
            },
        )
        for content_hash, skill in store.artifacts.items():
            if skill.content_hash != content_hash:
                raise MaterializationError("artifact content hash mismatch during restore")
        for version_id, lineage in store.lineages.items():
            if version_id != lineage.version_id:
                raise MaterializationError("lineage map key does not match version_id")
            if lineage.content_hash not in store.artifacts:
                raise MaterializationError("lineage references a missing artifact")
            if (
                lineage.parent_version_id is not None
                and lineage.parent_version_id not in store.lineages
            ):
                raise MaterializationError("lineage references a missing parent")
            if len(lineage.patch_ids) != len(lineage.patch_fingerprints):
                raise MaterializationError("lineage patch ids/fingerprints are misaligned")
        for version_id in store.lineages:
            visited: set[str] = set()
            current: str | None = version_id
            while current is not None:
                if current in visited:
                    raise MaterializationError("lineage graph contains a cycle")
                visited.add(current)
                current = store.lineages[current].parent_version_id
        return store

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(canonical_json(self.to_dict()) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> MaterializationStore:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
