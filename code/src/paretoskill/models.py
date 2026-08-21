"""Immutable, JSON-serializable models for skills, patches, and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used for all identifiers."""

    return json.dumps(
        thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _normalize_path(value: str) -> str:
    candidate = value.replace("\\", "/")
    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"skill paths must be safe relative POSIX paths: {value!r}")
    return str(path)


def freeze_json(value: Any) -> Any:
    """Recursively copy/freeze a JSON-compatible value."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numeric values must be finite")
        return value
    raise ValueError(f"value is not JSON-compatible: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return mutable JSON containers for serialization without leaking internals."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    candidate: Any = {} if value is None else value
    if not isinstance(candidate, Mapping):
        raise ValueError("expected a JSON object mapping")
    frozen = freeze_json(candidate)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above.
        raise ValueError("expected a JSON object mapping")
    return frozen


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON boolean")
    return value


class PatchOperation(str, Enum):
    ADD = "add"
    DROP = "drop"
    REWRITE = "rewrite"
    COMPRESS = "compress"


@dataclass(frozen=True, slots=True)
class TraceEvidence:
    """Auditable evidence from one task-seed-target execution."""

    evidence_id: str
    task_id: str
    seed: int
    target_id: str
    outcome: bool
    verifier_summary: str
    trace_uri: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("evidence_id", "task_id", "target_id", "verifier_summary"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.trace_uri is not None and not isinstance(self.trace_uri, str):
            raise ValueError("trace_uri must be a string or null")
        if not isinstance(self.tags, (list, tuple)):
            raise ValueError("evidence tags must be an array")
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _strict_bool(self.outcome, "outcome")
        if any(not isinstance(tag, str) or not tag for tag in self.tags):
            raise ValueError("evidence tags must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "verifier_summary": self.verifier_summary,
            "trace_uri": self.trace_uri,
            "tags": list(self.tags),
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TraceEvidence:
        return cls(
            evidence_id=value["evidence_id"],
            task_id=value["task_id"],
            seed=value["seed"],
            target_id=value["target_id"],
            outcome=_strict_bool(value["outcome"], "outcome"),
            verifier_summary=value["verifier_summary"],
            trace_uri=value.get("trace_uri"),
            tags=tuple(value.get("tags", ())),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class Patch:
    """A deterministic, evidence-grounded edit against one parent version."""

    patch_id: str
    operation: PatchOperation
    target_path: str
    parent_version_id: str
    evidence_ids: tuple[str, ...]
    content: str = ""
    match_text: str | None = None
    applicability: str = "always"
    risk_category: str = "unspecified"
    sequence: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.patch_id, str)
            or not self.patch_id.strip()
            or not isinstance(self.parent_version_id, str)
            or not self.parent_version_id.strip()
        ):
            raise ValueError("patch_id and parent_version_id must be non-empty")
        for field_name in ("target_path", "content", "applicability", "risk_category"):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"{field_name} must be a string")
        if self.match_text is not None and not isinstance(self.match_text, str):
            raise ValueError("match_text must be a string or null")
        if not isinstance(self.evidence_ids, (list, tuple)):
            raise ValueError("patch evidence ids must be an array")
        object.__setattr__(self, "operation", PatchOperation(self.operation))
        object.__setattr__(self, "target_path", _normalize_path(self.target_path))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if not self.evidence_ids:
            raise ValueError("every patch must cite at least one trace evidence id")
        if any(
            not isinstance(evidence_id, str) or not evidence_id
            for evidence_id in self.evidence_ids
        ):
            raise ValueError("patch evidence ids must be non-empty strings")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("patch sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("patch sequence must be non-negative")
        if self.operation in {PatchOperation.DROP, PatchOperation.REWRITE} and not self.match_text:
            raise ValueError(f"{self.operation.value} patches require match_text")
        if self.operation in {
            PatchOperation.ADD,
            PatchOperation.REWRITE,
            PatchOperation.COMPRESS,
        } and not self.content:
            raise ValueError(f"{self.operation.value} patches require content")

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "operation": self.operation.value,
            "target_path": self.target_path,
            "parent_version_id": self.parent_version_id,
            "evidence_ids": list(self.evidence_ids),
            "content": self.content,
            "match_text": self.match_text,
            "applicability": self.applicability,
            "risk_category": self.risk_category,
            "sequence": self.sequence,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Patch:
        return cls(
            patch_id=value["patch_id"],
            operation=PatchOperation(value["operation"]),
            target_path=value["target_path"],
            parent_version_id=value["parent_version_id"],
            evidence_ids=tuple(value["evidence_ids"]),
            content=value.get("content", ""),
            match_text=value.get("match_text"),
            applicability=value.get("applicability", "always"),
            risk_category=value.get("risk_category", "unspecified"),
            sequence=value.get("sequence", 0),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class Skill:
    """A skill directory represented as normalized relative text files."""

    name: str
    files: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("skill name must be non-empty")
        if not isinstance(self.files, Mapping):
            raise ValueError("skill files must be a mapping")
        normalized: dict[str, str] = {}
        for raw_path, raw_content in self.files.items():
            if not isinstance(raw_path, str) or not isinstance(raw_content, str):
                raise ValueError("skill paths and file contents must be strings")
            path = _normalize_path(raw_path)
            if path in normalized:
                raise ValueError(f"duplicate normalized skill path: {path}")
            normalized[path] = raw_content.replace("\r\n", "\n").replace("\r", "\n")
        if "SKILL.md" not in normalized:
            raise ValueError("a skill directory must contain root SKILL.md")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(normalized.items()))))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    @property
    def content_hash(self) -> str:
        # Metadata is intentionally excluded: execution reuse is keyed by directory content.
        return stable_hash({"format": 1, "files": dict(self.files)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "files": dict(self.files),
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Skill:
        return cls(
            name=value["name"],
            files=value["files"],
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class VersionLineage:
    """One derivation path; multiple paths may point at the same content hash."""

    version_id: str
    parent_version_id: str | None
    content_hash: str
    patch_ids: tuple[str, ...] = ()
    patch_fingerprints: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    materializer_version: str = "paretoskill-materializer/v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not self.version_id.strip() or not _is_sha256(
            self.content_hash
        ):
            raise ValueError("version_id must be non-empty and content_hash must be SHA-256")
        if self.parent_version_id is not None and (
            not isinstance(self.parent_version_id, str) or not self.parent_version_id.strip()
        ):
            raise ValueError("parent_version_id must be a non-empty string or null")
        for name in ("patch_ids", "patch_fingerprints", "evidence_ids"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise ValueError(f"lineage {name} must be an array")
        object.__setattr__(self, "patch_ids", tuple(self.patch_ids))
        object.__setattr__(self, "patch_fingerprints", tuple(self.patch_fingerprints))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if len(self.patch_ids) != len(self.patch_fingerprints):
            raise ValueError("lineage patch ids and fingerprints must align")
        if any(not isinstance(patch_id, str) or not patch_id for patch_id in self.patch_ids):
            raise ValueError("lineage patch ids must be non-empty strings")
        if any(not _is_sha256(fingerprint) for fingerprint in self.patch_fingerprints):
            raise ValueError("lineage patch fingerprints must be SHA-256")
        if any(
            not isinstance(evidence_id, str) or not evidence_id
            for evidence_id in self.evidence_ids
        ):
            raise ValueError("lineage evidence ids must be non-empty strings")
        if not isinstance(self.materializer_version, str) or not self.materializer_version.strip():
            raise ValueError("materializer_version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "content_hash": self.content_hash,
            "patch_ids": list(self.patch_ids),
            "patch_fingerprints": list(self.patch_fingerprints),
            "evidence_ids": list(self.evidence_ids),
            "materializer_version": self.materializer_version,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VersionLineage:
        return cls(
            version_id=value["version_id"],
            parent_version_id=value.get("parent_version_id"),
            content_hash=value["content_hash"],
            patch_ids=tuple(value.get("patch_ids", ())),
            patch_fingerprints=tuple(value.get("patch_fingerprints", ())),
            evidence_ids=tuple(value.get("evidence_ids", ())),
            materializer_version=value.get(
                "materializer_version", "paretoskill-materializer/v1"
            ),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class SkillVersion:
    skill: Skill
    lineage: VersionLineage

    def __post_init__(self) -> None:
        if not isinstance(self.skill, Skill) or not isinstance(self.lineage, VersionLineage):
            raise ValueError("SkillVersion requires Skill and VersionLineage values")
        if self.skill.content_hash != self.lineage.content_hash:
            raise ValueError("skill content does not match lineage content_hash")

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill.to_dict(), "lineage": self.lineage.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkillVersion:
        return cls(
            skill=Skill.from_dict(value["skill"]),
            lineage=VersionLineage.from_dict(value["lineage"]),
        )


def make_base_version(skill: Skill, *, label: str = "base") -> SkillVersion:
    content_hash = skill.content_hash
    version_id = f"{label}-{stable_hash({'parent': None, 'content_hash': content_hash})[:16]}"
    return SkillVersion(
        skill=skill,
        lineage=VersionLineage(
            version_id=version_id,
            parent_version_id=None,
            content_hash=content_hash,
            metadata={"kind": "base"},
        ),
    )
