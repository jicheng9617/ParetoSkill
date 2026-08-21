"""Strict local asset loading for configured experiments.

This module performs no downloads and has no network-capable code.  Every real
run must pass content-pin preflight before these assets are consumed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import Patch, Skill, SkillVersion, TraceEvidence, make_base_version


class RuntimeAssetError(ValueError):
    """Raised when a configured local experiment asset is malformed."""


def _read_collection(path: Path, *, collection_key: str) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise RuntimeAssetError(f"asset does not exist: {path}")
    try:
        if path.suffix.lower() == ".jsonl":
            values: Any = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix.lower() == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise RuntimeAssetError(f"unsupported asset format: {path.suffix}")
    except (json.JSONDecodeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeAssetError(f"cannot parse asset {path}: {exc}") from exc
    if isinstance(values, Mapping):
        values = values.get(collection_key)
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeAssetError(
            f"{path} must contain an array or {collection_key!r} array of objects"
        )
    return list(values)


def load_base_skill(path: str | Path, *, name: str | None = None) -> SkillVersion:
    """Load a base skill from JSON, a single text file, or a text-only directory."""

    source = Path(path).resolve()
    if not source.exists():
        raise RuntimeAssetError(f"base skill does not exist: {source}")
    if source.is_symlink():
        raise RuntimeAssetError("base skill root may not be a symbolic link")
    if source.is_file() and source.suffix.lower() == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeAssetError(f"cannot parse base skill JSON {source}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise RuntimeAssetError("base skill JSON must contain an object")
        if "skill" in value and "lineage" in value:
            version = SkillVersion.from_dict(value)
            if version.lineage.parent_version_id is not None:
                raise RuntimeAssetError("configured base SkillVersion must have no parent")
            return version
        skill = Skill.from_dict(value)
        return make_base_version(skill)

    files: dict[str, str] = {}
    if source.is_file():
        try:
            files["SKILL.md"] = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeAssetError(f"base skill must be UTF-8 text: {source}") from exc
    elif source.is_dir():
        for item in sorted(source.rglob("*"), key=lambda entry: entry.as_posix()):
            if item.is_symlink():
                raise RuntimeAssetError(f"base skill may not contain symbolic links: {item}")
            if not item.is_file():
                continue
            relative = item.relative_to(source).as_posix()
            try:
                files[relative] = item.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeAssetError(
                    f"base skill directory must contain UTF-8 text only: {item}"
                ) from exc
    else:  # pragma: no cover - resolved existence and file/dir cover normal filesystems.
        raise RuntimeAssetError(f"unsupported base skill path: {source}")
    if "SKILL.md" not in files or not files["SKILL.md"].strip():
        raise RuntimeAssetError("base skill requires a non-empty SKILL.md")
    return make_base_version(
        Skill(name=name or source.stem or source.name, files=files),
    )


def load_trace_evidence(path: str | Path) -> dict[str, TraceEvidence]:
    """Load a frozen JSON/JSONL/YAML trace-evidence collection."""

    source = Path(path).resolve()
    rows = _read_collection(source, collection_key="evidence")
    evidence: dict[str, TraceEvidence] = {}
    for index, row in enumerate(rows, start=1):
        try:
            item = TraceEvidence.from_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeAssetError(
                f"invalid trace evidence at {source} item {index}: {exc}"
            ) from exc
        if item.evidence_id in evidence:
            raise RuntimeAssetError(f"duplicate evidence_id: {item.evidence_id}")
        evidence[item.evidence_id] = item
    if not evidence:
        raise RuntimeAssetError("trace evidence collection may not be empty")
    return evidence


def load_patch_pool(
    path: str | Path,
    *,
    base_version_id: str,
    evidence: Mapping[str, TraceEvidence],
) -> tuple[Patch, ...]:
    """Load a patch pool and bind an explicit ``$BASE`` parent placeholder."""

    if not isinstance(base_version_id, str) or not base_version_id.strip():
        raise RuntimeAssetError("base_version_id must be non-empty")
    source = Path(path).resolve()
    rows = _read_collection(source, collection_key="patches")
    patches: list[Patch] = []
    identifiers: set[str] = set()
    for index, row in enumerate(rows, start=1):
        prepared = dict(row)
        if prepared.get("parent_version_id") == "$BASE":
            prepared["parent_version_id"] = base_version_id
        try:
            patch = Patch.from_dict(prepared)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeAssetError(f"invalid patch at {source} item {index}: {exc}") from exc
        if patch.parent_version_id != base_version_id:
            raise RuntimeAssetError(
                f"patch {patch.patch_id!r} targets {patch.parent_version_id!r}; "
                f"configured pool must target {base_version_id!r} or use '$BASE'"
            )
        if patch.patch_id in identifiers:
            raise RuntimeAssetError(f"duplicate patch_id: {patch.patch_id}")
        missing = set(patch.evidence_ids) - set(evidence)
        if missing:
            raise RuntimeAssetError(
                f"patch {patch.patch_id!r} cites unknown evidence: {sorted(missing)}"
            )
        identifiers.add(patch.patch_id)
        patches.append(patch)
    if not patches:
        raise RuntimeAssetError("patch pool may not be empty")
    # Canonical ordering makes subset proposals and experiment IDs stable.
    return tuple(sorted(patches, key=lambda patch: (patch.sequence, patch.patch_id)))


def bind_patch_parent(patch: Patch, parent: SkillVersion) -> Patch:
    """Rebind a generated/pool patch to a selected parent without mutating it."""

    return replace(patch, parent_version_id=parent.lineage.version_id)
