"""Small task-ID manifest validation without loading benchmark payloads."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import PLACEHOLDER


class TaskManifestError(ValueError):
    pass


def _task_id(item: Any, *, path: Path) -> str:
    if isinstance(item, str):
        task_id = item
    elif isinstance(item, Mapping):
        raw = item.get("task_id", item.get("id"))
        task_id = raw if isinstance(raw, str) else ""
    else:
        task_id = ""
    if not task_id.strip():
        raise TaskManifestError(f"task entry has no task_id/id in {path}")
    return task_id


def load_task_ids(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    if not source.is_file():
        raise TaskManifestError(f"task manifest does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            items = payload.get("task_ids", payload.get("tasks"))
        else:
            items = payload
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            items = payload.get("task_ids", payload.get("tasks"))
        else:
            items = payload
    else:
        raise TaskManifestError(f"unsupported task manifest format: {source.suffix}")
    if not isinstance(items, list):
        raise TaskManifestError(f"task manifest must contain a list: {source}")
    identifiers = tuple(_task_id(item, path=source) for item in items)
    duplicates = sorted(
        task_id for task_id, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise TaskManifestError(f"duplicate task ids in {source}: {duplicates[:5]}")
    return identifiers


def validate_declared_splits(
    configuration: Mapping[str, Any],
    *,
    split_ids: Iterable[str] | None = None,
    base_directory: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    raw_splits = configuration.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise TaskManifestError("configuration.splits must be a mapping")
    selected = set(raw_splits) if split_ids is None else set(split_ids)
    loaded: dict[str, tuple[str, ...]] = {}
    for split_id in sorted(selected):
        if split_id not in raw_splits:
            raise TaskManifestError(f"unknown requested split: {split_id}")
        split = raw_splits[split_id]
        if not isinstance(split, Mapping):
            raise TaskManifestError(f"split {split_id} must be a mapping")
        path = str(split.get("manifest", ""))
        if not path or PLACEHOLDER.search(path):
            raise TaskManifestError(f"split {split_id} manifest path is unresolved")
        source = Path(path)
        if not source.is_absolute() and base_directory is not None:
            source = Path(base_directory) / source
        declared_digest = split.get("manifest_sha256")
        if (
            not isinstance(declared_digest, str)
            or PLACEHOLDER.search(declared_digest)
            or len(declared_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in declared_digest
            )
        ):
            raise TaskManifestError(
                f"split {split_id} manifest_sha256 is unresolved or invalid"
            )
        if not source.is_file():
            raise TaskManifestError(f"task manifest does not exist: {source}")
        observed_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed_digest != declared_digest:
            raise TaskManifestError(
                f"split {split_id} manifest digest mismatch: "
                f"expected {declared_digest}, observed {observed_digest}"
            )
        identifiers = load_task_ids(source)
        expected_count = split.get("expected_count")
        if expected_count is not None and (
            isinstance(expected_count, bool) or not isinstance(expected_count, int)
        ):
            raise TaskManifestError(f"split {split_id} expected_count must be an integer")
        if expected_count is not None and len(identifiers) != expected_count:
            raise TaskManifestError(
                f"split {split_id} has {len(identifiers)} tasks, expected {expected_count}"
            )
        loaded[split_id] = identifiers

    for split_id, identifiers in loaded.items():
        declaration = raw_splits[split_id]
        assert isinstance(declaration, Mapping)
        for other_id in declaration.get("disjoint_from", []):
            if other_id not in loaded:
                continue
            overlap = set(identifiers) & set(loaded[other_id])
            if overlap:
                raise TaskManifestError(
                    f"declared-disjoint splits {split_id} and {other_id} overlap: "
                    f"{sorted(overlap)[:5]}"
                )
    return loaded
