"""Result schema, content-addressed cache, budget ledger, and resumable checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluation import EvaluationRecord
from .models import canonical_json


class StorageError(RuntimeError):
    pass


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(slots=True)
class ResultCache:
    """One immutable JSON record per evaluation request cache key."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, cache_key: str) -> Path:
        invalid_character = any(
            character not in "0123456789abcdef" for character in cache_key
        )
        if len(cache_key) != 64 or invalid_character:
            raise StorageError("cache keys must be lowercase SHA-256 hex")
        return self.root / cache_key[:2] / f"{cache_key}.json"

    def get(self, cache_key: str) -> EvaluationRecord | None:
        path = self._path(cache_key)
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        record = EvaluationRecord.from_dict(row)
        if record.cache_key != cache_key:
            raise StorageError(f"cache record key mismatch: {path}")
        return record

    def keys(self) -> set[str]:
        if not self.root.exists():
            return set()
        keys: set[str] = set()
        for path in self.root.rglob("*.json"):
            cache_key = path.stem
            # Validate both filename shape and stored record integrity.
            if self.get(cache_key) is None:  # pragma: no cover - rglob found it.
                raise StorageError(f"cache file disappeared during scan: {path}")
            keys.add(cache_key)
        return keys

    def put(self, record: EvaluationRecord) -> None:
        path = self._path(record.cache_key)
        if path.exists():
            existing = EvaluationRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.to_dict() != record.to_dict():
                raise StorageError(f"immutable cache collision for {record.cache_key}")
            return
        _atomic_json_write(path, record.to_dict())


@dataclass(slots=True)
class JsonlResultStore:
    path: Path

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: EvaluationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record.to_dict()) + "\n")

    def replace(self, records: Iterable[EvaluationRecord]) -> None:
        """Atomically replace the ledger with one row per evaluation identity."""

        rows = tuple(records)
        identities = [record.evaluation_identity for record in rows]
        if len(set(identities)) != len(identities):
            raise StorageError("cannot persist duplicate evaluation identities")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(rows, key=lambda item: item.evaluation_identity):
                handle.write(canonical_json(record.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def read_all(self) -> list[EvaluationRecord]:
        if not self.path.exists():
            return []
        records: list[EvaluationRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(EvaluationRecord.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise StorageError(
                        f"invalid result record at {self.path}:{line_number}: {exc}"
                    ) from exc
        return records


@dataclass(slots=True)
class ExecutionBudget:
    limit: int
    consumed: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.limit, self.consumed)
        ):
            raise ValueError("execution budget values must be integers")
        if self.limit < 0 or self.consumed < 0 or self.consumed > self.limit:
            raise ValueError("invalid execution budget")

    @property
    def remaining(self) -> int:
        return self.limit - self.consumed

    def consume(self, amount: int = 1) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError("budget consumption must be an integer")
        if amount < 0:
            raise ValueError("cannot consume a negative budget")
        if amount > self.remaining:
            raise StorageError(
                f"task-execution budget exhausted: requested {amount}, remaining {self.remaining}"
            )
        self.consumed += amount


@dataclass(slots=True)
class ExperimentCheckpoint:
    experiment_id: str
    completed_cache_keys: set[str] = field(default_factory=set)
    proposed_candidate_ids: list[str] = field(default_factory=list)
    task_executions_consumed: int = 0
    round_index: int = 0
    archive_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise StorageError("checkpoint experiment_id must be non-empty")
        self.completed_cache_keys = set(self.completed_cache_keys)
        self.proposed_candidate_ids = list(self.proposed_candidate_ids)
        if any(
            not isinstance(cache_key, str)
            or len(cache_key) != 64
            or any(character not in "0123456789abcdef" for character in cache_key)
            for cache_key in self.completed_cache_keys
        ):
            raise StorageError("checkpoint contains an invalid cache key")
        if any(
            not isinstance(candidate_id, str) or not candidate_id.strip()
            for candidate_id in self.proposed_candidate_ids
        ):
            raise StorageError("checkpoint contains an invalid proposed candidate id")
        if len(set(self.proposed_candidate_ids)) != len(self.proposed_candidate_ids):
            raise StorageError("checkpoint contains duplicate proposed candidate ids")
        for name, value in (
            ("task_executions_consumed", self.task_executions_consumed),
            ("round_index", self.round_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StorageError(f"checkpoint {name} must be a non-negative integer")
        if self.task_executions_consumed < len(self.completed_cache_keys):
            raise StorageError("checkpoint consumed count is smaller than its completed cache set")
        if self.archive_path is not None:
            if not isinstance(self.archive_path, str) or not self.archive_path.strip():
                raise StorageError("checkpoint archive_path must be a non-empty string or null")
            if "\x00" in self.archive_path:
                raise StorageError("checkpoint archive_path contains a null byte")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "completed_cache_keys": sorted(self.completed_cache_keys),
            "proposed_candidate_ids": list(self.proposed_candidate_ids),
            "task_executions_consumed": self.task_executions_consumed,
            "round_index": self.round_index,
            "archive_path": self.archive_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentCheckpoint:
        if value.get("schema_version") != 1:
            raise StorageError("unsupported checkpoint schema")
        if "experiment_id" not in value:
            raise StorageError("checkpoint is missing experiment_id")
        raw_completed = value.get("completed_cache_keys", [])
        raw_proposed = value.get("proposed_candidate_ids", [])
        if not isinstance(raw_completed, list) or not isinstance(raw_proposed, list):
            raise StorageError("checkpoint cache keys and candidate ids must be arrays")
        consumed = value.get("task_executions_consumed", 0)
        round_index = value.get("round_index", 0)
        return cls(
            experiment_id=value["experiment_id"],
            completed_cache_keys=set(raw_completed),
            proposed_candidate_ids=list(raw_proposed),
            task_executions_consumed=consumed,
            round_index=round_index,
            archive_path=value.get("archive_path"),
        )

    def save(self, path: str | Path) -> None:
        _atomic_json_write(Path(path), self.to_dict())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_experiment_id: str,
        max_task_executions: int | None = None,
        expected_output_root: str | Path | None = None,
    ) -> ExperimentCheckpoint:
        source = Path(path)
        checkpoint = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        if checkpoint.experiment_id != expected_experiment_id:
            raise StorageError(
                "checkpoint belongs to a different experiment: "
                f"{checkpoint.experiment_id} != {expected_experiment_id}"
            )
        if max_task_executions is not None:
            if (
                isinstance(max_task_executions, bool)
                or not isinstance(max_task_executions, int)
                or max_task_executions < 0
            ):
                raise ValueError("max_task_executions must be a non-negative integer")
            if checkpoint.task_executions_consumed > max_task_executions:
                raise StorageError("checkpoint exceeds the configured task-execution budget")
        if checkpoint.archive_path is not None and expected_output_root is not None:
            root = Path(expected_output_root).resolve()
            archive = Path(checkpoint.archive_path)
            resolved = (
                archive.resolve()
                if archive.is_absolute()
                else (source.parent / archive).resolve()
            )
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise StorageError(
                    "checkpoint archive_path escapes the experiment output root"
                ) from exc
        return checkpoint


def cache_records(cache: ResultCache, records: Iterable[EvaluationRecord]) -> None:
    for record in records:
        cache.put(record)
