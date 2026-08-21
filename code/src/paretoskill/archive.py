"""Bounded, confidence-aware non-dominated archive with JSON recovery."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .objectives import (
    DEFAULT_ACTIVE_OBJECTIVES,
    DominanceMode,
    FeasibilityConstraints,
    ObjectiveName,
    dominance_vector,
    dominates,
    feasibility,
    normalize_active_objectives,
)
from .statistics import ObjectiveSummary
from .models import freeze_mapping, thaw_json


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    candidate_id: str
    content_hash: str
    objectives: ObjectiveSummary
    evaluation_cost: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if (
            not isinstance(self.content_hash, str)
            or len(self.content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.content_hash)
        ):
            raise ValueError("content_hash must be lowercase SHA-256 hex")
        if not isinstance(self.objectives, ObjectiveSummary):
            raise ValueError("objectives must be an ObjectiveSummary")
        if isinstance(self.evaluation_cost, bool) or not isinstance(
            self.evaluation_cost, int
        ):
            raise ValueError("evaluation_cost must be an integer")
        if self.evaluation_cost < 0:
            raise ValueError("evaluation_cost must be non-negative")
        # Fail early instead of producing non-standard NaN/Infinity JSON.
        if not all(math.isfinite(value) for value in self.objectives.point_vector()):
            raise ValueError("objective values must be finite")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "objectives": self.objectives.to_dict(),
            "evaluation_cost": self.evaluation_cost,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchiveEntry":
        objectives = data.get("objectives")
        if not isinstance(objectives, Mapping):
            raise ValueError("objectives must be an object")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            candidate_id=data["candidate_id"],
            content_hash=data["content_hash"],
            objectives=ObjectiveSummary.from_dict(objectives),
            evaluation_cost=_strict_int(data.get("evaluation_cost", 1), "evaluation_cost"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    candidate_id: str
    removed_ids: tuple[str, ...] = ()
    evaluations_spent: int = 0


class ParetoArchive:
    """A deterministic bounded archive.

    New, non-duplicate candidates consume evaluation budget even when infeasible
    or dominated, reflecting the work needed to obtain their objective summary.
    Repeated content hashes and over-budget attempts consume no additional budget.
    Capacity pruning retains objective extremes first, then high crowding distance,
    with ``candidate_id`` as the deterministic final tie-breaker.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        max_size: int,
        evaluation_budget: int,
        constraints: FeasibilityConstraints,
        dominance_mode: DominanceMode = "uncertainty",
        active_objectives: Iterable[ObjectiveName] = DEFAULT_ACTIVE_OBJECTIVES,
    ) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be positive")
        if (
            isinstance(evaluation_budget, bool)
            or not isinstance(evaluation_budget, int)
            or evaluation_budget < 0
        ):
            raise ValueError("evaluation_budget must be non-negative")
        if not isinstance(constraints, FeasibilityConstraints):
            raise ValueError("constraints must be FeasibilityConstraints")
        if dominance_mode not in {"uncertainty", "point"}:
            raise ValueError(f"unknown dominance mode: {dominance_mode!r}")
        self.max_size = max_size
        self.evaluation_budget = evaluation_budget
        self.constraints = constraints
        self.dominance_mode = dominance_mode
        self.active_objectives = normalize_active_objectives(active_objectives)
        self.evaluations_spent = 0
        self._entries: dict[str, ArchiveEntry] = {}
        self._evaluated: dict[str, ArchiveEntry] = {}
        self._seen_hashes: set[str] = set()
        self._seen_candidate_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def remaining_budget(self) -> int:
        return self.evaluation_budget - self.evaluations_spent

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    @property
    def evaluated_entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._evaluated[key] for key in sorted(self._evaluated))

    @property
    def scientific_front(self) -> tuple[ArchiveEntry, ...]:
        return self._feasible_front()

    def get(self, candidate_id: str) -> ArchiveEntry | None:
        return self._entries.get(candidate_id)

    def admit(self, entry: ArchiveEntry) -> AdmissionDecision:
        """Attempt admission and return an explainable stable decision."""

        if entry.candidate_id in self._seen_candidate_ids:
            return self._decision(False, "duplicate_candidate_id", entry)
        if entry.content_hash in self._seen_hashes:
            return self._decision(False, "duplicate_content_hash", entry)
        if entry.evaluation_cost > self.remaining_budget:
            return self._decision(False, "evaluation_budget_exceeded", entry)

        self.evaluations_spent += entry.evaluation_cost
        self._seen_hashes.add(entry.content_hash)
        self._seen_candidate_ids.add(entry.candidate_id)
        self._evaluated[entry.candidate_id] = entry
        feasible = feasibility(
            entry.objectives, self.constraints, mode=self.dominance_mode
        )
        if not feasible.feasible:
            return self._decision(
                False, "infeasible:" + ",".join(feasible.reasons), entry
            )

        previous_ids = set(self._entries)
        full_front = self._feasible_front()
        full_front_ids = {current.candidate_id for current in full_front}
        keep_ids = self._capacity_keep_ids(full_front, self.max_size)
        self._entries = {
            current.candidate_id: current
            for current in full_front
            if current.candidate_id in keep_ids
        }
        if entry.candidate_id not in full_front_ids:
            dominators = [
                current.candidate_id
                for current in full_front
                if dominates(
                    current.objectives,
                    entry.objectives,
                    mode=self.dominance_mode,
                    active_objectives=self.active_objectives,
                )
            ]
            return self._decision(
                False,
                "dominated_by:" + ",".join(sorted(dominators)),
                entry,
            )
        removed_ids = previous_ids - set(self._entries)
        removed = tuple(sorted(removed_ids - {entry.candidate_id}))
        if entry.candidate_id not in self._entries:
            return self._decision(False, "capacity_pruned", entry, removed)
        reason = "accepted"
        if any(candidate_id not in full_front_ids for candidate_id in removed_ids):
            reason = "accepted_removed_dominated"
        elif removed_ids:
            reason = "accepted_capacity_pruned"
        return self._decision(True, reason, entry, removed)

    def _feasible_front(self) -> tuple[ArchiveEntry, ...]:
        feasible_entries = [
            entry
            for entry in self.evaluated_entries
            if feasibility(
                entry.objectives,
                self.constraints,
                mode=self.dominance_mode,
            ).feasible
        ]
        return tuple(
            entry
            for entry in feasible_entries
            if not any(
                dominates(
                    other.objectives,
                    entry.objectives,
                    mode=self.dominance_mode,
                    active_objectives=self.active_objectives,
                )
                for other in feasible_entries
                if other.candidate_id != entry.candidate_id
            )
        )

    def _decision(
        self,
        accepted: bool,
        reason: str,
        entry: ArchiveEntry,
        removed_ids: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        return AdmissionDecision(
            accepted=accepted,
            reason=reason,
            candidate_id=entry.candidate_id,
            removed_ids=removed_ids,
            evaluations_spent=self.evaluations_spent,
        )

    def _capacity_keep_ids(
        self, entries: Iterable[ArchiveEntry], capacity: int
    ) -> set[str]:
        population = sorted(entries, key=lambda entry: entry.candidate_id)
        if len(population) <= capacity:
            return {entry.candidate_id for entry in population}
        vectors = {
            entry.candidate_id: dominance_vector(
                entry.objectives,
                mode=self.dominance_mode,
                active_objectives=self.active_objectives,
            )
            for entry in population
        }
        extreme_count = {entry.candidate_id: 0 for entry in population}
        crowding = {entry.candidate_id: 0.0 for entry in population}

        for objective_index in range(len(self.active_objectives)):
            ordered = sorted(
                population,
                key=lambda entry: (
                    vectors[entry.candidate_id][objective_index],
                    entry.candidate_id,
                ),
            )
            minimum = vectors[ordered[0].candidate_id][objective_index]
            maximum = vectors[ordered[-1].candidate_id][objective_index]
            span = maximum - minimum
            if span <= 0.0:
                continue
            for current in population:
                value = vectors[current.candidate_id][objective_index]
                if value == minimum or value == maximum:
                    extreme_count[current.candidate_id] += 1
                    crowding[current.candidate_id] = math.inf
            for position in range(1, len(ordered) - 1):
                candidate_id = ordered[position].candidate_id
                if math.isinf(crowding[candidate_id]):
                    continue
                previous = vectors[ordered[position - 1].candidate_id][objective_index]
                following = vectors[ordered[position + 1].candidate_id][objective_index]
                crowding[candidate_id] += (following - previous) / span

        ranked = sorted(
            population,
            key=lambda entry: (
                -extreme_count[entry.candidate_id],
                -crowding[entry.candidate_id],
                entry.candidate_id,
            ),
        )
        return {entry.candidate_id for entry in ranked[:capacity]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "max_size": self.max_size,
            "evaluation_budget": self.evaluation_budget,
            "evaluations_spent": self.evaluations_spent,
            "dominance_mode": self.dominance_mode,
            "active_objectives": list(self.active_objectives),
            "constraints": {
                "accuracy_floor": self.constraints.accuracy_floor,
                "token_budget": self.constraints.token_budget,
                "accuracy_delta_floor": self.constraints.accuracy_delta_floor,
                "enabled": self.constraints.enabled,
            },
            "seen_hashes": sorted(self._seen_hashes),
            "seen_candidate_ids": sorted(self._seen_candidate_ids),
            "entries": [entry.to_dict() for entry in self.entries],
            "evaluated_entries": [entry.to_dict() for entry in self.evaluated_entries],
            "scientific_front": [entry.to_dict() for entry in self.scientific_front],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, allow_nan=False
        )

    def save(self, path: str | Path) -> None:
        """Atomically persist the archive snapshot."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(self.to_json() + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParetoArchive":
        if _strict_int(data.get("schema_version", -1), "schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported archive schema version")
        raw_constraints = data.get("constraints")
        if not isinstance(raw_constraints, Mapping):
            raise ValueError("constraints must be an object")
        enabled = raw_constraints.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("constraints.enabled must be a boolean")
        archive = cls(
            max_size=_strict_int(data["max_size"], "max_size"),
            evaluation_budget=_strict_int(
                data["evaluation_budget"], "evaluation_budget"
            ),
            constraints=FeasibilityConstraints(
                accuracy_floor=_strict_number(
                    raw_constraints["accuracy_floor"], "constraints.accuracy_floor"
                ),
                token_budget=_strict_number(
                    raw_constraints["token_budget"], "constraints.token_budget"
                ),
                accuracy_delta_floor=(
                    _strict_number(
                        raw_constraints["accuracy_delta_floor"],
                        "constraints.accuracy_delta_floor",
                    )
                    if raw_constraints.get("accuracy_delta_floor") is not None
                    else None
                ),
                enabled=enabled,
            ),
            dominance_mode=str(data["dominance_mode"]),  # type: ignore[arg-type]
            active_objectives=data.get(
                "active_objectives", DEFAULT_ACTIVE_OBJECTIVES
            ),  # type: ignore[arg-type]
        )
        archive.evaluations_spent = _strict_int(
            data["evaluations_spent"], "evaluations_spent"
        )
        if not 0 <= archive.evaluations_spent <= archive.evaluation_budget:
            raise ValueError("evaluations_spent is outside the configured budget")
        raw_entries = data.get("entries", [])
        raw_evaluated = data.get("evaluated_entries", [])
        raw_scientific = data.get("scientific_front", [])
        raw_hashes = data.get("seen_hashes", [])
        raw_candidate_ids = data.get("seen_candidate_ids", [])
        if not all(
            isinstance(value, list)
            for value in (
                raw_entries,
                raw_evaluated,
                raw_scientific,
                raw_hashes,
                raw_candidate_ids,
            )
        ):
            raise ValueError("archive entry ledgers and seen identifiers must be arrays")
        for raw_entry in raw_evaluated:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("evaluated archive entry must be an object")
            entry = ArchiveEntry.from_dict(raw_entry)
            if entry.candidate_id in archive._evaluated:
                raise ValueError("duplicate candidate_id in evaluated archive ledger")
            archive._evaluated[entry.candidate_id] = entry
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("archive entry must be an object")
            entry = ArchiveEntry.from_dict(raw_entry)
            if entry.candidate_id in archive._entries:
                raise ValueError("duplicate candidate_id in archive snapshot")
            archive._entries[entry.candidate_id] = entry
        if len(archive._entries) > archive.max_size:
            raise ValueError("archive snapshot exceeds max_size")
        if any(not isinstance(value, str) for value in raw_hashes + raw_candidate_ids):
            raise ValueError("seen identifiers must be strings")
        archive._seen_hashes = set(raw_hashes)
        archive._seen_candidate_ids = set(raw_candidate_ids)
        live_hashes = {entry.content_hash for entry in archive._entries.values()}
        if len(live_hashes) != len(archive._entries):
            raise ValueError("duplicate live content_hash in archive snapshot")
        if not live_hashes <= archive._seen_hashes:
            raise ValueError("live entry hash missing from seen_hashes")
        if not set(archive._entries) <= archive._seen_candidate_ids:
            raise ValueError("live candidate id missing from seen_candidate_ids")
        if set(archive._evaluated) != archive._seen_candidate_ids:
            raise ValueError("evaluated ledger and seen candidate ids differ")
        evaluated_hashes = {entry.content_hash for entry in archive._evaluated.values()}
        if evaluated_hashes != archive._seen_hashes:
            raise ValueError("evaluated ledger and seen content hashes differ")
        evaluated_cost = sum(
            entry.evaluation_cost for entry in archive._evaluated.values()
        )
        if evaluated_cost != archive.evaluations_spent:
            raise ValueError("evaluated ledger cost differs from evaluations_spent")
        computed_front = archive._feasible_front()
        if any(not isinstance(item, Mapping) for item in raw_scientific):
            raise ValueError("scientific front entry must be an object")
        restored_front = tuple(ArchiveEntry.from_dict(item) for item in raw_scientific)
        if [entry.to_dict() for entry in computed_front] != [
            entry.to_dict() for entry in restored_front
        ]:
            raise ValueError("scientific front does not match evaluated ledger")
        keep_ids = archive._capacity_keep_ids(computed_front, archive.max_size)
        computed_working = tuple(
            entry for entry in computed_front if entry.candidate_id in keep_ids
        )
        if [entry.to_dict() for entry in archive.entries] != [
            entry.to_dict() for entry in computed_working
        ]:
            raise ValueError("working archive does not match evaluated ledger")
        live_entries = archive.entries
        for entry in live_entries:
            if not feasibility(
                entry.objectives,
                archive.constraints,
                mode=archive.dominance_mode,
            ).feasible:
                raise ValueError("archive snapshot contains an infeasible entry")
        for index, left in enumerate(live_entries):
            for right in live_entries[index + 1 :]:
                if dominates(
                    left.objectives,
                    right.objectives,
                    mode=archive.dominance_mode,
                    active_objectives=archive.active_objectives,
                ) or dominates(
                    right.objectives,
                    left.objectives,
                    mode=archive.dominance_mode,
                    active_objectives=archive.active_objectives,
                ):
                    raise ValueError("archive snapshot contains dominated entries")
        return archive

    @classmethod
    def from_json(cls, payload: str) -> "ParetoArchive":
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("archive JSON root must be an object")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> "ParetoArchive":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
