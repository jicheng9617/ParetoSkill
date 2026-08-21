"""Pure, held-out analysis helpers for the frozen ParetoSkill experiment.

This module deliberately has no provider, filesystem, or runner dependencies.
It consumes already-produced :class:`EvaluationRecord` and
:class:`ObjectiveSummary` values, so final-test analysis can be reviewed and
tested independently from execution.

Undefined scientific quantities are represented by :class:`DefinedScalar`.
Invalid or incomplete inputs raise :class:`FinalAnalysisError`; an empty
feasible approximation, by contrast, is a valid analysis state with explicit
metric-specific semantics.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .evaluation import EvaluationRecord
from .metrics import (
    additive_epsilon_indicator,
    coverage,
    hypervolume,
    inverted_generational_distance,
    nondominated,
    normalize_vectors,
    strictly_dominates,
)
from .objectives import FeasibilityConstraints, feasibility
from .statistics import ObjectiveSummary, PairedObservation, paired_bootstrap


FinalRole = Literal["id", "transfer", "diagnostic"]
RunKey = tuple[str, int]


class FinalAnalysisError(ValueError):
    """Raised when frozen final-analysis inputs are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class DefinedScalar:
    """A finite scalar or an explicit reason why that scalar is undefined."""

    defined: bool
    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.defined, bool):
            raise ValueError("defined must be a boolean")
        if self.defined:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or not math.isfinite(float(self.value))
            ):
                raise ValueError("a defined scalar must have a finite numeric value")
            if self.reason is not None:
                raise ValueError("a defined scalar cannot have an undefined reason")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("an undefined scalar requires a non-empty reason and null value")

    @classmethod
    def of(cls, value: float) -> DefinedScalar:
        return cls(True, value)

    @classmethod
    def undefined(cls, reason: str) -> DefinedScalar:
        return cls(False, None, reason)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"defined": self.defined, "value": self.value}
        if self.reason is not None:
            result["reason"] = self.reason
        return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalAnalysisError(f"{name} must be a mapping")
    return value


def _final_targets(configuration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_targets = _mapping(configuration.get("targets"), "configuration.targets")
    targets: dict[str, Mapping[str, Any]] = {}
    for target_id, raw_target in raw_targets.items():
        if not isinstance(target_id, str) or not target_id:
            raise FinalAnalysisError("target ids must be non-empty strings")
        target = _mapping(raw_target, f"target {target_id!r}")
        if target.get("phase") == "final_only":
            targets[target_id] = target
    if not targets:
        raise FinalAnalysisError("configuration declares no final_only targets")
    return targets


def resolve_final_target_roles(configuration: Mapping[str, Any]) -> dict[str, FinalRole]:
    """Resolve final targets into primary ID, primary transfer, or diagnostics.

    ``exclude_from_primary_pooled_metrics`` always wins over any role hint.  A
    manifest may declare roles through ``final_analysis.role_by_target`` or a
    per-target ``final_analysis_role``/``analysis_role``/``objective_role``.
    The checked-in manifest's ``primary_verified_report`` aggregation role is
    treated as ID, while remaining targets with a transfer group are transfer.
    Ambiguous targets fail closed instead of being silently pooled.
    """

    targets = _final_targets(configuration)
    raw_analysis = configuration.get("final_analysis", {})
    analysis = _mapping(raw_analysis, "configuration.final_analysis")
    raw_overrides = analysis.get("role_by_target", {})
    overrides = _mapping(raw_overrides, "final_analysis.role_by_target")
    unknown_overrides = set(overrides) - set(targets)
    if unknown_overrides:
        raise FinalAnalysisError(
            f"final role overrides reference unknown targets: {sorted(unknown_overrides)!r}"
        )

    roles: dict[str, FinalRole] = {}
    for target_id, target in sorted(targets.items()):
        if target.get("exclude_from_primary_pooled_metrics") is True:
            roles[target_id] = "diagnostic"
            continue
        raw_role = overrides.get(target_id)
        if raw_role is None:
            for key in ("final_analysis_role", "analysis_role", "objective_role"):
                if target.get(key) is not None:
                    raw_role = target[key]
                    break
        if raw_role == "excluded":
            raw_role = "diagnostic"
        if raw_role in {"id", "transfer", "diagnostic"}:
            roles[target_id] = raw_role
            continue
        aggregation_role = target.get("aggregation_role")
        if aggregation_role == "primary_verified_report":
            roles[target_id] = "id"
        elif aggregation_role == "separate_full_collection_report":
            roles[target_id] = "diagnostic"
        elif isinstance(target.get("transfer_group"), str) and target["transfer_group"]:
            roles[target_id] = "transfer"
        else:
            raise FinalAnalysisError(
                f"final target {target_id!r} has no unambiguous analysis role"
            )
    if "id" not in roles.values():
        raise FinalAnalysisError("final analysis requires at least one ID target")
    if "transfer" not in roles.values():
        raise FinalAnalysisError("final analysis requires at least one transfer target")
    return roles


def _identity_value(mapping: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    return tuple(str(mapping.get(key, "<missing>")) for key in keys)


def _execution_signature(
    target: Mapping[str, Any], configuration: Mapping[str, Any]
) -> tuple[str, ...]:
    models = _mapping(configuration.get("models", {}), "configuration.models")
    harnesses = _mapping(configuration.get("harnesses", {}), "configuration.harnesses")
    domains = _mapping(configuration.get("domains", {}), "configuration.domains")
    controls = _mapping(
        configuration.get("shared_search_controls", {}),
        "configuration.shared_search_controls",
    )
    model_key = target.get("model")
    harness_key = target.get("harness")
    domain_key = target.get("domain")
    if not all(isinstance(value, str) and value for value in (model_key, harness_key, domain_key)):
        raise FinalAnalysisError("final targets require model, harness, and domain ids")
    model = _mapping(models.get(model_key), f"model {model_key!r}")
    harness = _mapping(harnesses.get(harness_key), f"harness {harness_key!r}")
    domain = _mapping(domains.get(domain_key), f"domain {domain_key!r}")
    return (
        "model",
        str(model_key),
        *_identity_value(model, "model_id", "revision"),
        "harness",
        str(harness_key),
        *_identity_value(harness, "adapter", "repository_revision"),
        "domain-adapter",
        *_identity_value(domain, "adapter", "adapter_revision"),
        "verifier",
        *_identity_value(controls, "verifier", "verifier_revision"),
    )


def resolve_final_execution_groups(configuration: Mapping[str, Any]) -> dict[str, str]:
    """Return conservative execution groups used for overlap deduplication.

    Targets remain isolated by default.  Automatic coalescing occurs only when
    at least one target in an otherwise identical execution signature opts in
    through ``reuse_verified_execution_when_canonical_keys_match``.  Explicit
    groups may instead be frozen in ``final_analysis.execution_identity_groups``
    or per-target ``canonical_execution_group``.
    """

    targets = _final_targets(configuration)
    analysis = _mapping(configuration.get("final_analysis", {}), "final_analysis")
    overrides = _mapping(
        analysis.get("execution_identity_groups", {}),
        "final_analysis.execution_identity_groups",
    )
    unknown_overrides = set(overrides) - set(targets)
    if unknown_overrides:
        raise FinalAnalysisError(
            "execution identity groups reference unknown targets: "
            f"{sorted(unknown_overrides)!r}"
        )

    signatures = {
        target_id: _execution_signature(target, configuration)
        for target_id, target in targets.items()
    }
    reusable = {
        signature
        for target_id, signature in signatures.items()
        if targets[target_id].get("reuse_verified_execution_when_canonical_keys_match")
        is True
    }
    groups: dict[str, str] = {}
    for target_id, target in sorted(targets.items()):
        explicit = overrides.get(target_id, target.get("canonical_execution_group"))
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit:
                raise FinalAnalysisError("execution identity groups must be non-empty strings")
            groups[target_id] = f"explicit:{explicit}"
        elif signatures[target_id] in reusable:
            payload = json.dumps(signatures[target_id], ensure_ascii=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            groups[target_id] = f"canonical:{digest}"
        else:
            groups[target_id] = f"target:{target_id}"
    return groups


@dataclass(frozen=True, slots=True)
class FinalRecordPartition:
    """Deduplicated final records separated by their scientific role."""

    id_records: tuple[EvaluationRecord, ...]
    transfer_records: tuple[EvaluationRecord, ...]
    diagnostic_records: tuple[EvaluationRecord, ...]
    target_roles: tuple[tuple[str, FinalRole], ...]
    execution_groups: tuple[tuple[str, str], ...]
    input_record_count: int
    dropped_overlap_count: int

    @property
    def primary_records(self) -> tuple[EvaluationRecord, ...]:
        return self.id_records + self.transfer_records

    def to_dict(self) -> dict[str, object]:
        return {
            "input_record_count": self.input_record_count,
            "deduplicated_record_count": len(self.primary_records)
            + len(self.diagnostic_records),
            "dropped_overlap_count": self.dropped_overlap_count,
            "id_record_count": len(self.id_records),
            "transfer_record_count": len(self.transfer_records),
            "diagnostic_record_count": len(self.diagnostic_records),
            "target_roles": dict(self.target_roles),
            "execution_groups": dict(self.execution_groups),
        }


def _record_sort_key(record: EvaluationRecord) -> tuple[object, ...]:
    return (
        record.candidate_id,
        record.task_id,
        record.seed,
        record.target_id,
        record.cache_key,
    )


def partition_final_records(
    records: Iterable[EvaluationRecord], configuration: Mapping[str, Any]
) -> FinalRecordPartition:
    """Classify and deduplicate final rows by canonical task execution identity.

    The deduplication identity is candidate/content, canonical task id,
    execution seed, and the frozen execution group.  A primary record takes
    precedence over its excluded diagnostic duplicate.  Two conflicting
    outcomes or a collision between ID and transfer roles is rejected.
    """

    rows = tuple(records)
    if not rows:
        raise FinalAnalysisError("final analysis requires at least one evaluation record")
    if any(not isinstance(row, EvaluationRecord) for row in rows):
        raise FinalAnalysisError("records must contain EvaluationRecord values")
    roles = resolve_final_target_roles(configuration)
    groups = resolve_final_execution_groups(configuration)
    unknown = sorted({row.target_id for row in rows} - set(roles))
    if unknown:
        raise FinalAnalysisError(f"records contain non-final or unknown targets: {unknown!r}")

    hashes: dict[str, set[str]] = {}
    base_flags: dict[str, set[bool]] = {}
    for row in rows:
        hashes.setdefault(row.candidate_id, set()).add(row.content_hash)
        base_flags.setdefault(row.candidate_id, set()).add(row.is_base)
    drift = sorted(candidate_id for candidate_id, values in hashes.items() if len(values) != 1)
    if drift:
        raise FinalAnalysisError(f"candidate content drift in final records: {drift!r}")
    mixed_base = sorted(
        candidate_id for candidate_id, values in base_flags.items() if len(values) != 1
    )
    if mixed_base:
        raise FinalAnalysisError(f"candidate base-role drift in final records: {mixed_base!r}")

    grouped: dict[tuple[str, str, str, int, str], list[EvaluationRecord]] = {}
    for row in rows:
        identity = (
            row.candidate_id,
            row.content_hash,
            row.task_id,
            row.seed,
            groups[row.target_id],
        )
        grouped.setdefault(identity, []).append(row)

    selected: list[EvaluationRecord] = []
    dropped = 0
    role_priority = {"id": 0, "transfer": 1, "diagnostic": 2}
    for identity, duplicates in sorted(grouped.items()):
        target_counts = Counter(row.target_id for row in duplicates)
        repeated_target = sorted(target for target, count in target_counts.items() if count > 1)
        if repeated_target:
            raise FinalAnalysisError(
                f"duplicate final records for target identity {identity!r}: {repeated_target!r}"
            )
        outcomes = {
            (row.result.correct, row.result.input_tokens, row.result.output_tokens)
            for row in duplicates
        }
        if len(outcomes) != 1:
            raise FinalAnalysisError(
                f"conflicting outcomes for overlapping task identity {identity!r}"
            )
        primary_roles = {roles[row.target_id] for row in duplicates} - {"diagnostic"}
        if len(primary_roles) > 1:
            raise FinalAnalysisError(
                f"overlapping task identity spans ID and transfer roles: {identity!r}"
            )
        transfer_groups = {
            row.transfer_group or row.target_id
            for row in duplicates
            if roles[row.target_id] == "transfer"
        }
        if len(transfer_groups) > 1:
            raise FinalAnalysisError(
                f"overlapping transfer identity spans groups: {identity!r}"
            )
        chosen = min(
            duplicates,
            key=lambda row: (role_priority[roles[row.target_id]], row.target_id),
        )
        selected.append(chosen)
        dropped += len(duplicates) - 1

    by_role: dict[FinalRole, list[EvaluationRecord]] = {
        "id": [],
        "transfer": [],
        "diagnostic": [],
    }
    for row in selected:
        by_role[roles[row.target_id]].append(row)
    return FinalRecordPartition(
        id_records=tuple(sorted(by_role["id"], key=_record_sort_key)),
        transfer_records=tuple(sorted(by_role["transfer"], key=_record_sort_key)),
        diagnostic_records=tuple(sorted(by_role["diagnostic"], key=_record_sort_key)),
        target_roles=tuple(sorted(roles.items())),
        execution_groups=tuple(sorted(groups.items())),
        input_record_count=len(rows),
        dropped_overlap_count=dropped,
    )


def build_final_objective_summaries(
    records: Iterable[EvaluationRecord],
    configuration: Mapping[str, Any],
    *,
    candidate_ids: Iterable[str] | None = None,
    confidence_level: float | None = None,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, ObjectiveSummary]:
    """Build role-correct paired summaries from final task outcomes.

    Diagnostics excluded by the manifest never contribute to any of the four
    primary objectives.  Every requested candidate must have the exact same
    deduplicated primary task matrix as the single declared base candidate.
    """

    partition = partition_final_records(records, configuration)
    primary = partition.primary_records
    if not partition.id_records or not partition.transfer_records:
        raise FinalAnalysisError("deduplicated final records require both ID and transfer rows")
    base_ids = {row.candidate_id for row in primary if row.is_base}
    if len(base_ids) != 1:
        raise FinalAnalysisError("final records require exactly one base candidate id")
    base_id = next(iter(base_ids))
    available = sorted({row.candidate_id for row in primary if not row.is_base})
    requested = available if candidate_ids is None else list(candidate_ids)
    if not requested:
        raise FinalAnalysisError("no final candidate ids were requested")
    if len(set(requested)) != len(requested) or any(
        not isinstance(candidate_id, str) or not candidate_id for candidate_id in requested
    ):
        raise FinalAnalysisError("candidate_ids must be unique non-empty strings")
    if base_id in requested:
        raise FinalAnalysisError("the base candidate cannot be summarized as a treatment")
    missing_candidates = sorted(set(requested) - set(available))
    if missing_candidates:
        raise FinalAnalysisError(
            f"requested candidates have no final rows: {missing_candidates!r}"
        )

    roles = dict(partition.target_roles)
    targets = _final_targets(configuration)
    base_by_key = {
        (row.target_id, row.task_id, row.seed): row for row in primary if row.is_base
    }
    expected_keys = set(base_by_key)
    if len(base_by_key) != sum(row.is_base for row in primary):
        raise FinalAnalysisError("base final rows contain duplicate task identities")
    transfer_groups = {
        row.transfer_group or str(targets[row.target_id].get("transfer_group") or row.target_id)
        for row in primary
        if row.is_base and roles[row.target_id] == "transfer"
    }
    if not transfer_groups:
        raise FinalAnalysisError("final records contain no transfer groups")

    statistics = _mapping(configuration.get("statistics", {}), "configuration.statistics")
    level = (
        float(statistics.get("confidence_level", 0.95))
        if confidence_level is None
        else float(confidence_level)
    )
    replicates = (
        int(statistics.get("bootstrap_replicates", 2_000))
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    if bootstrap_seed is None:
        blocks = _mapping(configuration.get("task_seed_blocks", {}), "task_seed_blocks")
        resolved_seed = int(blocks.get("bootstrap_seed", 0))
    else:
        resolved_seed = bootstrap_seed
    minimum_blocks = int(statistics.get("minimum_effective_blocks_for_archive", 2))
    raw_constraints = _mapping(configuration.get("constraints", {}), "constraints")
    raw_token = _mapping(raw_constraints.get("token_budget", {}), "constraints.token_budget")
    token_upper = raw_token.get("budget")
    token_upper_bound = float(token_upper) if token_upper is not None else None

    summaries: dict[str, ObjectiveSummary] = {}
    for candidate_id in sorted(requested):
        candidate_rows = [row for row in primary if row.candidate_id == candidate_id]
        keys = {(row.target_id, row.task_id, row.seed) for row in candidate_rows}
        if keys != expected_keys or len(keys) != len(candidate_rows):
            missing = sorted(expected_keys - keys)[:3]
            extra = sorted(keys - expected_keys)[:3]
            raise FinalAnalysisError(
                f"candidate {candidate_id!r} final matrix mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )
        observations: list[PairedObservation] = []
        for row in sorted(candidate_rows, key=_record_sort_key):
            base = base_by_key[(row.target_id, row.task_id, row.seed)]
            role = roles[row.target_id]
            if role not in {"id", "transfer"}:
                raise AssertionError("diagnostic record leaked into primary records")
            group = None
            if role == "transfer":
                group = row.transfer_group or str(
                    targets[row.target_id].get("transfer_group") or row.target_id
                )
            observations.append(
                PairedObservation(
                    task_id=row.task_id,
                    seed=row.seed,
                    split=role,
                    target=row.target_id,
                    group=group,
                    candidate_correct=row.result.correct,
                    base_correct=base.result.correct,
                    input_tokens=row.result.input_tokens,
                    output_tokens=row.result.output_tokens,
                )
            )
        summaries[candidate_id] = paired_bootstrap(
            observations,
            confidence_level=level,
            replicates=replicates,
            seed=resolved_seed,
            expected_transfer_groups=transfer_groups,
            min_effective_blocks=minimum_blocks,
            token_cost_upper_bound=token_upper_bound,
        )
    return summaries


def _constraints_from_configuration(
    configuration: Mapping[str, Any],
) -> FeasibilityConstraints:
    raw = _mapping(configuration.get("constraints"), "configuration.constraints")
    accuracy = _mapping(raw.get("id_accuracy_floor"), "constraints.id_accuracy_floor")
    tokens = _mapping(raw.get("token_budget"), "constraints.token_budget")
    try:
        epsilon = float(accuracy["epsilon"])
        token_budget = float(tokens["budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalAnalysisError("resolved numeric final constraints are required") from exc
    return FeasibilityConstraints.from_paired_epsilon(
        epsilon=epsilon,
        token_budget=token_budget,
        enabled=raw.get("enabled") is not False,
    )


def _normalization_from_configuration(
    configuration: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    protocol = _mapping(
        configuration.get("selection_protocol"), "configuration.selection_protocol"
    )
    raw = protocol.get("normalization_ranges")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise FinalAnalysisError(
            "selection_protocol.normalization_ranges must contain four frozen pairs"
        )
    try:
        ranges = tuple((float(pair[0]), float(pair[1])) for pair in raw)
    except (IndexError, TypeError, ValueError) as exc:
        raise FinalAnalysisError("normalization ranges must be numeric pairs") from exc
    if any(
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or maximum <= minimum
        for minimum, maximum in ranges
    ):
        raise FinalAnalysisError("normalization ranges must be finite with positive width")
    return ranges


def _reference_from_configuration(configuration: Mapping[str, Any]) -> tuple[float, ...]:
    statistics = _mapping(configuration.get("statistics"), "configuration.statistics")
    hypervolume_spec = _mapping(
        statistics.get("hypervolume_reference"), "statistics.hypervolume_reference"
    )
    raw = hypervolume_spec.get("normalized_maximize_point")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise FinalAnalysisError("a four-dimensional frozen hypervolume reference is required")
    try:
        reference = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise FinalAnalysisError("hypervolume reference must be numeric") from exc
    if not all(math.isfinite(value) for value in reference):
        raise FinalAnalysisError("hypervolume reference must be finite")
    return reference


def _front_ids(
    candidate_ids: Iterable[str], vectors: Mapping[str, Sequence[float]]
) -> tuple[str, ...]:
    ids = tuple(sorted(set(candidate_ids)))
    return tuple(
        candidate_id
        for candidate_id in ids
        if not any(
            strictly_dominates(vectors[other_id], vectors[candidate_id])
            for other_id in ids
            if other_id != candidate_id
        )
    )


@dataclass(frozen=True, slots=True)
class HeldoutRunFront:
    method_id: str
    search_seed: int
    candidate_ids: tuple[str, ...]
    feasible_candidate_ids: tuple[str, ...]
    front_candidate_ids: tuple[str, ...]
    hypervolume: DefinedScalar
    coverage_of_pooled_reference: DefinedScalar
    additive_epsilon: DefinedScalar
    inverted_generational_distance: DefinedScalar

    @property
    def run_id(self) -> str:
        return f"{self.method_id}::seed={self.search_seed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "search_seed": self.search_seed,
            "candidate_ids": list(self.candidate_ids),
            "feasible_candidate_ids": list(self.feasible_candidate_ids),
            "front_candidate_ids": list(self.front_candidate_ids),
            "hypervolume": self.hypervolume.to_dict(),
            "coverage_of_pooled_reference": self.coverage_of_pooled_reference.to_dict(),
            "additive_epsilon": self.additive_epsilon.to_dict(),
            "inverted_generational_distance": self.inverted_generational_distance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HeldoutFrontAnalysis:
    """Per-method/seed held-out fronts against one pooled feasible reference front."""

    defined: bool
    undefined_reason: str | None
    pooled_front_candidate_ids: tuple[str, ...]
    normalized_vectors: tuple[tuple[str, tuple[float, ...]], ...]
    normalization_ranges: tuple[tuple[float, float], ...]
    hypervolume_reference: tuple[float, ...]
    runs: tuple[HeldoutRunFront, ...]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "defined": self.defined,
            "pooled_front_candidate_ids": list(self.pooled_front_candidate_ids),
            "normalized_vectors": {
                candidate_id: list(vector) for candidate_id, vector in self.normalized_vectors
            },
            "normalization_ranges": [list(bounds) for bounds in self.normalization_ranges],
            "hypervolume_reference": list(self.hypervolume_reference),
            "runs": {run.run_id: run.to_dict() for run in self.runs},
        }
        if self.undefined_reason is not None:
            result["reason"] = self.undefined_reason
        return result


def analyze_heldout_fronts(
    summaries: Mapping[str, ObjectiveSummary],
    run_candidates: Mapping[RunKey, Iterable[str]],
    configuration: Mapping[str, Any],
    *,
    constraints: FeasibilityConstraints | None = None,
    feasibility_mode: Literal["uncertainty", "point"] = "uncertainty",
) -> HeldoutFrontAnalysis:
    """Reconstruct feasible held-out method fronts and standard MOO metrics.

    Objective vectors use the final-test point estimates; frozen conservative
    constraints determine feasibility.  Hypervolume of an empty approximation
    is defined as zero. Coverage against a non-empty pooled reference is also
    defined as zero for an empty approximation. Additive epsilon and IGD need a
    non-empty approximation and are therefore explicitly undefined in that case.
    If no run has any feasible point, all reference-front metrics are undefined.
    """

    if not isinstance(summaries, Mapping) or not summaries:
        raise FinalAnalysisError("held-out analysis requires candidate summaries")
    if not isinstance(run_candidates, Mapping) or not run_candidates:
        raise FinalAnalysisError("held-out analysis requires method/seed candidate sets")
    ranges = _normalization_from_configuration(configuration)
    reference = _reference_from_configuration(configuration)
    resolved_constraints = constraints or _constraints_from_configuration(configuration)

    normalized_runs: dict[RunKey, tuple[str, ...]] = {}
    referenced: set[str] = set()
    for key, raw_ids in run_candidates.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not key[0]
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
        ):
            raise FinalAnalysisError("run keys must be (non-empty method id, integer seed)")
        ids = tuple(raw_ids)
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise FinalAnalysisError(f"run {key!r} requires non-empty candidate ids")
        if len(set(ids)) != len(ids):
            raise FinalAnalysisError(f"run {key!r} contains duplicate candidate ids")
        missing = sorted(set(ids) - set(summaries))
        if missing:
            raise FinalAnalysisError(f"run {key!r} lacks summaries for {missing!r}")
        normalized_runs[key] = tuple(sorted(ids))
        referenced.update(ids)
    if any(
        not isinstance(summaries[candidate_id], ObjectiveSummary)
        for candidate_id in referenced
    ):
        raise FinalAnalysisError("summaries must contain ObjectiveSummary values")

    raw_vectors = {
        candidate_id: summaries[candidate_id].point_vector() for candidate_id in referenced
    }
    vectors = normalize_vectors(raw_vectors, ranges)
    feasible_ids = {
        candidate_id
        for candidate_id in referenced
        if feasibility(
            summaries[candidate_id], resolved_constraints, mode=feasibility_mode
        ).feasible
    }
    pooled_ids = _front_ids(feasible_ids, vectors)
    pooled_vectors = nondominated(vectors[candidate_id] for candidate_id in pooled_ids)
    pooled_defined = bool(pooled_vectors)

    runs: list[HeldoutRunFront] = []
    for (method_id, search_seed), candidate_ids in sorted(normalized_runs.items()):
        run_feasible = tuple(
            candidate_id for candidate_id in candidate_ids if candidate_id in feasible_ids
        )
        run_front_ids = _front_ids(run_feasible, vectors)
        run_front = nondominated(vectors[candidate_id] for candidate_id in run_front_ids)
        hv = DefinedScalar.of(hypervolume(run_front, reference))
        if not pooled_defined:
            reason = "no_feasible_candidates_across_runs"
            run_coverage = DefinedScalar.undefined(reason)
            epsilon = DefinedScalar.undefined(reason)
            igd = DefinedScalar.undefined(reason)
        else:
            run_coverage = DefinedScalar.of(coverage(run_front, pooled_vectors))
            if not run_front:
                reason = "run_has_no_feasible_candidates"
                epsilon = DefinedScalar.undefined(reason)
                igd = DefinedScalar.undefined(reason)
            else:
                epsilon = DefinedScalar.of(
                    additive_epsilon_indicator(run_front, pooled_vectors)
                )
                igd = DefinedScalar.of(
                    inverted_generational_distance(run_front, pooled_vectors)
                )
        runs.append(
            HeldoutRunFront(
                method_id=method_id,
                search_seed=search_seed,
                candidate_ids=candidate_ids,
                feasible_candidate_ids=run_feasible,
                front_candidate_ids=run_front_ids,
                hypervolume=hv,
                coverage_of_pooled_reference=run_coverage,
                additive_epsilon=epsilon,
                inverted_generational_distance=igd,
            )
        )
    return HeldoutFrontAnalysis(
        defined=pooled_defined,
        undefined_reason=None if pooled_defined else "no_feasible_candidates_across_runs",
        pooled_front_candidate_ids=pooled_ids,
        normalized_vectors=tuple(sorted(vectors.items())),
        normalization_ranges=ranges,
        hypervolume_reference=reference,
        runs=tuple(runs),
    )


@dataclass(frozen=True, slots=True)
class PairedSeedContrast:
    comparison_method: str
    seeds: tuple[int, ...]
    differences: tuple[float, ...]
    missing_or_undefined_seeds: tuple[int, ...]
    mean_difference: DefinedScalar
    exact_two_sided_p: DefinedScalar
    holm_adjusted_p: DefinedScalar

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison_method": self.comparison_method,
            "seeds": list(self.seeds),
            "differences": list(self.differences),
            "missing_or_undefined_seeds": list(self.missing_or_undefined_seeds),
            "mean_difference": self.mean_difference.to_dict(),
            "exact_two_sided_p": self.exact_two_sided_p.to_dict(),
            "holm_adjusted_p": self.holm_adjusted_p.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PairedHolmAnalysis:
    treatment_method: str
    seeds: tuple[int, ...]
    family_complete: bool
    contrasts: tuple[PairedSeedContrast, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "treatment_method": self.treatment_method,
            "seeds": list(self.seeds),
            "family_complete": self.family_complete,
            "contrasts": {
                contrast.comparison_method: contrast.to_dict()
                for contrast in self.contrasts
            },
        }


def _metric_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalAnalysisError("paired seed metrics must be finite numbers or null")
    result = float(value)
    if not math.isfinite(result):
        raise FinalAnalysisError("paired seed metrics must be finite numbers or null")
    return result


def _exact_two_sided_sign_flip(differences: Sequence[float]) -> float:
    if not differences:
        raise ValueError("sign-flip inference requires paired differences")
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(
            sum(sign * difference for sign, difference in zip(signs, differences, strict=True))
            / len(differences)
        )
        total += 1
        if permuted + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def paired_three_seed_sign_flip_holm(
    run_metric_values: Mapping[RunKey, float | None],
    *,
    treatment_method: str,
    comparison_methods: Iterable[str],
    seeds: Iterable[int],
    higher_is_better: bool = True,
) -> PairedHolmAnalysis:
    """Compute exact paired three-seed sign-flip tests with Holm correction.

    Values are paired by the exact declared search seed.  With only three pairs,
    the two-sided randomization distribution enumerates all eight sign flips.
    A missing value makes that contrast undefined.  If any predeclared contrast
    is incomplete, Holm-adjusted p-values for the whole family are undefined;
    complete contrasts still expose their descriptive mean and raw exact p-value.
    """

    if not isinstance(run_metric_values, Mapping):
        raise FinalAnalysisError("run_metric_values must be a mapping")
    if not isinstance(treatment_method, str) or not treatment_method:
        raise FinalAnalysisError("treatment_method must be non-empty")
    seed_tuple = tuple(seeds)
    if (
        len(seed_tuple) != 3
        or len(set(seed_tuple)) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_tuple)
    ):
        raise FinalAnalysisError("the frozen paired design requires exactly three unique seeds")
    comparison_tuple = tuple(comparison_methods)
    if (
        not comparison_tuple
        or len(set(comparison_tuple)) != len(comparison_tuple)
        or any(not isinstance(method, str) or not method for method in comparison_tuple)
        or treatment_method in comparison_tuple
    ):
        raise FinalAnalysisError("comparison_methods must be unique non-treatment ids")
    if not isinstance(higher_is_better, bool):
        raise FinalAnalysisError("higher_is_better must be a boolean")

    provisional: list[dict[str, Any]] = []
    for comparison in comparison_tuple:
        differences: list[float] = []
        missing: list[int] = []
        for seed in seed_tuple:
            treatment = _metric_value(run_metric_values.get((treatment_method, seed)))
            baseline = _metric_value(run_metric_values.get((comparison, seed)))
            if treatment is None or baseline is None:
                missing.append(seed)
                continue
            differences.append(
                treatment - baseline if higher_is_better else baseline - treatment
            )
        if missing:
            provisional.append(
                {
                    "comparison": comparison,
                    "differences": tuple(differences),
                    "missing": tuple(missing),
                    "mean": DefinedScalar.undefined("missing_or_undefined_seed_metric"),
                    "raw": DefinedScalar.undefined("missing_or_undefined_seed_metric"),
                }
            )
        else:
            provisional.append(
                {
                    "comparison": comparison,
                    "differences": tuple(differences),
                    "missing": (),
                    "mean": DefinedScalar.of(sum(differences) / len(differences)),
                    "raw": DefinedScalar.of(_exact_two_sided_sign_flip(differences)),
                }
            )
    family_complete = all(item["raw"].defined for item in provisional)
    adjusted: dict[str, float] = {}
    if family_complete:
        ordered = sorted(
            provisional,
            key=lambda item: (item["raw"].value, item["comparison"]),
        )
        running = 0.0
        family_size = len(ordered)
        for index, item in enumerate(ordered):
            raw_p = item["raw"].value
            assert raw_p is not None
            running = max(running, min(1.0, (family_size - index) * raw_p))
            adjusted[item["comparison"]] = running

    contrasts: list[PairedSeedContrast] = []
    for item in provisional:
        comparison = item["comparison"]
        if family_complete:
            holm = DefinedScalar.of(adjusted[comparison])
        else:
            holm = DefinedScalar.undefined("predeclared_holm_family_incomplete")
        contrasts.append(
            PairedSeedContrast(
                comparison_method=comparison,
                seeds=seed_tuple,
                differences=item["differences"],
                missing_or_undefined_seeds=item["missing"],
                mean_difference=item["mean"],
                exact_two_sided_p=item["raw"],
                holm_adjusted_p=holm,
            )
        )
    return PairedHolmAnalysis(
        treatment_method=treatment_method,
        seeds=seed_tuple,
        family_complete=family_complete,
        contrasts=tuple(contrasts),
    )


def deployment_candidate_union(
    deployment_payloads: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the deterministic union selected by all defined deployment policies."""

    if isinstance(deployment_payloads, Mapping):
        if "policies" in deployment_payloads:
            payloads: tuple[Mapping[str, Any], ...] = (deployment_payloads,)
        else:
            payloads = tuple(
                _mapping(value, "deployment payload") for value in deployment_payloads.values()
            )
    else:
        payloads = tuple(deployment_payloads)
    if not payloads:
        raise FinalAnalysisError("at least one deployment payload is required")
    selected: set[str] = set()
    for raw_payload in payloads:
        payload = _mapping(raw_payload, "deployment payload")
        primary = payload.get("primary_candidate_id")
        if primary is not None:
            if not isinstance(primary, str) or not primary:
                raise FinalAnalysisError("primary_candidate_id must be a string or null")
            selected.add(primary)
        policies = _mapping(payload.get("policies"), "deployment payload policies")
        for policy_id, raw_selection in policies.items():
            if not isinstance(policy_id, str) or not policy_id:
                raise FinalAnalysisError("deployment policy ids must be non-empty strings")
            selection = _mapping(raw_selection, f"deployment policy {policy_id!r}")
            defined = selection.get("defined")
            if not isinstance(defined, bool):
                raise FinalAnalysisError("deployment policy defined flags must be booleans")
            candidate_id = selection.get("candidate_id")
            if defined:
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise FinalAnalysisError(
                        "defined deployment policies require a candidate_id"
                    )
                selected.add(candidate_id)
            elif candidate_id is not None and (
                not isinstance(candidate_id, str) or not candidate_id
            ):
                raise FinalAnalysisError(
                    "undefined deployment candidate_id must be a string or null"
                )
    return tuple(sorted(selected))


def _id_set(values: Iterable[str], name: str) -> set[str]:
    materialized = tuple(values)
    if any(not isinstance(value, str) or not value for value in materialized):
        raise FinalAnalysisError(f"{name} must contain non-empty string ids")
    return set(materialized)


@dataclass(frozen=True, slots=True)
class ValidatedFalseAdmissionRate:
    """False screen admission rate among candidates that reached full validation."""

    rate: DefinedScalar
    false_admission_count: int
    eligible_admission_count: int
    excluded_unvalidated_ids: tuple[str, ...]
    false_admission_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rate": self.rate.to_dict(),
            "false_admission_count": self.false_admission_count,
            "eligible_admission_count": self.eligible_admission_count,
            "excluded_unvalidated_ids": list(self.excluded_unvalidated_ids),
            "false_admission_ids": list(self.false_admission_ids),
            "denominator_definition": "screen_admitted_intersection_full_validated",
        }


def validated_false_archive_admission_rate(
    screen_admitted_ids: Iterable[str],
    full_validated_ids: Iterable[str],
    full_archive_ids: Iterable[str],
) -> ValidatedFalseAdmissionRate:
    """Compute a false-admission rate with a legally observed denominator.

    A screen admission cannot be labelled false unless it received full
    validation. Therefore the denominator is ``screen admitted AND full
    validated``; screen admissions never promoted to full validation are listed
    but excluded. The full archive must itself be a subset of fully validated
    candidates.
    """

    screened = _id_set(screen_admitted_ids, "screen_admitted_ids")
    validated = _id_set(full_validated_ids, "full_validated_ids")
    archive = _id_set(full_archive_ids, "full_archive_ids")
    invalid_archive = sorted(archive - validated)
    if invalid_archive:
        raise FinalAnalysisError(
            f"full archive contains candidates without full validation: {invalid_archive!r}"
        )
    eligible = screened & validated
    false_ids = eligible - archive
    excluded = screened - validated
    if eligible:
        rate = DefinedScalar.of(len(false_ids) / len(eligible))
    else:
        rate = DefinedScalar.undefined("no_screen_admissions_received_full_validation")
    return ValidatedFalseAdmissionRate(
        rate=rate,
        false_admission_count=len(false_ids),
        eligible_admission_count=len(eligible),
        excluded_unvalidated_ids=tuple(sorted(excluded)),
        false_admission_ids=tuple(sorted(false_ids)),
    )


__all__ = [
    "DefinedScalar",
    "FinalAnalysisError",
    "FinalRecordPartition",
    "HeldoutFrontAnalysis",
    "HeldoutRunFront",
    "PairedHolmAnalysis",
    "PairedSeedContrast",
    "ValidatedFalseAdmissionRate",
    "analyze_heldout_fronts",
    "build_final_objective_summaries",
    "deployment_candidate_union",
    "paired_three_seed_sign_flip_holm",
    "partition_final_records",
    "resolve_final_execution_groups",
    "resolve_final_target_roles",
    "validated_false_archive_admission_rate",
]
