"""Configuration-driven search and final-evaluation orchestration.

The runner is provider-neutral.  Network safety is enforced by ``PairedEvaluator``
for every request, while this module owns deterministic candidate generation,
paired screen/full evaluation, method selection, budgets, and resumable outputs.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import inspect
import json
import locale
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from .ablations import AblationRegistry
from .archive import ArchiveEntry, ParetoArchive
from .baselines import (
    AccuracyOnlyPlugin,
    BaselinePlugin,
    MOCHAPlugin,
    PluginRegistry,
    ScoredCandidate,
)
from .config import ExperimentManifest
from .costs import (
    generation_usage_by_provider,
    reconcile_provider_costs,
    usage_by_provider,
)
from .deployment import DeploymentCandidate, select_deployment_candidate
from .evaluation import (
    EvaluationCandidate,
    EvaluationRecord,
    Harness,
    PairedEvaluator,
    TargetSpec,
    TaskSeedBlock,
    expected_evaluation_matrix,
    validate_evaluation_matrix,
)
from .final_analysis import (
    FinalAnalysisError,
    FinalRecordPartition,
    analyze_heldout_fronts,
    build_final_objective_summaries,
    deployment_candidate_union,
    paired_three_seed_sign_flip_holm,
    partition_final_records,
    resolve_final_execution_groups,
    resolve_final_target_roles,
    validated_false_archive_admission_rate,
)
from .failures import FailureEvent
from .materialize import MaterializationError, MaterializationStore, Materializer
from .metrics import (
    additive_epsilon_indicator,
    coverage,
    hypervolume,
    inverted_generational_distance,
    nondominated,
)
from .models import Patch, SkillVersion, TraceEvidence, canonical_json
from .objectives import (
    DEFAULT_ACTIVE_OBJECTIVES,
    FeasibilityConstraints,
    feasibility,
    normalize_active_objectives,
)
from .providers import NetworkPolicy, Provider
from .proposer import (
    ArchiveConditioner,
    EvidenceBundle,
    MutationProposer,
    MutationRequest,
    ObjectiveDirection,
)
from .search_strategies import (
    AdapterBackedBinarySubsetController,
    BinarySubsetBayesianAdapter,
    BernoulliUniqueStream,
    CommonCandidateStream,
    EvoTopKController,
    ExternalOptimizerRequired,
    MOCHAController,
    NSGAIIController,
    SearchSpaceExhausted,
    SearchStrategyError,
    SubsetSearchController,
    make_binary_subset_controller,
    restore_search_controller,
)
from .statistics import ObjectiveSummary, PairedObservation, PointObjectives, paired_bootstrap
from .storage import JsonlResultStore, ResultCache, StorageError


RunStage = Literal["smoke", "search", "final", "all"]


class ExperimentRunError(RuntimeError):
    """Raised before or during a configured experiment orchestration failure."""


@dataclass(frozen=True, slots=True)
class PhaseRuntime:
    targets: tuple[TargetSpec, ...]
    blocks: tuple[TaskSeedBlock, ...]
    harnesses: Mapping[str, Harness]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "harnesses", dict(self.harnesses))
        if not self.targets or not self.blocks:
            raise ValueError("a runtime phase requires targets and task-seed blocks")
        missing = {target.harness_id for target in self.targets} - set(self.harnesses)
        if missing:
            raise ValueError(f"runtime phase is missing harnesses: {sorted(missing)}")


@dataclass(slots=True)
class ExperimentRuntime:
    base: SkillVersion
    patches: tuple[Patch, ...]
    evidence: Mapping[str, TraceEvidence]
    providers: Mapping[str, Provider]
    phases: Mapping[str, PhaseRuntime]
    proposer_factory: (
        Callable[..., MutationProposer] | None
    ) = None
    binary_optimizer_factory: (
        Callable[
            [tuple[str, ...], int, Mapping[str, Any]],
            BinarySubsetBayesianAdapter,
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        self.patches = tuple(self.patches)
        self.evidence = dict(self.evidence)
        self.providers = dict(self.providers)
        self.phases = dict(self.phases)
        if not self.patches:
            raise ValueError("configured search requires a non-empty patch pool")
        missing = {
            target.provider_id
            for phase in self.phases.values()
            for target in phase.targets
        } - set(self.providers)
        if missing:
            raise ValueError(f"runtime is missing providers: {sorted(missing)}")
        if self.proposer_factory is not None and not callable(self.proposer_factory):
            raise ValueError("proposer_factory must be callable or null")
        if self.binary_optimizer_factory is not None and not callable(
            self.binary_optimizer_factory
        ):
            raise ValueError("binary_optimizer_factory must be callable or null")


@dataclass(frozen=True, slots=True)
class MethodRunSummary:
    method_id: str
    search_seed: int
    output_directory: Path
    proposed_candidates: int
    screened_candidates: int
    promoted_candidates: int
    selected_candidate_ids: tuple[str, ...]
    plugin_native_selected_candidate_ids: tuple[str, ...]
    logical_task_executions: int
    physical_provider_executions: int
    budget_limit: int | None
    budget_complete: bool
    rejected_materializations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "search_seed": self.search_seed,
            "output_directory": str(self.output_directory),
            "proposed_candidates": self.proposed_candidates,
            "screened_candidates": self.screened_candidates,
            "promoted_candidates": self.promoted_candidates,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "plugin_native_selected_candidate_ids": list(
                self.plugin_native_selected_candidate_ids
            ),
            "final_frozen_candidate_ids": list(self.selected_candidate_ids),
            "logical_task_executions": self.logical_task_executions,
            "physical_provider_executions": self.physical_provider_executions,
            "budget_limit": self.budget_limit,
            "budget_complete": self.budget_complete,
            "rejected_materializations": self.rejected_materializations,
        }


@dataclass(frozen=True, slots=True)
class ConfiguredRunSummary:
    experiment_id: str
    stage: RunStage
    output_directory: Path
    method_runs: tuple[MethodRunSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "output_directory": str(self.output_directory),
            "method_runs": [run.to_dict() for run in self.method_runs],
            "method_run_count": len(self.method_runs),
        }


@dataclass(frozen=True, slots=True)
class FinalRunSummary:
    experiment_id: str
    output_directory: Path
    candidate_count: int
    target_count: int
    task_outcomes: int
    logical_task_executions: int
    physical_provider_executions: int
    budget_limit: int | None
    budget_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "stage": "final",
            "output_directory": str(self.output_directory),
            "candidate_count": self.candidate_count,
            "target_count": self.target_count,
            "task_outcomes": self.task_outcomes,
            "logical_task_executions": self.logical_task_executions,
            "physical_provider_executions": self.physical_provider_executions,
            "budget_limit": self.budget_limit,
            "budget_complete": self.budget_complete,
        }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_failure_events(path: Path) -> tuple[FailureEvent, ...]:
    """Read and strictly validate a sanitized failure-event ledger."""

    if not path.exists():
        return ()
    events: list[FailureEvent] = []
    observed: dict[str, Mapping[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExperimentRunError(f"cannot read failure-event artifact: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError("event must be a JSON object")
            event = FailureEvent.from_dict(raw)
            serialized = event.to_dict()
            if dict(raw) != serialized:
                raise ValueError("event contains missing or unexpected fields")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExperimentRunError(
                f"invalid failure event at {path}:{line_number}: {exc}"
            ) from exc
        existing = observed.get(event.failure_id)
        if existing is not None:
            if canonical_json(existing) != canonical_json(serialized):
                raise ExperimentRunError(
                    f"conflicting failure_id in failure-event artifact: {event.failure_id}"
                )
            continue
        observed[event.failure_id] = serialized
        events.append(event)
    return tuple(events)


def _replace_failure_events(path: Path, events: Iterable[FailureEvent]) -> None:
    """Atomically replace a ledger with first-seen, failure-id-deduplicated events."""

    ordered: list[FailureEvent] = []
    observed: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if not isinstance(event, FailureEvent):
            raise ExperimentRunError("failure-event ledger accepts only FailureEvent values")
        serialized = event.to_dict()
        existing = observed.get(event.failure_id)
        if existing is not None:
            if canonical_json(existing) != canonical_json(serialized):
                raise ExperimentRunError(
                    f"conflicting failure_id while writing ledger: {event.failure_id}"
                )
            continue
        observed[event.failure_id] = serialized
        ordered.append(event)
    _atomic_jsonl(path, (event.to_dict() for event in ordered))


def _persist_failure_events(path: Path, events: Iterable[FailureEvent]) -> None:
    """Append sanitized events transactionally while making retries resume-idempotent."""

    _replace_failure_events(path, (*_read_failure_events(path), *tuple(events)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _git_state(code_root: Path) -> dict[str, Any]:
    def invoke(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=code_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
            shell=False,
        )
        return completed.stdout.strip()

    try:
        head = invoke("rev-parse", "HEAD").decode("ascii")
        status = invoke("status", "--porcelain=v1", "-z")
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return {"available": False, "head": None, "dirty": None}
    return {
        "available": True,
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _run_metadata(
    manifest: ExperimentManifest,
    *,
    stage: str,
    started_at_utc: str,
    complete: bool,
) -> dict[str, Any]:
    from . import __version__

    captured_environment = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")
    }
    return {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "declared_experiment_id": manifest.data["experiment"]["id"],
        "profile": manifest.profile,
        "stage": stage,
        "complete": complete,
        "started_at_utc": started_at_utc,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "package_version": __version__,
        "implementation_digest": manifest.implementation_digest,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "locale": locale.getlocale(),
        "timezone": str(dt.datetime.now().astimezone().tzinfo),
        "command": list(sys.argv),
        "environment_allowlist": captured_environment,
        "git": _git_state(manifest.code_root),
        "environment_dumped": False,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentRunError(f"cannot read required JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ExperimentRunError(f"required JSON artifact is not an object: {path}")
    return value


def _exact_sign_flip_pvalue(differences: list[float]) -> float:
    if not differences:
        raise ValueError("paired sign-flip test requires differences")
    observed = abs(sum(differences) / len(differences))
    extreme = 0
    total = 1 << len(differences)
    for mask in range(total):
        permuted = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(differences)
        ) / len(differences)
        extreme += abs(permuted) >= observed - 1e-15
    return extreme / total


def _root_frontier_analysis(
    configuration: Mapping[str, Any],
    metric_payloads: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    ranges = _normalization_ranges(configuration, strict=False)
    run_vectors: dict[tuple[str, int], list[tuple[float, ...]]] = {}
    run_hypervolumes: dict[tuple[str, int], float] = {}
    constraints = _constraints(configuration)
    for run_identity, payload in metric_payloads.items():
        full = payload.get("full")
        analysis = payload.get("frontier_analysis")
        if not isinstance(full, Mapping) or not isinstance(analysis, Mapping):
            raise ExperimentRunError("method metrics lack full/frontier analysis")
        hypervolume_value = analysis.get("feasible_hypervolume")
        if isinstance(hypervolume_value, (int, float)) and not isinstance(
            hypervolume_value, bool
        ):
            run_hypervolumes[run_identity] = float(hypervolume_value)
        vectors: list[tuple[float, ...]] = []
        if ranges is not None:
            for raw_summary in full.values():
                if not isinstance(raw_summary, Mapping):
                    raise ExperimentRunError("method full summary is malformed")
                summary = ObjectiveSummary.from_dict(raw_summary)
                if not feasibility(summary, constraints).feasible:
                    continue
                normalized = []
                for value, (minimum, maximum) in zip(
                    summary.point_objectives().maximize_vector(),
                    ranges,
                    strict=True,
                ):
                    normalized.append(
                        0.0
                        if maximum <= minimum
                        else (value - minimum) / (maximum - minimum)
                    )
                vectors.append(tuple(normalized))
        run_vectors[run_identity] = vectors
    pooled_front = (
        nondominated(vector for vectors in run_vectors.values() for vector in vectors)
        if ranges is not None
        else ()
    )
    per_run: dict[str, Any] = {}
    for (method_id, seed), vectors in sorted(run_vectors.items()):
        key = f"{method_id}::seed-{seed}"
        if not pooled_front or not vectors:
            per_run[key] = {
                "defined": False,
                "feasible_hypervolume": run_hypervolumes.get((method_id, seed)),
                "coverage_of_pooled_front": None,
                "additive_epsilon_indicator": None,
                "igd_to_pooled_front": None,
            }
            continue
        run_front = nondominated(vectors)
        per_run[key] = {
            "defined": True,
            "feasible_hypervolume": run_hypervolumes.get((method_id, seed)),
            "coverage_of_pooled_front": coverage(run_front, pooled_front),
            "additive_epsilon_indicator": additive_epsilon_indicator(
                run_front, pooled_front
            ),
            "igd_to_pooled_front": inverted_generational_distance(
                run_front, pooled_front
            ),
            "front_size": len(run_front),
        }

    comparison = configuration["statistics"]["heldout_comparisons"]
    baseline_ids = list(comparison.get("matched_run_ids", []))
    raw_contrasts: list[dict[str, Any]] = []
    pareto_by_seed = {
        seed: value
        for (method_id, seed), value in run_hypervolumes.items()
        if method_id == "paretoskill"
    }
    for baseline_id in baseline_ids:
        baseline_by_seed = {
            seed: value
            for (method_id, seed), value in run_hypervolumes.items()
            if method_id == baseline_id
        }
        shared = sorted(set(pareto_by_seed) & set(baseline_by_seed))
        required_seed_count = len(configuration["task_seed_blocks"]["search_seeds"])
        if len(shared) != required_seed_count:
            raw_contrasts.append(
                {
                    "baseline_method_id": baseline_id,
                    "defined": False,
                    "shared_seeds": shared,
                    "reason": "incomplete paired search-seed hypervolume values",
                }
            )
            continue
        differences = [pareto_by_seed[seed] - baseline_by_seed[seed] for seed in shared]
        raw_contrasts.append(
            {
                "baseline_method_id": baseline_id,
                "defined": True,
                "shared_seeds": shared,
                "paired_differences": differences,
                "mean_difference": sum(differences) / len(differences),
                "raw_p_value": _exact_sign_flip_pvalue(differences),
            }
        )
    defined = [contrast for contrast in raw_contrasts if contrast.get("defined")]
    previous_adjusted = 0.0
    for rank, contrast in enumerate(
        sorted(defined, key=lambda item: (item["raw_p_value"], item["baseline_method_id"])),
        start=1,
    ):
        adjusted = min(
            1.0,
            max(
                previous_adjusted,
                (len(defined) - rank + 1) * float(contrast["raw_p_value"]),
            ),
        )
        contrast["holm_adjusted_p_value"] = adjusted
        previous_adjusted = adjusted
    return {
        "normalization_defined": ranges is not None,
        "pooled_empirical_front": [list(vector) for vector in pooled_front],
        "per_run": per_run,
        "paired_holm_search_seed_contrasts": raw_contrasts,
        "primary_endpoint": comparison.get("primary_endpoint"),
        "paired_design": True,
    }


def _write_search_output_contract(
    root: Path,
    manifest: ExperimentManifest,
    result: ConfiguredRunSummary,
) -> None:
    contract_runs = {
        (run.method_id, run.search_seed): run for run in result.method_runs
    }
    for state_path in sorted((root / "methods").glob("*/seed-*/run_state.json")):
        state = _read_json_object(state_path)
        summary = state.get("summary")
        if (
            state.get("experiment_id") != manifest.experiment_id
            or state.get("execution_complete") is not True
            or not isinstance(summary, Mapping)
        ):
            continue
        try:
            recovered = MethodRunSummary(
                method_id=str(summary["method_id"]),
                search_seed=int(summary["search_seed"]),
                output_directory=state_path.parent,
                proposed_candidates=int(summary["proposed_candidates"]),
                screened_candidates=int(summary["screened_candidates"]),
                promoted_candidates=int(summary["promoted_candidates"]),
                selected_candidate_ids=tuple(summary["selected_candidate_ids"]),
                plugin_native_selected_candidate_ids=tuple(
                    summary.get(
                        "plugin_native_selected_candidate_ids",
                        summary["selected_candidate_ids"],
                    )
                ),
                logical_task_executions=int(summary["logical_task_executions"]),
                physical_provider_executions=int(
                    summary["physical_provider_executions"]
                ),
                budget_limit=(
                    None
                    if summary.get("budget_limit") is None
                    else int(summary["budget_limit"])
                ),
                budget_complete=bool(summary["budget_complete"]),
                rejected_materializations=int(summary["rejected_materializations"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentRunError(f"malformed method run summary: {state_path}") from exc
        contract_runs[(recovered.method_id, recovered.search_seed)] = recovered
    all_runs = tuple(
        contract_runs[key] for key in sorted(contract_runs, key=lambda item: (item[0], item[1]))
    )
    observed_matrix = {(run.method_id, run.search_seed) for run in all_runs}
    expected_matrix = _expected_search_run_matrix(manifest.data)
    matrix_complete = observed_matrix == expected_matrix
    records: list[EvaluationRecord] = []
    candidates: dict[str, Mapping[str, Any]] = {}
    lineages: dict[str, Mapping[str, Any]] = {}
    archive_indexes: list[dict[str, Any]] = []
    front_indexes: list[dict[str, Any]] = []
    metric_indexes: list[dict[str, Any]] = []
    metric_payloads: dict[tuple[str, int], Mapping[str, Any]] = {}
    deployment_indexes: list[dict[str, Any]] = []
    proposal_input_tokens = 0
    proposal_output_tokens = 0
    proposal_physical_calls = 0
    cache_keys: set[str] = set()
    failure_events: list[FailureEvent] = []
    for run in all_runs:
        directory = run.output_directory
        failure_path = directory / "failure_events.jsonl"
        if not failure_path.is_file():
            raise ExperimentRunError(
                f"completed method run is missing failure-event artifact: {failure_path}"
            )
        failure_events.extend(_read_failure_events(failure_path))
        records.extend(
            JsonlResultStore(directory / "screen_task_outcomes.jsonl").read_all()
        )
        records.extend(
            JsonlResultStore(directory / "full_task_outcomes.jsonl").read_all()
        )
        for line_number, line in enumerate(
            (directory / "candidates.jsonl").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                version_id = row["lineage"]["version_id"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ExperimentRunError(
                    f"invalid candidate artifact at {directory}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping) or not isinstance(version_id, str):
                raise ExperimentRunError("candidate artifact has invalid identity")
            existing = candidates.get(version_id)
            if existing is not None and canonical_json(existing) != canonical_json(row):
                raise ExperimentRunError(f"conflicting candidate version: {version_id}")
            candidates[version_id] = row
            lineage = row["lineage"]
            assert isinstance(lineage, Mapping)
            lineages[version_id] = lineage
        for name, destination in (
            ("archive.json", archive_indexes),
            ("scientific_front.json", front_indexes),
        ):
            path = directory / name
            if path.is_file():
                destination.append(
                    {
                        "method_id": run.method_id,
                        "search_seed": run.search_seed,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _sha256_file(path),
                    }
                )
        metrics_path = directory / "metrics.json"
        metric_payloads[(run.method_id, run.search_seed)] = _read_json_object(
            metrics_path
        )
        metric_indexes.append(
            {
                "method_id": run.method_id,
                "search_seed": run.search_seed,
                "path": metrics_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(metrics_path),
            }
        )
        deployment_path = directory / "deployment_selection.json"
        deployment_payload = _read_json_object(deployment_path)
        deployment_indexes.append(
            {
                "method_id": run.method_id,
                "search_seed": run.search_seed,
                "path": deployment_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(deployment_path),
                "primary_policy": deployment_payload.get("primary_policy"),
                "primary_candidate_id": deployment_payload.get(
                    "primary_candidate_id"
                ),
            }
        )
        proposal_path = directory / "proposal_events.jsonl"
        if proposal_path.is_file():
            for line in proposal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                proposal_input_tokens += int(event.get("input_tokens", 0))
                proposal_output_tokens += int(event.get("output_tokens", 0))
                proposal_physical_calls += event.get("cache_hit") is not True
        state = _read_json_object(directory / "run_state.json")
        declared_keys = state.get("cache_keys", [])
        if not isinstance(declared_keys, list) or any(
            not isinstance(value, str) for value in declared_keys
        ):
            raise ExperimentRunError("method run_state cache_keys is malformed")
        cache_keys.update(declared_keys)

    merged_records = _merge_records(records)
    _replace_failure_events(root / "failure_events.jsonl", failure_events)
    JsonlResultStore(root / "task_outcomes.jsonl").replace(merged_records)
    _atomic_jsonl(
        root / "candidates.jsonl",
        [candidates[version_id] for version_id in sorted(candidates)],
    )
    _atomic_jsonl(
        root / "lineage.jsonl",
        [lineages[version_id] for version_id in sorted(lineages)],
    )
    _atomic_json(
        root / "archive.json",
        {"schema_version": 1, "kind": "archive_index", "runs": archive_indexes},
    )
    _atomic_json(
        root / "scientific_front.json",
        {"schema_version": 1, "kind": "front_index", "runs": front_indexes},
    )
    _atomic_json(
        root / "metrics.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "stage": result.stage,
            "method_runs": [run.to_dict() for run in all_runs],
            "metric_artifacts": metric_indexes,
            "frontier_comparison": _root_frontier_analysis(
                manifest.data, metric_payloads
            ),
        },
    )
    _atomic_json(
        root / "deployment_selection.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "data_role": "validation_only",
            "runs": deployment_indexes,
        },
    )
    logical_candidate_rows = [record for record in merged_records if not record.is_base]
    experiment_cache = ResultCache(root / "cache")
    cached_records = [
        record
        for cache_key in sorted(experiment_cache.keys())
        if (record := experiment_cache.get(cache_key)) is not None
    ]
    cached_proposals: list[Mapping[str, Any]] = []
    for cache_path in sorted(
        (root / "methods").glob("*/seed-*/proposal_cache/*/*.json")
    ):
        payload = _read_json_object(cache_path)
        generation = payload.get("generation")
        if not isinstance(generation, Mapping):
            raise ExperimentRunError(f"proposal cache is malformed: {cache_path}")
        cached_proposals.append(generation)
    try:
        proposer_model_id = manifest.data["proposer"]["model"]
        proposer_provider_id = manifest.data["models"][proposer_model_id]["provider"]
        evaluation_costs = reconcile_provider_costs(
            usage_by_provider(cached_records, manifest.data),
            manifest.data["providers"],
        )
        proposal_costs = reconcile_provider_costs(
            generation_usage_by_provider(
                cached_proposals,
                default_provider_id=proposer_provider_id,
            ),
            manifest.data["providers"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunError(f"cannot reconcile provider prices: {exc}") from exc
    _atomic_json(
        root / "token_accounting.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "logical_task_executions": sum(
                run.logical_task_executions for run in all_runs
            ),
            "logical_evaluation_tokens": sum(
                record.result.total_tokens for record in logical_candidate_rows
            ),
            "physical_evaluation_provider_calls": len(cached_records),
            "physical_evaluation_tokens": sum(
                record.result.total_tokens for record in cached_records
            ),
            "proposal_provider_calls": len(cached_proposals),
            "proposal_input_tokens": sum(
                int(generation.get("input_tokens", 0))
                for generation in cached_proposals
            ),
            "proposal_output_tokens": sum(
                int(generation.get("output_tokens", 0))
                for generation in cached_proposals
            ),
            "evaluation_cost_reconciliation": evaluation_costs,
            "proposal_cost_reconciliation": proposal_costs,
            "current_invocation_proposal_event_tokens": {
                "input": proposal_input_tokens,
                "output": proposal_output_tokens,
                "physical_calls": proposal_physical_calls,
            },
            "cache_reuse_reported_separately": True,
        },
    )
    _atomic_json(
        root / "checkpoint.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "implementation_digest": manifest.implementation_digest,
            "stage": result.stage,
            "invocation_complete": True,
            "execution_complete": result.stage == "smoke" or matrix_complete,
            "budget_complete": (
                all(run.budget_complete for run in all_runs)
                and (result.stage == "smoke" or matrix_complete)
            ),
            "method_seed_matrix_complete": matrix_complete,
            "expected_method_seed_matrix": [
                {"method_id": method_id, "search_seed": seed}
                for method_id, seed in sorted(expected_matrix)
            ],
            "observed_method_seed_matrix": [
                {"method_id": method_id, "search_seed": seed}
                for method_id, seed in sorted(observed_matrix)
            ],
            "method_runs": [run.to_dict() for run in all_runs],
            "cache_keys": sorted(cache_keys),
        },
    )


def _update_final_output_contract(
    root: Path,
    manifest: ExperimentManifest,
    *,
    records: list[EvaluationRecord],
    summary: FinalRunSummary,
    started_at_utc: str,
) -> None:
    task_path = root / "task_outcomes.jsonl"
    if not task_path.is_file():
        raise ExperimentRunError("final stage requires the frozen search output contract")
    final_failure_path = root / "final" / "failure_events.jsonl"
    if not final_failure_path.is_file():
        raise ExperimentRunError("final stage lacks its failure-event artifact")
    _persist_failure_events(
        root / "failure_events.jsonl",
        _read_failure_events(final_failure_path),
    )
    search_records = JsonlResultStore(task_path).read_all()
    JsonlResultStore(task_path).replace(_merge_records(search_records, records))

    final_metrics_path = root / "final" / "metrics.json"
    final_metrics_payload = _read_json_object(final_metrics_path)
    primary_analysis = final_metrics_payload.get("primary_analysis")
    if not isinstance(primary_analysis, Mapping):
        raise ExperimentRunError("final metrics lack primary held-out analysis")
    heldout_fronts = primary_analysis.get("heldout_fronts")
    paired_holm = primary_analysis.get("paired_three_seed_sign_flip_holm")
    partition = primary_analysis.get("record_partition")
    if not all(
        isinstance(value, Mapping)
        for value in (heldout_fronts, paired_holm, partition)
    ):
        raise ExperimentRunError("final primary analysis summary is malformed")
    assert isinstance(heldout_fronts, Mapping)
    assert isinstance(paired_holm, Mapping)
    assert isinstance(partition, Mapping)
    analysis_summary = {
        "defined": primary_analysis.get("defined") is True,
        "data_role": primary_analysis.get("data_role"),
        "id_record_count": partition.get("id_record_count"),
        "transfer_record_count": partition.get("transfer_record_count"),
        "diagnostic_record_count": partition.get("diagnostic_record_count"),
        "dropped_overlap_count": partition.get("dropped_overlap_count"),
        "heldout_front_defined": heldout_fronts.get("defined") is True,
        "pooled_front_candidate_ids": heldout_fronts.get(
            "pooled_front_candidate_ids", []
        ),
        "method_seed_run_count": len(heldout_fronts.get("runs", {})),
        "paired_holm_family_complete": paired_holm.get("family_complete") is True,
        "primary_endpoint": primary_analysis.get("primary_endpoint"),
    }
    metrics = _read_json_object(root / "metrics.json")
    metrics["final"] = {
        "path": final_metrics_path.relative_to(root).as_posix(),
        "sha256": _sha256_file(final_metrics_path),
        "summary": summary.to_dict(),
        "analysis_summary": analysis_summary,
    }
    metrics["heldout_frontier_comparison"] = {
        "data_role": "heldout_final_only",
        "path": final_metrics_path.relative_to(root).as_posix(),
        **analysis_summary,
    }
    _atomic_json(root / "metrics.json", metrics)

    accounting = _read_json_object(root / "token_accounting.json")
    final_candidate_records = [record for record in records if not record.is_base]
    previous_final = accounting.get("final")
    previous_physical = (
        int(previous_final.get("physical_provider_calls", 0))
        if isinstance(previous_final, Mapping)
        else 0
    )
    accounting["final"] = {
        "logical_task_executions": summary.logical_task_executions,
        "logical_evaluation_tokens": sum(
            record.result.total_tokens for record in final_candidate_records
        ),
        "physical_provider_calls": max(
            previous_physical, summary.physical_provider_executions
        ),
        "budget_limit": summary.budget_limit,
        "budget_complete": summary.budget_complete,
    }
    accounting["total_logical_task_executions"] = int(
        accounting.get("logical_task_executions", 0)
    ) + summary.logical_task_executions
    accounting["total_logical_evaluation_tokens"] = int(
        accounting.get("logical_evaluation_tokens", 0)
    ) + sum(record.result.total_tokens for record in final_candidate_records)
    experiment_cache = ResultCache(root / "cache")
    all_cached_records = [
        record
        for cache_key in sorted(experiment_cache.keys())
        if (record := experiment_cache.get(cache_key)) is not None
    ]
    accounting["physical_evaluation_provider_calls"] = len(all_cached_records)
    accounting["physical_evaluation_tokens"] = sum(
        record.result.total_tokens for record in all_cached_records
    )
    try:
        accounting["evaluation_cost_reconciliation"] = reconcile_provider_costs(
            usage_by_provider(all_cached_records, manifest.data),
            manifest.data["providers"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunError(f"cannot reconcile final provider prices: {exc}") from exc
    _atomic_json(root / "token_accounting.json", accounting)

    checkpoint = _read_json_object(root / "checkpoint.json")
    existing_cache_keys = checkpoint.get("cache_keys", [])
    if not isinstance(existing_cache_keys, list):
        raise ExperimentRunError("root checkpoint cache_keys is malformed")
    checkpoint.update(
        stage="final",
        execution_complete=True,
        budget_complete=(
            checkpoint.get("budget_complete") is True and summary.budget_complete
        ),
        final=summary.to_dict(),
        final_candidate_manifest_sha256=_read_json_object(
            root / "final" / "candidate_manifest.json"
        )["sha256"],
        cache_keys=sorted(
            set(existing_cache_keys) | {record.cache_key for record in records}
        ),
    )
    _atomic_json(root / "checkpoint.json", checkpoint)
    _atomic_json(
        root / "run_metadata.json",
        _run_metadata(
            manifest,
            stage="final",
            started_at_utc=started_at_utc,
            complete=True,
        ),
    )


def _compatible(target: TargetSpec, block: TaskSeedBlock) -> bool:
    task = block.task
    return (
        task.domain_id == target.domain_id
        and (target.split_id is None or task.split_id == target.split_id)
        and (
            target.objective_role is None
            or task.objective_role == target.objective_role
        )
        and (target.task_group == "*" or task.group_id == target.task_group)
    )


def _matrix_size(targets: Iterable[TargetSpec], blocks: Iterable[TaskSeedBlock]) -> int:
    block_list = tuple(blocks)
    return sum(sum(_compatible(target, block) for block in block_list) for target in targets)


def _validate_search_budget_matrix(
    configuration: Mapping[str, Any], phase: PhaseRuntime, *, smoke: bool
) -> None:
    if smoke:
        return
    budgets = configuration["budgets"]
    full = budgets["full"]
    screen = budgets["screen"]
    target_ids = {target.target_id for target in phase.targets}
    expected_targets = set(full["target_ids"])
    if target_ids != expected_targets or set(screen["target_ids"]) != expected_targets:
        raise ExperimentRunError(
            "runtime search targets do not match frozen screen/full target_ids"
        )
    expected_per_target = int(full["tasks_per_target"]) * int(
        full["execution_seeds_per_task"]
    )
    counts = {
        target.target_id: sum(_compatible(target, block) for block in phase.blocks)
        for target in phase.targets
    }
    wrong = {
        target_id: count
        for target_id, count in counts.items()
        if count != expected_per_target
    }
    if wrong:
        raise ExperimentRunError(
            "runtime full matrix violates frozen per-target cardinality: "
            f"expected={expected_per_target}, observed={wrong}"
        )
    if _matrix_size(phase.targets, phase.blocks) != int(full["task_executions"]):
        raise ExperimentRunError("runtime full matrix does not equal the frozen full budget")
    screen_blocks = _select_screen_blocks(
        phase,
        per_target=int(screen["tasks_per_target"]),
        seeds={int(seed) for seed in screen["selected_execution_seeds"]},
        salt=str(screen["selection_salt"]),
    )
    if _matrix_size(phase.targets, screen_blocks) != int(screen["task_executions"]):
        raise ExperimentRunError("runtime screen matrix does not equal the frozen screen budget")


def _select_screen_blocks(
    phase: PhaseRuntime,
    *,
    per_target: int,
    seeds: set[int],
    salt: str,
) -> tuple[TaskSeedBlock, ...]:
    selected: dict[tuple[str, str, int, str], TaskSeedBlock] = {}
    for target in phase.targets:
        compatible = [
            block
            for block in phase.blocks
            if block.seed in seeds and _compatible(target, block)
        ]
        ranked = sorted(
            compatible,
            key=lambda block: (
                hashlib.sha256(
                    f"{salt}\0{target.target_id}\0{block.block_id}\0{block.seed}".encode()
                ).hexdigest(),
                block.block_id,
                block.seed,
            ),
        )
        chosen = ranked[:per_target]
        if len(chosen) != per_target:
            raise ExperimentRunError(
                f"target {target.target_id!r} has {len(chosen)} screen blocks; "
                f"{per_target} required"
            )
        for block in chosen:
            key = (
                block.task.task_id,
                str(block.task.objective_role),
                block.seed,
                str(block.task.split_id),
            )
            selected[key] = block
    return tuple(sorted(selected.values(), key=lambda block: (block.block_id, block.seed)))


def _observations(
    records: Iterable[EvaluationRecord], candidate_id: str
) -> tuple[PairedObservation, ...]:
    rows = tuple(records)
    base_by_block = {record.block_key: record for record in rows if record.is_base}
    candidate_rows = [record for record in rows if record.candidate_id == candidate_id]
    observations: list[PairedObservation] = []
    for record in candidate_rows:
        try:
            base = base_by_block[record.block_key]
        except KeyError as exc:
            raise ExperimentRunError(
                f"candidate {candidate_id!r} has no paired base row for {record.block_key}"
            ) from exc
        observations.append(
            PairedObservation(
                task_id=record.task_id,
                seed=record.seed,
                split=record.split,
                target=record.target_id,
                group=(record.transfer_group or record.target_id)
                if record.split == "transfer"
                else None,
                candidate_correct=record.result.correct,
                base_correct=base.result.correct,
                input_tokens=record.result.input_tokens,
                output_tokens=record.result.output_tokens,
            )
        )
    if not observations:
        raise ExperimentRunError(f"candidate {candidate_id!r} has no evaluation rows")
    return tuple(observations)


def _constraints(configuration: Mapping[str, Any]) -> FeasibilityConstraints:
    raw = configuration["constraints"]
    return FeasibilityConstraints.from_paired_epsilon(
        epsilon=float(raw["id_accuracy_floor"]["epsilon"]),
        token_budget=float(raw["token_budget"]["budget"]),
        enabled=raw.get("enabled") is not False,
    )


def _summaries(
    records: Iterable[EvaluationRecord],
    candidate_ids: Iterable[str],
    *,
    targets: tuple[TargetSpec, ...],
    configuration: Mapping[str, Any],
    smoke: bool,
) -> dict[str, ObjectiveSummary]:
    statistics = configuration["statistics"]
    transfer_groups = {
        target.transfer_group or target.target_id
        for target in targets
        if target.objective_role == "transfer"
    }
    replicates = int(statistics["bootstrap_replicates"])
    if smoke:
        replicates = min(replicates, 200)
    return {
        candidate_id: paired_bootstrap(
            _observations(records, candidate_id),
            confidence_level=float(statistics["confidence_level"]),
            replicates=replicates,
            seed=int(configuration["task_seed_blocks"]["bootstrap_seed"]),
            expected_transfer_groups=transfer_groups,
            min_effective_blocks=int(
                statistics.get("minimum_effective_blocks_for_archive", 2)
            ),
            token_cost_upper_bound=float(
                configuration["constraints"]["token_budget"]["budget"]
            ),
        )
        for candidate_id in candidate_ids
    }


def _hard_easy_rates(
    records: Iterable[EvaluationRecord], candidate_ids: Iterable[str], *, count: int
) -> tuple[dict[str, tuple[float, float]], int]:
    rows = tuple(records)
    base_id_rows = [row for row in rows if row.is_base and row.split == "id"]
    if not base_id_rows:
        raise ExperimentRunError("Ctx2Skill selection requires ID rows")
    by_task: dict[str, list[bool]] = {}
    for row in base_id_rows:
        by_task.setdefault(row.task_id, []).append(row.result.correct)
    ranked = sorted(
        by_task,
        key=lambda task_id: (
            sum(by_task[task_id]) / len(by_task[task_id]),
            task_id,
        ),
    )
    probe_count = min(count, len(ranked) // 2)
    if probe_count < 1:
        raise ExperimentRunError("Ctx2Skill selection requires two disjoint probe tasks")
    hard = set(ranked[:probe_count])
    easy = set(ranked[-probe_count:])
    rates: dict[str, tuple[float, float]] = {}
    for candidate_id in candidate_ids:
        candidate_rows = [
            row
            for row in rows
            if row.candidate_id == candidate_id and row.split == "id"
        ]
        hard_values = [row.result.correct for row in candidate_rows if row.task_id in hard]
        easy_values = [row.result.correct for row in candidate_rows if row.task_id in easy]
        if not hard_values or not easy_values:
            raise ExperimentRunError(
                f"candidate {candidate_id!r} is missing Ctx2Skill probe rows"
            )
        rates[candidate_id] = (
            sum(hard_values) / len(hard_values),
            sum(easy_values) / len(easy_values),
        )
    return rates, probe_count


def _scored_candidates(
    versions: Mapping[str, SkillVersion | EvaluationCandidate],
    summaries: Mapping[str, ObjectiveSummary],
    *,
    records: Iterable[EvaluationRecord],
    configuration: Mapping[str, Any],
    include_ctx_rates: bool,
) -> tuple[ScoredCandidate, ...]:
    constraints = _constraints(configuration)
    ctx_count = 10
    for variant in configuration["methods"]["fixed_scalarization"]["variants"]:
        if variant.get("id") == "ctx2skill_hard_easy_product":
            ctx_count = int(variant.get("tasks_per_probe", 10))
            break
    ctx_rates, ctx_observed_count = (
        _hard_easy_rates(records, summaries, count=ctx_count)
        if include_ctx_rates
        else ({}, 0)
    )
    scored: list[ScoredCandidate] = []
    for candidate_id, summary in summaries.items():
        candidate = versions[candidate_id]
        version = candidate.version if isinstance(candidate, EvaluationCandidate) else candidate
        result = feasibility(summary, constraints)
        metadata: dict[str, Any] = {
            "feasible": result.feasible,
            "feasibility_reasons": list(result.reasons),
            "evaluation_cost": sum(
                record.candidate_id == candidate_id for record in records
            ),
        }
        if candidate_id in ctx_rates:
            hard, easy = ctx_rates[candidate_id]
            metadata.update(
                hard_probe_success_rate=hard,
                easy_probe_success_rate=easy,
                hard_easy_probe_tasks_per_side=ctx_observed_count,
                hard_easy_probe_tasks_requested=ctx_count,
            )
        scored.append(
            ScoredCandidate(
                candidate_id=candidate_id,
                patch_ids=version.lineage.patch_ids,
                objectives=summary.point_objectives(),
                metadata=metadata,
                summary=summary,
                content_hash=(
                    candidate.content_hash
                    if isinstance(candidate, EvaluationCandidate)
                    else version.skill.content_hash
                ),
            )
        )
    return tuple(sorted(scored, key=lambda item: item.candidate_id))


def _normalization_ranges(
    configuration: Mapping[str, Any], *, strict: bool
) -> tuple[tuple[float, float], ...] | None:
    raw = configuration["selection_protocol"].get("normalization_ranges")
    if raw is None:
        if strict:
            raise ExperimentRunError(
                "real/replay search requires selection_protocol.normalization_ranges"
            )
        return None
    if not isinstance(raw, list) or len(raw) != 4:
        raise ExperimentRunError("normalization_ranges must contain four [min,max] pairs")
    try:
        ranges = tuple((float(item[0]), float(item[1])) for item in raw)
    except (IndexError, TypeError, ValueError) as exc:
        raise ExperimentRunError("normalization_ranges entries must be numeric pairs") from exc
    if any(maximum < minimum for minimum, maximum in ranges):
        raise ExperimentRunError("normalization range maximum must be at least its minimum")
    return ranges


def _normalized_scored_vectors(
    candidates: Iterable[ScoredCandidate],
    ranges: tuple[tuple[float, float], ...],
    *,
    feasible_only: bool,
) -> dict[str, tuple[float, float, float, float]]:
    normalized: dict[str, tuple[float, float, float, float]] = {}
    for candidate in candidates:
        if feasible_only and candidate.metadata.get("feasible") is not True:
            continue
        values: list[float] = []
        for value, (minimum, maximum) in zip(
            candidate.vector, ranges, strict=True
        ):
            values.append(
                0.0
                if maximum <= minimum
                else (float(value) - minimum) / (maximum - minimum)
            )
        normalized[candidate.candidate_id] = tuple(values)  # type: ignore[assignment]
    return normalized


def _hypervolume_reference(configuration: Mapping[str, Any]) -> tuple[float, ...]:
    raw = configuration["statistics"]["hypervolume_reference"].get(
        "normalized_maximize_point"
    )
    if not isinstance(raw, list) or len(raw) != 4:
        raise ExperimentRunError(
            "hypervolume_reference.normalized_maximize_point must contain four values"
        )
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ExperimentRunError("hypervolume reference must be numeric") from exc


def _feasible_hypervolume(
    candidates: Iterable[ScoredCandidate],
    *,
    ranges: tuple[tuple[float, float], ...],
    reference: tuple[float, ...],
) -> float:
    vectors = _normalized_scored_vectors(
        candidates, ranges, feasible_only=True
    )
    return hypervolume(vectors.values(), reference)


def _budget_curve(
    *,
    candidate_order: Iterable[str],
    screen_scored: tuple[ScoredCandidate, ...],
    promoted_order: Iterable[str],
    full_scored: tuple[ScoredCandidate, ...],
    configuration: Mapping[str, Any],
    ranges: tuple[tuple[float, float], ...] | None,
) -> list[dict[str, Any]]:
    checkpoints = sorted(
        int(value)
        for value in configuration["budgets"]["search_total_per_method"].get(
            "budget_curve_task_executions", []
        )
    )
    if not checkpoints:
        return []
    if ranges is None:
        return [
            {
                "task_executions": checkpoint,
                "feasible_hypervolume": None,
                "defined": False,
                "reason": "normalization_ranges_unavailable",
            }
            for checkpoint in checkpoints
        ]
    reference = _hypervolume_reference(configuration)
    screen_by_id = {candidate.candidate_id: candidate for candidate in screen_scored}
    full_by_id = {candidate.candidate_id: candidate for candidate in full_scored}
    events: list[tuple[int, ScoredCandidate]] = []
    screen_cost = int(configuration["budgets"]["screen"]["task_executions"])
    incremental_full = int(
        configuration["budgets"]["full"][
            "incremental_task_executions_after_screen"
        ]
    )
    spent = 0
    for candidate_id in candidate_order:
        candidate = screen_by_id.get(candidate_id)
        if candidate is None:
            continue
        spent += screen_cost
        events.append((spent, candidate))
    for candidate_id in promoted_order:
        candidate = full_by_id.get(candidate_id)
        if candidate is None:
            continue
        spent += incremental_full
        events.append((spent, candidate))
    active: dict[str, ScoredCandidate] = {}
    event_index = 0
    curve: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        while event_index < len(events) and events[event_index][0] <= checkpoint:
            _, candidate = events[event_index]
            active[candidate.candidate_id] = candidate
            event_index += 1
        curve.append(
            {
                "task_executions": checkpoint,
                "feasible_hypervolume": _feasible_hypervolume(
                    active.values(), ranges=ranges, reference=reference
                ),
                "evaluated_candidates": len(active),
                "defined": True,
                "run_reached_checkpoint": bool(events and events[-1][0] >= checkpoint),
            }
        )
    return curve


def _deployment_payload(
    *,
    method_id: str,
    search_seed: int,
    full_scored: tuple[ScoredCandidate, ...],
    base_summary: ObjectiveSummary,
    configuration: Mapping[str, Any],
    ranges: tuple[tuple[float, float], ...] | None,
) -> dict[str, Any]:
    candidates = tuple(
        DeploymentCandidate(
            candidate.candidate_id,
            candidate.content_hash,
            candidate.summary,
        )
        for candidate in full_scored
        if candidate.content_hash is not None and candidate.summary is not None
    )
    epsilon = float(configuration["constraints"]["id_accuracy_floor"]["epsilon"])
    accuracy_floor = max(0.0, base_summary.id_accuracy.lcb - epsilon)
    token_budget = float(configuration["constraints"]["token_budget"]["budget"])
    selections: dict[str, Any] = {}
    for raw_policy in configuration["deployment"]["policies"]:
        policy_id = str(raw_policy["id"])
        try:
            selected = select_deployment_candidate(
                policy_id,  # type: ignore[arg-type]
                candidates,
                accuracy_floor=accuracy_floor,
                token_budget=token_budget,
                pessimistic=True,
                frozen_ranges=ranges,
            )
            selections[policy_id] = {
                "defined": True,
                "candidate_id": selected.candidate_id,
                "content_hash": selected.content_hash,
            }
        except ValueError as exc:
            selections[policy_id] = {
                "defined": False,
                "candidate_id": None,
                "reason": str(exc),
            }
    primary = str(configuration["deployment"]["primary_policy"])
    return {
        "schema_version": 1,
        "method_id": method_id,
        "search_seed": search_seed,
        "selection_data_role": "validation_only",
        "pessimistic": True,
        "accuracy_floor": accuracy_floor,
        "token_budget": token_budget,
        "primary_policy": primary,
        "primary_candidate_id": selections.get(primary, {}).get("candidate_id"),
        "policies": selections,
    }


def _materialize_subsets(
    base: SkillVersion,
    patches: tuple[Patch, ...],
    evidence: Mapping[str, TraceEvidence],
    subsets: Iterable[tuple[str, ...]],
    *,
    label: str,
) -> tuple[dict[str, SkillVersion], MaterializationStore, list[dict[str, str]]]:
    by_patch = {patch.patch_id: patch for patch in patches}
    materializer = Materializer()
    store = MaterializationStore()
    store.add(base)
    versions: dict[str, SkillVersion] = {}
    seen_hashes: set[str] = {base.skill.content_hash}
    rejected: list[dict[str, str]] = []
    for subset in subsets:
        unknown = set(subset) - set(by_patch)
        if unknown:
            raise ExperimentRunError(f"proposal references unknown patches: {sorted(unknown)}")
        if not subset:
            continue
        try:
            version = materializer.materialize(
                base,
                tuple(by_patch[patch_id] for patch_id in subset),
                evidence=evidence,
                label=label,
            )
        except MaterializationError as exc:
            rejected.append({"patch_ids": ",".join(subset), "reason": str(exc)})
            continue
        if version.skill.content_hash in seen_hashes:
            rejected.append({"patch_ids": ",".join(subset), "reason": "duplicate_content"})
            continue
        seen_hashes.add(version.skill.content_hash)
        version = store.add(version)
        versions[version.lineage.version_id] = version
    return versions, store, rejected


def _materialize_subset_batch(
    base: SkillVersion,
    patches: tuple[Patch, ...],
    evidence: Mapping[str, TraceEvidence],
    subsets: Iterable[tuple[str, ...]],
    *,
    label: str,
    store: MaterializationStore,
    execution_candidates: Mapping[str, SkillVersion | EvaluationCandidate],
    rejected: list[dict[str, str]],
) -> dict[tuple[str, ...], SkillVersion]:
    """Materialize one ask batch while preserving subset-to-feedback identity."""

    by_patch = {patch.patch_id: patch for patch in patches}
    materializer = Materializer()
    seen_hashes = {
        candidate.content_hash
        if isinstance(candidate, EvaluationCandidate)
        else candidate.skill.content_hash
        for candidate in execution_candidates.values()
    }
    seen_hashes.add(base.skill.content_hash)
    materialized: dict[tuple[str, ...], SkillVersion] = {}
    for subset in subsets:
        unknown = set(subset) - set(by_patch)
        if unknown:
            raise ExperimentRunError(
                f"proposal references unknown patches: {sorted(unknown)}"
            )
        try:
            version = materializer.materialize(
                base,
                tuple(by_patch[patch_id] for patch_id in subset),
                evidence=evidence,
                label=label,
            )
        except MaterializationError as exc:
            rejected.append({"patch_ids": ",".join(subset), "reason": str(exc)})
            continue
        if version.skill.content_hash in seen_hashes:
            rejected.append(
                {"patch_ids": ",".join(subset), "reason": "duplicate_content"}
            )
            continue
        seen_hashes.add(version.skill.content_hash)
        materialized[subset] = store.add(version)
    return materialized


def _invalid_feedback(
    subset: tuple[str, ...], reason: str, configuration: Mapping[str, Any]
) -> ScoredCandidate:
    digest = hashlib.sha256(
        canonical_json({"patch_ids": list(subset), "reason": reason}).encode("utf-8")
    ).hexdigest()
    token_budget = float(configuration["constraints"]["token_budget"]["budget"])
    return ScoredCandidate(
        candidate_id=f"invalid-{digest[:24]}",
        patch_ids=subset,
        objectives=PointObjectives(0.0, 0.0, token_budget + 1.0, 1.0),
        metadata={
            "materialization_valid": False,
            "evaluation_cost": 0,
            "reason": reason,
        },
        content_hash=digest,
    )


def _static_subsets(
    plugin: BaselinePlugin,
    plugin_id: str,
    patch_ids: tuple[str, ...],
    *,
    maximum_candidates: int,
    seed: int,
    configuration: Mapping[str, Any],
    smoke: bool,
) -> tuple[tuple[str, ...], ...]:
    if smoke or plugin_id in {"no_skill", "base_skill", "trace2skill_all"}:
        return plugin.propose_subsets(
            patch_ids, max_candidates=maximum_candidates, seed=seed
        )
    space = (1 << len(set(patch_ids))) - 1
    count = min(maximum_candidates, space)
    simple = configuration["methods"]["simple_patch_composition"]
    if plugin_id == "simple_patch_composition":
        return BernoulliUniqueStream(
            patch_ids,
            seed,
            inclusion_probability=float(simple.get("inclusion_probability", 0.5)),
            maximum_consecutive_duplicates=int(
                simple.get("maximum_consecutive_duplicate_proposals", 1_000)
            ),
        ).ask(count)
    common_ids = {
        "ctx2skill_hard_easy_product",
        "mocha_chebyshev_hvc",
        "passive_archive",
        "paretoskill",
    }
    if plugin_id.startswith("fixed_scalarization/") or plugin_id in common_ids:
        return CommonCandidateStream.from_bernoulli(
            patch_ids,
            candidate_count=count,
            seed=seed,
            maximum_consecutive_duplicates=int(
                simple.get("maximum_consecutive_duplicate_proposals", 1_000)
            ),
        ).ask(count)
    return plugin.propose_subsets(
        patch_ids, max_candidates=maximum_candidates, seed=seed
    )


def _subset_controller(
    runtime: ExperimentRuntime,
    plugin: BaselinePlugin,
    plugin_id: str,
    patch_ids: tuple[str, ...],
    *,
    maximum_candidates: int,
    search_seed: int,
    configuration: Mapping[str, Any],
) -> tuple[SubsetSearchController, int] | None:
    simple = configuration["methods"]["simple_patch_composition"]
    duplicate_limit = int(
        simple.get("maximum_consecutive_duplicate_proposals", 1_000)
    )
    space = (1 << len(set(patch_ids))) - 1
    candidate_count = min(maximum_candidates, space)
    if plugin_id == "trace2skill_accuracy_subset":
        if runtime.binary_optimizer_factory is None:
            raise ExternalOptimizerRequired(
                "Trace2Skill-style binary subset search requires the frozen "
                "BinarySubsetBayesianAdapter before any provider execution"
            )
        method = configuration["methods"]["trace2skill_accuracy_subset"]
        adapter = runtime.binary_optimizer_factory(patch_ids, search_seed, method)
        controller = make_binary_subset_controller(
            patch_ids,
            seed=search_seed,
            adapter=adapter,
            maximum_consecutive_duplicates=duplicate_limit,
        )
        protocol = method.get("optimizer_protocol", {})
        return controller, int(protocol.get("batch_size", 1))
    common = CommonCandidateStream.from_bernoulli(
        patch_ids,
        candidate_count=candidate_count,
        seed=search_seed,
        maximum_consecutive_duplicates=duplicate_limit,
    )
    if plugin_id == "evoskill_scalar_topk":
        method = configuration["methods"]["evoskill_scalar_topk"]
        return EvoTopKController(common, top_k=int(method["top_k"])), min(
            20, candidate_count
        )
    if plugin_id == "skillmoo_nsga2":
        method = configuration["methods"]["skillmoo_nsga2"]
        population = int(method["population_size"])
        offspring = int(method["offspring_size"])
        if maximum_candidates < population:
            raise ExperimentRunError(
                "NSGA-II screen capacity is smaller than its frozen population"
            )
        mutation = method.get("mutation", {})
        mutation_probability = (
            1.0 / len(patch_ids)
            if mutation.get("per_locus_probability") == "reciprocal_patch_pool_size"
            else float(mutation["per_locus_probability"])
        )
        crossover = method.get("crossover", {})
        return NSGAIIController(
            patch_ids,
            seed=search_seed,
            population_size=population,
            offspring_size=offspring,
            crossover_probability=float(crossover.get("probability", 0.9)),
            per_locus_parent_probability=float(
                crossover.get("per_locus_parent_probability", 0.5)
            ),
            mutation_probability=mutation_probability,
            maximum_consecutive_duplicates=duplicate_limit,
            initial_stream=common,
        ), offspring
    if plugin_id == "mocha_chebyshev_hvc":
        if not isinstance(plugin, MOCHAPlugin):
            raise ExperimentRunError("MOCHA method did not resolve to MOCHAPlugin")
        return MOCHAController(
            common,
            seed=search_seed,
            logical_task_execution_budget=int(
                configuration["budgets"]["search_total_per_method"][
                    "task_executions"
                ]
            ),
            task_executions_per_candidate=int(
                configuration["budgets"]["screen"]["task_executions"]
            ),
            plugin=plugin,
        ), 1
    return None


@dataclass(slots=True)
class _SubsetControllerProgress:
    controller: SubsetSearchController
    execution_candidates: dict[str, SkillVersion | EvaluationCandidate]
    store: MaterializationStore
    rejected: list[dict[str, str]]
    screen_records: list[EvaluationRecord]
    screen_summaries: dict[str, ObjectiveSummary]
    screen_scored: tuple[ScoredCandidate, ...]
    proposed: int


def _search_controller_checkpoint_payload(
    *,
    manifest: ExperimentManifest,
    method_id: str,
    search_seed: int,
    execution_complete: bool,
    maximum_candidates: int,
    batch_size: int,
    patch_ids: tuple[str, ...],
    controller: SubsetSearchController,
    proposed: int,
    execution_candidates: Mapping[str, SkillVersion | EvaluationCandidate],
    store: MaterializationStore,
    rejected: Iterable[Mapping[str, str]],
    screen_records: Iterable[EvaluationRecord],
) -> dict[str, Any]:
    records = tuple(screen_records)
    return {
        "schema_version": 2,
        "experiment_id": manifest.experiment_id,
        "implementation_digest": manifest.implementation_digest,
        "method_id": method_id,
        "search_seed": search_seed,
        "execution_complete": execution_complete,
        "maximum_candidates": maximum_candidates,
        "batch_size": batch_size,
        "patch_ids": list(patch_ids),
        "proposed_candidates": proposed,
        "execution_candidate_order": list(execution_candidates),
        "controller": dict(controller.state_dict()),
        "rejected_materializations": [dict(item) for item in rejected],
        # The materialization snapshot is embedded in the atomic commit record.
        # Cache rows are immutable and are restored by the keys below.  This
        # avoids a torn multi-file checkpoint if a process exits between writes.
        "materializations": store.to_dict(),
        "screen_cache_keys": sorted(record.cache_key for record in records),
        "screen_record_count": len(records),
    }


def _controller_consumed_subsets(
    state: Mapping[str, Any], patch_ids: tuple[str, ...]
) -> set[tuple[str, ...]]:
    state_type = state.get("state_type")
    if state_type == AdapterBackedBinarySubsetController.STATE_TYPE:
        rows = state.get("observations")
        field = "observations"
    elif state_type == EvoTopKController.STATE_TYPE:
        rows = state.get("observed_subsets")
        field = "observed_subsets"
    elif state_type == NSGAIIController.STATE_TYPE:
        rows = state.get("seen")
        field = "seen"
    elif state_type == MOCHAController.STATE_TYPE:
        rows = state.get("decisions")
        field = "decisions"
    else:
        raise ExperimentRunError(
            f"unsupported partial controller state type: {state_type!r}"
        )
    if not isinstance(rows, list):
        raise ExperimentRunError(f"controller checkpoint {field} must be an array")
    subsets: list[tuple[str, ...]] = []
    universe = set(patch_ids)
    for row in rows:
        raw_subset = row.get("patch_ids") if isinstance(row, Mapping) else row
        if not isinstance(raw_subset, list):
            raise ExperimentRunError(
                f"controller checkpoint {field} contains a malformed patch subset"
            )
        subset = tuple(raw_subset)
        if (
            not subset
            or any(not isinstance(value, str) or value not in universe for value in subset)
            or len(set(subset)) != len(subset)
        ):
            raise ExperimentRunError(
                f"controller checkpoint {field} contains an invalid patch subset"
            )
        canonical = tuple(value for value in patch_ids if value in set(subset))
        if canonical != subset:
            raise ExperimentRunError(
                f"controller checkpoint {field} patch order is not canonical"
            )
        subsets.append(subset)
    if len(set(subsets)) != len(subsets):
        raise ExperimentRunError(
            f"controller checkpoint {field} contains duplicate patch subsets"
        )
    return set(subsets)


def _restore_subset_controller_progress(
    *,
    expected_controller: SubsetSearchController,
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    phase: PhaseRuntime,
    configuration: Mapping[str, Any],
    method_id: str,
    search_seed: int,
    maximum_candidates: int,
    batch_size: int,
    patch_ids: tuple[str, ...],
    screen_blocks: tuple[TaskSeedBlock, ...],
    cache: ResultCache,
    output: Path,
) -> _SubsetControllerProgress | None:
    checkpoint_path = output / "search_controller.json"
    convenience_paths = (
        output / "materializations.partial.json",
        output / "screen_task_outcomes.partial.jsonl",
    )
    if not checkpoint_path.is_file():
        if any(path.exists() for path in convenience_paths):
            raise ExperimentRunError(
                "orphaned partial controller artifacts exist without a checkpoint"
            )
        return None
    checkpoint = _read_json_object(checkpoint_path)
    expected_identity = {
        "schema_version": 2,
        "experiment_id": manifest.experiment_id,
        "implementation_digest": manifest.implementation_digest,
        "method_id": method_id,
        "search_seed": search_seed,
        "maximum_candidates": maximum_candidates,
        "batch_size": batch_size,
        "patch_ids": list(patch_ids),
    }
    mismatched = [
        name
        for name, expected in expected_identity.items()
        if checkpoint.get(name) != expected
    ]
    if mismatched:
        raise ExperimentRunError(
            "partial controller checkpoint identity/configuration mismatch: "
            f"{mismatched}"
        )
    if not isinstance(checkpoint.get("execution_complete"), bool):
        raise ExperimentRunError("controller checkpoint execution_complete must be boolean")
    raw_controller = checkpoint.get("controller")
    if not isinstance(raw_controller, Mapping):
        raise ExperimentRunError("controller checkpoint state must be an object")
    if raw_controller.get("pending") != []:
        raise ExperimentRunError(
            "persisted controller checkpoint must not contain an unresolved ask batch"
        )
    binary_adapter = (
        expected_controller.adapter
        if isinstance(expected_controller, AdapterBackedBinarySubsetController)
        else None
    )
    try:
        restored_controller = restore_search_controller(
            raw_controller,
            binary_adapter=binary_adapter,
        )
    except (ValueError, SearchStrategyError) as exc:
        raise ExperimentRunError("invalid persisted search controller state") from exc
    if type(restored_controller) is not type(expected_controller):
        raise ExperimentRunError("restored controller type does not match configured method")
    restored_patch_ids = getattr(restored_controller, "patch_ids", None)
    if restored_patch_ids != patch_ids:
        raise ExperimentRunError("restored controller patch universe mismatch")

    proposed = checkpoint.get("proposed_candidates")
    if isinstance(proposed, bool) or not isinstance(proposed, int) or proposed < 1:
        raise ExperimentRunError("controller proposed_candidates must be positive")
    consumed_subsets = _controller_consumed_subsets(raw_controller, patch_ids)
    if proposed != len(consumed_subsets):
        raise ExperimentRunError(
            "controller proposed count does not match its consumed subset state"
        )

    raw_store = checkpoint.get("materializations")
    if not isinstance(raw_store, Mapping):
        raise ExperimentRunError("controller materialization snapshot must be an object")
    try:
        store = MaterializationStore.from_dict(raw_store)  # type: ignore[arg-type]
        restored_base = store.get(runtime.base.lineage.version_id)
    except (KeyError, MaterializationError, TypeError, ValueError) as exc:
        raise ExperimentRunError("invalid controller materialization snapshot") from exc
    if restored_base.to_dict() != runtime.base.to_dict():
        raise ExperimentRunError("controller materialization base does not match runtime")
    raw_order = checkpoint.get("execution_candidate_order")
    if (
        not isinstance(raw_order, list)
        or any(not isinstance(value, str) or not value for value in raw_order)
        or len(set(raw_order)) != len(raw_order)
    ):
        raise ExperimentRunError("controller execution_candidate_order is malformed")
    expected_lineages = {runtime.base.lineage.version_id, *raw_order}
    if set(store.lineages) != expected_lineages:
        raise ExperimentRunError(
            "controller candidate order does not match materialization lineages"
        )
    if len(raw_order) > maximum_candidates:
        raise ExperimentRunError("controller checkpoint exceeds candidate capacity")
    execution_candidates: dict[str, SkillVersion | EvaluationCandidate] = {
        version_id: store.get(version_id) for version_id in raw_order
    }
    materialized_subsets = {
        tuple(candidate.lineage.patch_ids)
        for candidate in execution_candidates.values()
        if isinstance(candidate, SkillVersion)
    }
    if not materialized_subsets <= consumed_subsets:
        raise ExperimentRunError(
            "controller materializations contain subsets absent from controller state"
        )

    raw_rejected = checkpoint.get("rejected_materializations")
    if not isinstance(raw_rejected, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("patch_ids"), str)
        or not isinstance(item.get("reason"), str)
        for item in raw_rejected
    ):
        raise ExperimentRunError("controller rejected_materializations is malformed")
    rejected = [
        {"patch_ids": str(item["patch_ids"]), "reason": str(item["reason"])}
        for item in raw_rejected
    ]
    rejected_keys = [item["patch_ids"] for item in rejected]
    if len(set(rejected_keys)) != len(rejected_keys):
        raise ExperimentRunError("controller rejected subsets contain duplicates")
    consumed_keys = {",".join(subset) for subset in consumed_subsets}
    accounted_keys = {
        *(",".join(subset) for subset in materialized_subsets),
        *rejected_keys,
    }
    if accounted_keys != consumed_keys:
        raise ExperimentRunError(
            "controller materialized/rejected subsets do not cover consumed proposals"
        )

    raw_cache_keys = checkpoint.get("screen_cache_keys")
    if (
        not isinstance(raw_cache_keys, list)
        or any(not isinstance(value, str) for value in raw_cache_keys)
        or len(set(raw_cache_keys)) != len(raw_cache_keys)
    ):
        raise ExperimentRunError("controller screen_cache_keys is malformed")
    screen_records: list[EvaluationRecord] = []
    for cache_key in raw_cache_keys:
        try:
            record = cache.get(cache_key)
        except (OSError, StorageError, ValueError) as exc:
            raise ExperimentRunError("controller checkpoint cache is corrupt") from exc
        if record is None:
            raise ExperimentRunError(
                f"controller checkpoint cache row is missing: {cache_key}"
            )
        if record.experiment_id != manifest.experiment_id:
            raise ExperimentRunError("controller checkpoint cache experiment mismatch")
        screen_records.append(record)
    screen_records.sort(key=lambda item: item.evaluation_identity)
    if checkpoint.get("screen_record_count") != len(screen_records):
        raise ExperimentRunError("controller screen record count is inconsistent")
    if not execution_candidates:
        if screen_records:
            raise ExperimentRunError(
                "controller checkpoint has screen rows without materialized candidates"
            )
        screen_summaries: dict[str, ObjectiveSummary] = {}
        screen_scored: tuple[ScoredCandidate, ...] = ()
    else:
        try:
            expected_matrix = expected_evaluation_matrix(
                base=runtime.base,
                candidates=tuple(execution_candidates.values()),
                targets=phase.targets,
                blocks=screen_blocks,
            )
            validate_evaluation_matrix(screen_records, expected_matrix)
            screen_summaries = _summaries(
                screen_records,
                execution_candidates,
                targets=phase.targets,
                configuration=configuration,
                smoke=False,
            )
            screen_scored = _scored_candidates(
                execution_candidates,
                screen_summaries,
                records=screen_records,
                configuration=configuration,
                include_ctx_rates=False,
            )
        except (ExperimentRunError, ValueError) as exc:
            raise ExperimentRunError(
                "controller checkpoint screen evaluation matrix is invalid"
            ) from exc
    return _SubsetControllerProgress(
        controller=restored_controller,
        execution_candidates=execution_candidates,
        store=store,
        rejected=rejected,
        screen_records=screen_records,
        screen_summaries=screen_summaries,
        screen_scored=screen_scored,
        proposed=proposed,
    )


def _controller_promotion_pool(
    controller: SubsetSearchController | None,
    screen_scored: tuple[ScoredCandidate, ...],
) -> tuple[ScoredCandidate, ...]:
    """Restrict adaptive-method promotion to the controller's final survivors."""

    if isinstance(controller, EvoTopKController):
        survivor_ids = {candidate.candidate_id for candidate in controller.incumbents}
    elif isinstance(controller, NSGAIIController):
        survivor_ids = {candidate.candidate_id for candidate in controller.population}
    elif isinstance(controller, MOCHAController):
        survivor_ids = set(controller.accepted_ids)
    else:
        return screen_scored
    return tuple(
        candidate
        for candidate in screen_scored
        if candidate.candidate_id in survivor_ids
    )


def _run_subset_controller(
    *,
    controller: SubsetSearchController,
    batch_size: int,
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    phase: PhaseRuntime,
    configuration: Mapping[str, Any],
    method_id: str,
    search_seed: int,
    maximum_candidates: int,
    screen_blocks: tuple[TaskSeedBlock, ...],
    cache: ResultCache,
    policy: NetworkPolicy,
    output: Path,
    progress: _SubsetControllerProgress | None = None,
) -> tuple[
    dict[str, SkillVersion | EvaluationCandidate],
    MaterializationStore,
    list[dict[str, str]],
    list[EvaluationRecord],
    dict[str, ObjectiveSummary],
    tuple[ScoredCandidate, ...],
    int,
]:
    if progress is None:
        store = MaterializationStore()
        store.add(runtime.base)
        execution_candidates: dict[str, SkillVersion | EvaluationCandidate] = {}
        rejected: list[dict[str, str]] = []
        screen_records: list[EvaluationRecord] = []
        screen_summaries: dict[str, ObjectiveSummary] = {}
        screen_scored: tuple[ScoredCandidate, ...] = ()
        proposed = 0
    else:
        if progress.controller is not controller:
            raise ExperimentRunError("restored progress/controller identity mismatch")
        store = progress.store
        execution_candidates = progress.execution_candidates
        rejected = progress.rejected
        screen_records = progress.screen_records
        screen_summaries = progress.screen_summaries
        screen_scored = progress.screen_scored
        proposed = progress.proposed
    proposal_cap = maximum_candidates + int(
        configuration["methods"]["simple_patch_composition"].get(
            "maximum_consecutive_duplicate_proposals", 1_000
        )
    )
    while len(execution_candidates) < maximum_candidates and proposed < proposal_cap:
        remaining = maximum_candidates - len(execution_candidates)
        requested = min(batch_size, remaining)
        if isinstance(controller, NSGAIIController) and not controller.population:
            requested = controller.population_size
        try:
            subsets = controller.ask(requested)
        except SearchSpaceExhausted:
            break
        if not subsets:
            break
        proposed += len(subsets)
        materialized = _materialize_subset_batch(
            runtime.base,
            runtime.patches,
            runtime.evidence,
            subsets,
            label=method_id.replace("/", "-"),
            store=store,
            execution_candidates=execution_candidates,
            rejected=rejected,
        )
        for version in materialized.values():
            execution_candidates[version.lineage.version_id] = version
        new_records = (
            _evaluate(
                experiment_id=manifest.experiment_id,
                base=runtime.base,
                candidates=tuple(materialized.values()),
                phase=phase,
                blocks=screen_blocks,
                providers=runtime.providers,
                cache=cache,
                policy=policy,
                retry_limit=configuration["task_seed_blocks"]["retry_limit"],
                failure_event_path=output / "failure_events.jsonl",
            )
            if materialized
            else []
        )
        screen_records = _merge_records(screen_records, new_records)
        batch_summaries = (
            _summaries(
                new_records,
                (version.lineage.version_id for version in materialized.values()),
                targets=phase.targets,
                configuration=configuration,
                smoke=False,
            )
            if materialized
            else {}
        )
        screen_summaries.update(batch_summaries)
        batch_scored = _scored_candidates(
            {
                version.lineage.version_id: version
                for version in materialized.values()
            },
            batch_summaries,
            records=new_records,
            configuration=configuration,
            include_ctx_rates=False,
        )
        by_subset = {tuple(item.patch_ids): item for item in batch_scored}
        rejection_by_subset = {
            tuple(item["patch_ids"].split(",")): item["reason"]
            for item in rejected
            if item["patch_ids"]
        }
        feedback = tuple(
            by_subset.get(subset)
            or _invalid_feedback(
                subset,
                rejection_by_subset.get(subset, "materialization_rejected"),
                configuration,
            )
            for subset in subsets
        )
        controller.tell(feedback)
        screen_scored = tuple(
            sorted((*screen_scored, *batch_scored), key=lambda item: item.candidate_id)
        )
        _atomic_json(
            output / "search_controller.json",
            _search_controller_checkpoint_payload(
                manifest=manifest,
                method_id=method_id,
                search_seed=search_seed,
                execution_complete=False,
                maximum_candidates=maximum_candidates,
                batch_size=batch_size,
                patch_ids=tuple(patch.patch_id for patch in runtime.patches),
                controller=controller,
                proposed=proposed,
                execution_candidates=execution_candidates,
                store=store,
                rejected=rejected,
                screen_records=screen_records,
            ),
        )
        store.save(output / "materializations.partial.json")
        JsonlResultStore(output / "screen_task_outcomes.partial.jsonl").replace(
            screen_records
        )
    return (
        execution_candidates,
        store,
        rejected,
        screen_records,
        screen_summaries,
        screen_scored,
        proposed,
    )


def _merge_records(*collections: Iterable[EvaluationRecord]) -> list[EvaluationRecord]:
    merged: dict[tuple[str, str, str, int], EvaluationRecord] = {}
    for record in (item for collection in collections for item in collection):
        existing = merged.get(record.evaluation_identity)
        if existing is not None and existing.to_dict() != record.to_dict():
            raise ExperimentRunError(
                f"conflicting repeated evaluation identity: {record.evaluation_identity}"
            )
        merged[record.evaluation_identity] = record
    return list(sorted(merged.values(), key=lambda item: item.evaluation_identity))


def _active_objectives(configuration: Mapping[str, Any]):
    objectives = configuration["objectives"]
    return normalize_active_objectives(
        name
        for name in DEFAULT_ACTIVE_OBJECTIVES
        if objectives[name].get("enabled") is True
    )


def _new_archive(configuration: Mapping[str, Any]) -> ParetoArchive:
    dominance = configuration["statistics"]["dominance"]["primary"]
    return ParetoArchive(
        max_size=int(
            configuration["selection_protocol"]["archive_capacity"]["max_entries"]
        ),
        evaluation_budget=int(
            configuration["budgets"]["search_total_per_method"]["task_executions"]
        ),
        constraints=_constraints(configuration),
        dominance_mode="point" if dominance == "point_estimates" else "uncertainty",
        active_objectives=_active_objectives(configuration),
    )


def _admit_scored(
    archive: ParetoArchive, scored: Iterable[ScoredCandidate]
) -> None:
    for candidate in sorted(scored, key=lambda item: item.candidate_id):
        if candidate.summary is None or candidate.content_hash is None:
            raise ExperimentRunError("archive candidates require summaries and content hashes")
        cost = candidate.metadata.get("evaluation_cost", 1)
        if isinstance(cost, bool) or not isinstance(cost, int):
            raise ExperimentRunError("archive evaluation_cost must be an integer")
        archive.admit(
            ArchiveEntry(
                candidate_id=candidate.candidate_id,
                content_hash=candidate.content_hash,
                objectives=candidate.summary,
                evaluation_cost=cost,
            )
        )


def _directional_evidence(
    evidence: Mapping[str, TraceEvidence],
) -> dict[ObjectiveDirection, tuple[TraceEvidence, ...]]:
    aliases = {
        ObjectiveDirection.ACCURACY: {"id_accuracy", "accuracy"},
        ObjectiveDirection.TRANSFER: {"worst_target_transfer", "transfer"},
        ObjectiveDirection.COST: {"token_cost", "cost", "compress"},
        ObjectiveDirection.REGRESSION: {"paired_regression", "regression"},
    }
    result: dict[ObjectiveDirection, tuple[TraceEvidence, ...]] = {}
    for direction, names in aliases.items():
        matching = tuple(
            item
            for item in sorted(evidence.values(), key=lambda trace: trace.evidence_id)
            if names & {tag.lower() for tag in item.tags}
        )
        if matching:
            result[direction] = matching
    return result


def _archive_conditioned_expand(
    *,
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    phase: PhaseRuntime,
    configuration: Mapping[str, Any],
    search_seed: int,
    maximum_candidates: int,
    screen_blocks: tuple[TaskSeedBlock, ...],
    cache: ResultCache,
    policy: NetworkPolicy,
    store: MaterializationStore,
    execution_candidates: dict[str, SkillVersion | EvaluationCandidate],
    screen_records: list[EvaluationRecord],
    screen_summaries: dict[str, ObjectiveSummary],
    screen_scored: tuple[ScoredCandidate, ...],
    rejected: list[dict[str, str]],
    smoke: bool,
    output: Path,
) -> tuple[
    list[EvaluationRecord],
    dict[str, ObjectiveSummary],
    tuple[ScoredCandidate, ...],
    int,
    ParetoArchive,
    list[Mapping[str, Any]],
]:
    archive = _new_archive(configuration)
    base_id = runtime.base.lineage.version_id
    base_summary = _summaries(
        screen_records,
        (base_id,),
        targets=phase.targets,
        configuration=configuration,
        smoke=smoke,
    )[base_id]
    base_scored = _scored_candidates(
        {base_id: runtime.base},
        {base_id: base_summary},
        records=screen_records,
        configuration=configuration,
        include_ctx_rates=False,
    )
    _admit_scored(archive, (*base_scored, *screen_scored))
    if runtime.proposer_factory is None:
        if not smoke and len(execution_candidates) < maximum_candidates:
            raise ExperimentRunError(
                "ParetoSkill search requires a configured archive-conditioned proposer"
            )
        return (
            screen_records,
            screen_summaries,
            screen_scored,
            0,
            archive,
            [],
        )
    evidence_by_direction = _directional_evidence(runtime.evidence)
    if not evidence_by_direction:
        raise ExperimentRunError(
            "ParetoSkill proposer requires objective-tagged trace evidence"
        )
    factory = runtime.proposer_factory
    factory_arguments = (
        store.get,
        search_seed,
        configuration,
        output / "proposal_cache",
    )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise ExperimentRunError("proposer factory signature cannot be inspected") from exc
    proposer: MutationProposer | None = None
    for argument_count in (4, 3, 2):
        try:
            signature.bind(*factory_arguments[:argument_count])
        except TypeError:
            continue
        proposer = factory(*factory_arguments[:argument_count])
        break
    if proposer is None:
        raise ExperimentRunError(
            "proposer factory must accept parent_resolver, seed, and optional "
            "configuration/cache arguments"
        )
    conditioner = ArchiveConditioner()
    generated = 0
    attempts = 0
    duplicate_limit = int(
        configuration["methods"]["simple_patch_composition"].get(
            "maximum_consecutive_duplicate_proposals", 1000
        )
    )
    materializer = Materializer()
    while len(execution_candidates) < maximum_candidates:
        if not archive.entries:
            raise ExperimentRunError("ParetoSkill working archive is empty")
        attempts += 1
        if attempts > maximum_candidates + duplicate_limit:
            raise ExperimentRunError("ParetoSkill proposal duplicate limit was exhausted")
        parent_entry, direction = conditioner.choose(
            archive.entries,
            evidence_by_direction=evidence_by_direction,
            seed=search_seed + attempts,
        )
        parent = store.get(parent_entry.candidate_id)
        bundle = EvidenceBundle(
            direction,
            evidence_by_direction[direction],
            notes=("objective-specific archive-conditioned evidence",),
        )
        patch = proposer.propose(
            MutationRequest(
                parent_version_id=parent.lineage.version_id,
                parent_candidate_id=parent_entry.candidate_id,
                direction=direction,
                evidence=bundle,
                sequence=attempts - 1,
            )
        )
        try:
            version = materializer.materialize(
                parent,
                (patch,),
                evidence=runtime.evidence,
                label="paretoskill",
            )
        except MaterializationError as exc:
            rejected.append({"patch_ids": patch.patch_id, "reason": str(exc)})
            continue
        if version.skill.content_hash in {
            candidate.content_hash
            if isinstance(candidate, EvaluationCandidate)
            else candidate.skill.content_hash
            for candidate in execution_candidates.values()
        }:
            rejected.append({"patch_ids": patch.patch_id, "reason": "duplicate_content"})
            continue
        store.add(version)
        execution_candidates[version.lineage.version_id] = version
        new_records = _evaluate(
            experiment_id=manifest.experiment_id,
            base=runtime.base,
            candidates=(version,),
            phase=phase,
            blocks=screen_blocks,
            providers=runtime.providers,
            cache=cache,
            policy=policy,
            retry_limit=configuration["task_seed_blocks"]["retry_limit"],
            failure_event_path=output / "failure_events.jsonl",
        )
        screen_records = _merge_records(screen_records, new_records)
        summary = _summaries(
            new_records,
            (version.lineage.version_id,),
            targets=phase.targets,
            configuration=configuration,
            smoke=smoke,
        )[version.lineage.version_id]
        screen_summaries[version.lineage.version_id] = summary
        scored = _scored_candidates(
            {version.lineage.version_id: version},
            {version.lineage.version_id: summary},
            records=new_records,
            configuration=configuration,
            include_ctx_rates=False,
        )[0]
        screen_scored = tuple(
            sorted((*screen_scored, scored), key=lambda item: item.candidate_id)
        )
        _admit_scored(archive, (scored,))
        generated += 1
    raw_events = getattr(proposer, "events", [])
    events = [
        {
            "patch_id": event.patch_id,
            "request_cache_key": event.request_cache_key,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "latency_ms": event.latency_ms,
            "provider_metadata": dict(event.provider_metadata),
            "cache_hit": event.cache_hit,
        }
        for event in raw_events
    ]
    return (
        screen_records,
        screen_summaries,
        screen_scored,
        generated,
        archive,
        events,
    )


def _evaluate(
    *,
    experiment_id: str,
    base: SkillVersion,
    candidates: tuple[SkillVersion | EvaluationCandidate, ...],
    phase: PhaseRuntime,
    blocks: tuple[TaskSeedBlock, ...],
    providers: Mapping[str, Provider],
    cache: ResultCache,
    policy: NetworkPolicy,
    retry_limit: int,
    failure_event_path: Path,
) -> list[EvaluationRecord]:
    harnesses = dict(phase.harnesses)
    for harness in harnesses.values():
        if hasattr(harness, "cache"):
            setattr(harness, "cache", cache)
    evaluator = PairedEvaluator(network_policy=policy, retry_limit=retry_limit)
    try:
        records = evaluator.evaluate(
            experiment_id=experiment_id,
            base=base,
            candidates=candidates,
            targets=phase.targets,
            blocks=blocks,
            providers=providers,
            harnesses=harnesses,
        )
    except BaseException:
        _persist_failure_events(failure_event_path, evaluator.failure_events)
        raise
    _persist_failure_events(failure_event_path, evaluator.failure_events)
    return records


def _method_specs(
    configuration: Mapping[str, Any], requested: Iterable[str] | None
) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    methods = configuration["methods"]
    specs: list[tuple[str, str, Mapping[str, Any]]] = []
    for method_id, method in methods.items():
        if method_id == "fixed_scalarization":
            for variant in method["variants"]:
                variant_id = str(variant["id"])
                plugin_id = (
                    "ctx2skill_hard_easy_product"
                    if variant_id == "ctx2skill_hard_easy_product"
                    else f"fixed_scalarization/{variant_id}"
                )
                specs.append((str(variant["run_id"]), plugin_id, configuration))
        else:
            specs.append((str(method_id), str(method_id), configuration))
    raw_ablations = configuration.get("ablations", [])
    if isinstance(raw_ablations, list):
        registry = AblationRegistry.from_manifest(raw_ablations)
        for ablation in raw_ablations:
            ablation_id = str(ablation["id"])
            specs.append(
                (
                    f"ablation_{ablation_id}",
                    str(ablation["base_method"]),
                    registry.apply(ablation_id, configuration),
                )
            )
    if requested is None:
        return tuple(specs)
    selected = set(requested)
    known = {method_id for method_id, _, _ in specs}
    unknown = selected - known
    if unknown:
        raise ExperimentRunError(f"unknown requested methods: {sorted(unknown)}")
    return tuple(spec for spec in specs if spec[0] in selected)


def _resume_method_if_complete(
    *,
    output: Path,
    manifest: ExperimentManifest,
    method_id: str,
    search_seed: int,
    cache: ResultCache,
    smoke: bool,
) -> MethodRunSummary | None:
    state_path = output / "run_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        summary = state["summary"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ExperimentRunError(f"invalid method run state: {state_path}") from exc
    if state.get("experiment_id") != manifest.experiment_id:
        raise ExperimentRunError("method run state belongs to another experiment")
    if state.get("execution_complete") is not True:
        return None
    if state.get("smoke") is not smoke:
        return None
    if summary.get("method_id") != method_id or summary.get("search_seed") != search_seed:
        raise ExperimentRunError("method run state identity mismatch")
    required = {
        "screen_task_outcomes.jsonl",
        "full_task_outcomes.jsonl",
        "selected_candidates.jsonl",
        "metrics.json",
        "materializations.json",
        "failure_events.jsonl",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise ExperimentRunError(f"completed method run is missing artifacts: {missing}")
    _read_failure_events(output / "failure_events.jsonl")
    declared_keys = state.get("cache_keys")
    if not isinstance(declared_keys, list) or not set(declared_keys) <= cache.keys():
        raise ExperimentRunError("completed method run cache is incomplete or corrupt")
    return MethodRunSummary(
        method_id=method_id,
        search_seed=search_seed,
        output_directory=output,
        proposed_candidates=int(summary["proposed_candidates"]),
        screened_candidates=int(summary["screened_candidates"]),
        promoted_candidates=int(summary["promoted_candidates"]),
        selected_candidate_ids=tuple(summary["selected_candidate_ids"]),
        plugin_native_selected_candidate_ids=tuple(
            summary.get(
                "plugin_native_selected_candidate_ids",
                summary["selected_candidate_ids"],
            )
        ),
        logical_task_executions=int(summary["logical_task_executions"]),
        physical_provider_executions=0,
        budget_limit=(
            None if summary.get("budget_limit") is None else int(summary["budget_limit"])
        ),
        budget_complete=bool(summary["budget_complete"]),
        rejected_materializations=int(summary["rejected_materializations"]),
    )


def _run_method(
    *,
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    phase: PhaseRuntime,
    method_id: str,
    plugin_id: str,
    configuration: Mapping[str, Any],
    search_seed: int,
    output: Path,
    cache: ResultCache,
    policy: NetworkPolicy,
    smoke: bool,
    smoke_candidate_limit: int | None = None,
    smoke_blocks_per_target: int | None = None,
) -> MethodRunSummary:
    resumed = _resume_method_if_complete(
        output=output,
        manifest=manifest,
        method_id=method_id,
        search_seed=search_seed,
        cache=cache,
        smoke=smoke,
    )
    if resumed is not None:
        return resumed
    ranges = _normalization_ranges(configuration, strict=not smoke)
    registry = PluginRegistry.from_manifest(
        configuration,
        normalization_ranges=ranges,
        strict_frozen_inputs=not smoke,
    )
    plugin = registry.get(plugin_id)
    budget_spec = configuration["budgets"]["search_total_per_method"]
    budget_limit = int(budget_spec["task_executions"])
    maximum_candidates = int(budget_spec["maximum_unique_screened_candidates"])
    maximum_promoted = int(budget_spec["maximum_promoted_candidates"])
    if smoke:
        configured_limit = 2 if smoke_candidate_limit is None else smoke_candidate_limit
        if (
            isinstance(configured_limit, bool)
            or not isinstance(configured_limit, int)
            or configured_limit < 1
        ):
            raise ExperimentRunError("smoke candidate limit must be a positive integer")
        maximum_candidates = min(maximum_candidates, configured_limit)
        maximum_promoted = min(maximum_promoted, 1)

    patch_ids = tuple(patch.patch_id for patch in runtime.patches)
    conditioned_generation = (
        plugin_id == "paretoskill"
        and configuration["methods"]["paretoskill"].get(
            "generation_conditioned_on_archive"
        )
        is True
    )
    initial_candidate_limit = maximum_candidates
    if conditioned_generation:
        initial_candidate_limit = (
            1
            if smoke
            else min(maximum_candidates, max(4, min(20, len(runtime.patches) + 2)))
        )
    screen_spec = configuration["budgets"]["screen"]
    # Two blocks are the minimum useful uncertainty smoke and also permit the
    # frozen hard/easy selector to form disjoint probes.
    if smoke:
        screen_per_target = (
            2 if smoke_blocks_per_target is None else smoke_blocks_per_target
        )
        if (
            isinstance(screen_per_target, bool)
            or not isinstance(screen_per_target, int)
            or screen_per_target < 2
        ):
            raise ExperimentRunError(
                "smoke blocks_per_target must be an integer of at least two"
            )
    else:
        screen_per_target = int(screen_spec["tasks_per_target"])
    screen_seeds = {
        int(seed) for seed in screen_spec.get("selected_execution_seeds", [17])
    }
    screen_blocks = _select_screen_blocks(
        phase,
        per_target=screen_per_target,
        seeds=screen_seeds,
        salt=str(screen_spec["selection_salt"]),
    )
    cache_before = cache.keys()
    ctx = plugin_id == "ctx2skill_hard_easy_product"
    controller_spec = (
        _subset_controller(
            runtime,
            plugin,
            plugin_id,
            patch_ids,
            maximum_candidates=maximum_candidates,
            search_seed=search_seed,
            configuration=configuration,
        )
        if not smoke
        and plugin_id
        in {
            "trace2skill_accuracy_subset",
            "evoskill_scalar_topk",
            "skillmoo_nsga2",
            "mocha_chebyshev_hvc",
        }
        else None
    )
    controller: SubsetSearchController | None = None
    subsets: tuple[tuple[str, ...], ...] = ()
    if controller_spec is not None:
        controller, controller_batch_size = controller_spec
        controller_progress = _restore_subset_controller_progress(
            expected_controller=controller,
            manifest=manifest,
            runtime=runtime,
            phase=phase,
            configuration=configuration,
            method_id=method_id,
            search_seed=search_seed,
            maximum_candidates=maximum_candidates,
            batch_size=controller_batch_size,
            patch_ids=patch_ids,
            screen_blocks=screen_blocks,
            cache=cache,
            output=output,
        )
        if controller_progress is not None:
            controller = controller_progress.controller
        (
            execution_candidates,
            store,
            rejected,
            screen_records,
            screen_summaries,
            screen_scored,
            proposed_candidates,
        ) = _run_subset_controller(
            controller=controller,
            batch_size=controller_batch_size,
            manifest=manifest,
            runtime=runtime,
            phase=phase,
            configuration=configuration,
            method_id=method_id,
            search_seed=search_seed,
            maximum_candidates=maximum_candidates,
            screen_blocks=screen_blocks,
            cache=cache,
            policy=policy,
            output=output,
            progress=controller_progress,
        )
    else:
        subsets = _static_subsets(
            plugin,
            plugin_id,
            patch_ids,
            maximum_candidates=initial_candidate_limit,
            seed=search_seed,
            configuration=configuration,
            smoke=smoke,
        )
        versions, store, rejected = _materialize_subsets(
            runtime.base,
            runtime.patches,
            runtime.evidence,
            subsets,
            label=method_id.replace("/", "-"),
        )
        versions = dict(list(versions.items())[:maximum_candidates])
        if plugin_id == "base_skill":
            execution_candidates = {runtime.base.lineage.version_id: runtime.base}
            candidate_values: tuple[SkillVersion | EvaluationCandidate, ...] = ()
        elif plugin_id == "no_skill":
            no_skill = EvaluationCandidate.no_skill(runtime.base)
            execution_candidates = {no_skill.candidate_id: no_skill}
            candidate_values = (no_skill,)
        else:
            execution_candidates = dict(versions)
            candidate_values = tuple(versions.values())
        if not execution_candidates:
            raise ExperimentRunError(
                f"method {method_id!r} produced no unique materializable candidates"
            )
        screen_records = _evaluate(
            experiment_id=manifest.experiment_id,
            base=runtime.base,
            candidates=candidate_values,
            phase=phase,
            blocks=screen_blocks,
            providers=runtime.providers,
            cache=cache,
            policy=policy,
            retry_limit=configuration["task_seed_blocks"]["retry_limit"],
            failure_event_path=output / "failure_events.jsonl",
        )
        screen_summaries = _summaries(
            screen_records,
            execution_candidates,
            targets=phase.targets,
            configuration=configuration,
            smoke=smoke,
        )
        screen_scored = _scored_candidates(
            execution_candidates,
            screen_summaries,
            records=screen_records,
            configuration=configuration,
            include_ctx_rates=ctx and smoke,
        )
        proposed_candidates = len(subsets)
    if not execution_candidates:
        raise ExperimentRunError(
            f"method {method_id!r} produced no unique materializable candidates"
        )
    generated_candidates = 0
    search_archive: ParetoArchive | None = None
    proposal_events: list[Mapping[str, Any]] = []
    if conditioned_generation:
        (
            screen_records,
            screen_summaries,
            screen_scored,
            generated_candidates,
            search_archive,
            proposal_events,
        ) = _archive_conditioned_expand(
            manifest=manifest,
            runtime=runtime,
            phase=phase,
            configuration=configuration,
            search_seed=search_seed,
            maximum_candidates=maximum_candidates,
            screen_blocks=screen_blocks,
            cache=cache,
            policy=policy,
            store=store,
            execution_candidates=execution_candidates,
            screen_records=screen_records,
            screen_summaries=screen_summaries,
            screen_scored=screen_scored,
            rejected=rejected,
            smoke=smoke,
            output=output,
        )
    promotion_plugin: BaselinePlugin = (
        AccuracyOnlyPlugin()
        if ctx and not smoke
        else plugin
    )
    promotion_pool = _controller_promotion_pool(controller, screen_scored)
    if isinstance(controller, (MOCHAController, NSGAIIController)) and not promotion_pool:
        method_name = "MOCHA" if isinstance(controller, MOCHAController) else "NSGA-II"
        raise ExperimentRunError(
            f"{method_name} retained no valid materialized candidate for promotion"
        )
    promoted_scored = promotion_plugin.select(
        promotion_pool,
        count=min(maximum_promoted, len(promotion_pool)),
    )
    promoted = {
        candidate.candidate_id: execution_candidates[candidate.candidate_id]
        for candidate in promoted_scored
    }
    promoted_values = tuple(
        value
        for value in promoted.values()
        if not (
            isinstance(value, SkillVersion)
            and value.lineage.version_id == runtime.base.lineage.version_id
        )
    )
    full_blocks = screen_blocks if smoke else phase.blocks
    full_records = _evaluate(
        experiment_id=manifest.experiment_id,
        base=runtime.base,
        candidates=promoted_values,
        phase=phase,
        blocks=full_blocks,
        providers=runtime.providers,
        cache=cache,
        policy=policy,
        retry_limit=configuration["task_seed_blocks"]["retry_limit"],
        failure_event_path=output / "failure_events.jsonl",
    )
    full_summaries = _summaries(
        full_records,
        promoted,
        targets=phase.targets,
        configuration=configuration,
        smoke=smoke,
    )
    full_scored = _scored_candidates(
        promoted,
        full_summaries,
        records=full_records,
        configuration=configuration,
        include_ctx_rates=ctx,
    )
    base_full_summary = _summaries(
        full_records,
        (runtime.base.lineage.version_id,),
        targets=phase.targets,
        configuration=configuration,
        smoke=smoke,
    )[runtime.base.lineage.version_id]
    archive_method = plugin_id in {"passive_archive", "paretoskill"}
    selected_count = min(
        len(full_scored),
        int(configuration["selection_protocol"]["archive_capacity"]["max_entries"])
        if archive_method
        else 1,
    )
    selected = plugin.select(full_scored, count=max(1, selected_count))
    plugin_native_selected_ids = tuple(item.candidate_id for item in selected)
    deployment_payload = _deployment_payload(
        method_id=method_id,
        search_seed=search_seed,
        full_scored=full_scored,
        base_summary=base_full_summary,
        configuration=configuration,
        ranges=ranges,
    )
    try:
        frozen_selected_ids = tuple(
            sorted(
                set(plugin_native_selected_ids)
                | set(deployment_candidate_union(deployment_payload))
            )
        )
    except FinalAnalysisError as exc:
        raise ExperimentRunError(
            f"deployment selection cannot be frozen for final evaluation: {exc}"
        ) from exc
    unknown_frozen = sorted(set(frozen_selected_ids) - set(execution_candidates))
    if unknown_frozen:
        raise ExperimentRunError(
            "deployment selection references unknown materializations: "
            f"{unknown_frozen!r}"
        )
    deployment_policy_ids: dict[str, list[str]] = {}
    raw_policy_selections = deployment_payload.get("policies", {})
    if isinstance(raw_policy_selections, Mapping):
        for policy_id, raw_selection in raw_policy_selections.items():
            if not isinstance(policy_id, str) or not isinstance(raw_selection, Mapping):
                continue
            candidate_id = raw_selection.get("candidate_id")
            if raw_selection.get("defined") is True and isinstance(candidate_id, str):
                deployment_policy_ids.setdefault(candidate_id, []).append(policy_id)

    screen_candidate_rows = [
        row for row in screen_records if row.candidate_id in execution_candidates
    ]
    full_candidate_rows = [row for row in full_records if row.candidate_id in promoted]
    screen_identities = {row.evaluation_identity for row in screen_candidate_rows}
    incremental_full = sum(
        row.evaluation_identity not in screen_identities for row in full_candidate_rows
    )
    logical_spent = len(screen_candidate_rows) + incremental_full
    comparison_role = configuration["methods"].get(method_id, {}).get(
        "comparison_role"
    )
    matched = comparison_role in {"matched_search", "matched_style_adaptation"}
    if method_id.startswith("fixed_scalarization_") or method_id.startswith("ablation_"):
        matched = True
    if logical_spent > budget_limit:
        raise ExperimentRunError(
            f"method {method_id!r} exceeded logical budget: {logical_spent}>{budget_limit}"
        )
    budget_complete = not matched or logical_spent == budget_limit or smoke

    output.mkdir(parents=True, exist_ok=True)
    JsonlResultStore(output / "screen_task_outcomes.jsonl").replace(screen_records)
    JsonlResultStore(output / "full_task_outcomes.jsonl").replace(full_records)
    _atomic_jsonl(
        output / "candidates.jsonl",
        [
            runtime.base.to_dict(),
            *[
                candidate.version.to_dict()
                if isinstance(candidate, EvaluationCandidate)
                else candidate.to_dict()
                for candidate_id, candidate in sorted(execution_candidates.items())
                if candidate_id != runtime.base.lineage.version_id
            ],
        ],
    )
    _atomic_jsonl(
        output / "selected_candidates.jsonl",
        [
            {
                "schema_version": 1,
                "method_id": method_id,
                "search_seed": search_seed,
                "candidate_id": candidate_id,
                "injection_mode": (
                    execution_candidates[candidate_id].injection_mode
                    if isinstance(
                        execution_candidates[candidate_id], EvaluationCandidate
                    )
                    else "skill"
                ),
                "version": (
                    execution_candidates[candidate_id].version.to_dict()
                    if isinstance(
                        execution_candidates[candidate_id], EvaluationCandidate
                    )
                    else execution_candidates[candidate_id].to_dict()
                ),
                "selection_sources": {
                    "plugin_native": candidate_id in plugin_native_selected_ids,
                    "deployment_policy_ids": sorted(
                        deployment_policy_ids.get(candidate_id, [])
                    ),
                    "primary_deployment_policy": (
                        candidate_id == deployment_payload.get("primary_candidate_id")
                    ),
                },
            }
            for candidate_id in frozen_selected_ids
        ],
    )
    store.save(output / "materializations.json")
    if archive_method and search_archive is None:
        search_archive = _new_archive(configuration)
        _admit_scored(search_archive, screen_scored)
    if search_archive is not None:
        search_archive.save(output / "screen_archive.json")
    final_archive: ParetoArchive | None = None
    if archive_method:
        final_archive = _new_archive(configuration)
        _admit_scored(final_archive, full_scored)
        final_archive.save(output / "archive.json")
        _atomic_json(
            output / "scientific_front.json",
            {
                "schema_version": 1,
                "entries": [entry.to_dict() for entry in final_archive.scientific_front],
            },
        )
    normalized_reference = (
        _hypervolume_reference(configuration) if ranges is not None else None
    )
    feasible_full = [
        candidate
        for candidate in full_scored
        if candidate.metadata.get("feasible") is True
    ]
    false_admission_analysis = None
    if search_archive is not None and final_archive is not None:
        try:
            false_admission_analysis = validated_false_archive_admission_rate(
                (
                    entry.candidate_id
                    for entry in search_archive.entries
                    if entry.candidate_id != runtime.base.lineage.version_id
                ),
                (candidate.candidate_id for candidate in full_scored),
                (
                    entry.candidate_id
                    for entry in final_archive.scientific_front
                    if entry.candidate_id != runtime.base.lineage.version_id
                ),
            )
        except FinalAnalysisError as exc:
            raise ExperimentRunError(
                f"false archive admission accounting is inconsistent: {exc}"
            ) from exc
    frontier_analysis = {
        "normalization_defined": ranges is not None,
        "feasible_hypervolume": (
            _feasible_hypervolume(
                full_scored,
                ranges=ranges,
                reference=normalized_reference,
            )
            if ranges is not None and normalized_reference is not None
            else None
        ),
        "feasibility_rate": (
            len(feasible_full) / len(full_scored) if full_scored else 0.0
        ),
        "feasible_front_size": (
            len(
                nondominated(
                    _normalized_scored_vectors(
                        feasible_full, ranges, feasible_only=False
                    ).values()
                )
            )
            if ranges is not None and feasible_full
            else 0
        ),
        "working_archive_size": (
            len(final_archive.entries) if final_archive is not None else None
        ),
        "scientific_front_size": (
            len(final_archive.scientific_front) if final_archive is not None else None
        ),
        "false_archive_admission_rate": (
            false_admission_analysis.rate.value
            if false_admission_analysis is not None
            else None
        ),
        "false_archive_admission": (
            false_admission_analysis.to_dict()
            if false_admission_analysis is not None
            else None
        ),
        "budget_curve": _budget_curve(
            candidate_order=execution_candidates,
            screen_scored=screen_scored,
            promoted_order=promoted,
            full_scored=full_scored,
            configuration=configuration,
            ranges=ranges,
        ),
    }
    _atomic_json(output / "deployment_selection.json", deployment_payload)
    _atomic_jsonl(output / "proposal_events.jsonl", proposal_events)
    if controller is not None:
        _atomic_json(
            output / "search_controller.json",
            _search_controller_checkpoint_payload(
                manifest=manifest,
                method_id=method_id,
                search_seed=search_seed,
                execution_complete=True,
                maximum_candidates=maximum_candidates,
                batch_size=controller_batch_size,
                patch_ids=patch_ids,
                controller=controller,
                proposed=proposed_candidates,
                execution_candidates=execution_candidates,
                store=store,
                rejected=rejected,
                screen_records=screen_records,
            ),
        )
    _atomic_json(
        output / "metrics.json",
        {
            "schema_version": 1,
            "method_id": method_id,
            "search_seed": search_seed,
            "screen": {
                candidate_id: summary.to_dict()
                for candidate_id, summary in sorted(screen_summaries.items())
            },
            "full": {
                candidate_id: summary.to_dict()
                for candidate_id, summary in sorted(full_summaries.items())
            },
            "selected_candidate_ids": list(frozen_selected_ids),
            "plugin_native_selected_candidate_ids": list(
                plugin_native_selected_ids
            ),
            "final_frozen_candidate_ids": list(frozen_selected_ids),
            "screen_candidate_order": list(execution_candidates),
            "promoted_candidate_order": list(promoted),
            "frontier_analysis": frontier_analysis,
            "deployment_selection": deployment_payload,
            "proposal_accounting": {
                "physical_provider_executions": sum(
                    event.get("cache_hit") is not True for event in proposal_events
                ),
                "cache_hits": sum(
                    event.get("cache_hit") is True for event in proposal_events
                ),
                "input_tokens": sum(
                    int(event.get("input_tokens", 0)) for event in proposal_events
                ),
                "output_tokens": sum(
                    int(event.get("output_tokens", 0)) for event in proposal_events
                ),
                "latency_ms": sum(
                    float(event.get("latency_ms", 0.0)) for event in proposal_events
                ),
            },
        },
    )
    cache_after = cache.keys()
    proposal_provider_executions = sum(
        event.get("cache_hit") is not True for event in proposal_events
    )
    summary = MethodRunSummary(
        method_id=method_id,
        search_seed=search_seed,
        output_directory=output,
        proposed_candidates=proposed_candidates + generated_candidates,
        screened_candidates=len(execution_candidates),
        promoted_candidates=len(promoted),
        selected_candidate_ids=frozen_selected_ids,
        plugin_native_selected_candidate_ids=plugin_native_selected_ids,
        logical_task_executions=logical_spent,
        physical_provider_executions=(
            len(cache_after - cache_before) + proposal_provider_executions
        ),
        budget_limit=budget_limit if matched else None,
        budget_complete=budget_complete,
        rejected_materializations=len(rejected),
    )
    _atomic_json(
        output / "run_state.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "complete": summary.budget_complete,
            "execution_complete": True,
            "smoke": smoke,
            "summary": summary.to_dict(),
            "rejected_materializations": rejected,
            "cache_keys": sorted(
                {record.cache_key for record in (*screen_records, *full_records)}
            ),
        },
    )
    return summary


def run_configured_search(
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    *,
    output_root: str | Path,
    policy: NetworkPolicy,
    method_ids: Iterable[str] | None = None,
    search_seeds: Iterable[int] | None = None,
    smoke: bool = False,
) -> ConfiguredRunSummary:
    """Execute the configured search matrix using local assets and supplied adapters."""

    if "search" not in runtime.phases:
        raise ExperimentRunError("runtime has no search phase")
    configuration = manifest.data
    requested_methods = None if method_ids is None else tuple(method_ids)
    requested_seeds = None if search_seeds is None else tuple(search_seeds)
    smoke_spec: Mapping[str, Any] = {}
    smoke_candidate_limits: Mapping[str, Any] = {}
    smoke_blocks_per_target: int | None = None
    smoke_namespace: str | None = None
    smoke_ceiling: int | None = None
    if smoke:
        runner = configuration.get("runner", {})
        if not isinstance(runner, Mapping):
            raise ExperimentRunError("runner must be a mapping")
        raw_smoke = runner.get("smoke")
        if raw_smoke is not None:
            if not isinstance(raw_smoke, Mapping):
                raise ExperimentRunError("runner.smoke must be a mapping")
            smoke_spec = raw_smoke
            declared_methods = smoke_spec.get("methods")
            declared_seeds = smoke_spec.get("search_seeds")
            if (
                not isinstance(declared_methods, list)
                or not declared_methods
                or any(not isinstance(value, str) or not value for value in declared_methods)
                or len(set(declared_methods)) != len(declared_methods)
            ):
                raise ExperimentRunError("runner.smoke.methods must be a unique string list")
            if (
                not isinstance(declared_seeds, list)
                or not declared_seeds
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in declared_seeds
                )
                or len(set(declared_seeds)) != len(declared_seeds)
            ):
                raise ExperimentRunError(
                    "runner.smoke.search_seeds must be a unique integer list"
                )
            if requested_methods is None:
                requested_methods = tuple(declared_methods)
            else:
                if (
                    not requested_methods
                    or any(
                        not isinstance(value, str) or not value
                        for value in requested_methods
                    )
                    or len(set(requested_methods)) != len(requested_methods)
                ):
                    raise ExperimentRunError(
                        "requested smoke methods must be a unique string sequence"
                    )
                if not set(requested_methods) <= set(declared_methods):
                    raise ExperimentRunError(
                        "smoke methods must be a subset of runner.smoke.methods"
                    )
            if requested_seeds is None:
                requested_seeds = tuple(declared_seeds)
            else:
                if (
                    not requested_seeds
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in requested_seeds
                    )
                    or len(set(requested_seeds)) != len(requested_seeds)
                ):
                    raise ExperimentRunError(
                        "requested smoke seeds must be a unique integer sequence"
                    )
                if not set(requested_seeds) <= set(declared_seeds):
                    raise ExperimentRunError(
                        "smoke seeds must be a subset of runner.smoke.search_seeds"
                    )
            maximum = smoke_spec.get("max_candidates")
            blocks_per_target = smoke_spec.get("blocks_per_target")
            ceiling = smoke_spec.get("logical_task_execution_ceiling")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in (maximum, ceiling)
            ) or (
                isinstance(blocks_per_target, bool)
                or not isinstance(blocks_per_target, int)
                or blocks_per_target < 2
            ):
                raise ExperimentRunError(
                    "runner.smoke candidate/block/ceiling values are invalid"
                )
            smoke_blocks_per_target = blocks_per_target
            smoke_ceiling = ceiling
            raw_limits = smoke_spec.get("candidate_limits", {})
            if not isinstance(raw_limits, Mapping) or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
                for key, value in raw_limits.items()
            ):
                raise ExperimentRunError(
                    "runner.smoke.candidate_limits must contain positive bounded integers"
                )
            smoke_candidate_limits = raw_limits
            namespace = smoke_spec.get("separate_output_namespace")
            if not isinstance(namespace, str) or not namespace.strip() or Path(
                namespace
            ).name != namespace:
                raise ExperimentRunError(
                    "runner.smoke.separate_output_namespace must be one path segment"
                )
            smoke_namespace = namespace
        elif requested_methods is None or requested_seeds is None:
            raise ExperimentRunError(
                "a manifest without runner.smoke must receive explicit methods and seeds"
            )
    _validate_search_budget_matrix(
        configuration,
        runtime.phases["search"],
        smoke=smoke,
    )
    raw_seeds = tuple(
        requested_seeds
        if requested_seeds is not None
        else configuration["task_seed_blocks"]["search_seeds"]
    )
    if (
        not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ExperimentRunError("search seeds must be a non-empty unique sequence")
    seeds = raw_seeds
    if requested_methods is not None and (
        not requested_methods
        or any(not isinstance(value, str) or not value for value in requested_methods)
        or len(set(requested_methods)) != len(requested_methods)
    ):
        raise ExperimentRunError("requested methods must be a non-empty unique sequence")
    specs = _method_specs(configuration, requested_methods)
    if smoke_spec:
        declared_targets = smoke_spec.get("search_targets")
        if (
            not isinstance(declared_targets, list)
            or not declared_targets
            or any(not isinstance(value, str) for value in declared_targets)
            or set(declared_targets)
            != {target.target_id for target in runtime.phases["search"].targets}
        ):
            raise ExperimentRunError(
                "smoke runtime targets do not match runner.smoke.search_targets"
            )
        assert smoke_blocks_per_target is not None
        screen_spec = configuration["budgets"]["screen"]
        smoke_blocks = _select_screen_blocks(
            runtime.phases["search"],
            per_target=smoke_blocks_per_target,
            seeds={
                int(seed)
                for seed in screen_spec.get("selected_execution_seeds", [17])
            },
            salt=str(screen_spec["selection_salt"]),
        )
        matrix_size = _matrix_size(
            runtime.phases["search"].targets, smoke_blocks
        )
        default_limit = int(smoke_spec["max_candidates"])
        maximum_logical = 0
        for method_id, plugin_id, _ in specs:
            run_count = 1 if plugin_id in {"no_skill", "base_skill"} else len(seeds)
            if plugin_id == "base_skill":
                candidate_limit = 0
            elif plugin_id == "no_skill":
                candidate_limit = 1
            else:
                candidate_limit = int(
                    smoke_candidate_limits.get(method_id, default_limit)
                )
            maximum_logical += run_count * candidate_limit * matrix_size
        assert smoke_ceiling is not None
        if maximum_logical > smoke_ceiling:
            raise ExperimentRunError(
                "declared smoke method×seed matrix can exceed its logical ceiling: "
                f"{maximum_logical}>{smoke_ceiling}; no provider request was made"
            )
    selected_plugins = {plugin_id for _, plugin_id, _ in specs}
    if (
        not smoke
        and "trace2skill_accuracy_subset" in selected_plugins
        and runtime.binary_optimizer_factory is None
    ):
        raise ExternalOptimizerRequired(
            "the selected formal comparison includes Trace2Skill-style binary "
            "Bayesian search, but no frozen BinarySubsetBayesianAdapter is installed; "
            "no provider execution was started"
        )
    if not smoke and "skillmoo_nsga2" in selected_plugins:
        population = int(
            configuration["methods"]["skillmoo_nsga2"]["population_size"]
        )
        subset_space = (1 << len({patch.patch_id for patch in runtime.patches})) - 1
        if subset_space < population:
            raise ExperimentRunError(
                "the patch pool is too small for the frozen NSGA-II population"
            )
    root = Path(output_root).resolve() / manifest.experiment_id
    if smoke_namespace is not None:
        root /= smoke_namespace
    root.mkdir(parents=True, exist_ok=True)
    stage: RunStage = "smoke" if smoke else "search"
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest.save(root / "resolved_manifest.yaml")
    _atomic_json(
        root / "run_metadata.json",
        _run_metadata(
            manifest,
            stage=stage,
            started_at_utc=started_at_utc,
            complete=False,
        ),
    )
    cache = ResultCache(root / "cache")
    summaries: list[MethodRunSummary] = []
    for method_id, plugin_id, effective_configuration in specs:
        run_seeds = seeds[:1] if plugin_id in {"no_skill", "base_skill"} else seeds
        for seed in run_seeds:
            summaries.append(
                _run_method(
                    manifest=manifest,
                    runtime=runtime,
                    phase=runtime.phases["search"],
                    method_id=method_id,
                    plugin_id=plugin_id,
                    configuration=effective_configuration,
                    search_seed=seed,
                    output=root / "methods" / method_id / f"seed-{seed}",
                    cache=cache,
                    policy=policy,
                    smoke=smoke,
                    smoke_candidate_limit=(
                        int(
                            smoke_candidate_limits.get(
                                method_id, smoke_spec["max_candidates"]
                            )
                        )
                        if smoke_spec and plugin_id not in {"no_skill", "base_skill"}
                        else None
                    ),
                    smoke_blocks_per_target=smoke_blocks_per_target,
                )
            )
    result = ConfiguredRunSummary(
        experiment_id=manifest.experiment_id,
        stage=stage,
        output_directory=root,
        method_runs=tuple(summaries),
    )
    _write_search_output_contract(root, manifest, result)
    _atomic_json(root / "run_summary.json", result.to_dict())
    _atomic_json(
        root / "run_metadata.json",
        _run_metadata(
            manifest,
            stage=stage,
            started_at_utc=started_at_utc,
            complete=True,
        ),
    )
    return result


def _expected_search_run_matrix(
    configuration: Mapping[str, Any],
) -> set[tuple[str, int]]:
    runner = configuration.get("runner", {})
    final_spec = runner.get("final", {}) if isinstance(runner, Mapping) else {}
    if not isinstance(final_spec, Mapping):
        raise ExperimentRunError("runner.final must be a mapping")
    required_methods = final_spec.get("required_method_ids")
    if required_methods is not None and (
        not isinstance(required_methods, list)
        or not required_methods
        or any(not isinstance(value, str) for value in required_methods)
    ):
        raise ExperimentRunError("runner.final.required_method_ids is malformed")
    raw_seeds = final_spec.get(
        "required_search_seeds",
        configuration["task_seed_blocks"]["search_seeds"],
    )
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ExperimentRunError("runner final search-seed requirement is malformed")
    specs = _method_specs(configuration, required_methods)
    expected: set[tuple[str, int]] = set()
    for method_id, plugin_id, _ in specs:
        run_seeds = raw_seeds[:1] if plugin_id in {"no_skill", "base_skill"} else raw_seeds
        expected.update((method_id, int(seed)) for seed in run_seeds)
    return expected


def _load_selected_conditions(
    root: Path,
    *,
    configuration: Mapping[str, Any],
    experiment_id: str,
) -> tuple[
    dict[str, SkillVersion | EvaluationCandidate],
    dict[str, list[dict[str, Any]]],
]:
    conditions: dict[str, SkillVersion | EvaluationCandidate] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    identity_to_id: dict[tuple[str, str], str] = {}
    observed_runs: set[tuple[str, int]] = set()
    paths = sorted((root / "methods").glob("*/seed-*/selected_candidates.jsonl"))
    if not paths:
        raise ExperimentRunError(
            "final evaluation requires completed search selected_candidates.jsonl files"
        )
    for path in paths:
        state_path = path.with_name("run_state.json")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentRunError(
                f"selected candidates have no valid run state: {state_path}"
            ) from exc
        if not isinstance(state, Mapping) or state.get("execution_complete") is not True:
            raise ExperimentRunError(f"search execution is incomplete: {state_path}")
        if state.get("experiment_id") != experiment_id:
            raise ExperimentRunError(f"search execution has wrong experiment id: {state_path}")
        if state.get("smoke") is True:
            raise ExperimentRunError(
                f"smoke candidates cannot enter final evaluation: {state_path}"
            )
        if state.get("complete") is not True:
            raise ExperimentRunError(
                f"budget-incomplete search cannot enter final evaluation: {state_path}"
            )
        summary = state.get("summary")
        if not isinstance(summary, Mapping):
            raise ExperimentRunError(f"search run summary is malformed: {state_path}")
        method_id = summary.get("method_id")
        search_seed = summary.get("search_seed")
        if (
            not isinstance(method_id, str)
            or isinstance(search_seed, bool)
            or not isinstance(search_seed, int)
        ):
            raise ExperimentRunError(f"search run identity is malformed: {state_path}")
        run_identity = (method_id, search_seed)
        if run_identity in observed_runs:
            raise ExperimentRunError(f"duplicate completed search run: {run_identity}")
        observed_runs.add(run_identity)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping) or row.get("schema_version") != 1:
                    raise ValueError("unsupported selected-candidate schema")
                candidate_id = row["candidate_id"]
                injection_mode = row["injection_mode"]
                version = SkillVersion.from_dict(row["version"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExperimentRunError(
                    f"invalid selected candidate at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(candidate_id, str) or injection_mode not in {"skill", "none"}:
                raise ExperimentRunError(
                    f"invalid selected execution condition at {path}:{line_number}"
                )
            identity = (
                version.skill.content_hash if injection_mode == "skill" else "no-skill",
                injection_mode,
            )
            canonical_id = identity_to_id.setdefault(identity, candidate_id)
            if canonical_id not in conditions:
                conditions[canonical_id] = (
                    EvaluationCandidate.no_skill(version, candidate_id=canonical_id)
                    if injection_mode == "none"
                    else version
                )
            sources.setdefault(canonical_id, []).append(
                {
                    "method_id": row.get("method_id"),
                    "search_seed": row.get("search_seed"),
                    "source_candidate_id": candidate_id,
                    "selection_sources": row.get("selection_sources", {}),
                    "path": str(path),
                }
            )
    expected_runs = _expected_search_run_matrix(configuration)
    if observed_runs != expected_runs:
        missing = sorted(expected_runs - observed_runs)
        extra = sorted(observed_runs - expected_runs)
        raise ExperimentRunError(
            "final evaluation requires the exact frozen method×seed matrix; "
            f"missing={missing}, extra={extra}"
        )
    return conditions, sources


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_final_statistics(
    records: Iterable[EvaluationRecord],
    candidate_ids: Iterable[str],
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rows = tuple(records)
    base = {record.block_key: record for record in rows if record.is_base}
    alpha = 1.0 - confidence_level
    output: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        candidate_rows = [record for record in rows if record.candidate_id == candidate_id]
        by_target: dict[str, list[tuple[EvaluationRecord, EvaluationRecord]]] = {}
        for record in candidate_rows:
            if record.block_key not in base:
                raise ExperimentRunError(
                    f"final row for {candidate_id!r} has no paired base execution"
                )
            by_target.setdefault(record.target_id, []).append(
                (record, base[record.block_key])
            )
        target_metrics: dict[str, Any] = {}
        for target_id, pairs in sorted(by_target.items()):
            task_clusters: dict[
                str, list[tuple[EvaluationRecord, EvaluationRecord]]
            ] = {}
            for pair in pairs:
                task_clusters.setdefault(pair[0].task_id, []).append(pair)
            task_ids = sorted(task_clusters)

            def estimates(
                sampled_tasks: list[str],
            ) -> tuple[float, float, float | None, float]:
                sampled = [pair for task_id in sampled_tasks for pair in task_clusters[task_id]]
                candidate_accuracy = sum(pair[0].result.correct for pair in sampled) / len(
                    sampled
                )
                base_accuracy = sum(pair[1].result.correct for pair in sampled) / len(sampled)
                eligible = [pair for pair in sampled if pair[1].result.correct]
                regression = (
                    sum(not pair[0].result.correct for pair in eligible) / len(eligible)
                    if eligible
                    else None
                )
                tokens = sum(pair[0].result.total_tokens for pair in sampled) / len(sampled)
                return candidate_accuracy, candidate_accuracy - base_accuracy, regression, tokens

            point = estimates(task_ids)
            bootstrap: list[tuple[float, float, float | None, float]] = []
            rng_seed = seed ^ int(
                hashlib.sha256(f"{candidate_id}\0{target_id}".encode()).hexdigest()[:16],
                16,
            )
            rng = random.Random(rng_seed)
            for _ in range(replicates):
                bootstrap.append(
                    estimates([task_ids[rng.randrange(len(task_ids))] for _ in task_ids])
                )

            def interval(index: int) -> list[float] | None:
                values = [
                    float(value[index])
                    for value in bootstrap
                    if value[index] is not None
                ]
                if not values:
                    return None
                return [
                    _percentile(values, alpha / 2.0),
                    _percentile(values, 1.0 - alpha / 2.0),
                ]

            target_metrics[target_id] = {
                "task_clusters": len(task_ids),
                "task_seed_rows": len(pairs),
                "accuracy": {"point": point[0], "ci": interval(0)},
                "paired_accuracy_delta": {"point": point[1], "ci": interval(1)},
                "paired_regression": {"point": point[2], "ci": interval(2)},
                "token_cost": {"point": point[3], "ci": interval(3)},
            }
        all_pairs = [pair for pairs in by_target.values() for pair in pairs]
        output[candidate_id] = {
            "targets": target_metrics,
            "diagnostic_summary": {
                "data_role": "per_target_diagnostic_only",
                "excluded_from_primary_inference": True,
                "target_count": len(target_metrics),
                "task_seed_rows": len(all_pairs),
                "pooled_mixed_target_metrics": None,
                "reason": (
                    "ID, transfer, and excluded diagnostic targets are not pooled; "
                    "use primary_analysis.objective_summaries"
                ),
            },
        }
    return output


def _base_final_objective_summary(
    partition: FinalRecordPartition,
    *,
    base_id: str,
    configuration: Mapping[str, Any],
) -> ObjectiveSummary:
    roles = dict(partition.target_roles)
    raw_targets = configuration.get("targets")
    if not isinstance(raw_targets, Mapping):
        raise FinalAnalysisError("configuration.targets must be a mapping")
    base_rows = [
        record
        for record in partition.primary_records
        if record.is_base and record.candidate_id == base_id
    ]
    if not base_rows:
        raise FinalAnalysisError("deduplicated final records contain no base rows")
    observations: list[PairedObservation] = []
    transfer_groups: set[str] = set()
    for record in base_rows:
        role = roles[record.target_id]
        if role not in {"id", "transfer"}:
            raise FinalAnalysisError("diagnostic row leaked into base objective summary")
        raw_target = raw_targets.get(record.target_id)
        if not isinstance(raw_target, Mapping):
            raise FinalAnalysisError(
                f"final target {record.target_id!r} is missing from configuration"
            )
        group = None
        if role == "transfer":
            group = record.transfer_group or str(
                raw_target.get("transfer_group") or record.target_id
            )
            transfer_groups.add(group)
        observations.append(
            PairedObservation(
                task_id=record.task_id,
                seed=record.seed,
                split=role,
                target=record.target_id,
                group=group,
                candidate_correct=record.result.correct,
                base_correct=record.result.correct,
                input_tokens=record.result.input_tokens,
                output_tokens=record.result.output_tokens,
            )
        )
    statistics = configuration["statistics"]
    return paired_bootstrap(
        observations,
        confidence_level=float(statistics["confidence_level"]),
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(configuration["task_seed_blocks"]["bootstrap_seed"]),
        expected_transfer_groups=transfer_groups,
        min_effective_blocks=int(
            statistics.get("minimum_effective_blocks_for_archive", 2)
        ),
        token_cost_upper_bound=float(
            configuration["constraints"]["token_budget"]["budget"]
        ),
    )


def _final_method_candidate_sets(
    sources: Mapping[str, list[dict[str, Any]]],
    *,
    available_candidate_ids: set[str],
    expected_runs: set[tuple[str, int]],
) -> dict[tuple[str, int], tuple[str, ...]]:
    by_run: dict[tuple[str, int], set[str]] = {
        run_identity: set() for run_identity in expected_runs
    }
    for candidate_id, candidate_sources in sources.items():
        if candidate_id not in available_candidate_ids:
            continue
        for source in candidate_sources:
            method_id = source.get("method_id")
            search_seed = source.get("search_seed")
            if (
                not isinstance(method_id, str)
                or isinstance(search_seed, bool)
                or not isinstance(search_seed, int)
            ):
                continue
            run_identity = (method_id, search_seed)
            if run_identity in by_run:
                by_run[run_identity].add(candidate_id)
    empty = sorted(run_identity for run_identity, ids in by_run.items() if not ids)
    if empty:
        raise FinalAnalysisError(
            f"completed search runs have no frozen final candidates: {empty!r}"
        )
    return {
        run_identity: tuple(sorted(candidate_ids))
        for run_identity, candidate_ids in sorted(by_run.items())
    }


def _final_primary_analysis(
    records: Iterable[EvaluationRecord],
    candidate_ids: Iterable[str],
    *,
    base_id: str,
    sources: Mapping[str, list[dict[str, Any]]],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_ids = tuple(sorted(set(candidate_ids) - {base_id}))
    partition = partition_final_records(records, configuration)
    summaries = build_final_objective_summaries(
        records,
        configuration,
        candidate_ids=treatment_ids,
    )
    summaries[base_id] = _base_final_objective_summary(
        partition,
        base_id=base_id,
        configuration=configuration,
    )
    expected_runs = _expected_search_run_matrix(configuration)
    run_candidates = _final_method_candidate_sets(
        sources,
        available_candidate_ids=set(summaries),
        expected_runs=expected_runs,
    )
    heldout = analyze_heldout_fronts(
        summaries,
        run_candidates,
        configuration,
        constraints=_constraints(configuration),
    )

    comparison = configuration["statistics"]["heldout_comparisons"]
    holm_family = comparison.get("holm_family", {})
    raw_baselines = (
        holm_family.get("matched_run_ids", [])
        if isinstance(holm_family, Mapping)
        else []
    )
    runner = configuration.get("runner", {})
    final_runner = runner.get("final", {}) if isinstance(runner, Mapping) else {}
    raw_seeds = (
        final_runner.get("required_search_seeds")
        if isinstance(final_runner, Mapping)
        else None
    )
    if raw_seeds is None:
        raw_seeds = configuration["task_seed_blocks"]["search_seeds"]
    seeds = tuple(raw_seeds) if isinstance(raw_seeds, list) else ()
    if (
        len(seeds) == 3
        and len(set(seeds)) == 3
        and isinstance(raw_baselines, list)
        and raw_baselines
        and all(isinstance(method_id, str) for method_id in raw_baselines)
    ):
        run_hypervolumes = {
            (run.method_id, run.search_seed): run.hypervolume.value
            for run in heldout.runs
        }
        paired = paired_three_seed_sign_flip_holm(
            run_hypervolumes,
            treatment_method="paretoskill",
            comparison_methods=raw_baselines,
            seeds=seeds,
        )
        paired_payload = paired.to_dict()
        paired_payload["defined"] = paired.family_complete
        if not paired.family_complete:
            paired_payload["reason"] = "predeclared_holm_family_incomplete"
    else:
        paired_payload = {
            "defined": False,
            "family_complete": False,
            "reason": "exact paired inference requires three frozen seeds and a Holm family",
            "seeds": list(seeds),
            "contrasts": {},
        }
    return {
        "defined": True,
        "data_role": "heldout_final_only",
        "record_partition": partition.to_dict(),
        "objective_summaries": {
            candidate_id: summary.to_dict()
            for candidate_id, summary in sorted(summaries.items())
        },
        "method_seed_candidate_ids": {
            f"{method_id}::seed={search_seed}": list(ids)
            for (method_id, search_seed), ids in sorted(run_candidates.items())
        },
        "heldout_fronts": heldout.to_dict(),
        "paired_three_seed_sign_flip_holm": paired_payload,
        "primary_endpoint": comparison.get("primary_endpoint"),
    }


def _validate_final_budget_matrix(
    configuration: Mapping[str, Any],
    phase: PhaseRuntime,
    *,
    candidate_count: int,
) -> tuple[int, int]:
    final_budget = configuration["budgets"]["final"]
    expected_targets = final_budget.get("final_target_ids")
    if (
        not isinstance(expected_targets, list)
        or not expected_targets
        or any(not isinstance(target_id, str) for target_id in expected_targets)
        or len(set(expected_targets)) != len(expected_targets)
    ):
        raise ExperimentRunError("final_target_ids must be a non-empty unique list")
    observed_targets = [target.target_id for target in phase.targets]
    if observed_targets != expected_targets:
        raise ExperimentRunError(
            "final runtime targets do not match the frozen final_target_ids: "
            f"observed={observed_targets}, expected={expected_targets}"
        )
    configured_budget = final_budget.get("task_executions")
    if (
        isinstance(configured_budget, bool)
        or not isinstance(configured_budget, int)
        or configured_budget <= 0
    ):
        raise ExperimentRunError(
            "final task_executions must be a resolved positive integer"
        )
    if candidate_count < 1:
        raise ExperimentRunError("final evaluation requires at least one non-base condition")
    matrix_size = _matrix_size(phase.targets, phase.blocks)
    if matrix_size <= 0:
        raise ExperimentRunError("final runtime matrix is empty")
    for target in phase.targets:
        target_rows = sum(_compatible(target, block) for block in phase.blocks)
        if target_rows <= 0:
            raise ExperimentRunError(
                f"final target {target.target_id!r} has no compatible task blocks"
            )
    expected_logical = candidate_count * matrix_size
    if expected_logical != configured_budget:
        raise ExperimentRunError(
            "frozen final budget does not match candidate×target-task-seed matrix: "
            f"{candidate_count}×{matrix_size}={expected_logical}, "
            f"configured={configured_budget}; no provider request was made"
        )
    return configured_budget, expected_logical


def _validate_final_analysis_preflight(
    configuration: Mapping[str, Any], phase: PhaseRuntime
) -> None:
    """Reject malformed held-out analysis inputs before any provider request."""

    try:
        roles = resolve_final_target_roles(configuration)
        resolve_final_execution_groups(configuration)
        ranges = _normalization_ranges(configuration, strict=True)
        reference = _hypervolume_reference(configuration)
        _constraints(configuration)
    except (FinalAnalysisError, KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunError(f"invalid frozen final-analysis configuration: {exc}") from exc
    if ranges is None or any(maximum <= minimum for minimum, maximum in ranges):
        raise ExperimentRunError(
            "invalid frozen final-analysis configuration: normalization ranges "
            "must have positive width"
        )
    if not all(math.isfinite(value) for value in reference):
        raise ExperimentRunError(
            "invalid frozen final-analysis configuration: hypervolume reference "
            "must be finite"
        )
    runtime_target_ids = {target.target_id for target in phase.targets}
    if runtime_target_ids != set(roles):
        raise ExperimentRunError(
            "invalid frozen final-analysis configuration: runtime/config final "
            "target identities differ"
        )
    statistics = configuration.get("statistics")
    comparison = (
        statistics.get("heldout_comparisons")
        if isinstance(statistics, Mapping)
        else None
    )
    family = comparison.get("holm_family") if isinstance(comparison, Mapping) else None
    baselines = family.get("matched_run_ids") if isinstance(family, Mapping) else None
    if (
        not isinstance(baselines, list)
        or not baselines
        or any(not isinstance(method_id, str) or not method_id for method_id in baselines)
        or len(set(baselines)) != len(baselines)
        or "paretoskill" in baselines
    ):
        raise ExperimentRunError(
            "invalid frozen final-analysis configuration: Holm family must contain "
            "unique non-ParetoSkill method ids"
        )
    runner = configuration.get("runner", {})
    final_runner = runner.get("final", {}) if isinstance(runner, Mapping) else {}
    seeds = (
        final_runner.get("required_search_seeds")
        if isinstance(final_runner, Mapping)
        else None
    )
    if seeds is None:
        blocks = configuration.get("task_seed_blocks")
        seeds = blocks.get("search_seeds") if isinstance(blocks, Mapping) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ExperimentRunError(
            "invalid frozen final-analysis configuration: paired search seeds "
            "must be non-empty unique integers"
        )


def run_configured_final(
    manifest: ExperimentManifest,
    runtime: ExperimentRuntime,
    *,
    output_root: str | Path,
    policy: NetworkPolicy,
) -> FinalRunSummary:
    """Evaluate frozen search outputs on every declared final-only target."""

    if "final" not in runtime.phases:
        raise ExperimentRunError("runtime has no final phase")
    _validate_final_analysis_preflight(manifest.data, runtime.phases["final"])
    root = Path(output_root).resolve() / manifest.experiment_id
    conditions, sources = _load_selected_conditions(
        root,
        configuration=manifest.data,
        experiment_id=manifest.experiment_id,
    )
    base_id = runtime.base.lineage.version_id
    candidate_ids = sorted(conditions)
    if base_id not in candidate_ids:
        candidate_ids.insert(0, base_id)
        conditions[base_id] = runtime.base
        sources[base_id] = [{"method_id": "base_skill", "search_seed": None}]
    candidates = tuple(
        condition
        for candidate_id, condition in sorted(conditions.items())
        if candidate_id != base_id
    )
    final_output = root / "final"
    candidate_manifest = {
        "schema_version": 1,
        "experiment_id": manifest.experiment_id,
        "conditions": [
            {
                "candidate_id": candidate_id,
                "injection_mode": (
                    condition.injection_mode
                    if isinstance(condition, EvaluationCandidate)
                    else "skill"
                ),
                "content_hash": (
                    condition.content_hash
                    if isinstance(condition, EvaluationCandidate)
                    else condition.skill.content_hash
                ),
                "sources": sources[candidate_id],
            }
            for candidate_id, condition in sorted(conditions.items())
        ],
    }
    candidate_manifest["sha256"] = hashlib.sha256(
        canonical_json(candidate_manifest).encode("utf-8")
    ).hexdigest()
    manifest_path = final_output / "candidate_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != candidate_manifest:
            raise ExperimentRunError(
                "frozen final candidate manifest changed; choose a new experiment id"
            )
    _atomic_json(manifest_path, candidate_manifest)
    budget_limit, expected_logical = _validate_final_budget_matrix(
        manifest.data,
        runtime.phases["final"],
        candidate_count=len(candidates),
    )
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(
        root / "run_metadata.json",
        _run_metadata(
            manifest,
            stage="final",
            started_at_utc=started_at_utc,
            complete=False,
        ),
    )
    cache = ResultCache(root / "cache")
    cache_before = cache.keys()
    records = _evaluate(
        experiment_id=manifest.experiment_id,
        base=runtime.base,
        candidates=candidates,
        phase=runtime.phases["final"],
        blocks=runtime.phases["final"].blocks,
        providers=runtime.providers,
        cache=cache,
        policy=policy,
        retry_limit=manifest.data["task_seed_blocks"]["retry_limit"],
        failure_event_path=final_output / "failure_events.jsonl",
    )
    statistics = manifest.data["statistics"]
    final_diagnostics = _paired_final_statistics(
        records,
        candidate_ids,
        confidence_level=float(statistics["confidence_level"]),
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(manifest.data["task_seed_blocks"]["bootstrap_seed"]),
    )
    try:
        primary_analysis = _final_primary_analysis(
            records,
            candidate_ids,
            base_id=base_id,
            sources=sources,
            configuration=manifest.data,
        )
    except (FinalAnalysisError, ValueError) as exc:
        raise ExperimentRunError(f"cannot compute held-out final analysis: {exc}") from exc
    JsonlResultStore(final_output / "task_outcomes.jsonl").replace(records)
    _atomic_json(
        final_output / "metrics.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "paired_design": True,
            "bootstrap_replicates": int(statistics["bootstrap_replicates"]),
            "confidence_level": float(statistics["confidence_level"]),
            "primary_analysis": primary_analysis,
            "candidate_target_diagnostics": final_diagnostics,
            "candidates": final_diagnostics,
            "candidate_sources": sources,
        },
    )
    logical = sum(not record.is_base for record in records)
    if logical != expected_logical:
        raise ExperimentRunError(
            "final evaluator returned an incomplete or duplicate logical matrix"
        )
    summary = FinalRunSummary(
        experiment_id=manifest.experiment_id,
        output_directory=final_output,
        candidate_count=len(candidate_ids),
        target_count=len(runtime.phases["final"].targets),
        task_outcomes=len(records),
        logical_task_executions=logical,
        physical_provider_executions=len(cache.keys() - cache_before),
        budget_limit=budget_limit,
        budget_complete=logical == budget_limit,
    )
    _atomic_json(
        final_output / "run_state.json",
        {
            "schema_version": 1,
            "complete": summary.budget_complete,
            "execution_complete": True,
            "summary": summary.to_dict(),
            "candidate_manifest_sha256": candidate_manifest["sha256"],
            "cache_keys": sorted({record.cache_key for record in records}),
        },
    )
    _update_final_output_contract(
        root,
        manifest,
        records=records,
        summary=summary,
        started_at_utc=started_at_utc,
    )
    return summary
