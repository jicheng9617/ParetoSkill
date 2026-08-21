"""Offline synthetic dry-run exercising manifests, cache, statistics, and archive recovery."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import ArchiveEntry, ParetoArchive
from .config import ExperimentManifest
from .evaluation import (
    EvaluationRecord,
    PairedEvaluator,
    ProviderHarness,
    TargetSpec,
    TaskSeedBlock,
    TaskSpec,
)
from .materialize import MaterializationStore, Materializer
from .models import (
    Patch,
    PatchOperation,
    Skill,
    SkillVersion,
    TraceEvidence,
    canonical_json,
    make_base_version,
    stable_hash,
)
from .objectives import FeasibilityConstraints
from .providers import MockProvider, ModelSpec
from .statistics import ObjectiveSummary, PairedObservation, paired_bootstrap
from .storage import ExperimentCheckpoint, JsonlResultStore, ResultCache, StorageError


@dataclass(frozen=True, slots=True)
class DryRunSummary:
    experiment_id: str
    output_directory: Path
    candidate_count: int
    unique_task_executions: int
    reused_task_executions: int
    archive_size: int
    archive_candidate_ids: tuple[str, ...]
    external_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "output_directory": str(self.output_directory),
            "candidate_count": self.candidate_count,
            "unique_task_executions": self.unique_task_executions,
            "reused_task_executions": self.reused_task_executions,
            "archive_size": self.archive_size,
            "archive_candidate_ids": list(self.archive_candidate_ids),
            "external_calls": self.external_calls,
            "synthetic_only": True,
        }


def _synthetic_versions() -> tuple[SkillVersion, tuple[SkillVersion, ...], MaterializationStore]:
    base = make_base_version(
        Skill(
            name="paretoskill-synthetic-fixture",
            files={
                "SKILL.md": (
                    "# Synthetic Fixture\n\n"
                    "Synthetic verbose guidance used only by the offline fixture.\n\n"
                    "Synthetic risky guidance used only by the offline fixture."
                )
            },
            metadata={"synthetic_only": True},
        )
    )
    evidence = {
        "fixture-failure": TraceEvidence(
            evidence_id="fixture-failure",
            task_id="synthetic-id-0",
            seed=17,
            target_id="synthetic-id",
            outcome=False,
            verifier_summary="offline fixture failure, not an empirical observation",
            tags=("synthetic", "failure"),
        ),
        "fixture-transfer": TraceEvidence(
            evidence_id="fixture-transfer",
            task_id="synthetic-transfer-0",
            seed=17,
            target_id="synthetic-transfer-a",
            outcome=True,
            verifier_summary="offline fixture transfer evidence",
            tags=("synthetic", "transfer"),
        ),
    }
    patch_pool = (
        Patch(
            patch_id="fixture-rewrite-risk",
            operation=PatchOperation.REWRITE,
            target_path="SKILL.md",
            parent_version_id=base.lineage.version_id,
            evidence_ids=("fixture-failure",),
            match_text="Synthetic risky guidance used only by the offline fixture.",
            content="Synthetic guarded guidance.",
            risk_category="regression",
            sequence=0,
        ),
        Patch(
            patch_id="fixture-compress",
            operation=PatchOperation.COMPRESS,
            target_path="SKILL.md",
            parent_version_id=base.lineage.version_id,
            evidence_ids=("fixture-failure",),
            match_text="Synthetic verbose guidance used only by the offline fixture.",
            content="Synthetic concise guidance.",
            risk_category="cost",
            sequence=1,
        ),
        Patch(
            patch_id="fixture-add-transfer",
            operation=PatchOperation.ADD,
            target_path="SKILL.md",
            parent_version_id=base.lineage.version_id,
            evidence_ids=("fixture-transfer",),
            content="Synthetic target-invariant procedure.",
            risk_category="transfer",
            sequence=2,
        ),
    )
    materializer = Materializer()
    candidates = (
        materializer.materialize(base, (patch_pool[0],), evidence=evidence, label="fixture"),
        materializer.materialize(base, (patch_pool[1],), evidence=evidence, label="fixture"),
        materializer.materialize(base, patch_pool, evidence=evidence, label="fixture"),
    )
    store = MaterializationStore()
    for version in (base, *candidates):
        store.add(version)
    return base, candidates, store


def _synthetic_matrix() -> tuple[tuple[TargetSpec, ...], tuple[TaskSeedBlock, ...]]:
    model = ModelSpec("offline-deterministic-mock", "mock", "fixture-v1", {"temperature": 0})
    targets = (
        TargetSpec(
            "synthetic-id",
            "mock",
            model,
            "provider-structured",
            "synthetic",
            "id",
        ),
        TargetSpec(
            "synthetic-transfer-a",
            "mock",
            model,
            "provider-structured",
            "synthetic",
            "transfer",
        ),
        TargetSpec(
            "synthetic-transfer-b",
            "mock",
            model,
            "provider-structured",
            "synthetic",
            "transfer",
        ),
    )
    blocks: list[TaskSeedBlock] = []
    for split, count in (("id", 4), ("transfer", 3)):
        for index in range(count):
            task = TaskSpec(
                task_id=f"synthetic-{split}-{index}",
                split=split,
                domain_id="synthetic",
                group_id=split,
                payload={"synthetic_index": index, "split": split},
            )
            blocks.append(TaskSeedBlock(f"{split}-{index}", task, seed=17))
    return targets, tuple(blocks)


def _paired_observations(
    records: list[EvaluationRecord], candidate_id: str
) -> tuple[PairedObservation, ...]:
    base_by_block = {record.block_key: record for record in records if record.is_base}
    candidate_records = [record for record in records if record.candidate_id == candidate_id]
    observations: list[PairedObservation] = []
    for record in candidate_records:
        base = base_by_block[record.block_key]
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
    return tuple(observations)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _git_state(repository_root: Path) -> dict[str, Any]:
    """Return non-secret implementation provenance without mutating git state."""

    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all", "--", "code")
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": stable_hash(status or ""),
    }


def run_synthetic_dry_run(
    manifest: ExperimentManifest,
    *,
    output_root: str | Path | None = None,
    resume: bool = True,
) -> DryRunSummary:
    """Run a tiny deterministic fixture. It cannot access real data or external providers."""

    if manifest.profile != "dry_run" or not manifest.is_offline:
        raise ValueError("synthetic dry-run requires the offline dry_run profile")
    profile = manifest.data["runtime_profiles"]["dry_run"]
    uses_mock = profile["provider_override"] == "mock"
    uses_fixture = profile["data_adapter"] == "builtin_synthetic_fixture"
    if not uses_mock or not uses_fixture:
        raise ValueError("dry-run must use the built-in synthetic fixture and mock provider")

    if output_root is None:
        configured = Path(str(manifest.data["outputs"]["root"]))
        code_root = manifest.code_root
        root = configured if configured.is_absolute() else code_root / configured
    else:
        root = Path(output_root).resolve()
    output = root / manifest.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.exists() and not resume:
        raise StorageError(
            f"dry-run output already exists at {output}; use resume or select another output root"
        )
    checkpoint = (
        ExperimentCheckpoint.load(
            checkpoint_path,
            expected_experiment_id=manifest.experiment_id,
            max_task_executions=int(
                manifest.data["budgets"]["search_total_per_method"]["task_executions"]
            ),
            expected_output_root=output,
        )
        if checkpoint_path.exists()
        else ExperimentCheckpoint(experiment_id=manifest.experiment_id)
    )

    base, candidates, materializations = _synthetic_versions()
    targets, blocks = _synthetic_matrix()
    cache = ResultCache(output / "cache")
    cached_before = cache.keys()
    # The cache is the execution source of truth. A checkpoint may have been
    # persisted just before/after a cache write when a process was interrupted.
    checkpoint.completed_cache_keys.intersection_update(cached_before)
    result_store = JsonlResultStore(output / "task_outcomes.jsonl")
    harness = ProviderHarness(cache=cache)
    records = PairedEvaluator().evaluate(
        experiment_id=manifest.experiment_id,
        base=base,
        candidates=candidates,
        targets=targets,
        blocks=blocks,
        providers={"mock": MockProvider()},
        harnesses={harness.harness_id: harness},
    )

    unique_keys = {record.cache_key for record in records}
    new_keys = unique_keys - cached_before
    # The content-addressed cache is the execution source of truth. Rebuild the
    # task ledger atomically so a crash during an earlier append cannot create
    # duplicates or leave a partial JSONL tail on resume.
    result_store.replace(records)
    checkpoint.completed_cache_keys.update(unique_keys)
    checkpoint.task_executions_consumed = len(checkpoint.completed_cache_keys)

    summaries: dict[str, ObjectiveSummary] = {}
    expected_transfer_groups = {
        target.target_id for target in targets if target.task_group == "transfer"
    }
    minimum_effective_blocks = int(
        manifest.data["statistics"].get("minimum_effective_blocks_for_archive", 2)
    )
    for version in (base, *candidates):
        summaries[version.lineage.version_id] = paired_bootstrap(
            _paired_observations(records, version.lineage.version_id),
            confidence_level=0.95,
            replicates=500,
            seed=32452843,
            expected_transfer_groups=expected_transfer_groups,
            min_effective_blocks=minimum_effective_blocks,
            token_cost_upper_bound=float(
                manifest.data["constraints"]["token_budget"]["budget"]
            ),
        )
    epsilon = float(manifest.data["constraints"]["id_accuracy_floor"]["epsilon"])
    token_budget = float(manifest.data["constraints"]["token_budget"]["budget"])
    constraints = FeasibilityConstraints.from_paired_epsilon(
        epsilon=epsilon,
        token_budget=token_budget,
        enabled=bool(manifest.data["constraints"]["enabled"]),
    )
    archive = ParetoArchive(
        max_size=int(manifest.data["selection_protocol"]["archive_capacity"]["max_entries"]),
        evaluation_budget=int(
            manifest.data["budgets"]["search_total_per_method"]["task_executions"]
        ),
        constraints=constraints,
        dominance_mode="uncertainty",
    )
    decisions = []
    for version in (base, *candidates):
        decision = archive.admit(
            ArchiveEntry(
                candidate_id=version.lineage.version_id,
                content_hash=version.skill.content_hash,
                objectives=summaries[version.lineage.version_id],
                evaluation_cost=sum(
                    record.candidate_id == version.lineage.version_id for record in records
                ),
                metadata={"synthetic_only": True},
            )
        )
        decisions.append(
            {
                "candidate_id": decision.candidate_id,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "removed_ids": list(decision.removed_ids),
            }
        )

    manifest.save(output / "resolved_manifest.yaml")
    materializations.save(output / "materializations.json")
    archive.save(output / "archive.json")
    _write_json(
        output / "scientific_front.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "synthetic_only": True,
            "entries": [entry.to_dict() for entry in archive.scientific_front],
        },
    )
    checkpoint.archive_path = str(output / "archive.json")
    checkpoint.save(checkpoint_path)
    _write_jsonl(
        output / "candidates.jsonl",
        [version.to_dict() for version in (base, *candidates)],
    )
    _write_jsonl(
        output / "lineage.jsonl",
        [version.lineage.to_dict() for version in (base, *candidates)],
    )
    _write_json(
        output / "metrics.json",
        {
            "schema_version": 1,
            "synthetic_only": True,
            "objectives": {
                candidate_id: summary.to_dict()
                for candidate_id, summary in sorted(summaries.items())
            },
            "archive_decisions": decisions,
        },
    )
    _write_json(
        output / "run_metadata.json",
        {
            "schema_version": 1,
            "experiment_id": manifest.experiment_id,
            "profile": manifest.profile,
            "python": platform.python_version(),
            "implementation_digest": manifest.implementation_digest,
            "git": _git_state(manifest.code_root.parent),
            "synthetic_only": True,
            "external_calls": 0,
            "unresolved_external_placeholders": list(manifest.unresolved_placeholders),
            "cache_keys": len(unique_keys),
            "new_cache_keys_this_invocation": len(new_keys),
        },
    )
    # Dry-runs do not represent real billing, but keep the required accounting artifact.
    _write_json(
        output / "token_accounting.json",
        {
            "schema_version": 1,
            "synthetic_only": True,
            "input_tokens": sum(record.result.input_tokens for record in records),
            "output_tokens": sum(record.result.output_tokens for record in records),
            "price": 0.0,
        },
    )

    return DryRunSummary(
        experiment_id=manifest.experiment_id,
        output_directory=output,
        candidate_count=1 + len(candidates),
        unique_task_executions=len(new_keys),
        reused_task_executions=len(unique_keys & cached_before),
        archive_size=len(archive),
        archive_candidate_ids=tuple(entry.candidate_id for entry in archive.entries),
    )
