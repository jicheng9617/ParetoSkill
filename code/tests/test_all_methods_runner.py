"""Offline integration coverage for every configured search method.

The fixture deliberately supplies two blocks per target so uncertainty-aware
selection and the Ctx2Skill hard/easy split are both exercised without any
external provider.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import paretoskill.experiment_runner as experiment_runner_module
from paretoskill.baselines import AccuracyOnlyPlugin, ScoredCandidate
from paretoskill.config import load_manifest
from paretoskill.evaluation import (
    ProviderHarness,
    TargetSpec,
    TaskSeedBlock,
    TaskSpec,
)
from paretoskill.experiment_runner import (
    ExperimentRunError,
    ExperimentRuntime,
    PhaseRuntime,
    _controller_promotion_pool,
    _method_specs,
    _run_method,
    run_configured_final,
    run_configured_search,
)
from paretoskill.failures import (
    PairedEvaluationFailure,
    ProviderTransportFailureBeforeResponse,
)
from paretoskill.models import (
    Patch,
    PatchOperation,
    Skill,
    TraceEvidence,
    make_base_version,
)
from paretoskill.providers import (
    ExecutionRequest,
    ExecutionResult,
    ModelSpec,
    NetworkPolicy,
)
from paretoskill.proposer import DeterministicMockProposer
from paretoskill.search_strategies import CommonCandidateStream, NSGAIIController
from paretoskill.statistics import PointObjectives
from paretoskill.storage import ResultCache


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"

EXPECTED_METHOD_IDS = (
    "no_skill",
    "base_skill",
    "simple_patch_composition",
    "trace2skill_all",
    "trace2skill_accuracy_subset",
    "fixed_scalarization_accuracy_only",
    "fixed_scalarization_accuracy_cost_equal",
    "fixed_scalarization_balanced_four_objective",
    "fixed_scalarization_ctx2skill_hard_easy_product",
    "evoskill_scalar_topk",
    "skillmoo_nsga2",
    "mocha_chebyshev_hvc",
    "passive_archive",
    "paretoskill",
    "ablation_no_uncertainty_bounds",
    "ablation_no_regression_objective",
    "ablation_no_transfer_objective",
    "ablation_passive_archive",
    "ablation_no_feasibility_gate",
    "ablation_evidence_blind_generation",
    "ablation_lineage_blind_generation",
    "ablation_patch_subset_only",
)


@dataclass(slots=True)
class AlwaysCorrectProvider:
    """Deterministic local provider that makes feasibility unambiguous."""

    provider_id: str = "mock"
    is_external: bool = False
    calls: int = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        return ExecutionResult(
            correct=True,
            input_tokens=16 + len(request.skill_files),
            output_tokens=4,
            latency_ms=0.0,
            trace={"mode": "all-methods-offline-fixture"},
            provider_metadata={"offline": True},
        )


@dataclass(slots=True)
class ControlledFailureProvider:
    """Offline provider that fails a configured number of physical attempts."""

    failure_count: int
    ordinary: bool = False
    provider_id: str = "mock"
    is_external: bool = False
    calls: int = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.calls += 1
        if self.calls <= self.failure_count:
            if self.ordinary:
                raise RuntimeError("ordinary provider bug")
            raise ProviderTransportFailureBeforeResponse(
                evidence_sha256="a" * 64
            )
        return ExecutionResult(
            correct=True,
            input_tokens=16 + len(request.skill_files),
            output_tokens=4,
            latency_ms=0.0,
            trace={"mode": "controlled-failure-offline-fixture"},
            provider_metadata={"offline": True},
        )


def _runtime(
    provider: AlwaysCorrectProvider,
    *,
    tasks_per_role: int = 2,
    include_final: bool = False,
) -> ExperimentRuntime:
    base = make_base_version(Skill("all-methods-fixture", {"SKILL.md": "# Base\n"}))
    evidence = TraceEvidence(
        evidence_id="evidence-1",
        task_id="id-task-0",
        seed=17,
        target_id="id-target",
        outcome=False,
        verifier_summary="synthetic offline evidence",
        tags=("id_accuracy",),
    )
    patch = Patch(
        patch_id="patch-1",
        operation=PatchOperation.ADD,
        target_path="SKILL.md",
        parent_version_id=base.lineage.version_id,
        evidence_ids=(evidence.evidence_id,),
        content="Use a checked procedure.",
        sequence=0,
    )
    model = ModelSpec("fixture-model", "mock", "fixture-v1", {"temperature": 0.0})
    targets = (
        TargetSpec(
            "id-target",
            "mock",
            model,
            "provider-structured",
            "id-domain",
            "*",
            split_id="id-split",
            objective_role="id",
        ),
        TargetSpec(
            "transfer-target",
            "mock",
            model,
            "provider-structured",
            "transfer-domain",
            "*",
            split_id="transfer-split",
            transfer_group="domain",
            objective_role="transfer",
        ),
    )
    blocks = tuple(
        TaskSeedBlock(
            f"{role}-block-{index}",
            TaskSpec(
                task_id=f"{role}-task-{index}",
                split=role,
                domain_id=f"{role}-domain",
                group_id=role,
                payload={"index": index},
                split_id=f"{role}-split",
                objective_role=role,
            ),
            17,
        )
        for role in ("id", "transfer")
        for index in range(tasks_per_role)
    )
    phases = {
        "search": PhaseRuntime(
            targets=targets,
            blocks=blocks,
            harnesses={"provider-structured": ProviderHarness()},
        )
    }
    if include_final:
        final_targets = (
            TargetSpec(
                "final-id-target",
                "mock",
                model,
                "provider-structured",
                "final-id-domain",
                "*",
                split_id="final-id-split",
                objective_role="id",
            ),
            TargetSpec(
                "final-transfer-target",
                "mock",
                model,
                "provider-structured",
                "final-transfer-domain",
                "*",
                split_id="final-transfer-split",
                transfer_group="heldout-domain",
                objective_role="transfer",
            ),
        )
        final_blocks = tuple(
            TaskSeedBlock(
                f"final-{role}-block-{index}",
                TaskSpec(
                    task_id=f"final-{role}-task-{index}",
                    split=role,
                    domain_id=f"final-{role}-domain",
                    group_id=f"final-{role}",
                    payload={"index": index},
                    split_id=f"final-{role}-split",
                    objective_role=role,
                ),
                17,
            )
            for role in ("id", "transfer")
            for index in range(2)
        )
        phases["final"] = PhaseRuntime(
            targets=final_targets,
            blocks=final_blocks,
            harnesses={"provider-structured": ProviderHarness()},
        )
    return ExperimentRuntime(
        base=base,
        patches=(patch,),
        evidence={evidence.evidence_id: evidence},
        providers={provider.provider_id: provider},
        phases=phases,
    )


def _small_formal_manifest():
    manifest = load_manifest(CONFIG, profile="dry_run")
    data = copy.deepcopy(dict(manifest.data))
    data["budgets"]["screen"].update(
        {
            "task_executions": 2,
            "tasks_per_target": 1,
            "execution_seeds_per_task": 1,
            "target_ids": ["id-target", "transfer-target"],
        }
    )
    data["budgets"]["full"].update(
        {
            "task_executions": 6,
            "task_executions_including_screen_subset": 6,
            "incremental_task_executions_after_screen": 4,
            "tasks_per_target": 3,
            "execution_seeds_per_task": 1,
            "target_ids": ["id-target", "transfer-target"],
        }
    )
    data["domains"]["final-id-domain"] = {
        "adapter": "fixture-final-id",
        "adapter_revision": "fixture-v1",
    }
    data["domains"]["final-transfer-domain"] = {
        "adapter": "fixture-final-transfer",
        "adapter_revision": "fixture-v1",
    }
    data["targets"]["final-id-target"] = {
        "phase": "final_only",
        "model": "skill_user_small",
        "harness": "spreadsheet_primary",
        "domain": "final-id-domain",
        "split": "final-id-split",
        "final_analysis_role": "id",
    }
    data["targets"]["final-transfer-target"] = {
        "phase": "final_only",
        "model": "skill_user_small",
        "harness": "spreadsheet_primary",
        "domain": "final-transfer-domain",
        "split": "final-transfer-split",
        "transfer_group": "heldout-domain",
        "final_analysis_role": "transfer",
    }
    for target_id in tuple(data["targets"]):
        if (
            data["targets"][target_id].get("phase") == "final_only"
            and target_id not in {"final-id-target", "final-transfer-target"}
        ):
            del data["targets"][target_id]
    data["budgets"]["final"].update(
        {
            "task_executions": 4,
            "final_target_ids": ["final-id-target", "final-transfer-target"],
        }
    )
    data.setdefault("runner", {})["final"] = {
        "required_method_ids": ["trace2skill_all"],
        "required_search_seeds": [104729],
    }
    data["selection_protocol"]["normalization_ranges"] = [
        [0.0, 1.0],
        [0.0, 1.0],
        [-4096.0, 0.0],
        [-1.0, 0.0],
    ]
    return replace(manifest, data=data)


def _controller_formal_manifest():
    manifest = _small_formal_manifest()
    data = copy.deepcopy(dict(manifest.data))
    data["budgets"]["search_total_per_method"].update(
        {
            "task_executions": 16,
            "maximum_unique_screened_candidates": 6,
            "screen_allocation_task_executions": 12,
            "maximum_promoted_candidates": 1,
            "incremental_promotion_allocation_task_executions": 4,
            "allocation_sum_check": 16,
        }
    )
    data["methods"]["trace2skill_accuracy_subset"]["optimizer_protocol"][
        "batch_size"
    ] = 2
    data["methods"]["trace2skill_accuracy_subset"][
        "logical_task_execution_budget_per_search_seed"
    ] = 16
    data["methods"]["evoskill_scalar_topk"].update(
        {
            "logical_task_execution_budget_per_search_seed": 16,
            "top_k": 2,
        }
    )
    data["methods"]["skillmoo_nsga2"].update(
        {
            "logical_task_execution_budget_per_search_seed": 16,
            "population_size": 4,
            "offspring_size": 2,
        }
    )
    return replace(manifest, data=data)


def _bounded_smoke_manifest(*, ceiling: int = 16):
    manifest = load_manifest(CONFIG, profile="dry_run")
    data = copy.deepcopy(dict(manifest.data))
    data["runner"] = {
        "smoke": {
            "max_candidates": 2,
            "candidate_limits": {
                "simple_patch_composition": 1,
                "paretoskill": 2,
            },
            "blocks_per_target": 2,
            "search_seeds": [104729],
            "methods": [
                "no_skill",
                "base_skill",
                "simple_patch_composition",
                "paretoskill",
            ],
            "search_targets": ["id-target", "transfer-target"],
            "logical_task_execution_ceiling": ceiling,
            "separate_output_namespace": "smoke",
            "never_promote_results_to_main_comparison": True,
        }
    }
    return replace(manifest, data=data)


def _runtime_with_patch_pool(
    provider: AlwaysCorrectProvider,
    *,
    patch_count: int = 5,
) -> ExperimentRuntime:
    runtime = _runtime(provider, tasks_per_role=3)
    evidence_id = next(iter(runtime.evidence))
    runtime.patches = tuple(
        Patch(
            patch_id=f"patch-{index}",
            operation=PatchOperation.ADD,
            target_path="SKILL.md",
            parent_version_id=runtime.base.lineage.version_id,
            evidence_ids=(evidence_id,),
            content=f"Apply independent checked rule {index}.",
            sequence=index,
        )
        for index in range(patch_count)
    )
    return runtime


class RecordingBinaryAdapter:
    adapter_id = "tests.runner_binary_adapter/v1"

    def __init__(self) -> None:
        self.ask_batches: list[tuple[tuple[str, ...], ...]] = []
        self.tell_batches: list[tuple[tuple[str, ...], ...]] = []
        self.observation_sizes: list[int] = []
        self.actual_ask_calls = 0
        self.actual_tell_calls = 0
        self.load_state_calls = 0

    def ask(
        self,
        *,
        patch_ids,
        count,
        seed,
        seen_subsets,
        observations,
    ):
        del seed
        self.actual_ask_calls += 1
        self.observation_sizes.append(len(observations))
        seen = set(seen_subsets)
        available = tuple(
            tuple(
                patch_id
                for index, patch_id in enumerate(patch_ids)
                if mask & (1 << index)
            )
            for mask in range(1, 1 << len(patch_ids))
            if tuple(
                patch_id
                for index, patch_id in enumerate(patch_ids)
                if mask & (1 << index)
            )
            not in seen
        )
        batch = available[:count]
        assert len(batch) == count
        self.ask_batches.append(batch)
        return batch

    def tell(self, scored_values):
        self.actual_tell_calls += 1
        self.tell_batches.append(
            tuple(tuple(candidate.patch_ids) for candidate in scored_values)
        )

    def state_dict(self):
        return {
            "ask_batches": [[list(subset) for subset in batch] for batch in self.ask_batches],
            "tell_batches": [
                [list(subset) for subset in batch] for batch in self.tell_batches
            ],
            "observation_sizes": list(self.observation_sizes),
        }

    def load_state_dict(self, state):
        self.load_state_calls += 1
        self.ask_batches = [
            tuple(tuple(subset) for subset in batch) for batch in state["ask_batches"]
        ]
        self.tell_batches = [
            tuple(tuple(subset) for subset in batch) for batch in state["tell_batches"]
        ]
        self.observation_sizes = list(state["observation_sizes"])


def test_manifest_expands_to_the_frozen_method_matrix():
    manifest = load_manifest(CONFIG, profile="dry_run")
    observed = tuple(spec[0] for spec in _method_specs(manifest.data, None))
    assert observed == EXPECTED_METHOD_IDS


@pytest.mark.parametrize(
    ("method_id", "state_type"),
    (
        ("trace2skill_accuracy_subset", "adapter_backed_binary_subset"),
        ("evoskill_scalar_topk", "evo_top_k_controller"),
        ("skillmoo_nsga2", "nsga2_subset_controller"),
        ("mocha_chebyshev_hvc", "mocha_subset_controller"),
    ),
)
def test_formal_adaptive_subset_controllers_close_the_small_frozen_budget(
    tmp_path: Path,
    method_id: str,
    state_type: str,
):
    manifest = _controller_formal_manifest()
    provider = AlwaysCorrectProvider()
    runtime = _runtime_with_patch_pool(provider)
    adapters: list[RecordingBinaryAdapter] = []
    if method_id == "trace2skill_accuracy_subset":

        def binary_optimizer_factory(patch_ids, seed, method):
            assert len(patch_ids) == 5
            assert seed == 104729
            assert method["optimizer_protocol"]["batch_size"] == 2
            adapter = RecordingBinaryAdapter()
            adapters.append(adapter)
            return adapter

        runtime.binary_optimizer_factory = binary_optimizer_factory

    summary = _run_method(
        manifest=manifest,
        runtime=runtime,
        phase=runtime.phases["search"],
        method_id=method_id,
        plugin_id=method_id,
        configuration=manifest.data,
        search_seed=104729,
        output=tmp_path / method_id,
        cache=ResultCache(tmp_path / "cache"),
        policy=NetworkPolicy(),
        smoke=False,
    )

    assert summary.proposed_candidates == summary.screened_candidates == 6
    assert summary.promoted_candidates == 1
    assert summary.logical_task_executions == summary.budget_limit == 16
    assert summary.budget_complete is True
    checkpoint = json.loads(
        (summary.output_directory / "search_controller.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["proposed_candidates"] == 6
    assert checkpoint["execution_complete"] is True
    controller = checkpoint["controller"]
    assert controller["state_type"] == state_type
    assert controller["pending"] == []

    if method_id == "trace2skill_accuracy_subset":
        assert len(adapters) == 1
        assert [len(batch) for batch in adapters[0].ask_batches] == [2, 2, 2]
        assert [len(batch) for batch in adapters[0].tell_batches] == [2, 2, 2]
        assert adapters[0].observation_sizes == [0, 2, 4]
        assert len(controller["observations"]) == 6
        assert controller["adapter_state"]["observation_sizes"] == [0, 2, 4]
    elif method_id == "evoskill_scalar_topk":
        assert len(controller["observed_subsets"]) == 6
        assert len(controller["incumbents"]) == 2
        assert controller["stream"]["cursor"] == 6
    elif method_id == "skillmoo_nsga2":
        assert len(controller["seen"]) == 6
        assert len(controller["population"]) == 4
        assert controller["generation"] == 1
    else:
        assert len(controller["decisions"]) == 6
        assert controller["logical_task_executions_spent"] == 12
        assert controller["stream"]["cursor"] == 6


def test_partial_trace_controller_checkpoint_resumes_without_replaying_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = _controller_formal_manifest()
    output = tmp_path / "interrupted" / "trace2skill_accuracy_subset"
    cache = ResultCache(tmp_path / "interrupted" / "cache")
    adapters: list[RecordingBinaryAdapter] = []

    def binary_optimizer_factory(patch_ids, seed, method):
        del patch_ids, seed, method
        adapter = RecordingBinaryAdapter()
        adapters.append(adapter)
        return adapter

    first_provider = AlwaysCorrectProvider()
    first_runtime = _runtime_with_patch_pool(first_provider)
    first_runtime.binary_optimizer_factory = binary_optimizer_factory
    original_atomic_json = experiment_runner_module._atomic_json
    injected = False

    class SimulatedCrash(RuntimeError):
        pass

    def crash_after_first_committed_batch(path, payload):
        nonlocal injected
        original_atomic_json(path, payload)
        if (
            not injected
            and path.name == "search_controller.json"
            and payload.get("execution_complete") is False
            and payload.get("proposed_candidates") == 2
        ):
            injected = True
            raise SimulatedCrash("synthetic interruption after atomic checkpoint")

    monkeypatch.setattr(
        experiment_runner_module,
        "_atomic_json",
        crash_after_first_committed_batch,
    )
    with pytest.raises(SimulatedCrash, match="synthetic interruption"):
        _run_method(
            manifest=manifest,
            runtime=first_runtime,
            phase=first_runtime.phases["search"],
            method_id="trace2skill_accuracy_subset",
            plugin_id="trace2skill_accuracy_subset",
            configuration=manifest.data,
            search_seed=104729,
            output=output,
            cache=cache,
            policy=NetworkPolicy(),
            smoke=False,
        )

    partial = json.loads(
        (output / "search_controller.json").read_text(encoding="utf-8")
    )
    assert partial["schema_version"] == 2
    assert partial["execution_complete"] is False
    assert partial["proposed_candidates"] == 2
    assert len(partial["controller"]["observations"]) == 2
    assert partial["execution_candidate_order"]
    assert partial["materializations"]["lineages"]
    # The injected crash happens before convenience partial files are written;
    # the atomic checkpoint alone must be sufficient for recovery.
    assert not (output / "materializations.partial.json").exists()
    assert adapters[0].actual_ask_calls == adapters[0].actual_tell_calls == 1
    first_batch_provider_calls = first_provider.calls
    assert first_batch_provider_calls > 0

    second_provider = AlwaysCorrectProvider()
    second_runtime = _runtime_with_patch_pool(second_provider)
    second_runtime.binary_optimizer_factory = binary_optimizer_factory
    resumed = _run_method(
        manifest=manifest,
        runtime=second_runtime,
        phase=second_runtime.phases["search"],
        method_id="trace2skill_accuracy_subset",
        plugin_id="trace2skill_accuracy_subset",
        configuration=manifest.data,
        search_seed=104729,
        output=output,
        cache=cache,
        policy=NetworkPolicy(),
        smoke=False,
    )

    assert resumed.budget_complete is True
    assert len(adapters) == 2
    restored_adapter = adapters[1]
    assert restored_adapter.load_state_calls == 1
    assert restored_adapter.actual_ask_calls == restored_adapter.actual_tell_calls == 2
    assert [len(batch) for batch in restored_adapter.ask_batches] == [2, 2, 2]
    assert restored_adapter.observation_sizes == [0, 2, 4]

    monkeypatch.setattr(
        experiment_runner_module,
        "_atomic_json",
        original_atomic_json,
    )
    clean_provider = AlwaysCorrectProvider()
    clean_runtime = _runtime_with_patch_pool(clean_provider)
    clean_adapters: list[RecordingBinaryAdapter] = []

    def clean_factory(patch_ids, seed, method):
        del patch_ids, seed, method
        adapter = RecordingBinaryAdapter()
        clean_adapters.append(adapter)
        return adapter

    clean_runtime.binary_optimizer_factory = clean_factory
    clean = _run_method(
        manifest=manifest,
        runtime=clean_runtime,
        phase=clean_runtime.phases["search"],
        method_id="trace2skill_accuracy_subset",
        plugin_id="trace2skill_accuracy_subset",
        configuration=manifest.data,
        search_seed=104729,
        output=tmp_path / "clean" / "trace2skill_accuracy_subset",
        cache=ResultCache(tmp_path / "clean" / "cache"),
        policy=NetworkPolicy(),
        smoke=False,
    )
    assert resumed.logical_task_executions == clean.logical_task_executions == 16
    assert first_batch_provider_calls + second_provider.calls == clean_provider.calls
    assert second_provider.calls < clean_provider.calls
    assert clean_adapters[0].actual_ask_calls == 3


def test_nsga_promotion_pool_is_restricted_to_final_population() -> None:
    patch_ids = ("p0", "p1", "p2")
    initial = CommonCandidateStream(
        patch_ids,
        (("p0",), ("p1",)),
        seed=31,
    )
    controller = NSGAIIController(
        patch_ids,
        seed=37,
        population_size=2,
        offspring_size=1,
        initial_stream=initial,
    )
    pending = controller.ask()
    population_scores = tuple(
        ScoredCandidate(
            candidate_id=f"population-{index}",
            patch_ids=subset,
            objectives=PointObjectives(
                id_accuracy=0.5 + index / 10,
                worst_target_transfer=0.5,
                token_cost=50.0,
                paired_regression=0.1,
            ),
        )
        for index, subset in enumerate(pending)
    )
    controller.tell(population_scores)
    outside = ScoredCandidate(
        candidate_id="screened-but-not-survivor",
        patch_ids=("p2",),
        objectives=PointObjectives(1.0, 1.0, 1.0, 0.0),
    )

    promotion_pool = _controller_promotion_pool(
        controller,
        (*population_scores, outside),
    )

    assert {candidate.candidate_id for candidate in promotion_pool} == {
        candidate.candidate_id for candidate in controller.population
    }
    assert outside.candidate_id not in {
        candidate.candidate_id for candidate in promotion_pool
    }


@pytest.mark.parametrize(
    "method_id",
    EXPECTED_METHOD_IDS,
)
def test_every_method_has_a_resumable_offline_smoke(
    tmp_path: Path, method_id: str
):
    manifest = load_manifest(CONFIG, profile="dry_run")
    first_provider = AlwaysCorrectProvider()
    first = run_configured_search(
        manifest,
        _runtime(first_provider),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=(method_id,),
        search_seeds=(104729,),
        smoke=True,
    )
    second_provider = AlwaysCorrectProvider()
    second = run_configured_search(
        manifest,
        _runtime(second_provider),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=(method_id,),
        search_seeds=(104729,),
        smoke=True,
    )

    assert len(first.method_runs) == len(second.method_runs) == 1
    first_run = first.method_runs[0]
    second_run = second.method_runs[0]
    assert first_run.method_id == second_run.method_id == method_id
    assert first_run.physical_provider_executions == first_provider.calls > 0
    assert second_run.physical_provider_executions == second_provider.calls == 0
    assert first_run.logical_task_executions == second_run.logical_task_executions
    assert first_run.logical_task_executions == 4
    assert first_run.budget_complete is second_run.budget_complete is True
    assert first_run.selected_candidate_ids

    selected_path = first_run.output_directory / "selected_candidates.jsonl"
    selected_rows = [
        json.loads(line)
        for line in selected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["candidate_id"] for row in selected_rows] == list(
        first_run.selected_candidate_ids
    )
    if method_id == "no_skill":
        assert selected_rows[0]["injection_mode"] == "none"
        assert selected_rows[0]["candidate_id"] == "no-skill"
    else:
        assert all(row["injection_mode"] == "skill" for row in selected_rows)
    run_state = json.loads(
        (first_run.output_directory / "run_state.json").read_text(encoding="utf-8")
    )
    assert run_state["complete"] is True
    assert run_state["execution_complete"] is True
    missing_contract_files = [
        name
        for name in manifest.data["outputs"]["required_files"]
        if not (first.output_directory / name).is_file()
    ]
    assert missing_contract_files == []
    root_checkpoint = json.loads(
        (first.output_directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert root_checkpoint["execution_complete"] is True


def test_successful_retry_is_persisted_and_resume_deduplicates_events(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(CONFIG, profile="dry_run")
    provider = ControlledFailureProvider(failure_count=1)
    first = run_configured_search(
        manifest,
        _runtime(provider),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=True,
    )
    method_ledger = first.method_runs[0].output_directory / "failure_events.jsonl"
    root_ledger = first.output_directory / "failure_events.jsonl"
    method_rows = [
        json.loads(line)
        for line in method_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root_rows = [
        json.loads(line)
        for line in root_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(method_rows) == len(root_rows) == 1
    assert method_rows == root_rows
    assert method_rows[0]["will_retry"] is True
    assert method_rows[0]["retry_limit"] == 1

    resumed_provider = AlwaysCorrectProvider()
    resumed = run_configured_search(
        manifest,
        _runtime(resumed_provider),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=True,
    )
    resumed_rows = [
        json.loads(line)
        for line in (resumed.output_directory / "failure_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert resumed_provider.calls == 0
    assert resumed_rows == root_rows
    assert len({row["failure_id"] for row in resumed_rows}) == 1


def test_exhausted_retry_persists_every_verified_attempt(tmp_path: Path) -> None:
    manifest = load_manifest(CONFIG, profile="dry_run")
    provider = ControlledFailureProvider(failure_count=2)
    runtime = _runtime(provider)
    output = tmp_path / "exhausted-method"
    with pytest.raises(PairedEvaluationFailure) as raised:
        _run_method(
            manifest=manifest,
            runtime=runtime,
            phase=runtime.phases["search"],
            method_id="trace2skill_all",
            plugin_id="trace2skill_all",
            configuration=manifest.data,
            search_seed=104729,
            output=output,
            cache=ResultCache(tmp_path / "exhausted-cache"),
            policy=NetworkPolicy(),
            smoke=True,
        )
    rows = [
        json.loads(line)
        for line in (output / "failure_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert provider.calls == 2
    assert len(rows) == len(raised.value.events) == 2
    assert [row["attempt_number"] for row in rows] == [1, 2]
    assert [row["will_retry"] for row in rows] == [True, False]
    assert len({row["failure_id"] for row in rows}) == 2


def test_ordinary_exception_persists_no_synthetic_failure_event(tmp_path: Path) -> None:
    manifest = load_manifest(CONFIG, profile="dry_run")
    provider = ControlledFailureProvider(failure_count=1, ordinary=True)
    runtime = _runtime(provider)
    output = tmp_path / "ordinary-method"
    with pytest.raises(RuntimeError, match="ordinary provider bug"):
        _run_method(
            manifest=manifest,
            runtime=runtime,
            phase=runtime.phases["search"],
            method_id="trace2skill_all",
            plugin_id="trace2skill_all",
            configuration=manifest.data,
            search_seed=104729,
            output=output,
            cache=ResultCache(tmp_path / "ordinary-cache"),
            policy=NetworkPolicy(),
            smoke=True,
        )
    assert (output / "failure_events.jsonl").read_text(encoding="utf-8") == ""


def test_matched_subset_stream_excludes_the_empty_control():
    patch_ids = tuple(f"patch-{index}" for index in range(9))
    subsets = AccuracyOnlyPlugin().propose_subsets(
        patch_ids,
        max_candidates=386,
        seed=104729,
    )

    assert len(subsets) == 386
    assert all(subsets)


def test_formal_search_does_not_resume_a_smoke_run(tmp_path: Path):
    manifest = _small_formal_manifest()
    smoke_provider = AlwaysCorrectProvider()
    smoke = run_configured_search(
        manifest,
        _runtime(smoke_provider, tasks_per_role=3),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=True,
    )

    formal_provider = AlwaysCorrectProvider()
    formal = run_configured_search(
        manifest,
        _runtime(formal_provider, tasks_per_role=3),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=False,
    )

    assert smoke.method_runs[0].logical_task_executions == 4
    assert formal.method_runs[0].logical_task_executions == 6
    assert formal.method_runs[0].physical_provider_executions == formal_provider.calls == 4
    state = json.loads(
        (formal.method_runs[0].output_directory / "run_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["smoke"] is False


def test_final_candidate_manifest_drift_fails_before_provider_execution(
    tmp_path: Path,
):
    manifest = _small_formal_manifest()
    initial_provider = AlwaysCorrectProvider()
    runtime = _runtime(
        initial_provider,
        tasks_per_role=3,
        include_final=True,
    )
    search = run_configured_search(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=False,
    )
    first_final = run_configured_final(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
    )
    assert first_final.budget_complete is True
    final_metrics = json.loads(
        (first_final.output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    primary = final_metrics["primary_analysis"]
    assert primary["defined"] is True
    assert primary["record_partition"]["id_record_count"] > 0
    assert primary["record_partition"]["transfer_record_count"] > 0
    assert primary["heldout_fronts"]["defined"] is True
    assert all(
        "aggregate" not in diagnostic
        for diagnostic in final_metrics["candidate_target_diagnostics"].values()
    )
    root_metrics = json.loads(
        (search.output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    assert root_metrics["heldout_frontier_comparison"]["data_role"] == (
        "heldout_final_only"
    )
    assert root_metrics["final"]["analysis_summary"]["heldout_front_defined"] is True

    selected_path = search.method_runs[0].output_directory / "selected_candidates.jsonl"
    no_skill_row = {
        "schema_version": 1,
        "method_id": "drift-fixture",
        "search_seed": 104729,
        "candidate_id": "no-skill",
        "injection_mode": "none",
        "version": runtime.base.to_dict(),
    }
    with selected_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(no_skill_row, sort_keys=True) + "\n")

    drift_provider = AlwaysCorrectProvider()
    drift_runtime = _runtime(
        drift_provider,
        tasks_per_role=3,
        include_final=True,
    )
    with pytest.raises(
        ExperimentRunError,
        match="frozen final candidate manifest changed",
    ):
        run_configured_final(
            manifest,
            drift_runtime,
            output_root=tmp_path,
            policy=NetworkPolicy(),
        )
    assert drift_provider.calls == 0


def test_final_analysis_config_is_rejected_before_provider_calls(tmp_path: Path):
    manifest = _small_formal_manifest()
    data = copy.deepcopy(dict(manifest.data))
    data["selection_protocol"]["normalization_ranges"][0] = [1.0, 1.0]
    manifest = replace(manifest, data=data)
    provider = AlwaysCorrectProvider()

    with pytest.raises(
        ExperimentRunError,
        match="invalid frozen final-analysis configuration",
    ):
        run_configured_final(
            manifest,
            _runtime(provider, tasks_per_role=3, include_final=True),
            output_root=tmp_path,
            policy=NetworkPolicy(),
        )

    assert provider.calls == 0


def test_selected_artifact_freezes_native_and_all_deployment_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    manifest = _small_formal_manifest()
    configuration = copy.deepcopy(dict(manifest.data))
    configuration["budgets"]["search_total_per_method"].update(
        {
            "task_executions": 12,
            "maximum_unique_screened_candidates": 2,
            "maximum_promoted_candidates": 2,
        }
    )
    configuration["statistics"]["minimum_effective_blocks_for_archive"] = 1
    runtime = _runtime_with_patch_pool(AlwaysCorrectProvider(), patch_count=2)

    def two_policy_payload(**kwargs):
        ids = sorted(candidate.candidate_id for candidate in kwargs["full_scored"])
        assert len(ids) == 2
        return {
            "schema_version": 1,
            "method_id": kwargs["method_id"],
            "search_seed": kwargs["search_seed"],
            "primary_policy": "first",
            "primary_candidate_id": ids[0],
            "policies": {
                "first": {"defined": True, "candidate_id": ids[0]},
                "second": {"defined": True, "candidate_id": ids[1]},
            },
        }

    monkeypatch.setattr(experiment_runner_module, "_deployment_payload", two_policy_payload)
    summary = _run_method(
        manifest=manifest,
        runtime=runtime,
        phase=runtime.phases["search"],
        method_id="fixed_scalarization_accuracy_only",
        plugin_id="fixed_scalarization/accuracy_only",
        configuration=configuration,
        search_seed=104729,
        output=tmp_path / "union-run",
        cache=ResultCache(tmp_path / "cache"),
        policy=NetworkPolicy(),
        smoke=False,
    )
    selected_rows = [
        json.loads(line)
        for line in (summary.output_directory / "selected_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    assert len(summary.plugin_native_selected_candidate_ids) == 1
    assert len(summary.selected_candidate_ids) == 2
    assert {row["candidate_id"] for row in selected_rows} == set(
        summary.selected_candidate_ids
    )
    assert sum(row["selection_sources"]["plugin_native"] for row in selected_rows) == 1
    assert {
        policy
        for row in selected_rows
        for policy in row["selection_sources"]["deployment_policy_ids"]
    } == {"first", "second"}


def test_false_archive_admission_uses_only_full_validated_denominator(
    tmp_path: Path,
):
    manifest = load_manifest(CONFIG, profile="dry_run")
    runtime = _runtime_with_patch_pool(AlwaysCorrectProvider(), patch_count=2)
    summary = _run_method(
        manifest=manifest,
        runtime=runtime,
        phase=runtime.phases["search"],
        method_id="passive_archive",
        plugin_id="passive_archive",
        configuration=manifest.data,
        search_seed=104729,
        output=tmp_path / "false-admission-run",
        cache=ResultCache(tmp_path / "cache"),
        policy=NetworkPolicy(),
        smoke=True,
        smoke_candidate_limit=2,
    )
    metrics = json.loads(
        (summary.output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    analysis = metrics["frontier_analysis"]["false_archive_admission"]

    assert analysis["denominator_definition"] == (
        "screen_admitted_intersection_full_validated"
    )
    assert analysis["eligible_admission_count"] == 1
    assert len(analysis["excluded_unvalidated_ids"]) == 1
    assert analysis["rate"]["value"] == 0.0


def test_final_outputs_three_seed_exact_holm_and_heldout_method_fronts(
    tmp_path: Path,
):
    manifest = _small_formal_manifest()
    data = copy.deepcopy(dict(manifest.data))
    seeds = [101, 103, 107]
    data["runner"]["final"] = {
        "required_method_ids": ["trace2skill_all", "paretoskill"],
        "required_search_seeds": seeds,
    }
    data["statistics"]["heldout_comparisons"]["holm_family"][
        "matched_run_ids"
    ] = ["trace2skill_all"]
    data["budgets"]["search_total_per_method"].update(
        {
            "task_executions": 6,
            "maximum_unique_screened_candidates": 1,
            "screen_allocation_task_executions": 2,
            "maximum_promoted_candidates": 1,
            "incremental_promotion_allocation_task_executions": 4,
            "allocation_sum_check": 6,
        }
    )
    data["methods"]["paretoskill"][
        "logical_task_execution_budget_per_search_seed"
    ] = 6
    data["statistics"]["minimum_effective_blocks_for_archive"] = 1
    manifest = replace(manifest, data=data)
    provider = AlwaysCorrectProvider()
    runtime = _runtime(provider, tasks_per_role=3, include_final=True)
    runtime.proposer_factory = lambda *args: DeterministicMockProposer()

    search = run_configured_search(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all", "paretoskill"),
        search_seeds=seeds,
        smoke=False,
    )
    final = run_configured_final(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
    )
    metrics = json.loads(
        (final.output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    primary = metrics["primary_analysis"]
    paired = primary["paired_three_seed_sign_flip_holm"]

    assert len(primary["heldout_fronts"]["runs"]) == 6
    assert paired["family_complete"] is True
    assert paired["defined"] is True
    contrast = paired["contrasts"]["trace2skill_all"]
    assert contrast["seeds"] == seeds
    assert contrast["exact_two_sided_p"]["defined"] is True
    assert contrast["holm_adjusted_p"]["defined"] is True
    root_metrics = json.loads(
        (search.output_directory / "metrics.json").read_text(encoding="utf-8")
    )
    assert root_metrics["heldout_frontier_comparison"][
        "paired_holm_family_complete"
    ] is True


def test_final_budget_mismatch_fails_before_provider_execution(tmp_path: Path):
    manifest = _small_formal_manifest()
    data = copy.deepcopy(dict(manifest.data))
    data["budgets"]["final"]["task_executions"] = 3
    manifest = replace(manifest, data=data)
    search_provider = AlwaysCorrectProvider()
    run_configured_search(
        manifest,
        _runtime(search_provider, tasks_per_role=3),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=False,
    )
    final_provider = AlwaysCorrectProvider()
    with pytest.raises(
        ExperimentRunError,
        match="frozen final budget does not match",
    ):
        run_configured_final(
            manifest,
            _runtime(final_provider, tasks_per_role=3, include_final=True),
            output_root=tmp_path,
            policy=NetworkPolicy(),
        )
    assert final_provider.calls == 0


def test_final_requires_exact_frozen_method_seed_matrix(tmp_path: Path):
    manifest = _small_formal_manifest()
    run_configured_search(
        manifest,
        _runtime(AlwaysCorrectProvider(), tasks_per_role=3),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("base_skill",),
        search_seeds=(104729,),
        smoke=False,
    )
    final_provider = AlwaysCorrectProvider()
    with pytest.raises(
        ExperimentRunError,
        match="exact frozen method×seed matrix",
    ):
        run_configured_final(
            manifest,
            _runtime(final_provider, tasks_per_role=3, include_final=True),
            output_root=tmp_path,
            policy=NetworkPolicy(),
        )
    assert final_provider.calls == 0


def test_paretoskill_smoke_exercises_archive_conditioned_generation(tmp_path: Path):
    manifest = load_manifest(CONFIG, profile="dry_run")
    provider = AlwaysCorrectProvider()
    runtime = _runtime(provider)
    factory_calls = 0

    def proposer_factory(parent_resolver, seed, effective_configuration):
        nonlocal factory_calls
        del parent_resolver, seed, effective_configuration
        factory_calls += 1
        return DeterministicMockProposer()

    runtime.proposer_factory = proposer_factory
    summary = run_configured_search(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("paretoskill",),
        search_seeds=(104729,),
        smoke=True,
    ).method_runs[0]

    assert factory_calls == 1
    assert summary.screened_candidates == 2
    assert summary.proposed_candidates == 2
    assert (summary.output_directory / "screen_archive.json").is_file()
    assert (summary.output_directory / "archive.json").is_file()
    assert (summary.output_directory / "scientific_front.json").is_file()


@pytest.mark.parametrize(
    "method_id",
    ("passive_archive", "ablation_passive_archive"),
)
def test_passive_archive_variants_do_not_call_the_proposer(
    tmp_path: Path,
    method_id: str,
):
    manifest = load_manifest(CONFIG, profile="dry_run")
    runtime = _runtime(AlwaysCorrectProvider())

    def forbidden_factory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("passive archive must not construct a proposer")

    runtime.proposer_factory = forbidden_factory
    summary = run_configured_search(
        manifest,
        runtime,
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=(method_id,),
        search_seeds=(104729,),
        smoke=True,
    ).method_runs[0]

    assert summary.selected_candidate_ids


def test_declared_smoke_contract_defaults_are_bounded_and_namespaced(
    tmp_path: Path,
):
    manifest = _bounded_smoke_manifest()
    provider = AlwaysCorrectProvider()
    result = run_configured_search(
        manifest,
        _runtime(provider),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        smoke=True,
    )

    assert [run.method_id for run in result.method_runs] == [
        "no_skill",
        "base_skill",
        "simple_patch_composition",
        "paretoskill",
    ]
    assert {run.search_seed for run in result.method_runs} == {104729}
    assert sum(run.logical_task_executions for run in result.method_runs) <= 16
    assert result.output_directory.name == "smoke"
    assert result.output_directory.parent.name == manifest.experiment_id
    accounting = json.loads(
        (result.output_directory / "token_accounting.json").read_text(encoding="utf-8")
    )
    assert accounting["evaluation_cost_reconciliation"]["all_prices_defined"] is False
    assert accounting["proposal_cost_reconciliation"]["total_monetary_cost"] is None


def test_smoke_ceiling_drift_fails_before_provider_calls(tmp_path: Path):
    manifest = _bounded_smoke_manifest(ceiling=15)
    provider = AlwaysCorrectProvider()
    with pytest.raises(ExperimentRunError, match="no provider request was made"):
        run_configured_search(
            manifest,
            _runtime(provider),
            output_root=tmp_path,
            policy=NetworkPolicy(),
            smoke=True,
        )
    assert provider.calls == 0
