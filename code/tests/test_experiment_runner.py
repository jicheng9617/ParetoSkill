from pathlib import Path

from paretoskill.config import load_manifest
from paretoskill.evaluation import (
    ProviderHarness,
    TargetSpec,
    TaskSeedBlock,
    TaskSpec,
)
from paretoskill.experiment_runner import (
    ExperimentRuntime,
    PhaseRuntime,
    run_configured_search,
)
from paretoskill.models import (
    Patch,
    PatchOperation,
    Skill,
    TraceEvidence,
    make_base_version,
)
from paretoskill.providers import MockProvider, ModelSpec, NetworkPolicy


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


def _runtime() -> ExperimentRuntime:
    base = make_base_version(Skill("fixture", {"SKILL.md": "# Fixture\n"}))
    evidence = TraceEvidence(
        evidence_id="e-1",
        task_id="id-1",
        seed=17,
        target_id="id-target",
        outcome=False,
        verifier_summary="fixture",
        tags=("id_accuracy",),
    )
    patch = Patch(
        patch_id="p-1",
        operation=PatchOperation.ADD,
        target_path="SKILL.md",
        parent_version_id=base.lineage.version_id,
        evidence_ids=(evidence.evidence_id,),
        content="Use a checked procedure.",
        sequence=0,
    )
    model = ModelSpec("mock-model", "mock", "fixture-v1", {"temperature": 0.0})
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
            f"{role}-{index}",
            TaskSpec(
                task_id=f"{role}-{index}",
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
        for index in range(2)
    )
    phase = PhaseRuntime(
        targets=targets,
        blocks=blocks,
        harnesses={"provider-structured": ProviderHarness()},
    )
    return ExperimentRuntime(
        base=base,
        patches=(patch,),
        evidence={evidence.evidence_id: evidence},
        providers={"mock": MockProvider()},
        phases={"search": phase},
    )


def test_configured_smoke_run_is_resumable_and_offline(tmp_path):
    manifest = load_manifest(CONFIG, profile="dry_run")
    first = run_configured_search(
        manifest,
        _runtime(),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=True,
    )
    second = run_configured_search(
        manifest,
        _runtime(),
        output_root=tmp_path,
        policy=NetworkPolicy(),
        method_ids=("trace2skill_all",),
        search_seeds=(104729,),
        smoke=True,
    )

    assert first.stage == "smoke"
    assert first.method_runs[0].physical_provider_executions > 0
    assert second.method_runs[0].physical_provider_executions == 0
    run_output = first.method_runs[0].output_directory
    assert (run_output / "selected_candidates.jsonl").is_file()
    assert (run_output / "run_state.json").is_file()
