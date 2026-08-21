from __future__ import annotations

from dataclasses import replace

import pytest

from paretoskill.evaluation import (
    EvaluationCandidate,
    PairedEvaluator,
    ProviderHarness,
    TargetSpec,
    TaskSeedBlock,
    TaskSpec,
    expected_evaluation_matrix,
    validate_evaluation_matrix,
)
from paretoskill.materialize import Materializer
from paretoskill.models import (
    Patch,
    PatchOperation,
    Skill,
    TraceEvidence,
    make_base_version,
)
from paretoskill.providers import (
    DisabledExternalProvider,
    ExecutionRequest,
    ExecutionResult,
    MockProvider,
    ModelSpec,
    NetworkDisabledError,
    NetworkPolicy,
    ReplayMissError,
    ReplayProvider,
)
from paretoskill.storage import ExperimentCheckpoint, ResultCache, StorageError


def request() -> ExecutionRequest:
    return ExecutionRequest(
        experiment_id="exp",
        candidate_id="candidate",
        content_hash="a" * 64,
        task_id="task",
        seed=1,
        target_id="target",
        model=ModelSpec("mock-model", "mock", "fixture-v1"),
        skill_files={"SKILL.md": "fixture"},
        task_payload={"value": 1},
    )


def test_network_requires_three_explicit_switches(monkeypatch) -> None:
    monkeypatch.delenv("PARETOSKILL_ENABLE_NETWORK", raising=False)
    provider = DisabledExternalProvider(
        "future-api",
        NetworkPolicy(config_allows_network=True, cli_allows_network=True),
    )
    with pytest.raises(NetworkDisabledError, match="No network request"):
        provider.execute(request())

    monkeypatch.setenv("PARETOSKILL_ENABLE_NETWORK", "1")
    provider = DisabledExternalProvider(
        "future-api",
        NetworkPolicy(config_allows_network=True, cli_allows_network=False),
    )
    with pytest.raises(NetworkDisabledError):
        provider.execute(request())

    with pytest.raises(ValueError, match="correct must be a boolean"):
        type(MockProvider().execute(request()))("false", 1, 1, 1.0)


def test_mock_is_deterministic_and_replay_never_falls_through() -> None:
    result = MockProvider().execute(request())
    assert result == MockProvider().execute(request())
    assert result.provider_metadata["offline"] is True

    replay = ReplayProvider(records={request().cache_key: result})
    assert replay.execute(request()) == result
    missing = replace(request(), task_id="different")
    with pytest.raises(ReplayMissError, match="fallback is forbidden"):
        replay.execute(missing)

    malformed = result.to_dict()
    malformed["correct"] = "false"
    with pytest.raises(ValueError, match="JSON boolean"):
        type(result).from_dict(malformed)


def test_paired_evaluator_uses_identical_blocks_and_storage_resumes(tmp_path) -> None:
    base = make_base_version(Skill("demo", {"SKILL.md": "base"}))
    patch = Patch(
        "p1",
        PatchOperation.ADD,
        "SKILL.md",
        base.lineage.version_id,
        ("e1",),
        content="candidate",
    )
    candidate = Materializer().materialize(
        base,
        [patch],
        evidence={
            "e1": TraceEvidence(
                "e1", "task-1", 11, "mock-target", False, "synthetic failure"
            )
        },
    )
    target = TargetSpec(
        target_id="mock-target",
        provider_id="mock",
        model=ModelSpec("mock-model", "mock", "fixture-v1"),
        harness_id="provider-structured",
        domain_id="synthetic",
        task_group="id",
    )
    block = TaskSeedBlock(
        "block-1",
        TaskSpec("task-1", "id", "synthetic", "id", {"fixture": True}),
        11,
    )
    records = PairedEvaluator().evaluate(
        experiment_id="exp",
        base=base,
        candidates=[candidate],
        targets=[target],
        blocks=[block],
        providers={"mock": MockProvider()},
        harnesses={"provider-structured": ProviderHarness()},
    )
    assert len(records) == 2
    assert {record.block_key for record in records} == {("task-1", 11, "mock-target")}
    assert sum(record.is_base for record in records) == 1

    cache = ResultCache(tmp_path / "cache")
    for record in records:
        cache.put(record)
        assert cache.get(record.cache_key) == record

    checkpoint = ExperimentCheckpoint(
        experiment_id="exp",
        completed_cache_keys={record.cache_key for record in records},
        task_executions_consumed=2,
    )
    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    assert ExperimentCheckpoint.load(path, expected_experiment_id="exp") == checkpoint
    with pytest.raises(StorageError, match="different experiment"):
        ExperimentCheckpoint.load(path, expected_experiment_id="other")


def test_no_skill_is_an_explicit_distinct_execution_condition(tmp_path) -> None:
    class CaptureProvider:
        provider_id = "capture"
        is_external = False

        def __init__(self) -> None:
            self.requests: list[ExecutionRequest] = []

        def execute(self, execution_request: ExecutionRequest) -> ExecutionResult:
            self.requests.append(execution_request)
            return ExecutionResult(True, 1, 1, 1.0)

    base = make_base_version(Skill("demo", {"SKILL.md": "base"}))
    no_skill = EvaluationCandidate.no_skill(base)
    target = TargetSpec(
        target_id="capture-target",
        provider_id="capture",
        model=ModelSpec("capture-model", "capture", "fixture-v1"),
        harness_id="provider-structured",
        domain_id="synthetic",
        task_group="id",
    )
    block = TaskSeedBlock(
        "block",
        TaskSpec("task", "id", "synthetic", "id"),
        7,
    )
    provider = CaptureProvider()
    records = PairedEvaluator().evaluate(
        experiment_id="exp",
        base=base,
        candidates=(no_skill,),
        targets=(target,),
        blocks=(block,),
        providers={"capture": provider},
        harnesses={
            "provider-structured": ProviderHarness(
                cache=ResultCache(tmp_path / "cache")
            )
        },
    )

    assert len(provider.requests) == 2
    by_mode = {request.metadata["injection_mode"]: request for request in provider.requests}
    assert by_mode["skill"].skill_files == {"SKILL.md": "base"}
    assert by_mode["none"].skill_files == {}
    assert by_mode["none"].candidate_id == "no-skill"
    assert by_mode["none"].content_hash != base.skill.content_hash
    assert by_mode["none"].cache_key != by_mode["skill"].cache_key

    records_by_id = {record.candidate_id: record for record in records}
    assert records_by_id["no-skill"].result.provider_metadata["injection_mode"] == "none"
    assert records_by_id[base.lineage.version_id].result.provider_metadata[
        "injection_mode"
    ] == "skill"
    assert records_by_id["no-skill"].is_base is False


def test_evaluator_gates_external_adapter_even_if_adapter_does_not(monkeypatch) -> None:
    class UnsafeExternal:
        provider_id = "unsafe"
        is_external = True
        called = False

        def execute(self, execution_request):
            del execution_request
            self.called = True
            raise AssertionError("safety gateway was bypassed")

    monkeypatch.delenv("PARETOSKILL_ENABLE_NETWORK", raising=False)
    provider = UnsafeExternal()
    base = make_base_version(Skill("demo", {"SKILL.md": "base"}))
    target = TargetSpec(
        target_id="unsafe-target",
        provider_id="unsafe",
        model=ModelSpec("unsafe-model", "unsafe", "fixture-v1"),
        harness_id="provider-structured",
        domain_id="synthetic",
        task_group="id",
    )
    block = TaskSeedBlock(
        "block",
        TaskSpec("task", "id", "synthetic", "id"),
        1,
    )
    with pytest.raises(NetworkDisabledError):
        PairedEvaluator().evaluate(
            experiment_id="exp",
            base=base,
            candidates=(),
            targets=(target,),
            blocks=(block,),
            providers={"unsafe": provider},
            harnesses={"provider-structured": ProviderHarness()},
        )
    assert provider.called is False


def test_evaluation_matrix_rejects_missing_and_duplicate_blocks() -> None:
    base = make_base_version(Skill("demo", {"SKILL.md": "base"}))
    target = TargetSpec(
        target_id="target",
        provider_id="mock",
        model=ModelSpec("mock-model", "mock", "fixture-v1"),
        harness_id="provider-structured",
        domain_id="synthetic",
        task_group="id",
    )
    block = TaskSeedBlock("block", TaskSpec("task", "id", "synthetic", "id"), 1)
    expected = expected_evaluation_matrix(
        base=base,
        candidates=(),
        targets=(target,),
        blocks=(block,),
    )
    with pytest.raises(ValueError, match="matrix mismatch"):
        validate_evaluation_matrix((), expected)

    duplicate = TaskSeedBlock("other-block-id", block.task, block.seed)
    with pytest.raises(ValueError, match="duplicate task-seed execution"):
        expected_evaluation_matrix(
            base=base,
            candidates=(),
            targets=(target,),
            blocks=(block, duplicate),
        )

    incompatible = replace(target, domain_id="other-domain")
    with pytest.raises(ValueError, match="no compatible task-seed blocks"):
        expected_evaluation_matrix(
            base=base,
            candidates=(),
            targets=(incompatible,),
            blocks=(block,),
        )


def test_checkpoint_strict_schema_budget_and_archive_containment(tmp_path) -> None:
    with pytest.raises(StorageError, match="non-negative integer"):
        ExperimentCheckpoint.from_dict(
            {
                "schema_version": 1,
                "experiment_id": "exp",
                "task_executions_consumed": "1",
            }
        )

    run_root = tmp_path / "run"
    checkpoint_path = run_root / "checkpoint.json"
    checkpoint = ExperimentCheckpoint(
        experiment_id="exp",
        task_executions_consumed=2,
        archive_path="../outside/archive.json",
    )
    checkpoint.save(checkpoint_path)
    with pytest.raises(StorageError, match="escapes"):
        ExperimentCheckpoint.load(
            checkpoint_path,
            expected_experiment_id="exp",
            expected_output_root=run_root,
        )
    with pytest.raises(StorageError, match="exceeds"):
        ExperimentCheckpoint.load(
            checkpoint_path,
            expected_experiment_id="exp",
            max_task_executions=1,
        )
