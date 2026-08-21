from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from paretoskill.evaluation import (
    EvaluationCandidate,
    EvaluationRecord,
    PairedEvaluator,
    ProviderHarness,
    TargetSpec,
    TaskSeedBlock,
    TaskSpec,
)
from paretoskill.failures import (
    FailureEvent,
    HarnessCrashBeforeAgentOutput,
    InfrastructureFailureCategory,
    PairedEvaluationFailure,
    ProviderTransportFailureBeforeResponse,
    SandboxStartFailure,
    VerifiedInfrastructureFailure,
    VerifierInfrastructureFailure,
)
from paretoskill.models import Skill, make_base_version
from paretoskill.providers import MockProvider, ModelSpec, TransportError


EVIDENCE_SHA256 = "a" * 64


@dataclass(slots=True)
class ScriptedHarness:
    failures: list[BaseException]
    harness_id: str = "provider-structured"
    harness_revision: str = "failure-fixture-v1"
    calls: list[tuple[object, ...]] = field(default_factory=list)
    delegate: ProviderHarness = field(default_factory=ProviderHarness)

    def evaluate(
        self,
        *,
        provider,
        experiment_id: str,
        candidate: EvaluationCandidate,
        target: TargetSpec,
        block: TaskSeedBlock,
        is_base: bool,
    ) -> EvaluationRecord:
        self.calls.append(
            (
                experiment_id,
                candidate.candidate_id,
                target.target_id,
                block.task.task_id,
                block.seed,
                is_base,
                id(candidate),
                id(target),
                id(block),
                id(provider),
            )
        )
        if self.failures:
            raise self.failures.pop(0)
        return self.delegate.evaluate(
            provider=provider,
            experiment_id=experiment_id,
            candidate=candidate,
            target=target,
            block=block,
            is_base=is_base,
        )


def evaluation_fixture(harness: ScriptedHarness):
    base = make_base_version(Skill("failure-fixture", {"SKILL.md": "base"}))
    target = TargetSpec(
        target_id="target",
        provider_id="mock",
        model=ModelSpec("mock-model", "mock", "fixture-v1"),
        harness_id=harness.harness_id,
        domain_id="synthetic",
        task_group="id",
    )
    block = TaskSeedBlock(
        "block",
        TaskSpec(
            "task",
            "id",
            "synthetic",
            "id",
            {"private_task_body": "hashed-before-event"},
        ),
        43,
    )
    return {
        "experiment_id": "failure-exp",
        "base": base,
        "candidates": (),
        "targets": (target,),
        "blocks": (block,),
        "providers": {"mock": MockProvider()},
        "harnesses": {harness.harness_id: harness},
    }


@pytest.mark.parametrize(
    ("factory", "category"),
    (
        (
            SandboxStartFailure,
            InfrastructureFailureCategory.SANDBOX_START_FAILURE,
        ),
        (
            ProviderTransportFailureBeforeResponse,
            InfrastructureFailureCategory.PROVIDER_TRANSPORT_FAILURE_BEFORE_RESPONSE,
        ),
        (
            HarnessCrashBeforeAgentOutput,
            InfrastructureFailureCategory.HARNESS_CRASH_BEFORE_AGENT_OUTPUT,
        ),
        (
            VerifierInfrastructureFailure,
            InfrastructureFailureCategory.VERIFIER_INFRASTRUCTURE_FAILURE,
        ),
    ),
)
def test_only_four_explicit_verified_categories_are_retry_eligible(
    factory: Callable[..., VerifiedInfrastructureFailure],
    category: InfrastructureFailureCategory,
) -> None:
    failure = factory(evidence_sha256=EVIDENCE_SHA256)
    assert failure.category is category
    harness = ScriptedHarness([failure])
    evaluator = PairedEvaluator(retry_limit=1)

    records = evaluator.evaluate(**evaluation_fixture(harness))

    assert len(records) == 1
    assert len(harness.calls) == 2
    assert evaluator.failure_events[0].category is category


def test_verified_category_type_and_retry_limit_are_strict() -> None:
    assert {category.value for category in InfrastructureFailureCategory} == {
        "sandbox_start_failure",
        "provider_transport_failure_before_response",
        "harness_crash_before_agent_output",
        "verifier_infrastructure_failure",
    }
    with pytest.raises(TypeError, match="InfrastructureFailureCategory"):
        VerifiedInfrastructureFailure(  # type: ignore[arg-type]
            "sandbox_start_failure",
            evidence_sha256=EVIDENCE_SHA256,
        )
    with pytest.raises(ValueError):
        InfrastructureFailureCategory("model_refusal")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        SandboxStartFailure(evidence_sha256="raw evidence is forbidden")
    with pytest.raises(ValueError, match="retry_limit"):
        PairedEvaluator(retry_limit=-1)
    with pytest.raises(ValueError, match="retry_limit"):
        PairedEvaluator(retry_limit=True)


def test_successful_retry_keeps_sanitized_event_and_exact_same_invocation() -> None:
    secret = "credential-like-value-do-not-serialize raw provider response"
    failure = SandboxStartFailure(evidence_sha256=EVIDENCE_SHA256)
    failure.__cause__ = RuntimeError(secret)
    harness = ScriptedHarness([failure])
    evaluator = PairedEvaluator(retry_limit=1)

    records = evaluator.evaluate(**evaluation_fixture(harness))

    assert len(records) == 1
    assert len(harness.calls) == 2
    assert harness.calls[0] == harness.calls[1]
    assert records[0].task_id == "task"
    assert records[0].seed == 43
    events = evaluator.failure_events
    assert len(events) == 1
    event = events[0]
    assert event.attempt_number == 1
    assert event.retry_limit == 1
    assert event.will_retry is True
    serialized = json.dumps(event.to_dict(), sort_keys=True)
    assert FailureEvent.from_dict(json.loads(serialized)) == event
    assert secret not in serialized
    assert "private_task_body" not in serialized
    assert "response" not in event.to_dict()
    assert "credential" not in event.to_dict()


def test_retry_exhaustion_raises_complete_events_without_missing_rows() -> None:
    failures = [
        ProviderTransportFailureBeforeResponse(evidence_sha256=character * 64)
        for character in ("a", "b", "c")
    ]
    harness = ScriptedHarness(failures)
    evaluator = PairedEvaluator(retry_limit=2)

    with pytest.raises(PairedEvaluationFailure) as captured:
        evaluator.evaluate(**evaluation_fixture(harness))

    assert len(harness.calls) == 3
    assert harness.calls[0] == harness.calls[1] == harness.calls[2]
    events = captured.value.events
    assert events == evaluator.failure_events
    assert [event.attempt_number for event in events] == [1, 2, 3]
    assert [event.will_retry for event in events] == [True, True, False]
    assert {event.seed for event in events} == {43}
    assert {event.task_id for event in events} == {"task"}
    assert len({event.request_fingerprint for event in events}) == 1
    payload = captured.value.to_dict()
    assert len(payload["failure_events"]) == 3
    assert json.loads(json.dumps(payload)) == payload


def test_default_retry_limit_zero_attempts_once_and_raises() -> None:
    harness = ScriptedHarness(
        [VerifierInfrastructureFailure(evidence_sha256=EVIDENCE_SHA256)]
    )
    evaluator = PairedEvaluator()

    with pytest.raises(PairedEvaluationFailure) as captured:
        evaluator.evaluate(**evaluation_fixture(harness))

    assert len(harness.calls) == 1
    assert len(captured.value.events) == 1
    assert captured.value.events[0].attempt_number == 1
    assert captured.value.events[0].will_retry is False


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("model_refusal"),
        ValueError("invalid_agent_output"),
        RuntimeError("tool_misuse"),
        TimeoutError("task_timeout_after_valid_start"),
        AssertionError("verifier_assertion_failure"),
        TransportError("unverified provider transport error"),
        OSError("ordinary infrastructure-looking exception"),
    ),
)
def test_noneligible_or_unverified_errors_are_never_retried_or_reclassified(
    error: BaseException,
) -> None:
    harness = ScriptedHarness([error])
    evaluator = PairedEvaluator(retry_limit=5)

    with pytest.raises(type(error), match=str(error)):
        evaluator.evaluate(**evaluation_fixture(harness))

    assert len(harness.calls) == 1
    assert evaluator.failure_events == ()
