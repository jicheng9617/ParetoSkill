from __future__ import annotations

import json

import pytest

from paretoskill.archive import ArchiveEntry
from paretoskill.models import Skill, TraceEvidence, make_base_version
from paretoskill.proposer import (
    ArchiveConditioner,
    EvidenceBundle,
    MutationRequest,
    ObjectiveDirection,
    ProviderMutationProposer,
)
from paretoskill.providers import (
    ExecutionResult,
    GeneratedResponse,
    ModelSpec,
    NetworkPolicy,
    ReplayProvider,
    SafetyGatedProvider,
)
from paretoskill.statistics import MetricEstimate, ObjectiveSummary


def evidence(evidence_id: str = "e1") -> TraceEvidence:
    return TraceEvidence(evidence_id, "task", 1, "target", False, "failure")


def estimate(point: float, pessimistic: float, *, upper: float | None = None) -> MetricEstimate:
    upper_value = pessimistic if upper is None else upper
    low = min(point, pessimistic, upper_value)
    high = max(point, pessimistic, upper_value)
    return MetricEstimate(point, low, high, pessimistic, upper_value, 10)


def summary(
    accuracy: tuple[float, float],
    transfer: tuple[float, float],
) -> ObjectiveSummary:
    return ObjectiveSummary(
        id_accuracy=estimate(*accuracy, upper=max(accuracy)),
        worst_target_transfer=estimate(*transfer, upper=max(transfer)),
        token_cost=estimate(10.0, 10.0, upper=10.0),
        paired_regression=estimate(0.1, 0.1, upper=0.1),
        confidence_level=0.95,
        bootstrap_replicates=100,
        bootstrap_seed=1,
        block_count=10,
    )


def test_mutation_request_requires_matching_direction_and_valid_sequence() -> None:
    bundle = EvidenceBundle(ObjectiveDirection.REGRESSION, (evidence(),))
    with pytest.raises(ValueError, match="direction must match"):
        MutationRequest("version", "candidate", ObjectiveDirection.ACCURACY, bundle, 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        MutationRequest("version", "candidate", ObjectiveDirection.REGRESSION, bundle, -1)
    with pytest.raises(ValueError, match="duplicate evidence"):
        EvidenceBundle(ObjectiveDirection.ACCURACY, (evidence(), evidence()))


def test_archive_conditioner_uses_pessimistic_direction_values() -> None:
    entries = (
        ArchiveEntry("a", "a" * 64, summary((0.9, 0.1), (0.2, 0.2))),
        ArchiveEntry("b", "b" * 64, summary((0.5, 0.5), (0.5, 0.5))),
    )
    trace = evidence()
    parent, direction = ArchiveConditioner().choose(
        entries,
        evidence_by_direction={
            ObjectiveDirection.ACCURACY: (trace,),
            ObjectiveDirection.TRANSFER: (trace,),
        },
        seed=1,
    )
    assert parent.candidate_id == "a"
    assert direction is ObjectiveDirection.ACCURACY


def test_provider_mutation_proposer_parses_auditable_json_patch() -> None:
    class FakeGenerationProvider:
        provider_id = "author"

        def generate(self, request):
            assert request.task_payload["mode"] == "paretoskill_patch_proposal"
            output = {
                "operation": "add",
                "target_path": "SKILL.md",
                "content": "Use the verified fallback.",
                "match_text": None,
                "applicability": "when the primary path fails",
                "risk_category": "id_accuracy",
            }
            return GeneratedResponse(
                response_text="fixture-json",
                parsed_output=output,
                input_tokens=12,
                output_tokens=8,
                latency_ms=1.0,
                provider_metadata={"provider": "author"},
            )

    base = make_base_version(Skill("fixture", {"SKILL.md": "# Fixture\n"}))
    bundle = EvidenceBundle(ObjectiveDirection.ACCURACY, (evidence(),))
    proposer = ProviderMutationProposer(
        proposer_id="author-proposer",
        experiment_id="experiment",
        provider=FakeGenerationProvider(),
        model=ModelSpec("author-model", "author", "v1"),
        parent_resolver=lambda version_id: base
        if version_id == base.lineage.version_id
        else (_ for _ in ()).throw(KeyError(version_id)),
        prompt_template="Return a grounded patch.",
        seed=7,
    )
    patch = proposer.propose(
        MutationRequest(
            base.lineage.version_id,
            base.lineage.version_id,
            ObjectiveDirection.ACCURACY,
            bundle,
            0,
        )
    )

    assert patch.operation.value == "add"
    assert patch.evidence_ids == ("e1",)
    assert proposer.events[0].input_tokens == 12


def test_provider_mutation_proposer_replays_frozen_generation_end_to_end(
    tmp_path,
) -> None:
    output = {
        "operation": "add",
        "target_path": "SKILL.md",
        "content": "Use the replayed verified fallback.",
        "match_text": None,
        "applicability": "when the primary path fails",
        "risk_category": "id_accuracy",
    }

    class RecordingGenerationProvider:
        provider_id = "replay"
        is_external = False

        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return GeneratedResponse(
                response_text=json.dumps(output, sort_keys=True),
                parsed_output=output,
                input_tokens=21,
                output_tokens=9,
                latency_ms=2.5,
                finish_reason="stop",
                provider_metadata={"provider": "recording-fixture"},
            )

    base = make_base_version(Skill("fixture", {"SKILL.md": "# Fixture\n"}))
    mutation = MutationRequest(
        base.lineage.version_id,
        base.lineage.version_id,
        ObjectiveDirection.ACCURACY,
        EvidenceBundle(ObjectiveDirection.ACCURACY, (evidence(),)),
        3,
    )

    def make_proposer(provider):
        return ProviderMutationProposer(
            proposer_id="replay-proposer",
            experiment_id="experiment",
            provider=provider,
            model=ModelSpec("author-model", "replay", "v1"),
            parent_resolver=lambda version_id: base
            if version_id == base.lineage.version_id
            else (_ for _ in ()).throw(KeyError(version_id)),
            prompt_template="Return one grounded patch.",
            seed=7,
        )

    recording_provider = RecordingGenerationProvider()
    recorded_proposer = make_proposer(recording_provider)
    recorded_patch = recorded_proposer.propose(mutation)
    assert len(recording_provider.requests) == 1
    request = recording_provider.requests[0]

    frozen_result = ExecutionResult(
        correct=False,
        input_tokens=21,
        output_tokens=9,
        latency_ms=2.5,
        trace={
            "response_text": json.dumps(output, sort_keys=True),
            "model_output": output,
            "finish_reason": "stop",
        },
        provider_metadata={"provider": "recording-fixture"},
    )
    replay_path = tmp_path / "proposal-replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {"cache_key": request.cache_key, "result": frozen_result.to_dict()},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    replay = SafetyGatedProvider(
        ReplayProvider.from_jsonl(replay_path, provider_id="replay"),
        NetworkPolicy(),
    )
    replayed_proposer = make_proposer(replay)
    replayed_patch = replayed_proposer.propose(mutation)

    assert replayed_patch == recorded_patch
    assert len(replayed_proposer.events) == 1
    event = replayed_proposer.events[0]
    assert event.request_cache_key == request.cache_key
    assert (event.input_tokens, event.output_tokens, event.latency_ms) == (21, 9, 2.5)
    assert event.provider_metadata["replay"] is True
    assert event.provider_metadata["provider"] == "recording-fixture"


def test_provider_mutation_proposer_resumes_from_atomic_generation_cache(
    tmp_path,
) -> None:
    class CountingProvider:
        provider_id = "author"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return GeneratedResponse(
                response_text="fixture",
                parsed_output={
                    "operation": "add",
                    "target_path": "SKILL.md",
                    "content": "Cached verified guidance.",
                    "match_text": None,
                },
                input_tokens=11,
                output_tokens=4,
                latency_ms=1.0,
            )

    base = make_base_version(Skill("fixture", {"SKILL.md": "# Fixture\n"}))
    request = MutationRequest(
        base.lineage.version_id,
        base.lineage.version_id,
        ObjectiveDirection.ACCURACY,
        EvidenceBundle(ObjectiveDirection.ACCURACY, (evidence(),)),
        0,
    )
    cache = tmp_path / "proposal-cache"

    def proposer(provider):
        return ProviderMutationProposer(
            proposer_id="cached-proposer",
            experiment_id="experiment",
            provider=provider,
            model=ModelSpec("author-model", "author", "v1"),
            parent_resolver=lambda _version_id: base,
            prompt_template="Return one grounded patch.",
            cache_directory=cache,
        )

    first_provider = CountingProvider()
    first = proposer(first_provider)
    first_patch = first.propose(request)
    second_provider = CountingProvider()
    second = proposer(second_provider)
    second_patch = second.propose(request)

    assert first_patch == second_patch
    assert first_provider.calls == 1
    assert second_provider.calls == 0
    assert first.events[0].cache_hit is False
    assert second.events[0].cache_hit is True
