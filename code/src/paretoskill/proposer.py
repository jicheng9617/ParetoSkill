"""Archive-conditioned mutation protocols with auditable generation caching."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .archive import ArchiveEntry
from .metrics import crowding_distance
from .models import (
    Patch,
    PatchOperation,
    SkillVersion,
    TraceEvidence,
    canonical_json,
    stable_hash,
    thaw_json,
)
from .providers import (
    ExecutionRequest,
    GeneratedResponse,
    ModelSpec,
    NetworkPolicy,
    ProviderError,
)


class ObjectiveDirection(str, Enum):
    ACCURACY = "id_accuracy"
    TRANSFER = "worst_target_transfer"
    COST = "token_cost"
    REGRESSION = "paired_regression"


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    direction: ObjectiveDirection
    traces: tuple[TraceEvidence, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", ObjectiveDirection(self.direction))
        object.__setattr__(self, "traces", tuple(self.traces))
        object.__setattr__(self, "notes", tuple(self.notes))
        if not self.traces:
            raise ValueError("archive-conditioned proposals require trace evidence")
        if any(not isinstance(trace, TraceEvidence) for trace in self.traces):
            raise ValueError("evidence bundle traces must be TraceEvidence instances")
        evidence_ids = [trace.evidence_id for trace in self.traces]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence bundle contains duplicate evidence ids")
        if any(not isinstance(note, str) or not note.strip() for note in self.notes):
            raise ValueError("evidence notes must be non-empty strings")


@dataclass(frozen=True, slots=True)
class MutationRequest:
    parent_version_id: str
    parent_candidate_id: str
    direction: ObjectiveDirection
    evidence: EvidenceBundle
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent_version_id, str) or not self.parent_version_id.strip():
            raise ValueError("parent_version_id must be non-empty")
        if not isinstance(self.parent_candidate_id, str) or not self.parent_candidate_id.strip():
            raise ValueError("parent_candidate_id must be non-empty")
        object.__setattr__(self, "direction", ObjectiveDirection(self.direction))
        if not isinstance(self.evidence, EvidenceBundle):
            raise ValueError("mutation evidence must be an EvidenceBundle")
        if self.evidence.direction is not self.direction:
            raise ValueError("mutation direction must match its evidence bundle")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("mutation sequence must be a non-negative integer")


class MutationProposer(Protocol):
    proposer_id: str

    def propose(self, request: MutationRequest) -> Patch: ...


class GenerationProvider(Protocol):
    provider_id: str

    def generate(self, request: ExecutionRequest) -> GeneratedResponse: ...


@dataclass(slots=True)
class ArchiveConditioner:
    """Choose a sparse parent and its weakest normalized pessimistic objective."""

    def choose(
        self,
        entries: tuple[ArchiveEntry, ...],
        *,
        evidence_by_direction: Mapping[ObjectiveDirection, tuple[TraceEvidence, ...]],
        seed: int,
    ) -> tuple[ArchiveEntry, ObjectiveDirection]:
        if not entries:
            raise ValueError("cannot condition generation on an empty archive")
        candidate_ids = [entry.candidate_id for entry in entries]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("archive conditioning requires unique candidate ids")
        vectors = {
            entry.candidate_id: entry.objectives.pessimistic_vector() for entry in entries
        }
        distances = crowding_distance(vectors)
        rng = random.Random(seed)
        ranked = sorted(
            entries,
            key=lambda entry: (
                -distances[entry.candidate_id],
                entry.candidate_id,
            ),
        )
        best_distance = distances[ranked[0].candidate_id]
        tied = [entry for entry in ranked if distances[entry.candidate_id] == best_distance]
        parent = tied[rng.randrange(len(tied))]

        available = [
            direction
            for direction in ObjectiveDirection
            if evidence_by_direction.get(direction)
        ]
        if not available:
            raise ValueError("no objective-specific evidence is available")
        parent_vector = parent.objectives.pessimistic_vector()
        normalized: dict[ObjectiveDirection, float] = {}
        for index, direction in enumerate(ObjectiveDirection):
            values = [entry.objectives.pessimistic_vector()[index] for entry in entries]
            minimum, maximum = min(values), max(values)
            normalized[direction] = (
                1.0
                if maximum <= minimum
                else (parent_vector[index] - minimum) / (maximum - minimum)
            )
        direction = min(available, key=lambda item: (normalized[item], item.value))
        return parent, direction


@dataclass(slots=True)
class DeterministicMockProposer:
    """Create auditable synthetic edits for tests; predictions are never scores."""

    proposer_id: str = "deterministic-mock"
    target_path: str = "SKILL.md"

    def propose(self, request: MutationRequest) -> Patch:
        evidence_ids = tuple(trace.evidence_id for trace in request.evidence.traces)
        if request.direction in {ObjectiveDirection.ACCURACY, ObjectiveDirection.TRANSFER}:
            operation = PatchOperation.ADD
            content = (
                f"Synthetic dry-run guidance for {request.direction.value}; "
                f"evidence={','.join(sorted(evidence_ids))}."
            )
            match_text = None
        elif request.direction is ObjectiveDirection.COST:
            operation = PatchOperation.COMPRESS
            content = "Synthetic concise guidance."
            match_text = "Synthetic verbose guidance used only by the offline fixture."
        else:
            operation = PatchOperation.DROP
            content = ""
            match_text = "Synthetic risky guidance used only by the offline fixture."
        return Patch(
            patch_id=f"mock-{request.sequence:04d}-{request.direction.value}",
            operation=operation,
            target_path=self.target_path,
            parent_version_id=request.parent_version_id,
            evidence_ids=evidence_ids,
            content=content,
            match_text=match_text,
            applicability="synthetic dry-run only",
            risk_category=request.direction.value,
            sequence=request.sequence,
            metadata={"predicted_effect_is_not_a_score": True},
        )


@dataclass(frozen=True, slots=True)
class ProposalEvent:
    patch_id: str
    request_cache_key: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider_metadata: Mapping[str, Any]
    cache_hit: bool = False


@dataclass(slots=True)
class ProviderMutationProposer:
    """Turn a policy-gated model generation into a strictly validated Patch."""

    proposer_id: str
    experiment_id: str
    provider: GenerationProvider
    model: ModelSpec
    parent_resolver: Callable[[str], SkillVersion]
    prompt_template: str
    allowed_operations: tuple[PatchOperation, ...] = tuple(PatchOperation)
    target_path_default: str = "SKILL.md"
    seed: int = 0
    include_evidence_details: bool = True
    include_parent_lineage: bool = True
    include_ancestral_patch_history: bool = True
    cache_directory: Path | None = None
    events: list[ProposalEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.proposer_id, self.experiment_id, self.prompt_template)
        ):
            raise ValueError("proposer id, experiment id, and prompt template are required")
        if self.model.provider_id != self.provider.provider_id:
            raise ValueError("proposer model and provider ids must match")
        if not callable(self.parent_resolver):
            raise ValueError("parent_resolver must be callable")
        self.allowed_operations = tuple(PatchOperation(item) for item in self.allowed_operations)
        if not self.allowed_operations:
            raise ValueError("at least one patch operation must be allowed")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("proposer seed must be an integer")
        for name in (
            "include_evidence_details",
            "include_parent_lineage",
            "include_ancestral_patch_history",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.cache_directory is not None:
            self.cache_directory = Path(self.cache_directory).resolve()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_directory is None:
            return None
        return self.cache_directory / cache_key[:2] / f"{cache_key}.json"

    @staticmethod
    def _generation_payload(
        cache_key: str, generation: GeneratedResponse
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_cache_key": cache_key,
            "generation": {
                "response_text": generation.response_text,
                "parsed_output": thaw_json(generation.parsed_output),
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "latency_ms": generation.latency_ms,
                "finish_reason": generation.finish_reason,
                "provider_metadata": thaw_json(generation.provider_metadata),
            },
        }

    def _load_cached_generation(
        self, cache_key: str
    ) -> GeneratedResponse | None:
        path = self._cache_path(cache_key)
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != 1
                or payload.get("request_cache_key") != cache_key
                or not isinstance(payload.get("generation"), Mapping)
            ):
                raise ValueError("proposal cache envelope is invalid")
            generation = payload["generation"]
            assert isinstance(generation, Mapping)
            return GeneratedResponse(
                response_text=generation["response_text"],
                parsed_output=generation["parsed_output"],
                input_tokens=generation["input_tokens"],
                output_tokens=generation["output_tokens"],
                latency_ms=generation["latency_ms"],
                finish_reason=generation.get("finish_reason"),
                provider_metadata=generation.get("provider_metadata", {}),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(f"invalid proposal cache entry: {path}") from exc

    def _store_cached_generation(
        self, cache_key: str, generation: GeneratedResponse
    ) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(
            canonical_json(self._generation_payload(cache_key, generation)) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _execution_request(self, request: MutationRequest) -> ExecutionRequest:
        parent = self.parent_resolver(request.parent_version_id)
        evidence = (
            [trace.to_dict() for trace in request.evidence.traces]
            if self.include_evidence_details
            else [
                {
                    "evidence_id": trace.evidence_id,
                    "outcome": trace.outcome,
                    "tags": list(trace.tags),
                }
                for trace in request.evidence.traces
            ]
        )
        parent_payload = parent.to_dict()
        if not self.include_parent_lineage:
            parent_payload = {
                "skill": parent.skill.to_dict(),
                "lineage": {"version_id": parent.lineage.version_id},
            }
        elif not self.include_ancestral_patch_history:
            parent_payload["lineage"] = {
                **parent_payload["lineage"],
                "patch_ids": [],
                "patch_fingerprints": [],
                "evidence_ids": [],
            }
        task_payload = {
            "mode": "paretoskill_patch_proposal",
            "instruction": self.prompt_template,
            "required_output": {
                "operation": [operation.value for operation in self.allowed_operations],
                "target_path": "relative skill file path",
                "content": "replacement or added content; empty only for drop",
                "match_text": "required for drop/rewrite/compress; null for add",
                "applicability": "brief applicability condition",
                "risk_category": request.direction.value,
            },
            "objective_direction": request.direction.value,
            "parent": parent_payload,
            "evidence": evidence,
            "notes": list(request.evidence.notes),
            "constraints": [
                "Return exactly one JSON object and no prose.",
                "Cite only the supplied evidence; do not invent evaluation scores.",
                "Do not modify files outside the supplied skill artifact.",
            ],
        }
        return ExecutionRequest(
            experiment_id=self.experiment_id,
            candidate_id=request.parent_candidate_id,
            content_hash=parent.skill.content_hash,
            task_id=f"proposal-{request.sequence:08d}",
            seed=self.seed + request.sequence,
            target_id=self.proposer_id,
            model=self.model,
            skill_files=parent.skill.files,
            task_payload=task_payload,
            metadata={
                "role": "proposer",
                "objective_direction": request.direction.value,
                "response_schema": "paretoskill-patch/v1",
            },
        )

    def propose(self, request: MutationRequest) -> Patch:
        execution = self._execution_request(request)
        generation = self._load_cached_generation(execution.cache_key)
        cache_hit = generation is not None
        if generation is None:
            generation = self.provider.generate(execution)
            self._store_cached_generation(execution.cache_key, generation)
        value = thaw_json(generation.parsed_output)
        if not isinstance(value, Mapping):
            raise ProviderError("proposer output must be one JSON object")
        required = {"operation", "target_path", "content"}
        missing = required - set(value)
        if missing:
            raise ProviderError(f"proposer output is missing fields: {sorted(missing)}")
        try:
            operation = PatchOperation(value["operation"])
        except (TypeError, ValueError) as exc:
            raise ProviderError("proposer output has an invalid patch operation") from exc
        if operation not in self.allowed_operations:
            raise ProviderError(f"proposer operation {operation.value!r} is not allowed")
        target_path = value.get("target_path", self.target_path_default)
        content = value.get("content")
        match_text = value.get("match_text")
        if not isinstance(target_path, str) or not target_path.strip():
            raise ProviderError("proposer target_path must be a non-empty string")
        if not isinstance(content, str):
            raise ProviderError("proposer content must be a string")
        if match_text is not None and not isinstance(match_text, str):
            raise ProviderError("proposer match_text must be a string or null")
        if operation in {
            PatchOperation.DROP,
            PatchOperation.REWRITE,
            PatchOperation.COMPRESS,
        } and not match_text:
            raise ProviderError(f"{operation.value} proposal requires match_text")
        if operation is PatchOperation.DROP:
            content = ""
        evidence_ids = tuple(trace.evidence_id for trace in request.evidence.traces)
        identity = {
            "proposer": self.proposer_id,
            "parent": request.parent_version_id,
            "direction": request.direction.value,
            "sequence": request.sequence,
            "evidence_ids": evidence_ids,
            "output": value,
        }
        patch = Patch(
            patch_id=f"proposal-{request.sequence:08d}-{stable_hash(identity)[:12]}",
            operation=operation,
            target_path=target_path,
            parent_version_id=request.parent_version_id,
            evidence_ids=evidence_ids,
            content=content,
            match_text=match_text,
            applicability=str(value.get("applicability", "unspecified")),
            risk_category=str(value.get("risk_category", request.direction.value)),
            sequence=request.sequence,
            metadata={
                "proposer_id": self.proposer_id,
                "predicted_effect_is_not_a_score": True,
                "response_sha256": stable_hash(
                    canonical_json(thaw_json(generation.parsed_output))
                ),
            },
        )
        self.events.append(
            ProposalEvent(
                patch_id=patch.patch_id,
                request_cache_key=execution.cache_key,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                latency_ms=generation.latency_ms,
                provider_metadata=generation.provider_metadata,
                cache_hit=cache_hit,
            )
        )
        return patch


@dataclass(slots=True)
class DisabledLLMProposerAdapter:
    """Future adapter seam. It intentionally has no SDK, endpoint, or request code."""

    proposer_id: str
    policy: NetworkPolicy = field(default_factory=NetworkPolicy)

    def propose(self, request: MutationRequest) -> Patch:
        del request
        self.policy.require_external_enabled()
        raise ProviderError(
            f"LLM proposer {self.proposer_id!r} is a disabled protocol adapter; "
            "no external request was made"
        )
