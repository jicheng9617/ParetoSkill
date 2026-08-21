"""Provider/model/harness/domain-independent paired evaluation contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol

from .failures import (
    FailureEvent,
    PairedEvaluationFailure,
    VerifiedInfrastructureFailure,
)
from .models import SkillVersion, freeze_mapping, stable_hash
from .providers import (
    ExecutionRequest,
    ExecutionResult,
    ModelSpec,
    NetworkPolicy,
    Provider,
    SafetyGatedProvider,
)


EvaluationIdentity = tuple[str, str, str, int]
InjectionMode = Literal["skill", "none"]
NO_SKILL_CONTENT_HASH = stable_hash({"format": 1, "files": {}})


def _lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    split: str
    domain_id: str
    group_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    split_id: str | None = None
    objective_role: str | None = None

    def __post_init__(self) -> None:
        identifiers = (self.task_id, self.split, self.domain_id, self.group_id)
        if not all(value.strip() for value in identifiers):
            raise ValueError("task identifiers must be non-empty")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        resolved_split_id = self.split if self.split_id is None else self.split_id
        resolved_role = self.split if self.objective_role is None else self.objective_role
        if not isinstance(resolved_split_id, str) or not resolved_split_id.strip():
            raise ValueError("task split_id must be non-empty")
        if resolved_role not in {"id", "transfer", "final", "diagnostic"}:
            raise ValueError("task objective_role is invalid")
        object.__setattr__(self, "split_id", resolved_split_id)
        object.__setattr__(self, "objective_role", resolved_role)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    target_id: str
    provider_id: str
    model: ModelSpec
    harness_id: str
    domain_id: str
    task_group: str
    split_id: str | None = None
    transfer_group: str | None = None
    objective_role: str | None = None

    def __post_init__(self) -> None:
        identifiers = (
            self.target_id,
            self.provider_id,
            self.harness_id,
            self.domain_id,
            self.task_group,
        )
        if not all(isinstance(value, str) and value.strip() for value in identifiers):
            raise ValueError("target identifiers must be non-empty strings")
        if self.provider_id != self.model.provider_id:
            raise ValueError("target provider_id must match its model provider_id")
        if self.split_id is not None and (
            not isinstance(self.split_id, str) or not self.split_id.strip()
        ):
            raise ValueError("target split_id must be a non-empty string or null")
        if self.transfer_group is not None and (
            not isinstance(self.transfer_group, str) or not self.transfer_group.strip()
        ):
            raise ValueError("target transfer_group must be a non-empty string or null")
        resolved_role = self.objective_role
        if resolved_role is None and self.task_group in {"id", "transfer"}:
            resolved_role = self.task_group
        if resolved_role is not None and resolved_role not in {
            "id",
            "transfer",
            "final",
            "diagnostic",
        }:
            raise ValueError("target objective_role is invalid")
        object.__setattr__(self, "objective_role", resolved_role)


def target_specs_from_manifest(
    configuration: Mapping[str, Any], *, phase: str | None = None
) -> tuple[TargetSpec, ...]:
    """Build provider-neutral target specs from a resolved experiment manifest."""

    raw_models = configuration.get("models")
    raw_targets = configuration.get("targets")
    raw_objectives = configuration.get("objectives")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_models, raw_targets, raw_objectives)
    ):
        raise ValueError("manifest models, targets, and objectives must be mappings")
    assert isinstance(raw_models, Mapping)
    assert isinstance(raw_targets, Mapping)
    assert isinstance(raw_objectives, Mapping)
    id_spec = raw_objectives.get("id_accuracy")
    transfer_spec = raw_objectives.get("worst_target_transfer")
    if not isinstance(id_spec, Mapping) or not isinstance(transfer_spec, Mapping):
        raise ValueError("manifest objective target sets are incomplete")
    id_targets = set(id_spec.get("target_ids", ()))
    transfer_targets = set(transfer_spec.get("target_ids", ()))
    result: list[TargetSpec] = []
    for target_id, raw_target in sorted(raw_targets.items()):
        if not isinstance(target_id, str) or not isinstance(raw_target, Mapping):
            raise ValueError("manifest target entries must map string ids to objects")
        target_phase = raw_target.get("phase")
        if phase is not None and target_phase != phase:
            continue
        model_key = raw_target.get("model")
        if not isinstance(model_key, str) or not isinstance(
            raw_models.get(model_key), Mapping
        ):
            raise ValueError(f"target {target_id!r} references an invalid model")
        raw_model = raw_models[model_key]
        assert isinstance(raw_model, Mapping)
        if target_id in id_targets:
            objective_role = "id"
        elif target_id in transfer_targets:
            objective_role = "transfer"
        elif target_phase == "final_only":
            objective_role = "final"
        else:
            objective_role = "diagnostic"
        decoding = raw_model.get("decoding", {})
        if not isinstance(decoding, Mapping):
            raise ValueError(f"model {model_key!r} decoding must be a mapping")
        result.append(
            TargetSpec(
                target_id=target_id,
                provider_id=raw_model["provider"],
                model=ModelSpec(
                    model_id=raw_model["model_id"],
                    provider_id=raw_model["provider"],
                    revision=raw_model["revision"],
                    decoding=decoding,
                ),
                harness_id=raw_target["harness"],
                domain_id=raw_target["domain"],
                task_group="*",
                split_id=raw_target["split"],
                transfer_group=raw_target.get("transfer_group"),
                objective_role=objective_role,
            )
        )
    if not result:
        raise ValueError(f"manifest declares no targets for phase {phase!r}")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TaskSeedBlock:
    block_id: str
    task: TaskSpec
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("block_id must be non-empty")
        if not isinstance(self.task, TaskSpec):
            raise ValueError("task-seed blocks require a TaskSpec")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("task-seed block seed must be an integer")

    @property
    def pair_key(self) -> str:
        return f"{self.task.task_id}::seed={self.seed}"


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    """One execution condition, including an explicit no-skill control."""

    candidate_id: str
    version: SkillVersion
    injection_mode: InjectionMode = "skill"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("evaluation candidate_id must be non-empty")
        if not isinstance(self.version, SkillVersion):
            raise ValueError("evaluation candidate requires a SkillVersion")
        if self.injection_mode not in {"skill", "none"}:
            raise ValueError("injection_mode must be 'skill' or 'none'")
        if (
            self.injection_mode == "skill"
            and self.candidate_id != self.version.lineage.version_id
        ):
            raise ValueError("skill injection candidate_id must equal its version_id")
        if (
            self.injection_mode == "none"
            and self.candidate_id == self.version.lineage.version_id
        ):
            raise ValueError("no-skill candidate_id must differ from its anchor version_id")

    @classmethod
    def skill(cls, version: SkillVersion) -> EvaluationCandidate:
        return cls(version.lineage.version_id, version, "skill")

    @classmethod
    def no_skill(
        cls, anchor: SkillVersion, *, candidate_id: str = "no-skill"
    ) -> EvaluationCandidate:
        return cls(candidate_id, anchor, "none")

    @property
    def content_hash(self) -> str:
        return (
            self.version.skill.content_hash
            if self.injection_mode == "skill"
            else NO_SKILL_CONTENT_HASH
        )

    @property
    def skill_files(self) -> Mapping[str, str]:
        return self.version.skill.files if self.injection_mode == "skill" else {}


CandidateLike = SkillVersion | EvaluationCandidate


def _as_evaluation_candidate(value: CandidateLike) -> EvaluationCandidate:
    if isinstance(value, EvaluationCandidate):
        return value
    if isinstance(value, SkillVersion):
        return EvaluationCandidate.skill(value)
    raise ValueError("candidates must be SkillVersion or EvaluationCandidate values")


def _compatible(target: TargetSpec, block: TaskSeedBlock) -> bool:
    if block.task.domain_id != target.domain_id:
        return False
    if target.split_id is not None and block.task.split_id != target.split_id:
        return False
    if (
        target.objective_role is not None
        and block.task.objective_role != target.objective_role
    ):
        return False
    return target.task_group == "*" or block.task.group_id == target.task_group


def _retry_request_fingerprint(
    *,
    experiment_id: str,
    candidate: EvaluationCandidate,
    target: TargetSpec,
    block: TaskSeedBlock,
    harness: Harness,
) -> str:
    """Hash the exact retry invocation without serializing task/skill contents."""

    harness_revision = getattr(harness, "harness_revision", None)
    if harness_revision is not None and not isinstance(harness_revision, str):
        raise ValueError("harness_revision must be a string or null")
    return stable_hash(
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "candidate_id": candidate.candidate_id,
            "content_hash": candidate.content_hash,
            "injection_mode": candidate.injection_mode,
            "target_id": target.target_id,
            "provider_id": target.provider_id,
            "model_id": target.model.model_id,
            "model_revision": target.model.revision,
            "decoding_sha256": stable_hash(target.model.decoding),
            "harness_id": target.harness_id,
            "harness_revision": harness_revision,
            "domain_id": target.domain_id,
            "split_id": target.split_id,
            "block_id": block.block_id,
            "task_id": block.task.task_id,
            "task_payload_sha256": stable_hash(block.task.payload),
            "seed": block.seed,
        }
    )


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    experiment_id: str
    candidate_id: str
    content_hash: str
    target_id: str
    task_id: str
    group_id: str
    split: str
    seed: int
    result: ExecutionResult
    cache_key: str
    is_base: bool = False
    split_id: str | None = None
    transfer_group: str | None = None

    def __post_init__(self) -> None:
        identifiers = (
            self.experiment_id,
            self.candidate_id,
            self.target_id,
            self.task_id,
            self.group_id,
            self.split,
        )
        if not all(isinstance(value, str) and value.strip() for value in identifiers):
            raise ValueError("evaluation identifiers must be non-empty strings")
        if not isinstance(self.content_hash, str) or not _lower_sha256(self.content_hash):
            raise ValueError("evaluation content_hash must be lowercase SHA-256 hex")
        if not isinstance(self.cache_key, str) or not _lower_sha256(self.cache_key):
            raise ValueError("evaluation cache_key must be lowercase SHA-256 hex")
        if not isinstance(self.result, ExecutionResult):
            raise ValueError("evaluation result must be an ExecutionResult")
        if not isinstance(self.is_base, bool):
            raise ValueError("is_base must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("evaluation seed must be an integer")
        resolved_split_id = self.split if self.split_id is None else self.split_id
        if not isinstance(resolved_split_id, str) or not resolved_split_id.strip():
            raise ValueError("evaluation split_id must be non-empty")
        if self.transfer_group is not None and (
            not isinstance(self.transfer_group, str) or not self.transfer_group.strip()
        ):
            raise ValueError("evaluation transfer_group must be a string or null")
        object.__setattr__(self, "split_id", resolved_split_id)

    @property
    def block_key(self) -> tuple[str, int, str]:
        return (self.task_id, self.seed, self.target_id)

    @property
    def evaluation_identity(self) -> EvaluationIdentity:
        return (self.candidate_id, self.target_id, self.task_id, self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "target_id": self.target_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "split": self.split,
            "split_id": self.split_id,
            "transfer_group": self.transfer_group,
            "seed": self.seed,
            "result": self.result.to_dict(),
            "cache_key": self.cache_key,
            "is_base": self.is_base,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationRecord:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported evaluation record schema")
        is_base = value.get("is_base", False)
        if not isinstance(is_base, bool):
            raise ValueError("is_base must be a JSON boolean")
        return cls(
            experiment_id=value["experiment_id"],
            candidate_id=value["candidate_id"],
            content_hash=value["content_hash"],
            target_id=value["target_id"],
            task_id=value["task_id"],
            group_id=value["group_id"],
            split=value["split"],
            seed=value["seed"],
            result=ExecutionResult.from_dict(value["result"]),
            cache_key=value["cache_key"],
            is_base=is_base,
            split_id=value.get("split_id", value["split"]),
            transfer_group=value.get("transfer_group"),
        )


class Harness(Protocol):
    harness_id: str

    def evaluate(
        self,
        *,
        provider: Provider,
        experiment_id: str,
        candidate: EvaluationCandidate,
        target: TargetSpec,
        block: TaskSeedBlock,
        is_base: bool,
    ) -> EvaluationRecord: ...


class EvaluationCache(Protocol):
    def get(self, cache_key: str) -> EvaluationRecord | None: ...

    def put(self, record: EvaluationRecord) -> None: ...


class DomainAdapter(Protocol):
    domain_id: str

    def tasks(self, split: str) -> Iterable[TaskSpec]: ...


def expected_evaluation_matrix(
    *,
    base: SkillVersion,
    candidates: Iterable[CandidateLike],
    targets: Iterable[TargetSpec],
    blocks: Iterable[TaskSeedBlock],
) -> frozenset[EvaluationIdentity]:
    """Build and validate the exact candidate-target-task-seed execution matrix."""

    execution_candidates = (
        EvaluationCandidate.skill(base),
        *tuple(_as_evaluation_candidate(candidate) for candidate in candidates),
    )
    target_list = tuple(targets)
    block_list = tuple(blocks)
    candidate_ids = [candidate.candidate_id for candidate in execution_candidates]
    target_ids = [target.target_id for target in target_list]
    block_ids = [block.block_id for block in block_list]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("paired evaluation requires unique candidate version ids")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("paired evaluation requires unique target ids")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("paired evaluation requires unique task-seed block ids")
    if not target_list:
        raise ValueError("paired evaluation requires at least one target")
    if not block_list:
        raise ValueError("paired evaluation requires at least one task-seed block")

    expected: set[EvaluationIdentity] = set()
    for target in target_list:
        compatible = [
            block
            for block in block_list
            if _compatible(target, block)
        ]
        if not compatible:
            raise ValueError(f"target {target.target_id!r} has no compatible task-seed blocks")
        target_keys = [(block.task.task_id, block.seed) for block in compatible]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError(
                f"target {target.target_id!r} has duplicate task-seed execution identities"
            )
        expected.update(
            (candidate_id, target.target_id, block.task.task_id, block.seed)
            for candidate_id in candidate_ids
            for block in compatible
        )
    return frozenset(expected)


def validate_evaluation_matrix(
    records: Iterable[EvaluationRecord],
    expected: frozenset[EvaluationIdentity],
) -> tuple[EvaluationRecord, ...]:
    """Reject missing, extra, or duplicate task-level outcomes before statistics."""

    rows = tuple(records)
    identities = [record.evaluation_identity for record in rows]
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate evaluation records: {duplicates[:3]}")
    actual = set(identities)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "evaluation matrix mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return rows


@dataclass(slots=True)
class ProviderHarness:
    """Generic harness for providers that return already-verifiable structured outcomes."""

    harness_id: str = "provider-structured"
    harness_revision: str = "fixture-v1"
    cache: EvaluationCache | None = None

    def evaluate(
        self,
        *,
        provider: Provider,
        experiment_id: str,
        candidate: EvaluationCandidate,
        target: TargetSpec,
        block: TaskSeedBlock,
        is_base: bool,
    ) -> EvaluationRecord:
        if target.harness_id != self.harness_id:
            raise ValueError(
                f"target requests harness {target.harness_id!r}, got {self.harness_id!r}"
            )
        if not _compatible(target, block):
            raise ValueError("task block is incompatible with target domain/split/group")
        request = ExecutionRequest(
            experiment_id=experiment_id,
            candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash,
            task_id=block.task.task_id,
            seed=block.seed,
            target_id=target.target_id,
            model=target.model,
            skill_files=candidate.skill_files,
            task_payload=block.task.payload,
            metadata={
                "group_id": block.task.group_id,
                "split": target.objective_role or block.task.objective_role,
                "split_id": target.split_id or block.task.split_id,
                "transfer_group": target.transfer_group,
                "harness_id": self.harness_id,
                "harness_revision": self.harness_revision,
                "injection_mode": candidate.injection_mode,
            },
        )
        cached = self.cache.get(request.cache_key) if self.cache is not None else None
        result = cached.result if cached is not None else provider.execute(request)
        result = ExecutionResult(
            correct=result.correct,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            trace=result.trace,
            provider_metadata={
                **dict(result.provider_metadata),
                "injection_mode": candidate.injection_mode,
            },
        )
        record = EvaluationRecord(
            experiment_id=experiment_id,
            candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash,
            target_id=target.target_id,
            task_id=block.task.task_id,
            group_id=block.task.group_id,
            split=target.objective_role or block.task.objective_role or block.task.split,
            seed=block.seed,
            result=result,
            cache_key=request.cache_key,
            is_base=is_base,
            split_id=target.split_id or block.task.split_id,
            transfer_group=target.transfer_group,
        )
        if cached is None and self.cache is not None:
            self.cache.put(record)
        return record


@dataclass(slots=True)
class PairedEvaluator:
    """Execute base and candidates on exactly the same declared blocks and targets."""

    network_policy: NetworkPolicy = field(default_factory=NetworkPolicy)
    retry_limit: int = 0
    _failure_events: list[FailureEvent] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.retry_limit, bool)
            or not isinstance(self.retry_limit, int)
            or self.retry_limit < 0
        ):
            raise ValueError("retry_limit must be a non-negative integer")

    @property
    def failure_events(self) -> tuple[FailureEvent, ...]:
        """Sanitized verified-infrastructure events from the latest evaluate call."""

        return tuple(self._failure_events)

    def _evaluate_one(
        self,
        *,
        harness: Harness,
        provider: Provider,
        experiment_id: str,
        candidate: EvaluationCandidate,
        target: TargetSpec,
        block: TaskSeedBlock,
        is_base: bool,
    ) -> EvaluationRecord:
        request_fingerprint = _retry_request_fingerprint(
            experiment_id=experiment_id,
            candidate=candidate,
            target=target,
            block=block,
            harness=harness,
        )
        for attempt_index in range(self.retry_limit + 1):
            try:
                return harness.evaluate(
                    provider=provider,
                    experiment_id=experiment_id,
                    candidate=candidate,
                    target=target,
                    block=block,
                    is_base=is_base,
                )
            except VerifiedInfrastructureFailure as failure:
                event = FailureEvent(
                    category=failure.category,
                    evidence_sha256=failure.evidence_sha256,
                    experiment_id=experiment_id,
                    candidate_id=candidate.candidate_id,
                    target_id=target.target_id,
                    task_id=block.task.task_id,
                    seed=block.seed,
                    request_fingerprint=request_fingerprint,
                    attempt_number=attempt_index + 1,
                    retry_limit=self.retry_limit,
                    will_retry=attempt_index < self.retry_limit,
                )
                self._failure_events.append(event)
                if event.will_retry:
                    continue
                raise PairedEvaluationFailure(tuple(self._failure_events)) from None
        raise AssertionError("retry loop exited without a result or explicit failure")

    def evaluate(
        self,
        *,
        experiment_id: str,
        base: SkillVersion,
        candidates: Iterable[CandidateLike],
        targets: Iterable[TargetSpec],
        blocks: Iterable[TaskSeedBlock],
        providers: Mapping[str, Provider],
        harnesses: Mapping[str, Harness],
    ) -> list[EvaluationRecord]:
        self._failure_events = []
        candidate_list = [
            EvaluationCandidate.skill(base),
            *[_as_evaluation_candidate(candidate) for candidate in candidates],
        ]
        target_list = sorted(targets, key=lambda target: target.target_id)
        block_list = sorted(blocks, key=lambda block: (block.block_id, block.seed))
        expected = expected_evaluation_matrix(
            base=base,
            candidates=candidate_list[1:],
            targets=target_list,
            blocks=block_list,
        )
        records: list[EvaluationRecord] = []
        for target in target_list:
            raw_provider = providers[target.provider_id]
            if raw_provider.provider_id != target.provider_id:
                raise ValueError("provider registry key does not match provider.provider_id")
            provider = SafetyGatedProvider(
                raw_provider,
                self.network_policy,
            )
            harness = harnesses[target.harness_id]
            for block in block_list:
                if not _compatible(target, block):
                    continue
                for candidate in candidate_list:
                    records.append(
                        self._evaluate_one(
                            harness=harness,
                            provider=provider,
                            experiment_id=experiment_id,
                            candidate=candidate,
                            target=target,
                            block=block,
                            is_base=candidate.candidate_id == base.lineage.version_id,
                        )
                    )
        return list(validate_evaluation_matrix(records, expected))
