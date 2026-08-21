"""Fail-closed, serialization-safe infrastructure failure contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Mapping

from .models import stable_hash


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@unique
class InfrastructureFailureCategory(str, Enum):
    """The four manifest-frozen retry-eligible infrastructure categories."""

    SANDBOX_START_FAILURE = "sandbox_start_failure"
    PROVIDER_TRANSPORT_FAILURE_BEFORE_RESPONSE = (
        "provider_transport_failure_before_response"
    )
    HARNESS_CRASH_BEFORE_AGENT_OUTPUT = "harness_crash_before_agent_output"
    VERIFIER_INFRASTRUCTURE_FAILURE = "verifier_infrastructure_failure"


class VerifiedInfrastructureFailure(RuntimeError):
    """Explicit verifier-adjudicated failure; never inferred from other errors.

    The exception intentionally accepts no raw message, response, credential,
    request payload, or arbitrary metadata.  Callers retain raw evidence in
    their protected infrastructure log and provide only its SHA-256 digest.
    """

    __slots__ = ("category", "evidence_sha256")

    def __init__(
        self,
        category: InfrastructureFailureCategory,
        *,
        evidence_sha256: str,
    ) -> None:
        if not isinstance(category, InfrastructureFailureCategory):
            raise TypeError(
                "verified infrastructure category must be an "
                "InfrastructureFailureCategory"
            )
        self.category = category
        self.evidence_sha256 = _sha256(evidence_sha256, "evidence_sha256")
        super().__init__(f"verified infrastructure failure: {category.value}")


class SandboxStartFailure(VerifiedInfrastructureFailure):
    def __init__(self, *, evidence_sha256: str) -> None:
        super().__init__(
            InfrastructureFailureCategory.SANDBOX_START_FAILURE,
            evidence_sha256=evidence_sha256,
        )


class ProviderTransportFailureBeforeResponse(VerifiedInfrastructureFailure):
    def __init__(self, *, evidence_sha256: str) -> None:
        super().__init__(
            InfrastructureFailureCategory.PROVIDER_TRANSPORT_FAILURE_BEFORE_RESPONSE,
            evidence_sha256=evidence_sha256,
        )


class HarnessCrashBeforeAgentOutput(VerifiedInfrastructureFailure):
    def __init__(self, *, evidence_sha256: str) -> None:
        super().__init__(
            InfrastructureFailureCategory.HARNESS_CRASH_BEFORE_AGENT_OUTPUT,
            evidence_sha256=evidence_sha256,
        )


class VerifierInfrastructureFailure(VerifiedInfrastructureFailure):
    def __init__(self, *, evidence_sha256: str) -> None:
        super().__init__(
            InfrastructureFailureCategory.VERIFIER_INFRASTRUCTURE_FAILURE,
            evidence_sha256=evidence_sha256,
        )


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """One sanitized, immutable failed attempt for an exact evaluation request."""

    category: InfrastructureFailureCategory
    evidence_sha256: str
    experiment_id: str
    candidate_id: str
    target_id: str
    task_id: str
    seed: int
    request_fingerprint: str
    attempt_number: int
    retry_limit: int
    will_retry: bool

    SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.category, InfrastructureFailureCategory):
            raise TypeError("FailureEvent category must be explicit and retry-eligible")
        _sha256(self.evidence_sha256, "evidence_sha256")
        _sha256(self.request_fingerprint, "request_fingerprint")
        for name in ("experiment_id", "candidate_id", "target_id", "task_id"):
            _identifier(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("FailureEvent seed must be an integer")
        for name, minimum in (("attempt_number", 1), ("retry_limit", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"FailureEvent {name} must be at least {minimum}")
        if self.attempt_number > self.retry_limit + 1:
            raise ValueError("FailureEvent attempt exceeds the retry policy")
        if not isinstance(self.will_retry, bool):
            raise ValueError("FailureEvent will_retry must be boolean")
        if self.will_retry != (self.attempt_number <= self.retry_limit):
            raise ValueError("FailureEvent retry flag contradicts attempt/retry_limit")

    @property
    def failure_id(self) -> str:
        return stable_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "category": self.category.value,
            "evidence_sha256": self.evidence_sha256,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "request_fingerprint": self.request_fingerprint,
            "attempt_number": self.attempt_number,
            "retry_limit": self.retry_limit,
            "will_retry": self.will_retry,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "failure_id": self.failure_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureEvent:
        if value.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported FailureEvent schema")
        try:
            category = InfrastructureFailureCategory(value["category"])
        except (KeyError, ValueError) as exc:
            raise ValueError("FailureEvent has an ineligible category") from exc
        event = cls(
            category=category,
            evidence_sha256=value["evidence_sha256"],
            experiment_id=value["experiment_id"],
            candidate_id=value["candidate_id"],
            target_id=value["target_id"],
            task_id=value["task_id"],
            seed=value["seed"],
            request_fingerprint=value["request_fingerprint"],
            attempt_number=value["attempt_number"],
            retry_limit=value["retry_limit"],
            will_retry=value["will_retry"],
        )
        if value.get("failure_id") != event.failure_id:
            raise ValueError("FailureEvent failure_id mismatch")
        return event


class PairedEvaluationFailure(RuntimeError):
    """A verified infrastructure failure exhausted its exact-request retries."""

    __slots__ = ("events",)

    def __init__(self, events: tuple[FailureEvent, ...]) -> None:
        if not events or any(not isinstance(event, FailureEvent) for event in events):
            raise ValueError("PairedEvaluationFailure requires FailureEvent values")
        if events[-1].will_retry:
            raise ValueError("the final FailureEvent must mark retry exhaustion")
        self.events = tuple(events)
        super().__init__(
            "paired evaluation stopped after verified infrastructure retry exhaustion; "
            f"events={len(self.events)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "error_type": "paired_evaluation_failure",
            "failure_events": [event.to_dict() for event in self.events],
        }
