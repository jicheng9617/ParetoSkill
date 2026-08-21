"""Validation-only deployment rules over a frozen archive."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .statistics import ObjectiveSummary


@dataclass(frozen=True, slots=True)
class DeploymentCandidate:
    candidate_id: str
    content_hash: str
    objectives: ObjectiveSummary

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("deployment candidate_id must be non-empty")
        if (
            not isinstance(self.content_hash, str)
            or len(self.content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.content_hash)
        ):
            raise ValueError("deployment content_hash must be lowercase SHA-256")
        if not isinstance(self.objectives, ObjectiveSummary):
            raise ValueError("deployment objectives must be an ObjectiveSummary")


def _values(
    candidate: DeploymentCandidate, *, pessimistic: bool
) -> tuple[float, float, float, float]:
    summary = candidate.objectives
    if pessimistic:
        return (
            summary.id_accuracy.lcb,
            summary.worst_target_transfer.lcb,
            summary.token_cost.ucb,
            summary.paired_regression.ucb,
        )
    return (
        summary.id_accuracy.point,
        summary.worst_target_transfer.point,
        summary.token_cost.point,
        summary.paired_regression.point,
    )


def _tie_key(candidate: DeploymentCandidate, *, pessimistic: bool) -> tuple[float, float, str]:
    _, _, tokens, regression = _values(candidate, pessimistic=pessimistic)
    return (regression, tokens, candidate.content_hash)


def select_min_tokens(
    candidates: tuple[DeploymentCandidate, ...],
    *,
    accuracy_floor: float,
    pessimistic: bool = True,
) -> DeploymentCandidate:
    if (
        isinstance(accuracy_floor, bool)
        or not isinstance(accuracy_floor, (int, float))
        or not math.isfinite(float(accuracy_floor))
        or not 0.0 <= float(accuracy_floor) <= 1.0
    ):
        raise ValueError("accuracy_floor must be finite and in [0, 1]")
    eligible = [
        candidate
        for candidate in candidates
        if _values(candidate, pessimistic=pessimistic)[0] >= accuracy_floor
    ]
    if not eligible:
        raise ValueError("no deployment candidate satisfies the accuracy floor")
    return min(
        eligible,
        key=lambda candidate: (
            _values(candidate, pessimistic=pessimistic)[2],
            *_tie_key(candidate, pessimistic=pessimistic),
        ),
    )


def select_max_transfer(
    candidates: tuple[DeploymentCandidate, ...],
    *,
    token_budget: float,
    pessimistic: bool = True,
) -> DeploymentCandidate:
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, (int, float))
        or not math.isfinite(float(token_budget))
        or float(token_budget) < 0.0
    ):
        raise ValueError("token_budget must be finite and non-negative")
    eligible = [
        candidate
        for candidate in candidates
        if _values(candidate, pessimistic=pessimistic)[2] <= token_budget
    ]
    if not eligible:
        raise ValueError("no deployment candidate satisfies the token budget")
    return min(
        eligible,
        key=lambda candidate: (
            -_values(candidate, pessimistic=pessimistic)[1],
            *_tie_key(candidate, pessimistic=pessimistic),
        ),
    )


def select_knee(
    candidates: tuple[DeploymentCandidate, ...],
    *,
    pessimistic: bool = True,
    frozen_ranges: tuple[tuple[float, float], ...] | None = None,
) -> DeploymentCandidate:
    if not candidates:
        raise ValueError("knee selection requires at least one candidate")
    raw = {
        candidate.candidate_id: (
            _values(candidate, pessimistic=pessimistic)[0],
            _values(candidate, pessimistic=pessimistic)[1],
            -_values(candidate, pessimistic=pessimistic)[2],
            -_values(candidate, pessimistic=pessimistic)[3],
        )
        for candidate in candidates
    }
    if frozen_ranges is None:
        raise ValueError(
            "knee selection requires validation-frozen maximize-space ranges"
        )
    if len(frozen_ranges) != 4:
        raise ValueError("knee selection requires four frozen ranges")
    for minimum, maximum in frozen_ranges:
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("knee normalization ranges must be finite")
        if maximum < minimum:
            raise ValueError("knee normalization maximum cannot be below minimum")

    def distance(candidate: DeploymentCandidate) -> float:
        vector = raw[candidate.candidate_id]
        normalized = [
            1.0 if maximum <= minimum else (value - minimum) / (maximum - minimum)
            for value, (minimum, maximum) in zip(vector, frozen_ranges, strict=True)
        ]
        return math.dist(normalized, [1.0] * 4)

    return min(
        candidates,
        key=lambda candidate: (
            distance(candidate),
            *_tie_key(candidate, pessimistic=pessimistic),
        ),
    )


PolicyName = Literal[
    "min_tokens_subject_to_accuracy_floor",
    "max_worst_transfer_under_token_budget",
    "normalized_knee_point",
]


def select_deployment_candidate(
    policy: PolicyName,
    candidates: tuple[DeploymentCandidate, ...],
    *,
    accuracy_floor: float,
    token_budget: float,
    pessimistic: bool = True,
    frozen_ranges: tuple[tuple[float, float], ...] | None = None,
) -> DeploymentCandidate:
    if policy == "min_tokens_subject_to_accuracy_floor":
        return select_min_tokens(
            candidates, accuracy_floor=accuracy_floor, pessimistic=pessimistic
        )
    if policy == "max_worst_transfer_under_token_budget":
        return select_max_transfer(
            candidates, token_budget=token_budget, pessimistic=pessimistic
        )
    if policy == "normalized_knee_point":
        return select_knee(
            candidates,
            pessimistic=pessimistic,
            frozen_ranges=frozen_ranges,
        )
    raise ValueError(f"unknown deployment policy: {policy}")
