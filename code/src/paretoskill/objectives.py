"""Dominance and deployment feasibility rules for ParetoSkill objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .statistics import ObjectiveSummary


DominanceMode = Literal["uncertainty", "point"]
ObjectiveName = Literal[
    "id_accuracy",
    "worst_target_transfer",
    "token_cost",
    "paired_regression",
]
DEFAULT_ACTIVE_OBJECTIVES: tuple[ObjectiveName, ...] = (
    "id_accuracy",
    "worst_target_transfer",
    "token_cost",
    "paired_regression",
)


@dataclass(frozen=True, slots=True)
class FeasibilityConstraints:
    """Pre-declared deployment constraints.

    The legacy absolute ``accuracy_floor`` remains supported. For the primary
    paired design, ``accuracy_delta_floor`` checks the candidate-minus-base ID
    accuracy delta directly. ``enabled=False`` is the frozen no-feasibility
    ablation. Uncertainty-aware feasibility uses lower accuracy bounds and the
    token upper bound.
    """

    accuracy_floor: float
    token_budget: float
    accuracy_delta_floor: float | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.accuracy_floor, bool) or not isinstance(
            self.accuracy_floor, (int, float)
        ):
            raise ValueError("accuracy_floor must be numeric")
        if not 0.0 <= self.accuracy_floor <= 1.0:
            raise ValueError("accuracy_floor must be in [0, 1]")
        if isinstance(self.token_budget, bool) or not isinstance(
            self.token_budget, (int, float)
        ):
            raise ValueError("token_budget must be numeric")
        if not math.isfinite(self.token_budget) or self.token_budget < 0.0:
            raise ValueError("token_budget must be finite and non-negative")
        if self.accuracy_delta_floor is not None and (
            isinstance(self.accuracy_delta_floor, bool)
            or not isinstance(self.accuracy_delta_floor, (int, float))
            or not math.isfinite(self.accuracy_delta_floor)
            or not -1.0 <= self.accuracy_delta_floor <= 1.0
        ):
            raise ValueError("accuracy_delta_floor must be in [-1, 1]")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

    @classmethod
    def from_base(
        cls, *, base_id_accuracy: float, epsilon: float, token_budget: float
    ) -> "FeasibilityConstraints":
        if (
            isinstance(base_id_accuracy, bool)
            or not isinstance(base_id_accuracy, (int, float))
            or not 0.0 <= base_id_accuracy <= 1.0
        ):
            raise ValueError("base_id_accuracy must be in [0, 1]")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not 0.0 <= epsilon <= 1.0
        ):
            raise ValueError("epsilon must be in [0, 1]")
        return cls(max(0.0, base_id_accuracy - epsilon), token_budget)

    @classmethod
    def from_paired_epsilon(
        cls, *, epsilon: float, token_budget: float, enabled: bool = True
    ) -> "FeasibilityConstraints":
        """Require paired ID-accuracy delta LCB to be at least ``-epsilon``."""

        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not 0.0 <= epsilon <= 1.0
        ):
            raise ValueError("epsilon must be in [0, 1]")
        return cls(
            accuracy_floor=0.0,
            token_budget=token_budget,
            accuracy_delta_floor=-epsilon,
            enabled=enabled,
        )


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    feasible: bool
    accuracy_value: float
    token_value: float
    reasons: tuple[str, ...]
    accuracy_delta_value: float | None = None


def feasibility(
    objectives: ObjectiveSummary,
    constraints: FeasibilityConstraints,
    *,
    mode: DominanceMode = "uncertainty",
    tolerance: float = 1e-12,
) -> FeasibilityResult:
    """Check conservative (or ablation point-estimate) feasibility."""

    if mode == "uncertainty":
        accuracy = objectives.id_accuracy.lcb
        tokens = objectives.token_cost.ucb
    elif mode == "point":
        accuracy = objectives.id_accuracy.point
        tokens = objectives.token_cost.point
    else:
        raise ValueError(f"unknown dominance mode: {mode!r}")
    delta_metric = objectives.id_accuracy_delta
    delta = None
    if delta_metric is not None:
        delta = delta_metric.lcb if mode == "uncertainty" else delta_metric.point
    if not constraints.enabled:
        return FeasibilityResult(True, accuracy, tokens, (), delta)

    reasons: list[str] = []
    if accuracy + tolerance < constraints.accuracy_floor:
        reasons.append("accuracy_floor")
    if constraints.accuracy_delta_floor is not None:
        if delta is None:
            reasons.append("accuracy_delta_unavailable")
        elif delta + tolerance < constraints.accuracy_delta_floor:
            reasons.append("accuracy_delta_floor")
    if tokens - tolerance > constraints.token_budget:
        reasons.append("token_budget")
    return FeasibilityResult(not reasons, accuracy, tokens, tuple(reasons), delta)


def normalize_active_objectives(
    active_objectives: Iterable[ObjectiveName] | None,
) -> tuple[ObjectiveName, ...]:
    """Validate an objective mask and return canonical objective order."""

    if active_objectives is None:
        return DEFAULT_ACTIVE_OBJECTIVES
    requested = tuple(active_objectives)
    if not requested:
        raise ValueError("at least one objective must be active")
    if len(set(requested)) != len(requested):
        raise ValueError("active_objectives cannot contain duplicates")
    unknown = set(requested) - set(DEFAULT_ACTIVE_OBJECTIVES)
    if unknown:
        raise ValueError(f"unknown active objectives: {sorted(unknown)!r}")
    return tuple(name for name in DEFAULT_ACTIVE_OBJECTIVES if name in requested)


def dominance_vector(
    objectives: ObjectiveSummary,
    *,
    mode: DominanceMode = "uncertainty",
    active_objectives: Iterable[ObjectiveName] | None = None,
) -> tuple[float, ...]:
    names = normalize_active_objectives(active_objectives)
    if mode == "uncertainty":
        vector = objectives.pessimistic_vector()
    elif mode == "point":
        vector = objectives.point_vector()
    else:
        raise ValueError(f"unknown dominance mode: {mode!r}")
    values = dict(zip(DEFAULT_ACTIVE_OBJECTIVES, vector, strict=True))
    return tuple(values[name] for name in names)


def dominates(
    left: ObjectiveSummary,
    right: ObjectiveSummary,
    *,
    mode: DominanceMode = "uncertainty",
    tolerance: float = 1e-12,
    active_objectives: Iterable[ObjectiveName] | None = None,
) -> bool:
    """Return whether ``left`` Pareto-dominates ``right`` in maximize space."""

    names = normalize_active_objectives(active_objectives)
    left_vector = dominance_vector(left, mode=mode, active_objectives=names)
    right_vector = dominance_vector(right, mode=mode, active_objectives=names)
    weakly_better = all(
        lhs + tolerance >= rhs
        for lhs, rhs in zip(left_vector, right_vector, strict=True)
    )
    strictly_better = any(
        lhs > rhs + tolerance
        for lhs, rhs in zip(left_vector, right_vector, strict=True)
    )
    return weakly_better and strictly_better
