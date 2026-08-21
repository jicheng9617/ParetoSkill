"""Paired, block-level statistics for ParetoSkill evaluations.

The bootstrap resamples stratified ``(task_id, seed)`` blocks rather than
individual target observations.  Consequently every candidate/base pair and
every target measured for a task execution stays together in a replicate.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One candidate/base outcome belonging to a paired task-seed block.

    ``split`` is either ``"id"`` or ``"transfer"``.  Transfer observations
    must name a non-empty ``group``; ID observations may leave it empty.  Token
    cost is the total input plus output tokens for the candidate execution.
    """

    task_id: str
    seed: int
    split: str
    target: str
    candidate_correct: bool
    base_correct: bool
    input_tokens: int = 0
    output_tokens: int = 0
    group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be non-empty")
        _strict_int(self.seed, "seed")
        if self.split not in {"id", "transfer"}:
            raise ValueError("split must be 'id' or 'transfer'")
        if self.split == "transfer" and (
            not isinstance(self.group, str) or not self.group
        ):
            raise ValueError("transfer observations require a group")
        if not isinstance(self.candidate_correct, bool) or not isinstance(
            self.base_correct, bool
        ):
            raise ValueError("paired outcomes must be booleans")
        _strict_int(self.input_tokens, "input_tokens")
        _strict_int(self.output_tokens, "output_tokens")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def block_key(self) -> tuple[str, int]:
        return (self.task_id, self.seed)

    @property
    def observation_key(self) -> tuple[str, int, str, str, str | None]:
        return (self.task_id, self.seed, self.split, self.target, self.group)

    @property
    def token_cost(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class PointObjectives:
    """The four raw objectives, with costs represented as positive quantities."""

    id_accuracy: float
    worst_target_transfer: float
    token_cost: float
    paired_regression: float

    def __post_init__(self) -> None:
        for name in ("id_accuracy", "worst_target_transfer", "paired_regression"):
            value = _strict_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        token_cost = _strict_float(self.token_cost, "token_cost")
        if token_cost < 0.0:
            raise ValueError("token_cost must be non-negative")

    def maximize_vector(self) -> tuple[float, float, float, float]:
        """Return the common maximize-oriented representation."""

        return (
            self.id_accuracy,
            self.worst_target_transfer,
            -self.token_cost,
            -self.paired_regression,
        )


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Point estimate, percentile CI, and one-sided confidence bounds."""

    point: float
    ci_low: float
    ci_high: float
    lcb: float
    ucb: float
    sample_size: int

    def __post_init__(self) -> None:
        values = tuple(
            _strict_float(getattr(self, name), name)
            for name in ("point", "ci_low", "ci_high", "lcb", "ucb")
        )
        _strict_int(self.sample_size, "sample_size")
        if self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if self.ci_low > self.ci_high or self.lcb > self.ucb:
            raise ValueError("invalid confidence-bound ordering")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "lcb": self.lcb,
            "ucb": self.ucb,
            "sample_size": self.sample_size,
            "defined": self.sample_size > 0,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MetricEstimate":
        return cls(
            point=_strict_float(data["point"], "point"),
            ci_low=_strict_float(data["ci_low"], "ci_low"),
            ci_high=_strict_float(data["ci_high"], "ci_high"),
            lcb=_strict_float(data["lcb"], "lcb"),
            ucb=_strict_float(data["ucb"], "ucb"),
            sample_size=_strict_int(data["sample_size"], "sample_size"),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveSummary:
    """Confidence-aware estimates of the ParetoSkill objective vector."""

    id_accuracy: MetricEstimate
    worst_target_transfer: MetricEstimate
    token_cost: MetricEstimate
    paired_regression: MetricEstimate
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    block_count: int
    id_accuracy_delta: MetricEstimate | None = None

    def __post_init__(self) -> None:
        for name in (
            "id_accuracy",
            "worst_target_transfer",
            "token_cost",
            "paired_regression",
        ):
            if not isinstance(getattr(self, name), MetricEstimate):
                raise ValueError(f"{name} must be a MetricEstimate")
        if self.id_accuracy_delta is not None and not isinstance(
            self.id_accuracy_delta, MetricEstimate
        ):
            raise ValueError("id_accuracy_delta must be a MetricEstimate or null")
        confidence = _strict_float(self.confidence_level, "confidence_level")
        _strict_int(self.bootstrap_replicates, "bootstrap_replicates")
        _strict_int(self.bootstrap_seed, "bootstrap_seed")
        _strict_int(self.block_count, "block_count")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.bootstrap_replicates < 0 or self.block_count < 1:
            raise ValueError("invalid bootstrap metadata")
        for name, metric in (
            ("id_accuracy", self.id_accuracy),
            ("worst_target_transfer", self.worst_target_transfer),
            ("paired_regression", self.paired_regression),
        ):
            values = (metric.point, metric.ci_low, metric.ci_high, metric.lcb, metric.ucb)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"{name} estimates must be in [0, 1]")
        if self.id_accuracy_delta is not None:
            delta_values = (
                self.id_accuracy_delta.point,
                self.id_accuracy_delta.ci_low,
                self.id_accuracy_delta.ci_high,
                self.id_accuracy_delta.lcb,
                self.id_accuracy_delta.ucb,
            )
            if not all(-1.0 <= value <= 1.0 for value in delta_values):
                raise ValueError("id_accuracy_delta estimates must be in [-1, 1]")
        token_values = (
            self.token_cost.point,
            self.token_cost.ci_low,
            self.token_cost.ci_high,
            self.token_cost.lcb,
            self.token_cost.ucb,
        )
        if not all(value >= 0.0 for value in token_values):
            raise ValueError("token_cost estimates must be non-negative")

    def point_objectives(self) -> PointObjectives:
        return PointObjectives(
            id_accuracy=self.id_accuracy.point,
            worst_target_transfer=self.worst_target_transfer.point,
            token_cost=self.token_cost.point,
            paired_regression=self.paired_regression.point,
        )

    def point_vector(self) -> tuple[float, float, float, float]:
        return self.point_objectives().maximize_vector()

    def pessimistic_vector(self) -> tuple[float, float, float, float]:
        """LCB for benefits and negated UCB for costs/regressions."""

        return (
            self.id_accuracy.lcb,
            self.worst_target_transfer.lcb,
            -self.token_cost.ucb,
            -self.paired_regression.ucb,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id_accuracy": self.id_accuracy.to_dict(),
            "worst_target_transfer": self.worst_target_transfer.to_dict(),
            "token_cost": self.token_cost.to_dict(),
            "paired_regression": self.paired_regression.to_dict(),
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "block_count": self.block_count,
            "id_accuracy_delta": (
                self.id_accuracy_delta.to_dict()
                if self.id_accuracy_delta is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ObjectiveSummary":
        def metric(name: str) -> MetricEstimate:
            value = data[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            return MetricEstimate.from_dict(value)

        raw_delta = data.get("id_accuracy_delta")
        if raw_delta is not None and not isinstance(raw_delta, Mapping):
            raise ValueError("id_accuracy_delta must be an object or null")
        return cls(
            id_accuracy=metric("id_accuracy"),
            worst_target_transfer=metric("worst_target_transfer"),
            token_cost=metric("token_cost"),
            paired_regression=metric("paired_regression"),
            confidence_level=_strict_float(data["confidence_level"], "confidence_level"),
            bootstrap_replicates=_strict_int(
                data["bootstrap_replicates"], "bootstrap_replicates"
            ),
            bootstrap_seed=_strict_int(data["bootstrap_seed"], "bootstrap_seed"),
            block_count=_strict_int(data["block_count"], "block_count"),
            id_accuracy_delta=(
                MetricEstimate.from_dict(raw_delta)
                if isinstance(raw_delta, Mapping)
                else None
            ),
        )


def validate_observations(
    observations: Iterable[PairedObservation],
    *,
    expected_transfer_groups: Iterable[str] | None = None,
) -> tuple[PairedObservation, ...]:
    """Materialize observations and reject ambiguous duplicate measurements."""

    result = tuple(observations)
    if not result:
        raise ValueError("at least one paired observation is required")
    seen: set[tuple[str, int, str, str, str | None]] = set()
    for observation in result:
        if observation.observation_key in seen:
            raise ValueError(f"duplicate observation: {observation.observation_key!r}")
        seen.add(observation.observation_key)
    if not any(row.split == "id" for row in result):
        raise ValueError("at least one ID observation is required")
    if not any(row.split == "transfer" for row in result):
        raise ValueError("at least one transfer observation is required")
    if expected_transfer_groups is not None:
        expected = {str(group) for group in expected_transfer_groups}
        if not expected or any(not group for group in expected):
            raise ValueError("expected_transfer_groups must contain non-empty names")
        observed = {
            row.group
            for row in result
            if row.split == "transfer" and row.group is not None
        }
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "transfer group mismatch: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
    return result


def aggregate_objectives(
    observations: Iterable[PairedObservation],
    *,
    expected_transfer_groups: Iterable[str] | None = None,
) -> PointObjectives:
    """Aggregate ID accuracy, worst-group transfer, tokens, and regression.

    Regression is conditioned on base-correct observations. When no base-correct
    observation exists, the metric is undefined. Its numeric sentinel is one
    (worst case), and the bootstrap summary marks ``sample_size=0``/``defined=false``.
    """

    rows = validate_observations(
        observations, expected_transfer_groups=expected_transfer_groups
    )
    id_rows = [row for row in rows if row.split == "id"]
    transfer_rows = [row for row in rows if row.split == "transfer"]
    id_accuracy = sum(row.candidate_correct for row in id_rows) / len(id_rows)

    groups: dict[str, list[PairedObservation]] = defaultdict(list)
    for row in transfer_rows:
        assert row.group is not None  # established by PairedObservation validation
        groups[row.group].append(row)
    transfer_rates = [
        sum(row.candidate_correct for row in group_rows) / len(group_rows)
        for group_rows in groups.values()
    ]
    worst_transfer = min(transfer_rates)
    token_cost = sum(row.token_cost for row in rows) / len(rows)

    eligible = [row for row in rows if row.base_correct]
    regression = (
        sum(not row.candidate_correct for row in eligible) / len(eligible)
        if eligible
        else 1.0
    )
    return PointObjectives(id_accuracy, worst_transfer, token_cost, regression)


def _quantile(values: Sequence[float], probability: float) -> float:
    """Deterministic, linearly interpolated empirical quantile (stdlib only)."""

    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _estimate(
    point: float,
    samples: Sequence[float],
    confidence_level: float,
    sample_size: int,
) -> MetricEstimate:
    alpha = 1.0 - confidence_level
    return MetricEstimate(
        point=point,
        ci_low=_quantile(samples, alpha / 2.0),
        ci_high=_quantile(samples, 1.0 - alpha / 2.0),
        lcb=_quantile(samples, alpha),
        ucb=_quantile(samples, confidence_level),
        sample_size=sample_size,
    )


def paired_bootstrap(
    observations: Iterable[PairedObservation],
    *,
    confidence_level: float = 0.95,
    replicates: int = 2_000,
    seed: int = 0,
    expected_transfer_groups: Iterable[str] | None = None,
    min_effective_blocks: int = 2,
    token_cost_upper_bound: float | None = None,
) -> ObjectiveSummary:
    """Compute deterministic paired block-bootstrap objective estimates.

    The implementation is intentionally serial and deterministic.  A replicate
    samples unique task-seed blocks with replacement, then retains all rows in
    each selected block. Replicates in which a metric is structurally absent are
    skipped only for that metric. Bounded metrics with fewer than
    ``min_effective_blocks`` receive their full-support conservative interval.
    Token cost requires an explicit finite upper bound in that case. If regression
    has no eligible base-correct block, its interval is set to ``[0, 1]``.
    """

    rows = validate_observations(
        observations, expected_transfer_groups=expected_transfer_groups
    )
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be between zero and one")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    if (
        isinstance(min_effective_blocks, bool)
        or not isinstance(min_effective_blocks, int)
        or min_effective_blocks < 1
    ):
        raise ValueError("min_effective_blocks must be positive")
    if token_cost_upper_bound is not None and (
        isinstance(token_cost_upper_bound, bool)
        or not isinstance(token_cost_upper_bound, (int, float))
        or not math.isfinite(token_cost_upper_bound)
        or token_cost_upper_bound < 0.0
    ):
        raise ValueError("token_cost_upper_bound must be finite and non-negative")

    grouped: dict[tuple[str, int], list[PairedObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.block_key].append(row)
    block_keys = sorted(grouped)
    # Preserve the count of each observed target-composition stratum.  In the
    # common design every task-seed block contains all targets and there is one
    # stratum.  When benchmark groups use disjoint task sets, this prevents a
    # bootstrap replicate from accidentally omitting an entire target group.
    strata: dict[
        tuple[tuple[str, str, str], ...], list[tuple[str, int]]
    ] = defaultdict(list)
    for key in block_keys:
        signature = tuple(
            sorted((row.split, row.target, row.group or "") for row in grouped[key])
        )
        strata[signature].append(key)
    transfer_group_names = {
        row.group for row in rows if row.split == "transfer" and row.group is not None
    }
    point = aggregate_objectives(rows)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "id_accuracy": [],
        "id_accuracy_delta": [],
        "worst_target_transfer": [],
        "token_cost": [],
        "paired_regression": [],
    }

    # Sampling blocks can omit all rows for a split if tasks do not share split
    # labels.  Such a replicate is not evidence for the absent metric.
    for _ in range(replicates):
        sampled_rows: list[PairedObservation] = []
        for signature in sorted(strata):
            stratum_keys = strata[signature]
            for _block in stratum_keys:
                chosen = stratum_keys[rng.randrange(len(stratum_keys))]
                sampled_rows.extend(grouped[chosen])
        id_rows = [row for row in sampled_rows if row.split == "id"]
        transfer_rows = [row for row in sampled_rows if row.split == "transfer"]
        if id_rows:
            samples["id_accuracy"].append(
                sum(row.candidate_correct for row in id_rows) / len(id_rows)
            )
            samples["id_accuracy_delta"].append(
                sum(
                    int(row.candidate_correct) - int(row.base_correct)
                    for row in id_rows
                )
                / len(id_rows)
            )
        sampled_transfer_groups = {
            row.group
            for row in transfer_rows
            if row.group is not None
        }
        # Never let a replicate that happened to omit the weakest target group
        # turn the minimum into an optimistic estimate.
        if transfer_rows and sampled_transfer_groups == transfer_group_names:
            by_group: dict[str, list[PairedObservation]] = defaultdict(list)
            for row in transfer_rows:
                assert row.group is not None
                by_group[row.group].append(row)
            samples["worst_target_transfer"].append(
                min(
                    sum(row.candidate_correct for row in group_rows)
                    / len(group_rows)
                    for group_rows in by_group.values()
                )
            )
        samples["token_cost"].append(
            sum(row.token_cost for row in sampled_rows) / len(sampled_rows)
        )
        eligible = [row for row in sampled_rows if row.base_correct]
        if eligible:
            samples["paired_regression"].append(
                sum(not row.candidate_correct for row in eligible) / len(eligible)
            )

    # Extremely small/stratified datasets can produce no valid replicate for a
    # split.  The full estimate remains usable, but the bound becomes maximally
    # conservative for a benefit (LCB=0) or bounded rate cost (UCB=1).
    def benefit_estimate(name: str, value: float, size: int) -> MetricEstimate:
        metric_samples = samples[name]
        if size < min_effective_blocks or not metric_samples:
            return MetricEstimate(value, 0.0, 1.0, 0.0, 1.0, size)
        return _estimate(value, metric_samples, confidence_level, size)

    id_blocks = {row.block_key for row in rows if row.split == "id"}
    transfer_blocks_by_group: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        if row.split == "transfer":
            assert row.group is not None
            transfer_blocks_by_group[row.group].add(row.block_key)
    transfer_count = min(map(len, transfer_blocks_by_group.values()))
    regression_blocks = {row.block_key for row in rows if row.base_correct}
    id_count = len(id_blocks)
    regression_count = len(regression_blocks)
    regression_samples = samples["paired_regression"]
    if regression_count == 0 or not regression_samples:
        regression_estimate = MetricEstimate(1.0, 0.0, 1.0, 0.0, 1.0, 0)
    elif regression_count < min_effective_blocks:
        regression_estimate = MetricEstimate(
            point.paired_regression, 0.0, 1.0, 0.0, 1.0, regression_count
        )
    else:
        regression_estimate = _estimate(
            point.paired_regression,
            regression_samples,
            confidence_level,
            regression_count,
        )

    id_rows = [row for row in rows if row.split == "id"]
    id_delta_point = sum(
        int(row.candidate_correct) - int(row.base_correct) for row in id_rows
    ) / len(id_rows)
    if id_count < min_effective_blocks or not samples["id_accuracy_delta"]:
        id_delta_estimate = MetricEstimate(
            id_delta_point, -1.0, 1.0, -1.0, 1.0, id_count
        )
    else:
        id_delta_estimate = _estimate(
            id_delta_point,
            samples["id_accuracy_delta"],
            confidence_level,
            id_count,
        )

    if len(block_keys) < min_effective_blocks:
        if token_cost_upper_bound is None:
            raise ValueError(
                "token_cost_upper_bound is required when effective blocks are "
                "below min_effective_blocks"
            )
        if token_cost_upper_bound < point.token_cost:
            raise ValueError("token_cost_upper_bound cannot be below the point estimate")
        token_estimate = MetricEstimate(
            point.token_cost,
            0.0,
            token_cost_upper_bound,
            0.0,
            token_cost_upper_bound,
            len(block_keys),
        )
    else:
        token_estimate = _estimate(
            point.token_cost,
            samples["token_cost"],
            confidence_level,
            len(block_keys),
        )

    return ObjectiveSummary(
        id_accuracy=benefit_estimate("id_accuracy", point.id_accuracy, id_count),
        worst_target_transfer=benefit_estimate(
            "worst_target_transfer", point.worst_target_transfer, transfer_count
        ),
        token_cost=token_estimate,
        paired_regression=regression_estimate,
        confidence_level=confidence_level,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        block_count=len(block_keys),
        id_accuracy_delta=id_delta_estimate,
    )
