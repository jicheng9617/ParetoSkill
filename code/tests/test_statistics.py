from __future__ import annotations

from dataclasses import replace

import pytest

from paretoskill.objectives import (
    FeasibilityConstraints,
    dominates,
    feasibility,
)
from paretoskill.statistics import (
    MetricEstimate,
    ObjectiveSummary,
    PairedObservation,
    aggregate_objectives,
    paired_bootstrap,
)


def observations() -> list[PairedObservation]:
    return [
        PairedObservation("a", 0, "id", "source", True, True, 10, 2),
        PairedObservation("b", 0, "id", "source", False, True, 8, 2),
        PairedObservation("c", 0, "id", "source", True, False, 9, 2),
        PairedObservation("a", 0, "transfer", "small", True, True, 11, 2, "model-small"),
        PairedObservation("b", 0, "transfer", "small", False, True, 9, 2, "model-small"),
        PairedObservation("a", 1, "transfer", "large", True, True, 12, 2, "model-large"),
        PairedObservation("b", 1, "transfer", "large", True, False, 10, 2, "model-large"),
    ]


def summary(values: tuple[float, float, float, float], width: float = 0.0) -> ObjectiveSummary:
    accuracy, transfer, tokens, regression = values

    def rate(value: float) -> MetricEstimate:
        low = max(0.0, value - width)
        high = min(1.0, value + width)
        return MetricEstimate(value, low, high, low, high, 10)

    def cost(value: float) -> MetricEstimate:
        return MetricEstimate(
            value, value - width, value + width, value - width, value + width, 10
        )

    return ObjectiveSummary(
        rate(accuracy), rate(transfer), cost(tokens), rate(regression), 0.95, 100, 0, 10
    )


def test_aggregate_worst_group_and_conditional_regression() -> None:
    result = aggregate_objectives(observations())
    assert result.id_accuracy == pytest.approx(2 / 3)
    assert result.worst_target_transfer == pytest.approx(0.5)
    # Three of five base-correct outcomes remain correct.
    assert result.paired_regression == pytest.approx(2 / 5)
    assert result.token_cost == pytest.approx(83 / 7)


def test_bootstrap_is_deterministic_and_bounds_are_oriented() -> None:
    first = paired_bootstrap(observations(), replicates=400, seed=71)
    second = paired_bootstrap(observations(), replicates=400, seed=71)
    assert first == second
    assert first.id_accuracy.lcb <= first.id_accuracy.point
    assert first.token_cost.ucb >= first.token_cost.point
    assert first.paired_regression.ucb >= first.paired_regression.point
    assert first.block_count == 5
    assert first.id_accuracy.sample_size == 3
    assert first.worst_target_transfer.sample_size == 2
    assert first.token_cost.sample_size == 5
    assert first.paired_regression.sample_size == 3
    assert first.id_accuracy_delta is not None
    assert first.id_accuracy_delta.point == pytest.approx(0.0)
    assert first.id_accuracy_delta.sample_size == 3
    assert ObjectiveSummary.from_dict(first.to_dict()) == first
    assert first.pessimistic_vector()[2] == -first.token_cost.ucb


def test_no_base_correct_rows_gets_conservative_regression_ucb() -> None:
    rows = [
        PairedObservation("a", 0, "id", "src", True, False),
        PairedObservation("a", 0, "transfer", "x", True, False, group="g"),
    ]
    result = paired_bootstrap(rows, replicates=20, token_cost_upper_bound=0.0)
    assert result.paired_regression.point == 1.0
    assert result.paired_regression.sample_size == 0
    assert result.paired_regression.ucb == 1.0
    assert result.paired_regression.to_dict()["defined"] is False


def test_point_and_uncertainty_dominance_can_differ() -> None:
    point_better_but_uncertain = summary((0.9, 0.8, 90.0, 0.05), width=0.1)
    stable = summary((0.85, 0.75, 95.0, 0.1), width=0.0)
    assert dominates(point_better_but_uncertain, stable, mode="point")
    assert not dominates(point_better_but_uncertain, stable, mode="uncertainty")


def test_feasibility_uses_accuracy_lcb_and_token_ucb() -> None:
    candidate = summary((0.9, 0.8, 95.0, 0.1), width=0.1)
    constraints = FeasibilityConstraints(accuracy_floor=0.85, token_budget=95.05)
    pessimistic = feasibility(candidate, constraints, mode="uncertainty")
    assert not pessimistic.feasible
    assert pessimistic.reasons == ("accuracy_floor", "token_budget")
    assert feasibility(candidate, constraints, mode="point").feasible


def test_rejects_duplicate_observation_keys() -> None:
    row = PairedObservation("a", 0, "id", "source", True, True)
    transfer = PairedObservation("a", 0, "transfer", "other", True, True, group="g")
    with pytest.raises(ValueError, match="duplicate observation"):
        aggregate_objectives([row, row, transfer])


def test_requires_the_complete_declared_transfer_group_set() -> None:
    with pytest.raises(ValueError, match=r"missing=\['model-missing'\]"):
        paired_bootstrap(
            observations(),
            expected_transfer_groups={
                "model-small",
                "model-large",
                "model-missing",
            },
            replicates=20,
        )


def test_too_few_effective_blocks_get_conservative_bounded_metric_bounds() -> None:
    rows = [
        PairedObservation("only", 0, "id", "src", True, True),
        PairedObservation(
            "only", 0, "transfer", "target", True, True, group="group"
        ),
    ]
    result = paired_bootstrap(
        rows,
        replicates=20,
        min_effective_blocks=2,
        token_cost_upper_bound=10.0,
    )
    assert result.id_accuracy.lcb == 0.0
    assert result.worst_target_transfer.lcb == 0.0
    assert result.paired_regression.ucb == 1.0
    assert result.token_cost.ucb == 10.0
    assert result.id_accuracy_delta is not None
    assert result.id_accuracy_delta.lcb == -1.0
    assert result.id_accuracy_delta.ucb == 1.0


def test_paired_delta_feasibility_and_legacy_absolute_floor_coexist() -> None:
    candidate = replace(
        summary((0.9, 0.8, 90.0, 0.05)),
        id_accuracy_delta=MetricEstimate(0.0, -0.1, 0.1, -0.1, 0.1, 10),
    )
    paired = FeasibilityConstraints.from_paired_epsilon(
        epsilon=0.05, token_budget=100.0
    )
    result = feasibility(candidate, paired, mode="uncertainty")
    assert not result.feasible
    assert result.reasons == ("accuracy_delta_floor",)
    assert result.accuracy_delta_value == -0.1
    assert feasibility(candidate, paired, mode="point").feasible

    legacy = FeasibilityConstraints(accuracy_floor=0.85, token_budget=100.0)
    assert feasibility(candidate, legacy).feasible


def test_feasibility_gate_and_active_objective_mask_are_executable() -> None:
    infeasible = summary((0.1, 0.1, 500.0, 0.9))
    disabled = FeasibilityConstraints(
        accuracy_floor=0.9, token_budget=1.0, enabled=False
    )
    assert feasibility(infeasible, disabled).feasible

    left = summary((0.8, 0.5, 100.0, 0.1))
    right = summary((0.8, 0.6, 100.0, 0.2))
    assert not dominates(left, right)
    assert dominates(
        left,
        right,
        active_objectives=("id_accuracy", "token_cost", "paired_regression"),
    )
    assert dominates(
        right,
        left,
        active_objectives=("id_accuracy", "worst_target_transfer", "token_cost"),
    )
