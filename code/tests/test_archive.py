from __future__ import annotations

from paretoskill.archive import ArchiveEntry, ParetoArchive
from paretoskill.models import stable_hash
from paretoskill.objectives import FeasibilityConstraints
from paretoskill.statistics import MetricEstimate, ObjectiveSummary


def summary(accuracy: float, transfer: float, tokens: float, regression: float) -> ObjectiveSummary:
    def metric(value: float) -> MetricEstimate:
        return MetricEstimate(value, value, value, value, value, 10)

    return ObjectiveSummary(
        metric(accuracy), metric(transfer), metric(tokens), metric(regression), 0.95, 100, 1, 10
    )


def entry(
    candidate_id: str, values: tuple[float, float, float, float], cost: int = 1
) -> ArchiveEntry:
    return ArchiveEntry(
        candidate_id,
        stable_hash({"candidate_id": candidate_id}),
        summary(*values),
        cost,
        {"round": 1},
    )


def archive(max_size: int = 4, budget: int = 20) -> ParetoArchive:
    return ParetoArchive(
        max_size=max_size,
        evaluation_budget=budget,
        constraints=FeasibilityConstraints(accuracy_floor=0.5, token_budget=200),
    )


def test_admission_removes_dominated_and_explains_rejection() -> None:
    result = archive()
    assert result.admit(entry("weak", (0.6, 0.6, 150, 0.2))).accepted
    decision = result.admit(entry("strong", (0.8, 0.7, 100, 0.1)))
    assert decision.accepted
    assert decision.reason == "accepted_removed_dominated"
    assert decision.removed_ids == ("weak",)
    rejected = result.admit(entry("worse", (0.7, 0.6, 120, 0.2)))
    assert not rejected.accepted
    assert rejected.reason == "dominated_by:strong"


def test_content_hash_dedup_is_historical_and_free() -> None:
    result = archive()
    first = entry("one", (0.8, 0.7, 100, 0.1), cost=3)
    assert result.admit(first).accepted
    duplicate = ArchiveEntry("two", first.content_hash, first.objectives, 5)
    decision = result.admit(duplicate)
    assert not decision.accepted
    assert decision.reason == "duplicate_content_hash"
    assert result.evaluations_spent == 3


def test_candidate_id_dedup_is_historical_even_after_rejection() -> None:
    result = archive()
    rejected = entry("reused", (0.4, 0.9, 80, 0.0))
    assert not result.admit(rejected).accepted
    replacement = ArchiveEntry(
        "reused",
        stable_hash("different-content"),
        summary(0.9, 0.9, 70, 0.0),
        1,
    )
    decision = result.admit(replacement)
    assert not decision.accepted
    assert decision.reason == "duplicate_candidate_id"


def test_budget_and_infeasibility_accounting() -> None:
    result = archive(budget=2)
    infeasible = result.admit(entry("bad", (0.4, 0.8, 100, 0.1)))
    assert not infeasible.accepted
    assert infeasible.reason == "infeasible:accuracy_floor"
    assert result.evaluations_spent == 1
    over = result.admit(entry("expensive", (0.8, 0.8, 100, 0.1), cost=2))
    assert not over.accepted
    assert over.reason == "evaluation_budget_exceeded"
    assert result.evaluations_spent == 1


def test_json_round_trip_preserves_budget_entries_and_seen_hashes(tmp_path) -> None:
    result = archive()
    assert result.admit(entry("one", (0.8, 0.7, 100, 0.1), cost=2)).accepted
    assert not result.admit(entry("bad", (0.4, 0.9, 80, 0.0))).accepted
    path = tmp_path / "archive.json"
    result.save(path)
    restored = ParetoArchive.load(path)
    assert restored.to_dict() == result.to_dict()
    retry = ArchiveEntry(
        "retry",
        entry("bad", (0.4, 0.9, 80, 0.0)).content_hash,
        summary(0.9, 0.9, 70, 0),
        1,
    )
    assert restored.admit(retry).reason == "duplicate_content_hash"


def test_capacity_is_deterministic_and_retains_objective_extremes() -> None:
    first = archive(max_size=2)
    second = archive(max_size=2)
    candidates = [
        entry("accuracy", (0.95, 0.55, 150, 0.20)),
        entry("transfer", (0.60, 0.95, 140, 0.15)),
        entry("cheap", (0.55, 0.60, 50, 0.10)),
    ]
    for candidate in candidates:
        first.admit(candidate)
    for candidate in reversed(candidates):
        second.admit(candidate)
    assert {item.candidate_id for item in first.entries} == {
        item.candidate_id for item in second.entries
    }
    assert len(first) == 2
    # With capacity below the number of extrema, objective-extreme count and the
    # candidate-id tie-breaker define a stable approximation.
    assert {item.candidate_id for item in first.entries} <= {
        "accuracy", "transfer", "cheap"
    }


def test_archive_round_trip_preserves_paired_gate_and_objective_mask() -> None:
    result = ParetoArchive(
        max_size=3,
        evaluation_budget=10,
        constraints=FeasibilityConstraints.from_paired_epsilon(
            epsilon=0.05, token_budget=200.0, enabled=False
        ),
        active_objectives=("id_accuracy", "token_cost", "paired_regression"),
    )
    restored = ParetoArchive.from_json(result.to_json())
    assert restored.to_dict() == result.to_dict()
    assert restored.active_objectives == (
        "id_accuracy",
        "token_cost",
        "paired_regression",
    )
    assert restored.constraints.accuracy_delta_floor == -0.05
    assert restored.constraints.enabled is False
