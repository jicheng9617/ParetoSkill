from __future__ import annotations

import math

import pytest

from paretoskill.baselines import MOCHAPlugin, NSGAIIPlugin, ScoredCandidate, builtin_plugins
from paretoskill.metrics import (
    additive_epsilon_indicator,
    coverage,
    crowding_distance,
    hypervolume,
    hypervolume_contributions,
    inverted_generational_distance,
)
from paretoskill.statistics import PointObjectives


def candidate(candidate_id: str, vector: tuple[float, float, float, float]) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id,
        (),
        PointObjectives(vector[0], vector[1], -vector[2], -vector[3]),
    )


def test_frontier_metrics_have_known_small_examples() -> None:
    points = {"a": (1.0, 0.5), "b": (0.5, 1.0)}
    assert math.isclose(hypervolume(points.values(), (0.0, 0.0)), 0.75)
    contributions = hypervolume_contributions(points, (0.0, 0.0))
    assert contributions == {"a": 0.25, "b": 0.25}
    assert coverage([(1.0, 1.0)], points.values()) == 1.0
    assert additive_epsilon_indicator(points.values(), [(1.0, 1.0)]) == 0.5
    assert inverted_generational_distance([(1.0, 1.0)], [(1.0, 1.0)]) == 0.0


def test_hypervolume_rejects_malformed_or_nonfinite_vectors() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        hypervolume([(1.0, 1.0), (1.0,)], (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        hypervolume([(1.0, math.nan)], (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        hypervolume([(1.0, 1.0)], (0.0, math.inf))


def test_crowding_ignores_constant_objectives() -> None:
    distances = crowding_distance(
        {
            "a": (0.5, 0.5, 0.0),
            "b": (0.0, 1.0, 0.0),
            "c": (1.0, 0.0, 0.0),
        }
    )
    assert math.isfinite(distances["a"])
    assert distances["b"] == math.inf
    assert distances["c"] == math.inf


def test_plugins_cover_frozen_manifest_and_multiobjective_selection() -> None:
    plugins = builtin_plugins()
    assert {
        "no_skill",
        "base_skill",
        "simple_patch_composition",
        "trace2skill_all",
        "trace2skill_accuracy_subset",
        "fixed_scalarization",
        "evoskill_scalar_topk",
        "skillmoo_nsga2",
        "mocha_chebyshev_hvc",
        "passive_archive",
        "paretoskill",
    } <= set(plugins)

    pool = (
        candidate("accuracy", (1.0, 0.2, -10.0, -0.2)),
        candidate("transfer", (0.7, 1.0, -12.0, -0.1)),
        candidate("dominated", (0.6, 0.1, -20.0, -0.4)),
    )
    selected = NSGAIIPlugin().select(pool, count=2)
    assert {item.candidate_id for item in selected} == {"accuracy", "transfer"}
    assert len(MOCHAPlugin().select(pool, count=2)) == 2


def test_seeded_patch_subset_generation_is_reproducible() -> None:
    plugin = builtin_plugins()["simple_patch_composition"]
    left = plugin.propose_subsets(("p3", "p1", "p2"), max_candidates=7, seed=123)
    right = plugin.propose_subsets(("p2", "p3", "p1"), max_candidates=7, seed=123)
    assert left == right
    assert () not in left
    assert ("p1", "p2", "p3") in left
    constrained = plugin.propose_subsets(("p1", "p2"), max_candidates=2, seed=123)
    assert constrained == (("p1", "p2"), ("p1",))
