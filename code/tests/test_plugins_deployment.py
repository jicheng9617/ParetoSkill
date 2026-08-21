from __future__ import annotations

from pathlib import Path

from paretoskill.ablations import AblationRegistry
from paretoskill.config import load_manifest
from paretoskill.deployment import (
    DeploymentCandidate,
    select_knee,
    select_max_transfer,
    select_min_tokens,
)
from paretoskill.statistics import MetricEstimate, ObjectiveSummary


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


def estimate(value: float) -> MetricEstimate:
    return MetricEstimate(value, value, value, value, value, 10)


def candidate(
    candidate_id: str, accuracy: float, transfer: float, tokens: float, regression: float
) -> DeploymentCandidate:
    summary = ObjectiveSummary(
        id_accuracy=estimate(accuracy),
        worst_target_transfer=estimate(transfer),
        token_cost=estimate(tokens),
        paired_regression=estimate(regression),
        confidence_level=0.95,
        bootstrap_replicates=100,
        bootstrap_seed=1,
        block_count=10,
    )
    return DeploymentCandidate(candidate_id, candidate_id * 64, summary)


def test_manifest_ablations_are_executable_overrides() -> None:
    manifest = load_manifest(CONFIG, environment={})
    registry = AblationRegistry.from_manifest(manifest.data["ablations"])
    assert len(registry.plugins) == 8
    ablated = registry.apply("no_regression_objective", manifest.data)
    assert ablated["objectives"]["paired_regression"]["enabled"] is False
    assert manifest.data["objectives"]["paired_regression"]["enabled"] is True


def test_three_frozen_deployment_policies() -> None:
    pool = (
        candidate("a", 0.9, 0.6, 100.0, 0.1),
        candidate("b", 0.8, 0.9, 120.0, 0.05),
        candidate("c", 0.95, 0.5, 80.0, 0.2),
    )
    assert select_min_tokens(pool, accuracy_floor=0.9).candidate_id == "c"
    assert select_max_transfer(pool, token_budget=110).candidate_id == "a"
    frozen_ranges = (
        (0.0, 1.0),
        (0.0, 1.0),
        (-150.0, 0.0),
        (-1.0, 0.0),
    )
    assert select_knee(pool, frozen_ranges=frozen_ranges).candidate_id in {"a", "b", "c"}
