from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from paretoskill.archive import ArchiveEntry, ParetoArchive
from paretoskill.baselines import (
    Ctx2SkillFixedProductPlugin,
    MOCHAPlugin,
    PluginRegistry,
    ScoredCandidate,
    SimplePatchCompositionPlugin,
)
from paretoskill.config import ConfigError, load_manifest, validate_manifest
from paretoskill.deployment import DeploymentCandidate, select_knee
from paretoskill.evaluation import target_specs_from_manifest
from paretoskill.objectives import FeasibilityConstraints
from paretoskill.statistics import MetricEstimate, ObjectiveSummary, PointObjectives


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


def _estimate(value: float) -> MetricEstimate:
    return MetricEstimate(value, value, value, value, value, 10)


def _summary(
    accuracy: float, transfer: float, tokens: float, regression: float
) -> ObjectiveSummary:
    return ObjectiveSummary(
        id_accuracy=_estimate(accuracy),
        worst_target_transfer=_estimate(transfer),
        token_cost=_estimate(tokens),
        paired_regression=_estimate(regression),
        confidence_level=0.95,
        bootstrap_replicates=100,
        bootstrap_seed=17,
        block_count=10,
    )


def _scored(
    candidate_id: str,
    *,
    accuracy: float,
    transfer: float = 0.5,
    tokens: float = 100.0,
    regression: float = 0.1,
    metadata: dict[str, Any] | None = None,
) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=candidate_id,
        patch_ids=(),
        objectives=PointObjectives(accuracy, transfer, tokens, regression),
        metadata={} if metadata is None else metadata,
    )


def _deployment(
    candidate_id: str,
    *,
    accuracy: float,
    transfer: float,
    tokens: float,
    regression: float,
) -> DeploymentCandidate:
    return DeploymentCandidate(
        candidate_id=candidate_id,
        content_hash=candidate_id * 64,
        objectives=_summary(accuracy, transfer, tokens, regression),
    )


def test_relaxed_registry_constructs_every_synthetic_manifest_plugin() -> None:
    manifest = load_manifest(CONFIG, profile="dry_run", environment={})

    registry = PluginRegistry.from_manifest(
        manifest.data,
        normalization_ranges=None,
        strict_frozen_inputs=False,
    )

    expected = {
        *manifest.data["methods"],
        "ctx2skill_hard_easy_product",
        "fixed_scalarization/accuracy_only",
        "fixed_scalarization/accuracy_cost_equal",
        "fixed_scalarization/balanced_four_objective",
    }
    assert expected <= set(registry.plugins)
    assert registry.get("no_skill").artifact_kind == "no_skill_injection"
    assert registry.get("paretoskill").archive_conditioned_generation is True


def test_manifest_targets_preserve_split_role_and_transfer_group() -> None:
    manifest = load_manifest(CONFIG, profile="dry_run", environment={})
    targets = {
        target.target_id: target
        for target in target_specs_from_manifest(manifest.data, phase="search")
    }

    assert targets["id_small_primary"].split_id == "id_validation"
    assert targets["id_small_primary"].objective_role == "id"
    assert targets["id_small_primary"].transfer_group is None
    assert targets["cross_scale_large"].objective_role == "transfer"
    assert targets["cross_scale_large"].transfer_group == "model_scale"


def test_simple_patch_composition_filters_infeasible_then_ties_on_tokens() -> None:
    plugin = SimplePatchCompositionPlugin()
    pool = (
        _scored("infeasible-high", accuracy=0.99, tokens=1, metadata={"feasible": False}),
        _scored("lower-accuracy", accuracy=0.80, tokens=1, metadata={"feasible": True}),
        _scored("accurate-expensive", accuracy=0.90, tokens=120, metadata={"feasible": True}),
        _scored("accurate-cheap", accuracy=0.90, tokens=80, metadata={"feasible": True}),
    )

    selected = plugin.select(pool, count=3)

    assert [item.candidate_id for item in selected] == [
        "accurate-cheap",
        "accurate-expensive",
        "lower-accuracy",
    ]
    assert all(item.metadata["feasible"] is True for item in selected)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"hard_probe_success_rate": 0.5},
        {"hard_probe_success_rate": True, "easy_probe_success_rate": 0.5},
        {"hard_probe_success_rate": "0.5", "easy_probe_success_rate": 0.5},
        {"hard_probe_success_rate": float("nan"), "easy_probe_success_rate": 0.5},
        {"hard_probe_success_rate": -0.01, "easy_probe_success_rate": 0.5},
        {"hard_probe_success_rate": 0.5, "easy_probe_success_rate": 1.01},
    ],
)
def test_ctx2skill_probe_rates_are_strict(metadata: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="probe|numeric|finite"):
        candidate = _scored("candidate", accuracy=0.8, metadata=metadata)
        Ctx2SkillFixedProductPlugin().select((candidate,), count=1)


def test_ctx2skill_selects_largest_valid_hard_easy_product() -> None:
    pool = (
        _scored(
            "balanced",
            accuracy=0.8,
            metadata={"hard_probe_success_rate": 0.8, "easy_probe_success_rate": 0.8},
        ),
        _scored(
            "hard-only",
            accuracy=0.9,
            metadata={"hard_probe_success_rate": 1.0, "easy_probe_success_rate": 0.5},
        ),
    )

    selected = Ctx2SkillFixedProductPlugin().select(pool, count=1)

    assert selected[0].candidate_id == "balanced"


def test_mocha_seeded_parent_and_acceptance_are_reproducible() -> None:
    ranges = ((0.0, 1.0), (0.0, 1.0), (-200.0, 0.0), (-1.0, 0.0))
    first = MOCHAPlugin(ranges=ranges, require_frozen_ranges=True)
    second = MOCHAPlugin(ranges=ranges, require_frozen_ranges=True)
    incumbents = (
        _scored("accuracy", accuracy=0.9, transfer=0.5, tokens=100, regression=0.2),
        _scored("transfer", accuracy=0.6, transfer=0.9, tokens=120, regression=0.1),
    )
    proposal = _scored(
        "proposal", accuracy=0.8, transfer=0.8, tokens=90, regression=0.08
    )

    assert first.sample_weights(seed=31415) == second.sample_weights(seed=31415)
    assert first.select_parent(incumbents, seed=31415).candidate_id == second.select_parent(
        tuple(reversed(incumbents)), seed=31415
    ).candidate_id
    accepted = first.accept_proposal(
        proposal,
        incumbents,
        task_executions_spent=250,
        task_execution_budget=1000,
        seed=2718,
    )
    repeated = second.accept_proposal(
        proposal,
        tuple(reversed(incumbents)),
        task_executions_spent=250,
        task_execution_budget=1000,
        seed=2718,
    )
    assert repeated is accepted


def test_knee_requires_frozen_ranges_and_ignores_added_dominated_candidate() -> None:
    pool = (
        _deployment("a", accuracy=0.9, transfer=0.7, tokens=100, regression=0.1),
        _deployment("b", accuracy=0.8, transfer=0.9, tokens=120, regression=0.05),
    )
    ranges = ((0.0, 1.0), (0.0, 1.0), (-200.0, 0.0), (-1.0, 0.0))

    with pytest.raises(ValueError, match="validation-frozen"):
        select_knee(pool)

    selected = select_knee(pool, frozen_ranges=ranges)
    dominated = _deployment(
        "d", accuracy=0.7, transfer=0.5, tokens=160, regression=0.2
    )
    selected_with_dominated = select_knee(pool + (dominated,), frozen_ranges=ranges)

    assert selected.candidate_id == "a"
    assert selected_with_dominated.candidate_id == selected.candidate_id


def test_bounded_archive_is_order_invariant_but_scientific_front_is_unbounded() -> None:
    candidates = tuple(
        ArchiveEntry(
            candidate_id=f"tradeoff-{index}",
            content_hash=f"{index:064x}",
            objectives=_summary(
                accuracy=0.55 + 0.10 * index,
                transfer=0.95 - 0.10 * index,
                tokens=100,
                regression=0.1,
            ),
        )
        for index in range(5)
    )

    def build(order: tuple[ArchiveEntry, ...]) -> ParetoArchive:
        result = ParetoArchive(
            max_size=2,
            evaluation_budget=20,
            constraints=FeasibilityConstraints(accuracy_floor=0.5, token_budget=200),
        )
        for item in order:
            result.admit(item)
        return result

    forward = build(candidates)
    reverse = build(tuple(reversed(candidates)))

    assert [item.candidate_id for item in forward.entries] == [
        item.candidate_id for item in reverse.entries
    ]
    assert {item.candidate_id for item in forward.entries} == {"tradeoff-0", "tradeoff-4"}
    assert len(forward.entries) == 2
    assert len(forward.scientific_front) == 5
    assert {item.candidate_id for item in forward.scientific_front} == {
        item.candidate_id for item in candidates
    }


@pytest.mark.parametrize(
    ("path", "malformed"),
    [
        (("runtime_profiles", "real"), []),
        (("datasets",), []),
        (("targets",), []),
        (("splits", "evolution_trace", "expected_count"), {}),
        (("task_seed_blocks", "search_seeds"), [[1]]),
        (("objectives", "id_accuracy", "target_ids"), None),
        (("budgets", "screen", "target_ids"), {}),
        (("methods",), []),
        (("ablations",), {}),
        (("outputs", "required_files"), {}),
    ],
)
def test_malformed_manifest_types_are_wrapped_as_config_errors(
    path: tuple[str, ...], malformed: object
) -> None:
    data = copy.deepcopy(load_manifest(CONFIG, profile="dry_run", environment={}).data)
    current = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = malformed

    with pytest.raises(ConfigError):
        validate_manifest(data, profile="dry_run")


def test_profile_application_type_error_is_wrapped_as_config_error(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["deployment"]["policies"] = [None]
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_manifest(malformed, profile="dry_run", environment={})
