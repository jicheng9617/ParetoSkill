from __future__ import annotations

import hashlib

import pytest

from paretoskill.evaluation import EvaluationRecord
from paretoskill.final_analysis import (
    FinalAnalysisError,
    analyze_heldout_fronts,
    build_final_objective_summaries,
    deployment_candidate_union,
    paired_three_seed_sign_flip_holm,
    partition_final_records,
    resolve_final_execution_groups,
    resolve_final_target_roles,
    validated_false_archive_admission_rate,
)
from paretoskill.objectives import FeasibilityConstraints
from paretoskill.providers import ExecutionResult
from paretoskill.statistics import MetricEstimate, ObjectiveSummary


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configuration() -> dict[str, object]:
    return {
        "models": {
            "small": {"model_id": "small-model", "revision": "model-r1"},
            "large": {"model_id": "large-model", "revision": "model-r1"},
        },
        "harnesses": {
            "primary": {"adapter": "primary-adapter", "repository_revision": "h-r1"},
            "alternate": {"adapter": "alternate-adapter", "repository_revision": "h-r2"},
        },
        "domains": {
            "verified": {"adapter": "spreadsheet", "adapter_revision": "domain-r1"},
            "full": {"adapter": "spreadsheet", "adapter_revision": "domain-r1"},
            "ood-one": {"adapter": "ood-one", "adapter_revision": "ood-r1"},
            "ood-two": {"adapter": "ood-two", "adapter_revision": "ood-r1"},
        },
        "shared_search_controls": {
            "verifier": "verifier-adapter",
            "verifier_revision": "verifier-r1",
        },
        "targets": {
            "final-id": {
                "phase": "final_only",
                "model": "small",
                "harness": "primary",
                "domain": "verified",
                "aggregation_role": "primary_verified_report",
            },
            "final-transfer-one": {
                "phase": "final_only",
                "model": "small",
                "harness": "primary",
                "domain": "ood-one",
                "transfer_group": "ood-one",
            },
            "final-transfer-two": {
                "phase": "final_only",
                "model": "small",
                "harness": "primary",
                "domain": "ood-two",
                "transfer_group": "ood-two",
            },
            "final-full-diagnostic": {
                "phase": "final_only",
                "model": "small",
                "harness": "primary",
                "domain": "full",
                "transfer_group": "expanded-source",
                "final_analysis_role": "transfer",
                "exclude_from_primary_pooled_metrics": True,
                "reuse_verified_execution_when_canonical_keys_match": True,
            },
        },
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_replicates": 20,
            "minimum_effective_blocks_for_archive": 2,
            "hypervolume_reference": {"normalized_maximize_point": [0, 0, 0, 0]},
        },
        "task_seed_blocks": {"bootstrap_seed": 41},
        "constraints": {
            "enabled": True,
            "id_accuracy_floor": {"epsilon": 0.1},
            "token_budget": {"budget": 1_000},
        },
        "selection_protocol": {
            "normalization_ranges": [[0, 1], [0, 1], [-200, 0], [-1, 0]]
        },
    }


def _record(
    *,
    target_id: str,
    candidate_id: str,
    task_id: str,
    correct: bool,
    seed: int = 1,
    is_base: bool = False,
    total_tokens: int = 10,
    transfer_group: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        experiment_id="final-experiment",
        candidate_id=candidate_id,
        content_hash=_digest(f"content:{candidate_id}"),
        target_id=target_id,
        task_id=task_id,
        group_id=transfer_group or "source",
        split="final",
        split_id="heldout",
        transfer_group=transfer_group,
        seed=seed,
        result=ExecutionResult(
            correct=correct,
            input_tokens=total_tokens - 2,
            output_tokens=2,
            latency_ms=1.0,
        ),
        cache_key=_digest(f"{target_id}:{candidate_id}:{task_id}:{seed}"),
        is_base=is_base,
    )


def test_final_roles_exclude_diagnostic_even_with_transfer_hint() -> None:
    roles = resolve_final_target_roles(_configuration())

    assert roles == {
        "final-full-diagnostic": "diagnostic",
        "final-id": "id",
        "final-transfer-one": "transfer",
        "final-transfer-two": "transfer",
    }


def test_overlap_dedup_uses_canonical_execution_signature_and_task_identity() -> None:
    config = _configuration()
    rows: list[EvaluationRecord] = []
    for candidate_id, is_base, correct in (
        ("base", True, True),
        ("candidate", False, False),
    ):
        rows.extend(
            (
                _record(
                    target_id="final-id",
                    candidate_id=candidate_id,
                    task_id="canonical-overlap",
                    correct=correct,
                    is_base=is_base,
                ),
                _record(
                    target_id="final-full-diagnostic",
                    candidate_id=candidate_id,
                    task_id="canonical-overlap",
                    correct=correct,
                    is_base=is_base,
                ),
                _record(
                    target_id="final-full-diagnostic",
                    candidate_id=candidate_id,
                    task_id="full-only",
                    correct=correct,
                    is_base=is_base,
                ),
                # The same canonical task id on a different domain adapter is a
                # separate transfer execution and must not be collapsed.
                _record(
                    target_id="final-transfer-one",
                    candidate_id=candidate_id,
                    task_id="canonical-overlap",
                    correct=correct,
                    is_base=is_base,
                    transfer_group="ood-one",
                ),
            )
        )

    groups = resolve_final_execution_groups(config)
    partition = partition_final_records(rows, config)

    assert groups["final-id"] == groups["final-full-diagnostic"]
    assert groups["final-id"] != groups["final-transfer-one"]
    assert len(partition.id_records) == 2
    assert len(partition.transfer_records) == 2
    assert len(partition.diagnostic_records) == 2
    assert partition.dropped_overlap_count == 2
    assert partition.to_dict()["deduplicated_record_count"] == 6


def test_overlap_dedup_rejects_conflicting_reused_outcomes() -> None:
    config = _configuration()
    rows = [
        _record(
            target_id="final-id",
            candidate_id="candidate",
            task_id="same",
            correct=True,
        ),
        _record(
            target_id="final-full-diagnostic",
            candidate_id="candidate",
            task_id="same",
            correct=False,
        ),
    ]

    with pytest.raises(FinalAnalysisError, match="conflicting outcomes"):
        partition_final_records(rows, config)


def test_build_final_summaries_keeps_roles_separate_and_excludes_diagnostics() -> None:
    config = _configuration()
    rows: list[EvaluationRecord] = []
    specifications = (
        ("final-id", "id-1", 1, None, True),
        ("final-id", "id-2", 2, None, False),
        ("final-transfer-one", "one-1", 1, "ood-one", True),
        ("final-transfer-one", "one-2", 2, "ood-one", True),
        ("final-transfer-two", "two-1", 1, "ood-two", True),
        ("final-transfer-two", "two-2", 2, "ood-two", False),
    )
    for target_id, task_id, seed, group, candidate_correct in specifications:
        rows.append(
            _record(
                target_id=target_id,
                candidate_id="base",
                task_id=task_id,
                seed=seed,
                correct=True,
                is_base=True,
                transfer_group=group,
            )
        )
        rows.append(
            _record(
                target_id=target_id,
                candidate_id="candidate",
                task_id=task_id,
                seed=seed,
                correct=candidate_correct,
                transfer_group=group,
            )
        )
    # These deliberately extreme diagnostic outcomes must not affect any
    # primary point estimate.
    for candidate_id, is_base in (("base", True), ("candidate", False)):
        rows.append(
            _record(
                target_id="final-full-diagnostic",
                candidate_id=candidate_id,
                task_id="diagnostic-only",
                correct=False,
                is_base=is_base,
                total_tokens=999,
            )
        )

    summary = build_final_objective_summaries(
        rows,
        config,
        bootstrap_replicates=20,
    )["candidate"]

    assert summary.id_accuracy.point == pytest.approx(0.5)
    assert summary.worst_target_transfer.point == pytest.approx(0.5)
    assert summary.token_cost.point == pytest.approx(10.0)
    assert summary.paired_regression.point == pytest.approx(2 / 6)
    assert summary.id_accuracy_delta is not None
    assert summary.id_accuracy_delta.point == pytest.approx(-0.5)


def _estimate(
    point: float, *, lcb: float | None = None, ucb: float | None = None
) -> MetricEstimate:
    lower = point if lcb is None else lcb
    upper = point if ucb is None else ucb
    return MetricEstimate(
        point=point,
        ci_low=lower,
        ci_high=upper,
        lcb=lower,
        ucb=upper,
        sample_size=3,
    )


def _summary(
    identity: float,
    transfer: float,
    tokens: float,
    regression: float,
    *,
    delta_lcb: float = 0.0,
) -> ObjectiveSummary:
    return ObjectiveSummary(
        id_accuracy=_estimate(identity),
        worst_target_transfer=_estimate(transfer),
        token_cost=_estimate(tokens),
        paired_regression=_estimate(regression),
        id_accuracy_delta=_estimate(delta_lcb),
        confidence_level=0.95,
        bootstrap_replicates=20,
        bootstrap_seed=1,
        block_count=3,
    )


def test_heldout_fronts_reconstruct_per_run_and_pooled_metrics() -> None:
    config = _configuration()
    summaries = {
        "a": _summary(0.80, 0.70, 100, 0.10),
        "b": _summary(0.85, 0.60, 80, 0.05),
        "c": _summary(0.70, 0.60, 120, 0.20),
        "infeasible": _summary(0.90, 0.90, 50, 0.01, delta_lcb=-0.20),
    }
    constraints = FeasibilityConstraints.from_paired_epsilon(
        epsilon=0.10,
        token_budget=200,
    )

    result = analyze_heldout_fronts(
        summaries,
        {
            ("method-one", 11): ("a", "c"),
            ("method-two", 11): ("b",),
            ("empty-run", 11): ("infeasible",),
        },
        config,
        constraints=constraints,
    )
    runs = {run.method_id: run for run in result.runs}

    assert result.defined is True
    assert result.pooled_front_candidate_ids == ("a", "b")
    assert runs["method-one"].front_candidate_ids == ("a",)
    assert runs["method-two"].front_candidate_ids == ("b",)
    assert runs["method-one"].hypervolume.defined
    assert runs["method-one"].hypervolume.value > 0
    assert runs["method-one"].coverage_of_pooled_reference.value == pytest.approx(0.5)
    assert runs["method-one"].additive_epsilon.defined
    assert runs["method-one"].inverted_generational_distance.defined
    assert runs["empty-run"].hypervolume.value == 0.0
    assert runs["empty-run"].coverage_of_pooled_reference.value == 0.0
    assert runs["empty-run"].additive_epsilon.defined is False
    assert runs["empty-run"].additive_epsilon.reason == "run_has_no_feasible_candidates"
    assert runs["empty-run"].inverted_generational_distance.defined is False


def test_heldout_fronts_explicitly_undefined_without_any_feasible_point() -> None:
    result = analyze_heldout_fronts(
        {"only": _summary(0.9, 0.9, 50, 0.01, delta_lcb=-0.2)},
        {("method", 1): ("only",)},
        _configuration(),
        constraints=FeasibilityConstraints.from_paired_epsilon(
            epsilon=0.1,
            token_budget=200,
        ),
    )

    assert result.defined is False
    assert result.undefined_reason == "no_feasible_candidates_across_runs"
    assert result.runs[0].hypervolume.value == 0.0
    assert result.runs[0].coverage_of_pooled_reference.defined is False
    assert result.runs[0].additive_epsilon.defined is False
    assert result.runs[0].inverted_generational_distance.defined is False


def test_three_seed_exact_sign_flip_and_holm_are_paired_by_seed() -> None:
    values = {
        ("paretoskill", 1): 3.0,
        ("paretoskill", 2): 3.0,
        ("paretoskill", 3): 3.0,
        ("baseline-a", 1): 1.0,
        ("baseline-a", 2): 1.0,
        ("baseline-a", 3): 1.0,
        ("baseline-b", 1): 2.0,
        ("baseline-b", 2): 4.0,
        ("baseline-b", 3): 2.0,
    }

    analysis = paired_three_seed_sign_flip_holm(
        values,
        treatment_method="paretoskill",
        comparison_methods=("baseline-a", "baseline-b"),
        seeds=(1, 2, 3),
    )
    contrasts = {item.comparison_method: item for item in analysis.contrasts}

    assert analysis.family_complete is True
    assert contrasts["baseline-a"].differences == (2.0, 2.0, 2.0)
    assert contrasts["baseline-a"].exact_two_sided_p.value == pytest.approx(0.25)
    assert contrasts["baseline-a"].holm_adjusted_p.value == pytest.approx(0.5)
    assert contrasts["baseline-b"].differences == (1.0, -1.0, 1.0)
    assert contrasts["baseline-b"].exact_two_sided_p.value == pytest.approx(1.0)
    assert contrasts["baseline-b"].holm_adjusted_p.value == pytest.approx(1.0)


def test_incomplete_predeclared_holm_family_has_no_adjusted_p_values() -> None:
    values = {
        ("paretoskill", 1): 3.0,
        ("paretoskill", 2): 3.0,
        ("paretoskill", 3): 3.0,
        ("complete", 1): 1.0,
        ("complete", 2): 1.0,
        ("complete", 3): 1.0,
        ("incomplete", 1): 1.0,
        ("incomplete", 2): None,
        ("incomplete", 3): 1.0,
    }

    analysis = paired_three_seed_sign_flip_holm(
        values,
        treatment_method="paretoskill",
        comparison_methods=("complete", "incomplete"),
        seeds=(1, 2, 3),
    )
    contrasts = {item.comparison_method: item for item in analysis.contrasts}

    assert analysis.family_complete is False
    assert contrasts["complete"].exact_two_sided_p.defined is True
    assert contrasts["complete"].holm_adjusted_p.defined is False
    assert contrasts["complete"].holm_adjusted_p.reason == "predeclared_holm_family_incomplete"
    assert contrasts["incomplete"].exact_two_sided_p.defined is False
    assert contrasts["incomplete"].missing_or_undefined_seeds == (2,)


def test_deployment_candidate_union_collects_defined_policy_outputs() -> None:
    payloads = {
        "run-one": {
            "primary_candidate_id": "candidate-a",
            "policies": {
                "balanced": {"defined": True, "candidate_id": "candidate-a"},
                "cheap": {"defined": True, "candidate_id": "candidate-b"},
                "missing": {"defined": False, "candidate_id": None},
            },
        },
        "run-two": {
            "primary_candidate_id": "candidate-c",
            "policies": {
                "balanced": {"defined": True, "candidate_id": "candidate-c"},
                "cheap": {"defined": False, "candidate_id": "ignored"},
            },
        },
    }

    assert deployment_candidate_union(payloads) == (
        "candidate-a",
        "candidate-b",
        "candidate-c",
    )


def test_false_admission_denominator_only_counts_full_validated_admissions() -> None:
    result = validated_false_archive_admission_rate(
        screen_admitted_ids=("a", "b", "not-promoted"),
        full_validated_ids=("a", "b", "full-only"),
        full_archive_ids=("a", "full-only"),
    )

    assert result.rate.value == pytest.approx(0.5)
    assert result.false_admission_count == 1
    assert result.eligible_admission_count == 2
    assert result.false_admission_ids == ("b",)
    assert result.excluded_unvalidated_ids == ("not-promoted",)
    assert result.to_dict()["denominator_definition"] == (
        "screen_admitted_intersection_full_validated"
    )


def test_false_admission_rate_is_undefined_for_empty_legal_denominator() -> None:
    result = validated_false_archive_admission_rate(("screened",), (), ())

    assert result.rate.defined is False
    assert result.rate.reason == "no_screen_admissions_received_full_validation"
    assert result.eligible_admission_count == 0


def test_false_admission_rejects_unvalidated_full_archive_member() -> None:
    with pytest.raises(FinalAnalysisError, match="without full validation"):
        validated_false_archive_admission_rate(("a",), ("a",), ("a", "unknown"))
