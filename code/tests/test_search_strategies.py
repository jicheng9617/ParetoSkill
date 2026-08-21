from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from typing import Any

import pytest

from paretoskill.baselines import MOCHAPlugin, ScoredCandidate
from paretoskill.search_strategies import (
    AdapterBackedBinarySubsetController,
    BernoulliUniqueStream,
    CommonCandidateStream,
    EvoTopKController,
    ExternalOptimizerRequired,
    InitialDesignBinarySubsetController,
    MOCHAController,
    NSGAIIController,
    SearchSpaceExhausted,
    SearchStrategyError,
    make_binary_subset_controller,
    restore_search_controller,
)
from paretoskill.statistics import PointObjectives


PATCHES = tuple(f"p{index:02d}" for index in range(10))


def scored(subset: tuple[str, ...], *, salt: str = "", accuracy: float | None = None):
    digest = hashlib.sha256(("\0".join(subset) + salt).encode()).hexdigest()
    bit_sum = sum(int(patch_id[1:]) + 1 for patch_id in subset)
    return ScoredCandidate(
        candidate_id=f"candidate-{digest[:16]}",
        patch_ids=subset,
        objectives=PointObjectives(
            id_accuracy=(bit_sum % 97) / 100 if accuracy is None else accuracy,
            worst_target_transfer=(bit_sum % 83) / 100,
            token_cost=float(len(subset) * 10),
            paired_regression=(bit_sum % 71) / 100,
        ),
        content_hash=digest,
    )


def round_trip(value: Mapping[str, Any]) -> dict[str, Any]:
    restored = json.loads(json.dumps(value, sort_keys=True))
    assert isinstance(restored, dict)
    return restored


def test_bernoulli_unique_stream_is_nonempty_reproducible_and_resumable() -> None:
    left = BernoulliUniqueStream(PATCHES[:6], 104729, 0.5, 1_000)
    right = BernoulliUniqueStream(tuple(reversed(PATCHES[:6])), 104729, 0.5, 1_000)
    first = left.ask(12)
    assert first == right.ask(12)
    assert len(first) == len(set(first)) == 12
    assert all(subset and tuple(sorted(subset)) == subset for subset in first)

    restored = BernoulliUniqueStream.from_state(round_trip(left.state_dict()))
    assert restored.ask(8) == left.ask(8)
    assert restored.seen == left.seen

    with pytest.raises(ValueError, match="greater than zero"):
        BernoulliUniqueStream(("p",), 1, 0.0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        BernoulliUniqueStream(("p",), True)


def test_bernoulli_duplicate_cap_and_space_exhaustion_are_explicit() -> None:
    stream = BernoulliUniqueStream(("only",), 3, 1.0, 2)
    assert stream.ask(1) == (("only",),)
    with pytest.raises(SearchSpaceExhausted, match="only 0 remaining"):
        stream.ask(1)

    concentrated = BernoulliUniqueStream(("a", "b"), 3, 1.0, 2)
    assert concentrated.ask(1) == (("a", "b"),)
    before_failure = concentrated.state_dict()
    with pytest.raises(SearchSpaceExhausted, match="duplicate"):
        concentrated.ask(1)
    assert concentrated.state_dict() == before_failure


def test_common_candidate_stream_can_be_forked_across_methods() -> None:
    common = CommonCandidateStream.from_bernoulli(
        PATCHES[:7], candidate_count=30, seed=17
    )
    method_a = common.fork()
    method_b = common.fork()
    assert method_a.ask(11) == method_b.ask(11)
    assert method_a.ask(7) == method_b.ask(7)
    assert common.cursor == 0

    restored = CommonCandidateStream.from_state(round_trip(method_a.state_dict()))
    assert restored.ask(100) == method_a.ask(100)
    assert restored.remaining == 0


def test_binary_bayesian_factory_never_mislabels_builtin_random_design() -> None:
    with pytest.raises(ExternalOptimizerRequired, match="no GP implementation"):
        make_binary_subset_controller(PATCHES, seed=7)

    controller = make_binary_subset_controller(
        PATCHES,
        seed=7,
        mode="initial_design_only",
        initial_design_size=4,
    )
    assert isinstance(controller, InitialDesignBinarySubsetController)
    assert controller.optimizer_kind == "seeded_initial_design_only_not_gp"
    batch = controller.ask(4)
    controller.tell(scored(subset) for subset in batch)
    with pytest.raises(ExternalOptimizerRequired, match="stops after its initial design"):
        controller.ask(1)

    restored = InitialDesignBinarySubsetController.from_state(
        round_trip(controller.state_dict())
    )
    assert [value.candidate_id for value in restored.observations] == [
        value.candidate_id for value in controller.observations
    ]
    generic = restore_search_controller(round_trip(controller.state_dict()))
    assert isinstance(generic, InitialDesignBinarySubsetController)


class LexicographicAdapter:
    adapter_id = "tests.lexicographic_binary_adapter/v1"

    def __init__(self) -> None:
        self.tell_count = 0

    def ask(
        self,
        *,
        patch_ids: tuple[str, ...],
        count: int,
        seed: int,
        seen_subsets: tuple[tuple[str, ...], ...],
        observations: tuple[ScoredCandidate, ...],
    ) -> tuple[tuple[str, ...], ...]:
        del seed, observations
        seen = set(seen_subsets)
        candidates = (
            combination
            for size in range(1, len(patch_ids) + 1)
            for combination in itertools.combinations(patch_ids, size)
            if combination not in seen
        )
        return tuple(itertools.islice(candidates, count))

    def tell(self, scored_values: tuple[ScoredCandidate, ...]) -> None:
        self.tell_count += len(scored_values)

    def state_dict(self) -> Mapping[str, Any]:
        return {"tell_count": self.tell_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.tell_count = int(state["tell_count"])


def test_external_binary_adapter_is_strictly_wrapped_and_resumable() -> None:
    adapter = LexicographicAdapter()
    controller = make_binary_subset_controller(PATCHES[:4], seed=11, adapter=adapter)
    assert isinstance(controller, AdapterBackedBinarySubsetController)
    first = controller.ask(3)
    assert first == (("p00",), ("p01",), ("p02",))
    with pytest.raises(SearchStrategyError, match="pending batch"):
        controller.ask(1)
    controller.tell(scored(subset) for subset in reversed(first))
    assert adapter.tell_count == 3

    restored_adapter = LexicographicAdapter()
    restored = AdapterBackedBinarySubsetController.from_state(
        round_trip(controller.state_dict()), adapter=restored_adapter
    )
    assert restored_adapter.tell_count == 3
    assert restored.ask(2) == controller.ask(2)
    with pytest.raises(ExternalOptimizerRequired, match="requires its external adapter"):
        restore_search_controller(round_trip(controller.state_dict()))


def test_evo_top_k_consumes_the_same_stream_and_restores_state() -> None:
    stream = CommonCandidateStream.from_bernoulli(
        PATCHES[:6], candidate_count=16, seed=2027
    )
    evo = EvoTopKController(stream.fork(), top_k=3)
    batch = evo.ask(8)
    accuracies = (0.1, 0.7, 0.2, 0.9, 0.8, 0.3, 0.4, 0.6)
    evo.tell(
        scored(subset, salt=str(index), accuracy=accuracy)
        for index, (subset, accuracy) in enumerate(zip(batch, accuracies, strict=True))
    )
    assert [item.objectives.id_accuracy for item in evo.incumbents] == [0.9, 0.8, 0.7]

    restored = EvoTopKController.from_state(round_trip(evo.state_dict()))
    next_batch = evo.ask(4)
    assert restored.ask(4) == next_batch
    next_scores = tuple(
        scored(subset, salt=f"next-{index}", accuracy=0.95 - 0.01 * index)
        for index, subset in enumerate(next_batch)
    )
    evo.tell(next_scores)
    restored.tell(next_scores)
    assert restored.state_dict() == evo.state_dict()


def test_mocha_controller_is_seeded_and_uses_plugin_parent_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        ("p00",),
        ("p01",),
        ("p00", "p01"),
        ("p02",),
    )
    common = CommonCandidateStream(PATCHES[:3], candidates, seed=2027)
    ranges = ((0.0, 1.0), (0.0, 1.0), (-100.0, 0.0), (-1.0, 0.0))
    plugin = MOCHAPlugin(ranges=ranges, require_frozen_ranges=True)
    parent_calls: list[tuple[tuple[str, ...], int]] = []
    acceptance_calls: list[tuple[str, tuple[str, ...], int, int]] = []
    original_parent = MOCHAPlugin.select_parent
    original_acceptance = MOCHAPlugin.accept_proposal

    def record_parent(
        self: MOCHAPlugin,
        values: tuple[ScoredCandidate, ...],
        *,
        seed: int,
    ) -> ScoredCandidate:
        parent_calls.append((tuple(value.candidate_id for value in values), seed))
        return original_parent(self, values, seed=seed)

    def record_acceptance(
        self: MOCHAPlugin,
        proposal: ScoredCandidate,
        incumbents: tuple[ScoredCandidate, ...],
        *,
        task_executions_spent: int,
        task_execution_budget: int,
        seed: int,
    ) -> bool:
        acceptance_calls.append(
            (
                proposal.candidate_id,
                tuple(value.candidate_id for value in incumbents),
                task_executions_spent,
                seed,
            )
        )
        return original_acceptance(
            self,
            proposal,
            incumbents,
            task_executions_spent=task_executions_spent,
            task_execution_budget=task_execution_budget,
            seed=seed,
        )

    monkeypatch.setattr(MOCHAPlugin, "select_parent", record_parent)
    monkeypatch.setattr(MOCHAPlugin, "accept_proposal", record_acceptance)
    left = MOCHAController(
        common.fork(),
        seed=31415,
        logical_task_execution_budget=40,
        task_executions_per_candidate=10,
        plugin=plugin,
    )
    right = MOCHAController(
        common.fork(),
        seed=31415,
        logical_task_execution_budget=40,
        task_executions_per_candidate=10,
        plugin=MOCHAPlugin(ranges=ranges, require_frozen_ranges=True),
    )

    first = left.ask()
    assert first == right.ask() == (("p00",),)
    first_score = scored(first[0], salt="first", accuracy=0.7)
    left.tell((first_score,))
    right.tell((first_score,))
    assert [call[2] for call in acceptance_calls] == [10, 10]
    assert left.accepted_ids == (first_score.candidate_id,)
    assert left.logical_task_executions_spent == 10
    assert left.logical_budget_progress == 0.25

    second = left.ask()
    assert second == right.ask() == (("p01",),)
    parent_seed = left.pending_parent_seeds[0]
    assert parent_seed is not None
    assert parent_calls[:2] == [
        ((first_score.candidate_id,), parent_seed),
        ((first_score.candidate_id,), parent_seed),
    ]
    expected_parent = plugin.select_parent(left.incumbents, seed=parent_seed)
    assert left.pending_parent_ids == (expected_parent.candidate_id,)
    second_score = scored(second[0], salt="second", accuracy=0.9)
    incumbents_before = left.incumbents
    left.tell((second_score,))
    right.tell((second_score,))
    assert [call[2] for call in acceptance_calls[-2:]] == [20, 20]
    decision = left.decision_log[-1]
    expected_acceptance = plugin.accept_proposal(
        second_score,
        incumbents_before,
        task_executions_spent=20,
        task_execution_budget=40,
        seed=decision["acceptance_seed"],
    )
    assert decision["accepted"] is expected_acceptance
    assert decision["parent_id"] == expected_parent.candidate_id
    assert left.state_dict() == right.state_dict()


def test_mocha_controller_json_roundtrip_restores_pending_and_progress() -> None:
    candidates = (
        ("p00",),
        ("p01",),
        ("p02",),
        ("p00", "p02"),
    )
    common = CommonCandidateStream(PATCHES[:3], candidates, seed=7)
    controller = MOCHAController(
        common,
        seed=2718,
        logical_task_execution_budget=28,
        task_executions_per_candidate=7,
    )
    first = controller.ask()
    controller.tell((scored(first[0], salt="initial", accuracy=0.8),))
    pending = controller.ask(2)

    state = round_trip(controller.state_dict())
    restored = MOCHAController.from_state(state)
    generic = restore_search_controller(state)
    assert isinstance(generic, MOCHAController)
    assert restored.state_dict() == controller.state_dict() == generic.state_dict()
    assert restored.pending_parent_ids == controller.pending_parent_ids
    assert restored.pending_parent_seeds == controller.pending_parent_seeds

    feedback = tuple(
        scored(subset, salt=f"pending-{index}", accuracy=0.6 + index / 10)
        for index, subset in enumerate(pending)
    )
    controller.tell(feedback)
    restored.tell(feedback)
    generic.tell(feedback)
    assert restored.state_dict() == controller.state_dict() == generic.state_dict()
    assert controller.logical_task_executions_spent == 21
    assert len(controller.decision_log) == 3
    with pytest.raises(SearchStrategyError, match="logical budget"):
        controller.ask(2)


def test_mocha_invalid_materialization_is_zero_cost_and_never_accepted() -> None:
    common = CommonCandidateStream(
        PATCHES[:2],
        (("p00",), ("p01",)),
        seed=19,
    )
    controller = MOCHAController(
        common,
        seed=23,
        logical_task_execution_budget=10,
        task_executions_per_candidate=5,
    )
    invalid_subset = controller.ask()[0]
    invalid_digest = hashlib.sha256(b"invalid-materialization").hexdigest()
    invalid = ScoredCandidate(
        candidate_id="invalid-materialization",
        patch_ids=invalid_subset,
        objectives=PointObjectives(0.0, 0.0, 10_000.0, 1.0),
        metadata={
            "materialization_valid": False,
            "evaluation_cost": 0,
            "reason": "synthetic conflict",
        },
        content_hash=invalid_digest,
    )
    controller.tell((invalid,))

    assert controller.incumbents == ()
    assert controller.accepted_ids == ()
    assert controller.logical_task_executions_spent == 0
    assert controller.logical_budget_progress == 0.0
    invalid_decision = controller.decision_log[-1]
    assert invalid_decision["materialization_valid"] is False
    assert invalid_decision["evaluation_cost"] == 0
    assert invalid_decision["accepted"] is False
    assert invalid_decision["acceptance_seed"] is None
    assert invalid_decision["logical_task_executions_before"] == 0
    assert invalid_decision["logical_task_executions_after"] == 0

    restored = MOCHAController.from_state(round_trip(controller.state_dict()))
    valid_subset = controller.ask()[0]
    assert restored.ask() == (valid_subset,)
    assert controller.pending_parent_ids == restored.pending_parent_ids == (None,)
    valid = scored(valid_subset, salt="first-valid", accuracy=0.8)
    controller.tell((valid,))
    restored.tell((valid,))
    assert controller.state_dict() == restored.state_dict()
    assert controller.accepted_ids == (valid.candidate_id,)
    assert controller.logical_task_executions_spent == 5


def test_nsga2_default_20_by_20_ask_tell_is_deterministic_and_resumable() -> None:
    common = CommonCandidateStream.from_bernoulli(
        PATCHES, candidate_count=20, seed=31337
    )
    left = NSGAIIController(PATCHES, seed=271828, initial_stream=common)
    right = NSGAIIController(PATCHES, seed=271828, initial_stream=common)
    assert left.population_size == left.offspring_size == 20

    initial = left.ask()
    assert initial == right.ask()
    assert len(initial) == len(set(initial)) == 20
    initial_scores = tuple(scored(subset, salt="initial") for subset in initial)
    left.tell(initial_scores)
    right.tell(initial_scores)

    offspring = left.ask()
    assert offspring == right.ask()
    assert len(offspring) == len(set(offspring)) == 20
    assert not set(offspring) & set(initial)
    assert all(offspring)

    restored = NSGAIIController.from_state(round_trip(left.state_dict()))
    offspring_scores = tuple(scored(subset, salt="offspring") for subset in offspring)
    left.tell(offspring_scores)
    right.tell(offspring_scores)
    restored.tell(offspring_scores)
    assert left.generation == right.generation == restored.generation == 1
    assert len(left.population) == 20
    assert left.state_dict() == right.state_dict() == restored.state_dict()

    next_left = left.ask()
    assert next_left == right.ask() == restored.ask()


def test_nsga2_rejects_invalid_seed_space_and_feedback() -> None:
    with pytest.raises(ValueError, match="population_size exceeds"):
        NSGAIIController(("a", "b", "c"), seed=1)
    controller = NSGAIIController(PATCHES[:5], seed=1, population_size=4, offspring_size=2)
    pending = controller.ask()
    with pytest.raises(SearchStrategyError, match="exactly 4"):
        controller.tell(scored(subset) for subset in pending[:3])
