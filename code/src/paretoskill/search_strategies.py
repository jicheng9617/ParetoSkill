"""Deterministic, dependency-light subset-search controllers.

The controllers in this module deliberately separate candidate generation from
evaluation.  ``ask`` returns non-empty patch-id subsets and ``tell`` consumes
the corresponding :class:`~paretoskill.baselines.ScoredCandidate` values.  All
built-in strategies are local, deterministic under an integer seed, and expose
JSON-compatible state for checkpoint/restart.

This module does *not* contain a Gaussian-process implementation.  The binary
Bayesian controller requires an explicit external adapter.  Its built-in
fallback is named and limited to the seeded initial design so an experiment can
never silently report random search as GP Bayesian optimization.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .baselines import MOCHAPlugin, ScoredCandidate, nondominated_fronts
from .metrics import crowding_distance
from .models import freeze_mapping, thaw_json
from .statistics import ObjectiveSummary, PointObjectives


Subset = tuple[str, ...]


class SearchStrategyError(RuntimeError):
    """Base error for invalid state transitions or unavailable search logic."""


class SearchSpaceExhausted(SearchStrategyError):
    """Raised when a requested unique proposal cannot be generated."""


class ExternalOptimizerRequired(SearchStrategyError):
    """Raised when a real optimizer is required but no adapter was supplied."""


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _patch_universe(patch_ids: Iterable[str]) -> tuple[str, ...]:
    raw = tuple(patch_ids)
    if not raw:
        raise ValueError("patch_ids must be non-empty")
    if any(not isinstance(patch_id, str) or not patch_id for patch_id in raw):
        raise ValueError("patch_ids must contain non-empty strings")
    if len(set(raw)) != len(raw):
        raise ValueError("patch_ids must be unique")
    return tuple(sorted(raw))


def _normalize_subset(subset: Iterable[str], patch_ids: Sequence[str]) -> Subset:
    raw = tuple(subset)
    if not raw:
        raise ValueError("patch subsets must be non-empty")
    if any(not isinstance(patch_id, str) or not patch_id for patch_id in raw):
        raise ValueError("patch subsets must contain non-empty strings")
    if len(set(raw)) != len(raw):
        raise ValueError("patch subsets must not contain duplicate ids")
    unknown = set(raw) - set(patch_ids)
    if unknown:
        raise ValueError(f"patch subset contains unknown ids: {sorted(unknown)!r}")
    return tuple(patch_id for patch_id in patch_ids if patch_id in set(raw))


def _json_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_json_tuple(item) for item in value)
    return value


def _random_state(rng: random.Random) -> list[Any]:
    """Return ``random.Random`` state using JSON-compatible lists."""

    def as_list(value: Any) -> Any:
        if isinstance(value, tuple):
            return [as_list(item) for item in value]
        return value

    state = as_list(rng.getstate())
    if not isinstance(state, list):  # pragma: no cover - random state is a tuple.
        raise SearchStrategyError("unexpected random generator state")
    return state


def _restore_random(seed: int, state: Any) -> random.Random:
    if not isinstance(state, list):
        raise ValueError("rng_state must be a JSON array")
    rng = random.Random(seed)
    try:
        rng.setstate(_json_tuple(state))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid rng_state") from exc
    return rng


def _subset_rows(values: Iterable[Subset]) -> list[list[str]]:
    return [list(value) for value in values]


def _read_subsets(value: Any, patch_ids: Sequence[str], name: str) -> tuple[Subset, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if any(not isinstance(item, list) for item in value):
        raise ValueError(f"{name} entries must be arrays")
    result = tuple(_normalize_subset(item, patch_ids) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique subsets")
    return result


@dataclass(slots=True)
class BernoulliUniqueStream:
    """Seeded Bernoulli stream over unique, non-empty patch subsets.

    Empty or already-seen draws are rejected.  The duplicate cap counts all
    consecutive rejected draws, including the forbidden empty subset; this
    gives a finite failure mode even for very small inclusion probabilities.
    """

    patch_ids: tuple[str, ...]
    seed: int
    inclusion_probability: float = 0.5
    maximum_consecutive_duplicates: int = 1_000
    _seen: set[Subset] = field(default_factory=set, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    STATE_TYPE = "bernoulli_unique_stream"
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        self.patch_ids = _patch_universe(self.patch_ids)
        self.seed = _strict_int(self.seed, "seed")
        self.inclusion_probability = _strict_float(
            self.inclusion_probability,
            "inclusion_probability",
            minimum=0.0,
            maximum=1.0,
        )
        if self.inclusion_probability <= 0.0:
            raise ValueError("inclusion_probability must be greater than zero")
        self.maximum_consecutive_duplicates = _strict_int(
            self.maximum_consecutive_duplicates,
            "maximum_consecutive_duplicates",
            minimum=1,
        )
        self._rng = random.Random(self.seed)

    @property
    def seen(self) -> tuple[Subset, ...]:
        return tuple(sorted(self._seen))

    @property
    def remaining(self) -> int:
        return (1 << len(self.patch_ids)) - 1 - len(self._seen)

    def ask(self, count: int) -> tuple[Subset, ...]:
        requested = _strict_int(count, "count", minimum=0)
        if requested == 0:
            return ()
        if requested > self.remaining:
            raise SearchSpaceExhausted(
                f"requested {requested} unique subsets with only {self.remaining} remaining"
            )
        original_rng_state = self._rng.getstate()
        candidate_seen = set(self._seen)
        proposals: list[Subset] = []
        rejected = 0
        try:
            while len(proposals) < requested:
                subset = tuple(
                    patch_id
                    for patch_id in self.patch_ids
                    if self._rng.random() < self.inclusion_probability
                )
                if not subset or subset in candidate_seen:
                    rejected += 1
                    if rejected >= self.maximum_consecutive_duplicates:
                        raise SearchSpaceExhausted(
                            "maximum consecutive duplicate/empty Bernoulli draws reached"
                        )
                    continue
                candidate_seen.add(subset)
                proposals.append(subset)
                rejected = 0
        except SearchSpaceExhausted:
            self._rng.setstate(original_rng_state)
            raise
        self._seen = candidate_seen
        return tuple(proposals)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "patch_ids": list(self.patch_ids),
            "seed": self.seed,
            "inclusion_probability": self.inclusion_probability,
            "maximum_consecutive_duplicates": self.maximum_consecutive_duplicates,
            "seen": _subset_rows(sorted(self._seen)),
            "rng_state": _random_state(self._rng),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> BernoulliUniqueStream:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported Bernoulli stream state")
        stream = cls(
            patch_ids=tuple(state["patch_ids"]),
            seed=state["seed"],
            inclusion_probability=state["inclusion_probability"],
            maximum_consecutive_duplicates=state["maximum_consecutive_duplicates"],
        )
        stream._seen = set(_read_subsets(state["seen"], stream.patch_ids, "seen"))
        stream._rng = _restore_random(stream.seed, state["rng_state"])
        return stream


@dataclass(slots=True)
class CommonCandidateStream:
    """Finite, immutable candidate order that can be forked across methods."""

    patch_ids: tuple[str, ...]
    candidates: tuple[Subset, ...]
    seed: int
    _cursor: int = field(default=0, init=False, repr=False)

    STATE_TYPE = "common_candidate_stream"
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        self.patch_ids = _patch_universe(self.patch_ids)
        self.seed = _strict_int(self.seed, "seed")
        normalized = tuple(
            _normalize_subset(candidate, self.patch_ids) for candidate in self.candidates
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("common candidate stream must contain unique subsets")
        self.candidates = normalized

    @classmethod
    def from_bernoulli(
        cls,
        patch_ids: Iterable[str],
        *,
        candidate_count: int,
        seed: int,
        inclusion_probability: float = 0.5,
        maximum_consecutive_duplicates: int = 1_000,
    ) -> CommonCandidateStream:
        count = _strict_int(candidate_count, "candidate_count", minimum=0)
        universe = _patch_universe(patch_ids)
        source = BernoulliUniqueStream(
            universe,
            seed,
            inclusion_probability,
            maximum_consecutive_duplicates,
        )
        return cls(universe, source.ask(count), seed)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def remaining(self) -> int:
        return len(self.candidates) - self._cursor

    def ask(self, count: int) -> tuple[Subset, ...]:
        requested = _strict_int(count, "count", minimum=0)
        end = min(len(self.candidates), self._cursor + requested)
        result = self.candidates[self._cursor : end]
        self._cursor = end
        return result

    def fork(self, *, preserve_cursor: bool = False) -> CommonCandidateStream:
        forked = CommonCandidateStream(self.patch_ids, self.candidates, self.seed)
        if preserve_cursor:
            forked._cursor = self._cursor
        return forked

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "patch_ids": list(self.patch_ids),
            "seed": self.seed,
            "candidates": _subset_rows(self.candidates),
            "cursor": self._cursor,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> CommonCandidateStream:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported common stream state")
        stream = cls(
            patch_ids=tuple(state["patch_ids"]),
            candidates=tuple(tuple(item) for item in state["candidates"]),
            seed=state["seed"],
        )
        cursor = _strict_int(state["cursor"], "cursor", minimum=0)
        if cursor > len(stream.candidates):
            raise ValueError("common stream cursor exceeds candidate count")
        stream._cursor = cursor
        return stream


def _scored_to_dict(candidate: ScoredCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "patch_ids": list(candidate.patch_ids),
        "objectives": {
            "id_accuracy": candidate.objectives.id_accuracy,
            "worst_target_transfer": candidate.objectives.worst_target_transfer,
            "token_cost": candidate.objectives.token_cost,
            "paired_regression": candidate.objectives.paired_regression,
        },
        "metadata": thaw_json(candidate.metadata),
        "summary": candidate.summary.to_dict() if candidate.summary is not None else None,
        "content_hash": candidate.content_hash,
    }


def _scored_from_dict(value: Mapping[str, Any], patch_ids: Sequence[str]) -> ScoredCandidate:
    raw_objectives = value.get("objectives")
    if not isinstance(raw_objectives, Mapping):
        raise ValueError("serialized candidate objectives must be an object")
    raw_summary = value.get("summary")
    if raw_summary is not None and not isinstance(raw_summary, Mapping):
        raise ValueError("serialized candidate summary must be an object or null")
    return ScoredCandidate(
        candidate_id=value["candidate_id"],
        patch_ids=_normalize_subset(value["patch_ids"], patch_ids),
        objectives=PointObjectives(
            id_accuracy=raw_objectives["id_accuracy"],
            worst_target_transfer=raw_objectives["worst_target_transfer"],
            token_cost=raw_objectives["token_cost"],
            paired_regression=raw_objectives["paired_regression"],
        ),
        metadata=value.get("metadata", {}),
        summary=(
            ObjectiveSummary.from_dict(raw_summary)
            if isinstance(raw_summary, Mapping)
            else None
        ),
        content_hash=value.get("content_hash"),
    )


def _validate_feedback(
    scored: Iterable[ScoredCandidate],
    pending: Sequence[Subset],
    patch_ids: Sequence[str],
) -> tuple[ScoredCandidate, ...]:
    values = tuple(scored)
    if not pending:
        raise SearchStrategyError("tell called without a pending ask batch")
    if len(values) != len(pending):
        raise SearchStrategyError(
            f"tell requires exactly {len(pending)} scored candidates; received {len(values)}"
        )
    if any(not isinstance(value, ScoredCandidate) for value in values):
        raise ValueError("tell requires ScoredCandidate values")
    normalized = tuple(_normalize_subset(value.patch_ids, patch_ids) for value in values)
    if len(set(normalized)) != len(normalized):
        raise SearchStrategyError("tell feedback contains duplicate patch subsets")
    if set(normalized) != set(pending):
        raise SearchStrategyError("tell feedback does not match the pending ask batch")
    if len({value.candidate_id for value in values}) != len(values):
        raise SearchStrategyError("tell feedback contains duplicate candidate ids")
    by_subset = dict(zip(normalized, values, strict=True))
    return tuple(by_subset[subset] for subset in pending)


@runtime_checkable
class BinarySubsetBayesianAdapter(Protocol):
    """Protocol for an externally supplied binary Bayesian optimizer.

    The adapter is responsible for its surrogate and acquisition function.  It
    must be deterministic under ``seed`` and return unique, non-empty subsets.
    """

    adapter_id: str

    def ask(
        self,
        *,
        patch_ids: tuple[str, ...],
        count: int,
        seed: int,
        seen_subsets: tuple[Subset, ...],
        observations: tuple[ScoredCandidate, ...],
    ) -> tuple[Subset, ...]: ...

    def tell(self, scored: tuple[ScoredCandidate, ...]) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(slots=True)
class InitialDesignBinarySubsetController:
    """Seeded Bernoulli initial design only; explicitly not Bayesian/GP search."""

    patch_ids: tuple[str, ...]
    seed: int
    initial_design_size: int = 20
    inclusion_probability: float = 0.5
    maximum_consecutive_duplicates: int = 1_000
    _stream: BernoulliUniqueStream = field(init=False, repr=False)
    _pending: tuple[Subset, ...] = field(default=(), init=False, repr=False)
    _observations: list[ScoredCandidate] = field(default_factory=list, init=False, repr=False)

    STATE_TYPE = "binary_subset_initial_design_only"
    STATE_VERSION = 1
    optimizer_kind = "seeded_initial_design_only_not_gp"

    def __post_init__(self) -> None:
        self.patch_ids = _patch_universe(self.patch_ids)
        self.seed = _strict_int(self.seed, "seed")
        self.initial_design_size = _strict_int(
            self.initial_design_size, "initial_design_size", minimum=1
        )
        if self.initial_design_size > (1 << len(self.patch_ids)) - 1:
            raise ValueError("initial_design_size exceeds the non-empty subset space")
        self._stream = BernoulliUniqueStream(
            self.patch_ids,
            self.seed,
            self.inclusion_probability,
            self.maximum_consecutive_duplicates,
        )

    @property
    def observations(self) -> tuple[ScoredCandidate, ...]:
        return tuple(self._observations)

    def ask(self, count: int = 1) -> tuple[Subset, ...]:
        requested = _strict_int(count, "count", minimum=0)
        if self._pending:
            raise SearchStrategyError("tell must resolve the pending batch before ask")
        remaining = self.initial_design_size - len(self._observations)
        if requested > remaining:
            raise ExternalOptimizerRequired(
                "the built-in binary subset fallback stops after its initial design; "
                "supply a BinarySubsetBayesianAdapter for model-based proposals"
            )
        self._pending = self._stream.ask(requested)
        return self._pending

    def tell(self, scored: Iterable[ScoredCandidate]) -> None:
        values = _validate_feedback(scored, self._pending, self.patch_ids)
        self._observations.extend(values)
        self._pending = ()

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "patch_ids": list(self.patch_ids),
            "seed": self.seed,
            "initial_design_size": self.initial_design_size,
            "inclusion_probability": self.inclusion_probability,
            "maximum_consecutive_duplicates": self.maximum_consecutive_duplicates,
            "stream": self._stream.state_dict(),
            "pending": _subset_rows(self._pending),
            "observations": [_scored_to_dict(value) for value in self._observations],
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any]
    ) -> InitialDesignBinarySubsetController:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported initial-design controller state")
        controller = cls(
            patch_ids=tuple(state["patch_ids"]),
            seed=state["seed"],
            initial_design_size=state["initial_design_size"],
            inclusion_probability=state["inclusion_probability"],
            maximum_consecutive_duplicates=state["maximum_consecutive_duplicates"],
        )
        raw_stream = state.get("stream")
        if not isinstance(raw_stream, Mapping):
            raise ValueError("stream state must be an object")
        controller._stream = BernoulliUniqueStream.from_state(raw_stream)
        if (
            controller._stream.patch_ids != controller.patch_ids
            or controller._stream.seed != controller.seed
            or controller._stream.inclusion_probability
            != controller.inclusion_probability
            or controller._stream.maximum_consecutive_duplicates
            != controller.maximum_consecutive_duplicates
        ):
            raise ValueError("nested stream configuration mismatch")
        controller._pending = _read_subsets(state["pending"], controller.patch_ids, "pending")
        raw_observations = state.get("observations")
        if not isinstance(raw_observations, list):
            raise ValueError("observations must be an array")
        controller._observations = [
            _scored_from_dict(value, controller.patch_ids) for value in raw_observations
        ]
        if set(controller._pending) & {
            tuple(value.patch_ids) for value in controller._observations
        }:
            raise ValueError("pending and observed subsets overlap")
        accounted_for = {
            *(tuple(value.patch_ids) for value in controller._observations),
            *controller._pending,
        }
        if accounted_for != set(controller._stream.seen):
            raise ValueError("initial-design stream and feedback state are inconsistent")
        if (
            len(controller._observations) + len(controller._pending)
            > controller.initial_design_size
        ):
            raise ValueError("initial-design state exceeds configured size")
        return controller


@dataclass(slots=True)
class AdapterBackedBinarySubsetController:
    """Strict ask/tell wrapper around a caller-provided Bayesian adapter."""

    patch_ids: tuple[str, ...]
    seed: int
    adapter: BinarySubsetBayesianAdapter
    _pending: tuple[Subset, ...] = field(default=(), init=False, repr=False)
    _seen: set[Subset] = field(default_factory=set, init=False, repr=False)
    _observations: list[ScoredCandidate] = field(default_factory=list, init=False, repr=False)
    _ask_index: int = field(default=0, init=False, repr=False)

    STATE_TYPE = "adapter_backed_binary_subset"
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        self.patch_ids = _patch_universe(self.patch_ids)
        self.seed = _strict_int(self.seed, "seed")
        if not isinstance(self.adapter, BinarySubsetBayesianAdapter):
            raise ValueError("adapter does not implement BinarySubsetBayesianAdapter")
        if not isinstance(self.adapter.adapter_id, str) or not self.adapter.adapter_id:
            raise ValueError("adapter_id must be a non-empty string")

    @property
    def observations(self) -> tuple[ScoredCandidate, ...]:
        return tuple(self._observations)

    def ask(self, count: int = 1) -> tuple[Subset, ...]:
        requested = _strict_int(count, "count", minimum=0)
        if self._pending:
            raise SearchStrategyError("tell must resolve the pending batch before ask")
        if requested > (1 << len(self.patch_ids)) - 1 - len(self._seen):
            raise SearchSpaceExhausted("requested adapter batch exceeds remaining subset space")
        proposal_seed = self.seed ^ self._ask_index
        raw = self.adapter.ask(
            patch_ids=self.patch_ids,
            count=requested,
            seed=proposal_seed,
            seen_subsets=tuple(sorted(self._seen)),
            observations=tuple(self._observations),
        )
        if not isinstance(raw, tuple) or len(raw) != requested:
            raise SearchStrategyError("external adapter returned the wrong proposal count/type")
        proposals = tuple(_normalize_subset(value, self.patch_ids) for value in raw)
        if len(set(proposals)) != len(proposals) or set(proposals) & self._seen:
            raise SearchStrategyError("external adapter returned duplicate/seen proposals")
        self._pending = proposals
        self._seen.update(proposals)
        self._ask_index += 1
        return proposals

    def tell(self, scored: Iterable[ScoredCandidate]) -> None:
        values = _validate_feedback(scored, self._pending, self.patch_ids)
        self.adapter.tell(values)
        self._observations.extend(values)
        self._pending = ()

    def state_dict(self) -> dict[str, Any]:
        adapter_state = self.adapter.state_dict()
        if not isinstance(adapter_state, Mapping):
            raise SearchStrategyError("external adapter state_dict must return a mapping")
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "patch_ids": list(self.patch_ids),
            "seed": self.seed,
            "adapter_id": self.adapter.adapter_id,
            "adapter_state": thaw_json(freeze_mapping(adapter_state)),
            "pending": _subset_rows(self._pending),
            "seen": _subset_rows(sorted(self._seen)),
            "observations": [_scored_to_dict(value) for value in self._observations],
            "ask_index": self._ask_index,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        adapter: BinarySubsetBayesianAdapter,
    ) -> AdapterBackedBinarySubsetController:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported adapter-backed controller state")
        if state.get("adapter_id") != adapter.adapter_id:
            raise ValueError("adapter id does not match serialized controller state")
        controller = cls(tuple(state["patch_ids"]), state["seed"], adapter)
        controller._pending = _read_subsets(state["pending"], controller.patch_ids, "pending")
        controller._seen = set(_read_subsets(state["seen"], controller.patch_ids, "seen"))
        if not set(controller._pending) <= controller._seen:
            raise ValueError("pending proposals must appear in seen subsets")
        raw_observations = state.get("observations")
        if not isinstance(raw_observations, list):
            raise ValueError("observations must be an array")
        controller._observations = [
            _scored_from_dict(value, controller.patch_ids) for value in raw_observations
        ]
        if len({value.candidate_id for value in controller._observations}) != len(
            controller._observations
        ):
            raise ValueError("observations contain duplicate candidate ids")
        if not {tuple(value.patch_ids) for value in controller._observations} <= controller._seen:
            raise ValueError("observed candidates must appear in seen subsets")
        controller._ask_index = _strict_int(state["ask_index"], "ask_index", minimum=0)
        adapter_state = state.get("adapter_state")
        if not isinstance(adapter_state, Mapping):
            raise ValueError("adapter_state must be an object")
        adapter.load_state_dict(adapter_state)
        return controller


def make_binary_subset_controller(
    patch_ids: Iterable[str],
    *,
    seed: int,
    adapter: BinarySubsetBayesianAdapter | None = None,
    mode: Literal["bayesian", "initial_design_only"] = "bayesian",
    initial_design_size: int = 20,
    inclusion_probability: float = 0.5,
    maximum_consecutive_duplicates: int = 1_000,
) -> AdapterBackedBinarySubsetController | InitialDesignBinarySubsetController:
    """Construct a binary subset controller without mislabelling the fallback.

    ``mode='bayesian'`` always requires an external adapter.  The only builtin
    option is the explicitly named ``initial_design_only`` mode.
    """

    universe = _patch_universe(patch_ids)
    if mode == "bayesian":
        if adapter is None:
            raise ExternalOptimizerRequired(
                "binary Bayesian search requires an external "
                "BinarySubsetBayesianAdapter; no GP implementation is bundled"
            )
        return AdapterBackedBinarySubsetController(universe, seed, adapter)
    if mode != "initial_design_only":
        raise ValueError(f"unknown binary subset controller mode: {mode!r}")
    if adapter is not None:
        raise ValueError("initial_design_only mode does not accept an optimizer adapter")
    return InitialDesignBinarySubsetController(
        universe,
        seed,
        initial_design_size,
        inclusion_probability,
        maximum_consecutive_duplicates,
    )


def _rank_and_crowding(
    candidates: Sequence[ScoredCandidate],
) -> tuple[dict[str, int], dict[str, float]]:
    fronts = nondominated_fronts(tuple(candidates))
    ranks: dict[str, int] = {}
    distances: dict[str, float] = {}
    for rank, front in enumerate(fronts):
        vectors = {candidate.candidate_id: candidate.vector for candidate in front}
        front_distances = crowding_distance(vectors)
        for candidate in front:
            ranks[candidate.candidate_id] = rank
            distances[candidate.candidate_id] = front_distances[candidate.candidate_id]
    return ranks, distances


@dataclass(slots=True)
class NSGAIIController:
    """Binary NSGA-II ask/tell controller using rank and crowding tournaments."""

    patch_ids: tuple[str, ...]
    seed: int
    population_size: int = 20
    offspring_size: int = 20
    crossover_probability: float = 0.9
    per_locus_parent_probability: float = 0.5
    mutation_probability: float | None = None
    maximum_consecutive_duplicates: int = 1_000
    initial_stream: CommonCandidateStream | None = None
    _rng: random.Random = field(init=False, repr=False)
    _initializer: BernoulliUniqueStream | CommonCandidateStream = field(
        init=False, repr=False
    )
    _population: list[ScoredCandidate] = field(default_factory=list, init=False, repr=False)
    _pending: tuple[Subset, ...] = field(default=(), init=False, repr=False)
    _seen: set[Subset] = field(default_factory=set, init=False, repr=False)
    _generation: int = field(default=0, init=False, repr=False)

    STATE_TYPE = "nsga2_subset_controller"
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        self.patch_ids = _patch_universe(self.patch_ids)
        self.seed = _strict_int(self.seed, "seed")
        self.population_size = _strict_int(
            self.population_size, "population_size", minimum=2
        )
        self.offspring_size = _strict_int(
            self.offspring_size, "offspring_size", minimum=1
        )
        self.crossover_probability = _strict_float(
            self.crossover_probability,
            "crossover_probability",
            minimum=0.0,
            maximum=1.0,
        )
        self.per_locus_parent_probability = _strict_float(
            self.per_locus_parent_probability,
            "per_locus_parent_probability",
            minimum=0.0,
            maximum=1.0,
        )
        if self.mutation_probability is None:
            self.mutation_probability = 1.0 / len(self.patch_ids)
        self.mutation_probability = _strict_float(
            self.mutation_probability,
            "mutation_probability",
            minimum=0.0,
            maximum=1.0,
        )
        self.maximum_consecutive_duplicates = _strict_int(
            self.maximum_consecutive_duplicates,
            "maximum_consecutive_duplicates",
            minimum=1,
        )
        space = (1 << len(self.patch_ids)) - 1
        if self.population_size > space:
            raise ValueError("population_size exceeds the non-empty subset space")
        self._rng = random.Random(self.seed ^ 0x4E53474132)
        if self.initial_stream is not None:
            if self.initial_stream.patch_ids != self.patch_ids:
                raise ValueError("initial stream patch universe does not match NSGA-II")
            self._initializer = self.initial_stream.fork()
        else:
            self._initializer = BernoulliUniqueStream(
                self.patch_ids,
                self.seed,
                0.5,
                self.maximum_consecutive_duplicates,
            )

    @property
    def population(self) -> tuple[ScoredCandidate, ...]:
        return tuple(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    def _initial_ask(self, count: int) -> tuple[Subset, ...]:
        if self._initializer.remaining < count:
            raise SearchSpaceExhausted(
                "initial candidate stream ended before population could be filled"
            )
        proposals = self._initializer.ask(count)
        if len(proposals) != count:
            raise SearchSpaceExhausted("initial candidate stream ended before population filled")
        self._seen.update(proposals)
        return proposals

    def _tournament(self) -> ScoredCandidate:
        ranks, distances = _rank_and_crowding(self._population)
        left = self._population[self._rng.randrange(len(self._population))]
        right = self._population[self._rng.randrange(len(self._population))]

        def key(value: ScoredCandidate) -> tuple[float, float, str]:
            return (
                float(ranks[value.candidate_id]),
                -distances[value.candidate_id],
                value.content_hash or value.candidate_id,
            )

        return min((left, right), key=key)

    def _offspring(self) -> Subset:
        parent_a = set(self._tournament().patch_ids)
        parent_b = set(self._tournament().patch_ids)
        if self._rng.random() < self.crossover_probability:
            selected = {
                patch_id
                for patch_id in self.patch_ids
                if (
                    patch_id in parent_a
                    if self._rng.random() < self.per_locus_parent_probability
                    else patch_id in parent_b
                )
            }
        else:
            selected = set(parent_a)
        for patch_id in self.patch_ids:
            if self._rng.random() < self.mutation_probability:
                if patch_id in selected:
                    selected.remove(patch_id)
                else:
                    selected.add(patch_id)
        if not selected:
            selected.add(self.patch_ids[self._rng.randrange(len(self.patch_ids))])
        return tuple(patch_id for patch_id in self.patch_ids if patch_id in selected)

    def ask(self, count: int | None = None) -> tuple[Subset, ...]:
        if self._pending:
            raise SearchStrategyError("tell must resolve the pending batch before ask")
        default_count = self.population_size if not self._population else self.offspring_size
        requested = default_count if count is None else _strict_int(count, "count", minimum=0)
        if not self._population and requested != self.population_size:
            raise SearchStrategyError(
                "the initial NSGA-II ask must request exactly population_size proposals"
            )
        remaining = (1 << len(self.patch_ids)) - 1 - len(self._seen)
        if requested > remaining:
            raise SearchSpaceExhausted(
                f"requested {requested} NSGA-II proposals with only {remaining} remaining"
            )
        if not self._population:
            self._pending = self._initial_ask(requested)
            return self._pending
        original_rng_state = self._rng.getstate()
        proposals: list[Subset] = []
        rejected = 0
        try:
            while len(proposals) < requested:
                proposal = self._offspring()
                if proposal in self._seen or proposal in proposals:
                    rejected += 1
                    if rejected >= self.maximum_consecutive_duplicates:
                        raise SearchSpaceExhausted(
                            "maximum consecutive duplicate NSGA-II proposals reached"
                        )
                    continue
                proposals.append(proposal)
                rejected = 0
        except SearchSpaceExhausted:
            self._rng.setstate(original_rng_state)
            raise
        self._pending = tuple(proposals)
        self._seen.update(proposals)
        return self._pending

    def tell(self, scored: Iterable[ScoredCandidate]) -> None:
        values = _validate_feedback(scored, self._pending, self.patch_ids)
        if not self._population:
            self._population = list(values)
            if len(self._population) != self.population_size:
                raise SearchStrategyError(
                    "initial NSGA-II tell must fill the configured population"
                )
        else:
            existing_ids = {value.candidate_id for value in self._population}
            duplicate_ids = existing_ids & {value.candidate_id for value in values}
            if duplicate_ids:
                raise SearchStrategyError(
                    "NSGA-II offspring candidate ids overlap the parent population: "
                    f"{sorted(duplicate_ids)!r}"
                )
            combined = tuple(self._population) + values
            survivors: list[ScoredCandidate] = []
            for front in nondominated_fronts(combined):
                remaining = self.population_size - len(survivors)
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    survivors.extend(front)
                    continue
                distances = crowding_distance(
                    {candidate.candidate_id: candidate.vector for candidate in front}
                )
                survivors.extend(
                    sorted(
                        front,
                        key=lambda value: (
                            -distances[value.candidate_id],
                            value.content_hash or value.candidate_id,
                        ),
                    )[:remaining]
                )
            self._population = survivors
            self._generation += 1
        self._pending = ()

    def state_dict(self) -> dict[str, Any]:
        initializer_state = self._initializer.state_dict()
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "patch_ids": list(self.patch_ids),
            "seed": self.seed,
            "population_size": self.population_size,
            "offspring_size": self.offspring_size,
            "crossover_probability": self.crossover_probability,
            "per_locus_parent_probability": self.per_locus_parent_probability,
            "mutation_probability": self.mutation_probability,
            "maximum_consecutive_duplicates": self.maximum_consecutive_duplicates,
            "rng_state": _random_state(self._rng),
            "initializer": initializer_state,
            "population": [_scored_to_dict(value) for value in self._population],
            "pending": _subset_rows(self._pending),
            "seen": _subset_rows(sorted(self._seen)),
            "generation": self._generation,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> NSGAIIController:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported NSGA-II controller state")
        raw_initializer = state.get("initializer")
        if not isinstance(raw_initializer, Mapping):
            raise ValueError("initializer state must be an object")
        initializer_type = raw_initializer.get("state_type")
        initial_stream = (
            CommonCandidateStream.from_state(raw_initializer)
            if initializer_type == CommonCandidateStream.STATE_TYPE
            else None
        )
        controller = cls(
            patch_ids=tuple(state["patch_ids"]),
            seed=state["seed"],
            population_size=state["population_size"],
            offspring_size=state["offspring_size"],
            crossover_probability=state["crossover_probability"],
            per_locus_parent_probability=state["per_locus_parent_probability"],
            mutation_probability=state["mutation_probability"],
            maximum_consecutive_duplicates=state["maximum_consecutive_duplicates"],
            initial_stream=initial_stream,
        )
        if initializer_type == BernoulliUniqueStream.STATE_TYPE:
            controller._initializer = BernoulliUniqueStream.from_state(raw_initializer)
        elif initializer_type == CommonCandidateStream.STATE_TYPE:
            controller._initializer = CommonCandidateStream.from_state(raw_initializer)
        else:
            raise ValueError("unknown NSGA-II initializer state type")
        if controller._initializer.patch_ids != controller.patch_ids:
            raise ValueError("initializer patch universe mismatch")
        controller._rng = _restore_random(controller.seed ^ 0x4E53474132, state["rng_state"])
        raw_population = state.get("population")
        if not isinstance(raw_population, list):
            raise ValueError("population must be an array")
        controller._population = [
            _scored_from_dict(value, controller.patch_ids) for value in raw_population
        ]
        if len({value.candidate_id for value in controller._population}) != len(
            controller._population
        ):
            raise ValueError("restored NSGA-II population has duplicate candidate ids")
        if controller._population and len(controller._population) != controller.population_size:
            raise ValueError("restored NSGA-II population has the wrong size")
        controller._pending = _read_subsets(state["pending"], controller.patch_ids, "pending")
        controller._seen = set(_read_subsets(state["seen"], controller.patch_ids, "seen"))
        required_seen = {
            *(tuple(value.patch_ids) for value in controller._population),
            *controller._pending,
        }
        if not required_seen <= controller._seen:
            raise ValueError("NSGA-II population/pending subsets must appear in seen")
        initializer_seen = (
            set(controller._initializer.seen)
            if isinstance(controller._initializer, BernoulliUniqueStream)
            else set(controller._initializer.candidates[: controller._initializer.cursor])
        )
        if not initializer_seen <= controller._seen:
            raise ValueError("consumed NSGA-II initializer subsets must appear in seen")
        controller._generation = _strict_int(state["generation"], "generation", minimum=0)
        return controller


@dataclass(slots=True)
class EvoTopKController:
    """Consume a common stream and retain deterministic ID-accuracy top-k."""

    stream: CommonCandidateStream
    top_k: int = 10
    _pending: tuple[Subset, ...] = field(default=(), init=False, repr=False)
    _incumbents: list[ScoredCandidate] = field(default_factory=list, init=False, repr=False)
    _observed_subsets: set[Subset] = field(default_factory=set, init=False, repr=False)

    STATE_TYPE = "evo_top_k_controller"
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        if not isinstance(self.stream, CommonCandidateStream):
            raise ValueError("Evo top-k requires a CommonCandidateStream")
        self.top_k = _strict_int(self.top_k, "top_k", minimum=1)

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return self.stream.patch_ids

    @property
    def incumbents(self) -> tuple[ScoredCandidate, ...]:
        return tuple(self._incumbents)

    def ask(self, count: int = 1) -> tuple[Subset, ...]:
        if self._pending:
            raise SearchStrategyError("tell must resolve the pending batch before ask")
        self._pending = self.stream.ask(count)
        return self._pending

    @staticmethod
    def _selection_key(candidate: ScoredCandidate) -> tuple[float, str, str]:
        return (
            -candidate.objectives.id_accuracy,
            candidate.content_hash or candidate.candidate_id,
            candidate.candidate_id,
        )

    def tell(self, scored: Iterable[ScoredCandidate]) -> None:
        values = _validate_feedback(scored, self._pending, self.patch_ids)
        subsets = {tuple(value.patch_ids) for value in values}
        if subsets & self._observed_subsets:
            raise SearchStrategyError("Evo top-k received a previously observed subset")
        self._observed_subsets.update(subsets)
        self._incumbents = sorted(
            (*self._incumbents, *values), key=self._selection_key
        )[: self.top_k]
        self._pending = ()

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "stream": self.stream.state_dict(),
            "top_k": self.top_k,
            "pending": _subset_rows(self._pending),
            "incumbents": [_scored_to_dict(value) for value in self._incumbents],
            "observed_subsets": _subset_rows(sorted(self._observed_subsets)),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> EvoTopKController:
        if state.get("state_type") != cls.STATE_TYPE or state.get("state_version") != 1:
            raise ValueError("unsupported Evo top-k controller state")
        raw_stream = state.get("stream")
        if not isinstance(raw_stream, Mapping):
            raise ValueError("stream state must be an object")
        controller = cls(CommonCandidateStream.from_state(raw_stream), state["top_k"])
        controller._pending = _read_subsets(state["pending"], controller.patch_ids, "pending")
        raw_incumbents = state.get("incumbents")
        if not isinstance(raw_incumbents, list):
            raise ValueError("incumbents must be an array")
        controller._incumbents = [
            _scored_from_dict(value, controller.patch_ids) for value in raw_incumbents
        ]
        if len(controller._incumbents) > controller.top_k:
            raise ValueError("restored Evo incumbents exceed top_k")
        controller._observed_subsets = set(
            _read_subsets(
                state["observed_subsets"],
                controller.patch_ids,
                "observed_subsets",
            )
        )
        if not {tuple(value.patch_ids) for value in controller._incumbents} <= (
            controller._observed_subsets
        ):
            raise ValueError("Evo incumbents must appear in observed subsets")
        if set(controller._pending) & controller._observed_subsets:
            raise ValueError("Evo pending and observed subsets overlap")
        consumed = set(controller.stream.candidates[: controller.stream.cursor])
        if consumed != controller._observed_subsets | set(controller._pending):
            raise ValueError(
                "Evo stream cursor is inconsistent with observed and pending subsets"
            )
        return controller


def _mocha_plugin_to_dict(plugin: MOCHAPlugin) -> dict[str, Any]:
    """Serialize the immutable protocol parameters used by ``MOCHAPlugin``."""

    return {
        "weights": list(plugin.weights),
        "ranges": (
            [list(bounds) for bounds in plugin.ranges]
            if plugin.ranges is not None
            else None
        ),
        "plugin_id": plugin.plugin_id,
        "require_frozen_ranges": plugin.require_frozen_ranges,
        "dirichlet_alpha": plugin.dirichlet_alpha,
        "annealing_rate": plugin.annealing_rate,
        "reference_point": list(plugin.reference_point),
    }


def _mocha_plugin_from_dict(value: Mapping[str, Any]) -> MOCHAPlugin:
    raw_ranges = value.get("ranges")
    if raw_ranges is not None:
        if not isinstance(raw_ranges, list) or any(
            not isinstance(bounds, list) or len(bounds) != 2 for bounds in raw_ranges
        ):
            raise ValueError("MOCHA ranges must be null or an array of pairs")
        ranges = tuple(tuple(bounds) for bounds in raw_ranges)
    else:
        ranges = None
    raw_weights = value.get("weights")
    raw_reference = value.get("reference_point")
    if not isinstance(raw_weights, list):
        raise ValueError("MOCHA weights must be an array")
    if not isinstance(raw_reference, list):
        raise ValueError("MOCHA reference_point must be an array")
    require_frozen = value.get("require_frozen_ranges")
    if not isinstance(require_frozen, bool):
        raise ValueError("MOCHA require_frozen_ranges must be boolean")
    plugin_id = value.get("plugin_id")
    if not isinstance(plugin_id, str) or not plugin_id:
        raise ValueError("MOCHA plugin_id must be a non-empty string")
    return MOCHAPlugin(
        weights=tuple(raw_weights),  # type: ignore[arg-type]
        ranges=ranges,  # type: ignore[arg-type]
        plugin_id=plugin_id,
        require_frozen_ranges=require_frozen,
        dirichlet_alpha=value.get("dirichlet_alpha"),
        annealing_rate=value.get("annealing_rate"),
        reference_point=tuple(raw_reference),  # type: ignore[arg-type]
    )


@dataclass(slots=True)
class MOCHAController:
    """Run MOCHA acceptance over a matched ``CommonCandidateStream``.

    The common stream freezes which patch subsets are measured.  Before each
    ask, the existing :class:`MOCHAPlugin` performs seeded Dirichlet-Chebyshev
    parent selection over accepted incumbents.  During tell, the same plugin's
    HVC-to-Chebyshev annealing gate decides whether each measured proposal joins
    the incumbent set.  Parent choices are recorded for auditability; they do
    not alter the matched common-candidate order.

    Logical task executions are charged when feedback is accepted by ``tell``,
    including rejected MOCHA proposals.  A pending batch reserves its full
    charge, so an ask can never overrun the declared logical budget.
    """

    stream: CommonCandidateStream
    seed: int
    logical_task_execution_budget: int
    task_executions_per_candidate: int = 1
    plugin: MOCHAPlugin = field(default_factory=MOCHAPlugin)
    _pending: tuple[Subset, ...] = field(default=(), init=False, repr=False)
    _pending_parent_ids: tuple[str | None, ...] = field(
        default=(), init=False, repr=False
    )
    _pending_parent_seeds: tuple[int | None, ...] = field(
        default=(), init=False, repr=False
    )
    _incumbents: list[ScoredCandidate] = field(
        default_factory=list, init=False, repr=False
    )
    _accepted_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _observed_subsets: set[Subset] = field(default_factory=set, init=False, repr=False)
    _decisions: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _logical_task_executions_spent: int = field(default=0, init=False, repr=False)

    STATE_TYPE = "mocha_subset_controller"
    STATE_VERSION = 2
    _PARENT_SEED_SALT = 0x504152454E54
    _ACCEPTANCE_SEED_SALT = 0x414343455054

    def __post_init__(self) -> None:
        if not isinstance(self.stream, CommonCandidateStream):
            raise ValueError("MOCHA requires a CommonCandidateStream")
        self.seed = _strict_int(self.seed, "seed")
        self.logical_task_execution_budget = _strict_int(
            self.logical_task_execution_budget,
            "logical_task_execution_budget",
            minimum=1,
        )
        self.task_executions_per_candidate = _strict_int(
            self.task_executions_per_candidate,
            "task_executions_per_candidate",
            minimum=1,
        )
        if self.task_executions_per_candidate > self.logical_task_execution_budget:
            raise ValueError(
                "task_executions_per_candidate exceeds the logical task budget"
            )
        if not isinstance(self.plugin, MOCHAPlugin):
            raise ValueError("plugin must be a MOCHAPlugin")

    @property
    def patch_ids(self) -> tuple[str, ...]:
        return self.stream.patch_ids

    @property
    def incumbents(self) -> tuple[ScoredCandidate, ...]:
        return tuple(self._incumbents)

    @property
    def accepted_ids(self) -> tuple[str, ...]:
        return tuple(self._accepted_ids)

    @property
    def logical_task_executions_spent(self) -> int:
        return self._logical_task_executions_spent

    @property
    def logical_budget_progress(self) -> float:
        return (
            self._logical_task_executions_spent
            / self.logical_task_execution_budget
        )

    @property
    def pending_parent_ids(self) -> tuple[str | None, ...]:
        return self._pending_parent_ids

    @property
    def pending_parent_seeds(self) -> tuple[int | None, ...]:
        return self._pending_parent_seeds

    @property
    def decision_log(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(freeze_mapping(decision) for decision in self._decisions)

    def _parent_seed(self, proposal_index: int) -> int:
        return self.seed ^ self._PARENT_SEED_SALT ^ proposal_index

    def _acceptance_seed(self, proposal_index: int) -> int:
        return self.seed ^ self._ACCEPTANCE_SEED_SALT ^ proposal_index

    def ask(self, count: int = 1) -> tuple[Subset, ...]:
        requested = _strict_int(count, "count", minimum=0)
        if self._pending:
            raise SearchStrategyError("tell must resolve the pending batch before ask")
        remaining_budget = (
            self.logical_task_execution_budget
            - self._logical_task_executions_spent
        )
        affordable = remaining_budget // self.task_executions_per_candidate
        if requested > affordable:
            raise SearchStrategyError(
                f"requested {requested} MOCHA proposals with logical budget for "
                f"only {affordable}"
            )
        pending = self.stream.ask(requested)
        parent_ids: list[str | None] = []
        parent_seeds: list[int | None] = []
        first_index = len(self._decisions)
        for offset, _subset in enumerate(pending):
            if not self._incumbents:
                parent_ids.append(None)
                parent_seeds.append(None)
                continue
            parent_seed = self._parent_seed(first_index + offset)
            parent = self.plugin.select_parent(
                tuple(self._incumbents), seed=parent_seed
            )
            parent_ids.append(parent.candidate_id)
            parent_seeds.append(parent_seed)
        self._pending = pending
        self._pending_parent_ids = tuple(parent_ids)
        self._pending_parent_seeds = tuple(parent_seeds)
        return pending

    def tell(self, scored: Iterable[ScoredCandidate]) -> None:
        values = _validate_feedback(scored, self._pending, self.patch_ids)
        subsets = {tuple(value.patch_ids) for value in values}
        if subsets & self._observed_subsets:
            raise SearchStrategyError("MOCHA received a previously observed subset")
        previous_ids = {
            decision["candidate_id"] for decision in self._decisions
        }
        candidate_ids = {value.candidate_id for value in values}
        if candidate_ids & previous_ids:
            raise SearchStrategyError("MOCHA received a previously observed candidate id")

        incumbents = list(self._incumbents)
        accepted_ids = list(self._accepted_ids)
        decisions = list(self._decisions)
        spent = self._logical_task_executions_spent
        for offset, proposal in enumerate(values):
            proposal_index = len(decisions)
            spent_before = spent
            materialization_valid = proposal.metadata.get(
                "materialization_valid", True
            )
            if not isinstance(materialization_valid, bool):
                raise SearchStrategyError(
                    "MOCHA materialization_valid feedback must be boolean"
                )
            raw_evaluation_cost = proposal.metadata.get(
                "evaluation_cost",
                self.task_executions_per_candidate if materialization_valid else None,
            )
            if (
                isinstance(raw_evaluation_cost, bool)
                or not isinstance(raw_evaluation_cost, int)
            ):
                raise SearchStrategyError(
                    "MOCHA feedback evaluation_cost must be an integer"
                )
            expected_cost = (
                self.task_executions_per_candidate if materialization_valid else 0
            )
            if raw_evaluation_cost != expected_cost:
                raise SearchStrategyError(
                    "MOCHA feedback evaluation_cost does not match materialization status"
                )
            evaluation_cost = raw_evaluation_cost
            spent += evaluation_cost
            acceptance_seed = (
                self._acceptance_seed(proposal_index)
                if materialization_valid
                else None
            )
            accepted = (
                self.plugin.accept_proposal(
                    proposal,
                    tuple(incumbents),
                    task_executions_spent=spent,
                    task_execution_budget=self.logical_task_execution_budget,
                    seed=acceptance_seed,
                )
                if acceptance_seed is not None
                else False
            )
            if accepted:
                incumbents.append(proposal)
                accepted_ids.append(proposal.candidate_id)
            decisions.append(
                {
                    "proposal_index": proposal_index,
                    "candidate_id": proposal.candidate_id,
                    "patch_ids": list(proposal.patch_ids),
                    "parent_id": self._pending_parent_ids[offset],
                    "parent_seed": self._pending_parent_seeds[offset],
                    "materialization_valid": materialization_valid,
                    "evaluation_cost": evaluation_cost,
                    "acceptance_seed": acceptance_seed,
                    "accepted": accepted,
                    "logical_task_executions_before": spent_before,
                    "logical_task_executions_after": spent,
                    "logical_budget_progress": (
                        spent / self.logical_task_execution_budget
                    ),
                }
            )

        self._incumbents = incumbents
        self._accepted_ids = accepted_ids
        self._decisions = decisions
        self._logical_task_executions_spent = spent
        self._observed_subsets.update(subsets)
        self._pending = ()
        self._pending_parent_ids = ()
        self._pending_parent_seeds = ()

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_type": self.STATE_TYPE,
            "state_version": self.STATE_VERSION,
            "stream": self.stream.state_dict(),
            "seed": self.seed,
            "logical_task_execution_budget": self.logical_task_execution_budget,
            "task_executions_per_candidate": self.task_executions_per_candidate,
            "plugin": _mocha_plugin_to_dict(self.plugin),
            "pending": _subset_rows(self._pending),
            "pending_parent_ids": list(self._pending_parent_ids),
            "pending_parent_seeds": list(self._pending_parent_seeds),
            "incumbents": [_scored_to_dict(value) for value in self._incumbents],
            "accepted_ids": list(self._accepted_ids),
            "observed_subsets": _subset_rows(sorted(self._observed_subsets)),
            "decisions": thaw_json(self._decisions),
            "logical_task_executions_spent": self._logical_task_executions_spent,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> MOCHAController:
        if (
            state.get("state_type") != cls.STATE_TYPE
            or state.get("state_version") != cls.STATE_VERSION
        ):
            raise ValueError("unsupported MOCHA controller state")
        raw_stream = state.get("stream")
        raw_plugin = state.get("plugin")
        if not isinstance(raw_stream, Mapping):
            raise ValueError("stream state must be an object")
        if not isinstance(raw_plugin, Mapping):
            raise ValueError("MOCHA plugin state must be an object")
        controller = cls(
            CommonCandidateStream.from_state(raw_stream),
            state["seed"],
            state["logical_task_execution_budget"],
            state["task_executions_per_candidate"],
            _mocha_plugin_from_dict(raw_plugin),
        )
        controller._pending = _read_subsets(
            state["pending"], controller.patch_ids, "pending"
        )
        raw_parent_ids = state.get("pending_parent_ids")
        raw_parent_seeds = state.get("pending_parent_seeds")
        if not isinstance(raw_parent_ids, list) or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in raw_parent_ids
        ):
            raise ValueError("pending_parent_ids must contain strings or null")
        if not isinstance(raw_parent_seeds, list) or any(
            value is not None and (isinstance(value, bool) or not isinstance(value, int))
            for value in raw_parent_seeds
        ):
            raise ValueError("pending_parent_seeds must contain integers or null")
        if not (
            len(controller._pending)
            == len(raw_parent_ids)
            == len(raw_parent_seeds)
        ):
            raise ValueError("MOCHA pending parent metadata length mismatch")
        controller._pending_parent_ids = tuple(raw_parent_ids)
        controller._pending_parent_seeds = tuple(raw_parent_seeds)

        raw_incumbents = state.get("incumbents")
        if not isinstance(raw_incumbents, list):
            raise ValueError("incumbents must be an array")
        controller._incumbents = [
            _scored_from_dict(value, controller.patch_ids)
            for value in raw_incumbents
        ]
        if len({value.candidate_id for value in controller._incumbents}) != len(
            controller._incumbents
        ):
            raise ValueError("restored MOCHA incumbents have duplicate candidate ids")
        raw_accepted_ids = state.get("accepted_ids")
        if not isinstance(raw_accepted_ids, list) or any(
            not isinstance(value, str) or not value for value in raw_accepted_ids
        ):
            raise ValueError("accepted_ids must contain non-empty strings")
        controller._accepted_ids = list(raw_accepted_ids)
        if controller._accepted_ids != [
            value.candidate_id for value in controller._incumbents
        ]:
            raise ValueError("accepted_ids must match MOCHA incumbents in order")
        controller._observed_subsets = set(
            _read_subsets(
                state["observed_subsets"],
                controller.patch_ids,
                "observed_subsets",
            )
        )

        raw_decisions = state.get("decisions")
        if not isinstance(raw_decisions, list) or any(
            not isinstance(value, Mapping) for value in raw_decisions
        ):
            raise ValueError("decisions must be an array of objects")
        decisions: list[dict[str, Any]] = []
        decision_subsets: list[Subset] = []
        accepted_from_decisions: list[str] = []
        accepted_so_far: set[str] = set()
        seen_candidate_ids: set[str] = set()
        expected_spent = 0
        for index, raw_decision in enumerate(raw_decisions):
            decision = thaw_json(freeze_mapping(raw_decision))
            if decision.get("proposal_index") != index:
                raise ValueError("MOCHA decision proposal indexes must be contiguous")
            candidate_id = decision.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("MOCHA decision candidate_id must be non-empty")
            if candidate_id in seen_candidate_ids:
                raise ValueError("MOCHA decisions have duplicate candidate ids")
            seen_candidate_ids.add(candidate_id)
            subset = _normalize_subset(decision.get("patch_ids", ()), controller.patch_ids)
            decision["patch_ids"] = list(subset)
            decision_subsets.append(subset)
            if not isinstance(decision.get("accepted"), bool):
                raise ValueError("MOCHA decision accepted must be boolean")
            materialization_valid = decision.get("materialization_valid")
            if not isinstance(materialization_valid, bool):
                raise ValueError(
                    "MOCHA decision materialization_valid must be boolean"
                )
            evaluation_cost = decision.get("evaluation_cost")
            if isinstance(evaluation_cost, bool) or not isinstance(evaluation_cost, int):
                raise ValueError("MOCHA decision evaluation_cost must be an integer")
            required_cost = (
                controller.task_executions_per_candidate
                if materialization_valid
                else 0
            )
            if evaluation_cost != required_cost:
                raise ValueError(
                    "MOCHA decision evaluation cost contradicts materialization status"
                )
            if not materialization_valid and decision["accepted"]:
                raise ValueError("MOCHA cannot accept an invalid materialization")
            parent_id = decision.get("parent_id")
            parent_seed = decision.get("parent_seed")
            if parent_id is None:
                if parent_seed is not None:
                    raise ValueError("MOCHA null parent must have a null parent seed")
            else:
                if not isinstance(parent_id, str) or parent_id not in accepted_so_far:
                    raise ValueError(
                        "MOCHA decision parent must be a previously accepted incumbent"
                    )
                if parent_seed != controller._parent_seed(index):
                    raise ValueError("MOCHA decision parent seed is invalid")
            if materialization_valid and not accepted_so_far and not decision["accepted"]:
                raise ValueError("MOCHA must accept a proposal when no incumbent exists")
            if decision["accepted"]:
                accepted_from_decisions.append(candidate_id)
                accepted_so_far.add(candidate_id)
            expected_before = expected_spent
            expected_after = expected_before + evaluation_cost
            if (
                decision.get("logical_task_executions_before") != expected_before
                or decision.get("logical_task_executions_after") != expected_after
            ):
                raise ValueError("MOCHA decision logical budget accounting is invalid")
            expected_seed = (
                controller._acceptance_seed(index) if materialization_valid else None
            )
            if decision.get("acceptance_seed") != expected_seed:
                raise ValueError("MOCHA decision acceptance seed is invalid")
            expected_progress = expected_after / controller.logical_task_execution_budget
            progress = decision.get("logical_budget_progress")
            if (
                isinstance(progress, bool)
                or not isinstance(progress, (int, float))
                or not math.isclose(float(progress), expected_progress)
            ):
                raise ValueError("MOCHA decision logical budget progress is invalid")
            expected_spent = expected_after
            decisions.append(decision)
        if len(set(decision_subsets)) != len(decision_subsets):
            raise ValueError("MOCHA decisions have duplicate patch subsets")
        if set(decision_subsets) != controller._observed_subsets:
            raise ValueError("MOCHA decisions must match observed subsets")
        if accepted_from_decisions != controller._accepted_ids:
            raise ValueError("MOCHA decisions do not match accepted_ids")
        subset_by_candidate_id = dict(zip(
            (decision["candidate_id"] for decision in decisions),
            decision_subsets,
            strict=True,
        ))
        if any(
            tuple(incumbent.patch_ids)
            != subset_by_candidate_id[incumbent.candidate_id]
            for incumbent in controller._incumbents
        ):
            raise ValueError("MOCHA incumbent patch subsets do not match decisions")
        controller._decisions = decisions

        spent = _strict_int(
            state["logical_task_executions_spent"],
            "logical_task_executions_spent",
            minimum=0,
        )
        if spent != expected_spent or spent > controller.logical_task_execution_budget:
            raise ValueError("MOCHA logical task execution total is inconsistent")
        controller._logical_task_executions_spent = spent
        accepted_set = set(controller._accepted_ids)
        if any(
            parent_id is not None and parent_id not in accepted_set
            for parent_id in controller._pending_parent_ids
        ):
            raise ValueError("MOCHA pending parent is not an accepted incumbent")
        for offset, (parent_id, parent_seed) in enumerate(
            zip(
                controller._pending_parent_ids,
                controller._pending_parent_seeds,
                strict=True,
            )
        ):
            if parent_id is None:
                if parent_seed is not None or controller._incumbents:
                    raise ValueError("MOCHA pending null parent metadata is invalid")
            elif parent_seed != controller._parent_seed(len(decisions) + offset):
                raise ValueError("MOCHA pending parent seed is invalid")
        consumed = set(controller.stream.candidates[: controller.stream.cursor])
        if consumed != controller._observed_subsets | set(controller._pending):
            raise ValueError(
                "MOCHA stream cursor is inconsistent with observed and pending subsets"
            )
        reserved = len(controller._pending) * controller.task_executions_per_candidate
        if spent + reserved > controller.logical_task_execution_budget:
            raise ValueError("MOCHA pending proposals exceed the logical task budget")
        return controller


@runtime_checkable
class SubsetSearchController(Protocol):
    """Minimal runner integration contract shared by ask/tell controllers."""

    def ask(self, count: int = 1) -> tuple[Subset, ...]: ...

    def tell(self, scored: Iterable[ScoredCandidate]) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...


def restore_search_controller(
    state: Mapping[str, Any],
    *,
    binary_adapter: BinarySubsetBayesianAdapter | None = None,
) -> SubsetSearchController:
    """Restore a controller checkpoint through one runner-facing entrypoint."""

    state_type = state.get("state_type")
    if state_type == InitialDesignBinarySubsetController.STATE_TYPE:
        return InitialDesignBinarySubsetController.from_state(state)
    if state_type == AdapterBackedBinarySubsetController.STATE_TYPE:
        if binary_adapter is None:
            raise ExternalOptimizerRequired(
                "restoring adapter-backed binary search requires its external adapter"
            )
        return AdapterBackedBinarySubsetController.from_state(
            state, adapter=binary_adapter
        )
    if state_type == NSGAIIController.STATE_TYPE:
        return NSGAIIController.from_state(state)
    if state_type == EvoTopKController.STATE_TYPE:
        return EvoTopKController.from_state(state)
    if state_type == MOCHAController.STATE_TYPE:
        return MOCHAController.from_state(state)
    raise ValueError(f"unknown search controller state type: {state_type!r}")
