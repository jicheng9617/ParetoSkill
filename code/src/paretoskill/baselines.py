"""Baseline and ablation plugin contracts with clean-room reference strategies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .archive import ArchiveEntry, ParetoArchive
from .metrics import crowding_distance, hypervolume_contributions, strictly_dominates
from .models import freeze_mapping
from .objectives import (
    DEFAULT_ACTIVE_OBJECTIVES,
    DominanceMode,
    FeasibilityConstraints,
    ObjectiveName,
    normalize_active_objectives,
)
from .statistics import ObjectiveSummary, PointObjectives


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate_id: str
    patch_ids: tuple[str, ...]
    objectives: PointObjectives
    metadata: Mapping[str, Any] = field(default_factory=dict)
    summary: ObjectiveSummary | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        object.__setattr__(self, "patch_ids", tuple(self.patch_ids))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        if self.content_hash is not None and (
            len(self.content_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.content_hash)
        ):
            raise ValueError("content_hash must be a lowercase SHA-256 digest")

    @property
    def vector(self) -> tuple[float, float, float, float]:
        return self.objectives.maximize_vector()


class BaselinePlugin(Protocol):
    plugin_id: str
    archive_conditioned_generation: bool

    def propose_subsets(
        self, patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
    ) -> tuple[tuple[str, ...], ...]: ...

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]: ...


def _validate_count(count: int) -> None:
    if count < 1:
        raise ValueError("selection count must be positive")


def _diverse_subsets(
    patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    if max_candidates < 0:
        raise ValueError("max_candidates cannot be negative")
    ordered = tuple(sorted(set(patch_ids)))
    if not ordered or max_candidates == 0:
        return ()
    nonempty_space_size = (1 << len(ordered)) - 1
    # Base/no-skill are separate controls. Counting the empty subset here would
    # silently reduce every matched method's materialized candidate budget by one.
    initial = {ordered, *((patch_id,) for patch_id in ordered)}
    rng = random.Random(seed)
    while len(initial) < max_candidates and ordered:
        size = rng.randint(1, len(ordered))
        initial.add(tuple(sorted(rng.sample(ordered, size))))
        if len(initial) == nonempty_space_size:
            break

    def priority(subset: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        if subset == ordered:
            return (0, len(subset), subset)
        return (1, len(subset), subset)

    return tuple(sorted(initial, key=priority))[:max_candidates]


def nondominated_fronts(
    candidates: tuple[ScoredCandidate, ...],
) -> tuple[tuple[ScoredCandidate, ...], ...]:
    remaining = {candidate.candidate_id: candidate for candidate in candidates}
    fronts: list[tuple[ScoredCandidate, ...]] = []
    while remaining:
        front = tuple(
            candidate
            for candidate in sorted(remaining.values(), key=lambda item: item.candidate_id)
            if not any(
                strictly_dominates(other.vector, candidate.vector)
                for other in remaining.values()
                if other.candidate_id != candidate.candidate_id
            )
        )
        fronts.append(front)
        for candidate in front:
            del remaining[candidate.candidate_id]
    return tuple(fronts)


@dataclass(slots=True)
class BaseControlPlugin:
    plugin_id: str
    artifact_kind: str = "configured_base_skill"
    archive_conditioned_generation: bool = False

    def propose_subsets(
        self, patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
    ) -> tuple[tuple[str, ...], ...]:
        del patch_ids, seed
        return ((),) if max_candidates else ()

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))[:count]


@dataclass(slots=True)
class FullMergePlugin:
    plugin_id: str = "trace2skill_all"
    archive_conditioned_generation: bool = False

    def propose_subsets(
        self, patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
    ) -> tuple[tuple[str, ...], ...]:
        del seed
        return (tuple(sorted(set(patch_ids))),) if max_candidates else ()

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        return candidates[: min(count, len(candidates))]


@dataclass(slots=True)
class AccuracyOnlyPlugin:
    plugin_id: str = "trace2skill_accuracy_subset"
    archive_conditioned_generation: bool = False

    def propose_subsets(
        self, patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
    ) -> tuple[tuple[str, ...], ...]:
        return _diverse_subsets(patch_ids, max_candidates=max_candidates, seed=seed)

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        return tuple(
            sorted(
                candidates,
                key=lambda item: (-item.objectives.id_accuracy, item.candidate_id),
            )[:count]
        )


@dataclass(slots=True)
class SimplePatchCompositionPlugin(AccuracyOnlyPlugin):
    """Seeded subsets selected by conservative feasibility, accuracy, then tokens."""

    plugin_id: str = "simple_patch_composition"

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        eligible: list[ScoredCandidate] = []
        for candidate in candidates:
            feasible = candidate.metadata.get("feasible")
            if not isinstance(feasible, bool):
                raise ValueError(
                    "simple patch composition requires a boolean conservative "
                    "'feasible' annotation"
                )
            if feasible:
                eligible.append(candidate)
        return tuple(
            sorted(
                eligible,
                key=lambda item: (
                    -item.objectives.id_accuracy,
                    item.objectives.token_cost,
                    item.candidate_id,
                ),
            )[:count]
        )


@dataclass(slots=True)
class FixedScalarizationPlugin:
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    ranges: tuple[tuple[float, float], ...] | None = None
    plugin_id: str = "fixed_scalarization"
    archive_conditioned_generation: bool = False
    require_frozen_ranges: bool = False

    def __post_init__(self) -> None:
        if len(self.weights) != 4 or any(weight < 0 for weight in self.weights):
            raise ValueError("scalarization requires four non-negative weights")
        if sum(self.weights) <= 0:
            raise ValueError("at least one scalarization weight must be positive")
        if self.ranges is not None:
            if len(self.ranges) != 4:
                raise ValueError("scalarization requires four frozen ranges")
            for minimum, maximum in self.ranges:
                if not math.isfinite(minimum) or not math.isfinite(maximum):
                    raise ValueError("scalarization ranges must be finite")
                if maximum < minimum:
                    raise ValueError("scalarization range maximum must not be below minimum")
        if self.require_frozen_ranges and self.ranges is None:
            raise ValueError("this scalarization requires pre-frozen normalization ranges")

    def propose_subsets(
        self, patch_ids: tuple[str, ...], *, max_candidates: int, seed: int
    ) -> tuple[tuple[str, ...], ...]:
        return _diverse_subsets(patch_ids, max_candidates=max_candidates, seed=seed)

    def _ranges(self, candidates: tuple[ScoredCandidate, ...]) -> tuple[tuple[float, float], ...]:
        if self.ranges is not None:
            return self.ranges
        if self.require_frozen_ranges:
            raise ValueError("frozen normalization ranges were not supplied")
        vectors = [candidate.vector for candidate in candidates]
        return tuple(
            (min(vector[index] for vector in vectors), max(vector[index] for vector in vectors))
            for index in range(4)
        )

    def scores(self, candidates: tuple[ScoredCandidate, ...]) -> dict[str, float]:
        if not candidates:
            return {}
        ranges = self._ranges(candidates)
        scores: dict[str, float] = {}
        for candidate in candidates:
            normalized = []
            for value, (minimum, maximum) in zip(candidate.vector, ranges, strict=True):
                normalized.append(
                    0.0
                    if maximum <= minimum
                    else (value - minimum) / (maximum - minimum)
                )
            scores[candidate.candidate_id] = sum(
                weight * value for weight, value in zip(self.weights, normalized, strict=True)
            )
        return scores

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        scores = self.scores(candidates)
        return tuple(
            sorted(candidates, key=lambda item: (-scores[item.candidate_id], item.candidate_id))[
                :count
            ]
        )


@dataclass(slots=True)
class Ctx2SkillFixedProductPlugin(AccuracyOnlyPlugin):
    plugin_id: str = "ctx2skill_hard_easy_product"

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        for candidate in candidates:
            has_hard = "hard_probe_success_rate" in candidate.metadata
            has_easy = "easy_probe_success_rate" in candidate.metadata
            if not has_hard or not has_easy:
                raise ValueError("Ctx2Skill adaptation requires frozen hard/easy probe rates")
            for name in ("hard_probe_success_rate", "easy_probe_success_rate"):
                value = candidate.metadata[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be numeric")
                if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"{name} must be finite and in [0, 1]")
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -float(item.metadata["hard_probe_success_rate"])
                    * float(item.metadata["easy_probe_success_rate"]),
                    item.candidate_id,
                ),
            )[:count]
        )


@dataclass(slots=True)
class EvoSkillTopKPlugin(AccuracyOnlyPlugin):
    top_k: int = 10
    plugin_id: str = "evoskill_scalar_topk"

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        return super().select(candidates, count=min(count, self.top_k))


@dataclass(slots=True)
class NSGAIIPlugin(AccuracyOnlyPlugin):
    plugin_id: str = "skillmoo_nsga2"

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        selected: list[ScoredCandidate] = []
        for front in nondominated_fronts(candidates):
            remaining = count - len(selected)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                selected.extend(front)
                continue
            vectors = {candidate.candidate_id: candidate.vector for candidate in front}
            distances = crowding_distance(vectors)
            selected.extend(
                sorted(
                    front,
                    key=lambda item: (-distances[item.candidate_id], item.candidate_id),
                )[:remaining]
            )
        return tuple(selected)


@dataclass(slots=True)
class MOCHAPlugin(FixedScalarizationPlugin):
    """Clean-room stateful adaptation of MOCHA's Chebyshev/HVC schedule.

    The iterative search driver should call :meth:`select_parent` and
    :meth:`accept_proposal`.  ``select`` is only the deterministic final survivor
    rule required by the common baseline protocol.
    """

    plugin_id: str = "mocha_chebyshev_hvc"
    dirichlet_alpha: float = 1.0
    annealing_rate: float = 4.0
    reference_point: tuple[float, float, float, float] = (
        -1e-12,
        -1e-12,
        -1e-12,
        -1e-12,
    )

    def __post_init__(self) -> None:
        super(MOCHAPlugin, self).__post_init__()
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("MOCHA Dirichlet alpha must be finite and positive")
        if not math.isfinite(self.annealing_rate) or self.annealing_rate <= 0.0:
            raise ValueError("MOCHA annealing rate must be finite and positive")
        if len(self.reference_point) != 4 or any(
            not math.isfinite(value) for value in self.reference_point
        ):
            raise ValueError("MOCHA reference point must contain four finite values")

    def _normalized(
        self, candidates: tuple[ScoredCandidate, ...]
    ) -> dict[str, tuple[float, float, float, float]]:
        if not candidates:
            return {}
        ranges = self._ranges(candidates)
        normalized: dict[str, tuple[float, float, float, float]] = {}
        for candidate in candidates:
            values = tuple(
                0.0
                if maximum <= minimum
                else min(1.0, max(0.0, (value - minimum) / (maximum - minimum)))
                for value, (minimum, maximum) in zip(
                    candidate.vector, ranges, strict=True
                )
            )
            normalized[candidate.candidate_id] = values  # type: ignore[assignment]
        return normalized

    def sample_weights(self, *, seed: int) -> tuple[float, float, float, float]:
        rng = random.Random(seed)
        draws = tuple(rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in range(4))
        total = sum(draws)
        return tuple(value / total for value in draws)  # type: ignore[return-value]

    @staticmethod
    def _chebyshev(
        vector: tuple[float, ...], weights: tuple[float, ...]
    ) -> float:
        # All normalized objectives are maximize-space; smaller distance to the
        # ideal point is better. Values are clipped by _normalized.
        return max(
            weight * (1.0 - value)
            for weight, value in zip(weights, vector, strict=True)
        )

    def select_parent(
        self, candidates: tuple[ScoredCandidate, ...], *, seed: int
    ) -> ScoredCandidate:
        if not candidates:
            raise ValueError("MOCHA parent selection requires at least one candidate")
        normalized = self._normalized(candidates)
        weights = self.sample_weights(seed=seed)
        return min(
            candidates,
            key=lambda item: (
                self._chebyshev(normalized[item.candidate_id], weights),
                item.candidate_id,
            ),
        )

    def accept_proposal(
        self,
        proposal: ScoredCandidate,
        incumbents: tuple[ScoredCandidate, ...],
        *,
        task_executions_spent: int,
        task_execution_budget: int,
        seed: int,
    ) -> bool:
        """Apply a reproducible annealed HVC-to-Chebyshev acceptance gate."""

        if task_execution_budget <= 0:
            raise ValueError("MOCHA task execution budget must be positive")
        if not 0 <= task_executions_spent <= task_execution_budget:
            raise ValueError("MOCHA spent budget must lie within the total budget")
        combined = incumbents + (proposal,)
        normalized = self._normalized(combined)
        proposal_id = proposal.candidate_id
        contributions = hypervolume_contributions(normalized, self.reference_point)
        hvc_accept = contributions[proposal_id] > 0.0
        if not incumbents:
            return True
        weights = self.sample_weights(seed=seed)
        proposal_score = self._chebyshev(normalized[proposal_id], weights)
        incumbent_score = min(
            self._chebyshev(normalized[item.candidate_id], weights)
            for item in incumbents
        )
        chebyshev_accept = proposal_score <= incumbent_score
        progress = task_executions_spent / task_execution_budget
        hvc_probability = math.exp(-self.annealing_rate * progress)
        gate = random.Random(seed ^ 0x4D4F434841).random()
        return hvc_accept if gate < hvc_probability else chebyshev_accept

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        if not candidates:
            return ()
        normalized = self._normalized(candidates)
        contributions = hypervolume_contributions(normalized, self.reference_point)
        weights = self.weights
        chebyshev = {
            candidate.candidate_id: self._chebyshev(
                normalized[candidate.candidate_id], weights
            )
            for candidate in candidates
        }
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    contributions[item.candidate_id] <= 0.0,
                    chebyshev[item.candidate_id],
                    -contributions[item.candidate_id],
                    item.candidate_id,
                ),
            )[:count]
        )


@dataclass(slots=True)
class ArchiveSelectionPlugin(AccuracyOnlyPlugin):
    """Selection backed by the uncertainty-aware archive, not point NSGA-II."""

    constraints: FeasibilityConstraints | None = None
    archive_capacity: int = 128
    evaluation_budget: int = 2**63 - 1
    dominance_mode: DominanceMode = "uncertainty"
    active_objectives: tuple[ObjectiveName, ...] = DEFAULT_ACTIVE_OBJECTIVES
    plugin_id: str = "archive_selection"
    archive_conditioned_generation: bool = False

    def select(
        self, candidates: tuple[ScoredCandidate, ...], *, count: int
    ) -> tuple[ScoredCandidate, ...]:
        _validate_count(count)
        if self.constraints is None:
            raise ValueError("archive selection requires frozen feasibility constraints")
        if any(candidate.summary is None for candidate in candidates):
            raise ValueError("archive selection requires uncertainty ObjectiveSummary values")
        archive = ParetoArchive(
            max_size=max(count, self.archive_capacity),
            evaluation_budget=self.evaluation_budget,
            constraints=self.constraints,
            dominance_mode=self.dominance_mode,
            active_objectives=self.active_objectives,
        )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate in sorted(candidates, key=lambda item: item.candidate_id):
            evaluation_cost = candidate.metadata.get("evaluation_cost", 1)
            if isinstance(evaluation_cost, bool) or not isinstance(evaluation_cost, int):
                raise ValueError("candidate evaluation_cost must be an integer")
            if candidate.content_hash is None:
                raise ValueError(
                    "archive selection requires the materialized skill content_hash"
                )
            assert candidate.summary is not None
            archive.admit(
                ArchiveEntry(
                    candidate_id=candidate.candidate_id,
                    content_hash=candidate.content_hash,
                    objectives=candidate.summary,
                    evaluation_cost=evaluation_cost,
                )
            )
        # The scientific front is unbounded; deployment/reporting can select a
        # deterministic subset without changing the search-time working archive.
        front = archive.scientific_front
        if len(front) <= count:
            selected_ids = {entry.candidate_id for entry in front}
        else:
            selected_ids = archive._capacity_keep_ids(front, count)
        return tuple(by_id[candidate_id] for candidate_id in sorted(selected_ids))


@dataclass(slots=True)
class PassiveArchivePlugin(ArchiveSelectionPlugin):
    plugin_id: str = "passive_archive"
    archive_conditioned_generation: bool = False


@dataclass(slots=True)
class ParetoSkillPlugin(ArchiveSelectionPlugin):
    plugin_id: str = "paretoskill"
    archive_conditioned_generation: bool = True


def builtin_plugins() -> dict[str, BaselinePlugin]:
    plugins: tuple[BaselinePlugin, ...] = (
        BaseControlPlugin("no_skill", artifact_kind="no_skill_injection"),
        BaseControlPlugin("base_skill", artifact_kind="configured_base_skill"),
        SimplePatchCompositionPlugin(),
        FullMergePlugin(),
        AccuracyOnlyPlugin(),
        FixedScalarizationPlugin(),
        Ctx2SkillFixedProductPlugin(),
        EvoSkillTopKPlugin(),
        NSGAIIPlugin(),
        MOCHAPlugin(),
        PassiveArchivePlugin(),
        ParetoSkillPlugin(),
    )
    return {plugin.plugin_id: plugin for plugin in plugins}


@dataclass(slots=True)
class PluginRegistry:
    plugins: dict[str, BaselinePlugin] = field(default_factory=builtin_plugins)

    @classmethod
    def from_manifest(
        cls,
        configuration: Mapping[str, Any],
        *,
        normalization_ranges: tuple[tuple[float, float], ...] | None = None,
        strict_frozen_inputs: bool = True,
    ) -> "PluginRegistry":
        """Instantiate the declared comparison, including scalar variants.

        Frozen pilot ranges are deliberately supplied as data, rather than
        recomputed from the candidate pool. A real driver should keep
        ``strict_frozen_inputs=True``; the relaxed mode exists only for synthetic
        interface tests.
        """

        methods = configuration.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("manifest methods must be a mapping")
        missing = {
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
        } - set(methods)
        if missing:
            raise ValueError(f"manifest is missing baseline methods: {sorted(missing)}")

        raw_constraints = configuration.get("constraints")
        if not isinstance(raw_constraints, Mapping):
            raise ValueError("manifest constraints must be a mapping")
        accuracy = raw_constraints.get("id_accuracy_floor")
        token = raw_constraints.get("token_budget")
        if not isinstance(accuracy, Mapping) or not isinstance(token, Mapping):
            raise ValueError("manifest feasibility constraints are incomplete")
        epsilon = accuracy.get("epsilon")
        token_budget = token.get("budget")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("resolved accuracy epsilon must be numeric")
        if isinstance(token_budget, bool) or not isinstance(token_budget, (int, float)):
            raise ValueError("resolved token budget must be numeric")
        constraints = FeasibilityConstraints.from_paired_epsilon(
            epsilon=float(epsilon),
            token_budget=float(token_budget),
            enabled=raw_constraints.get("enabled") is not False,
        )

        raw_objectives = configuration.get("objectives")
        if not isinstance(raw_objectives, Mapping):
            raise ValueError("manifest objectives must be a mapping")
        active_objectives = normalize_active_objectives(
            name
            for name in DEFAULT_ACTIVE_OBJECTIVES
            if isinstance(raw_objectives.get(name), Mapping)
            and raw_objectives[name].get("enabled") is True
        )
        raw_statistics = configuration.get("statistics")
        if not isinstance(raw_statistics, Mapping):
            raise ValueError("manifest statistics must be a mapping")
        raw_dominance = raw_statistics.get("dominance")
        if not isinstance(raw_dominance, Mapping):
            raise ValueError("manifest dominance configuration is incomplete")
        dominance_mode: DominanceMode = (
            "point" if raw_dominance.get("primary") == "point_estimates" else "uncertainty"
        )
        raw_selection = configuration.get("selection_protocol")
        raw_budgets = configuration.get("budgets")
        if not isinstance(raw_selection, Mapping) or not isinstance(raw_budgets, Mapping):
            raise ValueError("manifest selection/budget configuration is incomplete")
        capacity_spec = raw_selection.get("archive_capacity")
        search_budget_spec = raw_budgets.get("search_total_per_method")
        if not isinstance(capacity_spec, Mapping) or not isinstance(
            search_budget_spec, Mapping
        ):
            raise ValueError("manifest archive capacity/search budget is incomplete")
        capacity = int(capacity_spec["max_entries"])
        evaluation_budget = int(search_budget_spec["task_executions"])

        fixed_spec = methods["fixed_scalarization"]
        if not isinstance(fixed_spec, Mapping):
            raise ValueError("fixed_scalarization method must be a mapping")
        variants = fixed_spec.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("fixed_scalarization must declare variants")
        scalar_plugins: list[BaselinePlugin] = []
        main_scalar: FixedScalarizationPlugin | None = None
        ctx_plugin: Ctx2SkillFixedProductPlugin | None = None
        for raw_variant in variants:
            if not isinstance(raw_variant, Mapping):
                raise ValueError("scalarization variant must be a mapping")
            variant_id = str(raw_variant.get("id", ""))
            if variant_id == "ctx2skill_hard_easy_product":
                ctx_plugin = Ctx2SkillFixedProductPlugin()
                scalar_plugins.append(ctx_plugin)
                continue
            raw_weights = raw_variant.get("maximize_space_weights")
            if not isinstance(raw_weights, list) or len(raw_weights) != 4:
                if strict_frozen_inputs:
                    raise ValueError(
                        f"scalar variant {variant_id!r} lacks four frozen maximize-space weights"
                    )
                defaults = {
                    "accuracy_only": (1.0, 0.0, 0.0, 0.0),
                    "accuracy_cost_equal": (0.5, 0.0, 0.5, 0.0),
                    "balanced_four_objective": (0.25, 0.25, 0.25, 0.25),
                }
                weights = defaults.get(variant_id)
                if weights is None:
                    raise ValueError(f"unknown scalarization variant: {variant_id!r}")
            else:
                weights = tuple(float(value) for value in raw_weights)
            plugin = FixedScalarizationPlugin(
                weights=weights,  # type: ignore[arg-type]
                ranges=normalization_ranges,
                plugin_id=f"fixed_scalarization/{variant_id}",
                require_frozen_ranges=strict_frozen_inputs,
            )
            scalar_plugins.append(plugin)
            if variant_id == "balanced_four_objective":
                main_scalar = plugin
        if main_scalar is None or ctx_plugin is None:
            raise ValueError("fixed scalarization must include balanced and Ctx2Skill variants")

        evo_spec = methods["evoskill_scalar_topk"]
        mocha_spec = methods["mocha_chebyshev_hvc"]
        if not isinstance(evo_spec, Mapping) or not isinstance(mocha_spec, Mapping):
            raise ValueError("EvoSkill/MOCHA method specs must be mappings")
        mocha_protocol = mocha_spec.get("protocol", {})
        if not isinstance(mocha_protocol, Mapping):
            raise ValueError("MOCHA protocol must be a mapping")
        if strict_frozen_inputs:
            required_mocha = {"dirichlet_alpha", "annealing_rate"}
            if not required_mocha <= set(mocha_protocol):
                raise ValueError("MOCHA protocol parameters are not frozen")

        plugins: list[BaselinePlugin] = [
            BaseControlPlugin("no_skill", artifact_kind="no_skill_injection"),
            BaseControlPlugin("base_skill", artifact_kind="configured_base_skill"),
            SimplePatchCompositionPlugin(),
            FullMergePlugin(),
            AccuracyOnlyPlugin(),
            *scalar_plugins,
            EvoSkillTopKPlugin(top_k=int(evo_spec["top_k"])),
            NSGAIIPlugin(),
            MOCHAPlugin(
                ranges=normalization_ranges,
                require_frozen_ranges=strict_frozen_inputs,
                dirichlet_alpha=float(mocha_protocol.get("dirichlet_alpha", 1.0)),
                annealing_rate=float(mocha_protocol.get("annealing_rate", 4.0)),
            ),
            PassiveArchivePlugin(
                constraints=constraints,
                archive_capacity=capacity,
                evaluation_budget=evaluation_budget,
                dominance_mode=dominance_mode,
                active_objectives=active_objectives,
            ),
            ParetoSkillPlugin(
                constraints=constraints,
                archive_capacity=capacity,
                evaluation_budget=evaluation_budget,
                dominance_mode=dominance_mode,
                active_objectives=active_objectives,
            ),
        ]
        registry = cls(plugins={})
        registry.register(
            FixedScalarizationPlugin(
                weights=main_scalar.weights,
                ranges=main_scalar.ranges,
                plugin_id="fixed_scalarization",
                require_frozen_ranges=main_scalar.require_frozen_ranges,
            )
        )
        for plugin in plugins:
            if plugin.plugin_id not in registry.plugins:
                registry.register(plugin)
        return registry

    def register(self, plugin: BaselinePlugin) -> None:
        if plugin.plugin_id in self.plugins:
            raise ValueError(f"duplicate plugin id: {plugin.plugin_id}")
        self.plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> BaselinePlugin:
        try:
            return self.plugins[plugin_id]
        except KeyError as exc:
            raise ValueError(f"unknown baseline plugin: {plugin_id}") from exc
