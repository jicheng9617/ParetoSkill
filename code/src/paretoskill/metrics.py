"""Dependency-free frontier metrics used by matched-budget comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


Vector = tuple[float, ...]


def weakly_dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right) or not left:
        raise ValueError("objective vectors must have the same non-zero dimension")
    return all(lhs >= rhs for lhs, rhs in zip(left, right, strict=True))


def strictly_dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return weakly_dominates(left, right) and any(
        lhs > rhs for lhs, rhs in zip(left, right, strict=True)
    )


def nondominated(points: Iterable[Sequence[float]]) -> tuple[Vector, ...]:
    unique = sorted({tuple(float(value) for value in point) for point in points})
    return tuple(
        point
        for index, point in enumerate(unique)
        if not any(
            strictly_dominates(other, point)
            for other_index, other in enumerate(unique)
            if index != other_index
        )
    )


def hypervolume(points: Iterable[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact dominated hypervolume for maximize-oriented points.

    The recursive slab algorithm is intended for small archives, not large-scale
    optimization internals. Points that do not weakly dominate the fixed
    reference contribute zero.
    """

    reference_vector = tuple(float(value) for value in reference)
    if not reference_vector:
        raise ValueError("reference point must be non-empty")
    if not all(math.isfinite(value) for value in reference_vector):
        raise ValueError("reference point must contain only finite values")
    vectors: list[Vector] = []
    for point in points:
        vector = tuple(float(value) for value in point)
        if len(vector) != len(reference_vector):
            raise ValueError("hypervolume point dimension mismatch")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("hypervolume points must contain only finite values")
        vectors.append(vector)
    eligible = [
        vector
        for vector in vectors
        if weakly_dominates(vector, reference_vector)
    ]
    if not eligible:
        return 0.0

    def recurse(vectors: list[Vector], ref: Vector) -> float:
        if len(ref) == 1:
            return max(vector[0] for vector in vectors) - ref[0]
        levels = sorted({ref[-1], *(vector[-1] for vector in vectors)})
        volume = 0.0
        for lower, upper in zip(levels[:-1], levels[1:], strict=True):
            if upper <= lower:
                continue
            active = [vector[:-1] for vector in vectors if vector[-1] >= upper]
            if active:
                volume += (upper - lower) * recurse(active, ref[:-1])
        return volume

    return recurse(list(nondominated(eligible)), reference_vector)


def hypervolume_contributions(
    points: Mapping[str, Sequence[float]], reference: Sequence[float]
) -> dict[str, float]:
    total = hypervolume(points.values(), reference)
    return {
        point_id: max(
            0.0,
            total
            - hypervolume(
                (value for other_id, value in points.items() if other_id != point_id),
                reference,
            ),
        )
        for point_id in sorted(points)
    }


def coverage(left: Iterable[Sequence[float]], right: Iterable[Sequence[float]]) -> float:
    """C(left, right): fraction of right weakly dominated by at least one left point."""

    left_points = [tuple(point) for point in left]
    right_points = [tuple(point) for point in right]
    if not right_points:
        raise ValueError("coverage requires a non-empty reference set")
    return sum(
        any(weakly_dominates(candidate, reference) for candidate in left_points)
        for reference in right_points
    ) / len(right_points)


def additive_epsilon_indicator(
    approximation: Iterable[Sequence[float]], reference_front: Iterable[Sequence[float]]
) -> float:
    """Unary additive epsilon indicator in maximize orientation (lower is better)."""

    approximation_points = [tuple(point) for point in approximation]
    reference_points = [tuple(point) for point in reference_front]
    if not approximation_points or not reference_points:
        raise ValueError("epsilon indicator requires two non-empty sets")
    return max(
        min(
            max(
                reference_value - approximation_value
                for approximation_value, reference_value in zip(
                    candidate, reference, strict=True
                )
            )
            for candidate in approximation_points
        )
        for reference in reference_points
    )


def inverted_generational_distance(
    approximation: Iterable[Sequence[float]], reference_front: Iterable[Sequence[float]]
) -> float:
    approximation_points = [tuple(point) for point in approximation]
    reference_points = [tuple(point) for point in reference_front]
    if not approximation_points or not reference_points:
        raise ValueError("IGD requires two non-empty sets")
    return sum(
        min(
            math.dist(candidate, reference)
            for candidate in approximation_points
        )
        for reference in reference_points
    ) / len(reference_points)


def normalize_vectors(
    points: Mapping[str, Sequence[float]],
    ranges: Sequence[tuple[float, float]],
) -> dict[str, Vector]:
    normalized: dict[str, Vector] = {}
    for point_id, point in points.items():
        if len(point) != len(ranges):
            raise ValueError("normalization dimension mismatch")
        values: list[float] = []
        for value, (minimum, maximum) in zip(point, ranges, strict=True):
            if maximum <= minimum:
                raise ValueError("normalization ranges must have positive width")
            values.append((float(value) - minimum) / (maximum - minimum))
        normalized[point_id] = tuple(values)
    return normalized


def crowding_distance(points: Mapping[str, Sequence[float]]) -> dict[str, float]:
    if not points:
        return {}
    dimensions = {len(tuple(point)) for point in points.values()}
    if len(dimensions) != 1 or not next(iter(dimensions)):
        raise ValueError("crowding points must share a non-zero dimension")
    distances = {point_id: 0.0 for point_id in points}
    for objective_index in range(next(iter(dimensions))):
        ordered = sorted(
            points,
            key=lambda point_id: (points[point_id][objective_index], point_id),
        )
        minimum = points[ordered[0]][objective_index]
        maximum = points[ordered[-1]][objective_index]
        if maximum <= minimum:
            continue
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        for index in range(1, len(ordered) - 1):
            point_id = ordered[index]
            if math.isinf(distances[point_id]):
                continue
            previous_value = points[ordered[index - 1]][objective_index]
            next_value = points[ordered[index + 1]][objective_index]
            distances[point_id] += (next_value - previous_value) / (maximum - minimum)
    return distances


def false_admission_rate(screen_ids: Iterable[str], full_archive_ids: Iterable[str]) -> float:
    screened = set(screen_ids)
    if not screened:
        return 0.0
    return len(screened - set(full_archive_ids)) / len(screened)
