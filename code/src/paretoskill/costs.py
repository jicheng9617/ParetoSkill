"""Provider-token usage and frozen-price reconciliation.

The runner treats token counts as the scientific cost signal.  Monetary cost is
reported only when every contributing provider has a finite, non-negative,
dated price declaration; unresolved placeholders never become zero-cost calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .evaluation import EvaluationRecord


class CostAccountingError(ValueError):
    """A usage row or frozen provider price is malformed."""


@dataclass(frozen=True, slots=True)
class ProviderPrice:
    currency: str
    input_per_million_tokens: float
    output_per_million_tokens: float
    source_url: str
    retrieved_at_utc: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProviderPrice:
        try:
            currency = value["currency"]
            source_url = value["source_url"]
            retrieved_at_utc = value["retrieved_at_utc"]
            input_price = value["input_per_million_tokens"]
            output_price = value["output_per_million_tokens"]
        except KeyError as exc:
            raise CostAccountingError(f"provider price is missing {exc.args[0]!r}") from exc
        strings = (currency, source_url, retrieved_at_utc)
        if any(not isinstance(item, str) or not item.strip() for item in strings):
            raise CostAccountingError("provider price provenance must be non-empty strings")
        prices = (input_price, output_price)
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in prices
        ):
            raise CostAccountingError("provider token prices must be finite and non-negative")
        return cls(
            currency=currency,
            input_per_million_tokens=float(input_price),
            output_per_million_tokens=float(output_price),
            source_url=source_url,
            retrieved_at_utc=retrieved_at_utc,
        )

    def cost(self, *, input_tokens: int, output_tokens: int) -> float:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens)
        ):
            raise CostAccountingError("token usage must contain non-negative integers")
        return (
            input_tokens * self.input_per_million_tokens
            + output_tokens * self.output_per_million_tokens
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "input_per_million_tokens": self.input_per_million_tokens,
            "output_per_million_tokens": self.output_per_million_tokens,
            "source_url": self.source_url,
            "retrieved_at_utc": self.retrieved_at_utc,
        }


def _configured_provider_id(
    record: EvaluationRecord, configuration: Mapping[str, Any]
) -> str:
    reported = record.result.provider_metadata.get("provider")
    if isinstance(reported, str) and reported.strip():
        return reported
    try:
        target = configuration["targets"][record.target_id]
        model = configuration["models"][target["model"]]
        provider_id = model["provider"]
    except (KeyError, TypeError) as exc:
        raw_models = configuration.get("models", {})
        configured = {
            raw_model.get("provider")
            for raw_model in raw_models.values()
            if isinstance(raw_model, Mapping)
            and isinstance(raw_model.get("provider"), str)
            and raw_model.get("provider")
        }
        if len(configured) != 1:
            raise CostAccountingError(
                f"cannot resolve provider for target {record.target_id!r}"
            ) from exc
        provider_id = next(iter(configured))
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise CostAccountingError("configured provider id must be a non-empty string")
    return provider_id


def usage_by_provider(
    records: Iterable[EvaluationRecord], configuration: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    """Aggregate actual execution rows by their provider without pricing them."""

    usage: dict[str, dict[str, int]] = {}
    for record in records:
        if not isinstance(record, EvaluationRecord):
            raise CostAccountingError("provider usage requires EvaluationRecord values")
        provider_id = _configured_provider_id(record, configuration)
        bucket = usage.setdefault(
            provider_id,
            {"provider_calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        bucket["provider_calls"] += 1
        bucket["input_tokens"] += record.result.input_tokens
        bucket["output_tokens"] += record.result.output_tokens
    return {provider_id: usage[provider_id] for provider_id in sorted(usage)}


def generation_usage_by_provider(
    generations: Iterable[Mapping[str, Any]], *, default_provider_id: str
) -> dict[str, dict[str, int]]:
    """Aggregate cached proposer generations without relying on secret-bearing data."""

    if not isinstance(default_provider_id, str) or not default_provider_id.strip():
        raise CostAccountingError("default generation provider id must be non-empty")
    usage: dict[str, dict[str, int]] = {}
    for generation in generations:
        if not isinstance(generation, Mapping):
            raise CostAccountingError("generation usage requires mapping values")
        metadata = generation.get("provider_metadata", {})
        if not isinstance(metadata, Mapping):
            raise CostAccountingError("generation provider_metadata must be a mapping")
        reported = metadata.get("provider")
        provider_id = (
            reported
            if isinstance(reported, str) and reported.strip()
            else default_provider_id
        )
        input_tokens = generation.get("input_tokens")
        output_tokens = generation.get("output_tokens")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens)
        ):
            raise CostAccountingError(
                "generation token usage must contain non-negative integers"
            )
        bucket = usage.setdefault(
            provider_id,
            {"provider_calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        bucket["provider_calls"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
    return {provider_id: usage[provider_id] for provider_id in sorted(usage)}


def reconcile_provider_costs(
    usage: Mapping[str, Mapping[str, int]],
    providers: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen price provenance, preserving undefined prices explicitly."""

    rows: dict[str, Any] = {}
    currency_totals: dict[str, float] = {}
    all_defined = True
    for provider_id in sorted(usage):
        values = usage[provider_id]
        try:
            calls = values["provider_calls"]
            input_tokens = values["input_tokens"]
            output_tokens = values["output_tokens"]
        except KeyError as exc:
            raise CostAccountingError(f"usage row is missing {exc.args[0]!r}") from exc
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (calls, input_tokens, output_tokens)
        ):
            raise CostAccountingError("usage counts must be non-negative integers")
        row: dict[str, Any] = {
            "provider_calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        raw_provider = providers.get(provider_id)
        raw_price = raw_provider.get("price") if isinstance(raw_provider, Mapping) else None
        try:
            if not isinstance(raw_price, Mapping):
                raise CostAccountingError("provider has no frozen price mapping")
            price = ProviderPrice.from_mapping(raw_price)
        except CostAccountingError as exc:
            all_defined = False
            row.update(
                {
                    "price_defined": False,
                    "monetary_cost": None,
                    "undefined_reason": str(exc),
                }
            )
        else:
            monetary_cost = price.cost(
                input_tokens=input_tokens, output_tokens=output_tokens
            )
            currency_totals[price.currency] = (
                currency_totals.get(price.currency, 0.0) + monetary_cost
            )
            row.update(
                {
                    "price_defined": True,
                    "price": price.to_dict(),
                    "monetary_cost": monetary_cost,
                }
            )
        rows[provider_id] = row
    single_currency_total = (
        next(iter(currency_totals.values()))
        if all_defined and len(currency_totals) == 1
        else None
    )
    return {
        "schema_version": 1,
        "all_prices_defined": all_defined,
        "providers": rows,
        "currency_totals": {
            currency: currency_totals[currency] for currency in sorted(currency_totals)
        },
        "total_monetary_cost": single_currency_total,
        "currency": (
            next(iter(currency_totals))
            if all_defined and len(currency_totals) == 1
            else None
        ),
    }


__all__ = [
    "CostAccountingError",
    "ProviderPrice",
    "generation_usage_by_provider",
    "reconcile_provider_costs",
    "usage_by_provider",
]
