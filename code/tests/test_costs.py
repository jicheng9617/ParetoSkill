from __future__ import annotations

from paretoskill.costs import (
    ProviderPrice,
    generation_usage_by_provider,
    reconcile_provider_costs,
)


def test_provider_price_and_reconciliation_are_exactly_token_based() -> None:
    price = ProviderPrice.from_mapping(
        {
            "currency": "USD",
            "input_per_million_tokens": 2.0,
            "output_per_million_tokens": 8.0,
            "source_url": "https://provider.example/pricing",
            "retrieved_at_utc": "2026-08-21T00:00:00Z",
        }
    )
    assert price.cost(input_tokens=1_000_000, output_tokens=500_000) == 6.0

    result = reconcile_provider_costs(
        {
            "provider-a": {
                "provider_calls": 3,
                "input_tokens": 1_000_000,
                "output_tokens": 500_000,
            }
        },
        {"provider-a": {"price": price.to_dict()}},
    )
    assert result["all_prices_defined"] is True
    assert result["currency"] == "USD"
    assert result["total_monetary_cost"] == 6.0


def test_unresolved_price_is_never_treated_as_zero_cost() -> None:
    result = reconcile_provider_costs(
        {
            "provider-a": {
                "provider_calls": 1,
                "input_tokens": 10,
                "output_tokens": 5,
            }
        },
        {
            "provider-a": {
                "price": {
                    "currency": "${PRICE_CURRENCY}",
                    "input_per_million_tokens": "${INPUT_PRICE}",
                    "output_per_million_tokens": "${OUTPUT_PRICE}",
                    "source_url": "${PRICE_URL}",
                    "retrieved_at_utc": "${PRICE_DATE}",
                }
            }
        },
    )
    assert result["all_prices_defined"] is False
    assert result["total_monetary_cost"] is None
    assert result["providers"]["provider-a"]["monetary_cost"] is None


def test_generation_usage_prefers_recorded_provider_and_has_frozen_fallback() -> None:
    usage = generation_usage_by_provider(
        (
            {
                "input_tokens": 4,
                "output_tokens": 2,
                "provider_metadata": {"provider": "actual"},
            },
            {
                "input_tokens": 3,
                "output_tokens": 1,
                "provider_metadata": {},
            },
        ),
        default_provider_id="configured",
    )
    assert usage == {
        "actual": {"provider_calls": 1, "input_tokens": 4, "output_tokens": 2},
        "configured": {"provider_calls": 1, "input_tokens": 3, "output_tokens": 1},
    }
