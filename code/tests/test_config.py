from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import paretoskill.config as config_module
from paretoskill.config import ConfigError, load_manifest, validate_manifest


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"
REAL_V2_TEMPLATE = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "iclr2027_real_v2.template.yaml"
)


def _base_data() -> dict:
    return copy.deepcopy(load_manifest(CONFIG, profile="dry_run", environment={}).data)


def _dynamic_adapter_data() -> dict:
    data = _base_data()
    data["active_runtime_profile"] = "replay"
    data["runtime"] = {"dataset_roots": {}}
    data["safety"]["allow_dynamic_runtime_adapter_imports"] = True
    data["safety"]["allowed_dynamic_runtime_adapter_prefixes"] = ["approved"]
    pin = "a" * 64
    data["harnesses"] = {
        harness_id: {
            "adapter": "approved.adapters:build_harness",
            "adapter_sha256": pin,
        }
        for harness_id in data["harnesses"]
    }
    data["domains"] = {
        domain_id: {
            "adapter": "approved.adapters:build_domain",
            "adapter_sha256": pin,
        }
        for domain_id in data["domains"]
    }
    return data


def _smoke_data() -> dict:
    data = _base_data()
    template = yaml.safe_load(REAL_V2_TEMPLATE.read_text(encoding="utf-8"))
    data["runner"] = copy.deepcopy(template["runner"])
    return data


def test_checked_in_manifest_validates_offline_and_has_stable_id() -> None:
    first = load_manifest(CONFIG, profile="dry_run", environment={})
    second = load_manifest(CONFIG, profile="dry_run", environment={})
    assert first.experiment_id == second.experiment_id
    assert first.is_offline
    assert first.data["constraints"]["id_accuracy_floor"]["epsilon"] == 0.05
    assert first.data["outputs"]["root"] == "runs/dry-run"
    assert first.unresolved_placeholders


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_retry_limit_must_be_a_non_negative_integer(value: object) -> None:
    data = _base_data()
    data["task_seed_blocks"]["retry_limit"] = value
    with pytest.raises(ConfigError, match="retry_limit must be a non-negative integer"):
        validate_manifest(data, profile="dry_run")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "eligible_for_retry_and_exclusion",
            [
                "sandbox_start_failure",
                "provider_transport_failure_before_response",
                "harness_crash_before_agent_output",
            ],
            "exactly match the four frozen eligible categories",
        ),
        (
            "eligible_for_retry_and_exclusion",
            "sandbox_start_failure",
            "must be a list of strings",
        ),
        (
            "never_exclude_as_infrastructure",
            ["model_refusal", "invalid_agent_output", "tool_misuse"],
            "exactly match the frozen never-exclude categories",
        ),
        (
            "never_exclude_as_infrastructure",
            ["model_refusal", 3],
            "must be a list of strings",
        ),
    ],
)
def test_failure_taxonomy_must_exactly_match_the_frozen_protocol(
    field: str, value: object, match: str
) -> None:
    data = _base_data()
    data["task_seed_blocks"]["failure_taxonomy"][field] = value
    with pytest.raises(ConfigError, match=match):
        validate_manifest(data, profile="dry_run")


def test_real_profile_fails_closed_on_unresolved_values() -> None:
    with pytest.raises(ConfigError, match="unresolved placeholders"):
        load_manifest(CONFIG, profile="real", environment={})


def test_real_v2_template_keeps_dynamic_runtime_imports_disabled() -> None:
    raw = yaml.safe_load(REAL_V2_TEMPLATE.read_text(encoding="utf-8"))
    safety = raw["safety"]
    assert safety["allow_dynamic_runtime_adapter_imports"] is False
    assert safety["allowed_dynamic_runtime_adapter_prefixes"] == [
        "${PARETOSKILL_RUNTIME_ADAPTER_ALLOWED_MODULE_PREFIX}"
    ]


def test_disabled_runtime_adapter_gate_accepts_unresolved_prefix_placeholder() -> None:
    data = _base_data()
    data["safety"]["allow_dynamic_runtime_adapter_imports"] = False
    data["safety"]["allowed_dynamic_runtime_adapter_prefixes"] = [
        "${PARETOSKILL_RUNTIME_ADAPTER_ALLOWED_MODULE_PREFIX}"
    ]
    validate_manifest(data, profile="dry_run")


def test_dynamic_runtime_adapter_contract_validates_for_replay() -> None:
    validate_manifest(_dynamic_adapter_data(), profile="replay")


def test_dynamic_runtime_adapters_reject_dry_run_profile() -> None:
    with pytest.raises(ConfigError, match="require the real or replay profile"):
        validate_manifest(_dynamic_adapter_data(), profile="dry_run")


def test_dynamic_runtime_adapters_require_boolean_gate() -> None:
    data = _base_data()
    data["safety"]["allow_dynamic_runtime_adapter_imports"] = "true"
    with pytest.raises(ConfigError, match="must be boolean"):
        validate_manifest(data, profile="dry_run")


@pytest.mark.parametrize(
    ("reference", "pin", "match"),
    [
        ("outside.adapters:build_harness", "a" * 64, "outside the allowlist"),
        ("approved.adapters.build_harness", "a" * 64, "module:factory"),
        ("approved.adapters:build_harness", "A" * 64, "exact lowercase SHA-256"),
    ],
)
def test_dynamic_runtime_adapter_reference_and_pin_are_strict(
    reference: str, pin: str, match: str
) -> None:
    data = _dynamic_adapter_data()
    data["harnesses"]["spreadsheet_primary"]["adapter"] = reference
    data["harnesses"]["spreadsheet_primary"]["adapter_sha256"] = pin
    with pytest.raises(ConfigError, match=match):
        validate_manifest(data, profile="replay")


def test_dynamic_runtime_adapter_prefix_must_be_resolved() -> None:
    data = _dynamic_adapter_data()
    data["safety"]["allowed_dynamic_runtime_adapter_prefixes"] = [
        "${PARETOSKILL_RUNTIME_ADAPTER_ALLOWED_MODULE_PREFIX}"
    ]
    with pytest.raises(ConfigError, match="resolved module names"):
        validate_manifest(data, profile="replay")


def test_frozen_runner_smoke_contract_validates() -> None:
    validate_manifest(_smoke_data(), profile="dry_run")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("methods", ["unknown"], "unknown run IDs"),
        ("search_seeds", [999], "subset of task_seed_blocks.search_seeds"),
        (
            "search_targets",
            [
                "id_small_primary",
                "cross_scale_large",
                "cross_harness_small",
                "ood_wtq_small",
            ],
            "equal the configured search targets",
        ),
        ("max_candidates", False, "max_candidates must be a positive integer"),
        ("blocks_per_target", 1, "blocks_per_target must be an integer >= 2"),
        ("logical_task_execution_ceiling", 39, "declared matrix exceeds"),
        ("separate_output_namespace", "../smoke", "one safe path segment"),
        (
            "never_promote_results_to_main_comparison",
            False,
            "never_promote_results_to_main_comparison must be true",
        ),
        (
            "candidate_limits",
            {"misspelled_method": 1},
            "keys must name declared smoke methods",
        ),
    ],
)
def test_runner_smoke_contract_rejects_drift(
    field: str, value: object, match: str
) -> None:
    data = _smoke_data()
    data["runner"]["smoke"][field] = value
    with pytest.raises(ConfigError, match=match):
        validate_manifest(data, profile="dry_run")


def test_validation_rejects_split_drift(tmp_path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["splits"]["id_validation"]["expected_count"] = 41
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="160/40/200"):
        load_manifest(path, profile="dry_run", environment={})


def test_validation_rejects_evo_promotion_budget_drift(tmp_path) -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["methods"]["evoskill_scalar_topk"]["top_k"] = 10
    path = tmp_path / "bad-evo-budget.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="top_k must equal the frozen promotion count"):
        load_manifest(path, profile="dry_run", environment={})


def test_implementation_digest_supports_non_editable_wheel_layout(
    tmp_path, monkeypatch
) -> None:
    package = tmp_path / "site-packages" / "paretoskill"
    package.mkdir(parents=True)
    installed_config = package / "config.py"
    installed_config.write_text("VALUE = 1\n", encoding="utf-8")
    (package / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "__file__", str(installed_config))

    manifest = load_manifest(CONFIG, profile="dry_run", environment={})
    assert manifest.code_root == package
    assert len(manifest.implementation_digest) == 64
    assert manifest.experiment_id.startswith("paretoskill-iclr2027-v1-")
