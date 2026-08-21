"""Versioned YAML loading, environment resolution, validation, and experiment IDs."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import canonical_json


class ConfigError(ValueError):
    pass


PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "experiment",
    "runtime_profiles",
    "safety",
    "reproducibility",
    "providers",
    "models",
    "harnesses",
    "domains",
    "datasets",
    "splits",
    "task_seed_blocks",
    "targets",
    "objectives",
    "constraints",
    "statistics",
    "budgets",
    "methods",
    "ablations",
    "metrics",
    "selection_protocol",
    "deployment",
    "outputs",
    "shared_search_controls",
    "proposer",
}

REQUIRED_METHODS = {
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
}

REQUIRED_ABLATIONS = {
    "no_uncertainty_bounds",
    "no_regression_objective",
    "no_transfer_objective",
    "passive_archive",
    "no_feasibility_gate",
    "evidence_blind_generation",
    "lineage_blind_generation",
    "patch_subset_only",
}

REQUIRED_OBJECTIVES = {
    "id_accuracy": ("maximize", "lower"),
    "worst_target_transfer": ("maximize", "lower"),
    "token_cost": ("minimize", "upper"),
    "paired_regression": ("minimize", "upper"),
}

BUILTIN_RUNTIME_ADAPTERS = {
    "builtin",
    "builtin_local_domain",
    "builtin_verified_response",
}

ELIGIBLE_INFRASTRUCTURE_FAILURES = (
    "sandbox_start_failure",
    "provider_transport_failure_before_response",
    "harness_crash_before_agent_output",
    "verifier_infrastructure_failure",
)

NEVER_EXCLUDE_AS_INFRASTRUCTURE = (
    "model_refusal",
    "invalid_agent_output",
    "tool_misuse",
    "task_timeout_after_valid_start",
    "verifier_assertion_failure",
)


def _mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be a mapping")
        return {}
    return value


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_placeholder(key) or _contains_placeholder(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return isinstance(value, str) and PLACEHOLDER.search(value) is not None


def _resolve_value(
    value: Any,
    *,
    environment: Mapping[str, str],
    path: str,
    unresolved: list[str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_value(
                item,
                environment=environment,
                path=f"{path}.{key}" if path else str(key),
                unresolved=unresolved,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_value(
                item,
                environment=environment,
                path=f"{path}[{index}]",
                unresolved=unresolved,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    matches = tuple(PLACEHOLDER.finditer(value))
    if not matches:
        return value
    missing = [match.group(1) for match in matches if match.group(1) not in environment]
    if missing:
        unresolved.append(f"{path}: {', '.join(sorted(set(missing)))}")
        return value
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        # Parse numbers/booleans/null while leaving ordinary strings unchanged.
        return yaml.safe_load(environment[matches[0].group(1)])
    result = value
    for match in matches:
        result = result.replace(match.group(0), environment[match.group(1)])
    return result


def _set_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ConfigError(f"cannot apply runtime override at {dotted_path}")
        current = child
    current[parts[-1]] = value


def _apply_profile(data: dict[str, Any], profile: str) -> None:
    profiles = data.get("runtime_profiles", {})
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ConfigError(f"unknown runtime profile: {profile}")
    profile_data = profiles[profile]
    if not isinstance(profile_data, dict):
        raise ConfigError(f"runtime_profiles.{profile} must be a mapping")
    data["active_runtime_profile"] = profile
    if profile == "dry_run":
        synthetic = profile_data.get("synthetic_only_values", {})
        if not isinstance(synthetic, dict):
            raise ConfigError("runtime_profiles.dry_run.synthetic_only_values must be a mapping")
        override_paths = {
            "accuracy_tolerance": "constraints.id_accuracy_floor.epsilon",
            "token_budget": "constraints.token_budget.budget",
            "archive_capacity": "selection_protocol.archive_capacity.max_entries",
            "final_task_executions": "budgets.final.task_executions",
            "results_root": "outputs.root",
        }
        for key, target_path in override_paths.items():
            if key not in synthetic:
                raise ConfigError(f"dry-run synthetic value is missing: {key}")
            _set_path(data, target_path, synthetic[key])
        _set_path(
            data,
            "selection_protocol.working_archive.capacity.max_entries",
            synthetic["archive_capacity"],
        )
        deployment = data.get("deployment")
        if not isinstance(deployment, dict):
            raise ConfigError("deployment must be a mapping")
        policies = deployment.get("policies")
        if not isinstance(policies, list) or any(
            not isinstance(policy, dict) for policy in policies
        ):
            raise ConfigError("deployment.policies must be an array of mappings")
        _set_path(
            data,
            "deployment.policies",
            [
                {
                    **policy,
                    **(
                        {"token_budget": synthetic["token_budget"]}
                        if policy.get("id") == "max_worst_transfer_under_token_budget"
                        else {}
                    ),
                }
                for policy in policies
            ],
        )


def _validate_manifest(data: Mapping[str, Any], *, profile: str) -> None:
    errors: list[str] = []
    missing_top = REQUIRED_TOP_LEVEL - set(data)
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")
    if str(data.get("schema_version")) != "1.0":
        errors.append("schema_version must be '1.0'")

    experiment = _mapping(data.get("experiment"), "experiment", errors)
    if not str(experiment.get("id", "")).strip():
        errors.append("experiment.id must be non-empty")
    minimum_search_seeds = experiment.get("minimum_search_seeds")
    if (
        isinstance(minimum_search_seeds, bool)
        or not isinstance(minimum_search_seeds, int)
        or minimum_search_seeds < 3
    ):
        errors.append("experiment.minimum_search_seeds must be at least 3")
    if profile == "dry_run" and experiment.get("mode") != "dry-run":
        errors.append("dry_run profile requires experiment.mode='dry-run'")

    safety = _mapping(data.get("safety"), "safety", errors)
    if safety.get("offline_by_default") is not True:
        errors.append("safety.offline_by_default must be true")
    if not safety.get("required_network_env") or not safety.get("required_network_value"):
        errors.append("the explicit network environment switch must be declared")
    allow_runtime_adapters = safety.get(
        "allow_dynamic_runtime_adapter_imports", False
    )
    if not isinstance(allow_runtime_adapters, bool):
        errors.append(
            "safety.allow_dynamic_runtime_adapter_imports must be boolean"
        )
    raw_runtime_prefixes = safety.get(
        "allowed_dynamic_runtime_adapter_prefixes"
    )
    runtime_prefixes: tuple[str, ...] = ()
    if raw_runtime_prefixes is not None:
        if (
            not isinstance(raw_runtime_prefixes, list)
            or any(
                not isinstance(prefix, str)
                or not prefix
                or prefix != prefix.strip()
                for prefix in raw_runtime_prefixes
            )
            or len(set(raw_runtime_prefixes)) != len(raw_runtime_prefixes)
        ):
            errors.append(
                "safety.allowed_dynamic_runtime_adapter_prefixes must be a "
                "unique non-empty string list"
            )
        else:
            runtime_prefixes = tuple(raw_runtime_prefixes)
    if allow_runtime_adapters is True:
        if profile not in {"real", "replay"}:
            errors.append(
                "dynamic runtime adapters require the real or replay profile"
            )
        if not runtime_prefixes:
            errors.append(
                "dynamic runtime adapters require non-empty allowed module prefixes"
            )
        elif any(
            PLACEHOLDER.search(prefix) is not None
            or not all(part.isidentifier() for part in prefix.split("."))
            for prefix in runtime_prefixes
        ):
            errors.append(
                "dynamic runtime adapter prefixes must be resolved module names"
            )
    if profile == "dry_run":
        if safety.get("allow_network") is not False:
            errors.append("dry-run manifest must set safety.allow_network=false")
        if safety.get("forbid_api_calls_in_dry_run") is not True:
            errors.append("dry-run must explicitly forbid API calls")
        active_profile = _mapping(
            _mapping(data.get("runtime_profiles"), "runtime_profiles", errors).get("dry_run"),
            "runtime_profiles.dry_run",
            errors,
        )
        if active_profile.get("network") is not False:
            errors.append("dry-run profile network must be false")
        if active_profile.get("provider_override") not in set(
            safety.get("allowed_dry_run_providers", [])
        ):
            errors.append("dry-run provider must be in allowed_dry_run_providers")

    providers = _mapping(data.get("providers"), "providers", errors)
    profiles = _mapping(data.get("runtime_profiles"), "runtime_profiles", errors)
    for profile_id in ("dry_run", "replay", "real"):
        runtime = _mapping(
            profiles.get(profile_id), f"runtime_profiles.{profile_id}", errors
        )
        if not isinstance(runtime.get("provider_override"), str):
            errors.append(
                f"runtime_profiles.{profile_id}.provider_override must be a string"
            )
        if not isinstance(runtime.get("data_adapter"), str):
            errors.append(f"runtime_profiles.{profile_id}.data_adapter must be a string")
        if not isinstance(runtime.get("network"), bool):
            errors.append(f"runtime_profiles.{profile_id}.network must be boolean")
        if not isinstance(runtime.get("resolve_external_placeholders"), bool):
            errors.append(
                f"runtime_profiles.{profile_id}.resolve_external_placeholders must be boolean"
            )
        if runtime.get("provider_override") not in providers:
            errors.append(
                f"runtime_profiles.{profile_id}.provider_override references an unknown provider"
            )
    for provider_id, provider_value in providers.items():
        provider = _mapping(provider_value, f"providers.{provider_id}", errors)
        if not isinstance(provider.get("enabled"), bool):
            errors.append(f"providers.{provider_id}.enabled must be boolean")
        if not isinstance(provider.get("network"), bool):
            errors.append(f"providers.{provider_id}.network must be boolean")
        if not str(provider.get("kind", "")).strip():
            errors.append(f"providers.{provider_id}.kind must be non-empty")
    active_runtime = profiles.get(profile)
    if isinstance(active_runtime, dict):
        active_provider_id = active_runtime.get("provider_override")
        active_provider = providers.get(active_provider_id)
        if isinstance(active_provider, dict):
            if active_provider.get("enabled") is not True:
                errors.append("active runtime provider must be enabled")
            if profile in {"dry_run", "replay"} and active_provider.get("network") is not False:
                errors.append("offline runtime providers must declare network=false")
        if profile == "dry_run" and active_runtime.get("data_adapter") != (
            "builtin_synthetic_fixture"
        ):
            errors.append("dry-run must use the built-in synthetic data adapter")

    reproducibility = _mapping(
        data.get("reproducibility"), "reproducibility", errors
    )
    if reproducibility.get("reject_path_only_identity") is not True:
        errors.append("reproducibility.reject_path_only_identity must be true")
    if (
        reproducibility.get("require_sha256_or_immutable_revision_for_external_inputs")
        is not True
    ):
        errors.append("external inputs must require a digest or immutable revision")
    identity_inputs = reproducibility.get("identity_inputs")
    required_identity_inputs = {
        "task_manifest_digests",
        "base_skill_content_digest",
        "patch_pool_digest",
        "provider_adapter_digest",
        "harness_adapter_digest",
        "dependency_lock_digest",
    }
    if not isinstance(identity_inputs, list) or not required_identity_inputs <= set(
        value for value in identity_inputs if isinstance(value, str)
    ):
        errors.append("reproducibility.identity_inputs is incomplete")
    models = _mapping(data.get("models"), "models", errors)
    for model_id, model_value in models.items():
        model = _mapping(model_value, f"models.{model_id}", errors)
        if model.get("provider") not in providers:
            errors.append(f"models.{model_id}.provider references an unknown provider")

    datasets = _mapping(data.get("datasets"), "datasets", errors)
    splits = _mapping(data.get("splits"), "splits", errors)
    for split_id, split_value in splits.items():
        split = _mapping(split_value, f"splits.{split_id}", errors)
        if split.get("dataset") not in datasets:
            errors.append(f"splits.{split_id}.dataset references an unknown dataset")
        for other in split.get("disjoint_from", []):
            if other not in splits:
                errors.append(f"splits.{split_id}.disjoint_from references {other!r}")
    try:
        count_values = (
            splits["evolution_trace"]["expected_count"],
            splits["id_validation"]["expected_count"],
            splits["heldout_verified"]["expected_count"],
            datasets["spreadsheetbench_verified"]["total_tasks"],
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in count_values
        ):
            raise TypeError("split counts must be integers")
        evolution_count, validation_count, heldout_count, verified_total = count_values
        primary_counts = (evolution_count, validation_count, heldout_count, verified_total)
        if primary_counts != (160, 40, 200, 400):
            errors.append("primary split must remain 160/40/200 out of Verified-400")
        if evolution_count + validation_count + heldout_count != verified_total:
            errors.append("primary split counts do not sum to the dataset total")
    except (KeyError, TypeError, ValueError):
        errors.append("primary split count fields are incomplete")

    harnesses = _mapping(data.get("harnesses"), "harnesses", errors)
    domains = _mapping(data.get("domains"), "domains", errors)
    if allow_runtime_adapters is True:
        runtime_spec = _mapping(data.get("runtime"), "runtime", errors)
        if _contains_placeholder(runtime_spec):
            errors.append(
                "runtime must be fully resolved before dynamic adapter loading"
            )
        for collection_name, collection in (
            ("harnesses", harnesses),
            ("domains", domains),
        ):
            for adapter_id, raw_spec in collection.items():
                location = f"{collection_name}.{adapter_id}"
                spec = _mapping(raw_spec, location, errors)
                if _contains_placeholder(spec):
                    errors.append(
                        f"{location} must be fully resolved before dynamic adapter loading"
                    )
                    continue
                reference = spec.get("adapter")
                if reference in BUILTIN_RUNTIME_ADAPTERS:
                    continue
                if not isinstance(reference, str) or reference.count(":") != 1:
                    errors.append(f"{location}.adapter must use 'module:factory'")
                    continue
                module_name, factory_name = reference.split(":", 1)
                reference_parts = (
                    *module_name.split("."),
                    *factory_name.split("."),
                )
                if (
                    not module_name
                    or not factory_name
                    or not all(part.isidentifier() for part in reference_parts)
                ):
                    errors.append(
                        f"{location}.adapter has an invalid module or factory name"
                    )
                    continue
                if runtime_prefixes and not any(
                    module_name == prefix or module_name.startswith(prefix + ".")
                    for prefix in runtime_prefixes
                ):
                    errors.append(f"{location}.adapter module is outside the allowlist")
                adapter_sha256 = spec.get("adapter_sha256")
                if not isinstance(adapter_sha256, str) or re.fullmatch(
                    r"[0-9a-f]{64}", adapter_sha256
                ) is None:
                    errors.append(
                        f"{location}.adapter_sha256 must be an exact lowercase SHA-256 pin"
                    )
    targets = _mapping(data.get("targets"), "targets", errors)
    for target_id, target_value in targets.items():
        target = _mapping(target_value, f"targets.{target_id}", errors)
        for key, collection in (
            ("model", models),
            ("harness", harnesses),
            ("domain", domains),
            ("split", splits),
        ):
            if target.get(key) not in collection:
                errors.append(f"targets.{target_id}.{key} references an unknown value")

    blocks = _mapping(data.get("task_seed_blocks"), "task_seed_blocks", errors)
    search_seeds = blocks.get("search_seeds", [])
    execution_seeds = blocks.get("execution_seeds", [])
    if not isinstance(search_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in search_seeds
    ):
        errors.append("search seeds must be an array of integers")
        search_seeds = []
    if not isinstance(execution_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in execution_seeds
    ):
        errors.append("execution seeds must be an array of integers")
        execution_seeds = []
    if len(set(search_seeds)) < 3:
        errors.append("at least three unique search seeds are required")
    if not execution_seeds or len(set(execution_seeds)) != len(execution_seeds):
        errors.append("execution seeds must be non-empty and unique")
    candidates_shared = blocks.get("shared_across_candidates") is True
    base_paired = blocks.get("paired_against_base") is True
    if not candidates_shared or not base_paired:
        errors.append("paired blocks must be shared across candidates and base")
    retry_limit = blocks.get("retry_limit")
    if (
        isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or retry_limit < 0
    ):
        errors.append("task_seed_blocks.retry_limit must be a non-negative integer")
    if blocks.get("infrastructure_retry_uses_same_seed") is not True:
        errors.append(
            "task_seed_blocks.infrastructure_retry_uses_same_seed must be true"
        )
    if blocks.get("exclude_only_verified_infrastructure_failures") is not True:
        errors.append(
            "task_seed_blocks.exclude_only_verified_infrastructure_failures must be true"
        )
    taxonomy = _mapping(
        blocks.get("failure_taxonomy"),
        "task_seed_blocks.failure_taxonomy",
        errors,
    )
    eligible = taxonomy.get("eligible_for_retry_and_exclusion")
    if not isinstance(eligible, list) or any(
        not isinstance(value, str) for value in eligible
    ):
        errors.append(
            "task_seed_blocks.failure_taxonomy.eligible_for_retry_and_exclusion "
            "must be a list of strings"
        )
    elif tuple(eligible) != ELIGIBLE_INFRASTRUCTURE_FAILURES:
        errors.append(
            "task_seed_blocks.failure_taxonomy.eligible_for_retry_and_exclusion "
            "must exactly match the four frozen eligible categories"
        )
    never_exclude = taxonomy.get("never_exclude_as_infrastructure")
    if not isinstance(never_exclude, list) or any(
        not isinstance(value, str) for value in never_exclude
    ):
        errors.append(
            "task_seed_blocks.failure_taxonomy.never_exclude_as_infrastructure "
            "must be a list of strings"
        )
    elif tuple(never_exclude) != NEVER_EXCLUDE_AS_INFRASTRUCTURE:
        errors.append(
            "task_seed_blocks.failure_taxonomy.never_exclude_as_infrastructure "
            "must exactly match the frozen never-exclude categories"
        )

    objectives = _mapping(data.get("objectives"), "objectives", errors)
    if set(objectives) != set(REQUIRED_OBJECTIVES):
        errors.append("the primary configuration must define exactly the four frozen objectives")
    for objective_id, (direction, bound) in REQUIRED_OBJECTIVES.items():
        objective = _mapping(objectives.get(objective_id), f"objectives.{objective_id}", errors)
        if objective.get("enabled") is not True:
            errors.append(f"objectives.{objective_id} must be enabled")
        if objective.get("direction") != direction or objective.get("pessimistic_bound") != bound:
            errors.append(f"objectives.{objective_id} has the wrong direction/bound")
        for target_id in objective.get("target_ids", []):
            if target_id not in targets:
                errors.append(f"objectives.{objective_id} references unknown target {target_id!r}")
    id_target_ids = set(objectives.get("id_accuracy", {}).get("target_ids", []))
    transfer_target_ids = set(
        objectives.get("worst_target_transfer", {}).get("target_ids", [])
    )
    if id_target_ids & transfer_target_ids:
        errors.append("ID and transfer objective target sets must be disjoint")
    for target_id, target_value in targets.items():
        if not isinstance(target_value, dict):
            continue
        if target_value.get("phase") not in {"search", "final_only"}:
            errors.append(f"targets.{target_id}.phase is invalid")
        transfer_group = target_value.get("transfer_group")
        if target_id in transfer_target_ids and (
            not isinstance(transfer_group, str) or not transfer_group
        ):
            errors.append(f"transfer target {target_id} requires transfer_group")
        if target_id in id_target_ids and transfer_group is not None:
            errors.append(f"ID target {target_id} must not define transfer_group")

    constraints = _mapping(data.get("constraints"), "constraints", errors)
    if not isinstance(constraints.get("enabled"), bool):
        errors.append("constraints.enabled must be boolean")
    accuracy_constraint = _mapping(
        constraints.get("id_accuracy_floor"), "constraints.id_accuracy_floor", errors
    )
    epsilon = accuracy_constraint.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        errors.append("constraints.id_accuracy_floor.epsilon must be numeric in [0, 1]")
    if not str(accuracy_constraint.get("expression", "")).strip():
        errors.append("constraints.id_accuracy_floor.expression must be non-empty")
    token_constraint = _mapping(
        constraints.get("token_budget"), "constraints.token_budget", errors
    )
    token_budget = token_constraint.get("budget")
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, (int, float))
        or not math.isfinite(float(token_budget))
        or not float(token_budget) >= 0.0
    ):
        errors.append("constraints.token_budget.budget must be non-negative")
    if not str(token_constraint.get("expression", "")).strip():
        errors.append("constraints.token_budget.expression must be non-empty")

    statistics = _mapping(data.get("statistics"), "statistics", errors)
    if statistics.get("paired_design") is not True:
        errors.append("statistics.paired_design must be true")
    replicates = statistics.get("bootstrap_replicates")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
    ):
        errors.append("statistics.bootstrap_replicates must be an integer >= 100")
    minimum_blocks = statistics.get("minimum_effective_blocks_for_archive")
    if (
        isinstance(minimum_blocks, bool)
        or not isinstance(minimum_blocks, int)
        or minimum_blocks < 2
    ):
        errors.append(
            "statistics.minimum_effective_blocks_for_archive must be an integer >= 2"
        )
    confidence = statistics.get("confidence_level")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        errors.append("statistics.confidence_level must be numeric in (0, 1)")
    dominance = _mapping(statistics.get("dominance"), "statistics.dominance", errors)
    if dominance.get("primary") not in {"pessimistic_bounds", "point_estimates"}:
        errors.append("statistics.dominance.primary has an unsupported value")
    if dominance.get("weak_all_strict_one") is not True:
        errors.append("statistics.dominance.weak_all_strict_one must be true")

    budgets = _mapping(data.get("budgets"), "budgets", errors)
    if budgets.get("accounting_unit") != "task_executions":
        errors.append("budget accounting must use task_executions")
    for stage_name in ("screen", "full"):
        stage = _mapping(budgets.get(stage_name), f"budgets.{stage_name}", errors)
        tasks_per_target = stage.get("tasks_per_target")
        execution_seeds_per_task = stage.get("execution_seeds_per_task")
        stage_targets = stage.get("target_ids")
        if (
            isinstance(tasks_per_target, bool)
            or not isinstance(tasks_per_target, int)
            or tasks_per_target < 1
            or isinstance(execution_seeds_per_task, bool)
            or not isinstance(execution_seeds_per_task, int)
            or execution_seeds_per_task < 1
            or not isinstance(stage_targets, list)
            or any(not isinstance(value, str) for value in stage_targets)
        ):
            errors.append(f"budgets.{stage_name} matrix fields have invalid types")
            continue
        expected = tasks_per_target * len(stage_targets) * execution_seeds_per_task
        if stage.get("task_executions") != expected:
            errors.append(f"budgets.{stage_name}.task_executions does not match its matrix")
    screen = _mapping(budgets.get("screen"), "budgets.screen", errors)
    full = _mapping(budgets.get("full"), "budgets.full", errors)
    if screen.get("is_subset_of_full_matrix") is not True:
        errors.append("the 40-execution screen must be declared as a subset of full")
    if full.get("task_executions_including_screen_subset") != full.get("task_executions"):
        errors.append("full budget must be total including the screen subset")
    if full.get("incremental_task_executions_after_screen") != (
        full.get("task_executions", 0) - screen.get("task_executions", 0)
    ):
        errors.append("full incremental budget must equal full total minus screen")
    search_total = _mapping(
        budgets.get("search_total_per_method"),
        "budgets.search_total_per_method",
        errors,
    )
    allocations = (
        search_total.get("screen_allocation_task_executions"),
        search_total.get("incremental_promotion_allocation_task_executions"),
    )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in allocations):
        if sum(allocations) != search_total.get("task_executions"):
            errors.append("search screen/promotion allocations must sum to the method budget")
    else:
        errors.append("search budget allocations must be integers")

    maximum_screened = search_total.get("maximum_unique_screened_candidates")
    maximum_promoted = search_total.get("maximum_promoted_candidates")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (maximum_screened, maximum_promoted)
    ):
        errors.append("search candidate and promotion limits must be positive integers")
    elif all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (screen.get("task_executions"), full.get("task_executions"))
    ):
        expected_screen_allocation = maximum_screened * screen["task_executions"]
        expected_promotion_allocation = maximum_promoted * (
            full["task_executions"] - screen["task_executions"]
        )
        if allocations[0] != expected_screen_allocation:
            errors.append(
                "search screen allocation must equal maximum screened candidates "
                "times the screen matrix"
            )
        if allocations[1] != expected_promotion_allocation:
            errors.append(
                "search promotion allocation must equal maximum promoted candidates "
                "times the incremental full matrix"
            )

    methods = _mapping(data.get("methods"), "methods", errors)
    if REQUIRED_METHODS - set(methods):
        errors.append(f"required methods are missing: {sorted(REQUIRED_METHODS - set(methods))}")
    for method_id, method_value in methods.items():
        method = _mapping(method_value, f"methods.{method_id}", errors)
        if not str(method.get("family", "")).strip():
            errors.append(f"methods.{method_id}.family must be non-empty")
        if not str(method.get("search", "")).strip():
            errors.append(f"methods.{method_id}.search must be non-empty")
        optimizer = method.get("optimizer")
        if isinstance(optimizer, str) and "_or_" in optimizer:
            errors.append(
                f"methods.{method_id}.optimizer must freeze one optimizer, not {optimizer!r}"
            )
    evo = _mapping(
        methods.get("evoskill_scalar_topk"),
        "methods.evoskill_scalar_topk",
        errors,
    )
    evo_top_k = evo.get("top_k")
    if (
        isinstance(evo_top_k, bool)
        or not isinstance(evo_top_k, int)
        or evo_top_k < 1
    ):
        errors.append("methods.evoskill_scalar_topk.top_k must be a positive integer")
    elif isinstance(maximum_promoted, int) and evo_top_k != maximum_promoted:
        errors.append(
            "EvoSkill matched adaptation top_k must equal the frozen promotion count"
        )
    fixed_method = _mapping(
        methods.get("fixed_scalarization"), "methods.fixed_scalarization", errors
    )
    variants = fixed_method.get("variants")
    if not isinstance(variants, list) or len(variants) < 4:
        errors.append("fixed_scalarization must declare at least four variants")
    else:
        variant_ids: set[str] = set()
        for index, raw_variant in enumerate(variants):
            variant = _mapping(
                raw_variant, f"methods.fixed_scalarization.variants[{index}]", errors
            )
            variant_id = variant.get("id")
            if not isinstance(variant_id, str) or not variant_id:
                errors.append(f"fixed scalar variant {index} must have an id")
                continue
            variant_ids.add(variant_id)
            if variant.get("logical_task_execution_budget_per_search_seed") != 30000:
                errors.append(
                    f"fixed scalar variant {variant_id} must receive its own 30000 budget"
                )
            if variant_id != "ctx2skill_hard_easy_product":
                weights = variant.get("maximize_space_weights")
                if not isinstance(weights, list) or len(weights) != 4 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in weights
                ):
                    errors.append(
                        f"fixed scalar variant {variant_id} needs four numeric weights"
                    )
        if {
            "accuracy_only",
            "accuracy_cost_equal",
            "balanced_four_objective",
            "ctx2skill_hard_easy_product",
        } - variant_ids:
            errors.append("the four frozen scalar variants are incomplete")
    mocha = _mapping(methods.get("mocha_chebyshev_hvc"), "methods.mocha", errors)
    mocha_protocol = _mapping(
        mocha.get("protocol"), "methods.mocha_chebyshev_hvc.protocol", errors
    )
    for key in ("dirichlet_alpha", "annealing_rate"):
        value = mocha_protocol.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(f"MOCHA {key} must be a positive finite number")
    ablations = data.get("ablations", [])
    if not isinstance(ablations, list):
        errors.append("ablations must be a list")
    else:
        ablation_ids = {
            item.get("id") for item in ablations if isinstance(item, dict) and item.get("id")
        }
        if REQUIRED_ABLATIONS - ablation_ids:
            errors.append(
                f"required ablations are missing: {sorted(REQUIRED_ABLATIONS - ablation_ids)}"
            )
        for index, item in enumerate(ablations):
            ablation = _mapping(item, f"ablations[{index}]", errors)
            if ablation.get("base_method") not in methods:
                errors.append(f"ablations[{index}].base_method references an unknown method")
            if not isinstance(ablation.get("overrides"), dict) or not ablation.get("overrides"):
                errors.append(f"ablations[{index}].overrides must be a non-empty mapping")

    raw_runner = data.get("runner")
    if raw_runner is not None:
        runner = _mapping(raw_runner, "runner", errors)
        raw_smoke = runner.get("smoke")
        if raw_smoke is not None:
            smoke = _mapping(raw_smoke, "runner.smoke", errors)
            smoke_methods = smoke.get("methods")
            valid_smoke_methods = (
                isinstance(smoke_methods, list)
                and bool(smoke_methods)
                and all(
                    isinstance(method_id, str) and bool(method_id)
                    for method_id in smoke_methods
                )
                and len(set(smoke_methods)) == len(smoke_methods)
            )
            if not valid_smoke_methods:
                errors.append(
                    "runner.smoke.methods must be a non-empty unique string list"
                )
                smoke_methods = []

            executable_methods: dict[str, str] = {}
            for method_id, method_value in methods.items():
                if method_id != "fixed_scalarization":
                    if isinstance(method_id, str):
                        executable_methods[method_id] = method_id
                    continue
                if not isinstance(method_value, Mapping):
                    continue
                raw_variants = method_value.get("variants", [])
                if not isinstance(raw_variants, list):
                    continue
                for raw_variant in raw_variants:
                    if not isinstance(raw_variant, Mapping):
                        continue
                    run_id = raw_variant.get("run_id")
                    variant_id = raw_variant.get("id")
                    if isinstance(run_id, str) and isinstance(variant_id, str):
                        executable_methods[run_id] = (
                            "ctx2skill_hard_easy_product"
                            if variant_id == "ctx2skill_hard_easy_product"
                            else f"fixed_scalarization/{variant_id}"
                        )
            if isinstance(ablations, list):
                for raw_ablation in ablations:
                    if not isinstance(raw_ablation, Mapping):
                        continue
                    ablation_id = raw_ablation.get("id")
                    base_method = raw_ablation.get("base_method")
                    if isinstance(ablation_id, str) and isinstance(base_method, str):
                        executable_methods[f"ablation_{ablation_id}"] = base_method
            if valid_smoke_methods:
                unknown_methods = sorted(
                    set(smoke_methods) - set(executable_methods)
                )
                if unknown_methods:
                    errors.append(
                        f"runner.smoke.methods reference unknown run IDs: {unknown_methods}"
                    )

            smoke_seeds = smoke.get("search_seeds")
            valid_smoke_seeds = (
                isinstance(smoke_seeds, list)
                and bool(smoke_seeds)
                and all(
                    isinstance(seed, int) and not isinstance(seed, bool)
                    for seed in smoke_seeds
                )
                and len(set(smoke_seeds)) == len(smoke_seeds)
            )
            if not valid_smoke_seeds:
                errors.append(
                    "runner.smoke.search_seeds must be a non-empty unique integer list"
                )
                smoke_seeds = []
            elif not set(smoke_seeds) <= set(search_seeds):
                errors.append(
                    "runner.smoke.search_seeds must be a subset of task_seed_blocks.search_seeds"
                )

            smoke_targets = smoke.get("search_targets")
            valid_smoke_targets = (
                isinstance(smoke_targets, list)
                and bool(smoke_targets)
                and all(
                    isinstance(target_id, str) and bool(target_id)
                    for target_id in smoke_targets
                )
                and len(set(smoke_targets)) == len(smoke_targets)
            )
            expected_search_targets = {
                target_id
                for target_id, target_value in targets.items()
                if isinstance(target_id, str)
                and isinstance(target_value, Mapping)
                and target_value.get("phase") == "search"
            }
            if not valid_smoke_targets:
                errors.append(
                    "runner.smoke.search_targets must be a non-empty unique string list"
                )
                smoke_targets = []
            elif set(smoke_targets) != expected_search_targets:
                errors.append(
                    "runner.smoke.search_targets must equal the configured search targets"
                )

            max_candidates = smoke.get("max_candidates")
            blocks_per_target = smoke.get("blocks_per_target")
            logical_ceiling = smoke.get("logical_task_execution_ceiling")
            if (
                isinstance(max_candidates, bool)
                or not isinstance(max_candidates, int)
                or max_candidates < 1
            ):
                errors.append("runner.smoke.max_candidates must be a positive integer")
            if (
                isinstance(blocks_per_target, bool)
                or not isinstance(blocks_per_target, int)
                or blocks_per_target < 2
            ):
                errors.append("runner.smoke.blocks_per_target must be an integer >= 2")
            if (
                isinstance(logical_ceiling, bool)
                or not isinstance(logical_ceiling, int)
                or logical_ceiling < 1
            ):
                errors.append(
                    "runner.smoke.logical_task_execution_ceiling must be a positive integer"
                )

            raw_candidate_limits = smoke.get("candidate_limits", {})
            candidate_limits: dict[str, int] = {}
            if not isinstance(raw_candidate_limits, Mapping):
                errors.append("runner.smoke.candidate_limits must be a mapping")
            elif isinstance(max_candidates, int) and not isinstance(
                max_candidates, bool
            ):
                for method_id, limit in raw_candidate_limits.items():
                    if not isinstance(method_id, str) or method_id not in smoke_methods:
                        errors.append(
                            "runner.smoke.candidate_limits keys must name declared smoke methods"
                        )
                    if (
                        isinstance(limit, bool)
                        or not isinstance(limit, int)
                        or limit < 1
                        or limit > max_candidates
                    ):
                        errors.append(
                            "runner.smoke.candidate_limits values must be positive "
                            "integers no greater than max_candidates"
                        )
                    elif isinstance(method_id, str):
                        candidate_limits[method_id] = limit

            namespace = smoke.get("separate_output_namespace")
            if (
                not isinstance(namespace, str)
                or not namespace
                or namespace != namespace.strip()
                or Path(namespace).name != namespace
                or namespace in {".", ".."}
            ):
                errors.append(
                    "runner.smoke.separate_output_namespace must be one safe path segment"
                )
            if smoke.get("never_promote_results_to_main_comparison") is not True:
                errors.append(
                    "runner.smoke.never_promote_results_to_main_comparison must be true"
                )

            if (
                valid_smoke_methods
                and not set(smoke_methods) - set(executable_methods)
                and valid_smoke_seeds
                and valid_smoke_targets
                and isinstance(max_candidates, int)
                and not isinstance(max_candidates, bool)
                and max_candidates >= 1
                and isinstance(blocks_per_target, int)
                and not isinstance(blocks_per_target, bool)
                and blocks_per_target >= 2
                and isinstance(logical_ceiling, int)
                and not isinstance(logical_ceiling, bool)
                and logical_ceiling >= 1
            ):
                matrix_size = len(smoke_targets) * blocks_per_target
                maximum_logical = 0
                for method_id in smoke_methods:
                    plugin_id = executable_methods[method_id]
                    run_count = (
                        1
                        if plugin_id in {"no_skill", "base_skill"}
                        else len(smoke_seeds)
                    )
                    if plugin_id == "base_skill":
                        candidate_limit = 0
                    elif plugin_id == "no_skill":
                        candidate_limit = 1
                    else:
                        candidate_limit = candidate_limits.get(
                            method_id, max_candidates
                        )
                    maximum_logical += (
                        run_count * candidate_limit * matrix_size
                    )
                if maximum_logical > logical_ceiling:
                    errors.append(
                        "runner.smoke declared matrix exceeds "
                        "logical_task_execution_ceiling: "
                        f"{maximum_logical}>{logical_ceiling}"
                    )

    selection = _mapping(data.get("selection_protocol"), "selection_protocol", errors)
    if selection.get("search_uses_final_splits") is not False:
        errors.append("selection_protocol.search_uses_final_splits must be false")
    archive_capacity = _mapping(
        selection.get("archive_capacity"), "selection_protocol.archive_capacity", errors
    )
    max_entries = archive_capacity.get("max_entries")
    if (
        isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries < 1
    ):
        errors.append("selection_protocol.archive_capacity.max_entries must be positive")
    working_archive = _mapping(
        selection.get("working_archive"), "selection_protocol.working_archive", errors
    )
    working_capacity = _mapping(
        working_archive.get("capacity"),
        "selection_protocol.working_archive.capacity",
        errors,
    )
    if working_capacity.get("max_entries") != max_entries:
        errors.append("working archive capacity and compatibility alias must match")
    working_admission = _mapping(
        working_archive.get("admission"),
        "selection_protocol.working_archive.admission",
        errors,
    )
    admission_alias = _mapping(
        selection.get("archive_admission"),
        "selection_protocol.archive_admission",
        errors,
    )
    for key in (
        "require_conservative_feasibility",
        "require_not_robustly_dominated",
        "content_hash_deduplicate",
    ):
        if working_admission.get(key) is not True or admission_alias.get(key) is not True:
            errors.append(f"archive admission {key} must be true in both declarations")
    scientific_front = _mapping(
        selection.get("scientific_front"), "selection_protocol.scientific_front", errors
    )
    if scientific_front.get("capacity", "missing") is not None:
        errors.append("scientific front must be explicitly unbounded")
    deployment = _mapping(data.get("deployment"), "deployment", errors)
    if deployment.get("never_tune_on_final_test") is not True:
        errors.append("deployment policies must never tune on final test")

    metrics = _mapping(data.get("metrics"), "metrics", errors)
    for group in ("primary", "frontier", "efficiency", "diagnostic"):
        values = metrics.get(group)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            errors.append(f"metrics.{group} must be a non-empty string array")

    controls = _mapping(
        data.get("shared_search_controls"), "shared_search_controls", errors
    )
    for key in ("base_skill", "trace_store", "patch_pool", "verifier"):
        if not isinstance(controls.get(key), str) or not controls.get(key):
            errors.append(f"shared_search_controls.{key} must be non-empty")
    proposer = _mapping(data.get("proposer"), "proposer", errors)
    for key in ("require_trace_evidence", "require_parent_version"):
        if proposer.get(key) is not True:
            errors.append(f"proposer.{key} must be true")

    outputs = _mapping(data.get("outputs"), "outputs", errors)
    if not isinstance(outputs.get("root"), str) or not outputs.get("root"):
        errors.append("outputs.root must be a non-empty path string")
    required_files = outputs.get("required_files")
    mandatory_artifacts = {
        "resolved_manifest.yaml",
        "run_metadata.json",
        "task_outcomes.jsonl",
        "candidates.jsonl",
        "archive.json",
        "scientific_front.json",
        "metrics.json",
        "lineage.jsonl",
        "token_accounting.json",
        "checkpoint.json",
    }
    if not isinstance(required_files, list) or not mandatory_artifacts <= set(
        value for value in required_files if isinstance(value, str)
    ):
        errors.append("outputs.required_files is missing required resumable artifacts")

    if profile == "real":
        real_profile = _mapping(profiles.get("real"), "runtime_profiles.real", errors)
        if real_profile.get("preflight_action") == "reject_checked_in_manifest":
            errors.append(
                "the checked-in manifest is structurally frozen for offline use; "
                "create a new versioned real manifest"
            )
        else:
            if experiment.get("mode") != "real":
                errors.append("real profile requires experiment.mode='real'")
            if real_profile.get("network") is not True:
                errors.append("an executable real profile must declare network=true")
            if safety.get("allow_network") is not True:
                errors.append("an executable real manifest must set safety.allow_network=true")

        def check_digests(value: Any, path: str = "") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    child = f"{path}.{key}" if path else str(key)
                    if key == "sha256" or str(key).endswith("_sha256"):
                        if not isinstance(item, str) or not re.fullmatch(
                            r"[0-9a-f]{64}", item
                        ):
                            errors.append(f"{child} must be a resolved lowercase SHA-256")
                    else:
                        check_digests(item, child)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    check_digests(item, f"{path}[{index}]")

        check_digests(data)

    if errors:
        raise ConfigError("invalid experiment manifest:\n- " + "\n- ".join(errors))


def validate_manifest(data: Mapping[str, Any], *, profile: str) -> None:
    """Validate structure and cross references without leaking YAML type errors."""

    try:
        _validate_manifest(data, profile=profile)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid experiment manifest value: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    data: Mapping[str, Any]
    source_path: Path
    profile: str
    unresolved_placeholders: tuple[str, ...]

    @property
    def code_root(self) -> Path:
        module_path = Path(__file__).resolve()
        source_root = module_path.parents[2]
        if (source_root / "pyproject.toml").is_file():
            return source_root
        # A normal wheel contains the package, not the repository-level
        # pyproject/schemas directories.  Treat the installed package directory
        # as the implementation root so non-editable installs remain runnable.
        return module_path.parent

    @property
    def implementation_digest(self) -> str:
        """Hash execution-affecting in-repository code and schemas."""

        code_root = self.code_root
        pyproject = code_root / "pyproject.toml"
        source_root = code_root / "src"
        schema_root = code_root / "schemas"
        if pyproject.is_file():
            paths = [pyproject, *source_root.rglob("*.py"), *schema_root.rglob("*.json")]
        else:
            paths = list(code_root.rglob("*.py"))
        if not paths:
            raise ConfigError("cannot locate installed ParetoSkill implementation files")
        digest = hashlib.sha256()
        for path in sorted(paths, key=lambda item: item.as_posix()):
            relative = path.relative_to(code_root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @property
    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "resolved_manifest": self.data,
                "implementation_digest": self.implementation_digest,
            }
        )

    @property
    def experiment_id(self) -> str:
        digest = hashlib.sha256(self.canonical_payload.encode("utf-8")).hexdigest()[:16]
        declared = str(self.data["experiment"]["id"])
        return f"{declared}-{digest}"

    @property
    def is_offline(self) -> bool:
        return not bool(self.data["runtime_profiles"][self.profile].get("network", False))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(dict(self.data), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )


def load_manifest(
    path: str | Path,
    *,
    profile: str = "dry_run",
    environment: Mapping[str, str] | None = None,
    require_all_placeholders: bool | None = None,
) -> ExperimentManifest:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("experiment YAML root must be a mapping")
    data = copy.deepcopy(raw)
    _apply_profile(data, profile)
    profile_settings = data["runtime_profiles"][profile]
    resolution_environment = (
        (os.environ if environment is None else environment)
        if profile_settings.get("resolve_external_placeholders", False)
        else {}
    )
    unresolved: list[str] = []
    data = _resolve_value(
        data,
        environment=resolution_environment,
        path="",
        unresolved=unresolved,
    )
    assert isinstance(data, dict)
    require = profile == "real" if require_all_placeholders is None else require_all_placeholders
    if require and unresolved:
        raise ConfigError(
            "unresolved placeholders are forbidden for this profile:\n- "
            + "\n- ".join(sorted(unresolved))
        )
    validate_manifest(data, profile=profile)
    return ExperimentManifest(
        data=data,
        source_path=source,
        profile=profile,
        unresolved_placeholders=tuple(sorted(unresolved)),
    )
