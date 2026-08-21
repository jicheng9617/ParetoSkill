"""Build a configured experiment runtime without performing provider calls."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import ExperimentManifest
from .experiment_runner import ExperimentRuntime, PhaseRuntime
from .models import PatchOperation, SkillVersion
from .proposer import MutationProposer, ProviderMutationProposer
from .providers import ModelSpec, NetworkPolicy, Provider, ProviderError, build_provider
from .runtime_assets import load_base_skill, load_patch_pool, load_trace_evidence
from .runtime_data import RuntimeDataError, load_runtime_matrix
from .search_strategies import BinarySubsetBayesianAdapter


class RuntimeFactoryError(RuntimeError):
    """Raised when resolved configuration cannot produce an execution runtime."""


def _resolved_path(value: Any, *, base_directory: Path, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeFactoryError(f"{location} must be a non-empty local path")
    source = Path(value)
    return source.resolve() if source.is_absolute() else (base_directory / source).resolve()


def _provider_factory_spec(provider_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    nested = raw.get("factory")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise RuntimeFactoryError(f"providers.{provider_id}.factory must be a mapping")
        return dict(nested)
    kind = raw.get("type", raw.get("kind"))
    if kind == "mock":
        return {"type": "mock", "provider_id": provider_id}
    if kind == "replay":
        return {
            "type": "replay",
            "provider_id": provider_id,
            "path": raw.get("path", raw.get("replay_path")),
        }
    if kind == "openai_compatible":
        allowed = {
            "base_url",
            "api_key_env",
            "timeout_seconds",
            "max_retries",
            "retry_backoff_seconds",
            "max_input_tokens",
            "max_output_tokens",
            "max_response_bytes",
            "allow_insecure_http",
        }
        return {
            "type": "openai_compatible",
            "provider_id": provider_id,
            **{name: raw[name] for name in allowed if name in raw},
        }
    raise RuntimeFactoryError(
        f"provider {provider_id!r} has no executable factory (kind={kind!r})"
    )


def _active_configuration(manifest: ExperimentManifest) -> dict[str, Any]:
    configuration = copy.deepcopy(dict(manifest.data))
    profile = configuration["runtime_profiles"][manifest.profile]
    provider_override = profile.get("provider_override")
    if provider_override in {None, ""}:
        return configuration
    if provider_override not in configuration["providers"]:
        raise RuntimeFactoryError(
            f"runtime provider_override references unknown provider {provider_override!r}"
        )
    # Replay must preserve model ids/revisions and request payloads while routing
    # every execution through the frozen local replay provider.
    if manifest.profile == "replay":
        for model in configuration["models"].values():
            if not isinstance(model, dict):
                raise RuntimeFactoryError("model specifications must be mappings")
            model["provider"] = provider_override
    return configuration


def _build_providers(
    configuration: Mapping[str, Any], *, policy: NetworkPolicy
) -> dict[str, Provider]:
    raw_providers = configuration.get("providers")
    raw_models = configuration.get("models")
    if not isinstance(raw_providers, Mapping) or not isinstance(raw_models, Mapping):
        raise RuntimeFactoryError("configuration providers/models must be mappings")
    required_ids = {
        model.get("provider")
        for model in raw_models.values()
        if isinstance(model, Mapping)
    }
    if any(not isinstance(provider_id, str) for provider_id in required_ids):
        raise RuntimeFactoryError("every model must name a provider")
    result: dict[str, Provider] = {}
    for provider_id in sorted(required_ids):
        assert isinstance(provider_id, str)
        raw = raw_providers.get(provider_id)
        if not isinstance(raw, Mapping):
            raise RuntimeFactoryError(f"provider {provider_id!r} is missing")
        try:
            result[provider_id] = build_provider(
                _provider_factory_spec(provider_id, raw),
                policy=policy,
            )
        except (ProviderError, TypeError, ValueError) as exc:
            raise RuntimeFactoryError(
                f"cannot construct provider {provider_id!r}: {exc}"
            ) from exc
    return result


def _proposer_factory(
    configuration: Mapping[str, Any],
    *,
    providers: Mapping[str, Provider],
    experiment_id: str,
    base_directory: Path,
) -> Callable[[Callable[[str], SkillVersion], int], MutationProposer]:
    proposer = configuration.get("proposer")
    models = configuration.get("models")
    if not isinstance(proposer, Mapping) or not isinstance(models, Mapping):
        raise RuntimeFactoryError("configuration proposer/models must be mappings")
    model_key = proposer.get("model")
    raw_model = models.get(model_key) if isinstance(model_key, str) else None
    if not isinstance(raw_model, Mapping):
        raise RuntimeFactoryError("proposer.model references an unknown model")
    provider_id = raw_model.get("provider")
    if not isinstance(provider_id, str) or provider_id not in providers:
        raise RuntimeFactoryError("proposer model provider is unavailable")
    prompt_path = _resolved_path(
        proposer.get("prompt_template"),
        base_directory=base_directory,
        location="proposer.prompt_template",
    )
    if not prompt_path.is_file():
        raise RuntimeFactoryError(f"proposer prompt does not exist: {prompt_path}")
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeFactoryError("proposer prompt must be UTF-8 text") from exc
    if not prompt.strip():
        raise RuntimeFactoryError("proposer prompt may not be empty")
    decoding = raw_model.get("decoding", {})
    if not isinstance(decoding, Mapping):
        raise RuntimeFactoryError("proposer model decoding must be a mapping")
    model = ModelSpec(
        model_id=str(raw_model["model_id"]),
        provider_id=provider_id,
        revision=str(raw_model["revision"]),
        decoding=decoding,
    )
    raw_operations = proposer.get("allowed_operations", [item.value for item in PatchOperation])
    if not isinstance(raw_operations, list):
        raise RuntimeFactoryError("proposer.allowed_operations must be an array")
    try:
        operations = tuple(PatchOperation(item) for item in raw_operations)
    except ValueError as exc:
        raise RuntimeFactoryError("proposer.allowed_operations contains an invalid value") from exc

    def factory(
        parent_resolver: Callable[[str], SkillVersion],
        seed: int,
        effective_configuration: Mapping[str, Any] | None = None,
        proposal_cache_directory: Path | None = None,
    ) -> MutationProposer:
        active = configuration if effective_configuration is None else effective_configuration
        active_proposer = active.get("proposer", proposer)
        active_controls = active.get("shared_search_controls", {})
        if not isinstance(active_proposer, Mapping) or not isinstance(
            active_controls, Mapping
        ):
            raise RuntimeFactoryError("effective proposer/search controls are malformed")
        active_operations = active_controls.get(
            "patch_operations", [operation.value for operation in operations]
        )
        if not isinstance(active_operations, list):
            raise RuntimeFactoryError("effective patch_operations must be an array")
        return ProviderMutationProposer(
            proposer_id="paretoskill-proposer",
            experiment_id=experiment_id,
            provider=providers[provider_id],  # type: ignore[arg-type]
            model=model,
            parent_resolver=parent_resolver,
            prompt_template=prompt,
            allowed_operations=tuple(PatchOperation(item) for item in active_operations),
            seed=seed,
            include_evidence_details=(
                active_proposer.get("verifier_evidence_visible") is not False
            ),
            include_parent_lineage=(
                active_proposer.get("parent_lineage_visible") is not False
            ),
            include_ancestral_patch_history=(
                active_proposer.get("ancestral_patch_history_visible") is not False
            ),
            cache_directory=proposal_cache_directory,
        )

    return factory


def _binary_optimizer_factory(
    configuration: Mapping[str, Any],
) -> Callable[
    [tuple[str, ...], int, Mapping[str, Any]], BinarySubsetBayesianAdapter
] | None:
    """Load an explicitly allowlisted, content-pinned local BO adapter factory."""

    methods = configuration.get("methods")
    safety = configuration.get("safety")
    if methods is None:
        return None
    if not isinstance(methods, Mapping) or not isinstance(safety, Mapping):
        raise RuntimeFactoryError("configuration methods/safety must be mappings")
    method = methods.get("trace2skill_accuracy_subset")
    if method is None:
        return None
    if not isinstance(method, Mapping):
        raise RuntimeFactoryError("trace2skill_accuracy_subset method is malformed")
    reference = method.get("optimizer_adapter")
    if not isinstance(reference, str) or not reference.strip():
        return None
    if safety.get("allow_dynamic_optimizer_imports") is not True:
        return None
    prefixes = safety.get("allowed_dynamic_optimizer_prefixes")
    if not isinstance(prefixes, list) or not prefixes or any(
        not isinstance(prefix, str) or not prefix.strip() for prefix in prefixes
    ):
        raise RuntimeFactoryError(
            "dynamic optimizer imports require non-empty allowed module prefixes"
        )
    if reference.count(":") != 1:
        raise RuntimeFactoryError(
            "optimizer_adapter must use the explicit 'module:factory' form"
        )
    module_name, attribute_path = reference.split(":", 1)
    if not module_name or not attribute_path or not all(
        part.isidentifier() for part in (*module_name.split("."), *attribute_path.split("."))
    ):
        raise RuntimeFactoryError("optimizer_adapter contains an invalid module/attribute")
    if not any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in prefixes
    ):
        raise RuntimeFactoryError("optimizer adapter module is outside the allowlist")
    expected_sha = method.get("optimizer_adapter_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise RuntimeFactoryError("optimizer adapter requires a lowercase SHA-256 pin")
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeFactoryError("optimizer adapter module cannot be located")
    origin = Path(spec.origin).resolve()
    if not origin.is_file():
        raise RuntimeFactoryError("optimizer adapter must resolve to a local file")
    actual_sha = hashlib.sha256(origin.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeFactoryError("optimizer adapter module SHA-256 does not match")
    module = importlib.import_module(module_name)
    factory_object: Any = module
    for part in attribute_path.split("."):
        if not hasattr(factory_object, part):
            raise RuntimeFactoryError("optimizer adapter factory attribute is missing")
        factory_object = getattr(factory_object, part)
    if not callable(factory_object):
        raise RuntimeFactoryError("optimizer adapter factory must be callable")

    def factory(
        patch_ids: tuple[str, ...],
        seed: int,
        method_spec: Mapping[str, Any],
    ) -> BinarySubsetBayesianAdapter:
        try:
            adapter = factory_object(
                patch_ids=patch_ids,
                seed=seed,
                method_spec=method_spec,
            )
        except Exception as exc:
            raise RuntimeFactoryError(
                f"optimizer adapter factory failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(adapter, BinarySubsetBayesianAdapter):
            raise RuntimeFactoryError(
                "optimizer adapter factory did not return BinarySubsetBayesianAdapter"
            )
        return adapter

    return factory


def _phase_runtime(
    configuration: Mapping[str, Any],
    *,
    phase: str,
    base_directory: Path,
) -> PhaseRuntime:
    """Build one closed phase, preserving any pinned custom harness instances."""

    matrix = load_runtime_matrix(
        configuration,
        phase=phase,
        base_directory=base_directory,
    )
    return PhaseRuntime(
        targets=matrix.targets,
        blocks=matrix.blocks,
        harnesses=matrix.harnesses,
    )


def build_experiment_runtime(
    manifest: ExperimentManifest,
    *,
    policy: NetworkPolicy,
    include_search: bool = True,
    include_final: bool = False,
) -> ExperimentRuntime:
    """Build all local inputs/adapters; no provider method is called here."""

    configuration = _active_configuration(manifest)
    base_directory = manifest.source_path.parent
    runtime = configuration.get("runtime", {})
    controls = configuration.get("shared_search_controls", {})
    if not isinstance(runtime, Mapping) or not isinstance(controls, Mapping):
        raise RuntimeFactoryError("runtime/shared_search_controls must be mappings")
    base_path = runtime.get("base_skill_path", controls.get("base_skill"))
    trace_path = runtime.get("trace_store_path", controls.get("trace_store"))
    patch_path = runtime.get("patch_pool_path", controls.get("patch_pool"))
    base = load_base_skill(
        _resolved_path(
            base_path,
            base_directory=base_directory,
            location="runtime.base_skill_path",
        )
    )
    evidence = load_trace_evidence(
        _resolved_path(
            trace_path,
            base_directory=base_directory,
            location="runtime.trace_store_path",
        )
    )
    patches = load_patch_pool(
        _resolved_path(
            patch_path,
            base_directory=base_directory,
            location="runtime.patch_pool_path",
        ),
        base_version_id=base.lineage.version_id,
        evidence=evidence,
    )
    providers = _build_providers(configuration, policy=policy)
    phases: dict[str, PhaseRuntime] = {}
    try:
        if include_search:
            phases["search"] = _phase_runtime(
                configuration,
                phase="search",
                base_directory=base_directory,
            )
        if include_final:
            phases["final"] = _phase_runtime(
                configuration,
                phase="final_only",
                base_directory=base_directory,
            )
    except RuntimeDataError as exc:
        raise RuntimeFactoryError(f"cannot build runtime task matrix: {exc}") from exc
    proposer_factory = _proposer_factory(
        configuration,
        providers=providers,
        experiment_id=manifest.experiment_id,
        base_directory=base_directory,
    )
    binary_optimizer_factory = (
        _binary_optimizer_factory(configuration) if include_search else None
    )
    return ExperimentRuntime(
        base=base,
        patches=patches,
        evidence=evidence,
        providers=providers,
        phases=phases,
        proposer_factory=proposer_factory,
        binary_optimizer_factory=binary_optimizer_factory,
    )
