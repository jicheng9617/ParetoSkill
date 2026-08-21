"""Explicitly enabled, content-pinned local runtime adapter loading.

The default runtime never imports adapter references from a manifest.  A copied
real/replay manifest must opt in, allowlist the module namespace, and pin the
exact bytes of every module before its factory can be imported or invoked.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.machinery
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .evaluation import Harness, TaskSpec


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"\$\{[^}]+\}")
_REMOTE_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_BUILTIN_ADAPTERS = frozenset(
    {"builtin", "builtin_local_domain", "builtin_verified_response"}
)


class RuntimeAdapterError(RuntimeError):
    """A dynamic runtime adapter failed its safety or factory contract."""


Factory = Callable[..., Any]


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeAdapterError(f"{location} must be a mapping")
    return value


def _require_resolved(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeAdapterError(f"{location} keys must be strings")
            _require_resolved(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_resolved(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and _PLACEHOLDER.search(value):
        raise RuntimeAdapterError(f"{location} contains an unresolved placeholder")


def _identifier(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _PLACEHOLDER.search(value)
    ):
        raise RuntimeAdapterError(f"{location} must be a resolved non-empty string")
    return value


def _resolve_dataset_roots(
    runtime: Mapping[str, Any], *, base_directory: Path
) -> Mapping[str, str]:
    raw_roots = runtime.get("dataset_roots", {})
    if not isinstance(raw_roots, Mapping):
        raise RuntimeAdapterError("runtime.dataset_roots must be a mapping")
    resolved: dict[str, str] = {}
    for raw_id, raw_path in sorted(raw_roots.items(), key=lambda item: str(item[0])):
        dataset_id = _identifier(raw_id, location="runtime.dataset_roots key")
        path_value = _identifier(
            raw_path, location=f"runtime.dataset_roots.{dataset_id}"
        )
        if _REMOTE_PATH.match(path_value):
            raise RuntimeAdapterError(
                f"runtime.dataset_roots.{dataset_id} must be a local directory"
            )
        path = Path(path_value)
        path = path.resolve() if path.is_absolute() else (base_directory / path).resolve()
        if not path.is_dir():
            raise RuntimeAdapterError(
                f"runtime.dataset_roots.{dataset_id} is not an existing directory"
            )
        resolved[dataset_id] = str(path)
    return MappingProxyType(resolved)


def _parse_reference(reference: str, *, location: str) -> tuple[str, str]:
    if reference.count(":") != 1:
        raise RuntimeAdapterError(f"{location} must use 'module:factory'")
    module_name, attribute_path = reference.split(":", 1)
    parts = (*module_name.split("."), *attribute_path.split("."))
    if not module_name or not attribute_path or not all(part.isidentifier() for part in parts):
        raise RuntimeAdapterError(f"{location} has an invalid module or factory name")
    return module_name, attribute_path


def _find_local_module_spec(module_name: str) -> importlib.machinery.ModuleSpec | None:
    """Locate dotted filesystem modules without importing their parent packages."""

    search_path: Iterable[str] | None = None
    qualified = ""
    module_spec: importlib.machinery.ModuleSpec | None = None
    parts = module_name.split(".")
    for index, part in enumerate(parts):
        qualified = part if not qualified else f"{qualified}.{part}"
        module_spec = importlib.machinery.PathFinder.find_spec(qualified, search_path)
        if module_spec is None:
            return None
        if index < len(parts) - 1:
            locations = module_spec.submodule_search_locations
            if locations is None:
                return None
            search_path = tuple(locations)
    return module_spec


def _load_pinned_factory(
    *,
    reference: str,
    expected_sha256: Any,
    prefixes: tuple[str, ...],
    location: str,
) -> Factory:
    module_name, attribute_path = _parse_reference(reference, location=location)
    if not any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in prefixes
    ):
        raise RuntimeAdapterError(f"{location} module is outside the allowlist")
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise RuntimeAdapterError(f"{location}_sha256 must be a lowercase SHA-256 pin")
    try:
        module_spec = _find_local_module_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeAdapterError(f"{location} module cannot be located") from exc
    if module_spec is None or module_spec.origin is None:
        raise RuntimeAdapterError(f"{location} module cannot be located")
    origin = Path(module_spec.origin).resolve()
    if not origin.is_file():
        raise RuntimeAdapterError(f"{location} module must resolve to a local file")
    actual_sha256 = hashlib.sha256(origin.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeAdapterError(f"{location} module SHA-256 does not match")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeAdapterError(
            f"{location} module import failed: {type(exc).__name__}"
        ) from exc
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != origin:
        raise RuntimeAdapterError(f"{location} imported module origin changed")
    factory: Any = module
    for part in attribute_path.split("."):
        if not hasattr(factory, part):
            raise RuntimeAdapterError(f"{location} factory attribute is missing")
        factory = getattr(factory, part)
    if not callable(factory):
        raise RuntimeAdapterError(f"{location} factory must be callable")
    return factory


@dataclass(frozen=True, slots=True)
class RuntimeAdapterRegistry:
    """Loaded factories plus the immutable resolved context passed to them."""

    domain_factories: Mapping[str, Factory]
    harness_factories: Mapping[str, Factory]
    domain_specs: Mapping[str, Mapping[str, Any]]
    harness_specs: Mapping[str, Mapping[str, Any]]
    runtime_spec: Mapping[str, Any]
    dataset_roots: Mapping[str, str]
    base_directory: Path

    @property
    def enabled(self) -> bool:
        return bool(self.domain_factories or self.harness_factories)

    def adapt_domain(
        self,
        *,
        domain_id: str,
        tasks: Iterable[TaskSpec],
        phase: str | None,
    ) -> tuple[TaskSpec, ...]:
        source_tasks = tuple(tasks)
        factory = self.domain_factories.get(domain_id)
        if factory is None:
            return source_tasks
        try:
            produced = factory(
                domain_id=domain_id,
                tasks=source_tasks,
                spec=self.domain_specs[domain_id],
                runtime_spec=self.runtime_spec,
                dataset_roots=self.dataset_roots,
                base_directory=self.base_directory,
                phase=phase,
            )
            if isinstance(produced, TaskSpec):
                result = (produced,)
            else:
                result = tuple(produced)
        except Exception as exc:
            raise RuntimeAdapterError(
                f"domain adapter factory for {domain_id!r} failed: {type(exc).__name__}"
            ) from exc
        if not result:
            raise RuntimeAdapterError(
                f"domain adapter factory for {domain_id!r} returned no tasks"
            )
        if any(not isinstance(task, TaskSpec) for task in result):
            raise RuntimeAdapterError(
                f"domain adapter factory for {domain_id!r} must return TaskSpec values"
            )
        if any(task.domain_id != domain_id for task in result):
            raise RuntimeAdapterError(
                f"domain adapter factory for {domain_id!r} returned another domain"
            )
        identities = [
            (task.split_id, task.task_id, task.objective_role, task.group_id)
            for task in result
        ]
        if len(set(identities)) != len(identities):
            raise RuntimeAdapterError(
                f"domain adapter factory for {domain_id!r} returned duplicate tasks"
            )
        return result

    def build_harness(
        self,
        *,
        harness_id: str,
        default_harness: Harness,
        verifiers: Mapping[str, Any],
        phase: str | None,
    ) -> Harness:
        factory = self.harness_factories.get(harness_id)
        if factory is None:
            return default_harness
        try:
            harness = factory(
                harness_id=harness_id,
                default_harness=default_harness,
                verifiers=verifiers,
                spec=self.harness_specs[harness_id],
                runtime_spec=self.runtime_spec,
                dataset_roots=self.dataset_roots,
                base_directory=self.base_directory,
                phase=phase,
            )
        except Exception as exc:
            raise RuntimeAdapterError(
                f"harness adapter factory for {harness_id!r} failed: {type(exc).__name__}"
            ) from exc
        if getattr(harness, "harness_id", None) != harness_id:
            raise RuntimeAdapterError(
                f"harness adapter factory for {harness_id!r} returned the wrong id"
            )
        if not callable(getattr(harness, "evaluate", None)):
            raise RuntimeAdapterError(
                f"harness adapter factory for {harness_id!r} must return a Harness"
            )
        return harness


def load_runtime_adapter_registry(
    configuration: Mapping[str, Any],
    *,
    domain_ids: Iterable[str],
    harness_ids: Iterable[str],
    base_directory: str | Path,
) -> RuntimeAdapterRegistry:
    """Load only explicitly enabled and required runtime adapter factories."""

    root = Path(base_directory).resolve()
    safety = _mapping(configuration.get("safety", {}), location="safety")
    runtime = _mapping(configuration.get("runtime", {}), location="runtime")
    empty = RuntimeAdapterRegistry(
        domain_factories=MappingProxyType({}),
        harness_factories=MappingProxyType({}),
        domain_specs=MappingProxyType({}),
        harness_specs=MappingProxyType({}),
        runtime_spec=MappingProxyType({}),
        dataset_roots=MappingProxyType({}),
        base_directory=root,
    )
    if safety.get("allow_dynamic_runtime_adapter_imports") is not True:
        return empty
    profile = configuration.get("active_runtime_profile")
    if profile not in {"real", "replay"}:
        raise RuntimeAdapterError(
            "dynamic runtime adapters require active_runtime_profile real or replay"
        )
    raw_prefixes = safety.get("allowed_dynamic_runtime_adapter_prefixes")
    if (
        not isinstance(raw_prefixes, list)
        or not raw_prefixes
        or any(not isinstance(prefix, str) or not prefix.strip() for prefix in raw_prefixes)
    ):
        raise RuntimeAdapterError(
            "dynamic runtime adapters require non-empty allowed module prefixes"
        )
    prefixes = tuple(raw_prefixes)
    _require_resolved(runtime, location="runtime")
    dataset_roots = _resolve_dataset_roots(runtime, base_directory=root)
    runtime_copy = copy.deepcopy(dict(runtime))
    runtime_copy["dataset_roots"] = dict(dataset_roots)

    loaded: dict[tuple[str, str], Factory] = {}

    def factories_for(
        collection_name: str, identifiers: Iterable[str]
    ) -> tuple[dict[str, Factory], dict[str, Mapping[str, Any]]]:
        collection = _mapping(
            configuration.get(collection_name), location=collection_name
        )
        factories: dict[str, Factory] = {}
        specs: dict[str, Mapping[str, Any]] = {}
        for adapter_id in sorted(set(identifiers)):
            _identifier(adapter_id, location=f"{collection_name} adapter id")
            raw_spec = _mapping(
                collection.get(adapter_id),
                location=f"{collection_name}.{adapter_id}",
            )
            _require_resolved(raw_spec, location=f"{collection_name}.{adapter_id}")
            reference = _identifier(
                raw_spec.get("adapter"),
                location=f"{collection_name}.{adapter_id}.adapter",
            )
            if reference in _BUILTIN_ADAPTERS:
                continue
            expected_sha256 = raw_spec.get("adapter_sha256")
            cache_key = (reference, str(expected_sha256))
            factory = loaded.get(cache_key)
            if factory is None:
                factory = _load_pinned_factory(
                    reference=reference,
                    expected_sha256=expected_sha256,
                    prefixes=prefixes,
                    location=f"{collection_name}.{adapter_id}.adapter",
                )
                loaded[cache_key] = factory
            factories[adapter_id] = factory
            specs[adapter_id] = MappingProxyType(copy.deepcopy(dict(raw_spec)))
        return factories, specs

    domain_factories, domain_specs = factories_for("domains", domain_ids)
    harness_factories, harness_specs = factories_for("harnesses", harness_ids)
    return RuntimeAdapterRegistry(
        domain_factories=MappingProxyType(domain_factories),
        harness_factories=MappingProxyType(harness_factories),
        domain_specs=MappingProxyType(domain_specs),
        harness_specs=MappingProxyType(harness_specs),
        runtime_spec=MappingProxyType(runtime_copy),
        dataset_roots=dataset_roots,
        base_directory=root,
    )


__all__ = [
    "RuntimeAdapterError",
    "RuntimeAdapterRegistry",
    "load_runtime_adapter_registry",
]
