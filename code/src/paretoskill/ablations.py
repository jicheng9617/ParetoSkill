"""Composable configuration-override plugins for the frozen ablation matrix."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class AblationPlugin(Protocol):
    plugin_id: str

    def apply(self, configuration: Mapping[str, Any]) -> dict[str, Any]: ...


def _assign_dotted(configuration: dict[str, Any], path: str, value: Any) -> None:
    current = configuration
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"ablation path does not resolve to a mapping: {path}")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"ablation path does not exist: {path}")
    current[parts[-1]] = copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class OverrideAblation:
    plugin_id: str
    overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.plugin_id or not self.overrides:
            raise ValueError("an ablation must have an id and at least one override")

    def apply(self, configuration: Mapping[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(dict(configuration))
        for path, value in sorted(self.overrides.items()):
            _assign_dotted(updated, str(path), value)
        updated["active_ablation"] = self.plugin_id
        return updated


@dataclass(slots=True)
class AblationRegistry:
    plugins: dict[str, AblationPlugin] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, ablations: list[Mapping[str, Any]]) -> AblationRegistry:
        registry = cls()
        for specification in ablations:
            plugin = OverrideAblation(
                plugin_id=str(specification["id"]),
                overrides=specification["overrides"],
            )
            registry.register(plugin)
        return registry

    def register(self, plugin: AblationPlugin) -> None:
        if plugin.plugin_id in self.plugins:
            raise ValueError(f"duplicate ablation plugin: {plugin.plugin_id}")
        self.plugins[plugin.plugin_id] = plugin

    def apply(self, plugin_id: str, configuration: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.plugins[plugin_id].apply(configuration)
        except KeyError as exc:
            raise ValueError(f"unknown ablation plugin: {plugin_id}") from exc
