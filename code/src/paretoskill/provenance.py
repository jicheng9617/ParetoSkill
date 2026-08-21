"""Local content-pin verification for reproducible real-run preflight."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    pass


def content_sha256(path: str | Path) -> str:
    """Hash a file or a canonical relative-path/bytes directory tree."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise ProvenanceError(f"content pins may not target symlinks: {candidate}")
    source = candidate.resolve()
    if source.is_file():
        return hashlib.sha256(source.read_bytes()).hexdigest()
    if not source.is_dir():
        raise ProvenanceError(f"pinned content path does not exist: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    for item in files:
        if item.is_symlink():
            raise ProvenanceError(f"pinned directory contains a symlink: {item}")
        relative = item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve(path: str, base_directory: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_directory / candidate


def _pin(
    container: Mapping[str, Any],
    *,
    path_key: str,
    digest_key: str,
    label: str,
    base_directory: Path,
) -> tuple[str, Path, str]:
    raw_path = container.get(path_key)
    raw_digest = container.get(digest_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ProvenanceError(f"{label}.{path_key} must be a resolved path")
    if (
        not isinstance(raw_digest, str)
        or len(raw_digest) != 64
        or any(character not in "0123456789abcdef" for character in raw_digest)
    ):
        raise ProvenanceError(f"{label}.{digest_key} must be lowercase SHA-256")
    return f"{label}.{path_key}", _resolve(raw_path, base_directory), raw_digest


def declared_local_pins(
    configuration: Mapping[str, Any], *, base_directory: str | Path
) -> tuple[tuple[str, Path, str], ...]:
    """Collect execution-affecting paths whose bytes are locally verifiable."""

    root = Path(base_directory).resolve()
    pins: list[tuple[str, Path, str]] = []
    controls = configuration.get("shared_search_controls")
    if not isinstance(controls, Mapping):
        raise ProvenanceError("shared_search_controls must be a mapping")
    for path_key, digest_key in (
        ("base_skill", "base_skill_sha256"),
        ("trace_store", "trace_store_sha256"),
        ("patch_pool", "patch_pool_sha256"),
    ):
        pins.append(
            _pin(
                controls,
                path_key=path_key,
                digest_key=digest_key,
                label="shared_search_controls",
                base_directory=root,
            )
        )

    reproducibility = configuration.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise ProvenanceError("reproducibility must be a mapping")
    dependency_lock = reproducibility.get("dependency_lock")
    if not isinstance(dependency_lock, Mapping):
        raise ProvenanceError("reproducibility.dependency_lock must be a mapping")
    pins.append(
        _pin(
            dependency_lock,
            path_key="path",
            digest_key="sha256",
            label="reproducibility.dependency_lock",
            base_directory=root,
        )
    )

    proposer = configuration.get("proposer")
    if not isinstance(proposer, Mapping):
        raise ProvenanceError("proposer must be a mapping")
    pins.append(
        _pin(
            proposer,
            path_key="prompt_template",
            digest_key="prompt_sha256",
            label="proposer",
            base_directory=root,
        )
    )
    return tuple(pins)


def verify_local_content_pins(
    configuration: Mapping[str, Any], *, base_directory: str | Path
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for label, path, expected in declared_local_pins(
        configuration, base_directory=base_directory
    ):
        observed = content_sha256(path)
        if observed != expected:
            raise ProvenanceError(
                f"{label} content digest mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )
        verified[label] = observed
    return verified


def digest_manifest_files(paths: Iterable[str | Path]) -> str:
    """Combine already-small task manifests into one deterministic digest."""

    digest = hashlib.sha256()
    resolved = (Path(value).resolve() for value in paths)
    for path in sorted(resolved, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
