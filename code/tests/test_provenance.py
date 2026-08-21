from __future__ import annotations

from pathlib import Path

import pytest

from paretoskill.provenance import (
    ProvenanceError,
    content_sha256,
    verify_local_content_pins,
)


def test_file_and_directory_content_pins_are_deterministic(tmp_path: Path) -> None:
    file_path = tmp_path / "records.jsonl"
    file_path.write_text("{}\n", encoding="utf-8")
    directory = tmp_path / "skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# Demo\n", encoding="utf-8")

    first = content_sha256(directory)
    second = content_sha256(directory)

    assert first == second
    assert first != content_sha256(file_path)


def test_declared_local_pins_are_verified_against_bytes(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name in ("base", "traces", "patches", "lock", "prompt"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    configuration = {
        "shared_search_controls": {
            "base_skill": str(paths["base"]),
            "base_skill_sha256": content_sha256(paths["base"]),
            "trace_store": str(paths["traces"]),
            "trace_store_sha256": content_sha256(paths["traces"]),
            "patch_pool": str(paths["patches"]),
            "patch_pool_sha256": content_sha256(paths["patches"]),
        },
        "reproducibility": {
            "dependency_lock": {
                "path": str(paths["lock"]),
                "sha256": content_sha256(paths["lock"]),
            }
        },
        "proposer": {
            "prompt_template": str(paths["prompt"]),
            "prompt_sha256": content_sha256(paths["prompt"]),
        },
    }

    verified = verify_local_content_pins(configuration, base_directory=tmp_path)
    assert len(verified) == 5

    paths["patches"].write_text("changed", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="digest mismatch"):
        verify_local_content_pins(configuration, base_directory=tmp_path)
