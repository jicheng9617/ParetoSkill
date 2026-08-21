from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

from paretoskill.config import ExperimentManifest
from paretoskill.providers import NetworkPolicy, ReplayProvider
from paretoskill.runtime_factory import (
    RuntimeFactoryError,
    _binary_optimizer_factory,
    build_experiment_runtime,
)
from paretoskill.search_strategies import BinarySubsetBayesianAdapter


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optimizer_configuration(
    *,
    reference: str,
    sha256: str = "0" * 64,
    allow_imports: bool | None = True,
    prefixes: list[str] | None = None,
) -> dict:
    safety: dict = {
        "allowed_dynamic_optimizer_prefixes": prefixes or ["approved_optimizer"]
    }
    if allow_imports is not None:
        safety["allow_dynamic_optimizer_imports"] = allow_imports
    return {
        "methods": {
            "trace2skill_accuracy_subset": {
                "optimizer_adapter": reference,
                "optimizer_adapter_sha256": sha256,
            }
        },
        "safety": safety,
    }


@pytest.mark.parametrize("allow_imports", [None, False])
def test_binary_optimizer_factory_is_disabled_without_explicit_true_flag(
    allow_imports,
):
    configuration = _optimizer_configuration(
        reference="module_that_must_not_be_imported:build_adapter",
        allow_imports=allow_imports,
    )

    assert _binary_optimizer_factory(configuration) is None


def test_binary_optimizer_factory_defaults_to_none_without_method_configuration():
    assert _binary_optimizer_factory({}) is None


def test_binary_optimizer_factory_rejects_module_outside_allowlist():
    configuration = _optimizer_configuration(
        reference="unapproved_optimizer:build_adapter",
        prefixes=["approved_optimizer"],
    )

    with pytest.raises(RuntimeFactoryError, match="outside the allowlist"):
        _binary_optimizer_factory(configuration)


def test_binary_optimizer_factory_rejects_module_sha_mismatch(
    tmp_path,
    monkeypatch,
):
    module_name = "paretoskill_test_bad_sha_optimizer"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "def build_adapter(**kwargs):\n    raise AssertionError('must not import')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    configuration = _optimizer_configuration(
        reference=f"{module_name}:build_adapter",
        sha256="0" * 64,
        prefixes=[module_name],
    )

    with pytest.raises(RuntimeFactoryError, match="SHA-256 does not match"):
        _binary_optimizer_factory(configuration)

    assert module_name not in sys.modules


def test_binary_optimizer_factory_loads_pinned_local_factory_per_seed(
    tmp_path,
    monkeypatch,
):
    module_name = "paretoskill_test_pinned_optimizer"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
class FakeBinaryOptimizer:
    adapter_id = "fake-binary-optimizer"

    def __init__(self, *, patch_ids, seed, method_spec):
        self.patch_ids = tuple(patch_ids)
        self.seed = seed
        self.method_spec = dict(method_spec)
        self.observations = []

    def ask(self, *, patch_ids, count, seed, seen_subsets, observations):
        return ()

    def tell(self, scored):
        self.observations.extend(scored)

    def state_dict(self):
        return {"seed": self.seed, "observations": list(self.observations)}

    def load_state_dict(self, state):
        self.observations = list(state.get("observations", ()))


def build_adapter(*, patch_ids, seed, method_spec):
    return FakeBinaryOptimizer(
        patch_ids=patch_ids,
        seed=seed,
        method_spec=method_spec,
    )
""".lstrip(),
        encoding="utf-8",
    )
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    configuration = _optimizer_configuration(
        reference=f"{module_name}:build_adapter",
        sha256=module_sha256,
        prefixes=[module_name],
    )

    try:
        factory = _binary_optimizer_factory(configuration)
        assert factory is not None

        first = factory(("patch-a", "patch-b"), 17, {"batch_size": 2})
        second = factory(("patch-a", "patch-b"), 23, {"batch_size": 2})

        assert isinstance(first, BinarySubsetBayesianAdapter)
        assert isinstance(second, BinarySubsetBayesianAdapter)
        assert first is not second
        assert first.seed == 17
        assert second.seed == 23
        first.tell(("first-only",))
        assert first.state_dict()["observations"] == ["first-only"]
        assert second.state_dict()["observations"] == []
    finally:
        sys.modules.pop(module_name, None)


def test_runtime_factory_builds_replay_matrix_without_provider_calls(tmp_path):
    base = tmp_path / "SKILL.md"
    base.write_text("# Base\n", encoding="utf-8")
    evidence = tmp_path / "evidence.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "evidence_id": "e-1",
                "task_id": "task-1",
                "seed": 17,
                "target_id": "id-target",
                "outcome": False,
                "verifier_summary": "fixture",
                "tags": ["id_accuracy"],
                "metadata": {},
            }
        ],
    )
    patches = tmp_path / "patches.jsonl"
    _write_jsonl(
        patches,
        [
            {
                "patch_id": "p-1",
                "operation": "add",
                "target_path": "SKILL.md",
                "parent_version_id": "$BASE",
                "evidence_ids": ["e-1"],
                "content": "Use the checked procedure.",
                "sequence": 0,
            }
        ],
    )
    tasks = tmp_path / "tasks.jsonl"
    task_digest = _write_jsonl(
        tasks,
        [
            {
                "schema_version": 1,
                "task_id": f"task-{index}",
                "split_id": "shared",
                "domain_id": "tables",
                "group_id": "all",
                "objective_roles": ["id", "transfer"],
                "payload": {"question": f"question-{index}"},
                "verifier": {
                    "kind": "exact_match",
                    "expected": f"answer-{index}",
                },
            }
            for index in range(2)
        ],
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text("", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return one evidence-grounded JSON patch.", encoding="utf-8")
    configuration = {
        "experiment": {"id": "fixture"},
        "active_runtime_profile": "replay",
        "runtime_profiles": {"replay": {"provider_override": "replay"}},
        "safety": {"allow_external_verifier_commands": False},
        "runtime": {
            "base_skill_path": str(base),
            "trace_store_path": str(evidence),
            "patch_pool_path": str(patches),
        },
        "shared_search_controls": {},
        "providers": {
            "external": {"kind": "external_protocol"},
            "replay": {"kind": "replay", "replay_path": str(replay)},
        },
        "models": {
            "author": {
                "provider": "external",
                "model_id": "author-model",
                "revision": "v1",
                "decoding": {"max_output_tokens": 64},
            },
            "user": {
                "provider": "external",
                "model_id": "user-model",
                "revision": "v1",
                "decoding": {"temperature": 0.0},
            },
        },
        "proposer": {
            "model": "author",
            "prompt_template": str(prompt),
            "allowed_operations": ["add", "drop", "rewrite", "compress"],
        },
        "harnesses": {"verified": {"adapter": "builtin"}},
        "domains": {"tables": {"adapter": "builtin"}},
        "splits": {
            "shared": {
                "manifest": str(tasks),
                "manifest_sha256": task_digest,
                "expected_count": 2,
            }
        },
        "targets": {
            "id-target": {
                "model": "user",
                "harness": "verified",
                "domain": "tables",
                "split": "shared",
                "phase": "search",
                "transfer_group": None,
            },
            "transfer-target": {
                "model": "user",
                "harness": "verified",
                "domain": "tables",
                "split": "shared",
                "phase": "search",
                "transfer_group": "domain",
            },
        },
        "objectives": {
            "id_accuracy": {"target_ids": ["id-target"]},
            "worst_target_transfer": {"target_ids": ["transfer-target"]},
        },
        "task_seed_blocks": {"execution_seeds": [17]},
    }
    manifest = ExperimentManifest(
        data=configuration,
        source_path=tmp_path / "config.yaml",
        profile="replay",
        unresolved_placeholders=(),
    )

    runtime = build_experiment_runtime(
        manifest,
        policy=NetworkPolicy(),
        include_search=True,
        include_final=False,
    )

    assert isinstance(runtime.providers["replay"], ReplayProvider)
    assert {target.provider_id for target in runtime.phases["search"].targets} == {
        "replay"
    }
    assert len(runtime.phases["search"].blocks) == 4
    assert runtime.proposer_factory is not None
    proposer = runtime.proposer_factory(
        lambda version_id: runtime.base,
        104729,
    )
    assert proposer.proposer_id == "paretoskill-proposer"
