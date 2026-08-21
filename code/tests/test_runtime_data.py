from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from paretoskill.evaluation import EvaluationCandidate, PairedEvaluator
from paretoskill.models import Skill, make_base_version
from paretoskill.providers import ExecutionRequest, ExecutionResult, GeneratedResponse
from paretoskill.runtime_data import (
    RuntimeDataError,
    VerifiedResponseHarness,
    VerifierSpec,
    load_local_task_manifest,
    load_runtime_matrix,
)
from paretoskill.runtime_factory import _phase_runtime


def _task(
    task_id: str,
    *,
    roles: list[str] | None = None,
    verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "split_id": "shared",
        "domain_id": "tables",
        "group_id": "all",
        "objective_roles": ["id", "transfer"] if roles is None else roles,
        "payload": {"question": f"question for {task_id}"},
        "verifier": (
            {
                "kind": "exact_match",
                "expected": f"answer-{task_id}",
                "strip": True,
                "case_sensitive": False,
            }
            if verifier is None
            else verifier
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_pins(*paths: str | Path) -> dict[str, str]:
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    }


def _configuration(path: Path, digest: str, count: int) -> dict[str, Any]:
    return {
        "active_runtime_profile": "replay",
        "safety": {"allow_external_verifier_commands": False},
        "providers": {"local": {"kind": "fixture"}},
        "models": {
            "model": {
                "provider": "local",
                "model_id": "local-model",
                "revision": "fixture-v1",
                "decoding": {"temperature": 0},
            }
        },
        "harnesses": {"verified": {"adapter": "local"}},
        "domains": {"tables": {"adapter": "local"}},
        "splits": {
            "shared": {
                "manifest": str(path),
                "manifest_sha256": digest,
                "expected_count": count,
            }
        },
        "targets": {
            "id-target": {
                "model": "model",
                "harness": "verified",
                "domain": "tables",
                "split": "shared",
                "phase": "search",
                "transfer_group": None,
            },
            "transfer-target": {
                "model": "model",
                "harness": "verified",
                "domain": "tables",
                "split": "shared",
                "phase": "search",
                "transfer_group": "cross-model",
            },
        },
        "objectives": {
            "id_accuracy": {"target_ids": ["id-target"]},
            "worst_target_transfer": {"target_ids": ["transfer-target"]},
        },
        "task_seed_blocks": {"execution_seeds": [11, 7]},
    }


def test_local_jsonl_builds_closed_role_target_seed_matrix(tmp_path: Path) -> None:
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one"), _task("two")])

    matrix = load_runtime_matrix(
        _configuration(source, digest, 2), phase="search", base_directory=tmp_path
    )
    targets, blocks, harnesses = matrix

    assert [target.target_id for target in targets] == ["id-target", "transfer-target"]
    assert targets[0].split_id == "shared"
    assert targets[0].objective_role == "id"
    assert targets[1].objective_role == "transfer"
    assert targets[1].transfer_group == "cross-model"
    assert targets[1].model.model_id == "local-model"
    assert targets[1].harness_id == "verified"
    assert targets[1].domain_id == "tables"
    assert len(blocks) == 8  # 2 logical tasks x 2 roles x 2 execution seeds
    assert {block.task.split_id for block in blocks} == {"shared"}
    assert {block.task.objective_role for block in blocks} == {"id", "transfer"}
    assert {block.seed for block in blocks} == {7, 11}
    assert set(harnesses) == {"verified"}
    assert len(tuple(matrix.domains["tables"].tasks("shared"))) == 4
    assert matrix.manifests["shared"].logical_task_count == 2


def test_json_envelope_is_supported_and_hash_count_are_mandatory(tmp_path: Path) -> None:
    source = tmp_path / "tasks.json"
    source.write_text(
        json.dumps({"schema_version": 1, "tasks": [_task("one")]}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    loaded = load_local_task_manifest(
        source, expected_sha256=digest, expected_count=1
    )

    assert loaded.source_task_ids == ("one",)
    assert len(loaded.tasks) == 2
    with pytest.raises(RuntimeDataError, match="digest mismatch"):
        load_local_task_manifest(source, expected_sha256="0" * 64, expected_count=1)
    with pytest.raises(RuntimeDataError, match="has 1 tasks, expected 2"):
        load_local_task_manifest(source, expected_sha256=digest, expected_count=2)


def test_loader_rejects_duplicate_ids_extra_fields_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate_digest = _write_jsonl(duplicate, [_task("same"), _task("same")])
    with pytest.raises(RuntimeDataError, match="duplicate task ids"):
        load_local_task_manifest(
            duplicate, expected_sha256=duplicate_digest, expected_count=2
        )

    extra = tmp_path / "extra.jsonl"
    extra_row = _task("one") | {"unexpected": True}
    extra_digest = _write_jsonl(extra, [extra_row])
    with pytest.raises(RuntimeDataError, match="schema mismatch"):
        load_local_task_manifest(extra, expected_sha256=extra_digest, expected_count=1)

    repeated_key = tmp_path / "repeated-key.jsonl"
    repeated_key.write_text(
        '{"schema_version":1,"schema_version":1,"task_id":"one"}\n',
        encoding="utf-8",
    )
    repeated_digest = hashlib.sha256(repeated_key.read_bytes()).hexdigest()
    with pytest.raises(RuntimeDataError, match="duplicate JSON object key"):
        load_local_task_manifest(
            repeated_key, expected_sha256=repeated_digest, expected_count=1
        )


def test_loader_never_accepts_urls_or_unfrozen_counts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeDataError, match="URLs are forbidden"):
        load_local_task_manifest(
            "https://example.invalid/tasks.jsonl",
            expected_sha256="0" * 64,
            expected_count=1,
        )
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    configuration["splits"]["shared"]["expected_count"] = None
    with pytest.raises(RuntimeDataError, match="must be frozen"):
        load_runtime_matrix(configuration, phase="search")


def test_matrix_fails_on_missing_role_duplicate_seed_and_missing_reference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one", roles=["id"])])
    configuration = _configuration(source, digest, 1)
    with pytest.raises(RuntimeDataError, match="matrix is not closed"):
        load_runtime_matrix(configuration, phase="search")

    configuration = _configuration(source, digest, 1)
    configuration["task_seed_blocks"]["execution_seeds"] = [7, 7]
    with pytest.raises(RuntimeDataError, match="seeds contain duplicates"):
        load_runtime_matrix(configuration, phase="search")

    configuration = _configuration(source, digest, 1)
    configuration["targets"]["id-target"]["model"] = "missing"
    with pytest.raises(RuntimeDataError, match="unknown model"):
        load_runtime_matrix(configuration, phase="search")

    configuration = _configuration(source, digest, 1)
    with pytest.raises(RuntimeDataError, match="target_ids contains duplicates"):
        load_runtime_matrix(
            configuration,
            phase="search",
            target_ids=["id-target", "id-target"],
        )


@pytest.mark.parametrize(
    ("spec", "response", "expected"),
    [
        (
            {"kind": "exact_match", "expected": "Yes", "case_sensitive": False},
            "yes",
            True,
        ),
        (
            {"kind": "json_boolean", "field_path": ["result", "passed"]},
            '{"result":{"passed":true}}',
            True,
        ),
        (
            {"kind": "json_field", "field_path": ["answers", 0], "expected": 42},
            {"answers": [42]},
            True,
        ),
        (
            {"kind": "builtin", "rule": "contains", "expected": "needle"},
            "a needle in output",
            True,
        ),
        (
            {
                "kind": "builtin",
                "rule": "numeric_tolerance",
                "expected": 1.0,
                "absolute_tolerance": 0.01,
            },
            "1.005",
            True,
        ),
        (
            {"kind": "json_boolean", "field_path": ["passed"]},
            "not-json",
            False,
        ),
    ],
)
def test_builtin_verifiers(spec: dict[str, Any], response: Any, expected: bool) -> None:
    assert VerifierSpec.from_mapping(spec).verify(response) is expected


def test_harness_ignores_provider_correct_and_handles_no_skill(tmp_path: Path) -> None:
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one", roles=["id"])])
    configuration = _configuration(source, digest, 1)
    configuration["targets"].pop("transfer-target")
    configuration["objectives"]["worst_target_transfer"]["target_ids"] = []
    matrix = load_runtime_matrix(configuration, phase="search", execution_seeds=[7])
    target = matrix.targets[0]
    block = matrix.blocks[0]

    class CaptureProvider:
        provider_id = "local"
        is_external = False

        def __init__(self) -> None:
            self.requests: list[ExecutionRequest] = []

        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            self.requests.append(request)
            return ExecutionResult(
                correct=False,
                input_tokens=3,
                output_tokens=2,
                latency_ms=1.0,
                trace={"raw_response": " ANSWER-ONE "},
            )

    provider = CaptureProvider()
    base = make_base_version(Skill("base", {"SKILL.md": "base instructions"}))
    no_skill = EvaluationCandidate.no_skill(base)
    records = PairedEvaluator().evaluate(
        experiment_id="local-eval",
        base=base,
        candidates=(no_skill,),
        targets=(target,),
        blocks=(block,),
        providers={"local": provider},
        harnesses=matrix.harnesses,
    )

    assert len(records) == 2
    assert all(record.result.correct for record in records)
    assert all(
        record.result.provider_metadata["correctness_source"] == "local_verifier"
        for record in records
    )
    by_mode = {request.metadata["injection_mode"]: request for request in provider.requests}
    assert by_mode["skill"].skill_files == {"SKILL.md": "base instructions"}
    assert by_mode["none"].skill_files == {}
    assert "expected" not in json.dumps(dict(by_mode["none"].task_payload))


def test_harness_uses_unscored_generate_bridge_when_available(tmp_path: Path) -> None:
    verifier = VerifierSpec.from_mapping(
        {"kind": "json_field", "field_path": ["x"], "expected": 2}
    )
    harness = VerifiedResponseHarness("verified", {"one": verifier})
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(
        source,
        [
            _task(
                "one",
                roles=["id"],
                verifier={"kind": "json_field", "field_path": ["x"], "expected": 2},
            )
        ],
    )
    configuration = _configuration(source, digest, 1)
    configuration["targets"].pop("transfer-target")
    configuration["objectives"]["worst_target_transfer"]["target_ids"] = []
    matrix = load_runtime_matrix(configuration, phase="search", execution_seeds=[7])

    class Generator:
        provider_id = "local"
        is_external = False

        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            raise AssertionError("execute must not be used when unscored generate is available")

        def generate(self, request: ExecutionRequest) -> GeneratedResponse:
            return GeneratedResponse("{\"x\":2}", {"x": 2}, 4, 2, 1.0)

    base = make_base_version(Skill("base", {"SKILL.md": "base"}))
    records = PairedEvaluator().evaluate(
        experiment_id="generated",
        base=base,
        candidates=(),
        targets=matrix.targets,
        blocks=matrix.blocks,
        providers={"local": Generator()},
        harnesses={"verified": harness},
    )

    assert records[0].result.correct is True
    assert records[0].result.provider_metadata["provider_reported_correct"] is None


def test_external_command_verifier_is_path_bounded_and_profile_gated(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "verify.py"
    helper.write_text(
        "import json, sys\n"
        "row = json.load(sys.stdin)\n"
        "print(json.dumps({'schema_version': 1, "
        "'correct': row['response'] == row['verifier_input']['expected']}))\n",
        encoding="utf-8",
    )
    spec = VerifierSpec.from_mapping(
        {
            "kind": "external_command",
            "argv": [sys.executable, str(helper)],
            "timeout_seconds": 5,
            "enabled_profiles": ["replay"],
            "allowed_path_roots": [str(Path(sys.executable).parent), str(tmp_path)],
            "path_sha256": _path_pins(sys.executable, helper),
            "cwd": str(tmp_path),
            "verifier_input": {"expected": "ok"},
        }
    )

    assert spec.verify(
        "ok", runtime_profile="replay", allow_external_commands=True
    ) is True
    with pytest.raises(RuntimeDataError, match="explicitly enabled real/replay"):
        spec.verify("ok", runtime_profile="dry_run", allow_external_commands=True)
    with pytest.raises(RuntimeDataError, match="explicitly enabled real/replay"):
        spec.verify("ok", runtime_profile="replay", allow_external_commands=False)

    helper.write_text("raise SystemExit(1)\n", encoding="utf-8")
    with pytest.raises(RuntimeDataError, match="pinned external verifier file changed"):
        spec.verify("ok", runtime_profile="replay", allow_external_commands=True)

    with pytest.raises(RuntimeDataError, match="outside allowed_path_roots"):
        VerifierSpec.from_mapping(
            {
                "kind": "external_command",
                "argv": [sys.executable, str(helper)],
                "timeout_seconds": 5,
                "enabled_profiles": ["replay"],
                "allowed_path_roots": [str(tmp_path)],
                "path_sha256": _path_pins(sys.executable, helper),
            }
        )


def test_external_verifier_matrix_requires_explicit_non_dry_run_enablement(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "verify.py"
    helper.write_text(
        "import json, sys\njson.load(sys.stdin)\n"
        "print(json.dumps({'schema_version': 1, 'correct': True}))\n",
        encoding="utf-8",
    )
    external = {
        "kind": "external_command",
        "argv": [sys.executable, str(helper)],
        "timeout_seconds": 5,
        "enabled_profiles": ["replay"],
        "allowed_path_roots": [str(Path(sys.executable).parent), str(tmp_path)],
        "path_sha256": _path_pins(sys.executable, helper),
    }
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one", verifier=external)])
    configuration = _configuration(source, digest, 1)

    with pytest.raises(RuntimeDataError, match="allow_external_verifier_commands"):
        load_runtime_matrix(configuration, phase="search")
    configuration["safety"]["allow_external_verifier_commands"] = True
    configuration["active_runtime_profile"] = "dry_run"
    with pytest.raises(RuntimeDataError, match="allow_external_verifier_commands"):
        load_runtime_matrix(configuration, phase="search")
    configuration["active_runtime_profile"] = "replay"
    matrix = load_runtime_matrix(configuration, phase="search", execution_seeds=[7])

    class ReplayFixture:
        provider_id = "local"
        is_external = False

        def execute(self, request: ExecutionRequest) -> ExecutionResult:
            return ExecutionResult(False, 1, 1, 0.0, trace={"raw_response": "ignored"})

    base = make_base_version(Skill("base", {"SKILL.md": "base"}))
    target = next(item for item in matrix.targets if item.objective_role == "id")
    block = next(item for item in matrix.blocks if item.task.objective_role == "id")
    record = matrix.harnesses["verified"].evaluate(
        provider=ReplayFixture(),
        experiment_id="external-verifier-fixture",
        candidate=EvaluationCandidate.skill(base),
        target=target,
        block=block,
        is_base=True,
    )
    assert record.result.correct is True


def test_dynamic_runtime_adapters_are_not_imported_without_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "paretoskill_test_runtime_adapter_disabled"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "raise AssertionError('disabled runtime adapter was imported')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    reference = f"{module_name}:factory"
    configuration["domains"]["tables"] = {
        "adapter": reference,
        "adapter_sha256": "0" * 64,
    }
    configuration["harnesses"]["verified"] = {
        "adapter": reference,
        "adapter_sha256": "0" * 64,
    }

    matrix = load_runtime_matrix(
        configuration,
        phase="search",
        base_directory=tmp_path,
        execution_seeds=[7],
    )

    assert module_name not in sys.modules
    assert isinstance(matrix.harnesses["verified"], VerifiedResponseHarness)
    assert all("adapted" not in block.task.payload for block in matrix.blocks)


def test_dynamic_runtime_adapter_rejects_allowlist_and_hash_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "paretoskill_test_runtime_adapter_rejected"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "def build_domain(**kwargs):\n    return kwargs['tasks']\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    configuration["safety"].update(
        {
            "allow_dynamic_runtime_adapter_imports": True,
            "allowed_dynamic_runtime_adapter_prefixes": ["approved_runtime_adapter"],
        }
    )
    configuration["domains"]["tables"] = {
        "adapter": f"{module_name}:build_domain",
        "adapter_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }
    configuration["harnesses"]["verified"] = {"adapter": "builtin"}

    with pytest.raises(RuntimeDataError, match="outside the allowlist"):
        load_runtime_matrix(configuration, phase="search", base_directory=tmp_path)
    assert module_name not in sys.modules

    configuration["safety"]["allowed_dynamic_runtime_adapter_prefixes"] = [
        module_name
    ]
    configuration["domains"]["tables"]["adapter_sha256"] = "0" * 64
    with pytest.raises(RuntimeDataError, match="SHA-256 does not match"):
        load_runtime_matrix(configuration, phase="search", base_directory=tmp_path)
    assert module_name not in sys.modules


def test_pinned_dynamic_domain_and_harness_factories_drive_runtime_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "paretoskill_test_runtime_adapter_success"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
from paretoskill.evaluation import TaskSpec

DOMAIN_CALLS = []
HARNESS_CALLS = []


def build_domain(*, domain_id, tasks, spec, runtime_spec, dataset_roots,
                 base_directory, phase):
    assert spec["task_type"] == "fixture"
    assert runtime_spec["dataset_roots"] == dataset_roots
    assert dataset_roots["fixture"] == str(base_directory / "dataset")
    DOMAIN_CALLS.append((domain_id, phase, len(tasks)))
    return tuple(
        TaskSpec(
            task_id=task.task_id,
            split=task.split,
            domain_id=task.domain_id,
            group_id=task.group_id,
            payload={**dict(task.payload), "adapted": True},
            split_id=task.split_id,
            objective_role=task.objective_role,
        )
        for task in tasks
    )


class CustomHarness:
    def __init__(self, harness_id, delegate):
        self.harness_id = harness_id
        self.delegate = delegate
        self.cache = None

    def evaluate(self, **kwargs):
        HARNESS_CALLS.append(("evaluate", kwargs["block"].task.payload["adapted"]))
        self.delegate.cache = self.cache
        return self.delegate.evaluate(**kwargs)


def build_harness(*, harness_id, default_harness, verifiers, spec,
                  runtime_spec, dataset_roots, base_directory, phase):
    assert spec["max_tool_steps"] == 7
    assert spec["sandbox_image"] == "fixture-sandbox"
    assert dataset_roots["fixture"] == str(base_directory / "dataset")
    assert verifiers
    HARNESS_CALLS.append(("build", harness_id, phase))
    return CustomHarness(harness_id, default_harness)
""".lstrip(),
        encoding="utf-8",
    )
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    configuration["safety"].update(
        {
            "allow_dynamic_runtime_adapter_imports": True,
            "allowed_dynamic_runtime_adapter_prefixes": [module_name],
        }
    )
    configuration["runtime"] = {"dataset_roots": {"fixture": str(dataset_root)}}
    configuration["domains"]["tables"] = {
        "adapter": f"{module_name}:build_domain",
        "adapter_sha256": module_sha256,
        "task_type": "fixture",
    }
    configuration["harnesses"]["verified"] = {
        "adapter": f"{module_name}:build_harness",
        "adapter_sha256": module_sha256,
        "max_tool_steps": 7,
        "sandbox_image": "fixture-sandbox",
        "sandbox_image_digest": "f" * 64,
    }

    try:
        phase_runtime = _phase_runtime(
            configuration,
            phase="search",
            base_directory=tmp_path,
        )
        imported = importlib.import_module(module_name)
        assert imported.DOMAIN_CALLS == [("tables", "search", 2)]
        assert imported.HARNESS_CALLS == [("build", "verified", "search")]
        assert all(
            block.task.payload["adapted"] is True for block in phase_runtime.blocks
        )
        assert type(phase_runtime.harnesses["verified"]).__name__ == "CustomHarness"

        class FixtureProvider:
            provider_id = "local"
            is_external = False

            def execute(self, request: ExecutionRequest) -> ExecutionResult:
                return ExecutionResult(
                    correct=False,
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=0.0,
                    trace={"raw_response": "answer-one"},
                )

        base = make_base_version(Skill("base", {"SKILL.md": "base"}))
        target = next(
            item for item in phase_runtime.targets if item.objective_role == "id"
        )
        block = next(
            item for item in phase_runtime.blocks if item.task.objective_role == "id"
        )
        record = phase_runtime.harnesses["verified"].evaluate(
            provider=FixtureProvider(),
            experiment_id="dynamic-runtime-adapter-fixture",
            candidate=EvaluationCandidate.skill(base),
            target=target,
            block=block,
            is_base=True,
        )
        assert record.result.correct is True
        assert imported.HARNESS_CALLS[-1] == ("evaluate", True)
    finally:
        sys.modules.pop(module_name, None)


def test_dynamic_domain_factory_output_must_preserve_closed_task_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "paretoskill_test_runtime_adapter_incomplete"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "def build_domain(**kwargs):\n    return tuple(kwargs['tasks'])[:-1]\n",
        encoding="utf-8",
    )
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    configuration["safety"].update(
        {
            "allow_dynamic_runtime_adapter_imports": True,
            "allowed_dynamic_runtime_adapter_prefixes": [module_name],
        }
    )
    configuration["domains"]["tables"] = {
        "adapter": f"{module_name}:build_domain",
        "adapter_sha256": module_sha256,
    }
    configuration["harnesses"]["verified"] = {"adapter": "builtin"}

    try:
        with pytest.raises(RuntimeDataError, match="matrix is not closed"):
            load_runtime_matrix(
                configuration,
                phase="search",
                base_directory=tmp_path,
                execution_seeds=[7],
            )
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    ("factory_name", "collection", "message"),
    [
        ("bad_domain", "domains", "must return TaskSpec values"),
        ("wrong_harness_id", "harnesses", "returned the wrong id"),
        ("missing_harness_evaluate", "harnesses", "must return a Harness"),
        ("exploding_domain", "domains", "factory for 'tables' failed: RuntimeError"),
    ],
)
def test_dynamic_runtime_adapter_factory_contracts_are_strict_and_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    collection: str,
    message: str,
) -> None:
    module_name = f"paretoskill_test_runtime_contract_{factory_name}"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        """
class HarnessValue:
    def __init__(self, harness_id, *, callable_evaluate=True):
        self.harness_id = harness_id
        if callable_evaluate:
            self.evaluate = lambda **kwargs: None


def bad_domain(**kwargs):
    return ("not-a-task-spec",)


def wrong_harness_id(**kwargs):
    return HarnessValue("another-harness")


def missing_harness_evaluate(**kwargs):
    return HarnessValue(kwargs["harness_id"], callable_evaluate=False)


def exploding_domain(**kwargs):
    raise RuntimeError("adapter-private-detail")
""".lstrip(),
        encoding="utf-8",
    )
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    source = tmp_path / "tasks.jsonl"
    digest = _write_jsonl(source, [_task("one")])
    configuration = _configuration(source, digest, 1)
    configuration["safety"].update(
        {
            "allow_dynamic_runtime_adapter_imports": True,
            "allowed_dynamic_runtime_adapter_prefixes": [module_name],
        }
    )
    configuration["domains"]["tables"] = {"adapter": "builtin"}
    configuration["harnesses"]["verified"] = {"adapter": "builtin"}
    adapter_id = "tables" if collection == "domains" else "verified"
    configuration[collection][adapter_id] = {
        "adapter": f"{module_name}:{factory_name}",
        "adapter_sha256": module_sha256,
    }

    try:
        with pytest.raises(RuntimeDataError, match=message):
            load_runtime_matrix(
                configuration,
                phase="search",
                base_directory=tmp_path,
                execution_seeds=[7],
            )
    finally:
        sys.modules.pop(module_name, None)
