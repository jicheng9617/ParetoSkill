from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import paretoskill.cli as cli
from paretoskill.config import ExperimentManifest, load_manifest
from paretoskill.evaluation import ProviderHarness, TargetSpec, TaskSeedBlock, TaskSpec
from paretoskill.experiment_runner import (
    ConfiguredRunSummary,
    ExperimentRuntime,
    PhaseRuntime,
)
from paretoskill.models import (
    Patch,
    PatchOperation,
    Skill,
    TraceEvidence,
    make_base_version,
)
from paretoskill.providers import ExecutionRequest, ExecutionResult, ModelSpec


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "iclr2027.yaml"


class OfflineFixtureProvider:
    provider_id = "mock"
    is_external = False

    def __init__(self, *, forbid_calls: bool = False) -> None:
        self.forbid_calls = forbid_calls
        self.calls = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.forbid_calls:
            raise AssertionError("preflight/unauthorized real stage called a provider")
        self.calls += 1
        return ExecutionResult(
            correct=True,
            input_tokens=10 + len(request.skill_files),
            output_tokens=2,
            latency_ms=0.0,
            trace={"mode": "offline-cli-fixture"},
            provider_metadata={"offline": True},
        )


def _manifest(profile: str, *, allow_network: bool = False) -> ExperimentManifest:
    base = load_manifest(CONFIG, profile="dry_run", environment={})
    data = copy.deepcopy(dict(base.data))
    data["safety"]["allow_network"] = allow_network
    data["runtime_profiles"]["real"]["network"] = True
    return replace(base, data=data, profile=profile)


def _runtime(
    provider: OfflineFixtureProvider, *, include_final: bool = False
) -> ExperimentRuntime:
    base = make_base_version(Skill("cli-fixture", {"SKILL.md": "# Base\n"}))
    evidence = TraceEvidence(
        evidence_id="evidence-1",
        task_id="id-task-0",
        seed=17,
        target_id="id-target",
        outcome=False,
        verifier_summary="offline CLI fixture",
        tags=("id_accuracy",),
    )
    patch = Patch(
        patch_id="patch-1",
        operation=PatchOperation.ADD,
        target_path="SKILL.md",
        parent_version_id=base.lineage.version_id,
        evidence_ids=(evidence.evidence_id,),
        content="Use a checked procedure.",
        sequence=0,
    )
    model = ModelSpec("fixture-model", "mock", "fixture-v1", {"temperature": 0.0})
    targets = (
        TargetSpec(
            "id-target",
            "mock",
            model,
            "provider-structured",
            "id-domain",
            "*",
            split_id="id-split",
            objective_role="id",
        ),
        TargetSpec(
            "transfer-target",
            "mock",
            model,
            "provider-structured",
            "transfer-domain",
            "*",
            split_id="transfer-split",
            transfer_group="domain",
            objective_role="transfer",
        ),
    )
    blocks = tuple(
        TaskSeedBlock(
            f"{role}-block-{index}",
            TaskSpec(
                task_id=f"{role}-task-{index}",
                split=role,
                domain_id=f"{role}-domain",
                group_id=role,
                payload={"index": index},
                split_id=f"{role}-split",
                objective_role=role,
            ),
            17,
        )
        for role in ("id", "transfer")
        for index in range(2)
    )
    phase = PhaseRuntime(
        targets=targets,
        blocks=blocks,
        harnesses={"provider-structured": ProviderHarness()},
    )
    phases = {"search": phase}
    if include_final:
        phases["final"] = phase
    return ExperimentRuntime(
        base=base,
        patches=(patch,),
        evidence={evidence.evidence_id: evidence},
        providers={"mock": provider},
        phases=phases,
    )


def _patch_local_preflight(
    monkeypatch: pytest.MonkeyPatch,
    manifest: ExperimentManifest,
    runtime_factory: Any,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {"splits": [], "pins": [], "build": []}

    def fake_load(path: Path, *, profile: str) -> ExperimentManifest:
        assert Path(path) == CONFIG
        assert profile == manifest.profile
        return manifest

    def fake_splits(configuration: Any, *, base_directory: Path) -> None:
        calls["splits"].append((configuration, base_directory))

    def fake_pins(configuration: Any, *, base_directory: Path) -> None:
        calls["pins"].append((configuration, base_directory))

    def fake_build(
        loaded: ExperimentManifest,
        *,
        policy: Any,
        include_search: bool,
        include_final: bool,
    ) -> ExperimentRuntime:
        calls["build"].append(
            (loaded, policy, include_search, include_final)
        )
        return runtime_factory()

    monkeypatch.setattr(cli, "load_manifest", fake_load)
    monkeypatch.setattr(cli, "validate_declared_splits", fake_splits)
    monkeypatch.setattr(cli, "verify_local_content_pins", fake_pins)
    monkeypatch.setattr(cli, "build_experiment_runtime", fake_build)
    return calls


def test_replay_preflight_validates_local_inputs_without_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest("replay")
    provider = OfflineFixtureProvider(forbid_calls=True)
    calls = _patch_local_preflight(
        monkeypatch,
        manifest,
        lambda: _runtime(provider, include_final=True),
    )
    monkeypatch.setattr(
        cli,
        "run_configured_search",
        lambda *args, **kwargs: pytest.fail("preflight entered search"),
    )
    monkeypatch.setattr(
        cli,
        "run_configured_final",
        lambda *args, **kwargs: pytest.fail("preflight entered final"),
    )

    exit_code = cli.main(
        [
            "run",
            str(CONFIG),
            "--profile",
            "replay",
            "--stage",
            "preflight",
            "--method",
            "trace2skill_all",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output == {
        "valid": True,
        "profile": "replay",
        "stage": "preflight",
        "experiment_id": manifest.experiment_id,
        "provider_ids": ["mock"],
        "search_targets": 2,
        "final_targets": 2,
        "provider_calls": 0,
    }
    assert provider.calls == 0
    assert len(calls["splits"]) == len(calls["pins"]) == 1
    assert calls["splits"][0][1] == manifest.source_path.parent
    assert calls["pins"][0][1] == manifest.source_path.parent
    assert calls["build"][0][2:] == (True, True)


def test_replay_smoke_runs_and_resumes_through_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    manifest = _manifest("replay")
    providers: list[OfflineFixtureProvider] = []

    def runtime_factory() -> ExperimentRuntime:
        provider = OfflineFixtureProvider()
        providers.append(provider)
        return _runtime(provider)

    _patch_local_preflight(monkeypatch, manifest, runtime_factory)
    arguments = [
        "run",
        str(CONFIG),
        "--profile",
        "replay",
        "--stage",
        "smoke",
        "--output-root",
        str(tmp_path),
        "--method",
        "trace2skill_all",
        "--seed",
        "104729",
    ]

    assert cli.main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli.main(arguments) == 0
    second = json.loads(capsys.readouterr().out)

    first_run = first["search"]["method_runs"][0]
    second_run = second["search"]["method_runs"][0]
    assert first["profile"] == second["profile"] == "replay"
    assert first["stage"] == second["stage"] == "smoke"
    assert first_run["physical_provider_executions"] == providers[0].calls > 0
    assert second_run["physical_provider_executions"] == providers[1].calls == 0
    assert first_run["logical_task_executions"] == second_run[
        "logical_task_executions"
    ]
    run_directory = Path(first_run["output_directory"])
    assert (run_directory / "run_state.json").is_file()
    assert (run_directory / "screen_task_outcomes.jsonl").is_file()
    assert (run_directory / "full_task_outcomes.jsonl").is_file()
    assert (run_directory / "selected_candidates.jsonl").is_file()

    assert cli.main(arguments + ["--no-resume"]) == 2
    error = capsys.readouterr().err
    assert "output already exists and --no-resume was requested" in error
    assert len(providers) == 3
    assert providers[2].calls == 0


def test_replay_search_stage_forwards_frozen_method_seed_and_non_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    manifest = _manifest("replay")
    provider = OfflineFixtureProvider(forbid_calls=True)
    _patch_local_preflight(monkeypatch, manifest, lambda: _runtime(provider))
    observed: dict[str, Any] = {}

    def fake_search(
        loaded: ExperimentManifest,
        runtime: ExperimentRuntime,
        **kwargs: Any,
    ) -> ConfiguredRunSummary:
        observed.update({"manifest": loaded, "runtime": runtime, **kwargs})
        return ConfiguredRunSummary(
            experiment_id=loaded.experiment_id,
            stage="search",
            output_directory=Path(kwargs["output_root"]) / loaded.experiment_id,
            method_runs=(),
        )

    monkeypatch.setattr(cli, "run_configured_search", fake_search)
    exit_code = cli.main(
        [
            "run",
            str(CONFIG),
            "--profile",
            "replay",
            "--stage",
            "search",
            "--output-root",
            str(tmp_path),
            "--method",
            "trace2skill_all",
            "--method",
            "base_skill",
            "--seed",
            "104729",
            "--seed",
            "130363",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["stage"] == "search"
    assert output["search"]["stage"] == "search"
    assert observed["manifest"] is manifest
    assert observed["method_ids"] == ["trace2skill_all", "base_skill"]
    assert observed["search_seeds"] == [104729, 130363]
    assert observed["smoke"] is False
    assert Path(observed["output_root"]) == tmp_path.resolve()
    assert provider.calls == 0


@pytest.mark.parametrize(
    ("config_authorized", "cli_authorized", "environment_authorized"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_real_execution_requires_all_three_network_authorizations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_authorized: bool,
    cli_authorized: bool,
    environment_authorized: bool,
) -> None:
    manifest = _manifest("real", allow_network=config_authorized)
    provider = OfflineFixtureProvider(forbid_calls=True)
    calls = _patch_local_preflight(monkeypatch, manifest, lambda: _runtime(provider))
    search_called = False

    def forbidden_search(*args: Any, **kwargs: Any) -> ConfiguredRunSummary:
        nonlocal search_called
        search_called = True
        raise AssertionError("unauthorized real run reached the search runner")

    monkeypatch.setattr(cli, "run_configured_search", forbidden_search)
    required_env = str(manifest.data["safety"]["required_network_env"])
    required_value = str(manifest.data["safety"]["required_network_value"])
    if environment_authorized:
        monkeypatch.setenv(required_env, required_value)
    else:
        monkeypatch.delenv(required_env, raising=False)

    arguments = ["run", str(CONFIG), "--profile", "real", "--stage", "smoke"]
    if cli_authorized:
        arguments.append("--allow-network")
    exit_code = cli.main(arguments)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "external providers are disabled" in captured.err
    assert "No network request was made" in captured.err
    assert captured.out == ""
    assert search_called is False
    assert provider.calls == 0
    assert calls["build"][0][1].config_allows_network is config_authorized
    assert calls["build"][0][1].cli_allows_network is cli_authorized
