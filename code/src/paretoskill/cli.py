"""Configuration-driven command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_manifest
from .experiment_runner import run_configured_final, run_configured_search
from .providers import NetworkPolicy
from .provenance import verify_local_content_pins
from .runtime_factory import build_experiment_runtime
from .runner import run_synthetic_dry_run
from .task_manifests import validate_declared_splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paretoskill")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate a versioned YAML manifest")
    validate.add_argument("config", type=Path)
    validate.add_argument("--profile", choices=("dry_run", "replay", "real"), default="dry_run")

    identify = subparsers.add_parser("experiment-id", help="print the reproducible manifest hash")
    identify.add_argument("config", type=Path)
    identify.add_argument("--profile", choices=("dry_run", "replay", "real"), default="dry_run")

    dry_run = subparsers.add_parser("dry-run", help="run the built-in offline synthetic fixture")
    dry_run.add_argument("config", type=Path)
    dry_run.add_argument("--output-root", type=Path)
    dry_run.add_argument("--no-resume", action="store_true")

    real_run = subparsers.add_parser(
        "run", help="run configured replay or explicitly authorized real stages"
    )
    real_run.add_argument("config", type=Path)
    real_run.add_argument("--profile", choices=("replay", "real"), required=True)
    real_run.add_argument(
        "--stage",
        choices=("preflight", "smoke", "search", "final", "all"),
        default="preflight",
    )
    real_run.add_argument("--output-root", type=Path)
    real_run.add_argument("--method", action="append", dest="methods")
    real_run.add_argument("--seed", action="append", dest="seeds", type=int)
    real_run.add_argument("--no-resume", action="store_true")
    real_run.add_argument("--allow-network", action="store_true")
    return parser


def _network_policy(manifest, *, cli_allows_network: bool) -> NetworkPolicy:
    safety = manifest.data["safety"]
    return NetworkPolicy(
        offline_by_default=bool(safety["offline_by_default"]),
        config_allows_network=bool(safety["allow_network"]),
        cli_allows_network=cli_allows_network,
        required_env=str(safety["required_network_env"]),
        required_value=str(safety["required_network_value"]),
    )


def _configured_output_root(manifest, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    configured = Path(str(manifest.data["outputs"]["root"]))
    return configured if configured.is_absolute() else manifest.code_root / configured


def _run(args: argparse.Namespace) -> int:
    if args.command == "validate-config":
        manifest = load_manifest(args.config, profile=args.profile)
        if args.profile == "real":
            validate_declared_splits(
                manifest.data, base_directory=manifest.source_path.parent
            )
            verify_local_content_pins(
                manifest.data, base_directory=manifest.source_path.parent
            )
        print(
            json.dumps(
                {
                    "valid": True,
                    "profile": manifest.profile,
                    "offline": manifest.is_offline,
                    "experiment_id": manifest.experiment_id,
                    "unresolved_placeholders": list(manifest.unresolved_placeholders),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "experiment-id":
        print(load_manifest(args.config, profile=args.profile).experiment_id)
        return 0
    if args.command == "dry-run":
        manifest = load_manifest(args.config, profile="dry_run")
        summary = run_synthetic_dry_run(
            manifest,
            output_root=args.output_root,
            resume=not args.no_resume,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0
    if args.command == "run":
        manifest = load_manifest(args.config, profile=args.profile)
        validate_declared_splits(
            manifest.data, base_directory=manifest.source_path.parent
        )
        verify_local_content_pins(
            manifest.data, base_directory=manifest.source_path.parent
        )
        policy = _network_policy(
            manifest,
            cli_allows_network=bool(args.allow_network),
        )
        include_search = args.stage in {"preflight", "smoke", "search", "all"}
        include_final = args.stage in {"preflight", "final", "all"}
        runtime = build_experiment_runtime(
            manifest,
            policy=policy,
            include_search=include_search,
            include_final=include_final,
        )
        if args.stage == "preflight":
            requires_binary_optimizer = args.methods is None or (
                "trace2skill_accuracy_subset" in args.methods
            )
            if (
                include_search
                and requires_binary_optimizer
                and runtime.binary_optimizer_factory is None
            ):
                raise ConfigError(
                    "preflight selected Trace2Skill-style binary Bayesian search, "
                    "but no allowlisted content-pinned BinarySubsetBayesianAdapter "
                    "is available; no provider request was made"
                )
            print(
                json.dumps(
                    {
                        "valid": True,
                        "profile": manifest.profile,
                        "stage": "preflight",
                        "experiment_id": manifest.experiment_id,
                        "provider_ids": sorted(runtime.providers),
                        "search_targets": len(runtime.phases.get("search", ()).targets)
                        if "search" in runtime.phases
                        else 0,
                        "final_targets": len(runtime.phases.get("final", ()).targets)
                        if "final" in runtime.phases
                        else 0,
                        "provider_calls": 0,
                    },
                    indent=2,
                )
            )
            return 0
        if args.profile == "real":
            policy.require_external_enabled()
        output_root = _configured_output_root(manifest, args.output_root)
        experiment_output = output_root / manifest.experiment_id
        if args.no_resume and experiment_output.exists() and any(
            experiment_output.iterdir()
        ):
            raise ConfigError(
                f"output already exists and --no-resume was requested: {experiment_output}"
            )
        payload: dict[str, object] = {}
        if args.stage in {"smoke", "search", "all"}:
            search = run_configured_search(
                manifest,
                runtime,
                output_root=output_root,
                policy=policy,
                method_ids=args.methods,
                search_seeds=args.seeds,
                smoke=args.stage == "smoke",
            )
            payload["search"] = search.to_dict()
        if args.stage in {"final", "all"}:
            final = run_configured_final(
                manifest,
                runtime,
                output_root=output_root,
                policy=policy,
            )
            payload["final"] = final.to_dict()
        print(
            json.dumps(
                {
                    "experiment_id": manifest.experiment_id,
                    "profile": manifest.profile,
                    "stage": args.stage,
                    **payload,
                },
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
