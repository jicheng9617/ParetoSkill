# ParetoSkill implementation

This directory contains the offline-safe implementation and frozen ICLR 2027 experiment protocol for **regression-aware multi-objective evolution of transferable agent skills**. It does not contain experiment results, benchmark data, API credentials, or paper artifacts.

The default path is deliberately incapable of contacting a model API. It includes deterministic synthetic dry-run support, local replay, a staged experiment runner, strict local task-manifest/verifier loading, and a built-in OpenAI-compatible HTTP provider that remains network-gated by default. No formal experiment has been run from this repository.

## Install

Python 3.11 or newer is required. Direct dependencies are exactly pinned in `pyproject.toml`; the sole runtime dependency is PyYAML.

```bash
cd code
python -m venv .venv
python -m pip install -e ".[dev]"
```

For an already provisioned environment, installation is optional:

```powershell
$env:PYTHONPATH = "src"
python -m paretoskill --version
```

## Pure offline validation and dry-run

From `code/`:

```powershell
$env:PYTHONPATH = "src"
python -m paretoskill validate-config configs/experiments/iclr2027.yaml
python -m paretoskill experiment-id configs/experiments/iclr2027.yaml
python -m paretoskill dry-run configs/experiments/iclr2027.yaml --output-root ../tmp/paretoskill-dry-run
```

The dry-run:

- materializes synthetic add/rewrite/compress patch compositions;
- records content hashes and multiple lineage records;
- evaluates base/candidates on identical synthetic task-seed-target blocks;
- computes the four objectives, deterministic paired bootstrap bounds, conservative feasibility, and archive admissions;
- writes result-schema JSONL, a content-addressed cache, bounded working archive,
  unbounded reconstructed scientific front, lineage, checkpoint, and token accounting;
- resumes without re-executing cached blocks; and
- reports `external_calls: 0` and `synthetic_only: true`.

The output is development evidence only. It must never be reported as a benchmark result.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python -m compileall -q src tests
```

If the optional formatter/linter is installed:

```powershell
python -m ruff check src tests
```

Tests cover objective aggregation, worst-group transfer, conditional pass-to-fail regression, bootstrap bounds, point/pessimistic dominance, conservative constraints, archive admission/deduplication/budget/capacity/recovery, deterministic materialization, evidence and lineage, config drift, task-manifest disjointness, provider safety, dynamic runtime-adapter import gates, replay misses, cache/resume, baseline selectors, frontier metrics, ablation overrides, and deployment policies.

## Configuration

There are two deliberately different configuration artifacts:

| File | Role | Network behavior |
|---|---|---|
| [`configs/experiments/iclr2027.yaml`](configs/experiments/iclr2027.yaml) | Frozen v1 protocol and executable synthetic dry-run | Always offline; its `real` profile intentionally rejects |
| [`configs/experiments/iclr2027_real_v2.template.yaml`](configs/experiments/iclr2027_real_v2.template.yaml) | Complete staged real-run template derived from v1 | Rejects by default; copy to a new versioned run manifest before activation |

Do not edit v1 into an online configuration and do not activate the v2 template in place. The rationale is in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); the adapter and operations checklist is in [`docs/REAL_RUN_GUIDE.md`](docs/REAL_RUN_GUIDE.md).

Important sections are:

| Section | Purpose |
|---|---|
| `runtime_profiles`, `safety`, `providers` | Offline default, mock/replay paths, network-gated OpenAI-compatible provider, and independent network/runtime-adapter gates |
| `reproducibility` | Content pins for task/artifact/adapter inputs, environment freeze and effective experiment identity |
| `models`, `harnesses`, `domains`, `targets` | Provider/model/harness/domain-independent target matrix |
| `datasets`, `splits`, `task_seed_blocks` | 160/40/200 freeze, untouched tests, shared paired blocks and seeds |
| `objectives`, `constraints`, `statistics` | Four raw objectives, accuracy/token feasibility, bootstrap and pessimistic dominance |
| `budgets`, `selection_protocol` | Screening/full task-execution allocation, promotion and archive rules |
| `methods`, `ablations` | Matched controls, scalar/native adaptations, multi-objective baselines, full method and required ablations |
| `metrics`, `deployment`, `outputs` | Frontier/efficiency metrics, validation-only selection rules and required result artifacts |

The v2 template additionally declares local `runtime` inputs/verifiers, the built-in `openai_compatible` provider, maximize-space `selection_protocol.normalization_ranges`, and the staged `runner` contract (`preflight`, `smoke`, `search`, `final`, `all`). Its API credential field contains only the environment-variable name `PARETOSKILL_OPENAI_COMPATIBLE_API_KEY`; no key value belongs in YAML, resolved manifests, logs, or shell history.

`${PARETOSKILL_*}` placeholders are intentional. The dry-run profile substitutes only synthetic thresholds/output locations and leaves external placeholders unresolved but unused. A real profile rejects every unresolved placeholder.

Experiment IDs are SHA-256-derived from the canonical resolved manifest plus an
automatic digest of `src/`, `schemas/`, and `pyproject.toml`. Real preflight also
recomputes declared task-manifest and local artifact digests, so changing bytes at
the same path, implementation code, a model revision, seed, target, budget, or
adapter pin produces a different ID.

## Safety model

External execution requires all of the following:

1. a versioned manifest with `safety.allow_network: true`;
2. an explicit CLI `--allow-network` flag;
3. the exact value declared by `safety.required_network_env` and `required_network_value`; and
4. a reviewed copied manifest with pinned local data/verifier inputs, plus any required external runtime or optimizer adapter.

The checked-in manifests set network access to false or use a rejecting template sentinel; therefore they cannot make a real request even if a credential exists in the environment. Replay misses fail instead of falling back to a provider.

The package provides `MockProvider`, strict `ReplayProvider`, and `OpenAICompatibleProvider`. The latter uses the standard-library HTTPS transport (no vendor SDK), reads a key only at call time, enforces bounded requests/retries/responses, and is wrapped by the three network gates. The built-in provider is not a benchmark adapter: real execution still requires pinned local task manifests, base skill/evidence/patch assets, local domain/harness/verifier definitions, and—for formal `trace2skill_accuracy_subset`—a reviewed binary-BO adapter. Dynamic optimizer loading is disabled by default and requires an allowlisted `module:factory` reference plus the exact local module SHA-256.

`LocalDomainAdapter` and `VerifiedResponseHarness` remain the default runtime adapters. A custom domain or harness adapter is trusted local Python code and is never imported unless the active profile is `real` or `replay`, `safety.allow_dynamic_runtime_adapter_imports: true`, its `module:factory` module matches `safety.allowed_dynamic_runtime_adapter_prefixes`, and `adapter_sha256` exactly matches the local module bytes. Use `builtin_local_domain` or `builtin_verified_response` in adapter fields when the built-ins are intentional. The factory receives the fully resolved adapter/runtime specification and resolved local dataset roots; ParetoSkill itself does not interpret sandbox images, tool-step limits, or execute adapter commands. An API key does not install or authorize these adapters, and the external-verifier command gate remains independent.

## Future authorized real run

Do not edit the frozen v1 file or activate the v2 template in place. Copy the v2 template to a new versioned YAML and assign a new experiment ID, then:

1. supply local task-ID manifests, their SHA-256 values, dataset roots, and
   content pins for the base skill, traces, patch pool, dependency lock, and prompt;
2. freeze exact model and serving revisions, decoding, harness commits, sandbox image, tool limits, prices and retry/exclusion rules;
3. set the predeclared accuracy tolerance, token budget, archive capacity, hypervolume reference and deterministic final budget;
4. configure the built-in OpenAI-compatible provider, local harness/domain/verifier task definitions, any reviewed runtime `module:factory` adapters with allowlist and exact SHA pins, and any required binary-BO optimizer adapter;
5. run `validate-config --profile real` and task-manifest overlap/count preflight;
6. obtain separate authorization for API/network use and spending; and
7. execute the staged sequence `preflight` → offline `replay` → authorized small `smoke` → `search` → `final`, using the commands and artifact checks in the real-run guide.

The v2 runner command shape is:

```text
python -m paretoskill run CONFIG --profile replay|real \
  --stage preflight|smoke|search|final|all [--output-root PATH] \
  [--method ID ...] [--seed N ...] [--no-resume] [--allow-network]
```

An API key alone is insufficient. Until the copied manifest passes local content-pin, task-manifest, verifier, runtime-adapter, and optional optimizer-adapter preflight, real execution must stop at preflight and make no request. Activation must never be an implicit mock-to-network fallback.

## Package map

```text
src/paretoskill/
  models.py             Skill, Patch, TraceEvidence, VersionLineage
  materialize.py        deterministic patch composition and content-addressed lineage store
  providers.py          mock/replay/OpenAI-compatible providers and network safety
  evaluation.py         harness/domain contracts and paired task-block execution
  runtime_data.py       strict local task manifests, domain adapter and verifiers
  runtime_adapters.py   opt-in allowlisted/SHA-pinned local domain and harness factories
  runtime_assets.py     pinned base skill, evidence and patch-pool loaders
  runtime_factory.py    config-to-provider/runtime assembly without I/O calls
  costs.py              configured price validation and token-cost accounting
  statistics.py         four objectives and paired block bootstrap bounds
  objectives.py         conservative feasibility and point/pessimistic dominance
  archive.py            bounded non-dominated archive and JSON recovery
  proposer.py           archive-conditioned evidence selection and provider mutation proposer
  baselines.py          baseline plugin registry and clean-room adapted selectors
  search_strategies.py  deterministic subset, NSGA-II, Evo top-k and BO-adapter controllers
  ablations.py          executable configuration-override plugins
  metrics.py            HV, HVC, coverage, epsilon, IGD, crowding and false admissions
  deployment.py         three validation-only deployment selection rules
  config.py             YAML schema/cross-reference validation and experiment ID
  provenance.py         canonical file/tree hashing and real-run content-pin preflight
  task_manifests.py     local task-ID count/overlap preflight
  storage.py            result cache, JSONL schema, budget ledger and checkpoints
  runner.py             legacy synthetic dry-run
  experiment_runner.py  staged screen/promotion/full search and frozen final runner
  final_analysis.py     paired final-stage summaries and uncertainty intervals
  cli.py                configuration-driven CLI
```

JSON schemas live under `schemas/`. Public-code provenance and license decisions are recorded in [`docs/IMPLEMENTATION_SOURCES.md`](docs/IMPLEMENTATION_SOURCES.md).

## Third-party code policy

No source code, prompts, released skills, evaluator code, or benchmark data from Trace2Skill, Ctx2Skill, SkillForge, EvoSkill, SkillMOO, MOCHA, SkillsBench, or SpreadsheetBench is included. Where a paper discloses an algorithmic rule, this repository implements it independently and records the source/decision. See the source audit before adding any upstream material.
