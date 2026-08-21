# ParetoSkill real-run adapter and operations guide

Status: execution contract for the versioned v2 real template. The repository includes a built-in OpenAI-compatible provider, local task-manifest/domain/verifier runtime, opt-in local domain/harness factory loading, a staged runner, and deterministic search-controller implementations. It does not include benchmark data, a configured endpoint, credentials, a released benchmark-specific domain/harness/verifier adapter, a binary-BO optimizer adapter, or empirical results.

## 1. Which manifest to use

- `configs/experiments/iclr2027.yaml` is the frozen v1 offline manifest. Use it for config validation and the synthetic dry-run only.
- `configs/experiments/iclr2027_real_v2.template.yaml` is a rejecting real-run template. Never activate or overwrite it in place.
- For an authorized run, copy v2 to a new filename such as `iclr2027_real_v2_run-YYYYMMDD.yaml`, change `experiment.id`, and preserve that resolved file with the outputs.

The copied manifest remains blocked until all `${PARETOSKILL_*}` values are resolved; local task/data, base-skill, evidence, patch-pool, verifier, harness, and domain content hashes match; required runtime and optimizer adapters are installed and reviewed; `runtime_profiles.real.preflight_action` is changed from `reject_checked_in_manifest` to `require_resolved_adapter_and_three_network_gates`; and `safety.allow_network` is deliberately changed to `true`. Those edits create a new experiment identity. An API key by itself is never sufficient.

## 2. Adapter contract

The configured provider ID is `openai_compatible`. The runner passes the strict `providers.openai_compatible.factory` mapping to the built-in provider factory. It must contain only the supported non-secret fields: `type`, `provider_id`, `base_url`, `api_key_env`, `timeout_seconds`, `max_retries`, `retry_backoff_seconds`, `max_input_tokens`, `max_output_tokens`, `max_response_bytes`, and `allow_insecure_http`. The adapter must read:

- `providers.openai_compatible.factory.base_url` and `api_key_env`;
- exact `models.*.model_id`, `revision`, and decoding settings;
- request timeout, bounded exponential retry policy, token limits, and response-byte limit;
- token usage returned by the endpoint, without estimating silently;
- the configured price snapshot for accounting only.

`api_key_env` is an environment-variable **name**, not a secret. The adapter obtains the value at call time from `PARETOSKILL_OPENAI_COMPATIBLE_API_KEY`. It must never serialize that value into `resolved_manifest.yaml`, metadata, cache keys, request logs, exception text, or checkpoints. Missing credentials fail explicitly. The endpoint must be HTTPS unless a separately reviewed local-only manifest explicitly allows insecure HTTP. Endpoint/model failures cannot fall back to another provider or revision.

`LocalDomainAdapter` and `VerifiedResponseHarness` are the built-in defaults. Use the explicit adapter IDs `builtin_local_domain` and `builtin_verified_response` when those implementations are intended. A configured custom adapter is not supplied or authorized by an API key, and its module is not imported while `safety.allow_dynamic_runtime_adapter_imports` is false. Dynamic loading is allowed only for a `real` or `replay` profile after setting that switch to true, resolving `safety.allowed_dynamic_runtime_adapter_prefixes`, expressing each adapter as `module:factory`, and pinning the exact local module bytes in `adapter_sha256`. The module must equal or be below an allowed prefix.

The domain factory contract is `factory(domain_id=..., tasks=tuple[TaskSpec, ...], spec=..., runtime_spec=..., dataset_roots=..., base_directory=..., phase=...) -> TaskSpec | Iterable[TaskSpec]`. Returned values must be `TaskSpec` instances with the requested domain ID and unique task identities, and the transformed tasks must still close every configured target matrix. The harness factory receives `harness_id`, `default_harness`, `verifiers`, and the same resolved `spec`/runtime/dataset/path/phase context; it must return a harness with the exact requested `harness_id` and a callable `evaluate`. Factory import/call errors are wrapped and fail closed. Adapter specs—including sandbox image, `max_tool_steps`, and resolved local dataset roots—are passed through; the runtime-adapter loader does not execute external commands or interpret those sandbox controls. The reviewed adapter implementation remains trusted local code and is responsible for its own behavior.

Each executable task supplies a local verifier specification (`exact_match`, JSON boolean/field, built-in, or an explicitly enabled pinned external command). The verifier path, revision, and digest are frozen in `runtime` and `shared_search_controls`. Only a failure explicitly raised and recorded as a verified infrastructure failure is retry-eligible under a configured retry policy; other exceptions fail closed. Do not infer automatic task exclusion from the manifest taxonomy alone. External verifier commands are disabled by the v2 template and require the additional explicit `safety.allow_external_verifier_commands: true` setting in the copied manifest; this is independent of provider-network and dynamic-adapter approval.

The search layer has deterministic common-candidate/Bernoulli streams, Evo top-k, NSGA-II, and MOCHA ask/tell controllers with serializable state, and an archive-conditioned mutation proposer. `trace2skill_accuracy_subset` preserves its frozen GP/expected-improvement claim only through the configured external binary-BO adapter. The only built-in alternative is explicitly named `initial_design_only`; it must not be reported as Bayesian optimization. The formal adapter reference uses `module:factory`, its module must match `safety.allowed_dynamic_optimizer_prefixes`, its local source bytes must match `optimizer_adapter_sha256`, and loading remains disabled until `safety.allow_dynamic_optimizer_imports: true` is set in the reviewed copied manifest. The factory contract is `factory(patch_ids=..., seed=..., method_spec=...) -> BinarySubsetBayesianAdapter`.

The staged runner interface is:

```text
python -m paretoskill run CONFIG --profile replay|real \
  --stage preflight|smoke|search|final|all [--output-root PATH] \
  [--method ID ...] [--seed N ...] [--no-resume] [--allow-network]
```

This is the adapter/runner acceptance interface. Before attempting any stage, confirm `python -m paretoskill run --help` exposes these options; an older preflight-only runner is not authorized to improvise or bypass them.

`--method` and `--seed` may be repeated. Omission means the full configured set. Filtered search commands are partial runs and are not complete primary comparisons until every configured method and seed is present.

## 3. Inputs to freeze

Resolve every placeholder reported by:

```powershell
rg -o '\$\{PARETOSKILL_[A-Z0-9_]+\}' configs/experiments/iclr2027_real_v2.template.yaml | Sort-Object -Unique
```

The required categories are:

1. Local task/data inputs: dataset roots; evolution, ID-validation, held-out, WTQ, HiTab, and full SpreadsheetBench manifests; release/conversion revisions; lowercase SHA-256 values.
2. Local search inputs: base skill, trace store, patch pool, proposer prompt, dependency lock, materializer, verifier, harness, domain, and provider adapter paths/revisions/digests; custom runtime adapter `module:factory` references, their allowlisted module prefix, and exact module SHA-256 pins.
3. Model endpoint: OpenAI-compatible base URL, exact model IDs/revisions, output limits, timeouts, bounded retry values, price source/retrieval time, and the API-key environment-variable value supplied outside files.
4. Experiment constants: accuracy tolerance, token budget, archive capacity, final task-execution total, hypervolume reference, and eight finite normalization bounds.
5. Execution environment: backend revision/digest, hardware, driver/runtime, deterministic-kernel policy, sandbox image/digest, and tool-step cap.

Normalization ranges are in maximize space and must remain ordered as:

```text
[
  [id_accuracy_min, id_accuracy_max],
  [worst_transfer_min, worst_transfer_max],
  [negative_token_cost_min, negative_token_cost_max],
  [negative_paired_regression_min, negative_paired_regression_max]
]
```

Each minimum must be strictly below its maximum. Freeze these values from the predeclared base/no-skill pilot before evaluating compared methods; never recompute them from method outcomes.

The dataset/split manifests used only for count, identity, and overlap preflight may be `.jsonl`, `.json`, `.yaml`, or `.yml`. They may contain string IDs or objects with `task_id`/`id`; JSON/YAML mappings may place the list under `task_ids` or `tasks`. IDs must be unique. These lightweight manifests do not supply executable payloads or verifiers.

By contrast, every path in `runtime.task_manifests` is an **execution manifest** and must be UTF-8 `.json` or `.jsonl`. A JSON file is exactly `{"schema_version": 1, "tasks": [...]}`; JSONL contains one task object per non-empty line. Every task object has exactly `schema_version`, `task_id`, `split_id`, `domain_id`, `group_id`, `objective_roles`, `payload`, and `verifier`; its `schema_version` is `1`, roles are non-empty and unique, payload is finite JSON, and verifier follows the strict runtime verifier schema. Relative paths resolve from the run manifest directory. Preflight/runtime loading recomputes digests, checks declared counts, and closes the target/task/seed matrix before any provider call.

## 4. Safety gates

A real provider call requires three independent gates:

1. the copied run manifest has `safety.allow_network: true` and a coherent real runtime/provider;
2. the command includes `--allow-network`;
3. `PARETOSKILL_ENABLE_NETWORK` exactly equals `I_UNDERSTAND_EXTERNAL_CALLS_MAY_COST_MONEY`.

The v2 template additionally has `preflight_action: reject_checked_in_manifest`, so it fails before these gates can authorize a call. Remove that sentinel only in a copied, reviewed manifest. The environment gate is not budget approval; obtain cost authorization separately. Replay never accepts `--allow-network`, and a replay miss fails without network fallback.

Local dynamic runtime adapters have a separate four-part gate: `real` or `replay` profile, `allow_dynamic_runtime_adapter_imports: true`, an exact allowed module prefix, and an exact lowercase SHA-256 pin for the local module containing each `module:factory`. The default false switch prevents import. Optimizer imports and external verifier commands retain their own independent switches. None of these gates is enabled by supplying an API key.

Do not print or dump the environment during diagnosis. Inject the API key through the approved secret manager or parent process, and verify only that the named variable is present—not its value.

## 5. Staged execution

The examples assume execution from `code/` and a copied manifest in `$RunConfig`.

```powershell
$env:PYTHONPATH = "src"
$RunConfig = "configs/experiments/iclr2027_real_v2_run-YYYYMMDD.yaml"
$OutputRoot = "../tmp/paretoskill-real"
```

### 5.1 Default-rejection check

Before copying/activation, confirm the template refuses real validation. A nonzero exit is expected and no request is made:

```powershell
python -m paretoskill validate-config configs/experiments/iclr2027_real_v2.template.yaml --profile real
```

### 5.2 Preflight

After resolving the copied manifest, preflight validates configuration, local paths/digests, split counts/overlaps, normalization ranges, targets, budgets, and output contract. It makes no provider call and therefore does not use `--allow-network`:

```powershell
python -m paretoskill validate-config $RunConfig --profile real
python -m paretoskill run $RunConfig --profile real --stage preflight --output-root $OutputRoot
```

Save the printed experiment ID and review the redacted `resolved_manifest.yaml` before any online stage.

### 5.3 Offline replay

Point `PARETOSKILL_REPLAY_PATH` to a complete local JSONL replay store. First run replay preflight, then the bounded smoke matrix. Neither command may use the network:

```powershell
python -m paretoskill run $RunConfig --profile replay --stage preflight --output-root $OutputRoot
python -m paretoskill run $RunConfig --profile replay --stage smoke --output-root $OutputRoot
```

The smoke contract uses **two relaxed blocks per target**, one search seed, and four named methods. Candidate limits are one no-skill condition, zero incremental base-skill conditions, one simple composition, and two ParetoSkill candidates, for at most 40 logical task executions across the whole invocation. The runner rejects a method/seed/target drift or ceiling overflow before any provider call. It is a safety/integration check, not a relaxation of the frozen formal screen or full protocol. Smoke outputs live in the separate `smoke/` namespace and cannot be promoted into the main comparison.

### 5.4 Authorized live smoke

Only after adapter review, preflight/replay success, secret injection, and spending approval, enable the three gates and run the same bounded smoke stage:

```powershell
$env:PARETOSKILL_ENABLE_NETWORK = "I_UNDERSTAND_EXTERNAL_CALLS_MAY_COST_MONEY"
python -m paretoskill run $RunConfig --profile real --stage smoke --output-root $OutputRoot --allow-network
```

Stop if the output exceeds the smoke candidate/two-block/budget ceiling, any model revision differs, token accounting is missing, retries exceed the manifest, or a required artifact is absent.

### 5.5 Search and final

Run the frozen search matrix with resume enabled by default:

```powershell
python -m paretoskill run $RunConfig --profile real --stage search --output-root $OutputRoot --allow-network
```

For isolating work, repeated filters are allowed; the checkpoint remains incomplete until the full matrix is present:

```powershell
python -m paretoskill run $RunConfig --profile real --stage search --output-root $OutputRoot --method paretoskill --seed 104729 --allow-network
```

The `final` stage must refuse unless search artifacts, scientific front, baseline selections, normalization constants, hypervolume reference, and deployment policy are frozen. It must not generate candidates or change selection:

```powershell
python -m paretoskill run $RunConfig --profile real --stage final --output-root $OutputRoot --allow-network
```

For a fresh authorized run, `--stage all` is shorthand for search, freeze, then final. Do not use it to bypass the review boundary. `--no-resume` disables checkpoint reuse; use it only with a new output directory. Resume must fail when the experiment ID or any content digest differs.

## 6. Output acceptance

Every completed requested stage writes into the resolved experiment-ID directory. Check the required set without displaying secrets:

```powershell
$RunDirectory = Join-Path $OutputRoot '<resolved-experiment-id>'
$Required = @(
  'resolved_manifest.yaml', 'run_metadata.json', 'task_outcomes.jsonl',
  'candidates.jsonl', 'archive.json', 'scientific_front.json',
  'lineage.jsonl', 'metrics.json', 'token_accounting.json', 'checkpoint.json'
)
$Missing = $Required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $RunDirectory $_)) }
if ($Missing) { throw "Missing required artifacts: $($Missing -join ', ')" }
```

Also verify:

- `resolved_manifest.yaml` contains no key value or unapproved environment data;
- `run_metadata.json` records the experiment ID, code/adapter revisions, backend/hardware, stage, command, and start/end time;
- `task_outcomes.jsonl` has no missing or unclassified block and uses the frozen task/target/seed keys;
- `token_accounting.json` reconciles logical budget, physical cache reuse, provider usage, and dated prices;
- `checkpoint.json` marks only actually completed stages and lists incomplete methods/seeds/targets;
- search uses no final split, final changes no archive/policy, and smoke is excluded from scientific metrics.

Unclassified exceptions and incomplete matrices fail closed; no missing block may be imputed. A retry is acceptable only when the output contains the explicit verified-infrastructure failure evidence and retry metadata required by the active evaluator policy. Do not assume automatic exclusion merely because the manifest declares an eligible category. Never fill a paper table from smoke, replay, synthetic, incomplete, or preflight outputs.
