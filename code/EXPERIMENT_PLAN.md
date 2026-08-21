# ParetoSkill ICLR 2027 Experiment Plan

Status: structural protocol freeze v1 plus a rejecting v2 real-run template; no empirical results have been produced.

The frozen protocol source of truth is [`configs/experiments/iclr2027.yaml`](configs/experiments/iclr2027.yaml). This document explains the decisions encoded there. That v1 file is a dry-run manifest and its `real` profile is a rejecting preflight stub, not a runnable online profile. [`configs/experiments/iclr2027_real_v2.template.yaml`](configs/experiments/iclr2027_real_v2.template.yaml) carries the same scientific protocol into an implemented staged provider/runtime/runner contract, but rejects real execution by default. A future real run requires copying v2 to a new versioned manifest and experiment ID, explicit activation of all three network gates, resolution of every required `${PARETOSKILL_*}` placeholder, matching local content pins, and all preflight checks below. Operational commands are in [`docs/REAL_RUN_GUIDE.md`](docs/REAL_RUN_GUIDE.md).

## 1. Questions and claims under test

The experiment is designed to answer four questions under matched **task-execution** budgets:

1. Does ParetoSkill find a larger, better-spread feasible frontier than scalar and multi-objective baselines?
2. Do its selected artifacts retain ID accuracy while improving worst-target transfer, token cost, or paired regression?
3. How quickly do feasible hypervolume and archive membership stabilize as task executions accumulate?
4. Which patch operations and evidence/lineage signals create real frontier improvements?

The central claim is falsified or materially weakened if objectives do not conflict, the archive collapses with more evaluation, accuracy-only or adapted multi-objective baselines match held-out feasible hypervolume, pessimistic bounds do not reduce false admissions, or archive-conditioned generation does not beat the passive archive at equal execution cost.

## 2. Safety and reproducibility contract

The checked-in configuration is deliberately offline:

- `experiment.mode: dry-run`;
- `safety.offline_by_default: true` and `safety.allow_network: false`;
- only `mock` and local `replay` providers are allowed in dry-run;
- `OpenAICompatibleProvider` is implemented but remains blocked by the rejecting template and three network gates;
- unresolved data paths, model revisions, prices, thresholds, or provider settings must fail a real-run preflight;
- no provider may be selected implicitly and no final-test result may flow back into search.

Network access for a future authorized run requires both a resolved provider configuration and the exact opt-in value specified by `safety.required_network_env`/`required_network_value`. That switch is a guard, not authorization to spend money. It also does not replace local task-manifest, verifier, base-skill/evidence/patch-pool, harness/domain, and content-pin preflight. The resolved manifest, with secrets removed, must be stored beside the results.

Experiment identity is the SHA-256 hash of the canonical **resolved and content-pinned** manifest. Before hashing, the resolver must embed SHA-256 digests or immutable revisions for every task manifest, dataset release, base skill, trace store, patch pool, verifier/provider/harness adapter, proposer prompt, materializer implementation, dependency lock, sandbox image, and execution backend. A path string alone is never an identity input. Thus changing content at the same path still creates a different experiment ID. Task outcomes and candidate evaluations are content-addressed and resumable.

## 3. Data freeze and leakage boundary

### Primary split

| Split | Count | Permitted use |
|---|---:|---|
| SpreadsheetBench-Verified evolution trace | 160 | Base traces, evidence extraction, patch generation |
| SpreadsheetBench-Verified ID validation | 40 | Search, confidence bounds, archive admission, deployment-policy fitting |
| SpreadsheetBench-Verified held-out | 200 | Final evaluation after all search outputs and policies are frozen |

The 160 and 40 tasks partition the 200-task evolution half; the 200 held-out tasks are untouched until final evaluation. The runner must reject overlapping task IDs.

WikiTableQuestions and HiTab conversions each have a separate 40-task transfer-validation manifest and a disjoint untouched test manifest. The full SpreadsheetBench test collection is final-only. Its canonical task IDs may overlap SpreadsheetBench-Verified: identical model--harness--task--seed executions are run once and reused, Verified-200 and full-collection aggregates are reported separately, and a task is never double-weighted in a pooled metric. The full-collection diagnostic must additionally stratify evolution-seen, ID-validation-seen, Verified-held-out, and full-only tasks; it is excluded from the primary held-out pooled endpoint. Their released sizes, exact IDs, identity mapping, and manifest digests are intentionally unresolved in v1 and must be supplied through the manifest environment variables; the runner must not infer a split after observing results.

Secondary scope tests on Ctx2Skill/CL-bench historical versions or a SkVM model-harness subset are not part of the primary matched comparison. They require a later versioned auxiliary configuration. The SkVM subset is admissible only after its released evaluator reproduces the documented no-skill/original-skill ordering.

## 4. Paired keys and resampling clusters

Every outcome/cache record is uniquely keyed by `(split_id, canonical_task_id, target_id, execution_seed)`. This is not the bootstrap cluster. Archive estimation resamples `(split_id, canonical_task_id, execution_seed)` clusters and retains every target observation belonging to a selected cluster; strata preserve split and transfer-group counts. Final method comparisons resample `(dataset_id, canonical_task_id)` clusters and retain all declared execution seeds and compatible targets for that task. Candidates, the base artifact, and compatible methods receive the same ordered outcomes. Infrastructure retries reuse the same seed and are not fresh observations.

The primary freeze uses three search seeds (`104729`, `130363`, `155921`), three execution seeds (`17`, `29`, `43`), and bootstrap RNG seed `32452843`. Seed choices are protocol constants, not claimed benchmark facts. A result is incomplete unless all declared search seeds are reported. Exclusion is allowed only for a logged, verifier-confirmed infrastructure failure under the common exclusion rule.

The base skill is evaluated on the same blocks before conditional regression is computed. When no block has `base_success = 1`, paired regression is undefined rather than silently set to zero.

## 5. Model-harness-domain target matrix

Exact checkpoints, revisions, adapter digests, serving backend, hardware, tool limits, sandbox image digest, dependency lock, and dated prices are `${PARETOSKILL_*}` placeholders. They must be embedded in the content-pinned manifest before the first main run.

| Phase | Target | Model role | Harness | Domain/split | Transfer group |
|---|---|---|---|---|---|
| Search | `id_small_primary` | source-family small user | primary | Verified ID-40 | ID objective, not transfer |
| Search | `cross_scale_large` | source-family large user | primary | Verified ID-40 | model scale |
| Search | `cross_harness_small` | source-family small user | alternate | Verified ID-40 | harness |
| Search | `ood_wtq_small` | source-family small user | primary | WTQ validation-40 | OOD domain |
| Search | `ood_hitab_small` | source-family small user | primary | HiTab validation-40 | OOD domain |
| Final | `final_verified_small` | source-family small user | primary | Verified held-out-200 | source held-out |
| Final | `final_verified_large` | source-family large user | primary | Verified held-out-200 | model scale |
| Final | `final_verified_heldout_family` | held-out model family | primary | Verified held-out-200 | model family |
| Final | `final_verified_alternate_harness` | source-family small user | alternate | Verified held-out-200 | harness |
| Final | `final_spreadsheetbench_full` | source-family small user | primary | full test | expanded source |
| Final | `final_wtq` / `final_hitab` | source-family small user | primary | untouched OOD tests | OOD domains |

The author model proposes patches but never grades them. Predicted effects from a proposer are metadata only; archive scores come exclusively from task execution.

## 6. Objectives and feasibility

All four objectives are retained as raw, interpretable measurements:

- **ID accuracy** (maximize): mean binary success on `id_small_primary`.
- **Worst-target transfer** (maximize): the minimum group mean over model-scale, harness, WTQ, and HiTab search targets.
- **Token cost** (minimize): mean input plus output tokens, including injected skill content and the executed trajectory.
- **Paired regression** (minimize): `P(candidate_fail | base_success)` on matched blocks.

Candidates must conservatively satisfy:

```text
LCB(candidate ID accuracy - base ID accuracy) >= -epsilon
UCB(candidate token cost) <= token budget B
```

The accuracy bound is computed from the paired per-block delta, not from separate candidate/base marginal lower bounds. `epsilon` and `B` are deployment choices, not paper facts. They remain `${PARETOSKILL_ACCURACY_TOLERANCE}` and `${PARETOSKILL_TOKEN_BUDGET}` until declared before the real comparison. Empty or nearly empty skills therefore cannot enter the feasible archive merely because they are cheap.

## 7. Evaluation budgets and promotion

Budgets count task executions, never proposals or generated artifacts.

| Stage | Frozen allocation | Purpose |
|---|---:|---|
| Screen | 40 executions/candidate | 8 SHA-256-selected tasks × 5 search targets × execution seed `17` |
| Full validation | 600 total executions/candidate | 40 tasks × 5 search targets × 3 execution seeds; includes the 40 screen outcomes |
| Promotion increment | 560 new executions/promoted candidate | Full matrix minus its cached screen subset |
| Total search | 30,000 executions/method/search seed | Shared hard cap for matched search methods |
| Budget curve | 7,500 / 15,000 / 22,500 / 30,000 | Predeclared efficiency checkpoints |
| Final | `${PARETOSKILL_FINAL_TASK_EXECUTIONS}` | All tasks in every frozen final manifest × all execution seeds |

The exact 30,000 schedule permits 386 unique screened candidates (15,440 executions) and 26 promotions (14,560 incremental executions), with the promotion order fixed by the selection rule. The screen subset uses SHA-256 with the literal salt `paretoskill-iclr2027-screen-v1`; collisions are resolved lexicographically. Every screen completes its 40 outcomes before a promotion/stop decision, so outcome-dependent partial screens cannot buy extra proposals. Screened candidates are promoted if conservatively feasible, if their interval crosses a feasibility boundary, or if interval overlap leaves them potentially non-dominated. A candidate that is provably unable to meet a constraint stops after screening. If a method cannot supply 386 unique valid screens or 26 eligible promotions, the run is marked incomplete and excluded from the matched primary comparison; unused allocation is never converted after seeing outcomes.

The final budget cannot be numeric until the unresolved full/OOD test manifests are frozen. It must equal the deterministic all-manifest allocation; post-outcome subsampling is forbidden.

## 8. Main comparison

All compatible search methods share the base traces, patch pool, verifier, materializer, outcome schedule, and logical task-execution cap. Physical cross-method cache reuse is allowed for efficiency, but a reused result consumes the same logical matched-budget units it would have consumed if executed anew. Controls and native methods with a different candidate space are reported separately and must not be mislabeled as matched comparisons.

| Config ID | Comparison implemented by the protocol |
|---|---|
| `no_skill` / `base_skill` | Empty-skill and configured-base controls |
| `simple_patch_composition` | Seeded random patch subsets with a simple feasible accuracy-then-token selector |
| `trace2skill_all` | Hierarchical aggregation/full merge of validated patches |
| `trace2skill_accuracy_subset` | Binary subset search selected by ID validation accuracy only |
| `fixed_scalarization` | Four independently budgeted runs: accuracy-only, accuracy-cost, equal four-objective weights, and fixed hard×easy product |
| `evoskill_scalar_topk` | Bounded top-k admitted and selected by a scalar held-out-validation score |
| `skillmoo_nsga2` | NSGA-II over the same materialized patch candidates and four measurements |
| `mocha_chebyshev_hvc` | Chebyshev selection plus hypervolume-contribution exploration over the same measurements |
| `passive_archive` | Pessimistic archive over an unconditioned common candidate stream |
| `paretoskill` | Pessimistic archive plus sparse-region parent choice and objective-conditioned evidence |

The Ctx2Skill-style entry is a fixed scalar selection rule, not a new objective: `hard_probe_success_rate × easy_probe_success_rate`. In the primary SpreadsheetBench adaptation it is evaluated over the common primary patch candidates, not CL-bench historical variants. Hard and easy probes are frozen from the bottom and top quartiles of the base task-level mean across execution seeds; ties are broken by canonical task ID and the two 10-task strata cannot overlap. A separate CL-bench historical-candidate reproduction requires an auxiliary manifest. EvoSkill's top-k is likewise treated as scalar rather than Pareto admission.

The executable parameter freeze is in the YAML. In summary: random composition uses a deterministic seeded Bernoulli-0.5 non-empty subset stream; accuracy-subset BO uses a required external binary-BO adapter with a frozen Matérn-5/2 GP and expected-improvement protocol (the built-in `initial_design_only` controller is not a GP/BO substitute); Evo uses a scalar top-26 incumbent set—26 is the promotion count required to close the matched `386×40 + 26×560 = 30,000` budget—and NSGA-II uses serializable ask/tell with population/offspring size 20, binary tournament, uniform crossover and `1/m` mutation; MOCHA uses the frozen Chebyshev/HVC policy over common evaluated candidates; ParetoSkill proposes batches of four with deterministic sparse-region/direction/operator tie-breaking. Each fixed-scalar variant receives an independent 30,000 logical-execution budget.

## 9. Required ablations

Each ablation keeps task blocks and task-execution budget fixed:

- point estimates instead of uncertainty bounds;
- remove paired regression from optimization;
- remove worst-target transfer from optimization;
- passive archive (archive does not guide generation);
- remove the feasibility gate;
- evidence-blind generation (failure labels only; verifier evidence hidden);
- lineage-blind generation (ancestry hidden from proposer);
- patch-subset-only (`add`/`drop`) versus full `add`/`drop`/`rewrite`/`compress` operations.

Evidence and lineage remain logged in their ablations so results are auditable; only the information exposed to candidate generation is removed. This avoids making the ablation irreproducible.

## 10. Statistics and reported metrics

Archive admission uses a 95% one-sided paired stratified bootstrap with 10,000 replicates over the archive clusters defined in Section 4. Benefit objectives use lower bounds; token and regression objectives use upper bounds. The accuracy feasibility gate instead bounds the paired candidate-minus-base accuracy delta. Robust dominance requires weak improvement in all pessimistic components and strict improvement in at least one, after conservative feasibility checking.

Final method differences use paired two-sided bootstrap intervals over task clusters, report the distribution over all three search seeds, and apply Holm correction only to the predeclared primary family: ParetoSkill versus each matched method/independently budgeted scalar variant on feasible hypervolume at 30,000 executions. Other objective and policy intervals are labeled secondary/exploratory. Worst-target transfer receives a predeclared lower-tail CVaR sensitivity analysis (`alpha = 0.25`). Hypervolume references and normalization ranges come from a frozen base/no-skill pilot manifest created before comparing methods.

Required output metrics are:

- the four raw objectives with paired intervals;
- feasible hypervolume, feasible coverage, additive epsilon indicator, and IGD to the pooled empirical front;
- feasibility rate, archive size, churn, and false admissions (screen admission rejected by full-validation pessimistic archive);
- task executions, search tokens, wall time, deployment tokens, amortized break-even reuse count, and budget curves;
- repair/regression counts, latency as a diagnostic, objective correlations, non-dominated fraction, patch-operation acceptance, and evidence-type acceptance.

Held-out hypervolume is post-hoc analysis only and must never affect search or selection.

## 11. Archive, final selection, and deployment

Admission first deduplicates by materialized content hash, then checks conservative feasibility and robust non-dominance. The online **working archive** is bounded by `${PARETOSKILL_ARCHIVE_CAPACITY}`; objective extremes are preserved before crowding-distance pruning. Separately, after each checkpoint and at finalization, an unbounded **scientific front** is reconstructed from every fully evaluated candidate under the same feasibility/dominance rule. Frontier metrics and the paper's scientific archive output use this unbounded reconstruction; the bounded archive is the search state. Every decision records the parent, patch/evidence IDs, objective estimates and bounds, and accept/reject reason.

Before any final manifest is opened, freeze:

- archive and baseline-selected outputs;
- normalization constants and hypervolume reference;
- accuracy tolerance, token budget, and archive capacity;
- all deployment-policy parameters.

The primary deployment policy is maximum worst-target transfer subject to the frozen token budget. Two additional policies are always reported: minimum tokens subject to the accuracy floor, and the normalized knee point. Policies are fitted on validation only. Ties are resolved by lower regression, then lower token cost, then lexicographic content hash.

## 12. Execution checklist

### Manifest boundary

- v1 remains the immutable offline/dry-run protocol artifact.
- v2 is the immutable, rejecting real-run template; it declares the built-in `openai_compatible` endpoint provider, local runtime/verifier inputs, timeout/retry policy, staged runner, checkpoint/resume contract, search-controller requirements, and required outputs.
- An authorized execution uses a copied v2 manifest with a new ID. The resolved copy freezes all paths, SHA-256 values, model/adapter revisions, prices, budgets, and maximize-space normalization ranges.
- The API key is never a manifest value. Only the environment-variable name is configured.

The runner stages are `preflight`, `smoke`, `search`, `final`, and `all`. Preflight cannot call a provider. Replay is offline and fails on a miss. Smoke uses the four named methods, one search seed, and **two relaxed blocks per search target**. Its frozen per-method candidate limits (no-skill one, base zero incremental, simple composition one, ParetoSkill two) make the whole smoke invocation at most 40 logical executions. The runner checks this bound before any provider call and writes smoke under its own namespace; smoke outputs never enter the main comparison. Formal screen/full allocations remain those frozen in `budgets`. Search uses only the frozen search targets. Final refuses until the search checkpoint and selection artifacts are frozen, then uses final targets without changing the archive or deployment policy.

### Offline development/dry-run

1. Load the YAML without custom YAML tags.
2. Validate schema types, cross-references, counts, disjointness declarations, at least three search seeds, and `shared_across_candidates: true`.
3. Confirm `mode = dry-run`, network is false, and the provider is `mock` or local `replay`.
4. Materialize synthetic candidates and print the resolved task-execution schedule without touching real datasets or providers.
5. Exercise checkpoint/resume and verify that the same resolved manifest yields the same experiment ID.

### Preflight before any future real run

1. Resolve and save exact data/task manifests; recompute every declared digest from local bytes, fail on any mismatch, and verify the 160/40/200 counts, canonical overlap mapping, and all disjointness checks.
2. Create a new manifest ID with `experiment.mode: real`; freeze model IDs and revisions, adapter/source digests, serving backend, hardware, decoding, tool-step cap, sandbox image digest, retry/exclusion taxonomy, dependency lock digest, proposer prompt digest, and materializer revision.
3. Resolve `epsilon`, token budget, archive capacity, final task-execution count, and pilot hypervolume reference.
4. Freeze the built-in OpenAI-compatible provider configuration/digest and dated token-price source; supply all local task manifests, verifier/harness/domain definitions, and content pins; install the required binary-BO adapter if running the Trace2Skill-style BO method; then set coherent real-profile/provider/network switches and obtain separate authorization for network use and cost. The checked-in manifest must continue to reject here.
5. Run base/no-skill pilot, then freeze strata, normalization, and hypervolume reference before main methods.
6. Run all methods on identical blocks and stop each matched search method at exactly 30,000 task executions per search seed.
7. Freeze archive, baseline outputs, and deployment policies; only then unlock final manifests.

## 13. Result artifacts

Each experiment directory must contain at least `resolved_manifest.yaml`, `run_metadata.json`, `task_outcomes.jsonl`, `candidates.jsonl`, bounded `archive.json`, unbounded `scientific_front.json`, `lineage.jsonl`, `metrics.json`, `token_accounting.json`, and `checkpoint.json`, plus periodic archive snapshots. Adaptive controllers checkpoint after each completed ask/tell batch; static methods rely on the per-execution content cache and write a completed-method checkpoint; the root contract is refreshed after each invocation. Resume is accepted only when candidate content hash, canonical task ID, target, seed, harness/model/provider revisions, and all content-pinned identity inputs match. Runtime metadata is captured from an explicit non-secret allowlist only; the process environment is never dumped or heuristically redacted.

No table cell in the paper should be filled until these artifacts pass completeness checks. Missing, failed, and excluded blocks must be reported explicitly rather than imputed.
