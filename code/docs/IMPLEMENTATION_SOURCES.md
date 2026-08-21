# Implementation Sources and License Review

Snapshot date: 2026-08-20.

This review uses only papers, author/maintainer repositories, official project pages, and official documentation. Repository HEADs were checked with `git ls-remote` on the snapshot date, and source/license links below are pinned to the recorded commits wherever a repository exists. No third-party repository was vendored or copied into ParetoSkill, and `tmp/oss_references` was not created. The implementation in `src/paretoskill/` is a clean-room implementation written for this project.

## Adoption summary

| Work | Paper and official implementation | License status at snapshot | Relevant modules or ideas | Decision |
|---|---|---|---|---|
| Trace2Skill | [paper](https://arxiv.org/abs/2603.25158), [pinned Qwen-Applications/Trace2Skill](https://github.com/Qwen-Applications/Trace2Skill/tree/3d0b52a140f002a512930252b613c49048f7d5ac) (`3d0b52a140f002a512930252b613c49048f7d5ac`) | The pinned repository README says Apache-2.0, but no root `LICENSE`, `NOTICE`, or `COPYING` file was found. Some released xlsx artifacts have a separate restrictive [artifact license at the same revision](https://github.com/Qwen-Applications/Trace2Skill/blob/3d0b52a140f002a512930252b613c49048f7d5ac/released_skills/trace2skill-xlsx-35B-combined/LICENSE.txt). | Parallel success/error analysis, trajectory-local patches, hierarchical consolidation, SpreadsheetBench runner and evaluator. | Adopt the paper-level evidence-grounded patch abstraction and matched spreadsheet protocol. Do not copy source or any released skill artifact because the repository-level grant is incomplete and artifact terms differ. |
| Ctx2Skill | [paper](https://arxiv.org/abs/2604.27660), [pinned S1s-Z/Ctx2Skill](https://github.com/S1s-Z/Ctx2Skill/tree/7776821017c42b0afff403647f28379b9fb54f96) (`7776821017c42b0afff403647f28379b9fb54f96`) | The pinned README says MIT; no root license file and no detected SPDX license were found. | Five-role self-play, historical skill versions, Cross-Time Replay, fixed hard-probe × easy-probe selection. | Clean-room fixed-product selector and historical-candidate adapter only. Do not copy its API-bound code or prompts. |
| SkillForge | [paper](https://arxiv.org/abs/2604.08618) | No paper-linked official code repository or code license was located. The similarly named public tools and Alibaba Cloud skill collections are not established as the paper implementation. | Domain-contextualized creation; Failure Analyzer → Diagnostician → Optimizer; knowledge/tool/clarification/style failure categories. | Paper-level protocol reference only. The evidence/risk fields are independently implemented; no source or prompt is copied. |
| EvoSkill | [paper](https://arxiv.org/abs/2603.02766), [pinned sentient-agi/EvoSkill](https://github.com/sentient-agi/EvoSkill/tree/36f6f04952293d7054145550c2b9f0b0411bff1c) (`36f6f04952293d7054145550c2b9f0b0411bff1c`) | Apache-2.0 with a root [LICENSE at the pinned revision](https://github.com/sentient-agi/EvoSkill/blob/36f6f04952293d7054145550c2b9f0b0411bff1c/LICENSE). | Cache, harness, evaluation, loop and registry boundaries; bounded scalar top-k behavior. | License-compatible, but no lines were copied. ParetoSkill independently implements provider/harness registries, cache, and an explicit scalar top-k diagnostic because EvoSkill's scalar frontier is not the required four-objective archive. |
| SkillMOO | [paper](https://arxiv.org/abs/2604.09297), [pinned gjz78910/SkillMOO](https://github.com/gjz78910/SkillMOO/tree/739af0bf90d9cfdd35006af64d249e0c7ecc8c81) (`739af0bf90d9cfdd35006af64d249e0c7ecc8c81`), [official replication record](https://zenodo.org/records/19489028) | Public source repository, but no root license file, README license grant, or detected SPDX license was found. | NSGA-II survivor selection, candidate operators, metrics, matched seeds and frozen reports. | No repository code is reused. `NSGAIIPlugin` is a clean-room implementation of non-dominated sorting and crowding based on the standard NSGA-II paper cited by the manuscript. |
| MOCHA | [paper](https://arxiv.org/abs/2605.19330) | No author/Adobe official code repository or code license was located. A paper-content license is not a software license. | Random scalarization, Chebyshev parent selection, hypervolume-contribution exploration/gating, annealed acceptance. | Clean-room mathematical adapter only. ParetoSkill implements Chebyshev ranking and exact small-archive hypervolume contribution without copying prompts or code. |
| SkillsBench | [paper](https://arxiv.org/abs/2602.12670), [project](https://www.skillsbench.ai/), [pinned benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench/tree/9a1f4dd5f7659f75707435da3ce854b6e48321d1) (`9a1f4dd5f7659f75707435da3ce854b6e48321d1`), [v1.1 release](https://github.com/benchflow-ai/skillsbench/releases/tag/v1.1) | Apache-2.0 with a root [LICENSE at the pinned revision](https://github.com/benchflow-ai/skillsbench/blob/9a1f4dd5f7659f75707435da3ce854b6e48321d1/LICENSE). Individual task assets can have upstream terms and still require per-task review. | Native `task.md` packages, environment/skill/oracle/verifier layout, with-skill/no-skill pairing, registry hashes, local Docker path. | Future external benchmark adapter, pinned release, and user-supplied checkout only. Nothing is vendored; the built-in dry-run uses original synthetic fixtures. |
| SpreadsheetBench | [paper](https://arxiv.org/abs/2406.14991), [project page](https://spreadsheetbench.github.io/), [pinned RUCKBReasoning/SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench/tree/49b73a94775fb489063f60ca1865e3a650079a79) (`49b73a94775fb489063f60ca1865e3a650079a79`), [officially linked Verified-400 artifact](https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/ab0b742b0fc95b946f212d80ac7771b5531272e4/spreadsheetbench_verified_400.tar.gz) | The pinned README declares CC BY-SA 4.0 for the project; no root license file or detected SPDX code license was found. The linked Verified-400 artifact page declares CC BY-SA 4.0 and exposes an artifact SHA-256; exact data artifacts must still be pinned and reviewed at real-run freeze. | Official task manifests, spreadsheet evaluator, formula recalculation/parity checks, Verified-400 split. | Treat code and data as externally supplied benchmark material. Do not copy the evaluator or dataset into the core package. A future adapter must pin the checkout/dataset hashes and carry attribution/share-alike obligations separately. |

## What is and is not reused

No source lines, prompts, released skills, benchmark data, or evaluator assets from the works above are included. The following are independent implementations of published abstractions:

- `Patch` plus deterministic add/drop/rewrite/compress materialization: motivated by trajectory-local patch search, but implemented from the ParetoSkill manuscript's own data model.
- Provider/harness registries, result cache, and checkpointing: general interface patterns, not copied from EvoSkill.
- `NSGAIIPlugin`: standard non-dominated sorting and crowding-distance adaptation; no SkillMOO code used.
- `MOCHAPlugin`: clean-room normalized Chebyshev ranking plus hypervolume contribution; no official MOCHA code was available.
- `Ctx2SkillFixedProductPlugin`: the paper's disclosed hard × easy scalar rule over already measured candidates.
- `ParetoArchive`: project-specific pessimistic four-objective admission, conservative constraints, historical content-hash deduplication, capacity pruning, and recovery.

If future work copies or modifies Apache-2.0 code from EvoSkill or SkillsBench, the change must include file-level attribution, the upstream license/notice, the exact upstream revision, and a description of modifications. That is not the case in this snapshot.

## Direct dependencies

| Dependency | Role | Pin | Official source | License |
|---|---|---:|---|---|
| PyYAML | runtime YAML parser | `6.0.3` | [yaml/pyyaml tag 6.0.3](https://github.com/yaml/pyyaml/tree/6.0.3) | MIT ([pinned license](https://github.com/yaml/pyyaml/blob/6.0.3/LICENSE)) |
| setuptools | PEP 517 build backend | `81.0.0` | [pypa/setuptools v81.0.0](https://github.com/pypa/setuptools/tree/v81.0.0) | MIT ([pinned license](https://github.com/pypa/setuptools/blob/v81.0.0/LICENSE)) |
| pytest | development/test only | `9.1.1` | [pytest-dev/pytest 9.1.1](https://github.com/pytest-dev/pytest/tree/9.1.1) | MIT ([pinned license](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE)) |
| Ruff | development lint only | `0.13.2` | [astral-sh/ruff 0.13.2](https://github.com/astral-sh/ruff/tree/0.13.2) | MIT ([pinned license](https://github.com/astral-sh/ruff/blob/0.13.2/LICENSE)) |

These versions match `pyproject.toml`; none are vendored. A complete environment lock digest and sandbox image digest remain required real-experiment identity inputs because transitive dependencies and platform wheels are environment-specific.

## License-sensitive TODOs before real experiments

1. Re-check every upstream revision and license; repository terms can change after this snapshot.
2. Obtain and record the exact SpreadsheetBench data/evaluator attribution and share-alike handling for generated distributions.
3. Review licenses of every SkillsBench task, Docker image, oracle, and asset selected for an auxiliary study.
4. Do not use Trace2Skill released xlsx skill artifacts unless their specific artifact terms independently permit the intended operation.
5. Record any future adapter's source, revision, license, local modifications, and notices in this file before running it.
