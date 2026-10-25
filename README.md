# GuardrailServe

Research artifacts for **GuardrailServe: Admission–Recovery Interactions in Large Language Model Serving**.

**Authors:** Hanbo Ma, Zhongguo Li, Zezhi Tang, Jun Yang, and Zhengtao Ding.

GuardrailServe coordinates request admission, KV-cache recovery, and a conditional token budget in LLM serving. This repository provides the source snapshots and compact experimental evidence accompanying the manuscript.

## Get the submission snapshot

Download [GuardrailServe_Reproducibility.zip](GuardrailServe_Reproducibility.zip) and extract it from this repository's root:

```bash
unzip -o GuardrailServe_Reproducibility.zip
cd reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_cpu.txt
python verify_registered.py
python evidence/r12/verify_reported.py
python evidence/final_controls/verify_previous.py
python evidence/threshold_control/verify_threshold.py
(cd evidence/pfs_conformance && python -m unittest -v test_pfs_fidelity.py && python check_block_peak.py)
```

On Windows, activate the environment with `.venv\Scripts\activate` and run the last two scripts from `evidence/pfs_conformance`.

The archive is the complete compact companion package for this submission. The source snapshots and verification scripts are also available as browsable files under [reproducibility/](reproducibility/). Download and extract the archive before running them: metric JSON/CSV inputs are distributed inside the archive.

## Artifact map

| Directory inside the extracted archive | Contents |
| --- | --- |
| `evidence/r12/` | Shared-engine PFS comparisons, component ablations, CUDA Graph, and budget sensitivity |
| `evidence/final_controls/` | Model transfer, baseline controls, gamma arrivals, and overhead measurements |
| `evidence/threshold_control/` | Admission-ceiling controls |
| `evidence/r15_r17/` | Independent component controls, longer arrivals, development studies, and held-out replay/Dolly confirmation |
| `evidence/pfs_conformance/` | Checksum-pinned PFS references and CPU decision-conformance tests |
| `evidence/diagnostics/` | Service flags and measurement-boundary diagnostics |
| `source_snapshots/` | Available R15–R17 Python source snapshots |
| `source_coverage.json` | Exact source-hash coverage by registered study |
| `manifest.json` | SHA-256 hashes of the frozen companion inputs |

## Verification and reproduction scope

CPU scripts reconstruct reported statistics from compact per-run exports. `verify_registered.py` checks packaged input integrity, 188 registered exact tests, and 29 Holm families. Historical bootstrap reconstruction and PFS decision tests are provided separately.

This package does not include model weights, all historical prompt/history pools, full raw server logs, or a self-contained GPU environment. Study registries and source coverage identify what is supplied. Deployment paths in historical snapshots require local configuration for GPU re-execution. No new GPU experiments were run to prepare this repository.

## Third-party materials and citation

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the pinned PFS source and Apache 2.0 license. Azure and Dolly are upstream data sources; follow their original access and licensing terms. Replay inputs use timestamps/token counts and surrogate prompts, as described in the manuscript.

Use [CITATION.cff](CITATION.cff) to cite this artifact. A Zenodo DOI has not yet been issued. Repository visibility does not itself grant a project-wide reuse license; see [RIGHTS.md](RIGHTS.md).
