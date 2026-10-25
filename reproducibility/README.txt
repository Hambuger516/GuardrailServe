GuardrailServe - final submission — companion reproducibility archive (25 October 2026)

Scope
This archive accompanies the paper and supplementary PDF. It contains compact
run exports, registered contrasts, CPU reconstruction scripts, source snapshots,
and the pinned PFS arithmetic conformance tests. It is separate from the clean
Overleaf source ZIP. No new GPU experiments were run for this packaging step.

Directory map
evidence/r12/              Shared-engine, component, graph and sensitivity studies
evidence/final_controls/   Model transfer, six-policy baseline and gamma arrivals
evidence/threshold_control/Admission-ceiling controls
evidence/pfs_conformance/  Public-artifact arithmetic and integer-block checks
evidence/diagnostics/     Completion/service flags and measurement boundaries
evidence/r15_r17/          Independent controls, longer arrivals, development,
                          and held-out replay/Dolly confirmation (17 bundles)
source_snapshots/         Available R15, R16 and R17 Python source snapshots
source_coverage.json      Per-registry source hashes and included exact matches

CPU verification (from the extracted archive directory)
python3 -m pip install -r requirements_cpu.txt
python3 verify_registered.py
python3 evidence/r12/verify_reported.py
python3 evidence/final_controls/verify_previous.py
python3 evidence/threshold_control/verify_threshold.py
cd evidence/pfs_conformance
python3 -m unittest -v test_pfs_fidelity.py
python3 check_block_peak.py

The first verifier checks file integrity, 188 registered exact sign-flip tests,
and their within-study/model Holm families. The historical reconstruction scripts
also recompute paired bootstrap intervals from compact per-run counters. They
write derived verification receipts next to their inputs; immutable input hashes
are listed in manifest.json. Existing receipts are historical provenance, not
claims that the original GPU runs were repeated during packaging.

Re-execution scope
These compact exports support independent numerical reconstruction. This is not
a self-contained GPU experiment image: model weights, complete prompt/history
pools, raw server event logs and every historical source revision are excluded.
Frozen study configurations are in each registry; software/input preparation and
measurement boundaries are described in supplementary Section S1 and the study
sections. Snapshot scripts can retain original deployment paths and need local
configuration before GPU use. A null exact match in source_coverage.json means
that precise historical source file is not supplied here. Do not substitute a
later source revision and claim bit-for-bit historical reproduction. In particular,
the shared R16 analysis script was amended after earlier C2/E analyses; the audit
receipt records that fact. No private model credentials are required for CPU checks.

Statistics and interpretation
Trace sets, rather than individual requests, are the independent analysis units.
Pointwise confidence intervals and multiplicity-adjusted tests answer different
questions. Nonsignificance is not equivalence. Replay uses sampled arrivals,
clipped lengths and length-matched surrogate prompts. Preserve all registered
outcomes and flagged traces when reconstructing results.
