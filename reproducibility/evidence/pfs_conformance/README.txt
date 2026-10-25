PFS decision conformance (CPU only)
Requires Python and NumPy. From this directory run:
  python fetch_reference.py
  python -m unittest test_pfs_fidelity -v
  python check_block_peak.py
Fetching the two checksum-pinned official class files needs public GitHub access;
Overleaf compilation does not execute these tests or require network access.
The official files are fetched from commit d93ff69; their unmodified class bodies
are AST-extracted to skip GPU imports. The seven tests contain 120 seeded token
decision comparisons. The separate block test exhaustively enumerates time.
This does not claim native LightLLM runtime/throughput equivalence. Integer blocks,
the common class queue and scheduler, warm observed initialization, chunked prefill,
native recomputation and router cadence are disclosed integration boundaries.
