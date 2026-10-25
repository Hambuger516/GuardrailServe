Supporting diagnosis of existing experiments; no new GPU runs.
cohort_boundary_check.json reconstructs SLO attainment from the frozen R12 run
counters, including drain completions and every planned request in the denominator.
flagged_pairs.json contains the eight raw-run diagnostic summaries, class counts,
observed maximum inter-token gaps and SHA256s of the underlying event/metric files.
The raw request/event records were read on the experiment host; they remain in the
original archive and are not copied into this Overleaf source ZIP. Selected flags
do not estimate a population frequency or supply a recovery-off counterfactual.
The stored post-warmup reconstruction counters may include shutdown scheduling;
the diagnostic preemption counts are bounded at the service observation cutoff.
