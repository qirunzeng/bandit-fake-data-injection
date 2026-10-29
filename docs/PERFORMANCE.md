# Simulator optimization

The simulation still processes every genuine and fake record and consumes the same random draws in the same order. The attack definitions, thresholds, cost accounting, PBI scheduling rules, and matched-budget heuristic are unchanged.

Changes:

- Cache empirical means when their arm receives a record, avoiding division for every arm at every decision. Completion checks recompute means from the ledger to check the cache.
- Cache immutable learner settings and classify attack methods once per run.
- Compute the UCB exploration term only for UCB; TS keeps the same posterior draws.
- Stop scanning attack triggers once all arms have plans. Check PBI batches when the earliest batch is due, while retaining simultaneous-arm dispatch. Maintain the heuristic's remaining count when an injection occurs.
- Construct validation error strings only when a check fails.

## Validation

The optimized full run contains 17,920 method runs. Both raw CSV files are byte-identical to the pre-optimization publication results. Additional old/new comparisons cover small horizons, sensitivity grids, batch size one, recorded trajectories, and million-record runs. The first-trigger audit, integer reward search, and single-/multi-worker reproducibility checks also agree exactly on the same platform.

To compare two compiled versions:

```bash
python3 experiments/check_equivalence.py --baseline /path/to/previous/simulator
```

## Measurement

On the recorded Darwin arm64 machine, seven alternating single-worker benchmark pairs had median wall times of **0.4731 s before** and **0.4331 s after** (8.5% less elapsed time). Each invocation runs both learners and all four methods at T=1,000,000 and includes startup and CSV output. This is a local benchmark, not a promised speedup on every machine or workload.

Compiler details, all timings, source hashes, and result hashes are in `results/validation/optimization.json`. No numerical approximations or fast-forwarding were introduced.
