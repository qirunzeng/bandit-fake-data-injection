# Bandit Fake Data Injection

Reproducible experiments for targeted fake-data-injection attacks on stochastic bandits, with UCB and Gaussian Thompson Sampling learners.

This repository continues [richardhoff88/online_learning_torank](https://github.com/richardhoff88/online_learning_torank). The upstream Git history and [original attribution](docs/UPSTREAM_README.md) are retained. The original implementations and results are archived under `legacy/`; the corrected experiment pipeline is under `experiments/`.

## Reproduce the corrected experiments

Requirements: Python 3.9+ and a C++17 compiler (`clang++` or `g++`). The simulation and analysis use no third-party Python packages.

```bash
python3 experiments/run.py --threads 6
python3 experiments/report.py --output results/report
```

The first command builds and checks the simulator, then runs 20 repetitions at each of 112 horizons: K = 10, 100, 500, then 1,000 to 9,000 in steps of 1,000, and 10,000 to 1,000,000 in steps of 10,000. All measured points enter the curves; approximately 11 symbols are shown for readability. The second audits the counts and costs and exports the figures. Both panels use the same run endpoints at each horizon: final attack cost and final target-arm selection ratio. Both horizontal axes are linear. Straight segments connect measurements; no spline smoothing is used. Integer attack thresholds can produce genuine plateaus and jumps. Different horizons are simulated separately because the attack threshold depends on T; neither panel uses prefixes of a fixed-horizon run.

Optional vector PDF export requires pdfLaTeX with TikZ and PGFPlots:

```bash
python3 experiments/build_figures.py --report results/report
```

The current publication run is in `results/horizon/`; the earlier broad sweeps remain in `results/corrected/`. Its manifest records source hashes, input-data provenance, result hashes, and the experiment protocol.

## Main settings

| Setting | UCB | Thompson Sampling |
|---|---|---|
| Repetitions per setting | 20 | 20 |
| Arms | 10 sampled movie columns | Means 0.90, 0.85, ..., 0.45 |
| Genuine rewards | Bernoulli at the movie interaction rate | Gaussian, standard deviation 0.05 |
| Confidence parameter | 0.05 | 0.05 |
| Default attack parameter | 0.2 | 1.0 |
| Shared bounded fake reward | -70 | -70 |
| PBI maximum batch size | 10 | 10 |
| Maximum horizon | 1,000,000 records | 1,000,000 records |
| Horizon settings | 112 | 112 |

Both genuine and fake samples consume the same horizon. Every fake sample updates the same counts and empirical means used by the learner. Attack cost is always the sum of absolute deviations from the **true arm mean**; the number of injected samples is recorded separately.

The fixed injection interval is -70 to 1 for every bounded method. This is a broad suppression stress setting, not a binary-feedback validity constraint. Unit-range injection experiments are retained in the archived broad sweeps. LI remains an unbounded reference.

## Methods and feasibility

- **SBI:** compute the bounded injection budget at the pull-count threshold and process the batch without interleaved genuine actions.
- **PBI:** check the signed gaps and positive integer waiting times before executing the periodic schedule. Infeasible arm-level plans are recorded and withheld; their trials remain in every average. All arms due at the same epoch are checked against one pre-injection state and injected before any genuine pull. Only batches already overdue at that epoch are recertified.
- **LI:** inject one unrestricted sample per triggered non-target arm.
- **Matched-budget heuristic:** supply the actual per-arm SBI budget to a random-timing control. Counts and true-mean costs match SBI exactly; target-arm selections need not match. This control uses the paired SBI run's budget rather than estimating a budget online.

The shared reward -70 is the least negative **integer** passing the original 11-horizon calibration grid: all 880 SBI/PBI runs complete the nine non-target arms with no infeasible schedules and no subsequent pulls, including between batches. Every integer from -1 through -69 has a recorded failure witness. The additional T = K point contains only initialization, so its injection count and cost are zero and it adds no calibration constraint. The denser plotting grid retains this calibrated reward. Calibration uses the displayed seeds; it is not a universal or out-of-sample success guarantee. See [the scheduling audit](docs/PBI_AUDIT.md) and `results/reward-search/`.

## Results and additional checks

The current run contains **17,920 method runs** (2 learners × 112 horizons × 20 trials × 4 methods). At 1,000,000 records, UCB target-arm ratios are **99.68% for both SBI and PBI**; TS SBI/PBI both reach **99.99%**. Infeasible trials are included. The heuristic has the same endpoint cost as SBI but much lower target-arm ratios.

Recorded outputs include:

- `runs.csv`: seeds, parameters, costs, target ratios, feasibility flags, per-arm counts, suppression diagnostics, and simultaneous-batch counters.
- `trajectories.csv`: empty apart from its header for the endpoint-only experiment; no separate trajectory runs are needed.
- `manifest.json`: input, code, and result hashes.
- Exported `summary.csv`: sample counts, costs, feasibility, and target-arm ratios at every horizon.
- Archived sensitivity results remain in `results/corrected/`; regenerate them with `--sweeps --output results/corrected`.

To run a small pilot or an explicit parameter probe:

```bash
python3 experiments/run.py --quick --output results/pilot
python3 experiments/report.py --source results/pilot --audit-only
python3 experiments/run.py --ratio-only --lower-bound -36 --batch-size 10 \
  --seed-base 2026102901 --output results/probe
python3 experiments/check_reproducibility.py
```

The publication run uses seed base 3141592653, separate from the exploratory parameter probes. Arm-wise reward streams are paired across methods. Single-worker and multi-worker execution produce byte-identical results on the same platform.

See [the correction record](docs/CORRECTIONS.md) for changes from the archived implementation.
