# Bandit Fake Data Injection

Reproducible experiments for targeted fake-data-injection attacks on stochastic bandits, with UCB and Gaussian Thompson Sampling learners.

This repository continues [richardhoff88/online_learning_torank](https://github.com/richardhoff88/online_learning_torank). The upstream Git history and [original attribution](docs/UPSTREAM_README.md) are retained. The original implementations and results are archived under `legacy/`; the corrected experiment pipeline is under `experiments/`.

## Reproduce the corrected experiments

Requirements: Python 3.9+ and a C++17 compiler (`clang++` or `g++`). The simulation and analysis use no third-party Python packages.

```bash
python3 experiments/run.py --threads 6
python3 experiments/report.py --output results/report
```

The first command builds the simulator, runs its correctness checks, verifies the supplied MovieLens matrix, and generates raw results. The second independently audits the sample counts and costs before exporting six TikZ figures and three tables.

Optional vector PDF export requires pdfLaTeX with TikZ and PGFPlots:

```bash
python3 experiments/build_figures.py --report results/report
```

The committed full run is in `results/corrected/`. Its manifest records source hashes, input-data provenance, result hashes, and the experiment protocol.

## Main settings

| Setting | UCB | Thompson Sampling |
|---|---|---|
| Repetitions per setting | 20 | 20 |
| Arms | 10 sampled movie columns | Means 0.90, 0.85, ..., 0.45 |
| Genuine rewards | Bernoulli at the movie interaction rate | Gaussian, standard deviation 0.05 |
| Confidence parameter | 0.05 | 0.05 |
| Default attack parameter | 0.2 | 1.0 |
| Shared bounded fake reward | -64 | -64 |
| PBI maximum batch size | 10 | 10 |
| Main trajectory horizon | 100,000 records | 100,000 records |

Both genuine and fake samples consume the same horizon. Every fake sample updates the same counts and empirical means used by the learner. Attack cost is always the sum of absolute deviations from the **true arm mean**; the number of injected samples is recorded separately.

The fixed injection interval is -64 to 1 for every bounded method. This is a broad suppression stress setting, not a binary-feedback validity constraint. Unit-range injection experiments are included separately. LI remains an unbounded reference.

## Methods and feasibility

- **SBI:** compute the bounded injection budget at the pull-count threshold and process the batch without interleaved genuine actions.
- **PBI:** check the signed gaps and positive integer waiting times before executing the periodic schedule. Infeasible arm-level plans are recorded and withheld; their trials remain in every average. Delayed batches are recertified before execution.
- **LI:** inject one unrestricted sample per triggered non-target arm.
- **Matched-budget heuristic:** supply the actual per-arm SBI budget to a random-timing control. Counts and true-mean costs match SBI exactly; target-arm selections need not match. This control uses the paired SBI run's budget rather than estimating a budget online.

At the main setting, each arm's required injection fits into one PBI batch. SBI and PBI therefore coincide there. The batch and reward sensitivity experiments report where the periodic certificate becomes infeasible; the main comparison does not establish a benefit from multiple periodic interventions. The checks do not resolve the outstanding mathematical conditions of the general PBI theorem.

## Results and additional checks

The full run contains **7,360 method runs**. At 100,000 records, mean SBI/PBI target-arm ratios are approximately **97.36% for UCB** and **99.88% for TS**. Failures and infeasible trials are included. The heuristic's low target ratios despite identical costs demonstrate the role of timing.

Recorded outputs include:

- `runs.csv`: seeds, parameters, costs, target ratios, feasibility flags, and per-arm genuine/fake/prescribed counts.
- `trajectories.csv`: measured target-arm ratio checkpoints.
- `manifest.json`: input, code, and result hashes.
- Exported `summary.csv` and sensitivity tables: sample counts, costs, and feasibility alongside target-arm ratios.

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
