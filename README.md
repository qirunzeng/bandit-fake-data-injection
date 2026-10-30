# Bandit Fake Data Injection

[Code](experiments/) for fake-data-injection experiments with UCB and Thompson Sampling.

## Run

Requires Python 3.9+ and a C++17 compiler (`clang++` or `g++`). Run from the repository root:

```bash
python3 experiments/run.py
python3 experiments/report.py
```

To generate PDF figures, install pdfLaTeX with TikZ and PGFPlots, then run:

```bash
python3 experiments/build_figures.py
```

## Results

Saved figures:

| Learner | Attack cost | Target-arm selection ratio |
|---|---|---|
| UCB | [PDF](results/report/fig/ucb-cost-horizon.pdf) | [PDF](results/report/fig/ucb-selection-ratio.pdf) |
| Thompson Sampling | [PDF](results/report/fig/ts-cost-horizon.pdf) | [PDF](results/report/fig/ts-selection-ratio.pdf) |

- [Raw results](results/horizon/runs.csv)
- [Summary](results/report/fig/data/summary.csv)
