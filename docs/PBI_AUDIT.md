# PBI scheduling audit

## Simultaneous due arms

The old dispatcher selected one due arm, injected its batch, then restarted scheduling. It did not insert a genuine pull while other arms remained due, but it incorrectly treated a peer batch from the same epoch as delayed and recertified it against the advanced state.

The dispatcher now collects all due arms and checks genuinely overdue batches before applying any injection. Every accepted arm receives its per-arm batch before another genuine pull. A regression test forces two simultaneous due arms and verifies both injections, no intermediate genuine pull, and no false recertification. A separate test checks truncation at the shared horizon.

A full rerun at the former reward -64 leaves all pre-existing result columns unchanged across 1,760 runs. These runs contain no simultaneous due arms. The two UCB rejections at T = 1,000,000 therefore have a different cause.

## First-batch feasibility

Write L for the target-arm lower confidence bound and g = L - post_mean for the first batch's signed gap. The proof needs

    3 sigma sqrt(log(t + f + R) / (N_i + f)) <= g.

Squaring is valid only when g is nonnegative. A nonnegative waiting interval also requires

    exp(g^2 (N_i + f) / (9 sigma^2)) >= t + f.

For a positive integer cooldown the right side becomes t + f + 1. Concentration of genuine rewards does not ensure either first-batch condition. The stated reward assumption makes the full injection budget feasible, but does not require the first f samples to suffice.

For the first publication trial at T = 1,000,000 and reward -64, the first arm triggers at t = 1176 with N_i = 346. Its budget is 12, but the first batch contains only 10 samples. The signed gap is positive (0.07030), yet the first waiting bound is -1183.81. There were no earlier injections or competing due arms. At reward -69 the bound is still negative (-59.06); at -70 it is 7125.55.

## Counterexample with no concentration failure

Take deterministic genuine rewards with means (0.90, 0.85, ..., 0.50, 0.30), K = 10, sigma = 0.5, delta = 0.05, delta0 = 0.2, f = 10, and T = 1,000,000. Deterministic distributions are 0.25-sub-Gaussian, and all genuine empirical means equal their true means: the concentration event holds with probability one.

At the first trigger, t = 1201, N_i = 346 and N_K = 25. With reward -64:

| Quantity | Value |
|---|---:|
| Reward-assumption cutoff | -5.40382 |
| Full prescribed budget | 11 |
| Target lower confidence bound | -0.716933 |
| Empirical mean after the first 10 samples | -0.923034 |
| Signed gap | 0.206100 |
| First waiting-time upper bound | -381.464674 |

Thus the reward assumption holds strictly, but no nonnegative waiting time satisfies the proof's first-batch sufficient inequality. This identifies a missing feasibility premise in that proof; it does not prove that all attacks fail at this reward or repair the theorem. The simulator retains signed-gap and positive-cooldown checks rather than silently clamping a negative wait to zero.

Reproduce the numerical audit from the repository root:

```bash
clang++ -std=c++17 -O3 -pthread experiments/audit_pbi.cpp -o build/audit_pbi
./build/audit_pbi
```

## Finite-grid reward calibration

The shared integer reward is searched in order -1, -2, ... . The calibration grid has two learners, 11 nontrivial horizons, 20 fixed repetitions and two attacks (SBI/PBI): 880 runs per fully evaluated candidate. Each rejected integer has one failure witness; screening stops early on that witness. A passing candidate must finish all nine non-target arms with zero infeasible schedules and zero genuine pulls of an arm after its first injection, including between batches.

The first passing integer is -70. All integers -1 through -69 fail. Repeating the search after the simultaneous-batch correction gives the same result. This is calibration on the publication seeds, not a mathematical guarantee or an independent estimate of success probability. Both plotted metrics still use the same 20 runs at each varying horizon; LI and the heuristic remain in the final four-method comparison.

```bash
clang++ -std=c++17 -O3 -pthread experiments/search_reward.cpp -o build/search_reward
./build/search_reward
```

Failure witnesses and first-trigger calculations are recorded in `results/reward-search/`; the complete calibrated run is in `results/horizon/`.

The plotted grid additionally includes T = K = 10. This endpoint contains initialization only: every method has zero injection cost and target-arm ratio 1/K. It imposes no additional condition on the calibrated reward.
