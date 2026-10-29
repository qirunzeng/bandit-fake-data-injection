# Correction record

The corrected pipeline starts from upstream commit `dc2d3df799ec11d5d6111a5d8986c6d0dda41550`. The original code and saved plots are kept in `legacy/` for attribution and comparison. They are not inputs to the corrected figures.

| Area | Corrected behavior |
|---|---|
| Horizon | Exactly T genuine-plus-fake records, including initialization, for every method. |
| Learner statistics | One shared sample-count and empirical-mean state. Fake samples increment both and advance the same counter. |
| UCB rule | Empirical mean plus 3 sigma sqrt(log(next record)/count), with sigma=0.5 for Bernoulli rewards. |
| Trigger | Ceiling of log(T)/delta0 squared, matching the stated attack threshold. |
| MovieLens rewards | Genuine Bernoulli samples from column means; column means are not reused as deterministic feedback. |
| TS initialization | Actual genuine initial rewards populate the state used for all subsequent posterior draws and injection decisions. |
| TS noise | Standard deviation 0.05 for every genuine observation; no switch to an attack-specific reward noise. |
| Bounded rewards | Same fixed range and fake reward for SBI, PBI, and the heuristic. No adaptive lowering of the allowed bound. |
| Cost | Absolute fake-reward deviation from the true arm mean, for each actually processed fake record. |
| Heuristic | Exactly matched realized per-arm SBI counts, with randomized timing and no substituted saved arrays. |
| PBI gaps | Check their signs before squaring. Reject infeasible waiting intervals rather than clamping them to zero. |
| PBI execution | Serialize batches, use the actual final batch size, and recertify a delayed batch. Keep all infeasible trials. |
| Randomness | Explicit SplitMix64/Box-Muller streams, paired arm-wise observations, recorded seeds, and deterministic task collection. |
| Reporting | 20 repetitions for each learner, sample standard deviations, exact measured markers, true counts and costs. |

## PBI interpretation

The implementation computes a sufficient scheduling certificate. Rejecting a plan means this certificate could not justify the proposed schedule; it does not prove that every possible periodic attack must fail. Rechecking delayed batches is an implementation safeguard under the global record stream. This behavior is documented as certificate-checked PBI, rather than claiming the original theorem holds without additional conditions.

At fake reward -64 and batch size 10, the per-arm budgets in the main endpoint experiments fit in one batch. Coincident SBI/PBI curves in that setting do not establish a periodic-injection advantage. The batch-size and reward-strength sweeps include smaller, infeasible settings.

## Scope of the reward range

The main fixed range, -64 to 1, permits much stronger negative feedback than a system that enforces binary clicks. It is used as a controlled suppression setting. The separate unit-range injection check keeps fake rewards between zero and one; the tested SBI and checked PBI strategies cannot certify their plans there. This observation is not an impossibility theorem for other attacks.

## Validation

The simulator checks record totals, genuine/fake accounting, bounded rewards, unchanged target-arm provenance, horizon truncation, and a negative-gap counterexample. The reporting script independently recomputes bounded-method costs from the raw per-arm counts and true means, and verifies the heuristic's exact budget match. The reproducibility check compares complete pilot CSV files between one and three worker threads. CI runs the pilot, audit, and reproducibility checks.
