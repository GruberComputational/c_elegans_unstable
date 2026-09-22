# Supplementary table: model predictions and post-intervention survival

## A. Whole-cohort model scores

| Condition | Model | N | log L | k | Nominal AIC |
| --- | --- | --- | --- | --- | --- |
| DMSO day 10 | Langevin | 232 | -404.61 | 3 | 815.22 |
| DMSO day 10 | Gompertz | 232 | -342.64 | 2 | 689.28 |
| Auxin day 10 | Langevin | 81 | -212.62 | 2 | 429.24 |
| Auxin day 10 | Gompertz | 81 | -164.52 | 2 | 333.04 |
| Auxin day 21 | Langevin | 247 | -459.86 | 0 | 919.72 |
| Auxin day 21 | Gompertz | 247 | -688.39 | 0 | 1376.79 |

## B. Conditional post-day-21 survival

| Model | End day | Observed RMST | Predicted RMST | Delta RMST | p (greater) | p (curve) |
| --- | --- | --- | --- | --- | --- | --- |
| Gompertz | 35 | 7.80 | 2.65 | +5.15 | 1.00e-04 | 1.00e-04 |
| Langevin | 35 | 7.80 | 8.10 | -0.31 | 0.690 | 0.520 |
| Gompertz | 39 | 7.90 | 2.65 | +5.25 | 1.00e-04 | 1.00e-04 |
| Langevin | 39 | 7.90 | 8.89 | -0.99 | 0.908 | 0.067 |

## C. Paired predictive comparison (Langevin minus Gompertz)

| Follow-up | N | Mean difference | 95% bootstrap CI | Normal p |
| --- | --- | --- | --- | --- |
| Full follow-up | 59 | +4.04 | [2.60, 5.59] | 1.84e-07 |
| Through day 35 | 59 | +3.95 | [2.58, 5.40] | 6.12e-08 |
| Through day 39 | 59 | +4.04 | [2.60, 5.59] | 1.84e-07 |

A: Whole-cohort grouped log-likelihoods use unmodified model probabilities. k counts parameters estimated from the named condition; Auxin day 21 is held out (k=0). Langevin retains the original simulation-based training parameters; Gompertz is refitted without floors. Nominal AIC is descriptive, not a significance test. -- denotes unresolved simulated probability, for which no finite score is reported. B: RMST is inspection-time restricted remaining survival in days, conditional on the 59 animals observed after day 21. Delta is observed minus predicted. The one-sided test concerns greater RMST; the curve test uses integrated squared distance. 10,000 replicates; minimum reported p=1.00e-04. C: Differences are nats per worm; positive values favor Langevin. Intervals use 20,000 paired worm-bootstrap replicates; normal p is a two-sided asymptotic diagnostic, not a bootstrap p-value. Full follow-up and day 39 coincide numerically because all observed deaths occurred by day 39. B/C assume independent worms and fixed fitted predictions, excluding training-fit and simulation uncertainty. Endpoints are retrospective, p-values unadjusted, and analyses exploratory.

# Figure caption

Survival curves for (A) DMSO day 10, (B) Auxin day 10, and (C) Auxin day 21. Colored steps show observed Kaplan-Meier survival; black dashed curves show Langevin predictions; magenta dash-dot curves show Gompertz predictions. Panels A and B show training conditions. Panel C shows held-out two-phase predictions using parameters from the training conditions, with a vertical dotted line at intervention on day 21. All curves show unconditional survival from the original cohort; the conditional post-day-21 tests are reported in the supplementary table. Langevin curves use 1,000,000 simulated paths per curve.
