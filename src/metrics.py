"""
Goodness-of-fit statistics for comparing a fitted/simulated survival curve
against an empirical Kaplan-Meier curve.
"""

import numpy as np


def goodness_of_fit(
    S_model_at_emp: np.ndarray, S_emp: np.ndarray
) -> dict[str, float]:
    """
    Compute RMSE and KS distance between a model curve and the empirical
    curve, evaluated at the empirical curve's own distinct observed ages
    (not an arbitrary dense grid, so the statistic is anchored to actual
    data points).

    - RMSE: root-mean-square difference in S(t), in the same [0, 1] units
      as the survival probability itself -- a simple, reportable summary of
      typical deviation across the whole curve.
    - KS: the Kolmogorov-Smirnov distance, max_t |S_model(t) - S_emp(t)| --
      the standard goodness-of-fit statistic for comparing a fitted
      distribution against an empirical one, reporting the single worst
      point of disagreement rather than an average.

    Args:
        S_model_at_emp: Model/simulated survival probabilities, evaluated
            at the same ages as `S_emp`.
        S_emp: Empirical (Kaplan-Meier) survival probabilities.

    Returns:
        Dict with keys "rmse" and "ks".
    """
    S_model_at_emp = np.asarray(S_model_at_emp, dtype=float)
    S_emp = np.asarray(S_emp, dtype=float)
    diff = S_model_at_emp - S_emp
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "ks": float(np.max(np.abs(diff))),
    }
