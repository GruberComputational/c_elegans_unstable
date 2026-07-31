"""
Analytic Gompertz survival model, and simulation-based median lifespan.

`survival_analytic` is the closed-form expression from Podolskiy et al.
(arXiv:1502.04307v2), Supplementary Information Appendix B, Section 2 ("A
toy model: absorbing wall at a large z = Z"), specialized to a population
started at z = 0 (the "very narrow initial distribution", sigma_0 -> 0,
z_0 = 0 case) -- matching the z0=0.0 default in model.simulate_fpt_and_state:

    survival_analytic : Eq. B30, S(t) = erf(sqrt(alpha*Z^2 / (2*sigma_sq*X(t))))

This formula is exact for the pure-linear drift v(z) = alpha*z with an
absorbing wall at z = Z (the beta -> infinity case of the general
nonlinearity phi(z) = 1 + (z/Z)^beta in Eq. B27). The paper argues (end of
Appendix B, Section 2.c) that this closed form is nevertheless a good
approximation for any finite beta -- including our beta = 1 simulation,
v(z) = alpha*z + g*z^2 with g = alpha/Z -- as long as the population is in
the weak-nonlinearity/"Gompertzian" regime, alpha*Z^2/sigma_sq >> 1 (Eq.
B12).

`median_survival_time` does NOT use this closed form: see its own
docstring for why it instead simulates the actual beta=1 model directly.

sigma_sq below is Delta, the noise VARIANCE rate from the paper (see
model.py's module docstring) -- not a standard deviation.
"""

from typing import Optional

import numpy as np
from lifelines import KaplanMeierFitter
from scipy.special import erf

from model import simulate_fpt_and_state


def survival_analytic(
    t: np.ndarray, alpha: float, Z: float, sigma_sq: float
) -> np.ndarray:
    """Closed-form survival function S(t), Eq. B30."""
    t = np.asarray(t, dtype=float)
    # expm1 avoids catastrophic cancellation of exp(2*alpha*t) - 1 for
    # small t, where X(t) is small but not zero.
    X = np.expm1(2.0 * alpha * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = np.sqrt(alpha * Z**2 / (2.0 * sigma_sq * X))
    return erf(arg)


def median_survival_time(
    alpha: float,
    Z: float,
    sigma_sq: float,
    n_paths: int = 20_000,
    dt: float = 0.1,
    t_max: float = 300.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    The model's implied median time-to-death, read off the empirical
    Kaplan-Meier curve of `n_paths` simulated first-passage times from the
    actual beta=1 model (`model.simulate_fpt_and_state`, drift
    v(z) = alpha*z + g*z^2 with g = alpha/Z) -- not from `survival_analytic`.

    `survival_analytic` (Eq. B30) is only exact for the beta -> infinity
    (pure-linear-drift, hard-absorbing-wall) limit, and is stated by the
    paper to remain a good approximation for our beta=1 model only in the
    weak-nonlinearity regime, alpha*Z^2/sigma_sq >> 1 (Eq. B12) -- e.g.
    `GAMMA_MIN=50` in auto_fit_parameters_mle.py. A caller sweeping
    parameters (see notebooks/99_parameter_sensitivity.py) can easily land
    well below that threshold at the edges of a sweep, where Eq. B30 no
    longer reliably tracks the simulated model -- simulating directly is
    correct at any gamma, at the cost of actually running the simulation
    instead of a cheap root-find.

    This turns the abstract, unitless threshold Z into something directly
    comparable to the data: a predicted lifespan in days.

    Returns np.nan if fewer than half of the simulated paths die within
    `t_max` (the empirical survival curve never reaches 0.5) -- widen
    `t_max`, or treat as a sign the parameters are implausible.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    g = alpha / Z
    T, _, _ = simulate_fpt_and_state(
        n_paths, alpha, g, sigma_sq, Z, z0=0.0, dt=dt, t_max=t_max, rng=rng,
    )
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=T < t_max)
    median = kmf.median_survival_time_
    return float(median) if np.isfinite(median) else np.nan
