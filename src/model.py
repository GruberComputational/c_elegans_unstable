"""
Shared Langevin / first-passage-time (FPT) simulation utilities.

See Podolskiy et al. (arXiv:1502.04307v2) for the derivation linking these
functions to the paper's Langevin/first-passage-time model.
"""

from typing import Optional, Union

import numpy as np
from lifelines import KaplanMeierFitter


def simulate_fpt_and_state(
    n_paths: int,
    a: float,
    g: float,
    sigma_sq: float,
    z_death: float,
    z_reflect: float = 0.0,
    z0: Union[float, np.ndarray] = 0.0,
    dt: float = 0.1,
    t_max: float = 40.0,
    rng: Optional[np.random.Generator] = None,
    noise: Optional[np.ndarray] = None,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-Maruyama FPT simulation returning final state z(t_max).
    z0 can be a scalar or a (n_paths,) array (used for stage-2 hand-off).

    sigma_sq is the noise VARIANCE rate (paper's Delta, i.e. sigma^2 in the
    usual SDE convention dz = v(z) dt + sigma * dW). We inject
    sqrt(sigma_sq) * dW below so that Var[dz_noise] = sigma_sq * dt, matching
    E[dF'(t) dF'(t')] = Delta * delta(t - t') from the paper.

    beta: the exponent in the paper's general nonlinearity family,
    phi(z) = 1 + (z/Z)^beta (Eq. B27), giving drift v(z) = a*z*phi(z) =
    a*z + a*z^(beta+1)/Z^beta. `g` here is that coefficient a/Z^beta, i.e.
    g*z^(beta+1) is the nonlinear term -- NOT simply a/Z unless beta=1 (our
    original, default choice: v(z) = a*z + g*z^2, matching every existing
    caller). Larger beta makes the nonlinearity switch on much more sharply
    right at z=Z (more cliff-like) rather than gradually; see conversation
    history for why this is worth testing against DMSO's sharp late-life
    cutoff, which the beta=1 model's tail doesn't fully capture.

    noise: optional pre-generated (n_paths, n_steps) array of iid N(0, 1)
    draws, used in place of fresh rng draws. This is what lets curve-fitting
    code (see fit_simulation.py) evaluate this function at different
    (a, g, sigma_sq) with the *same* underlying randomness per path/step --
    "common random numbers" -- so the fitting objective varies smoothly with
    the parameters instead of jumping around due to fresh Monte Carlo noise
    at every evaluation. It must be indexed by each path's *original*
    position 0..n_paths-1 (not by its position within the shrinking alive
    set), which is why we index noise[idx_a, step] below rather than just
    drawing noise.size == idx_a.size fresh values: a path that dies later
    under one parameter set than another must still see the same noise
    draws at every step it remains alive, regardless of which other paths
    have already died.

    Args:
        n_paths: Number of independent paths to simulate.
        a: Linear drift coefficient.
        g: Nonlinear drift coefficient, a/Z^beta (see above).
        sigma_sq: Noise variance rate (paper's Delta).
        z_death: Absorbing threshold; a path is dead once z >= z_death.
        z_reflect: Reflecting boundary; z can't go below this value.
        z0: Initial state, scalar or per-path (n_paths,) array.
        dt: Euler-Maruyama time step.
        t_max: Simulation horizon; surviving paths are right-censored here.
        rng: Random generator to draw fresh noise from, if `noise` is not
            given. Defaults to `np.random.default_rng(42)`.
        noise: Optional pre-generated (n_paths, n_steps) iid N(0, 1) draws,
            for common-random-numbers curve fitting (see above).
        beta: Exponent in the drift nonlinearity v(z) = a*z + g*z^(beta+1).

    Returns:
        T: (n_paths,) first-passage times, `t_max` for paths still alive
            at the end of the simulation.
        z_final: (n_paths,) state at t_max (whatever it was at death, for
            paths that died, since z is not clamped to z_death itself).
        alive: (n_paths,) bool, True for paths that never crossed
            `z_death` within `t_max`.
    """
    if noise is None and rng is None:
        rng = np.random.default_rng(42)
    z = np.full(n_paths, float(z0)) if np.isscalar(z0) else np.asarray(z0, dtype=float).copy()
    fpt = np.full(n_paths, np.nan)
    alive = np.ones(n_paths, dtype=bool)
    sigma = np.sqrt(sigma_sq)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(round(t_max / dt))
    t = 0.0
    for step in range(n_steps):
        if not alive.any():
            break
        idx_a = np.where(alive)[0]
        z_a = z[idx_a]
        if noise is not None:
            step_noise = noise[idx_a, step] * sqrt_dt
        else:
            step_noise = rng.standard_normal(idx_a.size) * sqrt_dt
        z_new = z_a + (a * z_a + g * z_a**(beta + 1.0)) * dt + sigma * step_noise
        # Reflecting boundary at z_reflect: state can't fall below it
        # (mirror image of any overshoot back into the domain).
        z_new = np.where(z_new < z_reflect, 2.0 * z_reflect - z_new, z_new)
        crossed = z_new >= z_death
        fpt[idx_a[crossed]] = t + dt
        alive[idx_a[crossed]] = False
        z[idx_a] = z_new
        t += dt
    T = np.where(np.isnan(fpt), t_max, fpt)
    return T, z.copy(), alive


def simulate_trajectories(
    n_paths: int,
    a: float,
    g: float,
    sigma_sq: float,
    z_death: float,
    z_reflect: float = 0.0,
    z0: Union[float, np.ndarray] = 0.0,
    dt: float = 0.1,
    t_max: float = 40.0,
    rng: Optional[np.random.Generator] = None,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like simulate_fpt_and_state, but records and returns the *full* z(t)
    path for every one of n_paths, rather than just the final state --
    intended for a small n_paths, for visualizing individual trajectories.
    simulate_fpt_and_state deliberately doesn't keep this history, since it
    would be memory-heavy at the thousands-of-paths scale used for fitting;
    this is a separate, single-purpose function for the small-n_paths
    plotting case instead.

    Args:
        n_paths: Number of independent paths to simulate.
        a: Linear drift coefficient.
        g: Nonlinear drift coefficient, a/Z^beta.
        sigma_sq: Noise variance rate.
        z_death: Absorbing threshold; a path is dead once z >= z_death.
        z_reflect: Reflecting boundary; z can't go below this value.
        z0: Initial state, scalar or per-path (n_paths,) array.
        dt: Euler-Maruyama time step.
        t_max: Simulation horizon.
        rng: Random generator. Defaults to `np.random.default_rng(42)`.
        beta: Exponent in the drift nonlinearity v(z) = a*z + g*z^(beta+1).

    Returns:
        t_grid: (n_steps+1,) array of time points, 0..t_max.
        z_paths: (n_steps+1, n_paths) array of z(t) for each path. A path's
            entries are NaN for every time point *after* it first crosses
            z_death (its own death), including the reflecting-boundary
            behavior at z_reflect while still alive -- so plotting each
            column directly (e.g. ax.plot(t_grid, z_paths)) naturally stops
            the line exactly where that worm died, rather than drawing an
            unphysical flat "corpse" continuation or a line that keeps
            evolving past the absorbing threshold.
        alive: (n_paths,) bool, True for paths that never crossed
            `z_death` within `t_max` -- use this (not a NaN check on
            `z_paths[-1]`) to tell surviving paths apart from ones that
            happened to die on the very last recorded step, whose final
            entry is clipped to `z_death` rather than left NaN.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    z = np.full(n_paths, float(z0)) if np.isscalar(z0) else np.asarray(z0, dtype=float).copy()
    alive = np.ones(n_paths, dtype=bool)
    sigma = np.sqrt(sigma_sq)
    sqrt_dt = np.sqrt(dt)
    n_steps = int(round(t_max / dt))

    t_grid = np.linspace(0.0, t_max, n_steps + 1)
    z_paths = np.full((n_steps + 1, n_paths), np.nan)
    z_paths[0] = z

    for step in range(n_steps):
        idx_a = np.where(alive)[0]
        if idx_a.size == 0:
            break
        z_a = z[idx_a]
        noise = rng.standard_normal(idx_a.size) * sqrt_dt
        z_new = z_a + (a * z_a + g * z_a**(beta + 1.0)) * dt + sigma * noise
        z_new = np.where(z_new < z_reflect, 2.0 * z_reflect - z_new, z_new)
        crossed = z_new >= z_death
        z[idx_a] = z_new
        alive[idx_a[crossed]] = False
        # Only paths alive at the *start* of this step get a recorded value
        # here -- paths already dead before this step stay NaN (set by the
        # np.full initialization above), which is what makes their line stop
        # rather than continue.
        z_paths[step + 1, idx_a] = z_new
        # Euler-Maruyama's discrete step generally overshoots z_death rather
        # than landing exactly on it; clip the death step's recorded value
        # to z_death so plotted trajectories visibly stop at the absorbing
        # boundary instead of appearing to cross it.
        z_paths[step + 1, idx_a[crossed]] = z_death

    return t_grid, z_paths, alive


def km_from_fpt(
    T: np.ndarray, t_eval: np.ndarray, t_max: Optional[float] = None
) -> np.ndarray:
    """
    Kaplan-Meier survival curve evaluated at t_eval from an FPT array, using
    lifelines (a well-established, independently-validated survival-analysis
    package) rather than a hand-rolled KM implementation.

    Paths that never crossed z_death within the simulation window have
    FPT == t_max (see simulate_fpt_and_state); these are right-censored
    rather than treated as deaths, matching standard survival-analysis
    convention -- we only know they survived at least to t_max, not that
    they died at exactly t_max.

    Args:
        T: (n_paths,) first-passage times, as returned by
            `simulate_fpt_and_state`/`simulate_two_phase`.
        t_eval: Ages at which to evaluate the fitted survival curve.
        t_max: The simulation horizon used to produce `T`, i.e. the value
            that marks a path as censored rather than dead. Defaults to
            `T.max()`.

    Returns:
        Array parallel to `t_eval` with the fitted S(t) at each age.
    """
    if t_max is None:
        t_max = T.max()
    event_observed = T < t_max
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=event_observed)
    # survival_function_at_times looks up the right-continuous step value at
    # each requested time directly (S(t) = survival probability at the last
    # event time <= t), rather than linearly interpolating between the
    # sparse, unique event times -- which would visibly smear out the drops
    # in what should be a step function.
    return kmf.survival_function_at_times(t_eval).to_numpy()


def simulate_two_phase(
    n_paths: int,
    alpha1: float,
    Z1: float,
    alpha2: float,
    Z2: float,
    sigma_sq: float,
    t_switch: float,
    t_max: float,
    dt: float,
    noise1: Optional[np.ndarray] = None,
    noise2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate n_paths worms through two phases and return their absolute
    first-passage (death) times, t_max for survivors -- same convention as
    langevin.simulate_fpt_and_state, so the result feeds km_from_fpt
    directly.

    Phase 1 ((alpha1, Z1), t in [0, t_switch]): starts every worm at z=0.
    Phase 2 ((alpha2, Z2), t in [t_switch, t_max]): only run for worms still
    alive at t_switch, continuing from *that worm's own* z at hand-off
    (simulate_fpt_and_state's z0 argument accepts a per-path array for
    exactly this reason) -- not reset to 0, since the intervention doesn't
    erase accumulated state.

    noise1, noise2: optional common-random-numbers matrices, shaped
    (n_paths, n_steps_of_that_phase) -- see fit_simulation.make_common_noise.
    Indexed by each worm's *original* index (0..n_paths-1) in both phases
    (langevin.simulate_fpt_and_state already handles this correctly via its
    own `idx_a` indexing -- see that function's docstring), so which worms
    happen to survive phase 1 under a given (alpha1, Z1) does not change
    which noise draws a given worm sees in phase 2.

    Args:
        n_paths: Number of independent worms to simulate.
        alpha1: Linear drift coefficient during phase 1.
        Z1: Death threshold during phase 1.
        alpha2: Linear drift coefficient during phase 2.
        Z2: Death threshold during phase 2.
        sigma_sq: Noise variance rate, shared by both phases.
        t_switch: Absolute time at which phase 1 ends and phase 2 begins.
        t_max: Absolute simulation horizon (end of phase 2).
        dt: Euler-Maruyama time step, shared by both phases.
        noise1: Optional common-random-numbers matrix for phase 1, shaped
            (n_paths, n_steps_phase1).
        noise2: Optional common-random-numbers matrix for phase 2, shaped
            (n_paths, n_steps_phase2).

    Returns:
        (n_paths,) absolute first-passage (death) times, `t_max` for
        paths that survive both phases.
    """
    T1, z1_final, alive1 = simulate_fpt_and_state(
        n_paths, alpha1, alpha1 / Z1, sigma_sq, z_death=Z1,
        z0=0.0, dt=dt, t_max=t_switch, noise=noise1,
    )
    T_total = T1.copy()

    alive_idx = np.where(alive1)[0]
    if alive_idx.size > 0:
        noise2_subset = noise2[alive_idx] if noise2 is not None else None
        T2, _, _ = simulate_fpt_and_state(
            alive_idx.size, alpha2, alpha2 / Z2, sigma_sq, z_death=Z2,
            z0=z1_final[alive_idx], dt=dt, t_max=t_max - t_switch, noise=noise2_subset,
        )
        T_total[alive_idx] = t_switch + T2

    return T_total


def simulate_two_phase_trajectories(
    n_paths: int,
    alpha1: float,
    Z1: float,
    alpha2: float,
    Z2: float,
    sigma_sq: float,
    t_switch: float,
    t_max: float,
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like simulate_two_phase, but returns full z(t) trajectories across both
    phases (for visualization), analogous to langevin.simulate_trajectories
    -- and to that function's single-phase counterpart, used for the DMSO
    trajectory plot.

    "Alive at switch" is taken from simulate_trajectories' own `alive`
    return value, not inferred from NaN-ness of z_paths1[-1] -- a worm
    whose death is recorded on phase 1's very last step has that entry
    clipped to `Z1` rather than left NaN, so a NaN check would wrongly
    treat it as a phase-1 survivor and hand it off into phase 2 with
    `z0 = Z1`.

    Args:
        n_paths: Number of independent worms to simulate.
        alpha1: Linear drift coefficient during phase 1.
        Z1: Death threshold during phase 1.
        alpha2: Linear drift coefficient during phase 2.
        Z2: Death threshold during phase 2.
        sigma_sq: Noise variance rate, shared by both phases.
        t_switch: Absolute time at which phase 1 ends and phase 2 begins.
        t_max: Absolute simulation horizon (end of phase 2).
        dt: Euler-Maruyama time step, shared by both phases.
        rng: Random generator, shared across both phases. Defaults to
            `np.random.default_rng(42)`.

    Returns:
        t_grid: (n_steps_total+1,) absolute time points, 0..t_max.
        z_paths: (n_steps_total+1, n_paths) array. NaN after a path's own
            death, in either phase -- a worm that died during phase 1 never
            appears in phase 2 at all (it doesn't get reset or re-simulated),
            so its column stays NaN for the remainder of the array, exactly
            as langevin.simulate_trajectories already does within a phase.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t_grid1, z_paths1, alive_at_switch = simulate_trajectories(
        n_paths, alpha1, alpha1 / Z1, sigma_sq, Z1, z0=0.0, dt=dt, t_max=t_switch, rng=rng,
    )
    alive_idx = np.where(alive_at_switch)[0]

    n_steps2 = int(round((t_max - t_switch) / dt))
    z_paths2_full = np.full((n_steps2 + 1, n_paths), np.nan)
    if alive_idx.size > 0:
        z0_phase2 = z_paths1[-1, alive_idx]
        t_grid2, z_paths2_alive, _ = simulate_trajectories(
            alive_idx.size, alpha2, alpha2 / Z2, sigma_sq, Z2,
            z0=z0_phase2, dt=dt, t_max=t_max - t_switch, rng=rng,
        )
        z_paths2_full[:, alive_idx] = z_paths2_alive
    else:
        t_grid2 = np.linspace(0.0, t_max - t_switch, n_steps2 + 1)

    # Drop phase 2's first row (t=0 there, i.e. t_switch overall) -- it's
    # the same handoff value already recorded as phase 1's last row, so
    # keeping both would duplicate that time point.
    t_grid = np.concatenate([t_grid1, t_grid2[1:] + t_switch])
    z_paths = np.concatenate([z_paths1, z_paths2_full[1:]], axis=0)
    return t_grid, z_paths