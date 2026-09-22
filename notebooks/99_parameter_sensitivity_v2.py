# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dynamics
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parameter sensitivity: what happens when only `alpha` or only `g` changes?
#
# This is an exploratory supplementary analysis: the Langevin drift is
# `v(z) = alpha*z + g*z^2`, so `alpha` and `g` are the two literal
# coefficients in the equation as written. This notebook checks how
# sensitive the model's predicted survival curve is to each one
# individually, holding the *other coefficient* (not `Z`) fixed, since
# `Z = alpha/g` is itself derived from whichever of `alpha`/`g` is held
# fixed:
#
# - **Sweep A**: vary `alpha`, hold `g` fixed at its DMSO-fitted value.
#   `Z = alpha/g` then moves *with* `alpha`.
# - **Sweep B**: vary `g`, hold `alpha` fixed at its DMSO-fitted value.
#   `Z = alpha/g` then moves *inversely* with `g`.
#
# **Baseline**: `(alpha, g, sigma)` are taken directly from
# `notebooks/0_auto_fit_parameters_mle.py`'s Step 1 fit -- the same numbers
# used in `2_pub_figures_auxin10_switch.py`'s published figures, not a
# separate, independently-fit baseline.
#
# `sigma_sq` is held fixed throughout this sweep, since the question here is
# specifically about `alpha` and `g` -- not because `sigma_sq` is assumed
# constant in general (it is itself a fitted quantity in
# `0_auto_fit_parameters_mle.py`, estimated jointly with `alpha`/`g` from
# the DMSO data by maximum likelihood, then held fixed across conditions
# there).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTEBOOK_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SRC_DIR = NOTEBOOK_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

NOTES_DIR = NOTEBOOK_DIR / "notes"
NOTES_DIR.mkdir(exist_ok=True)

from model import km_from_fpt, simulate_fpt_and_state, simulate_trajectories
from survival import median_survival_time, survival_analytic

FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]  # +/-50%, +/-25%, baseline

# %% [markdown]
# ## Baseline: the DMSO operating point actually used in Figures 1-3
#
# Hardcoded from `2_pub_figures_auxin10_switch.py`'s "Fixed parameters"
# cell rather than refit here, so this analysis can never silently drift
# from what the published figures actually show.

# %%
alpha_base, log_g_base, sigma_base = 0.1553, -2.5439, 1.5941
g_base = 10.0 ** log_g_base
Z_base = alpha_base / g_base
SIGMA_SQ = sigma_base ** 2
gamma_base = alpha_base * Z_base**2 / SIGMA_SQ
print(f"Baseline (DMSO, from 2_pub_figures_auxin10_switch.py): alpha={alpha_base:.5g}, Z={Z_base:.5g}, "
      f"g={g_base:.5g}, sigma_sq={SIGMA_SQ:.5g}, gamma={gamma_base:.5g}")

t_plot = np.linspace(0, 30, 300)

# %% [markdown]
# ## Note on the exploratory curves
#
# The quick `S(t)` plots in Sweeps A/B below use the closed-form
# `survival_analytic` (Eq. B30) purely for speed; that expression is an
# approximation that holds for `gamma >> 1`. All reported median lifespans
# (`median_survival_time`) and the manuscript figure further down are
# simulated directly from the beta=1 model with the reflecting boundary at
# z=0, so they are valid across the whole sweep.

# %% [markdown]
# ## Sweep A: vary `alpha`, hold `g` fixed

# %%
rows_a = []
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(FACTORS)))
for factor, color in zip(FACTORS, colors):
    alpha = alpha_base * factor
    Z = alpha / g_base  # g fixed -> Z moves with alpha
    gamma = alpha * Z**2 / SIGMA_SQ
    mrdt = np.log(2) / alpha
    med_t = median_survival_time(alpha, Z, SIGMA_SQ)
    rows_a.append({"factor": factor, "alpha": alpha, "Z": Z, "gamma": gamma, "mrdt": mrdt, "median_lifespan": med_t})
    ax.plot(t_plot, survival_analytic(t_plot, alpha, Z, SIGMA_SQ), color=color,
            label=f"alpha x{factor:.3g} (alpha={alpha:.3g}, Z={Z:.3g})")
ax.set_xlabel("Time (days)")
ax.set_ylabel("S(t)")
ax.set_title("Sweep A: vary alpha, g fixed at DMSO baseline")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

sweep_a = pd.DataFrame(rows_a)
print(sweep_a.to_string(index=False))

# %% [markdown]
# ## Sweep B: vary `g`, hold `alpha` fixed

# %%
rows_b = []
fig, ax = plt.subplots(figsize=(8, 6))
for factor, color in zip(FACTORS, colors):
    g = g_base * factor
    Z = alpha_base / g  # alpha fixed -> Z moves inversely with g
    gamma = alpha_base * Z**2 / SIGMA_SQ
    mrdt = np.log(2) / alpha_base
    med_t = median_survival_time(alpha_base, Z, SIGMA_SQ)
    rows_b.append({"factor": factor, "alpha": alpha_base, "Z": Z, "g": g, "gamma": gamma, "mrdt": mrdt, "median_lifespan": med_t})
    ax.plot(t_plot, survival_analytic(t_plot, alpha_base, Z, SIGMA_SQ), color=color,
            label=f"g x{factor:.3g} (g={g:.3g}, Z={Z:.3g})")
ax.set_xlabel("Time (days)")
ax.set_ylabel("S(t)")
ax.set_title("Sweep B: vary g, alpha fixed at DMSO baseline")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

sweep_b = pd.DataFrame(rows_b)
print(sweep_b.to_string(index=False))

# %% [markdown]
# ## Direct comparison: which parameter moves the outcome more?
#
# Using `median_survival_time` as the outcome metric (captures both
# parameters' effects, unlike MRDT), compare the range spanned by the
# same +/-50%/+/-25% perturbation applied to each parameter individually.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(FACTORS))
width = 0.35
ax.bar(x - width / 2, sweep_a["median_lifespan"], width, label="vary alpha (g fixed)", color="tab:blue")
ax.bar(x + width / 2, sweep_b["median_lifespan"], width, label="vary g (alpha fixed)", color="tab:orange")
ax.set_xticks(x)
ax.set_xticklabels([f"x{f:.2g}" for f in FACTORS])
baseline_median = sweep_a.loc[sweep_a["factor"] == 1.0, "median_lifespan"].item()
ax.axhline(baseline_median, color="black", lw=1, ls=":", label="baseline")
ax.set_xlabel("Perturbation factor")
ax.set_ylabel("Median simulated lifespan (days)")
ax.set_title("Outcome sensitivity: alpha-only vs. g-only perturbation")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

range_a = sweep_a["median_lifespan"].max() - sweep_a["median_lifespan"].min()
range_b = sweep_b["median_lifespan"].max() - sweep_b["median_lifespan"].min()
print(f"Median-lifespan range across the sweep: alpha-only = {range_a:.3g} days, g-only = {range_b:.3g} days")
print(f"gamma range: alpha-only = [{sweep_a['gamma'].min():.4g}, {sweep_a['gamma'].max():.4g}]   "
      f"g-only = [{sweep_b['gamma'].min():.4g}, {sweep_b['gamma'].max():.4g}]")

# %% [markdown]
# **Result**: decreasing either parameter extends median survival and
# increasing either shortens it. For equal relative changes, `alpha` has the
# somewhat larger effect on the median lifespan.

# %% [markdown]
# ## Shape of the change: interquartile range
#
# `alpha` and `g` change the survival curve differently: in the Gompertzian
# closed form (Eq. B30) time enters only through `gamma * exp(-2*alpha*t)`,
# so a change in `g` (via `gamma`) translates the curve in time, while a
# change in `alpha` also rescales time and so stretches or compresses it.
# We quantify the shape with the interquartile range of the simulated
# lifespans, IQR = T25 - T75 (times at which S = 0.25 and S = 0.75).
# Same seed and settings as `median_survival_time`, so the paths are the
# ones behind the reported T50.

# %%
from lifelines import KaplanMeierFitter
from model import simulate_fpt_and_state


def lifespan_iqr(alpha, Z, sigma_sq, n_paths=20_000, dt=0.1, t_max=300.0, seed=0):
    T, _, _ = simulate_fpt_and_state(
        n_paths, alpha, alpha / Z, sigma_sq, Z, z0=0.0, dt=dt, t_max=t_max,
        rng=np.random.default_rng(seed),
    )
    sf = KaplanMeierFitter().fit(T, event_observed=T < t_max).survival_function_
    t_at = lambda p: float(sf.index[np.argmax(sf.values[:, 0] <= p)])
    return t_at(0.25) - t_at(0.75)


sweep_a["iqr"] = [lifespan_iqr(a, z, SIGMA_SQ) for a, z in zip(sweep_a["alpha"], sweep_a["Z"])]
sweep_b["iqr"] = [lifespan_iqr(a, z, SIGMA_SQ) for a, z in zip(sweep_b["alpha"], sweep_b["Z"])]
print(sweep_a[["factor", "median_lifespan", "iqr"]].to_string(index=False))
print(sweep_b[["factor", "median_lifespan", "iqr"]].to_string(index=False))

# %% [markdown]
# ## Connecting back to trajectories (Figure 2 style)
#
# The same two sweeps, shown as representative simulated trajectories
# rather than population survival curves, to make the effect on individual
# dynamics as visually direct as Figure 2 itself.

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), sharey=True)

for ax, (sweep_name, param_name) in zip(axes, [("alpha", "alpha (g fixed)"), ("g", "g (alpha fixed)")]):
    for factor, color in zip([0.5, 1.0, 1.5], plt.cm.viridis(np.linspace(0, 1, 3))):
        if sweep_name == "alpha":
            alpha, g = alpha_base * factor, g_base
        else:
            alpha, g = alpha_base, g_base * factor
        Z = alpha / g
        t_grid, z_paths, _ = simulate_trajectories(
            8, alpha, g, SIGMA_SQ, Z, z0=0.0, dt=0.1, t_max=30.0, rng=np.random.default_rng(3),
        )
        for i in range(8):
            ax.plot(t_grid, z_paths[:, i], color=color, lw=1.0, alpha=0.7,
                    label=f"x{factor:.3g}" if i == 0 else None)
    ax.set_xlabel("Time (days)")
    ax.set_title(f"Vary {param_name}")
    ax.legend(fontsize=8, title="factor")

axes[0].set_ylabel("z(t)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# 1. `alpha` and `g` were swept individually (+/-25%, +/-50%) around the
#    DMSO fitted baseline, holding the other coefficient and `D` fixed.
# 2. Decreasing either parameter extends median survival; for equal
#    relative changes `alpha` has the somewhat larger effect.
# 3. The shape of the change differs: changing `g` shifts the survival
#    curve in time with nearly unchanged spread (IQR), whereas changing
#    `alpha` also stretches or compresses it.

# %% [markdown]
# ## Manuscript-ready outputs (Supplementary Note 1)
#
# Everything above is the exploratory working-through of the sensitivity
# question. This section renders the two deliverables actually referenced
# by `notebooks/supplementary_note1_sensitivity.tex` -- one figure and one
# LaTeX table -- using the manuscript's own notation (`alpha`, `g`,
# `Z = alpha/g`, `D` with `sigma_sq = 2*D`) and its plotting style
# (`2_pub_figures_auxin10_switch.py`'s rcParams).
#
# The curves plotted here are **simulated directly**
# (`model.simulate_fpt_and_state` + `model.km_from_fpt`, the same beta=1
# model and Kaplan-Meier machinery used throughout this project), not the
# closed-form `survival_analytic` used for the quick exploratory plots
# above -- so this figure is correct across the whole sweep, including
# Sweep A's low-`gamma` edge (see the note above).

# %%
matplotlib_pub_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": False,
}
plt.rcParams.update(matplotlib_pub_rc)
cm = 1 / 2.54

D_base = SIGMA_SQ / 2.0
N_SIM = 20_000
DT_SIM = 0.1  # matches the dt used for fitting in 0_auto_fit_parameters_mle.py
T_MAX_SIM = 60.0
t_eval = np.linspace(0.0, T_MAX_SIM, 400)


def simulated_survival_curve(alpha, Z, sigma_sq, seed):
    """Simulated S(t) on `t_eval`, from the real beta=1 model -- not the
    closed-form Eq. B30 approximation."""
    g = alpha / Z
    T, _, _ = simulate_fpt_and_state(
        N_SIM, alpha, g, sigma_sq, Z, z0=0.0, dt=DT_SIM, t_max=T_MAX_SIM,
        rng=np.random.default_rng(seed),
    )
    return km_from_fpt(T, t_eval, t_max=T_MAX_SIM)


fig_s1, axes_s1 = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharey=True)
sweep_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(FACTORS)))

for factor, color in zip(FACTORS, sweep_colors):
    alpha = alpha_base * factor
    Z = alpha / g_base
    S = simulated_survival_curve(alpha, Z, SIGMA_SQ, seed=10)
    t50 = median_survival_time(alpha, Z, SIGMA_SQ, n_paths=N_SIM, dt=DT_SIM, t_max=300.0, rng=np.random.default_rng(10))
    axes_s1[0].plot(t_eval, S, color=color, label=rf"$\alpha$ x{factor:.3g}")
    if np.isfinite(t50):
        axes_s1[0].plot(t50, 0.5, "o", color=color, ms=3, mec="black", mew=0.3, zorder=5)

for factor, color in zip(FACTORS, sweep_colors):
    g = g_base * factor
    Z = alpha_base / g
    S = simulated_survival_curve(alpha_base, Z, SIGMA_SQ, seed=11)
    t50 = median_survival_time(alpha_base, Z, SIGMA_SQ, n_paths=N_SIM, dt=DT_SIM, t_max=300.0, rng=np.random.default_rng(11))
    axes_s1[1].plot(t_eval, S, color=color, label=rf"$g$ x{factor:.3g}")
    if np.isfinite(t50):
        axes_s1[1].plot(t50, 0.5, "o", color=color, ms=3, mec="black", mew=0.3, zorder=5)

axes_s1[0].axhline(0.5, color="gray", lw=0.6, ls=":", zorder=0)
axes_s1[1].axhline(0.5, color="gray", lw=0.6, ls=":", zorder=0)
axes_s1[0].set_title(r"Sweep A: vary $\alpha$ ($g$ fixed)")
axes_s1[1].set_title(r"Sweep B: vary $g$ ($\alpha$ fixed)")
axes_s1[0].set_xlabel("Time (days)")
axes_s1[1].set_xlabel("Time (days)")
axes_s1[0].set_ylabel("Survival probability $S(t)$")
axes_s1[0].legend(loc="upper right")
axes_s1[1].legend(loc="upper right")
axes_s1[0].set_xlim(0, T_MAX_SIM)
axes_s1[1].set_xlim(0, T_MAX_SIM)
axes_s1[0].set_ylim(-0.03, 1.03)

fig_s1.tight_layout(pad=0.6)
fig_s1.savefig(NOTES_DIR / "supp_note1_fig.png", bbox_inches="tight", dpi=600)
plt.show()

# %% [markdown]
# ### LaTeX table
#
# Combines both sweeps into the single table `\input` by
# `supplementary_note1_sensitivity.tex`.

# %%
from decimal import Decimal, ROUND_HALF_UP


def _fmt(x, decimals):
    """Round half-up from the value's shortest repr, so e.g. 0.1035 prints as
    0.104 (matching main-text Table I) rather than float-artefact 0.103."""
    if pd.isna(x):
        return "--"
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


table_a = sweep_a.copy()
table_a.insert(0, "sweep", r"$\alpha$ varies")
table_a["g"] = g_base

table_b = sweep_b.copy()
table_b.insert(0, "sweep", r"$g$ varies")

table_cols = ["sweep", "factor", "alpha", "g", "Z", "median_lifespan", "iqr"]
supp_table = pd.concat(
    [table_a[table_cols], table_b[table_cols]], ignore_index=True
)
supp_table["factor"] = supp_table["factor"].map(lambda f: "--" if pd.isna(f) else f"{f:g}")
supp_table["alpha"] = supp_table["alpha"].map(lambda v: _fmt(v, 4 if v < 0.1 else 3))
supp_table["g"] = (supp_table["g"] * 1e3).map(lambda v: _fmt(v, 2))
supp_table["Z"] = supp_table["Z"].map(lambda v: _fmt(v, 1))
supp_table["median_lifespan"] = supp_table["median_lifespan"].map(lambda v: _fmt(v, 1))
supp_table["iqr"] = supp_table["iqr"].map(lambda v: _fmt(v, 1))
supp_table = supp_table.rename(columns={
    "sweep": "Perturbation",
    "factor": "Factor",
    "alpha": r"$\alpha$ (d$^{-1}$)",
    "g": r"$g$ ($10^{-3}$)",
    "Z": r"$z_{max}$",
    "median_lifespan": r"$T_{50}$ (d)",
    "iqr": r"IQR (d)",
})

latex_body = supp_table.to_latex(
    index=False, escape=False,
    column_format="l" + "r" * (len(table_cols) - 1),
)
# Rule between the two sweeps.
body_lines = latex_body.split("\n")
first_row = body_lines.index(r"\midrule") + 1
n_a = len(table_a)
body_lines.insert(first_row + n_a, r"\midrule")
latex_body = "\n".join(body_lines)

latex_table = (
    "% Auto-generated by notebooks/99_parameter_sensitivity.py -- do not edit by hand.\n"
    "\\begin{table}[htbp]\n\\centering\n"
    + latex_body
    + "\\caption{One-at-a-time sensitivity of survival to $\\alpha$ and $g$. "
      "Each parameter scaled by the given factor around the DMSO "
      "baseline ($\\alpha_0=" + _fmt(alpha_base, 3) + "$~d$^{-1}$, $g_0="
      + _fmt(g_base * 1e3, 2) + r"\times10^{-3}$, $z_{max,0}=" + _fmt(Z_base, 1)
      + "$) with the other held fixed. "
        "$D=" + _fmt(D_base, 2) + "$ throughout. $T_{50}$ is the median of $2\\times10^4$ "
        "simulated first-passage times (Euler--Maruyama, $\\Delta t=0.1$); "
        "IQR is $T_{25}-T_{75}$, the interquartile range of the same simulated lifespans.}"
      "\n\\label{tab:supp_note1_sensitivity}\n\\end{table}\n"
)
(NOTES_DIR / "supp_note1_table.tex").write_text(latex_table)
print(latex_table)
