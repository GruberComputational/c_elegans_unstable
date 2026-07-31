
"""
Loader for the raw (non-digitized) C. elegans lifespan data in
`data/raw/`.

Each file is a *grouped life table*, not one row per worm: every row is
`(Age at Death [days], Frequency [# worms], Censor)`. Summing `Frequency`
per file gives the true cohort sizes (232 / 81 / 247 for
DMSO_day_10 / Auxin_day_10 / Auxin_day_21).

IMPORTANT convention note: this file's `Censor` column is 1 for a censored
observation (removed/lost without being observed to die) and 0 for an
observed death -- the *opposite* of `lifelines`' `event_observed`
convention (1 = event/death, 0 = censored). We invert it once here
(`event_observed = 1 - censor`) so every downstream consumer
gets data in the convention it expects, rather than re-deriving the
inversion (and risking getting it backwards) in multiple places.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_RAW_FILES = {
    "DMSO_day_10": "dmso_day_10.csv",
    "Auxin_day_10": "auxin_day_10.csv",
    "Auxin_day_21": "auxin_day_21.csv",
}

@dataclass
class GroupedSurvivalData:
    """
    Grouped time-to-event data for one condition.

    t, event_observed, weight are parallel arrays: `weight[i]` worms were
    observed at duration `t[i]`, of which the outcome was death if
    `event_observed[i] == 1`, or censoring (removed/lost) if 0.
    """

    t: np.ndarray
    event_observed: np.ndarray
    weight: np.ndarray

    @property
    def n_total(self) -> int:
        return int(self.weight.sum())
    
def load_raw_survival_data() -> dict[str, GroupedSurvivalData]:
    """
    Load all three conditions' raw grouped life-table data.

    Returns:
        Dict mapping condition name ("DMSO_day_10", "Auxin_day_10",
        "Auxin_day_21") to its `GroupedSurvivalData`.
    """
    return {name: _load_one(RAW_DATA_DIR / fname) for name, fname in _RAW_FILES.items()}

def _load_one(path: Path) -> GroupedSurvivalData:
    """
    Read one raw grouped life-table CSV and convert it to
    `GroupedSurvivalData`, flipping the file's `Censor` convention to
    `lifelines`' `event_observed` convention (see module docstring).

    Args:
        path: Path to a raw life-table CSV with columns
            "Age at Death [D]", "Frequency [#]", and
            "Censor (yes =1, no = 0)".

    Returns:
        The parsed `GroupedSurvivalData` for that file.
    """
    df = pd.read_csv(path)
    t = df["Age at Death [D]"].to_numpy(dtype=float)
    weight = df["Frequency [#]"].to_numpy(dtype=float)
    censor = df["Censor (yes =1, no = 0)"].to_numpy(dtype=float)
    event_observed = 1.0 - censor  # see module docstring for the convention flip
    return GroupedSurvivalData(t=t, event_observed=event_observed, weight=weight)


def kaplan_meier_curve(data: GroupedSurvivalData) -> KaplanMeierFitter:
    """
    Fit a real Kaplan-Meier curve from grouped raw data, using the
    `weights` argument (a well-established `lifelines` feature for exactly
    this kind of duplicated/grouped observation) rather than expanding
    `weight` worms into `weight` repeated rows -- same result, no need to
    materialize a large per-worm array.

    Args:
        data: Grouped survival data for one condition.

    Returns:
        A `KaplanMeierFitter` already fitted to `data`.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(data.t, event_observed=data.event_observed, weights=data.weight)
    return kmf

def weight_at_times(data: GroupedSurvivalData, t_unique: np.ndarray) -> np.ndarray:
    """
    Total worm count (`weight`) observed at each age in `t_unique` -- i.e. how
    many worms that age's empirical S(t) estimate is actually based on. Used
    to weight per-age residuals by their statistical reliability when
    curve-matching (see e.g. `_weighted_log_loss` in
    notebooks/auto_fit_parameters_mle.py): an age backed by 1 worm is a far
    noisier estimate of the true survival probability than one backed by
    40, and should count proportionally less in a fit.

    Args:
        data: Grouped survival data for one condition.
        t_unique: Ages to look up, expected to be a subset of `data.t`'s
            distinct values (e.g. `kmf.timeline`) -- ages not present in
            `data.t` will report a weight of 0.

    Returns:
        Array parallel to `t_unique` with the summed weight at each age.
    """
    return np.array([data.weight[data.t == t].sum() for t in t_unique])
