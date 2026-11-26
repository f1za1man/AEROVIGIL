"""One set of metric functions for every detector (plan M9).

Every detector in the project ends up as the same two kinds of table, so they are all
scored by the functions here:

    windows   one row per scored window: true label and score. Gives accuracy,
              precision, recall, F1 and ROC-AUC at a threshold, and the confusion
              matrix.
    flights   one row per test flight: attacked or not, detected or not, delay from
              the measured onset, displacement at the first alarm, the whole drift
              when never caught, and the alarm count. Gives the detection rate, the
              delay, the undetected displacement and false alarms per flight-hour.

False alarms always come with their count, their hours (from timestamps) and an exact
Poisson 95% upper bound, because a handful of alarms in under an hour says little on
its own. Spread is shown two ways: mean and standard deviation over seeds where a
detector has randomness, and a bootstrap 95% interval over flights where it has none.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix, roc_auc_score

REFRACTORY_S = 10.0
BUDGET_PER_HOUR = 1.0


def poisson_upper(k, hours, level=0.95):
    """Exact one-sided upper bound on a Poisson rate from k events in `hours`."""
    if hours <= 0:
        return np.nan
    return float(chi2.ppf(level, 2 * (int(k) + 1)) / 2.0 / hours)


def refractory_count(times, seconds=REFRACTORY_S):
    """Alarms left after merging any that fall within `seconds` of the last kept."""
    n, last = 0, -np.inf
    for t in np.sort(np.asarray(times, float)):
        if t >= last + seconds:
            n, last = n + 1, t
    return n


def roc_auc(y, score):
    y = np.asarray(y)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def window_metrics(y, score, threshold):
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    A window is called an attack when its score is strictly above the threshold, the
    same rule every detector's alarm uses.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "windows": int(len(y)), "positives": int(y.sum()),
        "accuracy": float((tp + tn) / len(y)) if len(y) else np.nan,
        "precision": float(prec), "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "roc_auc": roc_auc(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }  # fmt: skip


def false_alarms(alarms, hours):
    """Count, hours, rate and the 95% upper bound, as one record."""
    k = int(np.sum(alarms))
    h = float(np.sum(hours))
    return {
        "fa_alarms": k, "fa_hours": h,
        "fa_per_hour": k / h if h > 0 else np.nan,
        "fa_upper_95": poisson_upper(k, h),
    }  # fmt: skip


def flight_metrics(flights):
    """Detection on attacked flights and false alarms on normal ones.

    Expects columns attacked, detected, delay_s, alarms and hours; displacement_m
    (offset at the first alarm) and final_offset_m (the whole drift) are optional.
    Undetected displacement is what the attacker got away with: the offset at the
    first alarm when caught, the whole drift when not.
    """
    att = flights[flights["attacked"].astype(bool)]
    nor = flights[~flights["attacked"].astype(bool)]
    hit = att["detected"].astype(bool)
    out = {
        "attacked_flights": int(len(att)),
        "detected": float(hit.mean()) if len(att) else np.nan,
        "median_delay_s": float(att.loc[hit, "delay_s"].median())
        if hit.any() else np.nan,
        "normal_flights": int(len(nor)),
        **false_alarms(nor["alarms"], nor["hours"]),
    }  # fmt: skip
    if {"displacement_m", "final_offset_m"} <= set(att.columns) and len(att):
        moved = np.where(hit, att["displacement_m"], att["final_offset_m"])
        out["median_undetected_displacement_m"] = float(np.nanmedian(moved))
    return out


def over_seeds(rows, keys, value_cols):
    """Mean and standard deviation over seeds of per-seed metric rows."""
    frame = pd.DataFrame(rows)
    g = frame.groupby(keys)
    out = g[value_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["seeds"] = g["seed"].nunique() if "seed" in frame else 1
    return out.reset_index()


def bootstrap_ci(values, stat=np.mean, n=2000, seed=0, level=0.95):
    """Percentile bootstrap interval of `stat` over the given per-flight values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.choice(v, len(v))) for _ in range(n)])
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
