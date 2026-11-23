"""Window scores -> alarms: false-alarm-budget threshold, refractory rule and flight metrics.

Same rules as Module 06: one shared budget, alarms counted once per refractory period,
detection = an alarm inside the attack span, displacement = spoofed offset when it fires.
A window alarms at its END time, so the alarm is causal.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def alarm_indices(score, t_end, thr, refractory_s):
    """Indices of alarm windows: score > thr, then silence for refractory_s."""
    out, skip_until = [], -np.inf
    for i in np.flatnonzero(score > thr):
        if t_end[i] >= skip_until:
            out.append(int(i))
            skip_until = t_end[i] + refractory_s
    return np.asarray(out, dtype=int)


def calibrate_threshold(flights, budget_per_hour, hop_s, refractory_s):
    """Smallest threshold whose alarm count on NORMAL flights stays inside the budget.

    flights: list of (score, t_end) arrays, one per normal flight. Each window stands for
    hop_s seconds of monitored time. Returns (threshold, hours, allowed_alarms, alarms_used).
    """
    hours = sum(len(s) for s, _ in flights) * hop_s / 3600.0
    allowed = budget_per_hour * hours
    all_scores = np.unique(np.concatenate([s for s, _ in flights]))
    candidates = np.concatenate([[all_scores[0] - 1e-9], all_scores])
    best = (float(all_scores[-1]), 0)
    for thr in candidates:  # ascending: first one that fits wins
        used = sum(len(alarm_indices(s, t, thr, refractory_s)) for s, t in flights)
        if used <= allowed:
            return float(thr), hours, allowed, used
    return best[0], hours, allowed, best[1]


def flight_metrics(score, t_end, offset_end, thr, refractory_s, start_s, end_s, hop_s):
    """Alarm time, delay, displacement and false alarms of one flight (window arrays)."""
    idx = alarm_indices(score, t_end, thr, refractory_s)
    out = {"n_alarms": int(len(idx))}
    if not np.isfinite(start_s):
        out.update(
            attacked=False,
            detected=False,
            first_alarm_s=np.nan,
            delay_s=np.nan,
            displacement_m=np.nan,
            false_alarms=int(len(idx)),
            normal_hours=len(score) * hop_s / 3600.0,
        )
        return out
    times = t_end[idx]
    inside = idx[(times >= start_s) & (times <= end_s)]
    detected = len(inside) > 0
    on = (t_end >= start_s) & (t_end <= end_s)
    if detected:
        disp = float(offset_end[inside[0]])
    else:
        disp = (
            float(np.nanmax(offset_end[on]))
            if on.any() and np.isfinite(offset_end[on]).any()
            else np.nan
        )
    pre = t_end < start_s
    out.update(
        attacked=True,
        detected=bool(detected),
        first_alarm_s=float(t_end[inside[0]]) if detected else np.nan,
        delay_s=float(t_end[inside[0]] - start_s) if detected else np.nan,
        displacement_m=disp,
        false_alarms=int((times < start_s).sum()),
        normal_hours=float(pre.sum()) * hop_s / 3600.0,
    )
    return out


def window_metrics(score, pos, normal, thr_fpr, fpr=0.01):
    """ROC-AUC, TPR at `fpr`, and precision / recall / F1 at the validation 1%-FPR threshold."""
    s_pos, s_neg = score[pos], score[normal]
    if len(s_pos) == 0 or len(s_neg) == 0:
        return {}
    y = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    s = np.concatenate([s_pos, s_neg])
    auc = float(roc_auc_score(y, s))
    tpr = float((s_pos > np.quantile(s_neg, 1.0 - fpr)).mean())
    tp, fp = float((s_pos > thr_fpr).sum()), float((s_neg > thr_fpr).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(s_pos)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "roc_auc": auc,
        f"tpr_at_{int(fpr * 100)}pct_fpr": tpr,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_pos": int(len(s_pos)),
        "n_neg": int(len(s_neg)),
    }
