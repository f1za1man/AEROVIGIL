r"""AEROVIGIL Module 10 step 2: calibrate, cross-check and score the fused detector.

Three detectors are scored on identical flights, at one shared false-alarm budget:

    kalman    Module 06's chi-square test on the horizontal NIS
    velocity  Module 09's velocity-consistency channel, standardised by median and MAD
    fused     an alarm when either fires; the budget is split between the two so the
              fused detector pays the same false-alarm price as each one alone

Thresholds come from the normal flights of the train and validation splits, and every
detection figure comes from the test split.

False alarms are measured two ways. The first, on the normal test flights only, is the
figure every earlier module reported; it rests on under an hour of flight, so it can
say little more than "one alarm in an hour". The second cross-calibrates over all 144
normal flights: the normals are cut into six folds by mission, and each fold is scored
with thresholds and a standardiser fitted on the other five. Every normal flight is
then scored by a detector that never saw it, and the false-alarm rate rests on all the
normal flight time in the dataset instead of a ninth of it. An exact Poisson upper bound
is reported with it, because a handful of alarms is still a handful.

Detection is per flight with a latch, the way Module 05 v2 scored it: a flight counts
as detected if any alarm falls inside its attack window, and the delay runs from the
measured onset, the first row of the ramp. Alarms before the window are the flight's
own false alarms and are counted as such.

Writes thresholds.json, flights.csv, threat_table.csv and false_alarms.csv in
data/processed/module10

Usage:
    python src\aerovigil\fusion\fuse.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from aerovigil.fusion.signals import OUT_DIR, ROOT, load

BUDGET_PER_HOUR = 1.0
REFRACTORY_S = 10.0
FOLDS = 6
DETECTORS = ("kalman", "velocity", "fused")
SEARCH_STEPS = 60


# ------------------------------------------------------------------ alarms --


def refractory(times, seconds=REFRACTORY_S):
    """Thin sorted exceedance times to one alarm per refractory period."""
    times = np.asarray(times, float)
    out, i, n = [], 0, len(times)
    while i < n:
        out.append(times[i])
        i = int(np.searchsorted(times, times[i] + seconds, side="left"))
    return np.asarray(out, float)


def kalman_hits(sig, thr):
    ok = sig["kf_valid"] & (sig["kf_nis"] > thr)
    return sig["kf_t"][ok]


def velocity_z(sig, mu, sd):
    z = np.abs((sig["vel"] - mu) / sd)
    return np.where(sig["vel_valid"], z, 0.0)


def velocity_hits(sig, thr, mu, sd):
    ok = sig["vel_valid"] & (velocity_z(sig, mu, sd) > thr)
    return sig["row_t"][ok]


def alarms(sig, det, th):
    """Alarm times of one detector on one flight, given its thresholds."""
    if det == "kalman":
        hits = kalman_hits(sig, th["kalman"])
    elif det == "velocity":
        hits = velocity_hits(sig, th["velocity"], th["mu"], th["sd"])
    else:
        hits = np.sort(np.concatenate([
            kalman_hits(sig, th["fused_kalman"]),
            velocity_hits(sig, th["fused_velocity"], th["mu"], th["sd"]),
        ]))  # fmt: skip
    return refractory(hits)


def hours_of(row, det):
    if det == "kalman":
        return float(row["kf_hours"])
    if det == "velocity":
        return float(row["vel_hours"])
    return float(max(row["kf_hours"], row["vel_hours"]))


# ------------------------------------------------------------- calibration --


def fit_standardiser(sigs):
    """Median and MAD of the velocity channel over clean flights."""
    pool = np.concatenate([s["vel"][s["vel_valid"]] for s in sigs])
    mu = float(np.median(pool))
    sd = float(max(1.4826 * np.median(np.abs(pool - mu)), 1e-9))
    return mu, sd


def lowest_threshold(count, lo, hi, allowed, steps=SEARCH_STEPS):
    """Smallest threshold in [lo, hi] whose alarm count fits, by bisection in log space.

    count(thr) only ever falls as thr rises, so bisection finds the boundary.
    """
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.001)
    if count(hi) > allowed:
        return float(hi)
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi))
        if count(mid) <= allowed:
            hi = mid
        else:
            lo = mid
    return float(hi)


def calibrate(sigs, rows, budget=BUDGET_PER_HOUR):
    """Thresholds for all three detectors on a set of clean flights."""
    mu, sd = fit_standardiser(sigs)

    def allowed_for(det, share=1.0):
        h = sum(hours_of(r, det) for r in rows)
        return max(1, int(np.floor(budget * share * h + 1e-9))), h

    k_max = max(float(np.max(s["kf_nis"][s["kf_valid"]], initial=1.0)) for s in sigs)
    v_max = max(float(np.max(velocity_z(s, mu, sd), initial=1.0)) for s in sigs)

    def k_count(thr):
        return sum(len(refractory(kalman_hits(s, thr))) for s in sigs)

    def v_count(thr):
        return sum(len(refractory(velocity_hits(s, thr, mu, sd))) for s in sigs)

    a_k, h_k = allowed_for("kalman")
    a_v, h_v = allowed_for("velocity")
    th = {"mu": mu, "sd": sd}
    th["kalman"] = lowest_threshold(k_count, 1.0, k_max * 1.01, a_k)
    th["velocity"] = lowest_threshold(v_count, 1.0, v_max * 1.01, a_v)
    a_f, h_f = allowed_for("fused")
    # Split the fused allowance between the two channels as evenly as possible. A
    # share of zero is allowed and switches that channel off, so a one-alarm budget
    # is honoured instead of being exceeded by a floor of one alarm per channel.
    share_v = (a_f + 1) // 2
    share_k = a_f - share_v
    while True:
        th["fused_kalman"] = (
            lowest_threshold(k_count, 1.0, k_max * 1.01, share_k)
            if share_k > 0
            else k_max * 1.01
        )
        th["fused_velocity"] = (
            lowest_threshold(v_count, 1.0, v_max * 1.01, share_v)
            if share_v > 0
            else v_max * 1.01
        )
        used = sum(len(alarms(s, "fused", th)) for s in sigs)
        if used <= a_f or (share_k == 0 and share_v == 0):
            break
        if share_k >= share_v:
            share_k -= 1
        else:
            share_v -= 1
    th["fused_shares"] = {"kalman": share_k, "velocity": share_v}
    th["hours"] = {"kalman": h_k, "velocity": h_v, "fused": h_f}
    th["allowed"] = {"kalman": a_k, "velocity": a_v, "fused": a_f}
    th["used"] = {
        det: int(sum(len(alarms(s, det, th)) for s in sigs)) for det in DETECTORS
    }
    return th


# ---------------------------------------------------------------- scoring --


def score_flight(sig, row, th):
    """Latched verdict of every detector on one flight."""
    out = {}
    for det in DETECTORS:
        a = alarms(sig, det, th)
        if not row["attacked"]:
            out[f"{det}_alarms"] = int(len(a))
            continue
        start, end, onset = sig["start_s"], sig["end_s"], sig["onset_s"]
        inside = a[(a >= start) & (a <= end)]
        hit = bool(len(inside))
        out[f"{det}_detected"] = hit
        out[f"{det}_delay_s"] = float(inside[0] - onset) if hit else np.nan
        out[f"{det}_displacement_m"] = (
            float(np.interp(inside[0], sig["row_t"], sig["offset_m"]))
            if hit
            else np.nan
        )
        out[f"{det}_alarms_before"] = int((a < start).sum())
    return out


def poisson_upper(k, hours, level=0.95):
    """Exact one-sided upper bound on a Poisson rate from k events in `hours`."""
    if hours <= 0:
        return np.nan
    return float(chi2.ppf(level, 2 * (k + 1)) / 2.0 / hours)


def folds_of(rows):
    """Deterministic folds, stratified by mission so every fold sees every mission."""
    order = rows.sort_values(["mission_type", "flight_id"]).reset_index(drop=True)
    order["fold"] = order.groupby("mission_type").cumcount() % FOLDS
    return dict(zip(order["flight_id"], order["fold"]))


def cross_validated_false_alarms(normal, sigs, budget):
    """Score every normal flight with thresholds fitted on the other folds."""
    fold = folds_of(normal)
    tally = {det: [0, 0.0] for det in DETECTORS}
    for f in range(FOLDS):
        held = normal[normal["flight_id"].map(fold) == f]
        fit = normal[normal["flight_id"].map(fold) != f]
        if held.empty:
            continue
        if fit.empty:
            raise SystemExit(
                f"Cross-validation needs normal flights outside fold {f}; "
                f"only {len(normal)} normal flights were found."
            )
        th = calibrate(
            [sigs[x] for x in fit["flight_id"]], fit.to_dict("records"), budget
        )
        for _, r in held.iterrows():
            for det in DETECTORS:
                tally[det][0] += len(alarms(sigs[r["flight_id"]], det, th))
                tally[det][1] += hours_of(r, det)
    rows = []
    for det, (k, h) in tally.items():
        rows.append({
            "detector": det, "scope": "cross-validated, all normal flights",
            "flights": len(normal), "alarms": k, "hours": h,
            "alarms_per_hour": k / h if h else np.nan,
            "upper_95_per_hour": poisson_upper(k, h),
        })  # fmt: skip
    return rows


def threat_groups(flights):
    """(label, mask) pairs: one row of the threat table each."""
    att = flights[flights["attacked"]]
    groups = [("Dataset's own attacks (simulator)", att["set"] == "orig")]
    main = att["set"] == "main"
    for mode, name in (("position_only", "position-only"),
                       ("consistent", "velocity-consistent")):  # fmt: skip
        m = main & (att["velocity_mode"] == mode)
        groups.append((f"Synthetic {name}, all families", m))
        for fam in sorted(att.loc[m, "attack_type"].unique()):
            groups.append((f"   {name} {fam}", m & (att["attack_type"] == fam)))
    sweep = att["set"] == "sweep"
    for rate in sorted(att.loc[sweep, "drift_rate_mps"].dropna().unique()):
        groups.append((
            f"Consistent drift {rate:g} m/s",
            sweep & (att["drift_rate_mps"] == rate),
        ))  # fmt: skip
    return att, [(name, m) for name, m in groups if m.any()]


def threat_table(flights):
    att, groups = threat_groups(flights)
    rows = []
    for name, m in groups:
        sub = att[m]
        row = {"attacker": name, "flights": len(sub)}
        for det in DETECTORS:
            row[f"{det}_detected"] = float(sub[f"{det}_detected"].mean())
            delays = sub[f"{det}_delay_s"].dropna()  # missed flights have no delay
            row[f"{det}_delay_s"] = float(delays.median()) if len(delays) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- run --


def run(output_dir=None, budget=BUDGET_PER_HOUR):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    index = pd.read_csv(out / "signals_index.csv")
    index["attacked"] = index["attacked"].astype(bool)
    print("=" * 78)
    print("AEROVIGIL - MODULE 10 FUSED DETECTOR (Kalman NIS + velocity consistency)")
    print("=" * 78)
    started = time.time()
    sigs = {
        r["flight_id"]: load(out / "signals" / r["file"]) for _, r in index.iterrows()
    }
    orig = index[index["set"] == "orig"]
    normal = orig[~orig["attacked"]]
    fit = normal[normal["split"].isin(["train", "validation"])]
    print(
        f"Flights {len(index)}   normal {len(normal)}   calibration normals "
        f"{len(fit)} (train + validation)   budget {budget:g} alarm/h   "
        f"refractory {REFRACTORY_S:g} s"
    )
    th = calibrate([sigs[x] for x in fit["flight_id"]], fit.to_dict("records"), budget)
    for det in DETECTORS:
        print(
            f"  {det:8s} calibrated on {th['hours'][det]:.2f} h: "
            f"at most {th['allowed'][det]} alarms, used {th['used'][det]}"
        )
    print(
        f"  thresholds: kalman NIS > {th['kalman']:.2f}, velocity z > "
        f"{th['velocity']:.2f}; fused uses {th['fused_kalman']:.2f} and "
        f"{th['fused_velocity']:.2f}"
    )
    (out / "thresholds.json").write_text(json.dumps(th, indent=2))
    test = index[index["split"] == "test"]
    scored = [
        {**r.to_dict(), **score_flight(sigs[r["flight_id"]], r, th)}
        for _, r in test.iterrows()
    ]
    flights = pd.DataFrame(scored)
    flights.to_csv(out / "flights.csv", index=False)
    table = threat_table(flights)
    table.to_csv(out / "threat_table.csv", index=False)
    fa_rows = []
    tn = flights[(flights["set"] == "orig") & (~flights["attacked"])]
    for det in DETECTORS:
        k = int(tn[f"{det}_alarms"].sum())
        h = float(sum(hours_of(r, det) for _, r in tn.iterrows()))
        fa_rows.append({
            "detector": det, "scope": "normal test flights only",
            "flights": len(tn), "alarms": k, "hours": h,
            "alarms_per_hour": k / h if h else np.nan,
            "upper_95_per_hour": poisson_upper(k, h),
        })  # fmt: skip
    fa_rows += cross_validated_false_alarms(normal, sigs, budget)
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(out / "false_alarms.csv", index=False)
    report(table, fa, out, time.time() - started)
    return table, fa, th


def report(table, fa, out, elapsed):
    pd.set_option("display.width", 240)
    show = table.copy()
    for det in DETECTORS:
        show[det] = [
            f"{d:4.0%} {s:5.1f}s" if np.isfinite(s) else f"{d:4.0%}    - "
            for d, s in zip(show[f"{det}_detected"], show[f"{det}_delay_s"])
        ]
    print("\nTHREAT TABLE (test flights): share detected and median delay from onset")
    print(show[["attacker", "flights", *DETECTORS]].to_string(index=False))
    print("\nFALSE ALARMS")
    print(fa.round(3).to_string(index=False))
    print(
        "\nNot in this table: Module 08's adaptive attacker, which eases in over about"
        "\n20 s and keeps position and velocity consistent. Against it the Kalman test"
        "\ncaught 5-14% and the consistency detector 0%, and Module 09 measured why on"
        "\nreal flights: no sensor in this dataset responds to it."
    )
    print(f"\nElapsed {elapsed:.0f} s   saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--budget", type=float, default=BUDGET_PER_HOUR)
    a = ap.parse_args()
    run(a.output_dir, a.budget)


if __name__ == "__main__":
    main()
