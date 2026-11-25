r"""AEROVIGIL Module 07 v2 step 4: accumulate the LSTM score across windows.

Module 07 v2's single-window latch loses to the Module 06 chi-square test on every
drift rate below 2 m/s: at 0.25 m/s the chi-square test catches 57% of flights and the
supervised LSTM 14%, and the autoencoder sits at chance until 2 m/s. The reason is not
capacity. A 0.25 m/s drift moves 1.25 m across a 5 s window, which is inside the GPS
noise, so no window carries the evidence. The Kalman filter wins because its state
carries evidence from the whole flight, while the LSTM looks at 5 s and forgets.

So this file gives the LSTM the same memory, with the same decision rule Module 06
used: a one-sided CUSUM over the window scores of a flight. Each score is standardised
by the calibration pool's median and MAD, a slack k is subtracted, and the running sum
alarms when it passes h. k and h are chosen on the calibration flights at the shared
1 alarm per flight-hour budget, so the comparison against Module 06 and against the
single-window latch is at equal cost.

Reads the scores train_v2.py wrote. Writes sequential_frontier.csv,
sequential_flights.csv and sequential_thresholds.csv in data/processed/module07_v2

Usage:
    python src\aerovigil\lstm\sequential_v2.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.lstm.build_sequences_v2 import OUT_DIR, ROOT
from aerovigil.lstm.evaluate_v2 import (
    BUDGET_PER_HOUR,
    by_seed,
    calibration_pool,
    detector_of,
    hours,
)
from aerovigil.lstm.train_v2 import load

K_GRID = (0.25, 0.5, 1.0)
H_GRID = tuple(np.geomspace(0.5, 20000.0, 120))
REFRACTORY_S = 10.0


def cusum(z, k):
    """One-sided CUSUM of a standardised score: max(0, previous + z - k)."""
    out = np.empty(len(z))
    acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, acc + v - k)
        out[i] = acc
    return out


def standardise(score, pool):
    """Centre and scale by the calibration pool, which no model was fitted on."""
    med = float(np.median(score[pool]))
    mad = float(np.median(np.abs(score[pool] - med)))
    return (score - med) / max(1.4826 * mad, 1e-9)


def flight_cusum(z, meta, k):
    """CUSUM of every flight, in window order, returned in the table's own order."""
    out = np.zeros(len(z))
    order = meta.sort_values(["flight_id", "window_start_s"]).index.to_numpy()
    fids = meta["flight_id"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, fids[1:] != fids[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        idx = order[a:b]
        out[idx] = cusum(z[idx], k)
    return out


def count_alarms(series, t_end, h):
    """Alarms above h, counted once per refractory period, and the first alarm index."""
    hit = np.flatnonzero(series > h)
    if not len(hit):
        return 0, None
    n, skip, first = 0, -np.inf, None
    for i in hit:
        if t_end[i] >= skip:
            n += 1
            skip = t_end[i] + REFRACTORY_S
            if first is None:
                first = int(i)
    return n, first


def calibrate(series, meta, pool, budget):
    """Smallest h whose alarm count on the calibration flights fits the budget."""
    allowed = max(1, round(budget * hours(int(pool.sum()))))
    sub = meta.loc[pool]
    groups = [
        (series[g.index.to_numpy()], g["window_end_s"].to_numpy())
        for _, g in sub.groupby("flight_id", sort=False)
    ]
    for h in H_GRID:
        used = sum(count_alarms(s, t, h)[0] for s, t in groups)
        if used <= allowed:
            return float(h), allowed, used
    return float(H_GRID[-1]), allowed, used


def evaluate(series, meta, h):
    """Per test flight: first alarm after the measured onset, delay, alarms per hour."""
    test = (meta["role"] == "test").to_numpy()
    sub = meta.loc[test]
    rows, normal_alarms, normal_windows = [], 0, 0
    for fid, g in sub.groupby("flight_id", sort=False):
        g = g.sort_values("window_start_s")
        idx = g.index.to_numpy()
        n, first = count_alarms(series[idx], g["window_end_s"].to_numpy(), h)
        attacked = g["attack_type"].iloc[0] != "none"
        since = g["seconds_since_onset_end"].to_numpy()
        after = np.flatnonzero((series[idx] > h) & (since >= 0)) if attacked else []
        if not attacked:
            normal_alarms += n
            normal_windows += len(g)
        rows.append({
            "flight_id": fid, "attack_set": g["attack_set"].iloc[0],
            "mission_type": g["mission_type"].iloc[0],
            "attack_type": g["attack_type"].iloc[0],
            "velocity_mode": g["velocity_mode"].iloc[0],
            "drift_rate_mps": g["drift_rate_mps"].iloc[0],
            "attacked": attacked, "alarms": n,
            "detected": bool(len(after)) if attacked else False,
            "delay_s": float(since[after[0]]) if len(after) else np.nan,
            "offset_at_alarm_m": (
                float(g["offset_end_m"].to_numpy()[after[0]]) if len(after) else np.nan
            ),
        })  # fmt: skip
    rate = normal_alarms / hours(normal_windows) if normal_windows else np.nan
    return pd.DataFrame(rows), float(rate)


def run(output_dir=None, budget=BUDGET_PER_HOUR):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    sdir = out / "scores"
    idx = pd.read_csv(sdir / "index.csv")
    _, meta = load(out)
    meta = meta.reset_index(drop=True)  # index labels must equal score positions
    pool = calibration_pool(meta)
    print("=" * 70)
    print("AEROVIGIL - MODULE 07 v2 SEQUENTIAL TEST (CUSUM over the window scores)")
    print("=" * 70)
    print(
        f"Models {len(idx)}   calibration windows {int(pool.sum()):,} "
        f"({hours(int(pool.sum())):.2f} h)   budget {budget:g} alarm/h"
    )
    flights, thresholds = [], []
    for _, r in idx.iterrows():
        if r["protocol"] == "unseen" and r["held_out"] != "linear_drift":
            continue  # the sweep is a linear drift: only that hold-out is honest on it
        z = standardise(np.load(sdir / f"{r['name']}.npy"), pool)
        for k in K_GRID:
            series = flight_cusum(z, meta, k)
            h, allowed, used = calibrate(series, meta, pool, budget)
            fl, rate = evaluate(series, meta, h)
            fl.insert(0, "seed", r["seed"])
            fl.insert(0, "k", k)
            fl.insert(0, "detector", detector_of(r))
            flights.append(fl)
            thresholds.append({
                "name": r["name"], "detector": detector_of(r), "seed": r["seed"],
                "k": k, "h": h, "allowed_alarms": allowed, "alarms_used": used,
                "alarms_per_hour_test": rate,
            })  # fmt: skip
            sw = fl[(fl["attack_set"] == "sweep") & (fl["drift_rate_mps"] <= 0.25)]
            print(
                f"  {r['name']:34s} k={k:<5g} h={h:7.2f}  "
                f"{rate:.2f} alarms/h on test normals   "
                f"detected {sw['detected'].mean() if len(sw) else np.nan:.2f} "
                f"at 0.25 m/s and below"
            )
    fl = pd.concat(flights, ignore_index=True)
    fl.to_csv(out / "sequential_flights.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "sequential_thresholds.csv", index=False)
    sweep = fl[fl["attack_set"] == "sweep"]
    frontier = by_seed(sweep[sweep["attacked"]], ["detector", "k", "drift_rate_mps"])
    frontier.to_csv(out / "sequential_frontier.csv", index=False)
    report(fl, frontier, out)
    return fl, frontier


def report(fl, frontier, out):
    pd.set_option("display.width", 220)
    print("\nSEQUENTIAL STEALTH FRONTIER: detection rate per drift rate")
    print(
        frontier.pivot_table(
            index="drift_rate_mps", columns=["detector", "k"], values="detection_rate"
        )
        .round(3)
        .to_string()
    )
    print("\nMEDIAN DELAY (s) from the measured onset")
    print(
        frontier.pivot_table(
            index="drift_rate_mps", columns=["detector", "k"], values="median_delay_s"
        )
        .round(1)
        .to_string()
    )
    print("\nFor comparison, the single-window latch and Module 06's chi-square test:")
    ref = pd.DataFrame(
        {
            "module06_chi2": [0.05, 0.05, 0.57, 1.00, 1.00, 1.00, 1.00],
            "lstm_ae_window": [0.048, 0.048, 0.048, 0.048, 0.111, 1.000, 1.000],
            "lstm_sup_window": [0.016, 0.016, 0.143, 0.603, 0.873, 0.984, 1.000],
        },
        index=[0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],
    )
    ref.index.name = "drift_rate_mps"
    print(ref.to_string())
    main = fl[fl["attack_set"] != "sweep"]
    print("\nDETECTION ON THE MAIN ATTACK SET (velocity mode is where it was weak)")
    print(
        by_seed(main[main["attacked"]], ["detector", "k", "velocity_mode"])
        .round(3)
        .to_string(index=False)
    )
    print(f"\nSaved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET_PER_HOUR)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.output_dir, a.budget)


if __name__ == "__main__":
    main()
