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
alarms when it passes h, chosen on the calibration flights at the shared 1 alarm per
flight-hour budget.

The plain CUSUM does not survive that calibration. A sum that never forgets grows with
flight length, and these flights run from 46 s (straight) to 265 s (curved), so one h
cannot fit both: the first run calibrated to 2 alarms in 1.59 h of calibration flight
and then fired 9 to 14 alarms per hour on the normal test flights. The statistic is not
stationary, so the budget does not transfer and the detection numbers it buys are not
comparable with Module 06's.

The fix is a decay: acc <- max(0, decay * acc + z - k). A decay below 1 gives the sum a
finite memory of about 1 / (1 - decay) windows, which makes it stationary while still
accumulating a slow drift across far more than one window. decay 1.0 is kept in the
grid so the non-stationary case stays visible in the output.

Every detection number in the tables is printed next to the false-alarm rate it cost,
because a detector that is over budget has not beaten anything.

Reads the scores train_v2.py wrote. Writes sequential_frontier.csv,
sequential_flights.csv, sequential_cost.csv and sequential_thresholds.csv in
data/processed/module07_v2

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
DECAY_GRID = (1.0, 0.97, 0.90)
H_GRID = tuple(np.geomspace(0.2, 20000.0, 130))
REFRACTORY_S = 10.0
RATES = (0.05, 0.10, 0.25, 0.50, 1.00)
MODULE06 = {0.05: 0.05, 0.10: 0.05, 0.25: 0.57, 0.50: 1.00, 1.00: 1.00}


def cusum(z, k, decay=1.0):
    """One-sided CUSUM with a finite memory: max(0, decay * previous + z - k)."""
    out = np.empty(len(z))
    acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, decay * acc + v - k)
        out[i] = acc
    return out


def standardise(score, pool):
    """Centre and scale by the calibration pool, which no model was fitted on."""
    med = float(np.median(score[pool]))
    mad = float(np.median(np.abs(score[pool] - med)))
    return (score - med) / max(1.4826 * mad, 1e-9)


def flight_cusum(z, meta, k, decay=1.0):
    """CUSUM of every flight, in window order, returned in the table's own order."""
    out = np.zeros(len(z))
    order = meta.sort_values(["flight_id", "window_start_s"]).index.to_numpy()
    fids = meta["flight_id"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, fids[1:] != fids[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)]):
        idx = order[a:b]
        out[idx] = cusum(z[idx], k, decay)
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
    used = None
    for h in H_GRID:
        used = sum(count_alarms(s, t, h)[0] for s, t in groups)
        if used <= allowed:
            return float(h), allowed, used
    return float(H_GRID[-1]), allowed, used


def evaluate(series, meta, h):
    """Per test flight: first alarm after the measured onset, delay, alarms per hour."""
    sub = meta.loc[(meta["role"] == "test").to_numpy()]
    rows, normal_alarms, normal_windows = [], 0, 0
    for fid, g in sub.groupby("flight_id", sort=False):
        g = g.sort_values("window_start_s")
        idx = g.index.to_numpy()
        n, _ = count_alarms(series[idx], g["window_end_s"].to_numpy(), h)
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
    print(f"k {list(K_GRID)}   decay {list(DECAY_GRID)}   ({len(idx)} models)\n")
    flights, thresholds, done = [], [], 0
    for _, r in idx.iterrows():
        if r["protocol"] == "unseen" and r["held_out"] != "linear_drift":
            continue  # the sweep is a linear drift: only that hold-out is honest on it
        z = standardise(np.load(sdir / f"{r['name']}.npy"), pool)
        for k in K_GRID:
            for decay in DECAY_GRID:
                series = flight_cusum(z, meta, k, decay)
                h, allowed, used = calibrate(series, meta, pool, budget)
                fl, rate = evaluate(series, meta, h)
                fl.insert(0, "seed", r["seed"])
                fl.insert(0, "decay", decay)
                fl.insert(0, "k", k)
                fl.insert(0, "detector", detector_of(r))
                flights.append(fl)
                thresholds.append({
                    "name": r["name"], "detector": detector_of(r), "seed": r["seed"],
                    "k": k, "decay": decay, "h": h, "allowed_alarms": allowed,
                    "alarms_used": used, "alarms_per_hour_test": rate,
                })  # fmt: skip
                done += 1
        print(f"  {r['name']:34s} {done:3d} fits done")
    fl = pd.concat(flights, ignore_index=True)
    fl.to_csv(out / "sequential_flights.csv", index=False)
    thr = pd.DataFrame(thresholds)
    thr.to_csv(out / "sequential_thresholds.csv", index=False)
    sweep = fl[fl["attack_set"] == "sweep"]
    keys = ["detector", "k", "decay", "drift_rate_mps"]
    frontier = by_seed(sweep[sweep["attacked"]], keys)
    frontier.to_csv(out / "sequential_frontier.csv", index=False)
    cost = cost_table(frontier, thr, budget)
    cost.to_csv(out / "sequential_cost.csv", index=False)
    report(fl, frontier, cost, budget, out)
    return fl, frontier, cost


def cost_table(frontier, thr, budget):
    """Detection per drift rate next to the false-alarm rate it cost."""
    det = frontier.pivot_table(
        index=["detector", "k", "decay"], columns="drift_rate_mps",
        values="detection_rate",
    )  # fmt: skip
    det = det[[c for c in RATES if c in det.columns]]
    det.columns = [f"det_{c:g}" for c in det.columns]
    rate = (
        thr.groupby(["detector", "k", "decay"])["alarms_per_hour_test"]
        .mean()
        .rename("alarms_per_hour")
    )
    table = det.join(rate).reset_index()
    table["within_budget"] = table["alarms_per_hour"] <= budget + 1e-9
    beats = [c for c in table.columns if c.startswith("det_")]
    if len(beats) < len(RATES):  # a partial sweep cannot be compared with Module 06
        table["beats_module06"] = False
        return table.sort_values(["detector", "k", "decay"])
    table["beats_module06"] = [
        all(
            row[f"det_{r:g}"] >= MODULE06[r]
            for r in RATES
            if f"det_{r:g}" in beats and np.isfinite(row[f"det_{r:g}"])
        )
        for _, row in table.iterrows()
    ]
    return table.sort_values(["detector", "k", "decay"])


def report(fl, frontier, cost, budget, out):
    pd.set_option("display.width", 240)
    print(
        "\nCOST AND DETECTION (det_* is the detection rate at that drift rate in m/s)"
    )
    print(cost.round(3).to_string(index=False))
    print(f"\nModule 06's chi-square test, at the same {budget:g} alarm/h budget:")
    print("  " + "  ".join(f"det_{r:g}={MODULE06[r]:.2f}" for r in RATES))
    ok = cost[cost["within_budget"]]
    print(f"\n{len(ok)} of {len(cost)} settings stayed inside the budget.")
    if len(ok):
        print("Of those, the strongest at 0.25 m/s and below:")
        best = ok.sort_values(["det_0.25", "det_0.1"], ascending=False).head(5)
        print(best.round(3).to_string(index=False))
    if ok["beats_module06"].any():
        print("\nSettings that match or beat Module 06 at every rate, within budget:")
        print(ok[ok["beats_module06"]].round(3).to_string(index=False))
    else:
        print("\nNo setting within budget matches Module 06 at every rate.")
    good = set(map(tuple, ok[["detector", "k", "decay"]].to_numpy()))
    sub = frontier[
        [tuple(x) in good for x in frontier[["detector", "k", "decay"]].to_numpy()]
    ]
    if len(sub):
        print("\nMEDIAN DELAY (s) from the measured onset, within-budget settings only")
        print(
            sub.pivot_table(
                index="drift_rate_mps",
                columns=["detector", "k", "decay"],
                values="median_delay_s",
            )  # fmt: skip
            .round(1)
            .to_string()
        )
    main = fl[fl["attack_set"] != "sweep"]
    keep = [tuple(x) in good for x in main[["detector", "k", "decay"]].to_numpy()]
    main = main[keep]
    if len(main):
        print(
            "\nMAIN ATTACK SET, within-budget settings (velocity mode was the weak case)"
        )
        print(
            by_seed(main[main["attacked"]], ["detector", "k", "decay", "velocity_mode"])
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
