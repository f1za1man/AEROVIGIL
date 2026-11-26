r"""AEROVIGIL Module 10 step 3: Module 08's adaptive attacker against the fused detector.

Module 08 searched for the drift that best hides from Module 06's chi-square test, and
Module 09 replayed it against the consistency detector. Neither number was taken at
Module 10's thresholds, so this replays Module 08's winning parameters (one per mean
drift speed, plus the unconstrained winner) on the 21 normal test flights and scores
every flight with Module 10's three detectors exactly as fuse.py scores the threat
table: same thresholds, same refractory, same latch inside the attack window, same
delay from the measured onset.

The attacker is not re-searched against the fused detector. Its parameters were tuned
against the chi-square test alone, so this measures how far the best hiding strategy
found so far carries over to the fused detector, not the best possible attack on it.
Module 09 measured on real flights why a further search would not find much to
exploit: no sensor in this dataset responds to a smooth, velocity-consistent drift.

Needs fuse.py (thresholds.json) and Module 08 (adaptive_best.json).

Writes data/processed/module10/adaptive_flights.csv and adaptive_summary.csv

Usage:
    python src\aerovigil\fusion\adaptive.py
    python src\aerovigil\fusion\adaptive.py --limit 3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.adaptive.oracle import KalmanOracle
from aerovigil.adaptive.profile import profile_fn
from aerovigil.adaptive.search import normal_flights
from aerovigil.fusion.fuse import DETECTORS, score_flight
from aerovigil.fusion.signals import OUT_DIR, ROOT, Context, flight_signals

BEST = ROOT / "data" / "processed" / "module08" / "adaptive_best.json"
FRONTIER = ROOT / "data" / "processed" / "module08" / "adaptive_frontier.csv"


def label_of(key):
    """'mean_rate=0.25' -> 0.25; 'free' -> NaN."""
    return float(key.split("=")[1]) if key.startswith("mean_rate=") else np.nan


def replay(g, params, oracle, ctx, th):
    """Inject one adaptive attack into one clean flight and score all detectors."""
    attacked = oracle.attack(g, profile_fn(params))
    sig = flight_signals(attacked, ctx)
    out = score_flight(sig, {"attacked": True}, th)
    t, off = sig["row_t"], sig["offset_m"]
    on = (t >= sig["start_s"]) & (t <= sig["end_s"])
    final = float(np.nanmax(off[on])) if on.any() else 0.0
    window = max(sig["end_s"] - sig["start_s"], 1e-9)
    out["final_offset_m"] = final
    out["achieved_mean_mps"] = final / window
    return out


def summarise(flights):
    rows = []
    for key, sub in flights.groupby("search", sort=False):
        row = {"search": key, "mean_rate_mps": label_of(key), "flights": len(sub),
               "achieved_mean_mps": float(sub["achieved_mean_mps"].median()),
               "median_final_offset_m": float(sub["final_offset_m"].median())}  # fmt: skip
        for det in DETECTORS:
            hit = sub[f"{det}_detected"].astype(bool)
            row[f"{det}_detected"] = float(hit.mean())
            d = sub.loc[hit, f"{det}_delay_s"]
            row[f"{det}_delay_s"] = float(d.median()) if len(d) else np.nan
            # what the attacker got away with: the offset at the first alarm when
            # caught, the whole drift when not
            moved = np.where(hit, sub[f"{det}_displacement_m"], sub["final_offset_m"])
            row[f"{det}_median_displacement_m"] = float(np.median(moved))
        rows.append(row)
    return pd.DataFrame(rows)


def run(limit=None, output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    th_path = out / "thresholds.json"
    if not th_path.is_file():
        raise SystemExit(f"Missing {th_path}. Run fuse.py first.")
    if not BEST.is_file():
        raise SystemExit(f"Missing {BEST}. Run Module 08 (adaptive/search.py) first.")
    th = json.loads(th_path.read_text())
    best = json.loads(BEST.read_text())
    oracle, ctx = KalmanOracle(), Context()
    exam = normal_flights("test", limit)
    print("=" * 78)
    print("AEROVIGIL - MODULE 10 ADAPTIVE ATTACKER vs FUSED DETECTOR")
    print("=" * 78)
    print(
        f"Module 08 winners {len(best)}   normal test flights {len(exam)}   "
        f"thresholds from {th_path.name}"
    )
    started, rows = time.time(), []
    for key, params in best.items():
        for fid, mission, g in exam:
            rows.append({"search": key, "flight_id": fid, "mission_type": mission,
                         **replay(g, params, oracle, ctx, th)})  # fmt: skip
        print(f"  {key:16s} done   [{time.time() - started:.0f} s]")
    flights = pd.DataFrame(rows)
    flights.to_csv(out / "adaptive_flights.csv", index=False)
    summary = summarise(flights)
    if FRONTIER.is_file():
        m08 = pd.read_csv(FRONTIER)
        m08 = m08.assign(
            search=np.where(
                m08["search"] == "free",
                "free",
                "mean_rate=" + m08["mean_rate_mps"].map("{:g}".format),
            )
        )
        summary = summary.merge(
            m08[["search", "detection_rate"]].rename(
                columns={"detection_rate": "module08_kalman_detected"}
            ),
            on="search",
            how="left",
        )
    summary.to_csv(out / "adaptive_summary.csv", index=False)
    report(summary, time.time() - started, out)
    return summary


def report(summary, elapsed, out):
    show = pd.DataFrame({"attack": summary["search"],
                         "speed": summary["achieved_mean_mps"].round(2),
                         "drift_m": summary["median_final_offset_m"].round(0)})  # fmt: skip
    for det in DETECTORS:
        show[det] = [
            f"{d:4.0%} {m:6.0f}m"
            for d, m in zip(
                summary[f"{det}_detected"], summary[f"{det}_median_displacement_m"]
            )
        ]
    if "module08_kalman_detected" in summary:
        show["m08_kalman"] = summary["module08_kalman_detected"].map("{:.0%}".format)
    print("\nShare of test flights detected, and median metres the drone was moved")
    print("before the first alarm (the whole drift when never caught)")
    print(show.to_string(index=False))
    print(
        "\nm08_kalman is Module 08's own figure, at Module 06's threshold and alarm"
        "\nrules; the kalman column here uses Module 10's threshold."
    )
    print(f"\nElapsed {elapsed:.0f} s   saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="test flights to use")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.limit, a.output_dir)


if __name__ == "__main__":
    main()
