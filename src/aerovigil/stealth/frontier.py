r"""Plan M8 (Step D): the fixed-drift Stealth Frontier, and a self-test of the scorers.

Every test flight of Module 04 (the real attacks, the synthetic main set and the
7-rate drift sweep, plus the clean test flights) is scored by every detector of
detectors.py: five seeds for the random forest and both LSTMs, one run for the
deterministic physics detectors.

Self-test first. The same flights were scored in Step C by each detector's own code;
here they go through the one shared scorer. The verdicts must agree (detected or not)
on at least 98% of flights for every detector, or the Step D numbers are not trusted
and the run says so.

The Stealth Frontier (our own metric). For an attacker drifting at mean rate r,
D(r) is the distance the drone has been moved when the first alarm fires, or the whole
drift if no alarm ever fires, at the fixed budget of 1 false alarm per flight-hour.
A lower curve is a stronger detector. Here r is the Module 04 sweep's fixed rate,
velocity-consistent and eased in with the warp onset; adaptive.py adds the attacker
who reshapes the drift.

Writes data/processed/stealth/fixed_flights.csv, fixed_selftest.csv and
reports/tables/D_frontier_fixed.csv

Usage:
    python src\aerovigil\stealth\frontier.py                 (all, ~30 min)
    python src\aerovigil\stealth\frontier.py --part 1/3      (one third; merge with --merge)
    python src\aerovigil\stealth\frontier.py --merge
    python src\aerovigil\stealth\frontier.py --limit 5
"""

import argparse
import time

import numpy as np
import pandas as pd

from aerovigil.baselines.build_window_tables_v2 import OUT_DIR as M05_DIR
from aerovigil.evaluation import lstm_protocols as lp
from aerovigil.evaluation import physics_protocols as pp
from aerovigil.evaluation.metrics import bootstrap_ci, poisson_upper
from aerovigil.stealth import detectors as D
from aerovigil.windowing.make_windows_v2 import SECONDS

TABLE = D.ROOT / "reports" / "tables" / "D_frontier_fixed.csv"
AGREEMENT = 0.98
KEEP = ("detected", "delay_s", "displacement_m", "final_offset_m", "n_alarms",
        "alarms_before_onset", "achieved_mean_mps", "after_end_s")  # fmt: skip


def test_flights(limit=None):
    idx = pp.universe()
    test = idx[idx["split"] == "test"].copy()
    if limit:
        test = test.groupby(["set", "attacked"], group_keys=False).head(limit)
    return test.reset_index(drop=True)


def score_all(flights, dets, started):
    rows = []
    first = next(iter(dets.values()))
    for n, (_, r) in enumerate(flights.iterrows(), start=1):
        fl = first.flight(pd.read_parquet(r["base_path"]))
        for (name, seed), det in dets.items():
            v = det.verdict(fl)
            rows.append({"flight_id": r["flight_id"], "set": r["set"],
                         "mission_type": r["mission_type"],
                         "attack_type": r["attack_type"],
                         "velocity_mode": r["velocity_mode"],
                         "drift_rate_mps": r["drift_rate_mps"],
                         "attacked": bool(r["attacked"]),
                         "hours": hours_of(r, name),
                         "detector": name, "seed": seed,
                         **{k: v[k] for k in KEEP}})  # fmt: skip
        if n % 20 == 0 or n == len(flights):
            print(f"  {n}/{len(flights)} flights   [{time.time() - started:.0f} s]",
                  flush=True)  # fmt: skip
    return pd.DataFrame(rows)


def hours_of(row, name):
    """Monitored hours, the basis Step C used for each detector."""
    if name in ("kalman", "cusum"):
        return float(row["kf_hours"])
    if name == "fused":
        return float(max(row["kf_hours"], row["vel_hours"]))
    return float(row["vel_hours"])


# -------------------------------------------------------------- self-test --


def reference():
    """Step C's own per-flight verdicts on the same test flights (seen protocol)."""
    p = pd.read_csv(pp.OUT / "physics_flights.csv")
    p = p[p["protocol"] == "seen"][["flight_id", "detector", "detected"]]
    p["seed"] = 0
    r = pd.read_csv(
        M05_DIR / "all_models_flight_full.csv",
        usecols=["protocol", "model", "seed", "flight_id", "detected"],
    )
    r = r[(r["protocol"] == "seen") & (r["model"] == "random_forest")]
    r = r.rename(columns={"model": "detector"}).drop(columns="protocol")
    q = pd.read_csv(
        lp.PDIR / "flights.csv",
        usecols=["flight_id", "kind", "protocol", "seed", "detected"],
    )
    q = q[q["protocol"] == "seen"]
    q["detector"] = q["kind"].map({"ae": "lstm_ae", "clf": "lstm_supervised"})
    q = q.drop(columns=["kind", "protocol"])
    return pd.concat([p, r, q], ignore_index=True)


def selftest(flights):
    """Agreement with Step C. Step C's window latch (random forest, LSTMs) also
    credited an alarm raised by a window that still overlapped the attack but ended
    after it, i.e. up to one window length after the spoofing stopped; Step D, like
    Module 10 and the physics detectors, counts only alarms while the attack is on.
    The comparison applies Step C's rule to Step C's detectors, so it tests the scorer,
    and late_only counts the detections that the stricter rule removes."""
    att = flights[flights["attacked"]]
    m = att.merge(reference(), on=["flight_id", "detector", "seed"],
                  suffixes=("", "_stepC"))  # fmt: skip
    rows = []
    for det, g in m.groupby("detector"):
        here = g["detected"].astype(bool)
        late = pd.Series(False, index=g.index)
        if det in D.LEARNED:
            late = ~here & (g["after_end_s"] <= SECONDS)
        agree = (here | late) == g["detected_stepC"].astype(bool)
        rows.append({"detector": det, "flights_compared": len(g),
                     "agreement": float(agree.mean()),
                     "only_here": int((here & ~g["detected_stepC"]).sum()),
                     "only_stepC": int((~(here | late) & g["detected_stepC"]).sum()),
                     "late_only": int(late.sum()),
                     "ok": bool(agree.mean() >= AGREEMENT)})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------- frontier --


def moved(frame):
    """Metres moved before the first alarm; the whole drift when never caught."""
    return np.where(frame["detected"], frame["displacement_m"], frame["final_offset_m"])


def frontier(flights):
    """D(r) and detection rate per detector and rate, averaged over seeds per flight,
    with bootstrap 95% intervals over flights and the spread over seeds."""
    sw = flights[flights["set"] == "sweep"].assign(moved_m=lambda d: moved(d))
    rows = []
    for (det, rate), g in sw.groupby(["detector", "drift_rate_mps"]):
        per = g.groupby("flight_id").agg(det_=("detected", "mean"),
                                         moved=("moved_m", "mean"))  # fmt: skip
        seeds = g.groupby("seed").agg(det_=("detected", "mean"),
                                      moved=("moved_m", "median"))  # fmt: skip
        d_lo, d_hi = bootstrap_ci(per["det_"].to_numpy())
        m_lo, m_hi = bootstrap_ci(per["moved"].to_numpy(), stat=np.median)
        hit = g[g["detected"].astype(bool)]
        rows.append(
            {
                "detector": det,
                "drift_rate_mps": rate,
                "flights": len(per),
                "seeds": g["seed"].nunique(),
                "detected": float(per["det_"].mean()),
                "detected_ci_low": d_lo,
                "detected_ci_high": d_hi,
                "detected_sd_seeds": (
                    float(seeds["det_"].std(ddof=1)) if len(seeds) > 1 else np.nan
                ),
                "D_m": float(per["moved"].median()),
                "D_ci_low": m_lo,
                "D_ci_high": m_hi,
                "D_sd_seeds": (
                    float(seeds["moved"].std(ddof=1)) if len(seeds) > 1 else np.nan
                ),
                "median_delay_s": (
                    float(hit["delay_s"].median()) if len(hit) else np.nan
                ),
                "median_final_offset_m": float(g["final_offset_m"].median()),
            }
        )
    out = pd.DataFrame(rows)
    fa = false_alarm_rows(flights)
    return out.merge(fa, on="detector", how="left")


def false_alarm_rows(flights):
    """False alarms on the clean test flights, per detector (mean over seeds)."""
    clean = flights[~flights["attacked"]]
    rows = []
    for det, g in clean.groupby("detector"):
        per = g.groupby("seed").agg(k=("n_alarms", "sum"), h=("hours", "sum"))
        k, h = float(per["k"].mean()), float(per["h"].mean())
        rows.append({"detector": det, "fa_alarms": k, "fa_hours": h,
                     "fa_per_hour": k / h if h else np.nan,
                     "fa_upper_95": poisson_upper(int(round(k)), h)})  # fmt: skip
    return pd.DataFrame(rows)


def finish(flights, started):
    D.OUT.mkdir(parents=True, exist_ok=True)
    D.save_csv(flights, D.OUT / "fixed_flights.csv")
    check = selftest(flights)
    D.save_csv(check, D.OUT / "fixed_selftest.csv")
    table = frontier(flights)
    D.save_csv(table, TABLE)
    pd.set_option("display.width", 220)
    print(
        "\nSELF-TEST against Step C's own per-flight verdicts (attacked test flights)"
    )
    print(check.to_string(index=False))
    if not check["ok"].all():
        print("\n  !! a scorer disagrees with Step C: do not trust the frontier yet")
    print("\nFIXED-DRIFT STEALTH FRONTIER  (D = median metres moved before the first "
          "alarm; whole drift when never caught)")  # fmt: skip
    show = table.pivot(index="drift_rate_mps", columns="detector", values="D_m")
    print(show.round(1).to_string())
    print("\nshare of sweep flights detected")
    show = table.pivot(index="drift_rate_mps", columns="detector", values="detected")
    print(show.round(2).to_string())
    print(f"\nElapsed {time.time() - started:.0f} s   saved {TABLE}")
    return table, check


def run(limit=None, part=None):
    started = time.time()
    print("=" * 78)
    print("AEROVIGIL - PLAN M8: FIXED-DRIFT STEALTH FRONTIER (all detectors)")
    print("=" * 78)
    flights = test_flights(limit)
    if part:
        i, n = part
        flights = flights.iloc[i - 1 :: n]
    print(f"test flights {len(flights)}")
    dets = D.build()
    print(f"detectors {len(dets)} (with seeds)   [{time.time() - started:.0f} s]")
    out = score_all(flights, dets, started)
    if part:
        path = D.OUT / f"fixed_flights_part{part[0]}of{part[1]}.csv"
        D.atomic_write(path, lambda p: out.to_csv(p, index=False))
        print(f"saved {path}")
        return out
    return finish(out, started)


def merge():
    parts = sorted(D.OUT.glob("fixed_flights_part*of*.csv"))
    if not parts:
        raise SystemExit("no part files to merge")
    n = {int(p.stem.split("of")[-1]) for p in parts}
    if len(n) != 1 or len(parts) != n.pop():
        raise SystemExit(f"incomplete parts: {[p.name for p in parts]}")
    flights = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return finish(flights, time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--part", default=None, help="i/n: score every n-th flight from i")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    part = tuple(int(x) for x in a.part.split("/")) if a.part else None
    run(a.limit, part)


if __name__ == "__main__":
    main()
