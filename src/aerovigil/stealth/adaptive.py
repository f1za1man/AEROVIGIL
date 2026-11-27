r"""Plan M8 (Step D): the adaptive attacker against every detector.

Module 08 searched for the drift that hides best from Module 06's chi-square test; the
learned detectors and the CUSUM were never attacked. Here Module 08's own black-box
search (same hill climber, same profile, same measured-mean-speed constraint, same
evaluation budget) is run against each detector in turn:

    practice  5 validation flights   the attacker probes the detector here, and sees
                                     only alarms (when, and how far it got)
    exam      21 test flights        the winners are scored once, never probed

For each mean drift speed the attacker keeps whichever of two candidates did better on
the practice flights: its own winner against this detector, or Module 08's winner
against the chi-square test (transfer). The choice is made on practice flights, never
on the exam. The random forest and the LSTMs are searched at seed 42 and the winners
are examined against all five seeds.

The adaptive Stealth Frontier: D(r) is the median, over the exam flights, of the
distance moved before the first alarm (the whole drift when never caught), at the
1 alarm per flight-hour budget. Our own metric; a lower curve is a stronger detector.

Writes data/processed/stealth/adaptive/<detector>/{search.csv,best.json,practice.csv},
data/processed/stealth/adaptive_exam.csv and reports/tables/D_frontier_adaptive.csv

Usage:
    python src\aerovigil\stealth\adaptive.py --detector random_forest
    python src\aerovigil\stealth\adaptive.py --detector kalman --evals 4 --free-evals 4
    python src\aerovigil\stealth\adaptive.py --summarise
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from aerovigil.adaptive import search as m08
from aerovigil.adaptive.profile import DEFAULT, describe, profile_fn, to_unit
from aerovigil.evaluation.metrics import bootstrap_ci
from aerovigil.stealth import detectors as D

ADIR = D.OUT / "adaptive"
M08_BEST = D.ROOT / "data" / "processed" / "module08" / "adaptive_best.json"
TABLE = D.ROOT / "reports" / "tables" / "D_frontier_adaptive.csv"
SEARCH_SEED = 42


def key_of(rate):
    return "free" if rate is None else f"mean_rate={rate:g}"


def objective(det, practice, rate):
    if rate is None:
        return lambda p: m08.probe(det, practice, p)
    return lambda p: m08.constrained_objective(det, practice, p, rate, "consistent")


def search(name, evals=m08.EVALS, free_evals=m08.FREE_EVALS,
           flights=m08.SEARCH_FLIGHTS, rates=m08.MEAN_RATES, out_root=ADIR,
           det=None):  # fmt: skip
    """Search one detector; keep, per rate, the better of own winner and Module 08's."""
    started = time.time()
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    if det is None:
        seed = SEARCH_SEED if name in D.LEARNED else 0
        det = D.build((name,), seeds=(SEARCH_SEED,))[(name, seed)]
    practice = m08.normal_flights("validation", flights)
    m08_best = json.loads(M08_BEST.read_text())
    rng = np.random.default_rng(m08.SEED)
    print(f"[{name}] threshold {det.threshold:.4g}   practice flights {len(practice)}"
          f"   probes {len(rates)} x {evals} + {free_evals}", flush=True)  # fmt: skip
    best, hists, rows, warm = {}, [], [], None
    for rate in list(rates) + [None]:
        key, f = key_of(rate), objective(det, practice, rate)
        if rate is None:
            params, score, hist = m08.hill_climb(f, rng, free_evals, to_unit(DEFAULT))
        else:
            x0 = to_unit(m08.with_mean_rate(warm or DEFAULT, rate))
            params, score, hist = m08.hill_climb(f, rng, evals, x0)
            warm = params
        other = m08_best.get(key)
        other_score = f(other)[0] if other else -np.inf
        pick = "own" if score >= other_score else "module08"
        best[key] = params if pick == "own" else other
        hist.insert(0, "search", key)
        hists.append(hist)
        rows.append({"detector": name, "search": key, "own_score_m": score,
                     "module08_score_m": other_score, "picked": pick,
                     **best[key]})  # fmt: skip
        print(
            f"[{name}] {key:15s} own {score:8.2f} m   module08 {other_score:8.2f} m"
            f"   -> {pick}   [{time.time() - started:.0f} s]",
            flush=True,
        )
        print(f"        {describe(best[key])}", flush=True)
    pd.concat(hists, ignore_index=True).to_csv(out / "search.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "practice.csv", index=False)
    D.atomic_write(out / "best.json",
                   lambda p: p.write_text(json.dumps(best, indent=2)))  # fmt: skip
    return best


# ------------------------------------------------------------------- exam --


def exam(names=D.ALL, out_root=ADIR, dets=None, tag="before", limit=None):
    """Score every detector's winners on the test flights (every seed)."""
    started = time.time()
    dets = dets or D.build(names)
    flights = m08.normal_flights("test", limit)
    first = next(iter(dets.values()))
    rows = []
    for name in names:
        path = out_root / name / "best.json"
        if not path.is_file():
            print(f"  missing {path}: search {name} first")
            continue
        best = json.loads(path.read_text())
        mine = {k: d for k, d in dets.items() if k[0] == name}
        for key, params in best.items():
            fn = profile_fn(params)
            for fid, mission, g in flights:
                fl = first.flight(first.attacker.attack(g, fn))
                for (n, seed), det in mine.items():
                    rows.append({"tag": tag, "detector": n, "seed": seed,
                                 "search": key, "flight_id": fid,
                                 "mission_type": mission,
                                 **det.verdict(fl)})  # fmt: skip
        print(f"  exam {name:16s} [{time.time() - started:.0f} s]", flush=True)
    return pd.DataFrame(rows)


def rate_of(key):
    return float(key.split("=")[1]) if key.startswith("mean_rate=") else np.nan


def summarise(frame):
    """Adaptive D(r) and detection per detector, seeds averaged per flight."""
    rows = []
    f = frame.assign(moved_m=np.where(frame["detected"], frame["displacement_m"],
                                      frame["final_offset_m"]))  # fmt: skip
    for (tag, det, key), g in f.groupby(["tag", "detector", "search"], sort=False):
        per = g.groupby("flight_id").agg(
            det_=("detected", "mean"),
            moved=("moved_m", "mean"),
            speed=("achieved_mean_mps", "mean"),
        )
        seeds = g.groupby("seed").agg(det_=("detected", "mean"),
                                      moved=("moved_m", "median"))  # fmt: skip
        d_lo, d_hi = bootstrap_ci(per["det_"].to_numpy())
        m_lo, m_hi = bootstrap_ci(per["moved"].to_numpy(), stat=np.median)
        rows.append(
            {
                "tag": tag,
                "detector": det,
                "search": key,
                "mean_rate_mps": rate_of(key),
                "flights": len(per),
                "seeds": g["seed"].nunique(),
                "achieved_mean_mps": float(per["speed"].median()),
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
                "median_final_offset_m": float(g["final_offset_m"].median()),
                "alarms_before_onset": int(g["alarms_before_onset"].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_exam(limit=None):
    frame = exam(limit=limit)
    D.atomic_write(D.OUT / "adaptive_exam.csv",
                   lambda p: frame.to_csv(p, index=False))  # fmt: skip
    table = summarise(frame)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    report(table)
    return table


def report(table):
    pd.set_option("display.width", 220)
    print("\nADAPTIVE STEALTH FRONTIER (exam = 21 test flights never probed)")
    print(
        "D = median metres moved before the first alarm; whole drift when never caught"
    )
    print(table.pivot(index="search", columns="detector", values="D_m")
          .round(1).to_string())  # fmt: skip
    print("\nshare of exam flights detected")
    print(table.pivot(index="search", columns="detector", values="detected")
          .round(2).to_string())  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None, choices=D.ALL)
    ap.add_argument("--evals", type=int, default=m08.EVALS)
    ap.add_argument("--free-evals", type=int, default=m08.FREE_EVALS)
    ap.add_argument("--flights", type=int, default=m08.SEARCH_FLIGHTS)
    ap.add_argument("--summarise", action="store_true", help="exam all winners")
    ap.add_argument("--limit", type=int, default=None, help="exam flights")
    a = ap.parse_args()
    if a.detector:
        search(a.detector, a.evals, a.free_evals, a.flights)
    if a.summarise:
        run_exam(a.limit)


if __name__ == "__main__":
    main()
