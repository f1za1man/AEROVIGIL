r"""Plan M8 (Step D): the white-box worst case.

The black-box attacker of adaptive.py sees only alarms, practises on other flights and
must commit to one attack before the exam. The white-box attacker of the threat model
knows everything: the detector, its threshold, and the very flight it is attacking. It
can therefore simulate every candidate attack on that flight and fly the one that the
detector misses. Nothing stronger is possible within the attack family, so this is the
worst case, and it bounds what any cleverer search could still achieve.

Two measurements, on the 21 test flights:

  per mean speed  For each speed r the attacker tries every distinct adaptive winner at
                  that speed (Module 08's and each detector's own), each at its own
                  bearing and at +90, +180 and +270 degrees, plus a plain warp-onset
                  linear drift. A flight counts as detected only if every candidate is
                  caught. D(r) is the whole drift of the best hidden candidate, or the
                  largest displacement at the first alarm when none hides.

  free            Starting from the detector's own free winner, the attacker scales the
                  drift speed by bisection (12 steps, x0.01 to x100) to the largest
                  speed that stays hidden on that flight, and reports how far that
                  moves the drone. Done for the Kalman and fused detectors, the two the
                  plan names, and for the velocity channel they share.

Each candidate flight is built once and scored by every detector at once (seed 42 for
the learned ones), so all detectors face exactly the same candidates.

Writes data/processed/stealth/whitebox_flights.csv, whitebox_free.csv and
reports/tables/D_whitebox.csv

Usage:
    python src\aerovigil\stealth\whitebox.py
    python src\aerovigil\stealth\whitebox.py --limit 2 --part 1/3
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from aerovigil.adaptive import search as m08
from aerovigil.adaptive.profile import clip, profile_fn
from aerovigil.evaluation.metrics import bootstrap_ci
from aerovigil.stealth import detectors as D
from aerovigil.stealth.adaptive import ADIR, M08_BEST, key_of

TABLE = D.ROOT / "reports" / "tables" / "D_whitebox.csv"
TURNS = (0.0, 90.0, 180.0, 270.0)
FREE_DETECTORS = ("kalman", "fused", "velocity")
BISECT_STEPS = 12
SCALE_RANGE = (0.01, 100.0)
SEED = 42


def linear(rate):
    """A plain steady drift at `rate`, eased in over 20 s (the profile's own onset)."""
    return clip(
        {
            "rate_mps": rate,
            "onset_s": 20.0,
            "period_s": 600.0,
            "duty": 1.0,
            "bearing_deg": 0.0,
            "jitter_m": 0.0,
            "jitter_period_s": 10.0,
        }
    )


def candidates(rate, names):
    """Distinct profiles at one mean speed, each turned to four bearings."""
    key = key_of(rate)
    shapes = [json.loads(M08_BEST.read_text()).get(key)]
    for n in names:
        p = ADIR / n / "best.json"
        if p.is_file():
            shapes.append(json.loads(p.read_text()).get(key))
    if rate is not None:
        shapes.append(linear(rate))
    seen, out = set(), []
    for s in shapes:
        if not s:
            continue
        sig = json.dumps(s, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        for turn in TURNS:
            out.append(clip(dict(s, bearing_deg=s["bearing_deg"] + turn)))
    return out


def per_speed(dets, flights, started):
    first = next(iter(dets.values()))
    names = sorted({k[0] for k in dets})
    rows = []
    for rate in m08.MEAN_RATES:
        cands = candidates(rate, names)
        for fid, mission, g in flights:
            for c, params in enumerate(cands):
                fl = first.flight(first.attacker.attack(g, profile_fn(params)))
                for (n, _), det in dets.items():
                    v = det.verdict(fl)
                    rows.append({"mean_rate_mps": rate, "flight_id": fid,
                                 "mission_type": mission, "candidate": c,
                                 "detector": n, **v})  # fmt: skip
        print(f"  speed {rate:g}: {len(cands)} candidates x {len(flights)} flights   "
              f"[{time.time() - started:.0f} s]", flush=True)  # fmt: skip
    return pd.DataFrame(rows)


def worst_case(frame):
    """Per detector, speed and flight: hidden if any candidate hides."""
    rows = []
    for (det, rate, fid), g in frame.groupby(["detector", "mean_rate_mps",
                                              "flight_id"]):  # fmt: skip
        hidden = g[~g["detected"].astype(bool)]
        if len(hidden):
            moved, caught = float(hidden["final_offset_m"].max()), False
        else:
            moved, caught = float(g["displacement_m"].max()), True
        rows.append({"detector": det, "mean_rate_mps": rate, "flight_id": fid,
                     "detected": caught, "moved_m": moved,
                     "candidates": len(g)})  # fmt: skip
    per = pd.DataFrame(rows)
    out = []
    for (det, rate), g in per.groupby(["detector", "mean_rate_mps"]):
        d_lo, d_hi = bootstrap_ci(g["detected"].astype(float).to_numpy())
        m_lo, m_hi = bootstrap_ci(g["moved_m"].to_numpy(), stat=np.median)
        out.append({"detector": det, "mean_rate_mps": rate, "flights": len(g),
                    "candidates": int(g["candidates"].iloc[0]),
                    "detected": float(g["detected"].mean()),
                    "detected_ci_low": d_lo, "detected_ci_high": d_hi,
                    "D_m": float(g["moved_m"].median()),
                    "D_ci_low": m_lo, "D_ci_high": m_hi})  # fmt: skip
    return per, pd.DataFrame(out)


def free(dets, flights, started):
    """Largest hidden speed multiple of the detector's free winner, per flight."""
    first = next(iter(dets.values()))
    rows = []
    for name in FREE_DETECTORS:
        det = dets.get((name, 0))
        path = ADIR / name / "best.json"
        if det is None or not path.is_file():
            continue
        base = json.loads(path.read_text())["free"]
        for fid, mission, g in flights:
            lo, hi = np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])
            best = None
            for _ in range(BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                p = clip(dict(base, rate_mps=base["rate_mps"] * np.exp(mid)))
                v = det.verdict(first.flight(first.attacker.attack(g, profile_fn(p))))
                if v["detected"]:
                    hi = mid
                else:
                    lo, best = mid, (p, v)
            rows.append(
                {
                    "detector": name,
                    "flight_id": fid,
                    "mission_type": mission,
                    "hidden": best is not None,
                    "rate_mps": best[0]["rate_mps"] if best else np.nan,
                    "achieved_mean_mps": best[1]["achieved_mean_mps"] if best else 0.0,
                    "moved_m": best[1]["final_offset_m"] if best else 0.0,
                }
            )
        print(f"  free {name}: [{time.time() - started:.0f} s]", flush=True)
    return pd.DataFrame(rows)


def run(limit=None, part=None):
    started = time.time()
    print("=" * 78)
    print("AEROVIGIL - PLAN M8: WHITE-BOX WORST CASE (knows detector and flight)")
    print("=" * 78)
    flights = m08.normal_flights("test", limit)
    if part:
        flights = flights[part[0] - 1 :: part[1]]
    dets = D.build(seeds=(SEED,))
    print(f"exam flights {len(flights)}   detectors {len(dets)}   "
          f"[{time.time() - started:.0f} s]")  # fmt: skip
    frame = per_speed(dets, flights, started)
    fr = free(dets, flights, started)
    tag = f"_part{part[0]}of{part[1]}" if part else ""
    D.atomic_write(D.OUT / f"whitebox_flights{tag}.csv",
                   lambda p: frame.to_csv(p, index=False))  # fmt: skip
    D.atomic_write(D.OUT / f"whitebox_free{tag}.csv",
                   lambda p: fr.to_csv(p, index=False))  # fmt: skip
    if part:
        return frame
    return finish(frame, fr)


def finish(frame, fr):
    _, table = worst_case(frame)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    pd.set_option("display.width", 220)
    print("\nWHITE-BOX: share of test flights where every candidate was caught")
    print(table.pivot(index="mean_rate_mps", columns="detector", values="detected")
          .round(2).to_string())  # fmt: skip
    print("\nWHITE-BOX D(r): median metres moved before the first alarm")
    print(table.pivot(index="mean_rate_mps", columns="detector", values="D_m")
          .round(1).to_string())  # fmt: skip
    if len(fr):
        print("\nWHITE-BOX free: fastest hidden drift per flight (median)")
        print(fr.groupby("detector")[["hidden", "achieved_mean_mps", "moved_m"]]
              .median().round(2).to_string())  # fmt: skip
    return table


def merge():
    parts = sorted(D.OUT.glob("whitebox_flights_part*of*.csv"))
    if not parts:
        raise SystemExit("no part files")
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    fr = pd.concat([pd.read_csv(p) for p in
                    sorted(D.OUT.glob("whitebox_free_part*of*.csv"))],
                   ignore_index=True)  # fmt: skip
    D.atomic_write(D.OUT / "whitebox_flights.csv",
                   lambda p: frame.to_csv(p, index=False))  # fmt: skip
    D.atomic_write(D.OUT / "whitebox_free.csv", lambda p: fr.to_csv(p, index=False))
    return finish(frame, fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--part", default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    part = tuple(int(x) for x in a.part.split("/")) if a.part else None
    run(a.limit, part)


if __name__ == "__main__":
    main()
