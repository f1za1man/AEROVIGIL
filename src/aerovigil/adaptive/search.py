r"""AEROVIGIL Module 08 step 1: hill-climb the attack profile against the detector.

Two searches run here.

The constrained search fixes the mean drift rate to each of the rates Module 06's sweep
used and optimises everything else: the onset ramp, the move-then-hold cycle, the
bearing and the wobble. That makes the result directly comparable with the Module 06
frontier, and the comparison answers the question Module 08 exists to ask. Module 06's
frontier was measured against one fixed attack shape. If an attacker allowed to reshape
the attack reaches the same displacement at the same mean rate without being caught,
then that frontier was a property of the attack family and not of the detector, and the
reported 0.1 m/s is optimistic.

The free search drops the constraint and simply asks how far the drone can be moved
before an alarm.

Protocol. The attacker practises on validation flights and is then evaluated on test
flights it has never probed. Searching and reporting on the same flights would measure
nothing but overfitting, which is the same mistake Module 05 and Module 07 made in
their first versions.

Feedback. The climber sees one number per probe: the displacement reached before the
first alarm, averaged over the practice flights. Nothing else about the detector.

Writes adaptive_search.csv (every probe), adaptive_best.json (the winning parameters),
adaptive_frontier.csv (the test-flight comparison against Module 06) in
data/processed/module08

Usage:
    python src\aerovigil\adaptive\search.py
    python src\aerovigil\adaptive\search.py --evals 40 --search-flights 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.adaptive.oracle import ROOT, KalmanOracle
from aerovigil.adaptive.profile import (
    BOUNDS,
    DEFAULT,
    NAMES,
    burst_seconds,
    clip,
    describe,
    from_unit,
    mean_rate,
    profile_fn,
    to_unit,
)

M03_BASE = ROOT / "data" / "processed" / "module03_v2" / "base"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
OUT_DIR = ROOT / "data" / "processed" / "module08"
SEED = 42
EVALS = 60
FREE_EVALS = 120
SEARCH_FLIGHTS = 3
STEP0 = 0.25
STEP_UP, STEP_DOWN = 1.3, 0.85
STEP_MIN = 0.01
RESTART_AFTER = 20
MEAN_RATES = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00)
MODULE06 = {0.05: 0.05, 0.10: 0.05, 0.25: 0.57, 0.50: 1.00,
            1.00: 1.00, 2.00: 1.00, 5.00: 1.00}  # fmt: skip
PLAIN_ONSET_S = 1.5  # Module 04's own ONSET_RAMP_S, for the self-test
SELFTEST_TOLERANCE = 0.20  # detection-rate gap above which the oracle is suspect
# the free parameters of the constrained search: rate_mps is set by the constraint
CONSTRAINED = tuple(n for n in NAMES if n != "rate_mps")


def normal_flights(split, limit=None):
    """Clean flights of one split, as frames, cheapest mission mix first."""
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    rep = rep[(rep["attacked"] == 0) & (rep["flight_id"].map(splits) == split)]
    if limit:
        rep = rep.groupby("mission", group_keys=False).head(max(1, limit // 3))
        rep = rep.head(limit)
    out = []
    for _, r in rep.iterrows():
        path = M03_BASE / r["file"]
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Build the Module 03 v2 base first.")
        out.append((r["flight_id"], r["mission"], pd.read_parquet(path)))
    if not out:
        raise SystemExit(f"No normal {split} flights found.")
    return out


def with_mean_rate(free, mean_rate):
    """Full parameters from the constrained vector: rate_mps follows from the duty."""
    p = dict(free)
    lo, hi, _ = BOUNDS["rate_mps"]
    duty = float(np.clip(p.get("duty", 1.0), *BOUNDS["duty"][:2]))
    p["duty"] = duty
    p["rate_mps"] = float(np.clip(mean_rate / max(duty, 1e-6), lo, hi))
    return clip(p)


def probe(oracle, flights, params, mode="consistent", named=None):
    """One candidate against every practice flight: mean score and what happened.

    named: (attack_type, params) to use Module 04's own profile instead of the adaptive
    one. Only the self-test passes it.
    """
    rows = []
    fn = None if named else profile_fn(params)
    kind, kparams = named if named else ("adaptive", None)
    for fid, mission, g in flights:
        rows.append({"flight_id": fid, "mission_type": mission,
                     **oracle(g, fn, mode, kind, kparams)})  # fmt: skip
    frame = pd.DataFrame(rows)
    return float(frame["score"].mean()), frame


def hill_climb(evaluate, rng, evals, x0, step0=STEP0, free=None):
    """Climb in unit space: widen the step on success, shrink it on failure, restart."""
    mask = (
        np.ones(len(NAMES), bool)
        if free is None
        else np.array([n in free for n in NAMES])
    )
    best_x, (best_score, _) = np.array(x0, float), evaluate(from_unit(x0))
    history = [{"eval": 1, "score": best_score, "accepted": True, **from_unit(best_x)}]
    step, stale = step0, 0
    for i in range(2, evals + 1):
        cand = best_x.copy()
        cand[mask] = cand[mask] + rng.normal(0.0, step, mask.sum())
        cand = np.clip(cand, 0.0, 1.0)
        score, _ = evaluate(from_unit(cand))
        better = score > best_score
        if better:
            best_x, best_score, step, stale = cand, score, min(step * STEP_UP, 0.5), 0
        else:
            step, stale = max(step * STEP_DOWN, STEP_MIN), stale + 1
        if stale >= RESTART_AFTER:
            cand = rng.random(len(NAMES))
            cand[~mask] = best_x[~mask]
            score, _ = evaluate(from_unit(cand))
            if score > best_score:
                best_x, best_score = cand, score
            step, stale = step0, 0
        history.append({"eval": i, "score": score, "accepted": bool(better),
                        **from_unit(cand)})  # fmt: skip
    return from_unit(best_x), best_score, pd.DataFrame(history)


def evaluate_on(oracle, flights, params, mode="consistent", named=None):
    """Run a finished parameter set over the evaluation flights."""
    _, frame = probe(oracle, flights, params, mode, named)
    return {
        "n": len(frame),
        "detection_rate": float(frame["detected"].mean()),
        "median_displacement_m": float(frame["displacement_m"].median()),
        "median_final_offset_m": float(frame["final_offset_m"].median()),
        "median_delay_s": float(frame["delay_s"].median()),
    }, frame


def selftest(oracle, exam, rates=MEAN_RATES, mode="consistent"):
    """Reproduce Module 06's sweep through this oracle before trusting the search.

    Column module04 injects Module 04's own linear_drift through Module 04's own code
    path, ease-in included, so it must land on the frontier Module 06 published. That
    is what validates the oracle; a gap there means the oracle is measuring a different
    detector and nothing below it can be believed.

    Column eased is the same mean rate with the adaptive profile's onset, which ramps
    the velocity instead of scaling the offset. Module 04's smooth_start multiplies the
    whole offset by a smoothstep, so the reported velocity overshoots to 1.69 times the
    drift rate while it eases in. The filter is tuned to trust GPS velocity almost
    absolutely, so that overshoot is itself detectable. Where module04 and eased differ,
    Module 06 was detecting the switch-on and not the drift.
    """
    rows = []
    for rate in rates:
        plain, _ = evaluate_on(
            oracle, exam, None, mode,
            named=("linear_drift", {"direction_deg": 45.0, "drift_rate_mps": rate}),
        )  # fmt: skip
        eased, _ = evaluate_on(oracle, exam, clip({
            **DEFAULT, "rate_mps": rate, "duty": 1.0, "jitter_m": 0.0,
            "onset_s": PLAIN_ONSET_S,
        }), mode)  # fmt: skip
        published = MODULE06.get(rate, np.nan)
        rows.append({
            "drift_rate_mps": rate, "module06": published,
            "module04": plain["detection_rate"], "eased": eased["detection_rate"],
            "gap": abs(plain["detection_rate"] - published),
            "disp_module04_m": plain["median_displacement_m"],
            "disp_eased_m": eased["median_displacement_m"],
        })  # fmt: skip
    frame = pd.DataFrame(rows)
    print("ORACLE SELF-TEST")
    print("  module06 = the frontier Module 06 published")
    print("  module04 = Module 04's own linear_drift through this oracle (must match)")
    print("  eased    = same mean rate, velocity-ramped onset (the attacker's version)")
    print(frame.round(3).to_string(index=False))
    worst = float(frame["gap"].max())
    trusted = worst <= SELFTEST_TOLERANCE
    print(f"  largest module04 gap {worst:.3f} (tolerance {SELFTEST_TOLERANCE:g}) -> "
          f"{'ok, the oracle reproduces Module 06' if trusted else 'SUSPECT'}")  # fmt: skip
    onset = frame["module04"] - frame["eased"]
    if trusted and (onset > SELFTEST_TOLERANCE).any():
        hit = frame.loc[onset > SELFTEST_TOLERANCE, "drift_rate_mps"].tolist()
        print(
            "  NOTE: at "
            + ", ".join(f"{r:g}" for r in hit)
            + " m/s, Module 06's detection came from Module 04's switch-on"
        )
        print("  overshoot, not from the drift. Those frontier points are artefacts.")
    print()
    return frame, trusted


def run(evals=EVALS, free_evals=FREE_EVALS, search_flights=SEARCH_FLIGHTS,
        rates=MEAN_RATES, output_dir=None, mode="consistent", threshold=None,
        skip_selftest=False):  # fmt: skip
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    oracle = KalmanOracle(threshold=threshold)
    practice = normal_flights("validation", search_flights)
    exam = normal_flights("test")
    rng = np.random.default_rng(SEED)
    print("=" * 74)
    print("AEROVIGIL - MODULE 08 ADAPTIVE ATTACKER (black-box, alarms only)")
    print("=" * 74)
    print(
        f"Detector: Module 06 chi-square, threshold {oracle.threshold:.2f}, "
        f"refractory {oracle.refractory_s:g} s"
    )
    print(
        f"Practice flights {len(practice)} (validation)   "
        f"exam flights {len(exam)} (test)   velocity mode {mode}"
    )
    print(f"Probes: {len(rates)} x {evals} constrained + {free_evals} free\n")
    started, probes, best, rows = time.time(), [], {}, []
    warm = None
    trusted = True
    if not skip_selftest:
        check, trusted = selftest(oracle, exam, rates, mode)
        check.to_csv(out / "adaptive_selftest.csv", index=False)
    for rate in rates:

        def evaluate(params, rate=rate):
            return probe(oracle, practice, with_mean_rate(params, rate), mode)

        # warm start: neighbouring rates want similar onsets and duty cycles, so the
        # previous solution is a far better starting point than the default
        x0 = to_unit(with_mean_rate(warm or DEFAULT, rate))
        params, score, hist = hill_climb(evaluate, rng, evals, x0, free=CONSTRAINED)
        warm = params
        params = with_mean_rate(params, rate)
        hist.insert(0, "search", f"mean_rate={rate:g}")
        probes.append(hist)
        best[f"mean_rate={rate:g}"] = params
        summary, _ = evaluate_on(oracle, exam, params, mode)
        achieved = mean_rate(params)
        rows.append({"search": "constrained", "mean_rate_mps": rate,
                     "achieved_mean_mps": achieved,
                     "constraint_met": bool(abs(achieved - rate) <= 0.1 * rate),
                     "burst_s": burst_seconds(params),
                     "practice_score_m": score, **summary,
                     "module06_detection_rate": MODULE06.get(rate, np.nan),
                     **params})  # fmt: skip
        print(
            f"  mean {rate:<5g} practice {score:8.2f} m -> test detection "
            f"{summary['detection_rate']:.2f} (Module 06 {MODULE06.get(rate, float('nan')):.2f})"
            f"  displacement {summary['median_displacement_m']:8.2f} m"
            f"   [{time.time() - started:.0f} s]"
        )
        print(f"        {describe(params)}")

    def evaluate_free(params):
        return probe(oracle, practice, params, mode)

    params, score, hist = hill_climb(evaluate_free, rng, free_evals, to_unit(DEFAULT))
    hist.insert(0, "search", "free")
    probes.append(hist)
    best["free"] = params
    summary, exam_frame = evaluate_on(oracle, exam, params, mode)
    rows.append({"search": "free", "mean_rate_mps": np.nan,
                 "achieved_mean_mps": mean_rate(params), "constraint_met": True,
                 "burst_s": burst_seconds(params),
                 "practice_score_m": score, **summary,
                 "module06_detection_rate": np.nan, **params})  # fmt: skip
    print(
        f"\n  free      practice {score:8.2f} m -> test detection "
        f"{summary['detection_rate']:.2f}  displacement "
        f"{summary['median_displacement_m']:8.2f} m"
    )
    print(f"        {describe(params)}")
    frontier = pd.DataFrame(rows)
    frontier.to_csv(out / "adaptive_frontier.csv", index=False)
    pd.concat(probes, ignore_index=True).to_csv(
        out / "adaptive_search.csv", index=False
    )
    exam_frame.to_csv(out / "adaptive_free_test_flights.csv", index=False)
    (out / "adaptive_best.json").write_text(json.dumps(best, indent=2))
    report(frontier, oracle, out, time.time() - started, trusted)
    return frontier, best


def report(frontier, oracle, out, elapsed, trusted=True):
    pd.set_option("display.width", 240)
    con = frontier[frontier["search"] == "constrained"]
    print("\nADAPTIVE FRONTIER vs MODULE 06 (test flights, detection rate)")
    table = con[[
        "mean_rate_mps", "achieved_mean_mps", "constraint_met",
        "module06_detection_rate", "detection_rate", "median_displacement_m",
        "median_delay_s", "rate_mps", "burst_s", "period_s", "onset_s",
    ]].rename(columns={"detection_rate": "adaptive"})  # fmt: skip
    print(table.round(3).to_string(index=False))
    if not trusted:
        print("\nThe oracle self-test FAILED, so the rows below compare two different")
        print("detectors and mean nothing. Fix the oracle before reading any further.")
    if not con["constraint_met"].all():
        missed = con.loc[~con["constraint_met"], "mean_rate_mps"].tolist()
        print(
            "\nUnreachable mean rates (the rate bound caps them), excluded below: "
            + ", ".join(f"{r:g}" for r in missed)
        )
    con = con[con["constraint_met"]]
    at_cap = [
        name for name, (_, hi, _) in BOUNDS.items()
        if name in con and (con[name] >= hi - 1e-6).any()
    ]  # fmt: skip
    if at_cap:
        print(
            "\nParameters the search pushed onto their upper bound: "
            + ", ".join(at_cap)
            + ".\nThe frontier below is therefore a lower bound on what an attacker"
            " can do."
        )
    worse = con["detection_rate"] < con["module06_detection_rate"] - 1e-9
    print(
        f"\nThe adaptive attacker evaded more often than the fixed drift at "
        f"{int(worse.sum())} of {len(con)} rates."
    )
    if worse.any():
        print("Rates where Module 06's frontier was optimistic:")
        print(
            con.loc[
                worse,
                [
                    "mean_rate_mps",
                    "module06_detection_rate",
                    "detection_rate",
                    "median_displacement_m",
                ],
            ]  # fmt: skip
            .round(3)
            .to_string(index=False)
        )
    else:
        print("Module 06's frontier held up: reshaping the attack bought nothing.")
    free = frontier[frontier["search"] == "free"].iloc[0]
    print(
        f"\nBest evasion found: {free['median_displacement_m']:.1f} m of displacement "
        f"at a {free['detection_rate']:.2f} detection rate on the test flights."
    )
    print(f"\n{oracle.calls:,} probes in {elapsed:.0f} s   saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=EVALS)
    ap.add_argument("--free-evals", type=int, default=FREE_EVALS)
    ap.add_argument("--search-flights", type=int, default=SEARCH_FLIGHTS)
    ap.add_argument("--mode", default="consistent")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--skip-selftest", action="store_true")
    a = ap.parse_args()
    run(a.evals, a.free_evals, a.search_flights, MEAN_RATES, a.output_dir,
        a.mode, a.threshold, a.skip_selftest)  # fmt: skip


if __name__ == "__main__":
    main()
