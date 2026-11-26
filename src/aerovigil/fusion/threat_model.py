r"""AEROVIGIL Module 10 step 4: the threat-model summary, one table for the whole project.

Rows are attacker classes, columns are every detector the project built, and every
cell is a measured share of test flights detected with the median delay of the ones
caught. Nothing is re-run here: each cell is read from the per-flight results the
module itself saved, and pooled flight by flight (never by averaging summaries), so a
cell is exactly the share of the flights in that class.

Attacker classes
    real          the dataset's own simulator attacks (33 test flights)
    position-only synthetic, 5 families, GPS velocity left untouched (105)
    consistent    synthetic, 5 families, GPS velocity moved with the position (105)
    drift slow    consistent linear drift at 0.05-0.1 m/s (42)
    drift medium  consistent linear drift at 0.25 m/s (21)
    drift fast    consistent linear drift at 0.5-5 m/s (84)
    adaptive      Module 08's attacker, tuned against the chi-square test, mean drift
                  speed held at 0.05-5 m/s (147); and with no speed limit (21)

Detectors, each at its own module's one-alarm-per-flight-hour budget
    M05 RF        random forest on window features, event labels, trained on real +
                  synthetic attacks (Module 05 v2's headline configuration)
    M06 chi2      Module 06's chi-square test on the Kalman innovation
    M07 AE        LSTM autoencoder (normal flights only), mean over seeds
    M07 sup       supervised LSTM, mean over seeds
    M09 consist   Module 09's consistency detector (velocity + course channels)
    M10 kalman / velocity / fused   Module 10, one calibration for all three

Delays are each module's own definition: Modules 05-07 and 09 count from the label
window, Module 10 from the measured onset of the ramp. A cell reads "n/a" where a
module was never scored on that class (Module 05 v2 scored synthetic attacks by AUC
only; the adaptive attacker was replayed against Modules 06 and 10).

The first false-alarm row counts each detector's alarms on the same 21 normal test
flights. Hours come from Module 10's timestamp-based monitored time, on the same basis
fuse.py uses: the Kalman filter's own monitored time for the two chi-square tests (the
filter skips its warm-up and segments too short to filter), the row-level time for
the window and channel detectors, and the longer of the two for the fused detector.
That row rests on about 40 minutes of flight, so the second row adds Module 10's
cross-validated rate over all 144 normal flights, the reliable figure, for the three
Module 10 detectors (no other module was cross-validated).

Writes data/processed/module10/threat_model.csv (long) and threat_model_table.csv

Usage:
    python src\aerovigil\fusion\threat_model.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.fusion.fuse import poisson_upper
from aerovigil.fusion.signals import OUT_DIR, ROOT

P = ROOT / "data" / "processed"
CLASSES = (
    ("real", "Dataset's own attacks (simulator)"),
    ("position-only", "Synthetic, position only"),
    ("consistent", "Synthetic, velocity-consistent"),
    ("drift slow", "Consistent drift 0.05-0.1 m/s"),
    ("drift medium", "Consistent drift 0.25 m/s"),
    ("drift fast", "Consistent drift 0.5-5 m/s"),
    ("adaptive", "Adaptive, speed-limited 0.05-5 m/s"),
    ("adaptive free", "Adaptive, no speed limit"),
)
DETECTORS = ("M05 RF", "M06 chi2", "M07 AE", "M07 sup", "M09 consist",
             "M10 kalman", "M10 velocity", "M10 fused")  # fmt: skip


def attacker_class(attack_set, velocity_mode, rate):
    """One class per attacked flight, from the columns every module saved."""
    if attack_set == "orig":
        return "real"
    if attack_set in ("main", "syn"):
        return "position-only" if velocity_mode == "position_only" else "consistent"
    if attack_set == "sweep":
        if rate <= 0.1 + 1e-9:
            return "drift slow"
        return "drift medium" if rate <= 0.25 + 1e-9 else "drift fast"
    raise ValueError(f"unknown attack set {attack_set!r}")


def frame(det, sets, modes, rates, detected, delay, flight_ids):
    """Per-flight rows in the shared long format."""
    return pd.DataFrame({
        "detector": det,
        "attacker": [attacker_class(s, m, r) for s, m, r in zip(sets, modes, rates)],
        "flight_id": list(flight_ids),
        "detected": np.asarray(detected, bool),
        "delay_s": np.asarray(delay, float),
    })  # fmt: skip


def need(path):
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run that module first.")
    return pd.read_csv(path)


# ------------------------------------------------------------- per module --


def module05():
    r = need(P / "module05_v2" / "rescore_flight.csv")
    r = r[(r["task"] == "event") & (r["model"] == "random_forest")
          & (r["features"] == "all") & (r["sources"] == "orig+syn")]  # fmt: skip
    att, normal = r[r["attacked"]], r[~r["attacked"]]
    rows = frame("M05 RF", ["orig"] * len(att), ["na"] * len(att),
                 [np.nan] * len(att), att["detected"], att["delay_s"],
                 att["flight_id"])  # fmt: skip
    return rows, normal.set_index("flight_id")["alarms"]


def module06():
    a = need(P / "module06" / "alarms.csv")
    a = a[(a["detector"] == "nis_test") & (a["split"] == "test")]
    att, normal = a[a["attacked"]], a[~a["attacked"]]
    rows = frame("M06 chi2", att["set"], att["velocity_mode"],
                 att["drift_rate_mps"], att["detected"], att["delay_s"],
                 att["flight_id"])  # fmt: skip
    return rows, normal.set_index("flight_id")["n_alarms"]


def module07():
    f = need(P / "module07_v2" / "flights.csv")
    out, fa = [], {}
    for det, name in (("lstm_ae", "M07 AE"), ("lstm_supervised", "M07 sup")):
        d = f[(f["detector"] == det) & f["held_out"].isna()]
        att, normal = d[d["attacked"]], d[~d["attacked"]]
        # one row per flight and seed: pooling them gives the mean over seeds
        out.append(frame(name, att["attack_set"], att["velocity_mode"],
                         att["drift_rate_mps"], att["detected"], att["delay_s"],
                         att["flight_id"]))  # fmt: skip
        fa[name] = normal.groupby("flight_id")["alarms"].mean()
    return out, fa


def module09():
    a = need(P / "module09" / "alarms.csv")
    att, normal = a[a["attacked"]], a[~a["attacked"]]
    rows = frame("M09 consist", att["set"], att["velocity_mode"],
                 att["drift_rate_mps"], att["detected"], att["delay_s"],
                 att["flight_id"])  # fmt: skip
    return rows, normal.set_index("flight_id")["n_alarms"]


def module10(out_dir):
    f = need(out_dir / "flights.csv")
    att, normal = f[f["attacked"]], f[~f["attacked"]]
    rows, fa = [], {}
    for det in ("kalman", "velocity", "fused"):
        name = f"M10 {det}"
        rows.append(frame(name, att["set"], att["velocity_mode"],
                          att["drift_rate_mps"], att[f"{det}_detected"],
                          att[f"{det}_delay_s"], att["flight_id"]))  # fmt: skip
        fa[name] = normal.set_index("flight_id")[f"{det}_alarms"]
    return rows, fa


def adaptive(out_dir):
    """Module 08's own replay at Module 06's threshold, and Module 10's replay."""
    rows = []
    m08 = need(P / "module08" / "adaptive_free_test_flights.csv")
    front = need(P / "module08" / "adaptive_frontier.csv")
    # Module 08 saved per-flight results only for the free search; the constrained
    # rows are pooled from its frontier (21 flights each), which is exact for the
    # share detected. Their delays cannot be pooled from medians and are left out.
    con = front[front["search"] == "constrained"]
    hits = int(round(float((con["detection_rate"] * con["n"]).sum())))
    total = int(con["n"].sum())
    rows.append(pd.DataFrame({
        "detector": "M06 chi2", "attacker": "adaptive",
        "flight_id": [f"m08_{i}" for i in range(total)],
        "detected": np.arange(total) < hits, "delay_s": np.nan,
    }))  # fmt: skip
    rows.append(pd.DataFrame({
        "detector": "M06 chi2", "attacker": "adaptive free",
        "flight_id": m08["flight_id"], "detected": m08["detected"].astype(bool),
        "delay_s": m08["delay_s"],
    }))  # fmt: skip
    path = out_dir / "adaptive_flights.csv"
    if path.is_file():
        a = pd.read_csv(path)
        cls = np.where(a["search"] == "free", "adaptive free", "adaptive")
        for det in ("kalman", "velocity", "fused"):
            rows.append(pd.DataFrame({
                "detector": f"M10 {det}", "attacker": cls,
                "flight_id": a["flight_id"],
                "detected": a[f"{det}_detected"].astype(bool),
                "delay_s": a[f"{det}_delay_s"],
            }))  # fmt: skip
    return rows


# ---------------------------------------------------------------- assemble --


def collect(out_dir):
    long, fa = [], {}
    r, n = module05()
    long.append(r)
    fa["M05 RF"] = n
    r, n = module06()
    long.append(r)
    fa["M06 chi2"] = n
    r, n = module07()
    long += r
    fa.update(n)
    r, n = module09()
    long.append(r)
    fa["M09 consist"] = n
    r, n = module10(out_dir)
    long += r
    fa.update(n)
    long += adaptive(out_dir)
    return pd.concat(long, ignore_index=True), fa


def summarise(long):
    rows = []
    for (det, cls), g in long.groupby(["detector", "attacker"]):
        caught = g.loc[g["detected"], "delay_s"].dropna()
        rows.append({
            "detector": det, "attacker": cls, "flights": g["flight_id"].nunique(),
            "detected": float(g["detected"].mean()),
            "median_delay_s": float(caught.median()) if len(caught) else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows)


# Which monitored time each detector's alarms are divided by (see the top).
HOURS_BASIS = {"M06 chi2": "kalman", "M10 kalman": "kalman", "M10 fused": "fused"}


def monitored_hours(index, basis):
    """Per-flight monitored hours on one basis, the way fuse.hours_of counts them."""
    idx = index.set_index("flight_id")
    if basis == "kalman":
        return idx["kf_hours"]
    if basis == "fused":
        return idx[["kf_hours", "vel_hours"]].max(axis=1)
    return idx["vel_hours"]


def false_alarms(fa, out_dir):
    """Every detector's alarms on the same normal test flights."""
    index = need(out_dir / "signals_index.csv")
    rows = []
    for det in DETECTORS:
        counts = fa.get(det)
        if counts is None or not len(counts):
            continue
        basis = HOURS_BASIS.get(det, "rows")
        hours = monitored_hours(index, basis)
        if counts.index.difference(hours.index).size:
            raise SystemExit(f"{det}: normal flights missing from signals_index.csv")
        h = float(hours.reindex(counts.index).sum())
        k = float(counts.sum())
        rows.append({"detector": det, "scope": "normal test flights",
                     "hours_basis": basis, "flights": len(counts),
                     "alarms": k, "hours": h, "alarms_per_hour": k / h,
                     "upper_95_per_hour": poisson_upper(int(round(k)), h)})  # fmt: skip
    cv = need(out_dir / "false_alarms.csv")
    cv = cv[cv["scope"].str.startswith("cross-validated")]
    for _, r in cv.iterrows():
        rows.append({"detector": f"M10 {r['detector']}",
                     "scope": "cross-validated, all normal flights",
                     "hours_basis": HOURS_BASIS.get(f"M10 {r['detector']}", "rows"),
                     "flights": int(r["flights"]), "alarms": float(r["alarms"]),
                     "hours": float(r["hours"]),
                     "alarms_per_hour": float(r["alarms_per_hour"]),
                     "upper_95_per_hour": float(r["upper_95_per_hour"])})  # fmt: skip
    return pd.DataFrame(rows)


def cell(sub):
    if sub.empty:
        return "n/a"
    d, s = float(sub["detected"].iloc[0]), float(sub["median_delay_s"].iloc[0])
    return f"{d:4.0%} {s:5.1f}s" if np.isfinite(s) else f"{d:4.0%}    - "


def table(summary, fa):
    rows = []
    for key, label in CLASSES:
        n = summary.loc[summary["attacker"] == key, "flights"]
        row = {"attacker": label, "flights": int(n.max()) if len(n) else 0}
        for det in DETECTORS:
            row[det] = cell(
                summary[(summary["attacker"] == key) & (summary["detector"] == det)]
            )
        rows.append(row)
    for scope, label in (
        ("normal test flights", "False alarms/h, normal test flights"),
        ("cross-validated, all normal flights", "False alarms/h, cross-validated"),
    ):
        sub = fa[fa["scope"] == scope]
        row = {"attacker": label,
               "flights": int(sub["flights"].max()) if len(sub) else 0}  # fmt: skip
        for det in DETECTORS:
            x = sub[sub["detector"] == det]
            row[det] = (
                f"{float(x['alarms_per_hour'].iloc[0]):.2f}/h" if len(x) else "n/a"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(output_dir=None):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    long, fa_counts = collect(out)
    summary = summarise(long)
    summary.to_csv(out / "threat_model.csv", index=False)
    fa = false_alarms(fa_counts, out)
    fa.to_csv(out / "threat_model_false_alarms.csv", index=False)
    tab = table(summary, fa)
    tab.to_csv(out / "threat_model_table.csv", index=False)
    pd.set_option("display.width", 250)
    print("=" * 100)
    print("AEROVIGIL - THREAT MODEL: share of test flights detected, median delay")
    print("=" * 100)
    print(tab.to_string(index=False))
    if not (out / "adaptive_flights.csv").is_file():
        print("\n(Module 10 adaptive columns are empty: run adaptive.py first.)")
    print(
        "\nAll detectors at one false alarm per flight-hour on their own calibration."
        "\nThe test-flight false-alarm row rests on about 40 minutes of flight; the"
        "\ncross-validated row (Module 10 only) rests on all 144 normal flights."
        "\nDetails, hours and 95% upper bounds: threat_model_false_alarms.csv"
    )
    print(f"\nSaved to {out}")
    return tab, summary, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.output_dir)


if __name__ == "__main__":
    main()
