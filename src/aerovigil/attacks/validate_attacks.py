"""Validate the generated attack set against the Module 04 rules.

Usage:
    python src\\aerovigil\\attacks\\validate_attacks.py
    python src\\aerovigil\\attacks\\validate_attacks.py --sample 100
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    get_columns,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.attacks.injector import EARTH_R, EDGE_TOL_S, TRUTH_COLUMNS  # noqa: E402
from aerovigil.attacks.sampler import sample_attack  # noqa: E402

PARAM_RANGE_KEYS = {  # (manifest column, config key) per attack type
    "abrupt_jump": [("magnitude_m", "magnitude_m_range")],
    "constant_offset": [
        ("magnitude_m", "magnitude_m_range"),
        ("ramp_seconds", "ramp_seconds_range"),
    ],
    "linear_drift": [("drift_rate_mps", "drift_rate_mps_range")],
    "exponential_ramp": [
        ("initial_rate_mps", "initial_rate_mps_range"),
        ("time_constant_s", "time_constant_s_range"),
    ],
    "low_amplitude_wander": [("amplitude_m", "amplitude_m_range")],
}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "attacks" / "attacks.yaml")
    )
    ap.add_argument("--sample", type=int, default=None, help="check only N flights")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols, scale = get_columns(cfg, pre)
    out_root = (
        Path(a.output_dir) if a.output_dir else resolve(cfg["paths"]["output_dir"])
    )
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    man = pd.read_csv(out_root / "attack_manifest.csv")
    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    src_file = dict(zip(report["flight_id"].astype(str), report["file"]))
    src_attack_rows = dict(zip(report["flight_id"].astype(str), report["attack_rows"]))
    clean_dir = resolve(cfg["paths"]["clean_flights_dir"])
    tcfg = cfg["timing"]

    fails, stats = defaultdict(list), defaultdict(int)
    todo = (
        man if a.sample is None else man.sample(min(a.sample, len(man)), random_state=0)
    )

    print("=" * 65)
    print("AEROVIGIL - ATTACK VALIDATION")
    print("=" * 65)
    print(f"Manifest rows: {len(man):,}   Flights checked in detail: {len(todo):,}\n")

    changed = (
        {
            cols["lat"],
            cols["lon"],
            cols["vn"],
            cols["ve"],
            cols["label"],
            cols["flight"],
        }
        | ({cols["enabled"]} if cols["enabled"] else set())
        | ({cols["speed"]} if cols.get("speed") else set())
    )
    max_offset_err = 0.0

    for n, (_, m) in enumerate(todo.iterrows(), start=1):
        fid = m["attacked_flight_id"]
        s = pd.read_parquet(clean_dir / src_file[m["source_flight_id"]])
        g = pd.read_parquet(out_root / "attacked" / m["file"])

        if len(s) != len(g) or not np.array_equal(
            s[cols["time"]].to_numpy(), g[cols["time"]].to_numpy()
        ):
            fails["rows and timestamps identical to source"].append(fid)
            continue
        t = (g[cols["time"]].to_numpy(dtype="int64") - g[cols["time"]].iloc[0]) * scale
        tau = t - m["start_s"]
        active = (tau >= 0) & (tau <= m["duration_s"] + EDGE_TOL_S)
        lab = g[cols["label"]].to_numpy()

        # Threat model: only GPS position/velocity may change.
        untouched = [c for c in s.columns if c not in changed]
        bad = [c for c in untouched if not s[c].equals(g[c])]
        if bad:
            fails["IMU/baro/magnetometer/other columns untouched"].append(
                f"{fid}: {bad[:3]}"
            )

        # Nothing changes outside the attack window.
        gps4 = [cols["lat"], cols["lon"], cols["vn"], cols["ve"]]
        if not np.array_equal(
            s.loc[~active, gps4].to_numpy(), g.loc[~active, gps4].to_numpy()
        ):
            fails["GPS unchanged outside the attack window"].append(fid)
        if (g["spoof_offset_m"].to_numpy()[~active] != 0).any():
            fails["spoof offset is zero outside the attack window"].append(fid)

        # Recorded ground-truth offset equals the real position change.
        d = haversine(
            s[cols["lat"]].to_numpy(),
            s[cols["lon"]].to_numpy(),
            g[cols["lat"]].to_numpy(),
            g[cols["lon"]].to_numpy(),
        )
        err = np.abs(d - g["spoof_offset_m"].to_numpy()).max()
        max_offset_err = max(max_offset_err, err)
        if err > 0.05:
            fails["recorded offset equals haversine displacement (<0.05 m)"].append(
                f"{fid}: {err:.3f}"
            )

        # Labels and signed time since attack start.
        if (
            not np.array_equal(lab == 1, active)
            or int(active.sum()) != m["attack_rows"]
        ):
            fails["labels match the attack window"].append(fid)
        if cols["enabled"] and not np.array_equal(g[cols["enabled"]].to_numpy(), lab):
            fails["attack_enabled equals attack_label"].append(fid)
        tfs = g["time_from_attack_start_s"].to_numpy()
        if not (np.allclose(tfs, tau) and (tfs[lab == 1] >= 0).all()):
            fails["time_from_attack_start_s is consistent"].append(fid)

        # Velocity behaviour.
        dvn = g[cols["vn"]].to_numpy() - s[cols["vn"]].to_numpy()
        dve = g[cols["ve"]].to_numpy() - s[cols["ve"]].to_numpy()
        if m["velocity_mode"] == "position_only":
            if np.abs(dvn).max() > 0 or np.abs(dve).max() > 0:
                fails["position_only attacks leave velocity untouched"].append(fid)
        elif m["attack_type"] != "abrupt_jump":
            ta = t[active]
            if len(ta) > 2 and np.diff(ta).max() < 1.0:  # skip windows with log gaps
                for dv, off in (
                    (dvn, g["spoof_offset_north_m"]),
                    (dve, g["spoof_offset_east_m"]),
                ):
                    integral = (
                        np.trapezoid(dv[active], ta)
                        if hasattr(np, "trapezoid")
                        else np.trapz(dv[active], ta)
                    )
                    expected = off.to_numpy()[active][-1] - off.to_numpy()[active][0]
                    if abs(integral - expected) > max(0.5, 0.02 * abs(expected)):
                        fails["consistent velocity integrates to the offset"].append(
                            fid
                        )
                        break
                stats["velocity integral checked"] += 1
            else:
                stats["velocity integral skipped (log gap in attack)"] += 1
        # GPS speed column must follow the velocity components (or stay untouched).
        if cols.get("speed") and cols["speed"] in g.columns:
            dsp = g[cols["speed"]].to_numpy() - s[cols["speed"]].to_numpy()
            if m["velocity_mode"] == "position_only":
                speed_ok = np.abs(dsp).max() == 0
            else:
                expect = (
                    np.hypot(g[cols["vn"]], g[cols["ve"]]).to_numpy()
                    - np.hypot(s[cols["vn"]], s[cols["ve"]]).to_numpy()
                )
                speed_ok = np.allclose(dsp, expect, atol=1e-6)
            if not speed_ok:
                fails["GPS speed stays consistent with velocity components"].append(fid)

        # Timing bounds.
        if (
            m["start_s"] < tcfg["min_start_s"] - 1e-6
            or m["end_s"] > m["flight_duration_s"] + 1e-6
            or m["duration_s"] < tcfg["min_duration_s"] - 1e-6
        ):
            fails["timing within configured bounds"].append(fid)
        if n % 100 == 0 or n == len(todo):
            print(f"Checked {n}/{len(todo)} flights")

    # ---------------- manifest-level checks ----------------
    for typ, pairs in PARAM_RANGE_KEYS.items():
        sub = man[(man.attack_type == typ) & (man.attack_set == "main")]
        for col, key in pairs:
            lo, hi = cfg["attack_types"][typ][key]
            if len(sub) and not sub[col].between(lo - 1e-9, hi + 1e-9).all():
                fails["main-set parameters inside configured ranges"].append(
                    f"{typ}.{col}"
                )
    sweep = man[man.attack_set == "sweep"]
    if (
        len(sweep)
        and not sweep["drift_rate_mps"].isin(cfg["sweep"]["drift_rates_mps"]).all()
    ):
        fails["sweep rates equal configured rates"].append("sweep")

    sub = man.sample(min(30, len(man)), random_state=1)
    repro = 0
    for _, m in sub.iterrows():
        spec = sample_attack(
            cfg,
            m["source_flight_id"],
            m["flight_duration_s"],
            m["attack_set"],
            m["attack_type"],
            int(m["draw"]),
            m["drift_rate_mps"] if m["attack_set"] == "sweep" else None,
        )
        if (
            spec
            and abs(spec["start_s"] - m["start_s"]) < 1e-9
            and spec["velocity_mode"] == m["velocity_mode"]
            and spec["params"] == json.loads(m["params_json"])
        ):
            repro += 1
        else:
            fails["sampler reproduces manifest parameters"].append(
                m["attacked_flight_id"]
            )

    if not (man["split"] == man["source_flight_id"].map(split_map)).all():
        fails["attacked flight inherits source split"].append("manifest")
    if not man["attacked_flight_id"].is_unique:
        fails["attacked flight IDs unique"].append("manifest")
    if any(src_attack_rows[x] != 0 for x in man["source_flight_id"].unique()):
        fails["all source flights are fully normal"].append("manifest")
    types_per_source = (
        man[man.attack_set == "main"]
        .groupby("source_flight_id")["attack_type"]
        .nunique()
    )
    if not (types_per_source == len(cfg["enabled_attack_types"])).all():
        fails["every source flight has every attack type"].append("manifest")
    forbid = [
        c
        for c in TRUTH_COLUMNS
        if c not in pd.read_parquet(out_root / "attacked" / man["file"].iloc[0]).columns
    ]
    if forbid:
        fails["truth columns present"].append(str(forbid))

    # ---------------- report ----------------
    print("\n" + "-" * 65)
    print(
        f"Largest |recorded offset - haversine displacement| : {max_offset_err:.5f} m"
    )
    print(f"Sampler reproduced {repro}/{len(sub)} manifest rows exactly")
    for k, v in stats.items():
        print(f"{k}: {v}")
    all_checks = [
        "rows and timestamps identical to source",
        "IMU/baro/magnetometer/other columns untouched",
        "GPS unchanged outside the attack window",
        "spoof offset is zero outside the attack window",
        "recorded offset equals haversine displacement (<0.05 m)",
        "labels match the attack window",
        "attack_enabled equals attack_label",
        "time_from_attack_start_s is consistent",
        "position_only attacks leave velocity untouched",
        "consistent velocity integrates to the offset",
        "GPS speed stays consistent with velocity components",
        "timing within configured bounds",
        "main-set parameters inside configured ranges",
        "sweep rates equal configured rates",
        "sampler reproduces manifest parameters",
        "attacked flight inherits source split",
        "attacked flight IDs unique",
        "all source flights are fully normal",
        "every source flight has every attack type",
        "truth columns present",
    ]
    print("\nVALIDATION RESULTS")
    for c in all_checks:
        ok = c not in fails
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {c}"
            + ("" if ok else f"  -> {len(fails[c])} flight(s), e.g. {fails[c][:2]}")
        )
    print("=" * 65)
    if fails:
        raise SystemExit("Attack validation FAILED")


if __name__ == "__main__":
    main()
