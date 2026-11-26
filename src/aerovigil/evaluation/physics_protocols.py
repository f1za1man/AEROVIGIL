r"""Plan M9 for the physics detectors: seen, unseen missions, unseen attack types, CV.

Five detectors that are calibrated, not trained:

    kalman       Module 06's chi-square test on the horizontal NIS (Module 10 signals)
    velocity     Module 09's velocity-consistency channel (Module 10 signals)
    fused        either of the two, the budget split between them (Module 10)
    cusum        Module 06's two-sided CUSUM on the whitened innovations
    consistency  Module 09's detector: the larger of the velocity and course channels

None of them learns from attacks, apart from CUSUM's reference value k, which Module 06
picks on the validation attacks. So the protocols are:

    seen       calibrate on the normal train + validation flights, test on the test
               split (orig test flights and the synthetic test set)
    mission    calibrate on the normal train + validation flights of two missions, and
               choose CUSUM's k on their validation attacks; test on every flight of
               the third mission
    type       CUSUM's k chosen without family f; the other four never see attacks,
               so their unseen-type result is their seen result on family f
    cvfa       six folds over all 144 normal flights, stratified by mission: calibrate
               on five, count false alarms on the sixth

Same budget (1 alarm per flight-hour) and refractory (10 s) as every other detector;
monitored hours from timestamps; delays from the measured onset of the ramp.
Deterministic detectors have no seeds, so the spread is a bootstrap 95% interval over
flights.

Known caveat, reported with the results: the Kalman filter's noise levels
(module06/filter_noise.json) were tuned once on normal training flights of all three
missions, so in the unseen-mission protocol the filter itself is not mission-blind.
Only its thresholds are. Retuning the noise per fold would mean writing new
calibration files, which needs the owner's say-so first.

Writes data/processed/evaluation/physics_flights.csv, physics_calibration.csv and
reports/tables/C_physics.csv

Usage:
    python src\aerovigil\evaluation\physics_protocols.py
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.consistency.detector import (
    CHANNELS,
    alarm_indices,
    channel_values,
    monitored_hours,
    score_rows,
)
from aerovigil.consistency.detector import calibrate as consistency_calibrate
from aerovigil.consistency.detector import fit_standardiser as consistency_std
from aerovigil.evaluation.metrics import (
    BUDGET_PER_HOUR,
    REFRACTORY_S,
    bootstrap_ci,
    false_alarms,
)
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.fusion.fuse import DETECTORS as M10_DETECTORS
from aerovigil.fusion.fuse import alarms as m10_alarms
from aerovigil.fusion.fuse import calibrate as m10_calibrate
from aerovigil.fusion.fuse import folds_of, hours_of
from aerovigil.fusion.signals import M03_BASE, M04_BASE, load, monitored_seconds
from aerovigil.kalman.detectors import (
    W_COLS,
    calibrate_threshold,
    cusum_alarm_counts,
    cusum_alarms,
    fit_standardiser,
)

ROOT = Path(__file__).resolve().parents[3]
M06 = ROOT / "data" / "processed" / "module06"
M10 = ROOT / "data" / "processed" / "module10"
OUT = ROOT / "data" / "processed" / "evaluation"
CHAN_DIR = OUT / "consistency_channels"
TABLE = ROOT / "reports" / "tables" / "C_physics.csv"
MANIFEST = ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
MISSIONS = ("curved", "random", "straight")
FAMILIES = ("abrupt_jump", "constant_offset", "linear_drift", "exponential_ramp",
            "low_amplitude_wander", "original")  # fmt: skip
K_GRID = (0.25, 0.5, 1.0)
H_GRID = np.geomspace(1.0, 100000.0, 140)
DETECTORS = ("kalman", "velocity", "fused", "cusum", "consistency")


# ------------------------------------------------------------------ data --


def universe():
    """Every flight Module 10 has signals for, plus where its raw files live."""
    idx = pd.read_csv(M10 / "signals_index.csv")
    idx["attacked"] = idx["attacked"].astype(bool)
    rep = pd.read_csv(BASE_REPORT).set_index("flight_id")["file"]
    man = pd.read_csv(MANIFEST).set_index("attacked_flight_id")["file"]
    idx["base_path"] = [
        str(M03_BASE / rep[f]) if s == "orig" else str(M04_BASE / man[f])
        for f, s in zip(idx["flight_id"], idx["set"])
    ]
    idx["innov_path"] = [
        str(M06 / "innovations" / ("orig" if s == "orig" else
                                   "syn" if s == "main" else "sweep") / Path(p).name)
        for s, p in zip(idx["set"], idx["base_path"])
    ]  # fmt: skip
    return idx


def tuning_flights():
    """Validation attacks (real and synthetic main) for choosing CUSUM's k."""
    splits = pd.read_csv(ROOT / "data" / "interim" / "module03" / "flight_splits.csv")
    split = splits.set_index("flight_id")["split"]
    rep = pd.read_csv(BASE_REPORT)
    rows = [
        {"flight_id": r["flight_id"], "mission_type": r["mission"],
         "attack_type": "original",
         "innov_path": str(M06 / "innovations" / "orig" / r["file"])}
        for _, r in rep.iterrows()
        if r["attacked"] and split.get(r["flight_id"]) == "validation"
    ]  # fmt: skip
    man = pd.read_csv(MANIFEST)
    man = man[(man["split"] == "validation") & (man["attack_set"] == "main")]
    rows += [
        {"flight_id": r["attacked_flight_id"], "mission_type": r["mission_type"],
         "attack_type": r["attack_type"],
         "innov_path": str(M06 / "innovations" / "syn" / r["file"])}
        for _, r in man.iterrows()
    ]  # fmt: skip
    return pd.DataFrame(rows)


def innovations(path):
    df = pd.read_parquet(path, columns=["t_s", "valid", *W_COLS, "label", "offset_m",
                                         "rel_start_s"])  # fmt: skip
    return {
        "t": df["t_s"].to_numpy(float), "valid": df["valid"].to_numpy(bool),
        "w": df[W_COLS].to_numpy(float), "label": df["label"].to_numpy(float),
        "offset": df["offset_m"].to_numpy(float),
        "rel": df["rel_start_s"].to_numpy(float),
    }  # fmt: skip


def consistency_channels(fid, base_path):
    """Course and velocity channels on the base-table clock, cached per flight."""
    CHAN_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAN_DIR / f"{fid}.npz"
    stamp = f"{os.stat(base_path).st_size}:{os.stat(base_path).st_mtime_ns}"
    if path.is_file():
        with np.load(path) as z:
            if str(z["source"]) == stamp:
                return {k: z[k] for k in ("t", "values")}
    f = build_flight_v2(pd.read_parquet(base_path))
    out = {"t": f["t_s"].to_numpy(float), "values": channel_values(f, CHANNELS)}
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, source=stamp, **out)
    os.replace(tmp, path)
    return out


# ------------------------------------------------------------- verdicts --


def verdict(alarm_t, sig):
    """Latched verdict from alarm times on the base-table clock (Module 10's rule)."""
    if not np.isfinite(sig["start_s"]):
        return {"detected": False, "delay_s": np.nan, "displacement_m": np.nan,
                "alarms": int(len(alarm_t))}  # fmt: skip
    inside = alarm_t[(alarm_t >= sig["start_s"]) & (alarm_t <= sig["end_s"])]
    hit = len(inside) > 0
    return {
        "detected": hit,
        "delay_s": float(inside[0] - sig["onset_s"]) if hit else np.nan,
        "displacement_m": float(np.interp(inside[0], sig["row_t"], sig["offset_m"]))
        if hit else np.nan,
        "alarms": int((alarm_t < sig["start_s"]).sum()),
    }  # fmt: skip


def cusum_verdict(idx, inn):
    """Latched verdict on the filter's own clock. The innovations carry the label and
    the seconds since the measured onset tick by tick, so window, delay and offset are
    all read from the same ticks and the clock offset cancels out."""
    t, on = inn["t"], inn["label"] == 1
    if not on.any():
        return {"detected": False, "delay_s": np.nan, "displacement_m": np.nan,
                "alarms": int(len(idx))}  # fmt: skip
    start, end = t[on][0], t[on][-1]
    inside = idx[(t[idx] >= start) & (t[idx] <= end)]
    hit = len(inside) > 0
    delay = np.nan
    if hit:  # real flights carry no seconds-since-onset: time from the window start
        rel = inn["rel"][inside[0]]
        delay = float(rel) if np.isfinite(rel) else float(t[inside[0]] - start)
    return {
        "detected": hit,
        "delay_s": delay,
        "displacement_m": float(inn["offset"][inside[0]]) if hit else np.nan,
        "alarms": int((t[idx] < start).sum()),
    }  # fmt: skip


def final_offset(sig):
    on = (sig["row_t"] >= sig["start_s"]) & (sig["row_t"] <= sig["end_s"])
    return float(np.nanmax(sig["offset_m"][on])) if on.any() else np.nan


# ---------------------------------------------------------- calibration --


class Calibration:
    """Thresholds of all five detectors fitted on one set of clean flights."""

    def __init__(self, normals, sigs, inns, chans, tune=None, k_fixed=None):
        rows = normals.to_dict("records")
        self.m10 = m10_calibrate([sigs[f] for f in normals["flight_id"]], rows)
        # CUSUM: standardiser and h per k on the clean ticks, as Module 06 does
        ws, ts, vs, off = [], [], [], 0.0
        for f in normals["flight_id"]:
            inn = inns[f]
            ok = inn["valid"] & (inn["label"] == 0)
            ws.append(inn["w"])
            vs.append(ok)
            ts.append(inn["t"] + off)
            off += float(inn["t"].max()) + 1000.0
        w, v, t = np.vstack(ws), np.concatenate(vs), np.concatenate(ts)
        self.mu, self.sd = fit_standardiser(w[v])
        hours = sum(monitored_seconds(inns[f]["t"], inns[f]["valid"])
                    for f in normals["flight_id"]) / 3600.0  # fmt: skip
        allowed = max(1, int(round(BUDGET_PER_HOUR * hours)))
        z = (w - self.mu) / self.sd
        self.h = {
            k: calibrate_threshold(
                cusum_alarm_counts(z, v, t, k, H_GRID, REFRACTORY_S), H_GRID, allowed
            )
            for k in K_GRID
        }
        self.cusum_hours, self.cusum_allowed = hours, allowed
        self.k = k_fixed if k_fixed is not None else self.choose_k(tune, inns)
        # consistency: Module 09's standardiser and threshold
        vals = np.concatenate([chans[f]["values"] for f in normals["flight_id"]])
        self.c_mu, self.c_sd = consistency_std(vals)
        clean = []
        for f in normals["flight_id"]:
            s, live = score_rows(chans[f]["values"], self.c_mu, self.c_sd)
            clean.append((s, chans[f]["t"], live))
        self.c_thr, self.c_hours, self.c_allowed, _ = consistency_calibrate(
            clean, BUDGET_PER_HOUR, REFRACTORY_S
        )

    def cusum_idx(self, inn, k=None):
        k = self.k if k is None else k
        z = (inn["w"] - self.mu) / self.sd
        return cusum_alarms(z, inn["valid"], inn["t"], k, self.h[k], REFRACTORY_S)

    def choose_k(self, tune, inns):
        """Module 06's rule: highest detection rate on the tuning attacks, then the
        shortest median delay."""
        best, key = None, None
        for k in K_GRID:
            res = [cusum_verdict(self.cusum_idx(inns[f], k), inns[f])
                   for f in tune["flight_id"]]  # fmt: skip
            det = np.mean([r["detected"] for r in res])
            delays = [r["delay_s"] for r in res if r["detected"]]
            cand = (det, -np.median(delays) if delays else -np.inf)
            if key is None or cand > key:
                best, key = k, cand
        return best

    def record(self):
        return {"kalman_threshold": self.m10["kalman"],
                "velocity_threshold": self.m10["velocity"],
                "fused_kalman": self.m10["fused_kalman"],
                "fused_velocity": self.m10["fused_velocity"],
                "cusum_k": self.k, "cusum_h": self.h[self.k],
                "consistency_threshold": self.c_thr}  # fmt: skip


def score_flight(row, cal, sigs, inns, chans):
    """All five detectors on one flight: verdict plus monitored hours."""
    fid, sig = row["flight_id"], sigs[row["flight_id"]]
    out = []
    for det in M10_DETECTORS:
        out.append({"detector": det, "hours": hours_of(row, det),
                    **verdict(m10_alarms(sig, det, cal.m10), sig)})  # fmt: skip
    inn = inns[fid]
    out.append({"detector": "cusum",
                "hours": monitored_seconds(inn["t"], inn["valid"]) / 3600.0,
                **cusum_verdict(cal.cusum_idx(inn), inn)})  # fmt: skip
    ch = chans[fid]
    s, live = score_rows(ch["values"], cal.c_mu, cal.c_sd)
    idx = alarm_indices(s, ch["t"], live, cal.c_thr, REFRACTORY_S)
    out.append({"detector": "consistency", "hours": monitored_hours(live, ch["t"]),
                **verdict(ch["t"][idx], sig)})  # fmt: skip
    fo = final_offset(sig)
    for o in out:
        o.update({"flight_id": fid, "final_offset_m": fo})
    return out


# ------------------------------------------------------------- protocols --


def run():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = universe()
    tune = tuning_flights()
    print("=" * 78)
    print(
        "AEROVIGIL - PLAN M9: PHYSICS DETECTORS (seen / unseen missions / types / CV)"
    )
    print("=" * 78)
    sigs = {
        r["flight_id"]: load(M10 / "signals" / r["file"]) for _, r in idx.iterrows()
    }
    inns = {r["flight_id"]: innovations(r["innov_path"]) for _, r in idx.iterrows()}
    for _, r in tune.iterrows():
        inns.setdefault(r["flight_id"], innovations(r["innov_path"]))
    print(f"loaded signals and innovations   [{time.time() - started:.0f} s]")
    chans = {}
    for n, (_, r) in enumerate(idx.iterrows(), start=1):
        chans[r["flight_id"]] = consistency_channels(r["flight_id"], r["base_path"])
        if n % 200 == 0 or n == len(idx):
            print(
                f"  consistency channels {n}/{len(idx)}  [{time.time() - started:.0f} s]"
            )
    orig = idx[idx["set"] == "orig"]
    normals = orig[~orig["attacked"]]
    fit_normals = normals[normals["split"].isin(["train", "validation"])]
    test = idx[idx["split"] == "test"]
    rows, cals = [], []

    def score(protocol, held, cal, flights):
        cals.append({"protocol": protocol, "held": held, **cal.record()})
        for _, r in flights.iterrows():
            for o in score_flight(r, cal, sigs, inns, chans):
                rows.append({"protocol": protocol, "held": held,
                             "set": r["set"], "mission_type": r["mission_type"],
                             "attack_type": r["attack_type"],
                             "velocity_mode": r["velocity_mode"],
                             "drift_rate_mps": r["drift_rate_mps"],
                             "attacked": r["attacked"], **o})  # fmt: skip
        print(f"  {protocol:8s} {str(held):20s} k={cal.k:g}   "
              f"[{time.time() - started:.0f} s]")  # fmt: skip

    seen = Calibration(fit_normals, sigs, inns, chans, tune)
    score("seen", None, seen, test)
    for m in MISSIONS:
        keep = fit_normals["mission_type"] != m
        cal = Calibration(fit_normals[keep], sigs, inns, chans,
                          tune[tune["mission_type"] != m])  # fmt: skip
        score("mission", m, cal, idx[(idx["mission_type"] == m)])
    for fam in FAMILIES:
        k = seen.choose_k(tune[tune["attack_type"] != fam], inns)
        cal = seen if k == seen.k else Calibration(
            fit_normals, sigs, inns, chans, k_fixed=k)  # fmt: skip
        fl = test[(test["attack_type"] == fam) | ~test["attacked"]]
        score("type", fam, cal, fl)
    fold = folds_of(normals)
    for k in range(6):
        out = normals["flight_id"].map(fold) == k
        cal = Calibration(normals[~out], sigs, inns, chans, k_fixed=seen.k)
        score("cvfa", str(k), cal, normals[out])
    flights = pd.DataFrame(rows)
    flights.to_csv(OUT / "physics_flights.csv", index=False)
    pd.DataFrame(cals).to_csv(OUT / "physics_calibration.csv", index=False)
    table = summarise(flights)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    report(table, time.time() - started)
    return table


GROUP = {"seen": "seen", "mission": "unseen missions", "type": "unseen attack types",
         "cvfa": "cross-validated false alarms"}  # fmt: skip


def summarise(flights):
    rows = []
    for (proto, det), g in flights.groupby(["protocol", "detector"], sort=False):
        main = g[g["set"] != "sweep"]
        real = main[main["attacked"] & (main["set"] == "orig")]
        syn = main[main["attacked"] & (main["set"] == "main")]
        cv = syn[syn["velocity_mode"] == "consistent"]
        nor = main[~main["attacked"]]
        att = main[main["attacked"]]
        lo, hi = bootstrap_ci(att["detected"].astype(float))
        moved = np.where(att["detected"], att["displacement_m"], att["final_offset_m"])
        rows.append({
            "protocol": GROUP[proto], "detector": det,
            "attacked_flights": len(att),
            "detected": att["detected"].mean() if len(att) else np.nan,
            "detected_ci_low": lo, "detected_ci_high": hi,
            "real_detected": real["detected"].mean() if len(real) else np.nan,
            "real_delay_s": real.loc[real["detected"].astype(bool), "delay_s"].median(),
            "syn_detected": syn["detected"].mean() if len(syn) else np.nan,
            "consistent_detected": cv["detected"].mean() if len(cv) else np.nan,
            "median_delay_s": att.loc[att["detected"].astype(bool), "delay_s"].median(),
            "median_undetected_displacement_m": float(np.nanmedian(moved))
            if len(att) else np.nan,
            "normal_flights": len(nor),
            **false_alarms(nor["alarms"], nor["hours"]),
        })  # fmt: skip
    return pd.DataFrame(rows)


def report(table, elapsed):
    pd.set_option("display.width", 250)
    order = {"unseen missions": 0, "unseen attack types": 1, "seen": 2,
             "cross-validated false alarms": 3}  # fmt: skip
    t = table.sort_values(["protocol", "detector"],
                          key=lambda s: s.map(order).fillna(s) if s.name == "protocol"
                          else s)  # fmt: skip
    print(t[["protocol", "detector", "attacked_flights", "detected", "detected_ci_low",
             "detected_ci_high", "real_detected", "consistent_detected",
             "median_delay_s", "fa_alarms", "fa_hours",
             "fa_per_hour"]].round(3).to_string(index=False))  # fmt: skip
    print(f"\nElapsed {elapsed:.0f} s   saved to {OUT} and {TABLE}")


def main():
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
