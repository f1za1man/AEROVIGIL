r"""Plan M13 (Step H): precompute everything the demo dashboard shows.

The dashboard (app/app.py) runs fully offline and never trains, calibrates or runs a
detector itself: every scenario it can show is computed here, once, with the same
detectors, thresholds and explanations as Steps C-E (the Step D detector objects at
the Step C seen-protocol calibration, seed 42 for the learned models).

Scenarios
    flights     2 normal test flights per mission (curved, random, straight)
    attacks     linear drift, exponential ramp, constant offset (ramp then hold) and
                abrupt jump, each position-only and velocity-consistent, injected
                with Module 04's own injector (warp onset) from a quarter of the way
                into the flight to its end, heading 45 deg (north-east)
    strength    the "out-drift the detector" slider:
                  drift speed 0.05, 0.1, 0.25, 0.5, 1, 2, 5 m/s (linear drift; the
                  starting speed of an exponential ramp with a 20 s time constant;
                  the speed at which a constant offset ramps up to 50 m)
                  jump size 1, 2, 5, 10, 20, 50, 100 m (abrupt jump)
    also        each clean flight with no attack, and 1 real attacked test flight per
                mission (the dataset's own attack)

For each scenario and detector: every alarm time, Module 10's verdict (caught or not,
delay from the onset, how far the drone had been moved), the detector's signal over
time with its threshold, and Step E's three reasons and operator message for the
first alarm.

Writes data/processed/demo/scenarios.parquet, results.parquet and series/<id>.parquet
(one small file per scenario). Resumable: finished scenarios are skipped.

Usage:
    python src\aerovigil\demo\build.py          (~20-30 min the first time)
    python src\aerovigil\demo\build.py --limit 3
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from aerovigil.attacks.inject_v2 import inject
from aerovigil.consistency.detector import score_rows
from aerovigil.data.build_raw_base import METRES_PER_DEGREE
from aerovigil.evaluation import physics_protocols as pp
from aerovigil.explain import reasons as R
from aerovigil.explain.attribution import Explainer
from aerovigil.explain.run import ae_clean_error
from aerovigil.fusion.fuse import velocity_z
from aerovigil.kalman.plot_filter import cusum_path
from aerovigil.stealth import detectors as D

OUT = D.ROOT / "data" / "processed" / "demo"
SERIES = OUT / "series"
PARTS = OUT / "parts"
FLIGHTS_PER_MISSION = 2
RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
JUMPS_M = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
TYPES = ("linear_drift", "exponential_ramp", "constant_offset", "abrupt_jump")
MODES = ("position_only", "consistent")
DIRECTION_DEG = 45.0
START_FRACTION = 0.25
EXP_TIME_CONSTANT_S, EXP_CAP_M, CONST_OFFSET_M = 20.0, 500.0, 50.0
HZ = 5.0  # plotted signals: largest value in every 0.2 s
DETECTORS = ("kalman", "velocity", "fused", "cusum", "consistency", "random_forest",
             "lstm_ae", "lstm_supervised")  # fmt: skip
LABELS = {"kalman": "Kalman chi-square", "velocity": "Velocity check",
          "fused": "Fused (Kalman + velocity)", "cusum": "CUSUM",
          "consistency": "Consistency (velocity + course)",
          "random_forest": "Random forest", "lstm_ae": "LSTM autoencoder",
          "lstm_supervised": "Supervised LSTM"}  # fmt: skip
TYPE_LABELS = {"none": "No attack", "real": "Real attack (dataset)",
               "linear_drift": "Linear drift", "exponential_ramp": "Exponential ramp",
               "constant_offset": "Constant offset (ramp, then hold)",
               "abrupt_jump": "Abrupt jump"}  # fmt: skip


def params_for(typ, strength):
    p = {"direction_deg": DIRECTION_DEG}
    if typ == "linear_drift":
        p["drift_rate_mps"] = strength
    elif typ == "exponential_ramp":
        p.update(initial_rate_mps=strength, time_constant_s=EXP_TIME_CONSTANT_S,
                 max_offset_m=EXP_CAP_M)  # fmt: skip
    elif typ == "constant_offset":
        p.update(magnitude_m=CONST_OFFSET_M, ramp_seconds=CONST_OFFSET_M / strength)
    elif typ == "abrupt_jump":
        p["magnitude_m"] = strength
    else:
        raise ValueError(typ)
    return p


def strengths(typ):
    return JUMPS_M if typ == "abrupt_jump" else RATES


def scenario_id(flight_no, typ, mode="na", strength=None):
    s = "" if strength is None else f"_{strength:g}".replace(".", "p")
    return f"f{flight_no}_{typ}_{mode}{s}"


# ----------------------------------------------------------------- signals --


def enu(lat, lon, lat0, lon0):
    return ((lon - lon0) * METRES_PER_DEGREE * np.cos(np.radians(lat0)),
            (lat - lat0) * METRES_PER_DEGREE)  # fmt: skip


def thin(t, v, hz=HZ):
    """Largest value in every 1/hz seconds, so short spikes stay visible."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    f = pd.DataFrame({"b": np.floor(t[ok] * hz), "t": t[ok], "v": v[ok]})
    g = f.loc[f.groupby("b")["v"].idxmax()]
    return g["t"].to_numpy(), g["v"].to_numpy()


def signal(det, fl):
    """(t, value, threshold, what) of one detector's alarm signal on one flight."""
    name = det.name
    if name in ("kalman", "velocity", "fused"):
        sig, th = fl.sig, det.cal.m10
        kt, kv = sig["kf_t"][sig["kf_valid"]], sig["kf_nis"][sig["kf_valid"]]
        vz = velocity_z(sig, th["mu"], th["sd"])
        vt, vv = sig["row_t"][sig["vel_valid"]], vz[sig["vel_valid"]]
        if name == "kalman":
            return kt, kv, th["kalman"], "horizontal NIS"
        if name == "velocity":
            return vt, vv, th["velocity"], "|z| of the speed residual"
        t = np.r_[kt, vt]
        v = np.r_[kv / th["fused_kalman"], vv / th["fused_velocity"]]
        order = np.argsort(t, kind="stable")
        return t[order], v[order], 1.0, "larger channel / its threshold"
    if name == "cusum":
        inn, cal = fl.innovations, det.cal
        if inn is None:
            return np.zeros(0), np.zeros(0), cal.h[cal.k], "CUSUM sum"
        z = (inn["w"] - cal.mu) / cal.sd
        stat, _ = cusum_path(z, inn["valid"], inn["t"], cal.k, cal.h[cal.k],
                             D.REFRACTORY_S)  # fmt: skip
        return inn["row_t"], stat, cal.h[cal.k], "CUSUM sum"
    if name == "consistency":
        ch, cal = fl.channels, det.cal
        s, live = score_rows(ch["values"], cal.c_mu, cal.c_sd)
        return ch["t"][live], s[live], cal.c_thr, "largest channel |z|"
    ends, s = det.window_scores(fl)
    what = {"random_forest": "attack probability (5 s window)",
            "lstm_ae": "reconstruction error (5 s window)",
            "lstm_supervised": "attack logit (5 s window)"}[name]  # fmt: skip
    return np.asarray(ends, float), np.asarray(s, float), det.threshold, what


# -------------------------------------------------------------- scenarios --


def first_explained(fl, alarms):
    """The alarm to explain: the first one inside the attack, else the first one."""
    if not len(alarms):
        return None, ""
    if np.isfinite(fl.start):
        inside = alarms[(alarms >= fl.start) & (alarms <= fl.end)]
        if len(inside):
            return float(inside[0]), "first alarm during the attack"
        return float(alarms[0]), "false alarm (outside the attack)"
    return float(alarms[0]), "false alarm (no attack)"


def score_scenario(sid, meta, attacked, truth_latlon, dets, explainer):
    first = next(iter(dets.values()))
    fl = first.flight(attacked)
    lat0 = float(truth_latlon[0][0])
    lon0 = float(truth_latlon[1][0])
    t = attacked["t_s"].to_numpy(float)
    keep = np.unique(np.searchsorted(t, np.arange(t[0], t[-1], 1.0 / HZ)))
    te, tn = enu(truth_latlon[0][keep], truth_latlon[1][keep], lat0, lon0)
    se, sn = enu(attacked["latitude_deg"].to_numpy(float)[keep],
                 attacked["longitude_deg"].to_numpy(float)[keep], lat0, lon0)  # fmt: skip
    parts = [pd.DataFrame({"kind": "path", "t": t[keep], "true_e": te, "true_n": tn,
                           "spoof_e": se, "spoof_n": sn,
                           "offset_m": attacked["offset_m"].to_numpy(float)[keep]})]  # fmt: skip
    rows = []
    for (name, _), det in dets.items():
        alarms = np.sort(np.asarray(det.alarm_times(fl), float))
        v = fl.verdict(alarms)
        st, sv, thr, what = signal(det, fl)
        tt, vv = thin(st, sv)
        parts.append(pd.DataFrame({"kind": name, "t": tt, "value": vv}))
        at, why = first_explained(fl, alarms)
        ex = explainer.explain(det, fl, at) if at is not None else {}
        rows.append({
            "scenario": sid, "detector": name, "label": LABELS[name],
            "threshold": float(thr), "signal": what,
            "detected": bool(v["detected"]), "first_alarm_s": v["first_alarm_s"],
            "delay_s": v["delay_s"], "displacement_m": v["displacement_m"],
            "n_alarms": int(len(alarms)),
            "alarms_before_onset": int(v["alarms_before_onset"]),
            "alarm_times": json.dumps([round(float(a), 3) for a in alarms]),
            "explained_t": at, "explained_why": why,
            "message": ex.get("message", ""),
            "reasons": json.dumps([
                {"check": ex[f"reason{i}"], "source": ex[f"reason{i}_source"],
                 "value": ex[f"reason{i}_value"], "normal": ex[f"reason{i}_normal"],
                 "unit": R.unit_of(ex[f"reason{i}"]),
                 "text": R.text_of(ex[f"reason{i}"])}
                for i in (1, 2, 3) if f"reason{i}" in ex
            ], default=float),
        })  # fmt: skip
    series = pd.concat(parts, ignore_index=True)
    meta = {**meta, "scenario": sid,
            "attack_start_s": fl.start, "attack_end_s": fl.end,
            "onset_s": fl.onset, "final_offset_m": fl.final,
            "duration_s": float(t[-1] - t[0])}  # fmt: skip
    D.atomic_write(SERIES / f"{sid}.parquet", lambda p: series.to_parquet(p))
    D.atomic_write(PARTS / f"{sid}.json", lambda p: p.write_text(
        json.dumps({"meta": meta, "results": rows}, default=_jsonable)))  # fmt: skip


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def pick_flights():
    idx = pp.universe()
    test = idx[(idx["set"] == "orig") & (idx["split"] == "test")]
    normal = (test[~test["attacked"]].sort_values("flight_id")
              .groupby("mission_type").head(FLIGHTS_PER_MISSION))  # fmt: skip
    real = (test[test["attacked"]].sort_values("flight_id")
            .groupby("mission_type").head(1))  # fmt: skip
    return normal.reset_index(drop=True), real.reset_index(drop=True)


def jobs(limit=None):
    normal, real = pick_flights()
    out = []
    for i, r in normal.iterrows():
        info = {"flight_no": i, "flight_id": r["flight_id"],
                "mission_type": r["mission_type"], "base_path": r["base_path"]}  # fmt: skip
        out.append((scenario_id(i, "none"), {**info, "attack_type": "none",
                                             "velocity_mode": "na",
                                             "strength": np.nan}))  # fmt: skip
        for typ in TYPES:
            for mode in MODES:
                for s in strengths(typ):
                    out.append((scenario_id(i, typ, mode, s),
                                {**info, "attack_type": typ, "velocity_mode": mode,
                                 "strength": s}))  # fmt: skip
    for j, r in real.iterrows():
        i = len(normal) + j
        out.append((scenario_id(i, "real"), {
            "flight_no": i, "flight_id": r["flight_id"],
            "mission_type": r["mission_type"], "base_path": r["base_path"],
            "attack_type": "real", "velocity_mode": "na", "strength": np.nan}))  # fmt: skip
    return out[:limit] if limit else out


def build(limit=None):
    started = time.time()
    SERIES.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    todo = [(sid, m) for sid, m in jobs(limit)
            if not ((SERIES / f"{sid}.parquet").is_file()
                    and (PARTS / f"{sid}.json").is_file())]  # fmt: skip
    print(f"Demo scenarios: {len(jobs(limit))}, to build: {len(todo)}")
    if todo:
        dets = D.build(names=DETECTORS, seeds=(42,))
        dets = {k: dets[k] for k in sorted(dets, key=lambda k: DETECTORS.index(k[0]))}
        explainer = Explainer(R.build_normals(), ae_clean_error())
        cache = {}
        for n, (sid, meta) in enumerate(todo, start=1):
            if meta["base_path"] not in cache:
                cache.clear()
                cache[meta["base_path"]] = pd.read_parquet(meta["base_path"])
            g = cache[meta["base_path"]]
            typ = meta["attack_type"]
            if typ in ("none", "real"):
                attacked = g
                if typ == "real":
                    truth = (g["raw_latitude_deg"].to_numpy(float),
                             g["raw_longitude_deg"].to_numpy(float))  # fmt: skip
                else:
                    truth = (g["latitude_deg"].to_numpy(float),
                             g["longitude_deg"].to_numpy(float))  # fmt: skip
            else:
                dur = float(g["t_s"].iloc[-1])
                spec = {"attack_type": typ, "start_s": dur * START_FRACTION,
                        "duration_s": dur * (1 - START_FRACTION),
                        "params": params_for(typ, meta["strength"])}  # fmt: skip
                attacked, _ = inject(g, spec, meta["velocity_mode"], sid)
                truth = (g["latitude_deg"].to_numpy(float),
                         g["longitude_deg"].to_numpy(float))  # fmt: skip
            score_scenario(sid, meta, attacked, truth, dets, explainer)
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - started) / n
                print(f"  {n}/{len(todo)}  [{time.time() - started:.0f} s, "
                      f"~{rate * (len(todo) - n) / 60:.0f} min left]", flush=True)  # fmt: skip
    consolidate()
    print(f"Done in {(time.time() - started) / 60:.1f} min -> {OUT}")


def consolidate():
    metas, results = [], []
    for p in sorted(PARTS.glob("*.json")):
        d = json.loads(p.read_text())
        metas.append(d["meta"])
        results += d["results"]
    scen = pd.DataFrame(metas).drop(columns=["base_path"])
    scen["type_label"] = scen["attack_type"].map(TYPE_LABELS)
    D.atomic_write(OUT / "scenarios.parquet", lambda p: scen.to_parquet(p))
    D.atomic_write(OUT / "results.parquet",
                   lambda p: pd.DataFrame(results).to_parquet(p))  # fmt: skip
    return scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    build(ap.parse_args().limit)


if __name__ == "__main__":
    main()
