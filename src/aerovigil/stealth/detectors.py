r"""Plan M8 (Step D): every detector of the project as one kind of object.

The Stealth Frontier compares seven detectors on the same attacks, so they have to be
scored the same way. Each detector here turns one flight into a list of alarm times on
the base-table clock (t_s), thinned to one per 10 s refractory period, and every
verdict comes from those times by Module 10's rule: the first alarm inside the attack's
label window counts as the detection, the delay is taken from the measured onset (the
first ramp row) and the displacement is the spoofed offset at that moment.

    kalman, velocity, fused   Module 10's channels at the seen-protocol thresholds
    cusum                     Module 06's CUSUM on the whitened innovations
    consistency               Module 09's velocity + course channels
    random_forest             Module 05's best classical model (Step B setting)
    lstm_ae, lstm_supervised  Module 07's LSTMs, the Step C seen-protocol weights

All thresholds are the Step C seen-protocol ones: 1 alarm per flight-hour on clean
training/validation flights, never on a tested flight. The physics calibration is
re-fitted with Step C's own code and checked against physics_calibration.csv; the
random forest is refitted with Step C's setting and seed and checked against its saved
threshold; the LSTM weights and thresholds are loaded from Step C's files.

A detector is also an attacker's oracle (Module 08's contract): called with a clean
flight and an attack profile it injects the attack exactly as Module 08 and Module 10
did, scores it, and returns only what a black-box attacker may learn.
"""

import io
import json
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aerovigil.adaptive.oracle import KalmanOracle
from aerovigil.attacks.features_v2_m04 import PARAM_FILE
from aerovigil.baselines import all_models_v2 as am
from aerovigil.baselines.build_window_tables_v2 import OUT_DIR as M05_DIR
from aerovigil.baselines.build_window_tables_v2 import flight_windows
from aerovigil.consistency.detector import (
    CHANNELS,
    alarm_indices,
    channel_values,
    score_rows,
)
from aerovigil.data.build_raw_base import PHASE_RAMP
from aerovigil.evaluation import lstm_protocols as lp
from aerovigil.evaluation import physics_protocols as pp
from aerovigil.evaluation import store
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.fusion.fuse import alarms as m10_alarms
from aerovigil.fusion.fuse import refractory
from aerovigil.fusion.signals import VELOCITY, Context, truth_of
from aerovigil.kalman.detectors import W_COLS
from aerovigil.kalman.pipeline import run_flight
from aerovigil.lstm.models import build_model, predict
from aerovigil.lstm.sequences_v2 import flight_sequences
from aerovigil.preprocessing.normalize_features_v2 import CLIP
from aerovigil.tracking import log_run

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "processed" / "stealth"
MODELS = OUT / "models"
REFRACTORY_S = 10.0
SEEDS = (42, 43, 44, 45, 46)
PHYSICS = ("kalman", "velocity", "fused", "cusum", "consistency")
LEARNED = ("random_forest", "lstm_ae", "lstm_supervised")
ALL = PHYSICS + LEARNED
FAILED = {
    "detected": True, "first_alarm_s": np.nan, "delay_s": np.nan,
    "displacement_m": 0.0, "final_offset_m": 0.0, "score": 0.0,
    "attack_duration_s": np.nan, "achieved_mean_mps": 0.0,
    "n_alarms": 0, "alarms_before_onset": 0, "after_end_s": np.nan, "ok": False,
}  # fmt: skip


def atomic_write(path, write):
    """Write beside the target and rename, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    write(tmp)
    tmp.replace(path)


def save_csv(frame, path):
    """Write a table unless the file already holds the same values (to 1e-12), so a
    rebuild from saved results does not churn files with float round-off."""
    path = Path(path)
    if path.is_file():
        old = pd.read_csv(path)
        new = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        if list(old.columns) == list(new.columns) and old.shape == new.shape:
            num = new.select_dtypes("number").columns
            same_num = all(
                old[c].dtype.kind in "fiub"
                and np.allclose(old[c].to_numpy(float), new[c].to_numpy(float),
                                rtol=1e-12, atol=1e-12, equal_nan=True)
                for c in num
            )  # fmt: skip
            rest = [c for c in new.columns if c not in num]
            if same_num and old[rest].astype(str).equals(new[rest].astype(str)):
                return False
    atomic_write(path, lambda p: frame.to_csv(p, index=False))
    return True


# ------------------------------------------------------------------ flight --


class Norm:
    """Module 03 v2's normalisation, read quietly from its parameter file."""

    def __init__(self):
        p = json.loads(PARAM_FILE.read_text(encoding="utf-8"))
        self.features = list(p["features"])
        self.median = np.array([p["median"][f] for f in self.features], float)
        self.scale = np.array([p["scale"][f] for f in self.features], float)

    def __call__(self, f):
        got = [c for c in self.features if c in f.columns]
        if got != self.features:
            raise SystemExit("feature columns differ from the Module 03 v2 parameters")
        out = f.copy()
        z = (out[self.features].to_numpy(np.float64) - self.median) / self.scale
        out[self.features] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        return out


class Flight:
    """One (possibly attacked) base-table flight; each view is built once, on demand."""

    def __init__(self, g, ctx, norm):
        self.g, self.ctx, self.norm = g, ctx, norm
        t = g["t_s"].to_numpy(float)
        on = g["attack_label"].to_numpy(float) == 1
        self.row_t = t
        self.offset = g["offset_m"].to_numpy(float)
        if on.any():
            ramp = g["phase"].to_numpy() == PHASE_RAMP
            self.start, self.end = float(t[on][0]), float(t[on][-1])
            self.onset = float(t[ramp][0]) if ramp.any() else self.start
            self.final = (
                float(np.nanmax(self.offset[on]))
                if np.isfinite(self.offset[on]).any()
                else 0.0
            )
        else:
            self.start = self.end = self.onset = np.nan
            self.final = np.nan

    @cached_property
    def features(self):
        f = build_flight_v2(self.g)
        if len(f) != len(self.g):
            raise SystemExit("build_flight_v2 changed the row count")
        return f

    @cached_property
    def filter(self):
        """Module 06's filter ticks, with their base-table times (row_t_s)."""
        c = self.ctx
        df = run_flight(self.g, c.cols, c.fcfg, c.conv, c.scale, c.noise,
                        truth_of(self.g))  # fmt: skip
        if df is None or not len(df):
            return None
        return df

    @cached_property
    def sig(self):
        """Module 10's signals, built as fusion.signals.flight_signals builds them."""
        df = self.filter
        if df is None:
            kf_t = kf_nis = np.zeros(0)
            kf_valid = np.zeros(0, bool)
        else:
            kf_t = df["row_t_s"].to_numpy(float)
            kf_nis = df["nis_h"].to_numpy(float)
            kf_valid = df["valid"].to_numpy(bool) & np.isfinite(kf_nis)
        vel = self.features[VELOCITY].to_numpy(float)
        return {"kf_t": kf_t, "kf_nis": kf_nis, "kf_valid": kf_valid,
                "row_t": self.row_t, "vel": vel, "vel_valid": np.isfinite(vel),
                "offset_m": self.offset, "start_s": self.start, "end_s": self.end,
                "onset_s": self.onset}  # fmt: skip

    @cached_property
    def innovations(self):
        df = self.filter
        if df is None:
            return None
        return {
            "t": df["t_s"].to_numpy(float),
            "row_t": df["row_t_s"].to_numpy(float),
            "valid": df["valid"].to_numpy(bool),
            "w": df[W_COLS].to_numpy(float),
        }

    @cached_property
    def channels(self):
        f = self.features
        return {"t": f["t_s"].to_numpy(float), "values": channel_values(f, CHANNELS)}

    @cached_property
    def normalized(self):
        return self.norm(self.features)

    @cached_property
    def windows(self):
        w, _ = flight_windows(self.normalized, self.norm.features, {})
        return w if w is not None else pd.DataFrame()

    @cached_property
    def sequences(self):
        x, m, _ = flight_sequences(self.normalized, self.norm.features,
                                   {"attack_set": "probe"}, False)  # fmt: skip
        return x, m

    def verdict(self, alarm_t):
        """Module 10's latched verdict, plus what Module 08's attacker is told."""
        alarm_t = np.asarray(alarm_t, float)
        if not np.isfinite(self.start):  # a clean flight: only its alarms count
            return dict(FAILED, detected=False, n_alarms=int(len(alarm_t)),
                        displacement_m=np.nan, final_offset_m=np.nan,
                        score=np.nan, ok=True)  # fmt: skip
        inside = alarm_t[(alarm_t >= self.start) & (alarm_t <= self.end)]
        after = alarm_t[alarm_t > self.end]
        hit = len(inside) > 0
        disp = float(np.interp(inside[0], self.row_t, self.offset)) if hit else np.nan
        window = max(self.end - self.start, 1e-9)
        return {
            "detected": bool(hit),
            "first_alarm_s": float(inside[0]) if hit else np.nan,
            "delay_s": float(inside[0] - self.onset) if hit else np.nan,
            "displacement_m": disp if hit else self.final,
            "final_offset_m": self.final,
            "score": disp if hit else self.final,
            "attack_duration_s": float(window),
            "achieved_mean_mps": float(self.final / window),
            "n_alarms": int(len(alarm_t)),
            "alarms_before_onset": int((alarm_t < self.start).sum()),
            # Step C's window latch also credited an alarm raised after the spoofing
            # had stopped, by a window that still overlapped it (the self-test needs it)
            "after_end_s": float(after[0] - self.end) if len(after) else np.nan,
            "ok": True,
        }


# --------------------------------------------------------------- detectors --


class Detector:
    """Alarm times on the base clock -> verdict; also Module 08's oracle contract."""

    name = "detector"
    refractory_s = REFRACTORY_S
    attacker = None  # shared KalmanOracle: Module 08's injection path, unchanged

    def __init__(self, ctx=None, norm=None):
        self.ctx = ctx or Context()
        self.norm = norm or Norm()
        self.calls = 0
        if Detector.attacker is None:
            Detector.attacker = KalmanOracle()

    def flight(self, g):
        return Flight(g, self.ctx, self.norm)

    def alarm_times(self, fl):
        raise NotImplementedError

    def verdict(self, fl):
        return fl.verdict(self.alarm_times(fl))

    def __call__(self, g, profile_fn=None, mode="consistent", attack_type="adaptive",
                 params=None):  # fmt: skip
        self.calls += 1
        attacked = self.attacker.attack(g, profile_fn, mode, attack_type, params)
        return self.verdict(self.flight(attacked))


class Physics(Detector):
    """One of Module 06 / 09 / 10's physics detectors at the seen calibration."""

    def __init__(self, name, cal, ctx=None, norm=None):
        super().__init__(ctx, norm)
        self.name, self.cal = name, cal
        rec = cal.record()
        self.threshold = {
            "kalman": rec["kalman_threshold"], "velocity": rec["velocity_threshold"],
            "fused": rec["fused_kalman"], "cusum": rec["cusum_h"],
            "consistency": rec["consistency_threshold"],
        }[name]  # fmt: skip

    def alarm_times(self, fl):
        if self.name in ("kalman", "velocity", "fused"):
            return m10_alarms(fl.sig, self.name, self.cal.m10)
        if self.name == "cusum":
            inn = fl.innovations
            if inn is None:
                return np.zeros(0)
            idx = self.cal.cusum_idx(inn)
            return inn["row_t"][idx] if len(idx) else np.zeros(0)
        ch = fl.channels
        s, live = score_rows(ch["values"], self.cal.c_mu, self.cal.c_sd)
        idx = alarm_indices(s, ch["t"], live, self.cal.c_thr, REFRACTORY_S)
        return ch["t"][idx]


class Windowed(Detector):
    """A model that scores windows: alarm at the end of every window above threshold."""

    def window_scores(self, fl):
        raise NotImplementedError

    def alarm_times(self, fl):
        ends, s = self.window_scores(fl)
        if not len(s):
            return np.zeros(0)
        order = np.argsort(ends, kind="stable")
        ends, s = np.asarray(ends, float)[order], np.asarray(s, float)[order]
        return refractory(ends[s > self.threshold], self.refractory_s)


class RandomForest(Windowed):
    """A Module 05 tree model on the window statistics (random forest by default)."""

    def __init__(self, model, threshold, cols, seed, ctx=None, norm=None,
                 name="random_forest"):  # fmt: skip
        super().__init__(ctx, norm)
        self.name = name
        self.model, self.threshold, self.cols, self.seed = model, threshold, cols, seed

    def window_scores(self, fl):
        w = fl.windows
        if not len(w):
            return np.zeros(0), np.zeros(0)
        return w["window_end_s"].to_numpy(float), am.score(
            self.name, self.model, w, self.cols
        )


class LSTM(Windowed):
    def __init__(self, kind, model, threshold, seed, ctx=None, norm=None):
        super().__init__(ctx, norm)
        self.name = "lstm_ae" if kind == "ae" else "lstm_supervised"
        self.kind, self.model, self.threshold, self.seed = kind, model, threshold, seed

    def window_scores(self, fl):
        x, m = fl.sequences
        if not len(x):
            return np.zeros(0), np.zeros(0)
        s, _ = predict(self.kind, self.model, x.astype(np.float32))
        return m["window_end_s"].to_numpy(float), np.asarray(s, float)


# ------------------------------------------------------------ calibration --


def physics_calibration():
    """Step C's seen-protocol calibration of the five physics detectors, re-fitted
    with Step C's own code and checked against what Step C saved."""
    idx = pp.universe()
    orig = idx[idx["set"] == "orig"]
    normals = orig[~orig["attacked"]]
    fit = normals[normals["split"].isin(["train", "validation"])]
    tune = pp.tuning_flights()
    sigs = {r["flight_id"]: pp.load(pp.M10 / "signals" / r["file"])
            for _, r in fit.iterrows()}  # fmt: skip
    inns = {r["flight_id"]: pp.innovations(r["innov_path"]) for _, r in fit.iterrows()}
    for _, r in tune.iterrows():
        inns.setdefault(r["flight_id"], pp.innovations(r["innov_path"]))
    chans = {r["flight_id"]: pp.consistency_channels(r["flight_id"], r["base_path"])
             for _, r in fit.iterrows()}  # fmt: skip
    cal = pp.Calibration(fit, sigs, inns, chans, tune)
    saved = pd.read_csv(pp.OUT / "physics_calibration.csv")
    saved = saved[saved["protocol"] == "seen"].iloc[0]
    for key, value in cal.record().items():
        if not np.isclose(float(saved[key]), float(value), rtol=1e-6):
            raise SystemExit(
                f"physics calibration differs from Step C: {key} {value} vs "
                f"{saved[key]}. Re-run physics_protocols.py first."
            )
    return cal


SHORT_NAME = {"random_forest": "rf", "decision_tree": "dt"}


def seen_params(name="random_forest"):
    t = pd.read_csv(M05_DIR / "all_models_tuning_full.csv")
    t = t[(t["protocol"] == "seen") & (t["model"] == name)]
    return json.loads(t.loc[t["val_auc"].idxmax(), "params"])


def saved_threshold(seed, name="random_forest"):
    w = pd.read_csv(M05_DIR / "all_models_window_full.csv")
    w = w[(w["protocol"] == "seen") & (w["model"] == name) & (w["seed"] == seed)]
    return float(w["threshold"].iloc[0]) if len(w) else np.nan


def fit_classical(name, seed, extra=None, tag="seen"):
    """Step B/C's seen tree model (random forest or decision tree) at one seed;
    `extra` adds training windows (the hardening set). The threshold is checked
    against Step C's. Cached under data/processed/stealth/models."""
    path = MODELS / f"{SHORT_NAME[name]}_{tag}_s{seed}.joblib"
    if path.is_file():
        d = joblib.load(path)
        return d["model"], d["threshold"], d["cols"]
    orig, syn = am.load(M05_DIR)
    cols = am.feature_columns(orig)
    tr, va, clean_va, _, _ = am.protocol_sets(orig, syn, None, "seen")
    if extra is not None:
        tr = pd.concat([tr, extra], ignore_index=True)
    params = seen_params(name)
    model = am.fit(name, params, tr, cols, seed)
    pool = am.clean_pool(name, params, tr, clean_va, model, cols, seed)
    thr, hours, allowed = am.threshold_for(pool)
    if tag == "seen":
        want = saved_threshold(seed, name)
        if np.isfinite(want) and not np.isclose(thr, want, rtol=1e-6):
            raise SystemExit(f"{name} seed {seed} threshold {thr} differs from "
                             f"Step C {want}")  # fmt: skip
    atomic_write(path, lambda p: joblib.dump({"model": model, "threshold": thr,
                                              "cols": cols, "pool_hours": hours,
                                              "allowed": allowed}, p))  # fmt: skip
    log_run("aerovigil-m8-hardening" if tag == "hard" else "aerovigil-m5-classical",
            f"{name} {tag} s{seed}", {"model": name, "tag": tag, "seed": seed, **params},
            {"threshold": thr, "calibration_hours": hours, "n_train": len(tr)},
            {"source": "run", "step": "D"})  # fmt: skip
    return model, thr, cols


def fit_random_forest(seed, extra=None, tag="seen"):
    return fit_classical("random_forest", seed, extra, tag)


def load_lstm(kind, seed, weights=None, threshold=None):
    """Step C's seen-protocol LSTM (or a hardened one, given its weights)."""
    name = f"{kind}_seen_s{seed}"
    job = {"kind": kind, "hidden": lp.BASE_SIZE[0], "layers": lp.BASE_SIZE[1]}
    model = build_model(kind, len(Norm().features), lp.params_for(job), seed)
    import torch

    state = torch.load(weights or lp.PDIR / "weights" / f"{name}.pt",
                       map_location="cpu")  # fmt: skip
    model.load_state_dict(state)
    model.eval()
    if threshold is None:
        f = store.read_csv(lp.PDIR / "flights.csv", usecols=["name", "threshold"])
        threshold = float(f.loc[f["name"] == name, "threshold"].iloc[0])
    return model, float(threshold)


def build(names=ALL, seeds=SEEDS, cal=None):
    """Every requested detector, keyed (name, seed); physics detectors have seed 0."""
    import torch

    torch.set_num_threads(1)
    ctx, norm = Context(), Norm()
    out = {}
    if any(n in PHYSICS for n in names):
        cal = cal or physics_calibration()
        for n in names:
            if n in PHYSICS:
                out[(n, 0)] = Physics(n, cal, ctx, norm)
    for seed in seeds:
        if "random_forest" in names:
            model, thr, cols = fit_random_forest(seed)
            out[("random_forest", seed)] = RandomForest(model, thr, cols, seed, ctx,
                                                        norm)  # fmt: skip
        for kind, n in (("ae", "lstm_ae"), ("clf", "lstm_supervised")):
            if n in names:
                model, thr = load_lstm(kind, seed)
                out[(n, seed)] = LSTM(kind, model, thr, seed, ctx, norm)
    return out
