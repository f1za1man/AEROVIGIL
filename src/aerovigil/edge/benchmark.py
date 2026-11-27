r"""Plan M11 (Step F): how small and fast is each detector, and what does int8 cost?

Everything is measured on this computer's CPU with ONE thread (PyTorch, ONNX Runtime,
scikit-learn and the BLAS libraries are all limited to one), one window at a time,
because a detector on board scores each new window as it arrives. No flight-computer
board is named or assumed: the numbers are this CPU's, and the run records which CPU
and which library versions produced them.

Learned detectors (the Step C seen-protocol models, all five seeds):

    lstm_ae, lstm_supervised   PyTorch -> ONNX (float32) -> ONNX dynamic int8. The int8
                               setting is chosen per model on the validation windows
                               (agreement of the alarm decisions with float32); the
                               test flights take no part in the choice. The int8 model
                               is scored twice: re-calibrated on the same clean
                               calibration windows at 1 alarm/h (what a deployment
                               would do), and with the float32 threshold kept.
    random_forest              scikit-learn -> ONNX (float32), at Step C's threshold.
                               int8 does not apply to a tree ensemble (onnx_models.py),
                               so smaller forests are trained and calibrated with Step
                               C's own code instead, to show what size buys.

Accuracy is scored by Step C's own functions (lstm_protocols.evaluate_job,
all_models_v2.evaluate_scores) on the same test windows and flights, and the
original-precision rows are checked against Step C's saved results before anything
is written: a mismatch stops the run.

Physics detectors (streaming.py): Module 10's velocity channel and Kalman channel run
one log row at a time. On every real test flight their signal and alarms are compared
with the batch code first, then every step is timed. The batch feature, window and
sequence builders that feed the learned models are timed per flight as well, since on
board they would run too.

Writes reports/tables/F_edge_models.csv     size, latency, accuracy per model and form
       reports/tables/F_int8_settings.csv   every int8 setting, validation and test
       reports/tables/F_compact_forests.csv smaller random forests
       reports/tables/F_streaming.csv       physics detectors: agreement and step time
       reports/tables/F_pipeline.csv        input building per window, batch code
       data/processed/edge/run_info.json    CPU, versions, settings
       data/processed/edge/models/          the ONNX files (rebuilt on every run)

Usage:
    python src\aerovigil\edge\benchmark.py              (~25 min)
    python src\aerovigil\edge\benchmark.py --quick      (smoke test: 1 seed, 4 flights)
"""

import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from aerovigil.baselines import all_models_v2 as am
from aerovigil.baselines.build_window_tables_v2 import OUT_DIR as M05
from aerovigil.edge import onnx_models as ox
from aerovigil.edge.streaming import KalmanCheck, VelocityCheck
from aerovigil.evaluation import lstm_protocols as lp
from aerovigil.evaluation import physics_protocols as pp
from aerovigil.evaluation.metrics import false_alarms
from aerovigil.features.build_features import GAP_FACTOR
from aerovigil.fusion.fuse import alarms as m10_alarms
from aerovigil.fusion.signals import VELOCITY, Context
from aerovigil.kalman.pipeline import segments
from aerovigil.lstm.build_sequences_v2 import OUT_DIR as M07
from aerovigil.lstm.models import predict
from aerovigil.lstm.train_v2 import load as load_sequences
from aerovigil.stealth import detectors as D

OUT = D.ROOT / "data" / "processed" / "edge"
MODELS = OUT / "models"
TABLES = D.ROOT / "reports" / "tables"
CALLS, WARMUP = 2000, 200
FORESTS = ((100, None), (50, None), (50, 12), (20, 12), (20, 8))  # trees, depth
LATENCY_SEED = 42
KINDS = {"ae": "lstm_ae", "clf": "lstm_supervised"}


def kb(path):
    return path.stat().st_size / 1024.0


def latency(fn, xs):
    """Median, 95th and 99th percentile milliseconds of fn on one window."""
    n = len(xs)
    for i in range(WARMUP):
        fn(xs[i % n][None])
    t = np.empty(CALLS)
    for i in range(CALLS):
        xi = xs[i % n][None]
        t0 = time.perf_counter_ns()
        fn(xi)
        t[i] = time.perf_counter_ns() - t0
    t /= 1e6
    return {"latency_ms_median": float(np.median(t)),
            "latency_ms_p95": float(np.percentile(t, 95)),
            "latency_ms_p99": float(np.percentile(t, 99))}  # fmt: skip


def one_thread():
    import torch

    torch.set_num_threads(1)
    return threadpool_limits(limits=1)


# ------------------------------------------------------------- scoring --


def per_seed(flights, windows, hours):
    """Step C's per-seed numbers from per-flight verdicts and window counts."""
    rows = []
    for seed, g in flights.groupby("seed"):
        if "attack_set" in g:
            g = g[g["attack_set"] != "sweep"]
        real = g[g["attacked"] & (g["origin"] == "orig")]
        syn = g[g["attacked"] & (g["origin"] == "syn")]
        nor = g[~g["attacked"]]
        w = windows[windows["seed"] == seed]
        tp, fp, fn, tn = (int(w[c].sum()) for c in ("tp", "fp", "fn", "tn"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "seed": seed,
            "real_detected": real["detected"].mean(),
            "real_delay_s": real.loc[real["detected"], "delay_s"].median(),
            "syn_detected": syn["detected"].mean(),
            "position_only_detected": syn.loc[
                syn["velocity_mode"] == "position_only", "detected"].mean(),
            "consistent_detected": syn.loc[
                syn["velocity_mode"] == "consistent", "detected"].mean(),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": float(w["roc_auc"].mean()),
            "threshold": float(w["threshold"].mean()),
            **false_alarms(nor["alarms"],
                           nor["flight_id"].map(hours).fillna(0.0)),
        })  # fmt: skip
    return pd.DataFrame(rows)


def mean_sd(per, keys):
    vals = [c for c in per.columns if c not in ("seed",)]
    out = dict(keys)
    out["seeds"] = int(per["seed"].nunique())
    for c in vals:
        out[c] = float(per[c].mean())
        out[c + "_sd"] = float(per[c].std()) if len(per) > 1 else np.nan
    return out


def check_same(mine, theirs, what, cols):
    for c in cols:
        a = np.asarray(mine[c], float)
        b = np.asarray(theirs[c], float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-12,
                                                 equal_nan=True):  # fmt: skip
            raise SystemExit(f"{what}: {c} differs from Step C ({a} vs {b})")


def change(ref, new, mask, thr_ref, thr_new):
    """How much a converted model moves the scores, and its alarm decisions."""
    a, b = ref[mask].astype(float), new[mask].astype(float)
    return {"max_score_change": float(np.max(np.abs(b - a))) if len(a) else np.nan,
            "median_rel_change": float(np.median(np.abs(b - a) /
                                                 (np.abs(a) + 1e-3))),
            "decision_agreement": float(np.mean((a > thr_ref) == (b > thr_new)))}  # fmt: skip


# --------------------------------------------------------------- LSTMs --


def lstm_part(seeds, hours):
    import torch

    x, meta = load_sequences(M07)
    x = x.astype(np.float32)
    clf_job = {"kind": "clf", "protocol": "seen", "held": None}
    _, val, _, _ = lp.masks(meta, clf_job, None)  # validation windows, both classes
    flights, windows, changes, sizes, models = [], [], [], {}, {}
    for kind in KINDS:
        for seed in seeds:
            job = {"name": f"{kind}_seen_s{seed}", "kind": kind, "protocol": "seen",
                   "held": None, "hidden": lp.BASE_SIZE[0],
                   "layers": lp.BASE_SIZE[1], "seed": seed}  # fmt: skip
            _, _, cal, test = lp.masks(meta, job, None)
            model, _ = D.load_lstm(kind, seed)
            models[(kind, seed)] = model
            torch.set_num_threads(2)
            native, _ = predict(kind, model, x)
            saved = np.load(lp.PDIR / "scores" / f"{job['name']}.npy")
            if np.max(np.abs(native - saved)) > 1e-4:
                raise SystemExit(f"{job['name']}: PyTorch scores differ from Step C")
            fp32 = ox.export_lstm(kind, model, MODELS / f"{kind}_s{seed}.onnx",
                                  x[:1])  # fmt: skip
            pt = lp.PDIR / "weights" / f"{job['name']}.pt"
            sizes[(kind, seed, "pytorch")] = kb(pt)
            sizes[(kind, seed, "onnx_fp32")] = kb(fp32)
            scores = {"pytorch": native,
                      "onnx_fp32": ox.score_all(ox.session(fp32, 2), x)}  # fmt: skip
            for s in ox.INT8_SETTINGS:
                q = MODELS / f"{kind}_s{seed}_int8_{s}.onnx"
                err = ox.quantize(fp32, q, s)
                if err:
                    raise SystemExit(f"int8 {s} failed on {job['name']}: {err}")
                sizes[(kind, seed, f"int8:{s}")] = kb(q)
                scores[f"int8:{s}"] = ox.score_all(ox.session(q, 2), x)
            thr_ref = lp.threshold_for(native[cal])[0]
            for name, s in scores.items():
                forms = [(name, None)]
                if name.startswith("int8:"):
                    forms.append((name + "|fp32_threshold", thr_ref))
                for form, fixed in forms:
                    f, w = lp.evaluate_job(job, s, meta, None, hours, fixed)
                    flights.append(f.assign(form=form))
                    windows.append({**w, "form": form})
                    thr = w["threshold"]
                    changes.append({"kind": kind, "seed": seed, "form": form,
                                    "size_kb": sizes[(kind, seed, name)],
                                    **{f"val_{k}": v for k, v in change(
                                        native, s, val, thr_ref, thr).items()},
                                    **{f"test_{k}": v for k, v in change(
                                        native, s, test, thr_ref, thr).items()}})  # fmt: skip
            print(f"  {job['name']}: {len(scores)} forms scored", flush=True)
    flights = pd.concat(flights, ignore_index=True)
    windows = pd.DataFrame(windows)
    changes = pd.DataFrame(changes)

    # the original-precision rows must be Step C's
    saved_w = pd.read_csv(lp.PDIR / "windows.csv")
    for kind in KINDS:
        mine = windows[(windows["form"] == "pytorch") & (windows["kind"] == kind)]
        theirs = saved_w[(saved_w["protocol"] == "seen") & (saved_w["kind"] == kind)
                         & (saved_w["seed"].isin(seeds))].sort_values("seed")  # fmt: skip
        check_same(mine.sort_values("seed"), theirs, f"{kind} windows",
                   ["threshold", "tp", "fp", "fn", "tn", "roc_auc"])  # fmt: skip
    saved_f = pd.read_csv(lp.PDIR / "flights.csv")
    saved_f = saved_f[(saved_f["protocol"] == "seen") & saved_f["seed"].isin(seeds)]
    mine_f = flights[flights["form"] == "pytorch"]
    a = mine_f.sort_values(["kind", "seed", "flight_id"])
    b = saved_f.sort_values(["kind", "seed", "flight_id"])
    check_same(a, b, "LSTM flights", ["detected", "alarms", "delay_s"])
    return x, meta, flights, windows, changes, models


def choose_int8(changes):
    """Per model, the int8 setting whose alarm decisions agree most with float32 on
    the validation windows (mean over seeds); ties go to the smaller file."""
    c = changes[changes["form"].str.startswith("int8:")
                & ~changes["form"].str.contains("fp32_threshold")]  # fmt: skip
    g = c.groupby(["kind", "form"])[["val_decision_agreement",
                                     "val_median_rel_change", "size_kb"]].mean()  # fmt: skip
    g = g.reset_index().sort_values(
        ["kind", "val_decision_agreement", "val_median_rel_change", "size_kb"],
        ascending=[True, False, True, True],
    )
    return {k: grp["form"].iloc[0].split(":", 1)[1] for k, grp in g.groupby("kind")}


# -------------------------------------------------------------- forests --


def forest_part(seeds, hours):
    orig, syn = am.load(M05)
    cols = am.feature_columns(orig)
    sets = am.protocol_sets(orig, syn, None, "seen")
    tr, _, clean_va, o_test, s_test = sets
    xo = o_test[cols].to_numpy("float32")
    xs = s_test[cols].to_numpy("float32")
    saved = pd.read_csv(M05 / "all_models_window_full.csv")
    saved = saved[(saved["protocol"] == "seen") & (saved["model"] == "random_forest")]
    windows, flights, rows, pipes = [], [], [], {}
    base = D.seen_params()
    for trees, depth in ((base["n_estimators"], base["max_depth"]),) + FORESTS:
        tag = f"t{trees}_d{depth or 'full'}"
        full = (trees, depth) == (base["n_estimators"], base["max_depth"])
        params = {**base, "n_estimators": trees, "max_depth": depth}
        for seed in seeds:
            if full:
                pipe, thr, _ = D.fit_random_forest(seed)
                d = joblib.load(D.MODELS / f"rf_seen_s{seed}.joblib")
                cal = (thr, d["pool_hours"], d["allowed"])
            else:
                path = MODELS / f"rf_{tag}_s{seed}.joblib"
                if path.is_file():
                    d = joblib.load(path)
                    pipe, cal = d["model"], d["calibration"]
                else:
                    pipe = am.fit("random_forest", params, tr, cols, seed)
                    pool = am.clean_pool("random_forest", params, tr, clean_va, pipe,
                                         cols, seed)  # fmt: skip
                    cal = am.threshold_for(pool)
                    D.atomic_write(path, lambda p: joblib.dump(
                        {"model": pipe, "calibration": cal}, p))  # fmt: skip
            pipes[(tag, seed)] = pipe
            so = am.score("random_forest", pipe, o_test, cols)
            ss = am.score("random_forest", pipe, s_test, cols)
            onnx_path = ox.export_forest(pipe, len(cols),
                                         MODELS / f"rf_{tag}_s{seed}.onnx")  # fmt: skip
            sess = ox.session(onnx_path, 2)
            qo, qs = ox.score_all(sess, xo), ox.score_all(sess, xs)
            refusal = None
            if full:
                refusal = ox.quantize(onnx_path, MODELS / f"rf_{tag}_s{seed}_int8.onnx",
                                      "default")  # fmt: skip
            for form, (a, b) in (("sklearn", (so, ss)), ("onnx_fp32", (qo, qs))):
                w, f = am.evaluate_scores("random_forest", params, sets, a, b, cal,
                                          hours, "seen", None, seed)  # fmt: skip
                windows.append({**w, "form": form, "forest": tag})
                flights.append(f.assign(form=form, forest=tag))
            allx = np.r_[so, ss]
            rows.append({"forest": tag, "trees": trees, "max_depth": depth or "none",
                         "seed": seed, "threshold": cal[0],
                         "nodes": int(sum(e.tree_.node_count
                                          for e in pipe[-1].estimators_)),
                         "pickle_kb": len(joblib_bytes(pipe)) / 1024.0,
                         "onnx_kb": kb(onnx_path), "int8_refusal": refusal,
                         **{f"test_{k}": v for k, v in change(
                             allx, np.r_[qo, qs], np.ones(len(allx), bool),
                             cal[0], cal[0]).items()}})  # fmt: skip
            if full:
                mine = pd.DataFrame([windows[-2]])
                theirs = saved[saved["seed"] == seed]
                check_same(mine, theirs, f"random forest seed {seed}",
                           ["threshold", "tp", "fp", "fn", "tn", "roc_auc",
                            "fa_alarms", "fa_hours"])  # fmt: skip
        print(f"  forest {tag}: {len(seeds)} seeds scored", flush=True)
    return (pd.DataFrame(windows), pd.concat(flights, ignore_index=True),
            pd.DataFrame(rows), pipes, xo)  # fmt: skip


def joblib_bytes(obj):
    import io

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


# ------------------------------------------------------------- physics --


def physics_part(limit):
    ctx, norm = Context(), D.Norm()
    th = D.physics_calibration().m10
    idx = pp.universe()
    test = idx[(idx["split"] == "test") & (idx["set"] == "orig")]
    if limit:
        test = test.groupby("attacked", group_keys=False).head(limit)
    stage_t = {k: 0.0 for k in ("features", "kalman_batch", "normalise",
                                "window_stats", "sequences")}  # fmt: skip
    counts = {"rows": 0, "windows": 0, "sequences": 0, "ticks": 0, "seconds": 0.0}
    vel_t, kf_row_t, kf_tick_t = [], [], []
    agree = {"vel_rows": 0, "vel_both_finite": 0, "vel_nan_match": 0,
             "vel_max_diff": 0.0, "vel_alarms_batch": 0, "vel_alarms_stream": 0,
             "vel_alarms_same": 0, "kf_ticks_batch": 0, "kf_ticks_stream": 0,
             "kf_ticks_matched": 0, "kf_ticks_stream_only": 0,
             "kf_ticks_stream_only_short_piece": 0,
             "kf_nis_h_max_rel": 0.0, "kf_nis_v_max_rel": 0.0,
             "kf_alarms_batch": 0, "kf_alarms_stream": 0, "kf_alarms_same": 0}  # fmt: skip
    started = time.time()
    for n, (_, r) in enumerate(test.iterrows(), start=1):
        g = pd.read_parquet(r["base_path"])
        fl = D.Flight(g, ctx, norm)
        for stage, get in (("features", lambda: fl.features),
                           ("kalman_batch", lambda: fl.filter),
                           ("normalise", lambda: fl.normalized),
                           ("window_stats", lambda: fl.windows),
                           ("sequences", lambda: fl.sequences)):  # fmt: skip
            t0 = time.perf_counter()
            get()
            stage_t[stage] += time.perf_counter() - t0
        t_s = g["t_s"].to_numpy(float)
        counts["rows"] += len(g)
        counts["seconds"] += float(t_s[-1] - t_s[0])
        counts["windows"] += len(fl.windows)
        counts["sequences"] += len(fl.sequences[0])

        # --- velocity channel, every row
        t_us = g["time_usec"].to_numpy("int64")
        tt = (t_us - t_us[0]) * 1e-6
        step = np.diff(tt)
        gap_s = GAP_FACTOR * np.median(step[step > 0])
        vc = VelocityCheck(gap_s, th["mu"], th["sd"], th["velocity"])
        cols = [t_s.tolist(), t_us.tolist()] + [
            g[c].to_numpy(float).tolist() for c in
            ("latitude_deg", "longitude_deg", "velocity_north", "velocity_east")
        ]  # fmt: skip
        vals = np.empty(len(g))
        v_alarm = []
        for i, row in enumerate(zip(*cols)):
            t0 = time.perf_counter_ns()
            hit = vc.step(*row)
            vel_t.append(time.perf_counter_ns() - t0)
            vals[i] = vc.value
            if hit:
                v_alarm.append(row[0])
        batch = fl.features[VELOCITY].to_numpy(float)
        stream = vals.astype(np.float32).astype(float)
        both = np.isfinite(batch) & np.isfinite(stream)
        agree["vel_rows"] += len(g)
        agree["vel_both_finite"] += int(both.sum())
        agree["vel_nan_match"] += int((np.isfinite(batch) == np.isfinite(stream)).sum())
        if both.any():
            agree["vel_max_diff"] = max(
                agree["vel_max_diff"], float(np.max(np.abs(batch - stream)[both]))
            )
        b_alarm = m10_alarms(fl.sig, "velocity", th)
        agree["vel_alarms_batch"] += len(b_alarm)
        agree["vel_alarms_stream"] += len(v_alarm)
        agree["vel_alarms_same"] += len(set(b_alarm.tolist()) & set(v_alarm))

        # --- Kalman channel, every complete row, a filter tick every 0.1 s
        keep = g["aligned_all"].to_numpy() == 1
        k = g[keep]
        kt_s = t_s[keep]
        kt_us = k["time_usec"].to_numpy("int64")
        kstep = np.diff((kt_us - kt_us[0]) * 1e-6)
        kgap = ctx.fcfg["gap_factor"] * np.median(kstep[kstep > 0])
        kc = KalmanCheck(ctx.fcfg, ctx.noise, kgap, th["kalman"])
        f = {c: k[c].to_numpy(float).tolist() for c in
             ("latitude_deg", "longitude_deg", "altitude", "pressure_altitude",
              "velocity_east", "velocity_north", "velocity_up")}  # fmt: skip
        acc = k[ctx.cols["accel"]].to_numpy(float).tolist()
        quat = k[ctx.cols["quat"]].to_numpy(float).tolist()
        ticks, k_alarm = [], []

        def t_of(i, kt_s=kt_s):
            return kt_s[i]

        rows = zip(kt_us.tolist(), f["latitude_deg"], f["longitude_deg"],
                   f["altitude"], f["pressure_altitude"], f["velocity_east"],
                   f["velocity_north"], f["velocity_up"], acc, quat)  # fmt: skip
        for row in rows:
            t0 = time.perf_counter_ns()
            hit = kc.step(t_of, *row)
            dt = time.perf_counter_ns() - t0
            (kf_tick_t if kc.ticks else kf_row_t).append(dt)
            ticks += kc.ticks
            k_alarm += hit
        k_alarm += kc.flush(t_of)
        ticks += kc.ticks
        df = fl.filter
        bt = df["row_t_s"].to_numpy(float)
        st = pd.DataFrame(ticks, columns=["row", "t", "valid", "nis_h", "nis_v"])
        st["row_t_s"] = kt_s[st["row"].to_numpy()]
        m = pd.merge(df[["row_t_s", "nis_h", "nis_v", "valid"]], st,
                     on="row_t_s", suffixes=("_b", "_s"))  # fmt: skip
        # ticks only the stream ran: the batch code drops log pieces shorter than
        # min_segment_s once it has seen the whole log; a stream cannot know yet
        extra = st.loc[~st["row_t_s"].isin(bt), "row"].to_numpy()
        kept_t = (kt_us - kt_us[0]) * 1e-6
        short = 0
        for s0, s1 in segments(kept_t, ctx.fcfg["gap_factor"]):
            if kept_t[s1 - 1] - kept_t[s0] < ctx.fcfg["min_segment_s"]:
                short += int(((extra >= s0) & (extra < s1)).sum())
        agree["kf_ticks_stream_only"] += len(extra)
        agree["kf_ticks_stream_only_short_piece"] += short
        agree["kf_ticks_batch"] += len(bt)
        agree["kf_ticks_stream"] += len(st)
        agree["kf_ticks_matched"] += int((m["valid_b"] == m["valid_s"]).sum())
        for c in ("nis_h", "nis_v"):
            rel = np.abs(m[f"{c}_b"] - m[f"{c}_s"]) / (np.abs(m[f"{c}_s"]) + 1e-6)
            agree[f"kf_{c}_max_rel"] = max(
                agree[f"kf_{c}_max_rel"], float(rel.max()) if len(rel) else 0.0
            )
        counts["ticks"] += len(st)
        b_alarm = m10_alarms(fl.sig, "kalman", th)
        s_alarm = kt_s[np.asarray(k_alarm, int)] if k_alarm else np.zeros(0)
        agree["kf_alarms_batch"] += len(b_alarm)
        agree["kf_alarms_stream"] += len(s_alarm)
        agree["kf_alarms_same"] += len(set(b_alarm.tolist()) & set(s_alarm.tolist()))
        if n % 10 == 0 or n == len(test):
            print(f"  physics {n}/{len(test)} flights  [{time.time() - started:.0f} s]",
                  flush=True)  # fmt: skip

    def us(a):
        a = np.asarray(a, float) / 1e3
        return {"step_us_median": float(np.median(a)),
                "step_us_p99": float(np.percentile(a, 99)),
                "step_us_max": float(a.max()), "steps": int(len(a)),
                "total_s": float(a.sum() / 1e6)}  # fmt: skip

    sec = counts["seconds"]
    vel = {"detector": "velocity", "step": "one log row", **us(vel_t)}
    kf_rows = {"detector": "kalman", "step": "one log row, no filter tick",
               **us(kf_row_t)}  # fmt: skip
    kf_ticks = {"detector": "kalman", "step": "one log row that runs a 0.1 s tick",
                **us(kf_tick_t)}  # fmt: skip
    for row in (vel, kf_rows, kf_ticks):
        row["cpu_share_of_flight_time"] = row["total_s"] / sec
    stream = pd.DataFrame([vel, kf_rows, kf_ticks])
    stream["flights"] = len(test)
    stream["flight_hours"] = sec / 3600.0
    for k_, v in agree.items():
        stream[k_] = v
    per = {"features": counts["rows"], "kalman_batch": counts["ticks"],
           "normalise": counts["rows"], "window_stats": counts["windows"],
           "sequences": counts["sequences"]}  # fmt: skip
    unit = {"features": "row", "kalman_batch": "filter tick", "normalise": "row",
            "window_stats": "5 s window", "sequences": "5 s sequence"}  # fmt: skip
    pipeline = pd.DataFrame([
        {"stage": s, "total_s": stage_t[s], "per": unit[s], "count": per[s],
         "ms_per_unit": 1e3 * stage_t[s] / max(per[s], 1),
         "cpu_share_of_flight_time": stage_t[s] / sec,
         "flights": len(test), "flight_hours": sec / 3600.0}
        for s in stage_t
    ])  # fmt: skip
    return stream, pipeline


# ------------------------------------------------------------- latency --


def latency_part(x, meta, models, choice, pipes, xo):
    """One thread, one window: native, ONNX float32 and ONNX int8."""
    import torch

    _, _, _, test = lp.masks(meta, {"kind": "clf", "protocol": "seen",
                                    "held": None}, None)  # fmt: skip
    xs = x[test][:1000]
    rows = []
    with one_thread():
        for kind, name in KINDS.items():
            model = models[(kind, LATENCY_SEED)]
            net = ox.ae_scorer(model) if kind == "ae" else model.eval()

            def native(xi, net=net):
                with torch.no_grad():
                    return net(torch.from_numpy(xi))

            forms = {"pytorch": native}
            fp32 = MODELS / f"{kind}_s{LATENCY_SEED}.onnx"
            forms["onnx_fp32"] = ox.scorer(ox.session(fp32, 1))
            q = MODELS / f"{kind}_s{LATENCY_SEED}_int8_{choice[kind]}.onnx"
            forms["onnx_int8"] = ox.scorer(ox.session(q, 1))
            for form, fn in forms.items():
                rows.append({"detector": name, "form": form, **latency(fn, xs)})
                print(f"  latency {name} {form}: {rows[-1]['latency_ms_median']:.3f} ms",
                      flush=True)  # fmt: skip
        xr = xo[:1000]
        for (tag, seed), pipe in pipes.items():
            if seed != LATENCY_SEED:
                continue
            clf = pipe[-1]
            old = clf.n_jobs
            clf.n_jobs = 1
            sess = ox.scorer(ox.session(MODELS / f"rf_{tag}_s{seed}.onnx", 1))
            for form, fn in (("sklearn", lambda xi, p=pipe: p.predict_proba(xi)[:, 1]),
                             ("onnx_fp32", sess)):  # fmt: skip
                rows.append({"detector": "random_forest", "forest": tag, "form": form,
                             **latency(fn, xr)})  # fmt: skip
                print(f"  latency forest {tag} {form}: "
                      f"{rows[-1]['latency_ms_median']:.3f} ms", flush=True)  # fmt: skip
            clf.n_jobs = old
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- run --


def run_info(choice, seeds):
    import onnx
    import onnxruntime
    import skl2onnx
    import sklearn
    import torch

    cpu = platform.processor()
    try:
        import subprocess

        cpu = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or cpu  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cpu": cpu, "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__, "skl2onnx": skl2onnx.__version__,
            "scikit_learn": sklearn.__version__, "threads": 1,
            "latency_calls": CALLS, "warmup_calls": WARMUP, "seeds": list(seeds),
            "latency_seed": LATENCY_SEED, "int8_choice": choice}  # fmt: skip


def models_table(lf, lw, changes, choice, ff, fw, frows, lat, hours):
    """F_edge_models.csv: one row per detector and form."""
    out = []
    named = {"pytorch": "pytorch", "onnx_fp32": "onnx_fp32"}
    for kind, name in KINDS.items():
        best = f"int8:{choice[kind]}"
        forms = {"pytorch": "pytorch", "onnx_fp32": "onnx_fp32", best: "onnx_int8",
                 best + "|fp32_threshold": "onnx_int8_fp32_threshold"}  # fmt: skip
        for src, label in forms.items():
            f = lf[(lf["kind"] == kind) & (lf["form"] == src)]
            w = lw[(lw["kind"] == kind) & (lw["form"] == src)]
            c = changes[(changes["kind"] == kind) & (changes["form"] == src)]
            lat_form = named.get(src, "onnx_int8")
            lrow = lat[(lat["detector"] == name) & (lat["form"] == lat_form)]
            out.append({**mean_sd(per_seed(f, w, hours),
                                  {"detector": name, "form": label,
                                   "int8_setting": choice[kind]
                                   if label.startswith("onnx_int8") else ""}),
                        "size_kb": float(c["size_kb"].mean()),
                        **lrow.iloc[0][["latency_ms_median", "latency_ms_p95",
                                        "latency_ms_p99"]].to_dict(),
                        "test_max_score_change": float(c["test_max_score_change"].max()),
                        "test_decision_agreement":
                            float(c["test_decision_agreement"].mean())})  # fmt: skip
    full = frows["forest"].iloc[0]
    for src in ("sklearn", "onnx_fp32"):
        f = ff[(ff["forest"] == full) & (ff["form"] == src)]
        w = fw[(fw["forest"] == full) & (fw["form"] == src)]
        r = frows[frows["forest"] == full]
        lrow = lat[(lat["detector"] == "random_forest") & (lat["forest"] == full)
                   & (lat["form"] == src)]  # fmt: skip
        out.append({**mean_sd(per_seed(f, w, hours),
                              {"detector": "random_forest", "form": src,
                               "int8_setting": ""}),
                    "size_kb": float((r["pickle_kb"] if src == "sklearn"
                                      else r["onnx_kb"]).mean()),
                    **lrow.iloc[0][["latency_ms_median", "latency_ms_p95",
                                    "latency_ms_p99"]].to_dict(),
                    "test_max_score_change": 0.0 if src == "sklearn"
                    else float(r["test_max_score_change"].max()),
                    "test_decision_agreement": 1.0 if src == "sklearn"
                    else float(r["test_decision_agreement"].mean())})  # fmt: skip
    out.append(
        {
            "detector": "random_forest",
            "form": "onnx_int8",
            "int8_setting": "not applicable",
            "note": "ONNX Runtime refused: " + str(frows["int8_refusal"].iloc[0]),
        }
    )
    return pd.DataFrame(out)


def int8_table(lf, lw, changes, choice, hours):
    out = []
    for kind, name in KINDS.items():
        for form in sorted(changes.loc[changes["kind"] == kind, "form"].unique()):
            f = lf[(lf["kind"] == kind) & (lf["form"] == form)]
            w = lw[(lw["kind"] == kind) & (lw["form"] == form)]
            c = changes[(changes["kind"] == kind) & (changes["form"] == form)]
            setting = form.split(":", 1)[-1].split("|")[0]
            s = mean_sd(per_seed(f, w, hours), {"detector": name, "form": form})
            out.append({"detector": name, "form": form,
                        "chosen": form.startswith("int8:") and setting == choice[kind],
                        "size_kb": float(c["size_kb"].mean()),
                        "val_decision_agreement": float(c["val_decision_agreement"].mean()),
                        "val_median_rel_change": float(c["val_median_rel_change"].mean()),
                        "test_decision_agreement":
                            float(c["test_decision_agreement"].mean()),
                        "test_max_score_change": float(c["test_max_score_change"].max()),
                        **{k: s[k] for k in ("real_detected", "syn_detected",
                                             "consistent_detected", "f1", "roc_auc",
                                             "fa_alarms", "fa_hours", "fa_per_hour",
                                             "threshold")}})  # fmt: skip
    return pd.DataFrame(out)


def forests_table(ff, fw, frows, lat, hours):
    out = []
    for tag, r in frows.groupby("forest", sort=False):
        f = ff[(ff["forest"] == tag) & (ff["form"] == "onnx_fp32")]
        w = fw[(fw["forest"] == tag) & (fw["form"] == "onnx_fp32")]
        lrow = lat[(lat["detector"] == "random_forest") & (lat["forest"] == tag)]
        s = mean_sd(per_seed(f, w, hours), {"forest": tag})
        out.append({"forest": tag, "trees": r["trees"].iloc[0],
                    "max_depth": r["max_depth"].iloc[0], "seeds": s["seeds"],
                    "nodes": float(r["nodes"].mean()), "onnx_kb": float(r["onnx_kb"].mean()),
                    "sklearn_ms_median": float(lrow.loc[lrow["form"] == "sklearn",
                                                        "latency_ms_median"].iloc[0]),
                    "onnx_ms_median": float(lrow.loc[lrow["form"] == "onnx_fp32",
                                                     "latency_ms_median"].iloc[0]),
                    "onnx_ms_p99": float(lrow.loc[lrow["form"] == "onnx_fp32",
                                                  "latency_ms_p99"].iloc[0]),
                    **{k: s[k] for k in ("real_detected", "syn_detected",
                                         "syn_detected_sd", "consistent_detected",
                                         "consistent_detected_sd", "f1", "roc_auc",
                                         "fa_alarms", "fa_hours", "fa_per_hour",
                                         "fa_upper_95")},
                    "onnx_vs_sklearn_agreement":
                        float(r["test_decision_agreement"].mean())})  # fmt: skip
    return pd.DataFrame(out)


def run(quick=False):
    started = time.time()
    seeds = D.SEEDS[:1] if quick else D.SEEDS
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    hours = pd.Series(lp.flight_hours())  # Step C's flight-hour table, both callers
    print("=" * 78)
    print("AEROVIGIL - PLAN M11: EDGE BENCHMARK (one CPU thread)")
    print("=" * 78)
    print("LSTMs: PyTorch, ONNX float32, ONNX int8 settings ...")
    x, meta, lf, lw, changes, models = lstm_part(seeds, hours)
    choice = choose_int8(changes)
    print(f"  int8 setting chosen on validation windows: {choice}")
    print(f"Random forests [{time.time() - started:.0f} s] ...")
    fw, ff, frows, pipes, xo = forest_part(seeds, hours)
    print(f"Latency, one thread [{time.time() - started:.0f} s] ...")
    lat = latency_part(x, meta, models, choice, pipes, xo)
    print(f"Physics detectors, streaming [{time.time() - started:.0f} s] ...")
    with one_thread():
        stream, pipeline = physics_part(4 if quick else None)
    tables = {
        "F_edge_models.csv": models_table(lf, lw, changes, choice, ff, fw, frows, lat,
                                          hours),
        "F_int8_settings.csv": int8_table(lf, lw, changes, choice, hours),
        "F_compact_forests.csv": forests_table(ff, fw, frows, lat, hours),
        "F_streaming.csv": stream,
        "F_pipeline.csv": pipeline,
    }  # fmt: skip
    out_dir = OUT / "quick" if quick else TABLES
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / name, index=False)
    info = run_info(choice, seeds)
    info["minutes"] = (time.time() - started) / 60.0
    (OUT / ("run_info_quick.json" if quick else "run_info.json")).write_text(
        json.dumps(info, indent=2)
    )
    report(tables, info)
    print(f"\nSaved to {out_dir}  [{(time.time() - started) / 60:.1f} min]")
    return tables


def report(tables, info):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\nCPU: {info['cpu']}   one thread   int8: {info['int8_choice']}")
    m = tables["F_edge_models.csv"]
    cols = ["detector", "form", "size_kb", "latency_ms_median", "latency_ms_p99",
            "real_detected", "syn_detected", "consistent_detected", "f1", "roc_auc",
            "fa_alarms", "fa_hours", "fa_per_hour", "test_decision_agreement"]  # fmt: skip
    print("\nMODELS (mean over seeds)")
    print(m[[c for c in cols if c in m]].round(4).to_string(index=False))
    print("\nSMALLER FORESTS (ONNX)")
    print(tables["F_compact_forests.csv"].round(4).to_string(index=False))
    s = tables["F_streaming.csv"]
    print("\nPHYSICS, STREAMING")
    print(
        s[
            [
                "detector",
                "step",
                "step_us_median",
                "step_us_p99",
                "step_us_max",
                "steps",
                "cpu_share_of_flight_time",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    agree = [c for c in s.columns if c.startswith(("vel_", "kf_"))]
    print(s[agree].iloc[0].to_string())
    print("\nINPUT BUILDING (batch code, per unit)")
    print(tables["F_pipeline.csv"].round(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 4 flights")
    run(ap.parse_args().quick)


if __name__ == "__main__":
    main()
