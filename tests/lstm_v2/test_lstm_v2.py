"""Module 07 v2 tests: binned sequences, roles, calibration and the latched alarms."""

import json

import numpy as np
import pandas as pd
import pytest

import aerovigil.lstm.build_sequences_v2 as bs
import aerovigil.lstm.evaluate_v2 as ev
import aerovigil.lstm.train_v2 as tr
from aerovigil.attacks.inject_v2 import KEEP, inject
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.lstm.sequences_v2 import T_STEPS, bin_means
from aerovigil.preprocessing.normalize_features_v2 import CLIP, NON_FEATURES

MISSIONS = {"curved": 150.0, "random": 120.0, "straight": 90.0}
TRUTH = (("offset_m", 0.0), ("spoof_truth", 0), ("phase", 0),
         ("seconds_since_onset", np.nan))  # fmt: skip
EXTRA = ("source_flight_id", "attack_type", "velocity_mode", "spoof_offset_north_m",
         "spoof_offset_east_m", "offset_rate_mps")  # fmt: skip
SYN_TYPES = ("linear_drift", "constant_offset")
SWEEP_RATES = (0.1, 1.0)


def fake_flight(fid, mission, duration, seed, hz=50):
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    t = np.arange(n) / hz
    vn, ve, vu = np.full(n, 4.0), np.full(n, 3.0), rng.normal(0, 1.0, n)
    speed = np.sqrt(vn**2 + ve**2 + vu**2) + rng.normal(0, 0.02, n)
    lat = 24.4 + np.degrees(np.cumsum(vn / hz) / 6371000.0)
    lon = 54.6 + np.degrees(np.cumsum(ve / hz) / (6371000.0 * np.cos(np.radians(24.4))))
    df = pd.DataFrame({
        "flight_id": fid, "mission_type": mission,
        "time_usec": (t * 1e6).astype("int64"), "t_s": t,
        "latitude_deg": lat, "longitude_deg": lon,
        "altitude": 50.0 + rng.normal(0, 0.01, n),
        "velocity": speed, "velocity_east": ve, "velocity_north": vn,
        "velocity_up": vu, "pressure_altitude": 51.0 + rng.normal(0, 0.1, n),
        "orientation_x": 0.0, "orientation_y": 0.0,
        "orientation_z": 0.0, "orientation_w": 1.0,
        "linear_acceleration_x": rng.normal(0, 1, n),
        "linear_acceleration_y": rng.normal(0, 1, n),
        "linear_acceleration_z": 9.8 + rng.normal(0, 1, n),
        "attack_label": 0.0, "attack_enabled": 0.0, "aligned_all": 1})  # fmt: skip
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in (
            "latitude_deg",
            "longitude_deg",
            "t_s",
        ):
            df[c] = df[c].astype(np.float32)
    return df[KEEP]


def spec_for(kind, duration, rate=None):
    """A ramp then a plateau, or a steady drift for the sweep."""
    if kind == "linear_drift":
        params = {"direction_deg": 45.0, "drift_rate_mps": rate or 0.4}
    else:
        params = {"direction_deg": 45.0, "magnitude_m": 9.7, "ramp_seconds": 4.0}
    start = 8.0 if rate is None else duration * 0.25
    return {"attack_type": "linear_drift" if kind == "linear_drift" else kind,
            "start_s": start, "duration_s": duration - start - 1.0,
            "params": params}  # fmt: skip


def attacked(g, duration):
    out, _ = inject(g, spec_for("constant_offset", duration), "position_only",
                    g["flight_id"].iloc[0])  # fmt: skip
    return out.drop(columns=[c for c in EXTRA if c in out.columns])


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Twenty original flights plus synthetic attacks, through the whole v2 chain."""
    root = tmp_path_factory.mktemp("m07v2")
    m03, m04, interim = root / "m03", root / "m04", root / "interim"
    for d in (m03 / "base", m03 / "normalized", m04 / "normalized", interim):
        d.mkdir(parents=True)
    splits = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
    rows, split_rows, raw, fit, feats = [], [], {}, [], None
    for i, split in enumerate(splits):
        mission = list(MISSIONS)[i % 3]
        duration = MISSIONS[mission]
        is_att = i % 2 == 1
        fid = f"{mission}__{'Attacked' if is_att else 'Normal'}__log_{i:02d}"
        g = fake_flight(fid, mission, duration, seed=i)
        base = g.copy()
        if is_att:
            g = attacked(g, duration)
        else:
            for c, v in TRUTH:
                g[c] = v
        g.to_parquet(m03 / "base" / f"{fid}.parquet", index=False)
        rows.append({"flight_id": fid, "mission": mission,
                     "attacked": int(is_att), "file": f"{fid}.parquet"})  # fmt: skip
        split_rows.append({"flight_id": fid, "split": split})
        raw[("orig", fid)] = (build_flight_v2(g), f"{fid}.parquet", m03 / "normalized")
        if not is_att:
            raw["base", fid] = (base, mission, duration, split)
    man = []
    for key, val in list(raw.items()):
        if key[0] != "base":
            continue
        fid, (base, mission, duration, split) = key[1], val
        jobs = [("main", t, m, np.nan) for t in SYN_TYPES
                for m in ("consistent", "position_only")]  # fmt: skip
        if split == "test":
            jobs += [("sweep", "linear_drift", "consistent", r) for r in SWEEP_RATES]
        for aset, typ, mode, rate in jobs:
            new_id = f"{fid}__{aset}_{typ}_{mode}_{rate:g}".replace(".", "p")
            out, _ = inject(base, spec_for(typ, duration, rate if aset == "sweep" else None),
                            mode, new_id)  # fmt: skip
            raw[("syn", new_id)] = (
                build_flight_v2(out),
                f"{new_id}.parquet",
                m04 / "normalized",
            )
            man.append({"attacked_flight_id": new_id, "source_flight_id": fid,
                        "split": split, "mission_type": mission, "attack_set": aset,
                        "attack_type": typ, "velocity_mode": mode,
                        "drift_rate_mps": rate, "file": f"{new_id}.parquet"})  # fmt: skip
    for key, val in raw.items():
        if key[0] == "base":
            continue
        f = val[0]
        feats = feats or [c for c in f.columns if c not in NON_FEATURES]
        fid = key[1]
        if (
            key[0] == "orig"
            and dict(
                split_rows and [(r["flight_id"], r["split"]) for r in split_rows]
            ).get(fid)
            == "train"
        ):
            fit.append(f.loc[f["phase"].isin([0, 1]), feats].to_numpy(np.float64))
    pd.DataFrame(rows).to_csv(interim / "base_report.csv", index=False)
    pd.DataFrame(split_rows).to_csv(interim / "splits.csv", index=False)
    pd.DataFrame(man).to_csv(interim / "manifest.csv", index=False)
    fit = np.concatenate(fit)
    median = np.nanmedian(fit, axis=0)
    q75, q25 = np.nanpercentile(fit, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)
    for key, val in raw.items():
        if key[0] == "base":
            continue
        f, fname, folder = val
        f = f.copy()
        z = (f[feats].to_numpy(np.float64) - median) / scale
        f[feats] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        f.to_parquet(folder / fname, index=False)
    (m03 / "normalization_parameters.json").write_text(json.dumps({"features": feats}))
    return {"root": root, "m03": m03, "m04": m04, "interim": interim, "features": feats}


@pytest.fixture(scope="module")
def built(project):
    """Point the builder at the fake project, run it once, and reuse its output."""
    out = project["root"] / "m07"
    bs.M03_NORM = project["m03"] / "normalized"
    bs.M04_NORM = project["m04"] / "normalized"
    bs.BASE_REPORT = project["interim"] / "base_report.csv"
    bs.MANIFEST = project["interim"] / "manifest.csv"
    bs.SPLITS = project["interim"] / "splits.csv"
    bs.OUT_DIR = out
    bs.run(("orig", "syn"), output_dir=out)
    return out


def test_bin_means_ignores_nan_and_counts_rows():
    x = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, np.nan], [7.0, np.nan]])
    t = np.array([0.0, 0.1, 0.3, 0.4])
    means, rows = bin_means(x, t, 0.25)
    assert rows.tolist() == [2, 2]
    assert means[0].tolist() == [2.0, 2.0]  # the NaN is skipped, not counted
    assert means[1][0] == 6.0 and np.isnan(means[1][1])  # a bin with no value stays NaN


def test_sequences_have_the_right_shape_and_no_nan(built, project):
    x = np.load(built / "X_orig.npy")
    meta = pd.read_parquet(built / "meta_orig.parquet")
    assert x.shape == (len(meta), T_STEPS, len(project["features"]))
    assert np.isfinite(x).all()
    assert meta["missing_fraction"].max() <= 0.10


def test_the_event_label_takes_the_plateau_out_of_the_question(built):
    meta = pd.read_parquet(built / "meta_orig.parquet")
    plateau = meta[meta["window_class"] == "plateau"]
    assert len(plateau) > 0
    assert plateau["y_window"].eq(1).all()  # Module 05 called every one an attack
    assert plateau["y_event"].eq(-1).all()  # the event label does not ask
    event = meta[meta["y_event"] == 1]
    assert set(event["window_class"]) == {"ramp"} or set(event["window_class"]) <= {
        "ramp",
        "snapback",
    }
    assert not event["attack_type"].eq("none").any()


def test_calibration_flights_are_normal_and_never_trained_on(built):
    meta = pd.read_parquet(built / "meta_orig.parquet")
    calib = meta[meta["role"] == "calib"]
    assert len(calib) > 0
    assert calib["attack_type"].eq("none").all()
    assert set(meta["role"]) == {"train", "calib", "validation", "test"}
    ae_tr, ae_va = tr.ae_masks(meta)
    clf_tr, clf_va = tr.clf_masks(meta)
    for m in (ae_tr, ae_va, clf_tr, clf_va):
        assert not meta.loc[m.to_numpy(), "role"].eq("calib").any()


def test_synthetic_main_keeps_only_attack_windows_but_the_sweep_keeps_all(built):
    syn = pd.read_parquet(built / "meta_syn.parquet")
    main = syn[syn["attack_set"] == "main"]
    sweep = syn[syn["attack_set"] == "sweep"]
    assert main["attack_fraction"].gt(0).all()  # no repeated clean copies
    assert len(sweep) > 0
    assert sweep["attack_fraction"].eq(0.0).any()  # the pre-attack stretch is kept
    assert set(sweep["drift_rate_mps"].unique()) == set(SWEEP_RATES)


def test_holding_a_family_out_removes_it_from_training_only(built):
    syn = pd.read_parquet(built / "meta_syn.parquet")
    orig = pd.read_parquet(built / "meta_orig.parquet")
    meta = pd.concat([orig, syn], ignore_index=True)
    full, _ = tr.clf_masks(meta)
    held, _ = tr.clf_masks(meta, "linear_drift")
    assert held.sum() < full.sum()
    assert not meta.loc[held.to_numpy(), "attack_type"].eq("linear_drift").any()
    assert meta.loc[full.to_numpy(), "attack_type"].eq("linear_drift").any()
    no_real, _ = tr.clf_masks(meta, "original")
    kept = meta.loc[no_real.to_numpy()]
    assert not kept[kept["y_event"] == 1]["attack_type"].eq("original").any()
    assert kept["attack_type"].eq("original").any()  # their clean stretch still counts


def test_threshold_honours_the_alarm_budget():
    scores = np.linspace(0, 1, 1440)  # 1440 windows is exactly one hour
    thr, n, h, allowed = ev.threshold(scores, 4.0)
    assert n == 1440 and h == pytest.approx(1.0) and allowed == 4
    assert (scores > thr).sum() == 3  # strictly above the fourth largest
    assert ev.threshold(scores, 0.0)[3] == 1  # never fewer than one


def test_latch_reports_delay_only_after_the_measured_onset(built):
    orig = pd.read_parquet(built / "meta_orig.parquet")
    syn = pd.read_parquet(built / "meta_syn.parquet")
    meta = pd.concat([orig, syn], ignore_index=True)
    score = np.ones(len(meta))  # every window alarms
    flights, rate = ev.latch(score, meta, 0.5)
    assert set(flights["flight_id"]) == set(
        meta.loc[meta["role"] == "test", "flight_id"]
    )
    att = flights[flights["attacked"]]
    assert att["detected"].all()
    assert (att["delay_s"] >= 0).all()
    assert att["false_alarm_before_onset"].any()
    assert rate > 0
    quiet, _ = ev.latch(score, meta, 2.0)  # nothing alarms
    assert quiet["alarms"].eq(0).all() and not quiet["detected"].any()


def test_train_and_evaluate_run_end_to_end(built):
    idx = tr.run(seeds=1, unseen=False, output_dir=built)
    assert set(idx["kind"]) == {"ae", "clf"}
    assert idx["protocol"].eq("seen").all()
    assert (idx["n_train"] > 0).all() and (idx["n_val"] > 0).all()
    wres, flights = ev.run(output_dir=built)
    assert wres["roc_auc_mean"].between(0, 1).all()
    assert "original" in set(wres["group"])
    assert set(flights["detector"]) == {"lstm_ae", "lstm_supervised"}
    for name in ("results_window.csv", "flights.csv", "false_alarms.csv",
                 "frontier.csv", "thresholds.csv"):  # fmt: skip
        assert (built / name).is_file()
    frontier = pd.read_csv(built / "frontier.csv")
    assert set(frontier["drift_rate_mps"].unique()) == set(SWEEP_RATES)
