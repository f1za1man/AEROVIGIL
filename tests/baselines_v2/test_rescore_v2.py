"""Module 05 v2 tests: phase-aware window tables and the latched re-scoring."""

import json

import numpy as np
import pandas as pd
import pytest

import aerovigil.baselines.build_window_tables_v2 as bw
import aerovigil.baselines.rescore_v2 as rs
from aerovigil.attacks.inject_v2 import KEEP, inject
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.preprocessing.normalize_features_v2 import CLIP, NON_FEATURES

MISSIONS = {"curved": 150.0, "random": 120.0, "straight": 60.0}
TRUTH = (
    ("offset_m", 0.0),
    ("spoof_truth", 0),
    ("phase", 0),
    ("seconds_since_onset", np.nan),
)
EXTRA = ("source_flight_id", "attack_type", "velocity_mode", "spoof_offset_north_m",
         "spoof_offset_east_m", "offset_rate_mps")  # fmt: skip


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


def attacked(g, mission, duration):
    """A ramp then a 9.7 m plateau; the random mission ends early, so it snaps back."""
    end = duration - 10 if mission == "random" else duration
    params = {"direction_deg": 45.0, "magnitude_m": 9.7, "ramp_seconds": 4.0}
    spec = {"attack_type": "constant_offset", "start_s": 8.0,
            "duration_s": end - 8.0, "params": params}  # fmt: skip
    out, _ = inject(g, spec, "position_only", g["flight_id"].iloc[0])
    return out.drop(columns=[c for c in EXTRA if c in out.columns])


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Twenty fake flights through the whole v2 chain: base, features, normalized."""
    root = tmp_path_factory.mktemp("m05v2")
    m03, interim = root / "m03", root / "interim"
    (m03 / "base").mkdir(parents=True)
    (m03 / "normalized").mkdir(parents=True)
    interim.mkdir(parents=True)
    splits = ["train"] * 12 + ["validation"] * 4 + ["test"] * 4
    rows, split_rows, cache, feats, fit = [], [], {}, None, []
    for i, split in enumerate(splits):
        mission = list(MISSIONS)[i % 3]
        duration = MISSIONS[mission]
        is_attacked = i % 2 == 1
        fid = f"{mission}__{'Attacked' if is_attacked else 'Normal'}__log_{i:02d}"
        g = fake_flight(fid, mission, duration, seed=i)
        if is_attacked:
            g = attacked(g, mission, duration)
        else:
            for c, v in TRUTH:
                g[c] = v
        g.to_parquet(m03 / "base" / f"{fid}.parquet", index=False)
        rows.append({
            "flight_id": fid, "mission": mission,
            "attacked": int(is_attacked), "file": f"{fid}.parquet",
        })  # fmt: skip
        split_rows.append({"flight_id": fid, "split": split})
        f = build_flight_v2(g)
        cache[fid] = f
        feats = feats or [c for c in f.columns if c not in NON_FEATURES]
        if split == "train":
            fit.append(f.loc[f["phase"].isin([0, 1]), feats].to_numpy(np.float64))
    pd.DataFrame(rows).to_csv(interim / "base_report.csv", index=False)
    pd.DataFrame(split_rows).to_csv(interim / "splits.csv", index=False)
    fit = np.concatenate(fit)
    median = np.nanmedian(fit, axis=0)
    q75, q25 = np.nanpercentile(fit, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)
    for r in rows:
        f = cache[r["flight_id"]].copy()
        z = (f[feats].to_numpy(np.float64) - median) / scale
        f[feats] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        f.to_parquet(m03 / "normalized" / r["file"], index=False)
    (m03 / "normalization_parameters.json").write_text(json.dumps({"features": feats}))
    return {"root": root, "m03": m03, "interim": interim, "features": feats}


@pytest.fixture
def paths(project, monkeypatch):
    monkeypatch.setattr(bw, "M03_NORM", project["m03"] / "normalized")
    monkeypatch.setattr(bw, "BASE_REPORT", project["interim"] / "base_report.csv")
    monkeypatch.setattr(bw, "SPLITS", project["interim"] / "splits.csv")
    monkeypatch.setattr(bw, "OUT_DIR", project["root"] / "m05")
    return project


def test_window_class_and_the_two_labels_disagree_on_the_plateau(paths):
    tables = bw.run(("orig",))
    orig = tables["orig"]
    assert set(orig["window_class"]) <= {
        "normal",
        "pre",
        "ramp",
        "plateau",
        "snapback",
        "post",
    }
    plateau = orig[orig["window_class"] == "plateau"]
    assert len(plateau) > 0
    assert (
        plateau["y_window"].eq(1).all()
    )  # Module 05 calls every plateau window an attack
    assert (
        plateau["y_event"].eq(-1).all()
    )  # the re-scoring takes it out of the question
    event = orig[orig["y_event"] == 1]
    assert set(event["window_class"]) == {"ramp", "snapback"}
    assert event["attack_type"].eq("original").all()
    clean = orig[orig["y_event"] == 0]
    assert set(clean["window_class"]) <= {"normal", "pre"}
    assert clean["spoof_fraction"].eq(0).all()


def test_snapback_appears_only_where_the_attack_ends_inside_the_log(paths):
    orig = bw.run(("orig",))["orig"]
    by_mission = orig[orig["window_class"] == "snapback"]["mission_type"].unique()
    assert set(by_mission) == {
        "random"
    }  # curved and straight run to the end of the log


def test_feature_subsets_are_the_module05_set_and_the_v2_set(paths):
    orig = bw.run(("orig",))["orig"]
    all_cols = rs.summary_columns(orig, "all")
    gps_cols = rs.summary_columns(orig, "gps")
    assert len(all_cols) == len(paths["features"]) * 6
    assert len(all_cols) - len(gps_cols) == 14 * 6  # the IMU features
    assert not any(c.startswith(("res_accel", "res_course")) for c in gps_cols)


def test_hours_counts_one_hop_per_window():
    frame = pd.DataFrame({"a": range(1440)})
    assert rs.hours(frame) == pytest.approx(1440 * 2.5 / 3600)


class Constant:
    """A stand-in model whose score is one chosen summary column."""

    def __init__(self, column_index):
        self.j = column_index

    def predict_proba(self, x):
        p = np.clip(np.asarray(x)[:, self.j], 0, 1)
        return np.column_stack([1 - p, p])


def test_latch_threshold_uses_validation_normals_only(paths):
    orig = bw.run(("orig",))["orig"]
    cols = rs.summary_columns(orig, "all")
    model = Constant(0)
    thr, n_pool, pool_h, allowed = rs.latch_threshold(model, orig, cols, 1.0)
    pool = orig[(orig["split"] == "validation") & (orig["attack_type"] == "none")]
    assert n_pool == len(pool) and pool_h == pytest.approx(rs.hours(pool))
    assert allowed == max(1, round(1.0 * pool_h))
    scores = model.predict_proba(pool[cols].to_numpy("float32"))[:, 1]
    assert (scores > thr).sum() == allowed - 1  # strictly above the k-th largest


def test_latch_marks_detection_delay_and_pre_onset_alarms(paths):
    orig = bw.run(("orig",))["orig"]
    cols = rs.summary_columns(orig, "all")
    flights, rate = rs.latch_evaluate(
        Constant(0), orig, cols, -1.0
    )  # everything alarms
    test_ids = set(orig.loc[orig["split"] == "test", "flight_id"])
    assert set(flights["flight_id"]) == test_ids
    att = flights[flights["attacked"]]
    assert att["detected"].all()
    assert (att["delay_s"] >= 0).all()
    assert att["false_alarm_before_onset"].all()  # it alarms from the first window
    assert rate > 0
    quiet, _ = rs.latch_evaluate(Constant(0), orig, cols, 2.0)  # nothing alarms
    assert quiet["alarms"].eq(0).all() and not quiet["detected"].any()


def test_rescore_runs_end_to_end_and_writes_both_tables(paths):
    bw.run(("orig",))
    out = paths["root"] / "m05"
    wtab, ftab = rs.run(alarms_per_hour=2.0, out_dir=out)
    assert set(wtab["task"]) == {"window", "event"}
    assert set(wtab["features"]) == {"gps", "all"}
    assert set(wtab["sources"]) == {"orig"}  # no synthetic table was built
    assert len(wtab) == 2 * 2 * 2
    assert wtab["auc_orig"].between(0, 1).all()
    assert "auc_syn_consistent" not in wtab  # no synthetic table was built
    assert len(ftab) == 8 * len(set(ftab["flight_id"]))
    assert (out / "rescore_window.csv").is_file()
    assert (out / "rescore_flight.csv").is_file()
