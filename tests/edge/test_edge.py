"""Plan M11 (Step F): streaming detectors match the batch code; exports match models."""

import numpy as np
import pandas as pd
import pytest

from aerovigil.edge.benchmark import choose_int8, per_seed
from aerovigil.edge.streaming import (
    CONVENTION,
    KalmanStream,
    TrailingMean,
    VelocityStream,
    world_accel_row,
)
from aerovigil.features.build_features import GAP_FACTOR, build_flight
from aerovigil.features.build_features_v2 import V1_CONFIG
from aerovigil.kalman.imu import world_accel
from aerovigil.kalman.pipeline import filter_segment, prepare_ticks

FCFG = {"rate_hz": 10, "warmup_s": 5.0, "min_segment_s": 8.0, "gap_factor": 10,
        "gravity": 9.80665, "sigma_pos_m": 1.0, "sigma_vel_mps": 0.3,
        "sigma_gps_alt_m": 1.0, "sigma_baro_m": 0.3, "sigma_vu_mps": 0.3,
        "sigma_acc": 0.18, "sigma_acc_no_imu": 2.0, "sigma_bias_walk": 0.02,
        "sigma_baro_bias_walk": 0.05, "p0_pos": 2.0, "p0_vel": 1.0, "p0_bias": 0.2,
        "p0_baro_bias": 2.0}  # fmt: skip
NOISE = {"pos": 0.04, "vel": 0.0002, "vert": 0.016}
COLS = {"time": "time_usec", "lat": "latitude_deg", "lon": "longitude_deg",
        "alt": "altitude", "vn": "velocity_north", "ve": "velocity_east",
        "vu": "velocity_up", "baro": "pressure_altitude",
        "accel": ["ax", "ay", "az"], "quat": ["qx", "qy", "qz", "qw"]}  # fmt: skip


def fake_flight(n=6000, seed=0):
    """An irregular ~170 Hz log with a 3 s gap, a short piece and missing speeds."""
    rng = np.random.default_rng(seed)
    step = rng.choice([4000, 8000], n)  # 250 Hz and 125 Hz rows, microseconds
    step[2500] = 3_000_000  # a gap
    step[2600] = 2_000_000  # a second gap, leaving a short piece
    t_us = 1_700_000_000_000 + np.cumsum(step)
    t = (t_us - t_us[0]) * 1e-6
    ve, vn = 3 * np.cos(t / 9), 2 * np.sin(t / 7)
    lat = 47.0 + np.cumsum(vn * np.r_[0, np.diff(t)]) / 6371000 * 180 / np.pi
    lon = 8.0 + np.cumsum(ve * np.r_[0, np.diff(t)]) / 6371000 * 180 / np.pi / 0.68
    q = rng.normal(0, 0.05, (n, 4))
    q[:, 3] = 1.0
    g = pd.DataFrame({
        "flight_id": "f", "mission_type": "curved", "time_usec": t_us,
        "attack_label": 0, "latitude_deg": lat + rng.normal(0, 1e-6, n),
        "longitude_deg": lon + rng.normal(0, 1e-6, n),
        "altitude": 30 + rng.normal(0, 0.3, n), "velocity_north": vn,
        "velocity_east": ve, "velocity_up": rng.normal(0, 0.1, n),
        "pressure_altitude": 30 + rng.normal(0, 0.2, n),
        "ax": rng.normal(0, 0.3, n), "ay": rng.normal(0, 0.3, n),
        "az": 9.8 + rng.normal(0, 0.3, n),
        "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
    })  # fmt: skip
    g.loc[rng.choice(n, 40, replace=False), "velocity_north"] = np.nan
    return g


def test_trailing_mean_window_edges_and_missing_values():
    m = TrailingMean(1000)
    assert m.push(0, 1.0) == 1.0
    assert m.push(500, np.nan) == 1.0
    assert m.push(999, 3.0) == 2.0
    assert m.push(1000, 5.0) == 4.0  # the sample at exactly t - span is out
    assert np.isnan(TrailingMean(10).push(0, np.nan))


def test_velocity_stream_matches_module03_row_for_row():
    g = fake_flight()
    batch = build_flight(g, V1_CONFIG, 1e-6)["res_horizontal_velocity_roll_mean"]
    t = (g["time_usec"].to_numpy() - g["time_usec"].iloc[0]) * 1e-6
    step = np.diff(t)
    s = VelocityStream(GAP_FACTOR * np.median(step[step > 0]))
    rows = zip(g["time_usec"].tolist(), g["latitude_deg"], g["longitude_deg"],
               g["velocity_north"], g["velocity_east"])  # fmt: skip
    got = np.array([s.step(*r) for r in rows], np.float32)
    b = batch.to_numpy(np.float32)
    assert (np.isfinite(got) == np.isfinite(b)).all()
    ok = np.isfinite(b)
    np.testing.assert_allclose(got[ok], b[ok], rtol=1e-5, atol=1e-6)


def test_world_accel_row_matches_module06():
    rng = np.random.default_rng(1)
    acc, q = rng.normal(0, 3, (50, 3)), rng.normal(0, 1, (50, 4))
    ref = world_accel(acc, q, CONVENTION, 9.80665)
    got = np.array([world_accel_row(a, qq, 9.80665) for a, qq in zip(acc, q)])
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_kalman_stream_matches_module06_tick_for_tick():
    g = fake_flight()
    ref = []
    for seg in prepare_ticks(g, COLS, FCFG, CONVENTION, 1e-6):
        r = filter_segment(seg, FCFG, NOISE)
        valid = (seg["t"] - seg["t0"]) >= FCFG["warmup_s"]
        ref += list(zip(seg["rows"], valid, r["nis_h"], r["nis_v"]))
    t = (g["time_usec"].to_numpy() - g["time_usec"].iloc[0]) * 1e-6
    step = np.diff(t)
    s = KalmanStream(FCFG, NOISE, FCFG["gap_factor"] * np.median(step[step > 0]))
    got = []
    cols = [g[c].tolist() for c in ("time_usec", "latitude_deg", "longitude_deg",
                                    "altitude", "pressure_altitude",
                                    "velocity_east", "velocity_north",
                                    "velocity_up")]  # fmt: skip
    acc, quat = (
        g[COLS["accel"]].to_numpy().tolist(),
        g[COLS["quat"]].to_numpy().tolist(),
    )
    for *row, a, q in zip(*cols, acc, quat):
        got += s.step(*row, a, q)
    got += s.flush()
    stream = {r: (v, h, vv) for r, _, v, h, vv in got}
    assert len(stream) >= len(ref)  # plus the short piece the batch code drops
    for row, valid, nis_h, nis_v in ref:
        v, h, vv = stream[row]
        assert v == valid
        assert h == pytest.approx(nis_h, rel=1e-9, abs=1e-12)
        assert vv == pytest.approx(nis_v, rel=1e-9, abs=1e-12)


def test_onnx_lstm_export_matches_pytorch_and_int8_is_smaller(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from aerovigil.edge import onnx_models as ox
    from aerovigil.lstm.models import build_model

    x = np.random.default_rng(0).normal(0, 1, (64, 20, 6)).astype(np.float32)
    for kind, params in (("ae", {"hidden": 16, "layers": 1, "latent": 8}),
                         ("clf", {"hidden": 16, "layers": 1})):  # fmt: skip
        m = build_model(kind, 6, params, seed=0).eval()
        p = ox.export_lstm(kind, m, tmp_path / f"{kind}.onnx", x[:1])
        net = ox.ae_scorer(m) if kind == "ae" else m
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy().reshape(-1)
        got = ox.score_all(ox.session(p), x)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
        q = tmp_path / f"{kind}_int8.onnx"
        assert ox.quantize(p, q, "reduce_range") is None
        assert q.stat().st_size < p.stat().st_size
        assert np.corrcoef(ox.score_all(ox.session(q), x), ref)[0, 1] > 0.95


def test_onnx_forest_matches_sklearn_and_int8_is_refused(tmp_path):
    pytest.importorskip("skl2onnx")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from aerovigil.edge import onnx_models as ox

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 5)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", RandomForestClassifier(20, random_state=0))]).fit(x, y)  # fmt: skip
    p = ox.export_forest(pipe, 5, tmp_path / "rf.onnx")
    got = ox.score_all(ox.session(p), x)
    assert np.mean(np.abs(got - pipe.predict_proba(x)[:, 1]) < 1e-5) > 0.99
    assert ox.quantize(p, tmp_path / "rf_int8.onnx", "default")  # a refusal message


def test_int8_choice_uses_validation_agreement_then_size():
    c = pd.DataFrame([
        {"kind": "ae", "form": "int8:a", "val_decision_agreement": 0.99,
         "val_median_rel_change": 0.01, "size_kb": 30},
        {"kind": "ae", "form": "int8:b", "val_decision_agreement": 0.999,
         "val_median_rel_change": 0.02, "size_kb": 40},
        {"kind": "ae", "form": "int8:b|fp32_threshold", "val_decision_agreement": 1.0,
         "val_median_rel_change": 0.0, "size_kb": 40},
        {"kind": "clf", "form": "int8:a", "val_decision_agreement": 0.99,
         "val_median_rel_change": 0.01, "size_kb": 30},
        {"kind": "clf", "form": "int8:b", "val_decision_agreement": 0.99,
         "val_median_rel_change": 0.01, "size_kb": 20},
    ])  # fmt: skip
    assert choose_int8(c) == {"ae": "b", "clf": "b"}


def test_per_seed_counts_false_alarms_on_normal_flights_only():
    f = pd.DataFrame({"seed": 1, "flight_id": ["a", "b", "c"],
                      "attacked": [True, False, False], "origin": "orig",
                      "velocity_mode": "na", "detected": [True, False, False],
                      "delay_s": [1.5, np.nan, np.nan], "alarms": [3, 1, 0]})  # fmt: skip
    w = pd.DataFrame({"seed": [1], "tp": [8], "fp": [2], "fn": [2], "tn": [88],
                      "roc_auc": [0.9], "threshold": [0.5]})  # fmt: skip
    r = per_seed(f, w, pd.Series({"a": 0.1, "b": 0.5, "c": 1.5})).iloc[0]
    assert r["fa_alarms"] == 1 and r["fa_hours"] == pytest.approx(2.0)
    assert r["real_detected"] == 1.0 and r["f1"] == pytest.approx(0.8)
