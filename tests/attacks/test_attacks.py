"""Module 04 tests: attack profiles, injection, sampling and split inheritance."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.generate_attacks import run  # noqa: E402
from aerovigil.attacks.injector import EARTH_R, inject_attack  # noqa: E402
from aerovigil.attacks.profiles import ATTACK_TYPES, offset_profile  # noqa: E402
from aerovigil.attacks.sampler import sample_attack  # noqa: E402

COLS = {
    "flight": "flight_id",
    "time": "time_usec",
    "label": "attack_label",
    "enabled": "attack_enabled",
    "lat": "latitude_deg",
    "lon": "longitude_deg",
    "vn": "velocity_north",
    "ve": "velocity_east",
    "speed": "velocity",
}

TIMING = {
    "min_start_s": 10.0,
    "min_duration_s": 10.0,
    "start_fraction_range": [0.2, 0.6],
    "duration_s_range": [20.0, 90.0],
}
TYPES_CFG = {
    "abrupt_jump": {"magnitude_m_range": [20.0, 500.0]},
    "constant_offset": {
        "magnitude_m_range": [5.0, 100.0],
        "ramp_seconds_range": [1.0, 5.0],
    },
    "linear_drift": {"drift_rate_mps_range": [0.05, 5.0]},
    "exponential_ramp": {
        "initial_rate_mps_range": [0.05, 1.0],
        "time_constant_s_range": [15.0, 60.0],
        "max_offset_m": 500.0,
    },
    "low_amplitude_wander": {
        "amplitude_m_range": [1.0, 10.0],
        "period_s_range": [8.0, 60.0],
    },
}


def make_flight(fid, duration_s=60.0, hz=250, seed=0):
    rng = np.random.default_rng(seed)
    n, dt = int(duration_s * hz), 1.0 / hz
    t = (np.arange(n) * dt * 1e6).astype("int64") + 1_000_000
    vn, ve = np.full(n, 4.0), np.full(n, 3.0)
    lat = 24.4 + np.degrees(np.cumsum(vn * dt) / EARTH_R)
    lon = 54.6 + np.degrees(np.cumsum(ve * dt) / (EARTH_R * np.cos(np.radians(lat))))
    return pd.DataFrame(
        {
            "flight_id": fid,
            "mission_type": "curved",
            "time_usec": t,
            "timestamp": t,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude": 50.0 + rng.normal(0, 0.01, n),
            "velocity_north": vn,
            "velocity_east": ve,
            "velocity_up": np.zeros(n),
            "velocity": np.hypot(vn, ve) + 0.01,
            "linear_acceleration_x": rng.normal(0, 1, n),
            "angular_velocity_z": rng.normal(0, 0.1, n),
            "magnetic_field_x": rng.normal(0, 0.1, n),
            "pressure_altitude": 51.0 + rng.normal(0, 0.1, n),
            "orientation_w": np.ones(n),
            "attack_label": np.zeros(n),
            "attack_enabled": np.zeros(n),
        }
    )


def spec(attack_type, params, start=20.0, duration=30.0, mode="consistent"):
    return {
        "attack_type": attack_type,
        "start_s": start,
        "duration_s": duration,
        "velocity_mode": mode,
        "params": params,
        "attacked_flight_id": "X__syn",
        "source_flight_id": "X",
    }


LINEAR = {"direction_deg": 90.0, "drift_rate_mps": 2.0}


def test_all_profiles_start_at_zero_offset():
    params = {
        "constant_offset": {"direction_deg": 10, "magnitude_m": 30, "ramp_seconds": 2},
        "linear_drift": LINEAR,
        "exponential_ramp": {
            "direction_deg": 10,
            "initial_rate_mps": 0.5,
            "time_constant_s": 20,
            "max_offset_m": 500,
        },
        "low_amplitude_wander": {
            "amplitude_m": 5,
            "periods_s": [10, 20, 40],
            "phases_north": [1, 2, 3],
            "phases_east": [4, 5, 6],
        },
    }
    for typ in (
        "constant_offset",
        "linear_drift",
        "exponential_ramp",
        "low_amplitude_wander",
    ):
        n, e, _, _ = offset_profile(typ, np.array([0.0]), params[typ])
        assert abs(n[0]) < 1e-9 and abs(e[0]) < 1e-9


def test_linear_drift_matches_formula_and_direction():
    tau = np.array([0.0, 10.0, 25.0])
    n, e, dn, de = offset_profile("linear_drift", tau, LINEAR)  # due east
    assert np.allclose(e, 2.0 * tau) and np.allclose(n, 0, atol=1e-9)
    assert np.allclose(de, 2.0) and np.allclose(dn, 0, atol=1e-9)


def test_exponential_ramp_is_capped():
    p = {
        "direction_deg": 0,
        "initial_rate_mps": 1.0,
        "time_constant_s": 15.0,
        "max_offset_m": 100.0,
    }
    n, _, dn, _ = offset_profile("exponential_ramp", np.array([0.0, 200.0]), p)
    assert n[1] == pytest.approx(100.0) and dn[1] == 0.0


def test_only_gps_columns_change_and_imu_baro_mag_untouched():
    g = make_flight("X")
    out = inject_attack(g, spec("linear_drift", LINEAR), COLS, 1e-6)
    changed = {
        "latitude_deg",
        "longitude_deg",
        "velocity_north",
        "velocity_east",
        "velocity",
        "attack_label",
        "attack_enabled",
        "flight_id",
    }
    for c in g.columns:
        if c not in changed:
            assert g[c].equals(out[c]), f"{c} was modified"


def test_nothing_changes_before_or_after_the_attack_window():
    g = make_flight("X")
    out = inject_attack(
        g, spec("linear_drift", LINEAR, start=20, duration=30), COLS, 1e-6
    )
    t = (g.time_usec - g.time_usec.iloc[0]) * 1e-6
    outside = (t < 20) | (t > 50)
    for c in (
        "latitude_deg",
        "longitude_deg",
        "velocity_north",
        "velocity_east",
        "velocity",
    ):
        assert np.array_equal(g.loc[outside, c], out.loc[outside, c])
    assert (out.loc[outside, "spoof_offset_m"] == 0).all()


def test_labels_and_signed_time_since_attack_start():
    g = make_flight("X")
    out = inject_attack(
        g, spec("linear_drift", LINEAR, start=20, duration=30), COLS, 1e-6
    )
    t = (g.time_usec - g.time_usec.iloc[0]) * 1e-6
    assert ((out.attack_label == 1) == ((t >= 20) & (t <= 50 + 1e-6))).all()
    assert (out.attack_enabled == out.attack_label).all()
    assert out.time_from_attack_start_s.iloc[0] < 0
    assert (out.loc[out.attack_label == 1, "time_from_attack_start_s"] >= 0).all()


def test_recorded_offset_equals_true_displacement():
    g = make_flight("X")
    out = inject_attack(
        g,
        spec("abrupt_jump", {"direction_deg": 45.0, "magnitude_m": 120.0}),
        COLS,
        1e-6,
    )
    p1, p2 = np.radians(g.latitude_deg), np.radians(out.latitude_deg)
    dl = np.radians(out.longitude_deg - g.longitude_deg)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_R * np.arcsin(np.sqrt(a))
    assert np.allclose(d, out.spoof_offset_m, atol=0.05)
    assert out.spoof_offset_m.max() == pytest.approx(120.0, abs=0.05)


def test_consistent_mode_velocity_integrates_to_offset():
    g = make_flight("X")
    out = inject_attack(g, spec("linear_drift", LINEAR, mode="consistent"), COLS, 1e-6)
    on = out.attack_label == 1
    t = ((g.time_usec - g.time_usec.iloc[0]) * 1e-6)[on].to_numpy()
    dve = (out.velocity_east - g.velocity_east)[on].to_numpy()
    integral = np.sum((dve[1:] + dve[:-1]) / 2 * np.diff(t))
    assert integral == pytest.approx(out.spoof_offset_east_m[on].iloc[-1], abs=0.05)


def test_position_only_mode_leaves_velocity_unchanged():
    g = make_flight("X")
    out = inject_attack(
        g, spec("linear_drift", LINEAR, mode="position_only"), COLS, 1e-6
    )
    assert g.velocity_north.equals(out.velocity_north) and g.velocity_east.equals(
        out.velocity_east
    )


def test_consistent_mode_speed_tracks_velocity_components():
    g = make_flight("X")
    out = inject_attack(g, spec("linear_drift", LINEAR, mode="consistent"), COLS, 1e-6)
    # the native 0.01 m/s residual is preserved on every row, attacked or not
    resid = out.velocity - np.hypot(out.velocity_north, out.velocity_east)
    assert np.allclose(resid, 0.01, atol=1e-9)
    assert (out.velocity - g.velocity).abs().max() > 1.0  # speed really moved
    outside = out.attack_label == 0
    assert g.loc[outside, "velocity"].equals(out.loc[outside, "velocity"])


def test_position_only_mode_leaves_speed_unchanged():
    g = make_flight("X")
    out = inject_attack(
        g, spec("linear_drift", LINEAR, mode="position_only"), COLS, 1e-6
    )
    assert g.velocity.equals(out.velocity)


def test_injector_does_not_mutate_the_input_flight():
    g = make_flight("X")
    before = g.copy()
    inject_attack(g, spec("linear_drift", LINEAR), COLS, 1e-6)
    pd.testing.assert_frame_equal(g, before)


# ------------------------------ sampler ------------------------------------
@pytest.fixture
def scfg():
    return {
        "random_seed": 42,
        "probability_consistent_velocity": 0.5,
        "timing": TIMING,
        "attack_types": TYPES_CFG,
        "sweep": {"start_fraction": 0.25, "velocity_mode": "consistent"},
    }


def test_sampler_is_deterministic_and_order_independent(scfg):
    a = sample_attack(scfg, "F1", 100.0, "main", "linear_drift", 0)
    sample_attack(
        scfg, "F2", 100.0, "main", "abrupt_jump", 0
    )  # unrelated call in between
    b = sample_attack(scfg, "F1", 100.0, "main", "linear_drift", 0)
    assert a == b
    assert a != sample_attack(scfg, "F3", 100.0, "main", "linear_drift", 0)


@pytest.mark.parametrize("typ", ATTACK_TYPES)
def test_sampled_timing_and_parameters_stay_in_bounds(scfg, typ):
    for k in range(40):
        s = sample_attack(scfg, f"F{k}", 28.5 + 5 * k, "main", typ, 0)
        assert s["start_s"] >= TIMING["min_start_s"]
        assert s["duration_s"] >= TIMING["min_duration_s"] - 1e-9
        assert s["start_s"] + s["duration_s"] <= 28.5 + 5 * k + 1e-9


def test_too_short_flight_is_skipped(scfg):
    assert sample_attack(scfg, "F", 15.0, "main", "linear_drift", 0) is None


def test_sweep_uses_the_requested_rate_and_runs_to_flight_end(scfg):
    s = sample_attack(scfg, "F", 100.0, "sweep", "linear_drift", 0, rate=0.5)
    assert s["params"]["drift_rate_mps"] == 0.5 and s["velocity_mode"] == "consistent"
    assert s["start_s"] + s["duration_s"] == pytest.approx(100.0)


# ------------------------ end-to-end: split inheritance ---------------------
def build_project(tmp_path):
    clean = tmp_path / "cleaned"
    clean.mkdir()
    rows, splits = [], []
    for i, sp in enumerate(["train", "train", "validation", "test"]):
        fid = f"Curved__Normal__log_{i}"
        g = make_flight(fid, duration_s=40 + 10 * i, seed=i)
        g.to_parquet(clean / f"{fid}.parquet", index=False)
        rows.append(
            {
                "flight_id": fid,
                "file": f"{fid}.parquet",
                "mission_type": "curved",
                "attack_rows": 0,
            }
        )
        splits.append({"flight_id": fid, "split": sp})
    pd.DataFrame(rows).to_csv(tmp_path / "cleaning_report.csv", index=False)
    pd.DataFrame(splits).to_csv(tmp_path / "flight_splits.csv", index=False)
    yaml.safe_dump(
        {
            "dataset": {
                "flight_id_column": "flight_id",
                "timestamp_column": "time_usec",
                "timestamp_unit": "microseconds",
            },
            "labels": {"attack_column": "attack_label"},
            "sensors": {
                "gps": {
                    "latitude": "latitude_deg",
                    "longitude": "longitude_deg",
                    "altitude": "altitude",
                    "velocity_north": "velocity_north",
                    "velocity_east": "velocity_east",
                    "velocity_up": "velocity_up",
                },
                "barometer": {"altitude": "pressure_altitude"},
            },
        },
        open(tmp_path / "pre.yaml", "w"),
    )
    cfg = {
        "paths": {
            "preprocessing_config": str(tmp_path / "pre.yaml"),
            "clean_flights_dir": str(clean),
            "cleaning_report": str(tmp_path / "cleaning_report.csv"),
            "flight_splits": str(tmp_path / "flight_splits.csv"),
            "output_dir": str(tmp_path / "out"),
        },
        "columns": {"attack_enabled": "attack_enabled", "gps_speed": "velocity"},
        "random_seed": 7,
        "draws_per_type": 1,
        "probability_consistent_velocity": 0.5,
        "enabled_attack_types": list(ATTACK_TYPES),
        "timing": TIMING,
        "attack_types": TYPES_CFG,
        "sweep": {
            "enabled": True,
            "splits": ["test"],
            "drift_rates_mps": [0.5, 2.0],
            "start_fraction": 0.25,
            "velocity_mode": "consistent",
        },
    }
    path = tmp_path / "attacks.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return path, splits


def test_attacked_flights_inherit_split_and_regeneration_is_identical(tmp_path):
    path, splits = build_project(tmp_path)
    man1 = run(str(path), output_dir=str(tmp_path / "out"))
    first = {
        f.name: pd.read_parquet(f)
        for f in (tmp_path / "out" / "attacked").glob("*.parquet")
    }
    man2 = run(str(path), output_dir=str(tmp_path / "out"))  # second run, same seed

    split_of = {s["flight_id"]: s["split"] for s in splits}
    assert (man1["split"] == man1["source_flight_id"].map(split_of)).all()
    assert (
        len(man1) == 4 * 5 + 2 * 1
    )  # 4 flights x 5 types + 1 test flight x 2 sweep rates
    assert man1.equals(man2)
    for f in (tmp_path / "out" / "attacked").glob("*.parquet"):
        pd.testing.assert_frame_equal(first[f.name], pd.read_parquet(f))
    # a source flight never appears in two splits
    assert man1.groupby("source_flight_id")["split"].nunique().max() == 1
