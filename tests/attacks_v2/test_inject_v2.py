"""Module 04 v2 tests: injection on the Module 03 v2 base (fake flights, no dataset)."""

import numpy as np
import pandas as pd
import pytest
import yaml

import aerovigil.attacks.generate_v2 as gv
from aerovigil.attacks.generate_v2 import run
from aerovigil.attacks.inject_v2 import KEEP, SCALARS, inject, speed_uses_3d
from aerovigil.attacks.profiles import ATTACK_TYPES
from aerovigil.data.build_raw_base import PHASE_PRE, PHASE_RAMP

CFG = {
    "random_seed": 7, "draws_per_type": 1, "probability_consistent_velocity": 0.5,
    "enabled_attack_types": list(ATTACK_TYPES),
    "timing": {"min_start_s": 10.0, "min_duration_s": 10.0,
               "start_fraction_range": [0.2, 0.6], "duration_s_range": [20.0, 90.0]},
    "attack_types": {
        "abrupt_jump": {"magnitude_m_range": [20.0, 500.0]},
        "constant_offset": {"magnitude_m_range": [5.0, 100.0], "ramp_seconds_range": [1.0, 5.0]},
        "linear_drift": {"drift_rate_mps_range": [0.05, 5.0]},
        "exponential_ramp": {"initial_rate_mps_range": [0.05, 1.0],
                             "time_constant_s_range": [15.0, 60.0], "max_offset_m": 500.0},
        "low_amplitude_wander": {"amplitude_m_range": [1.0, 10.0], "period_s_range": [8.0, 60.0]}},
    "sweep": {"enabled": True, "splits": ["test"], "drift_rates_mps": [0.5, 2.0],
              "start_fraction": 0.25, "velocity_mode": "consistent"},
}  # fmt: skip


def fake_flight(fid, duration=100.0, hz=50, seed=0, three_d=True):
    rng = np.random.default_rng(seed)
    n = int(duration * hz)
    t = np.arange(n) / hz
    vn, ve, vu = np.full(n, 4.0), np.full(n, 3.0), rng.normal(0, 1.0, n)
    speed = np.sqrt(vn**2 + ve**2 + vu**2) if three_d else np.hypot(vn, ve)
    speed = speed + rng.normal(0, 0.02, n)
    lat = 24.4 + np.degrees(np.cumsum(vn / hz) / 6371000.0)
    lon = 54.6 + np.degrees(np.cumsum(ve / hz) / (6371000.0 * np.cos(np.radians(24.4))))
    df = pd.DataFrame({
        "flight_id": fid, "mission_type": "curved",
        "time_usec": (t * 1e6).astype("int64") + 5_000_000, "t_s": t,
        "latitude_deg": lat, "longitude_deg": lon, "altitude": 50.0 + rng.normal(0, 0.01, n),
        "velocity": speed, "velocity_east": ve, "velocity_north": vn, "velocity_up": vu,
        "pressure_altitude": 51.0 + rng.normal(0, 0.1, n),
        "orientation_x": 0.0, "orientation_y": 0.0, "orientation_z": 0.0, "orientation_w": 1.0,
        "linear_acceleration_x": rng.normal(0, 1, n), "linear_acceleration_y": rng.normal(0, 1, n),
        "linear_acceleration_z": 9.8 + rng.normal(0, 1, n),
        "attack_label": 0.0, "attack_enabled": 0.0, "aligned_all": 1})  # fmt: skip
    for c in df.columns:  # the real base stores everything but these as float32
        if df[c].dtype == np.float64 and c not in (
            "latitude_deg",
            "longitude_deg",
            "t_s",
        ):
            df[c] = df[c].astype(np.float32)
    return df[KEEP]


def spec(typ, params, start=30.0, duration=40.0):
    return {
        "attack_type": typ,
        "start_s": start,
        "duration_s": duration,
        "params": params,
    }


JUMP = {"direction_deg": 90.0, "magnitude_m": 80.0}
DRIFT = {"direction_deg": 0.0, "drift_rate_mps": 1.0}


def test_measured_offset_equals_the_injected_profile():
    g = fake_flight("N")
    out, _ = inject(g, spec("linear_drift", DRIFT), "position_only", "A")
    t = g["t_s"]
    expected = np.where(
        (t >= 30) & (t <= 70), t - 30.0, 0.0
    )  # snaps back to 0 at the end
    assert np.allclose(out["offset_m"], expected, atol=0.01)


def test_only_gps_label_and_truth_columns_change():
    g = fake_flight("N")
    out, _ = inject(g, spec("linear_drift", DRIFT), "consistent", "A")
    changed = {"latitude_deg", "longitude_deg", "velocity", "velocity_east", "velocity_north",
               "attack_label", "attack_enabled", "flight_id"}  # fmt: skip
    for c in KEEP:
        if c not in changed:
            assert g[c].equals(out[c]), c


def test_position_only_leaves_every_velocity_column_untouched():
    g = fake_flight("N")
    out, _ = inject(g, spec("linear_drift", DRIFT), "position_only", "A")
    for c in ("velocity", "velocity_east", "velocity_north", "velocity_up"):
        assert g[c].equals(out[c])


@pytest.mark.parametrize("three_d", [True, False])
def test_consistent_speed_follows_the_components_and_keeps_source_noise(three_d):
    g = fake_flight("N", three_d=three_d)
    assert speed_uses_3d(g) is three_d
    out, _ = inject(
        g,
        spec("linear_drift", {"direction_deg": 90.0, "drift_rate_mps": 2.0}),
        "consistent",
        "A",
    )
    vn, ve, vu = out.velocity_north, out.velocity_east, out.velocity_up
    norm = np.sqrt(vn**2 + ve**2 + vu**2) if three_d else np.hypot(vn, ve)
    src_norm = (np.sqrt(g.velocity_north**2 + g.velocity_east**2 + g.velocity_up**2)
                if three_d else np.hypot(g.velocity_north, g.velocity_east))  # fmt: skip
    assert np.allclose(out.velocity - norm, g.velocity - src_norm, atol=1e-4)
    on = out.attack_label == 1
    assert (out.velocity[on] - g.velocity[on]).abs().max() > 0.5


def test_label_is_the_window_and_truth_follows_the_offset():
    g = fake_flight("N")
    out, _ = inject(g, spec("linear_drift", DRIFT), "position_only", "A")
    t = g["t_s"]
    assert ((out.attack_label == 1) == ((t >= 30) & (t <= 70 + 1e-6))).all()
    assert (out.attack_enabled == out.attack_label).all()
    assert out.loc[t < 30, "phase"].eq(PHASE_PRE).all()
    assert out.loc[out.spoof_truth == 1, "offset_m"].iloc[5:].min() > 0
    assert out.loc[out.phase == 4, "offset_m"].eq(0).all()  # snap-back to zero


def test_abrupt_jump_has_a_one_row_ramp_at_the_onset():
    g = fake_flight("N")
    out, _ = inject(g, spec("abrupt_jump", JUMP), "position_only", "A")
    assert (out.phase == PHASE_RAMP).sum() == 1
    first = int(np.flatnonzero(out.attack_label == 1)[0])
    assert out.phase.iloc[first] == PHASE_RAMP and out.offset_m.iloc[
        first
    ] == pytest.approx(80.0, abs=0.01)


def test_weak_wander_is_labelled_but_not_spoofed():
    g = fake_flight("N")
    p = {
        "amplitude_m": 0.3,
        "periods_s": [10, 20, 40],
        "phases_north": [1, 2, 3],
        "phases_east": [4, 5, 6],
    }
    out, _ = inject(g, spec("low_amplitude_wander", p), "position_only", "A")
    assert out.attack_label.sum() > 0 and out.offset_m.max() < 1.0
    assert out.spoof_truth.sum() <= 1


def test_injection_does_not_mutate_the_source_and_keeps_the_schema():
    g = fake_flight("N")
    before = g.copy()
    out, _ = inject(g, spec("abrupt_jump", JUMP), "consistent", "A")
    pd.testing.assert_frame_equal(g, before)
    assert list(out.columns)[: len(KEEP)] == KEEP and out["flight_id"].eq("A").all()


@pytest.fixture
def project(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    rows, splits = [], []
    for i, sp in enumerate(["train", "train", "validation", "test"]):
        fid = f"curved__Normal__log_{i}"
        fake_flight(fid, duration=60 + 20 * i, seed=i).to_parquet(
            base / f"{fid}.parquet", index=False
        )
        rows.append(
            {
                "flight_id": fid,
                "mission": "curved",
                "attacked": 0,
                "file": f"{fid}.parquet",
            }
        )
        splits.append({"flight_id": fid, "split": sp})
    pd.DataFrame(rows).to_csv(tmp_path / "report.csv", index=False)
    pd.DataFrame(splits).to_csv(tmp_path / "splits.csv", index=False)
    (tmp_path / "attacks.yaml").write_text(yaml.safe_dump(CFG))
    monkeypatch.setattr(gv, "BASE_DIR", base)
    monkeypatch.setattr(gv, "BASE_REPORT", tmp_path / "report.csv")
    monkeypatch.setattr(gv, "SPLITS", tmp_path / "splits.csv")
    monkeypatch.setattr(gv, "CONFIG", tmp_path / "attacks.yaml")
    return tmp_path


def test_run_pairs_the_modes_inherits_splits_and_is_reproducible(project):
    out_dir = project / "out"
    man1 = run(str(out_dir), min_free_gb=0.0)
    first = {f.name: pd.read_parquet(f) for f in (out_dir / "base").glob("*.parquet")}
    man2 = run(str(out_dir), min_free_gb=0.0)
    assert (
        len(man1) == 4 * 5 * 2 + 2
    )  # 4 flights x 5 types x 2 modes + 1 test flight x 2 rates
    assert man1.equals(man2)
    for name, df in first.items():
        pd.testing.assert_frame_equal(df, pd.read_parquet(out_dir / "base" / name))
    main = man1[man1.attack_set == "main"]
    for _, grp in main.groupby("pair_id"):
        assert set(grp.velocity_mode) == {"position_only", "consistent"}
        assert grp["start_s"].nunique() == 1 and grp["params_json"].nunique() == 1
        assert grp[SCALARS].fillna(-1).nunique().max() == 1
    split_of = pd.read_csv(project / "splits.csv").set_index("flight_id")["split"]
    assert (man1["split"] == man1["source_flight_id"].map(split_of)).all()
    assert man1.loc[man1.attack_set == "sweep", "velocity_mode"].eq("consistent").all()
