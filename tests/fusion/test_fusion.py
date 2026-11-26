"""Module 10 tests: timestamps, budget arithmetic, the latch and cross-validation."""

import json

import numpy as np
import pandas as pd
import pytest

import aerovigil.fusion.fuse as fu
from aerovigil.fusion.signals import (
    CACHE_VERSION,
    Context,
    StaleCache,
    attack_times,
    load,
    monitored_seconds,
    save,
    truth_of,
)
from aerovigil.kalman.pipeline import run_flight

KF_HZ = 10.0


def signal(duration=120.0, row_hz=170.0, attack=None, kf_spike=None, vel_spike=None,
           seed=0):  # fmt: skip
    """A handmade flight in the cached-signal format.

    attack:    (start, end, onset) of the label window, or None for a clean flight
    kf_spike:  (t0, t1, value) raises the Kalman NIS over that interval
    vel_spike: (t0, t1, value) raises the velocity channel over that interval
    """
    rng = np.random.default_rng(seed)
    kf_t = np.arange(0, duration, 1 / KF_HZ)
    nis = rng.chisquare(4, len(kf_t))
    row_t = np.arange(0, duration, 1 / row_hz)
    vel = 0.05 + rng.normal(0, 0.004, len(row_t))
    if kf_spike:
        a, b, v = kf_spike
        nis[(kf_t >= a) & (kf_t <= b)] = v
    if vel_spike:
        a, b, v = vel_spike
        vel[(row_t >= a) & (row_t <= b)] = v
    start, end, onset = attack if attack else (np.nan, np.nan, np.nan)
    offset = np.where(row_t >= (start if attack else np.inf), row_t - start, 0.0)
    return {
        "kf_t": kf_t, "kf_nis": nis, "kf_valid": kf_t >= 5.0,
        "row_t": row_t, "vel": vel, "vel_valid": np.ones(len(row_t), bool),
        "offset_m": offset, "start_s": start, "end_s": end, "onset_s": onset,
        "version": CACHE_VERSION,
    }  # fmt: skip


def row_of(sig, fid="f", **over):
    r = {
        "flight_id": fid, "attacked": bool(np.isfinite(sig["start_s"])),
        "kf_hours": monitored_seconds(sig["kf_t"], sig["kf_valid"]) / 3600,
        "vel_hours": monitored_seconds(sig["row_t"], sig["vel_valid"]) / 3600,
    }  # fmt: skip
    r.update(over)
    return r


# ------------------------------------------------------------- signals.py --


def test_monitored_time_ignores_the_logging_rate_and_skips_gaps():
    for hz in (125.0, 250.0):
        t = np.arange(0, 100, 1 / hz)
        assert monitored_seconds(t, np.ones(len(t), bool)) == pytest.approx(
            100, rel=1e-2
        )
    t = np.r_[np.arange(0, 40, 0.01), np.arange(200, 240, 0.01)]
    assert monitored_seconds(t, np.ones(len(t), bool)) == pytest.approx(80, rel=1e-2)
    v = np.ones(len(t), bool)
    v[: len(t) // 4] = False
    assert monitored_seconds(t, v) < 80


def test_truth_works_on_real_base_flights_without_spoof_columns():
    g = pd.DataFrame({
        "attack_label": [0, 1, 1], "offset_m": [0.0, 1.0, 2.0],
        "seconds_since_onset": [np.nan, 0.0, 0.1], "t_s": [0.0, 0.004, 0.008],
    })  # fmt: skip
    t = truth_of(g)
    assert np.array_equal(t["offset_e"], np.zeros(3))  # absent column, zero filler
    assert np.array_equal(t["offset_m"], [0.0, 1.0, 2.0])


def test_delay_runs_from_the_measured_ramp_not_the_label():
    g = pd.DataFrame({
        "t_s": np.arange(6.0), "attack_label": [0, 1, 1, 1, 1, 0],
        "phase": [0, 1, 2, 3, 3, 4],
    })  # fmt: skip
    assert attack_times(g) == (1.0, 4.0, 2.0)
    clean = g.assign(attack_label=0)
    assert all(np.isnan(x) for x in attack_times(clean))


def test_filter_ticks_are_on_the_base_table_clock_when_the_log_opens_with_dropouts():
    """Real logs open with ~18 IMU dropout rows. The filter drops them and counted
    time from the first row it kept, so its alarms ran 0.1 s early and an alarm at an
    abrupt onset was scored as a false alarm before the attack window."""
    hz, n, lead = 250.0, 250 * 30, 18
    t_s = np.arange(n) / hz
    g = pd.DataFrame({
        "time_usec": (t_s * 1e6).round().astype("int64"),
        "latitude_deg": 33.0 + 1e-6 * np.arange(n), "longitude_deg": np.full(n, 73.0),
        "altitude": np.full(n, 20.0), "pressure_altitude": np.full(n, 20.0),
        "velocity_north": np.full(n, 0.0278), "velocity_east": np.zeros(n),
        "velocity_up": np.zeros(n), "aligned_all": (np.arange(n) >= lead).astype(int),
        "attack_label": np.zeros(n), "offset_m": np.zeros(n),
        "seconds_since_onset": np.full(n, np.nan), "t_s": t_s,
    })  # fmt: skip
    ctx = Context()  # Module 06's real filter configuration
    df = run_flight(g, ctx.cols, ctx.fcfg, None, ctx.scale, ctx.noise, truth_of(g))
    shift = df["row_t_s"].to_numpy() - df["t_s"].to_numpy()
    assert shift == pytest.approx(np.full(len(df), lead / hz))  # the dropout lead
    assert df["row_t_s"].dtype == np.float64
    assert np.isin(df["row_t_s"].to_numpy(), t_s).all()  # exact base-table times


def test_row_times_survive_the_cache_exactly_so_an_onset_alarm_is_inside(tmp_path):
    """In float32, 101.764 is 101.76399994: an alarm on the onset row fell just
    before the window and was scored as a false alarm."""
    th = {"kalman": 1e9, "velocity": 6.0, "fused_kalman": 1e9, "fused_velocity": 6.0,
          "mu": 0.05, "sd": 0.004}  # fmt: skip
    sig = signal(duration=150.0, row_hz=250.0, attack=(101.764, 140.0, 101.764))
    sig["vel"][sig["row_t"] >= sig["start_s"]] = 1.0  # fires on the onset row itself
    save(tmp_path / "f.npz", sig)
    back = load(tmp_path / "f.npz")
    assert back["row_t"].dtype == np.float64
    out = fu.score_flight(back, row_of(back), th)
    assert out["velocity_detected"] and out["velocity_alarms_before"] == 0
    assert out["velocity_delay_s"] == pytest.approx(0.0, abs=1e-9)


def test_a_cache_file_from_a_regenerated_source_is_rejected(tmp_path):
    from aerovigil.fusion.signals import source_stamp

    src = tmp_path / "flight.parquet"
    src.write_bytes(b"first version")
    sig = {**signal(), "source": source_stamp(src)}
    save(tmp_path / "f.npz", sig)
    assert load(tmp_path / "f.npz", source_stamp(src))["kf_t"].size
    src.write_bytes(b"regenerated, same name")
    with pytest.raises(StaleCache):
        load(tmp_path / "f.npz", source_stamp(src))


def test_a_cache_file_from_an_older_version_is_rejected(tmp_path):
    old = {k: v for k, v in signal().items() if k != "version"}
    save(tmp_path / "old.npz", old)
    with pytest.raises(StaleCache):
        load(tmp_path / "old.npz")


# ---------------------------------------------------------------- fuse.py --


def test_refractory_keeps_one_alarm_per_period():
    t = np.arange(0, 35, 0.1)
    assert fu.refractory(t, 10.0).tolist() == pytest.approx([0.0, 10.0, 20.0, 30.0])
    assert len(fu.refractory(np.zeros(0))) == 0


def test_the_threshold_search_finds_the_boundary():
    def count(thr):
        return int(np.sum(np.array([1.0, 2.0, 5.0, 9.0]) > thr))

    thr = fu.lowest_threshold(count, 0.5, 20.0, allowed=1)
    assert 5.0 <= thr < 5.1  # just high enough to leave only the 9
    assert fu.lowest_threshold(count, 0.5, 2.0, allowed=0) == pytest.approx(2.0)


def test_every_detector_keeps_its_budget_including_a_one_alarm_budget():
    """The fused split once gave each channel a floor of one alarm and overspent."""
    sigs = [signal(seed=i) for i in range(4)]
    rows = [row_of(s, f"n{i}") for i, s in enumerate(sigs)]
    th = fu.calibrate(sigs, rows, budget=1.0)
    for det in fu.DETECTORS:
        assert th["used"][det] <= th["allowed"][det]
    assert th["allowed"]["fused"] == 1  # 4 x 2 minutes is under one hour
    assert th["fused_shares"]["kalman"] + th["fused_shares"]["velocity"] <= 1


def test_the_latch_counts_only_alarms_inside_the_window():
    th = {"kalman": 1e9, "velocity": 6.0, "fused_kalman": 1e9, "fused_velocity": 6.0,
          "mu": 0.05, "sd": 0.004}  # fmt: skip
    sig = signal(attack=(40.0, 100.0, 45.0), vel_spike=(50.0, 52.0, 1.0))
    out = fu.score_flight(sig, row_of(sig), th)
    assert out["velocity_detected"] and out["fused_detected"]
    assert not out["kalman_detected"]
    assert out["velocity_delay_s"] == pytest.approx(5.0, abs=0.01)  # 50 s - onset 45 s
    assert out["velocity_displacement_m"] == pytest.approx(10.0, abs=0.05)
    early = signal(attack=(40.0, 100.0, 45.0), vel_spike=(10.0, 12.0, 1.0))
    e = fu.score_flight(early, row_of(early), th)
    assert not e["velocity_detected"] and e["velocity_alarms_before"] == 1


def test_fusion_catches_what_either_channel_catches():
    th = {"kalman": 30.0, "velocity": 6.0, "fused_kalman": 30.0,
          "fused_velocity": 6.0, "mu": 0.05, "sd": 0.004}  # fmt: skip
    by_kf = signal(attack=(40, 100, 40), kf_spike=(60, 61, 99.0))
    by_vel = signal(attack=(40, 100, 40), vel_spike=(70, 71, 1.0))
    for sig in (by_kf, by_vel):
        assert fu.score_flight(sig, row_of(sig), th)["fused_detected"]


def test_poisson_bound_is_above_the_rate_and_shrinks_with_time():
    assert fu.poisson_upper(0, 1.0) == pytest.approx(3.0, rel=0.01)  # the rule of three
    assert fu.poisson_upper(2, 4.0) > 2 / 4.0
    assert fu.poisson_upper(2, 40.0) < fu.poisson_upper(2, 4.0)


def test_folds_are_stratified_by_mission():
    rows = pd.DataFrame({
        "flight_id": [f"{m}{i:02d}" for m in "abc" for i in range(12)],
        "mission_type": [m for m in "abc" for _ in range(12)],
    })  # fmt: skip
    fold = fu.folds_of(rows)
    rows["fold"] = rows["flight_id"].map(fold)
    counts = rows.groupby(["fold", "mission_type"]).size()
    assert len(counts) == fu.FOLDS * 3 and (counts == 2).all()


def test_cross_validation_scores_every_normal_flight_once():
    sigs = {f"{m}{i}": signal(seed=10 * j + i)
            for j, m in enumerate("abc") for i in range(6)}  # fmt: skip
    normal = pd.DataFrame([
        {**row_of(s, fid), "mission_type": fid[0]} for fid, s in sigs.items()
    ])  # fmt: skip
    rows = fu.cross_validated_false_alarms(normal, sigs, budget=1.0)
    for r in rows:
        assert r["flights"] == 18
        assert r["hours"] == pytest.approx(
            sum(fu.hours_of(x, r["detector"]) for _, x in normal.iterrows())
        )
        assert r["upper_95_per_hour"] >= r["alarms_per_hour"]


def test_run_end_to_end_from_the_cache(tmp_path):
    out = tmp_path / "m10"
    (out / "signals").mkdir(parents=True)
    rows = []
    for j, m in enumerate("abc"):
        for i in range(6):
            fid, split = f"{m}_normal_{i}", ("test" if i == 0 else "train")
            s = signal(seed=10 * j + i)
            save(out / "signals" / f"{fid}.npz", s)
            rows.append({**row_of(s, fid), "set": "orig", "split": split,
                         "mission_type": m, "attack_type": "none",
                         "velocity_mode": "na", "drift_rate_mps": np.nan,
                         "file": f"{fid}.npz"})  # fmt: skip
        fid = f"{m}_attack"
        s = signal(attack=(40, 110, 42), vel_spike=(55, 56, 2.0), seed=99 + j)
        save(out / "signals" / f"{fid}.npz", s)
        rows.append({**row_of(s, fid), "set": "orig", "split": "test",
                     "mission_type": m, "attack_type": "original",
                     "velocity_mode": "na", "drift_rate_mps": np.nan,
                     "file": f"{fid}.npz"})  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "signals_index.csv", index=False)
    table, fa, th = fu.run(output_dir=out)
    assert table.loc[0, "attacker"].startswith("Dataset")
    assert table.loc[0, "velocity_detected"] == 1.0
    assert set(fa["scope"]) == {"normal test flights only",
                                "cross-validated, all normal flights"}  # fmt: skip
    saved = json.loads((out / "thresholds.json").read_text())
    assert saved["used"]["fused"] <= saved["allowed"]["fused"]
    for name in ("flights.csv", "threat_table.csv", "false_alarms.csv"):
        assert (out / name).is_file()


def test_save_is_atomic_and_a_truncated_cache_file_is_detected(tmp_path):
    """An interrupted run once left a half-written cache file that crashed the next."""
    from aerovigil.fusion.signals import load

    path = tmp_path / "flight.npz"
    save(path, signal())
    assert not list(tmp_path.glob("*.partial.npz"))  # nothing left behind
    assert load(path)["kf_t"].size > 0
    path.write_bytes(path.read_bytes()[:300])  # what an interruption leaves
    with pytest.raises(Exception):
        load(path)  # signals.run catches this and rebuilds the flight
