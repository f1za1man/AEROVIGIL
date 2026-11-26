"""Module 09 tests: the course-and-velocity consistency detector."""

import json

import numpy as np
import pandas as pd
import pytest

import aerovigil.consistency.evaluate as ev
import aerovigil.consistency.oracle as orc
from aerovigil.adaptive.profile import DEFAULT, profile_fn
from aerovigil.attacks.inject_v2 import KEEP, inject
from aerovigil.consistency.detector import (
    CHANNELS,
    alarm_indices,
    calibrate,
    channel_values,
    fit_standardiser,
    monitored_hours,
    score_rows,
)
from aerovigil.features.build_features_v2 import build_flight_v2

HZ = 125.0
MISSIONS = {"curved": 120.0, "random": 110.0, "straight": 100.0}
TRUTH = (("offset_m", 0.0), ("spoof_truth", 0), ("phase", 0),
         ("seconds_since_onset", np.nan))  # fmt: skip


def nose_forward(fid, mission, duration, seed):
    """A flight whose attitude yaw tracks its course, as a real nose-forward log does."""
    rng = np.random.default_rng(seed)
    n = int(duration * HZ)
    t = np.arange(n) / HZ
    vn = 4.0 + 1.5 * np.sin(t / 25.0)
    ve = 3.0 + 1.5 * np.cos(t / 31.0)
    vu = rng.normal(0, 0.1, n)
    yaw = np.arctan2(vn, ve)
    lat = 24.4 + np.degrees(np.cumsum(vn / HZ) / 6371000.0)
    lon = 54.6 + np.degrees(np.cumsum(ve / HZ) / (6371000.0 * np.cos(np.radians(24.4))))
    df = pd.DataFrame({
        "flight_id": fid, "mission_type": mission,
        "time_usec": (t * 1e6).astype("int64"), "t_s": t,
        "latitude_deg": lat, "longitude_deg": lon,
        "altitude": 50.0 + rng.normal(0, 0.3, n),
        "velocity": np.sqrt(vn**2 + ve**2 + vu**2),
        "velocity_east": ve, "velocity_north": vn, "velocity_up": vu,
        "pressure_altitude": 51.0 + rng.normal(0, 0.3, n),
        "orientation_x": 0.0, "orientation_y": 0.0,
        "orientation_z": np.sin(yaw / 2), "orientation_w": np.cos(yaw / 2),
        "linear_acceleration_x": np.gradient(ve, t) + rng.normal(0, 0.05, n),
        "linear_acceleration_y": np.gradient(vn, t) + rng.normal(0, 0.05, n),
        "linear_acceleration_z": 9.80665 + rng.normal(0, 0.05, n),
        "attack_label": 0.0, "attack_enabled": 0.0, "aligned_all": 1})  # fmt: skip
    return df[KEEP]


def drifted(g, rate, onset, bearing=135.0, mode="consistent", start_fraction=0.25):
    duration = float(g["t_s"].iloc[-1])
    start = duration * start_fraction
    spec = {"attack_type": "adaptive", "start_s": start,
            "duration_s": duration - start, "params": {}}  # fmt: skip
    params = {**DEFAULT, "rate_mps": rate, "onset_s": onset, "duty": 1.0,
              "jitter_m": 0.0, "bearing_deg": bearing}  # fmt: skip
    out, _ = inject(g, spec, mode, "x", profile_fn(params))
    return out


def scored(frame_in, mu=None, sd=None):
    f = build_flight_v2(frame_in)
    values = channel_values(f)
    if mu is None:
        mu, sd = fit_standardiser(values)
    s, v = score_rows(values, mu, sd)
    return f["t_s"].to_numpy(float), s, v, values


@pytest.fixture(scope="module")
def clean_fit():
    """A standardiser from three clean nose-forward flights."""
    flights = [
        nose_forward(f"c{i}", m, d, i) for i, (m, d) in enumerate(MISSIONS.items())
    ]
    stack = []
    for g in flights:
        for c, val in TRUTH:
            g[c] = val
        stack.append(channel_values(build_flight_v2(g)))
    mu, sd = fit_standardiser(np.concatenate(stack))
    return flights, mu, sd


def test_a_slow_onset_hides_the_attack_from_acceleration_but_not_from_course(clean_fit):
    """The finding Module 09 is built on, as a test so it cannot quietly regress."""
    flights, _, _ = clean_fit
    g = flights[0]
    rows = {}
    for onset in (1.5, 20.0):
        f = build_flight_v2(drifted(g, 5.0, onset))
        t = f["t_s"].to_numpy()
        start = float(g["t_s"].iloc[-1]) * 0.25
        mid = (t > start + onset + 20) & (t < t[-1] - 5)
        pre = (t > 10) & (t < start - 5)
        rows[onset] = {
            name: (
                float(np.nanmean(np.abs(f.loc[pre, name]))),
                float(np.nanmean(np.abs(f.loc[mid, name]))),
            )
            for name in ("res_accel_horizontal", "res_course_heading")
        }
    for onset in (1.5, 20.0):
        acc_clean, acc_attack = rows[onset]["res_accel_horizontal"]
        crs_clean, crs_attack = rows[onset]["res_course_heading"]
        assert acc_attack < 2.0 * acc_clean  # acceleration sees nothing, either way
        assert crs_attack > 20.0 * crs_clean  # the course channel sees it, either way
    slow = rows[20.0]["res_course_heading"]
    fast = rows[1.5]["res_course_heading"]
    assert slow[1] > 0.5 * fast[1]  # easing in does not buy the attacker much here


def test_an_along_track_drift_only_partly_evades_the_course_channel(clean_fit):
    """The escape Module 10 will chase, and the reason a fixed bearing cannot take it.

    Aiming the drift along the mean course cuts the course residual by about three
    times, but this flight's course swings as it goes, so a fixed compass bearing is
    along-track only on average and the residual stays far above the clean level. To
    evade properly the attacker has to steer its drift with the victim's course, which
    its current fixed-bearing profile cannot do. That is Module 10's job.
    """
    flights, mu, sd = clean_fit
    g = flights[0]
    cross = build_flight_v2(drifted(g, 5.0, 20.0, bearing=135.0))
    along = build_flight_v2(drifted(g, 5.0, 20.0, bearing=53.0))
    clean = build_flight_v2(g)
    on = cross["attack_label"].to_numpy() == 1
    c = float(np.nanmean(np.abs(cross.loc[on, "res_course_heading"])))
    a = float(np.nanmean(np.abs(along.loc[on, "res_course_heading"])))
    base = float(np.nanmean(np.abs(clean["res_course_heading"])))
    assert c > 3.0 * a  # aiming along-track helps the attacker a lot
    assert a > 20.0 * base  # but a fixed bearing cannot hold it on a turning flight


def test_the_velocity_channel_is_what_sees_a_position_only_spoofer(clean_fit):
    flights, mu, sd = clean_fit
    g = flights[0]
    pos = build_flight_v2(drifted(g, 5.0, 20.0, mode="position_only"))
    con = build_flight_v2(drifted(g, 5.0, 20.0, mode="consistent"))
    on = pos["attack_label"].to_numpy() == 1
    v_pos = float(np.nanmean(np.abs(pos.loc[on, "res_horizontal_velocity"])))
    v_con = float(np.nanmean(np.abs(con.loc[on, "res_horizontal_velocity"])))
    assert v_pos > 5.0 * v_con  # the two channels cover the two kinds of attacker


def test_the_score_is_the_largest_channel_and_survives_a_nan(clean_fit):
    _, mu, sd = clean_fit
    values = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [np.nan, np.nan]])
    s, v = score_rows(values, np.zeros(2), np.ones(2))
    assert s.tolist() == [0.0, 10.0, 10.0, 0.0]
    assert v.tolist() == [True, True, True, False]
    half = score_rows(np.array([[np.nan, 4.0]]), np.zeros(2), np.ones(2))
    assert half[0][0] == 4.0 and bool(half[1][0])  # one live channel still counts


def test_alarms_respect_the_refractory_period():
    t = np.arange(0, 60, 0.5)
    s = np.full(len(t), 9.0)
    v = np.ones(len(t), bool)
    idx = alarm_indices(s, t, v, 1.0, refractory_s=10.0)
    assert len(idx) == 6  # 60 s of continuous exceedance, one alarm every 10 s
    assert idx[0] == 0
    assert not len(alarm_indices(s, t, v, 20.0, refractory_s=10.0))
    assert not len(alarm_indices(s, t, np.zeros(len(t), bool), 1.0))


def test_calibration_fits_the_alarm_budget(clean_fit):
    flights, mu, sd = clean_fit
    clean = []
    for g in flights:
        for c, val in TRUTH:
            g[c] = val
        t, s, v, _ = scored(g, mu, sd)
        clean.append((s, t, v))
    thr, hours, allowed, used = calibrate(clean, budget=1.0)
    assert hours == pytest.approx(sum(monitored_hours(v) for _, _, v in clean))
    assert used <= allowed and allowed >= 1
    assert np.isfinite(thr) and thr > 0


def test_the_detector_catches_the_drift_the_filter_misses(clean_fit):
    """End to end: calibrate on clean flights, then alarm on an eased 5 m/s drift."""
    flights, mu, sd = clean_fit
    clean = []
    for g in flights:
        for c, val in TRUTH:
            g[c] = val
        t, s, v, _ = scored(g, mu, sd)
        clean.append((s, t, v))
    thr, *_ = calibrate(clean, budget=1.0)
    caught = 0
    for g in flights:
        t, s, v, _ = scored(drifted(g, 5.0, 20.0), mu, sd)
        if len(alarm_indices(s, t, v, thr)):
            caught += 1
    assert caught == len(flights)  # Module 06's filter caught none of these


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A fake repo: calibration flights, test flights and a drift sweep."""
    base, m04, interim = tmp_path / "base", tmp_path / "m04", tmp_path / "interim"
    for d in (base, m04, interim):
        d.mkdir(parents=True)
    rows, splits, man = [], [], []
    for i, split in enumerate(["train"] * 4 + ["test"] * 2):
        mission = list(MISSIONS)[i % 3]
        fid = f"{mission}__Normal__log_{i:02d}"
        g = nose_forward(fid, mission, MISSIONS[mission], i)
        for c, val in TRUTH:
            g[c] = val
        g.to_parquet(base / f"{fid}.parquet", index=False)
        rows.append({"flight_id": fid, "mission": mission, "attacked": 0,
                     "file": f"{fid}.parquet"})  # fmt: skip
        splits.append({"flight_id": fid, "split": split})
        if split == "test":
            for rate in (0.5, 5.0):
                aid = f"{fid}__sweep_r{rate:g}".replace(".", "p")
                drifted(
                    nose_forward(fid, mission, MISSIONS[mission], i), rate, 20.0
                ).to_parquet(m04 / f"{aid}.parquet", index=False)
                man.append({
                    "attacked_flight_id": aid, "source_flight_id": fid, "split": split,
                    "mission_type": mission, "attack_set": "sweep",
                    "attack_type": "linear_drift", "velocity_mode": "consistent",
                    "drift_rate_mps": rate, "file": f"{aid}.parquet",
                })  # fmt: skip
    pd.DataFrame(rows).to_csv(interim / "base_report.csv", index=False)
    pd.DataFrame(splits).to_csv(interim / "splits.csv", index=False)
    pd.DataFrame(man).to_csv(interim / "manifest.csv", index=False)
    monkeypatch.setattr(ev, "M03_BASE", base)
    monkeypatch.setattr(ev, "M04_BASE", m04)
    monkeypatch.setattr(ev, "BASE_REPORT", interim / "base_report.csv")
    monkeypatch.setattr(ev, "SPLITS", interim / "splits.csv")
    monkeypatch.setattr(ev, "MANIFEST", interim / "manifest.csv")
    monkeypatch.setattr(ev, "OUT_DIR", tmp_path / "m09")
    return tmp_path


def test_evaluate_runs_and_writes_a_frontier(project):
    alarms, thr = ev.run(output_dir=project / "m09")
    out = project / "m09"
    assert thr > 0
    assert len(alarms) > 0
    cal = json.loads((out / "consistency_calibration.json").read_text())
    assert cal["channels"] == list(CHANNELS)
    assert cal["alarms_used"] <= cal["allowed_alarms"]
    assert len(cal["mu"]) == len(CHANNELS)
    frontier = pd.read_csv(out / "sweep_frontier.csv")
    assert set(frontier["drift_rate_mps"]) == {0.5, 5.0}
    assert (frontier["detection_rate"] > 0.5).all()  # the filter managed 0.048 here
    for name in ("alarms.csv", "summary_by_attack.csv", "false_alarms.csv"):
        assert (out / name).is_file()


def test_the_oracle_gives_the_attacker_only_alarm_feedback(project):
    ev.run(output_dir=project / "m09")
    oracle = orc.ConsistencyOracle(
        calibration=project / "m09" / "consistency_calibration.json"
    )
    g = pd.read_parquet(next((project / "base").glob("*.parquet")))
    r = oracle(g, profile_fn({**DEFAULT, "rate_mps": 5.0, "onset_s": 20.0,
                              "duty": 1.0, "jitter_m": 0.0}))  # fmt: skip
    assert set(r) == set(orc.FAILED)
    assert r["ok"] and oracle.calls == 1
    assert r["detected"]  # an eased 5 m/s cross-track drift does not get past this one
    assert r["achieved_mean_mps"] > 0
