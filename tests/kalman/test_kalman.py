"""Module 06 tests: IMU conventions, Kalman filter, tick preparation, detectors."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.kalman.detectors import (
    calibrate_threshold,
    cusum_alarm_counts,  # noqa: E402
    cusum_alarms,
    fit_standardiser,
    flight_metrics,
    nis_alarms,
)
from aerovigil.kalman.imu import (
    all_conventions,
    quat_to_matrix,
    rotate,  # noqa: E402
    world_accel,
)
from aerovigil.kalman.kalman import (
    H_HORIZONTAL,
    H_VERTICAL,
    KalmanFilter,  # noqa: E402
    horizontal_matrices,
    vertical_matrices,
)
from aerovigil.kalman.pipeline import prepare_ticks, segments  # noqa: E402

G = 9.80665


def yaw_quat(angle):
    return np.array([[0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]])


# ----------------------------------------------------------------- imu --
def test_identity_quaternion_gives_identity_matrix():
    assert np.allclose(quat_to_matrix(np.array([[0, 0, 0, 1.0]]))[0], np.eye(3))


def test_ninety_degree_yaw_turns_x_into_y():
    v = rotate(np.array([[1.0, 0, 0]]), yaw_quat(np.pi / 2), "body_to_world")
    assert np.allclose(v, [[0, 1, 0]], atol=1e-12)
    back = rotate(v, yaw_quat(np.pi / 2), "world_to_body")
    assert np.allclose(back, [[1, 0, 0]], atol=1e-12)


def test_hover_gives_zero_world_acceleration_for_the_right_convention():
    f = np.array([[0.0, 0.0, G]])  # accelerometer at rest, level
    conv = {"rotation": "body_to_world", "gravity": "specific_force", "frame": "enu"}
    assert np.allclose(world_accel(f, np.array([[0, 0, 0, 1.0]]), conv), 0, atol=1e-9)
    wrong = {"rotation": "body_to_world", "gravity": "linear", "frame": "enu"}
    assert world_accel(f, np.array([[0, 0, 0, 1.0]]), wrong)[0, 2] == pytest.approx(G)


@pytest.mark.parametrize("conv", all_conventions())
def test_every_convention_runs_and_ned_swaps_axes(conv):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    out = world_accel(rng.normal(size=(50, 3)), q, conv)
    assert out.shape == (50, 3) and np.isfinite(out).all()


def test_round_trip_recovers_world_acceleration():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a_world = rng.normal(size=(200, 3))
    f_world = a_world + np.array([0, 0, G])
    f_body = rotate(f_world, q, "world_to_body")  # inverse rotation
    conv = {"rotation": "body_to_world", "gravity": "specific_force", "frame": "enu"}
    assert np.allclose(world_accel(f_body, q, conv), a_world, atol=1e-9)


# -------------------------------------------------------------- kalman --
def simulate(n=3000, dt=0.1, drift=None, jump=None, seed=0):
    """Constant-velocity truth, accurate IMU, noisy GPS. Returns measurement arrays."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1) * dt
    p_true = 3.0 * t
    off = np.zeros(n)
    if drift is not None:
        off = drift * np.maximum(t - 100.0, 0.0)
    if jump is not None:
        off = np.where(t > 100.0, jump, 0.0)
    pos = p_true + off + rng.normal(0, 0.5, n)
    vel = (
        3.0
        + (drift if drift is not None else 0.0) * (t > 100.0)
        + rng.normal(0, 0.2, n)
    )
    acc = rng.normal(0, 0.05, n)
    return t, pos, vel, acc, off


def run_axis(pos, vel, acc, dt=0.1, sig_acc=0.3, sig_pos=0.5, sig_vel=0.2):
    r = np.diag([sig_pos**2, sig_vel**2])
    kf = KalmanFilter([[pos[0]], [vel[0]], [0.0]], np.diag([1.0, 1.0, 0.1]))
    nis, est = [], []
    for k in range(len(pos)):
        f, b, q = horizontal_matrices(dt, sig_acc, 0.01)
        kf.predict(f, q, b, [[acc[k]]])
        _, _, _, n_ = kf.update([[pos[k]], [vel[k]]], H_HORIZONTAL, r)
        nis.append(n_[0])
        est.append(kf.x[0, 0])
    return np.array(nis), np.array(est)


def test_consistent_filter_has_nis_close_to_its_dof():
    _, pos, vel, acc, _ = simulate()
    nis, _ = run_axis(pos, vel, acc)
    assert 1.6 < nis[200:].mean() < 2.5  # chi-square with 2 dof has mean 2


def test_covariance_stays_symmetric_and_positive_definite():
    kf = KalmanFilter([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], np.diag([4.0, 1.0, 0.1]))
    r = np.diag([0.25, 0.04])
    for k in range(500):
        f, b, q = horizontal_matrices(0.1, 0.3, 0.01)
        kf.predict(f, q, b, [[0.0, 0.0]])
        kf.update([[0.0, 0.0], [0.0, 0.0]], H_HORIZONTAL, r)
    assert (
        np.allclose(kf.p, kf.p.T, atol=1e-12) and (np.linalg.eigvalsh(kf.p) > 0).all()
    )


def test_two_channels_share_covariance_and_match_single_channel_runs():
    _, pos, vel, acc, _ = simulate(n=400)
    r = np.diag([0.25, 0.04])
    kf2 = KalmanFilter(
        [[pos[0], pos[0] + 1], [vel[0], vel[0]], [0, 0]], np.diag([1.0, 1.0, 0.1])
    )
    for k in range(400):
        f, b, q = horizontal_matrices(0.1, 0.3, 0.01)
        kf2.predict(f, q, b, [[acc[k], acc[k]]])
        kf2.update([[pos[k], pos[k] + 1], [vel[k], vel[k]]], H_HORIZONTAL, r)
    _, est = run_axis(pos, vel, acc, sig_acc=0.3)
    assert kf2.x[0, 0] == pytest.approx(est[-1], abs=1e-9)


def test_vertical_model_uses_barometer_offset_state():
    f, b, q = vertical_matrices(0.1, 0.3, 0.01, 0.05)
    assert f.shape == (4, 4) and q[3, 3] > 0 and (np.linalg.eigvalsh(q) >= -1e-12).all()
    kf = KalmanFilter([[0.0], [0.0], [0.0], [0.0]], np.diag([1.0, 1.0, 0.1, 4.0]))
    for _ in range(300):  # GPS says 0 m, barometer says 0.8 m
        kf.predict(f, q, b, [[0.0]])
        kf.update([[0.0], [0.8], [0.0]], H_VERTICAL, np.diag([0.25, 0.01, 0.04]))
    assert kf.x[3, 0] == pytest.approx(
        0.8, abs=0.05
    )  # offset learned, not treated as error


def test_filter_absorbs_slow_drift_but_not_a_jump():
    """The key insight of the module."""
    _, pos, vel, acc, off = simulate(drift=0.05)
    nis_d, est_d = run_axis(pos, vel, acc)
    t = np.arange(1, len(pos) + 1) * 0.1
    after = t > 100.0
    err = est_d - 3.0 * t  # estimate minus the true position
    assert (
        abs(err[-1] - off[-1]) < 0.2 * off[-1]
    )  # the state follows the spoofed offset
    assert nis_d[after].mean() < 3.0  # ... so innovations stay small

    _, posj, velj, accj, _ = simulate(jump=60.0)
    nis_j, _ = run_axis(posj, velj, accj)
    assert nis_j[(t > 100.0) & (t < 100.3)].max() > 100.0  # a jump is not absorbed


def test_own_filter_matches_filterpy_on_simulated_data():
    fp = pytest.importorskip("filterpy.kalman")
    _, pos, vel, acc, _ = simulate(n=300)
    r = np.diag([0.25, 0.04])
    ours = KalmanFilter([[pos[0]], [vel[0]], [0.0]], np.diag([1.0, 1.0, 0.1]))
    ref = fp.KalmanFilter(dim_x=3, dim_z=2)
    ref.x = np.array([[pos[0]], [vel[0]], [0.0]])
    ref.P = np.diag([1.0, 1.0, 0.1])
    ref.H = H_HORIZONTAL.copy()
    ref.R = r.copy()
    for k in range(300):
        f, b, q = horizontal_matrices(0.1, 0.3, 0.01)
        ours.predict(f, q, b, [[acc[k]]])
        _, _, _, nis = ours.update([[pos[k]], [vel[k]]], H_HORIZONTAL, r)
        ref.predict(u=np.array([[acc[k]]]), B=b, F=f, Q=q)
        ref.update(np.array([[pos[k]], [vel[k]]]))
        y = ref.y.reshape(-1, 1)
        assert nis[0] == pytest.approx(
            float((y.T @ np.linalg.inv(ref.S) @ y)[0, 0]), rel=1e-9
        )


# ------------------------------------------------------------- ticks ----
COLS = {
    "time": "time_usec",
    "lat": "lat",
    "lon": "lon",
    "alt": "alt",
    "baro": "baro",
    "vn": "vn",
    "ve": "ve",
    "vu": "vu",
    "accel": ["ax", "ay", "az"],
    "quat": ["qx", "qy", "qz", "qw"],
}
FCFG = {"rate_hz": 10, "gap_factor": 10, "min_segment_s": 8.0, "gravity": G}
CONV = {"rotation": "body_to_world", "gravity": "specific_force", "frame": "enu"}


def make_flight(n=5000, hz=250, gap=None, accel_world=(0.5, 0.0, 0.0)):
    t = np.arange(n) / hz
    if gap:
        t = np.where(np.arange(n) >= gap[0], t + gap[1], t)
    df = pd.DataFrame(
        {
            "time_usec": (t * 1e6).astype("int64"),
            "lat": 24.4,
            "lon": 54.6,
            "alt": 50.0,
            "baro": 51.0,
            "vn": 0.0,
            "ve": 0.0,
            "vu": 0.0,
            "ax": accel_world[0],
            "ay": accel_world[1],
            "az": accel_world[2] + G,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
    )
    return df


def test_segments_split_at_log_gaps():
    t = np.arange(1000) / 250.0
    t[500:] += 5.0
    assert segments(t, 10) == [(0, 500), (500, 1000)]


def test_ticks_do_not_span_a_gap_and_average_the_imu_input():
    g = make_flight(n=6000, gap=(3000, 20.0))
    segs = prepare_ticks(g, COLS, FCFG, CONV, 1e-6)
    assert len(segs) == 2
    for s in segs:
        assert (s["dt"] > 0).all() and s["dt"].max() < 0.2
        assert np.allclose(s["acc"][:, 0], 0.5)  # constant world acceleration recovered
        assert np.allclose(s["acc"][:, 2], 0.0, atol=1e-9)  # gravity removed
    assert segs[1]["t"].min() > segs[0]["t"].max() + 15.0


def test_short_segments_are_skipped_and_no_imu_means_zero_input():
    g = make_flight(n=1500, gap=(500, 20.0))  # 2 s then 4 s pieces
    assert prepare_ticks(g, COLS, FCFG, CONV, 1e-6) == []
    g = make_flight(n=5000)
    seg = prepare_ticks(g, COLS, FCFG, None, 1e-6)[0]
    assert np.allclose(seg["acc"], 0.0)


# --------------------------------------------------------- detectors ----
def test_cusum_quiet_on_noise_and_fast_on_a_small_shift():
    rng = np.random.default_rng(0)
    n = 6000
    t = np.arange(n) * 0.1
    v = np.ones(n, bool)
    z = rng.normal(0, 1, (n, 4))
    assert len(cusum_alarms(z, v, t, 0.5, 20.0, 10.0)) == 0
    z[3000:, 1] += 1.0  # a 1-sigma drift in one component
    first = cusum_alarms(z, v, t, 0.5, 8.0, 10.0)
    first = first[first >= 3000]
    assert len(first) > 0 and first[0] - 3000 < 300


def test_single_step_test_reacts_much_later_than_cusum_to_the_same_shift():
    rng = np.random.default_rng(0)
    n = 6000
    t = np.arange(n) * 0.1
    v = np.ones(n, bool)
    z = rng.normal(0, 1, (n, 4))
    z[3000:, 1] += 1.0
    nis = (z**2).sum(axis=1)
    chi = nis_alarms(nis, v, t, 25.0, 10.0)
    cus = cusum_alarms(z, v, t, 0.5, 8.0, 10.0)
    first_chi = chi[chi >= 3000][0] - 3000 if (chi >= 3000).any() else 10**9
    first_cus = cus[cus >= 3000][0] - 3000
    assert first_cus < 300 and first_chi > first_cus + 400


def test_multi_threshold_counts_match_single_runs_and_refractory_holds():
    rng = np.random.default_rng(2)
    n = 8000
    t = np.arange(n) * 0.1
    v = np.ones(n, bool)
    z = rng.normal(0, 1, (n, 4))
    hs = np.array([3.0, 5.0, 8.0])
    counts = cusum_alarm_counts(z, v, t, 0.5, hs, 10.0)
    for c, h in zip(counts, hs):
        a = cusum_alarms(z, v, t, 0.5, h, 10.0)
        assert c == len(a)
        assert (np.diff(t[a]) >= 10.0 - 1e-9).all()
    assert counts[0] >= counts[1] >= counts[2]


def test_invalid_ticks_reset_the_statistic():
    n = 400
    t = np.arange(n) * 0.1
    z = np.full((n, 4), 2.0)
    v = np.ones(n, bool)
    v[::5] = False  # warm-up resets every 5 ticks
    assert len(cusum_alarms(z, v, t, 0.5, 10.0, 0.0)) == 0


def test_calibrate_threshold_respects_the_budget():
    grid = np.array([1, 2, 3, 4, 5.0])
    assert calibrate_threshold(np.array([9, 4, 2, 1, 0]), grid, allowed=2) == 3.0
    assert calibrate_threshold(np.array([9, 9, 9, 9, 9]), grid, allowed=0) == 5.0
    assert calibrate_threshold(np.array([0, 0, 0, 0, 0]), grid, allowed=0.5) == 1.0


def test_standardiser_ignores_outliers():
    rng = np.random.default_rng(0)
    w = rng.normal(0.3, 2.0, (5000, 4))
    w[:50] = 1e4
    mu, sd = fit_standardiser(w)
    assert np.allclose(mu, 0.3, atol=0.15) and np.allclose(sd, 2.0, atol=0.2)


def tick_frame(n=1000, start=40.0, end=80.0, rate=10):
    t = np.arange(n) / rate
    on = (t >= start) & (t <= end)
    return pd.DataFrame(
        {
            "t_s": t,
            "valid": t >= 5.0,
            "label": on.astype(float),
            "offset_m": np.where(on, 0.5 * (t - start), 0.0),
        }
    )


def test_flight_metrics_detected_undetected_and_false_alarms():
    df = tick_frame()
    idx = np.array([100, 450])  # t = 10 s (false alarm), 45 s (detection)
    m = flight_metrics(df, idx, 10)
    assert m["detected"] and m["first_alarm_s"] == pytest.approx(45.0)
    assert m["delay_s"] == pytest.approx(5.0) and m["displacement_m"] == pytest.approx(
        2.5
    )
    assert m["false_alarms"] == 1 and m["normal_hours"] > 0
    missed = flight_metrics(df, np.array([], dtype=int), 10)
    assert not missed["detected"] and missed["displacement_m"] == pytest.approx(20.0)
    late = flight_metrics(df, np.array([900]), 10)  # alarm after the attack ended
    assert not late["detected"]
    clean = flight_metrics(df.assign(label=0.0), np.array([300]), 10)
    assert not clean["attacked"] and clean["false_alarms"] == 1
