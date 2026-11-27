"""Plan M12 (Step G): dead reckoning, magnetometer heading, survival curves."""

import numpy as np
import pandas as pd
import pytest

from aerovigil.kalman.pipeline import filter_segment
from aerovigil.mitigation import fallback as fb

FCFG = {"rate_hz": 10, "warmup_s": 5.0, "min_segment_s": 8.0, "gap_factor": 10,
        "gravity": 9.80665, "sigma_pos_m": 1.0, "sigma_vel_mps": 0.3,
        "sigma_gps_alt_m": 1.0, "sigma_baro_m": 0.3, "sigma_vu_mps": 0.3,
        "sigma_acc": 0.18, "sigma_acc_no_imu": 2.0, "sigma_bias_walk": 0.02,
        "sigma_baro_bias_walk": 0.05, "p0_pos": 2.0, "p0_vel": 1.0, "p0_bias": 0.2,
        "p0_baro_bias": 2.0}  # fmt: skip
NOISE = {"pos": 0.04, "vel": 0.0002, "vert": 0.016}


def segment(n=600, ve=2.0, vn=-1.0, acc=(0.0, 0.0, 0.0), seed=0):
    """A prepared filter segment (prepare_ticks' layout) of steady flight."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1) * 0.1
    return {
        "t": t, "dt": np.full(n, 0.1), "acc": np.tile(acc, (n, 1)).astype(float),
        "e": ve * t + rng.normal(0, 0.3, n), "n": vn * t + rng.normal(0, 0.3, n),
        "ve": ve + rng.normal(0, 0.05, n), "vn": vn + rng.normal(0, 0.05, n),
        "u": np.full(n, 20.0), "vu": np.zeros(n), "baro": np.full(n, 20.0),
        "init": {"e": 0.0, "n": 0.0, "ve": ve, "vn": vn, "u": 20.0, "vu": 0.0,
                 "baro": 20.0},
    }  # fmt: skip


def test_run_normal_is_module06_filter():
    seg = segment()
    est, states = fb.run_normal(seg, FCFG, NOISE)
    ref = filter_segment(seg, FCFG, NOISE)["est"]
    np.testing.assert_array_equal(est[:, :2], ref)
    assert len(states) == len(seg["t"]) + 1


def test_coasting_on_steady_flight_extrapolates_the_velocity():
    seg = segment()
    _, states = fb.run_normal(seg, FCFG, NOISE)
    k = 300
    dr = fb.coast(seg, states, FCFG, NOISE, k, len(seg["t"]))
    x0, v0, bias = states[k][0]  # position, velocity, learned accelerometer bias
    tt = np.arange(1, len(dr) + 1) * 0.1
    for i in (0, 1):  # no input: the learned bias is the only acceleration left
        want = x0[i] + v0[i] * tt - 0.5 * bias[i] * tt**2
        np.testing.assert_allclose(dr[:, i], want, atol=1e-6)
    assert np.abs(dr[:, 2] - 20.0).max() < 0.5  # height held by the barometer


def test_hold_ignores_the_accelerometer():
    seg = segment()
    _, states = fb.run_normal(seg, FCFG, NOISE)
    seg["acc"][300:, 0] = 0.5  # the drone speeds up East once GPS is gone
    imu = fb.coast(seg, states, FCFG, NOISE, 300, 600)
    hold = fb.coast(seg, states, FCFG, NOISE, 300, 600, "hold")
    gain = imu[-1, 0] - hold[-1, 0]
    assert gain == pytest.approx(0.5 * 0.5 * 30**2, rel=0.02)  # imu integrates it


def test_heading_fix_is_zero_before_the_switch_and_undoes_a_turn():
    seg = {"psi": np.r_[np.full(200, 0.3), np.full(100, 0.5)],
           "t_s": np.arange(300) * 0.1}  # fmt: skip
    fix = fb.heading_fix(seg, 150)
    assert (fix[:150] == 0).all()
    assert fix[150:200] == pytest.approx(0.0, abs=1e-12)
    assert fix[200:] == pytest.approx(-0.2)


def test_field_direction_uses_px4_body_axes():
    # level, nose east (identity attitude: FLU = ENU); a field pointing north and
    # down reads (x fwd, y right, z down) = (0, -B_n, +B_d) in PX4's body axes
    g = pd.DataFrame({"magnetic_field_x": [0.0], "magnetic_field_y": [-0.2],
                      "magnetic_field_z": [0.4], "qx": [0.0], "qy": [0.0],
                      "qz": [0.0], "qw": [1.0]})  # fmt: skip
    psi = fb.field_direction(g, ["qx", "qy", "qz", "qw"])
    assert psi[0] == pytest.approx(np.pi / 2)


def test_kaplan_meier_with_and_without_censoring():
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s = fb.kaplan_meier([1, 2, 3, 4], [True] * 4, grid)
    np.testing.assert_allclose(s, [1, 0.75, 0.5, 0.25, 0.0])
    # a run censored at 1.5 leaves 2 at risk at t = 2: 0.75 * 1/2
    s = fb.kaplan_meier([1, 1.5, 2, 3], [True, False, True, True], grid)
    np.testing.assert_allclose(s, [1, 0.75, 0.375, 0.0, np.nan])
    assert fb.km_quantile([1, 2, 3, 4], [True] * 4, 0.5) == 2.0
    assert np.isnan(fb.km_quantile([5, 6], [False, False], 0.5))


def test_first_exceed_and_crossover():
    since = np.arange(0, 20, 0.5)
    err = since * 2.0
    assert fb.first_exceed(since, err, 10.0) == (5.5, True)
    assert fb.first_exceed(since, err, 100.0) == (19.5, False)
    without = np.full(len(since), 8.0)
    blip = err.copy()
    blip[2] = 50.0  # one bad tick is not a crossover
    assert fb.crossover(since, blip, without) == (4.5, True)
    assert fb.crossover(since, np.zeros(len(since)), without) == (19.5, False)
