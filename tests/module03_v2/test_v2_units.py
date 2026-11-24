"""Module 03 v2 unit tests that need no dataset."""

import numpy as np
import pandas as pd

from aerovigil.data.build_raw_base import (
    PHASE_NORMAL,
    PHASE_PLATEAU,
    PHASE_POST,
    PHASE_PRE,
    PHASE_RAMP,
    horizontal_offset_m,
    label_phases,
)
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.windowing.make_windows_v2 import window_class


def synthetic_attack(n=4000, onset=1000, ramp=500, end=3000, label_end=3100):
    t = np.arange(n) * 0.004
    offset = np.zeros(n)
    offset[onset : onset + ramp] = np.linspace(0, 10, ramp)
    offset[onset + ramp : end] = 10.0
    label = np.zeros(n)
    label[onset:label_end] = 1
    return t, offset, label


def test_phases_follow_the_measured_offset():
    t, offset, label = synthetic_attack()
    phase, truth, since, facts = label_phases(offset, label, t, True)
    assert phase[0] == PHASE_PRE and phase[999] == PHASE_PRE
    assert phase[1000] == PHASE_RAMP and phase[1400] == PHASE_RAMP
    assert phase[1700] == PHASE_PLATEAU and phase[2999] == PHASE_PLATEAU
    assert phase[3500] == PHASE_POST
    assert truth[1000:3000].all() and not truth[:1000].any()
    assert np.isclose(facts["plateau_m"], 10.0)
    assert since[1000] == 0.0


def test_phase_order_never_goes_backwards():
    t, offset, label = synthetic_attack()
    phase, *_ = label_phases(offset, label, t, True)
    assert (np.diff(phase.astype(int)) >= 0).all()


def test_normal_flight_has_only_phase_zero():
    t, _offset, label = synthetic_attack()
    phase, truth, since, _ = label_phases(np.zeros(len(t)), 0 * label, t, False)
    assert (phase == PHASE_NORMAL).all() and not truth.any() and np.isnan(since).all()


def test_label_that_ends_early_is_not_spoof_truth():
    t, offset, label = synthetic_attack(end=3000, label_end=3100)
    _, truth, _, facts = label_phases(offset, label, t, True)
    assert facts["label_end_s"] > facts["spoof_end_s"]
    assert not truth[3001:].any()


def test_offset_in_metres():
    d = horizontal_offset_m(47.0 + 10 / 111320.0, 8.0, 47.0, 8.0)
    assert abs(float(d) - 10.0) < 1e-6


def test_window_classes():
    assert window_class(np.full(5, 0)) == "normal"
    assert window_class(np.full(5, 1)) == "pre"
    assert window_class(np.array([1, 1, 2, 2, 3])) == "ramp"
    assert window_class(np.full(5, 3)) == "plateau"
    assert window_class(np.array([3, 3, 3, 4, 4])) == "snapback"
    assert window_class(np.full(5, 4)) == "post"


def flight_frame(velocity_jump=0.0):
    n = 5000
    t = np.arange(n) * 0.004
    ve = 0.5 * t
    ve = np.where(t >= 10, ve + velocity_jump, ve)
    frame = pd.DataFrame(
        {
            "flight_id": "f",
            "mission_type": "straight",
            "time_usec": (t * 1e6).astype("int64"),
            "t_s": t,
            "latitude_deg": 47.0,
            "longitude_deg": 8.0 + 0.25 * t**2 / (111320 * np.cos(np.radians(47))),
            "altitude": 500.0,
            "pressure_altitude": 499.0,
            "velocity_east": ve,
            "velocity_north": 0.0,
            "velocity_up": 0.0,
            "attack_label": 0.0,
            "orientation_x": 0.0,
            "orientation_y": 0.0,
            "orientation_z": 0.0,
            "orientation_w": 1.0,
            "linear_acceleration_x": 0.5,
            "linear_acceleration_y": 0.0,
            "linear_acceleration_z": 9.80665,
            "offset_m": 0.0,
            "spoof_truth": 0,
            "phase": 0,
            "seconds_since_onset": np.nan,
            "aligned_all": 1,
        }
    )
    return frame


def test_imu_residuals_are_zero_for_consistent_motion():
    out = build_flight_v2(flight_frame())
    late = out[out["t_s"] > 3]
    assert late["res_accel_horizontal"].abs().max() < 1e-6
    assert late["res_accel_vertical"].abs().max() < 1e-6
    assert late["res_course_heading"].abs().max() < 1e-6


def test_imu_residual_sees_a_velocity_jump():
    out = build_flight_v2(flight_frame(velocity_jump=3.0))
    jump = out[(out["t_s"] >= 10) & (out["t_s"] < 11.5)]
    assert jump["res_accel_horizontal"].max() > 2.5
