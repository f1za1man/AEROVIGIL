r"""AEROVIGIL Module 03 v2: IMU-versus-GPS consistency residuals for one flight.

Uses the IMU convention that Module 06 validated (body_to_world / specific_force /
enu). Returns three residual series; build_features_v2.py turns them into features.

    res_accel_horizontal : |IMU acceleration (East, North) - acceleration implied by
                           the change of the reported GPS velocity|
    res_accel_vertical   : the same for the vertical axis
    res_course_heading   : sin(GPS course - IMU yaw), faded out below 2 m/s
"""

import numpy as np
import pandas as pd

from aerovigil.features.build_features import LAG_SECONDS, trailing, wrap_pi
from aerovigil.kalman.imu import quat_to_matrix, world_accel

CONVENTION = {"rotation": "body_to_world", "gravity": "specific_force", "frame": "enu"}
VELOCITY_SMOOTH_S = 0.2
COURSE_FULL_SPEED = 2.0
IMU_INPUTS = [
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
]


def imu_features(g, t, j, ok, dt):
    """Return a dict of IMU residual arrays for one flight (NaN where no IMU row)."""
    n = len(g)
    valid = g[IMU_INPUTS].notna().all(axis=1).to_numpy()
    a_world = np.full((n, 3), np.nan)
    yaw = np.full(n, np.nan)
    if valid.any():
        q = g.loc[valid, IMU_INPUTS[:4]].to_numpy(float)
        acc = g.loc[valid, IMU_INPUTS[4:]].to_numpy(float)
        a_world[valid] = world_accel(acc, q, CONVENTION)
        rot = quat_to_matrix(q)
        yaw[valid] = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    idx = pd.to_timedelta(t, unit="s")
    imu_e = trailing(a_world[:, 0], idx, LAG_SECONDS, "mean")
    imu_n = trailing(a_world[:, 1], idx, LAG_SECONDS, "mean")
    imu_u = trailing(a_world[:, 2], idx, LAG_SECONDS, "mean")

    def smooth(col):
        return trailing(g[col].to_numpy(float), idx, VELOCITY_SMOOTH_S, "mean")

    ve, vn, vu = (
        smooth("velocity_east"),
        smooth("velocity_north"),
        smooth("velocity_up"),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        gps_ae = np.where(ok, (ve - ve[j]) / dt, np.nan)
        gps_an = np.where(ok, (vn - vn[j]) / dt, np.nan)
        gps_au = np.where(ok, (vu - vu[j]) / dt, np.nan)
    ve_ref = trailing(g["velocity_east"].to_numpy(float), idx, LAG_SECONDS, "mean")
    vn_ref = trailing(g["velocity_north"].to_numpy(float), idx, LAG_SECONDS, "mean")
    course = np.arctan2(vn_ref, ve_ref)
    fade = np.clip(np.hypot(ve_ref, vn_ref) / COURSE_FULL_SPEED, 0.0, 1.0)
    return {
        "res_accel_horizontal": (np.hypot(imu_e - gps_ae, imu_n - gps_an), False),
        "res_accel_vertical": (imu_u - gps_au, True),
        "res_course_heading": (np.sin(wrap_pi(course - yaw)) * fade, True),
    }
