"""IMU conventions: rotate body-frame accelerometer data into a local ENU frame.

The dataset does not document its axis and gravity conventions, so they are
tested against the GPS velocity (calibrate_imu.py). A convention is a dict:

    rotation : 'body_to_world' (R @ f)  or 'world_to_body' (R.T @ f)
    gravity  : 'specific_force' (accelerometer includes gravity) or 'linear'
    frame    : 'enu' (world axes East-North-Up) or 'ned' (North-East-Down)
"""
import itertools

import numpy as np

G = 9.80665


def quat_to_matrix(q):
    """q: (n, 4) as (x, y, z, w). Returns (n, 3, 3) rotation matrices (normalised)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def rotate(vec_body, q, rotation):
    """Rotate (n, 3) vectors with the quaternion convention given by `rotation`."""
    r = quat_to_matrix(q)
    if rotation == "world_to_body":
        r = np.transpose(r, (0, 2, 1))
    elif rotation != "body_to_world":
        raise ValueError(f"Unknown rotation convention: {rotation}")
    return np.einsum("nij,nj->ni", r, np.asarray(vec_body, dtype=float))


def world_accel(acc_body, q, conv, g=G):
    """Acceleration in the local ENU frame (m/s^2), gravity removed."""
    a = rotate(acc_body, q, conv["rotation"])
    if conv["frame"] == "enu":
        if conv["gravity"] == "specific_force":
            a[:, 2] -= g                      # f = a - g_vec, g_vec = (0, 0, -g)
    elif conv["frame"] == "ned":
        if conv["gravity"] == "specific_force":
            a[:, 2] += g                      # g_vec = (0, 0, +g) in NED
        a = np.column_stack([a[:, 1], a[:, 0], -a[:, 2]])   # (N, E, D) -> (E, N, U)
    else:
        raise ValueError(f"Unknown frame: {conv['frame']}")
    return a


def all_conventions():
    keys = ("rotation", "gravity", "frame")
    options = (("body_to_world", "world_to_body"), ("specific_force", "linear"), ("enu", "ned"))
    return [dict(zip(keys, combo)) for combo in itertools.product(*options)]


def conv_name(conv):
    return f"{conv['rotation']}/{conv['gravity']}/{conv['frame']}"