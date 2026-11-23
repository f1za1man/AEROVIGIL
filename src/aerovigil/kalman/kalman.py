"""A small, explicit Kalman filter (written from scratch, cross-checked with FilterPy).

The filter handles several independent channels that share one covariance
(for example the East and North axes, which have identical dynamics and
noise). The state is an (n, c) matrix: n states for each of c channels.
"""

import numpy as np


class KalmanFilter:
    def __init__(self, x0, p0):
        self.x = np.array(x0, dtype=float).reshape(len(p0), -1)
        self.p = np.array(p0, dtype=float)

    def predict(self, f, q, b=None, u=None):
        """x = F x + B u ;  P = F P F' + Q.   u has shape (1, c)."""
        self.x = f @ self.x
        if b is not None and u is not None:
            self.x = self.x + b @ np.asarray(u, dtype=float).reshape(1, -1)
        self.p = f @ self.p @ f.T + q

    def update(self, z, h, r):
        """Joseph-form update. Returns the innovation, its covariance, the
        whitened innovation and the NIS (one value per channel)."""
        z = np.asarray(z, dtype=float).reshape(h.shape[0], -1)
        nu = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ nu
        i_kh = np.eye(len(self.p)) - k @ h
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T
        chol = np.linalg.cholesky(s)
        white = np.linalg.solve(chol, nu)  # (m, c), unit variance if consistent
        nis = (white**2).sum(axis=0)
        return nu, s, white, nis


# ---------------------------------------------------------------- models --
def horizontal_matrices(dt, sig_acc, sig_bias):
    """Per axis state [position, velocity, accel bias]; the IMU acceleration is the input."""
    f = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
    b = np.array([[0.5 * dt * dt], [dt], [0.0]])
    q = sig_acc**2 * np.array(
        [[dt**4 / 4, dt**3 / 2, 0.0], [dt**3 / 2, dt**2, 0.0], [0.0, 0.0, 0.0]]
    )
    q[2, 2] = sig_bias**2 * dt
    return f, b, q


def vertical_matrices(dt, sig_acc, sig_bias, sig_baro_bias):
    """Vertical state [height, velocity, accel bias, barometer bias]."""
    f3, b3, q3 = horizontal_matrices(dt, sig_acc, sig_bias)
    f = np.eye(4)
    f[:3, :3] = f3
    b = np.zeros((4, 1))
    b[:3] = b3
    q = np.zeros((4, 4))
    q[:3, :3] = q3
    q[3, 3] = sig_baro_bias**2 * dt
    return f, b, q


H_HORIZONTAL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
H_VERTICAL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # GPS altitude
        [1.0, 0.0, 0.0, 1.0],  # barometer = height + barometer bias
        [0.0, 1.0, 0.0, 0.0],
    ]
)  # GPS vertical velocity


def noise_matrices(fcfg, scale=1.0):
    """Measurement noise; `scale` multiplies the variances (set by tune_noise.py)."""
    r_h = np.diag([fcfg["sigma_pos_m"] ** 2, fcfg["sigma_vel_mps"] ** 2]) * scale
    r_v = (
        np.diag(
            [
                fcfg["sigma_gps_alt_m"] ** 2,
                fcfg["sigma_baro_m"] ** 2,
                fcfg["sigma_vu_mps"] ** 2,
            ]
        )
        * scale
    )
    return r_h, r_v
