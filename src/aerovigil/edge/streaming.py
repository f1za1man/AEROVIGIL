r"""Plan M11 (Step F): the two strongest physics detectors, one log row at a time.

Modules 06, 09 and 10 compute their signals over a whole logged flight at once (numpy
and pandas over every row). On board, a detector sees one row at a time and must
answer before the next one arrives, so this file re-writes the two detectors Module 10
fuses as streaming objects with a fixed amount of work per row:

    VelocityStream   Module 03's res_horizontal_velocity_roll_mean: the GPS velocity
                     implied by the position change over the last 1 s, minus the mean
                     reported GPS velocity over the same second, then its trailing 2 s
                     mean. VelocityCheck adds Module 10's z-score test and the 10 s
                     refractory period.
    KalmanStream     Module 06's filter: each IMU row is rotated into East-North-Up and
                     accumulated; every 0.1 s tick runs the horizontal and vertical
                     Kalman filters (predict + Joseph update) on the latest GPS and
                     barometer values. KalmanCheck adds Module 10's NIS test.

Both follow the batch code's rules exactly (lag sample, trailing-window edges, gap
segmentation, tick placement, warm-up), so on a logged flight they give the same
signal row for row; benchmark.py checks this on every real test flight before timing
them. Two settings come from the whole log in the batch code and are inputs here: the
gap length (10 x the median row step) and, for the filter, the first kept row as the
local origin. On board both are fixed in advance from the logging rate and the
take-off point.
"""

import math
from collections import deque

import numpy as np

from aerovigil.kalman.kalman import (
    H_HORIZONTAL,
    H_VERTICAL,
    KalmanFilter,
    horizontal_matrices,
    noise_matrices,
    vertical_matrices,
)

EARTH_R = 6371000.0
LAG_S = 1.0  # Module 03 LAG_SECONDS
ROLL_S = 2.0  # Module 03 ROLL_SECONDS
REFRACTORY_S = 10.0
CONVENTION = {"rotation": "body_to_world", "gravity": "specific_force",
              "frame": "enu"}  # fmt: skip


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class TrailingMean:
    """Mean of the finite values in (t - span, t], times in integer nanoseconds."""

    def __init__(self, span_ns):
        self.span, self.q, self.total, self.n = span_ns, deque(), 0.0, 0

    def push(self, t_ns, v):
        q = self.q
        while q and q[0][0] <= t_ns - self.span:
            _, old = q.popleft()
            if old == old:  # not NaN
                self.total -= old
                self.n -= 1
        q.append((t_ns, v))
        if v == v:
            self.total += v
            self.n += 1
        if not self.n:
            self.total = 0.0  # no drift carried across an all-NaN stretch
            return math.nan
        return self.total / self.n


class VelocityStream:
    """Module 03's res_horizontal_velocity_roll_mean, one row at a time."""

    def __init__(self, gap_s):
        self.gap_s = gap_s
        self.t0_us = None
        self.last_t = None
        self.seg = 0
        self.past = deque()  # (t, lat, lon, segment), oldest first
        self.vn_ref = TrailingMean(int(LAG_S * 1e9))
        self.ve_ref = TrailingMean(int(LAG_S * 1e9))
        self.roll = TrailingMean(int(ROLL_S * 1e9))

    def step(self, t_us, lat, lon, vn, ve):
        if self.t0_us is None:
            self.t0_us = t_us
        t = (t_us - self.t0_us) * 1e-6
        t_ns = (t_us - self.t0_us) * 1000
        if self.last_t is not None and t - self.last_t > self.gap_s:
            self.seg += 1
        self.last_t = t
        past = self.past
        cut = t - LAG_S
        while len(past) >= 2 and past[1][0] <= cut:
            past.popleft()
        vn_ref = self.vn_ref.push(t_ns, vn)
        ve_ref = self.ve_ref.push(t_ns, ve)
        res = math.nan
        if past and past[0][0] <= cut and past[0][3] == self.seg:
            tj, lat_j, lon_j, _ = past[0]
            dt = t - tj
            phi1, phi2 = math.radians(lat_j), math.radians(lat)
            dphi = phi2 - phi1
            dlmb = wrap_pi(math.radians(lon - lon_j))
            vn_pos = dphi * EARTH_R / dt
            ve_pos = dlmb * EARTH_R * math.cos((phi1 + phi2) / 2) / dt
            res = math.hypot(vn_pos - vn_ref, ve_pos - ve_ref)
        past.append((t, lat, lon, self.seg))
        return self.roll.push(t_ns, res)


class VelocityCheck:
    """Module 10's velocity channel: |z| above the threshold, 10 s refractory."""

    def __init__(self, gap_s, mu, sd, threshold):
        self.stream = VelocityStream(gap_s)
        self.mu, self.sd, self.thr = mu, sd, threshold
        self.last_alarm = -math.inf
        self.value = math.nan

    def step(self, t_s, t_us, lat, lon, vn, ve):
        v = self.value = self.stream.step(t_us, lat, lon, vn, ve)
        if v != v or abs((v - self.mu) / self.sd) <= self.thr:
            return False
        if t_s < self.last_alarm + REFRACTORY_S:
            return False
        self.last_alarm = t_s
        return True


def world_accel_row(acc, q, g):
    """One IMU row into East-North-Up, gravity removed (Module 06's convention:
    body_to_world rotation, specific force, ENU)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    ax, ay, az = acc
    e = (
        (1 - 2 * (y * y + z * z)) * ax
        + 2 * (x * y - z * w) * ay
        + 2 * (x * z + y * w) * az
    )
    nn = (
        2 * (x * y + z * w) * ax
        + (1 - 2 * (x * x + z * z)) * ay
        + 2 * (y * z - x * w) * az
    )
    u = (
        2 * (x * z - y * w) * ax
        + 2 * (y * z + x * w) * ay
        + (1 - 2 * (x * x + y * y)) * az
    )
    return e, nn, u - g


class KalmanStream:
    """Module 06's filters, fed one log row at a time; returns the ticks it ran."""

    def __init__(self, fcfg, noise, gap_s, conv=CONVENTION):
        if conv != CONVENTION:
            raise ValueError(f"streaming filter supports {CONVENTION} only")
        self.f = fcfg
        self.r_h, self.r_v = noise_matrices(fcfg, noise)
        self.dt_f = 1.0 / fcfg["rate_hz"]
        self.gap_s = gap_s
        self.origin = None
        self.seg = -1
        self.row = -1

    # ---------------------------------------------------------------- input --
    def step(self, t_us, lat, lon, alt, baro, ve, vn, vu, acc, quat):
        """Feed one complete row (IMU and barometer present). Returns the list of
        ticks run before this row was added: (row index, t, valid, nis_h, nis_v)."""
        if self.origin is None:
            self.origin = (t_us, lat, lon, alt)
        t0_us, lat0, lon0, alt0 = self.origin
        t = (t_us - t0_us) * 1e-6
        e = EARTH_R * math.radians(lon - lon0) * math.cos(math.radians(lat0))
        n = EARTH_R * math.radians(lat - lat0)
        meas = (e, n, ve, vn, alt - alt0, vu, baro - alt0)
        a = world_accel_row(acc, quat, self.f["gravity"])
        self.row += 1
        out = []
        if self.seg < 0 or t - self.last_t > self.gap_s:
            if self.seg >= 0:
                out += self.flush()
            self.start(t, meas)
        else:
            while self.t_seg0 + self.dt_f * self.k < t:
                out += self.tick()
            self.sum[0] += a[0]
            self.sum[1] += a[1]
            self.sum[2] += a[2]
            self.count += 1
        self.last_t, self.last_meas, self.last_row = t, meas, self.row
        return out

    def flush(self):
        """Run the ticks left at the end of a segment (a gap, or the end of the log)."""
        out = []
        n_ticks = math.floor((self.last_t - self.t_seg0) / self.dt_f)
        while self.k <= n_ticks:
            out += self.tick()
        return out

    # ------------------------------------------------------------- internals --
    def start(self, t, meas):
        f = self.f
        e, n, ve, vn, u, vu, baro = meas
        self.kf_h = KalmanFilter(
            [[e, n], [ve, vn], [0.0, 0.0]],
            np.diag([f["p0_pos"] ** 2, f["p0_vel"] ** 2, f["p0_bias"] ** 2]),
        )
        self.kf_v = KalmanFilter(
            [[u], [vu], [0.0], [baro - u]],
            np.diag([f["p0_pos"] ** 2, f["p0_vel"] ** 2, f["p0_bias"] ** 2,
                     f["p0_baro_bias"] ** 2]),
        )  # fmt: skip
        self.seg += 1
        self.t_seg0 = t
        self.k = 1
        self.sum = [0.0, 0.0, 0.0]
        self.count = 0
        self.t_prev = t

    def tick(self):
        """One filter tick at t_seg0 + k * dt; skipped when no new row arrived."""
        self.k += 1
        if not self.count:
            return []
        f = self.f
        t_b = self.last_t
        dt = t_b - self.t_prev
        acc = [s / self.count for s in self.sum]
        e, n, ve, vn, u, vu, baro = self.last_meas
        fh, bh, qh = horizontal_matrices(dt, f["sigma_acc"], f["sigma_bias_walk"])
        self.kf_h.predict(fh, qh, bh, np.array([[acc[0], acc[1]]]))
        _, _, _, nis = self.kf_h.update([[e, n], [ve, vn]], H_HORIZONTAL, self.r_h)
        nis_h = float(nis.sum())
        fv, bv, qv = vertical_matrices(dt, f["sigma_acc"], f["sigma_bias_walk"],
                                       f["sigma_baro_bias_walk"])  # fmt: skip
        self.kf_v.predict(fv, qv, bv, np.array([[acc[2]]]))
        _, _, _, nis = self.kf_v.update([[u], [baro], [vu]], H_VERTICAL, self.r_v)
        valid = (t_b - self.t_seg0) >= f["warmup_s"]
        self.sum = [0.0, 0.0, 0.0]
        self.count = 0
        self.t_prev = t_b
        return [(self.last_row, t_b, valid, nis_h, float(nis[0]))]


class KalmanCheck:
    """Module 10's Kalman channel: horizontal NIS above the threshold on a tick past
    the warm-up, 10 s refractory. Returns the alarm ticks' row indices."""

    def __init__(self, fcfg, noise, gap_s, threshold):
        self.stream = KalmanStream(fcfg, noise, gap_s)
        self.thr = threshold
        self.last_alarm = -math.inf
        self.ticks = []  # the ticks run by the last call

    def check(self, ticks, t_of_row):
        self.ticks = ticks
        out = []
        for row, _, valid, nis_h, _ in ticks:
            t = t_of_row(row)
            if valid and nis_h > self.thr and t >= self.last_alarm + REFRACTORY_S:
                self.last_alarm = t
                out.append(row)
        return out

    def step(self, t_of_row, *row):
        return self.check(self.stream.step(*row), t_of_row)

    def flush(self, t_of_row):
        return self.check(self.stream.flush(), t_of_row)
