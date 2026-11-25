"""Offset profiles for synthetic GPS-spoofing attacks (AEROVIGIL Module 04).
Every profile returns the spoofed offset in metres (north, east) and its time
derivative in m/s, as functions of tau = seconds since the attack started.
Direction convention: 0 deg = north, 90 deg = east.
"""

import numpy as np

ATTACK_TYPES = (
    "abrupt_jump",
    "constant_offset",
    "linear_drift",
    "exponential_ramp",
    "low_amplitude_wander",
)
SATURATION_KNEE = 0.8


def _saturate(s, ds, cap, knee=SATURATION_KNEE):
    """Bend the offset smoothly onto `cap` instead of clipping it.

    The old hard clip set the rate to zero the instant the offset reached the cap.
    At the cap an exponential ramp is moving at tens of m/s, so that is a step of
    tens of m/s in one sample: in velocity_mode consistent the reported velocity
    jumps, and the Kalman filter in Module 06 sees it immediately. The attack then
    gets detected for switching off, not for spoofing, which is not the thing under
    test.

    Below knee * cap this is the identity, so profiles that never approach the cap
    are bit-identical to before. Above it the offset follows a tanh onto the cap and
    the rate decays to zero continuously, matching value and slope at the knee.
    """
    u = s / cap
    x = np.clip((u - knee) / (1.0 - knee), 0.0, None)
    bent = cap * (knee + (1.0 - knee) * np.tanh(x))
    damp = 1.0 - np.tanh(x) ** 2
    over = u > knee
    return np.where(over, bent, s), np.where(over, ds * damp, ds)


def _unit(direction_deg):
    theta = np.radians(direction_deg)
    return float(np.cos(theta)), float(np.sin(theta))  # north, east


def _wander(tau, p):
    amp = p["amplitude_m"]
    periods = np.asarray(p["periods_s"], dtype=float)
    k = len(periods)
    w = 2.0 * np.pi / periods

    def component(phases):
        phases = np.asarray(phases, dtype=float)
        arg = np.outer(tau, w) + phases
        s = (np.sin(arg) - np.sin(phases)).sum(axis=1) * amp / k  # s(0) = 0
        ds = (np.cos(arg) * w).sum(axis=1) * amp / k
        return s, ds

    n, dn = component(p["phases_north"])
    e, de = component(p["phases_east"])
    return n, e, dn, de


def offset_profile(attack_type, tau, p):
    """Return (north_m, east_m, d_north_mps, d_east_mps) for tau >= 0."""
    tau = np.asarray(tau, dtype=float)
    if attack_type == "low_amplitude_wander":
        return _wander(tau, p)
    if attack_type == "abrupt_jump":
        s = np.full_like(tau, p["magnitude_m"])
        ds = np.zeros_like(tau)
    elif attack_type == "constant_offset":
        ramp = p["ramp_seconds"]
        s = p["magnitude_m"] * np.minimum(tau / ramp, 1.0)
        ds = np.where(tau < ramp, p["magnitude_m"] / ramp, 0.0)
    elif attack_type == "linear_drift":
        s = p["drift_rate_mps"] * tau
        ds = np.full_like(tau, p["drift_rate_mps"])
    elif attack_type == "exponential_ramp":
        r0, tc, cap = p["initial_rate_mps"], p["time_constant_s"], p["max_offset_m"]
        x = np.minimum(tau / tc, 50.0)
        s = r0 * tc * (np.exp(x) - 1.0)
        ds = r0 * np.exp(x)
        s, ds = _saturate(s, ds, cap)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    un, ue = _unit(p["direction_deg"])
    return s * un, s * ue, ds * un, ds * ue
