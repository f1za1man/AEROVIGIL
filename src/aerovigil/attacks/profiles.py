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
        capped = s >= cap
        s = np.where(capped, cap, s)
        ds = np.where(capped, 0.0, ds)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    un, ue = _unit(p["direction_deg"])
    return s * un, s * ue, ds * un, ds * ue