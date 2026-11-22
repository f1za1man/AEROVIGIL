"""Seeded, order-independent sampling of attack parameters and timing."""
import zlib

import numpy as np


def rng_for(seed, key):
    """One independent RNG per key, so results do not depend on loop order."""
    return np.random.default_rng([int(seed), zlib.crc32(key.encode("utf-8"))])


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_timing(rng, flight_duration_s, tcfg, start_fraction=None, until_end=False):
    """Return (start_s, duration_s) or None if the flight is too short."""
    d = float(flight_duration_s)
    earliest = tcfg["min_start_s"]
    latest = d - tcfg["min_duration_s"]
    if latest < earliest:
        return None
    if start_fraction is not None:
        start = min(max(earliest, start_fraction * d), latest)
    else:
        lo = min(max(earliest, tcfg["start_fraction_range"][0] * d), latest)
        hi = max(lo, min(tcfg["start_fraction_range"][1] * d, latest))
        start = float(rng.uniform(lo, hi))
    if until_end:
        duration = d - start
    else:
        duration = min(float(rng.uniform(*tcfg["duration_s_range"])), d - start)
    return float(start), float(duration)


def sample_params(rng, attack_type, acfg, fixed_rate=None):
    """Draw attack parameters. Magnitudes, rates and amplitudes are
    log-uniform; time constants, ramp times and periods are uniform."""
    p = {}
    if attack_type != "low_amplitude_wander":
        p["direction_deg"] = float(rng.uniform(0.0, 360.0))
    if attack_type == "abrupt_jump":
        p["magnitude_m"] = _log_uniform(rng, *acfg["magnitude_m_range"])
    elif attack_type == "constant_offset":
        p["magnitude_m"] = _log_uniform(rng, *acfg["magnitude_m_range"])
        p["ramp_seconds"] = float(rng.uniform(*acfg["ramp_seconds_range"]))
    elif attack_type == "linear_drift":
        p["drift_rate_mps"] = (float(fixed_rate) if fixed_rate is not None
                               else _log_uniform(rng, *acfg["drift_rate_mps_range"]))
    elif attack_type == "exponential_ramp":
        p["initial_rate_mps"] = _log_uniform(rng, *acfg["initial_rate_mps_range"])
        p["time_constant_s"] = float(rng.uniform(*acfg["time_constant_s_range"]))
        p["max_offset_m"] = float(acfg["max_offset_m"])
    elif attack_type == "low_amplitude_wander":
        p["amplitude_m"] = _log_uniform(rng, *acfg["amplitude_m_range"])
        p["periods_s"] = sorted(float(x) for x in
                                rng.uniform(*acfg["period_s_range"], size=3))
        p["phases_north"] = [float(x) for x in rng.uniform(0, 2 * np.pi, 3)]
        p["phases_east"] = [float(x) for x in rng.uniform(0, 2 * np.pi, 3)]
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return p


def sample_attack(cfg, source_id, flight_duration_s, attack_set, attack_type,
                  draw=0, rate=None):
    """Full, reproducible attack specification (without the new flight id)."""
    seed = cfg["random_seed"]
    key = f"{attack_set}|{source_id}|{attack_type}|{draw}|{rate}"
    rng = rng_for(seed, key)

    if attack_set == "sweep":
        sw = cfg["sweep"]
        timing = sample_timing(rng, flight_duration_s, cfg["timing"],
                               start_fraction=sw["start_fraction"], until_end=True)
        mode = sw["velocity_mode"]
    else:
        timing = sample_timing(rng, flight_duration_s, cfg["timing"])
        mode = ("consistent"
                if rng.uniform() < cfg["probability_consistent_velocity"]
                else "position_only")
    if timing is None:
        return None
    params = sample_params(rng, attack_type, cfg["attack_types"][attack_type],
                           fixed_rate=rate)
    return {
        "attack_type": attack_type, "start_s": timing[0],
        "duration_s": timing[1], "velocity_mode": mode, "params": params,
    }