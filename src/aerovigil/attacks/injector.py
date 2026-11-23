"""Inject a spoofing offset into one clean flight (log level only)."""

import numpy as np

from aerovigil.attacks.profiles import offset_profile

EARTH_R = 6371000.0  # metres
EDGE_TOL_S = 1e-6  # tolerance for the attack-window end (float rounding)


def inject_attack(g, spec, cols, scale):
    """Return a copy of flight `g` with a GPS offset applied.

    Only the GPS position (and, in 'consistent' mode, GPS horizontal velocity)
    is modified. IMU, barometer, magnetometer, orientation, GPS altitude and
    vertical velocity are never touched (threat model, Section 3 of the plan).

    spec keys: attack_type, start_s, duration_s, velocity_mode, params,
               attacked_flight_id, source_flight_id
    cols keys: flight, time, label, enabled (optional), lat, lon, vn, ve
    """
    out = g.copy()
    t_us = out[cols["time"]].to_numpy(dtype="int64")
    if len(t_us) < 2 or np.any(np.diff(t_us) < 0):
        raise ValueError("Flight must have >= 2 rows sorted by time")

    t = (t_us - t_us[0]) * scale  # seconds since flight start
    tau = t - spec["start_s"]  # seconds since attack start
    active = (tau >= 0) & (tau <= spec["duration_s"] + EDGE_TOL_S)

    n = np.zeros(len(t))
    e = np.zeros(len(t))
    dn = np.zeros(len(t))
    de = np.zeros(len(t))
    if active.any():
        n[active], e[active], dn[active], de[active] = offset_profile(
            spec["attack_type"], tau[active], spec["params"]
        )

    lat = out[cols["lat"]].to_numpy(dtype=float)
    lon = out[cols["lon"]].to_numpy(dtype=float)
    out[cols["lat"]] = lat + np.degrees(n / EARTH_R)
    out[cols["lon"]] = lon + np.degrees(e / (EARTH_R * np.cos(np.radians(lat))))

    if spec["velocity_mode"] == "consistent":
        # The spoofer also reports the velocity that matches its fake motion.
        vn0 = out[cols["vn"]].to_numpy(dtype=float)
        ve0 = out[cols["ve"]].to_numpy(dtype=float)
        vn1, ve1 = vn0 + dn, ve0 + de
        out[cols["vn"]] = vn1
        out[cols["ve"]] = ve1
        # GPS speed is horizontal speed (checked on real data). Keep it coherent,
        # preserving its native ~0.01 m/s residual; the delta is exactly 0 outside the attack.
        sp = cols.get("speed")
        if sp and sp in out.columns:
            out[sp] = out[sp].to_numpy(dtype=float) + (
                np.hypot(vn1, ve1) - np.hypot(vn0, ve0)
            )
    elif spec["velocity_mode"] != "position_only":
        raise ValueError(f"Unknown velocity_mode: {spec['velocity_mode']}")

    label = active.astype(out[cols["label"]].dtype)
    out[cols["label"]] = label
    if cols.get("enabled") and cols["enabled"] in out.columns:
        out[cols["enabled"]] = label

    out[cols["flight"]] = spec["attacked_flight_id"]
    out["source_flight_id"] = spec["source_flight_id"]
    out["attack_type"] = spec["attack_type"]
    out["time_from_attack_start_s"] = tau  # negative before the attack
    out["spoof_offset_north_m"] = n
    out["spoof_offset_east_m"] = e
    out["spoof_offset_m"] = np.hypot(n, e)
    return out


# Columns added by the generator. They are ground truth, never model inputs.
TRUTH_COLUMNS = [
    "source_flight_id",
    "attack_type",
    "time_from_attack_start_s",
    "spoof_offset_north_m",
    "spoof_offset_east_m",
    "spoof_offset_m",
]
