r"""AEROVIGIL Module 04 v2: inject the synthetic attacks into the Module 03 v2 base.

What changes against Module 04 v1:
  * the 144 normal flights come from the v2 base (Raw, every gps row kept, IMU rows
    aligned), not from the Merged-based cleaned flights;
  * every attack is made twice with identical parameters and timing, once with
    velocity_mode position_only and once with consistent, so the velocity effect
    is a paired comparison (v1 drew the mode at random: 69/75, 75/69 ...);
  * the truth columns use the Module 03 v2 convention: offset_m is measured, and
    phase / spoof_truth / seconds_since_onset come from the same label_phases()
    that labelled the real attacks, so synthetic and real attacks are comparable;
  * attack_label is still the injected window, but spoof_truth marks where the
    offset really is >= 1 m. The manifest keeps peak_offset_m, onset_s and spoof_end_s;
  * velocity_mode consistent also moves the scalar speed column, keeping the
    source flight's own speed noise, and eases the whole offset in over
    ONSET_RAMP_S seconds so the reported velocity never steps. A step at onset is
    a giveaway no careful spoofer would leave, and the Kalman filter in Module 06
    detects it in a single sample, which made the stealth frontier measure the
    switch-on rather than the drift.

This file holds the injector itself; generate_v2.py runs it over all flights.
"""

import numpy as np

from aerovigil.attacks.profiles import offset_profile
from aerovigil.data.build_raw_base import (
    METRES_PER_DEGREE,
    PHASE_RAMP,
    horizontal_offset_m,
    label_phases,
)
from aerovigil.features.imu_consistency import IMU_INPUTS

EDGE_TOL_S = 1e-6
ONSET_RAMP_S = 1.5
MODES = ("position_only", "consistent")
SHORT = {"position_only": "po", "consistent": "cv"}
KEEP = [
    "flight_id", "mission_type", "time_usec", "t_s", "latitude_deg", "longitude_deg",
    "altitude", "velocity", "velocity_east", "velocity_north", "velocity_up",
    "pressure_altitude", *IMU_INPUTS, "attack_label", "attack_enabled", "aligned_all",
]  # fmt: skip
SCALARS = [
    "direction_deg", "magnitude_m", "drift_rate_mps", "initial_rate_mps",
    "time_constant_s", "ramp_seconds", "max_offset_m", "amplitude_m",
]  # fmt: skip
TRUTH = [
    "source_flight_id", "attack_type", "velocity_mode", "spoof_offset_north_m",
    "spoof_offset_east_m", "offset_rate_mps", "offset_m", "spoof_truth", "phase",
    "seconds_since_onset",
]  # fmt: skip


def smooth_start(tau, seconds):
    """Smoothstep weight and its time derivative: 0 at tau=0, 1 after `seconds`,
    flat at both ends, so neither the offset nor its rate jumps at onset."""
    x = np.clip(tau / seconds, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x) / seconds


def speed_uses_3d(g):
    """True when the scalar speed column is the 3D norm of the velocity components."""
    vn, ve, vu = (
        g[c].to_numpy(float) for c in ("velocity_north", "velocity_east", "velocity_up")
    )
    sp = g["velocity"].to_numpy(float)
    err3 = np.nanmedian(np.abs(sp - np.sqrt(vn**2 + ve**2 + vu**2)))
    err2 = np.nanmedian(np.abs(sp - np.hypot(vn, ve)))
    return bool(err3 < err2)


def inject(g, spec, mode, new_id):
    """Return the attacked flight (base schema + truth columns) for one spec and mode."""
    t = g["t_s"].to_numpy(float)
    tau = t - spec["start_s"]
    active = (tau >= 0) & (tau <= spec["duration_s"] + EDGE_TOL_S)
    n, e, dn, de = (np.zeros(len(t)) for _ in range(4))
    if active.any():
        n[active], e[active], dn[active], de[active] = offset_profile(
            spec["attack_type"], tau[active], spec["params"]
        )
        if mode == "consistent":
            w, dw = smooth_start(tau[active], ONSET_RAMP_S)
            dn[active] = dn[active] * w + n[active] * dw
            de[active] = de[active] * w + e[active] * dw
            n[active] = n[active] * w
            e[active] = e[active] * w
    out = g.copy()
    lat0, lon0 = g["latitude_deg"].to_numpy(float), g["longitude_deg"].to_numpy(float)
    out["latitude_deg"] = lat0 + n / METRES_PER_DEGREE
    out["longitude_deg"] = lon0 + e / (METRES_PER_DEGREE * np.cos(np.radians(lat0)))
    if mode == "consistent":
        vn0, ve0, vu = (
            g[c].to_numpy(float)
            for c in ("velocity_north", "velocity_east", "velocity_up")
        )
        vn1, ve1 = vn0 + dn, ve0 + de
        if speed_uses_3d(g):
            shift = np.sqrt(vn1**2 + ve1**2 + vu**2) - np.sqrt(vn0**2 + ve0**2 + vu**2)
        else:
            shift = np.hypot(vn1, ve1) - np.hypot(vn0, ve0)
        out["velocity_north"], out["velocity_east"] = vn1, ve1
        out["velocity"] = g["velocity"].to_numpy(float) + shift
    label = active.astype(np.float32)
    out["attack_label"], out["attack_enabled"] = label, label
    offset = horizontal_offset_m(out["latitude_deg"], out["longitude_deg"], lat0, lon0)
    phase, truth, since, facts = label_phases(np.asarray(offset), label, t, True)
    if not (phase == PHASE_RAMP).any() and active.any():
        phase[int(np.flatnonzero(active)[0])] = PHASE_RAMP  # a step is a one-row ramp
    out["offset_m"], out["phase"], out["spoof_truth"] = offset, phase, truth
    out["seconds_since_onset"] = since
    out["source_flight_id"], out["attack_type"], out["velocity_mode"] = (
        g["flight_id"].iloc[0], spec["attack_type"], mode,
    )  # fmt: skip
    out["spoof_offset_north_m"], out["spoof_offset_east_m"] = n, e
    out["offset_rate_mps"] = np.hypot(dn, de)
    out["flight_id"] = new_id
    out = out[KEEP + TRUTH]
    for c in out.columns:
        if out[c].dtype == np.float64 and c not in (
            "latitude_deg",
            "longitude_deg",
            "t_s",
            "offset_m",
        ):
            out[c] = out[c].astype(np.float32)
    return out, facts
