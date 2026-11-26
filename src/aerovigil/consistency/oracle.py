r"""AEROVIGIL Module 09: the consistency detector as a black box the attacker may probe.

Same contract as the Kalman oracle in Module 08: inject an attack into a clean flight,
run the detector, and hand back only whether an alarm fired, when, and how far the
vehicle had been moved by then. The attacker never sees the channels, the standardiser
or the threshold. That is what makes Module 10's search a fair test rather than a
demonstration.

Usable only after evaluate.py has written consistency_calibration.json, because the
threshold has to come from the same 1 alarm per flight-hour budget as every other
detector in the project.
"""

from pathlib import Path

import numpy as np

from aerovigil.attacks.inject_v2 import inject
from aerovigil.consistency.detector import (
    REFRACTORY_S,
    alarm_indices,
    channel_values,
    score_rows,
)
from aerovigil.consistency.evaluate import OUT_DIR
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.kalman.common import load_json

START_FRACTION = 0.25
FAILED = {
    "detected": True, "first_alarm_s": np.nan, "delay_s": np.nan,
    "displacement_m": 0.0, "final_offset_m": 0.0, "score": 0.0,
    "attack_duration_s": np.nan, "achieved_mean_mps": 0.0,
    "n_alarms": 0, "alarms_before_onset": 0, "ok": False,
}  # fmt: skip


class ConsistencyOracle:
    """Module 09's course-and-velocity detector, callable on a freshly injected flight."""

    def __init__(self, calibration=None, threshold=None):
        path = (
            Path(calibration)
            if calibration
            else OUT_DIR / "consistency_calibration.json"
        )
        cal = load_json(path)
        if cal is None:
            raise SystemExit(
                f"Missing {path}. Run src\\aerovigil\\consistency\\evaluate.py first."
            )
        self.channels = tuple(cal["channels"])
        self.mu = np.asarray(cal["mu"], float)
        self.sd = np.asarray(cal["sd"], float)
        self.threshold = float(cal["threshold"] if threshold is None else threshold)
        self.refractory_s = float(cal.get("refractory_s", REFRACTORY_S))
        self.calls = 0

    def spec(self, duration_s, attack_type="adaptive", params=None):
        start = duration_s * START_FRACTION
        return {
            "attack_type": attack_type,
            "start_s": start,
            "duration_s": duration_s - start,
            "params": dict(params or {}),
        }

    def attack(self, g, profile_fn=None, mode="consistent", attack_type="adaptive",
               params=None):  # fmt: skip
        duration = float(g["t_s"].iloc[-1])
        out, _ = inject(
            g, self.spec(duration, attack_type, params), mode, "probe", profile_fn
        )
        return out

    def __call__(self, g, profile_fn=None, mode="consistent", attack_type="adaptive",
                 params=None):  # fmt: skip
        self.calls += 1
        attacked = self.attack(g, profile_fn, mode, attack_type, params)
        frame = build_flight_v2(attacked)
        values = channel_values(frame, self.channels)
        score, valid = score_rows(values, self.mu, self.sd)
        return self.verdict(
            score,
            frame["t_s"].to_numpy(float),
            valid,
            attacked["attack_label"].to_numpy(float) == 1,
            attacked["offset_m"].to_numpy(float),
        )

    def verdict(self, score, t, valid, on, offset):
        """Turn one detector run into the attacker's feedback."""
        idx = alarm_indices(score, t, valid, self.threshold, self.refractory_s)
        if not on.any():
            return dict(FAILED, n_alarms=int(len(idx)))
        start, end = float(t[on][0]), float(t[on][-1])
        inside = idx[(t[idx] >= start) & (t[idx] <= end)]
        detected = len(inside) > 0
        final = float(np.nanmax(offset[on])) if np.isfinite(offset[on]).any() else 0.0
        window = max(end - start, 1e-9)
        hit = float(offset[inside[0]]) if detected else np.nan
        return {
            "detected": bool(detected),
            "first_alarm_s": float(t[inside[0]]) if detected else np.nan,
            "delay_s": float(t[inside[0]] - start) if detected else np.nan,
            "displacement_m": hit if detected else final,
            "final_offset_m": final,
            "score": float(hit if detected else final),
            "attack_duration_s": float(window),
            "achieved_mean_mps": float(final / window),
            "n_alarms": int(len(idx)),
            "alarms_before_onset": int((t[idx] < start).sum()),
            "ok": True,
        }
