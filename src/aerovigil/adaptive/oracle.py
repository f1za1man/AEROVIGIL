r"""AEROVIGIL Module 08: the detector, as a black box the attacker may probe.

The attacker is allowed one thing: inject an attack into a flight and be told whether
an alarm went off, when, and how far the drone had been moved by then. It never sees
the filter state, the innovations, the threshold or the model weights. That is the
threat model worth reporting, because an attacker with the detector's internals is a
different and much easier problem.

The detector here is Module 06's chi-square test on the horizontal NIS, at the
threshold and refractory period Module 06 calibrated to 1 alarm per flight-hour. It is
the primary target because Module 07 established that it beats both LSTMs on stealthy
drift by a wide margin: at 0.25 m/s it catches 57% of flights against the supervised
LSTM's 11%, at the same alarm budget.

The autoencoder is not wired in yet. train_v2.py saves window scores but not model
weights, so scoring a freshly injected flight would mean retraining. Module 08 step 2
adds weight saving, and the hardening run then measures both detectors together.

Scoring. The attacker wants displacement, not silence: an attack that is never detected
but moves the drone 30 cm has achieved nothing. So the score of one attempt is the
offset the drone had reached when the first alarm inside the attack fired, or its full
final offset if no alarm ever did. Alarms before the attack starts are the flight's own
false alarms and are reported but not charged to the attacker.
"""

from pathlib import Path

import numpy as np

from aerovigil.attacks.inject_v2 import inject
from aerovigil.kalman.common import (
    UNIT_TO_SECONDS,
    get_columns,
    load_json,
    load_noise_scale,
    load_yaml,
    out_dir,
    resolve,
    truth_arrays,
)
from aerovigil.kalman.detectors import nis_alarms
from aerovigil.kalman.pipeline import run_flight

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "kalman" / "kalman.yaml"
START_FRACTION = 0.25
FAILED = {
    "detected": True, "first_alarm_s": np.nan, "delay_s": np.nan,
    "displacement_m": 0.0, "final_offset_m": 0.0, "score": 0.0,
    "attack_duration_s": np.nan, "achieved_mean_mps": 0.0,
    "n_alarms": 0, "alarms_before_onset": 0, "ok": False,
}  # fmt: skip


class KalmanOracle:
    """Module 06's chi-square detector, callable on a freshly injected flight."""

    def __init__(self, config=None, threshold=None):
        cfg = load_yaml(config or CONFIG)
        pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
        self.cfg, self.fcfg, self.dcfg = cfg, cfg["filter"], cfg["detector"]
        self.cols = get_columns(cfg, pre)
        unit = pre.get("dataset", {}).get("timestamp_unit")
        if unit not in UNIT_TO_SECONDS:
            raise SystemExit(
                f"dataset.timestamp_unit is {unit!r} in "
                f"{resolve(cfg['paths']['preprocessing_config'])}; "
                f"expected one of {sorted(UNIT_TO_SECONDS)}. "
                "The filter reads it the same way run_filter.py does."
            )
        self.scale = UNIT_TO_SECONDS[unit]
        base = out_dir(cfg)
        conv = load_json(base / "imu_convention.json")
        self.conv = conv["convention"] if conv else None
        self.noise = load_noise_scale(cfg)
        cal = load_json(base / "detector_calibration.json")
        if threshold is None:
            if cal is None:
                raise SystemExit(
                    "Run evaluate_detectors.py first "
                    "(detector_calibration.json is missing)."
                )
            threshold = cal["nis_threshold"]
        self.threshold = float(threshold)
        self.refractory_s = float(self.dcfg["refractory_s"])
        self.calls = 0

    def spec(self, duration_s, attack_type="adaptive", params=None):
        """Start a quarter of the way in and run to the end of the log."""
        start = duration_s * START_FRACTION
        return {
            "attack_type": attack_type,
            "start_s": start,
            "duration_s": duration_s - start,
            "params": dict(params or {}),
        }

    def attack(self, g, profile_fn=None, mode="consistent", attack_type="adaptive",
               params=None):  # fmt: skip
        """Inject one attack into a clean flight and return the attacked frame.

        With profile_fn the adaptive profile is used. Without it the named Module 04
        profile is used instead, through exactly the path Module 04 took, including
        its own ease-in: that is what the self-test needs in order to reproduce the
        Module 06 frontier rather than something that merely resembles it.
        """
        duration = float(g["t_s"].iloc[-1])
        spec = self.spec(duration, attack_type, params)
        out, _ = inject(g, spec, mode, "adaptive_probe", profile_fn)
        return out

    def __call__(self, g, profile_fn=None, mode="consistent", attack_type="adaptive",
                 params=None):  # fmt: skip
        """Run one probe: inject, filter, test, and report what the attacker learns."""
        self.calls += 1
        attacked = self.attack(g, profile_fn, mode, attack_type, params)
        df = run_flight(
            attacked, self.cols, self.fcfg, self.conv, self.scale, self.noise,
            truth_arrays(attacked),
        )  # fmt: skip
        if df is None or not len(df):
            return dict(FAILED)
        return self.verdict(df)

    def verdict(self, df):
        """Turn one filter run into the attacker's feedback."""
        t = df["t_s"].to_numpy(float)
        valid = df["valid"].to_numpy(bool)
        label = df["label"].to_numpy(float) == 1
        offset = df["offset_m"].to_numpy(float)
        idx = nis_alarms(
            df["nis_h"].to_numpy(float), valid, t, self.threshold, self.refractory_s
        )
        if not label.any():
            return dict(FAILED, n_alarms=int(len(idx)))
        start, end = float(t[label][0]), float(t[label][-1])
        alarm_t = t[idx]
        inside = idx[(alarm_t >= start) & (alarm_t <= end)]
        final = (
            float(np.nanmax(offset[label])) if np.isfinite(offset[label]).any() else 0.0
        )
        detected = len(inside) > 0
        hit = float(offset[inside[0]]) if detected else np.nan
        window = max(end - start, 1e-9)
        return {
            "detected": bool(detected),
            "first_alarm_s": float(t[inside[0]]) if detected else np.nan,
            "delay_s": float(t[inside[0]] - start) if detected else np.nan,
            "displacement_m": hit if detected else final,
            "final_offset_m": final,
            "score": float(hit if detected else final),
            # measured, not assumed: the mean speed the spoofed position actually
            # travelled at over the attack window. rate_mps x duty only equals this
            # when the duty cycle fits inside the window, which it often does not.
            "attack_duration_s": float(window),
            "achieved_mean_mps": float(final / window),
            "n_alarms": int(len(idx)),
            "alarms_before_onset": int((alarm_t < start).sum()),
            "ok": True,
        }
