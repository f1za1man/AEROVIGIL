r"""AEROVIGIL Module 08: the adaptive attacker's offset profile.

Module 06 and Module 07 measured the detectors against a fixed family of attacks whose
only free variable was the drift rate. The frontier that came out of that sits at about
0.1 m/s: below it the chi-square test catches 5% of flights, above 0.5 m/s it catches
all of them. But a real spoofer is not restricted to a steady drift. This profile gives
it the knobs a careful attacker would actually use, and Module 08's search then looks
for the combination that moves the drone furthest before an alarm.

    rate_mps         speed of the drift while it is moving
    onset_s          how long the whole attack eases in; a step at onset is fatal
    period_s         length of one move-then-hold cycle
    duty             share of each cycle spent moving (1.0 is a continuous drift)
    bearing_deg      direction of travel
    jitter_m         amplitude of a small wobble laid over the drift
    jitter_period_s  period of that wobble

The duty cycle is the interesting one. Holding still for part of every cycle lets a
CUSUM-style statistic decay back towards zero between bursts, so the same net
displacement can be bought at a lower accumulated score, while a single-sample test
such as the chi-square one should prefer the opposite. Whether that trade is worth
taking is exactly what the search decides, and it decides it from alarms alone.

The offset is the running integral of the velocity, so the two are consistent by
construction: a spoofer whose reported position and reported velocity disagree is
detectable without any filter at all.
"""

import numpy as np

# name: (low, high, scale)
BOUNDS = {
    "rate_mps": (0.01, 10.0, "log"),
    "onset_s": (0.5, 30.0, "lin"),
    "period_s": (5.0, 600.0, "lin"),
    "duty": (0.1, 1.0, "lin"),
    "bearing_deg": (0.0, 360.0, "wrap"),
    "jitter_m": (0.0, 2.0, "lin"),
    "jitter_period_s": (3.0, 60.0, "lin"),
}
NAMES = tuple(BOUNDS)
DEFAULT = {
    "rate_mps": 0.25,
    "onset_s": 3.0,
    "period_s": 60.0,
    "duty": 1.0,
    "bearing_deg": 45.0,
    "jitter_m": 0.0,
    "jitter_period_s": 10.0,
}


def clip(params):
    """Put every parameter back inside its bounds, wrapping the bearing."""
    out = {}
    for name, (lo, hi, scale) in BOUNDS.items():
        v = float(params[name])
        out[name] = v % hi if scale == "wrap" else float(np.clip(v, lo, hi))
    return out


def smoothstep(x):
    """0 at x<=0, 1 at x>=1, with a zero derivative at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gate(tau, period_s, duty, trans_s):
    """Move-then-hold cycle in [0, 1], eased over trans_s so the rate never steps."""
    if duty >= 1.0:
        return np.ones_like(tau)
    phase = np.mod(tau, period_s)
    on = duty * period_s
    trans = max(min(trans_s, on / 2.0, (period_s - on) / 2.0), 1e-6)
    rise = smoothstep(phase / trans)
    fall = 1.0 - smoothstep((phase - on) / trans)
    return np.minimum(rise, np.maximum(fall, 0.0))


def velocity(tau, params):
    """Spoofed velocity along the bearing, in m/s, as a function of tau >= 0."""
    p = clip(params)
    ease = smoothstep(tau / p["onset_s"])
    v = p["rate_mps"] * ease * gate(tau, p["period_s"], p["duty"], p["onset_s"])
    if p["jitter_m"] > 0.0:
        w = 2.0 * np.pi / p["jitter_period_s"]
        v = v + p["jitter_m"] * ease * w * np.cos(w * tau)
    return v


def offset_and_rate(tau, params):
    """(offset_m, rate_mps) along the bearing. The offset integrates the velocity."""
    tau = np.asarray(tau, dtype=float)
    if len(tau) == 0:
        return np.zeros(0), np.zeros(0)
    v = velocity(tau, params)
    dt = np.diff(tau, prepend=tau[0])
    s = np.cumsum(0.5 * (v + np.roll(v, 1)) * dt)
    s[0] = 0.0
    return s, v


def profile_fn(params):
    """A callable with the signature Module 04's injector expects."""
    p = clip(params)
    theta = np.radians(p["bearing_deg"])
    un, ue = float(np.cos(theta)), float(np.sin(theta))

    def fn(tau):
        s, v = offset_and_rate(tau, p)
        return s * un, s * ue, v * un, v * ue

    return fn


def to_vector(params):
    return np.array([float(params[n]) for n in NAMES])


def from_vector(vec):
    return clip(dict(zip(NAMES, map(float, vec))))


def to_unit(params):
    """Map the parameters onto [0, 1] so one step size suits all of them."""
    p, out = clip(params), []
    for name, (lo, hi, scale) in BOUNDS.items():
        v = p[name]
        if scale == "log":
            out.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
        elif scale == "wrap":
            out.append((v % hi) / hi)
        else:
            out.append((v - lo) / (hi - lo))
    return np.array(out)


def from_unit(vec):
    """Inverse of to_unit. Values outside [0, 1] are clipped, the bearing wraps."""
    out = {}
    for (name, (lo, hi, scale)), x in zip(BOUNDS.items(), np.asarray(vec, float)):
        if scale == "log":
            span = np.log(hi) - np.log(lo)
            out[name] = float(np.exp(np.log(lo) + float(np.clip(x, 0.0, 1.0)) * span))
        elif scale == "wrap":
            out[name] = float((x % 1.0) * hi)
        else:
            out[name] = float(lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo))
    return clip(out)


def mean_rate(params):
    """Average drift speed over a whole cycle, which is what the sweep rates mean."""
    p = clip(params)
    return p["rate_mps"] * p["duty"]


def burst_seconds(params):
    """How long each move phase lasts. Longer than the flight means a single burst."""
    p = clip(params)
    return p["duty"] * p["period_s"]


def describe(params):
    p = clip(params)
    return (
        f"rate {p['rate_mps']:.3f} m/s x duty {p['duty']:.2f} "
        f"(mean {mean_rate(p):.3f} m/s), {burst_seconds(p):.0f} s bursts every "
        f"{p['period_s']:.0f} s, onset {p['onset_s']:.1f} s, "
        f"bearing {p['bearing_deg']:.0f} deg, "
        f"jitter {p['jitter_m']:.2f} m / {p['jitter_period_s']:.0f} s"
    )
