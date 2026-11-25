"""The exponential ramp must bend onto its cap, not stop dead at it."""

import numpy as np
import pytest

from aerovigil.attacks.profiles import SATURATION_KNEE, offset_profile

CAP = 500.0
HZ = 125.0
BASE = {
    "direction_deg": 0.0,
    "initial_rate_mps": 0.5,
    "time_constant_s": 20.0,
    "max_offset_m": CAP,
}


def ramp(seconds=200.0, **over):
    tau = np.arange(0.0, seconds, 1.0 / HZ)
    north, _, d_north, _ = offset_profile("exponential_ramp", tau, {**BASE, **over})
    return tau, north, d_north


def raw(tau, rate, tc):
    """The uncapped exponential, for comparison."""
    return rate * tc * (np.exp(np.minimum(tau / tc, 50.0)) - 1.0)


def test_a_ramp_that_stays_under_the_knee_is_unchanged():
    tau, s, _ = ramp(initial_rate_mps=0.0005)
    assert s.max() < SATURATION_KNEE * CAP  # never reaches the bend
    assert np.allclose(s, raw(tau, 0.0005, BASE["time_constant_s"]))


def test_the_offset_approaches_the_cap_without_exceeding_it():
    _, s, _ = ramp()
    assert s.max() <= CAP + 1e-6
    assert s.max() > 0.99 * CAP  # it still gets there
    assert (np.diff(s) >= -1e-12).all()  # and never comes back down


def test_the_rate_decays_to_zero_instead_of_stepping():
    _, s, ds = ramp()
    assert np.abs(np.diff(ds)).max() < 0.1  # the hard clip stepped 25 m/s in one sample
    assert ds[-1] < 0.01  # the attacker does come to rest at the cap
    assert ds.max() > 10.0  # after passing through a fast stretch


def test_value_and_slope_match_where_the_bend_starts():
    tau, s, ds = ramp()
    r = raw(tau, BASE["initial_rate_mps"], BASE["time_constant_s"])
    knee = int(np.argmax(r >= SATURATION_KNEE * CAP))
    assert knee > 0
    assert s[knee] == pytest.approx(SATURATION_KNEE * CAP, rel=1e-3)
    before = ds[knee - 2]
    assert ds[knee] == pytest.approx(before, rel=0.05)  # no kink in the rate either


def test_the_other_profiles_are_untouched():
    tau = np.arange(0.0, 60.0, 1.0 / HZ)
    s, _, ds, _ = offset_profile(
        "linear_drift", tau, {"direction_deg": 0.0, "drift_rate_mps": 0.25}
    )
    assert np.allclose(s, 0.25 * tau)
    assert np.allclose(ds, 0.25)
