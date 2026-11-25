"""Module 08 tests: the evasive profile, the black-box oracle and the hill climber."""

import json

import numpy as np
import pandas as pd
import pytest

import aerovigil.adaptive.oracle as orc
import aerovigil.adaptive.search as sr
from aerovigil.adaptive.profile import (
    BOUNDS,
    DEFAULT,
    NAMES,
    clip,
    from_unit,
    gate,
    offset_and_rate,
    profile_fn,
    to_unit,
)
from aerovigil.attacks.inject_v2 import KEEP

HZ = 125.0
MISSIONS = {"curved": 120.0, "random": 100.0, "straight": 80.0}


def fake_flight(fid, mission, duration, seed):
    rng = np.random.default_rng(seed)
    n = int(duration * HZ)
    t = np.arange(n) / HZ
    vn = 4.0 + 0.5 * np.sin(t / 20.0)
    ve = 3.0 + 0.5 * np.cos(t / 17.0)
    vu = rng.normal(0, 0.1, n)
    lat = 24.4 + np.degrees(np.cumsum(vn / HZ) / 6371000.0)
    lon = 54.6 + np.degrees(np.cumsum(ve / HZ) / (6371000.0 * np.cos(np.radians(24.4))))
    df = pd.DataFrame({
        "flight_id": fid, "mission_type": mission,
        "time_usec": (t * 1e6).astype("int64"), "t_s": t,
        "latitude_deg": lat, "longitude_deg": lon,
        "altitude": 50.0 + rng.normal(0, 0.3, n),
        "velocity": np.sqrt(vn**2 + ve**2 + vu**2),
        "velocity_east": ve, "velocity_north": vn, "velocity_up": vu,
        "pressure_altitude": 51.0 + rng.normal(0, 0.3, n),
        "orientation_x": 0.0, "orientation_y": 0.0,
        "orientation_z": 0.0, "orientation_w": 1.0,
        "linear_acceleration_x": np.gradient(ve, t) + rng.normal(0, 0.2, n),
        "linear_acceleration_y": np.gradient(vn, t) + rng.normal(0, 0.2, n),
        "linear_acceleration_z": 9.80665 + rng.normal(0, 0.2, n),
        "attack_label": 0.0, "attack_enabled": 0.0, "aligned_all": 1})  # fmt: skip
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in (
            "latitude_deg",
            "longitude_deg",
            "t_s",
        ):
            df[c] = df[c].astype(np.float32)
    return df[KEEP]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Nine clean flights across the three splits, plus Module 06's saved calibration."""
    root = tmp_path_factory.mktemp("m08")
    base, interim, m06 = root / "base", root / "interim", root / "m06"
    for d in (base, interim, m06):
        d.mkdir(parents=True)
    rows, splits = [], []
    for i, split in enumerate(["train"] * 3 + ["validation"] * 3 + ["test"] * 3):
        mission = list(MISSIONS)[i % 3]
        fid = f"{mission}__Normal__log_{i:02d}"
        fake_flight(fid, mission, MISSIONS[mission], i).to_parquet(
            base / f"{fid}.parquet", index=False
        )
        rows.append({"flight_id": fid, "mission": mission, "attacked": 0,
                     "file": f"{fid}.parquet"})  # fmt: skip
        splits.append({"flight_id": fid, "split": split})
    pd.DataFrame(rows).to_csv(interim / "base_report.csv", index=False)
    pd.DataFrame(splits).to_csv(interim / "flight_splits.csv", index=False)
    json.dump({"convention": {"rotation": "body_to_world",
                              "gravity": "specific_force", "frame": "enu"}},
              (m06 / "imu_convention.json").open("w"))  # fmt: skip
    json.dump({"scale": {"pos": 0.05, "vel": 1e-6, "vert": 2.0}},
              (m06 / "filter_noise.json").open("w"))  # fmt: skip
    json.dump({"k": 0.5, "h": 100.0, "nis_threshold": 18.47, "mu": [0, 0, 0, 0],
               "sd": [1, 1, 1, 1], "calibration_hours": 4.0, "budget_per_hour": 1.0},
              (m06 / "detector_calibration.json").open("w"))  # fmt: skip
    return {"root": root, "base": base, "interim": interim, "m06": m06}


@pytest.fixture
def wired(project, monkeypatch):
    monkeypatch.setattr(sr, "M03_BASE", project["base"])
    monkeypatch.setattr(sr, "BASE_REPORT", project["interim"] / "base_report.csv")
    monkeypatch.setattr(sr, "SPLITS", project["interim"] / "flight_splits.csv")
    monkeypatch.setattr(orc, "out_dir", lambda cfg: project["m06"])
    return project


# ----------------------------------------------------------------- profile --


def test_the_offset_is_the_integral_of_the_reported_velocity():
    tau = np.arange(0.0, 150.0, 1.0 / HZ)
    for over in ({}, {"duty": 0.4, "period_s": 40.0}, {"jitter_m": 1.5}):
        s, v = offset_and_rate(tau, {**DEFAULT, **over})
        assert np.abs(np.gradient(s, tau) - v).max() < 1e-3
        assert s[0] == 0.0 and v[0] == 0.0  # no step at onset, in value or rate
        assert np.abs(np.diff(v)).max() < 0.05  # and none later either


def test_a_duty_cycle_holds_still_for_part_of_every_period():
    tau = np.arange(0.0, 120.0, 1.0 / HZ)
    g = gate(tau, 30.0, 0.4, 3.0)
    assert g.min() == pytest.approx(0.0, abs=1e-6)
    assert g.max() == pytest.approx(1.0, abs=1e-6)
    assert 0.3 < g.mean() < 0.5  # about the duty, blurred by the eased edges
    assert np.allclose(gate(tau, 30.0, 1.0, 3.0), 1.0)  # duty 1 is a plain drift


def test_the_duty_cycle_cuts_the_displacement_it_buys():
    tau = np.arange(0.0, 120.0, 1.0 / HZ)
    full, _ = offset_and_rate(tau, {**DEFAULT, "duty": 1.0})
    half, _ = offset_and_rate(tau, {**DEFAULT, "duty": 0.5, "period_s": 20.0})
    assert half.max() < full.max()
    assert half.max() > 0.3 * full.max()


def test_the_bearing_sets_the_direction():
    tau = np.arange(0.0, 60.0, 1.0 / HZ)
    n, e, dn, de = profile_fn({**DEFAULT, "bearing_deg": 90.0})(tau)
    assert np.abs(n).max() < 1e-9 and e.max() > 1.0
    n2, e2, _, _ = profile_fn({**DEFAULT, "bearing_deg": 0.0})(tau)
    assert np.abs(e2).max() < 1e-9 and n2.max() > 1.0


def test_unit_mapping_round_trips_and_stays_in_bounds():
    assert np.allclose(to_unit(from_unit(np.full(len(NAMES), 0.37))), 0.37, atol=1e-9)
    p = from_unit(np.zeros(len(NAMES)))
    q = from_unit(np.ones(len(NAMES)))
    for name, (lo, hi, scale) in BOUNDS.items():
        if scale == "wrap":
            continue
        assert p[name] == pytest.approx(lo)
        assert q[name] == pytest.approx(hi)
    wild = from_unit(np.full(len(NAMES), 5.0))
    assert clip(wild) == wild  # already inside the bounds


def test_the_mean_rate_constraint_sets_the_burst_rate():
    p = sr.with_mean_rate({**DEFAULT, "duty": 0.25}, 0.1)
    assert p["rate_mps"] == pytest.approx(0.4)
    assert p["duty"] == pytest.approx(0.25)
    capped = sr.with_mean_rate({**DEFAULT, "duty": 0.1}, 5.0)
    assert capped["rate_mps"] == pytest.approx(BOUNDS["rate_mps"][1])


# ------------------------------------------------------------------ oracle --


def test_the_oracle_reports_what_an_attacker_can_see(wired):
    oracle = orc.KalmanOracle()
    _, _, g = sr.normal_flights("test")[0]
    r = oracle(g, profile_fn(DEFAULT))
    assert set(r) == {"detected", "first_alarm_s", "delay_s", "displacement_m",
                      "final_offset_m", "score", "attack_duration_s",
                      "achieved_mean_mps", "n_alarms", "alarms_before_onset",
                      "ok"}  # fmt: skip
    assert r["ok"] and r["final_offset_m"] > 0.0
    assert r["achieved_mean_mps"] == pytest.approx(
        r["final_offset_m"] / r["attack_duration_s"]
    )
    assert oracle.calls == 1
    assert r["detected"] == np.isfinite(r["delay_s"])


def test_a_faster_drift_moves_further_and_an_abrupt_one_is_caught_early(wired):
    oracle = orc.KalmanOracle()
    _, _, g = sr.normal_flights("test")[0]
    slow = oracle(g, profile_fn({**DEFAULT, "rate_mps": 0.05}))
    fast = oracle(g, profile_fn({**DEFAULT, "rate_mps": 1.0}))
    assert fast["final_offset_m"] > slow["final_offset_m"] * 5
    abrupt = oracle(g, profile_fn({**DEFAULT, "rate_mps": 2.0, "onset_s": 0.5}))
    gentle = oracle(g, profile_fn({**DEFAULT, "rate_mps": 2.0, "onset_s": 30.0}))
    assert abrupt["detected"]
    assert abrupt["displacement_m"] < gentle["displacement_m"]


def test_the_oracle_uses_the_calibrated_threshold(wired):
    _, _, g = sr.normal_flights("test")[0]
    params = {**DEFAULT, "rate_mps": 2.0, "onset_s": 0.5}
    strict = orc.KalmanOracle(threshold=5.0)(g, profile_fn(params))
    loose = orc.KalmanOracle(threshold=1e9)(g, profile_fn(params))
    assert strict["n_alarms"] >= loose["n_alarms"]
    assert not loose["detected"]
    assert orc.KalmanOracle().threshold == pytest.approx(18.47)


# -------------------------------------------------------------- hill climb --


def test_the_climber_never_moves_a_parameter_it_was_told_to_hold():
    rng = np.random.default_rng(0)
    seen = []

    def evaluate(params):
        seen.append(params)
        return float(params["onset_s"]), None

    best, score, hist = sr.hill_climb(
        evaluate, rng, 25, to_unit(DEFAULT), free=("onset_s",)
    )
    assert len(hist) == 25
    assert {round(p["rate_mps"], 9) for p in seen} == {round(DEFAULT["rate_mps"], 9)}
    assert best["onset_s"] >= DEFAULT["onset_s"]  # it climbed, or at worst held
    assert score == pytest.approx(hist["score"].max())


def test_the_climber_finds_the_optimum_of_a_known_surface():
    rng = np.random.default_rng(1)
    target = 0.8

    def evaluate(params):
        u = to_unit(params)
        return float(-np.sum((u - target) ** 2)), None

    best, score, hist = sr.hill_climb(evaluate, rng, 200, np.zeros(len(NAMES)))
    assert score > -0.2
    assert np.abs(to_unit(best) - target).max() < 0.35
    assert hist["score"].iloc[0] < score


# -------------------------------------------------------------- end to end --


def test_the_search_runs_and_compares_itself_with_module06(wired):
    out = wired["root"] / "m08"
    frontier, best = sr.run(
        evals=4, free_evals=4, search_flights=1, rates=(0.1, 1.0), output_dir=out
    )
    assert set(frontier["search"]) == {"constrained", "free"}
    assert len(frontier) == 3
    assert frontier["detection_rate"].between(0, 1).all()
    assert set(best) == {"mean_rate=0.1", "mean_rate=1", "free"}
    con = frontier[frontier["search"] == "constrained"]
    assert (con["module06_detection_rate"] > 0).all()
    assert (con["median_final_offset_m"] > 0).all()
    for name in ("adaptive_frontier.csv", "adaptive_search.csv",
                 "adaptive_best.json", "adaptive_free_test_flights.csv"):  # fmt: skip
        assert (out / name).is_file()
    probes = pd.read_csv(out / "adaptive_search.csv")
    assert set(probes["search"]) == {"mean_rate=0.1", "mean_rate=1", "free"}
    assert len(probes) == 4 * 3
    # rate_mps is free now; the measured mean speed is what is pinned
    slow = con[con["mean_rate_mps"] == 0.1].iloc[0]
    assert slow["achieved_mean_mps"] == pytest.approx(0.1, rel=sr.RATE_TOLERANCE)
    assert "formula_mean_mps" in con.columns


def test_the_selftest_flags_an_oracle_that_disagrees_with_module06(wired):
    oracle = orc.KalmanOracle(threshold=1e9)  # a detector that never fires
    exam = sr.normal_flights("test")
    frame, trusted = sr.selftest(oracle, exam, rates=(0.5, 1.0))
    assert not trusted  # Module 06 caught all of these, so a silent oracle is suspect
    assert (frame["module04"] == 0.0).all() and (frame["eased"] == 0.0).all()
    assert frame["gap"].max() > sr.SELFTEST_TOLERANCE
    assert list(frame["drift_rate_mps"]) == [0.5, 1.0]


def test_the_selftest_separates_module04s_onset_from_the_eased_one(wired):
    """Module 04 scales the offset, so its reported velocity overshoots the rate."""
    oracle = orc.KalmanOracle()
    exam = sr.normal_flights("test")
    frame, _ = sr.selftest(oracle, exam, rates=(0.5,))
    assert {"module06", "module04", "eased", "gap"} <= set(frame.columns)
    assert frame["module04"].between(0, 1).all()
    assert frame["eased"].between(0, 1).all()
    assert (frame["disp_module04_m"] > 0).all()


def test_the_constraint_cannot_be_gamed_by_a_cycle_longer_than_the_flight(wired):
    """One long burst reads as a slow mean by formula, but not when it is measured."""
    oracle = orc.KalmanOracle()
    flights = sr.normal_flights("test")
    sneaky = {**DEFAULT, "rate_mps": 10.0, "duty": 0.1, "period_s": 600.0,
              "onset_s": 20.0}  # fmt: skip
    from aerovigil.adaptive.profile import mean_rate

    assert mean_rate(sneaky) == pytest.approx(1.0)  # what the formula claims
    _, frame = sr.probe(oracle, flights, sneaky)
    assert sr.achieved_mean(frame) > 2.0  # what it actually does
    honest, _ = sr.constrained_objective(oracle, flights, sneaky, 1.0, "consistent")
    plain = {**DEFAULT, "rate_mps": 1.0, "duty": 1.0, "onset_s": 20.0}
    good, gframe = sr.constrained_objective(oracle, flights, plain, 1.0, "consistent")
    assert sr.achieved_mean(gframe) == pytest.approx(1.0, rel=0.2)
    assert honest < good  # the penalty rejects the gamed candidate


def test_the_frontier_marks_a_mean_rate_the_rate_bound_cannot_reach(wired):
    out = wired["root"] / "m08b"
    hi = BOUNDS["rate_mps"][1]
    frontier, _ = sr.run(
        evals=2, free_evals=2, search_flights=1, rates=(0.1, hi * 2), output_dir=out
    )
    con = frontier[frontier["search"] == "constrained"]
    reachable = con[con["mean_rate_mps"] == 0.1].iloc[0]
    capped = con[con["mean_rate_mps"] == hi * 2].iloc[0]
    assert reachable["constraint_met"]
    assert not capped["constraint_met"]  # duty cannot exceed 1, so the mean is capped
    assert capped["achieved_mean_mps"] <= hi + 1e-9
    assert reachable["burst_s"] > 0


def test_mean_rate_and_burst_length_describe_the_duty_cycle():
    from aerovigil.adaptive.profile import burst_seconds, mean_rate

    p = {**DEFAULT, "rate_mps": 1.2, "duty": 0.25, "period_s": 80.0}
    assert mean_rate(p) == pytest.approx(0.3)
    assert burst_seconds(p) == pytest.approx(20.0)
    assert mean_rate({**DEFAULT, "duty": 1.0}) == pytest.approx(DEFAULT["rate_mps"])


@pytest.mark.parametrize("limit,expected", [(1, 1), (2, 2), (3, 3)])
def test_search_flights_asks_for_exactly_that_many(wired, limit, expected):
    """The earlier integer division turned --search-flights 5 into 3 flights."""
    assert len(sr.normal_flights("validation", limit)) == expected
    assert len(sr.normal_flights("validation")) >= 3  # no limit means all of them
