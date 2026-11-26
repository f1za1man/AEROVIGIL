"""Plan M9 evaluation framework: shared metrics and leak-free protocol masks."""

import numpy as np
import pandas as pd
import pytest

from aerovigil.evaluation import lstm_protocols as lp
from aerovigil.evaluation import metrics as mt
from aerovigil.evaluation.physics_protocols import cusum_verdict, verdict
from aerovigil.evaluation.run_all import combine

# ---------------------------------------------------------------- metrics --


def test_poisson_bound_and_false_alarm_record():
    assert mt.poisson_upper(0, 1.0) == pytest.approx(3.0, rel=0.01)
    r = mt.false_alarms([1, 2, 0], [0.5, 0.25, 0.25])
    assert (r["fa_alarms"], r["fa_hours"], r["fa_per_hour"]) == (3, 1.0, 3.0)
    assert r["fa_upper_95"] > 3.0


def test_window_metrics_and_confusion_matrix():
    m = mt.window_metrics([0, 0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9, 0.2], 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_flight_metrics_undetected_displacement():
    f = pd.DataFrame({
        "attacked": [True, True, False], "detected": [True, False, False],
        "delay_s": [2.0, np.nan, np.nan], "alarms": [0, 0, 2],
        "hours": [np.nan, np.nan, 0.5], "displacement_m": [3.0, np.nan, np.nan],
        "final_offset_m": [50.0, 80.0, np.nan],
    })  # fmt: skip
    r = mt.flight_metrics(f)
    assert r["detected"] == 0.5 and r["median_delay_s"] == 2.0
    assert r["median_undetected_displacement_m"] == pytest.approx(41.5)  # (3 + 80) / 2
    assert r["fa_per_hour"] == 4.0


def test_bootstrap_interval_brackets_the_mean():
    v = np.r_[np.ones(30), np.zeros(10)]
    lo, hi = mt.bootstrap_ci(v)
    assert lo < 0.75 < hi and 0 <= lo and hi <= 1


def test_refractory_count():
    assert mt.refractory_count([0, 5, 9.9, 10, 25]) == 3


# --------------------------------------------------------- physics verdicts --


def sig(start=10.0, end=20.0, onset=11.0):
    t = np.arange(0, 30, 0.01)
    return {"start_s": start, "end_s": end, "onset_s": onset, "row_t": t,
            "offset_m": np.clip(t - onset, 0, None)}  # fmt: skip


def test_verdict_counts_only_inside_the_window_and_times_from_onset():
    v = verdict(np.array([5.0, 12.0]), sig())
    assert v["detected"] and v["delay_s"] == pytest.approx(1.0)
    assert v["displacement_m"] == pytest.approx(1.0) and v["alarms"] == 1
    assert not verdict(np.array([25.0]), sig())["detected"]
    clean = verdict(np.array([3.0, 8.0]), sig(np.nan, np.nan, np.nan))
    assert not clean["detected"] and clean["alarms"] == 2


def test_cusum_verdict_reads_window_and_delay_from_the_ticks():
    t = np.arange(0, 30, 0.1)
    label = ((t >= 10) & (t <= 20)).astype(float)
    inn = {"t": t + 0.116, "label": label,  # the filter's clock, offset on purpose
           "rel": np.where(label == 1, t - 10.5, np.nan), "offset": t}  # fmt: skip
    i = int(np.searchsorted(t, 12.0))
    v = cusum_verdict(np.array([i]), inn)
    assert v["detected"] and v["delay_s"] == pytest.approx(1.5)


# ------------------------------------------------------------ LSTM masks --


def meta():
    rows = []
    for mission in ("curved", "random", "straight"):
        for role in ("train", "validation", "calib", "test"):
            for kind in ("none", "original"):
                fid = f"{mission}_{role}_{kind}"
                for y in (0,) if kind == "none" else (0, 1):
                    rows.append({"flight_id": fid, "source_flight_id": fid,
                                 "origin": "orig", "attack_set": "orig",
                                 "role": role, "mission_type": mission,
                                 "attack_type": kind, "y_event": y,
                                 "velocity_mode": "na"})  # fmt: skip
            if role != "calib":
                for fam in ("linear_drift", "abrupt_jump"):
                    src = f"{mission}_{role}_none"
                    rows.append({"flight_id": f"{src}__{fam}", "source_flight_id": src,
                                 "origin": "syn", "attack_set": "main", "role": role,
                                 "mission_type": mission, "attack_type": fam,
                                 "y_event": 1, "velocity_mode": "consistent"})  # fmt: skip
    return pd.DataFrame(rows)


def job(kind, protocol, held=None):
    return {"name": "x", "kind": kind, "protocol": protocol, "held": held,
            "hidden": 32, "layers": 1, "seed": 42}  # fmt: skip


@pytest.mark.parametrize("kind", ["ae", "clf"])
@pytest.mark.parametrize("held", ["curved", "random", "straight"])
def test_unseen_mission_is_never_fitted_or_calibrated(kind, held):
    m = meta()
    tr, va, cal, test = lp.masks(m, job(kind, "mission", held))
    for mask in (tr, va, cal):
        assert not (m.loc[mask, "mission_type"] == held).any()
    assert (m.loc[test, "mission_type"] == held).all() and test.any()


def test_autoencoder_never_sees_an_attack_window():
    m = meta()
    tr, va, _, _ = lp.masks(m, job("ae", "seen"))
    assert (m.loc[tr | va, "y_event"] == 0).all()
    assert (m.loc[tr | va, "attack_type"] == "none").all()


@pytest.mark.parametrize("fam", ["linear_drift", "original"])
def test_unseen_family_is_left_out_of_fitting(fam):
    m = meta()
    tr, va, _, _ = lp.masks(m, job("clf", "type", fam))
    pos = m.loc[tr | va]
    assert not ((pos["attack_type"] == fam) & (pos["y_event"] == 1)).any()


def test_cv_fold_drops_its_flights_and_their_synthetic_copies():
    m = meta()
    fold_of = lp.cv_fold_of(m)
    tr, va, cal, test = lp.masks(m, job("clf", "cvfa", "0"), fold_of)
    out = {f for f, k in fold_of.items() if k == 0}
    used = m.loc[tr | va | cal]
    assert not used["source_flight_id"].isin(out).any()
    assert set(m.loc[test, "flight_id"]) == out


def test_combined_table_leads_with_unseen_missions():
    phy = pd.DataFrame({
        "protocol": ["seen", "unseen missions"], "detector": ["kalman", "kalman"],
        "detected": [0.8, 0.9], "detected_ci_low": [0.7, 0.8],
        "detected_ci_high": [0.9, 1.0], "fa_alarms": [1, 2], "fa_hours": [1, 2],
        "fa_per_hour": [1, 1], "fa_upper_95": [4, 3],
    })  # fmt: skip
    out = combine(None, phy, None)
    assert out["protocol"].iloc[0] == "unseen missions"
    assert out["family"].iloc[0] == "physics"
