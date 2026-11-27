"""Plan M8 (Step D): verdict rules, window alarms, frontier and white-box aggregation."""

import numpy as np
import pandas as pd
import pytest

from aerovigil.stealth import detectors as D
from aerovigil.stealth.adaptive import rate_of
from aerovigil.stealth.adaptive import summarise as adaptive_summary
from aerovigil.stealth.frontier import moved
from aerovigil.stealth.whitebox import linear, worst_case


def frame(attacked=True):
    """A 30 s flight at 10 Hz: pre-phase from 10 s, ramp from 11 s, end 20 s."""
    t = np.arange(0, 30, 0.1)
    label = ((t >= 10) & (t <= 20)).astype(float) if attacked else np.zeros_like(t)
    phase = np.where((t >= 11) & (t <= 20), 2, np.where(label == 1, 1, 0))
    offset = np.where(label == 1, np.clip(t - 11, 0, None), 0.0)
    return pd.DataFrame({"t_s": t, "attack_label": label, "phase": phase,
                         "offset_m": offset})  # fmt: skip


def test_first_alarm_inside_the_label_window_is_the_detection():
    fl = D.Flight(frame(), None, None)
    v = fl.verdict([5.0, 13.0, 25.0])
    assert v["detected"] and v["delay_s"] == pytest.approx(2.0)  # from the ramp
    assert v["displacement_m"] == pytest.approx(2.0, abs=0.01)
    assert v["alarms_before_onset"] == 1 and v["n_alarms"] == 3
    assert v["final_offset_m"] == pytest.approx(9.0, abs=0.11)


def test_a_missed_attack_scores_the_whole_drift():
    fl = D.Flight(frame(), None, None)
    v = fl.verdict([2.0, 25.0])
    assert not v["detected"] and v["score"] == v["final_offset_m"]
    assert v["achieved_mean_mps"] == pytest.approx(9.0 / 10.0, rel=0.02)


def test_clean_flight_counts_alarms_only():
    v = D.Flight(frame(False), None, None).verdict([3.0, 8.0])
    assert not v["detected"] and v["n_alarms"] == 2 and v["ok"]


class FakeWindows(D.Windowed):
    def __init__(self, ends, scores, thr):
        self.ends, self.scores, self.threshold = ends, scores, thr

    def window_scores(self, fl):
        return np.asarray(self.ends, float), np.asarray(self.scores, float)


def test_window_alarms_are_sorted_thresholded_and_refractory():
    det = FakeWindows([17.5, 5.0, 7.5, 25.0], [0.9, 0.9, 0.9, 0.2], 0.5)
    assert det.alarm_times(None).tolist() == [5.0, 17.5]  # 7.5 inside 10 s of 5.0


def test_moved_is_displacement_when_caught_and_drift_when_not():
    f = pd.DataFrame({"detected": [True, False], "displacement_m": [2.0, np.nan],
                      "final_offset_m": [50.0, 80.0]})  # fmt: skip
    assert moved(f).tolist() == [2.0, 80.0]


def test_white_box_hides_if_any_candidate_hides():
    rows = []
    for c, (det, disp, fin) in enumerate([(True, 3.0, 40.0), (False, np.nan, 60.0)]):
        rows.append({"detector": "kalman", "mean_rate_mps": 0.5, "flight_id": "a",
                     "candidate": c, "detected": det, "displacement_m": disp,
                     "final_offset_m": fin})  # fmt: skip
    rows.append({"detector": "kalman", "mean_rate_mps": 0.5, "flight_id": "b",
                 "candidate": 0, "detected": True, "displacement_m": 4.0,
                 "final_offset_m": 40.0})  # fmt: skip
    per, table = worst_case(pd.DataFrame(rows))
    a, b = per.set_index("flight_id").loc["a"], per.set_index("flight_id").loc["b"]
    assert not a["detected"] and a["moved_m"] == 60.0
    assert b["detected"] and b["moved_m"] == 4.0
    assert table["detected"].iloc[0] == 0.5


def test_adaptive_summary_averages_seeds_per_flight():
    f = pd.DataFrame({
        "tag": "before", "detector": "random_forest", "search": "mean_rate=0.5",
        "seed": [42, 43, 42, 43], "flight_id": ["a", "a", "b", "b"],
        "detected": [True, False, False, False],
        "displacement_m": [1.0, np.nan, np.nan, np.nan],
        "final_offset_m": [30.0, 30.0, 50.0, 50.0],
        "achieved_mean_mps": 0.5, "alarms_before_onset": 0,
    })  # fmt: skip
    s = adaptive_summary(f).iloc[0]
    assert s["detected"] == pytest.approx(0.25) and s["seeds"] == 2
    assert s["D_m"] == pytest.approx((15.5 + 50.0) / 2)
    assert rate_of("mean_rate=0.25") == 0.25 and np.isnan(rate_of("free"))


def test_plain_linear_candidate_is_a_steady_drift():
    p = linear(0.25)
    assert p["duty"] == 1.0 and p["rate_mps"] == 0.25 and p["jitter_m"] == 0.0
