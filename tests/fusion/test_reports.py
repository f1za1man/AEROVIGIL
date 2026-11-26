"""Module 10 report tests: adaptive replay summary and the threat-model table."""

import numpy as np
import pandas as pd
import pytest

import aerovigil.fusion.threat_model as tm
from aerovigil.fusion.adaptive import label_of, summarise


def test_attacker_classes_follow_the_set_mode_and_rate():
    assert tm.attacker_class("orig", "na", np.nan) == "real"
    assert tm.attacker_class("main", "position_only", np.nan) == "position-only"
    assert tm.attacker_class("syn", "consistent", np.nan) == "consistent"
    assert tm.attacker_class("sweep", "consistent", 0.05) == "drift slow"
    assert tm.attacker_class("sweep", "consistent", 0.1) == "drift slow"
    assert tm.attacker_class("sweep", "consistent", 0.25) == "drift medium"
    assert tm.attacker_class("sweep", "consistent", 0.5) == "drift fast"
    with pytest.raises(ValueError):
        tm.attacker_class("other", "na", np.nan)


def test_summary_pools_flights_not_medians_and_counts_seeds_once():
    long = pd.DataFrame({
        "detector": "D", "attacker": "real",
        "flight_id": ["a", "a", "b", "b"],  # two seeds per flight
        "detected": [True, False, True, True], "delay_s": [1.0, np.nan, 3.0, 5.0],
    })  # fmt: skip
    s = tm.summarise(long).iloc[0]
    assert s["flights"] == 2
    assert s["detected"] == pytest.approx(0.75)  # the mean over seeds
    assert s["median_delay_s"] == pytest.approx(3.0)  # only caught flights


def test_table_marks_unmeasured_cells():
    summary = pd.DataFrame([{"detector": "M10 fused", "attacker": "real",
                             "flights": 3, "detected": 1.0,
                             "median_delay_s": 0.2}])  # fmt: skip
    fa = pd.DataFrame([
        {"detector": "M10 fused", "scope": "normal test flights", "flights": 21,
         "alarms_per_hour": 1.0},
        {"detector": "M10 fused", "scope": "cross-validated, all normal flights",
         "flights": 144, "alarms_per_hour": 1.33},
    ])  # fmt: skip
    tab = tm.table(summary, fa)
    real = tab[tab["attacker"].str.startswith("Dataset")].iloc[0]
    assert real["M10 fused"].startswith("100%") and real["M05 RF"] == "n/a"
    assert tab.iloc[-2]["M10 fused"] == "1.00/h"
    assert tab.iloc[-1]["M10 fused"] == "1.33/h" and tab.iloc[-1]["flights"] == 144
    assert tab.iloc[-1]["M06 chi2"] == "n/a"


def test_each_detector_is_divided_by_its_own_monitored_time():
    index = pd.DataFrame({"flight_id": ["a", "b"], "kf_hours": [0.03, 0.05],
                          "vel_hours": [0.04, 0.04]})  # fmt: skip
    assert tm.monitored_hours(index, "kalman").sum() == pytest.approx(0.08)
    assert tm.monitored_hours(index, "rows").sum() == pytest.approx(0.08)
    assert tm.monitored_hours(index, "fused").tolist() == [0.04, 0.05]
    assert tm.HOURS_BASIS["M10 kalman"] == "kalman"


def test_adaptive_labels_and_displacement_when_missed():
    assert label_of("mean_rate=0.25") == 0.25 and np.isnan(label_of("free"))
    flights = pd.DataFrame({
        "search": "mean_rate=1", "flight_id": ["a", "b"],
        "final_offset_m": [100.0, 200.0], "achieved_mean_mps": [1.0, 1.1],
        **{f"{d}_detected": [True, False] for d in ("kalman", "velocity", "fused")},
        **{f"{d}_delay_s": [2.0, np.nan] for d in ("kalman", "velocity", "fused")},
        **{f"{d}_displacement_m": [4.0, np.nan]
           for d in ("kalman", "velocity", "fused")},
    })  # fmt: skip
    s = summarise(flights).iloc[0]
    assert s["kalman_detected"] == 0.5 and s["kalman_delay_s"] == 2.0
    # caught at 4 m on one flight, never caught on the other (moved 200 m)
    assert s["kalman_median_displacement_m"] == pytest.approx(102.0)
