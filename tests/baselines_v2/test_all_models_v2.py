"""Plan M5 (all six classical models): splits, threshold, latch and metrics."""

import numpy as np
import pandas as pd
import pytest

import aerovigil.baselines.all_models_v2 as am


def frame(n_per=4, seed=0):
    """Windows from 2 flights per (mission, split, kind), real and synthetic."""
    rng = np.random.default_rng(seed)
    rows = []
    for mission in am.MISSIONS:
        for split in ("train", "validation", "test"):
            for kind in ("none", "original"):
                for f in range(2):
                    fid = f"{mission}_{split}_{kind}_{f}"
                    for w in range(n_per):
                        y = int(kind == "original" and w >= n_per // 2)
                        rows.append({
                            "flight_id": fid, "source_flight_id": fid,
                            "origin": "orig", "mission_type": mission,
                            "split": split, "attack_type": kind,
                            "velocity_mode": "na", "attack_set": "orig",
                            "window_start_s": 2.5 * w, "window_end_s": 2.5 * w + 5,
                            "seconds_since_onset_end": (
                                2.5 * (w - n_per // 2) if kind == "original"
                                else np.nan
                            ),
                            am.Y: y, "f__a": rng.normal(3 * y, 1),
                        })  # fmt: skip
    orig = pd.DataFrame(rows)
    syn = orig[orig[am.Y] == 1].assign(
        origin="syn", attack_set="main", velocity_mode="consistent",
        attack_type="linear_drift",
        flight_id=lambda d: d["flight_id"] + "__syn",
    )  # fmt: skip
    return orig, syn


@pytest.mark.parametrize("held", am.MISSIONS)
def test_unseen_mission_never_reaches_training_tuning_or_calibration(held):
    orig, syn = frame()
    tr, va, clean_va, o_test, s_test = am.protocol_sets(orig, syn, held)
    for part in (tr, va, clean_va):
        assert not (part["mission_type"] == held).any()
    assert (o_test["mission_type"] == held).all()
    assert (s_test["mission_type"] == held).all()
    # every flight of the held-out mission is tested, whatever its split
    assert set(o_test["split"]) == {"train", "validation", "test"}


def test_seen_protocol_is_flight_disjoint():
    orig, syn = frame()
    tr, va, clean_va, o_test, s_test = am.protocol_sets(orig, syn, None)
    test_ids = set(o_test["flight_id"]) | set(s_test["flight_id"])
    for part in (tr, va, clean_va):
        assert not test_ids & set(part["flight_id"])
    assert (o_test["split"] == "test").all()


def test_threshold_allows_the_budget_and_no_more():
    pool = np.arange(1440, dtype=float)  # 1440 windows x 2.5 s = 1 h
    thr, hours, allowed = am.threshold_for(pool, budget=1.0)
    assert hours == pytest.approx(1.0) and allowed == 1
    assert (pool > thr).sum() == 0  # the single allowed alarm sits AT the threshold
    thr2, _, allowed2 = am.threshold_for(pool, budget=3.0)
    assert allowed2 == 3 and (pool > thr2).sum() == 2


def test_refractory_merges_alarms_within_ten_seconds():
    assert am.refractory_count([0, 2.5, 5, 7.5, 10, 30]) == 3
    assert am.refractory_count([]) == 0


def test_latch_counts_detection_only_after_onset():
    orig, _ = frame()
    f = orig[orig["flight_id"] == "curved_test_original_0"]
    s = f["f__a"].to_numpy()
    out = am.latch(f, np.where(f[am.Y] == 1, 10.0, -10.0), 0.0).iloc[0]
    assert out["detected"] and out["delay_s"] == 0.0
    early = am.latch(f, np.where(f[am.Y] == 0, 10.0, -10.0), 0.0).iloc[0]
    assert not early["detected"]  # an alarm before onset is not a detection
    assert len(s) == 4


def test_window_metrics_match_the_confusion_matrix():
    m = am.window_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["f1"] == 0.5


def test_naive_bayes_scores_do_not_saturate():
    """Its probability rounded to 1.0 for many windows and blocked every alarm."""
    orig, _ = frame()
    pipe = am.fit("naive_bayes", {}, orig.assign(f__a=orig["f__a"] * 50), ["f__a"])
    s = am.score("naive_bayes", pipe, orig.assign(f__a=orig["f__a"] * 50), ["f__a"])
    assert len(np.unique(s)) > len(s) // 2  # log-odds keep the ranking apart
