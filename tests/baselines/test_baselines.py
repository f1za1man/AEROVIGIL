"""Module 05 tests: windows, labels, protocol leakage and metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.baselines.datasets import (assemble, balance_classes,  # noqa: E402
                                          build_test_sets, stat_columns)
from aerovigil.baselines.models import (build_model, evaluate, fit_model,  # noqa: E402
                                        grid, score_model, tpr_at_fpr)
from aerovigil.baselines.windows import (STATS, label_window, stat_names,  # noqa: E402
                                         summarise_flight, window_bounds)


def flight(n=2500, hz=250.0, k=3):
    t = np.arange(n) / hz
    x = np.column_stack([np.sin(t * (i + 1)) + i for i in range(k)])
    return t, x


# ------------------------------ windows ------------------------------------
def test_window_bounds_count_and_grid():
    t = np.arange(0, 20.0, 0.004)                        # 20 s at 250 Hz
    starts, a, b = window_bounds(t, 5.0, 2.5)
    assert np.allclose(np.diff(starts), 2.5)
    assert len(starts) == 6                              # 0 ... 12.5 s; 15 s would not fit
    assert (b - a == 1250).all()


def test_flight_shorter_than_one_window_gives_no_windows():
    t, x = flight(n=500)                                 # 2 s
    stats, meta, _ = summarise_flight(x, t, np.zeros(500), np.zeros(500, bool), 5.0, 2.5)
    assert len(meta) == 0 and stats.shape[0] == 0


def test_statistics_are_correct():
    t, x = flight()
    stats, meta, _ = summarise_flight(x, t, np.zeros(len(t)), np.zeros(len(t), bool), 5.0, 2.5)
    first = stats[0].reshape(x.shape[1], len(STATS))
    rows = (t >= 0) & (t < 5.0)
    xs = x[rows]
    assert np.allclose(first[:, 0], xs.mean(axis=0))
    assert np.allclose(first[:, 1], xs.std(axis=0))
    assert np.allclose(first[:, 2], xs.min(axis=0)) and np.allclose(first[:, 3], xs.max(axis=0))
    assert np.allclose(first[:, 4], xs[-1])
    assert np.allclose(first[:, 5], (xs[-1] - xs[0]) / (t[rows][-1] - t[rows][0]))
    assert len(stat_names(["a", "b", "c"])) == 3 * len(STATS)


def test_window_spanning_a_log_gap_is_dropped():
    t, x = flight()
    gap = np.zeros(len(t), bool); gap[1200] = True       # first sample after a small gap
    t = t.copy(); t[1200:] += 0.5
    _, meta, dropped = summarise_flight(x, t, np.zeros(len(t)), gap, 5.0, 2.5)
    for m in meta:                                       # no kept window contains the gap row
        i0 = np.searchsorted(t, m["window_start_s"])
        i1 = i0 + m["n_rows"]
        assert not gap[i0 + 1:i1].any()
    assert dropped["spans_gap"] + dropped["short_span"] >= 1


def test_window_with_too_many_nans_is_dropped():
    t, x = flight()
    x[:, 1] = np.nan                                     # a feature that is never valid
    _, meta, dropped = summarise_flight(x, t, np.zeros(len(t)), np.zeros(len(t), bool), 5.0, 2.5)
    assert len(meta) == 0 and dropped["too_many_nan"] > 0


def test_attack_fraction_per_window():
    t, x = flight()
    y = (t >= 2.5).astype(float)                         # attack from 2.5 s on
    _, meta, _ = summarise_flight(x, t, y, np.zeros(len(t), bool), 5.0, 2.5)
    assert meta[0]["attack_fraction"] == pytest.approx(0.5, abs=0.01)
    assert meta[1]["attack_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("fraction,expected", [(0.0, (0, False)), (0.02, (-1, True)),
                                               (0.49, (-1, True)), (0.5, (1, False)),
                                               (1.0, (1, False))])
def test_label_rule(fraction, expected):
    assert label_window(fraction, 0.5) == expected


# ------------------------ protocols and leakage ----------------------------
def make_tables():
    rows = []
    rng = np.random.default_rng(0)
    for mission in ("curved", "random", "straight"):
        for split, n_flights in (("train", 4), ("validation", 2), ("test", 2)):
            for j in range(n_flights):
                for origin in ("orig", "syn"):
                    if origin == "syn" and j % 2:
                        continue
                    fid = f"{mission}_{split}_{j}" + ("_syn" if origin == "syn" else "")
                    for w in range(6):
                        y = 1 if (origin == "syn" or (j % 2 and w > 2)) else 0
                        rows.append({"flight_id": fid, "source_flight_id": f"{mission}_{split}_{j}",
                                     "origin": origin, "split": split, "mission_type": mission,
                                     "attack_type": "linear_drift" if origin == "syn" else "original",
                                     "velocity_mode": ["consistent", "position_only"][j % 2]
                                     if origin == "syn" else "na",
                                     "y": y, "ambiguous": False, "a__mean": rng.normal(y, 1)})
    df = pd.DataFrame(rows)
    return df[df.origin == "orig"].reset_index(drop=True), df[df.origin == "syn"].reset_index(drop=True)


def test_seen_protocol_is_flight_disjoint():
    orig, syn = make_tables()
    d = assemble(orig, syn, None, ("orig", "syn"))
    ids = {k: set(d[k]["flight_id"]) for k in ("train", "val", "test_orig", "test_syn")}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test_orig"]
    assert not ids["val"] & ids["test_orig"] and not ids["train"] & ids["test_syn"]
    assert set(d["train"]["split"]) == {"train"} and set(d["val"]["split"]) == {"validation"}
    # a synthetic flight never lands in a different split from its source flight
    assert set(d["test_syn"]["split"]) == {"test"}


def test_unseen_mission_protocol_never_trains_on_the_held_out_type():
    orig, syn = make_tables()
    for held in ("curved", "random", "straight"):
        d = assemble(orig, syn, held, ("orig", "syn"))
        assert held not in set(d["train"]["mission_type"]) | set(d["val"]["mission_type"])
        assert set(d["test_orig"]["mission_type"]) == {held} == set(d["test_syn"]["mission_type"])


def test_orig_only_training_excludes_synthetic_windows_but_tests_keep_them():
    orig, syn = make_tables()
    d = assemble(orig, syn, None, ("orig",))
    assert set(d["train"]["origin"]) == {"orig"} and len(d["test_syn"]) > 0
    with pytest.raises(ValueError):
        assemble(orig, syn, None, ("syn",))


def test_ambiguous_windows_are_removed_everywhere():
    orig, syn = make_tables()
    orig.loc[orig.index[::7], "ambiguous"] = True
    d = assemble(orig, syn, None, ("orig", "syn"))
    for k in d:
        assert not d[k]["ambiguous"].any()


def test_balancing_equalises_classes_and_is_seeded():
    orig, syn = make_tables()
    df = pd.concat([orig, syn], ignore_index=True)
    b1, b2 = balance_classes(df, 1), balance_classes(df, 1)
    assert (b1["y"] == 1).sum() == (b1["y"] == 0).sum()
    pd.testing.assert_frame_equal(b1, b2)


def test_test_sets_share_one_normal_pool():
    orig, syn = make_tables()
    d = assemble(orig, syn, None, ("orig", "syn"))
    sets = build_test_sets(d["test_orig"], d["test_syn"])
    assert {"orig_test", "syn_all", "syn:consistent", "syn:position_only"} <= set(sets)
    neg = (d["test_orig"]["y"] == 0).sum()
    assert (sets["syn_all"]["y"] == 0).sum() == neg
    assert (sets["syn:consistent"]["y"] == 1).sum() + (sets["syn:position_only"]["y"] == 1).sum() \
        == (sets["syn_all"]["y"] == 1).sum()


def test_stat_columns_only_contain_summary_features():
    orig, _ = make_tables()
    assert stat_columns(orig) == ["a__mean"]


# ------------------------------- metrics ------------------------------------
def test_tpr_at_fpr_perfect_and_random_scores():
    y = np.array([0] * 100 + [1] * 100)
    assert tpr_at_fpr(y, np.r_[np.zeros(100), np.ones(100)], 0.01) == 1.0
    assert tpr_at_fpr(y, np.arange(200) % 7, 0.01) < 0.5


def test_evaluate_returns_none_for_single_class_and_correct_values():
    assert evaluate([1, 1, 1], [0.9, 0.8, 0.7], 0.5) is None
    m = evaluate([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)


@pytest.mark.parametrize("name,params", [
    ("logistic_regression", {"C": 1.0}), ("naive_bayes", {"var_smoothing": 1e-9}),
    ("knn", {"n_neighbors": 5}), ("decision_tree", {"max_depth": 4, "min_samples_leaf": 1}),
    ("random_forest", {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 1}),
    ("isolation_forest", {"n_estimators": 20, "max_samples": 5000, "max_features": 1.0})])
def test_every_model_fits_and_scores_with_nans_present(name, params):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    x = rng.normal(y[:, None] * 2.0, 1.0, size=(400, 6))
    x[rng.random(x.shape) < 0.02] = np.nan               # the imputer must cope
    pipe = fit_model(name, build_model(name, params, 0), x, y)
    s = score_model(name, pipe, x)
    assert s.shape == (400,) and np.isfinite(s).all()
    assert evaluate(y, s, 0.5)["roc_auc"] > 0.9


def test_grid_expands_all_combinations():
    assert len(grid({"a": [1, 2], "b": [None, 3, 4]})) == 6