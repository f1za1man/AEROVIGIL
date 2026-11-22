import json

import numpy as np
import pandas as pd

N_SAMPLES = 625
SPLITS = ("train", "validation", "test")


def test_window_size_consistent(paths):
    params = json.loads(paths["params"].read_text(encoding="utf-8"))
    k = len(params["feature_order"])
    for s in SPLITS:
        d = np.load(paths["windows"] / f"windows_{s}.npz")
        assert d["X"].shape[1:] == (N_SAMPLES, k), s
        assert d["X"].shape[0] == len(
            pd.read_csv(paths["windows"] / f"window_manifest_{s}.csv"))
        assert not np.isnan(d["X"]).any(), f"NaN in {s} windows"


def test_window_duration_is_constant(paths):
    m = pd.read_csv(paths["windows"] / "window_manifest.csv")
    duration = (m["end_time_usec"] - m["start_time_usec"]) / 1e6
    np.testing.assert_allclose(duration, (N_SAMPLES - 1) / 125, atol=1e-6)


def test_window_overlap_is_50_percent(paths):
    m = pd.read_csv(paths["windows"] / "window_manifest.csv")
    step_us = 2_500_000
    for fid, g in m.groupby("flight_id"):
        diffs = np.diff(np.sort(g["start_time_usec"].to_numpy()))
        if len(diffs):
            assert diffs.min() >= step_us - 1, fid
            remainder = np.abs(diffs - np.round(diffs / step_us) * step_us)
            assert remainder.max() <= 1, fid


def test_windows_do_not_mix_splits(paths):
    splits = pd.read_csv(paths["splits"]).set_index("flight_id")["split"]
    seen = {}
    for s in SPLITS:
        m = pd.read_csv(paths["windows"] / f"window_manifest_{s}.csv")
        assert (m["split"] == s).all()
        assert (m["flight_id"].map(splits) == s).all()
        seen[s] = set(m["flight_id"])
    assert not (seen["train"] & seen["validation"])
    assert not (seen["train"] & seen["test"])
    assert not (seen["validation"] & seen["test"])


def test_window_feature_order_matches_normalization(paths):
    params = json.loads(paths["params"].read_text(encoding="utf-8"))
    for s in SPLITS:
        d = np.load(paths["windows"] / f"windows_{s}.npz")
        assert [str(x) for x in d["feature_names"]] == params["feature_order"]


def test_labels_attached_for_evaluation_not_input(paths):
    d = np.load(paths["windows"] / "windows_train.npz")
    assert d["attack_fraction"].shape[0] == d["X"].shape[0]
    assert not any("attack" in str(n) or "label" in str(n)
                   for n in d["feature_names"])