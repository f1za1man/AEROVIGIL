import json

import numpy as np
import pandas as pd

LABEL_WORDS = ("attack", "label", "flight", "mission")


def test_timestamps_ordered_and_unique(paths):
    files = sorted(paths["cleaned"].glob("*.parquet"))
    assert len(files) == 360
    for f in files:
        t = pd.read_parquet(f, columns=["time_usec"])["time_usec"].to_numpy()
        assert (np.diff(t) > 0).all(), f"{f.name}: not strictly increasing"


def test_no_missing_flight_ids(paths):
    splits = pd.read_csv(paths["splits"])
    assert splits["flight_id"].notna().all()
    assert (splits["flight_id"].astype(str).str.len() > 0).all()
    ids = set()
    for f in sorted(paths["cleaned"].glob("*.parquet")):
        fid = pd.read_parquet(f, columns=["flight_id"])["flight_id"]
        assert fid.notna().all()
        assert fid.nunique() == 1, f"{f.name}: mixed flight IDs"
        ids.add(fid.iloc[0])
    assert len(ids) == 360
    assert ids == set(splits["flight_id"])


def test_split_flights_do_not_overlap(paths):
    s = pd.read_csv(paths["splits"])
    assert s["flight_id"].is_unique
    sets = {k: set(s.loc[s["split"] == k, "flight_id"])
            for k in ("train", "validation", "test")}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    assert sum(len(v) for v in sets.values()) == 360


def test_feature_columns_exclude_labels(paths):
    params = json.loads(paths["params"].read_text(encoding="utf-8"))
    for name in params["feature_order"]:
        assert not any(w in name.lower() for w in LABEL_WORDS), name
    first = sorted(paths["normalized"].glob("*.parquet"))[0]
    cols = pd.read_parquet(first).columns
    feature_cols = [c for c in cols if c not in (
        "flight_id", "mission_type", "time_usec", "attack_label",
        "after_gap")]
    assert feature_cols == params["feature_order"]


def test_normalization_fitted_on_training_flights_only(paths):
    params = json.loads(paths["params"].read_text(encoding="utf-8"))
    splits = pd.read_csv(paths["splits"]).set_index("flight_id")["split"]
    fit_ids = params["training_flight_ids"]
    assert params["fitted_on_split"] == "train"
    assert len(fit_ids) == (splits == "train").sum()
    assert all(splits[f] == "train" for f in fit_ids)

    # Recompute mean/std from training flights and compare
    names = params["feature_order"]
    n = np.zeros(len(names))
    s = np.zeros(len(names))
    ss = np.zeros(len(names))
    for f in sorted(paths["features"].glob("*.parquet")):
        g = pd.read_parquet(f, columns=["flight_id"] + names)
        if splits[g["flight_id"].iloc[0]] != "train":
            continue
        x = g[names].to_numpy(dtype="float64")
        n += (~np.isnan(x)).sum(axis=0)
        s += np.nansum(x, axis=0)
        ss += np.nansum(x ** 2, axis=0)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0))
    np.testing.assert_allclose(
        mean, [params["mean"][k] for k in names], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        std, [params["std"][k] for k in names], rtol=1e-3, atol=1e-6)