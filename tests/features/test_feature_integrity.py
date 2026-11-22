import numpy as np
import pandas as pd

KEY_COLUMNS = ["flight_id", "mission_type", "time_usec",
               "attack_label", "after_gap"]


def test_consistent_feature_ordering_across_flights(paths):
    files = sorted(paths["features"].glob("*.parquet"))
    reference = list(pd.read_parquet(files[0]).columns)
    for f in files:
        assert list(pd.read_parquet(f).columns) == reference, f.name
    for f in sorted(paths["normalized"].glob("*.parquet")):
        assert list(pd.read_parquet(f).columns) == reference, f.name


def test_row_counts_preserved_through_pipeline(paths):
    cleaned = sorted(paths["cleaned"].glob("*.parquet"))
    for c in cleaned:
        rows = len(pd.read_parquet(c, columns=["time_usec"]))
        assert len(pd.read_parquet(
            paths["features"] / c.name, columns=["time_usec"])) == rows
        assert len(pd.read_parquet(
            paths["normalized"] / c.name, columns=["time_usec"])) == rows


def test_labels_preserved_and_not_features(paths):
    cleaned_attack = features_attack = 0
    for c in sorted(paths["cleaned"].glob("*.parquet")):
        cleaned_attack += int(
            (pd.read_parquet(c, columns=["attack_label"])
             ["attack_label"] == 1).sum())
        features_attack += int(
            (pd.read_parquet(paths["features"] / c.name,
                             columns=["attack_label"])
             ["attack_label"] == 1).sum())
    assert cleaned_attack == features_attack == 4_302_145


def test_features_finite_where_defined(paths):
    f = sorted(paths["features"].glob("*.parquet"))[0]
    g = pd.read_parquet(f)
    values = g.drop(columns=KEY_COLUMNS).to_numpy()
    assert not np.isinf(values).any()