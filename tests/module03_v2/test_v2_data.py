"""Module 03 v2 tests on the generated data. Missing outputs fail (see conftest.py)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "processed" / "module03_v2"
REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
EXPECTED_FLIGHTS = 360
EXPECTED_SPLIT = {"train": 252, "validation": 54, "test": 54}
LABEL_WORDS = ("attack", "label", "flight", "phase", "spoof", "offset")


def base_files(require):
    require(V2 / "base")
    files = sorted((V2 / "base").glob("*.parquet"))
    assert files, "no base files"
    return files


def test_base_has_one_file_per_flight_and_matches_report(require):
    files = base_files(require)
    report = pd.read_csv(require(REPORT))
    assert len(files) == EXPECTED_FLIGHTS == len(report)
    rows = sum(len(pd.read_parquet(f, columns=["t_s"])) for f in files)
    assert rows == int(report["rows"].sum())


def test_base_timestamps_strictly_increase(require):
    for path in base_files(require):
        t = pd.read_parquet(path, columns=["time_usec"])["time_usec"].to_numpy()
        assert (np.diff(t) > 0).all(), path.name


def test_original_label_is_kept_unchanged(require):
    for path in base_files(require):
        g = pd.read_parquet(path, columns=["attack_label", "attack_enabled"])
        assert (g["attack_label"] == g["attack_enabled"]).all(), path.name


def test_normal_flights_have_no_spoofing_and_phases_are_ordered(require):
    for path in base_files(require):
        g = pd.read_parquet(path, columns=["flight_id", "phase", "spoof_truth"])
        phase = g["phase"].to_numpy().astype(int)
        if "__Normal__" in g["flight_id"].iloc[0]:
            assert (phase == 0).all() and not g["spoof_truth"].any(), path.name
        else:
            assert (np.diff(phase) >= 0).all(), path.name
            assert set(np.unique(phase)) <= {1, 2, 3, 4}, path.name


def test_no_spoofed_row_is_labelled_normal(require):
    report = pd.read_csv(require(REPORT))
    assert int(report["spoof_not_label_rows"].sum()) == 0


def test_feature_files_match_base_rows(require):
    base = {
        f.name: len(pd.read_parquet(f, columns=["t_s"])) for f in base_files(require)
    }
    require(V2 / "features")
    for name, rows in base.items():
        feat = require(V2 / "features" / name)
        assert len(pd.read_parquet(feat, columns=["t_s"])) == rows, name


def test_feature_metadata_has_no_label_columns(require):
    meta = pd.read_csv(require(V2 / "feature_metadata.csv"))
    bad = [f for f in meta["feature"] if any(w in f.lower() for w in LABEL_WORDS)]
    assert not bad, bad
    assert len(meta) == 36


def test_split_is_the_unchanged_seed_42_split(require):
    split = pd.read_csv(require(SPLITS))
    assert split["split"].value_counts().to_dict() == EXPECTED_SPLIT


def test_normalization_used_clean_training_rows_only(require):
    params = json.loads(require(V2 / "normalization_parameters.json").read_text())
    split = pd.read_csv(require(SPLITS))
    assert params["fit_phases"] == [0, 1]
    assert params["fit_flights"] == EXPECTED_SPLIT["train"]
    assert params["clip"] == 20.0
    assert all(v > 0 for v in params["scale"].values())
    assert len(params["features"]) == 36 and len(split) == EXPECTED_FLIGHTS


def test_normalized_values_are_clipped_and_centred_on_clean_rows(require):
    split = pd.read_csv(require(SPLITS)).set_index("flight_id")["split"]
    require(V2 / "normalized")
    clean = []
    for path in sorted((V2 / "normalized").glob("*.parquet"))[::6]:
        g = pd.read_parquet(path)
        feats = [c for c in g.columns if c.startswith(("res_", "gps_"))]
        assert np.nanmax(np.abs(g[feats].to_numpy())) <= 20.0 + 1e-4, path.name
        if split[g["flight_id"].iloc[0]] == "train":
            clean.append(g.loc[g["phase"].isin([0, 1]), feats].iloc[::10])
    centre = pd.concat(clean).median().abs()
    assert (centre < 1.0).all(), centre[centre >= 1.0]


def load_windows(require, name):
    return np.load(require(V2 / "windows" / f"windows_{name}.npz"))


def test_window_arrays_and_manifests_agree(require):
    manifest = pd.read_csv(require(V2 / "windows" / "window_manifest.csv"))
    for name in EXPECTED_SPLIT:
        data = load_windows(require, name)
        assert data["X"].shape[1:] == (625, 36)
        assert not np.isnan(data["X"]).any()
        assert len(data["X"]) == int((manifest["split"] == name).sum())
        assert len(data["window_class"]) == len(data["X"])


def test_windows_keep_the_seed_42_split(require):
    split = pd.read_csv(require(SPLITS)).set_index("flight_id")["split"]
    manifest = pd.read_csv(require(V2 / "windows" / "window_manifest.csv"))
    assert (manifest["split"] == manifest["flight_id"].map(split)).all()
    flights = {s: set(g["flight_id"]) for s, g in manifest.groupby("split")}
    assert not flights["train"] & flights["validation"]
    assert not flights["train"] & flights["test"]
    assert not flights["validation"] & flights["test"]


def test_window_classes_are_consistent_with_the_measured_offset(require):
    m = pd.read_csv(require(V2 / "windows" / "window_manifest.csv"))
    assert set(m["window_class"]) == {
        "normal",
        "pre",
        "ramp",
        "plateau",
        "snapback",
        "post",
    }
    train = m[m["split"] == "train"]
    assert set(train["window_class"]) == set(m["window_class"])
    normal = m[m["window_class"] == "normal"]
    assert (normal["spoof_fraction"] == 0).all() and (normal["any_attack"] == 0).all()
    plateau = m[m["window_class"] == "plateau"]
    assert plateau["max_offset_m"].median() > 5.0
    assert (m[m["window_class"] == "ramp"]["spoof_fraction"] > 0).all()
