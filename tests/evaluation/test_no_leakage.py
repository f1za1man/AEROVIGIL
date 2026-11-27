"""Plan M14 (Step I): no window, sequence or model ever crosses the flight split.

Checked on the real files every result was computed from (skipped where they have not
been built): the seed-42 flight split, Module 05's window tables, Module 07's LSTM
sequences, and the train / validation / calibration / test sets the protocols build
from them. A synthetic attack is a copy of a clean flight, so it must sit in the same
split as its source flight; otherwise the same flight would be trained on and tested.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
M05 = ROOT / "data" / "processed" / "module05_v2"
M07 = ROOT / "data" / "processed" / "module07_v2"
WINDOW_S, HOP_S = 5.0, 2.5


def need(*paths):
    for p in paths:
        if not p.is_file():
            pytest.skip(f"missing {p.relative_to(ROOT)}")


@pytest.fixture(scope="module")
def split():
    need(SPLITS)
    s = pd.read_csv(SPLITS)
    return s.set_index("flight_id")["split"]


@pytest.fixture(scope="module")
def windows():
    need(M05 / "windows_orig.parquet", M05 / "windows_syn.parquet")
    cols = ["flight_id", "source_flight_id", "origin", "split", "attack_type",
            "attack_set", "window_start_s", "window_end_s", "y_event"]  # fmt: skip
    return pd.concat([pd.read_parquet(M05 / f"windows_{s}.parquet", columns=cols)
                      for s in ("orig", "syn")], ignore_index=True)  # fmt: skip


@pytest.fixture(scope="module")
def sequences():
    need(M07 / "meta_orig.parquet", M07 / "meta_syn.parquet")
    return pd.concat([pd.read_parquet(M07 / f"meta_{s}.parquet")
                      for s in ("orig", "syn")], ignore_index=True)  # fmt: skip


def source(frame):
    return frame["source_flight_id"].fillna(frame["flight_id"])


def test_split_is_seed_42_252_54_54_and_each_flight_once(split):
    assert split.index.is_unique
    assert split.value_counts().to_dict() == {"train": 252, "validation": 54,
                                              "test": 54}  # fmt: skip


def test_every_window_carries_its_source_flights_split(split, windows):
    assert (windows["split"] == source(windows).map(split)).all()
    # a flight's windows never span two splits
    assert (windows.groupby("flight_id")["split"].nunique() == 1).all()


def test_every_window_lies_inside_one_flight_and_has_the_window_length(windows):
    span = windows["window_end_s"] - windows["window_start_s"]
    assert np.allclose(span, WINDOW_S, atol=0.05)
    assert (windows["window_start_s"] >= -1e-9).all()


def test_synthetic_attacks_come_only_from_normal_flights(split, windows):
    syn = windows[windows["origin"] == "syn"]
    src = syn["source_flight_id"].unique()
    orig = windows[windows["origin"] == "orig"].drop_duplicates("flight_id")
    normal = set(orig.loc[orig["attack_type"] == "none", "flight_id"])
    assert set(src) <= normal


def test_seen_protocol_sets_share_no_source_flight(windows):
    import aerovigil.baselines.all_models_v2 as am

    orig = windows[windows["origin"] == "orig"]
    syn = windows[(windows["origin"] == "syn") & (windows["attack_set"] == "main")]
    tr, va, clean_va, o_test, s_test = am.protocol_sets(orig, syn, None, "seen")
    test = set(source(o_test)) | set(source(s_test))
    assert not set(source(tr)) & set(source(va))
    assert not (set(source(tr)) | set(source(va))) & test
    assert set(source(clean_va)) <= set(source(va))


@pytest.mark.parametrize("held", ["curved", "random", "straight"])
def test_unseen_mission_sets_hold_out_the_whole_mission(windows, held):
    import aerovigil.baselines.all_models_v2 as am

    orig = windows[windows["origin"] == "orig"]
    syn = windows[(windows["origin"] == "syn") & (windows["attack_set"] == "main")]
    frames = am.protocol_sets(orig.assign(mission_type=orig["flight_id"].str.split(
        "__").str[0].str.lower()), syn.assign(mission_type=syn["flight_id"].str.split(
        "__").str[0].str.lower()), held, "mission")  # fmt: skip
    tr, va, _, o_test, s_test = frames
    assert not (tr["flight_id"].str.lower().str.startswith(held)).any()
    assert not (va["flight_id"].str.lower().str.startswith(held)).any()
    assert o_test["flight_id"].str.lower().str.startswith(held).all()


def test_lstm_sequences_keep_the_split_and_roles(split, sequences):
    s = sequences
    assert (s["split"] == source(s).map(split)).all()
    # role follows the split; calibration windows come from training flights only
    assert set(s.loc[s["split"] == "test", "role"]) == {"test"}
    assert set(s.loc[s["split"] == "validation", "role"]) == {"validation"}
    assert set(s.loc[s["split"] == "train", "role"]) <= {"train", "calib"}
    # the role follows the source flight, so a calibration flight's synthetic copies
    # are calibration too (and therefore never trained on); the source is a normal
    # training flight
    calib = s[s["role"] == "calib"]
    calib_sources = set(source(calib))
    orig = s[s["origin"] == "orig"].drop_duplicates("flight_id").set_index("flight_id")
    assert (orig.loc[list(calib_sources), "attack_type"] == "none").all()
    assert (orig.loc[list(calib_sources), "split"] == "train").all()
    assert (s.groupby(source(s))["role"].nunique() == 1).all()


def test_calibration_windows_are_clean_real_flights_only(sequences):
    from aerovigil.evaluation import lstm_protocols as lp

    s = sequences.reset_index(drop=True)
    _, _, cal, _ = lp.masks(s, {"kind": "clf", "protocol": "seen", "held": None})
    c = s[cal]
    assert len(c) and (c["origin"] == "orig").all()
    assert (c["attack_type"] == "none").all() and (c["y_event"] == 0).all()


def test_lstm_seen_masks_are_disjoint_by_flight(sequences):
    from aerovigil.evaluation import lstm_protocols as lp

    s = sequences.reset_index(drop=True)
    for kind in ("ae", "clf"):
        job = {"kind": kind, "protocol": "seen", "held": None}
        tr, va, cal, test = lp.masks(s, job)
        groups = [set(source(s[m])) for m in (tr, va, cal, test)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert not groups[i] & groups[j], (kind, i, j)
