"""Plan M14 (Step I): large results are read from their committed .gz copy."""

import pandas as pd

from aerovigil.evaluation import store


def test_gz_copy_reads_back_the_same_and_is_not_rewritten(tmp_path):
    p = tmp_path / "flights.csv"
    frame = pd.DataFrame({"flight_id": ["a", "b"], "detected": [True, False],
                          "delay_s": [0.25, float("nan")]})  # fmt: skip
    frame.to_csv(p, index=False)
    assert store.publish([p]) == [store.gz_of(p)]
    first = store.gz_of(p).read_bytes()
    assert store.publish([p]) == []  # unchanged result: the copy is left alone
    assert store.gz_of(p).read_bytes() == first
    p.unlink()  # a fresh clone holds only the .gz copy
    pd.testing.assert_frame_equal(store.read_csv(p), frame)


def test_missing_result_reads_as_none(tmp_path):
    assert store.read_csv(tmp_path / "nothing.csv") is None
    assert store.path_of(tmp_path / "nothing.csv") is None
