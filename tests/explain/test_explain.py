"""Plan M10 (Step E): checks vocabulary, reason ranking, messages, CUSUM attribution."""

import json

import numpy as np
import pytest

from aerovigil.attacks.features_v2_m04 import PARAM_FILE
from aerovigil.explain import reasons as R
from aerovigil.explain.attribution import cusum_sums, rank, shap_class1
from aerovigil.kalman.detectors import cusum_alarms


def test_every_feature_belongs_to_exactly_one_check():
    feats = json.loads(PARAM_FILE.read_text(encoding="utf-8"))["features"]
    fams = {f: R.family_of(f) for f in feats}
    assert set(fams.values()) == set(R.ROW_CHECKS)
    assert R.family_of("res_course_heading_abs__max") == "course"
    assert R.family_of("res_altitude_roll_std__mean") == "altitude"
    with pytest.raises(KeyError):
        R.family_of("nonsense")


def test_out_of_normal_reasons_come_first():
    scores = {"kf_position": 5.0, "kf_altitude": 3.0, "kf_velocity": 1.0}
    ratio = {k: np.nan for k in R.CHECKS}
    ratio.update({"kf_position": 6.0, "kf_altitude": 0.1, "kf_velocity": 0.5,
                  "horizontal_velocity": 3.0, "course": 0.2})  # fmt: skip
    top = rank(scores, ratio)
    assert [k for k, *_ in top] == ["kf_position", "horizontal_velocity",
                                    "kf_altitude"]  # fmt: skip
    assert [s for _, s, _ in top] == ["model", "supporting", "model"]
    assert top[0][2] == pytest.approx(5.0 / 9.0)


def test_always_three_reasons_even_when_the_model_names_one():
    ratio = {k: 0.5 for k in R.CHECKS}
    top = rank({"horizontal_velocity": 1.0}, ratio)
    assert len(top) == 3 and top[0][0] == "horizontal_velocity"


def test_sentence_says_within_normal_or_how_many_times_normal():
    assert "within normal" in R.sentence("kf_velocity", 0.02, 0.04)
    s = R.sentence("kf_position", 2.7, 0.45)
    assert "2.70 m" in s and "6x normal" in s


def test_message_has_three_numbered_reasons_and_consistent_list():
    m = R.message(
        "kalman",
        12.34,
        [("kf_position", 3.0, 0.45), ("course", 40.0, 22.0), ("altitude", 0.1, 0.25)],
        ["vertical_velocity"],
    )
    assert m.startswith("GPS SPOOFING ALARM at 12.3 s")
    assert "1)" in m and "2)" in m and "3)" in m and "Still consistent: climb" in m


def test_cusum_sums_reach_the_threshold_exactly_at_module06_alarms():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (600, 4))
    z[300:, 1] += 1.5  # a velocity drift
    t = np.arange(600) * 0.1
    valid = np.ones(600, bool)
    k, h = 0.5, 8.0
    idx = cusum_alarms(z, valid, t, k, h, 10.0)
    assert len(idx)
    for i in idx:
        s = cusum_sums(z, valid, t, k, h, 10.0, i)
        assert s.max() > h
        if i > 320:  # after the drift began, the drifting component is the cause
            assert int(np.argmax(s)) == 1
    before = cusum_sums(z, valid, t, k, h, 10.0, idx[0] - 1)
    assert before.max() <= h


def test_shap_layouts():
    a = np.zeros((1, 3, 2))
    a[..., 1] = 7
    assert shap_class1(a).tolist() == [[7, 7, 7]]
    assert shap_class1([np.zeros((1, 3)), np.ones((1, 3))]).sum() == 3
    assert shap_class1(np.ones((1, 3))).shape == (1, 3)
