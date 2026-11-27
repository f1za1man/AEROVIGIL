"""Plan M13 (Step H): every control of the demo dashboard works on the real demo data.

Runs the real app script headless (Streamlit's AppTest) against the precomputed demo
data, so it is skipped where that data has not been built (it needs the dataset)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app" / "app.py"
DATA = ROOT / "data" / "processed" / "demo" / "scenarios.parquet"

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="demo data not built")


def run(at):
    at.run(timeout=300)
    assert not at.exception, at.exception
    return at


def new():
    from streamlit.testing.v1 import AppTest

    return run(AppTest.from_file(str(APP), default_timeout=300))


def by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_opens_with_charts_table_and_explanation():
    at = new()
    assert at.title[0].value.startswith("AEROVIGIL")
    assert len(at.dataframe) >= 1
    assert len(at.metric) == 5


def test_every_flight_and_attack_type_renders():
    at = new()
    n_flights = len(by_label(at.sidebar.selectbox, "Flight").options)
    for f in range(n_flights):
        by_label(at.sidebar.selectbox, "Flight").select_index(f)
        run(at)
        n_types = len(by_label(at.sidebar.selectbox, "Attack").options)
        for k in range(n_types):
            by_label(at.sidebar.selectbox, "Attack").select_index(k)
            run(at)


def test_out_drift_slider_and_modes_change_the_result():
    import pandas as pd

    scen = pd.read_parquet(DATA)
    rates = sorted(scen.loc[scen["attack_type"] == "linear_drift", "strength"].unique())
    at = new()
    attack = by_label(at.sidebar.selectbox, "Attack")
    attack.select_index(attack.options.index("Linear drift"))
    run(at)
    seen = set()
    for mode in ("position_only", "consistent"):
        at.sidebar.radio[0].set_value(mode)
        run(at)
        for value in rates:
            at.sidebar.select_slider[0].set_value(value)
            run(at)
            seen.add(at.metric[4].value)
    assert len(seen) > 1  # how many detectors catch it depends on the drift


def test_explanation_picker_and_detector_filter():
    at = new()
    pick = [s for s in at.selectbox if s.label == "Alarm of"]
    if pick:
        for k in range(len(pick[0].options)):
            [s for s in at.selectbox if s.label == "Alarm of"][0].select_index(k)
            run(at)
            assert len(at.error) + len(at.warning) >= 1  # the operator message
    at.sidebar.multiselect[0].set_value(["kalman", "random_forest"])
    run(at)
    assert len(at.dataframe[0].value) == 2
