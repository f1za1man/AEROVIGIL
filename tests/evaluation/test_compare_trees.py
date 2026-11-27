"""Step J: the clean-rebuild comparison finds real changes and ignores round-off."""

import json

import pandas as pd

from aerovigil.evaluation import compare_trees as ct


def make(root, table, cal):
    (root / "reports").mkdir(parents=True)
    (root / "data" / "processed" / "module06").mkdir(parents=True)
    table.to_csv(root / "reports" / "t.csv", index=False)
    (root / "data" / "processed" / "module06" / "filter_noise.json").write_text(
        json.dumps(cal)
    )


def test_round_off_real_change_calibration_and_missing(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    t = pd.DataFrame({"name": ["x", "y"], "v": [1.0, 2.0]})
    make(a, t, {"pos": 0.04, "vel": 0.0001})
    make(b, t.assign(v=[1.0 + 1e-13, 2.0]), {"pos": 0.05, "vel": 0.0001})
    (a / "reports" / "only_here.csv").write_text("a\n1\n")
    files = ["reports/t.csv", "data/processed/module06/filter_noise.json",
             "reports/only_here.csv"]  # fmt: skip
    monkeypatch.setattr(ct, "ROOT", a)
    monkeypatch.setattr(ct, "tracked", lambda: files)
    r = ct.run(b).set_index("file")
    assert r.loc["reports/t.csv", "status"] == "round-off"
    cal = r.loc["data/processed/module06/filter_noise.json"]
    assert cal["calibration"] and cal["status"] == "differs" and "pos" in cal["note"]
    assert r.loc["reports/only_here.csv", "status"] == "missing in rebuild"

    make_b2 = tmp_path / "b2"
    make(make_b2, t.assign(v=[1.5, 2.0]), {"pos": 0.04, "vel": 0.0001})
    r2 = ct.run(make_b2).set_index("file")
    assert r2.loc["reports/t.csv", "status"] == "differs"
    assert r2.loc["data/processed/module06/filter_noise.json", "status"] == "same"
