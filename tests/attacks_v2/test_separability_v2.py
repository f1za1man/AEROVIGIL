"""Module 04 v2 tests: v2 features for the attack set and the per-phase separability."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from test_inject_v2 import CFG, fake_flight

import aerovigil.attacks.attack_separability as sep
import aerovigil.attacks.features_v2_m04 as fm
import aerovigil.attacks.generate_v2 as gv
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.preprocessing.normalize_features_v2 import CLIP, NON_FEATURES

TRUTH_DEFAULTS = (
    ("offset_m", 0.0),
    ("spoof_truth", 0.0),
    ("phase", 0),
    ("seconds_since_onset", np.nan),
)


def normal_features(g):
    """Features of a normal flight: the v2 base carries truth columns at zero."""
    g = g.copy()
    for c, v in TRUTH_DEFAULTS:
        g[c] = v
    return build_flight_v2(g)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A whole fake project: normal flights, fitted parameters, attacked flights."""
    m03, m04, interim = tmp_path / "m03", tmp_path / "m04", tmp_path / "interim"
    (m03 / "base").mkdir(parents=True)
    (m03 / "normalized").mkdir(parents=True)
    interim.mkdir(parents=True)
    rows, splits, feats = [], [], None
    for i, split in enumerate(["train"] * 4 + ["validation", "test"]):
        fid = f"curved__Normal__log_{i:02d}"
        g = fake_flight(fid, duration=70 + 10 * i, hz=50, seed=i)
        g.to_parquet(m03 / "base" / f"{fid}.parquet", index=False)
        rows.append(
            {
                "flight_id": fid,
                "mission": "curved",
                "attacked": 0,
                "file": f"{fid}.parquet",
            }
        )
        splits.append({"flight_id": fid, "split": split})
    pd.DataFrame(rows).to_csv(interim / "base_report.csv", index=False)
    pd.DataFrame(splits).to_csv(interim / "splits.csv", index=False)
    (tmp_path / "attacks.yaml").write_text(yaml.safe_dump(CFG))

    cache, fit = {}, []
    for r, s in zip(rows, splits):
        f = normal_features(pd.read_parquet(m03 / "base" / r["file"]))
        cache[r["file"]] = f
        feats = feats or [c for c in f.columns if c not in NON_FEATURES]
        if s["split"] == "train":
            fit.append(f[feats].to_numpy(np.float64))
    fit = np.concatenate(fit)
    median = np.nanmedian(fit, axis=0)
    q75, q25 = np.nanpercentile(fit, [75, 25], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)
    params = {
        "method": "median / (IQR / 1.349), clipped", "clip": CLIP,
        "scale_floor": 1e-6, "fit_phases": [0, 1], "fit_every_nth_row": 5,
        "fit_flights": 4, "fit_rows": len(fit), "features": feats,
        "median": dict(zip(feats, median.tolist())),
        "scale": dict(zip(feats, scale.tolist())),
    }  # fmt: skip
    (m03 / "normalization_parameters.json").write_text(json.dumps(params))
    for r in rows:
        f = cache[r["file"]].copy()
        z = (f[feats].to_numpy(np.float64) - median) / scale
        f[feats] = np.clip(z, -CLIP, CLIP).astype(np.float32)
        f.to_parquet(m03 / "normalized" / r["file"], index=False)

    monkeypatch.setattr(gv, "BASE_DIR", m03 / "base")
    monkeypatch.setattr(gv, "BASE_REPORT", interim / "base_report.csv")
    monkeypatch.setattr(gv, "SPLITS", interim / "splits.csv")
    monkeypatch.setattr(gv, "CONFIG", tmp_path / "attacks.yaml")
    monkeypatch.setattr(gv, "OUT_DIR", m04)
    monkeypatch.setattr(gv, "MANIFEST", interim / "attack_manifest_v2.csv")
    monkeypatch.setattr(fm, "PARAM_FILE", m03 / "normalization_parameters.json")
    monkeypatch.setattr(fm, "BASE_DIR", m04 / "base")
    monkeypatch.setattr(fm, "NORM_DIR", m04 / "normalized")
    monkeypatch.setattr(fm, "REPORT", interim / "feature_report.csv")
    monkeypatch.setattr(sep, "M03_NORM", m03 / "normalized")
    monkeypatch.setattr(sep, "BASE_REPORT", interim / "base_report.csv")
    monkeypatch.setattr(sep, "M04_NORM", m04 / "normalized")
    monkeypatch.setattr(sep, "MANIFEST", interim / "attack_manifest_v2.csv")
    monkeypatch.setattr(sep, "AUC_FILE", interim / "attack_block_auc.csv")
    man = gv.run(min_free_gb=0.0)
    return {"man": man, "m03": m03, "m04": m04, "interim": interim, "features": feats,
            "median": median, "scale": scale}  # fmt: skip


def test_auc_is_one_when_separated_and_half_when_identical():
    neg = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [np.nan, 5.0]])
    pos = np.array([[9.0, 5.0], [8.0, 5.0]])
    assert sep.auc(pos, neg) == pytest.approx([1.0, 0.5])
    assert np.isnan(sep.auc(np.full((2, 1), np.nan), neg[:, :1]))[0]


def test_blocks_are_five_seconds_and_classed_by_phase(pipeline):
    path = next((pipeline["m03"] / "normalized").glob("*.parquet"))
    stats, classes = sep.blocks(path, pipeline["features"])
    g = pd.read_parquet(path)
    expected = int(np.ceil(g["t_s"].iloc[-1] / sep.SECONDS))  # 5 s bins tile the flight
    assert stats.shape == (expected, len(pipeline["features"]))
    assert set(classes) == {"normal"}  # a normal flight has phase 0 only
    z = np.abs(g[pipeline["features"]].to_numpy(float))
    first = z[(g["t_s"] >= 0) & (g["t_s"] < sep.SECONDS)]
    assert stats[0] == pytest.approx(np.nanmax(first, axis=0), nan_ok=True)


def test_features_apply_the_module03_parameters_without_refitting(pipeline, capsys):
    fm.run(limit=4)
    files = sorted((pipeline["m04"] / "normalized").glob("*.parquet"))
    assert len(files) == 4
    out = pd.read_parquet(files[0])
    assert [c for c in out.columns if c not in NON_FEATURES] == pipeline["features"]
    raw = build_flight_v2(pd.read_parquet(pipeline["m04"] / "base" / files[0].name))
    want = (raw[pipeline["features"]].to_numpy(float) - pipeline["median"]) / pipeline[
        "scale"
    ]
    got = out[pipeline["features"]].to_numpy(float)
    assert np.allclose(got, np.clip(want, -CLIP, CLIP), atol=1e-3, equal_nan=True)
    assert (np.abs(got[np.isfinite(got)]) <= CLIP).all()
    assert "PASS" in capsys.readouterr().out


def test_features_refuse_a_parameter_file_that_does_not_match(pipeline):
    p = pipeline["m03"] / "normalization_parameters.json"
    cfg = json.loads(p.read_text())
    cfg["features"] = cfg["features"][:-1]
    p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="Feature columns differ"):
        fm.run(limit=1)


def test_separability_covers_every_cell_and_uses_normal_flights_as_negatives(pipeline):
    fm.run()
    table = sep.run()
    man = pipeline["man"]
    cells = man.groupby(["attack_set", "attack_type", "velocity_mode"]).ngroups
    assert table["attack_set"].isin(["main", "sweep"]).all()
    assert len(table) == len(
        table.drop_duplicates(
            ["attack_set", "attack_type", "velocity_mode", "window_class", "feature"]
        )
    )
    assert (
        table.groupby(["attack_set", "attack_type", "velocity_mode"]).ngroups == cells
    )
    assert set(table["window_class"]) <= set(sep.CLASSES)
    assert table["auc"].between(0.0, 1.0).all()
    assert table["feature"].nunique() == len(pipeline["features"])
    normal_rows = sum(
        len(pd.read_parquet(f))
        for f in (pipeline["m03"] / "normalized").glob("*.parquet")
    )
    assert (
        table["blocks"].max() <= normal_rows
    )  # positives only, negatives are separate
    assert (pipeline["interim"] / "attack_block_auc.csv").is_file()


def test_separability_needs_the_normalized_attack_flights(pipeline):
    with pytest.raises(SystemExit, match="Run features_v2_m04"):
        sep.run()
