"""Module 07 tests: windows, scaling, models, alarms, protocols and a tiny end-to-end run."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.alarms import (
    alarm_indices,
    calibrate_threshold,
    flight_metrics,  # noqa: E402
    window_metrics,
)
from aerovigil.lstm.datasets import (
    ATTACK_TYPES,
    ae_masks,
    clf_masks,
    eval_attack_groups,  # noqa: E402
    eval_pool,
)
from aerovigil.lstm.models import build_model, fit, predict  # noqa: E402
from aerovigil.lstm.sequences import (
    apply_scaler,
    bin_means,
    fit_scaler,  # noqa: E402
    flight_windows,
)

HZ, K = 125, 6


# ---------------------------------------------------------------- synthetic flights
def make_feats(duration_s=60.0, seed=0, attack=None, shift=3.0):
    """Feature table at 125 Hz: noise, plus +shift on the first 3 features inside the attack."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * HZ)
    t = np.arange(n) / HZ
    x = rng.normal(0, 1, (n, K))
    lab = np.zeros(n)
    if attack:
        lab = ((t >= attack[0]) & (t <= attack[1])).astype(float)
        x[:, :3] += shift * lab[:, None]
    return x, t, lab, np.zeros(n, dtype=bool)


def windows(x, t, lab, gap, **kw):
    return flight_windows(x, t, lab, gap, 5.0, 2.5, 0.25, **kw)


# ---------------------------------------------------------------- windows
def test_bin_means_match_a_manual_calculation_and_ignore_nan():
    t = np.arange(0, 1.0, 0.1)  # 10 rows, 0.1 s apart
    x = np.column_stack([np.arange(10.0), np.ones(10)])
    x[1, 0] = np.nan
    means, rows = bin_means(x, t, 0.5)
    assert rows.tolist() == [5, 5]
    assert means[0, 0] == pytest.approx(np.mean([0, 2, 3, 4]))
    assert means[1, 0] == pytest.approx(np.mean([5, 6, 7, 8, 9]))
    assert np.allclose(means[:, 1], 1.0)


def test_window_count_geometry_and_labels():
    x, t, lab, gap = make_feats(60.0, attack=(20.0, 35.0))
    X, meta, dropped = windows(x, t, lab, gap)
    assert X.shape == (
        22,
        20,
        K,
    )  # last start 54.99 s -> floor(54.99 / 2.5) + 1 windows
    m = pd.DataFrame(meta)
    assert np.allclose(np.diff(m["window_start_s"]), 2.5)
    assert (m.loc[m["window_end_s"] <= 20, "y"] == 0).all()
    assert (
        m.loc[(m["window_start_s"] >= 20) & (m["window_end_s"] <= 35), "y"] == 1
    ).all()
    mid = m[(m["window_start_s"] < 20) & (m["window_end_s"] > 20)]
    assert set(mid["y"]) <= {-1, 1, 0} and (mid["ambiguous"] | (mid["y"] != -1)).all()
    # a window with attack_fraction in (0, 0.5) is ambiguous
    amb = m[(m["attack_fraction"] > 0) & (m["attack_fraction"] < 0.5)]
    assert (amb["y"] == -1).all() and amb["ambiguous"].all()
    assert sum(dropped.values()) == 0


def test_attack_windows_have_higher_values_in_the_shifted_features():
    x, t, lab, gap = make_feats(60.0, attack=(20.0, 40.0))
    X, meta, _ = windows(x, t, lab, gap)
    y = np.array([m["y"] for m in meta])
    assert X[y == 1][:, :, :3].mean() > 2.5 and abs(X[y == 0][:, :, :3].mean()) < 0.2


def test_windows_spanning_a_gap_flag_are_dropped():
    x, t, lab, gap = make_feats(60.0)
    gap[int(30 * HZ)] = True  # first sample after a log gap at t = 30 s
    X, meta, dropped = windows(x, t, lab, gap)
    assert dropped["spans_gap"] >= 1
    for m in meta:
        assert not (m["window_start_s"] < 30.0 < m["window_end_s"])


def test_empty_bins_and_missing_values_drop_windows():
    x, t, lab, gap = make_feats(60.0)
    keep = ~((t >= 20.0) & (t < 21.0))  # remove 1 s of rows without a gap flag
    X, meta, dropped = windows(x[keep], t[keep], lab[keep], gap[keep])
    assert dropped["empty_bin"] >= 1
    x2 = x.copy()
    x2[:, 0] = np.nan  # one of six features always missing = 16.7%
    _, meta2, dropped2 = windows(x2, t, lab, gap, max_missing=0.10)
    assert len(meta2) == 0 and dropped2["too_many_nan"] > 0
    _, meta3, _ = windows(x2, t, lab, gap, max_missing=0.20)
    assert len(meta3) == 22


def test_bin_size_must_divide_the_hop():
    x, t, lab, gap = make_feats(20.0)
    with pytest.raises(ValueError):
        flight_windows(x, t, lab, gap, 5.0, 2.5, 0.3)


def test_short_flight_gives_no_windows():
    x, t, lab, gap = make_feats(4.0)
    X, meta, _ = windows(x, t, lab, gap)
    assert len(meta) == 0 and X.shape == (0, 20, K)


# ---------------------------------------------------------------- scaling
def test_scaler_is_robust_clips_and_fills_missing_values():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, (200, 20, 3)).astype(np.float32)
    X[0, 0, 0] = 1e6  # one wild outlier must not wreck the scale
    sc = fit_scaler(X, clip=8.0)
    assert sc["center"][0] == pytest.approx(5.0, abs=0.2) and sc["scale"][
        0
    ] == pytest.approx(2.0, rel=0.15)
    Y = X.copy()
    Y[1, 1, 1] = np.nan
    Z = apply_scaler(Y, sc)
    assert not np.isnan(Z).any() and Z.max() <= 8.0 + 1e-6 and Z[1, 1, 1] == 0.0


# ---------------------------------------------------------------- models
def test_model_shapes_and_scores():
    X = np.random.default_rng(0).normal(size=(30, 20, K)).astype(np.float32)
    ae = build_model("ae", K, {"hidden": 8, "layers": 2, "latent": 4, "dropout": 0.1})
    assert ae(torch.from_numpy(X)).shape == X.shape
    s, fe = predict("ae", ae, X)
    assert (
        s.shape == (30,)
        and fe.shape == (30, K)
        and np.allclose(s, fe.mean(axis=1), atol=1e-6)
    )
    clf = build_model("clf", K, {"hidden": 8, "layers": 1, "dropout": 0.1})
    s2, none = predict("clf", clf, X)
    assert s2.shape == (30,) and none is None


def _normal_and_attack(n=300, shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n, 20, K)).astype(np.float32)
    attack = rng.normal(0, 1, (n, 20, K)).astype(np.float32)
    attack[:, :, 2] += shift
    return normal, attack


def test_autoencoder_trained_on_normal_flags_a_shifted_feature():
    normal, attack = _normal_and_attack()
    p = {"hidden": 16, "layers": 1, "latent": 8, "dropout": 0.0, "lr": 0.003}
    model = build_model("ae", K, p, seed=0)
    model, h = fit("ae", model, normal[:200], None, normal[200:], None, p, 0, 25, 6)
    assert h["val"][h["best_epoch"]] <= h["val"][0]
    s_n, _ = predict("ae", model, normal[200:])
    s_a, fe = predict("ae", model, attack)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(np.r_[np.zeros(len(s_n)), np.ones(len(s_a))], np.r_[s_n, s_a])
    assert auc > 0.95
    assert (
        int(np.argmax(fe.mean(axis=0))) == 2
    )  # the per-feature error points at the shifted feature


def test_classifier_learns_the_attack_and_training_is_deterministic():
    normal, attack = _normal_and_attack(seed=1)
    X = np.concatenate([normal, attack])
    y = np.r_[np.zeros(len(normal)), np.ones(len(attack))].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    p = {"hidden": 16, "layers": 1, "dropout": 0.0, "lr": 0.01}

    def run():
        m = build_model("clf", K, p, seed=3)
        m, _ = fit("clf", m, X[:400], y[:400], X[400:], y[400:], p, 3, 40, 10)
        return predict("clf", m, X[400:])[0]

    a, b = run(), run()
    assert np.allclose(a, b, atol=1e-6)
    assert ((a > 0) == (y[400:] == 1)).mean() > 0.9


# ---------------------------------------------------------------- alarms
def test_alarm_indices_respect_the_refractory_period():
    score = np.array([0, 5, 5, 5, 0, 5, 5, 5], dtype=float)
    t_end = np.arange(8) * 2.5
    assert alarm_indices(score, t_end, 1.0, 10.0).tolist() == [1, 5]
    assert alarm_indices(score, t_end, 1.0, 0.0).tolist() == [1, 2, 3, 5, 6, 7]


def test_calibrated_threshold_stays_inside_the_budget():
    rng = np.random.default_rng(0)
    flights = [(rng.gamma(2.0, 1.0, 200), np.arange(200) * 2.5) for _ in range(10)]
    thr, hours, allowed, used = calibrate_threshold(flights, 2.0, 2.5, 10.0)
    assert hours == pytest.approx(10 * 200 * 2.5 / 3600)
    assert used <= allowed
    assert sum(len(alarm_indices(s, t, thr, 10.0)) for s, t in flights) == used
    # a stricter budget can only raise the threshold
    thr2, *_ = calibrate_threshold(flights, 0.5, 2.5, 10.0)
    assert thr2 >= thr


def test_flight_metrics_detected_missed_and_false_alarms():
    t_end = np.arange(1, 41) * 2.5 + 2.5  # 5 .. 102.5 s
    offset = np.where(t_end >= 50, (t_end - 50) * 2.0, 0.0)
    score = np.zeros(40)
    score[3] = 9.0  # false alarm at t = 12.5 s (before the attack)
    score[24] = 9.0  # detection at t = 65 s
    m = flight_metrics(score, t_end, offset, 1.0, 10.0, 50.0, 100.0, 2.5)
    assert m["detected"] and m["false_alarms"] == 1
    assert m["delay_s"] == pytest.approx(t_end[24] - 50.0)
    assert m["displacement_m"] == pytest.approx(offset[24])
    missed = flight_metrics(np.zeros(40), t_end, offset, 1.0, 10.0, 50.0, 100.0, 2.5)
    assert not missed["detected"] and np.isnan(missed["delay_s"])
    assert missed["displacement_m"] == pytest.approx(
        np.nanmax(offset[(t_end >= 50) & (t_end <= 100)])
    )
    normal = flight_metrics(
        score, t_end, np.full(40, np.nan), 1.0, 10.0, np.inf, np.inf, 2.5
    )
    assert not normal["attacked"] and normal["false_alarms"] == 2


def test_window_metrics_for_a_perfect_and_a_useless_score():
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(0, 1, 500), rng.normal(6, 1, 100)]
    pos = np.r_[np.zeros(500, bool), np.ones(100, bool)]
    m = window_metrics(s, pos, ~pos, float(np.quantile(s[~pos], 0.99)))
    assert m["roc_auc"] > 0.99 and m["tpr_at_1pct_fpr"] > 0.9 and m["f1"] > 0.9
    r = window_metrics(rng.normal(0, 1, 600), pos, ~pos, 2.3)
    assert 0.35 < r["roc_auc"] < 0.65


# ---------------------------------------------------------------- protocols
def make_meta():
    rows = []
    for split in ("train", "validation", "test"):
        for kind, n_flights in (("normal", 2), ("attacked", 1)):
            for i in range(n_flights):
                for y in [0] * 4 + ([1] * 3 if kind == "attacked" else []):
                    rows.append(
                        {
                            "flight_id": f"o_{split}_{kind}_{i}",
                            "source_flight_id": f"o_{split}_{kind}_{i}",
                            "origin": "orig",
                            "split": split,
                            "flight_kind": kind,
                            "y": y,
                            "attack_type": "original" if kind == "attacked" else "none",
                            "velocity_mode": "na",
                            "attack_fraction": float(y),
                        }
                    )
        for typ in ATTACK_TYPES:
            for mode in ("consistent", "position_only"):
                for y in (0, 1, 1):
                    rows.append(
                        {
                            "flight_id": f"s_{split}_{typ}_{mode}",
                            "source_flight_id": f"o_{split}_normal_0",
                            "origin": "syn",
                            "split": split,
                            "flight_kind": "attacked",
                            "y": y,
                            "attack_type": typ,
                            "velocity_mode": mode,
                            "attack_fraction": float(y),
                        }
                    )
    return pd.DataFrame(rows)


def test_autoencoder_sees_only_normal_windows_of_normal_train_flights():
    meta = make_meta()
    tr, va = ae_masks(meta)
    for mask, split in ((tr, "train"), (va, "validation")):
        sub = meta[mask]
        assert len(sub) > 0 and (sub["y"] == 0).all() and (sub["split"] == split).all()
        assert (sub["flight_kind"] == "normal").all() and (
            sub["origin"] == "orig"
        ).all()


@pytest.mark.parametrize("held", ATTACK_TYPES + ["original"])
def test_held_out_family_never_reaches_supervised_training_or_validation(held):
    meta = make_meta()
    tr, va = clf_masks(meta, held)
    used = meta[tr | va]
    assert (used["split"] != "test").all()
    if held == "original":
        assert not ((used["origin"] == "orig") & (used["y"] == 1)).any()
        assert ((used["origin"] == "syn") & (used["y"] == 1)).any()
    else:
        assert (
            not (used["attack_type"] == held).any()
            or not ((used["origin"] == "syn") & (used["attack_type"] == held)).any()
        )
        assert ((used["origin"] == "orig") & (used["y"] == 1)).any()
    seen_tr, _ = clf_masks(meta, None)
    assert seen_tr.sum() > tr.sum()


def test_negative_pool_and_attack_groups_use_original_test_flights_only_for_negatives():
    meta = make_meta()
    pool = meta[eval_pool(meta)]
    assert (
        (pool["origin"] == "orig").all()
        and (pool["split"] == "test").all()
        and (pool["y"] == 0).all()
    )
    groups = eval_attack_groups(meta)
    assert (
        "original" in groups
        and "synthetic_all" in groups
        and "linear_drift|position_only" in groups
    )
    for g, mask in groups.items():
        sub = meta[mask]
        assert (sub["split"] == "test").all() and (sub["y"] == 1).all()


# ---------------------------------------------------------------- tiny end-to-end run
def write_dataset(tmp):
    """Write windows.npz / windows_meta / flights / scaler in the Module 07 layout from synthetic flights."""
    out = tmp / "out"
    out.mkdir()
    Xs, metas, flights = [], [], []
    plan = (
        [("train", i) for i in range(6)]
        + [("validation", i) for i in range(6, 9)]
        + [("test", i) for i in range(9, 12)]
    )
    seed = 0
    for sp, i in plan:
        for kind, att, origin, typ, mode in [
            ("normal", None, "orig", "none", "na"),
            ("attacked", (20.0, 40.0), "orig", "original", "na"),
            ("attacked", (20.0, 40.0), "syn", "linear_drift", "position_only"),
            ("attacked", (20.0, 40.0), "syn", "abrupt_jump", "consistent"),
        ]:
            if origin == "orig" and kind == "attacked" and i % 2:
                continue
            seed += 1
            x, t, lab, gap = make_feats(60.0, seed=seed, attack=att)
            X, meta, _ = windows(x, t, lab, gap)
            m = pd.DataFrame(meta).drop(columns="last_row")
            fid = f"{origin}_{kind}_{typ}_{i}"
            m["offset_end_m"] = (
                np.where(m["window_end_s"] > 20, (m["window_end_s"] - 20) * 1.0, 0.0)
                if origin == "syn"
                else np.nan
            )
            m["rel_end_s"] = m["window_end_s"] - 20.0 if origin == "syn" else np.nan
            for k, v in {
                "flight_id": fid,
                "source_flight_id": f"orig_normal_none_{i}",
                "origin": origin,
                "split": sp,
                "mission_type": "curved",
                "flight_kind": kind,
                "attack_type": typ,
                "velocity_mode": mode,
                "drift_rate_mps": np.nan,
            }.items():
                m[k] = v
            Xs.append(X)
            metas.append(m)
            flights.append(
                {
                    "flight_id": fid,
                    "origin": origin,
                    "split": sp,
                    "flight_kind": kind,
                    "mission_type": "curved",
                    "attack_type": typ,
                    "velocity_mode": mode,
                    "drift_rate_mps": np.nan,
                    "attack_start_s": att[0] if att else np.inf,
                    "attack_end_s": att[1] if att else np.inf,
                    "windows": len(m),
                }
            )
    X, meta = np.concatenate(Xs), pd.concat(metas, ignore_index=True)
    tr = (
        (meta["origin"] == "orig")
        & (meta["split"] == "train")
        & (meta["flight_kind"] == "normal")
        & (meta["y"] == 0)
    ).to_numpy()
    sc = fit_scaler(X[tr], 8.0)
    sc["features"] = [f"f{i}" for i in range(K)]
    np.savez_compressed(out / "windows.npz", X=X)
    meta.to_parquet(out / "windows_meta.parquet", index=False)
    pd.DataFrame(flights).to_parquet(out / "flights.parquet", index=False)
    (out / "scaler.json").write_text(json.dumps(sc))
    yaml.safe_dump(
        {"processing": {"window_duration_seconds": 5.0, "window_overlap": 0.5}},
        open(tmp / "pre.yaml", "w"),
    )
    cfg = {
        "paths": {
            "preprocessing_config": str(tmp / "pre.yaml"),
            "output_dir": str(out),
            "kalman_dir": str(tmp / "none"),
        },
        "training": {
            "seed": 1,
            "n_final_seeds": 2,
            "n_unseen_seeds": 1,
            "max_epochs": 15,
            "patience": 5,
            "batch_size": 64,
            "weight_decay": 1e-5,
            "grad_clip": 1.0,
            "threads": 2,
        },
        "ae_params": {
            "hidden": 16,
            "layers": 1,
            "latent": 8,
            "dropout": 0.0,
            "lr": 0.003,
        },
        "clf_params": {"hidden": 16, "layers": 1, "dropout": 0.0, "lr": 0.003},
        "tuning": {"n_trials": 2, "max_epochs": 5, "patience": 3, "use_mlflow": False},
        "detector": {
            "false_alarm_budget_per_hour": 1.0,
            "refractory_s": 10.0,
            "operating_point_fpr": 0.01,
        },
    }
    path = tmp / "lstm.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return path, out


def test_pipeline_trains_scores_and_evaluates_end_to_end(tmp_path):
    from aerovigil.lstm import evaluate_lstm, train_models, tune_lstm

    path, out = write_dataset(tmp_path)
    best = tune_lstm.run(str(path), trials=2)
    assert set(best) == {"ae", "clf"} and best["ae"]["val_auc"] > 0.8
    idx = train_models.run(str(path))
    assert (idx["kind"] == "ae").sum() == 2 and (idx["protocol"] == "unseen").sum() == 6
    assert len(list((out / "scores").glob("*.npy"))) >= len(idx)
    agg, summary, fa, frontier = evaluate_lstm.run(str(path))
    ae = agg[(agg["detector"] == "lstm_ae") & (agg["group"] == "synthetic_all")]
    assert float(ae["roc_auc_mean"].iloc[0]) > 0.9
    held = agg[agg["detector"] == "lstm_supervised_unseen"]
    assert set(held["group"]) >= {"linear_drift", "abrupt_jump", "original"}
    for f in (
        "results_window.csv",
        "alarms.csv",
        "summary_by_attack.csv",
        "false_alarms.csv",
        "thresholds.json",
    ):
        assert (out / f).exists()
    ae_rows = summary[(summary["detector"] == "lstm_ae") & (summary["set"] == "syn")]
    assert (ae_rows["detection_rate"] > 0.5).all()
    # the thresholds come from validation flights only: calibration hours = validation normal windows
    thr = json.loads((out / "thresholds.json").read_text())
    assert all(v["alarms_used"] <= v["allowed_alarms"] + 1e-9 for v in thr.values())
