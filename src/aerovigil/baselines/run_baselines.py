"""Train, tune and evaluate the classical baselines (Module 05).

Usage:
    python src\\aerovigil\\baselines\\run_baselines.py
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import load_yaml, resolve  # noqa: E402
from aerovigil.baselines.datasets import (
    assemble,
    clean_values,  # noqa: E402
    build_test_sets,
    protocols,
    stat_columns,
)
from aerovigil.baselines.models import (
    STOCHASTIC,
    UNSUPERVISED,
    build_model,  # noqa: E402
    evaluate,
    fit_model,
    grid,
    score_model,
)

MAIN_SETS = ["orig_test", "syn_all", "syn:consistent", "syn:position_only"]
METRICS = ["precision", "recall", "f1", "roc_auc", "tpr_at_fpr"]


def aggregate(raw):
    """Mean and std over seeds, plus an 'unseen_mission_mean' protocol (mean over folds)."""
    keys = ["protocol", "model", "test_set"]
    agg = (
        raw.groupby(keys)
        .agg(
            n_seeds=("seed", "nunique"),
            n_pos=("n_pos", "first"),
            n_neg=("n_neg", "first"),
            **{f"{m}_mean": (m, "mean") for m in METRICS},
            **{f"{m}_std": (m, "std") for m in METRICS},
        )
        .reset_index()
    )
    agg[[f"{m}_std" for m in METRICS]] = agg[[f"{m}_std" for m in METRICS]].fillna(0.0)
    folds = agg[agg["protocol"].str.startswith("unseen_")]
    if len(folds):
        g = folds.groupby(["model", "test_set"])
        avg = g[[f"{m}_mean" for m in METRICS]].mean()
        std = (
            g[[f"{m}_mean" for m in METRICS]]
            .std(ddof=0)
            .rename(columns=lambda c: c.replace("_mean", "_std"))
        )
        extra = pd.concat(
            [avg, std, g[["n_pos", "n_neg"]].sum(), g["n_seeds"].max()], axis=1
        ).reset_index()
        extra["protocol"] = "unseen_mission_mean"
        agg = pd.concat([agg, extra], ignore_index=True)
    return agg


def run(config_path, output_dir=None):
    cfg = load_yaml(config_path)
    sources = list(cfg["training_sources"])
    seed = int(cfg["random_seed"])
    out_base = Path(output_dir) if output_dir else resolve(cfg["paths"]["output_dir"])
    out_base = out_base if out_base.is_absolute() else ROOT / out_base
    res_dir = out_base / f"results_{'+'.join(sources)}"
    res_dir.mkdir(parents=True, exist_ok=True)

    orig = clean_values(pd.read_parquet(out_base / "windows_orig.parquet"))
    syn = clean_values(pd.read_parquet(out_base / "windows_syn.parquet"))
    all_cols = stat_columns(orig)

    print("=" * 65)
    print("AEROVIGIL - CLASSICAL ML BASELINES")
    print("=" * 65)
    print(
        f"Training sources: {sources} | seed {seed} | final seeds {cfg['n_final_seeds']}"
    )
    print(
        f"Windows: original {len(orig):,}, synthetic {len(syn):,} | summary features {len(all_cols)}"
    )

    raw, tuning, saved_scores, importance = [], [], {}, None
    started = time.time()

    for protocol, held in protocols(cfg, orig):
        d = assemble(
            orig,
            syn,
            held,
            sources,
            balance_seed=seed if cfg.get("balance_train_val", True) else None,
        )
        cols = [c for c in all_cols if d["train"][c].notna().any()]
        xtr, ytr = d["train"][cols].to_numpy(float), d["train"]["y"].to_numpy(int)
        xva, yva = d["val"][cols].to_numpy(float), d["val"]["y"].to_numpy(int)
        sets = build_test_sets(d["test_orig"], d["test_syn"])
        print(
            f"\n--- protocol: {protocol} ---  train {len(xtr):,} (attack {int(ytr.sum()):,}) | "
            f"val {len(xva):,} (attack {int(yva.sum()):,}) | test sets {len(sets)}"
        )
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            raise SystemExit(
                f"Protocol {protocol}: training or validation has a single class."
            )
        overlap = (
            set(d["train"]["flight_id"])
            & set(d["val"]["flight_id"])
            & set(d["test_orig"]["flight_id"])
        )
        assert not overlap, "flight appears in train, val and test"

        for model, params in cfg["models"].items():
            # ---- tune on validation flights ----
            best, best_score = None, -np.inf
            for p in grid(params):
                pipe = fit_model(model, build_model(model, p, seed), xtr, ytr)
                s = score_model(model, pipe, xva)
                if model in UNSUPERVISED:
                    crit = evaluate(yva, s, np.inf)["roc_auc"]
                else:
                    crit = evaluate(yva, s, 0.5)["f1"]
                tuning.append(
                    {"protocol": protocol, "model": model, "criterion": crit, **p}
                )
                if crit > best_score:
                    best, best_score = p, crit

            # ---- final runs ----
            seeds = (
                [seed + k for k in range(int(cfg["n_final_seeds"]))]
                if model in STOCHASTIC
                else [seed]
            )
            for sd in seeds:
                pipe = fit_model(model, build_model(model, best, sd), xtr, ytr)
                if model in UNSUPERVISED:
                    s_val_norm = score_model(model, pipe, xva[yva == 0])
                    thr = float(
                        np.quantile(
                            s_val_norm, cfg["isolation_forest_threshold_quantile"]
                        )
                    )
                else:
                    thr = 0.5
                for set_name, df in sets.items():
                    y = df["y"].to_numpy(int)
                    s = score_model(model, pipe, df[cols].to_numpy(float))
                    m = evaluate(y, s, thr, cfg["operating_point_fpr"])
                    if m is None:
                        continue
                    raw.append(
                        {
                            "protocol": protocol,
                            "model": model,
                            "seed": sd,
                            "test_set": set_name,
                            "threshold": thr,
                            "best_params": str(best),
                            **m,
                        }
                    )
                    if (
                        protocol == "seen"
                        and sd == seed
                        and set_name in ("orig_test", "syn_all")
                    ):
                        saved_scores[f"{model}|{set_name}|y"] = y
                        saved_scores[f"{model}|{set_name}|s"] = s
                if protocol == "seen" and model == "random_forest" and sd == seed:
                    importance = pd.DataFrame(
                        {"feature": cols, "importance": pipe[-1].feature_importances_}
                    ).sort_values("importance", ascending=False)
            r = [
                x
                for x in raw
                if x["protocol"] == protocol
                and x["model"] == model
                and x["test_set"] == "orig_test"
            ]
            print(
                f"  {model:20s} best={best}  "
                f"orig_test F1={np.mean([x['f1'] for x in r]):.3f}  "
                f"({time.time() - started:.0f} s)"
            )

    raw_df = pd.DataFrame(raw)
    agg = aggregate(raw_df)
    raw_df.to_csv(res_dir / "results_raw.csv", index=False)
    pd.DataFrame(tuning).to_csv(res_dir / "tuning_log.csv", index=False)
    agg[agg["test_set"].isin(MAIN_SETS)].to_csv(
        res_dir / "results_main.csv", index=False
    )
    agg[~agg["test_set"].isin(MAIN_SETS)].to_csv(
        res_dir / "results_by_attack.csv", index=False
    )
    np.savez_compressed(res_dir / "scores_seen.npz", **saved_scores)
    if importance is not None:
        importance.to_csv(res_dir / "rf_importance_seen.csv", index=False)
    shutil.copy(config_path, res_dir / "baselines_config_used.yaml")

    # ---------------- printed results table ----------------
    print("\n" + "-" * 65)
    order = list(cfg["models"])
    shown = [p for p in ("seen", "unseen_mission_mean") if p in set(agg["protocol"])]
    for metric in ("f1", "roc_auc", "recall"):
        print(f"\n{metric.upper()} (mean over seeds / folds)")
        t = (
            agg[agg["test_set"].isin(MAIN_SETS) & agg["protocol"].isin(shown)]
            .pivot_table(
                index="model", columns=["test_set", "protocol"], values=f"{metric}_mean"
            )
            .reindex(order)
            .round(3)
        )
        print(t.to_string())
    print(f"\nSaved to: {res_dir}")
    print("=" * 65)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "baselines" / "baselines.yaml")
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.output_dir)


if __name__ == "__main__":
    main()
