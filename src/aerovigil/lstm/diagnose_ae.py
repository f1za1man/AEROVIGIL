r"""Step 8b: why does the AE raise more false alarms than its budget? (diagnostic, no training)

Compares the AE scores of NORMAL flights in the train, validation and test splits, per mission
type, and shows how many test flights cross the validation threshold. Also checks, for the
dataset's own attacks, whether attack windows score higher than the normal windows of the SAME flights.

Usage:
    python src\aerovigil\lstm\diagnose_ae.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.lstm.common import load_yaml, out_dir  # noqa: E402
from aerovigil.lstm.datasets import load_windows  # noqa: E402


def run(config, output_dir=None):
    cfg = load_yaml(config)
    out = out_dir(cfg, output_dir)
    _, meta = load_windows(out)
    idx = pd.read_csv(out / "scores" / "index.csv")
    thr = json.loads((out / "thresholds.json").read_text())
    ae = idx[idx["kind"] == "ae"].sort_values("seed")
    pd.set_option("display.width", 200)
    print("=" * 65)
    print("AEROVIGIL - AE DIAGNOSTIC")
    print("=" * 65)
    normal = (
        (meta["origin"] == "orig")
        & (meta["flight_kind"] == "normal")
        & (meta["y"] == 0)
    )
    rows_q, rows_f, rows_m = [], [], []
    for _, r in ae.iterrows():
        s = np.load(out / "scores" / f"{r['name']}.npy")
        t = thr[r["name"]]["threshold"]
        for sp in ("train", "validation", "test"):
            v = s[(normal & (meta["split"] == sp)).to_numpy()]
            rows_q.append(
                {
                    "seed": r["seed"],
                    "split": sp,
                    "windows": len(v),
                    "median": np.median(v),
                    "p90": np.quantile(v, 0.90),
                    "p99": np.quantile(v, 0.99),
                    "max": v.max(),
                    "share_above_thr": float((v > t).mean()),
                }
            )
        d = meta.loc[normal, ["flight_id", "split", "mission_type"]].assign(
            score=s[normal.to_numpy()]
        )
        fmax = (
            d.groupby(["flight_id", "split", "mission_type"])["score"]
            .max()
            .reset_index()
        )
        fmax["seed"], fmax["thr"] = r["seed"], t
        rows_f.append(fmax)
        orig_att = (
            (meta["origin"] == "orig") & (meta["y"] == 1) & (meta["split"] == "test")
        )
        same = meta.loc[orig_att, "flight_id"].unique()
        pre = (meta["flight_id"].isin(same) & (meta["y"] == 0)).to_numpy()
        rows_m.append(
            {
                "seed": r["seed"],
                "attack_windows_median": float(np.median(s[orig_att.to_numpy()])),
                "same_flights_normal_windows_median": float(np.median(s[pre])),
                "other_normal_test_median": float(
                    np.median(s[(normal & (meta["split"] == "test")).to_numpy()])
                ),
            }
        )
    q = pd.DataFrame(rows_q)
    print(
        "\nAE SCORE OF NORMAL WINDOWS BY SPLIT (mean over seeds)  [threshold is from validation flights]"
    )
    print(
        q.groupby("split")[
            ["windows", "median", "p90", "p99", "max", "share_above_thr"]
        ]
        .mean()
        .loc[["train", "validation", "test"]]
        .round(4)
        .to_string()
    )
    f = pd.concat(rows_f)
    f["above"] = f["score"] > f["thr"]
    print("\nPER-FLIGHT MAX SCORE vs THRESHOLD, normal flights (mean over seeds)")
    g = f.groupby("split").agg(
        flights=("flight_id", "nunique"),
        median_flight_max=("score", "median"),
        threshold=("thr", "mean"),
        share_flights_above=("above", "mean"),
    )
    print(g.loc[["train", "validation", "test"]].round(4).to_string())
    print("\nSHARE OF NORMAL TEST FLIGHTS THAT CROSS THE THRESHOLD, BY MISSION TYPE")
    print(
        f[f["split"] == "test"]
        .groupby("mission_type")
        .agg(
            flights=("flight_id", "nunique"),
            share_above=("above", "mean"),
            median_flight_max=("score", "median"),
        )
        .round(3)
        .to_string()
    )
    print(
        "\nSHARE OF NORMAL VALIDATION FLIGHTS THAT CROSS THE THRESHOLD, BY MISSION TYPE"
    )
    print(
        f[f["split"] == "validation"]
        .groupby("mission_type")
        .agg(
            flights=("flight_id", "nunique"),
            share_above=("above", "mean"),
            median_flight_max=("score", "median"),
        )
        .round(3)
        .to_string()
    )
    m = pd.DataFrame(rows_m)
    print(
        "\nORIGINAL ATTACKS (test): AE score of attack windows vs normal windows (mean over seeds)"
    )
    print(m.drop(columns="seed").mean().round(4).to_string())
    print(
        "\nIf attack windows score LOWER than the normal windows of the same flights, the AE cannot"
    )
    print(
        "separate the dataset's own attacks with these features (same finding as Module 05)."
    )
    print("=" * 65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.output_dir)


if __name__ == "__main__":
    main()
