r"""AEROVIGIL Module 03 v2: how separable is each window class from normal? (read only)

For every window the largest absolute normalized value of each feature is taken over
the 5 s, then each class (ramp, snapback, plateau, post) is compared with the clean
windows (normal flights and pre-attack) by ROC AUC. 0.5 means no separation.

Writes data/processed/module03_v2/windows/window_feature_auc.csv

Usage:
    python -m aerovigil.windowing.window_separability
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
WIN_DIR = ROOT / "data" / "processed" / "module03_v2" / "windows"
SPLITS = ("train", "validation", "test")
TARGETS = ["ramp", "snapback", "plateau", "post"]
CLEAN = ["normal", "pre"]
TOP = 5


def window_scores():
    parts, names, classes, split_col = [], None, [], []
    for s in SPLITS:
        data = np.load(WIN_DIR / f"windows_{s}.npz")
        names = [str(n) for n in data["feature_names"]]
        labels = np.array([str(n) for n in data["class_names"]])
        classes.append(labels[data["window_class"]])
        parts.append(np.abs(data["X"]).max(axis=1))
        split_col.append(np.full(len(parts[-1]), s))
    return (
        pd.DataFrame(np.concatenate(parts), columns=names),
        np.concatenate(classes),
        np.concatenate(split_col),
    )


def auc_table(scores, classes):
    clean = np.isin(classes, CLEAN)
    rows = []
    for target in TARGETS:
        hit = classes == target
        if not hit.any() or not clean.any():
            continue
        keep = hit | clean
        y = hit[keep].astype(int)
        for name in scores.columns:
            rows.append(
                {
                    "class": target,
                    "feature": name,
                    "auc": roc_auc_score(y, scores.loc[keep, name]),
                    "windows": int(hit.sum()),
                }
            )
    return pd.DataFrame(rows)


def main():
    scores, classes, split_col = window_scores()
    print("=" * 65)
    print("AEROVIGIL - WINDOW CLASS SEPARABILITY (max |z| per window)")
    print("=" * 65)
    print(
        f"windows: {len(scores):,}   clean (normal + pre): {int(np.isin(classes, CLEAN).sum()):,}"
    )
    table = auc_table(scores, classes)
    table.to_csv(WIN_DIR / "window_feature_auc.csv", index=False)
    pd.set_option("display.width", 200)
    for target in TARGETS:
        sub = table[table["class"] == target].sort_values("auc", ascending=False)
        if sub.empty:
            continue
        print(
            f"\n{target} windows ({int(sub['windows'].iloc[0]):,}) vs clean: best {TOP} features"
        )
        print(sub.head(TOP)[["feature", "auc"]].round(3).to_string(index=False))
    print("\nBEST SINGLE FEATURE PER CLASS, BY SPLIT (AUC vs clean in the same split)")
    for s in SPLITS:
        mask = split_col == s
        part = auc_table(scores[mask].reset_index(drop=True), classes[mask])
        best = part.sort_values("auc", ascending=False).groupby("class").head(1)
        print(f"\n{s}")
        print(best[["class", "feature", "auc"]].round(3).to_string(index=False))
    print("\nSaved:", WIN_DIR / "window_feature_auc.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
