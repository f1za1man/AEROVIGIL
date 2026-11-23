import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "preprocessing" / "preprocessing.yaml"
REPORT_FILE = ROOT / "data" / "processed" / "module03" / "cleaning_report.csv"
OUT_DIR = ROOT / "data" / "interim" / "module03"


def main():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["processing"]["random_seed"]
    sp = cfg["splitting"]
    assert (
        abs(
            sp["train_fraction"] + sp["validation_fraction"] + sp["test_fraction"] - 1.0
        )
        < 1e-9
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = pd.read_csv(REPORT_FILE)
    rep["has_attack"] = rep["attack_rows"] > 0
    rep["stratum"] = (
        rep["mission_type"] + "|" + np.where(rep["has_attack"], "attacked", "normal")
    )

    rng = np.random.default_rng(seed)
    assignment = {}
    for stratum, g in rep.groupby("stratum", sort=True):
        ids = sorted(g["flight_id"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * sp["test_fraction"]))
        n_val = int(round(n * sp["validation_fraction"]))
        if n >= 3:  # keep every partition represented
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for i, fid in enumerate(ids):
            if i < n_test:
                assignment[fid] = "test"
            elif i < n_test + n_val:
                assignment[fid] = "validation"
            else:
                assignment[fid] = "train"

    rep["split"] = rep["flight_id"].map(assignment)
    rep[["flight_id", "split"]].sort_values(["split", "flight_id"]).to_csv(
        OUT_DIR / "flight_splits.csv", index=False
    )

    print("=" * 65)
    print("AEROVIGIL - MISSION-DISJOINT SPLIT")
    print("=" * 65)
    print(f"Seed: {seed}   Flights: {len(rep)}\n")

    order = ["train", "validation", "test"]
    summary = {
        "seed": seed,
        "flights": {},
        "rows": {},
        "attack_rows": {},
        "by_mission_type": {},
        "by_flight_class": {},
    }
    for s in order:
        part = rep[rep["split"] == s]
        summary["flights"][s] = int(len(part))
        summary["rows"][s] = int(part["rows_out"].sum())
        summary["attack_rows"][s] = int(part["attack_rows"].sum())
    mt = pd.crosstab(rep["mission_type"], rep["split"])[order]
    fc = pd.crosstab(np.where(rep["has_attack"], "attacked", "normal"), rep["split"])[
        order
    ]
    summary["by_mission_type"] = mt.to_dict()
    summary["by_flight_class"] = fc.to_dict()
    with (OUT_DIR / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tot = pd.DataFrame(
        {
            "flights": summary["flights"],
            "rows": summary["rows"],
            "attack_rows": summary["attack_rows"],
        }
    ).loc[order]
    tot["flight_%"] = (tot["flights"] / tot["flights"].sum() * 100).round(1)
    tot["attack_row_%"] = (tot["attack_rows"] / tot["rows"] * 100).round(1)
    print(tot.to_string())
    print("\nMission type x split (flights)")
    print(mt.to_string())
    print("\nFlight class x split (flights)")
    print(fc.to_string())

    sets = {s: set(rep.loc[rep["split"] == s, "flight_id"]) for s in order}
    checks = {
        "Every flight assigned": rep["split"].notna().all(),
        "Train/validation disjoint": not (sets["train"] & sets["validation"]),
        "Train/test disjoint": not (sets["train"] & sets["test"]),
        "Validation/test disjoint": not (sets["validation"] & sets["test"]),
        "All flights covered": sum(len(v) for v in sets.values()) == 360,
        "Every mission type in every split": bool((mt > 0).all().all()),
        "Both flight classes in every split": bool((fc > 0).all().all()),
    }
    print("\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nSaved: {OUT_DIR / 'flight_splits.csv'}")
    print(f"Saved: {OUT_DIR / 'split_summary.json'}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Split checks FAILED")


if __name__ == "__main__":
    main()
