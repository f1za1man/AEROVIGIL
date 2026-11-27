r"""Step J: compare every result of a clean rebuild with the committed results.

Runs over the files Git tracks under reports/ and data/ in THIS repository and looks
for the same file in another copy of the project (the clean rebuild):

    CSV / CSV.GZ   same columns and rows; the largest absolute and relative difference
                   over numeric columns; text columns compared exactly
    JSON           numbers compared with the same tolerance, everything else exactly
    PNG and other  byte-identical or not (figures are redrawn in Step K anyway)

Differences below 1e-9 (absolute) count as floating-point round-off. Calibration files
(module06/*.json, module09/*.json, module10/thresholds.json) are listed first, because
none of them may be replaced without the owner seeing old and new.

Usage:
    python src\aerovigil\evaluation\compare_trees.py D:\AeroVigil_stepJ
    python src\aerovigil\evaluation\compare_trees.py D:\AeroVigil_stepJ --out report.csv
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TOL = 1e-9
CALIBRATION = ("data/processed/module06/", "data/processed/module09/",
               "data/processed/module10/thresholds.json")  # fmt: skip


def tracked():
    out = subprocess.run(["git", "ls-files", "reports", "data"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return [p for p in out.splitlines() if not p.endswith(".gitkeep")]


def is_calibration(path):
    return path.endswith(".json") and any(path.startswith(c) for c in CALIBRATION)


def compare_frames(a, b):
    if list(a.columns) != list(b.columns):
        return "columns differ", np.inf, np.inf
    if a.shape != b.shape:
        return f"rows {len(a)} -> {len(b)}", np.inf, np.inf
    num = a.select_dtypes("number").columns.intersection(
        b.select_dtypes("number").columns
    )
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    with np.errstate(all="ignore"):
        d = np.abs(x - y)
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.where(both_nan, 0.0, d)
        d = np.where(np.isnan(d), np.inf, d)  # NaN on one side only
        rel = d / np.maximum(np.abs(x), 1e-12)
    text = [c for c in a.columns if c not in num
            and not a[c].astype(str).equals(b[c].astype(str))]  # fmt: skip
    note = f"text differs in {text[:5]}" if text else ""
    dmax = float(d.max()) if d.size else 0.0
    rmax = float(rel.max()) if rel.size else 0.0
    if text:
        dmax = max(dmax, np.inf)
    return note, dmax, rmax


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
        return out
    return {prefix.rstrip("."): obj}


def compare_json(pa, pb):
    a, b = flatten(json.loads(pa.read_text())), flatten(json.loads(pb.read_text()))
    if set(a) != set(b):
        return "keys differ", np.inf, np.inf
    dmax = rmax = 0.0
    changed = []
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(float(va) - float(vb))
            dmax = max(dmax, d)
            rmax = max(rmax, d / max(abs(float(va)), 1e-12))
            if d > TOL:
                changed.append(f"{k}: {va} -> {vb}")
        elif va != vb:
            changed.append(f"{k}: {va} -> {vb}")
            dmax = np.inf
    return "; ".join(changed[:6]), dmax, rmax


def compare(path, other):
    a, b = ROOT / path, Path(other) / path
    if not b.is_file():
        return {"file": path, "status": "missing in rebuild"}
    try:
        if path.endswith((".csv", ".csv.gz")):
            note, d, r = compare_frames(pd.read_csv(a), pd.read_csv(b))
        elif path.endswith(".json"):
            note, d, r = compare_json(a, b)
        else:
            same = a.read_bytes() == b.read_bytes()
            return {"file": path, "status": "same" if same else "binary differs"}
    except Exception as e:  # noqa: BLE001  an unreadable file is reported, not fatal
        return {"file": path, "status": f"could not compare: {e}"}
    status = "same" if d == 0 else "round-off" if d <= TOL else "differs"
    return {"file": path, "status": status, "max_abs_diff": d, "max_rel_diff": r,
            "note": note}  # fmt: skip


def run(other, out=None):
    rows = [compare(p, other) for p in tracked()]
    t = pd.DataFrame(rows)
    t.insert(1, "calibration", t["file"].map(is_calibration))
    t = t.sort_values(["calibration", "status", "file"], ascending=[False, True, True])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(t["status"].value_counts().to_string())
    cal = t[t["calibration"]]
    print("\nCALIBRATION FILES")
    print(cal.drop(columns="calibration").to_string(index=False))
    bad = t[~t["status"].isin(["same", "round-off"]) & ~t["calibration"]]
    print("\nOTHER FILES THAT DIFFER OR ARE MISSING")
    print(
        bad.drop(columns="calibration").to_string(index=False) if len(bad) else "none"
    )
    if out:
        t.to_csv(out, index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("other", help="root folder of the clean rebuild")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.other, a.out)


if __name__ == "__main__":
    main()
