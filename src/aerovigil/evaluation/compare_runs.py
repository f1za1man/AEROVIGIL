r"""Old vs new results after the Module 04 regeneration (checklist Step A).

Module 04 was regenerated with the warp onset (no 69% velocity overshoot at switch-on)
and Modules 05-10 re-run on it. This reads the old results straight out of git, from
the last commit before the regeneration, and the new ones from the working tree, and
puts them side by side. Nothing is recomputed; every number comes from a saved file.

Writes reports/tables/A_old_vs_new.csv

Usage:
    python src\aerovigil\evaluation\compare_runs.py
    python src\aerovigil\evaluation\compare_runs.py --old 6d64964
"""

import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD_COMMIT = "3e2a94c"  # last commit before Module 04 was regenerated
OUT = ROOT / "reports" / "tables" / "A_old_vs_new.csv"


def old_csv(path, commit):
    text = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True
    )
    return pd.read_csv(io.StringIO(text))


def new_csv(path):
    return pd.read_csv(ROOT / path)


def pair(rows, module, metric, group, old, new):
    rows.append({"module": module, "metric": metric, "group": group,
                 "old": float(old), "new": float(new),
                 "change": float(new) - float(old)})  # fmt: skip


def module05(rows, commit):
    p = "data/processed/module05_v2/rescore_window.csv"
    key = ["task", "features", "model", "sources"]
    best = {"task": "event", "features": "all", "model": "random_forest",
            "sources": "orig+syn"}  # fmt: skip
    o, n = old_csv(p, commit), new_csv(p)
    sel = lambda d: d[(d[key] == pd.Series(best)).all(axis=1)].iloc[0]  # noqa: E731
    for col in ("auc_orig", "auc_syn_position_only", "auc_syn_consistent"):
        pair(rows, "M5 RF (event, all, orig+syn)", "AUC", col,
             sel(o)[col], sel(n)[col])  # fmt: skip


def module06(rows, commit):
    p = "data/processed/module06/sweep_frontier.csv"
    o, n = old_csv(p, commit), new_csv(p)
    for det in ("nis_test", "cusum"):
        a = o[o["detector"] == det].set_index("drift_rate_mps")
        b = n[n["detector"] == det].set_index("drift_rate_mps")
        for r in a.index:
            pair(rows, f"M6 {det}", "drift sweep detected", f"{r:g} m/s",
                 a.loc[r, "detection_rate"], b.loc[r, "detection_rate"])  # fmt: skip
    p = "data/processed/module06/summary_by_attack.csv"
    o, n = old_csv(p, commit), new_csv(p)
    k = ["set", "attack_type", "velocity_mode", "detector"]
    m = o.merge(n, on=k, suffixes=("_o", "_n"))
    m = m[(m["detector"] == "nis_test") & (m["velocity_mode"] == "consistent")]
    for _, r in m.iterrows():
        pair(rows, "M6 nis_test", "consistent detected", r["attack_type"],
             r["detection_rate_o"], r["detection_rate_n"])  # fmt: skip


def module07(rows, commit):
    p = "data/processed/module07_v2/flights.csv"
    o, n = old_csv(p, commit), new_csv(p)
    for det in ("lstm_ae", "lstm_supervised"):
        for mode in ("position_only", "consistent"):

            def rate(d, det=det, mode=mode):
                d = d[(d["detector"] == det) & d["held_out"].isna()]
                d = d[(d["attack_set"] == "main") & (d["velocity_mode"] == mode)]
                return d["detected"].mean()

            pair(rows, f"M7 {det}", "synthetic detected (mean over seeds)", mode,
                 rate(o), rate(n))  # fmt: skip


def module09(rows, commit):
    p = "data/processed/module09/false_alarms.csv"
    o, n = old_csv(p, commit), new_csv(p)
    pair(rows, "M9 consistency", "false alarms per hour", "normal test flights",
         o["alarms_per_hour"].iloc[0], n["alarms_per_hour"].iloc[0])  # fmt: skip


def module10(rows, commit):
    p = "data/processed/module10/threat_table.csv"
    o, n = old_csv(p, commit), new_csv(p)
    # the sweep rows were renamed: "Drift sweep 0.5 m/s (Module 04 onset)" is now
    # "Consistent drift 0.5 m/s", since with the warp onset they no longer measure
    # the switch-on
    o = o.assign(attacker=o["attacker"].str.strip().str.replace(
        r"^Drift sweep (.*) m/s \(Module 04 onset\)$", r"Consistent drift \1 m/s",
        regex=True))  # fmt: skip
    n = n.assign(attacker=n["attacker"].str.strip())
    m = o.merge(n, on="attacker", suffixes=("_o", "_n"))
    for _, r in m.iterrows():
        for det in ("kalman", "velocity", "fused"):
            pair(rows, f"M10 {det}", "detected", r["attacker"].strip(),
                 r[f"{det}_detected_o"], r[f"{det}_detected_n"])  # fmt: skip


def have_commit(commit):
    """True when the old results' commit is in this clone (a shallow clone lacks it)."""
    done = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                          cwd=ROOT, capture_output=True)  # fmt: skip
    return done.returncode == 0


def run(commit=OLD_COMMIT):
    if not have_commit(commit):
        print(f"Commit {commit} is not in this clone (shallow?): "
              f"{OUT.name} kept as committed")  # fmt: skip
        return pd.read_csv(OUT) if OUT.is_file() else None
    rows = []
    for f in (module05, module06, module07, module09, module10):
        f(rows, commit)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(f"Old = commit {commit}, new = working tree\n")
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {OUT}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD_COMMIT, help="commit holding the old results")
    run(ap.parse_args().old)


if __name__ == "__main__":
    main()
