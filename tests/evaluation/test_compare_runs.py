"""Step A: the old-vs-new table pairs renamed threat-table rows correctly."""

import pandas as pd

import aerovigil.evaluation.compare_runs as cr


def test_old_sweep_rows_pair_with_their_renamed_new_rows(monkeypatch):
    cols = {f"{d}_detected": [1.0] for d in ("kalman", "velocity", "fused")}
    old = pd.DataFrame({"attacker": ["   Drift sweep 0.25 m/s (Module 04 onset)"],
                        **cols})  # fmt: skip
    new = pd.DataFrame({"attacker": ["Consistent drift 0.25 m/s"],
                        **{k: [0.5] for k in cols}})  # fmt: skip
    monkeypatch.setattr(cr, "old_csv", lambda path, commit: old)
    monkeypatch.setattr(cr, "new_csv", lambda path: new)
    rows = []
    cr.module10(rows, "abc")
    assert len(rows) == 3
    assert {r["group"] for r in rows} == {"Consistent drift 0.25 m/s"}
    assert all(r["old"] == 1.0 and r["new"] == 0.5 for r in rows)
