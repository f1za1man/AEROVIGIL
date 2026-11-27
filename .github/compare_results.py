"""After the rebuild: how far does each changed table differ from the committed one?

Prints one line per changed CSV under reports/ and data/: the largest absolute and
relative difference over its numeric columns, and whether any text column changed.
Figures (PNG) are listed only: fonts render differently across operating systems.
"""

import io
import subprocess
import sys

import numpy as np
import pandas as pd


def changed():
    out = subprocess.run(["git", "status", "--porcelain", "--", "reports", "data"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return [line[3:] for line in out.splitlines() if line.startswith(" M")]


def compare(path):
    old = pd.read_csv(io.StringIO(subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, text=True,
        check=True).stdout))  # fmt: skip
    new = pd.read_csv(path)
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        return f"{path}: shape or columns changed {old.shape} -> {new.shape}"
    num = new.select_dtypes("number").columns
    a, b = old[num].to_numpy(float), new[num].to_numpy(float)
    with np.errstate(all="ignore"):
        d = np.abs(a - b)
        rel = d / np.maximum(np.abs(a), 1e-12)
    d, rel = np.nan_to_num(d), np.nan_to_num(rel)
    text = [c for c in new.columns if c not in num
            and not old[c].astype(str).equals(new[c].astype(str))]  # fmt: skip
    return f"{path}: max abs diff {d.max():.3g}, max rel diff {rel.max():.3g}" + (
        f", text changed in {text}" if text else ""
    )


def main():
    lines = []
    for p in changed():
        lines.append(compare(p) if p.endswith(".csv") else f"{p}: changed (binary)")
    print("\n".join(lines) or "no result changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
