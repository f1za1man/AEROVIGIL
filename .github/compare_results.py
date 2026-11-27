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


TOL = 1e-9  # below this a table differs only by floating-point round-off


def compare(path):
    """(message, True when the table really changed)."""
    old = pd.read_csv(io.StringIO(subprocess.run(
        ["git", "show", f"HEAD:{path}"], capture_output=True, text=True,
        check=True).stdout))  # fmt: skip
    new = pd.read_csv(path)
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        return f"{path}: shape or columns changed {old.shape} -> {new.shape}", True
    num = new.select_dtypes("number").columns
    a, b = old[num].to_numpy(float), new[num].to_numpy(float)
    with np.errstate(all="ignore"):
        d = np.abs(a - b)
        rel = d / np.maximum(np.abs(a), 1e-12)
    d, rel = np.nan_to_num(d), np.nan_to_num(rel)
    text = [c for c in new.columns if c not in num
            and not old[c].astype(str).equals(new[c].astype(str))]  # fmt: skip
    real = bool(text) or d.max() > TOL
    msg = f"{path}: max abs diff {d.max():.3g}, max rel diff {rel.max():.3g}" + (
        f", text changed in {text}" if text else ""
    )
    return msg, real


def main():
    """Prints the details, then a last line the workflow reads: RESULT same / differ.
    Figures are listed but never count: fonts render differently across systems."""
    lines, real = [], False
    for p in changed():
        if p.endswith(".csv"):
            msg, r = compare(p)
            lines.append(msg)
            real |= r
        else:
            lines.append(f"{p}: changed (binary)")
    print("\n".join(lines))
    print("RESULT differ" if real else "RESULT same")
    return 0


if __name__ == "__main__":
    sys.exit(main())
