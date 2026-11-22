import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
clean_dir = ROOT / "data" / "processed" / "module03" / "cleaned"
rep = pd.read_csv(ROOT / "data" / "processed" / "module03" / "cleaning_report.csv")
rep = rep[rep["attack_rows"] == 0]

rows = []
for f in rep["file"].iloc[:8]:
    g = pd.read_parquet(clean_dir / f)
    h = np.hypot(g.velocity_north, g.velocity_east)
    s3 = np.sqrt(g.velocity_north**2 + g.velocity_east**2 + g.velocity_up**2)
    rows.append({
        "flight": f[:30],
        "speed_min": g.velocity.min(), "speed_max": g.velocity.max(),
        "mae_vs_horizontal": (g.velocity - h).abs().mean(),
        "mae_vs_3d": (g.velocity - s3).abs().mean(),
        "max_abs_vs_horizontal": (g.velocity - h).abs().max(),
        "max_abs_vs_3d": (g.velocity - s3).abs().max(),
    })
print(pd.DataFrame(rows).round(4).to_string(index=False))