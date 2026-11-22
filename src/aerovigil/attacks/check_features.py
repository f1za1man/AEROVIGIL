"""Step 8: run attacked flights through the Module 03 feature builder.

Confirms the attacked flights are schema-compatible with the existing
pipeline and shows how visible each attack is to the 22 Module 03 features.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import get_columns, load_yaml, resolve  # noqa: E402
from aerovigil.features.build_features import build_flight  # noqa: E402

PER_GROUP = 4          # attacked flights per (attack type, velocity mode)
KEY = ["res_horizontal_velocity", "res_vertical_velocity_abs", "res_altitude_abs"]


def main():
    cfg = load_yaml(ROOT / "configs" / "attacks" / "attacks.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols, scale = get_columns(cfg, pre)
    c = {"flight": cols["flight"], "time": cols["time"], "label": cols["label"],
         "lat": cols["lat"], "lon": cols["lon"], "alt": cols["alt"],
         "vn": cols["vn"], "ve": cols["ve"], "vu": cols["vu"], "baro": cols["baro"]}
    out_root = resolve(cfg["paths"]["output_dir"])
    man = pd.read_csv(out_root / "attack_manifest.csv")
    main_set = man[man.attack_set == "main"]
    sample = (main_set.sort_values("attacked_flight_id")
              .groupby(["attack_type", "velocity_mode"]).head(PER_GROUP))

    print("=" * 65)
    print("AEROVIGIL - ATTACKED FLIGHTS THROUGH THE MODULE 03 FEATURE BUILDER")
    print("=" * 65)
    print(f"Flights processed: {len(sample)}\n")

    records = []
    for _, m in sample.iterrows():
        g = pd.read_parquet(out_root / "attacked" / m["file"])
        f = build_flight(g, c, scale)
        f["attack_type"], f["velocity_mode"] = m["attack_type"], m["velocity_mode"]
        records.append(f[["attack_type", "velocity_mode", "attack_label"] + KEY])
    allf = pd.concat(records, ignore_index=True)
    allf["state"] = np.where(allf["attack_label"] == 1, "attack", "normal")

    print("KEY FEATURES: median and 95th percentile, normal rows vs attack rows of the same flights")
    for k in KEY:
        t = (allf.groupby(["attack_type", "velocity_mode", "state"])[k]
             .agg(median="median", p95=lambda x: x.quantile(0.95))
             .unstack("state").round(4))
        print(f"\n{k}")
        print(t.to_string())
    print("\nNaN fraction (first second of flight / after gaps):",
          f"{allf[KEY].isna().mean().round(4).to_dict()}")
    print("=" * 65)


if __name__ == "__main__":
    main()