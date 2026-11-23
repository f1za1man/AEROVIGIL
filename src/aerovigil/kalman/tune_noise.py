"""Step 4: scale the measurement noise until the filter is statistically consistent.

For a consistent filter the normalised innovation squared (NIS) follows a
chi-square law. The median of chi-square(4) is 3.357, so the GPS/baro noise is
scaled until the median horizontal NIS on NORMAL TRAIN flights matches it.
Only normal training flights are used.

Usage:
    python src\\aerovigil\\kalman\\tune_noise.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import load_split_map  # noqa: E402
from aerovigil.kalman.common import (
    UNIT_TO_SECONDS,
    get_columns,
    load_convention,  # noqa: E402
    load_yaml,
    orig_truth,
    out_dir,
    resolve,
)
from aerovigil.kalman.pipeline import run_flight  # noqa: E402

MEDIAN_CHI2_4 = 3.3567
MEDIAN_CHI2_3 = 2.3660


def main():
    cfg = load_yaml(ROOT / "configs" / "kalman" / "kalman.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols = get_columns(cfg, pre)
    scale_t = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    conv = load_convention(cfg)
    tcfg, fcfg = cfg["noise_tuning"], cfg["filter"]

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    pool = report[
        (report["attack_rows"] == 0)
        & (report["flight_id"].astype(str).map(split_map) == "train")
    ]
    pool = pool.sample(min(tcfg["n_flights"], len(pool)), random_state=1)
    flights = [
        pd.read_parquet(resolve(cfg["paths"]["cleaned_dir"]) / f) for f in pool["file"]
    ]

    print("=" * 65)
    print("AEROVIGIL - FILTER NOISE TUNING")
    print("=" * 65)
    print(f"Normal TRAIN flights: {len(flights)} | IMU convention: {conv}")

    def run_all(scale):
        parts = []
        for g in flights:
            r = run_flight(
                g, cols, fcfg, conv, scale_t, scale, orig_truth(g, cols["label"])
            )
            if r is not None:
                parts.append(r[r["valid"]])
        d = pd.concat(parts)
        return d, float(np.median(d["nis_h"])), float(np.median(d["nis_v"]))

    scale, history = 1.0, []
    for it in range(tcfg["iterations"] + 1):
        d, med_h, med_v = run_all(scale)
        history.append(
            {
                "iteration": it,
                "scale": scale,
                "median_nis_h": med_h,
                "median_nis_v": med_v,
            }
        )
        print(
            f"  iteration {it}: scale {scale:8.3f} | median NIS horizontal {med_h:6.2f} "
            f"(target {MEDIAN_CHI2_4:.2f}) | vertical {med_v:5.2f} (target {MEDIAN_CHI2_3:.2f})"
        )
        if it == tcfg["iterations"]:
            break
        scale = float(
            np.clip(
                scale * (med_h / MEDIAN_CHI2_4) ** 0.8,
                tcfg["scale_min"],
                tcfg["scale_max"],
            )
        )

    white_std = d[["w_pos_e", "w_vel_e", "w_pos_n", "w_vel_n"]].std().round(3).to_dict()
    print("\nWhitened-innovation std (1.0 = consistent):", white_std)
    print(
        f"Valid ticks used: {len(d):,} ({len(d) / fcfg['rate_hz'] / 3600:.2f} flight-hours)"
    )
    json.dump(
        {
            "scale": scale,
            "history": history,
            "whitened_std": white_std,
            "median_nis_h": history[-1]["median_nis_h"],
            "median_nis_v": history[-1]["median_nis_v"],
        },
        (out_dir(cfg) / "filter_noise.json").open("w"),
        indent=2,
    )
    print(f"\nSaved: {out_dir(cfg) / 'filter_noise.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
