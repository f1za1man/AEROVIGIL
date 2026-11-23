"""Step 5: run the Kalman filters over every flight and save the innovations.

One innovation file (one row per filter tick) is written per flight:
    original flights   -> innovations/orig/
    Module 04 'main'   -> innovations/syn/
    Module 04 'sweep'  -> innovations/sweep/

Usage:
    python src\\aerovigil\\kalman\\run_filter.py
    python src\\aerovigil\\kalman\\run_filter.py --limit 3          (smoke test)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.kalman.common import (
    UNIT_TO_SECONDS,
    get_columns,
    load_convention,  # noqa: E402
    load_noise_scale,
    load_yaml,
    orig_truth,
    out_dir,
    resolve,
    truth_arrays,
)
from aerovigil.kalman.pipeline import run_flight  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "kalman" / "kalman.yaml")
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="flights per set (smoke test)"
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols = get_columns(cfg, pre)
    scale_t = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    conv, noise = load_convention(cfg), load_noise_scale(cfg)
    fcfg = cfg["filter"]
    base = Path(a.output_dir) if a.output_dir else out_dir(cfg)
    base = base if base.is_absolute() else ROOT / base

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    jobs = {
        "orig": [
            (r["flight_id"], resolve(cfg["paths"]["cleaned_dir"]) / r["file"], False)
            for _, r in report.iterrows()
        ],
        "syn": [
            (
                r["attacked_flight_id"],
                resolve(cfg["paths"]["attacked_dir"]) / r["file"],
                True,
            )
            for _, r in man[man.attack_set == "main"].iterrows()
        ],
        "sweep": [
            (
                r["attacked_flight_id"],
                resolve(cfg["paths"]["attacked_dir"]) / r["file"],
                True,
            )
            for _, r in man[man.attack_set == "sweep"].iterrows()
        ],
    }
    print("=" * 65)
    print("AEROVIGIL - KALMAN FILTER RUN")
    print("=" * 65)
    print(f"IMU convention: {conv if conv else 'NONE (constant-velocity model)'}")
    print(
        f"Noise scale   : {noise:.4f} | filter rate {fcfg['rate_hz']} Hz | warm-up {fcfg['warmup_s']} s"
    )
    started, total_ticks, skipped = time.time(), 0, 0
    for name, items in jobs.items():
        folder = base / "innovations" / name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.parquet"):
            old.unlink()
        if a.limit:
            items = items[: a.limit]
        for n, (fid, path, attacked) in enumerate(items, start=1):
            g = pd.read_parquet(path)
            truth = truth_arrays(g) if attacked else orig_truth(g, cols["label"])
            r = run_flight(g, cols, fcfg, conv, scale_t, noise, truth)
            if r is None:
                skipped += 1
                continue
            r.insert(0, "flight_id", str(fid))
            r.to_parquet(folder / path.name, index=False)
            total_ticks += len(r)
            if n % 100 == 0 or n == len(items):
                print(f"  {name:6s} {n}/{len(items)}  ({time.time() - started:.0f} s)")
    print(f"\nTicks written: {total_ticks:,} | flights skipped (too short): {skipped}")
    print(f"Elapsed: {time.time() - started:.0f} s | saved to {base / 'innovations'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
