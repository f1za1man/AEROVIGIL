"""Generate the synthetic attack set from a YAML config (Module 04).

Usage:
    python src\\aerovigil\\attacks\\generate_attacks.py
    python src\\aerovigil\\attacks\\generate_attacks.py --limit 2 --output-dir data\\interim\\module04\\smoke
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    get_columns,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.attacks.injector import inject_attack  # noqa: E402
from aerovigil.attacks.sampler import sample_attack  # noqa: E402

SCALAR_PARAMS = [
    "direction_deg",
    "magnitude_m",
    "drift_rate_mps",
    "initial_rate_mps",
    "time_constant_s",
    "ramp_seconds",
    "max_offset_m",
    "amplitude_m",
]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def attacked_id(source_id, attack_set, attack_type, draw, rate):
    if attack_set == "sweep":
        return f"{source_id}__sweep_{attack_type}_r{rate:g}".replace(".", "p")
    return f"{source_id}__syn_{attack_type}_{draw:02d}"


def run(config_path, limit=None, output_dir=None):
    cfg = load_yaml(config_path)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols, scale = get_columns(cfg, pre)

    out_root = Path(output_dir) if output_dir else resolve(cfg["paths"]["output_dir"])
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    flights_dir = out_root / "attacked"
    flights_dir.mkdir(parents=True, exist_ok=True)
    stale = list(flights_dir.glob("*.parquet"))
    for f in stale:
        f.unlink()

    clean_dir = resolve(cfg["paths"]["clean_flights_dir"])
    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))

    clean = report[report["attack_rows"] == 0].sort_values("flight_id")
    missing = [f for f in clean["flight_id"] if str(f) not in split_map]
    if missing:
        raise SystemExit(
            f"{len(missing)} clean flights have no split, e.g. {missing[:3]}"
        )
    clean = clean.assign(split=clean["flight_id"].astype(str).map(split_map))
    if limit:
        clean = clean.groupby("split", group_keys=False).head(limit)

    types = cfg["enabled_attack_types"]
    sw = cfg["sweep"]
    print("=" * 65)
    print("AEROVIGIL - ATTACK GENERATOR")
    print("=" * 65)
    print(
        f"Clean source flights : {len(clean)} "
        f"({clean['split'].value_counts().to_dict()})"
    )
    print(f"Attack types         : {types}")
    print(f"Output folder        : {flights_dir}  (removed {len(stale)} old files)")
    free_gb = shutil.disk_usage(out_root).free / 1e9
    print(f"Free disk space      : {free_gb:.1f} GB")

    manifest, skipped = [], []
    started = time.time()
    for n, (_, r) in enumerate(clean.iterrows(), start=1):
        src_id, split = str(r["flight_id"]), r["split"]
        g = pd.read_parquet(clean_dir / r["file"])
        t_us = g[cols["time"]].to_numpy(dtype="int64")
        duration = (t_us[-1] - t_us[0]) * scale

        jobs = [
            ("main", t, d, None) for t in types for d in range(cfg["draws_per_type"])
        ]
        if sw["enabled"] and split in sw["splits"]:
            jobs += [
                ("sweep", "linear_drift", 0, rate) for rate in sw["drift_rates_mps"]
            ]

        for attack_set, typ, draw, rate in jobs:
            spec = sample_attack(cfg, src_id, duration, attack_set, typ, draw, rate)
            if spec is None:
                skipped.append((src_id, attack_set, typ))
                continue
            new_id = attacked_id(src_id, attack_set, typ, draw, rate)
            spec.update(attacked_flight_id=new_id, source_flight_id=src_id)
            out = inject_attack(g, spec, cols, scale)
            fname = safe_name(new_id) + ".parquet"
            out.to_parquet(flights_dir / fname, index=False, compression="zstd")

            p = spec["params"]
            lab = out[cols["label"]].to_numpy()
            manifest.append(
                {
                    "attacked_flight_id": new_id,
                    "source_flight_id": src_id,
                    "split": split,
                    "mission_type": r["mission_type"],
                    "attack_set": attack_set,
                    "attack_type": typ,
                    "draw": draw,
                    "velocity_mode": spec["velocity_mode"],
                    **{k: p.get(k, np.nan) for k in SCALAR_PARAMS},
                    "start_s": spec["start_s"],
                    "duration_s": spec["duration_s"],
                    "end_s": spec["start_s"] + spec["duration_s"],
                    "flight_duration_s": duration,
                    "attack_start_time_usec": int(t_us[0] + spec["start_s"] / scale),
                    "rows": len(out),
                    "attack_rows": int((lab == 1).sum()),
                    "max_offset_m": float(out["spoof_offset_m"].max()),
                    "params_json": json.dumps(p),
                    "file": fname,
                }
            )
        if n % 20 == 0 or n == len(clean):
            print(
                f"Processed {n}/{len(clean)} source flights "
                f"({len(manifest)} attacked flights written)"
            )

    man = pd.DataFrame(manifest)
    man.to_csv(out_root / "attack_manifest.csv", index=False)
    shutil.copy(config_path, out_root / "attacks_config_used.yaml")

    # ---------------- summary ----------------
    print("\n" + "-" * 65)
    print(
        f"Attacked flights written : {len(man):,}   skipped (too short): {len(skipped)}"
    )
    print(f"Elapsed                  : {time.time() - started:.0f} s")
    print("\nFLIGHTS PER SPLIT x ATTACK SET")
    print(pd.crosstab(man["split"], man["attack_set"]).to_string())
    print("\nFLIGHTS PER ATTACK TYPE x VELOCITY MODE (main set)")
    print(
        pd.crosstab(
            man.loc[man.attack_set == "main", "attack_type"],
            man.loc[man.attack_set == "main", "velocity_mode"],
        ).to_string()
    )
    print("\nATTACK DURATION (s) AND PEAK SPOOFED OFFSET (m) BY TYPE (main set)")
    print(
        man[man.attack_set == "main"]
        .groupby("attack_type")[["duration_s", "max_offset_m"]]
        .describe()
        .loc[
            :,
            [
                ("duration_s", "min"),
                ("duration_s", "50%"),
                ("duration_s", "max"),
                ("max_offset_m", "min"),
                ("max_offset_m", "50%"),
                ("max_offset_m", "max"),
            ],
        ]
        .round(2)
        .to_string()
    )
    size_gb = sum(f.stat().st_size for f in flights_dir.glob("*.parquet")) / 1e9
    print(f"\nDisk used by attacked flights: {size_gb:.2f} GB")

    checks = {
        "One parquet file per manifest row": len(list(flights_dir.glob("*.parquet")))
        == len(man),
        "Attacked flight IDs are unique": man["attacked_flight_id"].is_unique,
        "Every attacked flight contains attack rows": bool(
            (man["attack_rows"] > 0).all()
        ),
        "Attacked flight inherits its source split": bool(
            (man["split"] == man["source_flight_id"].map(split_map)).all()
        ),
        "No attacked ID equals an original flight ID": not set(
            man["attacked_flight_id"]
        )
        & set(report["flight_id"].astype(str)),
    }
    print("\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nManifest: {out_root / 'attack_manifest.csv'}")
    print("=" * 65)
    if not all(checks.values()):
        raise SystemExit("Attack generation checks FAILED")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=str(ROOT / "configs" / "attacks" / "attacks.yaml")
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max clean source flights per split (smoke tests)",
    )
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.limit, a.output_dir)


if __name__ == "__main__":
    main()
