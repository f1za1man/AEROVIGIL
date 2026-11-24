r"""AEROVIGIL Module 04 v2: generate the synthetic attack set on the Module 03 v2 base.

For each of the 144 normal flights and each of the five attack types it draws one
attack (same seed and sampler as Module 04 v1) and writes two copies, position_only
and consistent, with identical parameters and timing. Test flights also get the
7-rate drift sweep (consistent only). See inject_v2.py for what changed against v1.

Needs configs/attacks/attacks.yaml, the Module 04 v1 profiles.py / sampler.py and the
Module 03 v2 base. Writes data/processed/module04_v2/base/<id>.parquet and
data/interim/module04_v2/attack_manifest_v2.csv.

Usage:
    python src\aerovigil\attacks\generate_v2.py
    python src\aerovigil\attacks\generate_v2.py --limit 1 --output-dir data\interim\module04_v2\smoke
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aerovigil.attacks.inject_v2 import KEEP, MODES, SCALARS, SHORT, inject
from aerovigil.attacks.profiles import ATTACK_TYPES
from aerovigil.attacks.sampler import sample_attack

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT / "data" / "processed" / "module03_v2" / "base"
SPLITS = ROOT / "data" / "interim" / "module03" / "flight_splits.csv"
BASE_REPORT = ROOT / "data" / "interim" / "module03_v2" / "base_report.csv"
CONFIG = ROOT / "configs" / "attacks" / "attacks.yaml"
OUT_DIR = ROOT / "data" / "processed" / "module04_v2"
MANIFEST = ROOT / "data" / "interim" / "module04_v2" / "attack_manifest_v2.csv"


def jobs_for(cfg, split):
    jobs = [
        ("main", typ, d, None)
        for typ in ATTACK_TYPES
        for d in range(cfg["draws_per_type"])
    ]
    sw = cfg["sweep"]
    if sw["enabled"] and split in sw["splits"]:
        jobs += [("sweep", "linear_drift", 0, r) for r in sw["drift_rates_mps"]]
    return jobs


def attack_id(src, attack_set, typ, draw, rate, mode):
    if attack_set == "sweep":
        return f"{src}__sweep_{typ}_r{rate:g}".replace(".", "p")
    return f"{src}__syn_{typ}_{draw:02d}_{SHORT[mode]}"


def run(output_dir=None, limit=None, min_free_gb=8.0):
    cfg = yaml.safe_load(CONFIG.read_text())
    out_root = Path(output_dir) if output_dir else OUT_DIR
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    flights_dir = out_root / "base"
    flights_dir.mkdir(parents=True, exist_ok=True)
    for f in flights_dir.glob("*.parquet"):
        f.unlink()
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    rep = pd.read_csv(BASE_REPORT)
    src = rep[rep["attacked"] == 0].sort_values("flight_id")
    src = src.assign(split=src["flight_id"].map(splits))
    if src["split"].isna().any():
        raise SystemExit(
            "normal flights without a split: "
            + str(src[src.split.isna()].flight_id.tolist()[:3])
        )
    if limit:
        src = src.groupby("split", group_keys=False).head(limit)
    free = shutil.disk_usage(out_root).free / 1e9
    print(f"Normal source flights: {len(src)}   free disk: {free:.1f} GB")
    if free < min_free_gb:
        raise SystemExit(
            f"Only {free:.1f} GB free; need {min_free_gb} GB (see --min-free-gb)"
        )
    rows, started = [], time.time()
    for k, (_, r) in enumerate(src.iterrows(), start=1):
        g = pd.read_parquet(BASE_DIR / r["file"], columns=KEEP)
        duration = float(g["t_s"].iloc[-1])
        for attack_set, typ, draw, rate in jobs_for(cfg, r["split"]):
            spec = sample_attack(
                cfg, r["flight_id"], duration, attack_set, typ, draw, rate
            )
            if spec is None:
                continue
            modes = MODES if attack_set == "main" else ("consistent",)
            for mode in modes:
                new_id = attack_id(r["flight_id"], attack_set, typ, draw, rate, mode)
                out, facts = inject(g, spec, mode, new_id)
                fname = new_id.replace(".", "p") + ".parquet"
                out.to_parquet(flights_dir / fname, index=False, compression="zstd")
                p = spec["params"]
                end_s = spec["start_s"] + spec["duration_s"]
                rows.append({
                    "attacked_flight_id": new_id, "source_flight_id": r["flight_id"],
                    "pair_id": f"{r['flight_id']}|{typ}|{draw}", "split": r["split"],
                    "mission_type": r["mission"], "attack_set": attack_set,
                    "attack_type": typ, "draw": draw, "velocity_mode": mode,
                    **{c: p.get(c, np.nan) for c in SCALARS},
                    "start_s": spec["start_s"], "duration_s": spec["duration_s"],
                    "end_s": end_s, "flight_duration_s": duration,
                    "ends_at_flight_end": bool(end_s >= duration - 0.05),
                    "rows": len(out), "label_rows": int(out["attack_label"].sum()),
                    "spoof_rows": int(out["spoof_truth"].sum()),
                    "peak_offset_m": float(out["offset_m"].max()),
                    "onset_s": facts["onset_s"], "ramp_end_s": facts["ramp_end_s"],
                    "spoof_end_s": facts["spoof_end_s"], "plateau_m": facts["plateau_m"],
                    "params_json": json.dumps(p), "file": fname,
                })  # fmt: skip
        if k % 20 == 0 or k == len(src):
            print(
                f"Processed {k}/{len(src)} source flights ({len(rows)} attacked flights)"
            )
    man = pd.DataFrame(rows)
    manifest = out_root / "attack_manifest_v2.csv" if output_dir else MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(manifest, index=False)
    report(man, flights_dir, manifest, time.time() - started)
    return man


def report(man, flights_dir, manifest, elapsed):
    main = man[man.attack_set == "main"]
    print(f"\nAttacked flights written: {len(man):,}   elapsed {elapsed:.0f} s")
    print("\nFLIGHTS PER SPLIT x ATTACK SET")
    print(pd.crosstab(man["split"], man["attack_set"]).to_string())
    print("\nMAIN SET: ATTACK TYPE x VELOCITY MODE (must be equal)")
    print(pd.crosstab(main["attack_type"], main["velocity_mode"]).to_string())
    print("\nPEAK OFFSET (m) and SECONDS WITH OFFSET >= 1 m, position_only copies")
    po = main[main.velocity_mode == "position_only"]
    spoofed = (po["spoof_end_s"] - po["onset_s"]).rename("spoofed_s")
    print(pd.concat([po["attack_type"], po["peak_offset_m"], spoofed], axis=1)
          .groupby("attack_type").describe().loc[:, [("peak_offset_m", "min"),
          ("peak_offset_m", "50%"), ("spoofed_s", "min"), ("spoofed_s", "50%")]].round(2).to_string())  # fmt: skip
    weak = main[(main.velocity_mode == "position_only") & (main.peak_offset_m < 2.0)]
    print(
        f"\nAttacked copies whose peak offset stays below 2 m: {len(weak)} "
        f"({weak['attack_type'].value_counts().to_dict()})"
    )
    print("\nATTACKS THAT RUN TO THE END OF THE FLIGHT (no snap-back inside the log)")
    print(pd.crosstab(main["mission_type"], main["ends_at_flight_end"]).to_string())
    size = sum(f.stat().st_size for f in flights_dir.glob("*.parquet")) / 1e9
    checks = {
        "one parquet per manifest row": len(list(flights_dir.glob("*.parquet"))) == len(man),
        "attacked ids unique": man["attacked_flight_id"].is_unique,
        "modes paired: equal counts per type": bool(
            (main.groupby(["attack_type", "velocity_mode"]).size().unstack().nunique(axis=1) == 1).all()),
        "every attacked flight has label rows": bool((man["label_rows"] > 0).all()),
    }  # fmt: skip
    print(f"\nDisk used: {size:.2f} GB\nManifest : {manifest}\n\nACCEPTANCE CHECKS")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("Module 04 v2 generation checks FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="source flights per split")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    a = ap.parse_args()
    run(a.output_dir, a.limit, a.min_free_gb)


if __name__ == "__main__":
    main()
