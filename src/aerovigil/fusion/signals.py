r"""AEROVIGIL Module 10 step 1: compute and cache the two detector signals per flight.

Module 10 fuses the two detectors that work on this dataset:

    kalman    Module 06's chi-square test on the horizontal NIS, one value per 10 Hz
              filter tick. Fast against abrupt onsets.
    velocity  Module 09's velocity-consistency channel, res_horizontal_velocity_roll_mean,
              one value per logged row. It caught 97% of the dataset's own attacks.

Module 09's course channel is left out on purpose. On real flights the nose does not
point along the track (clean spread 0.19 against 0.002 on nose-forward synthetic
flights), so including it only raises the shared threshold and blunts the velocity
channel.

Both signals are threshold-free, so they are computed once here and every calibration
and evaluation in fuse.py runs from the cache in seconds.

Monitored time is measured from the timestamps, not from a row count. Flights in this
dataset log at about 125 Hz or 250 Hz (172 rows per second on average), and Module 09's
detector divided rows by a fixed 125, which overstated its flight-hours by about 38%.

Both signals live on one clock, the base table's t_s. The filter drops IMU dropout rows
before it starts and counts time from the first row it keeps; real logs open with about
18 dropout rows, so its own clock runs 0.10-0.12 s behind t_s. An alarm at the instant
of an abrupt onset then fell just before the attack window and was scored as a false
alarm. The filter therefore carries each tick's t_s through, and that is what is cached.
Row times stay float64 for the same reason: in float32, 101.764 is 101.76399994, which
puts an alarm on the onset row a hair before the window it belongs to.

Writes data/processed/module10/signals/<flight_id>.npz and signals_index.csv

Usage:
    python src\aerovigil\fusion\signals.py
    python src\aerovigil\fusion\signals.py --limit 4
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from aerovigil.adaptive.oracle import KalmanOracle
from aerovigil.data.build_raw_base import PHASE_RAMP
from aerovigil.features.build_features_v2 import build_flight_v2
from aerovigil.kalman.pipeline import run_flight
from aerovigil.lstm.build_sequences_v2 import (
    BASE_REPORT,
    MANIFEST,
    ROOT,
    SPLITS,
    roles_for,
)

M03_BASE = ROOT / "data" / "processed" / "module03_v2" / "base"
M04_BASE = ROOT / "data" / "processed" / "module04_v2" / "base"
OUT_DIR = ROOT / "data" / "processed" / "module10"
VELOCITY = "res_horizontal_velocity_roll_mean"
MAX_GAP_S = 1.0  # a step longer than this is a logging gap, not monitored time
# Bumped whenever the cached content changes meaning; older files are rebuilt.
# 2: Kalman tick times on the base-table clock, row times kept in float64.
CACHE_VERSION = 2


def truth_of(g):
    """Ground truth the filter carries along. Real base flights have no spoof_offset_*
    columns, so those fall back to zeros; nothing here is ever a filter input."""
    n = len(g)

    def col(name):
        return g[name].to_numpy(float) if name in g.columns else np.zeros(n)

    return {
        "label": g["attack_label"].to_numpy(float),
        "offset_e": col("spoof_offset_east_m"),
        "offset_n": col("spoof_offset_north_m"),
        "offset_m": g["offset_m"].to_numpy(float),
        "rel_start": g["seconds_since_onset"].to_numpy(float),
        "t_s": g["t_s"].to_numpy(float),
    }


def monitored_seconds(t, valid, max_gap=MAX_GAP_S):
    """Time covered by valid samples, from the timestamps themselves."""
    t = np.asarray(t, float)
    valid = np.asarray(valid, bool)
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    both = valid[1:] & valid[:-1] & (dt > 0) & (dt <= max_gap)
    return float(dt[both].sum())


def attack_times(g):
    """(label start, label end, measured onset) in seconds, or NaNs for a clean flight.

    The label window decides whether an alarm counts as a detection. The measured
    onset, the first row of the ramp, is what delays are taken from, so a detector is
    not credited for the seconds before the offset had actually begun to grow.
    """
    t = g["t_s"].to_numpy(float)
    on = g["attack_label"].to_numpy(float) == 1
    if not on.any():
        return np.nan, np.nan, np.nan
    ramp = g["phase"].to_numpy() == PHASE_RAMP
    onset = float(t[ramp][0]) if ramp.any() else float(t[on][0])
    return float(t[on][0]), float(t[on][-1]), onset


class Context:
    """Module 06's filter configuration, borrowed from the tested Module 08 oracle."""

    def __init__(self):
        o = KalmanOracle()
        self.cols, self.fcfg, self.conv = o.cols, o.fcfg, o.conv
        self.scale, self.noise = o.scale, o.noise
        self.rate_hz = float(o.fcfg["rate_hz"])


def flight_signals(g, ctx):
    """Both signals and the truth needed to score them, for one flight."""
    df = run_flight(g, ctx.cols, ctx.fcfg, ctx.conv, ctx.scale, ctx.noise, truth_of(g))
    f = build_flight_v2(g)
    if len(f) != len(g):
        raise SystemExit(
            "build_flight_v2 changed the row count; the signals would misalign"
        )
    start, end, onset = attack_times(g)
    if df is None or not len(df):
        kf_t = kf_nis = np.zeros(0)
        kf_valid = np.zeros(0, bool)
    else:
        kf_t = df["row_t_s"].to_numpy(float)  # the base-table clock, see top
        kf_nis = df["nis_h"].to_numpy(float)
        kf_valid = df["valid"].to_numpy(bool) & np.isfinite(kf_nis)
    row_t = g["t_s"].to_numpy(float)
    vel = f[VELOCITY].to_numpy(float)
    return {
        "kf_t": kf_t, "kf_nis": kf_nis, "kf_valid": kf_valid,
        "row_t": row_t, "vel": vel, "vel_valid": np.isfinite(vel),
        "offset_m": g["offset_m"].to_numpy(float),
        "start_s": start, "end_s": end, "onset_s": onset,
        "version": CACHE_VERSION,
    }  # fmt: skip


ROW_ARRAYS = ("vel", "offset_m")  # row_t stays float64, see the top of this file


def save(path, sig):
    """Per-row values go to float32 to shrink the cache. Times do not: they are
    compared with the attack window, and a float32 time can land on the wrong side."""
    out = {}
    for k, v in sig.items():
        v = np.asarray(v)
        out[k] = v.astype(np.float32) if k in ROW_ARRAYS else v
    # Write beside the target and rename into place only once complete, so an
    # interrupted run can never leave a half-written file that looks like a cache hit.
    path = Path(path)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


class StaleCache(ValueError):
    """A cache file written by an older version of this module, or from a source
    flight that has since been regenerated."""


def source_stamp(path):
    """Size and modification time of a source flight. Regenerated Module 04 flights
    keep their names, so the name alone cannot tell a cache file is out of date."""
    st = Path(path).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def load(path, stamp=None):
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    if int(out.get("version", 0)) != CACHE_VERSION:
        raise StaleCache(path)
    if stamp is not None and str(out.get("source", "")) != stamp:
        raise StaleCache(path)
    for k in ("start_s", "end_s", "onset_s"):
        out[k] = float(out[k])
    return out


def jobs(limit=None):
    """Every original flight, plus the Module 04 v2 test-split attacks."""
    rep = pd.read_csv(BASE_REPORT).sort_values("flight_id")
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    role = roles_for(rep, splits)
    rows = []
    for _, r in rep.iterrows():
        fid = r["flight_id"]
        rows.append({
            "path": M03_BASE / r["file"], "flight_id": fid, "set": "orig",
            "split": splits[fid], "role": role[fid], "mission_type": r["mission"],
            "attack_type": "original" if r["attacked"] else "none",
            "velocity_mode": "na", "drift_rate_mps": np.nan,
            "attacked": bool(r["attacked"]),
        })  # fmt: skip
    if Path(MANIFEST).is_file():
        man = pd.read_csv(MANIFEST)
        for _, r in man[man["split"] == "test"].iterrows():
            rows.append({
                "path": M04_BASE / r["file"], "flight_id": r["attacked_flight_id"],
                "set": r["attack_set"], "split": "test", "role": "test",
                "mission_type": r["mission_type"], "attack_type": r["attack_type"],
                "velocity_mode": r["velocity_mode"],
                "drift_rate_mps": float(r.get("drift_rate_mps", np.nan)),
                "attacked": True,
            })  # fmt: skip
    if limit:
        by = {}
        for row in rows:
            by.setdefault((row["set"], row["attack_type"], row["split"]), []).append(
                row
            )
        rows = [r for group in by.values() for r in group[:limit]]
    return rows


def run(limit=None, output_dir=None, force=False):
    out = Path(output_dir) if output_dir else OUT_DIR
    out = out if out.is_absolute() else ROOT / out
    sig_dir = out / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    rows = jobs(limit)
    print("=" * 74)
    print("AEROVIGIL - MODULE 10 SIGNALS (Kalman NIS and velocity consistency)")
    print("=" * 74)
    print(f"Flights {len(rows)}   filter rate {ctx.rate_hz:g} Hz   cache {sig_dir}")
    for stale in sig_dir.glob("*.partial.npz"):
        stale.unlink()  # unfinished writes from an interrupted run
    started, index, rebuilt, stale = time.time(), [], 0, 0
    for n, r in enumerate(rows, start=1):
        path = r.pop("path")
        target = sig_dir / f"{r['flight_id']}.npz"
        sig = None
        if not force and target.is_file():
            try:
                sig = load(target, source_stamp(path) if path.is_file() else None)
            except StaleCache:
                stale += 1  # written before the last change to what is cached
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                rebuilt += 1  # left half-written by an interrupted run: rebuild it
        if sig is None:
            if not path.is_file():
                raise SystemExit(
                    f"Missing {path}. Build the Module 03/04 v2 base first."
                )
            sig = flight_signals(pd.read_parquet(path), ctx)
            sig["source"] = source_stamp(path)
            save(target, sig)
        r["kf_hours"] = monitored_seconds(sig["kf_t"], sig["kf_valid"]) / 3600.0
        r["vel_hours"] = monitored_seconds(sig["row_t"], sig["vel_valid"]) / 3600.0
        r["file"] = target.name
        index.append(r)
        if n % 50 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}  ({time.time() - started:.0f} s)")
    idx = pd.DataFrame(index)
    idx.to_csv(out / "signals_index.csv", index=False)
    normal = idx[~idx["attacked"]]
    print(
        f"\nNormal flights {len(normal)}: {normal['kf_hours'].sum():.2f} h monitored by "
        f"the filter, {normal['vel_hours'].sum():.2f} h by the velocity channel"
    )
    if stale:
        print(
            f"Recomputed {stale} cache file(s) from an older version or a "
            "regenerated source flight"
        )
    if rebuilt:
        print(
            f"Rebuilt {rebuilt} cache file(s) that an interrupted run had left broken"
        )
    print(f"Saved {len(idx)} flights in {time.time() - started:.0f} s")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true", help="recompute cached flights")
    a = ap.parse_args()
    run(a.limit, a.output_dir, a.force)


if __name__ == "__main__":
    main()
