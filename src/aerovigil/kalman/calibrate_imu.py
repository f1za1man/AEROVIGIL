"""Step 3: find the IMU axis and gravity convention from the data itself.

Every candidate convention turns the body-frame accelerometer into a world-frame
acceleration. The right one agrees with the acceleration implied by the GPS
velocity (and has no 1 g offset). Nothing is assumed about the dataset.

Usage:
    python src\\aerovigil\\kalman\\calibrate_imu.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    UNIT_TO_SECONDS,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.kalman.imu import all_conventions, conv_name, rotate, world_accel


def get_columns(cfg, pre):
    gps = pre["sensors"]["gps"]
    return {
        "time": pre["dataset"]["timestamp_column"],
        "lat": gps["latitude"],
        "lon": gps["longitude"],
        "alt": gps["altitude"],
        "vn": gps["velocity_north"],
        "ve": gps["velocity_east"],
        "vu": gps["velocity_up"],
        "baro": pre["sensors"]["barometer"]["altitude"],
        "accel": cfg["columns"]["accel"],
        "quat": cfg["columns"]["quat"],
        "mag": cfg["columns"]["mag"],
        "label": pre["labels"]["attack_column"],
    }


def lagged(t, lag):
    j = np.searchsorted(t, t - lag, side="right") - 1
    ok = (
        (j >= 0)
        & ((t - t[np.maximum(j, 0)]) > 0.8 * lag)
        & ((t - t[np.maximum(j, 0)]) < 1.3 * lag)
    )
    return np.maximum(j, 0), ok


def sample_pairs(g, cols, scale, lag):
    """Per-flight GPS-derived and IMU-derived mean acceleration over `lag` seconds."""
    t = (g[cols["time"]].to_numpy("int64") - g[cols["time"]].iloc[0]) * scale
    v = g[[cols["ve"], cols["vn"], cols["vu"]]].to_numpy(float)
    j, ok = lagged(t, lag)
    a_gps = (v - v[j]) / np.maximum(t - t[j], 1e-9)[:, None]
    return t, j, ok, a_gps


def circular_std_deg(angle):
    r = np.hypot(np.cos(angle).mean(), np.sin(angle).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12)))))


def main():
    cfg = load_yaml(ROOT / "configs" / "kalman" / "kalman.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols = get_columns(cfg, pre)
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    icfg = cfg["imu_calibration"]
    out_dir = resolve(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    if "attack_rows" not in report.columns:  # Module 03 v2 report
        report = report.rename(columns={"mission": "mission_type"})
        report["attack_rows"] = report["attacked"].astype(int)
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    pool = report[
        (report["attack_rows"] == 0)
        & (report["flight_id"].astype(str).map(split_map) == "train")
    ]
    pool = pool.sample(min(icfg["n_flights"], len(pool)), random_state=0)

    print("=" * 65)
    print("AEROVIGIL - IMU CONVENTION TEST")
    print("=" * 65)
    print(f"Normal TRAIN flights used: {len(pool)}")

    convs = all_conventions()
    stats = {
        conv_name(c): {
            "sse_h": 0.0,
            "sst_h": 0.0,
            "sse_v": 0.0,
            "sst_v": 0.0,
            "sum_u": 0.0,
            "n": 0,
        }
        for c in convs
    }
    acc_norm, quat_norm, gps_std, held = [], [], [], []
    mean_h, mean_v = [], []
    data = []
    for _, r in pool.iterrows():
        g = pd.read_parquet(resolve(cfg["paths"]["cleaned_dir"]) / r["file"])
        if "aligned_all" in g.columns:
            # Module 03 v2 keeps IMU and barometer dropouts as NaN rows. One NaN in
            # the running sum below makes every later value NaN, every score NaN, and
            # then no convention is accepted, so the rows go first, as in run_flight.
            g = g[g["aligned_all"].to_numpy() == 1].reset_index(drop=True)
        t, j, ok, a_gps = sample_pairs(g, cols, scale, icfg["lag_s"])
        acc = g[cols["accel"]].to_numpy(float)
        q = g[cols["quat"]].to_numpy(float)
        acc_norm.append(float(np.median(np.linalg.norm(acc, axis=1))))
        quat_norm.append(
            (
                float(np.linalg.norm(q, axis=1).min()),
                float(np.linalg.norm(q, axis=1).max()),
            )
        )
        gps_std.append(float(a_gps[ok][:, :2].std()))
        held.append(
            float(
                (
                    (np.diff(g[cols["lat"]].to_numpy(float)) == 0)
                    & (np.diff(g[cols["lon"]].to_numpy(float)) == 0)
                ).mean()
            )
        )
        data.append((g, t, j, ok, a_gps, acc, q))
        mean_h.append(a_gps[ok][:, :2])
        mean_v.append(a_gps[ok][:, 2])
    mh = np.concatenate(mean_h).mean(axis=0)
    mv = np.concatenate(mean_v).mean()

    for g, t, j, ok, a_gps, acc, q in data:
        cs0 = np.arange(len(t))
        for c in convs:
            a_w = world_accel(acc, q, c, cfg["filter"]["gravity"])
            cs = np.vstack([np.zeros((1, 3)), np.cumsum(a_w, axis=0)])
            cnt = np.maximum(cs0 - j, 1)[:, None]
            a_imu = (cs[cs0 + 1] - cs[j + 1]) / cnt
            s = stats[conv_name(c)]
            d = (a_gps - a_imu)[ok]
            ag = a_gps[ok]
            s["sse_h"] += float((d[:, :2] ** 2).sum())
            s["sst_h"] += float(((ag[:, :2] - mh) ** 2).sum())
            s["sse_v"] += float((d[:, 2] ** 2).sum())
            s["sst_v"] += float(((ag[:, 2] - mv) ** 2).sum())
            s["sum_u"] += float(a_imu[ok][:, 2].sum())
            s["n"] += int(ok.sum())

    rows = []
    for c in convs:
        s = stats[conv_name(c)]
        r2h = 1 - s["sse_h"] / max(s["sst_h"], 1e-9)
        r2v = 1 - s["sse_v"] / max(s["sst_v"], 1e-9)
        rows.append(
            {
                "convention": conv_name(c),
                "r2_horizontal": r2h,
                "r2_vertical": max(min(r2v, 1.0), -1.0),
                "mean_vertical_accel": s["sum_u"] / max(s["n"], 1),
                "score": r2h + max(min(r2v, 1.0), -1.0),
            }
        )
    table = (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )

    print(
        f"\nAccelerometer norm (median)     : {np.median(acc_norm):.2f} m/s^2 "
        f"({'includes gravity' if np.median(acc_norm) > 5 else 'gravity removed'})"
    )
    print(
        f"Quaternion norm range           : {min(x[0] for x in quat_norm):.4f} - "
        f"{max(x[1] for x in quat_norm):.4f}"
    )
    print(
        f"GPS-derived acceleration std    : {np.mean(gps_std):.3f} m/s^2 (horizontal, {icfg['lag_s']:g} s)"
    )
    print(
        f"GPS position repeated from previous row: {np.median(held):.1%} of rows "
        f"({'GPS is held between fixes' if np.median(held) > 0.2 else 'GPS changes almost every row'})"
    )
    print("\nCONVENTION SCORES (GPS-derived vs IMU acceleration)")
    print(table.round(3).to_string(index=False))

    best, second = table.iloc[0], table.iloc[1]
    accepted = bool(
        best["r2_horizontal"] >= icfg["min_horizontal_r2"]
        and abs(best["mean_vertical_accel"]) < icfg["max_abs_vertical_mean"]
        and best["score"] - second["score"] >= icfg["min_margin"]
    )
    conv = None
    if accepted:
        rot, grav, frame = best["convention"].split("/")
        conv = {"rotation": rot, "gravity": grav, "frame": frame}

    # Magnetometer: is the world-frame field direction constant (i.e. consistent with the attitude)?
    mag_info = None
    if conv is not None:
        spreads = []
        for g, t, j, ok, a_gps, acc, q in data:
            m_w = rotate(g[cols["mag"]].to_numpy(float), q, conv["rotation"])
            spreads.append(circular_std_deg(np.arctan2(m_w[:, 1], m_w[:, 0])))
        mag_info = float(np.median(spreads))

    print("\nRESULT")
    if conv is None:
        print(
            "  No convention passed the acceptance rule -> the filter will run WITHOUT the IMU"
        )
        print("  (constant-velocity model). Send me this output before continuing.")
    else:
        print(f"  Accepted convention: {conv}")
        print(
            f"  Horizontal R2 {best['r2_horizontal']:.3f}, margin over runner-up "
            f"{best['score'] - second['score']:.3f}"
        )
        print(
            f"  Magnetometer: world-frame horizontal field direction varies by "
            f"{mag_info:.1f} deg (circular std) -> "
            f"{'consistent with the attitude estimate' if mag_info < 15 else 'NOT constant'}"
        )
    target = out_dir / "imu_convention.json"
    previous = (
        json.loads(target.read_text()).get("convention") if target.is_file() else None
    )
    if conv is None and previous is not None and "--force" not in sys.argv:
        print(
            f"\n  NOT SAVED: {target} already holds the accepted convention {previous}."
            "\n  Overwriting it with 'no IMU' would make every later module run its"
            "\n  filter without the accelerometer. Re-run with --force to overwrite."
        )
        print("=" * 65)
        return
    with target.open("w") as f:
        json.dump(
            {
                "convention": conv,
                "accepted": accepted,
                "scores": table.round(4).to_dict(orient="records"),
                "magnetometer_circular_std_deg": mag_info,
                "gps_repeated_row_fraction": float(np.median(held)),
            },
            f,
            indent=2,
        )
    print(f"  Saved: {out_dir / 'imu_convention.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
