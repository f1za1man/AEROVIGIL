"""Step 8: cross-check our Kalman filter against FilterPy on real flights.

Both filters get the same matrices, inputs and measurements for the East axis.
Their NIS and position estimates must agree to numerical precision.

    pip install filterpy
    python src\\aerovigil\\kalman\\crosscheck_filterpy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.kalman.common import (
    UNIT_TO_SECONDS,
    get_columns,
    load_convention,  # noqa: E402
    load_noise_scale,
    load_yaml,
    resolve,
)
from aerovigil.kalman.kalman import (
    H_HORIZONTAL,
    KalmanFilter,  # noqa: E402
    horizontal_matrices,
    noise_matrices,
)
from aerovigil.kalman.pipeline import prepare_ticks  # noqa: E402


def main():
    try:
        from filterpy.kalman import KalmanFilter as FPKalman
    except ImportError:
        raise SystemExit("FilterPy is not installed. Run:  pip install filterpy")

    cfg = load_yaml(ROOT / "configs" / "kalman" / "kalman.yaml")
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    cols = get_columns(cfg, pre)
    scale_t = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    fcfg, conv, noise = cfg["filter"], load_convention(cfg), load_noise_scale(cfg)
    r_h, _ = noise_matrices(fcfg, noise)
    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    files = report.loc[report["attack_rows"] == 0, "file"].sample(3, random_state=0)

    print("=" * 65)
    print("AEROVIGIL - KALMAN FILTER vs FILTERPY")
    print("=" * 65)
    ok_all = True
    for f in files:
        g = pd.read_parquet(resolve(cfg["paths"]["cleaned_dir"]) / f)
        seg = prepare_ticks(g, cols, fcfg, conv, scale_t)[0]
        i = seg["init"]
        p0 = np.diag([fcfg["p0_pos"] ** 2, fcfg["p0_vel"] ** 2, fcfg["p0_bias"] ** 2])
        ours = KalmanFilter([[i["e"]], [i["ve"]], [0.0]], p0)
        fp = FPKalman(dim_x=3, dim_z=2)
        fp.x = np.array([[i["e"]], [i["ve"]], [0.0]])
        fp.P = p0.copy()
        fp.H = H_HORIZONTAL.copy()
        fp.R = r_h.copy()
        nis_a, nis_b, pos_a, pos_b = [], [], [], []
        for k in range(len(seg["t"])):
            fm, bm, qm = horizontal_matrices(
                seg["dt"][k], fcfg["sigma_acc"], fcfg["sigma_bias_walk"]
            )
            ours.predict(fm, qm, bm, [[seg["acc"][k, 0]]])
            _, _, _, nis = ours.update(
                [[seg["e"][k]], [seg["ve"][k]]], H_HORIZONTAL, r_h
            )
            fp.predict(u=np.array([[seg["acc"][k, 0]]]), B=bm, F=fm, Q=qm)
            fp.update(np.array([[seg["e"][k]], [seg["ve"][k]]]))
            y = fp.y.reshape(-1, 1)
            nis_a.append(nis[0])
            nis_b.append(float((y.T @ np.linalg.inv(fp.S) @ y)[0, 0]))
            pos_a.append(ours.x[0, 0])
            pos_b.append(float(fp.x[0, 0]))
        d_nis = float(np.max(np.abs(np.array(nis_a) - nis_b)))
        d_pos = float(np.max(np.abs(np.array(pos_a) - pos_b)))
        ok = d_nis < 1e-6 and d_pos < 1e-6
        ok_all &= ok
        print(
            f"  {f[:40]:40s} ticks {len(nis_a):5d} | max |NIS diff| {d_nis:.2e} | "
            f"max |position diff| {d_pos:.2e} | {'PASS' if ok else 'FAIL'}"
        )
    print(
        "\nRESULT:",
        "own filter matches FilterPy" if ok_all else "MISMATCH - send me this output",
    )
    print("=" * 65)
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
