r"""Step 3: check everything the Module 07 pipeline depends on, and estimate the run time."""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from aerovigil.attacks.common import (
    UNIT_TO_SECONDS,
    load_split_map,  # noqa: E402
    load_yaml,
    resolve,
)
from aerovigil.lstm.build_sequences import windows_of  # noqa: E402
from aerovigil.lstm.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lstm" / "lstm.yaml"))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    pre = load_yaml(resolve(cfg["paths"]["preprocessing_config"]))
    win_s = float(pre["processing"]["window_duration_seconds"])
    hop_s = win_s * (1.0 - float(pre["processing"]["window_overlap"]))
    scale = UNIT_TO_SECONDS[pre["dataset"]["timestamp_unit"]]
    wcfg, problems = cfg["windows"], []
    print("=" * 65)
    print("AEROVIGIL - MODULE 07 INPUT CHECK")
    print("=" * 65)
    versions = {"torch": torch.__version__, "scikit-learn": sklearn.__version__}
    for pkg in ("optuna", "mlflow"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except ImportError:
            versions[pkg] = "NOT INSTALLED"
            problems.append(f"{pkg} missing")
    print("Packages     :", versions)
    print(
        f"CPU threads  : {os.cpu_count()} (torch will use {cfg['training']['threads']}) | "
        f"CUDA available: {torch.cuda.is_available()}"
    )
    T, step = win_s / wcfg["bin_s"], hop_s / wcfg["bin_s"]
    print(
        f"Window {win_s:g} s, hop {hop_s:g} s, bin {wcfg['bin_s']:g} s -> {T:g} steps per window, "
        f"{step:g} bins per hop"
    )
    if abs(T - round(T)) > 1e-9 or abs(step - round(step)) > 1e-9:
        problems.append("bin_s does not divide the window and the hop")

    report = pd.read_csv(resolve(cfg["paths"]["cleaning_report"]))
    split_map, _, _ = load_split_map(resolve(cfg["paths"]["flight_splits"]))
    feat_dir = resolve(cfg["paths"]["features_dir"])
    missing = [f for f in report["file"] if not (feat_dir / f).exists()]
    print(
        f"\nFlights in cleaning report : {len(report)} | with a split: "
        f"{int(report['flight_id'].astype(str).isin(split_map).sum())}"
    )
    print(f"Feature files missing      : {len(missing)}")
    if missing:
        problems.append("feature files missing")
    man = pd.read_csv(resolve(cfg["paths"]["attack_manifest"]))
    man = man[man["attack_set"].isin(wcfg["synthetic_sets"])]
    man = man[(man["attack_set"] != "sweep") | man["split"].isin(wcfg["sweep_splits"])]
    miss_att = [
        x
        for x in man["file"]
        if not (resolve(cfg["paths"]["attacked_dir"]) / x).exists()
    ]
    print(
        f"Synthetic flights to window: {len(man)} ({man['attack_set'].value_counts().to_dict()}) | "
        f"files missing {len(miss_att)}"
    )
    if miss_att:
        problems.append("attacked files missing")

    sample = report.sample(min(5, len(report)), random_state=0)
    t0, n_win, names = time.time(), 0, None
    for _, r in sample.iterrows():
        f = pd.read_parquet(feat_dir / r["file"])
        X, m, _, names, _ = windows_of(f, win_s, hop_s, scale, wcfg)
        n_win += len(m)
    per_orig = (time.time() - t0) / len(sample)
    print(
        f"\nFeature columns: {len(names)} (expected {wcfg['expected_feature_count']}) | "
        f"'after_gap' present: {'after_gap' in f.columns}"
    )
    if len(names) != wcfg["expected_feature_count"] or "after_gap" not in f.columns:
        problems.append("feature layout")
    print(
        f"Windows per original flight (5 sampled): {n_win / len(sample):.0f} -> about "
        f"{n_win / len(sample) * len(report):,.0f} original windows | {per_orig:.2f} s per flight"
    )

    from aerovigil.features.build_features import build_flight

    gps = pre["sensors"]["gps"]
    c = {
        "flight": pre["dataset"]["flight_id_column"],
        "time": pre["dataset"]["timestamp_column"],
        "label": pre["labels"]["attack_column"],
        "lat": gps["latitude"],
        "lon": gps["longitude"],
        "alt": gps["altitude"],
        "vn": gps["velocity_north"],
        "ve": gps["velocity_east"],
        "vu": gps["velocity_up"],
        "baro": pre["sensors"]["barometer"]["altitude"],
    }
    t0 = time.time()
    for _, m_ in man.head(3).iterrows():
        g = pd.read_parquet(resolve(cfg["paths"]["attacked_dir"]) / m_["file"])
        windows_of(build_flight(g, c, scale), win_s, hop_s, scale, wcfg)
    per_att = (time.time() - t0) / 3
    print(
        f"Feature builder + windows on one attacked flight: {per_att:.2f} s -> about "
        f"{per_att * len(man) / 60:.0f} min for {len(man)} flights"
    )

    torch.set_num_threads(int(cfg["training"]["threads"]))
    x = torch.randn(3000, int(T), len(names))
    for kind in ("ae", "clf"):
        model = build_model(kind, len(names), {**cfg["ae_params"], **cfg["clf_params"]})
        opt = torch.optim.Adam(model.parameters())
        t0 = time.time()
        for i in range(0, 3000, cfg["training"]["batch_size"]):
            b = x[i : i + cfg["training"]["batch_size"]]
            opt.zero_grad()
            out = model(b)
            loss = ((out - b) ** 2).mean() if kind == "ae" else out.mean()
            loss.backward()
            opt.step()
        print(
            f"Training speed ({kind}): {time.time() - t0:.2f} s per epoch of 3,000 windows "
            f"(starting-size model)"
        )
    print(
        "\nRESULT:",
        "ready to build the sequences" if not problems else f"STOP - {problems}",
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
