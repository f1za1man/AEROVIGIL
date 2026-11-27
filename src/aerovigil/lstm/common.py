"""Shared helpers for the Module 07 scripts."""

import json
from pathlib import Path

import torch

from aerovigil.attacks.common import ROOT, load_yaml, resolve  # noqa: F401
from aerovigil.lstm.datasets import load_windows
from aerovigil.lstm.sequences import apply_scaler


def out_dir(cfg, output_dir=None):
    p = Path(output_dir) if output_dir else resolve(cfg["paths"]["output_dir"])
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_scaled(out):
    """(scaled X, raw X, meta, scaler dict)."""
    raw, meta = load_windows(out)
    scaler = json.loads((out / "scaler.json").read_text())
    return apply_scaler(raw, scaler), raw, meta, scaler


def model_params(cfg, out, kind):
    """Optuna result if it exists, otherwise the starting values in the config."""
    base = dict(cfg["ae_params"] if kind == "ae" else cfg["clf_params"])
    best = out / "best_params.json"
    if best.exists():
        base.update(json.loads(best.read_text()).get(kind, {}).get("params", {}))
    t = cfg["training"]
    base.update(
        batch_size=t["batch_size"],
        weight_decay=t["weight_decay"],
        grad_clip=t["grad_clip"],
    )
    return base


def set_threads(cfg):
    torch.set_num_threads(int(cfg["training"]["threads"]))


from aerovigil.tracking import Tracker  # noqa: E402,F401  the shared MLflow store
