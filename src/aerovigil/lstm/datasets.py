"""Leakage-safe train / validation / test selections over the window table."""

import numpy as np
import pandas as pd

ATTACK_TYPES = [
    "abrupt_jump",
    "constant_offset",
    "linear_drift",
    "exponential_ramp",
    "low_amplitude_wander",
]
HELD_OUT = ATTACK_TYPES + ["original"]


def load_windows(folder):
    """(X float32 (N, T, K) unscaled, meta DataFrame) written by build_sequences.py."""
    z = np.load(folder / "windows.npz")
    return z["X"], pd.read_parquet(folder / "windows_meta.parquet")


def ae_masks(meta):
    """One-class training: windows with no attack, from fully normal ORIGINAL flights only."""
    base = (
        (meta["origin"] == "orig")
        & (meta["flight_kind"] == "normal")
        & (meta["y"] == 0)
    )
    return base & (meta["split"] == "train"), base & (meta["split"] == "validation")


def clf_masks(meta, held_out=None):
    """Supervised training: original windows (normal + real attacks) plus synthetic attack windows.

    held_out removes ONE attack family from training and validation: a synthetic attack type,
    or 'original' for the dataset's own attacks. Test windows are never touched here.
    """
    orig = (meta["origin"] == "orig") & (meta["y"] >= 0) & meta["y"].isin([0, 1])
    syn_pos = (meta["origin"] == "syn") & (meta["y"] == 1)
    use = orig | syn_pos
    if held_out == "original":
        use &= ~((meta["origin"] == "orig") & (meta["y"] == 1))
    elif held_out is not None:
        use &= ~((meta["origin"] == "syn") & (meta["attack_type"] == held_out))
    return use & (meta["split"] == "train"), use & (meta["split"] == "validation")


def eval_pool(meta):
    """Normal windows of the original TEST flights: the shared negative class for every score."""
    return (meta["origin"] == "orig") & (meta["split"] == "test") & (meta["y"] == 0)


def eval_attack_groups(meta):
    """{group name: mask of positive test windows}: real attacks and each synthetic type."""
    g = {
        "original": (meta["origin"] == "orig")
        & (meta["split"] == "test")
        & (meta["y"] == 1)
    }
    syn = (meta["origin"] == "syn") & (meta["split"] == "test") & (meta["y"] == 1)
    for typ in ATTACK_TYPES:
        for mode in ("consistent", "position_only"):
            g[f"{typ}|{mode}"] = (
                syn & (meta["attack_type"] == typ) & (meta["velocity_mode"] == mode)
            )
        g[typ] = syn & (meta["attack_type"] == typ)
    g["synthetic_all"] = syn
    return {k: v for k, v in g.items() if v.any()}
