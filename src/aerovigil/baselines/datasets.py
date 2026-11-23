"""Assemble leakage-safe train / validation / test sets for each protocol."""

import numpy as np
import pandas as pd


def stat_columns(df):
    return [c for c in df.columns if "__" in c]


def clean_values(df):
    """Replace infinities with NaN in the summary columns (imputed later)."""
    cols = stat_columns(df)
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df


def _masks(held_out):
    if (
        held_out is None
    ):  # 'seen': flight-disjoint splits, every mission type in training
        return (
            lambda d: d["split"] == "train",
            lambda d: d["split"] == "validation",
            lambda d: d["split"] == "test",
        )
    # 'unseen mission': the held-out mission type is absent from training and tuning
    return (
        lambda d: (d["split"] == "train") & (d["mission_type"] != held_out),
        lambda d: (d["split"] == "validation") & (d["mission_type"] != held_out),
        lambda d: d["mission_type"] == held_out,
    )


def balance_classes(df, seed):
    """Down-sample the majority class (random, seeded) so both classes have equal
    counts. Used for training and validation only, never for test sets."""
    pos, neg = df[df["y"] == 1], df[df["y"] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = np.concatenate(
        [
            rng.choice(pos.index.to_numpy(), n, replace=False),
            rng.choice(neg.index.to_numpy(), n, replace=False),
        ]
    )
    return df.loc[np.sort(keep)].reset_index(drop=True)


def assemble(orig, syn, held_out=None, sources=("orig", "syn"), balance_seed=None):
    """Return a dict with train, val, test_orig and test_syn windows.

    Ambiguous windows are removed everywhere. If `balance_seed` is given, training
    and validation are class-balanced by down-sampling. Training and validation use only the
    sources in `sources`; the test side always contains both so that transfer to
    the synthetic attacks can be measured.
    """
    if "orig" not in sources:
        raise ValueError(
            "'orig' must be a training source: only the original flights "
            "contain normal windows."
        )
    tr, va, te = _masks(held_out)
    frames = {"orig": orig[~orig["ambiguous"]], "syn": syn[~syn["ambiguous"]]}
    train = pd.concat([frames[s][tr(frames[s])] for s in sources], ignore_index=True)
    val = pd.concat([frames[s][va(frames[s])] for s in sources], ignore_index=True)
    if balance_seed is not None:
        train, val = balance_classes(train, balance_seed), balance_classes(
            val, balance_seed
        )
    return {
        "train": train,
        "val": val,
        "test_orig": frames["orig"][te(frames["orig"])].reset_index(drop=True),
        "test_syn": frames["syn"][te(frames["syn"])].reset_index(drop=True),
    }


def build_test_sets(test_orig, test_syn):
    """Named evaluation sets. Synthetic attack windows are scored against the same
    pool of genuinely normal windows taken from the original test flights."""
    neg = test_orig[test_orig["y"] == 0]
    pos = test_syn[test_syn["y"] == 1]
    sets = {"orig_test": test_orig, "syn_all": pd.concat([pos, neg], ignore_index=True)}
    for mode in ("consistent", "position_only"):
        sets[f"syn:{mode}"] = pd.concat(
            [pos[pos["velocity_mode"] == mode], neg], ignore_index=True
        )
    for (typ, mode), g in pos.groupby(["attack_type", "velocity_mode"]):
        sets[f"syn:{typ}|{mode}"] = pd.concat([g, neg], ignore_index=True)
    return sets


def protocols(cfg, orig):
    out = []
    if cfg["protocols"]["seen"]:
        out.append(("seen", None))
    if cfg["protocols"]["unseen_mission"]:
        out += [(f"unseen_{m}", m) for m in sorted(orig["mission_type"].unique())]
    return out
