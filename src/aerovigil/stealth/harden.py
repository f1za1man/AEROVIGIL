r"""Plan M8 (Step D): adversarial hardening of the learned detectors.

The defender's move in the game: having seen the attacker's best evasive drifts, add
them to the training data, retrain, recalibrate at the same budget (1 false alarm per
flight-hour on clean flights), and see whether the attacker is pushed back.

    1. evasive set   Each normal training and validation flight gets the attacker's
                     winners against this model (adaptive.py, every mean speed and the
                     free search) and Module 08's winners, each at a random bearing.
                     They are injected through Module 08's path and cut into windows
                     and sequences exactly like Module 04's synthetic attacks
                     (attack-overlapping windows only, event labels from the measured
                     phases). Calibration-role and test flights are never used.
    2. retrain       The random forest (Step B's seen setting) and the supervised LSTM
                     (Step C's seen setting), five seeds each, on their usual training
                     data plus the evasive set; thresholds re-fitted on the same clean
                     calibration windows as before.
    3. re-attack     (a) the fixed-drift sweep and the usual test attacks, with false
                     alarms on the clean test flights; (b) a replay of the winners the
                     model was hardened against, on the 21 test flights; (c) a fresh
                     black-box search against the hardened model (seed 42), examined
                     against all five hardened seeds. (c) is the honest test: an
                     attacker who adapts again.

Writes data/processed/stealth/harden/*, data/processed/stealth/models/*_hard_*,
reports/tables/D_hardening.csv

Usage:
    python src\aerovigil\stealth\harden.py --model random_forest
    python src\aerovigil\stealth\harden.py --model lstm_supervised
    python src\aerovigil\stealth\harden.py --model random_forest --step research
    python src\aerovigil\stealth\harden.py --summarise
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from aerovigil.adaptive import search as m08
from aerovigil.adaptive.profile import clip, profile_fn
from aerovigil.baselines.build_window_tables_v2 import flight_windows
from aerovigil.evaluation import lstm_protocols as lp
from aerovigil.tracking import log_run
from aerovigil.lstm.build_sequences_v2 import BASE_REPORT, SPLITS, roles_for
from aerovigil.lstm.models import build_model, fit, predict
from aerovigil.lstm.sequences_v2 import flight_sequences
from aerovigil.lstm.train_v2 import MAX_EPOCHS, PATIENCE, load
from aerovigil.stealth import adaptive as adv
from aerovigil.stealth import detectors as D
from aerovigil.stealth import frontier as fr

HDIR = D.OUT / "harden"
HARD_SEARCH = D.OUT / "adaptive_hard"
TABLE = D.ROOT / "reports" / "tables" / "D_hardening.csv"
MODELS = ("random_forest", "lstm_supervised")
SEED = 42


# ------------------------------------------------------------ evasive set --


def winners(model):
    """The attacks to harden against: this model's winners and Module 08's."""
    out = []
    for source, path in (("own", adv.ADIR / model / "best.json"),
                         ("module08", adv.M08_BEST)):  # fmt: skip
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run adaptive.py --detector {model}.")
        for key, params in json.loads(path.read_text()).items():
            out.append((f"{source}_{key}", params))
    return out


def evasive_set(model, limit=None):
    """Windows (for the forest) and sequences (for the LSTM) of evasive attacks on the
    normal training and validation flights. Cached per model."""
    HDIR.mkdir(parents=True, exist_ok=True)
    wpath, xpath = HDIR / f"windows_{model}.parquet", HDIR / f"X_{model}.npy"
    mpath = HDIR / f"meta_{model}.parquet"
    if not limit and wpath.is_file() and xpath.is_file() and mpath.is_file():
        return pd.read_parquet(wpath), np.load(xpath), pd.read_parquet(mpath)
    started = time.time()
    rep = pd.read_csv(BASE_REPORT)
    splits = pd.read_csv(SPLITS).set_index("flight_id")["split"].to_dict()
    role = roles_for(rep, splits)
    det = D.Detector()
    norm = det.norm
    rng = np.random.default_rng(SEED)
    attacks = winners(model)
    wins, xs, metas, n = [], [], [], 0
    for split in ("train", "validation"):
        for fid, mission, g in m08.normal_flights(split, limit):
            if role.get(fid) == "calib":
                continue  # the LSTM calibrates on these; keep them untouched
            for key, params in attacks:
                p = clip(dict(params, bearing_deg=rng.uniform(0.0, 360.0)))
                fl = det.flight(det.attacker.attack(g, profile_fn(p)))
                info = {
                    "flight_id": f"{fid}__evasive_{key}",
                    "source_flight_id": fid,
                    "origin": "syn",
                    "split": split,
                    "mission_type": mission,
                    "attack_type": "evasive",
                    "velocity_mode": "consistent",
                    "attack_set": "main",
                }
                w, _ = flight_windows(fl.normalized, norm.features, info, True)
                if w is not None and len(w):
                    wins.append(w)
                x, m, _ = flight_sequences(
                    fl.normalized,
                    norm.features,
                    {**info, "role": role.get(fid, split), "drift_rate_mps": np.nan},
                    True,
                )
                if len(x):
                    xs.append(x)
                    metas.append(m)
                n += 1
            if n % 80 == 0:
                print(f"  evasive attacks {n}   [{time.time() - started:.0f} s]",
                      flush=True)  # fmt: skip
    windows = pd.concat(wins, ignore_index=True)
    x, meta = np.concatenate(xs), pd.concat(metas, ignore_index=True)
    if not limit:
        D.atomic_write(wpath, lambda p: windows.to_parquet(p, index=False))
        D.atomic_write(xpath, lambda p: np.save(p, x))
        D.atomic_write(mpath, lambda p: meta.to_parquet(p, index=False))
    print(f"  evasive set for {model}: {n} attacks, {len(windows)} windows, "
          f"{int((windows['y_event'] == 1).sum())} event windows, {len(x)} sequences"
          f"   [{time.time() - started:.0f} s]")  # fmt: skip
    return windows, x, meta


# --------------------------------------------------------------- retrain --


def hardened_rf(seed, windows):
    extra = windows[(windows["split"] == "train") & (windows["y_event"] >= 0)]
    model, thr, cols = D.fit_random_forest(seed, extra=extra, tag="hard")
    return D.RandomForest(model, thr, cols, seed)


def hardened_lstm(seed, x_e, meta_e):
    """Step C's seen supervised LSTM recipe with the evasive sequences added."""
    wpath = D.MODELS / f"clf_hard_s{seed}.pt"
    tpath = D.MODELS / f"clf_hard_s{seed}.json"
    if wpath.is_file() and tpath.is_file():
        thr = json.loads(tpath.read_text())["threshold"]
        model, thr = D.load_lstm("clf", seed, weights=wpath, threshold=thr)
        return D.LSTM("clf", model, thr, seed)
    import torch

    torch.set_num_threads(lp.THREADS)
    x, meta = load(lp.OUT_DIR)
    x = np.concatenate([x, x_e.astype(x.dtype)])
    meta = pd.concat([meta, meta_e[meta.columns.intersection(meta_e.columns)]],
                     ignore_index=True)  # fmt: skip
    job = {"name": f"clf_hard_s{seed}", "kind": "clf", "protocol": "seen",
           "held": None, "hidden": lp.BASE_SIZE[0], "layers": lp.BASE_SIZE[1],
           "seed": seed}  # fmt: skip
    tr, va, cal, _ = lp.masks(meta, job)
    y = meta["y_event"].to_numpy().astype(np.float32)
    params = lp.params_for(job)
    model, hist = fit("clf", build_model("clf", x.shape[-1], params, seed),
                      x[tr], y[tr], x[va], y[va], params, seed, MAX_EPOCHS,
                      PATIENCE)  # fmt: skip
    score, _ = predict("clf", model, x[cal])
    thr, hours, allowed = lp.threshold_for(np.asarray(score))
    D.MODELS.mkdir(parents=True, exist_ok=True)
    D.atomic_write(wpath, lambda p: torch.save(model.state_dict(), p))
    D.atomic_write(tpath, lambda p: p.write_text(json.dumps(
        {"threshold": thr, "calibration_hours": hours, "allowed": allowed,
         "best_epoch": hist["best_epoch"], "n_train": int(tr.sum()),
         "n_evasive_train": int((tr & (meta["attack_type"] == "evasive")
                                 .to_numpy()).sum())}, indent=2)))  # fmt: skip
    log_run("aerovigil-m8-hardening", f"lstm_supervised hardened s{seed}",
            {"model": "lstm_supervised", "seed": seed, **params},
            {"threshold": thr, "calibration_hours": hours,
             "best_epoch": hist["best_epoch"], "best_val_loss": hist["best_val"],
             "n_train": int(tr.sum())}, {"source": "run", "step": "D"})  # fmt: skip
    torch.set_num_threads(1)
    model.eval()
    return D.LSTM("clf", model, thr, seed)


def hardened(model, seeds=D.SEEDS):
    windows, x_e, meta_e = evasive_set(model)
    out = {}
    for seed in seeds:
        det = (hardened_rf(seed, windows) if model == "random_forest"
               else hardened_lstm(seed, x_e, meta_e))  # fmt: skip
        out[(model, seed)] = det
        print(f"  hardened {model} seed {seed}: threshold {det.threshold:.4g}",
              flush=True)  # fmt: skip
    return out


# -------------------------------------------------------------- re-attack --


def reattack(model, step="all"):
    started = time.time()
    print("=" * 78)
    print(f"AEROVIGIL - PLAN M8: HARDENING {model}")
    print("=" * 78)
    dets = hardened(model)
    out = HDIR / model
    out.mkdir(parents=True, exist_ok=True)
    if step in ("all", "fixed"):
        flights = fr.score_all(fr.test_flights(), dets, started)
        D.atomic_write(out / "fixed_flights.csv",
                       lambda p: flights.to_csv(p, index=False))  # fmt: skip
    if step in ("all", "replay"):
        frame = adv.exam((model,), adv.ADIR, dets, tag="after_replay")
        D.atomic_write(out / "replay_exam.csv", lambda p: frame.to_csv(p, index=False))
    if step in ("all", "research"):
        adv.search(model, out_root=HARD_SEARCH, det=dets[(model, SEED)])
        frame = adv.exam((model,), HARD_SEARCH, dets, tag="after_research")
        D.atomic_write(out / "research_exam.csv",
                       lambda p: frame.to_csv(p, index=False))  # fmt: skip
    print(f"\nElapsed {time.time() - started:.0f} s")


# --------------------------------------------------------------- summary --


def summarise():
    """Before vs after, per model: fixed sweep, real attacks, false alarms, adaptive."""
    before_fixed = pd.read_csv(D.OUT / "fixed_flights.csv")
    before_exam = pd.read_csv(D.OUT / "adaptive_exam.csv")
    rows = []
    for model in MODELS:
        d = HDIR / model
        if not (d / "fixed_flights.csv").is_file():
            continue
        after_fixed = pd.read_csv(d / "fixed_flights.csv")
        for tag, f in (("before", before_fixed[before_fixed["detector"] == model]),
                       ("after", after_fixed)):  # fmt: skip
            fx = fr.frontier(f).assign(tag=tag, attack="fixed sweep")
            rows.append(fx.rename(columns={"drift_rate_mps": "mean_rate_mps"}))
            rows.append(headline(f, model, tag))
        exams = [before_exam[before_exam["detector"] == model].assign(tag="before")]
        for name in ("replay_exam.csv", "research_exam.csv"):
            if (d / name).is_file():
                exams.append(pd.read_csv(d / name))
        ad = adv.summarise(pd.concat(exams, ignore_index=True))
        rows.append(ad.assign(attack="adaptive"))
    if not rows:
        raise SystemExit("nothing hardened yet")
    table = pd.concat(rows, ignore_index=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE, index=False)
    pd.set_option("display.width", 220)
    show = table[table["attack"].isin(["fixed sweep", "adaptive"])]
    print(
        show.pivot_table(
            index=["detector", "attack", "mean_rate_mps"], columns="tag", values="D_m"
        )
        .round(1)
        .to_string()
    )
    print(table[table["attack"] == "usual"].round(3).to_string(index=False))
    return table


def headline(flights, model, tag):
    """Real attacks, the Module 04 main set, and clean-flight false alarms."""
    real = flights[(flights["set"] == "orig") & flights["attacked"]]
    main = flights[flights["set"] == "main"]
    cv = main[main["velocity_mode"] == "consistent"]
    fa = fr.false_alarm_rows(flights)
    return pd.DataFrame([{
        "detector": model, "tag": tag, "attack": "usual",
        "real_detected": float(real["detected"].mean()),
        "real_delay_s": float(real.loc[real["detected"].astype(bool),
                                       "delay_s"].median()),
        "main_detected": float(main["detected"].mean()),
        "consistent_detected": float(cv["detected"].mean()),
        **fa.iloc[0].drop("detector").to_dict(),
    }])  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default=None)
    ap.add_argument(
        "--step",
        default="all",
        choices=["all", "evasive", "fixed", "replay", "research"],
    )
    ap.add_argument("--limit", type=int, default=None, help="evasive smoke test only")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.model and a.step == "evasive":
        evasive_set(a.model, a.limit)
    elif a.model:
        reattack(a.model, a.step)
    if a.summarise:
        summarise()


if __name__ == "__main__":
    main()
