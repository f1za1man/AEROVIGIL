r"""Plan M10 (Step E): explain every alarm on the test flights, cluster them, plot.

    1. Normals: the 95th percentile of every check on clean training flights.
    2. Every alarm that any detector raises on any test flight (real attacks, the
       Module 04 main set, the drift sweep and the clean flights' false alarms) gets
       its three reasons and an operator message (attribution.py). The run checks
       that every alarm has three reasons.
    3. SHAP, globally: mean |SHAP| per check for the random forest and the decision
       tree, on the test windows (attack and clean).
    4. K-Means on the alarms: each alarm described by how far every check is out of
       its normal range (log of value / normal), at the random forest's first alarm
       inside each attack. Do the clusters line up with the attack families? k from
       the silhouette score, plus k = 6 (five synthetic families and the real
       attacks); agreement by adjusted Rand index.

Writes data/processed/explain/alarms.csv, reports/tables/E_*.csv and
reports/figures/explain/E1-E2.

Usage:
    python src\aerovigil\explain\run.py              (~30 min)
    python src\aerovigil\explain\run.py --limit 3
    python src\aerovigil\explain\run.py --summarise  (tables and figures only)
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aerovigil.evaluation import lstm_protocols as lp  # noqa: E402
from aerovigil.evaluation import store  # noqa: E402
from aerovigil.explain import reasons as R  # noqa: E402
from aerovigil.explain.attribution import Explainer, shap_class1  # noqa: E402
from aerovigil.lstm.models import predict  # noqa: E402
from aerovigil.lstm.train_v2 import load  # noqa: E402
from aerovigil.stealth import detectors as D  # noqa: E402
from aerovigil.stealth import frontier as fr  # noqa: E402

TABLES = D.ROOT / "reports" / "tables"
FIG = D.ROOT / "reports" / "figures" / "explain"
SEED = 42
K_RANGE = range(2, 9)
FAMILY_K = 6
SHAP_WINDOWS = 1500
INFO = ("flight_id", "set", "mission_type", "attack_type", "velocity_mode",
        "drift_rate_mps", "attacked")  # fmt: skip


def ae_clean_error():
    """Mean per-feature reconstruction error of the seen AE on clean calibration
    windows (the scale against which an alarm window's errors are judged)."""
    path = R.OUT / "ae_clean_error.npy"
    if path.is_file():
        return np.load(path)
    x, meta = load(lp.OUT_DIR)
    job = {"kind": "ae", "protocol": "seen", "held": None}
    _, _, cal, _ = lp.masks(meta, job)
    model, _ = D.load_lstm("ae", SEED)
    _, fe = predict("ae", model, x[cal].astype(np.float32))
    err = np.maximum(np.asarray(fe).mean(axis=0), 1e-9)
    R.OUT.mkdir(parents=True, exist_ok=True)
    D.atomic_write(path, lambda p: np.save(p, err))
    return err


def detectors():
    dets = D.build(seeds=(SEED,))
    model, thr, cols = D.fit_classical("decision_tree", SEED)
    dets[("decision_tree", SEED)] = D.RandomForest(model, thr, cols, SEED,
                                                   name="decision_tree")  # fmt: skip
    return dets


def explain_all(limit=None):
    started = time.time()
    normals = R.build_normals()
    ex = Explainer(normals, ae_clean_error())
    dets = detectors()
    flights = fr.test_flights(limit)
    first = next(iter(dets.values()))
    rows = []
    for n, (_, r) in enumerate(flights.iterrows(), start=1):
        fl = first.flight(pd.read_parquet(r["base_path"]))
        for (name, _), det in dets.items():
            times = det.alarm_times(fl)
            for j, t in enumerate(times):
                inside = np.isfinite(fl.start) and fl.start <= t <= fl.end
                row = {**{k: r[k] for k in INFO}, "detector": name,
                       "alarm_index": j, "in_attack": bool(inside),
                       "seconds_after_onset": float(t - fl.onset)
                       if np.isfinite(fl.onset) else np.nan}  # fmt: skip
                row.update(ex.explain(det, fl, t))
                rows.append(row)
        if n % 25 == 0 or n == len(flights):
            print(f"  {n}/{len(flights)} flights, {len(rows)} alarms   "
                  f"[{time.time() - started:.0f} s]", flush=True)  # fmt: skip
    alarms = pd.DataFrame(rows)
    first_in = (
        alarms[alarms["in_attack"]]
        .groupby(["flight_id", "detector"])["alarm_t_s"]
        .transform("min")
    )
    alarms["first_in_attack"] = False
    alarms.loc[first_in.index, "first_in_attack"] = (
        alarms.loc[first_in.index, "alarm_t_s"] == first_in)  # fmt: skip
    D.atomic_write(R.OUT / "alarms.csv", lambda p: alarms.to_csv(p, index=False))
    return alarms, normals


# -------------------------------------------------------------- summaries --


def check_complete(alarms):
    have = alarms[[f"reason{i}" for i in (1, 2, 3)]].notna().all(axis=1)
    msg = alarms["message"].str.contains(r"1\).*2\).*3\)", regex=True)
    ok = bool(have.all() and msg.all())
    print(f"\nEvery alarm has three reasons and a message: {ok} "
          f"({int(have.sum())}/{len(alarms)})")  # fmt: skip
    if not ok:
        raise SystemExit("some alarms lack three reasons")
    return ok


def reason_table(alarms):
    """Share of first in-attack alarms whose top reason is each check."""
    a = alarms[alarms["first_in_attack"]].copy()
    a["family"] = np.where(a["set"] == "orig", "real attack", a["attack_type"])
    a.loc[a["set"] == "sweep", "family"] = "drift sweep"
    t = (a.groupby(["detector", "family"])["reason1"].value_counts(normalize=True)
         .rename("share").reset_index())  # fmt: skip
    counts = a.groupby(["detector", "family"]).size().rename("alarms").reset_index()
    return t.merge(counts, on=["detector", "family"])


def examples(alarms):
    """One operator message per detector and attack kind (the earliest alarm)."""
    a = alarms[alarms["first_in_attack"] | ~alarms["attacked"]].copy()
    a["kind"] = np.where(
        ~a["attacked"],
        "false alarm (clean flight)",
        np.where(
            a["set"] == "orig",
            "real attack",
            a["attack_type"] + " / " + a["velocity_mode"],
        ),
    )
    a = a.sort_values(["detector", "kind", "flight_id", "alarm_t_s"])
    return a.groupby(["detector", "kind"]).head(1)[
        ["detector", "kind", "flight_id", "alarm_t_s", "message"]
    ]


def shap_global():
    """Mean |SHAP| per check for the tree models on test windows."""
    import shap

    from aerovigil.baselines import all_models_v2 as am

    orig, syn = am.load(D.M05_DIR)
    _, _, _, o_test, s_test = am.protocol_sets(orig, syn, None, "seen")
    test = pd.concat([o_test, s_test], ignore_index=True)
    test = test[test["y_event"] >= 0]
    rng = np.random.default_rng(SEED)
    rows = []
    for name in ("random_forest", "decision_tree"):
        model, _, cols = D.fit_classical(name, SEED)
        ex = shap.TreeExplainer(model[-1])
        for label, part in (("attack windows", test[test["y_event"] == 1]),
                            ("clean windows", test[test["y_event"] == 0])):  # fmt: skip
            take = part.iloc[rng.choice(len(part), min(SHAP_WINDOWS, len(part)),
                                        replace=False)]  # fmt: skip
            z = model[:-1].transform(take[cols].to_numpy("float32"))
            sv = shap_class1(ex.shap_values(z, check_additivity=False))
            by = {}
            for c, v in zip(cols, np.abs(sv).mean(axis=0)):
                k = R.family_of(c)
                by[k] = by.get(k, 0.0) + float(v)
            for k, v in by.items():
                rows.append(
                    {
                        "model": name,
                        "windows": label,
                        "check": k,
                        "mean_abs_shap": v,
                        "n_windows": len(take),
                    }
                )
        print(f"  SHAP {name} done", flush=True)
    return pd.DataFrame(rows)


def kmeans(alarms):
    """Cluster the random forest's first in-attack alarms by their check profile."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    a = alarms[alarms["first_in_attack"] & (alarms["detector"] == "random_forest")
               & alarms["set"].isin(["orig", "main"])].copy()  # fmt: skip
    a["family"] = np.where(a["set"] == "orig", "real attack", a["attack_type"])
    cols = [f"ratio_{k}" for k in R.CHECKS]
    x = np.log1p(a[cols].fillna(a[cols].median()).clip(lower=0).to_numpy(float))
    x = StandardScaler().fit_transform(x)
    sil = {}
    if len(a) <= FAMILY_K:
        raise SystemExit(f"only {len(a)} alarms to cluster")
    for k in [k for k in K_RANGE if k < len(a)]:
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(x)
        sil[k] = float(silhouette_score(x, lab))
    best = max(sil, key=sil.get)
    out = {}
    for k in sorted({best, FAMILY_K}):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(x)
        out[k] = lab
    a["cluster"] = out[FAMILY_K]
    a["cluster_best_k"] = out[best]
    summary = pd.DataFrame([{"k": k, "silhouette": sil.get(k, np.nan),
                             "ari_vs_family": float(adjusted_rand_score(
                                 a["family"], lab)), "alarms": len(a)}
                            for k, lab in out.items()])  # fmt: skip
    summary["chosen_by"] = ["silhouette" if k == best else "families" for k in out]
    summary.insert(0, "description", "random forest, check ratios")
    sil_t = pd.DataFrame({"k": list(sil), "silhouette": list(sil.values())})
    return a, pd.concat([summary, alternatives(alarms)], ignore_index=True), sil_t


def alternatives(alarms):
    """The same k = 6 clustering on other descriptions of the alarms, to show the
    result does not hinge on one choice."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    base = alarms[alarms["first_in_attack"] & alarms["set"].isin(["orig", "main"])]
    fam = np.where(base["set"] == "orig", "real attack", base["attack_type"])
    base = base.assign(family=fam)
    ratio = [f"ratio_{k}" for k in R.CHECKS]
    score = [f"score_{k}" for k in R.ROW_CHECKS]
    rows = []
    for det, cols, label in (("random_forest", score, "random forest, SHAP shares"),
                             ("kalman", ratio, "Kalman, check ratios"),
                             ("fused", ratio, "fused, check ratios"),
                             ("lstm_supervised", ratio,
                              "supervised LSTM, check ratios")):  # fmt: skip
        d = base[base["detector"] == det]
        if len(d) <= FAMILY_K:
            continue
        x = d[cols].fillna(0).clip(lower=0).to_numpy(float)
        if cols is score:
            x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
        else:
            x = np.log1p(x)
        x = StandardScaler().fit_transform(x)
        lab = KMeans(FAMILY_K, n_init=20, random_state=SEED).fit_predict(x)
        rows.append({"description": label, "k": FAMILY_K, "silhouette": np.nan,
                     "ari_vs_family": float(adjusted_rand_score(d["family"], lab)),
                     "alarms": len(d), "chosen_by": "families"})  # fmt: skip
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures --

INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
SERIES = ("#257de0", "#f47230")


def fig_shap(g):
    if g is None or not len(g):
        return
    g = g[g["windows"] == "attack windows"]
    order = g[g["model"] == "random_forest"].sort_values("mean_abs_shap")["check"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(order))
    h = 0.38
    for i, (model, label) in enumerate(
        (("random_forest", "Random forest"), ("decision_tree", "Decision tree"))
    ):
        v = g[g["model"] == model].set_index("check").reindex(order)["mean_abs_shap"]
        ax.barh(y + (i - 0.5) * h, v, h * 0.92, color=SERIES[i], label=label)
    ax.set_yticks(y, [R.SHORT[c] for c in order])
    ax.set_xlabel("mean |SHAP| on attack windows (summed over each check's features)")
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Which sensor checks the tree models rely on", loc="left",
                 fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "E1_shap_by_check.png", dpi=160)
    plt.close(fig)


def wrap(name):
    """'low_amplitude_wander' -> one word per line, for axis labels."""
    return name.replace("_", " ").replace(" ", "\n")


def fig_clusters(a, summary):
    if a is None or not len(a):
        return
    tab = pd.crosstab(a["cluster"], a["family"], normalize="columns")
    fams = ["real attack", "abrupt_jump", "constant_offset", "linear_drift",
            "exponential_ramp", "low_amplitude_wander"]  # fmt: skip
    tab = tab.reindex(columns=[f for f in fams if f in tab.columns])
    counts = a["family"].value_counts()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 else INK)  # fmt: skip
    ax.set_xticks(range(tab.shape[1]),
                  [wrap(c) + f"\n(n={counts[c]})" for c in tab.columns],
                  fontsize=8)  # fmt: skip
    ax.set_yticks(range(tab.shape[0]), [f"cluster {c + 1}" for c in tab.index])
    main = summary[summary["description"] == "random forest, check ratios"]
    ari = float(main.loc[main["k"] == FAMILY_K, "ari_vs_family"].iloc[0])
    ax.set_title(f"K-Means (k={FAMILY_K}) on the random forest's first alarms: "
                 f"share of each attack family per cluster (ARI {ari:.2f})", loc="left",
                 fontsize=9.5, fontweight="bold", color=INK)  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("share of the family's alarms", color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "E2_alarm_clusters.png", dpi=160)
    plt.close(fig)


def summarise(alarms=None, with_shap=True):
    alarms = alarms if alarms is not None else store.read_csv(R.OUT / "alarms.csv")
    check_complete(alarms)
    TABLES.mkdir(parents=True, exist_ok=True)
    normals = R.build_normals()
    R.as_frame(normals).to_csv(TABLES / "E_normals.csv", index=False)
    reason_table(alarms).to_csv(TABLES / "E_top_reasons.csv", index=False)
    ex = examples(alarms)
    ex.to_csv(TABLES / "E_example_messages.csv", index=False)
    a, summary, sil = kmeans(alarms)
    summary.to_csv(TABLES / "E_kmeans.csv", index=False)
    sil.to_csv(TABLES / "E_kmeans_silhouette.csv", index=False)
    a[["flight_id", "family", "cluster", "cluster_best_k"]].to_csv(
        R.OUT / "alarm_clusters.csv", index=False
    )
    g = None
    if with_shap:
        g = shap_global()
        g.to_csv(TABLES / "E_shap_global.csv", index=False)
    elif (TABLES / "E_shap_global.csv").is_file():
        g = pd.read_csv(TABLES / "E_shap_global.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})  # fmt: skip
    fig_shap(g)
    fig_clusters(a, summary)
    report(alarms, summary, ex, g)


def report(alarms, summary, ex, g):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 400)
    first = alarms[alarms["first_in_attack"]]
    print("\nTOP REASON of each detector's first alarm inside an attack (share)")
    print(
        first.groupby("detector")["reason1"]
        .value_counts(normalize=True)
        .round(2)
        .rename("share")
        .reset_index()
        .to_string(index=False)
    )
    print("\nK-MEANS on the random forest's alarms")
    print(summary.round(3).to_string(index=False))
    if g is not None:
        print("\nSHAP, mean |SHAP| per check on attack windows")
        print(
            g[g["windows"] == "attack windows"]
            .pivot(index="check", columns="model", values="mean_abs_shap")
            .round(4)
        )
    print("\nEXAMPLE OPERATOR MESSAGES")
    for _, r in ex.head(12).iterrows():
        print(f"- [{r['detector']} / {r['kind']}] {r['message']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise(with_shap=not a.no_shap)
        return
    alarms, _ = explain_all(a.limit)
    summarise(alarms, with_shap=not a.no_shap)


if __name__ == "__main__":
    main()
