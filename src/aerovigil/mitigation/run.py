r"""Plan M12 (Step G): how far off is the drone after it stops trusting GPS?

Three experiments on the logged flights, all with fallback.py's dead reckoning:

    coast      normal flights: switch GPS off every 10 s along each flight (from 15 s
               after a filter segment starts) and coast for up to 120 s. This is how
               fast dead reckoning drifts with no attack at all; it gives the time the
               estimate stays within 10 m and 25 m of the truth (Kaplan-Meier: a run
               that ends before reaching the limit counts as "at least this long").
    real       the dataset's own attacked flights: switch at the fused detector's
               first alarm (Module 10's rule, Step C's seen calibration, as in Step D)
               and compare the error with and without the fallback until the attack
               ends.
    synthetic  Module 04's test attacks (position-only and velocity-consistent), the
               same way; truth is the source flight's GPS.

Settings chosen on the VALIDATION flights only (never the test ones):
    variant    imu (Module 06's attitude), mag (heading held by the magnetometer) or
               hold (no IMU, last velocity held: a baseline). Lowest median error 30 s
               into the coast runs of the validation normal flights.
    rollback   0, 1, 2, 5 or 10 s. Lowest median (over the validation real attacks)
               of the mean error with the fallback over the first 30 s after the
               alarm.

Writes reports/tables/G_choice.csv     the validation choice
       reports/tables/G_coast.csv      drift on normal test flights, by mission
       reports/tables/G_attacks.csv    with vs without, real and synthetic test attacks
       reports/figures/mitigation/G1_fallback_error.png
       reports/figures/mitigation/G2_within_limits.png
       data/processed/mitigation/{coast,attacks}_runs.csv (per run)
       data/processed/mitigation/curves.npz (error curves on a 0.5 s grid)

PX4 SITL is not used: switching the real autopilot to return-to-launch after an alarm
is left to the owner (see the checklist).

Usage:
    python src\aerovigil\mitigation\run.py            (~25 min)
    python src\aerovigil\mitigation\run.py --figures  (redraw from the saved runs)
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aerovigil.evaluation import physics_protocols as pp  # noqa: E402
from aerovigil.mitigation import fallback as fb  # noqa: E402
from aerovigil.stealth import detectors as D  # noqa: E402
from aerovigil.stealth.figures import GRID, INK, INK2, SERIES, style  # noqa: E402

OUT = D.ROOT / "data" / "processed" / "mitigation"
TABLES = D.ROOT / "reports" / "tables"
FIG = D.ROOT / "reports" / "figures" / "mitigation"
VARIANTS = ("imu", "mag", "hold")
ROLLBACKS = (0.0, 1.0, 2.0, 5.0, 10.0)
LIMITS = (10.0, 25.0)
COAST_EVERY_S, COAST_FIRST_S, COAST_HORIZON_S = 10.0, 15.0, 120.0
HORIZONS = (5, 10, 20, 30, 60, 120)
GRID_S = np.arange(0.0, 120.01, 0.5)
CHOICE_AT_S = 30.0


def on_grid(res, key):
    return np.interp(GRID_S, res["since_s"], res[key], right=np.nan)


def at(res, key, h):
    s = res["since_s"]
    return float(np.interp(h, s, res[key])) if len(s) and s[-1] >= h else np.nan


def run_row(res, **info):
    row = dict(info)
    row["covered_s"] = float(res["since_s"][-1]) if len(res["since_s"]) else 0.0
    for h in HORIZONS:
        row[f"with_{h}s"] = at(res, "err_with", h)
        row[f"without_{h}s"] = at(res, "err_without", h)
        row[f"vwith_{h}s"] = at(res, "verr_with", h)
    for lim in LIMITS:
        t, hit = fb.first_exceed(res["since_s"], res["err_with"], lim)
        row[f"t{lim:g}_s"], row[f"t{lim:g}_hit"] = t, hit
    t, hit = fb.crossover(res["since_s"], res["err_with"], res["err_without"])
    row["better_until_s"], row["better_until_hit"] = t, hit
    first = res["since_s"] <= CHOICE_AT_S
    row["mean_with_30s"] = float(np.mean(res["err_with"][first]))
    row["mean_without_30s"] = float(np.mean(res["err_without"][first]))
    row["max_without_m"] = float(np.max(res["err_without"]))
    return row


# -------------------------------------------------------------- experiments --


def coast_runs(flights, ctx, variants, tag):
    rows, curves = [], {}
    for n, (_, r) in enumerate(flights.iterrows(), start=1):
        ticks = fb.Ticks(pd.read_parquet(r["base_path"]), ctx)
        for si, seg in enumerate(ticks.segs):
            t = seg["t_s"]
            if not len(t):
                continue
            for sw in np.arange(t[0] + COAST_FIRST_S, t[-1] - 1.0, COAST_EVERY_S):
                for v in variants:
                    res = fb.fallback_run(ticks, sw, 0.0, v, COAST_HORIZON_S)
                    if res is None:
                        continue
                    rows.append(run_row(res, set=tag, flight_id=r["flight_id"],
                                        mission_type=r["mission_type"], variant=v,
                                        segment=si, switch_s=float(sw)))  # fmt: skip
                    curves.setdefault((tag, v), []).append(on_grid(res, "err_with"))
        print(f"  coast {tag}: {n}/{len(flights)} flights", flush=True)
    return rows, curves


def attack_flights(split):
    idx = pp.universe()
    real = idx[(idx["set"] == "orig") & (idx["split"] == split) & idx["attacked"]]
    syn = idx[(idx["set"] == "main") & (idx["split"] == split)]
    return real, syn


def attack_runs(flights, det, ctx, kind, combos, sources=None):
    """kind 'real' or 'synthetic'; combos: (variant, rollback_s) pairs to run on each
    detected flight. sources: synthetic id -> source base path."""
    rows, curves = [], {}
    for n, (_, r) in enumerate(flights.iterrows(), start=1):
        g = pd.read_parquet(r["base_path"])
        fl = D.Flight(g, det.ctx, det.norm)
        v = fl.verdict(det.alarm_times(fl))
        info = {"set": kind, "flight_id": r["flight_id"],
                "mission_type": r["mission_type"], "attack_type": r["attack_type"],
                "velocity_mode": r["velocity_mode"],
                "drift_rate_mps": r["drift_rate_mps"], "detected": v["detected"],
                "delay_s": v["delay_s"], "displacement_at_alarm_m": v["displacement_m"],
                "final_offset_m": v["final_offset_m"]}  # fmt: skip
        if not v["detected"]:
            rows.append({**info, "variant": "", "rollback_s": np.nan})
            continue
        truth = s = None
        if sources is not None:
            s = pd.read_parquet(sources[r["flight_id"]])
            if not (s["time_usec"].to_numpy() == g["time_usec"].to_numpy()).all():
                raise SystemExit(f"{r['flight_id']}: rows differ from its source")
            truth = tuple(s[c].to_numpy(float) for c in
                          ("raw_latitude_deg", "raw_longitude_deg", "raw_altitude"))  # fmt: skip
        ticks = fb.Ticks(g, ctx, truth, s)
        horizon = fl.end - v["first_alarm_s"]  # until the spoofing stops
        for var, rb in combos:
            res = fb.fallback_run(ticks, v["first_alarm_s"], rb, var, horizon)
            if res is None:  # the alarm fell outside every filter segment
                rows.append({**info, "variant": var, "rollback_s": rb,
                             "covered_s": 0.0})  # fmt: skip
                continue
            rows.append(run_row(res, **info, variant=var, rollback_s=rb,
                                attack_left_s=float(horizon)))  # fmt: skip
            key = (kind, r["velocity_mode"], var, rb)
            curves.setdefault(key, []).append(
                np.vstack([on_grid(res, "err_with"), on_grid(res, "err_without")])
            )
        if n % 20 == 0 or n == len(flights):
            print(f"  {kind}: {n}/{len(flights)} flights", flush=True)
    return rows, curves


# ------------------------------------------------------------------ summary --


def km_cols(g, prefix):
    out = {}
    for lim in LIMITS:
        t, e = g[f"{prefix}{lim:g}_s"], g[f"{prefix}{lim:g}_hit"].astype(bool)
        out[f"median_s_within_{lim:g}m"] = fb.km_quantile(t, e, 0.5)
        out[f"p10_s_within_{lim:g}m"] = fb.km_quantile(t, e, 0.1)
        out[f"share_left_{lim:g}m"] = float(e.mean())
    return out


def coast_table(runs):
    out = []
    for keys, g in runs.groupby(["set", "variant"], sort=False):
        for mission, gg in [("all", g)] + list(g.groupby("mission_type")):
            row = {"set": keys[0], "variant": keys[1], "mission": mission,
                   "runs": len(gg), "flights": gg["flight_id"].nunique()}  # fmt: skip
            for h in HORIZONS:
                col = gg[f"with_{h}s"].dropna()
                row[f"median_err_{h}s"] = float(col.median()) if len(col) else np.nan
                row[f"p90_err_{h}s"] = (float(col.quantile(0.9)) if len(col)
                                        else np.nan)  # fmt: skip
                row[f"runs_{h}s"] = int(len(col))
            row["median_vert_err_60s"] = float(gg["vwith_60s"].median())
            row.update(km_cols(gg, "t"))
            out.append(row)
    return pd.DataFrame(out)


def attack_table(runs):
    out = []
    groups = [("real", "all", runs["set"] == "real")]
    for mode in ("position_only", "consistent"):
        groups.append(("synthetic", mode, (runs["set"] == "synthetic")
                       & (runs["velocity_mode"] == mode)))  # fmt: skip
    for kind, mode, m in groups:
        sub = runs[m]
        base = sub.drop_duplicates("flight_id")
        det = base[base["detected"].astype(bool)]
        ran = sub[(sub["variant"] != "") & sub["better_until_s"].notna()]
        for (var, rb), g in ran.groupby(["variant",
                                                               "rollback_s"]):  # fmt: skip
            row = {"set": kind, "velocity_mode": mode, "variant": var,
                   "rollback_s": rb, "flights": len(base),
                   "detected": len(det), "median_delay_s": det["delay_s"].median(),
                   "median_displacement_at_alarm_m":
                       det["displacement_at_alarm_m"].median(),
                   "median_attack_left_s": g["attack_left_s"].median()}  # fmt: skip
            for h in HORIZONS:
                w, wo = g[f"with_{h}s"].dropna(), g[f"without_{h}s"].dropna()
                row[f"with_{h}s"] = float(w.median()) if len(w) else np.nan
                row[f"without_{h}s"] = float(wo.median()) if len(wo) else np.nan
                row[f"runs_{h}s"] = int(len(w))
            row["mean_with_30s"] = g["mean_with_30s"].median()
            row["mean_without_30s"] = g["mean_without_30s"].median()
            row["median_better_until_s"] = fb.km_quantile(
                g["better_until_s"], g["better_until_hit"].astype(bool), 0.5
            )
            row["share_better_whole_attack"] = float(
                (~g["better_until_hit"].astype(bool)).mean()
            )
            row["median_max_without_m"] = g["max_without_m"].median()
            row.update(km_cols(g, "t"))
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures --


def band(ax, curves, color, label):
    c = np.asarray(curves, float)
    n = np.sum(np.isfinite(c), axis=0)
    ok = n >= max(3, 0.2 * len(c))  # draw only where enough runs remain
    with np.errstate(all="ignore"):
        med = np.nanmedian(c, axis=0)
        lo, hi = np.nanpercentile(c, 25, axis=0), np.nanpercentile(c, 75, axis=0)
    ax.fill_between(GRID_S[ok], lo[ok], hi[ok], color=color, alpha=0.18, lw=0)
    ax.plot(GRID_S[ok], med[ok], color=color, lw=2, label=label)


def figures(curves, choice):
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    var, rb = choice["variant"], choice["rollback_s"]
    panels = [("real", "na", "Real attacks (dataset)"),
              ("synthetic", "position_only", "Synthetic, position only"),
              ("synthetic", "consistent", "Synthetic, velocity-consistent")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (kind, mode, title) in zip(axes, panels):
        c = curves.get(f"{kind}|{mode}|{var}|{rb:g}")
        if c is None:
            ax.set_visible(False)
            continue
        band(ax, c[:, 1], SERIES[1], "without fallback (trusts spoofed GPS)")
        band(ax, c[:, 0], SERIES[0], "with fallback (dead reckoning)")
        for lim in LIMITS:
            ax.axhline(lim, color=INK2, lw=0.8, ls="--")
            ax.text(119, lim * 1.08, f"{lim:g} m", ha="right", color=INK2, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 1000)
        ax.set_xlim(0, 120)
        ax.set_title(f"{title} (n = {len(c)})", loc="left")
        ax.set_xlabel("seconds since the alarm")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_yticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    axes[0].set_ylabel("horizontal position error (m)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Position error after a spoofing alarm: median and middle half of "
                 f"runs (fused detector, {var}, roll back {rb:g} s)",
                 x=0.01, ha="left", color=INK, fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "G1_fallback_error.png", dpi=160)
    plt.close(fig)

    runs = pd.read_csv(OUT / "coast_runs.csv")
    runs = runs[(runs["set"] == "test") & (runs["variant"] == var)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    grid = np.arange(0, 120.5, 0.5)
    for ax, mission in zip(axes, ["all", "curved", "random", "straight"]):
        g = runs if mission == "all" else runs[runs["mission_type"] == mission]
        for lim, color in zip(LIMITS, SERIES):
            s = fb.kaplan_meier(g[f"t{lim:g}_s"], g[f"t{lim:g}_hit"].astype(bool),
                                grid)  # fmt: skip
            ax.step(grid, 100 * s, where="post", color=color, lw=2,
                    label=f"within {lim:g} m")  # fmt: skip
        ax.set_title(f"{mission} ({len(g)} runs, {g['flight_id'].nunique()} flights)",
                     loc="left")  # fmt: skip
        ax.set_xlabel("seconds without GPS")
        if not g["t10_hit"].astype(bool).any():
            ax.text(60, 50, "flights too short:\nno run drifted 10 m\nbefore it ended",
                    ha="center", va="center", color=INK2, fontsize=8)  # fmt: skip
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 102)
        ax.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel("runs still within the limit (%)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("How long dead reckoning stays close to the truth on normal test "
                 "flights", x=0.01, ha="left", color=INK, fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "G2_within_limits.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------- run --


def choose(coast, real):
    rows = []
    c = coast[coast["set"] == "validation"]
    for v, g in c.groupby("variant"):
        rows.append({"what": "variant", "option": v, "runs": len(g),
                     "criterion": "median error 30 s into the coast (m)",
                     "value": float(g[f"with_{int(CHOICE_AT_S)}s"].median())})  # fmt: skip
    best_v = min((r for r in rows), key=lambda r: r["value"])["option"]
    r = real[(real["variant"] == best_v)]
    for rb, g in r.groupby("rollback_s"):
        rows.append({"what": "rollback_s", "option": rb, "runs": len(g),
                     "criterion": "median of the mean error over the first 30 s "
                                  "after the alarm (m)",
                     "value": float(g["mean_with_30s"].median())})  # fmt: skip
    t = pd.DataFrame(rows)
    rb_rows = t[t["what"] == "rollback_s"]
    best_rb = float(rb_rows.loc[rb_rows["value"].idxmin(), "option"])
    t["chosen"] = [(w == "variant" and o == best_v) or
                   (w == "rollback_s" and float(o) == best_rb)
                   for w, o in zip(t["what"], t["option"])]  # fmt: skip
    return t, {"variant": best_v, "rollback_s": best_rb}


def run(limit=None):
    started = time.time()
    cut = (
        (lambda f: f.groupby("mission_type", group_keys=False).head(limit))
        if limit
        else (lambda f: f)
    )
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("AEROVIGIL - PLAN M12: DEAD-RECKONING FALLBACK AFTER AN ALARM")
    print("=" * 78)
    det = D.build(names=("fused",), seeds=())[("fused", 0)]
    ctx = det.ctx
    idx = pp.universe()
    normal = idx[(idx["set"] == "orig") & ~idx["attacked"]]

    print("Validation: choose the variant and the roll-back ...")
    rows, _ = coast_runs(cut(normal[normal["split"] == "validation"]), ctx, VARIANTS,
                         "validation")  # fmt: skip
    coast_val = pd.DataFrame(rows)
    real_val, _ = attack_flights("validation")
    real_val = cut(real_val)
    rr, _ = attack_runs(real_val, det, ctx, "real",
                        [(v, rb) for v in VARIANTS for rb in ROLLBACKS])  # fmt: skip
    val_real = pd.DataFrame(rr)
    choice_t, choice = choose(coast_val, val_real[val_real["variant"] != ""])
    choice_t.to_csv(TABLES / "G_choice.csv", index=False)
    print(choice_t.round(3).to_string(index=False))
    print(f"  chosen: {choice}  [{time.time() - started:.0f} s]")

    print("Test: coast runs on normal flights ...")
    rows, ccurves = coast_runs(cut(normal[normal["split"] == "test"]), ctx, VARIANTS,
                               "test")  # fmt: skip
    coast = pd.concat([coast_val, pd.DataFrame(rows)], ignore_index=True)
    coast.to_csv(OUT / "coast_runs.csv", index=False)

    print(f"Test: attacks [{time.time() - started:.0f} s] ...")
    real_t, syn_t = attack_flights("test")
    real_t, syn_t = cut(real_t), cut(syn_t)
    man = pd.read_csv(pp.MANIFEST).set_index("attacked_flight_id")["source_flight_id"]
    paths = idx.set_index("flight_id")["base_path"]
    sources = {f: paths[man[f]] for f in syn_t["flight_id"]}
    # every variant at the chosen roll-back, every roll-back at the chosen variant
    combos = [(v, choice["rollback_s"]) for v in VARIANTS] + [
        (choice["variant"], rb) for rb in ROLLBACKS if rb != choice["rollback_s"]
    ]
    rows, curves = [], {}
    for kind, flights, src in (("real", real_t, None), ("synthetic", syn_t, sources)):
        rr, cc = attack_runs(flights, det, ctx, kind, combos, src)
        rows += rr
        curves.update(cc)
    attacks = pd.DataFrame(rows)
    attacks.to_csv(OUT / "attacks_runs.csv", index=False)
    np.savez_compressed(OUT / "curves.npz", **{
        "|".join(str(x if not isinstance(x, float) else f"{x:g}") for x in k):
        np.asarray(v) for k, v in curves.items()})  # fmt: skip

    tables = {"G_coast.csv": coast_table(coast[coast["set"] == "test"]),
              "G_attacks.csv": attack_table(attacks)}  # fmt: skip
    for name, t in tables.items():
        t.to_csv(TABLES / name, index=False)
    (OUT / "choice.txt").write_text(f"{choice['variant']} {choice['rollback_s']:g}\n")
    figures(load_curves(), choice)
    report(tables, choice)
    print(f"\nDone in {(time.time() - started) / 60:.1f} min")


def load_curves():
    z = np.load(OUT / "curves.npz")
    return {k: z[k] for k in z.files}


def report(tables, choice):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(f"\nChosen on validation: {choice}")
    c = tables["G_coast.csv"]
    cols = ["variant", "mission", "runs", "median_err_10s", "median_err_30s",
            "median_err_60s", "p90_err_30s", "median_s_within_10m", "p10_s_within_10m",
            "median_s_within_25m", "p10_s_within_25m", "median_vert_err_60s"]  # fmt: skip
    print("\nCOAST (normal test flights)")
    print(c[cols].round(2).to_string(index=False))
    a = tables["G_attacks.csv"]
    cols = ["set", "velocity_mode", "variant", "rollback_s", "flights", "detected",
            "median_delay_s", "median_displacement_at_alarm_m", "with_10s",
            "without_10s", "with_30s", "without_30s", "with_60s", "without_60s",
            "median_better_until_s", "share_better_whole_attack",
            "median_s_within_10m", "median_s_within_25m"]  # fmt: skip
    print("\nATTACKS (test)")
    print(a[cols].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: this many flights per mission and set",
    )
    args = ap.parse_args()
    if args.figures:
        v, rb = (OUT / "choice.txt").read_text().split()
        figures(load_curves(), {"variant": v, "rollback_s": float(rb)})
        return
    run(args.limit)


if __name__ == "__main__":
    main()
