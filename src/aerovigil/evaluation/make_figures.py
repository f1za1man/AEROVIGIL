r"""Step K: rebuild every final figure from saved results, then write FIGURES.md.

Each group lives in src/aerovigil/figures/ (one file per plan module) and reads only
saved results (and, for a few example plots, the processed flight tables). Output:
reports/figures/final/<plan module>_<name>.png (200 dpi) and .svg, plus
reports/figures/final/FIGURES.md with one line per figure: file, caption, data, script.

Usage:
    python src\aerovigil\evaluation\make_figures.py            every figure
    python src\aerovigil\evaluation\make_figures.py --only M5  one group (FIGURES.md
                                                              keeps the other lines)
"""

import argparse
import importlib
import re
import time

from aerovigil.figures import style as S

# (prefix, module) in report order
GROUPS = [
    ("M2", "data_figs"),
    ("M4", "attack_figs"),
    ("M5", "classical_figs"),
    ("M6", "kalman_figs"),
    ("M7", "lstm_figs"),
    ("M8", "stealth_figs"),
    ("R09", "consistency_figs"),
    ("R10", "fusion_figs"),
    ("M9", "evaluation_figs"),
    ("M10", "explain_figs"),
    ("M11", "edge_figs"),
    ("M12", "fallback_figs"),
    ("M13", "demo_figs"),
]
INDEX = S.OUT / "FIGURES.md"
HEADER = """# Final figures

Every figure of the project, rebuilt from the final results by one command:

    python src\\aerovigil\\evaluation\\make_figures.py

PNG at 200 dpi with an SVG copy of each. Names start with the plan module (M2-M13;
R09 / R10 = repo Modules 09 and 10, the consistency and fused detectors). Each
detector keeps one colour in every figure.

| Figure | What it shows | Data | Script |
|---|---|---|---|
"""


EARLIER = """
## Earlier figures and what replaces them

The module figures made during development (reports/figures/module03-module07) came
from the first version of the data and attacks (before the Module 03 v2 rebuild and the
removal of the velocity overshoot), so they were removed from the repository; Git
history keeps them. Nothing they showed is lost:

| Earlier figure | Replaced by |
|---|---|
| module03 feature_correlation, missing_values | M3_feature_correlation, M3_missing_values |
| module03 pca_train / pca_validation / pca_test | M3_pca_windows |
| module03 sampling_intervals | M2_logging_rate |
| module04 M4_fig1_offset_profiles | M4_attack_families |
| module04 M4_fig2_true_vs_spoofed_path | M3_example_flight_phases (path panel) |
| module04 M4_fig3_parameter_coverage | M4_parameter_coverage |
| module04 M4_fig4_sweep | M4_drift_sweep |
| module05 M5_fig1_f1_seen_vs_unseen | M5_metrics_seen_unseen, M9_protocols |
| module05 M5_fig2_roc_seen | M5_roc_curves |
| module05 M5_fig3_detection_by_attack | M5_detection_by_attack, M9_detection_by_attack |
| module05 M5_fig4_rf_importance | M5_rf_importance |
| module06 M6_fig1_innovation_plot | M6_innovation_nis |
| module06 M6_fig2_drift_absorption | M6_drift_absorption |
| module06 M6_fig3_detection_by_attack | M9_detection_by_attack |
| module06 M6_fig4_stealth_frontier (old overshoot artefact) | M8_oracle_selftest, M8_stealth_frontier |
| module07 M7_fig1_window_auc | M7_window_auc |
| module07 M7_fig2_ae_timeline | M7_reconstruction_error |
| module07 M7_fig3_detection_by_attack | M9_detection_by_attack |
| module07 M7_fig4_stealth_preview | M8_stealth_frontier |

The step figures in reports/figures/stealth (D1-D3), explain (E1-E2) and mitigation
(G1-G2) are still drawn by their own steps; M8_stealth_frontier, M8_detection_by_attacker,
M8_hardening, M10_* and M12_* are their final versions here.
"""


def row(r):
    data = "<br>".join(f"`{d}`" for d in r["data"])
    return f"| [{r['file']}]({r['file']}) | {r['caption']} | {data} | `{r['script']}` |"


def order_key(line):
    m = re.match(r"\| \[(\w+?)_", line)
    prefix = m.group(1) if m else ""
    names = [g[0] for g in GROUPS]
    head = "M3" if prefix == "M3" else prefix
    idx = (
        names.index("M2")
        if head in ("M2", "M3")
        else (names.index(head) if head in names else len(names))
    )
    return (idx, line)


def write_index(only):
    old = []
    if only and INDEX.is_file():
        keep = tuple(f"| [{p}_" for p in only)
        old = [ln for ln in INDEX.read_text(encoding="utf-8").splitlines()
               if ln.startswith("| [") and not ln.startswith(keep)]  # fmt: skip
    lines = sorted(old + [row(r) for r in S.REGISTRY], key=order_key)
    INDEX.write_text(HEADER + "\n".join(lines) + "\n" + EARLIER, encoding="utf-8")
    print(f"\n{len(lines)} figures listed in {INDEX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="group prefixes, e.g. M5 R10 (M2 also makes M3)")  # fmt: skip
    a = ap.parse_args()
    S.setup()
    started = time.time()
    only = set(a.only) if a.only else None
    for prefix, mod in GROUPS:
        if only and prefix not in only:
            continue
        print(f"{prefix}: {mod}  [{time.time() - started:.0f} s]", flush=True)
        importlib.import_module(f"aerovigil.figures.{mod}").make()
    covered = set()
    for p in only or []:
        covered |= {p, "M3"} if p == "M2" else {p}
    write_index(covered if only else None)
    print(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
