"""Step K: one look for every final figure.

Every detector keeps one colour, marker and line style in every figure. The eight
detectors take the eight slots of the validated categorical palette in its fixed order
(adjacent pairs clear the colour-blind checks); the five classical models that are not
detectors of the final comparison are drawn in greys with their own markers and line
styles, and every figure also names each series (legend or direct label), so colour is
never the only cue.

Figures are written to reports/figures/final as PNG (200 dpi) and SVG, and each call to
save() records the caption, data files and script for FIGURES.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "figures" / "final"
DPI = 200

INK = "#191919"
INK2 = "#666666"
GRID = "#e6e6e6"
SURFACE = "#ffffff"
NEUTRAL = "#9c9c9c"

# name: (label, colour, marker, line style)
MODELS = {
    "kalman": ("Kalman chi-square", "#257de0", "o", "-"),
    "velocity": ("Velocity check", "#f47230", "s", "-"),
    "fused": ("Fused (Kalman + velocity)", "#00b894", "D", "-"),
    "cusum": ("Kalman CUSUM", "#f2a600", "^", "--"),
    "consistency": ("Consistency", "#ee7fac", "v", "--"),
    "random_forest": ("Random forest", "#008a17", "P", "-"),
    "lstm_ae": ("LSTM autoencoder", "#6a4ce2", "X", "-."),
    "lstm_supervised": ("Supervised LSTM", "#ec4647", "*", "-."),
    "logistic_regression": ("Logistic regression", "#3d3d3d", "o", ":"),
    "naive_bayes": ("Naive Bayes", "#7a7a7a", "s", ":"),
    "knn": ("k-nearest neighbours", "#3d3d3d", "^", (0, (5, 2, 1, 2))),
    "decision_tree": ("Decision tree", "#a3a3a3", "D", ":"),
    "isolation_forest": ("Isolation forest", "#7a7a7a", "v", (0, (5, 2, 1, 2))),
}
PHYSICS = ["kalman", "velocity", "fused", "cusum", "consistency"]
LEARNED = ["random_forest", "lstm_ae", "lstm_supervised"]
DETECTORS = PHYSICS + LEARNED
CLASSICAL = ["logistic_regression", "naive_bayes", "knn", "decision_tree",
             "random_forest", "isolation_forest"]  # fmt: skip

# attack families and velocity modes (Module 04)
FAMILIES = {
    "abrupt_jump": "Abrupt jump",
    "constant_offset": "Constant offset",
    "linear_drift": "Linear drift",
    "exponential_ramp": "Exponential ramp",
    "low_amplitude_wander": "Low-amplitude wander",
}
MODES = {"position_only": "Position only", "consistent": "Velocity-consistent"}
MODE_STYLE = {"position_only": ("#257de0", "-"), "consistent": ("#f47230", "--")}

# phases of an attacked flight (Module 03 v2 codes 0-4)
PHASES = {0: ("normal", None), 1: ("pre-attack", "#efefef"),
          2: ("ramp", "#fde3a7"), 3: ("plateau", "#f9c0c0"),
          4: ("post-attack", "#cce3fd")}  # fmt: skip

REGISTRY = []


def label(name):
    return MODELS.get(name, (name.replace("_", " "),))[0]


def colour(name):
    return MODELS.get(name, (None, NEUTRAL))[1]


def line_kw(name, **kw):
    lab, col, mk, ls = MODELS[name]
    out = {"color": col, "marker": mk, "linestyle": ls, "label": lab,
           "linewidth": 2, "markersize": 6}  # fmt: skip
    out.update(kw)
    return out


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "svg.fonttype": "none",
    })  # fmt: skip


def save(fig, name, caption, data, script):
    """Write name.png (200 dpi) and name.svg and record the figure for FIGURES.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    place_note(fig)
    rasterize_dense(fig)
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    data = [data] if isinstance(data, str) else list(data)
    REGISTRY.append({"file": f"{name}.png", "caption": caption, "data": data,
                     "script": script})  # fmt: skip
    print(f"  {name}.png", flush=True)
    return OUT / f"{name}.png"


def note(fig, text):
    """A small source / method line under the figure (placed by save())."""
    fig._av_note = fig.text(0.01, 0.0, text, ha="left", va="top", fontsize=8,
                            color=INK2, wrap=True)  # fmt: skip


def place_note(fig):
    """Put the note just below everything else drawn on the figure."""
    t = getattr(fig, "_av_note", None)
    if t is None:
        return
    t.set_visible(False)
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r)
    t.set_visible(True)
    y = box.y0 / fig.get_size_inches()[1]  # tight bbox is in inches
    t.set_position((0.01, min(0.0, y) - 0.012))


def cell_label(family, mode):
    """Short three-line column label for a family / velocity-mode heatmap."""
    name = FAMILIES[family].replace("Low-amplitude", "Low-ampl.").replace(" ", "\n", 1)
    return f"{name}\n{'pos. only' if mode == 'position_only' else 'consistent'}"


def rasterize_dense(fig, points=3000):
    """Draw layers with thousands of points as an image inside the SVG (text, axes and
    light layers stay vector), so dense scatters and long time series stay small."""
    for ax in fig.axes:
        for c in ax.collections:
            if len(c.get_offsets()) > points:
                c.set_rasterized(True)
        for ln in ax.get_lines():
            if len(ln.get_xdata()) > 10 * points:
                ln.set_rasterized(True)
