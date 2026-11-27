r"""Plan M10 (Step E): which checks drove each alarm, detector by detector.

Each detector gets the attribution that is native to it, then every score is mapped
onto the shared checks of reasons.py:

    random_forest, decision_tree   SHAP values (TreeSHAP, exact for trees) of the
                                   alarm window, summed over each check's features
    lstm_ae                        per-feature reconstruction error of the alarm
                                   window, as excess over the same feature's mean
                                   error on clean calibration windows
    lstm_supervised                occlusion: the drop in the alarm window's logit
                                   when one check's features are set to their median
    kalman                         the horizontal NIS at the alarm tick split into its
                                   position and velocity terms, plus the vertical
                                   filter's NIS
    cusum                          the CUSUM sums per innovation component at the alarm
    velocity                       its one channel (horizontal speed consistency)
    consistency                    each channel's standardised value at the alarm
    fused                          whichever channel fired: Kalman or velocity

The top three reasons are the detector's own checks in order of attribution; if the
detector has fewer than three, the list is completed with the checks that are most out
of their normal range at the alarm ("supporting"). "Still consistent" lists every
check within its normal range.
"""

import numpy as np

from aerovigil.consistency.detector import COURSE, VELOCITY
from aerovigil.explain import reasons as R
from aerovigil.fusion.fuse import kalman_hits, velocity_hits
from aerovigil.lstm.models import predict
from aerovigil.stealth import detectors as D

ROW_WINDOW_S = 5.0  # the row checks are read over the 5 s before the alarm
FILTER_WINDOW_S = 1.0
ROW_Q = 90
CHANNEL_CHECK = {COURSE: "course", VELOCITY: "horizontal_velocity"}
COMPONENT_CHECK = ("kf_position", "kf_velocity", "kf_position", "kf_velocity")
TOP = 3


# ---------------------------------------------------------- measured values --


def values_at(fl, t):
    """Every check's measured value just before the alarm at time t."""
    out = {}
    rt = fl.row_t
    near = (rt > t - ROW_WINDOW_S) & (rt <= t)
    for k in R.ROW_CHECKS:
        v = R.row_measure(k, fl.features)[near]
        v = v[np.isfinite(v)]
        out[k] = float(np.percentile(v, ROW_Q)) if len(v) else np.nan
    df = fl.filter
    if df is not None:
        ft = df["row_t_s"].to_numpy(float)
        ok = (ft > t - FILTER_WINDOW_S) & (ft <= t) & df["valid"].to_numpy(bool)
        for k, v in R.filter_measures(df).items():
            vv = v[ok]
            vv = vv[np.isfinite(vv)]
            out[k] = float(vv.max()) if len(vv) else np.nan
    else:
        out.update({k: np.nan for k in R.FILTER_CHECKS})
    return out


# ------------------------------------------------------------ per detector --


def window_at(frame_end, t):
    """Index of the window that ends at the alarm time."""
    i = int(np.argmin(np.abs(np.asarray(frame_end, float) - t)))
    return i


def shap_class1(sv):
    """SHAP values for the attack class, whatever layout shap returns:
    (rows, features, classes), [class0, class1], or (rows, features)."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def tree_scores(det, fl, t, explainer):
    w = fl.windows
    i = window_at(w["window_end_s"], t)
    x = w[det.cols].iloc[[i]].to_numpy("float32")
    z = det.model[:-1].transform(x)
    sv = shap_class1(explainer.shap_values(z, check_additivity=False)).reshape(-1)
    out = {}
    for col, v in zip(det.cols, sv):
        k = R.family_of(col)
        out[k] = out.get(k, 0.0) + float(v)
    return out


def ae_scores(det, fl, t, clean_err):
    x, m = fl.sequences
    i = window_at(m["window_end_s"], t)
    _, fe = predict("ae", det.model, x[[i]].astype(np.float32))
    excess = np.maximum(np.asarray(fe).reshape(-1) / clean_err - 1.0, 0.0)
    out = {}
    for f, v in zip(det.norm.features, excess):
        k = R.family_of(f)
        out[k] = out.get(k, 0.0) + float(v)
    return out


def occlusion_scores(det, fl, t):
    x, m = fl.sequences
    i = window_at(m["window_end_s"], t)
    xi = x[[i]].astype(np.float32)
    base, _ = predict("clf", det.model, xi)
    feats = det.norm.features
    out = {}
    for k in R.ROW_CHECKS:
        idx = [j for j, f in enumerate(feats) if R.family_of(f) == k]
        xo = xi.copy()
        xo[..., idx] = 0.0  # the normalised median
        s, _ = predict("clf", det.model, xo)
        out[k] = float(np.asarray(base).reshape(-1)[0] - np.asarray(s).reshape(-1)[0])
    return out


def kalman_scores(fl, t):
    df = fl.filter
    if df is None:
        return {}
    i = window_at(df["row_t_s"], t)
    w = df[["w_pos_e", "w_vel_e", "w_pos_n", "w_vel_n"]].iloc[i].to_numpy(float)
    return {"kf_position": float(w[0] ** 2 + w[2] ** 2),
            "kf_velocity": float(w[1] ** 2 + w[3] ** 2),
            "kf_altitude": float(df["nis_v"].iloc[i])}  # fmt: skip


def cusum_sums(z, valid, t, k, h, refractory_s, stop):
    """Module 06's cusum_alarms loop, run to tick `stop`, returning each component's
    larger one-sided sum at that tick, before the alarm resets it."""
    sp, sn = np.zeros(z.shape[1]), np.zeros(z.shape[1])
    skip_until = -np.inf
    for i in range(stop + 1):
        if not valid[i] or t[i] < skip_until:
            sp[:] = 0.0
            sn[:] = 0.0
            continue
        sp = np.maximum(0.0, sp + z[i] - k)
        sn = np.maximum(0.0, sn - z[i] - k)
        if i == stop:
            break
        if max(sp.max(), sn.max()) > h:
            sp[:] = 0.0
            sn[:] = 0.0
            skip_until = t[i] + refractory_s
    return np.maximum(sp, sn)


def cusum_scores(det, fl, t):
    """Each innovation component's CUSUM sum at the alarm tick."""
    inn, cal = fl.innovations, det.cal
    if inn is None:
        return {}
    z = (inn["w"] - cal.mu) / cal.sd
    stop = window_at(inn["row_t"], t)
    sums = cusum_sums(z, inn["valid"], inn["t"], cal.k, cal.h[cal.k],
                      D.REFRACTORY_S, stop)  # fmt: skip
    out = {}
    for c, v in zip(COMPONENT_CHECK, sums):
        out[c] = out.get(c, 0.0) + float(v)
    return out


def channel_scores(det, fl, t):
    ch = fl.channels
    i = window_at(ch["t"], t)
    z = np.abs((ch["values"][i] - det.cal.c_mu) / det.cal.c_sd)
    from aerovigil.consistency.detector import CHANNELS

    return {CHANNEL_CHECK[c]: float(v) for c, v in zip(CHANNELS, z) if np.isfinite(v)}


def fused_scores(det, fl, t):
    th = det.cal.m10
    k = kalman_hits(fl.sig, th["fused_kalman"])
    if len(k) and np.min(np.abs(k - t)) < 1e-6:
        return kalman_scores(fl, t), "Kalman channel"
    v = velocity_hits(fl.sig, th["fused_velocity"], th["mu"], th["sd"])
    if len(v) and np.min(np.abs(v - t)) < 1e-6:
        return {"horizontal_velocity": 1.0}, "velocity channel"
    return kalman_scores(fl, t), "Kalman channel"


# ----------------------------------------------------------------- explain --


def rank(scores, ratio, top=TOP):
    """The three reasons, in this order:
    1. the detector's own checks whose value is out of normal, by attribution share;
    2. other checks out of normal, by how far out (supporting evidence);
    3. the detector's own checks within normal, by share;
    4. the remaining checks, by how close to their normal limit.
    So a value that is normal never outranks one that is not, and every reason the
    model itself gave is kept in its own order."""
    own = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0]
    total = sum(scores[k] for k in own) or 1.0

    def out(k):
        return np.isfinite(ratio.get(k, np.nan)) and ratio[k] > 1.0

    def by_ratio(ks):
        return sorted(ks, key=lambda k: -ratio[k] if np.isfinite(ratio[k]) else 1e9)

    others = [k for k in R.CHECKS if k not in own]
    order = (
        [(k, "model") for k in own if out(k)]
        + [(k, "supporting") for k in by_ratio([k for k in others if out(k)])]
        + [(k, "model") for k in own if not out(k)]
        + [
            (k, "supporting")
            for k in by_ratio(
                [k for k in others if not out(k) and np.isfinite(ratio[k])]
            )
        ]
    )
    return [(k, src, scores[k] / total if src == "model" else np.nan)
            for k, src in order[:top]]  # fmt: skip


class Explainer:
    """Turns (detector, flight, alarm time) into three reasons and a message."""

    def __init__(self, normals, clean_err=None):
        self.normals = normals
        self.clean_err = clean_err
        self.shap = {}

    def tree_explainer(self, det):
        key = (det.name, det.seed)
        if key not in self.shap:
            import shap

            self.shap[key] = shap.TreeExplainer(det.model[-1])
        return self.shap[key]

    def scores(self, det, fl, t):
        name, via = det.name, None
        if name in ("random_forest", "decision_tree"):
            s = tree_scores(det, fl, t, self.tree_explainer(det))
        elif name == "lstm_ae":
            s = ae_scores(det, fl, t, self.clean_err)
        elif name == "lstm_supervised":
            s = occlusion_scores(det, fl, t)
        elif name == "kalman":
            s = kalman_scores(fl, t)
        elif name == "cusum":
            s = cusum_scores(det, fl, t)
        elif name == "velocity":
            s = {"horizontal_velocity": 1.0}
        elif name == "consistency":
            s = channel_scores(det, fl, t)
        elif name == "fused":
            s, via = fused_scores(det, fl, t)
        else:
            raise KeyError(name)
        return s, via

    def explain(self, det, fl, t):
        scores, via = self.scores(det, fl, t)
        vals = values_at(fl, t)
        ratio = {k: (vals[k] / self.normals[k]
                     if np.isfinite(vals.get(k, np.nan)) and self.normals[k] > 0
                     else np.nan) for k in R.CHECKS}  # fmt: skip
        top = rank(scores, ratio)
        consistent = [k for k in R.ROW_CHECKS if np.isfinite(ratio[k]) and
                      ratio[k] <= 1.0 and k not in [c for c, *_ in top]]  # fmt: skip
        title = D_TITLES.get(det.name, det.name) + (f", {via}" if via else "")
        msg = R.message(det.name, t, [(k, vals.get(k, np.nan), self.normals[k])
                                      for k, *_ in top], consistent, title)  # fmt: skip
        row = {"alarm_t_s": float(t), "message": msg,
               "consistent": ";".join(consistent), "via": via or ""}  # fmt: skip
        for i, (k, src, share) in enumerate(top, start=1):
            row.update({f"reason{i}": k, f"reason{i}_source": src,
                        f"reason{i}_share": share,
                        f"reason{i}_value": vals.get(k, np.nan),
                        f"reason{i}_normal": self.normals[k],
                        f"reason{i}_ratio": ratio[k]})  # fmt: skip
        row.update({f"score_{k}": float(scores.get(k, 0.0)) for k in R.CHECKS})
        row.update({f"ratio_{k}": ratio[k] for k in R.CHECKS})
        return row


D_TITLES = {"kalman": "Kalman filter chi-square test", "cusum": "Kalman filter CUSUM",
            "velocity": "velocity consistency check", "fused": "fused detector",
            "consistency": "velocity and course consistency check",
            "random_forest": "random forest", "decision_tree": "decision tree",
            "lstm_ae": "LSTM autoencoder",
            "lstm_supervised": "supervised LSTM"}  # fmt: skip
