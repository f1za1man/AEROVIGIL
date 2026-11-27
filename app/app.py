r"""AEROVIGIL demo dashboard (plan M13, Step H): GPS-spoofing detection, offline.

Everything shown was computed beforehand by src\aerovigil\demo\build.py with the
project's own detectors (Step C seen-protocol thresholds: 1 false alarm per
flight-hour), so the app needs no internet, no GPU and no training.

Run:  .\make.ps1 demo        (Windows)   or   make demo
or:   python -m streamlit run app\app.py
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "demo"
# categorical slots 1-3 of the project palette; all three read on light and dark
TRUE_C, SPOOF_C, ALARM_C, BAND = "#257de0", "#f47230", "#00b894", "#f47230"
SPOOF_LABEL = "GPS position (spoofed)"
# sequential blues, light -> dark (how far the drone was pushed before the alarm)
BLUES = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
         "#08306b"]  # fmt: skip
MISSED = "#bdbdbd"
MISSION = {"curved": "Curved", "random": "Random", "straight": "Straight"}
MODE_LABEL = {"position_only": "Position only (GPS speed left honest)",
              "consistent": "Velocity-consistent (GPS speed spoofed to match)"}  # fmt: skip
ORDER = ["kalman", "velocity", "fused", "cusum", "consistency", "random_forest",
         "lstm_ae", "lstm_supervised"]  # fmt: skip
PHYSICS = {"kalman", "velocity", "fused", "cusum", "consistency"}


@st.cache_data
def load():
    scen = pd.read_parquet(DATA / "scenarios.parquet")
    res = pd.read_parquet(DATA / "results.parquet")
    return scen, res


@st.cache_data
def series(sid):
    return pd.read_parquet(DATA / "series" / f"{sid}.parquet")


def strength_label(typ, s):
    if typ == "abrupt_jump":
        return f"{s:g} m jump"
    return f"{s:g} m/s"


def flight_label(row):
    tag = " - real attack" if row["attack_type"] == "real" else ""
    return f"{MISSION[row['mission_type']]} flight {int(row.name) + 1}{tag}"


def fmt_s(x):
    return "" if pd.isna(x) else f"{x:.1f} s"


def fmt_m(x):
    return "" if pd.isna(x) else f"{x:.1f} m"


# ------------------------------------------------------------------ charts --


def attack_band(meta, y_field="y"):
    if pd.isna(meta["attack_start_s"]):
        return None
    d = pd.DataFrame({"start": [meta["attack_start_s"]], "end": [meta["attack_end_s"]]})
    return (
        alt.Chart(d).mark_rect(color=BAND, opacity=0.12).encode(x="start:Q", x2="end:Q")
    )


def path_chart(path, alarms):
    long = pd.concat([
        pd.DataFrame({"East (m)": path["true_e"], "North (m)": path["true_n"],
                      "t": path["t"], "path": "True position"}),
        pd.DataFrame({"East (m)": path["spoof_e"], "North (m)": path["spoof_n"],
                      "t": path["t"], "path": SPOOF_LABEL}),
    ])  # fmt: skip
    color = alt.Color("path:N", scale=alt.Scale(
        domain=["True position", SPOOF_LABEL],
        range=[TRUE_C, SPOOF_C]), legend=alt.Legend(orient="bottom", title=None))  # fmt: skip
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("East (m):Q", scale=alt.Scale(zero=False)),
        y=alt.Y("North (m):Q", scale=alt.Scale(zero=False)),
        color=color, order="t:Q",
        tooltip=[alt.Tooltip("t:Q", title="time (s)", format=".1f"), "path:N"],
    )  # fmt: skip
    layers = [lines]
    if len(alarms):
        layers.append(alt.Chart(alarms).mark_point(
            shape="diamond", size=110, filled=True, color=ALARM_C).encode(
            x="East (m):Q", y="North (m):Q",
            tooltip=["detector:N", alt.Tooltip("t:Q", title="alarm at (s)",
                                               format=".1f")]))  # fmt: skip
    return alt.layer(*layers).properties(height=360)


def offset_chart(path, meta, first):
    base = alt.Chart(path).mark_line(color=SPOOF_C, strokeWidth=2).encode(
        x=alt.X("t:Q", title="time (s)"),
        y=alt.Y("offset_m:Q", title="GPS pushed away from truth (m)"),
        tooltip=[alt.Tooltip("t:Q", format=".1f"), alt.Tooltip("offset_m:Q", format=".2f")],
    )  # fmt: skip
    layers = [base]
    band = attack_band(meta)
    if band is not None:
        layers.insert(0, band)
    if len(first):
        layers.append(
            alt.Chart(first)
            .mark_rule(color=ALARM_C, strokeDash=[4, 3])
            .encode(x="t:Q", tooltip=["detector:N", alt.Tooltip("t:Q", format=".1f")])
        )
    return alt.layer(*layers).properties(height=360)


def signal_chart(sig, r, meta, log_y):
    d = sig[["t", "value"]].copy()
    if log_y:
        d["value"] = d["value"].clip(lower=1e-3)
    y = alt.Y(
        "value:Q",
        title=None,
        axis=alt.Axis(tickCount=4),
        scale=alt.Scale(type="log") if log_y else alt.Scale(zero=False),
    )
    layers = []
    band = attack_band(meta)
    if band is not None:
        layers.append(band)
    layers.append(
        alt.Chart(d)
        .mark_line(color=TRUE_C, strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="time (s)"),
            y=y,
            tooltip=[
                alt.Tooltip("t:Q", format=".1f"),
                alt.Tooltip("value:Q", format=".3g"),
            ],
        )
    )
    thr = pd.DataFrame({"thr": [r["threshold"]]})
    layers.append(
        alt.Chart(thr)
        .mark_rule(color=SPOOF_C, strokeDash=[5, 3])
        .encode(
            y="thr:Q", tooltip=[alt.Tooltip("thr:Q", title="threshold", format=".3g")]
        )
    )
    times = json.loads(r["alarm_times"])
    if times:
        layers.append(alt.Chart(pd.DataFrame({"t": times})).mark_rule(
            color=ALARM_C, strokeWidth=1).encode(x="t:Q"))  # fmt: skip
    return alt.layer(*layers).properties(height=200)


def sweep_chart(table, current):
    d = table.copy()
    det = d["detected"].astype(bool).to_numpy()
    d["text"] = np.where(det, d["delay_s"].map(lambda x: f"{x:.1f}s"), "missed")
    # one hue, light -> dark with how far the drone was pushed (0.1 m .. 1 km, log)
    level = np.clip((np.log10(d["displacement_m"].clip(0.1, 1000)) + 1) / 4, 0, 1)
    d["fill"] = [BLUES[int(round(v * (len(BLUES) - 1)))] if ok else MISSED
                 for v, ok in zip(level, det)]  # fmt: skip
    d["ink"] = [("#ffffff" if v > 0.5 else "#191919") if ok else "#191919"
                for v, ok in zip(level, det)]  # fmt: skip
    d["current"] = d["strength_label"] == current
    order = list(dict.fromkeys(d["strength_label"]))
    base = alt.Chart(d).encode(
        x=alt.X("strength_label:N", sort=order, title="attack strength",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("label:N", sort=[d.loc[d["detector"] == k, "label"].iloc[0]
                                 for k in ORDER if (d["detector"] == k).any()],
                title=None, axis=alt.Axis(labelLimit=260)),
    )  # fmt: skip
    cells = base.mark_rect(stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color("fill:N", scale=None),
        tooltip=["label:N", "strength_label:N",
                 alt.Tooltip("delay_s:Q", title="caught after (s)", format=".1f"),
                 alt.Tooltip("displacement_m:Q", title="drone moved (m)", format=".1f")],
    )  # fmt: skip
    text = base.mark_text(fontSize=11).encode(
        text="text:N", color=alt.Color("ink:N", scale=None)
    )
    ring = base.transform_filter("datum.current").mark_rect(
        fill=None, stroke=ALARM_C, strokeWidth=2.5
    )
    return alt.layer(cells, text, ring).properties(height=34 * d["label"].nunique())


# --------------------------------------------------------------------- app --


def main():
    st.set_page_config(page_title="AEROVIGIL demo", layout="wide")
    if not (DATA / "scenarios.parquet").is_file():
        st.error(
            "The demo data is missing. Build it once (about 30 minutes): "
            "`python src\\aerovigil\\demo\\build.py`, or run `.\\make.ps1 demo`."
        )
        st.stop()
    scen, res = load()

    st.title("AEROVIGIL - GPS spoofing detection")
    st.caption(
        "PX4 SITL flights from the dataset. Every detector runs at the project's "
        "budget of 1 false alarm per flight-hour, set on clean training flights. "
        "Precomputed: works offline."
    )

    flights = (scen.drop_duplicates("flight_no").sort_values("flight_no")
               .set_index("flight_no"))  # fmt: skip
    with st.sidebar:
        st.header("Scenario")
        fno = st.selectbox("Flight", flights.index.tolist(),
                           format_func=lambda i: flight_label(flights.loc[i]))  # fmt: skip
        fs = scen[scen["flight_no"] == fno]
        types = [t for t in ["none", "real", "linear_drift", "exponential_ramp",
                             "constant_offset", "abrupt_jump"]
                 if (fs["attack_type"] == t).any()]  # fmt: skip
        labels = dict(zip(scen["attack_type"], scen["type_label"]))
        typ = st.selectbox("Attack", types, index=types.index("linear_drift") if
                           "linear_drift" in types else 0,
                           format_func=lambda t: labels.get(t, str(t)))  # fmt: skip
        ts = fs[fs["attack_type"] == typ]
        mode, strength = "na", np.nan
        if typ not in ("none", "real"):
            mode = st.radio("How careful is the attacker?", ["position_only", "consistent"],
                            format_func=MODE_LABEL.get)  # fmt: skip
            ts = ts[ts["velocity_mode"] == mode]
            options = sorted(ts["strength"].unique())
            st.subheader("Out-drift the detector")
            strength = st.select_slider(
                "Jump size" if typ == "abrupt_jump" else "Drift speed",
                options=options, value=options[len(options) // 2],
                format_func=lambda s: strength_label(typ, s),
                help="Slower drift is harder to see. Move it and watch which "
                     "detectors still fire, and how far the drone gets pushed first.",
            )  # fmt: skip
            ts = ts[ts["strength"] == strength]
        chosen = st.multiselect("Detectors", ORDER, default=ORDER,
                                format_func=lambda k: res.loc[res["detector"] == k,
                                                              "label"].iloc[0])  # fmt: skip
    meta = ts.iloc[0]
    sid = meta["scenario"]
    r = res[(res["scenario"] == sid) & res["detector"].isin(chosen)].copy()
    r["k"] = r["detector"].map(ORDER.index)
    r = r.sort_values("k")
    s = series(sid)
    path = s[s["kind"] == "path"].dropna(axis=1, how="all")

    # ---- headline
    c0, c1, c2, c3, c4 = st.columns(5)
    c0.metric("Attack", meta["type_label"].split(" (")[0])
    c1.metric("Strength", "-" if typ in ("none", "real") else
              strength_label(typ, strength))  # fmt: skip
    c2.metric("Attack starts", fmt_s(meta["attack_start_s"]) or "no attack")
    c3.metric("GPS pushed away by the end", fmt_m(meta["final_offset_m"]) or "0 m")
    caught = int(r["detected"].sum())
    if typ == "none":
        c4.metric("False alarms", int(r["n_alarms"].sum()))
    else:
        c4.metric("Detectors that caught it", f"{caught} of {len(r)}")

    # ---- path and offset
    first = r[r["first_alarm_s"].notna()][["label", "first_alarm_s"]].rename(
        columns={"label": "detector", "first_alarm_s": "t"}
    )
    marks = []
    for _, row in first.iterrows():
        i = int(np.argmin(np.abs(path["t"].to_numpy() - row["t"])))
        marks.append({"detector": row["detector"], "t": row["t"],
                      "East (m)": path["true_e"].iloc[i],
                      "North (m)": path["true_n"].iloc[i]})  # fmt: skip
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Where the drone is vs where GPS says it is")
        st.altair_chart(path_chart(path, pd.DataFrame(marks)), width="stretch")
        st.caption(
            "Diamonds: where the drone really was at each detector's first alarm."
        )
    with right:
        st.subheader("How far GPS has been pushed, and when each detector fired")
        st.altair_chart(offset_chart(path, meta, first), width="stretch")
        st.caption("Shaded: the attack. Dashed lines: each detector's first alarm.")

    # ---- comparison table
    st.subheader("Detectors side by side")
    table = pd.DataFrame({
        "Detector": r["label"],
        "Type": np.where(r["detector"].isin(PHYSICS), "physics", "learned"),
        "Caught it": np.where(r["detected"], "yes", "no") if typ != "none" else "-",
        "First alarm": r["first_alarm_s"].map(fmt_s),
        "Seconds after the attack began": r["delay_s"].map(fmt_s),
        "Drone moved before the alarm": np.where(r["detected"],
                                                 r["displacement_m"].map(fmt_m), ""),
        "False alarms": np.where(typ == "none", r["n_alarms"],
                                 r["alarms_before_onset"]),
    })  # fmt: skip
    st.dataframe(table, hide_index=True, width="stretch")

    # ---- out-drift sweep
    if typ not in ("none", "real"):
        st.subheader("Out-drift the detector: every strength at once")
        sweep = scen[(scen["flight_no"] == fno) & (scen["attack_type"] == typ)
                     & (scen["velocity_mode"] == mode)]  # fmt: skip
        sw = res[res["scenario"].isin(sweep["scenario"]) & res["detector"].isin(chosen)]
        sw = sw.merge(sweep[["scenario", "strength"]], on="scenario")
        sw["strength_label"] = sw["strength"].map(lambda v: strength_label(typ, v))
        sw = sw.sort_values("strength")
        st.altair_chart(sweep_chart(sw, strength_label(typ, strength)),
                        width="stretch")  # fmt: skip
        st.caption(
            "Each cell: how long after the attack began the detector fired "
            "(darker = the drone had been pushed further by then), or 'missed'. "
            "The green box is the strength picked on the left."
        )

    # ---- signals
    st.subheader("What each detector watches")
    st.caption(
        "Blue: the detector's signal. Orange dashed: its alarm threshold. Green "
        "lines: alarms. Shaded: the attack."
    )
    cols = st.columns(2)
    for i, (_, row) in enumerate(r.iterrows()):
        sig = s[s["kind"] == row["detector"]]
        log_y = row["detector"] in ("kalman", "velocity", "cusum", "consistency",
                                    "lstm_ae", "fused")  # fmt: skip
        with cols[i % 2]:
            st.markdown(f"**{row['label']}** - {row['signal']}"
                        + (" (log scale)" if log_y else ""))  # fmt: skip
            st.altair_chart(signal_chart(sig, row, meta, log_y), width="stretch",
                            height=240)  # fmt: skip

    # ---- explanation
    st.subheader("Why did it fire?")
    fired = r[r["message"] != ""]
    if not len(fired):
        st.info("No detector raised an alarm on this flight.")
        return
    pick = st.selectbox("Alarm of", fired["detector"].tolist(),
                        format_func=lambda k: fired.loc[fired["detector"] == k,
                                                        "label"].iloc[0])  # fmt: skip
    row = fired[fired["detector"] == pick].iloc[0]
    st.caption(f"Explaining the {row['explained_why']}.")
    (st.warning if "false alarm" in row["explained_why"] else st.error)(row["message"])
    reasons = pd.DataFrame(json.loads(row["reasons"]))
    if len(reasons):
        st.dataframe(pd.DataFrame({
            "#": range(1, len(reasons) + 1),
            "What disagrees": reasons["text"],
            "Measured": [f"{v:.3g} {u}" for v, u in zip(reasons["value"], reasons["unit"])],
            "Normal up to": [f"{v:.3g} {u}" for v, u in zip(reasons["normal"],
                                                            reasons["unit"])],
            "From": np.where(reasons["source"] == "model", "the detector itself",
                             "supporting check"),
        }), hide_index=True, width="stretch")  # fmt: skip


main()
