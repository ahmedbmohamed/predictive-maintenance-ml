"""Streamlit dashboard for the predictive maintenance model.

Three tabs:
* Fleet monitor: replays held-out sensor readings for 12 machines.
* What-if analysis: test any machine state, see the physics limits and a risk curve.
* Model report: how the model was chosen and how well it performs.

Run locally:  streamlit run dashboard/app.py
"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from pdm.config import RANDOM_STATE, REPORTS_DIR, TARGET, TEST_SIZE
from pdm.data import load_dataset
from pdm.features import add_engineered_features
from pdm.predictor import (
    HEAT_SPEED_RPM,
    HEAT_TEMP_DIFF_K,
    OVERSTRAIN_LIMIT,
    POWER_RANGE_W,
    Predictor,
)

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="wide")

GITHUB_URL = "https://github.com/ahmedbmohamed/predictive-maintenance-ml"
FLEET_SIZE = 12

# One color per status, used everywhere so the meaning never changes.
COLORS = {"OK": "#2E9E5B", "ANOMALY": "#E8871E", "FAILURE_RISK": "#D64545"}
LABELS = {"OK": "Normal", "ANOMALY": "Abnormal behaviour", "FAILURE_RISK": "Failure risk"}
ICONS = {"OK": "🟢", "ANOMALY": "🟠", "FAILURE_RISK": "🔴"}
ACCENT = "#0B4F8A"

FEATURE_NAMES = {
    "rotational_speed_rpm": "Rotational speed",
    "temp_diff_k": "Temperature gap",
    "strain_min_nm": "Strain (wear x torque)",
    "power_w": "Mechanical power",
    "tool_wear_min": "Tool wear",
    "torque_nm": "Torque",
    "type_L": "Product type L",
    "type_M": "Product type M",
    "type_H": "Product type H",
    "air_temp_k": "Air temperature",
    "process_temp_k": "Process temperature",
}

# Ready-made machine states for demos. Each one triggers a different failure mode.
SCENARIOS = {
    "Normal": dict(product_type="M", air=300.0, process=310.0, speed=1500, torque=40.0, wear=100),
    "Overstrain": dict(
        product_type="L", air=300.0, process=310.0, speed=1500, torque=65.0, wear=220
    ),
    "Heat dissipation": dict(
        product_type="M", air=302.0, process=310.0, speed=1300, torque=50.0, wear=30
    ),
    "Power failure": dict(
        product_type="M", air=300.0, process=310.0, speed=2800, torque=10.0, wear=50
    ),
}

# Small CSS layer for the status cards and gauges. Colors use transparency so
# they read well in both the light and the dark Streamlit theme.
st.markdown(
    """
    <style>
    .card {border-left: 6px solid var(--c); border-radius: 8px; padding: 10px 14px;
           background: color-mix(in srgb, var(--c) 9%, transparent); margin-bottom: 12px;}
    .card .title {font-weight: 700; font-size: 1rem;}
    .card .status {color: var(--c); font-weight: 700; margin: 2px 0 6px 0;}
    .card .meta {opacity: 0.7; font-size: 0.8rem;}
    .bar {height: 8px; border-radius: 4px; background: rgba(128,128,128,0.2); overflow: hidden;}
    .bar > div {height: 100%; background: var(--c);}
    .banner {border-radius: 10px; padding: 14px 18px; color: white; background: var(--c);}
    .banner .big {font-size: 1.6rem; font-weight: 800;}
    .gauge {margin: 6px 0 16px 0;}
    .gauge .head {display: flex; justify-content: space-between; font-size: 0.9rem;}
    .gauge .track {position: relative; height: 14px; border-radius: 7px;
                   background: rgba(214,69,69,0.25); margin-top: 4px;}
    .gauge .safe {position: absolute; top: 0; height: 100%; background: rgba(46,158,91,0.35);}
    .gauge .mark {position: absolute; top: -4px; width: 4px; height: 22px;
                  border-radius: 2px; background: var(--c);}
    .gauge .note {opacity: 0.7; font-size: 0.78rem; margin-top: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data loading


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor.load()


@st.cache_data
def load_report() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


@st.cache_data
def scored_test_set() -> pd.DataFrame:
    """Held-out test readings (never seen in training) with their predictions."""
    df = load_dataset()
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    test = df.iloc[test_idx].reset_index(drop=True)
    preds = pd.DataFrame(get_predictor().predict(test))
    return pd.concat([test, preds], axis=1)


def reading_from_state() -> pd.DataFrame:
    """Build a one-row DataFrame from the what-if sliders."""
    s = st.session_state
    return pd.DataFrame(
        [
            {
                "product_type": s.product_type,
                "air_temp_k": s.air,
                "process_temp_k": s.process,
                "rotational_speed_rpm": s.speed,
                "torque_nm": s.torque,
                "tool_wear_min": s.wear,
            }
        ]
    )


# ---------------------------------------------------------------- small HTML parts


def machine_card(number: int, row: pd.Series) -> str:
    color = COLORS[row["status"]]
    prob = row["failure_probability"]
    outcome = "Failed" if row[TARGET] == 1 else "No failure"
    return f"""
    <div class="card" style="--c:{color}">
      <div class="title">Machine {number:02d}
        <span class="meta">type {row["product_type"]}</span></div>
      <div class="status">{LABELS[row["status"]]}</div>
      <div class="bar"><div style="width:{prob * 100:.0f}%"></div></div>
      <div class="meta">Failure probability {prob:.0%} &nbsp;|&nbsp; Actual outcome: {outcome}</div>
    </div>"""


def gauge(label, value, unit, safe_low, safe_high, scale_max, note) -> str:
    """Horizontal gauge: green = safe zone, red = risk zone, marker = current value."""
    in_safe = safe_low <= value <= safe_high
    color = COLORS["OK"] if in_safe else COLORS["FAILURE_RISK"]

    def pct(x):
        return max(0.0, min(100.0, x / scale_max * 100))

    return f"""
    <div class="gauge" style="--c:{color}">
      <div class="head"><b>{label}</b>
        <span style="color:{color}"><b>{value:,.1f} {unit}</b></span></div>
      <div class="track">
        <div class="safe"
             style="left:{pct(safe_low)}%; width:{pct(safe_high) - pct(safe_low)}%"></div>
        <div class="mark" style="left:calc({pct(value)}% - 2px)"></div>
      </div>
      <div class="note">{note}</div>
    </div>"""


def apply_scenario(name: str) -> None:
    """Button callback: copy a preset into the slider values."""
    for key, value in SCENARIOS[name].items():
        st.session_state[key] = value


# ---------------------------------------------------------------- page sections


def sidebar(predictor: Predictor, report: dict) -> None:
    m = report["test_metrics"]
    with st.sidebar:
        st.markdown("### 🏭 Predictive Maintenance")
        st.write(
            "Predicts machine failures from sensor data and explains each alert "
            "with physical operating limits."
        )
        st.divider()
        st.markdown("**Model**")
        st.write(
            f"{report['selected_model'].replace('_', ' ').title()} v{predictor.info['version']}"
        )
        st.write(f"Decision threshold: **{predictor.info['threshold']:.2f}**")
        c1, c2 = st.columns(2)
        c1.metric("Recall", f"{m['recall']:.0%}")
        c2.metric("Precision", f"{m['precision']:.0%}")
        c1.metric("PR-AUC", f"{m['pr_auc']:.2f}")
        c2.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        st.caption(f"Evaluated on {report['dataset']['test_rows']:,} held-out readings.")
        st.divider()
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})")
        st.caption("Data: AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0)")


def header(data: pd.DataFrame) -> None:
    st.title("Predictive Maintenance Dashboard")
    st.caption(
        "Failure prediction and anomaly detection for industrial machines. "
        "All figures below come from readings the model never saw during training."
    )
    real = int(data[TARGET].sum())
    caught = int((data["failure_predicted"] & (data[TARGET] == 1)).sum())
    false_alarms = int((data["failure_predicted"] & (data[TARGET] == 0)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readings monitored", f"{len(data):,}")
    c2.metric("Failure alerts", int(data["failure_predicted"].sum()))
    c3.metric("Real failures caught", f"{caught} / {real}")
    c4.metric("False alarms", false_alarms)


def fleet_tab(data: pd.DataFrame) -> None:
    st.subheader("Fleet monitor")
    st.write("Twelve machines replay held-out readings. Press Play or move the time step.")
    max_step = len(data) // FLEET_SIZE - 1
    playing = st.toggle("▶ Play", value=False, help="Advance one time step every 2 seconds")

    # A fragment reruns on its own timer, without reloading the whole page.
    @st.fragment(run_every=2 if playing else None)
    def live_grid():
        if "step" not in st.session_state:
            st.session_state.step = 1  # step 1 shows two real failures
        if playing:
            st.session_state.step = (st.session_state.step + 1) % (max_step + 1)
        step = st.slider("Time step", 0, max_step, key="step")
        window = data.iloc[step * FLEET_SIZE : (step + 1) * FLEET_SIZE].reset_index(drop=True)

        counts = window["status"].value_counts()
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{ICONS['OK']} Normal", int(counts.get("OK", 0)))
        k2.metric(f"{ICONS['ANOMALY']} Abnormal", int(counts.get("ANOMALY", 0)))
        k3.metric(f"{ICONS['FAILURE_RISK']} Failure risk", int(counts.get("FAILURE_RISK", 0)))

        cols = st.columns(4)
        for i, row in window.iterrows():
            cols[i % 4].markdown(machine_card(i + 1, row), unsafe_allow_html=True)

    live_grid()

    st.subheader("Alert log")
    st.caption("Every reading in the test set that raised an alert, highest risk first.")
    alerts = data[data["status"] != "OK"].copy()
    alerts["status"] = alerts["status"].map(lambda s: f"{ICONS[s]} {LABELS[s]}")
    alerts["actual"] = alerts[TARGET].map({1: "Failed", 0: "No failure"})
    alerts["reasons"] = alerts["reasons"].apply(lambda r: " | ".join(r) if r else "")
    alerts = alerts.sort_values("failure_probability", ascending=False)
    st.dataframe(
        alerts[
            [
                "status",
                "failure_probability",
                "actual",
                "product_type",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
                "reasons",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={
            "status": "Status",
            "failure_probability": st.column_config.ProgressColumn(
                "Failure probability", format="percent", min_value=0, max_value=1
            ),
            "actual": "Actual outcome",
            "product_type": "Type",
            "rotational_speed_rpm": st.column_config.NumberColumn("Speed (rpm)"),
            "torque_nm": st.column_config.NumberColumn("Torque (Nm)"),
            "tool_wear_min": st.column_config.NumberColumn("Tool wear (min)"),
            "reasons": st.column_config.TextColumn("Why", width="large"),
        },
    )


def risk_curve(predictor: Predictor, reading: pd.DataFrame, variable: str) -> alt.Chart:
    """Failure probability while one sensor varies and the others stay fixed."""
    ranges = {
        "tool_wear_min": (0, 260, "Tool wear (min)"),
        "torque_nm": (3, 80, "Torque (Nm)"),
        "rotational_speed_rpm": (1150, 2900, "Rotational speed (rpm)"),
    }
    low, high, title = ranges[variable]
    values = np.linspace(low, high, 60)
    sweep = pd.concat([reading] * len(values), ignore_index=True)
    sweep[variable] = values
    sweep["probability"] = [p["failure_probability"] for p in predictor.predict(sweep)]

    threshold = predictor.info["threshold"]
    current = float(reading[variable].iloc[0])
    current_prob = predictor.predict(reading)[0]["failure_probability"]

    line = (
        alt.Chart(sweep)
        .mark_line(color=ACCENT, strokeWidth=3)
        .encode(
            x=alt.X(variable, title=title),
            y=alt.Y("probability", title="Failure probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip(variable, format=".0f"), alt.Tooltip("probability", format=".0%")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color=COLORS["FAILURE_RISK"], strokeDash=[6, 4])
        .encode(y="y")
    )
    label = (
        alt.Chart(pd.DataFrame({"y": [threshold], "x": [low]}))
        .mark_text(align="left", dy=-8, color=COLORS["FAILURE_RISK"])
        .encode(x="x", y="y", text=alt.value(f"Alert threshold {threshold:.0%}"))
    )
    point = (
        alt.Chart(pd.DataFrame({variable: [current], "probability": [current_prob]}))
        .mark_point(size=160, filled=True, color=COLORS["FAILURE_RISK"])
        .encode(x=variable, y="probability")
    )
    return (line + rule + label + point).properties(height=280)


def whatif_tab(predictor: Predictor) -> None:
    st.subheader("What-if analysis")
    st.write("Pick a scenario or move the sliders to test any machine state.")

    for key, value in SCENARIOS["Normal"].items():
        st.session_state.setdefault(key, value)

    buttons = st.columns(len(SCENARIOS))
    for col, name in zip(buttons, SCENARIOS, strict=True):
        col.button(name, on_click=apply_scenario, args=(name,), use_container_width=True)

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.radio(
            "Product type (quality variant)", ["L", "M", "H"], horizontal=True, key="product_type"
        )
        st.slider("Air temperature (K)", 295.0, 305.0, step=0.1, key="air")
        st.slider("Process temperature (K)", 305.0, 314.0, step=0.1, key="process")
        st.slider("Rotational speed (rpm)", 1150, 2900, step=10, key="speed")
        st.slider("Torque (Nm)", 3.0, 80.0, step=0.5, key="torque")
        st.slider("Tool wear (min)", 0, 260, step=1, key="wear")

    reading = reading_from_state()
    result = predictor.predict(reading)[0]
    features = add_engineered_features(reading).iloc[0]

    with right:
        color = COLORS[result["status"]]
        st.markdown(
            f"""<div class="banner" style="--c:{color}">
              <div>{ICONS[result["status"]]} {LABELS[result["status"]]}</div>
              <div class="big">{result["failure_probability"]:.1%} failure probability</div>
              <div>Anomaly score {result["anomaly_score"]:.3f}
                   ({"abnormal" if result["is_anomaly"] else "within normal behaviour"})</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("#### Physical operating limits")
        limit = OVERSTRAIN_LIMIT[st.session_state.product_type]
        slow = features["rotational_speed_rpm"] < HEAT_SPEED_RPM
        st.markdown(
            gauge(
                "Temperature gap (process minus air)",
                features["temp_diff_k"],
                "K",
                HEAT_TEMP_DIFF_K,
                15,
                15,
                f"Heat builds up below {HEAT_TEMP_DIFF_K} K when speed is under "
                f"{HEAT_SPEED_RPM} rpm (speed is {'low' if slow else 'high enough'} now).",
            )
            + gauge(
                "Mechanical power (torque x angular speed)",
                features["power_w"],
                "W",
                POWER_RANGE_W[0],
                POWER_RANGE_W[1],
                12_000,
                f"Safe range {POWER_RANGE_W[0]:,} to {POWER_RANGE_W[1]:,} W.",
            )
            + gauge(
                "Strain (tool wear x torque)",
                features["strain_min_nm"],
                "min.Nm",
                0,
                limit,
                max(limit * 1.4, features["strain_min_nm"] * 1.05),
                f"Overstrain limit for type {st.session_state.product_type}: {limit:,}.",
            ),
            unsafe_allow_html=True,
        )
        if result["reasons"]:
            st.markdown("**Why this alert**")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")

    st.markdown("#### Risk curve")
    variable = st.radio(
        "Vary one sensor, keep the others fixed",
        ["tool_wear_min", "torque_nm", "rotational_speed_rpm"],
        format_func={
            "tool_wear_min": "Tool wear",
            "torque_nm": "Torque",
            "rotational_speed_rpm": "Rotational speed",
        }.get,
        horizontal=True,
    )
    st.altair_chart(risk_curve(predictor, reading, variable), use_container_width=True)
    st.caption(
        "The red dot is the current machine. Above the dashed line the dashboard raises an alert."
    )


def report_tab(report: dict) -> None:
    st.subheader("Model report")
    m = report["test_metrics"]
    ab = report["ablation_without_engineered_features"]

    c1, c2, c3, c4 = st.columns(4)
    recall = m["true_positives"] / (m["true_positives"] + m["false_negatives"])
    c1.metric("Recall", f"{recall:.1%}", help="Share of real failures caught")
    c2.metric("Precision", f"{m['precision']:.1%}", help="Share of alerts that were real failures")
    c3.metric(
        "PR-AUC",
        f"{m['pr_auc']:.3f}",
        delta=f"+{m['pr_auc'] - ab['pr_auc']:.3f} from physics features",
    )
    c4.metric(
        "False alarms",
        m["false_positives"],
        delta=f"{m['false_positives'] - ab['false_positives']} vs raw sensors",
        delta_color="inverse",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**1. Model comparison** (5-fold cross-validation on the training set)")
        comp = pd.DataFrame(report["model_comparison_cv"]).T.reset_index(names="model")
        comp["model"] = comp["model"].str.replace("_", " ").str.title()
        comp["low"] = comp["cv_pr_auc_mean"] - comp["cv_pr_auc_std"]
        comp["high"] = comp["cv_pr_auc_mean"] + comp["cv_pr_auc_std"]
        chosen = report["selected_model"].replace("_", " ").title()
        order = comp.sort_values("cv_pr_auc_mean", ascending=False)["model"].tolist()
        bars = (
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X("cv_pr_auc_mean", title="PR-AUC", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("model", title=None, sort=order),
                color=alt.condition(
                    alt.datum.model == chosen, alt.value(ACCENT), alt.value("#9AA5B1")
                ),
                tooltip=["model", alt.Tooltip("cv_pr_auc_mean", format=".3f")],
            )
        )
        errors = (
            alt.Chart(comp)
            .mark_errorbar()
            .encode(x=alt.X("low", title="PR-AUC"), x2="high", y=alt.Y("model", sort=order))
        )
        st.altair_chart((bars + errors).properties(height=170), use_container_width=True)
        st.caption(
            "PR-AUC measures how well the rare failures (3.4% of readings) are found. "
            "Random Forest wins and is the deployed model."
        )

        st.markdown("**2. Recall by failure mode** (test set)")
        names = {
            "HDF": "Heat dissipation",
            "OSF": "Overstrain",
            "PWF": "Power",
            "TWF": "Tool wear",
            "RNF": "Random",
        }
        modes = pd.DataFrame(report["recall_by_failure_mode"]).T.reset_index(names="mode")
        modes["mode"] = modes["mode"].map(names)
        modes["color"] = np.where(modes["recall"] >= 0.8, COLORS["OK"], COLORS["FAILURE_RISK"])
        st.altair_chart(
            alt.Chart(modes)
            .mark_bar()
            .encode(
                x=alt.X("recall", title="Share caught", axis=alt.Axis(format="%")),
                y=alt.Y("mode", title=None, sort="-x", axis=alt.Axis(labelOverlap=False)),
                color=alt.Color("color", scale=None),
                tooltip=["mode", "count", alt.Tooltip("recall", format=".0%")],
            )
            .properties(height=200),
            use_container_width=True,
        )
        st.caption(
            "Heat, power and overstrain failures follow physical rules, so they are caught "
            "above 90%. Tool wear failures happen at a random point between 200 and 240 min "
            "and random failures do not depend on the sensors, so one reading cannot predict them."
        )

    with right:
        st.markdown("**3. Which inputs matter** (permutation importance on the test set)")
        imp = pd.Series(report["permutation_importance"]).rename_axis("feature")
        imp = imp.reset_index(name="importance")
        engineered = {"temp_diff_k", "power_w", "strain_min_nm"}
        imp["kind"] = np.where(imp["feature"].isin(engineered), "Physics feature", "Raw sensor")
        imp["feature"] = imp["feature"].map(FEATURE_NAMES)
        st.altair_chart(
            alt.Chart(imp)
            .mark_bar()
            .encode(
                x=alt.X("importance", title="Drop in PR-AUC when shuffled"),
                y=alt.Y(
                    "feature",
                    title=None,
                    sort="-x",
                    axis=alt.Axis(labelOverlap=False, labelLimit=220),
                ),
                color=alt.Color(
                    "kind",
                    scale=alt.Scale(
                        domain=["Physics feature", "Raw sensor"], range=[ACCENT, "#9AA5B1"]
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["feature", alt.Tooltip("importance", format=".3f")],
            )
            .properties(height=330),
            use_container_width=True,
        )
        st.caption(
            "Three of the four most useful inputs are the physics features built from "
            "the failure mechanisms: temperature gap, strain and power."
        )

        st.markdown(f"**4. Confusion matrix** (test set, threshold {m['threshold']:.2f})")
        cm = pd.DataFrame(
            [
                ("No failure", "No alert", m["true_negatives"]),
                ("No failure", "Alert", m["false_positives"]),
                ("Failure", "No alert", m["false_negatives"]),
                ("Failure", "Alert", m["true_positives"]),
            ],
            columns=["actual", "predicted", "count"],
        )
        cm["good"] = (cm["actual"] == "Failure") == (cm["predicted"] == "Alert")
        base = alt.Chart(cm).encode(
            x=alt.X(
                "predicted",
                title="Model decision",
                sort=["No alert", "Alert"],
                axis=alt.Axis(labelAngle=0, orient="top"),
            ),
            y=alt.Y("actual", title="Actual outcome", sort=["No failure", "Failure"]),
        )
        tiles = base.mark_rect(cornerRadius=6).encode(
            color=alt.condition(
                alt.datum.good, alt.value(COLORS["OK"]), alt.value(COLORS["FAILURE_RISK"])
            ),
            opacity=alt.value(0.75),
        )
        text = base.mark_text(fontSize=22, fontWeight="bold", color="white").encode(text="count")
        st.altair_chart((tiles + text).properties(height=220), use_container_width=True)
        healthy = m["true_negatives"] + m["false_positives"]
        st.caption(
            f"{m['true_positives']} failures caught, {m['false_negatives']} missed and "
            f"{m['false_positives']} false alarms among {healthy:,} healthy readings."
        )


# ---------------------------------------------------------------- page


predictor = get_predictor()
report = load_report()
data = scored_test_set()

sidebar(predictor, report)
header(data)
fleet, whatif, model_report = st.tabs(
    ["🏭 Fleet monitor", "🔧 What-if analysis", "📊 Model report"]
)
with fleet:
    fleet_tab(data)
with whatif:
    whatif_tab(predictor)
with model_report:
    report_tab(report)
