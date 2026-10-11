"""Streamlit dashboard: fleet monitoring, what-if analysis and model report.

Run locally:  streamlit run dashboard/app.py
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from pdm.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, TARGET, TEST_SIZE
from pdm.data import load_dataset
from pdm.predictor import Predictor

st.set_page_config(page_title="Predictive Maintenance", page_icon="🏭", layout="wide")

STATUS_STYLE = {
    "OK": ("🟢", "Normal"),
    "ANOMALY": ("🟠", "Abnormal behaviour"),
    "FAILURE_RISK": ("🔴", "Failure risk"),
}
FLEET_SIZE = 12


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor.load()


@st.cache_data
def held_out_readings() -> pd.DataFrame:
    """The test split the model never saw during training."""
    df = load_dataset()
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    return df.iloc[test_idx].reset_index(drop=True)


@st.cache_data
def scored_readings() -> pd.DataFrame:
    df = held_out_readings()
    preds = pd.DataFrame(get_predictor().predict(df))
    return pd.concat([df, preds], axis=1)


predictor = get_predictor()
st.title("🏭 Predictive Maintenance")
st.caption(
    f"Model: {predictor.info['model']} v{predictor.info['version']}, "
    f"decision threshold {predictor.info['threshold']:.2f}, trained {predictor.info['trained_at']}"
)

fleet_tab, whatif_tab, report_tab = st.tabs(["Fleet monitor", "What-if analysis", "Model report"])

with fleet_tab:
    st.subheader("Replay of held-out sensor readings")
    st.write(
        "Each machine replays readings from the test set, which the model never saw. "
        "Move the time step to see how alerts change."
    )
    data = scored_readings()
    step = st.slider("Time step", 0, len(data) // FLEET_SIZE - 1, 0)
    window = data.iloc[step * FLEET_SIZE : (step + 1) * FLEET_SIZE].reset_index(drop=True)

    cols = st.columns(4)
    for i, row in window.iterrows():
        icon, label = STATUS_STYLE[row["status"]]
        with cols[i % 4].container(border=True):
            st.markdown(f"**Machine {i + 1:02d}** (type {row['product_type']})")
            st.markdown(f"{icon} {label}")
            st.progress(
                min(row["failure_probability"], 1.0),
                text=f"Failure probability {row['failure_probability']:.0%}",
            )
            actual = "failed" if row[TARGET] == 1 else "no failure"
            st.caption(f"Actual outcome: {actual}")

    st.divider()
    st.subheader("Alert summary for the whole test set")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readings", len(data))
    c2.metric("Failure alerts", int(data["failure_predicted"].sum()))
    c3.metric("Anomaly alerts", int((data["status"] == "ANOMALY").sum()))
    caught = int((data["failure_predicted"] & (data[TARGET] == 1)).sum())
    c4.metric("Real failures caught", f"{caught} / {int(data[TARGET].sum())}")

    alerts = data[data["status"] != "OK"].copy()
    alerts["reasons"] = alerts["reasons"].apply(lambda r: "; ".join(r) if r else "")
    st.dataframe(
        alerts[
            [
                "product_type",
                "air_temp_k",
                "process_temp_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",
                "failure_probability",
                "status",
                TARGET,
                "reasons",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

with whatif_tab:
    st.subheader("Test a machine state")
    left, right = st.columns([1, 1])
    with left:
        product_type = st.radio("Product type", ["L", "M", "H"], horizontal=True)
        air = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
        speed = st.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
        torque = st.slider("Torque (Nm)", 3.0, 80.0, 40.0, 0.5)
        wear = st.slider("Tool wear (min)", 0, 260, 100, 1)
    reading = pd.DataFrame(
        [
            {
                "product_type": product_type,
                "air_temp_k": air,
                "process_temp_k": process,
                "rotational_speed_rpm": speed,
                "torque_nm": torque,
                "tool_wear_min": wear,
            }
        ]
    )
    result = predictor.predict(reading)[0]
    with right:
        icon, label = STATUS_STYLE[result["status"]]
        st.markdown(f"## {icon} {label}")
        st.metric("Failure probability", f"{result['failure_probability']:.1%}")
        st.metric(
            "Anomaly score",
            f"{result['anomaly_score']:.3f}",
            help="Isolation Forest score. Higher means further from normal behaviour.",
        )
        if result["reasons"]:
            st.markdown("**Why**")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.markdown("No operating limit exceeded.")

with report_tab:
    metrics_path = REPORTS_DIR / "metrics.json"
    report = json.loads(metrics_path.read_text())
    m = report["test_metrics"]
    st.subheader(f"Selected model: {report['selected_model']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{m['recall']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c3.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    st.markdown("**Model comparison (5-fold cross-validation on the training set)**")
    st.dataframe(pd.DataFrame(report["model_comparison_cv"]).T, use_container_width=True)

    st.markdown("**Recall by failure mode (test set)**")
    st.dataframe(pd.DataFrame(report["recall_by_failure_mode"]).T, use_container_width=True)

    f1, f2 = st.columns(2)
    f1.image(str(FIGURES_DIR / "precision_recall.png"))
    f2.image(str(FIGURES_DIR / "feature_importance.png"))
    st.image(str(FIGURES_DIR / "confusion_matrix.png"), width=380)
