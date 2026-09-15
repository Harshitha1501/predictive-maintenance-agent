"""
Predictive Maintenance Agent — Industrial AI Operations Center
Phase 5 Deliverable: Frontend UI & QA Dashboard
Lead: Neethu (Frontend UI & QA Lead)
Dataset: AI4I 2020 Predictive Maintenance Benchmark
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import altair as alt
import pandas as pd
import requests
import streamlit as st

# Add project root to sys.path so imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import SensorInput
from src.xai.agent import PredictiveMaintenanceAgent

# Set page configuration
st.set_page_config(
    page_title="AI4I Predictive Maintenance Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Industrial Dark/Modern Theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .status-card-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.22));
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }
    .status-card-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .action-card {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .physics-card {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Singleton Agent Instance
# -----------------------------------------------------------------------------
@st.cache_resource
def load_agent() -> PredictiveMaintenanceAgent:
    """Instantiates the unified AI agent once per session."""
    return PredictiveMaintenanceAgent()


# -----------------------------------------------------------------------------
# Preset Benchmark Scenarios
# -----------------------------------------------------------------------------
PRESET_SCENARIOS = {
    "Healthy Machine (Nominal)": {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 12.0,
        "description": "All telemetry operating in normal range with minimal tool degradation.",
    },
    "Overstrain Failure (OSF)": {
        "type": "L",
        "air_temperature": 298.4,
        "process_temperature": 308.2,
        "rotational_speed": 1282.0,
        "torque": 60.7,
        "tool_wear": 216.0,
        "description": "High torque applied to an already degraded cutting tool exceeding strain limits.",
    },
    "Heat Dissipation Failure (HDF)": {
        "type": "M",
        "air_temperature": 302.4,
        "process_temperature": 310.2,
        "rotational_speed": 1332.0,
        "torque": 52.3,
        "tool_wear": 142.0,
        "description": "Low temperature gradient (ΔT < 8.6K) at low spindle speed causing thermal buildup.",
    },
    "Power Failure (PWF)": {
        "type": "L",
        "air_temperature": 298.9,
        "process_temperature": 309.1,
        "rotational_speed": 2861.0,
        "torque": 4.6,
        "tool_wear": 143.0,
        "description": "Motor drive anomaly outside safe continuous operating envelope [3500W, 9000W].",
    },
    "Tool Wear Failure (TWF)": {
        "type": "L",
        "air_temperature": 298.8,
        "process_temperature": 308.9,
        "rotational_speed": 1455.0,
        "torque": 41.3,
        "tool_wear": 208.0,
        "description": "Cumulative tool wear exceeds physical durability limit (>= 200 min).",
    },
}

# -----------------------------------------------------------------------------
# Session State Initialization (Requirement 4)
# -----------------------------------------------------------------------------
if "telemetry" not in st.session_state:
    st.session_state.telemetry = PRESET_SCENARIOS["Healthy Machine (Nominal)"].copy()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "selected_preset_name" not in st.session_state:
    st.session_state.selected_preset_name = "Healthy Machine (Nominal)"


# -----------------------------------------------------------------------------
# Inference Execution Helper (Session State Caching)
# -----------------------------------------------------------------------------
def run_inference(telemetry_data: Dict[str, Any], execution_mode: str) -> Dict[str, Any]:
    """Executes inference via Direct Agent or FastAPI, caching in st.session_state."""
    if execution_mode == "FastAPI REST Service":
        try:
            resp = requests.post(
                f"{st.session_state.api_url}/api/v1/predict",
                json=telemetry_data,
                timeout=5.0,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
                return None
        except requests.exceptions.RequestException as ex:
            st.warning(f"FastAPI server unreachable at {st.session_state.api_url}. Falling back to Direct Agent.")
            agent = load_agent()
            return agent.analyze(telemetry_data)
    else:
        agent = load_agent()
        return agent.analyze(telemetry_data)


def safe_read_csv(file_or_path: Any) -> Optional[pd.DataFrame]:
    """Safely reads a CSV from a file-like buffer or filepath, rewinding buffer pointer if needed."""
    if file_or_path is None:
        return None
    try:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path)
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        return df
    except Exception as ex:
        st.error(f"Error reading CSV file: {ex}")
        return None


# -----------------------------------------------------------------------------
# Sidebar: Controls & Live Telemetry Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Operations Panel")
    st.caption("AI4I 2020 Predictive Maintenance System")

    st.markdown("### 🔌 Execution Mode")
    exec_mode = st.radio(
        "Inference Engine",
        options=["Direct Intelligent Agent", "FastAPI REST Service"],
        index=0,
        help="Direct Agent executes the local embedded ML & SHAP models. FastAPI queries Trusha's REST service.",
    )

    # Health check for API mode
    if exec_mode == "FastAPI REST Service":
        st.session_state.api_url = st.text_input("API Base URL", value=st.session_state.api_url)
        try:
            h_resp = requests.get(f"{st.session_state.api_url}/api/v1/health", timeout=2.0)
            if h_resp.status_code == 200:
                st.success("🟢 FastAPI Server Online")
            else:
                st.error("🔴 Server Status: Degraded")
        except Exception:
            st.warning("⚠️ FastAPI Server Offline (Will Fallback)")

    st.divider()

    has_uploaded_file = st.session_state.get("uploaded_file") is not None

    if has_uploaded_file:
        st.markdown("### 🎯 1-Click Test Scenarios")
        preset_choice = st.selectbox(
            "Load Known Machine State",
            options=list(PRESET_SCENARIOS.keys()),
            index=list(PRESET_SCENARIOS.keys()).index(st.session_state.selected_preset_name),
        )

        if st.button("Apply Selected Scenario", use_container_width=True):
            st.session_state.selected_preset_name = preset_choice
            st.session_state.telemetry = {
                k: v for k, v in PRESET_SCENARIOS[preset_choice].items() if k != "description"
            }
            st.session_state.user_tuned = True
            st.session_state.prediction_result = run_inference(st.session_state.telemetry, exec_mode)
            st.rerun()

        st.caption(f"ℹ️ {PRESET_SCENARIOS[preset_choice]['description']}")

        st.divider()

        st.markdown("### 🎛️ Live Sensor Sliders")

        curr = st.session_state.telemetry

        p_type = st.selectbox("Product Quality Variant (Type)", options=["L", "M", "H"], index=["L", "M", "H"].index(curr.get("type", "L")))
        air_t = st.slider("Air Temperature [K]", min_value=280.0, max_value=320.0, value=float(curr.get("air_temperature", 298.2)), step=0.1)
        proc_t = st.slider("Process Temperature [K]", min_value=290.0, max_value=330.0, value=float(curr.get("process_temperature", 308.6)), step=0.1)
        rpm = st.slider("Rotational Speed [rpm]", min_value=1000.0, max_value=3000.0, value=float(curr.get("rotational_speed", 1500.0)), step=10.0)
        torq = st.slider("Torque [Nm]", min_value=0.0, max_value=100.0, value=float(curr.get("torque", 40.0)), step=0.5)
        wear = st.slider("Tool Wear [min]", min_value=0.0, max_value=300.0, value=float(curr.get("tool_wear", 0.0)), step=1.0)

        # Update session state telemetry
        st.session_state.telemetry = {
            "type": p_type,
            "air_temperature": air_t,
            "process_temperature": proc_t,
            "rotational_speed": rpm,
            "torque": torq,
            "tool_wear": wear,
        }

        if st.button("⚡ Run Health Analysis", type="primary", use_container_width=True):
            st.session_state.user_tuned = True
            st.session_state.prediction_result = run_inference(st.session_state.telemetry, exec_mode)
            st.rerun()
    else:
        st.markdown("### 📁 Awaiting Telemetry Upload")
        st.caption("Upload a CSV file in the main view or Batch Fleet tab to activate live parameter tuning and 1-click test presets.")


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_live, tab_batch = st.tabs([
    "📊 Live Machine Diagnostics",
    "📁 Batch Fleet Telemetry",
])

# -----------------------------------------------------------------------------
# Tab 1: Live Machine Diagnostics
# -----------------------------------------------------------------------------
with tab_live:
    st.markdown('<div class="main-header">Predictive Maintenance AI Operations Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Sensor Telemetry, Multi-Class Failure Attribution & Prescriptive Action Planner</div>', unsafe_allow_html=True)

    # Primary CSV File Uploader Gatekeeper
    uploaded_file = st.file_uploader(
        "Upload Sensor Telemetry CSV to Initialize Live Diagnostics",
        type=["csv"],
        key="main_tab_csv_uploader",
        help="Upload an AI4I 2020 telemetry batch or single-machine CSV file to initialize analysis.",
    )

    # Sync with batch uploader if available
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    elif st.session_state.get("uploaded_file") is not None:
        uploaded_file = st.session_state.get("uploaded_file")

    if uploaded_file is not None:
        df_live = safe_read_csv(uploaded_file)
        if df_live is None or df_live.empty:
            st.error("Uploaded CSV file is empty or could not be parsed.")
            st.stop()

        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            row_idx = st.number_input(
                f"Select Machine Record from Uploaded CSV (Row 0 to {len(df_live) - 1})",
                min_value=0,
                max_value=len(df_live) - 1,
                value=0,
                step=1,
            )
        with col_sel2:
            file_name = getattr(uploaded_file, "name", "ai4i2020.csv")
            st.caption(f"📁 **{file_name}** ({len(df_live):,} total records)")

        selected_row = df_live.iloc[int(row_idx)].to_dict()
        agent_instance = load_agent()
        norm_dict = agent_instance.preprocessor.normalize_keys(selected_row)
        current_telemetry = {
            "type": str(norm_dict.get("Type", "L")),
            "air_temperature": float(norm_dict.get("Air temperature [K]", 298.2)),
            "process_temperature": float(norm_dict.get("Process temperature [K]", 308.6)),
            "rotational_speed": float(norm_dict.get("Rotational speed [rpm]", 1500.0)),
            "torque": float(norm_dict.get("Torque [Nm]", 40.0)),
            "tool_wear": float(norm_dict.get("Tool wear [min]", 0.0)),
        }

        if "telemetry" in st.session_state and st.session_state.get("user_tuned"):
            current_telemetry = st.session_state.telemetry
        else:
            st.session_state.telemetry = current_telemetry

        res = run_inference(current_telemetry, exec_mode)

        if res is not None:
            is_failure = (res.get("prediction") == "Machine Failure")
            fail_prob = float(res.get("failure_probability", 0.0))
            fail_type = res.get("failure_type", "No Failure")
            risk_lvl = res.get("risk_level", "Low")
            alert = res.get("alert", False)

        # 1. Primary Hero Status Card
        if is_failure:
            st.markdown(
                f"""
                <div class="status-card-danger">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="color: #ef4444; margin: 0;">🚨 CRITICAL ALERT: {fail_type.upper()} DETECTED</h2>
                            <p style="margin: 0.4rem 0 0 0; color: #fca5a5; font-size: 1.05rem;">
                                Equipment condition has breached physical reliability limits. Immediate intervention required.
                            </p>
                        </div>
                        <div>
                            <span class="badge-critical" style="font-size: 1.1rem; padding: 8px 16px;">FAIL PROBABILITY: {fail_prob*100:.1f}%</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card-healthy">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="color: #10b981; margin: 0;">✅ MACHINE HEALTH: NOMINAL / OPTIMAL</h2>
                            <p style="margin: 0.4rem 0 0 0; color: #6ee7b7; font-size: 1.05rem;">
                                All operating parameters and physical stress mechanics are within standard design thresholds.
                            </p>
                        </div>
                        <div>
                            <span class="badge-low" style="font-size: 1.1rem; padding: 8px 16px;">FAIL PROBABILITY: {fail_prob*100:.1f}%</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. KPI Metrics Grid
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="color: #9ca3af; font-size: 0.85rem;">BINARY PREDICTION</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {'#ef4444' if is_failure else '#10b981'};">
                        {res.get('prediction', 'Unknown')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="color: #9ca3af; font-size: 0.85rem;">FAILURE MODE</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #f59e0b;">
                        {fail_type}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi3:
            badge_class = f"badge-{risk_lvl.lower()}"
            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="color: #9ca3af; font-size: 0.85rem;">SYSTEM RISK LEVEL</div>
                    <div style="margin-top: 6px;">
                        <span class="{badge_class}">{risk_lvl.upper()}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi4:
            phys = res.get("physical_telemetry", {})
            power_w = phys.get("mechanical_power_watts", 0.0) if phys else 0.0
            st.markdown(
                f"""
                <div class="metric-container">
                    <div style="color: #9ca3af; font-size: 0.85rem;">MECHANICAL POWER</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8;">
                        {power_w:.0f} W
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Two-Column Diagnostic & Reasoning Section
        col_xai, col_rules = st.columns([1, 1])

        with col_xai:
            st.markdown("### 🧠 Explainable AI (SHAP Attribution)")
            st.caption("Top sensory factors driving the model's failure probability prediction")

            top_factors = res.get("top_factors", [])
            detailed_factors = res.get("detailed_factors", [])

            if detailed_factors:
                df_factors = pd.DataFrame(detailed_factors)
                chart = (
                    alt.Chart(df_factors)
                    .mark_bar(cornerRadius=4)
                    .encode(
                        x=alt.X("shap_value:Q", title="SHAP Importance (Impact on Prediction)"),
                        y=alt.Y("feature:N", sort="-x", title="Sensor Feature"),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(
                                domain=["Increases Risk", "Decreases Risk", "Neutral"],
                                range=["#ef4444", "#10b981", "#6b7280"],
                            ),
                            title="Attribution",
                        ),
                        tooltip=["feature", "value", "shap_value", "impact", "direction"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)
            elif top_factors:
                df_simple = pd.DataFrame(top_factors)
                st.dataframe(df_simple, use_container_width=True, hide_index=True)
            else:
                st.info("SHAP attribution operating with nominal weights.")

        with col_rules:
            st.markdown("### 🔬 Physical & Root-Cause Diagnosis")
            st.caption("Deterministic engineering mechanics from AI4I 2020 domain specifications")

            phys = res.get("physical_telemetry", {})
            if phys:
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.metric("Temp Differential (ΔT)", f"{phys.get('temperature_difference_k', 0.0):.1f} K", help="Threshold for HDF: ΔT < 8.6 K and Speed < 1380 RPM")
                with p_c2:
                    wear_torque = phys.get("overstrain_product_min_nm", 0.0)
                    threshold = phys.get("overstrain_threshold", 11000.0)
                    delta_strain = wear_torque - threshold
                    st.metric(
                        "Overstrain Product",
                        f"{wear_torque:.0f} min·Nm",
                        delta=f"{delta_strain:+.0f} vs Limit ({threshold:.0f})",
                        delta_color="inverse",
                        help="Threshold for OSF: Tool Wear × Torque > Variant Limit",
                    )

            st.markdown(
                f"""
                <div class="physics-card">
                    <strong style="color: #f59e0b;">Physics Root-Cause Explanation:</strong><br>
                    {res.get('root_cause', 'Nominal operations.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. Prescriptive Action Protocol
        st.markdown("### 🛠️ Prescriptive Maintenance Action Protocol")
        recommendation = res.get("recommendation", "Continue standard operation.")
        st.markdown(
            f"""
            <div class="action-card">
                <div style="font-size: 1.05rem; font-weight: 600; color: #60a5fa; margin-bottom: 0.3rem;">
                    RECOMMENDED TECHNICIAN PROCEDURE:
                </div>
                <div style="font-size: 1.05rem; color: #e2e8f0; line-height: 1.5;">
                    {recommendation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Warnings Banner
        warnings = res.get("warnings", [])
        if warnings:
            with st.expander("⚠️ Operational Parameter Boundary Alerts"):
                for w in warnings:
                    st.warning(w)
    else:
        st.info("📂 Please upload an telemetry CSV file in the Batch Fleet tab or main view to initialize predictive maintenance analysis.")


# -----------------------------------------------------------------------------
# Tab 2: Batch Fleet Telemetry (CSV Upload)
# -----------------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📁 Batch Fleet Machine Analysis")
    st.write("Upload an AI4I 2020 telemetry batch dataset (CSV) to analyze entire fleets of machinery simultaneously.")

    default_csv_path = PROJECT_ROOT / "data" / "ai4i2020.csv"
    batch_uploaded = st.file_uploader("Upload Sensor Telemetry CSV", type=["csv"], key="batch_csv_uploader")

    target_df = None
    if batch_uploaded is not None:
        st.session_state.uploaded_file = batch_uploaded
        target_df = safe_read_csv(batch_uploaded)
    elif st.session_state.get("uploaded_file") is not None:
        target_df = safe_read_csv(st.session_state.get("uploaded_file"))
    elif default_csv_path.exists():
        if st.button("Load Benchmark Dataset (`data/ai4i2020.csv`)"):
            st.session_state.uploaded_file = default_csv_path
            target_df = safe_read_csv(default_csv_path)
            st.rerun()

    if target_df is None:
        st.info("📂 Please upload an telemetry CSV file in the Batch Fleet tab or main view to initialize predictive maintenance analysis.")
    else:
        st.write(f"Loaded **{len(target_df):,}** machine records.")
        sample_size = st.slider("Select Batch Inspection Sample Size", min_value=10, max_value=min(500, len(target_df)), value=50, step=10)

        if st.button("🚀 Process Batch Fleet with Intelligent Agent"):
            agent = load_agent()
            subset = target_df.head(sample_size).copy()

            progress_bar = st.progress(0)
            predictions = []
            probabilities = []
            failure_modes = []
            risk_levels = []

            for i, (_, row) in enumerate(subset.iterrows()):
                row_dict = row.to_dict()
                analysis = agent.analyze(row_dict)
                predictions.append(analysis["prediction"])
                probabilities.append(round(analysis["failure_probability"], 3))
                failure_modes.append(analysis["failure_type"])
                risk_levels.append(analysis["risk_level"])
                progress_bar.progress((i + 1) / len(subset))

            subset["Agent Prediction"] = predictions
            subset["Fail Prob"] = probabilities
            subset["Failure Mode"] = failure_modes
            subset["Risk Level"] = risk_levels

            progress_bar.empty()

            # Display Batch Summary
            n_failures = (subset["Agent Prediction"] == "Machine Failure").sum()
            b1, b2, b3 = st.columns(3)
            b1.metric("Machines Analyzed", len(subset))
            b2.metric("Failures Detected", n_failures, delta=f"{n_failures/len(subset)*100:.1f}% Failure Rate", delta_color="inverse")
            b3.metric("Normal Machines", len(subset) - n_failures)

            # Failure Breakdown Chart
            st.markdown("#### Failure Distribution in Batch")
            counts = subset["Failure Mode"].value_counts().reset_index()
            counts.columns = ["Failure Mode", "Count"]
            b_chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Failure Mode:N", sort="-x"),
                    color=alt.Color("Failure Mode:N", legend=None),
                    tooltip=["Failure Mode", "Count"],
                )
                .properties(height=220)
            )
            st.altair_chart(b_chart, use_container_width=True)

            # Filter table by Risk Level
            st.markdown("#### Machine Records Table")
            show_only_failures = st.checkbox("Show Only Flagged Anomalies / Failures", value=True)
            if show_only_failures:
                display_df = subset[subset["Agent Prediction"] == "Machine Failure"]
            else:
                display_df = subset

            st.dataframe(display_df, use_container_width=True)

            # CSV Download
            csv_data = subset.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Annotated Batch Predictions (CSV)",
                data=csv_data,
                file_name="predictive_maintenance_batch_results.csv",
                mime="text/csv",
            )

