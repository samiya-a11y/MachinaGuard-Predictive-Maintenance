import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="MachinaGuard Industrial Health Monitor",
    page_icon="🛡️",
    layout="wide"
)

# Load Trained Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('machinaguard_model.pkl')
    scaler = joblib.load('machinaguard_scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# App Header
st.title("🛡️ MachinaGuard: IIoT Predictive Diagnostic Dashboard")
st.caption("AI-Powered Real-Time Milling Machine Telemetry & Breakdown Risk Forecaster")
st.markdown("---")

# Sidebar - Presets & Telemetry Controls
st.sidebar.header("🕹️ Simulation Presets")
preset = st.sidebar.selectbox(
    "Load Real-World Machine Scenario",
    ["Custom Input", "Optimal Baseline (Healthy)", "Severe Tool Friction (Failure)", "High Torque Overload (Failure)"]
)

# Set defaults based on preset selection
if preset == "Optimal Baseline (Healthy)":
    def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "Medium (M)", 298.1, 308.6, 1500, 38.0, 40
elif preset == "Severe Tool Friction (Failure)":
    def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "Low (L)", 301.0, 310.5, 1420, 52.0, 235
elif preset == "High Torque Overload (Failure)":
    def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "Low (L)", 302.5, 311.8, 1200, 68.0, 190
else:
    def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "Medium (M)", 300.0, 310.0, 1500, 40.0, 100

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Live Telemetry Feeds")
product_type = st.sidebar.selectbox(
    "Product Grade Type", 
    ["Low (L)", "Medium (M)", "High (H)"], 
    index=["Low (L)", "Medium (M)", "High (H)"].index(def_type)
)
air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 310.0, float(def_air), 0.1)
proc_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 320.0, float(def_proc), 0.1)
rot_speed = st.sidebar.number_input("Rotational Spindle Speed [rpm]", 1100, 2900, int(def_rpm), 10)
torque = st.sidebar.slider("Mechanical Torque [Nm]", 3.0, 80.0, float(def_torque), 0.5)
tool_wear = st.sidebar.slider("Cumulative Tool Wear [min]", 0, 260, int(def_wear), 1)

# Real-Time Feature Calculations
temp_diff = proc_temp - air_temp
power_factor = torque * rot_speed

# One-hot encoding values
type_m = 1 if product_type == "Medium (M)" else 0
type_l = 1 if product_type == "Low (L)" else 0

input_features = np.array([[
    air_temp, proc_temp, rot_speed, torque, tool_wear,
    type_m, type_l, temp_diff, power_factor
]])

# Top Row: Real-time Telemetry Metrics (KPI Cards)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Spindle Speed", f"{rot_speed} RPM")
col2.metric("Mechanical Torque", f"{torque} Nm")
col3.metric("Thermal Delta (ΔT)", f"{temp_diff:.2f} K")
col4.metric("Power Factor", f"{power_factor:,.0f} W-eq")

st.markdown("---")

# Prediction Execution
scaled_input = scaler.transform(input_features)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1] * 100

# Diagnostic Results Display
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.subheader("🔍 Automated Health Assessment")
    
    # Progress indicator for failure probability
    st.write(f"**Failure Probability Index:** {probability:.1f}%")
    st.progress(int(min(max(probability, 0), 100)))

    if prediction == 1:
        st.error("🚨 **CRITICAL ALERT: PREVENTIVE SHUTDOWN RECOMMENDED**")
        st.markdown(
            """
            * **Status:** Hardware operating in a high-failure zone.
            * **Primary Risk:** Mechanical fatigue or severe tool thermal breakdown.
            * **Action:** Disengage cutter feed immediately and inspect spindle assembly.
            """
        )
    else:
        st.success("✅ **STABLE STATE: MACHINERY OPERATIONAL**")
        st.markdown(
            """
            * **Status:** Telemetry within normal operational thresholds.
            * **Primary Risk:** None detected.
            * **Action:** Maintain planned production schedule.
            """
        )

with res_col2:
    st.subheader("📍 Operating Envelope Position")
    
    # 1. Dark background styling
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')   # Matches Streamlit outer dark background
    ax.set_facecolor('#1A1C24')          # Card surface background
    
    # 2. Text and axis colors
    ax.tick_params(colors='#E0E0E0', labelsize=9)
    ax.xaxis.label.set_color('#E0E0E0')
    ax.yaxis.label.set_color('#E0E0E0')
    ax.title.set_color('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_color('#444857')
    
    # 3. Danger zone and grid lines
    ax.axhspan(60, 80, color='#FF4B4B', alpha=0.25, label='High Torque Danger Zone')
    ax.grid(True, linestyle='--', color='#2D3139', alpha=0.8)
    
    # 4. Machine state marker
    point_color = '#FF4B4B' if prediction == 1 else '#00D26A'
    ax.scatter(rot_speed, torque, color=point_color, s=180, edgecolors='white', linewidth=1.8, zorder=5)
    ax.text(rot_speed + 35, torque + 1, "Current Machine State", 
            fontweight='bold', fontsize=9, color='#FFFFFF')
    
    # 5. Limits & labels
    ax.set_xlim(1100, 2900)
    ax.set_ylim(0, 85)
    ax.set_xlabel("Rotational Speed (RPM)", fontsize=9)
    ax.set_ylabel("Torque (Nm)", fontsize=9)
    ax.set_title("Current Machine vs. Safe Envelope", fontsize=11, pad=10)
    
    # 6. Legend styling
    legend = ax.legend(loc='upper right', fontsize=8, facecolor='#1A1C24', edgecolor='#444857')
    for text in legend.get_texts():
        text.set_color('#E0E0E0')
        
    st.pyplot(fig)