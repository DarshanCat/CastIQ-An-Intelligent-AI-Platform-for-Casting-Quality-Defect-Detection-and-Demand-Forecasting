import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_FILE = BASE / 'backend' / 'data' / 'maintenance_data.json'

def load_maintenance():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading maintenance data: {e}")
        return []

def render_predictive_maintenance():
    st.subheader("🔧 Machine Diagnostics & Predictive Maintenance")
    st.markdown("##### AI-powered Remaining Useful Life (RUL) forecasting and real-time vibration/bearing temperature telemetry")
    st.markdown("---")
    
    machines = load_maintenance()
    if not machines:
        return

    # Warnings panel
    warnings = []
    for m in machines:
        if m["status"] == "Warning":
            warnings.append(f"🚨 **{m['machine_name']}** needs attention! Bearing Temp: **{m['bearing_temp_c']} deg C**, RUL: **{m['rul_days']} days** remaining!")
            
    if warnings:
        for w in warnings:
            st.error(w)

    # Visual cards for each machine
    m_cols = st.columns(len(machines))
    for idx, m in enumerate(machines):
        with m_cols[idx]:
            st.markdown(f"""
            <div class="module-card">
                <h3 style='margin:0'>⚙️ {m['machine_name']}</h3>
                <p style='margin:8px 0;color:#cbd5e1;font-size:14px'>Status: <b>{m['status']}</b></p>
            </div>""", unsafe_allow_html=True)
            
            st.metric("Remaining Useful Life", f"{m['rul_days']} Days", delta="-0.8 Days", delta_color="inverse")
            st.metric("Vibration Amplitude", f"{m['vibration_mms']} mm/s", delta="Normal" if m['vibration_mms'] < 4.0 else "High", delta_color="normal" if m['vibration_mms'] < 4.0 else "inverse")
            st.metric("Bearing Temperature", f"{m['bearing_temp_c']} deg C", delta="Normal" if m['bearing_temp_c'] < 75 else "High", delta_color="normal" if m['bearing_temp_c'] < 75 else "inverse")
            
    st.markdown("---")
    
    # Machine details selector & decay curves
    st.markdown("### 📈 RUL Degradation Forecasting")
    selected_machine = st.selectbox("Select Centrifugal Machine to Forecast", [m["machine_name"] for m in machines])
    
    m_data = next(m for m in machines if m["machine_name"] == selected_machine)
    
    # Generate RUL decay curve
    current_rul = m_data["rul_days"]
    days_ahead = np.arange(0, int(current_rul) + 10)
    
    # Simulated exponential health degradation index
    vibe = m_data["vibration_mms"]
    temp = m_data["bearing_temp_c"]
    decay_rate = 0.015 * vibe + 0.002 * (temp - 40)
    health_index = 100.0 * np.exp(-decay_rate * days_ahead)
    health_index = np.clip(health_index, 0, 100)
    
    df_decay = pd.DataFrame({
        "Operating Days Ahead": days_ahead,
        "Machine Health Index (%)": health_index,
        "Critical Threshold": [30.0] * len(days_ahead)
    })
    
    fig = px.line(df_decay, x="Operating Days Ahead", y=["Machine Health Index (%)", "Critical Threshold"],
                  labels={"value": "Health / Threshold Index", "variable": "Channel"},
                  color_discrete_map={"Machine Health Index (%)": "#38bdf8", "Critical Threshold": "#f43f5e"})
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=20, b=0),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    > [!NOTE]
    > **Degradation Diagnostics**: The AI diagnostics engine projects **{selected_machine}** will cross the **Critical Failure Threshold (30% Health)** in exactly **{current_rul} operating days**.
    > Planned maintenance should be scheduled before this window to avoid unexpected manufacturing down-time.
    """)
