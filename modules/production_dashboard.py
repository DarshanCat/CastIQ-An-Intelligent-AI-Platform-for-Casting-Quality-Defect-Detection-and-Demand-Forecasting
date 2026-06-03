import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_FILE = BASE / 'backend' / 'data' / 'production_data.json'

def load_data():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return {}

def render_production_dashboard():
    st.subheader("🏭 Production & Yield Dashboard")
    st.markdown("##### Real-time centrifugal pipe manufacturing yields, melting efficiencies, and ladle status")
    st.markdown("---")
    
    data = load_data()
    if not data:
        return

    # Metric KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Output (Tonnes)", f"{data.get('total_tonnes_monthly', 0.0)} T", delta="+12.4 T")
    c2.metric("Melt Efficiency", f"{data.get('melt_efficiency_kwh_per_tonne', 0)} kWh/T", delta="-15 kWh/T", delta_color="inverse")
    c3.metric("Floor Scrap Rate", f"{data.get('scrap_rate_pct', 0.0)}%", delta="-0.4%", delta_color="inverse")

    st.markdown("---")
    
    # Visual graphs
    g1, g2 = st.columns([2, 1])
    
    with g1:
        st.markdown("### 📈 Daily Production vs Target")
        outputs = data.get("daily_outputs_tonnes", [])
        targets = data.get("daily_targets_tonnes", [])
        days = [f"Day {i+1}" for i in range(len(outputs))]
        
        df_plot = pd.DataFrame({
            "Day": days * 2,
            "Tonnes": outputs + targets,
            "Metric": ["Actual Output"] * len(outputs) + ["Target Output"] * len(targets)
        })
        
        fig = px.line(df_plot, x="Day", y="Tonnes", color="Metric", 
                      markers=True, line_shape="spline",
                      color_discrete_map={"Actual Output": "#38bdf8", "Target Output": "#f43f5e"})
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=0, r=0, t=20, b=0),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        st.markdown("### 📊 Defect Distribution")
        defect_data = pd.DataFrame({
            "Defect Type": ["Gas Blowholes", "Slag Inclusion", "Mold Cracks", "Shrinkage Cavities"],
            "Occurrences": [14, 8, 5, 3]
        })
        fig_pie = px.pie(defect_data, values="Occurrences", names="Defect Type",
                         color_discrete_sequence=px.colors.sequential.Blues)
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    
    # Recent heats list
    st.markdown("### 🧪 Recent Ladle Pours (Heats)")
    heats = data.get("recent_heats", [])
    if heats:
        df_heats = pd.DataFrame(heats)
        
        # Style status with color codes
        styled_rows = []
        for index, row in df_heats.iterrows():
            status = row['status']
            status_emoji = "🟢" if status == "Passed" else "🔴"
            styled_rows.append({
                "Heat ID": f"**{row['heat_id']}**",
                "Alloy Grade": row['grade'],
                "Pour Temperature (°C)": f"{row['pour_temp']} deg C",
                "Ladle Weight (kg)": f"{row['weight_kg']} kg",
                "Quality Status": f"{status_emoji} {status}"
            })
            
        st.dataframe(pd.DataFrame(styled_rows), use_container_width=True, hide_index=True)
