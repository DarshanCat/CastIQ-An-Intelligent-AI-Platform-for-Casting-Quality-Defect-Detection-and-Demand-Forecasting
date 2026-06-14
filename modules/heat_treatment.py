import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_heat_treatment():
    st.subheader("🔥 Heat Treatment & Phase Simulator")
    st.markdown("##### Metallurgical heat treatment cycle simulation - predicts microstructural phase changes and final mechanical properties")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Thermal Cycle Configuration")
        process = st.selectbox("Heat Treatment Process", 
                               ["Annealing (Ferritizing)", 
                                "Normalizing (Pearlitizing)", 
                                "Tempering (Ductilizing)"])
        
        # Default sliders based on process
        if "Annealing" in process:
            def_temp = 900
            def_hold = 2.0
            def_cool = 20.0
        elif "Normalizing" in process:
            def_temp = 920
            def_hold = 2.0
            def_cool = 150.0
        else:
            def_temp = 600
            def_hold = 3.0
            def_cool = 120.0
            
        temp = st.slider("Target Heat Temperature (°C)", 400, 1100, def_temp, step=10)
        hold_time = st.slider("Holding Duration (Hours)", 0.5, 6.0, def_hold, step=0.5)
        cool_rate = st.slider("Cooling Rate (°C/Hour)", 10.0, 300.0, def_cool, step=10.0)
        
        sim_btn = st.button("🔥 Run Metallurgical Simulation", width="stretch", type="primary")

    with col2:
        st.subheader("📊 Thermal Cycle Curve")
        
        # Generate simulation curve points
        # Stage 1: Heating (2 hours to temp)
        # Stage 2: Holding (hold_time hours at temp)
        # Stage 3: Cooling (from temp down to 100 deg C at cool_rate)
        heating_duration = 2.0
        cooling_duration = (temp - 100.0) / cool_rate
        
        t_heat = np.linspace(0, heating_duration, 20)
        temp_heat = np.linspace(25, temp, 20)
        
        t_hold = np.linspace(heating_duration, heating_duration + hold_time, 20)
        temp_hold = np.full(20, temp)
        
        t_cool = np.linspace(heating_duration + hold_time, heating_duration + hold_time + cooling_duration, 20)
        temp_cool = np.linspace(temp, 100, 20)
        
        times = np.concatenate([t_heat, t_hold, t_cool])
        temps = np.concatenate([temp_heat, temp_hold, temp_cool])
        
        df_curve = pd.DataFrame({
            "Time (Hours)": times,
            "Temperature (°C)": temps,
            "Cycle Stage": ["Heating"]*20 + ["Holding"]*20 + ["Cooling"]*20
        })
        
        fig = px.line(df_curve, x="Time (Hours)", y="Temperature (°C)", color="Cycle Stage",
                      line_shape="linear",
                      color_discrete_map={"Heating": "#f59e0b", "Holding": "#ef4444", "Cooling": "#3b82f6"})
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=0, r=0, t=20, b=0),
            height=300
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    
    # Mathematical microstructural calculations
    st.subheader("🔬 Simulated Microstructure & Mechanical Verdict")
    
    # Pearlite percentage increases with faster cooling and higher normalizing temperatures
    # Soft Ferrite increases with slow annealing cooling rates
    if cool_rate < 35.0:
        # Full annealing
        ferrite = 95.0 - (cool_rate / 10.0)
        pearlite = 100.0 - ferrite
    elif cool_rate > 200.0:
        # Fast cooling
        pearlite = 85.0 + (temp / 1000.0)
        ferrite = 100.0 - pearlite
    else:
        # Normalizing
        pearlite = 40.0 + (cool_rate / 3.0)
        ferrite = 100.0 - pearlite
        
    ferrite = np.clip(ferrite, 2, 98)
    pearlite = np.clip(pearlite, 2, 98)
    
    # Calculate mechanical properties based on composite rule of mixtures
    # Pure Ferrite: tensile = 350 MPa, hardness = 120 BHN
    # Pure Pearlite: tensile = 780 MPa, hardness = 250 BHN
    predicted_tensile = (ferrite / 100.0) * 350 + (pearlite / 100.0) * 780
    predicted_hardness = (ferrite / 100.0) * 120 + (pearlite / 100.0) * 250
    predicted_elongation = (ferrite / 100.0) * 22 + (pearlite / 100.0) * 8
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Ferrite Phase", f"{ferrite:.1f}%", delta="Soft ductile matrix")
    m2.metric("Predicted Pearlite Phase", f"{pearlite:.1f}%", delta="Hard high-strength matrix")
    m3.metric("Predicted Tensile Strength", f"{predicted_tensile:.0f} MPa", delta=f"{predicted_elongation:.1f}% Elongation")
    m4.metric("Predicted Hardness", f"{predicted_hardness:.0f} BHN", delta="Brinell Hardness Value")

    st.markdown("---")
    st.markdown("### 🧬 Microstructural Phase Composite Breakdown")
    df_pie = pd.DataFrame({
        "Microstructural Phase": ["Ferrite (Ductile)", "Pearlite (Strong)"],
        "Composition (%)": [ferrite, pearlite]
    })
    fig_pie = px.pie(df_pie, values="Composition (%)", names="Microstructural Phase",
                     color_discrete_sequence=["#38bdf8", "#818cf8"])
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=0),
        height=260
    )
    st.plotly_chart(fig_pie, width="stretch")
