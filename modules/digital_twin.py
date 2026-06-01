"""
🌐 Digital Twin Module
Real-time physics simulation of ductile iron centrifugal casting process.
Animates: pour → fill → solidify → cool → eject
File: modules/digital_twin.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path

# ── Physics constants ─────────────────────────────────────────
RHO_IRON   = 7100     # kg/m³  density of ductile iron
CP_IRON    = 500      # J/kg·K specific heat
K_IRON     = 35       # W/m·K  thermal conductivity
L_FUSION   = 247000   # J/kg   latent heat of fusion

def compute_liquidus(carbon, silicon):
    """TL = 1550 - 28C - 8Si  (practical foundry formula)"""
    return 1550 - 28*carbon - 8*silicon

def compute_solidus(carbon, silicon):
    """TS ≈ TL - 30 for ductile iron"""
    return compute_liquidus(carbon, silicon) - 30

def compute_CE(carbon, silicon, phosphorus):
    return carbon + silicon/3 + phosphorus/3

def compute_G(rpm, diameter):
    return (np.pi**2 * diameter * rpm**2) / (900 * 9.81)

def compute_Mg_eff(mg_added, treat_time):
    return mg_added * np.exp(-0.02 * treat_time)

def simulate_temperature_profile(pour_temp, T_liq, T_sol, mold_preheat,
                                  wall_mm, mold_type, n_steps=100):
    """
    Simulate temperature vs time during solidification.
    Uses Newton's law of cooling + latent heat plateau model.
    """
    B_map   = {'Permanent': 0.10, 'Sand': 0.15, 'Die': 0.08}
    B       = B_map[mold_type]
    t_sol   = B * (wall_mm**2)            # Chvorinov solidification time (s)
    t_cool  = t_sol * 3                   # total cooling to eject temp
    t_total = t_sol * 5
    t       = np.linspace(0, t_total, n_steps)

    T = np.zeros(n_steps)
    T[0] = pour_temp

    for i in range(1, n_steps):
        dt = t[i] - t[i-1]
        Ti = T[i-1]

        # Cooling coefficient depends on mold type
        h_map = {'Permanent': 0.015, 'Sand': 0.008, 'Die': 0.022}
        h = h_map[mold_type]

        if Ti > T_liq:
            # Liquid cooling phase
            dT = -h * (Ti - mold_preheat) * dt
        elif T_sol <= Ti <= T_liq:
            # Mushy zone — latent heat slows cooling
            dT = -h * (Ti - mold_preheat) * dt * 0.25
        else:
            # Solid cooling
            dT = -h * (Ti - mold_preheat) * dt * 0.6

        T[i] = max(Ti + dT, mold_preheat + 5)

    # Find key timestamps
    t_liq_idx  = np.argmax(T <= T_liq) if any(T <= T_liq) else n_steps-1
    t_sol_idx  = np.argmax(T <= T_sol) if any(T <= T_sol) else n_steps-1

    return t, T, t_liq_idx, t_sol_idx

def compute_nodularity_evolution(mg_added, treat_time, n_steps=100):
    """Nodularity evolves as Mg fades during/after pour."""
    t = np.linspace(treat_time, treat_time + 15, n_steps)
    Mg_t = mg_added * np.exp(-0.02 * t)
    nodularity = np.clip(0.60 + (Mg_t - 0.045)*8, 0, 1)
    return t - treat_time, nodularity

def render_digital_twin():
    st.title("🌐 Digital Twin")
    st.markdown("##### Real-time physics simulation of your casting process")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Process Parameters")

        carbon    = st.slider("Carbon %",         3.2, 3.9, 3.5, 0.01)
        silicon   = st.slider("Silicon %",        1.8, 2.8, 2.2, 0.01)
        phosphorus= st.slider("Phosphorus %",     0.01,0.08,0.03,0.005)
        mg_added  = st.slider("Magnesium Added %",0.030,0.085,0.055,0.001)
        treat_t   = st.slider("Treatment Time (min)", 2.0, 12.0, 5.0, 0.5)
        rpm       = st.slider("RPM",              400, 1000, 700, 10)
        mold_dia  = st.slider("Mold Diameter (m)",0.05,0.20, 0.12, 0.01)
        pour_temp = st.slider("Pour Temp (°C)",   1490,1570, 1520, 1)
        mold_pre  = st.slider("Mold Preheat (°C)",100, 350,  200,  5)
        wall_mm   = st.slider("Wall Thickness (mm)",8,  70,   25,  1)
        mold_type = st.selectbox("Mold Type", ['Permanent','Sand','Die'])

        simulate_btn = st.button("▶️ Run Simulation", use_container_width=True, type="primary")

    with col2:
        T_liq = compute_liquidus(carbon, silicon)
        T_sol = compute_solidus(carbon, silicon)
        CE    = compute_CE(carbon, silicon, phosphorus)
        G     = compute_G(rpm, mold_dia)
        Mg    = compute_Mg_eff(mg_added, treat_t)
        SH    = pour_temp - T_liq

        # ── Live KPIs (update without clicking) ───────────────
        st.subheader("🔬 Live Physics State")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("CE",         f"{CE:.3f}",   "✅" if 4.3<=CE<=4.6   else "⚠️ OOR")
        k2.metric("G-Factor",   f"{G:.1f}G",   "✅" if 40<=G<=100     else "⚠️ OOR")
        k3.metric("Superheat",  f"{SH:.0f}°C", "✅" if 60<=SH<=110    else "⚠️ OOR")
        k4.metric("Mg Eff %",   f"{Mg*100:.3f}","✅" if 0.035<=Mg<=0.055 else "⚠️ OOR")

        k5,k6,k7,k8 = st.columns(4)
        k5.metric("Liquidus",   f"{T_liq:.0f}°C")
        k6.metric("Solidus",    f"{T_sol:.0f}°C")
        B_map = {'Permanent':0.10,'Sand':0.15,'Die':0.08}
        t_sol_est = B_map[mold_type] * wall_mm**2
        k7.metric("Solid. Time",f"{t_sol_est:.0f}s")
        nodularity = np.clip(0.60 + (Mg-0.045)*8, 0, 1)
        k8.metric("Nodularity", f"{nodularity:.2f}", "✅ Ductile" if nodularity>0.7 else "⚠️ Low")

        st.markdown("---")

        if simulate_btn:
            # ── Simulate ──────────────────────────────────────
            with st.spinner("Running simulation..."):
                t_arr, T_arr, t_liq_i, t_sol_i = simulate_temperature_profile(
                    pour_temp, T_liq, T_sol, mold_pre, wall_mm, mold_type
                )
                t_mg, nod_arr = compute_nodularity_evolution(mg_added, treat_t)

            # ── Temperature-Time Curve ────────────────────────
            st.subheader("🌡️ Temperature Profile — Cooling Curve")
            fig_temp = go.Figure()

            # Zones
            fig_temp.add_hrect(y0=T_liq, y1=pour_temp+20, fillcolor="rgba(239,68,68,0.1)",
                               line_width=0, annotation_text="Liquid Zone")
            fig_temp.add_hrect(y0=T_sol, y1=T_liq, fillcolor="rgba(245,158,11,0.1)",
                               line_width=0, annotation_text="Mushy Zone (Solidification)")
            fig_temp.add_hrect(y0=mold_pre, y1=T_sol, fillcolor="rgba(59,130,246,0.08)",
                               line_width=0, annotation_text="Solid Zone")

            fig_temp.add_trace(go.Scatter(x=t_arr, y=T_arr,
                mode='lines', line=dict(color='#f59e0b', width=3),
                name='Temperature'))

            # Mark phase transitions
            if t_liq_i < len(t_arr):
                fig_temp.add_vline(x=t_arr[t_liq_i], line_dash='dot',
                                   line_color='#ef4444',
                                   annotation_text=f'Liq start {t_arr[t_liq_i]:.0f}s')
            if t_sol_i < len(t_arr):
                fig_temp.add_vline(x=t_arr[t_sol_i], line_dash='dot',
                                   line_color='#3b82f6',
                                   annotation_text=f'Full solid {t_arr[t_sol_i]:.0f}s')

            fig_temp.add_hline(y=T_liq, line_dash='dash', line_color='#ef4444',
                               annotation_text=f'TL={T_liq:.0f}°C')
            fig_temp.add_hline(y=T_sol, line_dash='dash', line_color='#3b82f6',
                               annotation_text=f'TS={T_sol:.0f}°C')

            fig_temp.update_layout(
                height=300, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='Time (s)', gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title='Temperature (°C)', gridcolor='rgba(255,255,255,0.08)'),
                showlegend=False, margin=dict(t=20,b=40,l=10,r=100)
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # ── Nodularity Evolution ──────────────────────────
            st.subheader("⭕ Nodularity Index Evolution (Mg Fade)")
            fig_nod = go.Figure()
            fig_nod.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(34,197,94,0.1)",
                              line_width=0, annotation_text="✅ Good Nodularity Zone")
            fig_nod.add_trace(go.Scatter(x=t_mg, y=nod_arr,
                mode='lines', line=dict(color='#22c55e', width=3)))
            fig_nod.add_hline(y=0.7, line_dash='dash', line_color='#f59e0b',
                              annotation_text='Min acceptable (0.70)')
            fig_nod.update_layout(
                height=220, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='Time after treatment (min)',
                           gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title='Nodularity Index', range=[0,1.05],
                           gridcolor='rgba(255,255,255,0.08)'),
                showlegend=False, margin=dict(t=10,b=40,l=10,r=120)
            )
            st.plotly_chart(fig_nod, use_container_width=True)

            # ── G-Factor Radial Visualization ────────────────
            st.subheader("🔄 Centrifugal G-Force Distribution")
            r_vals = np.linspace(0, mold_dia/2, 50)
            G_vals = [(np.pi**2 * 2*r * rpm**2)/(900*9.81) for r in r_vals]
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=r_vals*1000, y=G_vals,
                mode='lines', fill='tozeroy',
                line=dict(color='#3b82f6', width=2),
                fillcolor='rgba(59,130,246,0.2)'))
            fig_g.add_hrect(y0=40, y1=100, fillcolor="rgba(34,197,94,0.15)",
                            line_width=0, annotation_text="Optimal 40–100G")
            fig_g.update_layout(
                height=200, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='Radius (mm)', gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title='G-Factor', gridcolor='rgba(255,255,255,0.08)'),
                showlegend=False, margin=dict(t=10,b=40,l=10,r=120)
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # ── Summary Report ────────────────────────────────
            st.subheader("📋 Simulation Summary")
            solid_time = t_arr[t_sol_i] if t_sol_i < len(t_arr) else t_sol_est
            quality_est = np.clip(
                100 - abs(CE-4.45)*15 - max(0,abs(SH-85)-25)*0.3
                    - max(0,abs(G-70)-30)*0.2 - max(0,(0.045-Mg)*300)
                    + nodularity*6, 0, 100
            )
            st.success(f"""
**Simulation complete!**
- 🕒 Solidification time: **{solid_time:.0f} seconds**
- ⭕ Final nodularity: **{nod_arr[-1]:.2f}** ({'✅ Good' if nod_arr[-1]>0.7 else '⚠️ Low'})
- 🌡️ Mushy zone duration: **{(t_arr[t_sol_i]-t_arr[t_liq_i]):.0f}s** (shorter = finer microstructure)
- 🎯 Estimated quality score: **{quality_est:.1f}/100**
            """)
        else:
            st.info("👈 Set parameters and click **▶️ Run Simulation** to see the casting process modelled in real time")
            st.markdown("""
**What the Digital Twin simulates:**
- 🌡️ **Cooling curve** — temperature vs time through liquid → mushy → solid phases
- ⭕ **Nodularity decay** — how Mg fades after treatment (affects graphite shape)
- 🔄 **G-force distribution** — centrifugal force across mold radius
- 📋 **Summary report** — solidification time, nodularity, estimated quality

**Physics used:**
- Newton's law of cooling with latent heat plateau
- Chvorinov's Rule for solidification time
- Mg fade exponential model
- Centrifugal G-factor formula
            """)
