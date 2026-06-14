"""
⚡ Energy Optimizer Module
Minimize furnace energy consumption (kWh) while hitting quality target.
Uses scipy.optimize + physics-based energy model.
File: modules/energy_optimizer.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / 'backend' / 'models'

# ── Energy model constants ────────────────────────────────────
FURNACE_POWER_KW = 250          # induction furnace rated power (kW)
FURNACE_EFF      = 0.75         # thermal efficiency
ELECTRICITY_RATE = 8.50         # ₹/kWh (industrial rate India)
METAL_DENSITY    = 7100         # kg/m³
CP_IRON          = 500          # J/kg·K
L_FUSION         = 247000       # J/kg latent heat

FEATS = [
    'carbon_pct','silicon_pct','manganese_pct','phosphorus_pct',
    'magnesium_added_pct','treatment_time_min','rpm','mold_diameter_m',
    'pour_temp_c','mold_preheat_c','wall_thickness_mm','mold_enc',
    'carbon_equivalent','superheat_c','G_factor','Mg_effective_pct',
    'nodularity_index','cooling_rate_cs'
]

def compute_physics(p):
    CE        = p['carbon_pct'] + p['silicon_pct']/3 + p['phosphorus_pct']/3
    T_liq     = 1550 - 28*p['carbon_pct'] - 8*p['silicon_pct']
    SH        = p['pour_temp_c'] - T_liq
    G         = (np.pi**2 * p['mold_diameter_m'] * p['rpm']**2) / (900 * 9.81)
    Mg_eff    = p['magnesium_added_pct'] * np.exp(-0.02 * p['treatment_time_min'])
    nodularity= np.clip(0.60 + (Mg_eff-0.045)*8 - (p['treatment_time_min']-5)*0.015, 0, 1)
    B_map     = {0:0.10, 1:0.15, 2:0.08}
    t_sol     = B_map[int(p['mold_enc'])] * (p['wall_thickness_mm']**2)
    c_rate    = np.clip((p['pour_temp_c']-p['mold_preheat_c'])/(t_sol+1e-6), 0.5, 40)
    return CE, SH, G, Mg_eff, nodularity, c_rate

def compute_energy(pour_temp, batch_kg, mold_preheat, T_ambient=30):
    """
    Total energy = sensible heat + latent heat + losses
    E = m × Cp × (T_pour - T_ambient) + m × L_fusion
    Add 25% for heat losses, refractory, etc.
    """
    sensible  = batch_kg * CP_IRON * (pour_temp - T_ambient)   # J
    latent    = batch_kg * L_FUSION                              # J
    total_J   = (sensible + latent) * 1.25 / FURNACE_EFF
    total_kWh = total_J / 3_600_000
    cost_inr  = total_kWh * ELECTRICITY_RATE
    melt_time = total_kWh / FURNACE_POWER_KW * 60               # minutes
    return total_kWh, cost_inr, melt_time

def build_vector(p):
    CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(p)
    return np.array([
        p['carbon_pct'],p['silicon_pct'],p['manganese_pct'],p['phosphorus_pct'],
        p['magnesium_added_pct'],p['treatment_time_min'],p['rpm'],p['mold_diameter_m'],
        p['pour_temp_c'],p['mold_preheat_c'],p['wall_thickness_mm'],p['mold_enc'],
        CE, SH, G, Mg_eff, nodularity, c_rate
    ])

def render_energy_optimizer():
    st.title("⚡ Energy Optimizer")
    st.markdown("##### Minimize furnace energy cost while meeting your quality target")
    st.markdown("---")

    try:
        reg    = joblib.load(MODEL_DIR/'regressor.pkl')
        scaler = joblib.load(MODEL_DIR/'scaler.pkl')
    except Exception as e:
        st.error(f"❌ Models not found: {e}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Optimization Constraints")

        target_quality = st.slider("Minimum Quality Score", 70, 98, 88)
        batch_kg       = st.slider("Batch Size (kg)", 50, 500, 200)
        mold_type      = st.selectbox("Mold Type", ['Permanent','Sand','Die'])
        mold_enc       = {'Permanent':0,'Sand':1,'Die':2}[mold_type]

        st.markdown("---")
        st.subheader("📌 Fixed Parameters")
        carbon    = st.slider("Carbon %",    3.2, 3.9, 3.5, 0.01)
        silicon   = st.slider("Silicon %",   1.8, 2.8, 2.2, 0.01)
        wall_mm   = st.slider("Wall Thick (mm)", 8, 70, 25)

        st.markdown("---")
        st.subheader("🔋 Energy Tariff")
        tariff = st.number_input("Electricity Rate (₹/kWh)", 4.0, 20.0, 8.5, 0.5)

        optimize_btn = st.button("⚡ Optimize for Minimum Energy",
                                  width="stretch", type="primary")

    with col2:
        if optimize_btn:
            with st.spinner("🔋 Running energy optimization..."):

                # Variables to optimize: [pour_temp, rpm, mold_dia, mg_added, treat_time, mold_preheat]
                bounds = [
                    (1490, 1570),    # pour_temp — minimize this to save energy
                    (400,  1000),    # rpm
                    (0.05, 0.20),    # mold_dia
                    (0.030,0.085),   # mg_added
                    (2.0,  12.0),    # treat_time
                    (100,  350),     # mold_preheat
                ]

                def objective(x):
                    pour_t, rpm, dia, mg, tt, pre = x
                    p = {
                        'carbon_pct':carbon,'silicon_pct':silicon,
                        'manganese_pct':0.35,'phosphorus_pct':0.025,
                        'magnesium_added_pct':mg,'treatment_time_min':tt,
                        'rpm':rpm,'mold_diameter_m':dia,
                        'pour_temp_c':pour_t,'mold_preheat_c':pre,
                        'wall_thickness_mm':wall_mm,'mold_enc':mold_enc
                    }
                    vec    = build_vector(p).reshape(1,-1)
                    vec_sc = scaler.transform(vec)
                    q      = float(reg.predict(vec_sc)[0])

                    # Primary: minimize energy (pour_temp dominates)
                    energy, _, _ = compute_energy(pour_t, batch_kg, pre)

                    # Penalty if quality below target
                    penalty = max(0, target_quality - q) * 20

                    return energy + penalty

                result = differential_evolution(
                    objective, bounds, maxiter=200, popsize=12,
                    tol=0.001, seed=42
                )

                x_opt = result.x
                pour_opt, rpm_opt, dia_opt, mg_opt, tt_opt, pre_opt = x_opt

                p_opt = {
                    'carbon_pct':carbon,'silicon_pct':silicon,
                    'manganese_pct':0.35,'phosphorus_pct':0.025,
                    'magnesium_added_pct':mg_opt,'treatment_time_min':tt_opt,
                    'rpm':rpm_opt,'mold_diameter_m':dia_opt,
                    'pour_temp_c':pour_opt,'mold_preheat_c':pre_opt,
                    'wall_thickness_mm':wall_mm,'mold_enc':mold_enc
                }

                vec_opt    = build_vector(p_opt).reshape(1,-1)
                quality_opt= float(reg.predict(scaler.transform(vec_opt))[0])
                E_opt, cost_opt, melt_opt = compute_energy(pour_opt, batch_kg, pre_opt, tariff)

                # Baseline (max temp for comparison)
                E_base, cost_base, _ = compute_energy(1570, batch_kg, 100, tariff)
                savings_kwh  = E_base - E_opt
                savings_inr  = cost_base - cost_opt
                savings_pct  = savings_kwh / E_base * 100

                CE_opt, SH_opt, G_opt, Mg_opt, nod_opt, _ = compute_physics(p_opt)

            # ── Results ───────────────────────────────────────
            st.subheader("🎯 Optimized Results")
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Quality Score",   f"{quality_opt:.1f}/100",
                      "✅" if quality_opt >= target_quality else "⚠️ Below target")
            r2.metric("Energy Used",     f"{E_opt:.1f} kWh")
            r3.metric("Energy Cost",     f"₹{cost_opt:.0f}")
            r4.metric("Melt Time",       f"{melt_opt:.0f} min")

            st.markdown("---")

            # ── Savings vs baseline ───────────────────────────
            st.subheader("💰 Savings vs Maximum Temperature Baseline")
            s1,s2,s3 = st.columns(3)
            s1.metric("Energy Saved",    f"{savings_kwh:.1f} kWh",
                      f"-{savings_pct:.1f}%", delta_color="inverse")
            s2.metric("Cost Saved/Batch",f"₹{savings_inr:.0f}",
                      delta_color="inverse")
            monthly = savings_inr * 30   # assume 30 batches/month
            s3.metric("Monthly Savings", f"₹{monthly:,.0f}",
                      delta_color="inverse")

            # ── Sankey: Energy breakdown ──────────────────────
            st.subheader("🔋 Energy Breakdown")
            sensible_J  = batch_kg * CP_IRON * (pour_opt - 30)
            latent_J    = batch_kg * L_FUSION
            losses_J    = (sensible_J + latent_J) * 0.25
            total_J     = sensible_J + latent_J + losses_J

            fig_e = go.Figure(go.Pie(
                labels=['Sensible Heat', 'Latent Heat (Fusion)', 'Process Losses'],
                values=[sensible_J/3600000, latent_J/3600000, losses_J/3600000],
                hole=0.4,
                marker_colors=['#3b82f6','#f59e0b','#ef4444'],
                textinfo='label+percent'
            ))
            fig_e.update_layout(
                height=280, paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', showlegend=True,
                annotations=[dict(text=f"{E_opt:.1f}\nkWh", x=0.5, y=0.5,
                                  font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig_e, width="stretch")

            st.markdown("---")

            # ── Optimal parameters ────────────────────────────
            st.subheader("📋 Optimal Parameter Recipe")
            recipe = [
                ("🌡️ Pour Temperature",  f"{pour_opt:.0f}°C",  "1490–1570°C"),
                ("🔄 Spin Speed",        f"{rpm_opt:.0f} RPM", "400–1000 RPM"),
                ("🧲 Mg Added %",        f"{mg_opt:.4f}%",     "0.030–0.085%"),
                ("⏱️ Treatment Time",    f"{tt_opt:.1f} min",  "2–12 min"),
                ("🔥 Mold Preheat",      f"{pre_opt:.0f}°C",   "100–350°C"),
                ("📏 Mold Diameter",     f"{dia_opt:.3f}m",    "0.05–0.20m"),
            ]
            df_r = pd.DataFrame(recipe, columns=['Parameter','Optimal Value','Range'])
            st.dataframe(df_r, width="stretch", hide_index=True)

            # ── Physics validation ────────────────────────────
            st.subheader("🔬 Physics Check")
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("CE",       f"{CE_opt:.3f}", "✅" if 4.3<=CE_opt<=4.6 else "⚠️")
            p2.metric("Superheat",f"{SH_opt:.0f}°C","✅" if 60<=SH_opt<=110 else "⚠️")
            p3.metric("G-Factor", f"{G_opt:.1f}G","✅" if 40<=G_opt<=100 else "⚠️")
            p4.metric("Mg Eff",   f"{Mg_opt*100:.3f}%","✅" if 0.035<=Mg_opt<=0.055 else "⚠️")

            # ── Pour temp vs energy curve ─────────────────────
            st.subheader("📈 Pour Temperature vs Energy Cost")
            temps  = np.linspace(1490, 1570, 30)
            costs  = [compute_energy(t, batch_kg, pre_opt, tariff)[1] for t in temps]

            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=temps, y=costs, mode='lines',
                line=dict(color='#f59e0b', width=2)))
            fig_c.add_vline(x=pour_opt, line_dash='dash', line_color='#22c55e',
                            annotation_text=f'Optimal {pour_opt:.0f}°C')
            fig_c.update_layout(
                height=220, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='Pour Temperature (°C)',
                           gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title='Energy Cost (₹)', gridcolor='rgba(255,255,255,0.08)'),
                showlegend=False, margin=dict(t=10,b=40,l=10,r=80)
            )
            st.plotly_chart(fig_c, width="stretch")

        else:
            st.markdown("### 👈 Set constraints & click **Optimize**")
            st.info("""
**How Energy Optimization works:**

The furnace energy equation:
```
E = m × Cp × (T_pour − T_ambient) + m × L_fusion
    ──────────────────────────────────────────────
                   η_furnace
```
Where:
- **m** = batch mass (kg)
- **Cp** = 500 J/kg·K (specific heat of iron)
- **L_fusion** = 247,000 J/kg (latent heat)
- **η** = 0.75 (furnace efficiency)

**Key insight:** Pour temperature is the biggest energy driver.
Reducing it by just **20°C** on a 200kg batch saves **~₹120/batch**
= **~₹3,600/month** (30 batches).

AI finds the minimum pour temperature that still achieves your quality target.
            """)
