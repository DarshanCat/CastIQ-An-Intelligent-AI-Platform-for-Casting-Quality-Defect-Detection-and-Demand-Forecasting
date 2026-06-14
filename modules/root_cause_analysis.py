"""
🔍 Root Cause Analysis Module
Auto-diagnose WHY a batch failed using SHAP + physics rules + causal tree
File: modules/root_cause_analysis.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path

BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / 'backend' / 'models'
DATA_DIR  = BASE / 'backend' / 'data'

FEATS = [
    'carbon_pct','silicon_pct','manganese_pct','phosphorus_pct',
    'magnesium_added_pct','treatment_time_min','rpm','mold_diameter_m',
    'pour_temp_c','mold_preheat_c','wall_thickness_mm','mold_enc',
    'carbon_equivalent','superheat_c','G_factor','Mg_effective_pct',
    'nodularity_index','cooling_rate_cs'
]

FEAT_NAMES = {
    'carbon_pct':'Carbon %','silicon_pct':'Silicon %','manganese_pct':'Manganese %',
    'phosphorus_pct':'Phosphorus %','magnesium_added_pct':'Mg Added %',
    'treatment_time_min':'Treatment Time','rpm':'Spin Speed (RPM)',
    'mold_diameter_m':'Mold Diameter','pour_temp_c':'Pour Temperature',
    'mold_preheat_c':'Mold Preheat','wall_thickness_mm':'Wall Thickness',
    'mold_enc':'Mold Type','carbon_equivalent':'Carbon Equivalent (CE)',
    'superheat_c':'Superheat','G_factor':'G-Factor',
    'Mg_effective_pct':'Effective Mg %','nodularity_index':'Nodularity Index',
    'cooling_rate_cs':'Cooling Rate'
}

# Defect-cause mapping (metallurgical knowledge base)
DEFECT_LIBRARY = {
    'Porosity': {
        'causes': ['Low pour temperature','High gas content','Low G-factor','High phosphorus'],
        'physics': 'Gas cannot escape mushy zone → trapped as pores',
        'fix': 'Increase pour temp by 15°C, degas melt, increase RPM to raise G above 60',
        'icon': '🫧'
    },
    'Cold Shuts': {
        'causes': ['Pour temperature too low','Slow flow rate','Low superheat'],
        'physics': 'Metal solidifies before mold is fully filled',
        'fix': 'Increase pour temp, increase flow rate, check superheat > 60°C',
        'icon': '🧊'
    },
    'Shrinkage': {
        'causes': ['Insufficient feed metal','High carbon equivalent','Slow cooling'],
        'physics': 'Volume contracts during solidification without enough feed',
        'fix': 'Reduce CE below 4.5, optimize risers, increase cooling rate',
        'icon': '📉'
    },
    'Graphite Flotation': {
        'causes': ['CE too high (>4.6)','Low G-factor','High pour temp'],
        'physics': 'Graphite (lower density) floats to inner surface under low centrifugal force',
        'fix': 'Reduce CE to 4.3–4.5, increase RPM to get G > 50, reduce pour temp',
        'icon': '⬆️'
    },
    'Hot Tears': {
        'causes': ['G-factor too high (>100)','Rapid cooling','High Mn'],
        'physics': 'Tensile stresses exceed strength in semi-solid state',
        'fix': 'Reduce RPM to bring G below 100, increase mold preheat, reduce Mn',
        'icon': '💥'
    },
    'Poor Nodularity': {
        'causes': ['Low effective Mg (<0.035%)','Long treatment time','High sulphur'],
        'physics': 'Insufficient Mg → graphite forms as flakes not spheres → brittle',
        'fix': 'Increase Mg addition or reduce treatment-to-pour time below 8 min',
        'icon': '⭕'
    },
    'Segregation': {
        'causes': ['G-factor too high (>100)','High Mn content','Slow pour'],
        'physics': 'Centrifugal force separates elements by density',
        'fix': 'Reduce RPM, balance alloy composition, increase pour speed',
        'icon': '🔀'
    }
}

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

def diagnose_defects(p, CE, SH, G, Mg_eff, nodularity):
    """Rule-based defect diagnosis."""
    likely = []
    if G < 40 and SH < 60:
        likely.append(('Porosity', 0.85))
    elif G < 40:
        likely.append(('Porosity', 0.60))
    if SH < 50:
        likely.append(('Cold Shuts', 0.75))
    if CE > 4.6:
        likely.append(('Graphite Flotation', 0.80 if G < 50 else 0.55))
    if CE > 4.55:
        likely.append(('Shrinkage', 0.60))
    if G > 100:
        likely.append(('Hot Tears', 0.70))
        likely.append(('Segregation', 0.55))
    if Mg_eff < 0.035:
        likely.append(('Poor Nodularity', 0.90))
    elif Mg_eff < 0.040:
        likely.append(('Poor Nodularity', 0.55))

    # Deduplicate, sort by probability
    seen = {}
    for defect, prob in likely:
        if defect not in seen or seen[defect] < prob:
            seen[defect] = prob
    return sorted(seen.items(), key=lambda x: -x[1])

def build_vector(p):
    CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(p)
    return np.array([
        p['carbon_pct'],p['silicon_pct'],p['manganese_pct'],p['phosphorus_pct'],
        p['magnesium_added_pct'],p['treatment_time_min'],p['rpm'],p['mold_diameter_m'],
        p['pour_temp_c'],p['mold_preheat_c'],p['wall_thickness_mm'],p['mold_enc'],
        CE, SH, G, Mg_eff, nodularity, c_rate
    ])

def render_rca():
    st.title("🔍 Root Cause Analysis")
    st.markdown("##### Auto-diagnose WHY a casting batch failed — with corrective actions")
    st.markdown("---")

    try:
        clf    = joblib.load(MODEL_DIR/'classifier.pkl')
        reg    = joblib.load(MODEL_DIR/'regressor.pkl')
        scaler = joblib.load(MODEL_DIR/'scaler.pkl')
    except Exception as e:
        st.error(f"❌ Models not found: {e}")
        return

    tab1, tab2 = st.tabs(["🔎 Diagnose a Failed Batch", "📚 Defect Library"])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Diagnose a batch
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("**Enter the parameters of the failed batch below**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Alloy**")
            carbon    = st.slider("Carbon %",    3.0, 4.2, 3.8,  0.01, key='rca_c')
            silicon   = st.slider("Silicon %",   1.4, 3.2, 2.1,  0.01, key='rca_si')
            manganese = st.slider("Manganese %", 0.15,0.80,0.40, 0.01, key='rca_mn')
            phosphorus= st.slider("Phosphorus %",0.01,0.10,0.04, 0.005,key='rca_p')
        with col2:
            st.markdown("**Mg Treatment**")
            mg_added  = st.slider("Mg Added %",       0.020,0.090,0.025,0.001,key='rca_mg')
            treat_time= st.slider("Treatment Time (min)",2.0,15.0,11.0, 0.5, key='rca_tt')
            st.markdown("**Mold**")
            mold      = st.selectbox("Mold Type",['Permanent','Sand','Die'],key='rca_mt')
            mold_pre  = st.slider("Preheat (°C)",80, 400,110, key='rca_pre')
        with col3:
            st.markdown("**Process**")
            rpm       = st.slider("RPM",           200,1200,320, 10, key='rca_rpm')
            mold_dia  = st.slider("Mold Dia (m)",  0.05,0.25,0.22,0.01,key='rca_dia')
            pour_temp = st.slider("Pour Temp (°C)",1440,1600,1452,1,  key='rca_pt')
            wall_mm   = st.slider("Wall Thick (mm)",5,90,50, 1,       key='rca_wt')

        mold_enc = {'Permanent':0,'Sand':1,'Die':2}[mold]
        params   = {
            'carbon_pct':carbon,'silicon_pct':silicon,'manganese_pct':manganese,
            'phosphorus_pct':phosphorus,'magnesium_added_pct':mg_added,
            'treatment_time_min':treat_time,'rpm':rpm,'mold_diameter_m':mold_dia,
            'pour_temp_c':pour_temp,'mold_preheat_c':mold_pre,
            'wall_thickness_mm':wall_mm,'mold_enc':mold_enc
        }

        diagnose_btn = st.button("🔍 Run Root Cause Analysis",
                                  width="stretch", type="primary")

        if diagnose_btn:
            vec    = build_vector(params).reshape(1,-1)
            vec_sc = scaler.transform(vec)
            quality  = float(reg.predict(vec_sc)[0])
            rej_prob = float(clf.predict_proba(vec_sc)[0][1])
            CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(params)

            # ── Batch verdict ─────────────────────────────────
            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Quality Score",     f"{quality:.1f}/100")
            col_v2.metric("Reject Probability",f"{rej_prob*100:.1f}%")
            col_v3.metric("Verdict",           "🔴 REJECT" if rej_prob>0.5 else "🟢 PASS")

            st.markdown("---")

            # ── Physics violations ────────────────────────────
            st.subheader("🔬 Physics Violation Scan")
            violations = []
            checks = [
                ("Carbon Equivalent", CE,      4.3,  4.6,  f"{CE:.3f}"),
                ("Superheat",         SH,       60,   110,  f"{SH:.0f}°C"),
                ("G-Factor",          G,        40,   100,  f"{G:.1f}G"),
                ("Mg Effective %",    Mg_eff*100,3.5, 5.5,  f"{Mg_eff*100:.3f}%"),
                ("Nodularity Index",  nodularity,0.6, 1.0,  f"{nodularity:.2f}"),
            ]
            ok_count = 0
            for name, val, lo, hi, display in checks:
                ok = lo <= val <= hi
                if ok:
                    ok_count += 1
                    st.success(f"✅ **{name}** = {display} — Within optimal range")
                else:
                    delta = f"{lo-val:+.3f}" if val < lo else f"{val-hi:+.3f}"
                    direction = "Too LOW" if val < lo else "Too HIGH"
                    violations.append(name)
                    st.error(f"❌ **{name}** = {display} — **{direction}** by {abs(float(delta)):.3f}")

            st.markdown("---")

            # ── Defect diagnosis ──────────────────────────────
            st.subheader("🦠 Likely Defect Types")
            defects = diagnose_defects(params, CE, SH, G, Mg_eff, nodularity)

            if not defects:
                st.info("No specific defect pattern matched. Check raw inputs for inconsistencies.")
            else:
                for defect, prob in defects[:3]:
                    info = DEFECT_LIBRARY[defect]
                    with st.expander(f"{info['icon']} **{defect}** — {prob*100:.0f}% likely", expanded=(prob>0.7)):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Root Causes:**")
                            for cause in info['causes']:
                                st.markdown(f"• {cause}")
                            st.markdown(f"**Physics:** _{info['physics']}_")
                        with c2:
                            st.success(f"**✅ Corrective Action:**\n\n{info['fix']}")

            st.markdown("---")

            # ── SHAP top contributors ─────────────────────────
            st.subheader("📊 Top Contributing Parameters (SHAP)")
            with st.spinner("Computing SHAP..."):
                explainer = shap.TreeExplainer(clf)
                sv        = explainer.shap_values(vec_sc)
                sv        = sv[1][0] if isinstance(sv, list) else sv[0]

            feat_labels  = [FEAT_NAMES.get(f,f) for f in FEATS]
            sv_series    = pd.Series(np.abs(sv), index=feat_labels).sort_values(ascending=False).head(8)

            fig_shap = go.Figure(go.Bar(
                x=sv_series.values, y=sv_series.index, orientation='h',
                marker_color='#ef4444', text=[f"{v:.4f}" for v in sv_series.values],
                textposition='outside'
            ))
            fig_shap.update_layout(
                height=280, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='|SHAP Impact|', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                margin=dict(t=10,b=10,l=10,r=80)
            )
            st.plotly_chart(fig_shap, width="stretch")

            # ── Action plan ───────────────────────────────────
            st.subheader("📋 Recommended Action Plan")
            priority = 1
            action_rows = []
            if SH < 60:
                action_rows.append((priority, "🌡️ Pour Temperature", f"Increase to {pour_temp + (60-SH) + 10:.0f}°C", "Critical"))
                priority += 1
            if Mg_eff < 0.035:
                need_more = (0.045 - mg_added*np.exp(-0.02*treat_time)) / np.exp(-0.02*treat_time)
                action_rows.append((priority, "🧲 Magnesium Addition", f"Increase by {need_more*100:.3f}% OR reduce treatment time to <8 min", "Critical"))
                priority += 1
            if G < 40:
                need_rpm = np.sqrt(40 * 900 * 9.81 / (np.pi**2 * mold_dia))
                action_rows.append((priority, "🔄 Spin Speed", f"Increase RPM to minimum {need_rpm:.0f}", "High"))
                priority += 1
            if G > 100:
                max_rpm = np.sqrt(100 * 900 * 9.81 / (np.pi**2 * mold_dia))
                action_rows.append((priority, "🔄 Spin Speed", f"Reduce RPM to maximum {max_rpm:.0f}", "High"))
                priority += 1
            if CE > 4.6:
                action_rows.append((priority, "⚗️ Carbon Equivalent", f"Reduce CE by {CE-4.5:.2f} — lower C% or Si%", "High"))
                priority += 1
            if mold_pre < 150:
                action_rows.append((priority, "🔥 Mold Preheat", f"Increase preheat to minimum 150°C (currently {mold_pre}°C)", "Medium"))
                priority += 1

            if action_rows:
                df_act = pd.DataFrame(action_rows,
                    columns=['Priority','Parameter','Action','Severity'])
                st.dataframe(df_act, width="stretch", hide_index=True)
            else:
                st.success("✅ No specific parameter corrections needed — check raw material batch quality")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Defect Library
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("**Complete metallurgical defect reference for ductile iron centrifugal casting**")
        st.markdown("")
        for defect, info in DEFECT_LIBRARY.items():
            with st.expander(f"{info['icon']} **{defect}**"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Causes:**")
                    for cause in info['causes']:
                        st.markdown(f"• {cause}")
                with c2:
                    st.markdown("**Physics:**")
                    st.markdown(f"_{info['physics']}_")
                with c3:
                    st.success(f"**Fix:**\n\n{info['fix']}")
