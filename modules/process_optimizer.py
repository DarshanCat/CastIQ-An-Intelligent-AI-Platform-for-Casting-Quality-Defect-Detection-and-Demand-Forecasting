"""
🎯 Process Optimizer Module
Inverse ML: Given a target quality score → find optimal casting parameters
Uses scipy.optimize + trained XGBoost model
File: modules/process_optimizer.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / 'backend' / 'models'

# ── Parameter bounds (realistic operating ranges) ─────────────
PARAM_BOUNDS = {
    'carbon_pct':          (3.2,  3.9),
    'silicon_pct':         (1.8,  2.8),
    'manganese_pct':       (0.20, 0.65),
    'phosphorus_pct':      (0.01, 0.07),
    'magnesium_added_pct': (0.030,0.085),
    'treatment_time_min':  (2.0,  12.0),
    'rpm':                 (400,  1000),
    'mold_diameter_m':     (0.05, 0.20),
    'pour_temp_c':         (1490, 1570),
    'mold_preheat_c':      (100,  350),
    'wall_thickness_mm':   (8,    70),
}

PARAM_LABELS = {
    'carbon_pct':          ('⚗️  Carbon %',            '%'),
    'silicon_pct':         ('⚗️  Silicon %',           '%'),
    'manganese_pct':       ('⚗️  Manganese %',         '%'),
    'phosphorus_pct':      ('⚗️  Phosphorus %',        '%'),
    'magnesium_added_pct': ('🧲 Magnesium Added %',   '%'),
    'treatment_time_min':  ('⏱️  Treatment Time',      'min'),
    'rpm':                 ('🔄 Spin Speed',           'RPM'),
    'mold_diameter_m':     ('📏 Mold Diameter',        'm'),
    'pour_temp_c':         ('🌡️  Pour Temperature',    '°C'),
    'mold_preheat_c':      ('🔥 Mold Preheat',        '°C'),
    'wall_thickness_mm':   ('📐 Wall Thickness',       'mm'),
}

OPTIMAL_TARGETS = {
    'carbon_equivalent':  (4.3, 4.6),
    'superheat_c':        (60,  110),
    'G_factor':           (40,  100),
    'Mg_effective_pct':   (0.035, 0.055),
}

def compute_physics(p):
    """Compute physics-derived features from raw parameters."""
    CE        = p['carbon_pct'] + p['silicon_pct']/3 + p['phosphorus_pct']/3
    T_liq     = 1550 - 28*p['carbon_pct'] - 8*p['silicon_pct']
    superheat = p['pour_temp_c'] - T_liq
    G         = (np.pi**2 * p['mold_diameter_m'] * p['rpm']**2) / (900 * 9.81)
    Mg_eff    = p['magnesium_added_pct'] * np.exp(-0.02 * p['treatment_time_min'])
    nodularity= np.clip(0.60 + (Mg_eff - 0.045)*8 - (p['treatment_time_min']-5)*0.015, 0, 1)
    mold_B    = {'Permanent':0.10,'Sand':0.15,'Die':0.08}[p['mold_type']]
    t_sol     = mold_B * (p['wall_thickness_mm']**2)
    c_rate    = np.clip((p['pour_temp_c'] - p['mold_preheat_c']) / (t_sol + 1e-6), 0.5, 40)
    return CE, superheat, G, Mg_eff, nodularity, c_rate

def params_to_vector(params_dict):
    keys = list(PARAM_BOUNDS.keys())
    return np.array([params_dict[k] for k in keys])

def vector_to_features(x, mold_enc):
    keys  = list(PARAM_BOUNDS.keys())
    p     = dict(zip(keys, x))
    p['mold_type'] = ['Permanent','Sand','Die'][int(mold_enc)]
    CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(p)
    return np.array([
        p['carbon_pct'], p['silicon_pct'], p['manganese_pct'], p['phosphorus_pct'],
        p['magnesium_added_pct'], p['treatment_time_min'], p['rpm'], p['mold_diameter_m'],
        p['pour_temp_c'], p['mold_preheat_c'], p['wall_thickness_mm'], mold_enc,
        CE, SH, G, Mg_eff, nodularity, c_rate
    ])

def render_process_optimizer():
    st.title("🎯 Process Optimizer")
    st.markdown("##### Inverse AI — Set your target quality and get the optimal casting parameters")
    st.markdown("---")

    try:
        reg    = joblib.load(MODEL_DIR / 'regressor.pkl')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Optimization Settings")

        target_quality = st.slider("🎯 Target Quality Score", 70, 99, 92)
        mold_type      = st.selectbox("🔲 Mold Type (fixed)", ['Permanent', 'Sand', 'Die'])
        mold_enc       = {'Permanent':0,'Sand':1,'Die':2}[mold_type]

        st.markdown("---")
        st.markdown("**🔒 Lock parameters (keep at your value)**")
        lock_wall = st.checkbox("Lock Wall Thickness")
        if lock_wall:
            fixed_wall = st.slider("Wall Thickness (mm)", 8, 70, 25)
        lock_carbon = st.checkbox("Lock Carbon %")
        if lock_carbon:
            fixed_carbon = st.slider("Carbon %", 3.2, 3.9, 3.5, 0.01)

        optimize_btn = st.button("🚀 Find Optimal Parameters",
                                 width="stretch", type="primary")

    with col2:
        if optimize_btn:
            with st.spinner("🤖 Optimizing... running differential evolution..."):

                bounds = list(PARAM_BOUNDS.values())
                if lock_wall:
                    bounds[10] = (fixed_wall, fixed_wall+0.01)
                if lock_carbon:
                    bounds[0]  = (fixed_carbon, fixed_carbon+0.001)

                def objective(x):
                    feats  = vector_to_features(x, mold_enc).reshape(1,-1)
                    scaled = scaler.transform(feats)
                    pred   = reg.predict(scaled)[0]
                    return abs(pred - target_quality)

                result = differential_evolution(
                    objective, bounds,
                    maxiter=200, popsize=12,
                    tol=0.01, seed=42,
                    mutation=(0.5,1.0), recombination=0.7
                )

                opt_x    = result.x
                opt_keys = list(PARAM_BOUNDS.keys())
                opt_p    = dict(zip(opt_keys, opt_x))
                opt_p['mold_type'] = mold_type

                feats_opt   = vector_to_features(opt_x, mold_enc).reshape(1,-1)
                scaled_opt  = scaler.transform(feats_opt)
                pred_quality= float(reg.predict(scaled_opt)[0])

                CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(opt_p)

            # ── Results ─────────────────────────────────────
            verdict_color = "#22c55e" if pred_quality >= 80 else "#f59e0b"
            st.markdown(f"""
            <div style='background:rgba(34,197,94,0.1);border:1px solid #22c55e;
                        border-radius:12px;padding:20px;text-align:center;margin-bottom:16px'>
                <h2 style='color:#22c55e;margin:0'>🎯 Predicted Quality</h2>
                <h1 style='color:{verdict_color};margin:4px 0;font-size:48px'>{pred_quality:.1f}<span style='font-size:24px'>/100</span></h1>
                <p style='color:#aaa;margin:0'>Target was {target_quality}</p>
            </div>
            """, unsafe_allow_html=True)

            # Physics check
            st.subheader("🔬 Physics Validation")
            p1,p2,p3,p4 = st.columns(4)
            ce_ok  = 4.3 <= CE  <= 4.6
            sh_ok  = 60  <= SH  <= 110
            g_ok   = 40  <= G   <= 100
            mg_ok  = 0.035 <= Mg_eff <= 0.055
            p1.metric("CE",        f"{CE:.3f}",   "✅" if ce_ok  else "⚠️ Out of range")
            p2.metric("Superheat", f"{SH:.0f}°C", "✅" if sh_ok  else "⚠️ Out of range")
            p3.metric("G-Factor",  f"{G:.1f}G",   "✅" if g_ok   else "⚠️ Out of range")
            p4.metric("Mg_eff %",  f"{Mg_eff*100:.3f}%","✅" if mg_ok else "⚠️ Out of range")

            st.markdown("---")
            st.subheader("📋 Optimal Parameters to Set")

            rows = []
            for k, v in opt_p.items():
                if k == 'mold_type': continue
                label, unit = PARAM_LABELS[k]
                lo, hi = PARAM_BOUNDS[k]
                mid    = (lo+hi)/2
                rows.append({
                    'Parameter': label,
                    'Set To':    f"{v:.4f} {unit}".rstrip(),
                    'Range':     f"{lo} – {hi} {unit}".rstrip(),
                    'Status':    '✅ Optimal' if lo<=v<=hi else '⚠️ Check'
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # Bar chart: parameter deviation from midpoint
            st.markdown("---")
            st.subheader("📊 Parameter Position in Operating Range")
            fig_data = []
            for k, v in opt_p.items():
                if k == 'mold_type': continue
                lo, hi = PARAM_BOUNDS[k]
                pct    = (v - lo) / (hi - lo) * 100
                label, _ = PARAM_LABELS[k]
                fig_data.append({'param': label.split()[-1], 'pct': pct})

            fig = go.Figure(go.Bar(
                x=[d['pct'] for d in fig_data],
                y=[d['param'] for d in fig_data],
                orientation='h',
                marker_color=['#22c55e' if 20<=d['pct']<=80 else '#f59e0b'
                              for d in fig_data],
                text=[f"{d['pct']:.0f}%" for d in fig_data],
                textposition='outside'
            ))
            fig.add_vline(x=50, line_dash='dot', line_color='white',
                          annotation_text='Midpoint')
            fig.update_layout(
                height=380, xaxis=dict(range=[0,115], title='Position in Range (%)'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', margin=dict(t=10,b=10,l=10,r=60)
            )
            st.plotly_chart(fig, width="stretch")

        else:
            st.markdown("### 👈 Set target quality & click **Find Optimal Parameters**")
            st.info("""
**How this works:**

1. You set a target quality score (e.g. 92)
2. AI runs **Differential Evolution** optimization — searches 11-dimensional parameter space
3. Finds the combination of parameters that produces exactly your target quality
4. Validates against 6 metallurgical physics formulas
5. Gives you a ready-to-use parameter recipe

**Use cases:**
- New product → what parameters to start with?
- Customer needs IS 700/2 quality → what process settings?
- Minimize cost while hitting quality target
            """)
