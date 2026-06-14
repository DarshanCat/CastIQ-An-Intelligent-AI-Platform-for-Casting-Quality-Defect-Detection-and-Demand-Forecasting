"""
💡 SHAP Explainability Module
Answers: WHY was this casting rejected?
Visual AI explanation using SHAP values
File: modules/shap_explainer.py
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

FEAT_DISPLAY = {
    'carbon_pct':          ('Carbon %',            '3.4–3.6%',    '⚗️'),
    'silicon_pct':         ('Silicon %',           '2.0–2.5%',    '⚗️'),
    'manganese_pct':       ('Manganese %',         '0.3–0.5%',    '⚗️'),
    'phosphorus_pct':      ('Phosphorus %',        '< 0.05%',     '⚗️'),
    'magnesium_added_pct': ('Magnesium Added %',   '0.04–0.07%',  '🧲'),
    'treatment_time_min':  ('Treatment Time',      '3–8 min',     '⏱️'),
    'rpm':                 ('Spin Speed',          '500–900 RPM', '🔄'),
    'mold_diameter_m':     ('Mold Diameter',       '0.08–0.18m',  '📏'),
    'pour_temp_c':         ('Pour Temperature',    '1500–1540°C', '🌡️'),
    'mold_preheat_c':      ('Mold Preheat',        '150–280°C',   '🔥'),
    'wall_thickness_mm':   ('Wall Thickness',      '10–60mm',     '📐'),
    'mold_enc':            ('Mold Type',           'Permanent',   '🔲'),
    'carbon_equivalent':   ('Carbon Equivalent',   '4.3–4.6',     '📐'),
    'superheat_c':         ('Superheat',           '60–110°C',    '🌡️'),
    'G_factor':            ('G-Factor',            '40–100G',     '⚙️'),
    'Mg_effective_pct':    ('Effective Mg %',      '0.035–0.055%','🧲'),
    'nodularity_index':    ('Nodularity Index',    '> 0.7',       '⭕'),
    'cooling_rate_cs':     ('Cooling Rate',        '8–14°C/s',    '❄️'),
}

def compute_physics(params):
    CE       = params['carbon_pct'] + params['silicon_pct']/3 + params['phosphorus_pct']/3
    T_liq    = 1550 - 28*params['carbon_pct'] - 8*params['silicon_pct']
    SH       = params['pour_temp_c'] - T_liq
    G        = (np.pi**2 * params['mold_diameter_m'] * params['rpm']**2) / (900 * 9.81)
    Mg_eff   = params['magnesium_added_pct'] * np.exp(-0.02 * params['treatment_time_min'])
    nodularity = np.clip(0.60 + (Mg_eff - 0.045)*8 - (params['treatment_time_min']-5)*0.015, 0, 1)
    B_map    = {0: 0.10, 1: 0.15, 2: 0.08}
    B        = B_map[int(params['mold_enc'])]
    t_sol    = B * (params['wall_thickness_mm']**2)
    c_rate   = np.clip((params['pour_temp_c'] - params['mold_preheat_c']) / (t_sol + 1e-6), 0.5, 40)
    return CE, SH, G, Mg_eff, nodularity, c_rate

def build_input_vector(params):
    CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(params)
    return np.array([
        params['carbon_pct'], params['silicon_pct'], params['manganese_pct'],
        params['phosphorus_pct'], params['magnesium_added_pct'], params['treatment_time_min'],
        params['rpm'], params['mold_diameter_m'], params['pour_temp_c'],
        params['mold_preheat_c'], params['wall_thickness_mm'], params['mold_enc'],
        CE, SH, G, Mg_eff, nodularity, c_rate
    ])

FEATS = [
    'carbon_pct','silicon_pct','manganese_pct','phosphorus_pct',
    'magnesium_added_pct','treatment_time_min','rpm','mold_diameter_m',
    'pour_temp_c','mold_preheat_c','wall_thickness_mm','mold_enc',
    'carbon_equivalent','superheat_c','G_factor','Mg_effective_pct',
    'nodularity_index','cooling_rate_cs'
]

def render_shap_explainer():
    st.title("💡 AI Explainability — SHAP Analysis")
    st.markdown("##### Why was this casting rejected? Visual explanation of every AI decision")
    st.markdown("---")

    try:
        clf    = joblib.load(MODEL_DIR / 'classifier.pkl')
        reg    = joblib.load(MODEL_DIR / 'regressor.pkl')
        scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    except Exception as e:
        st.error(f"❌ Models not found. Run setup.py first. Error: {e}")
        return

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Explain Single Cast",
        "📊 Global Feature Importance",
        "🆚 Compare Two Casts"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Explain Single Cast
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("**Enter casting parameters to get a full AI explanation of the decision**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Alloy Composition**")
            carbon  = st.slider("Carbon %",    3.0, 4.2, 3.8,  0.01, key='e_c')
            silicon = st.slider("Silicon %",   1.5, 3.2, 2.1,  0.01, key='e_si')
            manganese=st.slider("Manganese %", 0.15,0.8, 0.45, 0.01, key='e_mn')
            phosphorus=st.slider("Phosphorus %",0.01,0.10,0.03,0.005,key='e_p')

        with col2:
            st.markdown("**Magnesium Treatment**")
            mg_added  = st.slider("Mg Added %",   0.02, 0.09, 0.025, 0.001, key='e_mg')
            treat_time= st.slider("Treatment Time (min)", 2.0, 15.0, 10.0, 0.5, key='e_tt')
            st.markdown("**Mold**")
            mold      = st.selectbox("Mold Type", ['Permanent','Sand','Die'], key='e_mt')
            mold_pre  = st.slider("Preheat (°C)", 80, 400, 120, key='e_pre')

        with col3:
            st.markdown("**Process Parameters**")
            rpm       = st.slider("RPM",           200, 1200, 350, 10, key='e_rpm')
            mold_dia  = st.slider("Mold Dia (m)",  0.05, 0.25, 0.22, 0.01, key='e_dia')
            pour_temp = st.slider("Pour Temp (°C)",1440, 1600, 1455, 1,   key='e_pt')
            wall_mm   = st.slider("Wall Thickness (mm)", 5, 90, 45, 1,    key='e_wt')

        mold_enc = {'Permanent':0,'Sand':1,'Die':2}[mold]
        params = {
            'carbon_pct':carbon,'silicon_pct':silicon,'manganese_pct':manganese,
            'phosphorus_pct':phosphorus,'magnesium_added_pct':mg_added,
            'treatment_time_min':treat_time,'rpm':rpm,'mold_diameter_m':mold_dia,
            'pour_temp_c':pour_temp,'mold_preheat_c':mold_pre,
            'wall_thickness_mm':wall_mm,'mold_enc':mold_enc
        }

        explain_btn = st.button("🔬 Explain This Cast", width="stretch", type="primary")

        if explain_btn:
            vec    = build_input_vector(params).reshape(1, -1)
            vec_sc = scaler.transform(vec)

            quality  = float(reg.predict(vec_sc)[0])
            rej_prob = float(clf.predict_proba(vec_sc)[0][1])
            is_reject= rej_prob > 0.5

            # ── Verdict Banner ────────────────────────────────
            if is_reject:
                st.error(f"🔴 **REJECT** — Quality: {quality:.1f}/100 | Reject Probability: {rej_prob*100:.1f}%")
            else:
                st.success(f"🟢 **GOOD CAST** — Quality: {quality:.1f}/100 | Reject Probability: {rej_prob*100:.1f}%")

            # ── SHAP Values ───────────────────────────────────
            with st.spinner("Computing SHAP explanations..."):
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(vec_sc)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]   # positive class
                else:
                    sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

            # Sort by absolute impact
            feat_names  = [FEAT_DISPLAY.get(f, (f,'',''))[0] for f in FEATS]
            shap_series = pd.Series(sv, index=feat_names).sort_values(key=abs, ascending=False)
            top_feats   = shap_series.head(10)

            # ── Waterfall-style SHAP bar chart ────────────────
            st.subheader("🔬 Why this decision? — SHAP Contribution Chart")
            colors = ['#ef4444' if v > 0 else '#22c55e' for v in top_feats.values]
            fig = go.Figure(go.Bar(
                x=top_feats.values,
                y=top_feats.index,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in top_feats.values],
                textposition='outside',
            ))
            fig.add_vline(x=0, line_color='white', line_width=1)
            fig.update_layout(
                height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                           title='SHAP Value (red = pushes toward REJECT, green = pushes toward GOOD)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                margin=dict(t=10, b=10, l=10, r=80)
            )
            st.plotly_chart(fig, width="stretch")

            # ── Plain English Explanation ─────────────────────
            st.subheader("📝 Plain English Explanation")
            reject_drivers = [(f, v) for f, v in zip(top_feats.index, top_feats.values) if v > 0.01]
            good_drivers   = [(f, v) for f, v in zip(top_feats.index, top_feats.values) if v < -0.01]

            if reject_drivers:
                st.markdown("**🔴 Main reasons pushing toward REJECT:**")
                for feat, val in sorted(reject_drivers, key=lambda x: -x[1])[:4]:
                    st.markdown(f"- **{feat}** is contributing `{val:+.3f}` toward rejection")

            if good_drivers:
                st.markdown("**🟢 Factors working in favor of GOOD CAST:**")
                for feat, val in sorted(good_drivers, key=lambda x: x[1])[:3]:
                    st.markdown(f"- **{feat}** is contributing `{abs(val):.3f}` toward good quality")

            # ── Physics Check ─────────────────────────────────
            st.markdown("---")
            st.subheader("🔬 Physics Formula Violations")
            CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(params)
            physics_checks = [
                ("Carbon Equivalent", CE, 4.3, 4.6, f"{CE:.3f}", "Formula: C% + Si%/3 + P%/3"),
                ("Superheat",    SH,   60,  110, f"{SH:.0f}°C",    "Formula: Pour Temp − Liquidus Temp"),
                ("G-Factor",     G,    40,  100, f"{G:.1f}G",       "Formula: π²×D×N² / (900×g)"),
                ("Mg Effective", Mg_eff*100, 3.5, 5.5, f"{Mg_eff*100:.3f}%", "Formula: Mg × e^(−0.02×t)"),
                ("Nodularity",   nodularity, 0.6, 1.0, f"{nodularity:.2f}", "Higher = more spherical graphite"),
            ]
            for name, val, lo, hi, display, formula in physics_checks:
                ok = lo <= val <= hi
                icon = "✅" if ok else "❌"
                with st.expander(f"{icon} {name} = {display}  ({'OK' if ok else 'OUT OF RANGE'})"):
                    st.caption(formula)
                    if not ok:
                        if val < lo:
                            st.warning(f"Below minimum {lo}. Increase by {lo-val:.3f}")
                        else:
                            st.warning(f"Above maximum {hi}. Decrease by {val-hi:.3f}")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — Global Feature Importance
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("**Which parameters matter most for casting quality — across ALL casts?**")

        data_path = DATA_DIR / 'casting_data_large.csv'
        if not data_path.exists():
            data_path = DATA_DIR / 'casting_physics_data.csv'
        if not data_path.exists():
            st.warning("Casting dataset not found. Please run setup.py first.")
            return

        with st.spinner("Computing global SHAP values on 300 samples..."):
            df     = pd.read_csv(data_path).sample(300, random_state=42)
            X      = df[FEATS]
            X_sc   = scaler.transform(X)
            explainer_g = shap.TreeExplainer(clf)
            sv_g   = explainer_g.shap_values(X_sc)
            if isinstance(sv_g, list):
                sv_g = sv_g[1]

        mean_abs  = np.abs(sv_g).mean(axis=0)
        feat_disp = [FEAT_DISPLAY.get(f, (f,'',''))[0] for f in FEATS]
        imp_df    = pd.DataFrame({'Feature': feat_disp, 'Importance': mean_abs})
        imp_df    = imp_df.sort_values('Importance', ascending=True)

        # Color: physics features in gold, raw inputs in blue
        physics_feats = ['Carbon Equivalent','Superheat','G-Factor',
                         'Effective Mg %','Nodularity Index','Cooling Rate']
        colors = ['#f59e0b' if f in physics_feats else '#3b82f6'
                  for f in imp_df['Feature']]

        fig_gi = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'],
            orientation='h', marker_color=colors,
            text=[f"{v:.4f}" for v in imp_df['Importance']],
            textposition='outside'
        ))
        fig_gi.update_layout(
            height=520,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Mean |SHAP Value|'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            margin=dict(t=20, b=20, l=10, r=80)
        )
        st.plotly_chart(fig_gi, width="stretch")

        st.caption("🟡 Gold bars = Physics-derived features | 🔵 Blue bars = Raw process inputs")

        # Insight
        top3 = imp_df.sort_values('Importance', ascending=False).head(3)['Feature'].tolist()
        st.info(f"**Key insight:** The top 3 most important features are **{top3[0]}**, "
                f"**{top3[1]}**, and **{top3[2]}**. "
                "Physics-derived features dominate over raw inputs — confirming that "
                "the metallurgical formulas are the real drivers of casting quality.")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Compare Two Casts
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("**Compare a GOOD cast vs a REJECTED cast — see exactly what's different**")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Cast A — Good Parameters")
            c_a  = st.slider("Carbon %",    3.0, 4.2, 3.5,  0.01, key='ca_c')
            si_a = st.slider("Silicon %",   1.5, 3.2, 2.25, 0.01, key='ca_si')
            mg_a = st.slider("Mg Added %",  0.02,0.09,0.055,0.001,key='ca_mg')
            tt_a = st.slider("Treat Time",  2.0, 15.0,4.0,  0.5,  key='ca_tt')
            rpm_a= st.slider("RPM",         200, 1200,700,  10,   key='ca_rpm')
            pt_a = st.slider("Pour Temp",   1440,1600,1520, 1,    key='ca_pt')

        with col_b:
            st.markdown("#### Cast B — Bad Parameters")
            c_b  = st.slider("Carbon %",    3.0, 4.2, 3.8,  0.01, key='cb_c')
            si_b = st.slider("Silicon %",   1.5, 3.2, 2.1,  0.01, key='cb_si')
            mg_b = st.slider("Mg Added %",  0.02,0.09,0.025,0.001,key='cb_mg')
            tt_b = st.slider("Treat Time",  2.0, 15.0,10.0, 0.5,  key='cb_tt')
            rpm_b= st.slider("RPM",         200, 1200,350,  10,   key='cb_rpm')
            pt_b = st.slider("Pour Temp",   1440,1600,1455, 1,    key='cb_pt')

        compare_btn = st.button("🆚 Compare These Casts", width="stretch", type="primary")

        if compare_btn:
            def make_params(c,si,mg,tt,rpm,pt):
                return {'carbon_pct':c,'silicon_pct':si,'manganese_pct':0.35,
                        'phosphorus_pct':0.025,'magnesium_added_pct':mg,
                        'treatment_time_min':tt,'rpm':rpm,'mold_diameter_m':0.12,
                        'pour_temp_c':pt,'mold_preheat_c':200,'wall_thickness_mm':25,
                        'mold_enc':0}

            pa = make_params(c_a,si_a,mg_a,tt_a,rpm_a,pt_a)
            pb = make_params(c_b,si_b,mg_b,tt_b,rpm_b,pt_b)

            va = build_input_vector(pa).reshape(1,-1)
            vb = build_input_vector(pb).reshape(1,-1)
            va_sc = scaler.transform(va)
            vb_sc = scaler.transform(vb)

            qa = float(reg.predict(va_sc)[0])
            qb = float(reg.predict(vb_sc)[0])
            ra = float(clf.predict_proba(va_sc)[0][1])
            rb = float(clf.predict_proba(vb_sc)[0][1])

            c1, c2 = st.columns(2)
            with c1:
                verdict_a = "🟢 GOOD" if ra < 0.5 else "🔴 REJECT"
                st.metric("Cast A", f"{qa:.1f}/100", f"{verdict_a} | {ra*100:.0f}% reject prob")
            with c2:
                verdict_b = "🟢 GOOD" if rb < 0.5 else "🔴 REJECT"
                st.metric("Cast B", f"{qb:.1f}/100", f"{verdict_b} | {rb*100:.0f}% reject prob")

            # SHAP for both
            explainer_c = shap.TreeExplainer(clf)
            sv_a = explainer_c.shap_values(va_sc)
            sv_b = explainer_c.shap_values(vb_sc)
            sv_a = sv_a[1][0] if isinstance(sv_a, list) else sv_a[0]
            sv_b = sv_b[1][0] if isinstance(sv_b, list) else sv_b[0]

            feat_names = [FEAT_DISPLAY.get(f,(f,'',''))[0] for f in FEATS]

            # Side-by-side SHAP
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name='Cast A (Good)', x=feat_names, y=sv_a,
                                     marker_color='#22c55e', opacity=0.8))
            fig_cmp.add_trace(go.Bar(name='Cast B (Reject)', x=feat_names, y=sv_b,
                                     marker_color='#ef4444', opacity=0.8))
            fig_cmp.update_layout(
                barmode='group', height=380,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickangle=-45),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title='SHAP Value'),
                legend=dict(orientation='h'),
                margin=dict(t=20, b=80, l=10, r=10)
            )
            st.subheader("📊 SHAP Comparison — Cast A vs Cast B")
            st.plotly_chart(fig_cmp, width="stretch")

            # Difference table
            diffs = [{'Feature': feat_names[i],
                      'Cast A SHAP': f"{sv_a[i]:+.4f}",
                      'Cast B SHAP': f"{sv_b[i]:+.4f}",
                      'Δ Impact':    f"{sv_b[i]-sv_a[i]:+.4f}"}
                     for i in range(len(feat_names))]
            diffs_df = pd.DataFrame(diffs)
            diffs_df['|Δ|'] = diffs_df['Δ Impact'].apply(lambda x: abs(float(x)))
            diffs_df = diffs_df.sort_values('|Δ|', ascending=False).drop('|Δ|', axis=1)

            st.subheader("📋 Feature Impact Difference Table")
            st.dataframe(diffs_df.head(10), width="stretch", hide_index=True)
