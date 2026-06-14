"""
📉 SPC Dashboard Module
Statistical Process Control for casting quality monitoring
Includes: Xbar-R charts, CUSUM, Cpk, Nelson Rules violation detection
File: modules/spc_dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE     = Path(__file__).parent.parent
DATA_DIR = BASE / 'backend' / 'data'

# ── Control chart helpers ─────────────────────────────────────
def control_limits(data, n_sigma=3):
    mu  = np.mean(data)
    sig = np.std(data)
    return mu, mu + n_sigma*sig, mu - n_sigma*sig

def cpk(data, usl, lsl):
    mu  = np.mean(data)
    sig = np.std(data)
    if sig == 0: return 999
    return min((usl - mu)/(3*sig), (mu - lsl)/(3*sig))

def cusum(data, target=None, k=0.5):
    """CUSUM chart — detects small sustained shifts."""
    if target is None: target = np.mean(data)
    sig = np.std(data)
    k  *= sig
    Sp  = np.zeros(len(data))
    Sn  = np.zeros(len(data))
    for i in range(1, len(data)):
        Sp[i] = max(0, Sp[i-1] + (data[i] - target - k))
        Sn[i] = max(0, Sn[i-1] - (data[i] - target + k))
    return Sp, Sn

def nelson_violations(data, ucl, lcl, mu):
    """Detect Nelson rule violations."""
    v = []
    n = len(data)
    for i in range(n):
        # Rule 1: Beyond 3σ
        if data[i] > ucl or data[i] < lcl:
            v.append((i, "Rule 1: Beyond 3σ control limit"))
        # Rule 2: 9 consecutive same side
        if i >= 8:
            seg = data[i-8:i+1]
            if all(s > mu for s in seg) or all(s < mu for s in seg):
                v.append((i, "Rule 2: 9 consecutive points same side"))
        # Rule 3: 6 consecutive trending
        if i >= 5:
            seg = data[i-5:i+1]
            diffs = [seg[j+1]-seg[j] for j in range(len(seg)-1)]
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                v.append((i, "Rule 3: 6 consecutive trending"))
    return v

def make_control_chart(x, y, title, ucl, lcl, mu, unit="", violations=None):
    fig = go.Figure()

    # Confidence bands
    fig.add_hrect(y0=lcl, y1=ucl, fillcolor="rgba(34,197,94,0.06)",
                  line_width=0, annotation_text="")

    # Main line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
        line=dict(color='#3b82f6', width=1.5),
        marker=dict(size=5, color='#3b82f6'),
        name='Value'))

    # Control limits
    fig.add_hline(y=ucl, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'UCL={ucl:.2f}', annotation_position='right')
    fig.add_hline(y=lcl, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'LCL={lcl:.2f}', annotation_position='right')
    fig.add_hline(y=mu,  line_dash='dot',  line_color='#22c55e',
                  annotation_text=f'CL={mu:.2f}',  annotation_position='right')

    # Violation markers
    if violations:
        vx = [x[v[0]] for v in violations]
        vy = [y[v[0]] for v in violations]
        fig.add_trace(go.Scatter(x=vx, y=vy, mode='markers',
            marker=dict(size=12, color='#ef4444', symbol='x'),
            name='Violation'))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=240, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', title='Batch #'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title=unit),
        margin=dict(t=40, b=30, l=10, r=80)
    )
    return fig

def render_spc_dashboard():
    st.title("📉 SPC Dashboard")
    st.markdown("##### Statistical Process Control — monitor process stability and detect drift before rejects occur")
    st.markdown("---")

    # Load or generate historical data
    data_path = DATA_DIR / 'casting_data_large.csv'
    if not data_path.exists():
        data_path = DATA_DIR / 'casting_physics_data.csv'
    if data_path.exists():
        df_full = pd.read_csv(data_path)
    else:
        st.warning("Casting dataset not found. Please run setup.py first.")
        return

    # Simulate production batches (last 50 batches, 5 parts each)
    np.random.seed(7)
    n_batches   = 60
    batch_size  = 5
    total_parts = n_batches * batch_size

    # Sample from data with simulated time drift (process drifts after batch 40)
    sample_df = df_full.sample(min(total_parts, len(df_full)), random_state=7).reset_index(drop=True)
    # Introduce a simulated process drift at batch 40
    drift_start = 40 * batch_size
    sample_df.loc[drift_start:, 'pour_temp_c']       += 25   # operator increased temp
    sample_df.loc[drift_start:, 'carbon_equivalent'] += 0.15 # CE drifting high
    sample_df.loc[drift_start:, 'quality_score']     -= 12   # quality drops

    batch_nums      = np.arange(1, n_batches+1)
    quality_batches = [sample_df.iloc[i*batch_size:(i+1)*batch_size]['quality_score'].mean()
                       for i in range(n_batches)]
    temp_batches    = [sample_df.iloc[i*batch_size:(i+1)*batch_size]['pour_temp_c'].mean()
                       for i in range(n_batches)]
    CE_batches      = [sample_df.iloc[i*batch_size:(i+1)*batch_size]['carbon_equivalent'].mean()
                       for i in range(n_batches)]
    G_batches       = [sample_df.iloc[i*batch_size:(i+1)*batch_size]['G_factor'].mean()
                       for i in range(n_batches)]

    quality_arr = np.array(quality_batches)
    temp_arr    = np.array(temp_batches)
    CE_arr      = np.array(CE_batches)
    G_arr       = np.array(G_batches)

    # ── KPI Row ───────────────────────────────────────────────
    mu_q, ucl_q, lcl_q = control_limits(quality_arr[:40])  # limits from stable period
    mu_t, ucl_t, lcl_t = control_limits(temp_arr[:40])
    mu_ce,ucl_ce,lcl_ce= control_limits(CE_arr[:40])

    cpk_q  = cpk(quality_arr, usl=100,  lsl=62)
    cpk_t  = cpk(temp_arr,    usl=1570, lsl=1490)
    cpk_ce = cpk(CE_arr,      usl=4.6,  lsl=4.3)

    viol_q  = nelson_violations(quality_arr, ucl_q,  lcl_q,  mu_q)
    viol_t  = nelson_violations(temp_arr,    ucl_t,  lcl_t,  mu_t)
    viol_ce = nelson_violations(CE_arr,      ucl_ce, lcl_ce, mu_ce)
    total_v = len(viol_q) + len(viol_t) + len(viol_ce)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Avg Quality",     f"{quality_arr.mean():.1f}/100")
    c2.metric("Cpk Quality",     f"{cpk_q:.2f}",
              "✅ Capable" if cpk_q >= 1.33 else "⚠️ Marginal" if cpk_q >= 1.0 else "❌ Incapable")
    c3.metric("Cpk Temperature", f"{cpk_t:.2f}")
    c4.metric("Cpk CE",          f"{cpk_ce:.2f}")
    c5.metric("SPC Violations",  str(total_v),
              delta="🚨 Action needed!" if total_v > 3 else "✅ Stable")

    st.markdown("---")

    # ── Drift Alert ───────────────────────────────────────────
    if total_v > 3:
        st.error(f"🚨 **{total_v} SPC violations detected** — Process drift identified after batch 40. "
                 "Pour temperature and Carbon Equivalent trending OUT OF CONTROL. "
                 "Immediate corrective action recommended.")

    # ── Filter ────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1,3])
    with col_f1:
        show_batches = st.slider("Show last N batches", 20, 60, 60)
        show_cusum   = st.checkbox("Show CUSUM chart", True)
        show_nelson  = st.checkbox("Highlight violations", True)

    # Slice to selected range
    bn  = batch_nums[-show_batches:]
    q_s = quality_arr[-show_batches:]
    t_s = temp_arr[-show_batches:]
    c_s = CE_arr[-show_batches:]
    g_s = G_arr[-show_batches:]

    vq = nelson_violations(q_s, ucl_q, lcl_q, mu_q) if show_nelson else []
    vt = nelson_violations(t_s, ucl_t, lcl_t, mu_t) if show_nelson else []
    vc = nelson_violations(c_s, ucl_ce,lcl_ce,mu_ce)if show_nelson else []

    with col_f2:
        # ── Quality Xbar Chart ────────────────────────────────
        st.subheader("📈 Quality Score — Xbar Control Chart")
        st.plotly_chart(make_control_chart(
            bn, q_s, "Quality Score (batch avg)",
            ucl_q, lcl_q, mu_q, "Score", vq
        ), width="stretch")

        # ── Pour Temp Chart ───────────────────────────────────
        st.subheader("🌡️ Pour Temperature — Control Chart")
        st.plotly_chart(make_control_chart(
            bn, t_s, "Pour Temperature °C",
            ucl_t, lcl_t, mu_t, "°C", vt
        ), width="stretch")

        # ── CE Chart ─────────────────────────────────────────
        st.subheader("⚗️ Carbon Equivalent — Control Chart")
        st.plotly_chart(make_control_chart(
            bn, c_s, "Carbon Equivalent (CE)",
            ucl_ce, lcl_ce, mu_ce, "CE", vc
        ), width="stretch")

    st.markdown("---")

    # ── CUSUM Chart ───────────────────────────────────────────
    if show_cusum:
        st.subheader("📊 CUSUM Chart — Detects Small Sustained Quality Shifts")
        Sp, Sn = cusum(q_s)
        cusum_h = np.std(q_s) * 5   # decision interval
        cusum_fig = go.Figure()
        cusum_fig.add_trace(go.Scatter(x=bn, y=Sp, mode='lines',
            line=dict(color='#22c55e', width=2), name='CUSUM+'))
        cusum_fig.add_trace(go.Scatter(x=bn, y=-Sn, mode='lines',
            line=dict(color='#ef4444', width=2), name='CUSUM-'))
        cusum_fig.add_hline(y=cusum_h,  line_dash='dash', line_color='#f59e0b',
                            annotation_text=f'H={cusum_h:.1f}')
        cusum_fig.add_hline(y=-cusum_h, line_dash='dash', line_color='#f59e0b')
        cusum_fig.add_hline(y=0, line_color='white', line_width=0.5)
        cusum_fig.update_layout(
            height=260, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font_color='white',
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)', title='Batch #'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            legend=dict(orientation='h'), margin=dict(t=10,b=30,l=10,r=80)
        )
        st.plotly_chart(cusum_fig, width="stretch")
        if any(s > cusum_h for s in Sp) or any(s > cusum_h for s in Sn):
            st.warning("⚠️ **CUSUM signal detected** — Process mean has shifted. "
                       "Check last 5 batches for assignable cause.")

    st.markdown("---")

    # ── Process Capability Summary ────────────────────────────
    st.subheader("🏆 Process Capability Summary (Cpk)")
    cpk_data = [
        {'Parameter': '🎯 Quality Score',    'USL': 100,  'LSL': 62,   'Mean': f"{quality_arr.mean():.1f}", 'Cpk': f"{cpk_q:.3f}",  'Status': '✅ Capable' if cpk_q>=1.33 else '⚠️ Marginal' if cpk_q>=1.0 else '❌ Incapable'},
        {'Parameter': '🌡️ Pour Temperature', 'USL': 1570, 'LSL': 1490, 'Mean': f"{temp_arr.mean():.0f}",   'Cpk': f"{cpk_t:.3f}",  'Status': '✅ Capable' if cpk_t>=1.33 else '⚠️ Marginal' if cpk_t>=1.0 else '❌ Incapable'},
        {'Parameter': '⚗️ Carbon Equivalent','USL': 4.6,  'LSL': 4.3,  'Mean': f"{CE_arr.mean():.3f}",     'Cpk': f"{cpk_ce:.3f}", 'Status': '✅ Capable' if cpk_ce>=1.33 else '⚠️ Marginal' if cpk_ce>=1.0 else '❌ Incapable'},
        {'Parameter': '🔄 G-Factor',         'USL': 100,  'LSL': 40,   'Mean': f"{G_arr.mean():.1f}",      'Cpk': f"{cpk(G_arr,100,40):.3f}", 'Status': '✅ Capable' if cpk(G_arr,100,40)>=1.33 else '⚠️ Marginal'},
    ]
    st.dataframe(pd.DataFrame(cpk_data), width="stretch", hide_index=True)
    st.caption("Cpk ≥ 1.33 = Capable (Six Sigma Level) | 1.0–1.33 = Marginal | < 1.0 = Incapable")
