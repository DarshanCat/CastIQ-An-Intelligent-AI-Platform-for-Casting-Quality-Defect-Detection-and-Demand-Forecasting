"""
🧬 Multi-Objective Optimizer Module
Balance Quality vs Cost vs Delivery Time simultaneously.
Uses NSGA-II (pymoo) to compute full Pareto front.
File: modules/multi_objective_optimizer.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE      = Path(__file__).parent.parent
MODEL_DIR = BASE / 'backend' / 'models'

FEATS = [
    'carbon_pct','silicon_pct','manganese_pct','phosphorus_pct',
    'magnesium_added_pct','treatment_time_min','rpm','mold_diameter_m',
    'pour_temp_c','mold_preheat_c','wall_thickness_mm','mold_enc',
    'carbon_equivalent','superheat_c','G_factor','Mg_effective_pct',
    'nodularity_index','cooling_rate_cs'
]

# Variable bounds for optimization
BOUNDS_LOW  = [3.2, 1.8, 0.20, 0.01, 0.030, 2.0, 400, 0.05, 1490, 100, 8]
BOUNDS_HIGH = [3.9, 2.8, 0.65, 0.07, 0.085, 12., 1000,0.20, 1570, 350, 70]

PARAM_NAMES = ['C%','Si%','Mn%','P%','Mg%','TreatTime',
               'RPM','MoldDia','PourTemp','Preheat','WallThick']

def compute_physics(x, mold_enc):
    c,si,mn,p,mg,tt,rpm,dia,pt,pre,wt = x
    CE        = c + si/3 + p/3
    T_liq     = 1550 - 28*c - 8*si
    SH        = pt - T_liq
    G         = (np.pi**2 * dia * rpm**2) / (900 * 9.81)
    Mg_eff    = mg * np.exp(-0.02 * tt)
    nodularity= np.clip(0.60 + (Mg_eff-0.045)*8 - (tt-5)*0.015, 0, 1)
    B_map     = {0:0.10, 1:0.15, 2:0.08}
    t_sol     = B_map[int(mold_enc)] * (wt**2)
    c_rate    = np.clip((pt-pre)/(t_sol+1e-6), 0.5, 40)
    return CE, SH, G, Mg_eff, nodularity, c_rate

def x_to_features(x, mold_enc):
    CE, SH, G, Mg_eff, nodularity, c_rate = compute_physics(x, mold_enc)
    return np.array([*x, mold_enc, CE, SH, G, Mg_eff*100, nodularity, c_rate])

def compute_cost(x, batch_kg=200, tariff=8.5):
    """Material + Energy cost per batch (₹)"""
    c,si,mn,p,mg,tt,rpm,dia,pt,pre,wt = x
    # Energy cost
    sensible  = batch_kg * 500 * (pt - 30)
    latent    = batch_kg * 247000
    energy_kWh= (sensible + latent) * 1.25 / 0.75 / 3_600_000
    energy_cost = energy_kWh * tariff
    # Mg cost (expensive)
    mg_cost   = mg * batch_kg * 350   # ₹350/kg for Mg-Fe nodularizer
    # Base metal cost proportional to alloy quality
    base_cost = batch_kg * (68 + (c-3.2)*15 + (si-1.8)*8)
    return energy_cost + mg_cost + base_cost * 0.01  # scaled

def compute_delivery(x, mold_enc):
    """Estimated delivery days from order"""
    _,_,_,_,_,_ = compute_physics(x, mold_enc)
    wt = x[10]
    B_map = {0:0.10, 1:0.15, 2:0.08}
    t_sol_min = B_map[int(mold_enc)] * (wt**2) / 60  # minutes
    melt_time = 90       # base melt time (min)
    setup_time= 30       # setup
    machining = 120      # finish machining
    dispatch  = 2        # days
    total_min = melt_time + setup_time + t_sol_min + machining
    total_days= total_min / 480 + dispatch  # 480 min/8hr shift
    return round(total_days, 1)

def run_pareto_nsga2(reg, scaler, mold_enc, n_gen=50, pop_size=80):
    """
    NSGA-II multi-objective optimization.
    Objectives: minimize [-quality, cost, delivery_days]
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling

        class CastingProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=11, n_obj=3, n_constr=0,
                    xl=np.array(BOUNDS_LOW),
                    xu=np.array(BOUNDS_HIGH)
                )
            def _evaluate(self, X, out, *args, **kwargs):
                f1, f2, f3 = [], [], []
                for x in X:
                    feats = x_to_features(x, mold_enc).reshape(1,-1)
                    q     = float(reg.predict(scaler.transform(feats))[0])
                    c     = compute_cost(x)
                    d     = compute_delivery(x, mold_enc)
                    f1.append(-q)      # maximize quality → minimize negative
                    f2.append(c)       # minimize cost
                    f3.append(d)       # minimize delivery days
                out["F"] = np.column_stack([f1, f2, f3])

        problem  = CastingProblem()
        algo     = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        res = pymoo_min(problem, algo, ('n_gen', n_gen), seed=1, verbose=False)
        return res.X, res.F
    except Exception as e:
        # Fallback: random sampling if pymoo fails
        np.random.seed(42)
        X_pop = np.random.uniform(BOUNDS_LOW, BOUNDS_HIGH, (400, 11))
        F_pop = []
        for x in X_pop:
            feats = x_to_features(x, mold_enc).reshape(1,-1)
            q = float(reg.predict(scaler.transform(feats))[0])
            c = compute_cost(x)
            d = compute_delivery(x, mold_enc)
            F_pop.append([-q, c, d])
        F_pop = np.array(F_pop)
        # Filter Pareto front
        pareto_mask = np.ones(len(F_pop), dtype=bool)
        for i in range(len(F_pop)):
            for j in range(len(F_pop)):
                if i != j and all(F_pop[j] <= F_pop[i]) and any(F_pop[j] < F_pop[i]):
                    pareto_mask[i] = False
                    break
        return X_pop[pareto_mask], F_pop[pareto_mask]

def render_multi_objective():
    st.title("🧬 Multi-Objective Optimizer")
    st.markdown("##### Find the Pareto-optimal balance between Quality, Cost, and Delivery")
    st.markdown("---")

    try:
        reg    = joblib.load(MODEL_DIR/'regressor.pkl')
        scaler = joblib.load(MODEL_DIR/'scaler.pkl')
    except Exception as e:
        st.error(f"❌ Models not found: {e}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Settings")
        mold_type  = st.selectbox("Mold Type", ['Permanent','Sand','Die'])
        mold_enc   = {'Permanent':0,'Sand':1,'Die':2}[mold_type]
        batch_kg   = st.slider("Batch Size (kg)", 50, 500, 200)
        tariff     = st.number_input("Electricity (₹/kWh)", 4.0, 20.0, 8.5, 0.5)
        n_gen      = st.slider("NSGA-II Generations", 20, 100, 50)
        pop_size   = st.slider("Population Size", 40, 120, 80)

        st.markdown("---")
        st.markdown("**🎯 Your Priority**")
        pref = st.radio("Optimize mainly for:", [
            "⚖️ Balanced (all equal)",
            "🏆 Maximize Quality",
            "💰 Minimize Cost",
            "🚀 Fastest Delivery"
        ])

        run_btn = st.button("🧬 Run NSGA-II Optimization",
                             use_container_width=True, type="primary")

    with col2:
        if run_btn:
            with st.spinner("🧬 Running NSGA-II multi-objective optimization..."):
                X_pareto, F_pareto = run_pareto_nsga2(
                    reg, scaler, mold_enc, n_gen, pop_size
                )

            quality_vals  = -F_pareto[:, 0]
            cost_vals     =  F_pareto[:, 1]
            delivery_vals =  F_pareto[:, 2]

            n_solutions = len(quality_vals)

            st.success(f"✅ Found **{n_solutions} Pareto-optimal solutions** — "
                       "each is a different trade-off. No one solution dominates all objectives.")

            # ── 3D Pareto front ───────────────────────────────
            st.subheader("🌐 3D Pareto Front")
            color_map = quality_vals
            fig_3d = go.Figure(go.Scatter3d(
                x=quality_vals, y=cost_vals, z=delivery_vals,
                mode='markers',
                marker=dict(
                    size=6, color=color_map,
                    colorscale='RdYlGn', showscale=True,
                    colorbar=dict(title='Quality', x=1.05)
                ),
                text=[f"Q:{q:.1f} C:₹{c:.0f} D:{d:.1f}d"
                      for q,c,d in zip(quality_vals,cost_vals,delivery_vals)],
                hovertemplate='Quality: %{x:.1f}<br>Cost: ₹%{y:.0f}<br>Days: %{z:.1f}<extra></extra>'
            ))
            fig_3d.update_layout(
                height=420,
                scene=dict(
                    xaxis_title='Quality Score',
                    yaxis_title='Cost (₹)',
                    zaxis_title='Delivery (days)',
                    bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                ),
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                margin=dict(t=20,b=20,l=0,r=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            # ── 2D scatter: Quality vs Cost ───────────────────
            st.subheader("📊 Quality vs Cost Trade-off")
            fig_2d = go.Figure(go.Scatter(
                x=cost_vals, y=quality_vals,
                mode='markers',
                marker=dict(size=8, color=delivery_vals,
                            colorscale='Blues', showscale=True,
                            colorbar=dict(title='Delivery (days)')),
                hovertemplate='Cost: ₹%{x:.0f}<br>Quality: %{y:.1f}<br><extra></extra>'
            ))
            fig_2d.update_layout(
                height=280, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(title='Cost (₹/batch)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title='Quality Score', gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=10,b=40,l=10,r=80)
            )
            st.plotly_chart(fig_2d, use_container_width=True)

            # ── Pick best for user preference ─────────────────
            st.subheader("🎯 Best Solution for Your Priority")
            if pref == "🏆 Maximize Quality":
                best_idx = np.argmax(quality_vals)
            elif pref == "💰 Minimize Cost":
                best_idx = np.argmin(cost_vals)
            elif pref == "🚀 Fastest Delivery":
                best_idx = np.argmin(delivery_vals)
            else:
                # Balanced: normalize each objective and sum
                q_n = (quality_vals - quality_vals.min()) / ((quality_vals.max() - quality_vals.min()) + 1e-9)
                c_n = (cost_vals.max() - cost_vals) / ((cost_vals.max() - cost_vals.min()) + 1e-9)
                d_n = (delivery_vals.max() - delivery_vals) / ((delivery_vals.max() - delivery_vals.min()) + 1e-9)
                best_idx = np.argmax(q_n + c_n + d_n)

            best_x = X_pareto[best_idx]
            b1,b2,b3 = st.columns(3)
            b1.metric("🏆 Quality Score",   f"{quality_vals[best_idx]:.1f}/100")
            b2.metric("💰 Cost/Batch",       f"₹{cost_vals[best_idx]:.0f}")
            b3.metric("🚀 Delivery",         f"{delivery_vals[best_idx]:.1f} days")

            # Best parameter recipe
            st.markdown("**Optimal Parameter Recipe:**")
            recipe_df = pd.DataFrame({
                'Parameter': PARAM_NAMES,
                'Value': [f"{v:.4f}" for v in best_x],
                'Unit': ['%','%','%','%','%','min','RPM','m','°C','°C','mm']
            })
            st.dataframe(recipe_df, use_container_width=True, hide_index=True)

            # ── All Pareto solutions table ─────────────────────
            st.markdown("---")
            st.subheader(f"📋 All {n_solutions} Pareto-Optimal Solutions")
            table_df = pd.DataFrame({
                'Solution': range(1, n_solutions+1),
                'Quality': quality_vals.round(1),
                'Cost (₹)': cost_vals.round(0).astype(int),
                'Delivery (days)': delivery_vals.round(1),
                'Rec.': ['⭐' if i == best_idx else '' for i in range(n_solutions)]
            }).sort_values('Quality', ascending=False)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

        else:
            st.markdown("### 👈 Configure and click **Run NSGA-II Optimization**")
            st.info("""
**What is Multi-Objective Optimization?**

In casting, you can't have everything at once:
- **Higher quality** → more expensive alloy, more time
- **Lower cost** → compromise on Mg% or pour temp
- **Faster delivery** → thinner wall, less solidification time

NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds the
**Pareto Front** — the set of solutions where improving one objective
requires sacrificing another.

**3 Objectives minimized simultaneously:**
1. 🏆 **Quality** → maximize (modelled as minimize negative)
2. 💰 **Cost** → minimize (energy + Mg + base metal)
3. 🚀 **Delivery** → minimize (melt + solidify + machine time)

You pick where on the Pareto front you want to operate
based on the customer's priority.
            """)
