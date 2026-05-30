import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N      = 5000
BASE   = Path(__file__).parent

def generate(n=N):
    # Mix: 70% normal ops, 30% with deliberate process deviations
    n_good = int(n * 0.70)
    n_bad  = n - n_good

    def sample(low, high, size): return np.random.uniform(low, high, size)

    def make_batch(size, bad=False):
        if bad:
            # Operators pushed parameters outside optimal
            c   = sample(3.0, 4.2, size)
            si  = sample(1.4, 3.2, size)
            mn  = sample(0.15, 0.80, size)
            p   = sample(0.02, 0.10, size)
            mg  = sample(0.020, 0.060, size)
            tt  = sample(5, 18, size)          # longer treatment = more fade
            rpm = sample(200, 600, size)        # too slow OR
            rpm = np.where(np.random.rand(size)>0.5, sample(1100, 1600, size), rpm)
            dia = sample(0.05, 0.25, size)
            pt  = sample(1440, 1610, size)      # under/over superheat
            pre = sample(50, 160, size)         # cold mold
            wt  = sample(5, 90, size)
        else:
            c   = sample(3.3, 3.8, size)
            si  = sample(1.9, 2.7, size)
            mn  = sample(0.25, 0.55, size)
            p   = sample(0.01, 0.05, size)
            mg  = sample(0.038, 0.075, size)
            tt  = sample(2, 8, size)
            rpm = sample(500, 900, size)
            dia = sample(0.08, 0.18, size)
            pt  = sample(1500, 1560, size)
            pre = sample(150, 300, size)
            wt  = sample(10, 60, size)
        mt = np.random.choice(['Permanent','Sand','Die'], size, p=[0.55,0.30,0.15])
        return c,si,mn,p,mg,tt,rpm,dia,pt,pre,wt,mt

    parts = [make_batch(n_good, bad=False), make_batch(n_bad, bad=True)]
    arrs  = [np.concatenate([p[i] for p in parts]) for i in range(11)]
    mold_type = np.concatenate([p[11] for p in parts])
    carbon_pct,silicon_pct,manganese_pct,phosphorus_pct,\
        mg_added,treat_time,rpm,mold_dia,pour_temp,preheat,wall_mm = arrs

    # Shuffle
    idx = np.random.permutation(n)
    carbon_pct,silicon_pct,manganese_pct,phosphorus_pct,\
        mg_added,treat_time,rpm,mold_dia,pour_temp,preheat,wall_mm = \
        [a[idx] for a in arrs]
    mold_type = mold_type[idx]

    # ── FORMULA 1: Carbon Equivalent ──────────────────────────
    CE     = carbon_pct + silicon_pct/3.0 + phosphorus_pct/3.0
    CE_pen = np.where(CE<4.3, (4.3-CE)*15, np.where(CE>4.6, (CE-4.6)*20, 0))

    # ── FORMULA 2: Liquidus & Superheat ───────────────────────
    T_liq  = 1550 - 28*carbon_pct - 8*silicon_pct
    SH     = pour_temp - T_liq
    SH_pen = np.where(SH<60, (60-SH)*0.35, np.where(SH>120, (SH-120)*0.28, 0))

    # ── FORMULA 3: G-Factor ───────────────────────────────────
    G      = (np.pi**2 * mold_dia * rpm**2) / (900 * 9.81)
    G_pen  = np.where(G<40, (40-G)*0.30, np.where(G>100, (G-100)*0.20, 0))

    # ── FORMULA 4: Mg Recovery ────────────────────────────────
    Mg_eff = mg_added * np.exp(-0.02 * treat_time)
    Mg_pen = np.where(Mg_eff<0.035, (0.035-Mg_eff)*800,
             np.where(Mg_eff>0.060, (Mg_eff-0.060)*500, 0))

    # ── FORMULA 5: Chvorinov ──────────────────────────────────
    B      = np.where(mold_type=='Permanent',0.10,np.where(mold_type=='Sand',0.15,0.08))
    t_sol  = B * (wall_mm**2)
    c_rate = np.clip((pour_temp-preheat)/(t_sol+1e-6), 0.5, 40)

    # ── FORMULA 6: Preheat ────────────────────────────────────
    pre_pen = np.where(preheat<150,(150-preheat)*0.10,
              np.where(preheat>300,(preheat-300)*0.08, 0))

    # ── Nodularity ────────────────────────────────────────────
    nodularity = np.clip(
        0.60 + (Mg_eff - 0.045)*8 - (treat_time-5)*0.015
             + np.random.normal(0, 0.04, n), 0, 1
    )

    # ── Quality ───────────────────────────────────────────────
    quality  = 100.0
    quality -= CE_pen; quality -= SH_pen; quality -= G_pen
    quality -= Mg_pen; quality -= pre_pen
    quality += nodularity*6
    quality -= np.abs(c_rate-10)*0.06
    quality += np.random.normal(0, 2.5, n)
    quality  = np.clip(quality, 0, 100)
    reject   = (quality<62).astype(int)

    mold_enc = np.array([{'Permanent':0,'Sand':1,'Die':2}[m] for m in mold_type])

    df = pd.DataFrame({
        'carbon_pct':carbon_pct.round(3),'silicon_pct':silicon_pct.round(3),
        'manganese_pct':manganese_pct.round(3),'phosphorus_pct':phosphorus_pct.round(4),
        'magnesium_added_pct':mg_added.round(4),'treatment_time_min':treat_time.round(1),
        'rpm':rpm.round(0).astype(int),'mold_diameter_m':mold_dia.round(3),
        'pour_temp_c':pour_temp.round(1),'mold_preheat_c':preheat.round(1),
        'wall_thickness_mm':wall_mm.round(1),'mold_type':mold_type,'mold_enc':mold_enc,
        # Physics features
        'carbon_equivalent':CE.round(4),'liquidus_temp_c':T_liq.round(1),
        'superheat_c':SH.round(1),'G_factor':G.round(2),
        'Mg_effective_pct':Mg_eff.round(5),'nodularity_index':nodularity.round(3),
        'solidification_time_s':t_sol.round(2),'cooling_rate_cs':c_rate.round(2),
        'quality_score':quality.round(2),'reject':reject
    })

    print(f"  Samples   : {n}")
    print(f"  Reject    : {reject.mean()*100:.1f}%")
    print(f"  Avg qual  : {quality.mean():.1f}")
    print(f"  CE        : {CE.min():.2f}–{CE.max():.2f}  (optimal 4.3–4.6)")
    print(f"  G         : {G.min():.0f}–{G.max():.0f}G  (optimal 40–100G)")
    print(f"  Superheat : {SH.min():.0f}–{SH.max():.0f}°C  (optimal 60–100°C)")
    out = BASE/'casting_physics_data.csv'
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return df

if __name__ == "__main__":
    generate()