"""
VSPL AI Platform — One-shot setup
Run this once: python setup.py
"""
import sys
import io

# Prevent UnicodeEncodeError on Windows standard output when printing emojis
if sys.platform.startswith('win') and (sys.stdout is None or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

BASE   = Path(__file__).parent
MDIR   = BASE / 'backend' / 'models'
DDIR   = BASE / 'backend' / 'data'
MDIR.mkdir(exist_ok=True)

print("=" * 55)
print("  VSPL AI Platform — Setup")
print("=" * 55)

# ── 1. Casting Quality Data & Model ───────────────────────────
print("\n[1/3] Training Casting Quality Predictor (Physics-Based)...")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

large_csv_path = DDIR / 'casting_data_large.csv'
if not large_csv_path.exists():
    print("   [!] Physics dataset not found. Generating now...")
    import subprocess
    subprocess.run([sys.executable, str(DDIR / 'generate_physics_data.py')], check=True)

cast_df = pd.read_csv(large_csv_path)

le_cast = LabelEncoder()
cast_df['mold_enc'] = le_cast.fit_transform(cast_df['mold_type'])

feats = [
    'carbon_pct', 'silicon_pct', 'manganese_pct', 'phosphorus_pct',
    'magnesium_added_pct', 'treatment_time_min', 'rpm', 'mold_diameter_m',
    'pour_temp_c', 'mold_preheat_c', 'wall_thickness_mm', 'mold_enc',
    'carbon_equivalent', 'superheat_c', 'G_factor',
    'Mg_effective_pct', 'nodularity_index', 'cooling_rate_cs'
]

X = cast_df[feats]
y_c = (cast_df['quality_score'] < 20.0).astype(int)
y_r = cast_df['quality_score']

Xt, Xe, yct, yce, yrt, yre = train_test_split(X, y_c, y_r, test_size=0.2, random_state=42)
sc = StandardScaler()
Xts = sc.fit_transform(Xt)
Xes = sc.transform(Xe)

clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(Xts, yct)
reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
reg.fit(Xts, yrt)

acc = accuracy_score(yce, clf.predict(Xes))
mae = mean_absolute_error(yre, reg.predict(Xes))
print(f"   ✅ Classifier accuracy : {acc*100:.1f}%")
print(f"   ✅ Regressor MAE       : {mae:.2f} pts")

joblib.dump(clf, MDIR/'classifier.pkl')
joblib.dump(reg, MDIR/'regressor.pkl')
joblib.dump(sc,  MDIR/'scaler.pkl')
joblib.dump(le_cast, MDIR/'label_encoder.pkl')
joblib.dump(feats, MDIR/'features.pkl')

# ── 2. Demand Forecasting ─────────────────────────────────────
print("\n[2/3] Training Demand Forecasting models...")

from prophet import Prophet

SECTORS = {
    'Hydraulics':        {'base': 42, 'trend': 0.4,  'seasonality': 6,  'noise': 4},
    'Wind Energy':       {'base': 35, 'trend': 0.8,  'seasonality': 10, 'noise': 5},
    'Locomotives':       {'base': 28, 'trend': 0.2,  'seasonality': 4,  'noise': 3},
    'Pharma':            {'base': 18, 'trend': 0.6,  'seasonality': 3,  'noise': 2},
    'Heavy Engineering': {'base': 55, 'trend': 0.3,  'seasonality': 8,  'noise': 6},
}

np.random.seed(7)
dates = pd.date_range('2023-01-01', periods=30, freq='MS')
rows  = []
for sec, cfg in SECTORS.items():
    for i, dt in enumerate(dates):
        qty = max(5, cfg['base'] + cfg['trend']*i
                  + cfg['seasonality']*np.sin(2*np.pi*i/12)
                  + np.random.normal(0, cfg['noise']))
        rows.append({'date': dt, 'sector': sec, 'orders': round(qty,1)})
demand_df = pd.DataFrame(rows)
demand_df.to_csv(DDIR/'demand_data.csv', index=False)

all_fc = []
for sec in SECTORS:
    sub = demand_df[demand_df['sector']==sec][['date','orders']].copy()
    sub.columns = ['ds','y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.1)
    m.fit(sub)
    fc = m.predict(m.make_future_dataframe(periods=6, freq='MS'))
    pred = fc[['ds','yhat','yhat_lower','yhat_upper']].copy()
    pred.columns = ['date','predicted','lower','upper']
    pred[['predicted','lower','upper']] = pred[['predicted','lower','upper']].clip(lower=0).round(1)
    merged = pred.merge(sub.rename(columns={'ds':'date','y':'actual'}), on='date', how='left')
    merged['sector'] = sec
    all_fc.append(merged)
    print(f"   ✅ {sec}")

combined = pd.concat(all_fc, ignore_index=True)
combined['date'] = combined['date'].astype(str)
combined.to_csv(MDIR/'demand_forecast.csv', index=False)

# ── 3. Alloy Recommendation Model ────────────────────────────
print("\n[3/3] Training Alloy Recommendation Engine...")

from sklearn.ensemble import RandomForestClassifier

with open(DDIR/'alloy_grades.json', encoding='utf-8') as f:
    GRADES = json.load(f)['grades']

np.random.seed(99)
arows = []
for _ in range(1000):
    g = np.random.choice(GRADES)
    corr_map = {'low':1,'medium':2,'high':3}
    arows.append({
        'tensile':    np.random.uniform(g['tensile_min'], g['tensile_max']),
        'hardness':   np.random.uniform(g['hardness_min'], g['hardness_max']),
        'temp':       np.random.uniform(100, g['temp_max']),
        'corrosion':  corr_map[g['corrosion']] + np.random.choice([-1,0,0,1]),
        'thickness':  np.random.uniform(5, 80),
        'grade':      g['grade']
    })
alloy_df = pd.DataFrame(arows)
alloy_df['corrosion'] = alloy_df['corrosion'].clip(1,3)

le_a  = LabelEncoder()
sc_a  = StandardScaler()
af    = ['tensile','hardness','temp','corrosion','thickness']
Xa    = sc_a.fit_transform(alloy_df[af])
ya    = le_a.fit_transform(alloy_df['grade'])
Xat, Xae, yat, yae = train_test_split(Xa, ya, test_size=0.2, random_state=42)
clf_a = RandomForestClassifier(n_estimators=200, random_state=42)
clf_a.fit(Xat, yat)
print(f"   ✅ Accuracy: {accuracy_score(yae, clf_a.predict(Xae))*100:.1f}%")

joblib.dump(clf_a, MDIR/'alloy_clf.pkl')
joblib.dump(sc_a,  MDIR/'alloy_scaler.pkl')
joblib.dump(le_a,  MDIR/'alloy_le.pkl')

print("\n" + "="*55)
print("  ✅ Setup complete! All models saved.")
print("  👉 Run: streamlit run app.py")
print("="*55)
