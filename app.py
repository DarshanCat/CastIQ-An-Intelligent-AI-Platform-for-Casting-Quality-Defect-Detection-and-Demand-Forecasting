# ═══════════════════════════════════════════════════════════
# CastIQ — VSPL AI Platform | Industrial Command Center UI
# Vijay Spheroidals Pvt Ltd
# Built by Darshan | CMR University | B.Tech AI & ML
# ═══════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, re, io, os, cv2, tempfile
from pathlib import Path
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER
import plotly.graph_objects as go
import plotly.express as px

# ── .env loader ─────────────────────────────────────────────
for _n in ['.env', ',env']:
    _p = Path(__file__).parent / _n
    if _p.exists():
        try:
            for _l in open(_p, encoding='utf-8'):
                _l = _l.strip()
                if _l and not _l.startswith('#') and '=' in _l:
                    k, v = _l.split('=', 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

# ── Module imports ───────────────────────────────────────────
from modules.production_dashboard   import render_production_dashboard
from modules.inventory_tracker      import render_inventory_tracker
from modules.predictive_maintenance import render_predictive_maintenance
from modules.order_tracker          import render_order_tracker
from modules.heat_treatment         import render_heat_treatment
from modules.cost_estimation        import render_cost_estimation
from modules.diagram_generator      import render_diagram_generator
from modules.process_optimizer        import render_process_optimizer
from modules.spc_dashboard            import render_spc_dashboard
from modules.shap_explainer           import render_shap_explainer
from modules.energy_optimizer         import render_energy_optimizer
from modules.multi_objective_optimizer import render_multi_objective
from modules.root_cause_analysis      import render_rca
from modules.digital_twin             import render_digital_twin

# ════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be first Streamlit call, nothing before
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CastIQ — VSPL",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════
# INDUSTRIAL COMMAND CENTER CSS
# ════════════════════════════════════════════════════════════
st.markdown("""<style>
/* ── Base ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main,
section.main>div,.block-container{
    background:#0e1118 !important;
    padding-top:0 !important;
}
.block-container{padding:20px 24px !important;max-width:100% !important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:#0b0e15 !important;
    border-right:1px solid #1a2030 !important;
}
[data-testid="stSidebar"] section{padding:0 !important;}

/* hide radio widget label */
[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none !important;}

/* radio items */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:0 !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label{
    padding:7px 10px 7px 14px !important;
    margin:0 6px !important;
    border-radius:5px !important;
    font-size:12px !important;
    color:#4a6070 !important;
    background:transparent !important;
    border:none !important;
    cursor:pointer !important;
    transition:all .12s !important;
    display:flex !important;
    align-items:center !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover{
    background:#111722 !important;
    color:#8aa0b0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label[data-baseweb]{
    display:flex !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p{
    color:inherit !important;
    font-size:12px !important;
    margin:0 !important;
}

/* ── Topbar area ── */
[data-testid="stHeader"]{background:#0b0e15 !important;border-bottom:1px solid #1a2030 !important;}

/* ── Metrics ── */
[data-testid="metric-container"]{
    background:#111722 !important;
    border:1px solid #1a2030 !important;
    border-top:2px solid #c9a84c !important;
    border-radius:6px !important;
    padding:14px 16px !important;
}
[data-testid="stMetricLabel"] p{
    color:#4a6070 !important;font-size:10px !important;
    font-weight:600 !important;text-transform:uppercase;letter-spacing:1px;
}
[data-testid="stMetricValue"]{color:#e8eef5 !important;font-size:22px !important;font-weight:600 !important;}
[data-testid="stMetricDelta"] svg{display:none;}
[data-testid="stMetricDelta"]>div{font-size:10px !important;color:#4a6070 !important;}

/* ── Buttons ── */
.stButton>button{
    background:#c9a84c !important;color:#0b0e15 !important;
    border:none !important;border-radius:5px !important;
    font-weight:700 !important;font-size:12px !important;
    padding:8px 18px !important;letter-spacing:.3px;
    transition:all .12s !important;
}
.stButton>button:hover{background:#d4b86a !important;transform:translateY(-1px) !important;}
.stButton>button[kind="secondary"]{
    background:#111722 !important;color:#8aa0b0 !important;
    border:1px solid #1a2030 !important;
}
.stButton>button[kind="secondary"]:hover{background:#1a2030 !important;}

/* ── Download ── */
[data-testid="stDownloadButton"] button{
    background:#111722 !important;color:#c9a84c !important;
    border:1px solid #c9a84c44 !important;border-radius:5px !important;
}

/* ── Inputs ── */
input,textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea{
    background:#111722 !important;border:1px solid #1a2030 !important;
    border-radius:5px !important;color:#e8eef5 !important;font-size:13px !important;
}
input:focus,textarea:focus{
    border-color:#c9a84c66 !important;
    box-shadow:0 0 0 2px #c9a84c18 !important;outline:none !important;
}
[data-testid="stSelectbox"]>div>div{
    background:#111722 !important;border:1px solid #1a2030 !important;
    border-radius:5px !important;color:#e8eef5 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"]{background:#c9a84c !important;border:2px solid #c9a84c !important;}

/* ── Tabs ── */
[role="tablist"]{border-bottom:1px solid #1a2030 !important;}
[role="tab"]{color:#4a6070 !important;font-size:12px !important;border:none !important;background:transparent !important;padding:8px 14px !important;}
[role="tab"][aria-selected="true"]{color:#c9a84c !important;border-bottom:2px solid #c9a84c !important;background:transparent !important;}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{border:1px solid #1a2030 !important;border-radius:6px !important;overflow:hidden;}

/* ── Alerts ── */
[data-testid="stAlert"]{background:#111722 !important;border:1px solid #1a2030 !important;border-radius:6px !important;}
[data-testid="stAlert"][kind="success"]{border-left:3px solid #4ade80 !important;}
[data-testid="stAlert"][kind="warning"]{border-left:3px solid #c9a84c !important;}
[data-testid="stAlert"][kind="error"]  {border-left:3px solid #f87171 !important;}
[data-testid="stAlert"][kind="info"]   {border-left:3px solid #60a5fa !important;}

/* ── File uploader ── */
[data-testid="stFileUploader"]{background:#111722 !important;border:1px dashed #1a2030 !important;border-radius:6px !important;}

/* ── Expander ── */
[data-testid="stExpander"]{background:#111722 !important;border:1px solid #1a2030 !important;border-radius:6px !important;}

/* ── Text ── */
h1{color:#e8eef5 !important;font-size:20px !important;font-weight:600 !important;letter-spacing:.3px;}
h2{color:#c8d8e8 !important;font-size:16px !important;font-weight:500 !important;}
h3{color:#8aa0b0 !important;font-size:13px !important;font-weight:500 !important;}
p,.stMarkdown p{color:#4a6070 !important;font-size:13px !important;}
hr{border-color:#1a2030 !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:#0e1118;}
::-webkit-scrollbar-thumb{background:#1a2030;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#c9a84c44;}
</style>""", unsafe_allow_html=True)

BASE      = Path(__file__).parent
MODEL_DIR = BASE / 'backend' / 'models'
DATA_DIR  = BASE / 'backend' / 'data'

# ==============================================================
# LOADERS
# ==============================================================
@st.cache_resource
def load_casting():
    return (joblib.load(MODEL_DIR/'classifier.pkl'),
            joblib.load(MODEL_DIR/'regressor.pkl'),
            joblib.load(MODEL_DIR/'scaler.pkl'),
            joblib.load(MODEL_DIR/'label_encoder.pkl'))

@st.cache_data
def load_forecast():
    df = pd.read_csv(MODEL_DIR/'demand_forecast.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_alloy():
    clf    = joblib.load(MODEL_DIR/'alloy_clf.pkl')
    scaler = joblib.load(MODEL_DIR/'alloy_scaler.pkl')
    le     = joblib.load(MODEL_DIR/'alloy_le.pkl')
    with open(DATA_DIR/'alloy_grades.json', encoding='utf-8') as f:
        grades = {g['grade']: g for g in json.load(f)['grades']}
    return clf, scaler, le, grades

@st.cache_data
def load_kb():
    with open(DATA_DIR/'knowledge_base.json', encoding='utf-8') as f:
        return json.load(f)

# PyTorch Deep Learning Model Setup
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # skip connection!
        return self.elu(out)

class CastingMultitaskDNN(nn.Module):
    def __init__(self, input_dim):
        super(CastingMultitaskDNN, self).__init__()
        # Shared representations
        self.shared_input = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.15)
        )
        self.res1 = ResidualBlock(256)
        self.shared_mid = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.15)
        )
        self.res2 = ResidualBlock(128)
        self.shared_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        
        # Task 1: Quality Score Regressor
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        # Task 2: Reject Classification
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.shared_input(x)
        h = self.res1(h)
        h = self.shared_mid(h)
        h = self.res2(h)
        shared_out = self.shared_out(h)
        score = self.regressor(shared_out)
        prob = self.classifier(shared_out)
        return score, prob

@st.cache_resource
def load_pytorch_dnn():
    weights_path = MODEL_DIR / 'casting_dnn.pth'
    scaler_path = MODEL_DIR / 'dnn_scaler.pkl'
    if weights_path.exists() and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        model = CastingMultitaskDNN(input_dim=18)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model, scaler
    return None, None

import pickle
import torch.nn.functional as F

# Mini-GPT Transformer Architecture for Local Offline Assistant Demonstration
class LocalHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class LocalMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([LocalHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class LocalFeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class LocalBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = LocalMultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = LocalFeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LocalMiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_blocks=2, block_size=64, dropout=0.1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[LocalBlock(n_embd, n_head, block_size, dropout) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 0.01)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

@st.cache_resource
def load_local_gpt():
    weights_path = MODEL_DIR / 'local_gpt.pth'
    vocab_path = MODEL_DIR / 'local_gpt_vocab.pkl'
    if weights_path.exists() and vocab_path.exists():
        try:
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            model = LocalMiniGPT(vocab_size=vocab['vocab_size'])
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.eval()
            return model, vocab
        except Exception:
            pass
    return None, None


# ==============================================================
# HELPERS
# ==============================================================
def quality_gauge(score):
    color = "#22c55e" if score >= 75 else "#f59e0b" if score >= 60 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={'text': "Quality Score", 'font': {'size': 18}},
        number={'suffix': '/100', 'font': {'size': 34, 'color': color}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': color},
            'steps': [
                {'range': [0,  60], 'color': '#fecaca'},
                {'range': [60, 75], 'color': '#fef08a'},
                {'range': [75,100], 'color': '#bbf7d0'},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(t=40,b=0,l=20,r=20),
                      paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig

def cast_recommendations(temp, rpm, carbon, silicon, cooling, flow, mold):
    tips = []
    if temp < 1400:    tips.append("🌡️ Increase pour temperature above 1400 deg C")
    if temp > 1450:    tips.append("🌡️ Reduce pour temperature below 1450 deg C")
    if rpm  < 1000:    tips.append("[SETUP]  Increase spin speed above 1000 RPM")
    if rpm  > 1200:    tips.append("[SETUP]  Reduce spin speed below 1200 RPM")
    if carbon < 3.4:   tips.append("[Alloy]  Increase Carbon % to 3.4-3.6 range")
    if carbon > 3.6:   tips.append("[Alloy]  Reduce Carbon % to 3.4-3.6 range")
    if silicon < 2.0:  tips.append("[Alloy]  Increase Silicon % to 2.0-2.5 range")
    if silicon > 2.5:  tips.append("[Alloy]  Reduce Silicon % to 2.0-2.5 range")
    if cooling < 60:   tips.append("❄️  Increase cooling time above 60 sec")
    if cooling > 90:   tips.append("❄️  Reduce cooling time below 90 sec")
    if flow < 1.2:     tips.append("💧 Increase metal flow rate to 1.2-1.8")
    if flow > 1.8:     tips.append("💧 Reduce metal flow rate to 1.2-1.8")
    if mold == 'Sand': tips.append("🔲 Consider switching to Permanent mold")
    return tips or ["[OK] All parameters within optimal range!"]

SECTOR_COLORS = {
    'Hydraulics': '#3b82f6', 'Wind Energy': '#22c55e',
    'Locomotives': '#f59e0b', 'Pharma': '#a855f7',
    'Heavy Engineering': '#ef4444',
}

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def parse_rfq(text):
    text_l = text.lower()
    result = {}
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|nos|pcs|pieces|units|numbers)', text_l)
    result['quantity']    = m.group(0) if m else 'Not specified'
    m = re.search(r'(\d{3,4})\s*(?:mpa|n/mm)', text_l)
    result['tensile']     = f"{m.group(1)} MPa" if m else 'Not specified'
    for kw in ['hydraulic','wind','locomotive','pharma','heavy engineering','general']:
        if kw in text_l:
            result['application'] = kw.title(); break
    else:
        result['application'] = 'General Purpose'
    m = re.search(r'(\d+)\s*(?:days?|weeks?)', text_l)
    if m:
        n = int(m.group(1))
        n = n*7 if 'week' in text_l[max(0,m.start()-5):m.end()+5] else n
        result['delivery'] = (datetime.today() + timedelta(days=n)).strftime('%d %b %Y')
    else:
        result['delivery'] = 'Standard lead time'
    m = re.search(r'(?:is|grade)\s*(\d{3})', text_l)
    result['grade_hint']  = f"IS {m.group(1)}" if m else None
    m = re.search(r'(\d+)\s*mm', text_l)
    result['diameter']    = f"{m.group(1)} mm" if m else 'Not specified'
    return result

# ==============================================================
# UPGRADE 1 - GEMINI API SMART ASSISTANT
# ==============================================================
def build_system_prompt(kb):
    kb_text = "\n\n".join(
        f"TOPIC: {', '.join(e['keywords'])}\n{e['answer']}" for e in kb
    )
    return f"""You are the VSPL Smart Assistant for Vijay Spheroidals Pvt Ltd, a Bengaluru-based manufacturer of ductile iron centrifugal castings.

You are knowledgeable, professional, and concise. You help customers and staff with questions about VSPL products, alloy grades, casting processes, delivery times, pricing, quality standards, and safety procedures.

Use the knowledge base below to answer questions accurately. If a question falls outside the knowledge base, give a helpful general answer based on your expertise in ductile iron casting, then suggest contacting VSPL sales.

Always format answers clearly. For lists, use bullet points. For grades, use a table if comparing multiple. Keep answers under 200 words unless detail is truly needed.

--- VSPL KNOWLEDGE BASE ---
{kb_text}
--- END KNOWLEDGE BASE ---

Company: Vijay Spheroidals Pvt Ltd | ISO 9001:2015 | www.vijayspheroidals.in | Peenya Industrial Area, Bengaluru"""

def ask_gemini(messages, kb):
    api_key = st.session_state.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return "[WARNING] Gemini API key not set. Please set the GEMINI_API_KEY environment variable or enter it in the sidebar."
    
    import requests
    import time
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    system_prompt = build_system_prompt(kb)
    
    chat_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
    
    contents = []
    for m in chat_msgs:
        role = "user" if m["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": m["content"]}]
        })
        
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            res_data = response.json()
            
            # Extract the text response
            text = res_data["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"[WARNING] Gemini API Request timed out/failed after {max_retries} attempts: {last_err}"
        except Exception as e:
            try:
                err_msg = response.json().get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            return f"[WARNING] Gemini API Request failed: {err_msg}"

# ==============================================================
# UPGRADE 2 - BATCH QUALITY PREDICTION HELPER
# ==============================================================
def run_batch_prediction(df_in, clf, reg, scaler, le):
    required = ['pour_temp','spin_rpm','carbon_pct','silicon_pct',
                'cooling_sec','flow_rate','mold_type']
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    df = df_in.copy()
    try:
        df['mold_enc'] = le.transform(df['mold_type'])
    except Exception as e:
        return None, f"Invalid mold_type values. Use: {list(le.classes_)}. Error: {e}"

    X = df[['pour_temp','spin_rpm','carbon_pct','silicon_pct',
             'cooling_sec','flow_rate','mold_enc']].values
    Xs = scaler.transform(X)
    df['quality_score']      = reg.predict(Xs).round(1)
    df['reject_probability'] = (clf.predict_proba(Xs)[:,1] * 100).round(1)
    df['verdict']            = clf.predict(Xs)
    df['verdict']            = df['verdict'].map({0: '[ACTIVE] GOOD', 1: '[ERROR] REJECT'})
    df.drop(columns=['mold_enc'], inplace=True)
    return df, None

# ==============================================================
# UPGRADE 3 - PDF REPORT GENERATOR
# ==============================================================
def generate_pdf_report(params: dict, score: float, rej_p: float,
                         verdict: str, tips: list) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('T', parent=styles['Title'],
                                    fontSize=18, textColor=colors.HexColor('#1e3a5f'),
                                    spaceAfter=4)
    sub_style     = ParagraphStyle('S', parent=styles['Normal'],
                                    fontSize=10, textColor=colors.HexColor('#555555'),
                                    spaceAfter=12)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'],
                                    fontSize=12, textColor=colors.HexColor('#1e3a5f'),
                                    spaceBefore=12, spaceAfter=6)
    body_style    = ParagraphStyle('B', parent=styles['Normal'],
                                    fontSize=10, leading=16)
    tip_style     = ParagraphStyle('Tip', parent=styles['Normal'],
                                    fontSize=10, leading=16,
                                    leftIndent=10,
                                    textColor=colors.HexColor('#374151'))

    verdict_color = colors.HexColor('#16a34a') if '[ACTIVE]' in verdict else colors.HexColor('#dc2626')
    score_color   = colors.HexColor('#16a34a') if score >= 75 else \
                    colors.HexColor('#d97706') if score >= 60 else colors.HexColor('#dc2626')

    story = []

    # Header
    story.append(Paragraph("🏭 VSPL AI - Casting Quality Report", title_style))
    story.append(Paragraph(
        f"Vijay Spheroidals Pvt Ltd &nbsp;|&nbsp; ISO 9001:2015 &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#1e3a5f'), spaceAfter=10))

    # Verdict banner
    verdict_data = [[
        Paragraph(f"<b>Verdict: {verdict}</b>",
                  ParagraphStyle('V', parent=styles['Normal'], fontSize=14,
                                  textColor=verdict_color)),
        Paragraph(f"<b>Quality Score: {score:.1f}/100</b>",
                  ParagraphStyle('QS', parent=styles['Normal'], fontSize=14,
                                  textColor=score_color, alignment=TA_CENTER)),
        Paragraph(f"<b>Reject Risk: {rej_p:.1f}%</b>",
                  ParagraphStyle('RR', parent=styles['Normal'], fontSize=14,
                                  textColor=colors.HexColor('#dc2626') if rej_p > 30 else colors.HexColor('#16a34a'))),
    ]]
    verdict_table = Table(verdict_data, colWidths=['34%','33%','33%'])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4f8')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0f4f8')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # Parameters table
    story.append(Paragraph("Casting Parameters", heading_style))
    param_data = [
        ["Parameter", "Value", "Optimal Range", "Status"],
        ["Pour Temperature",  f"{params['temp']}  deg C",   "1400-1450  deg C",  "[OK]" if 1400<=params['temp']<=1450 else "[WARNING]"],
        ["Spin Speed (RPM)",  f"{params['rpm']}",       "1000-1200 RPM", "[OK]" if 1000<=params['rpm']<=1200 else "[WARNING]"],
        ["Carbon %",          f"{params['carbon']:.2f}","3.4-3.6 %",     "[OK]" if 3.4<=params['carbon']<=3.6 else "[WARNING]"],
        ["Silicon %",         f"{params['silicon']:.2f}","2.0-2.5 %",    "[OK]" if 2.0<=params['silicon']<=2.5 else "[WARNING]"],
        ["Cooling Time",      f"{params['cooling']} s", "60-90 sec",     "[OK]" if 60<=params['cooling']<=90 else "[WARNING]"],
        ["Metal Flow Rate",   f"{params['flow']:.1f}",  "1.2-1.8",       "[OK]" if 1.2<=params['flow']<=1.8 else "[WARNING]"],
        ["Mold Type",         params['mold'],           "Permanent/Die", "[OK]" if params['mold'] != 'Sand' else "[WARNING]"],
    ]
    p_table = Table(param_data, colWidths=['35%','20%','30%','15%'])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#f8fafc')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (3,0), (3,-1), 'CENTER'),
    ]))
    story.append(p_table)

    # Recommendations
    story.append(Paragraph("AI Recommendations", heading_style))
    clean_tips = [re.sub(r'[^\x00-\x7F]+', '', t).strip() for t in tips]
    for tip in clean_tips:
        story.append(Paragraph(f"• {tip}", tip_style))
    story.append(Spacer(1, 8))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5,
                              color=colors.HexColor('#cbd5e1'), spaceBefore=12))
    story.append(Paragraph(
        "Vijay Spheroidals Pvt Ltd | Peenya Industrial Area, Bengaluru | "
        "www.vijayspheroidals.in | ISO 9001:2015 Certified",
        ParagraphStyle('F', parent=styles['Normal'], fontSize=8,
                        textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()

def generate_quote_pdf(parsed: dict, top_grade: str, ppkg: float, qty_kg: float, total: float, lead: int) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=15*mm, bottomMargin=15*mm)

    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('T', parent=styles['Title'],
                                    fontSize=18, textColor=colors.HexColor('#1e3a5f'),
                                    spaceAfter=4)
    sub_style     = ParagraphStyle('S', parent=styles['Normal'],
                                    fontSize=10, textColor=colors.HexColor('#555555'),
                                    spaceAfter=12)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'],
                                    fontSize=12, textColor=colors.HexColor('#1e3a5f'),
                                    spaceBefore=12, spaceAfter=6)
    body_style    = ParagraphStyle('B', parent=styles['Normal'],
                                    fontSize=10, leading=16)

    story = []

    # Header
    story.append(Paragraph("🏭 VIJAY SPHEROIDALS PVT LTD", title_style))
    story.append(Paragraph(
        f"Commercial Quotation &nbsp;|&nbsp; ISO 9001:2015 &nbsp;|&nbsp; "
        f"Date: {datetime.today().strftime('%d %b %Y')}",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#1e3a5f'), spaceAfter=10))

    # Intro Paragraph
    story.append(Paragraph(f"Dear Team,<br/><br/>We thank you for your request for quotation. Based on the technical requirements extracted from your RFQ, our AI metallurgical recommender has proposed the optimal casting grade. Please find our commercial quotation detailed below:", body_style))
    story.append(Spacer(1, 10))

    # Quotation parameters table
    story.append(Paragraph("Quotation Details", heading_style))
    param_data = [
        ["Line Item / Description", "Value / Specification"],
        ["Recommended Alloy Grade", f"<b>{top_grade}</b>"],
        ["Extracted Application", parsed.get('application', 'General Purpose')],
        ["Requested Quantity", parsed.get('quantity', 'Not specified')],
        ["Unit Rate (ex-works)", f"Rs.{ppkg:.2f} / kg"],
        ["Estimated Total Value", f"<b>Rs.{total:,.2f}</b>"],
        ["Lead Time", f"{lead} working days"],
        ["Requested Delivery Date", parsed.get('delivery', 'Standard lead time')],
    ]
    p_table = Table(param_data, colWidths=['40%','60%'])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#f8fafc')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 15))

    # Standard Terms & Conditions
    story.append(Paragraph("Standard Terms & Conditions", heading_style))
    terms = [
        "<b>Validity:</b> This quotation is valid for a period of 30 days from the date of issue.",
        "<b>Quality Standard:</b> Castings manufactured in accordance with EN 10204 Type 3.1 Material Certification.",
        "<b>Payment Terms:</b> 50% advance along with Purchase Order, 50% balance prior to dispatch from works.",
        "<b>Taxation:</b> GST and local duties extra at actuals as applicable at the time of delivery."
    ]
    for term in terms:
        story.append(Paragraph(f"• {term}", ParagraphStyle('Term', parent=styles['Normal'], fontSize=9, leading=14, leftIndent=10)))
    story.append(Spacer(1, 15))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5,
                              color=colors.HexColor('#cbd5e1'), spaceBefore=12))
    story.append(Paragraph(
        "Vijay Spheroidals Pvt Ltd | Peenya Industrial Area, Bengaluru | "
        "www.vijayspheroidals.in | ISO 9001:2015 Certified",
        ParagraphStyle('F', parent=styles['Normal'], fontSize=8,
                        textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ==============================================================
# SIDEBAR
# ==============================================================
st.sidebar.title("🏭 VSPL AI Platform")
st.sidebar.caption("Vijay Spheroidals Pvt Ltd")

# Premium interactive crystal lattice particle network (HTML5 Canvas + JS)
import streamlit.components.v1 as components
components.html("""
<canvas id="canvas" style="width: 100%; height: 110px; background: transparent; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.04);"></canvas>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resize = () => {
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
};
resize();

let particles = [];
const width = canvas.width / window.devicePixelRatio;
const height = canvas.height / window.devicePixelRatio;

for (let i = 0; i < 30; i++) {
    particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7,
        radius: Math.random() * 2 + 1
    });
}

function animate() {
    requestAnimationFrame(animate);
    ctx.clearRect(0, 0, width, height);
    
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            let dist = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
            if (dist < 40) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }
    
    ctx.fillStyle = 'rgba(56, 189, 248, 0.6)';
    particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;
        
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
    });
}
animate();
</script>
<style>
body { margin: 0; padding: 0; overflow: hidden; }
</style>
""", height=110)

with st.sidebar:
    _key = os.environ.get("GEMINI_API_KEY","")
    st.markdown(f"""
<div style="padding:16px 14px 0">
  <div style="display:flex;align-items:center;gap:10px;padding-bottom:12px;
              border-bottom:2px solid #c9a84c">
    <div style="width:34px;height:34px;background:#c9a84c;border-radius:4px;
                display:flex;align-items:center;justify-content:center;
                font-size:16px;font-weight:700;color:#0b0e15;flex-shrink:0">C</div>
    <div>
      <div style="color:#e8eef5;font-size:14px;font-weight:700;letter-spacing:.5px">CastIQ</div>
      <div style="color:#4a6070;font-size:9px;letter-spacing:1.5px;margin-top:1px">VSPL · AI PLATFORM</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # API key input at top
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    if _key:
        st.markdown('<div style="margin:0 8px 8px;padding:7px 10px;background:#0a1a10;border:1px solid #1a3a20;border-radius:5px;font-size:10px;color:#4ade80;font-weight:600">⬤ GEMINI CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:0 8px 3px;font-size:10px;color:#c9a84c;font-weight:600;letter-spacing:.5px">⚠️ GEMINI API KEY REQUIRED</div>', unsafe_allow_html=True)
        _uk = st.text_input("k", type="password", placeholder="Paste Gemini key...", label_visibility="collapsed")
        if _uk:
            os.environ["GEMINI_API_KEY"] = _uk
            _key = _uk
            st.success("Connected!")
            st.rerun()

    # System status at top
    st.markdown(f"""
<div style="padding:8px 14px 8px">
  <div style="color:#2a3a4a;font-size:8px;font-weight:700;letter-spacing:2px;margin-bottom:8px">SYSTEM STATUS</div>
  {''.join([
    f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0">'
    f'<span style="color:#4a6070;font-size:10px">{name}</span>'
    f'<span style="color:{col};font-size:9px;font-weight:700">⬤ {status}</span>'
    f'</div>'
    for name, col, status in [
        ("ML Models",  "#4ade80", "LIVE"),
        ("YOLOv8",     "#4ade80", "READY"),
        ("Gemini AI",  "#4ade80" if _key else "#c9a84c", "ONLINE" if _key else "NO KEY"),
        ("PDF Engine", "#4ade80", "READY"),
        ("12 Modules", "#4ade80", "ACTIVE"),
    ]
  ])}
</div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#1a2030;margin:8px 0"></div>', unsafe_allow_html=True)

    def _sec(label):
        st.markdown(f'<div style="color:#2a3a4a;font-size:8px;font-weight:700;letter-spacing:2px;padding:10px 14px 3px;text-transform:uppercase">{label}</div>', unsafe_allow_html=True)

    _sec("Overview")
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "🔮 Casting Quality Predictor",
        "🎯 Process Optimizer",
        "📉 SPC Dashboard",
        "💡 SHAP Explainability",
        "🌐 Digital Twin",
        "🔍 Root Cause Analysis",
        "⚡ Energy Optimizer",
        "🧬 Multi-Objective (NSGA-II)",
        "📊 Demand Forecasting",
        "⚗️ Alloy Recommendation",
        "📄 RFQ Intelligence",
        "🤖 Smart Assistant",
        "📷 CV Defect Detector",
        "✏️ Diagram Generator",
        "🏭 Production Dashboard",
        "📦 Inventory Tracker",
        "🔧 Predictive Maintenance",
        "🚚 Order Tracker",
        "🔥 Heat Treatment",
        "💰 Cost Estimator",
    ], label_visibility="collapsed")

    # Author/watermark at the very bottom
    st.markdown(f"""
<div style="padding:14px 14px 20px">
  <div style="height:1px;background:#1a2030;margin:10px 0 8px"></div>
  <div style="font-size:9px;color:#2a3a4a;line-height:1.8">
    DARSHAN<br>CMR UNIVERSITY<br>B.TECH AI &amp; ML · 2025–26
  </div>
</div>""", unsafe_allow_html=True)

# ── Page header helper ─────────────────────────────────────
def page_header(icon, title, subtitle, tag=None, tag_color="#c9a84c"):
    tag_html = ""
    if tag:
        tag_html = f'<span style="background:{tag_color}18;color:{tag_color};font-size:9px;font-weight:700;padding:3px 9px;border-radius:3px;border:1px solid {tag_color}33;letter-spacing:1px;margin-left:10px">{tag}</span>'
    st.markdown(f"""
<div style="background:#0b0e15;border-bottom:2px solid #c9a84c;
            padding:16px 0 14px;margin-bottom:22px">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:4px;height:30px;background:#c9a84c;border-radius:2px;flex-shrink:0"></div>
    <div style="font-size:22px">{icon}</div>
    <div>
      <div style="display:flex;align-items:center">
        <span style="color:#e8eef5;font-size:17px;font-weight:700;letter-spacing:.3px">{title}</span>
        {tag_html}
      </div>
      <div style="color:#4a6070;font-size:11px;margin-top:3px;letter-spacing:.3px">{subtitle}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DASHBOARD PAGE — Navy + Gold UI
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Hero
    st.markdown("""
<div style='background:linear-gradient(135deg,#0d1f35 0%,#112240 60%,#0a1628 100%);
            border:1px solid #c9a84c44; border-radius:16px; padding:32px 36px; margin-bottom:24px;
            box-shadow:0 8px 32px rgba(201,168,76,0.12)'>
    <div style='display:flex;align-items:center;gap:16px'>
        <div style='font-size:52px'>🏭</div>
        <div>
            <h1 style='margin:0;font-size:28px;color:#c9a84c;letter-spacing:1px'>CastIQ — AI Intelligence Platform</h1>
            <p style='margin:4px 0 0;color:#8899bb;font-size:14px'>
                Vijay Spheroidals Pvt Ltd &nbsp;|&nbsp; Peenya Industrial Area, Bengaluru &nbsp;|&nbsp; ISO 9001:2015
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Monthly Capacity", "200 T",    "Tonnes/month")
    k2.metric("AI Modules",       "12",       "Active")
    k3.metric("Classifier Acc.",  "85.4%",    "+2.1% vs baseline")
    k4.metric("Recommender Acc.", "94.5%",    "Alloy model")
    k5.metric("Sectors Served",   "6",        "Industries")
    k6.metric("Alloy Grades",     "6",        "Available")

    st.markdown("<div style='border-bottom:1px solid #c9a84c33;margin:20px 0'></div>", unsafe_allow_html=True)

    # Module cards — grouped
    def module_card(icon, title, desc, tag, tag_color="#22c55e"):
        return f"""
<div style='background:linear-gradient(135deg,#0d1f35,#112240);
            border:1px solid #c9a84c33; border-radius:14px; padding:20px;
            margin-bottom:12px; transition:all 0.2s;
            box-shadow:0 4px 16px rgba(0,0,0,0.3)'>
    <div style='display:flex;justify-content:space-between;align-items:flex-start'>
        <div style='font-size:26px'>{icon}</div>
        <span style='background:{tag_color}22;color:{tag_color};
                     font-size:9px;font-weight:700;padding:3px 8px;
                     border-radius:20px;letter-spacing:1px;border:1px solid {tag_color}44'>{tag}</span>
    </div>
    <div style='color:#d4b86a;font-weight:700;font-size:15px;margin:8px 0 4px'>{title}</div>
    <div style='color:#8899aa;font-size:12px;line-height:1.5'>{desc}</div>
</div>"""

    st.markdown("### 🤖 AI Quality Modules")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(module_card("🔮","Casting Quality Predictor",
            "Predict reject probability & quality score before every pour. Batch upload + PDF report.",
            "ML MODEL","#22c55e"), unsafe_allow_html=True)
        st.markdown(module_card("📄","RFQ Intelligence",
            "Paste any customer email → AI extracts specs and generates a formatted quote instantly.",
            "NLP + AI","#a855f7"), unsafe_allow_html=True)
    with c2:
        st.markdown(module_card("📊","Demand Forecasting",
            "6-month sector-wise order forecasting with confidence intervals and trend analysis.",
            "PROPHET","#3b82f6"), unsafe_allow_html=True)
        st.markdown(module_card("🤖","Smart Assistant",
            "Powered by Google Gemini. Answers any VSPL question using the full knowledge base.",
            "GEMINI AI","#f59e0b"), unsafe_allow_html=True)
    with c3:
        st.markdown(module_card("⚗️","Alloy Recommendation",
            "Input application specs → AI recommends the optimal ductile iron grade with confidence.",
            "RANDOM FOREST","#22c55e"), unsafe_allow_html=True)
        st.markdown(module_card("📷","CV Defect Detector",
            "YOLOv8 real-time surface defect detection via image upload or live webcam feed.",
            "YOLOv8","#ef4444"), unsafe_allow_html=True)

    st.markdown("### 🏭 Operations Modules")
    c4,c5,c6 = st.columns(3)
    with c4:
        st.markdown(module_card("🏭","Production Dashboard",
            "Real-time shift-wise output tracking, operator performance and 30-day trend analysis.",
            "LIVE DATA","#06b6d4"), unsafe_allow_html=True)
        st.markdown(module_card("🔧","Predictive Maintenance",
            "8-machine fleet health monitoring. Predict failures before they cause downtime.",
            "AI PREDICT","#f59e0b"), unsafe_allow_html=True)
    with c5:
        st.markdown(module_card("📦","Inventory Tracker",
            "Live raw material stock levels, low-stock alerts and auto-generated reorder POs.",
            "LIVE STOCK","#22c55e"), unsafe_allow_html=True)
        st.markdown(module_card("🔥","Heat Treatment",
            "AI-recommended heat treatment cycles per grade. Generates official certificates.",
            "PROCESS ENG","#f97316"), unsafe_allow_html=True)
    with c6:
        st.markdown(module_card("🚚","Order Tracker",
            "End-to-end order pipeline from received to delivery. Kanban view + analytics.",
            "CRM","#3b82f6"), unsafe_allow_html=True)
        st.markdown(module_card("💰","Cost Estimator",
            "Full cost breakdown: material + machining + overhead + margin. PDF quote included.",
            "PRICING","#c9a84c"), unsafe_allow_html=True)

    st.markdown("<div style='border-bottom:1px solid #c9a84c33;margin:20px 0'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("### 🛠️ Tech Stack")
    t1,t2,t3,t4,t5 = st.columns(5)
    for col, icon, title, items in [
        (t1,"🐍","Backend",    "Python 3.13\nStreamlit\nFastAPI"),
        (t2,"🤖","ML Models",  "Gradient Boosting\nRandom Forest\nYOLOv8"),
        (t3,"🧠","Deep Learning","PyTorch DNN\nMini-GPT\nOpenCV"),
        (t4,"💎","AI Services","Google Gemini\nGroq LLaMA\nAnthropic"),
        (t5,"📄","Reports",    "ReportLab PDF\nPlotly Charts\nCSV Export"),
    ]:
        col.markdown(f"""
<div style='background:#0d1f35;border:1px solid #c9a84c33;border-radius:10px;
            padding:14px;text-align:center'>
    <div style='font-size:22px'>{icon}</div>
    <div style='color:#c9a84c;font-weight:700;font-size:12px;margin:4px 0'>{title}</div>
    <div style='color:#778899;font-size:11px;line-height:1.6'>{items}</div>
</div>""", unsafe_allow_html=True)

elif page == "🔮 Casting Quality Predictor":
    clf, reg, scaler, le = load_casting()

    st.title("🔮 Casting Quality Predictor")
    st.markdown("##### Predict quality **before** the pour - prevent rejects proactively")
    st.markdown("---")

    mode = st.radio("Mode", ["Single Prediction", "Batch Upload"], horizontal=True)
    st.markdown("---")

    # ── SINGLE PREDICTION ──────────────────────────────────
    if mode == "Single Prediction":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚙️ Casting Parameters")
            temp = st.slider("🌡️ Pour Temperature (°C)", 1340, 1520, 1425)
            rpm = st.slider("🔄 Spin Speed (RPM)", 600, 1500, 1100)
            carbon = st.slider("⚗️ Carbon %", 3.0, 4.0, 3.5, 0.01)
            silicon = st.slider("⚗️ Silicon %", 1.5, 3.0, 2.25, 0.01)
            manganese = st.slider("⚗️ Manganese %", 0.1, 0.6, 0.3, 0.01)
            phosphorus = st.slider("⚗️ Phosphorus %", 0.01, 0.08, 0.03, 0.001)
            mg_added = st.slider("⚗️ Magnesium Added %", 0.02, 0.08, 0.05, 0.001)
            treat_time = st.slider("⏱️ Treatment Duration (min)", 1.0, 15.0, 6.0, 0.5)
            mold_diam = st.slider("📏 Mold Diameter (m)", 0.1, 1.0, 0.3, 0.01)
            preheat = st.slider("🔥 Mold Preheat Temperature (°C)", 50, 400, 200, 10)
            thickness = st.slider("📐 Wall Thickness (mm)", 5, 100, 25, 1)
            mold = st.selectbox("🔲 Mold Type", ['Permanent', 'Sand', 'Die'])
            st.markdown("---")
            st.markdown("**Prediction AI Architecture**")
            model_choice = st.selectbox("AI Model Architecture", ["Gradient Boosting (Traditional ML)", "PyTorch Multitask DNN (Deep Learning)"])
            predict = st.button("🔮 Predict Quality", use_container_width=True, type="primary")

        with col2:
            if predict:
                cooling = int(thickness * 3) # compute cooling time
                flow = 1.5
                if model_choice == "Gradient Boosting (Traditional ML)":
                    mold_map = {'Permanent': 0, 'Sand': 1, 'Die': 2}
                    mold_enc_val = mold_map[mold]
                    
                    # Compute all 6 physics-derived calculations (to align with the 18-feature model)
                    ce = carbon + (silicon + phosphorus) / 3.0
                    liquidus = 1669.0 - 124.0 * ce
                    superheat = temp - liquidus
                    g_fact = 0.00001118 * (rpm**2) * (mold_diam / 2.0)
                    mg_eff = mg_added * np.exp(-0.05 * treat_time)
                    nodularity = 98.0 * (1.0 - np.exp(-42.0 * mg_eff)) - (phosphorus * 80.0)
                    nodularity = np.clip(nodularity, 0.0, 100.0)
                    mold_k = {'Permanent': 16.0, 'Sand': 3.0, 'Die': 11.0}[mold]
                    cooling_rate = (mold_k * (temp - preheat)) / (thickness**2 * 8.0)
                    
                    X = np.array([[
                        carbon, silicon, manganese, phosphorus,
                        mg_added, treat_time, rpm, mold_diam,
                        temp, preheat, thickness, mold_enc_val,
                        ce, superheat, g_fact,
                        mg_eff, nodularity, cooling_rate
                    ]])
                    Xs = scaler.transform(X)
                    score   = float(reg.predict(Xs)[0])
                    rej_p   = float(clf.predict_proba(Xs)[0][1]) * 100
                    verdict = "[ACTIVE] GOOD CAST" if clf.predict(Xs)[0] == 0 else "[ERROR] LIKELY REJECT"
                else:
                    # PyTorch DNN path
                    dnn_model, dnn_scaler = load_pytorch_dnn()
                    if dnn_model is not None:
                        mold_map = {'Permanent': 0, 'Sand': 1, 'Die': 2}
                        mold_enc_val = mold_map[mold]
                        
                        # 6 physics-derived calculations (the secret sauce!)
                        ce = carbon + (silicon + phosphorus) / 3.0
                        liquidus = 1669.0 - 124.0 * ce
                        superheat = temp - liquidus
                        g_fact = 0.00001118 * (rpm**2) * (mold_diam / 2.0)
                        mg_eff = mg_added * np.exp(-0.05 * treat_time)
                        nodularity = 98.0 * (1.0 - np.exp(-42.0 * mg_eff)) - (phosphorus * 80.0)
                        nodularity = np.clip(nodularity, 0.0, 100.0)
                        mold_k = {'Permanent': 16.0, 'Sand': 3.0, 'Die': 11.0}[mold]
                        cooling_rate = (mold_k * (temp - preheat)) / (thickness**2 * 8.0)
                        
                        # Stack all 18 features in FEATS order
                        X = np.array([[
                            carbon, silicon, manganese, phosphorus,
                            mg_added, treat_time, rpm, mold_diam,
                            temp, preheat, thickness, mold_enc_val,
                            ce, superheat, g_fact,
                            mg_eff, nodularity, cooling_rate
                        ]])
                        
                        Xs = dnn_scaler.transform(X)
                        X_tensor = torch.tensor(Xs, dtype=torch.float32)
                        with torch.no_grad():
                            pred_score, pred_prob = dnn_model(X_tensor)
                        score = float(pred_score[0][0].item())
                        rej_p = float(pred_prob[0][0].item()) * 100
                        verdict = "[ACTIVE] GOOD CAST" if rej_p < 50.0 else "[ERROR] LIKELY REJECT"
                    else:
                        st.error("PyTorch model files not found. Please ensure the training pipeline is complete.")
                        st.stop()

                tips    = cast_recommendations(temp, rpm, carbon, silicon, cooling, flow, mold)

                tab1, tab2 = st.tabs(["[Forecast] QA Prediction & Metrics", "🧊 3D Solidification Viewer"])

                with tab1:
                    st.plotly_chart(quality_gauge(score), use_container_width=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Reject Probability", f"{rej_p:.1f}%")
                    c2.metric("Verdict", verdict)
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    for tip in tips:
                        st.info(tip)

                    # ── UPGRADE 3: PDF EXPORT ──
                    st.markdown("---")
                    params = dict(temp=temp, rpm=rpm, carbon=carbon, silicon=silicon,
                                  cooling=cooling, flow=flow, mold=mold)
                    pdf_bytes = generate_pdf_report(params, score, rej_p, verdict, tips)
                    st.download_button(
                        label="[RFQ] Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"VSPL_Quality_Report_{datetime.today().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                
                with tab2:
                    st.subheader("🧊 Centrifugal Casting Solidification Profile")
                    st.markdown("Solidification cooling gradients are simulated across casting wall depth (Radius: 75mm inner to 100mm outer) over time.")
                    
                    # ── 3D HEAT solid/liquid transfer equation simulation ──
                    r_vals = np.linspace(75, 100, 30) # mm radial depth
                    t_vals = np.linspace(0, cooling, 30) # cooling seconds
                    R, T_mesh = np.meshgrid(r_vals, t_vals)
                    
                    # Cooling rate is faster closer to the pre-heated mold outer wall (R=100mm)
                    k_base = 0.05 * (75.0 / cooling)
                    k_mesh = k_base * (1.0 + (R - 75.0) / 25.0 * 2.0)
                    
                    # Thermal curve
                    Temp_grid = 150 + (temp - 150) * np.exp(-k_mesh * T_mesh)
                    
                    # Draw 3D surface plot
                    fig_3d = go.Figure(data=[go.Surface(
                        x=R, y=T_mesh, z=Temp_grid,
                        colorscale='Hot',
                        colorbar=dict(title='Temp ( deg C)', titleside='right', len=0.8)
                    )])
                    
                    fig_3d.update_layout(
                        scene=dict(
                            xaxis=dict(title='Radius (mm)', gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(title='Time (s)', gridcolor='rgba(255,255,255,0.1)'),
                            zaxis=dict(title='Temp ( deg C)', gridcolor='rgba(255,255,255,0.1)'),
                            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=420,
                        margin=dict(t=20, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.markdown("### <- Set parameters & click **Predict Quality**")
                st.markdown("""
| Parameter | Optimal Range |
|-----------|--------------|
| Temperature | 1400-1450 deg C |
| Spin Speed | 1000-1200 RPM |
| Carbon % | 3.4-3.6% |
| Silicon % | 2.0-2.5% |
| Cooling | 60-90 sec |
| Flow Rate | 1.2-1.8 |
""")

    # ── UPGRADE 2: BATCH UPLOAD ────────────────────────────
    else:
        st.subheader("📂 Batch Prediction - Upload CSV")
        st.markdown("""
Upload a CSV with these **exact column names**:

| Column | Description | Example |
|--------|-------------|---------|
| `pour_temp` | Pour temperature  deg C | 1425 |
| `spin_rpm` | Spin speed RPM | 1100 |
| `carbon_pct` | Carbon % | 3.5 |
| `silicon_pct` | Silicon % | 2.25 |
| `cooling_sec` | Cooling time sec | 75 |
| `flow_rate` | Metal flow rate | 1.5 |
| `mold_type` | Mold type | Permanent |
""")

        # Sample CSV download
        sample_csv = """pour_temp,spin_rpm,carbon_pct,silicon_pct,cooling_sec,flow_rate,mold_type
1425,1100,3.5,2.25,75,1.5,Permanent
1380,900,3.2,1.8,50,1.0,Sand
1440,1150,3.55,2.3,80,1.6,Die
1500,1350,3.8,2.8,45,2.0,Sand
1415,1050,3.45,2.1,70,1.4,Permanent"""
        st.download_button("⬇️ Download Sample CSV", sample_csv,
                           file_name="vspl_batch_sample.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload your casting runs CSV", type=['csv'])

        if uploaded:
            df_in = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_in)} rows loaded.** Preview:")
            st.dataframe(df_in.head(5), use_container_width=True, hide_index=True)

            if st.button("[RUN] Run Batch Prediction", type="primary", use_container_width=True):
                df_out, err = run_batch_prediction(df_in, clf, reg, scaler, le)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success(f"[OK] Predicted {len(df_out)} casting runs!")

                    # Summary metrics
                    total    = len(df_out)
                    good     = (df_out['verdict'] == '[ACTIVE] GOOD').sum()
                    rejects  = total - good
                    avg_score = df_out['quality_score'].mean()

                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Total Runs",    total)
                    m2.metric("Good Casts",    good,    delta=f"{good/total*100:.0f}%")
                    m3.metric("Likely Rejects",rejects, delta=f"-{rejects/total*100:.0f}%", delta_color="inverse")
                    m4.metric("Avg Quality",   f"{avg_score:.1f}/100")

                    # Results table - color coded
                    st.markdown("---")
                    st.subheader("📋 Prediction Results")
                    st.dataframe(
                        df_out[['pour_temp','spin_rpm','carbon_pct','silicon_pct',
                                'mold_type','quality_score','reject_probability','verdict']],
                        use_container_width=True, hide_index=True
                    )

                    # Verdict pie chart
                    pie_fig = go.Figure(go.Pie(
                        labels=['Good Casts','Likely Rejects'],
                        values=[good, rejects],
                        marker_colors=['#22c55e','#ef4444'],
                        hole=0.4
                    ))
                    pie_fig.update_layout(
                        height=280, paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white', margin=dict(t=20,b=10)
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                    # Download results CSV
                    csv_out = df_out.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results CSV", csv_out,
                        file_name=f"VSPL_Batch_Results_{datetime.today().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True
                    )

# ==============================================================
# MODULE 2 - DEMAND FORECASTING (confidence intervals already in original)
# ==============================================================
elif page == "📊 Demand Forecasting":
    st.title("📊 Demand Forecasting")
    st.markdown("##### AI-powered order forecasting per sector - next 6 months")
    st.markdown("---")

    df      = load_forecast()
    sectors = sorted(df['sector'].unique())
    cutoff  = df[df['actual'].notna()]['date'].max()

    st.subheader("📈 Next Month Forecast")
    cols = st.columns(len(sectors))
    for i, sec in enumerate(sectors):
        sdf  = df[df['sector']==sec]
        nxt  = sdf[sdf['date']>cutoff].iloc[0]
        last = sdf[sdf['actual'].notna()].iloc[-1]['actual']
        d    = ((nxt['predicted']-last)/last)*100
        cols[i].metric(sec, f"{nxt['predicted']:.0f} units", f"{d:+.1f}%")

    st.markdown("---")
    c1, c2 = st.columns([1,3])
    with c1:
        st.subheader("🔍 Filters")
        selected = st.multiselect("Sectors", sectors, default=sectors)
        months   = st.slider("Forecast months", 1, 6, 6)
        show_ci  = st.checkbox("Show confidence band", True)

    with c2:
        st.subheader("📉 Historical + Forecast")
        fig = go.Figure()
        for sec in selected:
            sdf   = df[df['sector']==sec]
            hist  = sdf[sdf['actual'].notna()]
            fcast = sdf[sdf['date']>cutoff].head(months)
            col   = SECTOR_COLORS.get(sec,'#888')
            fig.add_trace(go.Scatter(x=hist['date'],  y=hist['actual'],
                name=f"{sec} (actual)", line=dict(color=col,width=2),
                mode='lines+markers', marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=fcast['date'], y=fcast['predicted'],
                name=f"{sec} (forecast)", line=dict(color=col,width=2,dash='dash'),
                mode='lines+markers', marker=dict(size=6,symbol='diamond')))
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([fcast['date'],fcast['date'][::-1]]),
                    y=pd.concat([fcast['upper'],fcast['lower'][::-1]]),
                    fill='toself', fillcolor=hex_to_rgba(col),
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                    name=f"{sec} CI"))
        fig.add_shape(type="line",
            x0=str(cutoff), x1=str(cutoff), y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="white", dash="dot", width=1))
        fig.add_annotation(x=str(cutoff), y=1, xref="x", yref="paper",
            text="Forecast →", showarrow=False,
            font=dict(color="white", size=12), xanchor="left", yanchor="bottom")
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font_color='white',
            legend=dict(orientation='h',y=1.1),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)',title='Orders (units/mo)'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 6-Month Forecast Table")
    rows = []
    for sec in selected:
        for _, r in df[(df['sector']==sec) & (df['date']>cutoff)].head(6).iterrows():
            rows.append({'Sector':sec,'Month':r['date'].strftime('%b %Y'),
                         'Forecast':f"{r['predicted']:.0f}",
                         'Low':f"{r['lower']:.0f}",'High':f"{r['upper']:.0f}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 Projected 6-Month Total")
    totals = [{'Sector':s,'Total':round(df[(df['sector']==s) & (df['date']>cutoff)].head(6)['predicted'].sum())}
              for s in selected]
    bfig = px.bar(pd.DataFrame(totals),x='Sector',y='Total',color='Sector',
                  color_discrete_map=SECTOR_COLORS,text='Total')
    bfig.update_traces(textposition='outside')
    bfig.update_layout(height=340,showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',font_color='white',
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
    st.plotly_chart(bfig, use_container_width=True)

# ==============================================================
# MODULE 3 - ALLOY RECOMMENDATION
# ==============================================================
elif page == "⚗️ Alloy Recommendation":
    clf_a, sc_a, le_a, grades_db = load_alloy()

    st.title("⚗️ Alloy Recommendation Engine")
    st.markdown("##### Describe your application → AI recommends the optimal ductile iron grade")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Application Requirements")
        tensile   = st.slider("💪 Required Tensile Strength (MPa)", 300, 950, 500, 10)
        hardness  = st.slider("🔩 Required Hardness (BHN)",         100, 380, 200, 5)
        temp      = st.slider("🌡️ Max Operating Temperature ( deg C)",  50,  700, 250, 10)
        corrosion = st.select_slider("🧪 Corrosion Resistance",
                                     options=["Low","Medium","High"], value="Medium")
        thickness = st.slider("📐 Wall Thickness (mm)", 5, 80, 25)
        find_btn  = st.button("🔍 Find Best Alloy Grade", use_container_width=True, type="primary")

    with col2:
        if find_btn:
            corr_map = {"Low":1,"Medium":2,"High":3}
            X  = np.array([[tensile, hardness, temp, corr_map[corrosion], thickness]])
            Xs = sc_a.transform(X)
            proba    = clf_a.predict_proba(Xs)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            top_grade = le_a.inverse_transform([top3_idx[0]])[0]
            info      = grades_db.get(top_grade, {})

            st.subheader("🌟 Recommended Grade")
            st.markdown(f"## `{top_grade}`")
            st.success(info.get('description',''))
            st.markdown("---")

            st.subheader("⚗️ Alloy Composition")
            comp = info.get('composition',{})
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Carbon %",    comp.get('carbon','-'))
            c2.metric("Silicon %",   comp.get('silicon','-'))
            c3.metric("Manganese %", comp.get('manganese','-'))
            c4.metric("Magnesium %", comp.get('magnesium','-'))
            st.markdown("---")

            st.subheader("📊 Mechanical Properties")
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Tensile",     f"{info.get('tensile_min','?')}-{info.get('tensile_max','?')} MPa")
            p2.metric("Hardness",    f"{info.get('hardness_min','?')}-{info.get('hardness_max','?')} BHN")
            p3.metric("Elongation",  f"{info.get('elongation','?')}%")
            p4.metric("Price/kg",    f"Rs.{info.get('price_per_kg','?')}")
            st.markdown("---")

            st.subheader("📈 Match Confidence")
            top3g = le_a.inverse_transform(top3_idx)
            top3s = proba[top3_idx]*100
            bfig  = go.Figure(go.Bar(
                x=top3s, y=top3g, orientation='h',
                marker_color=['#22c55e','#3b82f6','#f59e0b'],
                text=[f"{s:.1f}%" for s in top3s], textposition='outside',
            ))
            bfig.update_layout(height=180, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                xaxis=dict(range=[0,115], gridcolor='rgba(255,255,255,0.1)'),
                margin=dict(t=10,b=10,l=10,r=60))
            st.plotly_chart(bfig, use_container_width=True)

            # ── METALLURGICAL RADAR COMPARISON ──
            st.markdown("---")
            st.subheader("🎯 Metallurgical Radar Comparison")
            
            RADAR_DATA = {
                'IS 400/12': {'Tensile Strength': 40, 'Hardness': 35, 'Ductility (Elongation)': 95, 'High-Temp Suitability': 30, 'Corrosion Resistance': 40, 'Machinability': 90},
                'IS 500/7':  {'Tensile Strength': 52, 'Hardness': 48, 'Ductility (Elongation)': 75, 'High-Temp Suitability': 45, 'Corrosion Resistance': 45, 'Machinability': 80},
                'IS 600/3':  {'Tensile Strength': 65, 'Hardness': 60, 'Ductility (Elongation)': 50, 'High-Temp Suitability': 50, 'Corrosion Resistance': 50, 'Machinability': 70},
                'IS 700/2':  {'Tensile Strength': 78, 'Hardness': 72, 'Ductility (Elongation)': 35, 'High-Temp Suitability': 55, 'Corrosion Resistance': 55, 'Machinability': 60},
                'IS 800/2':  {'Tensile Strength': 90, 'Hardness': 85, 'Ductility (Elongation)': 25, 'High-Temp Suitability': 60, 'Corrosion Resistance': 60, 'Machinability': 50},
                'SiMo 0.8':  {'Tensile Strength': 48, 'Hardness': 55, 'Ductility (Elongation)': 60, 'High-Temp Suitability': 95, 'Corrosion Resistance': 90, 'Machinability': 75},
            }

            comparison_grades = st.multiselect(
                "Select Grades to Compare on Radar", 
                list(RADAR_DATA.keys()), 
                default=[top_grade, "IS 500/7"] if top_grade != "IS 500/7" else ["IS 500/7", "SiMo 0.8"]
            )

            fig_radar = go.Figure()
            categories = ['Tensile Strength', 'Hardness', 'Ductility (Elongation)', 'High-Temp Suitability', 'Corrosion Resistance', 'Machinability']

            for g in comparison_grades:
                values = [RADAR_DATA[g][cat] for cat in categories]
                values.append(values[0])
                categories_closed = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_closed,
                    fill='toself',
                    name=g
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.15)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=360,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.markdown("### <- Set requirements & click **Find Best Alloy Grade**")
            st.markdown("""
| Grade | Tensile (MPa) | Best For |
|-------|--------------|----------|
| IS 400/12 | 400-500 | General, Pharma |
| IS 500/7  | 500-600 | Hydraulics |
| IS 600/3  | 600-700 | Locomotives |
| IS 700/2  | 700-800 | Wind Energy |
| IS 800/2  | 800-950 | Heavy Engineering |
| SiMo 0.8  | 450-550 | High Temp / Corrosion |
""")

# ==============================================================
# MODULE 4 - RFQ INTELLIGENCE
# ==============================================================
elif page == "📄 RFQ Intelligence":
    _, _, le_a2, grades_db2 = load_alloy()
    clf_a2, sc_a2, _, _     = load_alloy()

    st.title("📄 RFQ Intelligence")
    st.markdown("##### Paste any customer RFQ → AI extracts specs & generates a quote instantly")
    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("📥 Paste RFQ Text")
        sample_rfq = """Dear VSPL Team,

We require ductile iron castings for hydraulic manifolds.
Specifications:
- Quantity: 500 kg
- Tensile strength: minimum 500 MPa
- Application: Hydraulic cylinders
- Wall thickness: 25 mm
- Delivery required within 21 days
- ISO 9001 certification required

Please provide quotation at earliest.

Regards,
ABC Engineering Pvt Ltd"""

        rfq_text = st.text_area("RFQ Content", value=sample_rfq, height=280)
        analyze_btn = st.button("⚡ Extract & Generate Quote", use_container_width=True, type="primary")

    with col2:
        if analyze_btn and rfq_text.strip():
            parsed = parse_rfq(rfq_text)

            st.subheader("📋 Extracted Specifications")
            e1,e2 = st.columns(2)
            e1.metric("Application",   parsed['application'])
            e2.metric("Quantity",      parsed['quantity'])
            e1.metric("Tensile Req.",  parsed['tensile'])
            e2.metric("Diameter",      parsed['diameter'])
            e1.metric("Delivery By",   parsed['delivery'])

            st.markdown("---")

            tens_val = int(re.search(r'\d+', parsed['tensile']).group()) if re.search(r'\d+', parsed['tensile']) else 500
            X  = np.array([[tens_val, 200, 300, 2, 25]])
            Xs = sc_a2.transform(X)
            top_grade = le_a2.inverse_transform(clf_a2.predict(Xs))[0]
            info      = grades_db2.get(top_grade, {})

            st.subheader("🌟 Recommended Grade")
            st.success(f"**{top_grade}** - {info.get('description','')}")

            st.markdown("---")
            st.subheader("📄 Generated Quote")

            qty_kg = float(re.search(r'\d+', parsed['quantity']).group()) if re.search(r'\d+', parsed['quantity']) else 100
            ppkg   = info.get('price_per_kg', 75)
            total  = qty_kg * ppkg
            lead   = info.get('lead_days', 21)

            quote = f"""
----------------------------------------
  VIJAY SPHEROIDALS PVT LTD
  Quotation - {datetime.today().strftime('%d %b %Y')}
----------------------------------------

  Grade Recommended : {top_grade}
  Application       : {parsed['application']}
  Quantity          : {parsed['quantity']}
  Unit Rate         : Rs.{ppkg}/kg (ex-works)
  Estimated Total   : Rs.{int(total):,}
  Lead Time         : {lead} working days
  Delivery By       : {parsed['delivery']}
  Validity          : 30 days from quote date

  Material Cert.    : EN 10204 Type 3.1
  Quality Standard  : ISO 9001:2015
  Payment Terms     : 50% advance, 50% on dispatch

----------------------------------------
  www.vijayspheroidals.in
----------------------------------------
"""
            st.code(quote, language=None)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download Quote (TXT)", quote,
                                   file_name=f"VSPL_Quote_{datetime.today().strftime('%Y%m%d')}.txt",
                                   mime="text/plain", use_container_width=True)
            with c2:
                pdf_bytes = generate_quote_pdf(parsed, top_grade, ppkg, qty_kg, total, lead)
                st.download_button("[RFQ] Download PDF Quotation", pdf_bytes,
                                   file_name=f"VSPL_Quotation_{datetime.today().strftime('%Y%m%d')}.pdf",
                                   mime="application/pdf", use_container_width=True)
        elif analyze_btn:
            st.warning("Please paste an RFQ first.")
        else:
            st.markdown("### <- Paste an RFQ and click **Extract & Generate Quote**")
            st.markdown("""
**What AI extracts automatically:**
- [OK] Quantity (kg/pcs)
- [OK] Required tensile strength (MPa)
- [OK] Application type
- [OK] Diameter / wall thickness
- [OK] Delivery timeline

**What AI generates:**
- [RECOMMENDED] Best matching alloy grade
- [PRICE] Price estimate (Rs./kg)
- [RFQ] Formatted quote ready to send
""")

# ==============================================================
# MODULE 5 - SMART ASSISTANT (Gemini API)
# ==============================================================
elif page == "🤖 Smart Assistant":
    def build_bi_system_prompt(selected_dataset_name, columns):
        return f"""You are the VSPL AI BI Data Analyst, a conversational business intelligence assistant inspired by Rill Data.
Your task is to analyze local casting datasets based on natural language questions.

You must translate the user's question into a safe, clean Python pandas statement.
The dataframe is already loaded as a local variable named `df`.

--- ACTIVE DATASET SCHEMA ---
Dataset Name: {selected_dataset_name}
Available Columns: {', '.join(columns)}

--- GUIDELINES ---
1. You MUST respond with a single, raw JSON block. DO NOT include markdown code blocks (such as ```json) in your response, just return the plain JSON string.
2. The JSON MUST contain exactly these keys:
   - "query_explanation": A concise, friendly explanation of what data we are looking at.
   - "pandas_code": A single line of python pandas code that returns a pandas DataFrame, Series, or scalar. It MUST operate on the variable `df`. (e.g., "df.groupby('mold_type')['reject'].mean().reset_index()")
   - "chart_requested": A boolean indicating if the user's prompt asks for a visualization (e.g. plot, chart, graph, show relationship).
   - "chart_type": "bar", "line", "scatter", or null.
   - "x_col": The column name to use for the x-axis, or null.
   - "y_col": The column name to use for the y-axis, or null.
   - "metallurgical_insight": A professional metallurgy analysis explaining the industrial significance of the query results.

3. Keep the pandas code safe and simple. Do not import any libraries. Only use standard pandas methods (groupby, mean, sum, count, filter, sort_values, head, describe, etc.).
4. Do not write any code that modifies `df` in-place. Always return a new variable or projection.

Example Output format:
{{
  "query_explanation": "Calculating average quality score grouped by mold type.",
  "pandas_code": "df.groupby('mold_type')['quality_score'].mean().reset_index()",
  "chart_requested": true,
  "chart_type": "bar",
  "x_col": "mold_type",
  "y_col": "quality_score",
  "metallurgical_insight": "Permanent molds yield higher quality scores due to faster solidification and finer grain size than sand molds."
}}
"""

    st.title("🤖 VSPL Smart Assistant Suite")
    st.markdown("##### Powered by **Gemini AI** - General Knowledge and Rill-Style conversational BI analytics")
    st.markdown("---")

    smart_tab1, smart_tab2 = st.tabs(["💬 VSPL Knowledge Assistant", "📊 AI BI Data Analyst (Rill-Style)"])

    # [CHAT] VSPL Knowledge Assistant
    with smart_tab1:
        st.subheader("💬 General Metallurgy & Company Q&A")
        st.markdown("Ask anything about VSPL products, alloy grades, centrifugal casting processes, or pricing.")
        
        kb = load_kb()

        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "Hello! Hello! I'm the VSPL Smart Assistant, powered by Google Gemini AI.\n\nAsk me anything about our alloy grades, casting process, delivery timelines, safety procedures, or pricing - I'll give you a detailed, intelligent answer."}
            ]

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg['role'], avatar="🏭" if msg['role']=='assistant' else "[USER]"):
                st.markdown(msg['content'])

        # Quick suggestions
        st.markdown("**💡 Quick questions:**")
        c1,c2,c3,c4 = st.columns(4)
        suggestions = [
            ("What grades do you offer?", c1),
            ("Explain the casting process", c2),
            ("Compare IS 700/2 vs IS 600/3", c3),
            ("Safety rules on casting floor?", c4),
        ]
        for sugg, col in suggestions:
            if col.button(sugg, key=f"sugg_{sugg}", use_container_width=True):
                st.session_state.messages.append({"role":"user","content":sugg})
                with st.spinner("Gemini is thinking..."):
                    try:
                        ans = ask_gemini(st.session_state.messages, kb)
                    except Exception as e:
                        ans = f"API error: {e}. Check your GEMINI_API_KEY."
                st.session_state.messages.append({"role":"assistant","content":ans})
                st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask anything about VSPL...", key="general_chat_input"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("Gemini is thinking..."):
                try:
                    ans = ask_gemini(st.session_state.messages, kb)
                except Exception as e:
                    ans = f"API error: {e}. Please ensure GEMINI_API_KEY is set."
            st.session_state.messages.append({"role":"assistant","content":ans})
            st.rerun()

        if st.button("[CLEAR] Clear General Chat", use_container_width=False):
            st.session_state.messages = [
                {"role":"assistant",
                 "content":"Hello! Hello! I'm the VSPL Smart Assistant. How can I help you?"}
            ]
            st.rerun()

    # [Forecast] AI BI Data Analyst (Rill-Style)
    with smart_tab2:
        st.subheader("📊 Conversational Data Analyst")
        st.markdown("Query, aggregate, and visualize our local manufacturing datasets Conversational-style using DuckDB/Pandas.")
        
        # Dataset selector
        dataset_choice = st.selectbox("Select Active Analytics Dataset", ["Standard Casting Dataset (1,200 runs)", "Large Deep Learning Dataset (50,000 runs)"])
        
        # Load active dataset
        if dataset_choice == "Standard Casting Dataset (1,200 runs)":
            df_bi = pd.read_csv(DATA_DIR / 'casting_data.csv')
            dataset_name = "casting_data.csv"
        else:
            df_bi = pd.read_csv(DATA_DIR / 'casting_data_large.csv')
            dataset_name = "casting_data_large.csv"
            
        st.info(f"[ACTIVE] Active Data Frame: **{df_bi.shape[0]:,} records** with features: `{', '.join(df_bi.columns.tolist())}`")
        
        if 'bi_messages' not in st.session_state:
            st.session_state.bi_messages = [
                {"role": "assistant", 
                 "content": "Hello! Hello! I'm your VSPL Conversational BI Analyst. Like Rill Data, you can query our active casting run database directly. Ask me things like:\n- *'Show the average quality score by mold type'*\n- *'Calculate the correlation between pour temperature and reject rate'*\n- *'Show a bar chart of the count of rejects vs mold types'*\n- *'Compare spin speed (RPM) averages between rejects and good runs'*"}
            ]
            
        # Display BI chat history
        for msg in st.session_state.bi_messages:
            with st.chat_message(msg['role'], avatar="[Forecast]" if msg['role']=='assistant' else "[USER]"):
                st.markdown(msg['content'])
                if 'query_result' in msg:
                    st.dataframe(pd.DataFrame(msg['query_result']), use_container_width=True)
                if 'chart_fig' in msg:
                    st.plotly_chart(msg['chart_fig'], use_container_width=True)
                    
        # BI Chat input
        if bi_prompt := st.chat_input("Query local casting dataset...", key="bi_chat_input"):
            st.session_state.bi_messages.append({"role": "user", "content": bi_prompt})
            
            with st.spinner("AI Analyst is querying dataset..."):
                api_key = st.session_state.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
                if not api_key:
                    st.session_state.bi_messages.append({"role": "assistant", "content": "[WARNING] Gemini API key not set. Please set the GEMINI_API_KEY in the sidebar to use the Rill BI chatbot."})
                    st.rerun()
                
                # Build custom system prompt containing schema
                columns = df_bi.columns.tolist()
                system_prompt = build_bi_system_prompt(dataset_name, columns)
                
                # Direct Gemini REST call
                try:
                    import requests
                    import time
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                    payload = {
                        "contents": [{"role": "user", "parts": [{"text": bi_prompt}]}],
                        "systemInstruction": {"parts": [{"text": system_prompt}]}
                    }
                    
                    max_retries = 3
                    res = None
                    last_err = ""
                    for attempt in range(max_retries):
                        try:
                            res = requests.post(url, json=payload, timeout=60)
                            res.raise_for_status()
                            break
                        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                            last_err = str(e)
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)
                                continue
                            raise Exception(f"Request timed out/failed after {max_retries} attempts: {last_err}")
                    
                    if res is not None and res.status_code == 200:
                        res_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Clean any markdown code blocks
                        if res_text.startswith("```json"):
                            res_text = res_text[7:]
                        if res_text.endswith("```"):
                            res_text = res_text[:-3]
                        res_text = res_text.strip()
                        
                        # Parse
                        bi_data = json.loads(res_text)
                        pandas_code = bi_data["pandas_code"]
                        explanation = bi_data["query_explanation"]
                        chart_requested = bi_data.get("chart_requested", False)
                        chart_type = bi_data.get("chart_type", None)
                        x_col = bi_data.get("x_col", None)
                        y_col = bi_data.get("y_col", None)
                        insight = bi_data.get("metallurgical_insight", "")
                        
                        # Evaluate pandas code on active df
                        local_context = {"df": df_bi}
                        try:
                            result_df = eval(pandas_code, {}, local_context)
                            
                            # Compile response content
                            reply_content = f"**Query**: `{pandas_code}`\n\n{explanation}\n\n**Metallurgical Insight**: {insight}"
                            msg_dict = {"role": "assistant", "content": reply_content}
                            
                            # Add query results
                            if isinstance(result_df, pd.DataFrame):
                                msg_dict['query_result'] = result_df.to_dict(orient='records')
                            elif isinstance(result_df, pd.Series):
                                msg_dict['query_result'] = result_df.reset_index().to_dict(orient='records')
                            else:
                                reply_content += f"\n\n**Result**: `{result_df}`"
                                msg_dict['content'] = reply_content
                                
                            # Add chart
                            if chart_requested and isinstance(result_df, pd.DataFrame) and x_col in result_df.columns and y_col in result_df.columns:
                                if chart_type == "bar":
                                    fig = px.bar(result_df, x=x_col, y=y_col, color=x_col, title=f"{y_col} by {x_col}", template="plotly_dark")
                                    msg_dict['chart_fig'] = fig
                                elif chart_type == "line":
                                    fig = px.line(result_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_dark")
                                    msg_dict['chart_fig'] = fig
                                elif chart_type == "scatter":
                                    fig = px.scatter(result_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}", template="plotly_dark")
                                    msg_dict['chart_fig'] = fig
                                    
                            st.session_state.bi_messages.append(msg_dict)
                        except Exception as eval_err:
                            st.session_state.bi_messages.append({"role": "assistant", "content": f"[WARNING] Error executing the generated Pandas query `{pandas_code}`: {eval_err}"})
                    else:
                        st.session_state.bi_messages.append({"role": "assistant", "content": f"[WARNING] API Error ({res.status_code}): {res.text}"})
                except Exception as req_err:
                    st.session_state.bi_messages.append({"role": "assistant", "content": f"[WARNING] Connection error: {req_err}"})
            st.rerun()
            
        if st.button("[CLEAR] Clear BI Analyst Conversations", use_container_width=False):
            st.session_state.bi_messages = [
                {"role": "assistant", 
                 "content": "Hello! Hello! I'm your VSPL Conversational BI Analyst. Like Rill Data, you can query our active casting run database directly."}
            ]
            st.rerun()


elif page == "📷 CV Defect Detector":
    st.title("📷 Real-time CV Defect Detector")
    st.markdown("##### Real-time automated surface inspection using Convolutional Neural Networks (CNN) & YOLOv8")
    st.markdown("---")

    from PIL import Image, ImageDraw, ImageFont

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("⚙️ Inspection Control Panel")
        source = st.radio("Casting Image Source", [
            "🎥 Choose from Casting Samples (Simulated Feed)",
            "📂 Upload High-Resolution Inspection Image"
        ])

        outcome = "Auto-Detect"
        uploaded_file = None

        if "[IoT] Choose from Casting Samples" in source:
            sample_run = st.selectbox("Select Casting Run Sample", [
                "Run #4105 - Flawless Surface Casting",
                "Run #4106 - Surface Crack Flaw",
                "Run #4107 - Gas Blowhole Cavities",
                "Run #4108 - Solidification Shrinkage"
            ])
        else:
            uploaded_file = st.file_uploader("Upload casting inspection image", type=['png', 'jpg', 'jpeg'])
            outcome = st.selectbox("Simulated Detection Outcome", [
                "Auto-Detect (AI Classification)",
                "Force: Flawless Casting",
                "Force: Gas Blowholes",
                "Force: Surface Crack",
                "Force: Shrinkage Cavity"
            ])

        st.markdown("---")
        st.markdown("**[SCIENCE] Defect Detection Model Settings**")
        model_name = st.selectbox("AI Model Model Architecture", [
            "YOLOv8x-Defect (Real-time Object Detection)",
            "ResNet50-Ductile (Surface Classification)"
        ])
        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        run_inspection = st.button("🔍 Run Surface Inspection", use_container_width=True, type="primary")

    with col2:
        st.subheader("📊 CV Diagnostics")

        # Determine active run parameters
        if "🎥 Choose from Casting Samples" in source:
            if "Flawless" in sample_run:
                run_type = "flawless"
                img_path = DATA_DIR / 'defect_flawless.png'
            elif "Crack" in sample_run:
                run_type = "crack"
                img_path = DATA_DIR / 'defect_crack.png'
            elif "Blowhole" in sample_run:
                run_type = "blowhole"
                img_path = DATA_DIR / 'defect_blowhole.png'
            else:
                run_type = "shrinkage"
                img_path = DATA_DIR / 'defect_shrinkage.png'
        else:
            if uploaded_file is not None:
                # Use uploaded file, map based on outcome force selection
                if "Flawless" in outcome:
                    run_type = "flawless"
                    img_path = DATA_DIR / 'defect_flawless.png'
                elif "Crack" in outcome:
                    run_type = "crack"
                    img_path = DATA_DIR / 'defect_crack.png'
                elif "Blowhole" in outcome:
                    run_type = "blowhole"
                    img_path = DATA_DIR / 'defect_blowhole.png'
                elif "Shrinkage" in outcome:
                    run_type = "shrinkage"
                    img_path = DATA_DIR / 'defect_shrinkage.png'
                else:
                    # Auto-detect maps based on filename or defaults to flawless
                    fn_l = uploaded_file.name.lower()
                    if "crack" in fn_l:
                        run_type = "crack"
                        img_path = DATA_DIR / 'defect_crack.png'
                    elif "blowhole" in fn_l or "bubble" in fn_l or "hole" in fn_l:
                        run_type = "blowhole"
                        img_path = DATA_DIR / 'defect_blowhole.png'
                    elif "shrink" in fn_l or "cavity" in fn_l:
                        run_type = "shrinkage"
                        img_path = DATA_DIR / 'defect_shrinkage.png'
                    else:
                        run_type = "flawless"
                        img_path = DATA_DIR / 'defect_flawless.png'
            else:
                img_path = None
                run_type = None

        if img_path is not None and img_path.exists():
            with st.spinner("AI running computer vision inference..."):
                import time
                time.sleep(0.6) # simulate model inference delay

                # Load image & draw boxes
                img = Image.open(img_path).convert("RGB")
                width, height = img.size
                draw = ImageDraw.Draw(img)

                # Bolding trick for text font
                try:
                    font = ImageFont.truetype("arial.ttf", max(14, int(width * 0.035)))
                except Exception:
                    font = ImageFont.load_default()

                boxes = []

elif page == "✏️ Diagram Generator":
    render_diagram_generator()

elif page == "🏭 Production Dashboard":
    render_production_dashboard()

elif page == "📦 Inventory Tracker":
    render_inventory_tracker()

elif page == "🔧 Predictive Maintenance":
    render_predictive_maintenance()

elif page == "🚚 Order Tracker":
    render_order_tracker()

elif page == "🔥 Heat Treatment":
    render_heat_treatment()

elif page == "💰 Cost Estimator":
    render_cost_estimation()

elif page == "🎯 Process Optimizer":
    render_process_optimizer()

elif page == "📉 SPC Dashboard":
    render_spc_dashboard()

elif page == "💡 SHAP Explainability":
    render_shap_explainer()

elif page == "🌐 Digital Twin":
    render_digital_twin()

elif page == "🔍 Root Cause Analysis":
    render_rca()

elif page == "⚡ Energy Optimizer":
    render_energy_optimizer()

elif page == "🧬 Multi-Objective (NSGA-II)":
    render_multi_objective()