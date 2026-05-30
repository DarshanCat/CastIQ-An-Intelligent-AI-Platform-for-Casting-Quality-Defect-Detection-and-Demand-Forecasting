# 🏭 VSPL AI Intelligence Platform

AI-powered manufacturing platform for **Vijay Spheroidals Pvt Ltd**.

---

## 🚀 Quick Start (Windows)

### Option A — Double-click
```
Double-click START.bat
```

### Option B — Manual
```bash
pip install -r requirements.txt
python setup.py
streamlit run app.py
```

Open browser → http://localhost:8501

---

## ⚙️ Setup — Anthropic API Key (for Smart Assistant)

The Smart Assistant uses the Claude API. Set your key before running:

**Windows (CMD):**
```
set ANTHROPIC_API_KEY=your-key-here
streamlit run app.py
```

**Windows (PowerShell):**
```
$env:ANTHROPIC_API_KEY="your-key-here"
streamlit run app.py
```

Get a free API key at: https://console.anthropic.com

---

## 📦 6 Modules

| Module | Description | Model / Tech |
|--------|-------------|--------------|
| 🔮 Casting Quality Predictor | Predict reject % before pour — single + batch | Gradient Boosting |
| 📊 Demand Forecasting | 6-month sector forecast with confidence intervals | Prophet |
| ⚗️ Alloy Recommendation | Recommend grade from specs | Random Forest |
| 📄 RFQ Intelligence | Extract specs + generate quote | NLP + Rule Engine |
| 🤖 Smart Assistant | Intelligent VSPL chatbot | **Claude API (Anthropic)** |
| 📄 PDF Reports | Download quality report as PDF | ReportLab |

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Casting Classifier | Accuracy | 85.4% |
| Casting Regressor | MAE | 3.95 pts |
| Alloy Recommender | Accuracy | 94.5% |

---

## 🏗️ Tech Stack

- **Frontend:** Streamlit + Plotly
- **ML:** scikit-learn, Prophet, XGBoost
- **AI Assistant:** Anthropic Claude API (claude-sonnet-4)
- **PDF Reports:** ReportLab
- **Data:** Pandas, NumPy, JSON

---

Built by **Darshan** | CMR University | B.Tech AI & ML 2025–26
# CastIQ-An-Intelligent-AI-Platform-for-Casting-Quality-Defect-Detection-and-Demand-Forecasting
