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

## ⚙️ Setup — Gemini API Key (for Smart Assistant & BI Chatbot)

The Smart Assistant and Gemini BI Chatbot use the Google Gemini API. The platform is configured to automatically load the key from the `.env` file in the project root:

```env
Gemini API key=your-gemini-api-key-here
```

Alternatively, you can set it as an environment variable before running:

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your-key-here
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

Get a free Gemini API key at: https://aistudio.google.com

---

## 📦 19 Integrated Advanced Modules

| Module | Description | Model / Tech |
|--------|-------------|--------------|
| 🔮 Casting Quality Predictor | Predict reject % before pour — single + batch | Gradient Boosting / PyTorch DNN |
| 🎯 Process Optimizer | Inverse ML searching for optimal recipe chemistry | Scipy Differential Evolution |
| 📉 SPC Dashboard | Control charts (Xbar, CUSUM) & Nelson rules | Statistical Process Control |
| 💡 SHAP Explainability | Single waterfall, Global importance, side-by-side SHAP | SHAP Explainable AI |
| 📊 Demand Forecasting | 6-month sector forecast with confidence intervals | Prophet |
| ⚗️ Alloy Recommendation | Recommend optimal ductile iron grade from specs | Random Forest |
| 📄 RFQ Intelligence | Extract specs and generate ex-works quotes | Regex NLP + metallurgical rules |
| 🤖 Smart Assistant | Intelligent VSPL metallurgical helper | **Google Gemini API** |
| 📷 CV Defect Detector | Real-time computer vision surface defect scanner | Custom CNN / YOLOv8 |
| 🏭 Production Dashboard | Centrifugal pipe yields & ladle heats logging | Real-time IoT Streaming |
| 📦 Inventory Tracker | Ledger for raw alloys and supplies with checkouts | Transaction Ledger |
| 🔧 Predictive Maintenance | Health diagnostics, vibration runout & RUL charts | Anomaly Detection |
| 🚚 Order Tracker | Pipeline tracks order stages from melt to dispatch | Stage tracker |
| 🔥 Heat Treatment | Simulate thermal cycles and microstructural strength | Thermal equations / Phase rules |
| 💰 Cost Estimator | Weight calculator & ex-works price quotes | Geometry calculations |
| 🧠 Deep Learning Hub | Downloader for 50k casting database & DNN control | PyTorch + Mini-GPT |
| 🌐 Digital Twin | Cooling curve simulation (latent heat plateau) | Physics simulation |
| 🔍 Root Cause Analysis | Diagnoses defect type and corrective action plan | SHAP + 7 physics rules |
| ⚡ Energy Optimizer | Minimizes pour temp to reduce furnace energy cost | Chvorinov's Rule + cp mass |
| 🧬 Multi-Objective (NSGA-II) | Quality + Cost + Delivery optimization | NSGA-II Genetic Pareto Front |

---

## 🏗️ Tech Stack

- **Frontend:** Streamlit + Plotly Graph Objects + Custom Glassmorphic HSL CSS
- **Traditional ML:** scikit-learn, Prophet
- **Deep Learning:** PyTorch Multitask Dense Residual Neural Networks, local Mini-GPT Transformer
- **Explainable AI:** SHAP (TreeExplainer, KernelExplainer)
- **Mathematical Optimization:** scipy (Differential Evolution), pymoo (NSGA-II)
- **Computer Vision:** Custom CNN, OpenCV, YOLOv8
- **AI Assistant:** Google Gemini API (gemini-2.5-flash)
- **PDF Reports:** ReportLab Flowables & Paragraph Styles

---

Built by **Darshan** | CMR University | B.Tech AI & ML 2025–26
# CastIQ-An-Intelligent-AI-Platform-for-Casting-Quality-Defect-Detection-and-Demand-Forecasting
