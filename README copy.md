# CastIQ — Intelligent AI Platform for Casting Quality, Defect Detection & Demand Forecasting

> An end-to-end AI/ML platform built for **Vijay Spheroidals Pvt Ltd (VSPL)**, a ductile iron casting manufacturer. CastIQ integrates 20 intelligent modules covering quality prediction, defect detection, demand forecasting, process optimization, explainability, and operations management — all inside a single Streamlit application with a custom Industrial Command Center UI.

---

## Screenshots

**Dashboard — Module Overview**
![CastIQ Dashboard](docs/screenshots/screenshot_dashboard.png)

**Casting Quality Predictor — Real-time Quality Score**
![Casting Quality Predictor](docs/screenshots/screenshot_quality_predictor.png)

**Process Optimizer — Inverse AI**
![Process Optimizer](docs/screenshots/screenshot_process_optimizer.png)

**SPC Dashboard — Statistical Process Control**
![SPC Dashboard](docs/screenshots/screenshot_spc_dashboard.png)

---

## Modules

| # | Module | What it does | Model / Tech |
|---|--------|-------------|--------------|
| 1 | 🔮 Casting Quality Predictor | Predict reject % and quality score before the pour — single batch and bulk CSV modes | Gradient Boosting + PyTorch Multitask DNN |
| 2 | 🎯 Process Optimizer | Inverse ML — finds the optimal chemistry recipe for a target quality score | Scipy Differential Evolution |
| 3 | 📉 SPC Dashboard | X-bar, R-chart, CUSUM control charts with Nelson rule violation alerts | Statistical Process Control |
| 4 | 💡 SHAP Explainability | Waterfall plots, global feature importance, and side-by-side SHAP comparison | SHAP TreeExplainer / KernelExplainer |
| 5 | 📊 Demand Forecasting | 6-month sector-wise order demand forecast with confidence intervals | Facebook Prophet |
| 6 | ⚗️ Alloy Recommendation | Recommends the optimal ductile iron grade from customer mechanical specs | Random Forest Classifier |
| 7 | 📄 RFQ Intelligence | Parses RFQ text, extracts specs, and generates an ex-works price quote PDF | Regex NLP + metallurgical rules |
| 8 | 🤖 Smart Assistant | Conversational metallurgical assistant with VSPL knowledge base | Google Gemini 2.5 Flash API |
| 9 | 📷 CV Defect Detector | Real-time surface defect classification from uploaded images or webcam | Custom CNN + YOLOv8 |
| 10 | 🏭 Production Dashboard | Live centrifugal pipe yield tracking and ladle heat logging | Real-time IoT simulation |
| 11 | 📦 Inventory Tracker | Raw alloy and consumables ledger with checkout and restocking | Transaction ledger |
| 12 | 🔧 Predictive Maintenance | Equipment health diagnostics, vibration runout charts, and RUL estimation | Anomaly detection |
| 13 | 🚚 Order Tracker | Visual pipeline tracking order stages from melt to dispatch | Stage tracker |
| 14 | 🔥 Heat Treatment Simulator | Simulates thermal cycles and predicts microstructural strength outcomes | Thermal equations + phase rules |
| 15 | 💰 Cost Estimator | Casting weight calculator and ex-works price quotation | Geometry calculations |
| 16 | 🧠 Deep Learning Hub | Download 50k casting dataset, train and evaluate the PyTorch DNN and Mini-GPT | PyTorch + custom Mini-GPT Transformer |
| 17 | 🌐 Digital Twin | Cooling curve simulation with latent heat plateau modelling | Physics-based simulation |
| 18 | 🔍 Root Cause Analysis | Diagnoses defect type and generates a corrective action plan | SHAP + 7 physics-based heuristic rules |
| 19 | ⚡ Energy Optimizer | Minimizes pour temperature to reduce furnace energy consumption | Chvorinov's Rule + cp × mass |
| 20 | 🧬 Multi-Objective Optimizer | Simultaneous Quality + Cost + Delivery optimization with Pareto front | NSGA-II Genetic Algorithm (pymoo) |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Streamlit, Plotly Graph Objects, custom Industrial Command Center CSS (navy + gold) |
| Traditional ML | scikit-learn (Gradient Boosting, Random Forest), Facebook Prophet |
| Deep Learning | PyTorch — Multitask Dense Residual DNN, local Mini-GPT Transformer |
| Explainable AI | SHAP (TreeExplainer, KernelExplainer), waterfall + beeswarm plots |
| Optimization | scipy Differential Evolution, pymoo NSGA-II |
| Computer Vision | OpenCV, Ultralytics YOLOv8, custom CNN |
| AI Assistant | Google Gemini API (gemini-2.5-flash) via REST |
| PDF Reports | ReportLab Flowables and Paragraph Styles |

---

## Project Structure

```
CastIQ/
├── app.py                        # Main Streamlit app (entry point)
├── setup.py                      # One-shot model training script
├── requirements.txt
├── START.bat                     # Windows quick-launch
├── .env                          # API keys (not committed)
│
├── modules/                      # 14 plug-in render modules
│   ├── process_optimizer.py
│   ├── spc_dashboard.py
│   ├── shap_explainer.py
│   ├── energy_optimizer.py
│   ├── multi_objective_optimizer.py
│   ├── root_cause_analysis.py
│   ├── digital_twin.py
│   ├── diagram_generator.py
│   └── ...
│
├── backend/
│   ├── models/                   # Pre-trained .pkl and .pth model files
│   └── data/                     # CSVs, JSONs, sample defect images
│
└── docs/
    └── screenshots/              # UI screenshots for README
```

---

## Quick Start

### Requirements

- Python 3.10+
- Windows (tested) / Linux / macOS
- A free [Google Gemini API key](https://aistudio.google.com) for the Smart Assistant

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/DarshanCat/CastIQ-An-Intelligent-AI-Platform-for-Casting-Quality-Defect-Detection-and-Demand-Forecasting.git
cd CastIQ-...

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (runs once, takes ~2–3 minutes)
python setup.py

# 4. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

On Windows you can also just double-click **START.bat**.

### Gemini API Key Setup

The Smart Assistant and AI Diagram Generator require a Gemini API key. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Or set it as an environment variable before running:

```bash
# Windows CMD
set GEMINI_API_KEY=your-key-here

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"

# Linux / macOS
export GEMINI_API_KEY=your-key-here
```

All other modules (ML prediction, optimization, defect detection, SPC, etc.) work fully offline without any API key.

---

## How It Works

### 1. Data & Model Training (`setup.py`)

Running `setup.py` once generates and trains three core models from scratch:

- **Casting Quality Predictor** — a Gradient Boosting classifier + regressor trained on a 50k-row physics-based synthetic dataset with 18 features (carbon %, silicon %, Mg pct, G-factor, nodularity index, cooling rate, etc.)
- **Demand Forecasting** — a Prophet model trained per sector (Hydraulics, Wind Energy, Locomotives, Pharma, Heavy Engineering) with 30 months of historical data
- **Alloy Recommendation** — a Random Forest classifier trained on 1,000 synthetic samples mapping mechanical specs to ductile iron grades (IS 400/18 through IS 800/2)

### 2. Deep Learning (`backend/train_deep_learning.py`)

A PyTorch **Multitask Dense Residual Neural Network** jointly predicts quality score (regression) and reject flag (classification) from the same 18-feature input. The architecture uses skip connections between dense layers to improve gradient flow on the small training set.

### 3. Computer Vision Defect Detector

Classifies four casting surface conditions — **Flawless**, **Blowhole**, **Crack**, and **Shrinkage** — from uploaded images using a custom CNN. YOLOv8 integration enables real-time webcam-based defect scanning.

### 4. NSGA-II Multi-Objective Optimizer

Uses the **pymoo** NSGA-II genetic algorithm to simultaneously optimize three conflicting objectives:
- Maximize casting quality score
- Minimize production cost per batch
- Minimize delivery lead time

The output is a Pareto front of non-dominated solutions visualized as a 3D scatter plot, with a "best solution picker" based on user preference (quality-first, cost-first, or balanced).

---

## Key Features

- **Offline-first** — all ML modules run locally; only the Smart Assistant needs internet
- **PDF report export** — every prediction module generates a downloadable ReportLab PDF report
- **Batch prediction** — upload a CSV of casting parameters and get quality scores for the entire batch
- **Explainable AI** — every prediction is accompanied by a SHAP waterfall plot showing which parameters drove the result
- **Industrial UI** — custom CSS with a navy + gold "command center" theme designed for shop-floor readability

---

## Dependencies

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
prophet>=1.1.4
joblib>=1.3.0
plotly>=5.18.0
reportlab>=4.0.0
google-generativeai>=0.4.0
groq>=0.4.2
opencv-python>=4.8.0.0
ultralytics>=8.1.0
scipy>=1.10.0
shap>=0.44.0
pymoo>=0.6.1
requests>=2.31.0
```

---

## About

Built as a B.Tech final-year project at **CMR University, Bengaluru** for the degree in Artificial Intelligence & Machine Learning (2025–26).

**Client:** Vijay Spheroidals Pvt Ltd (VSPL) — manufacturer of ductile iron centrifugal pipes, hydraulic manifolds, and precision castings.

**Author:** Darshan B G — [GitHub](https://github.com/DarshanCat) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## License

This project was built for academic and client demonstration purposes. All rights reserved.
