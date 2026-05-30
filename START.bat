@echo off
echo ================================================
echo   VSPL AI Platform - Setup and Launch
echo ================================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [2/3] Training all AI models (first time only)...
python setup.py

echo.
echo [3/3] Launching Streamlit app...
echo.
echo   Open browser at: http://localhost:8501
echo.
streamlit run app.py
