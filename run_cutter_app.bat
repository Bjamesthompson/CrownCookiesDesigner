@echo off
REM ───────────────────────────────────────────────────────────
REM 1) Go to your app folder
cd /d "C:\Users\piano\OneDrive\Pictures\CookieCutters"

REM 2) Launch Streamlit in a new window
start "Streamlit Server" py -3.10 -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501

REM 3) Wait up to 10 seconds for Streamlit to boot
timeout /t 2 /nobreak >nul

REM 4) Open your browser to the local URL
start "" "http://localhost:8501"

REM Done
