@echo off
REM — Go to the app folder
cd /d "C:\Users\piano\cookie_cutter_app"

REM — Activate the virtual environment
call .venv\Scripts\activate.bat

REM — Launch the Streamlit app
streamlit run ultimate_cookie_cutter_app.py

REM — Keep the window open if there’s an error
pause
