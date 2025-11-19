@echo off
echo ===============================================================================
echo  HFRI-NKUA AI CORPUS PLATFORM - 24/7 WITH WEB DASHBOARD
echo  National and Kapodistrian University of Athens (NKUA)
echo  Hellenic Foundation for Research and Innovation (HFRI)
echo ===============================================================================
echo.

cd /d "%~dp0"

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt --quiet
pip install spacy transformers nltk textblob trankit ollama fastapi uvicorn --quiet

echo.
echo ===============================================================================
echo  STARTING TWO SERVICES
echo ===============================================================================
echo.
echo 1. Web Dashboard (http://localhost:8000)
echo 2. Background Worker (24/7 processing)
echo.
echo Press Ctrl+C to stop both services
echo.

REM Start dashboard in background
start "HFRI-NKUA Dashboard" /MIN python web_dashboard.py

REM Wait 5 seconds for dashboard to start
timeout /t 5 /nobreak >nul

echo Dashboard started at: http://localhost:8000
echo.

REM Start worker in foreground
echo Starting background worker...
python unified_corpus_platform.py --cycles 0 --delay 30

pause
