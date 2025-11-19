@echo off
echo ===============================================
echo  UNIFIED CORPUS PLATFORM - WEB DASHBOARD
echo ===============================================
echo.

cd /d "%~dp0"

echo Installing requirements...
pip install fastapi uvicorn --quiet

echo.
echo Starting web dashboard...
echo Dashboard will open at: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo.

python web_dashboard.py

pause
