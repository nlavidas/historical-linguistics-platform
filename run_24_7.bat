@echo off
echo ===============================================================================
echo  HFRI-NKUA AI CORPUS PLATFORM - 24/7 AUTONOMOUS OPERATION
echo  National and Kapodistrian University of Athens (NKUA)
echo  Hellenic Foundation for Research and Innovation (HFRI)
echo ===============================================================================
echo.

cd /d "%~dp0"

:START
echo [%date% %time%] Starting 24/7 platform...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Install requirements
echo Installing/updating requirements...
pip install -r requirements.txt --quiet
pip install spacy transformers nltk textblob trankit ollama --quiet

REM Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

echo.
echo ===============================================================================
echo  PLATFORM STARTING - PRESS CTRL+C TO STOP
echo ===============================================================================
echo.
echo Multi-AI models will be initialized...
echo Platform will run continuously until stopped.
echo.
echo Dashboard available at: http://localhost:8000
echo.

REM Run platform with multi-AI annotator (continuous mode)
python unified_corpus_platform.py --cycles 0 --delay 30

REM If it exits, restart after 10 seconds
echo.
echo [%date% %time%] Platform stopped. Restarting in 10 seconds...
echo Press Ctrl+C to cancel restart.
timeout /t 10
goto START
