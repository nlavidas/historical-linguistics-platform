@echo off
echo ===============================================
echo  UNIFIED AI CORPUS PLATFORM
echo  Automatic Scraping, Parsing, and Annotation
echo ===============================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Installing requirements...
pip install -r requirements.txt --quiet

echo.
echo ===============================================
echo  Starting Platform...
echo ===============================================
echo.

python unified_corpus_platform.py %*

pause
