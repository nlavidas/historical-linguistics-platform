@echo off
title HFRI-NKUA Corpus Platform - Professional Dashboard

echo ===============================================================================
echo  HFRI-NKUA CORPUS PLATFORM - PROFESSIONAL INTERFACE
echo  National and Kapodistrian University of Athens
echo  Hellenic Foundation for Research and Innovation
echo ===============================================================================
echo.

cd /d "Z:\corpus_platform"

echo Installing production dependencies...
pip install --quiet --upgrade fastapi uvicorn aiohttp websockets

echo.
echo Starting professional dashboard...
echo.
echo Dashboard URL: http://localhost:8000
echo.
echo Features:
echo   - Professional academic interface
echo   - Real external source connections (Perseus, GitHub, Gutenberg)
echo   - Live processing statistics
echo   - Production-grade monitoring
echo.

python professional_dashboard.py

pause
