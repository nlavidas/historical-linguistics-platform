@echo off
echo ===============================================================================
echo  HFRI-NKUA AI CORPUS PLATFORM - VALIDATION
echo ===============================================================================
echo.

cd /d "%~dp0"

python validate_and_fix.py

echo.
echo ===============================================================================
echo  Validation complete. Check results above.
echo ===============================================================================
echo.
pause
