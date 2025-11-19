@echo off
:start
python professional_dashboard.py --production --port 8000
echo [%date% %time%] Server stopped, restarting...
goto start
