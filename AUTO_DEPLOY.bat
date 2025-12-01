@echo off
REM Autonomous Deployment Script for Windows
REM This syncs Z: drive to VM and runs autonomous setup

echo ========================================================================
echo AUTONOMOUS DEPLOYMENT TO VM
echo ========================================================================
echo.

REM Get VM IP (change if different)
set VM_IP=135.125.216.3
set VM_USER=ubuntu
set SSH_KEY=%USERPROFILE%\.ssh\id_rsa_ovh

echo Syncing Z:\corpus_platform to VM...
echo.

REM Sync everything to VM
scp -i "%SSH_KEY%" -r Z:\corpus_platform\* %VM_USER%@%VM_IP%:~/corpus_platform/

echo.
echo ✓ Sync complete
echo.
echo Running autonomous setup on VM...
echo.

REM Run autonomous setup on VM
ssh -i "%SSH_KEY%" %VM_USER%@%VM_IP% "cd ~/corpus_platform && chmod +x autonomous_vm_setup.sh && nohup ./autonomous_vm_setup.sh > setup.log 2>&1 &"

echo.
echo ========================================================================
echo DEPLOYMENT STARTED!
echo ========================================================================
echo.
echo The VM is now setting up autonomously in the background.
echo.
echo To check progress:
echo   ssh -i "%SSH_KEY%" %VM_USER%@%VM_IP%
echo   tail -f ~/corpus_platform/autonomous_vm_setup.log
echo.
echo When complete, access at:
echo   http://%VM_IP%
echo   http://%VM_IP%/ide/
echo.
echo Password: historical_linguistics_2025
echo ========================================================================
echo.
pause
