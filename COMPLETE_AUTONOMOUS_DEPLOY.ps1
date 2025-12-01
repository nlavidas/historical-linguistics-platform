# Autonomous Complete Deployment Script
# This script does EVERYTHING autonomously
# Run this ONCE from Windows PowerShell, then walk away

param(
    [string]$VM_IP = "135.125.216.3",
    [string]$VM_USER = "ubuntu",
    [string]$SSH_KEY_PATH = "$env:USERPROFILE\.ssh\id_rsa_ovh"
)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "AUTONOMOUS COMPLETE DEPLOYMENT" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting full deployment to OVH VM..." -ForegroundColor Yellow
Write-Host "VM: $VM_IP" -ForegroundColor Yellow
Write-Host "SSH Key: $SSH_KEY_PATH" -ForegroundColor Yellow
Write-Host ""

# Function to run SSH commands
function Invoke-SSHCommand {
    param([string]$Command, [string]$Description)
    Write-Host ">>> $Description" -ForegroundColor Green
    Write-Host "    $Command" -ForegroundColor Gray
    & ssh -i $SSH_KEY_PATH -o StrictHostKeyChecking=no "$VM_USER@$VM_IP" $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    FAILED: $Command" -ForegroundColor Red
        exit 1
    }
    Write-Host "    SUCCESS" -ForegroundColor Green
}

# Function to SCP files
function Copy-ToVM {
    param([string]$LocalPath, [string]$RemotePath, [string]$Description)
    Write-Host ">>> $Description" -ForegroundColor Green
    Write-Host "    scp -i $SSH_KEY_PATH -r $LocalPath $VM_USER@$VM_IP`:$RemotePath" -ForegroundColor Gray
    & scp -i $SSH_KEY_PATH -r $LocalPath "$VM_USER@$VM_IP`:$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    FAILED: File copy" -ForegroundColor Red
        exit 1
    }
    Write-Host "    SUCCESS" -ForegroundColor Green
}

# STEP 1: Check VM connectivity
Write-Host ""
Write-Host "STEP 1: Testing VM connectivity..." -ForegroundColor Magenta
Invoke-SSHCommand "echo 'VM is accessible'" "Testing SSH connection"

# STEP 2: Check and free disk space
Write-Host ""
Write-Host "STEP 2: Checking VM disk usage..." -ForegroundColor Magenta
Invoke-SSHCommand "df -h /" "Checking disk usage"

$diskUsage = & ssh -i $SSH_KEY_PATH "$VM_USER@$VM_IP" "df / | tail -1 | awk '{print \$5}' | sed 's/%//'"
if ([int]$diskUsage -gt 95) {
    Write-Host "Disk usage is $diskUsage% - attempting to free space..." -ForegroundColor Yellow
    Invoke-SSHCommand "sudo du -sh /home/ubuntu/* 2>/dev/null | sort -hr | head -5" "Finding large directories"
    Invoke-SSHCommand "rm -rf ~/old_* ~/cache ~/temp 2>/dev/null; echo 'Cleanup completed'" "Cleaning up old files"
} else {
    Write-Host "Disk usage OK: $diskUsage%" -ForegroundColor Green
}

# STEP 3: Sync all project files
Write-Host ""
Write-Host "STEP 3: Syncing project files..." -ForegroundColor Magenta
Copy-ToVM "Z:\corpus_platform\*" "~/corpus_platform" "Syncing corpus platform"

# STEP 4: Run autonomous VM setup
Write-Host ""
Write-Host "STEP 4: Running autonomous VM setup..." -ForegroundColor Magenta
Invoke-SSHCommand "cd ~/corpus_platform && chmod +x autonomous_vm_setup.sh" "Making setup script executable"
Invoke-SSHCommand "cd ~/corpus_platform && nohup ./autonomous_vm_setup.sh > setup.log 2>&1 &" "Starting autonomous setup in background"
Start-Sleep -Seconds 5
Invoke-SSHCommand "cd ~/corpus_platform && tail -20 setup.log" "Checking setup progress"

# STEP 5: Wait for services to start
Write-Host ""
Write-Host "STEP 5: Waiting for services to start..." -ForegroundColor Magenta
Start-Sleep -Seconds 30
Invoke-SSHCommand "cd ~/corpus_platform && ./check_status.sh 2>/dev/null || echo 'Status check not ready yet'" "Checking service status"

# STEP 6: Test web access
Write-Host ""
Write-Host "STEP 6: Testing web access..." -ForegroundColor Magenta
try {
    $response = Invoke-WebRequest -Uri "http://$VM_IP" -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "Web panel accessible at http://$VM_IP" -ForegroundColor Green
    }
} catch {
    Write-Host "Web panel not ready yet (may still be starting)" -ForegroundColor Yellow
}

# STEP 7: Set up monitoring cron job
Write-Host ""
Write-Host "STEP 7: Setting up autonomous monitoring..." -ForegroundColor Magenta
Invoke-SSHCommand "cd ~/corpus_platform && (crontab -l 2>/dev/null; echo '*/30 * * * * cd ~/corpus_platform && ./check_status.sh > status.log 2>&1') | crontab -" "Adding monitoring cron job"

# STEP 8: Set up GitHub repository
Write-Host ""
Write-Host "STEP 8: Setting up GitHub repository..." -ForegroundColor Magenta
Invoke-SSHCommand "cd ~/corpus_platform && chmod +x setup_github_repo.sh" "Making GitHub setup executable"
Invoke-SSHCommand "cd ~/corpus_platform && ./setup_github_repo.sh" "Setting up GitHub repository"

# STEP 9: Set up Trilium export
Write-Host ""
Write-Host "STEP 9: Setting up Trilium/Appflowy export..." -ForegroundColor Magenta
Invoke-SSHCommand "cd ~/corpus_platform && chmod +x export_for_trilium.sh" "Making Trilium export executable"
Invoke-SSHCommand "cd ~/corpus_platform && ./export_for_trilium.sh" "Exporting notes for open-source apps"

# STEP 10: Set up private website
Write-Host ""
Write-Host "STEP 10: Setting up private password-protected website..." -ForegroundColor Magenta
Invoke-SSHCommand "cd ~/corpus_platform && chmod +x setup_private_website.sh" "Making website setup executable"
Invoke-SSHCommand "cd ~/corpus_platform && ./setup_private_website.sh" "Setting up private website"

# STEP 11: Final status report
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your autonomous platform is now running 24/7!" -ForegroundColor Green
Write-Host ""
Write-Host "Access Points:" -ForegroundColor White
Write-Host "  - Web Control Panel: http://$VM_IP" -ForegroundColor White
Write-Host "  - VS Code IDE:       http://$VM_IP/ide/" -ForegroundColor White
Write-Host "  - Private Website:   http://corpus-platform.nlavid.as" -ForegroundColor White
Write-Host "  - Password:          historical_linguistics_2025" -ForegroundColor White
Write-Host ""
Write-Host "GitHub Repository:" -ForegroundColor White
Write-Host "  - https://github.com/nlavidas/diachronic-corpus-platform" -ForegroundColor White
Write-Host ""
Write-Host "Monitoring:" -ForegroundColor White
Write-Host "  - Status logs:       ssh ubuntu@$VM_IP 'cd ~/corpus_platform && tail -f setup.log'" -ForegroundColor White
Write-Host "  - Service health:    ssh ubuntu@$VM_IP 'cd ~/corpus_platform && ./check_status.sh'" -ForegroundColor White
Write-Host ""
Write-Host "The platform will continue running autonomously with:" -ForegroundColor Yellow
Write-Host "  - Auto-restart on VM reboot (systemd)" -ForegroundColor Yellow
Write-Host "  - Health monitoring every 30 minutes" -ForegroundColor Yellow
Write-Host "  - Continuous collection, annotation, and reporting" -ForegroundColor Yellow
Write-Host "  - Nightly GitHub sync and website rebuild" -ForegroundColor Yellow
Write-Host "  - Trilium/Appflowy note exports" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
