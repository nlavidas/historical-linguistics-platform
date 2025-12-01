# ============================================================================
# BACKUP AND MIGRATION SCRIPT
# Backs up Z: drive and prepares for OVH VPS migration
# ============================================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "BACKUP AND MIGRATION SCRIPT" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Configuration
$BackupDate = Get-Date -Format "yyyy-MM-dd_HHmm"
$BackupDir = "C:\Users\nlavi\Backups\$BackupDate"
$ZDrive = "Z:\"
$CorpusPlatform = "Z:\corpus_platform"
$GitHubUser = "nlavidas"

# Create backup directory
Write-Host "`n[1/6] Creating backup directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
New-Item -ItemType Directory -Force -Path "$BackupDir\corpus_platform" | Out-Null
New-Item -ItemType Directory -Force -Path "$BackupDir\z_drive_full" | Out-Null
Write-Host "Backup directory: $BackupDir" -ForegroundColor Green

# Backup corpus_platform
Write-Host "`n[2/6] Backing up corpus_platform..." -ForegroundColor Yellow
if (Test-Path $CorpusPlatform) {
    Copy-Item -Path "$CorpusPlatform\*" -Destination "$BackupDir\corpus_platform" -Recurse -Force
    Write-Host "Corpus platform backed up successfully" -ForegroundColor Green
} else {
    Write-Host "corpus_platform not found at $CorpusPlatform" -ForegroundColor Red
}

# Backup entire Z: drive
Write-Host "`n[3/6] Backing up entire Z: drive..." -ForegroundColor Yellow
Write-Host "This may take a while depending on size..." -ForegroundColor Gray
if (Test-Path $ZDrive) {
    # Get size first
    $ZSize = (Get-ChildItem -Path $ZDrive -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "Z: drive size: $([math]::Round($ZSize, 2)) GB" -ForegroundColor Gray
    
    # Copy everything
    robocopy $ZDrive "$BackupDir\z_drive_full" /E /R:1 /W:1 /MT:8 /NFL /NDL /NJH /NJS
    Write-Host "Z: drive backed up successfully" -ForegroundColor Green
} else {
    Write-Host "Z: drive not accessible" -ForegroundColor Red
}

# Create compressed archive
Write-Host "`n[4/6] Creating compressed archive..." -ForegroundColor Yellow
$ArchivePath = "C:\Users\nlavi\Backups\backup_$BackupDate.zip"
Compress-Archive -Path $BackupDir -DestinationPath $ArchivePath -Force
Write-Host "Archive created: $ArchivePath" -ForegroundColor Green

# Git push all repos
Write-Host "`n[5/6] Pushing to GitHub..." -ForegroundColor Yellow
Set-Location $CorpusPlatform
git add .
git commit -m "Backup before OVH VPS migration - $BackupDate" 2>$null
git push origin master
Write-Host "GitHub updated" -ForegroundColor Green

# Generate OVH VPS setup script
Write-Host "`n[6/6] Generating OVH VPS setup script..." -ForegroundColor Yellow

$OVHScript = @'
#!/bin/bash
# ============================================================================
# OVH VPS SETUP SCRIPT
# Run this on your new OVH VPS after SSH connection
# ============================================================================

echo "============================================"
echo "OVH VPS SETUP FOR GREEK LINGUISTICS PLATFORM"
echo "============================================"

# Update system
echo "[1/8] Updating system..."
apt update && apt upgrade -y

# Install dependencies
echo "[2/8] Installing dependencies..."
apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx ufw

# Setup firewall
echo "[3/8] Configuring firewall..."
ufw allow 22
ufw allow 80
ufw allow 443
ufw --force enable

# Create app directory
echo "[4/8] Setting up application..."
cd /root
rm -rf corpus_platform 2>/dev/null
git clone https://github.com/nlavidas/historical-linguistics-platform.git corpus_platform
cd corpus_platform

# Install Python dependencies
echo "[5/8] Installing Python packages..."
pip3 install -r requirements.txt

# Create systemd service for 24/7 operation
echo "[6/8] Creating systemd service..."
cat > /etc/systemd/system/greek-corpus.service << 'EOF'
[Unit]
Description=Greek Corpus Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/corpus_platform
ExecStart=/usr/bin/python3 -m streamlit run platform_app.py --server.port 80 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "[7/8] Starting service..."
systemctl daemon-reload
systemctl enable greek-corpus
systemctl start greek-corpus

# Show status
echo "[8/8] Setup complete!"
echo ""
echo "============================================"
echo "PLATFORM IS NOW RUNNING"
echo "============================================"
echo ""
echo "Access your platform at: http://$(curl -s ifconfig.me)"
echo ""
echo "Login credentials:"
echo "  Username: nlavidas"
echo "  Password: GreekCorpus2024!"
echo ""
echo "Useful commands:"
echo "  systemctl status greek-corpus  - Check status"
echo "  systemctl restart greek-corpus - Restart"
echo "  journalctl -u greek-corpus -f  - View logs"
echo ""
'@

$OVHScript | Out-File -FilePath "$BackupDir\ovh_vps_setup.sh" -Encoding UTF8
Write-Host "OVH setup script saved to: $BackupDir\ovh_vps_setup.sh" -ForegroundColor Green

# Summary
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "BACKUP COMPLETE" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup location: $BackupDir" -ForegroundColor White
Write-Host "Archive: $ArchivePath" -ForegroundColor White
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. ORDER OVH VPS:" -ForegroundColor White
Write-Host "   https://www.ovhcloud.com/en/vps/" -ForegroundColor Gray
Write-Host "   - VPS Starter: 3.50 EUR/month (2GB RAM)" -ForegroundColor Gray
Write-Host "   - VPS Essential: 6 EUR/month (4GB RAM) [RECOMMENDED]" -ForegroundColor Gray
Write-Host "   - VPS Comfort: 12 EUR/month (8GB RAM)" -ForegroundColor Gray
Write-Host ""
Write-Host "2. AFTER VPS IS READY:" -ForegroundColor White
Write-Host "   a. SSH to your new VPS:" -ForegroundColor Gray
Write-Host "      ssh root@YOUR_NEW_VPS_IP" -ForegroundColor Gray
Write-Host ""
Write-Host "   b. Copy and run the setup script:" -ForegroundColor Gray
Write-Host "      # Copy ovh_vps_setup.sh content and paste in terminal" -ForegroundColor Gray
Write-Host "      # Or upload it via SCP:" -ForegroundColor Gray
Write-Host "      scp $BackupDir\ovh_vps_setup.sh root@YOUR_NEW_VPS_IP:/root/" -ForegroundColor Gray
Write-Host "      ssh root@YOUR_NEW_VPS_IP 'chmod +x /root/ovh_vps_setup.sh && /root/ovh_vps_setup.sh'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. CANCEL OLD PUBLIC CLOUD (after testing):" -ForegroundColor White
Write-Host "   Go to OVH Manager > Public Cloud > Delete instance" -ForegroundColor Gray
Write-Host "   This will save you ~80 EUR/month!" -ForegroundColor Gray
Write-Host ""
