#!/bin/bash
# =============================================================================
# SETUP 24/7 AUTONOMOUS CORPUS COLLECTOR SERVICE
# Run this script on the server to install and start the service
# =============================================================================

set -e

echo "========================================"
echo "Setting up 24/7 Corpus Collector Service"
echo "========================================"

# Navigate to project
cd /root/corpus_platform

# Pull latest code
echo "[1/6] Pulling latest code..."
git pull origin master

# Create data directories
echo "[2/6] Creating directories..."
mkdir -p /root/corpus_platform/data/logs
mkdir -p /root/corpus_platform/data/cache

# Install Python dependencies
echo "[3/6] Installing dependencies..."
pip3 install requests --quiet 2>/dev/null || true

# Copy service file
echo "[4/6] Installing systemd service..."
cp corpus-collector.service /etc/systemd/system/corpus-collector.service

# Reload systemd
echo "[5/6] Reloading systemd..."
systemctl daemon-reload

# Enable and start service
echo "[6/6] Starting service..."
systemctl enable corpus-collector
systemctl restart corpus-collector

# Show status
echo ""
echo "========================================"
echo "Service Status:"
echo "========================================"
systemctl status corpus-collector --no-pager

echo ""
echo "========================================"
echo "DONE! Service is now running 24/7"
echo "========================================"
echo ""
echo "Useful commands:"
echo "  View logs:    journalctl -u corpus-collector -f"
echo "  Check status: systemctl status corpus-collector"
echo "  Stop service: systemctl stop corpus-collector"
echo "  Restart:      systemctl restart corpus-collector"
echo ""
echo "Log files:"
echo "  /root/corpus_platform/data/logs/autonomous_*.log"
echo "  /root/corpus_platform/data/logs/service.log"
echo ""
