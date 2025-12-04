#!/bin/bash
# Historical Linguistics Platform - OVH Server Installation Script
# University of Athens - Nikolaos Lavidas
#
# Usage: sudo ./install.sh
#
# This script installs the HLP platform as systemd services for 24/7 operation.

set -e

echo "=========================================="
echo "Historical Linguistics Platform Installer"
echo "=========================================="

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./install.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"

echo ""
echo "Creating log directory..."
mkdir -p /var/log/hlp
chown ubuntu:ubuntu /var/log/hlp

echo ""
echo "Creating data directory..."
mkdir -p /home/ubuntu/historical-linguistics-platform/data
chown ubuntu:ubuntu /home/ubuntu/historical-linguistics-platform/data

echo ""
echo "Installing Python dependencies..."
cd "$PROJECT_DIR"
pip3 install --upgrade pip
pip3 install fastapi uvicorn pydantic python-multipart requests stanza

echo ""
echo "Installing systemd services..."
cp "$SCRIPT_DIR/hlp-api.service" /etc/systemd/system/
cp "$SCRIPT_DIR/hlp-scheduler.service" /etc/systemd/system/

echo ""
echo "Reloading systemd..."
systemctl daemon-reload

echo ""
echo "Enabling services..."
systemctl enable hlp-api.service
systemctl enable hlp-scheduler.service

echo ""
echo "Starting services..."
systemctl start hlp-api.service
sleep 5
systemctl start hlp-scheduler.service

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Services status:"
systemctl status hlp-api.service --no-pager || true
echo ""
systemctl status hlp-scheduler.service --no-pager || true
echo ""
echo "Commands:"
echo "  View API logs:       sudo journalctl -u hlp-api -f"
echo "  View scheduler logs: sudo journalctl -u hlp-scheduler -f"
echo "  Restart API:         sudo systemctl restart hlp-api"
echo "  Restart scheduler:   sudo systemctl restart hlp-scheduler"
echo "  Stop all:            sudo systemctl stop hlp-api hlp-scheduler"
echo ""
echo "API endpoint: http://localhost:8000"
echo "API docs:     http://localhost:8000/docs"
echo ""
