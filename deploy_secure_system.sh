#!/bin/bash
"""
DEPLOY SECURE CORPUS PLATFORM WITH SMS PROTECTION
==================================================
Complete deployment with enhanced security and mobile notifications.
"""

echo "🔒 DEPLOYING SECURE CORPUS PLATFORM"
echo "====================================="
echo "Mobile security alerts: +30 6948066777"
echo ""

# Check if we're on the VM
if [[ ! -f "corpus_platform.db" ]]; then
    echo "❌ Error: Must run from corpus_platform directory on VM"
    exit 1
fi

echo "📦 Installing security dependencies..."
pip3 install twilio requests psutil flask --quiet

echo ""
echo "🛡️ Setting up SMS/Viber notifications..."
if [[ ! -f "sms_config.json" ]]; then
    echo "SMS configuration not found. Running setup..."
    chmod +x setup_secure_sms.sh
    ./setup_secure_sms.sh
else
    echo "SMS configuration already exists."
fi

echo ""
echo "🔧 Updating systemd services..."
sudo cp secure_monitoring.service /etc/systemd/system/monitoring.service
sudo systemctl daemon-reload

echo ""
echo "🚀 Starting secure services..."

# Stop old services
sudo systemctl stop monitoring 2>/dev/null || true

# Start secure monitoring
sudo systemctl start monitoring
sudo systemctl enable monitoring

echo ""
echo "🌐 Starting secure web panel..."
# Kill any existing web panel
pkill -f "secure_web_panel.py" 2>/dev/null || true
pkill -f "simple_web_panel.py" 2>/dev/null || true

# Start secure web panel
nohup python3 secure_web_panel.py > secure_web_panel.log 2>&1 &
echo $! > secure_web_panel.pid

echo ""
echo "📱 Testing SMS alerts..."
python3 secure_sms_notifier.py start

echo ""
echo "✅ SECURE DEPLOYMENT COMPLETE!"
echo ""
echo "🔐 Security Features Active:"
echo "• Two-factor authentication with SMS second password"
echo "• Real-time monitoring with mobile alerts"
echo "• Secure web control panel with session management"
echo "• Failed login protection and lockout"
echo "• System health monitoring with SMS notifications"
echo ""
echo "📞 Emergency Contact: +30 6948066777"
echo "📊 Web Panel: https://corpus-platform.nlavid.as"
echo "🔑 Primary Password: historical_linguistics_2025"
echo "📱 Second Password: Sent via SMS on login"
echo ""
echo "🧪 Test Commands:"
echo "• SMS test: python3 secure_sms_notifier.py test"
echo "• Password test: python3 secure_sms_notifier.py password"
echo "• Status check: sudo systemctl status monitoring"
echo ""
echo "📋 Security Documentation: SECURITY_DOCUMENTATION.md"
