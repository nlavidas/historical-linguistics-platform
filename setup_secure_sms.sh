#!/bin/bash
"""
SECURE SMS CONFIGURATION SETUP
===============================
Configure Twilio/Viber API keys for SMS notifications.
Mobile: +30 6948066777
"""

echo "🔒 SECURE SMS/VIBER CONFIGURATION SETUP"
echo "=========================================="
echo "Mobile notifications will be sent to: +30 6948066777"
echo ""

# Create config directory if it doesn't exist
mkdir -p ~/.config/corpus_platform

CONFIG_FILE="sms_config.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration file already exists. Do you want to update it? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

echo ""
echo "Choose SMS service:"
echo "1) Twilio (recommended for SMS)"
echo "2) Viber Business Messages"
echo "3) Both"
read -r service_choice

case $service_choice in
    1)
        enabled_services='["sms"]'
        ;;
    2)
        enabled_services='["viber"]'
        ;;
    3)
        enabled_services='["sms", "viber"]'
        ;;
    *)
        echo "Invalid choice. Using SMS only."
        enabled_services='["sms"]'
        ;;
esac

echo ""
echo "📱 CONFIGURING SMS SERVICE"
echo "You need to sign up for Twilio at: https://www.twilio.com/"
echo "Get your Account SID, Auth Token, and a phone number."
echo ""

if [[ "$enabled_services" == *'"sms"'* ]]; then
    echo "Enter your Twilio Account SID:"
    read -r twilio_sid

    echo "Enter your Twilio Auth Token:"
    read -rs twilio_token
    echo ""

    echo "Enter your Twilio phone number (e.g., +1234567890):"
    read -r twilio_from
fi

echo ""
echo "📲 CONFIGURING VIBER SERVICE (Optional)"
echo "For Viber, you need a Viber Business account."
echo "Get your API token from: https://developers.viber.com/"
echo ""

if [[ "$enabled_services" == *'"viber"'* ]]; then
    echo "Enter your Viber API Token (leave empty to skip):"
    read -r viber_token
fi

# Create configuration JSON
cat > "$CONFIG_FILE" << EOF
{
    "twilio_sid": "$twilio_sid",
    "twilio_token": "$twilio_token",
    "twilio_from": "$twilio_from",
    "viber_token": "$viber_token",
    "enabled_services": $enabled_services,
    "alert_types": {
        "platform_start": true,
        "platform_stop": true,
        "security_alert": true,
        "system_error": true,
        "daily_report": false
    }
}
EOF

echo ""
echo "✅ Configuration saved to $CONFIG_FILE"
echo ""
echo "🔐 SECURITY FEATURES ENABLED:"
echo "• Two-factor authentication with second password"
echo "• SMS alerts for platform start/stop"
echo "• Security monitoring with mobile notifications"
echo "• Session management with automatic logout"
echo "• Failed login attempt detection and lockout"
echo ""

# Install required Python packages
echo "📦 Installing required packages..."
pip3 install twilio requests psutil 2>/dev/null || echo "Some packages may need manual installation"

echo ""
echo "🚀 SETUP COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Test SMS: python3 secure_sms_notifier.py test"
echo "2. Start secure web panel: python3 secure_web_panel.py"
echo "3. Update systemd service: sudo systemctl daemon-reload"
echo ""

# Test the configuration
echo "🧪 Testing SMS configuration..."
python3 -c "
import sys
sys.path.append('.')
try:
    from secure_sms_notifier import SecureSMSNotifier
    notifier = SecureSMSNotifier()
    print('✅ SMS notifier loaded successfully')
    print(f'📱 Mobile number: {notifier.mobile_number}')
    print(f'📤 Enabled services: {notifier.config.get(\"enabled_services\", [])}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
