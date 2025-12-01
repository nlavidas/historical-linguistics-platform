#!/bin/bash
# Corpus Platform Production Setup Script
# Run this on the OVH VM as ubuntu user

set -e  # Exit on any error

echo "=== Corpus Platform Setup Starting ==="

# 1. Install dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade twilio psutil prometheus_client logstash_async

# 2. Create .env file if not exists
ENV_FILE="$HOME/corpus_platform/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file template..."
    cat > "$ENV_FILE" << 'EOF'
# Twilio Configuration
TWILIO_ACCOUNT_SID=""
TWILIO_AUTH_TOKEN=""
TWILIO_FROM_NUMBER=""

# Alert Configuration  
SMTP_SERVER=""
ALERT_EMAILS=""

# Other
TWILIO_TO_NUMBER="+306948066777"
EOF
    echo "Please edit $ENV_FILE with your actual values."
fi

# 3. Deploy systemd service
SERVICE_FILE="$HOME/corpus_platform/corpus_platform.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Deploying systemd service..."
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/"
sudo systemctl daemon-reload

# 4. Start and enable service
echo "Starting corpus platform service..."
sudo systemctl start corpus_platform
sudo systemctl enable corpus_platform

# 5. Wait and check status
sleep 5
sudo systemctl status corpus_platform --no-pager -l

# 6. Test basic endpoints
echo "Testing endpoints..."
curl -s -o /dev/null -w "Login: %{http_code}\n" http://localhost:5000/login
curl -s -o /dev/null -w "Debug: %{http_code}\n" http://localhost:5000/debug/status

echo "=== Setup Complete ==="
echo "Access: http://135.125.216.3"
echo "Edit $ENV_FILE for Twilio/SMTP"
echo "Run tests: python3 ~/corpus_platform/test_panel.py"
