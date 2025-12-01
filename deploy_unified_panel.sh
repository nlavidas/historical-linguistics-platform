#!/bin/bash
# DEPLOY UNIFIED WEB PANEL SCRIPT
# Run this on the VM to deploy the new unified interface

echo "🚀 Deploying Unified Web Panel..."

# Stop current web panel
echo "Stopping current web panel..."
pkill -f secure_clean
pkill -f secure_web
sleep 3

# Start unified panel
echo "Starting unified web panel..."
cd ~/corpus_platform
python3 unified_web_panel.py &
sleep 5

# Test functionality
echo "Testing web panel..."
if curl -s http://localhost/login | grep -q "Secure Corpus Platform"; then
    echo "✅ Login page working"
else
    echo "❌ Login page failed"
fi

if curl -s -c cookies.txt -X POST -d 'password=historical_linguistics_2025' http://localhost/login > /dev/null && curl -s -b cookies.txt http://localhost/ | grep -q "Historical Linguistics"; then
    echo "✅ Dashboard accessible"
else
    echo "❌ Dashboard failed"
fi

echo "🎉 Unified web panel deployed!"
echo "🌐 Access: http://135.125.216.3"
echo "🔑 Password: historical_linguistics_2025"
