#!/bin/bash
# =============================================================================
# RUN_NOW.sh - Execute this on the server to start collecting texts
# =============================================================================
#
# Usage:
#   sudo bash RUN_NOW.sh
#
# This script will:
# 1. Pull latest code from GitHub
# 2. Install dependencies
# 3. Run the auto-runner to collect texts
# 4. Set up systemd service for continuous operation
# =============================================================================

set -e

echo "============================================================"
echo "GREEK CORPUS PLATFORM - IMMEDIATE DEPLOYMENT"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo bash RUN_NOW.sh"
    exit 1
fi

# Configuration
PLATFORM_DIR="/root/corpus_platform"
DATA_DIR="/root/corpus_platform/data"
LOG_DIR="/var/log"

echo "1. Creating directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/cache/ud"
mkdir -p "$DATA_DIR/cache/perseus"
mkdir -p "$DATA_DIR/cache/gutenberg"

echo "2. Pulling latest code from GitHub..."
cd "$PLATFORM_DIR"
git pull origin master || echo "Git pull failed, continuing with existing code..."

echo "3. Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install requests beautifulsoup4 lxml pandas streamlit plotly

echo "4. Running initial text collection..."
echo ""
echo "============================================================"
echo "STARTING TEXT COLLECTION - THIS MAY TAKE 10-30 MINUTES"
echo "============================================================"
echo ""

python3 core/auto_runner.py --mode once --data-dir "$DATA_DIR"

echo ""
echo "5. Checking database..."
python3 -c "
import sqlite3
conn = sqlite3.connect('$DATA_DIR/corpus_platform.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM documents')
docs = cursor.fetchone()[0]
cursor.execute('SELECT SUM(sentence_count) FROM documents')
sents = cursor.fetchone()[0] or 0
cursor.execute('SELECT SUM(token_count) FROM documents')
toks = cursor.fetchone()[0] or 0
print(f'Documents: {docs}')
print(f'Sentences: {sents:,}')
print(f'Tokens: {toks:,}')
conn.close()
"

echo ""
echo "6. Setting up systemd service for continuous collection..."

cat > /etc/systemd/system/corpus-collector.service << 'EOF'
[Unit]
Description=Greek Corpus Text Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/corpus_platform
ExecStart=/usr/bin/python3 core/auto_runner.py --mode continuous --data-dir /root/corpus_platform/data --interval 6
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable corpus-collector
systemctl start corpus-collector

echo ""
echo "7. Restarting Streamlit platform..."
systemctl restart greek-corpus || echo "greek-corpus service not found, skipping..."

echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Services running:"
echo "  - corpus-collector: Continuous text collection (every 6 hours)"
echo "  - greek-corpus: Streamlit web interface"
echo ""
echo "Check status:"
echo "  systemctl status corpus-collector"
echo "  systemctl status greek-corpus"
echo ""
echo "View logs:"
echo "  tail -f /var/log/corpus_auto_runner.log"
echo ""
echo "Access platform:"
echo "  http://54.37.228.155:8501"
echo ""
echo "============================================================"
