#!/bin/bash
# Autonomous VM Setup Script
# This script installs all dependencies and starts the enhanced platform

set -e

echo "========================================================================"
echo "AUTONOMOUS VM SETUP - Diachronic Corpus Platform"
echo "========================================================================"

cd ~/corpus_platform

echo ""
echo ">>> Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install flask flask-socketio pytz scikit-learn numpy pandas beautifulsoup4 lxml requests eventlet

echo ""
echo ">>> Checking database..."
if [ ! -f "corpus_platform.db" ]; then
    echo "Creating database..."
    python3 -c "from DIACHRONIC_MULTILINGUAL_COLLECTOR import DiachronicMultilingualCollector; c = DiachronicMultilingualCollector(); print('Database initialized')"
fi

echo ""
echo ">>> Creating required directories..."
mkdir -p research_exports/{visual_reports,agent_reports,evaluation,valency_analysis,night_reports}
mkdir -p research_exports/incoming

echo ""
echo ">>> Testing new modules..."
python3 -c "import flask, flask_socketio, sklearn; print('✓ All modules imported successfully')"

echo ""
echo "========================================================================"
echo "SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "To start the live web control panel:"
echo "  python3 windsurf_web_panel.py"
echo ""
echo "Then open: http://135.125.216.3:8000"
echo ""
echo "The professional cycle is already running in the background."
echo "========================================================================"
