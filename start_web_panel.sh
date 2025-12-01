#!/bin/bash
# Start the Windsurf Web Control Panel
# This runs the Flask app with proper error handling

cd ~/corpus_platform

echo "========================================================================"
echo "Starting Windsurf Web Control Panel"
echo "========================================================================"
echo ""
echo "Access at: http://135.125.216.3:8000"
echo "Press Ctrl+C to stop"
echo ""
echo "========================================================================"

python3 windsurf_web_panel.py
