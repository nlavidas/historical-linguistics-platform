#!/bin/bash
# Master Orchestrator - Starts all platform components

echo "========================================================================"
echo "Starting Autonomous Historical Linguistics Platform"
echo "========================================================================"

# 1. Start Ollama (inference engine)
echo "Starting Ollama..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# 2. Start VS Code Server
echo "Starting VS Code Server..."
nohup code-server --bind-addr 0.0.0.0:8080 > /tmp/code-server.log 2>&1 &
sleep 3

# 3. Start Flask Web Control Panel
echo "Starting Web Control Panel..."
cd ~/corpus_platform
nohup python3 windsurf_web_panel.py > /tmp/web_panel.log 2>&1 &
sleep 3

# 4. Start Professional Dashboard (if exists)
if [ -f "professional_dashboard.py" ]; then
    echo "Starting Professional Dashboard..."
    nohup python3 professional_dashboard.py > /tmp/dashboard.log 2>&1 &
fi

echo ""
echo "========================================================================"
echo "Platform Started Successfully!"
echo "========================================================================"
echo ""
echo "Access Points:"
echo "  - Web Control Panel: http://localhost:8000"
echo "  - VS Code Server:    http://localhost:8080 (password: historical_linguistics_2025)"
echo "  - Ollama API:        http://localhost:11434"
echo ""
echo "Logs:"
echo "  - Ollama:       /tmp/ollama.log"
echo "  - VS Code:      /tmp/code-server.log"
echo "  - Web Panel:    /tmp/web_panel.log"
echo ""
echo "========================================================================"
