#!/bin/bash
# Autonomous VM Setup Agent
# This script runs on the VM and sets up EVERYTHING automatically
# No user interaction required!

set -e

LOG_FILE=~/corpus_platform/autonomous_vm_setup.log
STATUS_FILE=~/corpus_platform/vm_setup_status.json

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

update_status() {
    echo "{\"phase\": \"$1\", \"timestamp\": \"$(date -Iseconds)\"}" > "$STATUS_FILE"
}

log "========================================================================"
log "AUTONOMOUS VM SETUP AGENT STARTED"
log "========================================================================"

cd ~/corpus_platform

# PHASE 1: System Dependencies
log ""
log "PHASE 1: Installing System Dependencies"
update_status "phase1_system_deps"

sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    build-essential \
    nginx \
    sqlite3 \
    htop \
    tmux

log "✓ System dependencies installed"

# PHASE 2: Python Dependencies
log ""
log "PHASE 2: Installing Python Dependencies"
update_status "phase2_python_deps"

pip3 install --upgrade pip

# Install in batches to avoid timeout
pip3 install flask flask-socketio python-socketio eventlet pytz
pip3 install scikit-learn numpy pandas
pip3 install beautifulsoup4 lxml requests
pip3 install stanza transformers torch
pip3 install sentence-transformers chromadb
pip3 install langchain langchain-community
pip3 install crewai crewai-tools

log "✓ Python dependencies installed"

# PHASE 3: Install Ollama (Open-Source LLM Engine)
log ""
log "PHASE 3: Installing Ollama"
update_status "phase3_ollama"

if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    log "✓ Ollama installed"
else
    log "✓ Ollama already installed"
fi

# Start Ollama service
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

# Pull models
log "Pulling Qwen2.5-Coder model..."
ollama pull qwen2.5-coder:7b

log "Pulling embedding model..."
ollama pull nomic-embed-text

log "✓ Ollama models ready"

# PHASE 4: Install VS Code Server
log ""
log "PHASE 4: Installing VS Code Server"
update_status "phase4_vscode"

if ! command -v code-server &> /dev/null; then
    curl -fsSL https://code-server.dev/install.sh | sh
    log "✓ VS Code Server installed"
else
    log "✓ VS Code Server already installed"
fi

# Configure VS Code Server
mkdir -p ~/.config/code-server
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:8080
auth: password
password: historical_linguistics_2025
cert: false
EOF

log "✓ VS Code Server configured"

# PHASE 5: Download Stanza Models
log ""
log "PHASE 5: Downloading Stanza Models"
update_status "phase5_stanza"

python3 << PYEOF
import stanza
try:
    stanza.download('la')
    print("✓ Latin model downloaded")
except:
    print("⚠ Latin model download failed")

try:
    stanza.download('grc')
    print("✓ Ancient Greek model downloaded")
except:
    print("⚠ Greek model download failed")

try:
    stanza.download('en')
    print("✓ English model downloaded")
except:
    print("⚠ English model download failed")
PYEOF

log "✓ Stanza models downloaded"

# PHASE 6: Install CLTK
log ""
log "PHASE 6: Installing CLTK"
update_status "phase6_cltk"

pip3 install cltk

python3 << PYEOF
from cltk.data.fetch import FetchCorpus
try:
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.import_corpus("lat_models_cltk")
    print("✓ CLTK Latin models downloaded")
except:
    print("⚠ CLTK download skipped (optional)")
PYEOF

log "✓ CLTK installed"

# PHASE 7: Initialize ChromaDB
log ""
log "PHASE 7: Initializing Vector Database"
update_status "phase7_chromadb"

mkdir -p ~/corpus_platform/chroma_db

python3 << PYEOF
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="historical_corpus",
    metadata={"description": "Diachronic multilingual corpus"}
)
print(f"✓ ChromaDB initialized: {collection.count()} documents")
PYEOF

log "✓ Vector database initialized"

# PHASE 8: Setup Nginx Reverse Proxy
log ""
log "PHASE 8: Configuring Nginx"
update_status "phase8_nginx"

sudo tee /etc/nginx/sites-available/corpus-platform > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    # Web Control Panel
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # VS Code Server
    location /ide/ {
        proxy_pass http://localhost:8080/;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    # Ollama API
    location /api/ollama/ {
        proxy_pass http://localhost:11434/;
        proxy_set_header Host $host;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/corpus-platform /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

log "✓ Nginx configured"

# PHASE 9: Create Systemd Services
log ""
log "PHASE 9: Creating Systemd Services"
update_status "phase9_systemd"

# Ollama service
sudo tee /etc/systemd/system/ollama.service > /dev/null << EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# VS Code Server service
sudo tee /etc/systemd/system/code-server.service > /dev/null << EOF
[Unit]
Description=VS Code Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/corpus_platform
ExecStart=/usr/bin/code-server --bind-addr 0.0.0.0:8080
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Web Panel service
sudo tee /etc/systemd/system/web-panel.service > /dev/null << EOF
[Unit]
Description=Windsurf Web Control Panel
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/corpus_platform
ExecStart=/usr/bin/python3 windsurf_web_panel.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama code-server web-panel
sudo systemctl start ollama code-server web-panel

log "✓ Systemd services created and started"

# PHASE 10: Create Quick Start Scripts
log ""
log "PHASE 10: Creating Quick Start Scripts"
update_status "phase10_scripts"

# Status checker
cat > ~/corpus_platform/check_status.sh << 'EOF'
#!/bin/bash
echo "========================================================================"
echo "Platform Status"
echo "========================================================================"
echo ""
echo "Services:"
systemctl status ollama --no-pager | grep Active
systemctl status code-server --no-pager | grep Active
systemctl status web-panel --no-pager | grep Active
systemctl status nginx --no-pager | grep Active
echo ""
echo "Ports:"
netstat -tuln | grep -E ':(8000|8080|11434|80) '
echo ""
echo "Access Points:"
echo "  - Web Control Panel: http://$(hostname -I | awk '{print $1}')"
echo "  - VS Code IDE:       http://$(hostname -I | awk '{print $1}')/ide/"
echo "  - Password:          historical_linguistics_2025"
echo ""
echo "========================================================================"
EOF
chmod +x ~/corpus_platform/check_status.sh

# Restart all
cat > ~/corpus_platform/restart_all.sh << 'EOF'
#!/bin/bash
sudo systemctl restart ollama code-server web-panel nginx
echo "✓ All services restarted"
EOF
chmod +x ~/corpus_platform/restart_all.sh

# Stop all
cat > ~/corpus_platform/stop_all.sh << 'EOF'
#!/bin/bash
sudo systemctl stop ollama code-server web-panel
echo "✓ All services stopped"
EOF
chmod +x ~/corpus_platform/stop_all.sh

log "✓ Quick start scripts created"

# PHASE 11: Final Setup
log ""
log "PHASE 11: Final Configuration"
update_status "phase11_final"

# Create directories
mkdir -p ~/corpus_platform/research_exports/{visual_reports,agent_reports,evaluation,valency_analysis,night_reports}
mkdir -p ~/corpus_platform/research_exports/incoming

# Set permissions
chmod +x ~/corpus_platform/*.sh
chmod +x ~/corpus_platform/*.py

log "✓ Final configuration complete"

# COMPLETE
log ""
log "========================================================================"
log "AUTONOMOUS SETUP COMPLETE!"
log "========================================================================"
log ""
log "Platform is now running at:"
log "  - Web Control Panel: http://$(hostname -I | awk '{print $1}')"
log "  - VS Code IDE:       http://$(hostname -I | awk '{print $1}')/ide/"
log "  - Password:          historical_linguistics_2025"
log ""
log "Quick Commands:"
log "  - Check status:  ./check_status.sh"
log "  - Restart all:   ./restart_all.sh"
log "  - Stop all:      ./stop_all.sh"
log ""
log "All services are running as systemd daemons (auto-start on boot)"
log "========================================================================"

update_status "complete"

# Show status
./check_status.sh
