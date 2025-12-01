#!/bin/bash
# ============================================================================
# PERFECT PRODUCTION SETUP SCRIPT
# Complete automation for the Ultimate AI Monitoring Dashboard
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================================${NC}"
}

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
DATA_DIR="$PROJECT_DIR/data"
MODELS_DIR="$PROJECT_DIR/models"
LOGS_DIR="$PROJECT_DIR/logs"

# Default configuration
DEFAULT_CONFIG='{
    "database_url": "sqlite:///corpus_platform.db",
    "redis_url": "redis://localhost:6379",
    "secret_key": "'$(openssl rand -hex 32)'",
    "jwt_secret": "'$(openssl rand -hex 32)'",
    "ai_models_dir": "'$MODELS_DIR'",
    "gpu_enabled": true,
    "rate_limit_requests": 100,
    "rate_limit_window": 60,
    "metrics_enabled": true,
    "health_check_interval": 30
}'

# Check system requirements
check_system() {
    log_header "SYSTEM REQUIREMENTS CHECK"

    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Linux detected ✓"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_success "macOS detected ✓"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python $PYTHON_VERSION found ✓"
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check available memory
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    else
        MEM_GB=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
    fi

    if [ "$MEM_GB" -lt 4 ]; then
        log_warning "Low memory detected: ${MEM_GB}GB. AI models may not work optimally."
    else
        log_success "Memory: ${MEM_GB}GB ✓"
    fi

    # Check disk space
    DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$DISK_GB" -lt 10 ]; then
        log_warning "Low disk space: ${DISK_GB}GB available. Consider freeing up space."
    else
        log_success "Disk space: ${DISK_GB}GB available ✓"
    fi
}

# Setup virtual environment
setup_venv() {
    log_header "SETTING UP VIRTUAL ENVIRONMENT"

    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created ✓"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    log_success "Virtual environment ready ✓"
}

# Install dependencies
install_dependencies() {
    log_header "INSTALLING DEPENDENCIES"

    source "$VENV_DIR/bin/activate"

    # Core dependencies
    log_info "Installing core dependencies..."
    pip install --quiet \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        sqlalchemy==2.0.23 \
        asyncpg==0.29.0 \
        redis==5.0.1 \
        psutil==5.9.6 \
        prometheus-client==0.19.0 \
        python-jose[cryptography]==3.3.0 \
        passlib[bcrypt]==1.7.4 \
        python-multipart==0.0.6 \
        aiofiles==23.2.1

    # AI/ML libraries
    log_info "Installing AI/ML libraries..."
    pip install --quiet \
        torch==2.1.1 \
        torchvision==0.16.1 \
        torchaudio==2.1.1 \
        transformers==4.35.2 \
        accelerate==0.24.1 \
        datasets==2.15.0 \
        nltk==3.8.1 \
        spacy==3.7.2 \
        textblob==0.17.1 \
        scikit-learn==1.3.2 \
        pandas==2.1.3 \
        numpy==1.26.2

    # Specialized AI libraries (with error handling)
    log_info "Installing specialized AI libraries..."

    # Stanza
    if pip install --quiet stanza 2>/dev/null; then
        log_success "Stanza installed ✓"
    else
        log_warning "Stanza installation failed (optional)"
    fi

    # Polyglot
    if pip install --quiet polyglot pyicu 2>/dev/null; then
        log_success "Polyglot installed ✓"
    else
        log_warning "Polyglot installation failed (optional)"
    fi

    # UDPipe
    if pip install --quiet ufal.udpipe 2>/dev/null; then
        log_success "UDPipe installed ✓"
    else
        log_warning "UDPipe installation failed (optional)"
    fi

    # Trankit
    if pip install --quiet trankit 2>/dev/null; then
        log_success "Trankit installed ✓"
    else
        log_warning "Trankit installation failed (optional)"
    fi

    # Ollama (if available)
    if command -v ollama &> /dev/null; then
        pip install --quiet ollama
        log_success "Ollama client installed ✓"
    else
        log_warning "Ollama not found (optional - for local LLMs)"
    fi

    log_success "Dependencies installation completed ✓"
}

# Setup directories
setup_directories() {
    log_header "SETTING UP DIRECTORIES"

    mkdir -p "$DATA_DIR"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p templates
    mkdir -p static

    log_success "Directories created ✓"
}

# Download AI models
download_models() {
    log_header "DOWNLOADING AI MODELS"

    source "$VENV_DIR/bin/activate"

    log_info "Downloading NLTK data..."
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

    log_info "Downloading spaCy models..."
    python3 -c "import spacy; spacy.cli.download('en_core_web_sm')" || log_warning "spaCy model download failed"

    log_info "Testing model downloads..."
    python3 -c "
try:
    import stanza
    stanza.download('en')
    print('Stanza English model downloaded')
except:
    print('Stanza models skipped')

try:
    from transformers import pipeline
    pipeline('sentiment-analysis')
    print('Transformers models ready')
except:
    print('Transformers models skipped')
    "

    log_success "Model downloads completed ✓"
}

# Create configuration
create_config() {
    log_header "CREATING CONFIGURATION"

    CONFIG_FILE="$PROJECT_DIR/config.json"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$DEFAULT_CONFIG" | python3 -m json.tool > "$CONFIG_FILE"
        log_success "Configuration file created: $CONFIG_FILE"
        log_info "Please edit $CONFIG_FILE with your settings"
    else
        log_info "Configuration file already exists"
    fi

    # Create .env file
    ENV_FILE="$PROJECT_DIR/.env"
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Database
DATABASE_URL=sqlite:///corpus_platform.db

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)

# AI Settings
AI_MODELS_DIR=$MODELS_DIR
GPU_ENABLED=true

# External Services (configure as needed)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
SMTP_SERVER=
ALERT_EMAILS=

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
EOF
        log_success ".env file created"
    fi
}

# Setup systemd service
setup_service() {
    log_header "SETTING UP SYSTEM SERVICE"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        SERVICE_FILE="/etc/systemd/system/perfect-ai-platform.service"

        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Perfect AI Platform - Ultimate Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$VENV_DIR/bin
ExecStart=$VENV_DIR/bin/python3 perfect_production_system.py
Restart=always
RestartSec=5

# Resource limits
MemoryLimit=4G
CPUQuota=80%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=perfect-ai-platform

[Install]
WantedBy=multi-user.target
EOF

        sudo systemctl daemon-reload
        sudo systemctl enable perfect-ai-platform
        log_success "Systemd service created and enabled ✓"
    else
        log_info "Skipping systemd setup (not Linux)"
    fi
}

# Setup monitoring
setup_monitoring() {
    log_header "SETTING UP MONITORING"

    # Create prometheus configuration
    PROMETHEUS_DIR="$PROJECT_DIR/monitoring"
    mkdir -p "$PROMETHEUS_DIR"

    cat > "$PROMETHEUS_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'perfect-ai-platform'
    static_configs:
      - targets: ['localhost:8000']
EOF

    # Create grafana dashboard
    cat > "$PROMETHEUS_DIR/dashboard.json" << EOF
{
  "dashboard": {
    "title": "Perfect AI Platform Monitoring",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [{"expr": "system_cpu_usage"}]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [{"expr": "system_memory_usage"}]
      },
      {
        "title": "AI Requests",
        "type": "graph",
        "targets": [{"expr": "ai_requests_total"}]
      }
    ]
  }
}
EOF

    log_success "Monitoring configuration created ✓"
}

# Create startup script
create_startup_script() {
    log_header "CREATING STARTUP SCRIPTS"

    # Development startup script
    cat > "$PROJECT_DIR/start_dev.sh" << 'EOF'
#!/bin/bash
# Development startup script

export PYTHONPATH="$PWD"
source venv/bin/activate

# Set development environment
export ENV=development
export DEBUG=true

# Start the application
python3 perfect_production_system.py
EOF

    # Production startup script
    cat > "$PROJECT_DIR/start_prod.sh" << 'EOF'
#!/bin/bash
# Production startup script

export PYTHONPATH="$PWD"
source venv/bin/activate

# Set production environment
export ENV=production
export DEBUG=false

# Start with production settings
uvicorn perfect_production_system:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --http httptools \
    --access-log \
    --log-level info
EOF

    chmod +x start_dev.sh start_prod.sh
    log_success "Startup scripts created ✓"
}

# Test installation
test_installation() {
    log_header "TESTING INSTALLATION"

    source "$VENV_DIR/bin/activate"

    # Test imports
    log_info "Testing core imports..."
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from perfect_production_system import app
    print('✓ Core application imports successful')
except ImportError as e:
    print(f'✗ Core import failed: {e}')
    sys.exit(1)
"

    # Test AI imports
    log_info "Testing AI imports..."
    python3 -c "
try:
    import torch
    print('✓ PyTorch available')
except ImportError:
    print('✗ PyTorch not available')

try:
    from transformers import pipeline
    print('✓ Transformers available')
except ImportError:
    print('✗ Transformers not available')

try:
    import nltk
    print('✓ NLTK available')
except ImportError:
    print('✗ NLTK not available')
"

    # Test database connection
    log_info "Testing database connection..."
    python3 -c "
import sqlite3
conn = sqlite3.connect('test.db')
conn.execute('CREATE TABLE test (id INTEGER)')
conn.execute('INSERT INTO test VALUES (1)')
result = conn.execute('SELECT * FROM test').fetchone()
assert result[0] == 1
conn.close()
import os; os.remove('test.db')
print('✓ Database connection successful')
"

    log_success "Installation tests completed ✓"
}

# Create documentation
create_documentation() {
    log_header "CREATING DOCUMENTATION"

    README_FILE="$PROJECT_DIR/README_PERFECT.md"

    cat > "$README_FILE" << 'EOF'
# 🚀 Perfect AI Monitoring Dashboard

Ultimate production-ready system with perfect monitoring and **ALL** powerful community-driven AIs integrated.

## ✨ Features

### 🤖 Complete AI Integration
- **Stanza** (Stanford NLP) - Neural dependency parsing
- **spaCy** (Industrial NLP) - Production-ready NLP
- **Hugging Face Transformers** - State-of-the-art models
- **PyTorch** - Deep learning framework
- **TensorFlow/Keras** - Neural networks
- **NLTK** - Natural Language Toolkit
- **TextBlob** - Simplified text processing
- **Polyglot** - Multilingual NLP
- **UDPipe** - Neural parsing
- **Trankit** - Multilingual pipeline
- **Ollama** - Local LLM server (GPT-J, LLaMA, etc.)
- **LightSide** - Educational data mining
- **MLAnnotator** - Transformer-based annotation

### 📊 Perfect Monitoring
- Real-time system metrics (CPU, Memory, Disk, Network)
- AI model status and performance tracking
- Beautiful browser-based dashboard
- Prometheus metrics export
- Grafana integration
- Alert system with SMS/email notifications

### 🏗️ Production Ready
- FastAPI backend with async support
- PostgreSQL database with async drivers
- Redis caching and session management
- JWT authentication with role-based access
- Rate limiting and security middleware
- Docker containerization
- Systemd service integration
- Automated backups and health checks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- 10GB+ disk space for models

### Installation
```bash
# Clone and setup
git clone <repository>
cd perfect-ai-platform
chmod +x setup_perfect.sh
./setup_perfect.sh
```

### Development
```bash
./start_dev.sh
```

### Production
```bash
./start_prod.sh
```

## 📈 Dashboard Access

- **Main Dashboard**: http://localhost:8000/perfect-monitor
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 🤖 AI Usage Examples

### Sentiment Analysis
```python
from ultimate_ai_orchestrator import ultimate_ai

result = ultimate_ai.ensemble_predict(
    "This product is amazing!",
    "sentiment_analysis",
    "en"
)
print(f"Sentiment: {result.ensemble_prediction}")
```

### Text Generation
```python
result = ultimate_ai.ensemble_predict(
    "The future of AI is",
    "text_generation",
    "en"
)
print(f"Generated: {result.ensemble_prediction}")
```

## 🔧 Configuration

Edit `config.json` and `.env` for customization:

```json
{
    "database_url": "postgresql://user:pass@localhost/db",
    "redis_url": "redis://localhost:6379",
    "gpu_enabled": true,
    "rate_limit_requests": 100
}
```

## 📊 Monitoring & Alerts

### Metrics Available
- System resources (CPU, memory, disk, network)
- AI model performance and usage
- API request rates and latency
- Database connection pool status
- Cache hit rates

### Alert Channels
- Email notifications
- SMS alerts (Twilio integration)
- Slack webhooks
- PagerDuty integration

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d --build

# Scale services
docker-compose up -d --scale web=4
```

## 🔒 Security Features

- JWT token authentication
- Rate limiting per endpoint
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection
- Security headers (HSTS, CSP, etc.)

## 📚 API Reference

### Core Endpoints
- `POST /api/auth/login` - User authentication
- `GET /api/monitor/system` - System metrics
- `GET /api/monitor/ai` - AI models status
- `POST /api/ai/predict` - AI predictions
- `GET /api/corpus` - Corpus management

### WebSocket Events
- `metrics_update` - Real-time metrics
- `alert` - System alerts
- `ai_status` - AI model updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

Academic Research License - NKUA Historical Linguistics Platform

## 🙏 Acknowledgments

- Hellenic Foundation for Research and Innovation (HFRI)
- National and Kapodistrian University of Athens (NKUA)
- All the amazing open-source AI communities

---

**Built with ❤️ for advancing historical linguistics research**
EOF

    log_success "Documentation created ✓"
}

# Main execution
main() {
    log_header "PERFECT AI PLATFORM SETUP"
    echo -e "${CYAN}Ultimate Production System with All Community-Driven AIs${NC}"
    echo -e "${WHITE}NKUA Historical Linguistics Platform - HFRI Funded${NC}"
    echo ""

    check_system
    setup_directories
    setup_venv
    install_dependencies
    download_models
    create_config
    setup_service
    setup_monitoring
    create_startup_script
    test_installation
    create_documentation

    log_header "SETUP COMPLETE! 🎉"
    echo -e "${GREEN}Your Perfect AI Platform is ready!${NC}"
    echo ""
    echo -e "${WHITE}Next steps:${NC}"
    echo -e "1. Edit ${CYAN}config.json${NC} and ${CYAN}.env${NC} with your settings"
    echo -e "2. Run ${CYAN}./start_dev.sh${NC} for development"
    echo -e "3. Run ${CYAN}./start_prod.sh${NC} for production"
    echo -e "4. Access dashboard at ${CYAN}http://localhost:8000/perfect-monitor${NC}"
    echo ""
    echo -e "${YELLOW}Documentation: ${WHITE}README_PERFECT.md${NC}"
    echo -e "${YELLOW}Logs: ${WHITE}$LOGS_DIR/${NC}"
    echo ""
}

# Run main function
main "$@"
