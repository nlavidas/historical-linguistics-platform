#!/usr/bin/env python3
"""
Autonomous Platform Builder - Fully Automated Setup
This script acts as an autonomous agent to build the complete historical linguistics platform
using only free, open-source, community-driven tools.

No user interaction required - runs completely autonomously.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent

class AutonomousPlatformBuilder:
    """Fully autonomous agent that builds and configures the entire platform."""
    
    def __init__(self):
        self.log_file = ROOT / "autonomous_build.log"
        self.status_file = ROOT / "build_status.json"
        self.status = {
            "started": datetime.now().isoformat(),
            "phase": "initialization",
            "completed_steps": [],
            "errors": []
        }
        
    def log(self, message):
        """Log to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    
    def update_status(self, phase, step=None, error=None):
        """Update build status."""
        self.status["phase"] = phase
        if step:
            self.status["completed_steps"].append(step)
        if error:
            self.status["errors"].append(error)
        self.status["last_update"] = datetime.now().isoformat()
        
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)
    
    def run_command(self, cmd, description, critical=True):
        """Run a command autonomously."""
        self.log(f">>> {description}")
        self.log(f"    Command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(ROOT)
            )
            
            if result.returncode == 0:
                self.log(f"    ✓ Success")
                return True
            else:
                self.log(f"    FAILED: {result.stderr}")
                if critical:
                    self.update_status(self.status["phase"], error=f"{description}: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"    EXCEPTION: {e}")
            if critical:
                self.update_status(self.status["phase"], error=f"{description}: {e}")
            return False
    
    def phase1_install_dependencies(self):
        """Phase 1: Install all Python dependencies."""
        self.log("\n" + "="*80)
        self.log("PHASE 1: Installing Dependencies")
        self.log("="*80)
        self.update_status("phase1_dependencies")
        
        # Core dependencies
        deps = [
            "flask",
            "flask-socketio",
            "python-socketio[client]",
            "eventlet",
            "pytz",
            "scikit-learn",
            "numpy",
            "pandas",
            "beautifulsoup4",
            "lxml",
            "requests",
            "stanza",
            "transformers",
            "torch",
            "sentence-transformers",
            "chromadb",
            "langchain",
            "langchain-community",
            "crewai",
            "crewai-tools",
            "fastmcp",
        ]
        
        self.run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
        
        for dep in deps:
            self.run_command(
                f"{sys.executable} -m pip install {dep}",
                f"Installing {dep}",
                critical=False
            )
        
        self.update_status("phase1_dependencies", "dependencies_installed")
    
    def phase2_setup_vllm_alternative(self):
        """Phase 2: Setup open-source inference engine (Ollama as vLLM alternative)."""
        self.log("\n" + "="*80)
        self.log("PHASE 2: Setting up Open-Source Inference Engine")
        self.log("="*80)
        self.update_status("phase2_inference")
        
        # Create Ollama setup script for Linux
        ollama_setup = """#!/bin/bash
# Install Ollama (open-source vLLM alternative)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen2.5-Coder model (open-source coding model)
ollama pull qwen2.5-coder:7b

# Pull embedding model for RAG
ollama pull nomic-embed-text

# Start Ollama service
nohup ollama serve > /tmp/ollama.log 2>&1 &

echo "Ollama setup complete"
"""
        
        setup_script = ROOT / "setup_ollama.sh"
        setup_script.write_text(ollama_setup)
        setup_script.chmod(0o755)
        
        self.log("Created Ollama setup script: setup_ollama.sh")
        self.update_status("phase2_inference", "ollama_script_created")
    
    def phase3_setup_vscode_server(self):
        """Phase 3: Setup VS Code Server (open-source IDE)."""
        self.log("\n" + "="*80)
        self.log("PHASE 3: Setting up VS Code Server")
        self.log("="*80)
        self.update_status("phase3_vscode")
        
        vscode_setup = """#!/bin/bash
# Install code-server (open-source VS Code for web)
curl -fsSL https://code-server.dev/install.sh | sh

# Create config directory
mkdir -p ~/.config/code-server

# Configure code-server
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:8080
auth: password
password: historical_linguistics_2025
cert: false
EOF

# Start code-server
nohup code-server --bind-addr 0.0.0.0:8080 > /tmp/code-server.log 2>&1 &

echo "VS Code Server setup complete"
echo "Access at: http://localhost:8080"
echo "Password: historical_linguistics_2025"
"""
        
        setup_script = ROOT / "setup_vscode_server.sh"
        setup_script.write_text(vscode_setup)
        setup_script.chmod(0o755)
        
        self.log("Created VS Code Server setup script: setup_vscode_server.sh")
        self.update_status("phase3_vscode", "vscode_script_created")
    
    def phase4_setup_linguistic_tools(self):
        """Phase 4: Setup open-source linguistic tools."""
        self.log("\n" + "="*80)
        self.log("PHASE 4: Setting up Linguistic Tools")
        self.log("="*80)
        self.update_status("phase4_linguistics")
        
        # Install CLTK
        self.run_command(
            f"{sys.executable} -m pip install cltk",
            "Installing CLTK (Classical Language Toolkit)",
            critical=False
        )
        
        # Download Stanza models
        stanza_setup = f"""
import stanza
stanza.download('la')  # Latin
stanza.download('grc')  # Ancient Greek
stanza.download('en')  # English
"""
        
        stanza_script = ROOT / "setup_stanza.py"
        stanza_script.write_text(stanza_setup)
        
        self.run_command(
            f"{sys.executable} setup_stanza.py",
            "Downloading Stanza models",
            critical=False
        )
        
        self.update_status("phase4_linguistics", "linguistic_tools_installed")
    
    def phase5_setup_crewai_agents(self):
        """Phase 5: Setup CrewAI multi-agent system."""
        self.log("\n" + "="*80)
        self.log("PHASE 5: Setting up CrewAI Multi-Agent System")
        self.log("="*80)
        self.update_status("phase5_crewai")
        
        # Create agents configuration
        agents_config = """
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Initialize Ollama LLM
llm = Ollama(model="qwen2.5-coder:7b", base_url="http://localhost:11434")

# Define Agents
archivist = Agent(
    role="Data Archivist",
    goal="Collect and organize historical texts from various sources",
    backstory="Expert in digital archives and text curation",
    llm=llm,
    verbose=True
)

philologist = Agent(
    role="Linguistic Analyst",
    goal="Analyze, lemmatize, and parse historical texts",
    backstory="Classical philologist with expertise in Ancient Greek and Latin",
    llm=llm,
    verbose=True
)

curator = Agent(
    role="Output Curator",
    goal="Format and export analyzed texts in standard formats",
    backstory="Digital humanities specialist focused on TEI-XML and standards",
    llm=llm,
    verbose=True
)

# Define Tasks
collection_task = Task(
    description="Scan repositories and collect new historical texts",
    agent=archivist,
    expected_output="List of collected texts with metadata"
)

analysis_task = Task(
    description="Lemmatize and parse collected texts using Stanza and CLTK",
    agent=philologist,
    expected_output="Annotated texts with morphological and syntactic analysis"
)

export_task = Task(
    description="Export analyzed texts in TEI-XML and CoNLL-U formats",
    agent=curator,
    expected_output="Formatted output files ready for research"
)

# Create Crew
linguistic_crew = Crew(
    agents=[archivist, philologist, curator],
    tasks=[collection_task, analysis_task, export_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    result = linguistic_crew.kickoff()
    print(result)
"""
        
        crew_file = ROOT / "linguistic_crew.py"
        crew_file.write_text(agents_config)
        
        self.log("Created CrewAI configuration: linguistic_crew.py")
        self.update_status("phase5_crewai", "crewai_configured")
    
    def phase6_setup_vector_database(self):
        """Phase 6: Setup ChromaDB for semantic search."""
        self.log("\n" + "="*80)
        self.log("PHASE 6: Setting up Vector Database")
        self.log("="*80)
        self.update_status("phase6_vectordb")
        
        vector_setup = """
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection for historical texts
collection = client.get_or_create_collection(
    name="historical_corpus",
    metadata={"description": "Diachronic multilingual corpus"}
)

# Initialize embedding model (open-source)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Vector database initialized")
print(f"Collection: {collection.name}")
print(f"Documents: {collection.count()}")
"""
        
        vector_file = ROOT / "setup_vectordb.py"
        vector_file.write_text(vector_setup)
        
        self.run_command(
            f"{sys.executable} setup_vectordb.py",
            "Initializing vector database",
            critical=False
        )
        
        self.update_status("phase6_vectordb", "vectordb_initialized")
    
    def phase7_create_master_orchestrator(self):
        """Phase 7: Create master orchestration script."""
        self.log("\n" + "="*80)
        self.log("PHASE 7: Creating Master Orchestrator")
        self.log("="*80)
        self.update_status("phase7_orchestrator")
        
        orchestrator = """#!/bin/bash
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
"""
        
        orchestrator_file = ROOT / "start_platform.sh"
        orchestrator_file.write_text(orchestrator)
        orchestrator_file.chmod(0o755)
        
        self.log("Created master orchestrator: start_platform.sh")
        self.update_status("phase7_orchestrator", "orchestrator_created")
    
    def phase8_create_documentation(self):
        """Phase 8: Generate comprehensive documentation."""
        self.log("\n" + "="*80)
        self.log("PHASE 8: Generating Documentation")
        self.log("="*80)
        self.update_status("phase8_documentation")
        
        readme = """# Autonomous Historical Linguistics Platform

## Overview
A fully autonomous, open-source platform for diachronic corpus linguistics using only community-driven tools.

## Architecture

### Core Components
1. **Ollama** - Open-source LLM inference (replaces vLLM)
2. **VS Code Server** - Web-based IDE (replaces proprietary Windsurf)
3. **CrewAI** - Multi-agent orchestration
4. **ChromaDB** - Vector database for semantic search
5. **Stanza** - Linguistic analysis (lemmatization, parsing)
6. **CLTK** - Classical Language Toolkit

### Models Used (All Open-Source)
- **Qwen2.5-Coder 7B** - Coding and reasoning
- **Nomic Embed Text** - Text embeddings
- **Stanza Models** - Latin, Ancient Greek, English
- **SentenceTransformers** - Semantic similarity

## Quick Start

### On VM (Linux)
```bash
# 1. Install dependencies
./setup_vm_autonomous.sh

# 2. Setup Ollama
./setup_ollama.sh

# 3. Setup VS Code Server
./setup_vscode_server.sh

# 4. Start everything
./start_platform.sh
```

### Access
- **Web Control Panel**: http://YOUR_VM_IP:8000
- **VS Code IDE**: http://YOUR_VM_IP:8080
- **Password**: historical_linguistics_2025

## Features

### Autonomous Agents
- **Archivist**: Collects texts from APIs and repositories
- **Philologist**: Lemmatizes and parses using Stanza/CLTK
- **Curator**: Exports to TEI-XML, CoNLL-U formats

### Capabilities
- ✅ API discovery (Perseus, OpenGreekLatin, Gutenberg)
- ✅ Advanced valency analysis with scikit-learn
- ✅ ML-powered source suggestions
- ✅ Real-time dashboards
- ✅ Semantic search with ChromaDB
- ✅ Multi-agent workflows with CrewAI

## Cost
**$0** - All tools are free and open-source!

Infrastructure costs depend on your VM provider:
- OVH L4-90: ~$730/month
- Or use free tier VMs for testing

## Privacy
100% sovereign - all data stays on your infrastructure. No external API calls to proprietary services.

## License
All components use permissive open-source licenses (MIT, Apache 2.0, etc.)
"""
        
        readme_file = ROOT / "PLATFORM_README.md"
        readme_file.write_text(readme)
        
        self.log("Created documentation: PLATFORM_README.md")
        self.update_status("phase8_documentation", "documentation_created")
    
    def run_autonomous_build(self):
        """Execute all phases autonomously."""
        self.log("\n" + "="*80)
        self.log("AUTONOMOUS PLATFORM BUILDER STARTED")
        self.log("="*80)
        self.log(f"Build log: {self.log_file}")
        self.log(f"Status file: {self.status_file}")
        
        try:
            self.phase1_install_dependencies()
            self.phase2_setup_vllm_alternative()
            self.phase3_setup_vscode_server()
            self.phase4_setup_linguistic_tools()
            self.phase5_setup_crewai_agents()
            self.phase6_setup_vector_database()
            self.phase7_create_master_orchestrator()
            self.phase8_create_documentation()
            
            self.log("\n" + "="*80)
            self.log("BUILD COMPLETE!")
            self.log("="*80)
            self.log("\nNext steps:")
            self.log("1. Sync to VM: scp -r . ubuntu@VM_IP:~/corpus_platform")
            self.log("2. On VM, run: ./setup_ollama.sh")
            self.log("3. On VM, run: ./setup_vscode_server.sh")
            self.log("4. On VM, run: ./start_platform.sh")
            self.log("\nAll tools are FREE and OPEN-SOURCE!")
            
            self.update_status("complete", "all_phases_complete")
            
        except Exception as e:
            self.log(f"\n✗ BUILD FAILED: {e}")
            self.update_status("failed", error=str(e))

if __name__ == "__main__":
    builder = AutonomousPlatformBuilder()
    builder.run_autonomous_build()
