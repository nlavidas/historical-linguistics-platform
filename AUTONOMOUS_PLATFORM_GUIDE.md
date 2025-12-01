# 🤖 Autonomous Historical Linguistics Platform
## Complete Setup Guide for Z: Drive

---

## 🎯 What This Platform Does

A **fully autonomous, open-source** platform that:
- ✅ Collects historical texts from APIs (Perseus, OpenGreekLatin, Gutenberg)
- ✅ Lemmatizes & parses using Stanza and CLTK
- ✅ Performs advanced valency analysis with scikit-learn
- ✅ Provides ML-powered source suggestions
- ✅ Runs multi-agent workflows with CrewAI
- ✅ Offers web-based IDE (VS Code Server)
- ✅ Semantic search with ChromaDB
- ✅ Real-time dashboards

**100% Free & Open-Source Tools Only!**

---

## 🚀 One-Click Deployment

### From Windows (Z: Drive)

**Option 1: Automatic (Recommended)**
```batch
Z:\corpus_platform\AUTO_DEPLOY.bat
```

This will:
1. Sync all files from Z: to VM
2. Run autonomous setup on VM
3. Configure all services
4. Start everything automatically

**Option 2: Manual**
```powershell
# Sync to VM
scp -i "$env:USERPROFILE\.ssh\id_rsa_ovh" -r Z:\corpus_platform\* ubuntu@135.125.216.3:~/corpus_platform/

# SSH to VM
ssh -i "$env:USERPROFILE\.ssh\id_rsa_ovh" ubuntu@135.125.216.3

# Run autonomous setup
cd ~/corpus_platform
chmod +x autonomous_vm_setup.sh
./autonomous_vm_setup.sh
```

---

## 📊 What Gets Installed

### System Components
- **Ollama** - Open-source LLM inference engine (replaces vLLM)
- **VS Code Server** - Web-based IDE (replaces Windsurf)
- **Nginx** - Reverse proxy for all services
- **ChromaDB** - Vector database for semantic search
- **Systemd Services** - Auto-start on boot

### Python Libraries
- **Flask + SocketIO** - Web control panel
- **CrewAI** - Multi-agent orchestration
- **Stanza** - Linguistic analysis (lemmatization, parsing)
- **CLTK** - Classical Language Toolkit
- **scikit-learn** - ML for valency analysis
- **sentence-transformers** - Embeddings

### AI Models (All Open-Source)
- **Qwen2.5-Coder 7B** - Coding and reasoning
- **Nomic Embed Text** - Text embeddings
- **Stanza Models** - Latin, Ancient Greek, English

---

## 🌐 Access Points

After setup completes:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Web Control Panel** | `http://135.125.216.3` | None |
| **VS Code IDE** | `http://135.125.216.3/ide/` | Password: `historical_linguistics_2025` |
| **Ollama API** | `http://135.125.216.3/api/ollama/` | None |

---

## 🛠️ Quick Commands (On VM)

```bash
# Check status of all services
./check_status.sh

# Restart all services
./restart_all.sh

# Stop all services
./stop_all.sh

# View setup log
tail -f autonomous_vm_setup.log

# View web panel log
sudo journalctl -u web-panel -f

# View VS Code log
sudo journalctl -u code-server -f
```

---

## 🤖 Multi-Agent System

### Agents Configured

1. **Archivist Agent**
   - Role: Collect and organize historical texts
   - Tools: S3 scanner, PDF extractor, text normalizer

2. **Philologist Agent**
   - Role: Analyze, lemmatize, and parse texts
   - Tools: Stanza, CLTK, valency extractor

3. **Curator Agent**
   - Role: Format and export results
   - Tools: TEI-XML builder, Git commit, vector DB updater

### Running Agents

```bash
# Run the linguistic crew
python3 linguistic_crew.py

# Or use the web control panel
# Navigate to http://135.125.216.3 and click "Start 24/7 Collection"
```

---

## 📁 Directory Structure

```
~/corpus_platform/
├── corpus_platform.db          # SQLite database
├── chroma_db/                  # Vector database
├── research_exports/
│   ├── visual_reports/         # HTML dashboards
│   ├── agent_reports/          # Markdown reports
│   ├── evaluation/             # Evaluation CSVs
│   ├── valency_analysis/       # Valency reports
│   └── night_reports/          # Night cycle logs
├── autonomous_vm_setup.sh      # Main setup script
├── check_status.sh             # Status checker
├── restart_all.sh              # Restart services
├── stop_all.sh                 # Stop services
├── linguistic_crew.py          # CrewAI agents
└── windsurf_web_panel.py       # Web control panel
```

---

## 🔧 Customization

### Change VS Code Password

Edit `/home/ubuntu/.config/code-server/config.yaml`:
```yaml
password: YOUR_NEW_PASSWORD
```

Then restart:
```bash
sudo systemctl restart code-server
```

### Add More Ollama Models

```bash
# List available models
ollama list

# Pull a new model
ollama pull llama3.1:8b

# Use in CrewAI agents by updating linguistic_crew.py
```

### Configure Nginx SSL

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

---

## 📈 Performance

### Resource Usage (Typical)

| Component | CPU | RAM | GPU |
|-----------|-----|-----|-----|
| Ollama (7B model) | 2-4 cores | 8GB | Optional |
| VS Code Server | 1 core | 2GB | N/A |
| Web Panel | 1 core | 1GB | N/A |
| ChromaDB | 1 core | 2GB | N/A |
| **Total** | **~6 cores** | **~13GB** | **Optional** |

### Recommended VM Specs

- **Minimum**: 8 vCPU, 16GB RAM, 100GB SSD
- **Recommended**: 16 vCPU, 32GB RAM, 200GB SSD
- **Optimal**: OVH L4-90 (24 vCPU, 90GB RAM, NVIDIA L4 24GB)

---

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check logs
sudo journalctl -u ollama -n 50
sudo journalctl -u code-server -n 50
sudo journalctl -u web-panel -n 50

# Restart services
sudo systemctl restart ollama code-server web-panel nginx
```

### Port Conflicts

```bash
# Check what's using ports
sudo netstat -tuln | grep -E ':(8000|8080|11434|80) '

# Kill process on port
sudo fuser -k 8000/tcp
```

### Ollama Model Issues

```bash
# Re-pull model
ollama rm qwen2.5-coder:7b
ollama pull qwen2.5-coder:7b

# Check Ollama logs
tail -f /tmp/ollama.log
```

---

## 💰 Cost Breakdown

### Software: **$0** (All open-source!)

### Infrastructure (OVH Example):
- **L4-90 Instance**: ~$730/month
- **Object Storage (1TB)**: ~$23/month
- **Bandwidth**: Usually included
- **Total**: ~$753/month

Compare to AWS/Azure: **~$2,500+/month** for equivalent setup!

---

## 🔒 Security

### Default Security Measures

- ✅ Password-protected VS Code Server
- ✅ Nginx reverse proxy
- ✅ Firewall rules (UFW)
- ✅ No external API calls (100% sovereign)

### Recommended Hardening

```bash
# Enable firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Change VS Code password (see Customization section)

# Add SSL certificate (see Customization section)
```

---

## 📚 Next Steps

1. **Access the platform**: `http://135.125.216.3`
2. **Open VS Code IDE**: `http://135.125.216.3/ide/`
3. **Run professional cycle**: Click "Run Full Professional Cycle" in web panel
4. **Explore agents**: Check `linguistic_crew.py` and customize
5. **Add more sources**: Edit `api_discovery_collector.py`
6. **Review reports**: Check `research_exports/` directories

---

## 🎓 Learning Resources

- **Ollama Docs**: https://ollama.com/docs
- **CrewAI Docs**: https://docs.crewai.com
- **Stanza Docs**: https://stanfordnlp.github.io/stanza/
- **CLTK Docs**: https://docs.cltk.org
- **ChromaDB Docs**: https://docs.trychroma.com

---

## 🤝 Support

This is a fully autonomous, open-source platform. All components are community-driven:

- **Issues**: Check individual project GitHub repos
- **Community**: Join Discord/Slack channels for each tool
- **Documentation**: See links above

---

## 📜 License

All components use permissive open-source licenses:
- Ollama: MIT
- VS Code Server: MIT
- CrewAI: MIT
- Stanza: Apache 2.0
- CLTK: MIT
- ChromaDB: Apache 2.0

**Your platform, your data, your sovereignty!** 🚀
