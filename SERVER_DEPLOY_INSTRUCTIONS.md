# SERVER DEPLOYMENT INSTRUCTIONS

## 🚀 QUICK START - 24/7 AUTONOMOUS SYSTEM

Run this on the server to start the 24/7 collector:

```bash
ssh ubuntu@54.37.228.155
sudo su -
cd /root/corpus_platform
git pull origin master
chmod +x SETUP_247_SERVICE.sh
./SETUP_247_SERVICE.sh
```

This installs a systemd service that:
- ✅ Runs **continuously 24/7**
- ✅ Collects from **12+ Indo-European treebanks**
- ✅ Extracts **valency frames** automatically
- ✅ **Auto-restarts** on crash
- ✅ **Logs** everything

### Monitor the 24/7 System:
```bash
journalctl -u corpus-collector -f          # Live logs
systemctl status corpus-collector          # Status
tail -f /root/corpus_platform/data/logs/autonomous_*.log
```

---

## Current Status
The platform now has **REAL DATA**:
- **18 documents** from Universal Dependencies
- **80,240 sentences** 
- **940,180 tokens**
- **34,163 valency frames**

Languages included:
- Ancient Greek (PROIEL + Perseus)
- Modern Greek (GDT)
- Latin (PROIEL)
- Gothic (PROIEL)
- Old Church Slavonic (PROIEL)

## To Deploy on Server (54.37.228.155)

### Step 1: SSH and become root
```bash
ssh ubuntu@54.37.228.155
sudo su -
cd /root/corpus_platform
```

### Step 2: Pull latest code
```bash
git pull origin master
```

### Step 3: Run the data collection script
```bash
python3 ACTUALLY_RUN_NOW.py
```

This will:
1. Download all UD treebanks
2. Parse CoNLL-U files
3. Extract valency frames
4. Populate the database

Expected output:
- ~80,000 sentences
- ~940,000 tokens
- ~34,000 valency frames

### Step 4: Restart the platform
```bash
systemctl restart greek-corpus
```

### Step 5: Verify
Visit http://54.37.228.155:8501 and check:
- Documents count should be 18
- Sentences should be ~80,000
- Tokens should be ~940,000

## Database Location
- Server: `/root/corpus_platform/data/corpus_platform.db`
- Local: `Z:/corpus_platform/data/corpus_platform.db`

## What's New

### Live Agent System (`core/live_agent_system.py`)
- Real AI agents that ACTUALLY work
- Native vs Borrowed argument structure analysis (Yanovich methodology)
- Statistical bootstrap for valency patterns
- Integration with Hugging Face models

### Data Collection (`ACTUALLY_RUN_NOW.py`)
- Downloads from Universal Dependencies GitHub
- Parses CoNLL-U format
- Extracts valency frames automatically
- Caches downloaded files

### Platform Updates
- Database path auto-detection (Windows/Linux)
- Statistics now read from correct tables
- Works with new schema (separate sentences/tokens tables)

## Troubleshooting

### If database shows 0:
1. Check if `data/corpus_platform.db` exists
2. Run `python3 ACTUALLY_RUN_NOW.py` again
3. Check logs for download errors

### If platform won't start:
```bash
cd /root/corpus_platform
pip3 install -r requirements.txt
streamlit run platform_app.py --server.port 8501
```

### To check database contents:
```bash
sqlite3 /root/corpus_platform/data/corpus_platform.db "SELECT COUNT(*) FROM documents; SELECT COUNT(*) FROM sentences; SELECT COUNT(*) FROM tokens;"
```
