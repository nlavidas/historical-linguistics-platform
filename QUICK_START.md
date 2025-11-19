# 🚀 Quick Start Guide - Unified Corpus Platform

## Three Ways to Use the Platform

### **Option 1: Web Dashboard (Easiest)** ⭐ RECOMMENDED

1. **Start the dashboard**:
```bash
cd Z:\corpus_platform
start_dashboard.bat
```

2. **Open browser**: http://localhost:8000

3. **Add URLs** through the web interface

4. **Click "Start Pipeline"** and watch it work!

---

### **Option 2: Command Line**

```bash
cd Z:\corpus_platform

# Install requirements
pip install -r requirements.txt

# Download language models (one-time)
python -c "import stanza; stanza.download('grc')"
python -c "import stanza; stanza.download('en')"

# Add URLs
python unified_corpus_platform.py --add-urls \
  "https://example.com/text1.xml" \
  "https://example.com/text2.xml" \
  --language grc --priority 10

# Run pipeline
python unified_corpus_platform.py --cycles 10 --delay 5

# Check status
python unified_corpus_platform.py --status
```

---

### **Option 3: Docker (Production)** 🐳

```bash
cd Z:\corpus_platform

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Dashboard available at: http://localhost:8000

---

## Testing

Run the test suite:

```bash
python test_platform.py
```

Expected output:
```
✓ IMPORTS:   PASSED
✓ DATABASE:  PASSED
✓ SCRAPER:   PASSED
✓ PIPELINE:  PASSED
```

---

## Common Commands

### Add URLs from file
```bash
# Create urls.txt with one URL per line
python unified_corpus_platform.py --add-urls $(cat urls.txt) --language grc
```

### Run specific number of cycles
```bash
python unified_corpus_platform.py --cycles 5 --delay 10
```

### Check what's in the database
```bash
sqlite3 corpus_platform.db "SELECT url, status FROM corpus_items"
```

---

## File Structure

After running, you'll have:

```
corpus_platform/
├── corpus_platform.db       → Database (all tracking)
├── data/
│   ├── raw/                 → Downloaded texts
│   ├── parsed/              → Preprocessed texts
│   └── annotated/           → NLP annotations
└── corpus_platform.log      → Activity log
```

---

## Pipeline Stages

Each text goes through 3 automatic stages:

```
URL → [DISCOVERED] → Scraper
       ↓
    [SCRAPED] → Parser
       ↓
    [PARSED] → Annotator
       ↓
    [ANNOTATED] ✓ Done!
```

---

## Troubleshooting

### Error: "No module named 'stanza'"
```bash
pip install stanza
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
```

### Error: "No module named 'aiohttp'"
```bash
pip install -r requirements.txt
```

### Dashboard won't start
```bash
pip install fastapi uvicorn
python web_dashboard.py
```

### Want to see what's happening
```bash
tail -f corpus_platform.log
```

---

## Next Steps

1. ✅ Test with a few URLs
2. ✅ Check the web dashboard
3. ✅ Integrate with your existing Z:\py\ platform
4. ✅ Add more sources
5. ✅ Export annotations

---

## Integration with Other Systems

### Connect to ERC Research Platform (Z:\py\)
```python
from unified_corpus_platform import UnifiedCorpusDatabase

db = UnifiedCorpusDatabase()
annotated_items = db.get_items_by_status('annotated')

# Feed to Z:\py\erc_diachronic_valency_platform.py
for item in annotated_items:
    process_with_erc_platform(item['annotated_path'])
```

### Connect to Scraping Platform (Z:\CLAUDE CODE\)
```python
# Use scraped texts from CLAUDE CODE platform
from scraping_platform import ScrapingDatabase

scrape_db = ScrapingDatabase("Z:\\CLAUDE CODE\\scraping_platform.db")
results = scrape_db.get_completed_tasks()

# Add to unified platform
platform.add_source_urls([r['url'] for r in results])
```

---

## Support

- **Documentation**: See START_HERE.md
- **Examples**: See test_platform.py
- **Full analysis**: See Z:\PROJECT_ANALYSIS_AND_CONTINUATION.md

---

**Your platform is ready! Start with the web dashboard for the easiest experience.**
