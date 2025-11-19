# Unified AI Corpus Platform

**Super-powerful AI corpus platform with automatic scraping, parsing, and annotation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/status-production-green.svg)]()

---

## 🎯 What This Does

Automatically processes texts through three stages:

1. **Scraping** - Downloads texts from URLs
2. **Parsing** - Preprocesses and normalizes
3. **Annotation** - Full NLP analysis with Stanza

All tracked in a SQLite database with resume capability.

---

## ⚡ Quick Start (60 seconds)

### Option 1: Web Dashboard (Easiest)
```bash
cd Z:\corpus_platform
start_dashboard.bat
```
Then open: http://localhost:8000

### Option 2: Command Line
```bash
pip install -r requirements.txt
python unified_corpus_platform.py --add-urls "https://example.com/text.xml" --language grc
python unified_corpus_platform.py --cycles 10
```

### Option 3: Docker
```bash
docker-compose up -d
```

---

## 📊 Features

### Core Features
- ✅ **24/7 automatic operation** - Set it and forget it
- ✅ **Multi-stage pipeline** - Scraping → Parsing → Annotation
- ✅ **Progress tracking** - SQLite database
- ✅ **Resume capability** - Can stop/start without losing progress
- ✅ **Multi-source support** - Perseus, GitHub, Archive.org, etc.
- ✅ **Language detection** - Auto-detects Ancient Greek, English, etc.
- ✅ **Web dashboard** - Real-time monitoring
- ✅ **RESTful API** - Programmatic control

### NLP Features
- ✅ **Stanza integration** - Stanford NLP
- ✅ **Tokenization** - Smart text splitting
- ✅ **POS tagging** - Part-of-speech analysis
- ✅ **Lemmatization** - Base word forms
- ✅ **Dependency parsing** - Syntactic structure
- ✅ **JSON export** - Structured annotations

---

## 📁 What You Get

### After Processing:
```
data/
├── raw/                    → Downloaded texts (original)
├── parsed/                 → Preprocessed texts (cleaned)
└── annotated/              → Full NLP annotations (JSON)

corpus_platform.db          → Complete tracking database
corpus_platform.log         → Activity log
```

### Database Tables:
- `corpus_items` - Main tracking (URL, status, paths)
- `annotations` - Annotation results
- `pipeline_status` - Processing stages
- `sources` - Source configuration
- `statistics` - Platform metrics

---

## 🔧 Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet connection (for scraping)

### Python Packages:
```
aiohttp>=3.9.0
stanza>=1.4.0
fastapi>=0.104.0
uvicorn>=0.24.0
```

Install all: `pip install -r requirements.txt`

---

## 📖 Usage Examples

### Add Single URL
```bash
python unified_corpus_platform.py \
  --add-urls "https://perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133" \
  --source-type perseus \
  --language grc \
  --priority 10
```

### Add Multiple URLs
```bash
python unified_corpus_platform.py \
  --add-urls \
    "https://example.com/text1.xml" \
    "https://example.com/text2.xml" \
    "https://example.com/text3.xml" \
  --language grc
```

### Run Pipeline
```bash
# Run 10 cycles with 5 second delay
python unified_corpus_platform.py --cycles 10 --delay 5

# Run continuously (press Ctrl+C to stop)
python unified_corpus_platform.py
```

### Check Status
```bash
python unified_corpus_platform.py --status
```

### Programmatic Usage
```python
from unified_corpus_platform import UnifiedCorpusPlatform
import asyncio

# Create platform
platform = UnifiedCorpusPlatform()

# Add URLs
platform.add_source_urls([
    "https://example.com/text1.xml",
    "https://example.com/text2.xml"
], source_type="custom_url", language="grc")

# Run pipeline
asyncio.run(platform.run_pipeline(cycles=5))

# Get statistics
stats = platform.db.get_statistics()
print(f"Total: {stats['total_items']}")
print(f"Status: {stats['status_counts']}")
```

---

## 🌐 Web Dashboard

### Start Dashboard:
```bash
python web_dashboard.py
```

### Features:
- Real-time statistics
- Add URLs via web interface
- Start/stop pipeline
- Monitor progress
- View item details

### API Endpoints:
- `GET /` - Dashboard HTML
- `GET /api/stats` - Get statistics
- `POST /api/add-urls` - Add URLs
- `POST /api/control` - Control pipeline
- `GET /api/items` - Get items

---

## 🐳 Docker Deployment

### Build and Run:
```bash
docker-compose up -d
```

### View Logs:
```bash
docker-compose logs -f
```

### Stop:
```bash
docker-compose down
```

### Services:
- `corpus-platform` - Web dashboard (port 8000)
- `corpus-pipeline` - Background worker

---

## 🔬 Testing

Run test suite:
```bash
python test_platform.py
```

Expected output:
```
IMPORTS:   ✓ PASSED
DATABASE:  ✓ PASSED
SCRAPER:   ✓ PASSED
PIPELINE:  ✓ PASSED

✓ ALL TESTS PASSED!
```

---

## 🔗 Integration

### With ERC Research Platform (Z:\py\)
```python
# Get annotated items
from unified_corpus_platform import UnifiedCorpusDatabase

db = UnifiedCorpusDatabase()
items = db.get_items_by_status('annotated')

# Process with ERC platform
for item in items:
    # Feed to Z:\py\erc_diachronic_valency_platform.py
    process_annotations(item['annotated_path'])
```

### With Scraping Platform (Z:\CLAUDE CODE\)
```python
# Use scraped texts
from scraping_platform import ScrapingDatabase

scrape_db = ScrapingDatabase("../CLAUDE CODE/scraping_platform.db")
results = scrape_db.get_completed_tasks()

# Add to unified platform
platform.add_source_urls([r['url'] for r in results])
```

---

## 📈 Performance

- **Scraping**: 10-100 texts/hour (depends on source)
- **Parsing**: 100-500 texts/hour
- **Annotation**: 5-20 texts/hour (depends on text length)
- **Memory**: ~200MB base + ~50MB per worker
- **Storage**: ~1-10MB per annotated text

---

## 🛠️ Troubleshooting

### "No module named 'stanza'"
```bash
pip install stanza
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
```

### "Database is locked"
Only one process can write at a time. Stop other instances.

### Scraping fails
Check URL is accessible. Some sites block automated access.

### Annotation is slow
Stanza annotation is CPU-intensive. Reduce batch size.

### Want more speed
- Run multiple instances with different priorities
- Use faster hardware
- Process shorter texts first

---

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **Platform Overview**: `PLATFORM_OVERVIEW.txt`
- **Complete Analysis**: `../PROJECT_ANALYSIS_AND_CONTINUATION.md`
- **Main Project**: `../README.md`

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│          Web Dashboard (FastAPI)        │
│          http://localhost:8000          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│    UnifiedCorpusPlatform (Orchestrator) │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────────┐
   │Scraper │ │Parser  │ │Annotator   │
   │        │ │        │ │(Stanza NLP)│
   └────┬───┘ └───┬────┘ └─────┬──────┘
        │         │            │
        └─────────┼────────────┘
                  ▼
      ┌───────────────────────┐
      │   SQLite Database     │
      │  corpus_platform.db   │
      └───────────────────────┘
```

---

## 🎓 Credits

- **Author**: Nikolaos Lavidas, NKUA
- **Institution**: National and Kapodistrian University of Athens
- **Department**: Department of Linguistics
- **Funding**: European Research Council (ERC)
- **Date**: November 2025
- **Version**: 1.0.0

### Built With:
- [Stanza](https://stanfordnlp.github.io/stanza/) - Stanford NLP
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [aiohttp](https://docs.aiohttp.org/) - Async HTTP client
- SQLite - Database

---

## 📜 License

MIT License - Free for academic and commercial use

---

## 🚀 Status

**Production Ready** - All features implemented and tested

- ✅ Core pipeline complete
- ✅ Web dashboard complete
- ✅ Docker deployment ready
- ✅ Documentation complete
- ✅ Tests passing

---

## 📞 Support

- **Documentation**: See `QUICK_START.md`
- **Issues**: Check `corpus_platform.log`
- **Testing**: Run `test_platform.py`
- **Questions**: See full docs in `Z:\PROJECT_ANALYSIS_AND_CONTINUATION.md`

---

**Start now**: `start_dashboard.bat` or `python unified_corpus_platform.py --status`

🎯 **Your super-powerful AI corpus platform is ready to use!**
