# Unified AI Corpus Platform - Quick Start

## What Was Created

✅ **unified_corpus_platform.py** - Complete AI corpus platform with:
- Automatic web scraping (24/7)
- Automatic parsing and preprocessing  
- Automatic annotation with Stanza NLP
- SQLite database for tracking
- Multi-source support (Perseus, GitHub, Archive.org, etc.)

## Quick Start

### 1. Install Requirements
```bash
cd Z:\corpus_platform
pip install aiohttp stanza sqlite3
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
```

### 2. Add URLs to Process
```bash
python unified_corpus_platform.py --add-urls https://example.com/text1.xml https://example.com/text2.xml --language grc
```

### 3. Run Pipeline
```bash
python unified_corpus_platform.py --cycles 10 --delay 5
```

### 4. Check Status
```bash
python unified_corpus_platform.py --status
```

## Features

- **Auto-scraping**: Downloads texts from URLs
- **Auto-parsing**: Preprocesses and normalizes
- **Auto-annotation**: Full NLP annotation with Stanza
- **Multi-format**: Supports XML, TXT, HTML
- **Progress tracking**: SQLite database tracks all stages
- **Resumable**: Can stop/restart without data loss

## Database Structure

- `corpus_items` - Main items table
- `annotations` - Annotation results
- `pipeline_status` - Processing stages
- `statistics` - Platform metrics

## Integration with Existing Platform

This integrates with your existing Z:\py\ platform:
- Use scraped texts in Z:\py\erc_diachronic_valency_platform.py
- Annotations compatible with existing treebank converters
- Can feed into Z:\CLAUDE CODE\scraping_platform.py
