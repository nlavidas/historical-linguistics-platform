# Installation Guide - HFRI-NKUA AI Corpus Platform

## Quick Install (5 Minutes)

### Step 1: Install Python Packages
```bash
cd Z:\corpus_platform
pip install -r requirements.txt
pip install spacy transformers nltk textblob trankit ollama
```

### Step 2: Download AI Models

**Stanza (Required)**:
```bash
python -c "import stanza; stanza.download('grc')"
python -c "import stanza; stanza.download('en')"
```

**spaCy (Optional but recommended)**:
```bash
python -m spacy download en_core_web_sm
python -m spacy download el_core_news_sm
```

**NLTK Data (Optional)**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Step 3: Test Installation
```bash
python test_platform.py
python multi_ai_annotator.py
```

Expected: All tests pass ✅

---

## Available AI Models

### Tier 1: High-Accuracy Models (Recommended)
- ✅ **Stanza** - Stanford CoreNLP (Best for Ancient Greek)
- ✅ **spaCy** - Industrial NLP (Fast and accurate)
- ✅ **Transformers** - Hugging Face (BERT, etc.)

### Tier 2: Specialized Models
- ✅ **Trankit** - Multilingual transformer
- ✅ **NLTK** - Classic NLP toolkit
- ✅ **TextBlob** - Simple text processing

### Tier 3: LLM Integration
- ✅ **Ollama** - Local LLM (llama3.2, etc.)

---

## Installation Options

### Option A: Minimal (Stanza Only)
```bash
pip install stanza aiohttp
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
```
**Size**: ~500MB  
**Time**: 5 minutes

### Option B: Recommended (Multiple Models)
```bash
pip install stanza spacy transformers nltk aiohttp
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```
**Size**: ~2GB  
**Time**: 10 minutes

### Option C: Full Stack (All Models)
```bash
pip install stanza spacy transformers nltk textblob trankit ollama aiohttp fastapi uvicorn
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('grc'); stanza.download('en')"
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```
**Size**: ~5GB  
**Time**: 20 minutes

---

## Verify Installation

### Check Available Models
```bash
python -c "from multi_ai_annotator import MultiAIAnnotator; m = MultiAIAnnotator(); print(f'Available: {len(m.available_models)} models')"
```

### Test Multi-AI Annotation
```bash
python multi_ai_annotator.py
```

---

## Ollama Setup (Optional - For Local LLM)

### Install Ollama
1. Download from: https://ollama.ai
2. Install and run: `ollama serve`
3. Pull model: `ollama pull llama3.2`

### Test Ollama
```bash
ollama run llama3.2 "Hello"
```

---

## Docker Installation (Alternative)

### Using Docker Compose
```bash
cd Z:\corpus_platform
docker-compose up -d
```

This installs everything automatically!

---

## Troubleshooting

### "No module named 'stanza'"
```bash
pip install stanza
```

### "Can't find model 'grc'"
```bash
python -c "import stanza; stanza.download('grc')"
```

### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Low disk space
Install minimal option (Option A) - only 500MB

### Installation too slow
Use Option A first, add more models later

---

## System Requirements

**Minimum**:
- Python 3.8+
- 2GB RAM
- 1GB disk space
- Internet connection

**Recommended**:
- Python 3.10+
- 8GB RAM
- 5GB disk space
- Fast internet

**Optimal**:
- Python 3.11+
- 16GB RAM
- 10GB disk space
- GPU (optional, for Transformers)

---

## Next Steps

After installation:

1. **Test**: `python test_platform.py`
2. **View achievements**: Open `achievements.html` in browser
3. **Start 24/7**: Run `run_24_7_with_dashboard.bat`
4. **Add URLs**: Through web dashboard at http://localhost:8000

---

## Updates

To update all packages:
```bash
pip install --upgrade stanza spacy transformers nltk
python -c "import stanza; stanza.download('grc', force=True)"
python -m spacy download en_core_web_sm --upgrade
```

---

**Installation complete! Start with `run_24_7_with_dashboard.bat`**
