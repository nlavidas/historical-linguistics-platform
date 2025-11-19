"""
Local Models Configuration
Ensures all AI models are loaded from Z:\models\ directory
No repeated downloads needed - all models stored locally
"""

import os
from pathlib import Path

# Base directory for all models
MODELS_BASE = Path("Z:/models")

# Stanza NLP models
STANZA_DIR = MODELS_BASE / "stanza"
os.environ['STANZA_RESOURCES_DIR'] = str(STANZA_DIR)

# NLTK data
NLTK_DIR = MODELS_BASE / "nltk_data"
os.environ['NLTK_DATA'] = str(NLTK_DIR)

# Hugging Face / Transformers models
TRANSFORMERS_DIR = MODELS_BASE / "transformers"
os.environ['TRANSFORMERS_CACHE'] = str(TRANSFORMERS_DIR)
os.environ['HF_HOME'] = str(TRANSFORMERS_DIR)
os.environ['TORCH_HOME'] = str(TRANSFORMERS_DIR)

# spaCy models
SPACY_DIR = MODELS_BASE / "spacy"
os.environ['SPACY_DATA'] = str(SPACY_DIR)

# Ollama models (if used)
OLLAMA_DIR = MODELS_BASE / "ollama"
os.environ['OLLAMA_MODELS'] = str(OLLAMA_DIR)


def ensure_directories():
    """Create model directories if they don't exist"""
    for directory in [STANZA_DIR, NLTK_DIR, TRANSFORMERS_DIR, SPACY_DIR, OLLAMA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def verify_models():
    """Check if models are downloaded"""
    status = {
        'stanza': STANZA_DIR.exists() and any(STANZA_DIR.iterdir()),
        'nltk': NLTK_DIR.exists() and any(NLTK_DIR.iterdir()),
        'models_base': MODELS_BASE.exists()
    }
    return status


def get_stanza_model_path(language='grc'):
    """Get path to Stanza model for specific language"""
    return str(STANZA_DIR)


def get_nltk_data_path():
    """Get path to NLTK data"""
    return str(NLTK_DIR)


# Initialize on import
ensure_directories()

# Print confirmation when imported
if __name__ != "__main__":
    print(f"✓ Local models config loaded - using Z:\\models\\")
    status = verify_models()
    if not status['models_base']:
        print(f"⚠ Warning: Models directory not found. Run SETUP_MODELS_ONCE.bat")
