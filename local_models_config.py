"""
Local Models Configuration
Ensures all AI models are loaded from the appropriate directory based on environment.
Supports both Windows (Z: drive) and Linux/cloud deployments.

This module now uses the centralized config.py for path management.
"""

import os
import sys
from pathlib import Path

try:
    from config import config
    MODELS_BASE = config.base_models_dir
    STANZA_DIR = config.stanza_resources_dir
    NLTK_DIR = config.nltk_data_dir
    TRANSFORMERS_DIR = config.transformers_cache_dir
    SPACY_DIR = config.spacy_data_dir
    OLLAMA_DIR = config.ollama_models_dir
    _using_centralized_config = True
except ImportError:
    _using_centralized_config = False
    
    is_windows = sys.platform.startswith('win')
    z_drive_available = is_windows and Path("Z:/").exists()
    
    if z_drive_available:
        MODELS_BASE = Path("Z:/models")
    else:
        MODELS_BASE = Path(__file__).parent / "models"
    
    STANZA_DIR = MODELS_BASE / "stanza"
    NLTK_DIR = MODELS_BASE / "nltk_data"
    TRANSFORMERS_DIR = MODELS_BASE / "transformers"
    SPACY_DIR = MODELS_BASE / "spacy"
    OLLAMA_DIR = MODELS_BASE / "ollama"

os.environ['STANZA_RESOURCES_DIR'] = str(STANZA_DIR)
os.environ['NLTK_DATA'] = str(NLTK_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(TRANSFORMERS_DIR)
os.environ['HF_HOME'] = str(TRANSFORMERS_DIR)
os.environ['TORCH_HOME'] = str(TRANSFORMERS_DIR)
os.environ['SPACY_DATA'] = str(SPACY_DIR)
os.environ['OLLAMA_MODELS'] = str(OLLAMA_DIR)


def ensure_directories():
    """Create model directories if they don't exist"""
    directories = [STANZA_DIR, NLTK_DIR, TRANSFORMERS_DIR, SPACY_DIR, OLLAMA_DIR]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass
        except Exception:
            pass


def verify_models():
    """Check if models are downloaded"""
    def dir_has_content(d):
        try:
            return d.exists() and any(d.iterdir())
        except:
            return False
    
    status = {
        'stanza': dir_has_content(STANZA_DIR),
        'nltk': dir_has_content(NLTK_DIR),
        'transformers': TRANSFORMERS_DIR.exists(),
        'models_base': MODELS_BASE.exists()
    }
    return status


def get_stanza_model_path(language='grc'):
    """Get path to Stanza model for specific language"""
    return str(STANZA_DIR)


def get_nltk_data_path():
    """Get path to NLTK data"""
    return str(NLTK_DIR)


def get_models_base():
    """Get base models directory"""
    return str(MODELS_BASE)


ensure_directories()

if __name__ != "__main__":
    if _using_centralized_config:
        print(f"Local models config loaded - using centralized config: {MODELS_BASE}")
    else:
        print(f"Local models config loaded - using fallback: {MODELS_BASE}")
