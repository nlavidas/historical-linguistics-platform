"""
Centralized Configuration for Historical Linguistics Platform
Supports multiple environments: Windows (Z: drive), Linux, GitHub-based data

This configuration auto-detects the environment and provides appropriate paths.
All modules should import from this config instead of hardcoding paths.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlatformConfig:
    """
    Centralized configuration for the Historical Linguistics Platform.
    
    Supports three deployment modes:
    1. Windows with Z: drive (user's local setup)
    2. Linux/Unix with local paths
    3. GitHub-based data fetching (for CI/CD or cloud deployments)
    
    Usage:
        from config import config
        data_dir = config.data_dir
        stanza_dir = config.stanza_resources_dir
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._detect_environment()
        self._setup_paths()
        self._setup_environment_variables()
        
    def _detect_environment(self):
        """Detect the current environment and set base paths accordingly."""
        self.is_windows = sys.platform.startswith('win')
        self.is_linux = sys.platform.startswith('linux')
        self.is_macos = sys.platform.startswith('darwin')
        
        self.z_drive_available = False
        if self.is_windows:
            z_path = Path("Z:/")
            self.z_drive_available = z_path.exists()
        
        self.use_github_data = os.environ.get('USE_GITHUB_DATA', 'false').lower() == 'true'
        
        self.repo_root = Path(__file__).parent.resolve()
        
        if self.z_drive_available:
            self.deployment_mode = 'windows_z_drive'
            self.base_data_dir = Path("Z:/corpus_platform")
            self.base_models_dir = Path("Z:/models")
        elif self.use_github_data:
            self.deployment_mode = 'github'
            self.base_data_dir = self.repo_root / "data"
            self.base_models_dir = self.repo_root / "models"
        else:
            self.deployment_mode = 'local'
            self.base_data_dir = self.repo_root / "data"
            self.base_models_dir = self.repo_root / "models"
        
        logger.info(f"Platform config initialized: mode={self.deployment_mode}, "
                   f"data_dir={self.base_data_dir}")
    
    def _setup_paths(self):
        """Setup all path configurations."""
        self.data_dir = self.base_data_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.parsed_dir = self.data_dir / "parsed"
        self.annotated_dir = self.data_dir / "annotated"
        self.cache_dir = self.data_dir / "cache"
        self.exports_dir = self.base_data_dir / "research_exports"
        
        self.corpus_db_path = self.base_data_dir / "data" / "corpus_platform.db"
        self.valency_db_path = self.base_data_dir / "data" / "valency_lexicon.db"
        self.annotation_db_path = self.base_data_dir / "data" / "annotations.db"
        self.metadata_db_path = self.base_data_dir / "corpus_metadata.db"
        
        self.stanza_resources_dir = self.base_models_dir / "stanza"
        self.nltk_data_dir = self.base_models_dir / "nltk_data"
        self.transformers_cache_dir = self.base_models_dir / "transformers"
        self.spacy_data_dir = self.base_models_dir / "spacy"
        self.ollama_models_dir = self.base_models_dir / "ollama"
        self.lightside_models_dir = self.base_models_dir / "lightside"
        self.ml_annotator_dir = self.base_models_dir / "ml_annotator"
        
        self.logs_dir = self.base_data_dir / "logs"
        self.obsidian_notes_dir = self.exports_dir / "obsidian_notes"
        self.huggingface_export_dir = self.exports_dir / "huggingface_dataset"
        
    def _setup_environment_variables(self):
        """Set environment variables for external libraries."""
        os.environ['STANZA_RESOURCES_DIR'] = str(self.stanza_resources_dir)
        os.environ['NLTK_DATA'] = str(self.nltk_data_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.transformers_cache_dir)
        os.environ['HF_HOME'] = str(self.transformers_cache_dir)
        os.environ['TORCH_HOME'] = str(self.transformers_cache_dir)
        os.environ['SPACY_DATA'] = str(self.spacy_data_dir)
        os.environ['OLLAMA_MODELS'] = str(self.ollama_models_dir)
        
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_dir,
            self.parsed_dir,
            self.annotated_dir,
            self.cache_dir,
            self.exports_dir,
            self.logs_dir,
            self.obsidian_notes_dir,
            self.huggingface_export_dir,
            self.stanza_resources_dir,
            self.nltk_data_dir,
            self.transformers_cache_dir,
            self.spacy_data_dir,
            self.ollama_models_dir,
            self.lightside_models_dir,
            self.ml_annotator_dir,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.warning(f"Cannot create directory (permission denied): {directory}")
            except Exception as e:
                logger.warning(f"Cannot create directory {directory}: {e}")
        
        return self
    
    def get_db_path(self, db_name: str) -> Path:
        """Get path for a specific database."""
        db_paths = {
            'corpus': self.corpus_db_path,
            'valency': self.valency_db_path,
            'annotation': self.annotation_db_path,
            'metadata': self.metadata_db_path,
        }
        return db_paths.get(db_name, self.base_data_dir / "data" / f"{db_name}.db")
    
    def get_model_path(self, model_type: str) -> Path:
        """Get path for a specific model type."""
        model_paths = {
            'stanza': self.stanza_resources_dir,
            'nltk': self.nltk_data_dir,
            'transformers': self.transformers_cache_dir,
            'spacy': self.spacy_data_dir,
            'ollama': self.ollama_models_dir,
            'lightside': self.lightside_models_dir,
            'ml_annotator': self.ml_annotator_dir,
        }
        return model_paths.get(model_type, self.base_models_dir / model_type)
    
    def verify_models(self) -> Dict[str, bool]:
        """Check if models are available."""
        status = {
            'stanza': self.stanza_resources_dir.exists() and any(self.stanza_resources_dir.iterdir()) if self.stanza_resources_dir.exists() else False,
            'nltk': self.nltk_data_dir.exists() and any(self.nltk_data_dir.iterdir()) if self.nltk_data_dir.exists() else False,
            'transformers': self.transformers_cache_dir.exists(),
            'models_base': self.base_models_dir.exists(),
        }
        return status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current configuration status."""
        return {
            'deployment_mode': self.deployment_mode,
            'is_windows': self.is_windows,
            'is_linux': self.is_linux,
            'z_drive_available': self.z_drive_available,
            'use_github_data': self.use_github_data,
            'repo_root': str(self.repo_root),
            'base_data_dir': str(self.base_data_dir),
            'base_models_dir': str(self.base_models_dir),
            'models_status': self.verify_models(),
        }
    
    def __repr__(self):
        return f"PlatformConfig(mode={self.deployment_mode}, data={self.base_data_dir})"


config = PlatformConfig()


GITHUB_DATA_SOURCES = {
    'proiel_treebank': {
        'repo': 'proiel/proiel-treebank',
        'branch': 'master',
        'files': [
            'greek-nt.xml',
            'greek-herodotus.xml',
            'greek-sphrantzes.xml',
        ]
    },
    'first1k_greek': {
        'repo': 'OpenGreekAndLatin/First1KGreek',
        'branch': 'master',
        'base_path': 'data/',
    },
    'perseus_canonical': {
        'repo': 'PerseusDL/canonical-greekLit',
        'branch': 'master',
        'base_path': 'data/',
    },
    'ud_ancient_greek': {
        'repo': 'UniversalDependencies/UD_Ancient_Greek-PROIEL',
        'branch': 'master',
        'files': [
            'grc_proiel-ud-train.conllu',
            'grc_proiel-ud-dev.conllu',
            'grc_proiel-ud-test.conllu',
        ]
    },
    'ud_ancient_greek_perseus': {
        'repo': 'UniversalDependencies/UD_Ancient_Greek-Perseus',
        'branch': 'master',
        'files': [
            'grc_perseus-ud-train.conllu',
            'grc_perseus-ud-dev.conllu',
            'grc_perseus-ud-test.conllu',
        ]
    },
}


def get_github_raw_url(repo: str, branch: str, file_path: str) -> str:
    """Generate raw GitHub URL for a file."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"


def download_github_file(repo: str, branch: str, file_path: str, 
                         local_path: Optional[Path] = None) -> Optional[str]:
    """Download a file from GitHub."""
    import requests
    
    url = get_github_raw_url(repo, branch, file_path)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding='utf-8')
            logger.info(f"Downloaded {file_path} to {local_path}")
        
        return content
        
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("HISTORICAL LINGUISTICS PLATFORM - CONFIGURATION")
    print("=" * 60)
    
    status = config.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nEnsuring directories exist...")
    config.ensure_directories()
    
    print("\nConfiguration complete!")
