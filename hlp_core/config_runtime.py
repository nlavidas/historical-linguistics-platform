"""
HLP Core Config Runtime - Runtime Configuration Management

This module provides runtime configuration management, environment validation,
path resolution, and model registry functionality.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment modes for the platform"""
    LOCAL = "local"
    WINDOWS_Z_DRIVE = "windows_z"
    LINUX_SERVER = "linux_server"
    DOCKER = "docker"
    CLOUD = "cloud"
    GITHUB_ACTIONS = "github_actions"


class ModelType(Enum):
    """Types of AI models used in the platform"""
    STANZA = "stanza"
    SPACY = "spacy"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class LanguageSupport(Enum):
    """Language support levels"""
    FULL = "full"
    PARTIAL = "partial"
    EXPERIMENTAL = "experimental"
    NONE = "none"


@dataclass
class ModelInfo:
    """Information about an AI model"""
    model_id: str
    model_type: ModelType
    name: str
    version: str
    language: str
    
    path: Optional[str] = None
    url: Optional[str] = None
    
    size_mb: int = 0
    is_downloaded: bool = False
    is_loaded: bool = False
    
    capabilities: List[str] = field(default_factory=list)
    
    config: Dict[str, Any] = field(default_factory=dict)
    
    last_used: Optional[datetime] = None
    load_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "path": self.path,
            "url": self.url,
            "size_mb": self.size_mb,
            "is_downloaded": self.is_downloaded,
            "is_loaded": self.is_loaded,
            "capabilities": self.capabilities
        }


@dataclass
class LanguageConfig:
    """Configuration for a supported language"""
    code: str
    name: str
    support_level: LanguageSupport
    
    stanza_model: Optional[str] = None
    spacy_model: Optional[str] = None
    hf_models: List[str] = field(default_factory=list)
    
    tokenizer: Optional[str] = None
    lemmatizer: Optional[str] = None
    pos_tagger: Optional[str] = None
    parser: Optional[str] = None
    ner_model: Optional[str] = None
    
    unicode_normalization: str = "NFC"
    script: str = "Greek"
    
    periods: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "code": self.code,
            "name": self.name,
            "support_level": self.support_level.value,
            "stanza_model": self.stanza_model,
            "spacy_model": self.spacy_model,
            "hf_models": self.hf_models,
            "periods": self.periods
        }


class PathResolver:
    """Resolves paths based on deployment environment"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir or self._detect_base_dir()
        self._cache: Dict[str, Path] = {}
        self._lock = threading.Lock()
    
    def _detect_base_dir(self) -> Path:
        """Detect the base directory"""
        if os.environ.get("HLP_BASE_DIR"):
            return Path(os.environ["HLP_BASE_DIR"])
        
        if sys.platform.startswith("win"):
            z_drive = Path("Z:/")
            if z_drive.exists():
                corpus_dir = z_drive / "corpus_platform"
                if corpus_dir.exists():
                    return corpus_dir
        
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "config.py").exists() or (parent / "platform_app.py").exists():
                return parent
        
        return Path.cwd()
    
    @property
    def base_dir(self) -> Path:
        """Get base directory"""
        return self._base_dir
    
    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        return self._get_or_create("data", self._base_dir / "data")
    
    @property
    def models_dir(self) -> Path:
        """Get models directory"""
        return self._get_or_create("models", self._base_dir / "models")
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory"""
        return self._get_or_create("cache", self._base_dir / "cache")
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory"""
        return self._get_or_create("logs", self._base_dir / "logs")
    
    @property
    def exports_dir(self) -> Path:
        """Get exports directory"""
        return self._get_or_create("exports", self._base_dir / "exports")
    
    @property
    def temp_dir(self) -> Path:
        """Get temporary directory"""
        return self._get_or_create("temp", self._base_dir / "temp")
    
    @property
    def config_dir(self) -> Path:
        """Get configuration directory"""
        return self._get_or_create("config", self._base_dir / "config")
    
    @property
    def corpus_dir(self) -> Path:
        """Get corpus data directory"""
        return self._get_or_create("corpus", self.data_dir / "corpus")
    
    @property
    def treebank_dir(self) -> Path:
        """Get treebank directory"""
        return self._get_or_create("treebank", self.data_dir / "treebank")
    
    @property
    def raw_dir(self) -> Path:
        """Get raw data directory"""
        return self._get_or_create("raw", self.data_dir / "raw")
    
    @property
    def processed_dir(self) -> Path:
        """Get processed data directory"""
        return self._get_or_create("processed", self.data_dir / "processed")
    
    @property
    def annotated_dir(self) -> Path:
        """Get annotated data directory"""
        return self._get_or_create("annotated", self.data_dir / "annotated")
    
    def _get_or_create(self, key: str, path: Path) -> Path:
        """Get path from cache or create directory"""
        with self._lock:
            if key not in self._cache:
                path.mkdir(parents=True, exist_ok=True)
                self._cache[key] = path
            return self._cache[key]
    
    def get_db_path(self, db_name: str = "hlp_corpus.db") -> Path:
        """Get database file path"""
        return self.data_dir / db_name
    
    def get_model_path(self, model_type: ModelType, model_name: str) -> Path:
        """Get model directory path"""
        return self.models_dir / model_type.value / model_name
    
    def get_corpus_path(self, corpus_id: str) -> Path:
        """Get corpus directory path"""
        return self.corpus_dir / corpus_id
    
    def get_export_path(self, export_name: str, format: str = "xml") -> Path:
        """Get export file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{export_name}_{timestamp}.{format}"
        return self.exports_dir / filename
    
    def get_log_path(self, log_name: str = "hlp_platform.log") -> Path:
        """Get log file path"""
        return self.logs_dir / log_name
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{key_hash}.cache"
    
    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative path to absolute"""
        return self._base_dir / relative_path
    
    def to_dict(self) -> Dict[str, str]:
        """Get all paths as dictionary"""
        return {
            "base_dir": str(self._base_dir),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "cache_dir": str(self.cache_dir),
            "logs_dir": str(self.logs_dir),
            "exports_dir": str(self.exports_dir),
            "corpus_dir": str(self.corpus_dir),
            "treebank_dir": str(self.treebank_dir)
        }


class EnvironmentValidator:
    """Validates the runtime environment"""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self._validation_results: Dict[str, Tuple[bool, str]] = {}
    
    def validate_all(self) -> Dict[str, Tuple[bool, str]]:
        """Run all validations"""
        self._validation_results = {}
        
        self._validate_python_version()
        self._validate_directories()
        self._validate_dependencies()
        self._validate_environment_variables()
        self._validate_disk_space()
        self._validate_memory()
        
        return self._validation_results
    
    def _validate_python_version(self):
        """Validate Python version"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self._validation_results["python_version"] = (
                True, f"Python {version.major}.{version.minor}.{version.micro}"
            )
        else:
            self._validation_results["python_version"] = (
                False, f"Python 3.8+ required, found {version.major}.{version.minor}"
            )
    
    def _validate_directories(self):
        """Validate required directories exist and are writable"""
        dirs_to_check = [
            ("data_dir", self.path_resolver.data_dir),
            ("models_dir", self.path_resolver.models_dir),
            ("logs_dir", self.path_resolver.logs_dir),
            ("cache_dir", self.path_resolver.cache_dir)
        ]
        
        for name, path in dirs_to_check:
            if path.exists() and os.access(path, os.W_OK):
                self._validation_results[name] = (True, str(path))
            else:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self._validation_results[name] = (True, f"Created: {path}")
                except Exception as e:
                    self._validation_results[name] = (False, f"Cannot create: {e}")
    
    def _validate_dependencies(self):
        """Validate required Python packages"""
        required_packages = [
            ("sqlite3", "sqlite3"),
            ("json", "json"),
            ("pathlib", "pathlib"),
            ("dataclasses", "dataclasses"),
            ("typing", "typing"),
            ("logging", "logging"),
            ("threading", "threading"),
            ("queue", "queue"),
            ("hashlib", "hashlib"),
            ("re", "re"),
            ("collections", "collections"),
        ]
        
        optional_packages = [
            ("stanza", "stanza"),
            ("spacy", "spacy"),
            ("transformers", "transformers"),
            ("torch", "torch"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("requests", "requests"),
            ("aiohttp", "aiohttp"),
            ("fastapi", "fastapi"),
            ("streamlit", "streamlit"),
            ("plotly", "plotly"),
            ("lxml", "lxml"),
        ]
        
        for name, module in required_packages:
            try:
                __import__(module)
                self._validation_results[f"pkg_{name}"] = (True, "Available")
            except ImportError:
                self._validation_results[f"pkg_{name}"] = (False, "Missing (required)")
        
        for name, module in optional_packages:
            try:
                __import__(module)
                self._validation_results[f"pkg_{name}"] = (True, "Available")
            except ImportError:
                self._validation_results[f"pkg_{name}"] = (True, "Not installed (optional)")
    
    def _validate_environment_variables(self):
        """Validate environment variables"""
        env_vars = [
            ("HLP_BASE_DIR", False),
            ("STANZA_RESOURCES_DIR", False),
            ("NLTK_DATA", False),
            ("TRANSFORMERS_CACHE", False),
            ("HF_HOME", False),
            ("SPACY_DATA", False),
            ("OLLAMA_MODELS", False),
        ]
        
        for var_name, required in env_vars:
            value = os.environ.get(var_name)
            if value:
                self._validation_results[f"env_{var_name}"] = (True, value[:50])
            elif required:
                self._validation_results[f"env_{var_name}"] = (False, "Not set (required)")
            else:
                self._validation_results[f"env_{var_name}"] = (True, "Not set (optional)")
    
    def _validate_disk_space(self):
        """Validate available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.path_resolver.base_dir)
            free_gb = free / (1024 ** 3)
            
            if free_gb >= 10:
                self._validation_results["disk_space"] = (True, f"{free_gb:.1f} GB free")
            elif free_gb >= 1:
                self._validation_results["disk_space"] = (True, f"{free_gb:.1f} GB free (low)")
            else:
                self._validation_results["disk_space"] = (False, f"{free_gb:.2f} GB free (critical)")
        except Exception as e:
            self._validation_results["disk_space"] = (False, f"Cannot check: {e}")
    
    def _validate_memory(self):
        """Validate available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            
            if available_gb >= 4:
                self._validation_results["memory"] = (True, f"{available_gb:.1f} GB available")
            elif available_gb >= 1:
                self._validation_results["memory"] = (True, f"{available_gb:.1f} GB available (low)")
            else:
                self._validation_results["memory"] = (False, f"{available_gb:.2f} GB available (critical)")
        except ImportError:
            self._validation_results["memory"] = (True, "psutil not available")
        except Exception as e:
            self._validation_results["memory"] = (False, f"Cannot check: {e}")
    
    def is_valid(self) -> bool:
        """Check if all validations passed"""
        if not self._validation_results:
            self.validate_all()
        return all(result[0] for result in self._validation_results.values())
    
    def get_failures(self) -> Dict[str, str]:
        """Get failed validations"""
        if not self._validation_results:
            self.validate_all()
        return {k: v[1] for k, v in self._validation_results.items() if not v[0]}
    
    def get_report(self) -> str:
        """Get validation report as string"""
        if not self._validation_results:
            self.validate_all()
        
        lines = ["Environment Validation Report", "=" * 40]
        
        for name, (passed, message) in sorted(self._validation_results.items()):
            status = "PASS" if passed else "FAIL"
            lines.append(f"[{status}] {name}: {message}")
        
        lines.append("=" * 40)
        failures = self.get_failures()
        if failures:
            lines.append(f"FAILURES: {len(failures)}")
        else:
            lines.append("All validations passed")
        
        return "\n".join(lines)


class ModelRegistry:
    """Registry for AI models used in the platform"""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self._models: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models"""
        self.register(ModelInfo(
            model_id="stanza_grc",
            model_type=ModelType.STANZA,
            name="Ancient Greek",
            version="1.6.0",
            language="grc",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_grc_proiel",
            model_type=ModelType.STANZA,
            name="Ancient Greek (PROIEL)",
            version="1.6.0",
            language="grc",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse", "package": "proiel"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_la",
            model_type=ModelType.STANZA,
            name="Latin",
            version="1.6.0",
            language="la",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_la_proiel",
            model_type=ModelType.STANZA,
            name="Latin (PROIEL)",
            version="1.6.0",
            language="la",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse", "package": "proiel"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_cu",
            model_type=ModelType.STANZA,
            name="Old Church Slavonic",
            version="1.6.0",
            language="cu",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_got",
            model_type=ModelType.STANZA,
            name="Gothic",
            version="1.6.0",
            language="got",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse"}
        ))
        
        self.register(ModelInfo(
            model_id="stanza_xcl",
            model_type=ModelType.STANZA,
            name="Classical Armenian",
            version="1.6.0",
            language="xcl",
            capabilities=["tokenize", "pos", "lemma", "depparse"],
            config={"processors": "tokenize,pos,lemma,depparse"}
        ))
        
        self.register(ModelInfo(
            model_id="spacy_grc",
            model_type=ModelType.SPACY,
            name="Ancient Greek (spaCy)",
            version="3.0",
            language="grc",
            capabilities=["tokenize", "pos", "lemma"],
            url="https://huggingface.co/chcaa/grc_proiel_sm"
        ))
        
        self.register(ModelInfo(
            model_id="hf_bert_greek",
            model_type=ModelType.HUGGINGFACE,
            name="Greek BERT",
            version="1.0",
            language="el",
            capabilities=["embeddings", "classification"],
            url="nlpaueb/bert-base-greek-uncased-v1"
        ))
        
        self.register(ModelInfo(
            model_id="hf_ancient_greek_bert",
            model_type=ModelType.HUGGINGFACE,
            name="Ancient Greek BERT",
            version="1.0",
            language="grc",
            capabilities=["embeddings", "classification"],
            url="pranaydeeps/Ancient-Greek-BERT"
        ))
        
        self.register(ModelInfo(
            model_id="hf_latin_bert",
            model_type=ModelType.HUGGINGFACE,
            name="Latin BERT",
            version="1.0",
            language="la",
            capabilities=["embeddings", "classification"],
            url="dbmdz/bert-base-historic-multilingual-cased"
        ))
        
        self.register(ModelInfo(
            model_id="ollama_llama3",
            model_type=ModelType.OLLAMA,
            name="Llama 3",
            version="8b",
            language="multilingual",
            capabilities=["generation", "analysis", "annotation"],
            config={"model": "llama3:8b"}
        ))
        
        self.register(ModelInfo(
            model_id="ollama_mistral",
            model_type=ModelType.OLLAMA,
            name="Mistral",
            version="7b",
            language="multilingual",
            capabilities=["generation", "analysis", "annotation"],
            config={"model": "mistral:7b"}
        ))
    
    def register(self, model: ModelInfo):
        """Register a model"""
        with self._lock:
            self._models[model.model_id] = model
            
            model_path = self.path_resolver.get_model_path(model.model_type, model.name)
            model.path = str(model_path)
            model.is_downloaded = model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False
    
    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID"""
        return self._models.get(model_id)
    
    def get_by_language(self, language: str) -> List[ModelInfo]:
        """Get all models for a language"""
        return [m for m in self._models.values() if m.language == language]
    
    def get_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a type"""
        return [m for m in self._models.values() if m.model_type == model_type]
    
    def get_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get all models with a capability"""
        return [m for m in self._models.values() if capability in m.capabilities]
    
    def list_all(self) -> List[ModelInfo]:
        """List all registered models"""
        return list(self._models.values())
    
    def is_downloaded(self, model_id: str) -> bool:
        """Check if model is downloaded"""
        model = self.get(model_id)
        return model.is_downloaded if model else False
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded in memory"""
        return model_id in self._loaded_models
    
    def load_model(self, model_id: str) -> Any:
        """Load a model into memory"""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        model_info = self.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")
        
        start_time = datetime.now()
        loaded_model = None
        
        if model_info.model_type == ModelType.STANZA:
            loaded_model = self._load_stanza_model(model_info)
        elif model_info.model_type == ModelType.SPACY:
            loaded_model = self._load_spacy_model(model_info)
        elif model_info.model_type == ModelType.HUGGINGFACE:
            loaded_model = self._load_hf_model(model_info)
        elif model_info.model_type == ModelType.OLLAMA:
            loaded_model = self._load_ollama_model(model_info)
        
        if loaded_model:
            with self._lock:
                self._loaded_models[model_id] = loaded_model
                model_info.is_loaded = True
                model_info.last_used = datetime.now()
                model_info.load_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return loaded_model
    
    def _load_stanza_model(self, model_info: ModelInfo) -> Any:
        """Load a Stanza model"""
        try:
            import stanza
            
            config = model_info.config.copy()
            config["lang"] = model_info.language
            config["dir"] = str(self.path_resolver.models_dir / "stanza")
            
            if not model_info.is_downloaded:
                stanza.download(model_info.language, model_dir=config["dir"])
                model_info.is_downloaded = True
            
            nlp = stanza.Pipeline(**config)
            logger.info(f"Loaded Stanza model: {model_info.name}")
            return nlp
        except ImportError:
            logger.error("Stanza not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load Stanza model: {e}")
            return None
    
    def _load_spacy_model(self, model_info: ModelInfo) -> Any:
        """Load a spaCy model"""
        try:
            import spacy
            
            try:
                nlp = spacy.load(model_info.url or model_info.name)
            except OSError:
                logger.warning(f"Model not found, attempting download: {model_info.url}")
                return None
            
            logger.info(f"Loaded spaCy model: {model_info.name}")
            return nlp
        except ImportError:
            logger.error("spaCy not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return None
    
    def _load_hf_model(self, model_info: ModelInfo) -> Any:
        """Load a HuggingFace model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            cache_dir = str(self.path_resolver.models_dir / "transformers")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.url,
                cache_dir=cache_dir
            )
            model = AutoModel.from_pretrained(
                model_info.url,
                cache_dir=cache_dir
            )
            
            logger.info(f"Loaded HuggingFace model: {model_info.name}")
            return {"model": model, "tokenizer": tokenizer}
        except ImportError:
            logger.error("transformers not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            return None
    
    def _load_ollama_model(self, model_info: ModelInfo) -> Any:
        """Load/verify an Ollama model"""
        try:
            import requests
            
            ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            model_name = model_info.config.get("model", "llama3:8b")
            
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if model_name in model_names or model_name.split(":")[0] in [m.split(":")[0] for m in model_names]:
                    logger.info(f"Ollama model available: {model_name}")
                    return {"model_name": model_name, "url": ollama_url}
                else:
                    logger.warning(f"Ollama model not found: {model_name}")
                    return None
            else:
                logger.warning("Ollama server not responding")
                return None
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return None
    
    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        with self._lock:
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
                model_info = self.get(model_id)
                if model_info:
                    model_info.is_loaded = False
                logger.info(f"Unloaded model: {model_id}")
    
    def unload_all(self):
        """Unload all models"""
        with self._lock:
            for model_id in list(self._loaded_models.keys()):
                self.unload_model(model_id)
    
    def get_loaded_model(self, model_id: str) -> Any:
        """Get a loaded model"""
        return self._loaded_models.get(model_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "total_models": len(self._models),
            "downloaded_models": sum(1 for m in self._models.values() if m.is_downloaded),
            "loaded_models": len(self._loaded_models),
            "models_by_type": {
                mt.value: len(self.get_by_type(mt))
                for mt in ModelType
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary"""
        return {
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "loaded": list(self._loaded_models.keys()),
            "status": self.get_status()
        }


class RuntimeConfig:
    """Main runtime configuration class"""
    
    _instance: Optional["RuntimeConfig"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_dir: Optional[Path] = None):
        if self._initialized:
            return
        
        self.path_resolver = PathResolver(base_dir)
        self.environment_validator = EnvironmentValidator(self.path_resolver)
        self.model_registry = ModelRegistry(self.path_resolver)
        
        self._deployment_mode = self._detect_deployment_mode()
        self._language_configs: Dict[str, LanguageConfig] = {}
        self._settings: Dict[str, Any] = {}
        
        self._register_default_languages()
        self._load_settings()
        
        self._initialized = True
        logger.info(f"RuntimeConfig initialized: mode={self._deployment_mode.value}")
    
    def _detect_deployment_mode(self) -> DeploymentMode:
        """Detect the deployment mode"""
        if os.environ.get("DOCKER_CONTAINER"):
            return DeploymentMode.DOCKER
        
        if os.environ.get("GITHUB_ACTIONS"):
            return DeploymentMode.GITHUB_ACTIONS
        
        if os.environ.get("CLOUD_PROVIDER"):
            return DeploymentMode.CLOUD
        
        if sys.platform.startswith("win"):
            if Path("Z:/").exists():
                return DeploymentMode.WINDOWS_Z_DRIVE
            return DeploymentMode.LOCAL
        
        if Path("/etc/systemd/system").exists():
            return DeploymentMode.LINUX_SERVER
        
        return DeploymentMode.LOCAL
    
    def _register_default_languages(self):
        """Register default language configurations"""
        self.register_language(LanguageConfig(
            code="grc",
            name="Ancient Greek",
            support_level=LanguageSupport.FULL,
            stanza_model="stanza_grc_proiel",
            hf_models=["hf_ancient_greek_bert"],
            unicode_normalization="NFD",
            script="Greek",
            periods=["archaic", "classical", "hellenistic", "roman_imperial", 
                    "late_antique", "early_byzantine", "middle_byzantine", 
                    "late_byzantine", "early_modern"]
        ))
        
        self.register_language(LanguageConfig(
            code="la",
            name="Latin",
            support_level=LanguageSupport.FULL,
            stanza_model="stanza_la_proiel",
            hf_models=["hf_latin_bert"],
            unicode_normalization="NFC",
            script="Latin",
            periods=["archaic", "classical", "late_republic", "early_imperial",
                    "late_imperial", "late_antique", "medieval"]
        ))
        
        self.register_language(LanguageConfig(
            code="cu",
            name="Old Church Slavonic",
            support_level=LanguageSupport.PARTIAL,
            stanza_model="stanza_cu",
            unicode_normalization="NFC",
            script="Cyrillic",
            periods=["old_church_slavonic"]
        ))
        
        self.register_language(LanguageConfig(
            code="got",
            name="Gothic",
            support_level=LanguageSupport.PARTIAL,
            stanza_model="stanza_got",
            unicode_normalization="NFC",
            script="Gothic",
            periods=["gothic"]
        ))
        
        self.register_language(LanguageConfig(
            code="xcl",
            name="Classical Armenian",
            support_level=LanguageSupport.PARTIAL,
            stanza_model="stanza_xcl",
            unicode_normalization="NFC",
            script="Armenian",
            periods=["classical_armenian"]
        ))
        
        self.register_language(LanguageConfig(
            code="ang",
            name="Old English",
            support_level=LanguageSupport.EXPERIMENTAL,
            unicode_normalization="NFC",
            script="Latin",
            periods=["old_english"]
        ))
        
        self.register_language(LanguageConfig(
            code="non",
            name="Old Norse",
            support_level=LanguageSupport.EXPERIMENTAL,
            unicode_normalization="NFC",
            script="Latin",
            periods=["old_norse"]
        ))
        
        self.register_language(LanguageConfig(
            code="san",
            name="Sanskrit",
            support_level=LanguageSupport.EXPERIMENTAL,
            unicode_normalization="NFC",
            script="Devanagari",
            periods=["vedic", "classical_sanskrit"]
        ))
    
    def _load_settings(self):
        """Load settings from config file"""
        config_file = self.path_resolver.config_dir / "settings.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    self._settings = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")
        
        self._settings.setdefault("annotation", {
            "default_engine": "stanza",
            "batch_size": 100,
            "max_sentence_length": 500,
            "enable_gpu": False
        })
        
        self._settings.setdefault("pipeline", {
            "max_workers": 4,
            "retry_count": 3,
            "retry_delay": 5,
            "timeout": 300
        })
        
        self._settings.setdefault("server", {
            "cycle_interval": 300,
            "max_consecutive_errors": 5,
            "collection_batch_size": 10,
            "processing_batch_size": 50,
            "annotation_batch_size": 20
        })
        
        self._settings.setdefault("export", {
            "default_format": "proiel_xml",
            "include_metadata": True,
            "pretty_print": True
        })
        
        self._settings.setdefault("ui", {
            "theme": "light",
            "items_per_page": 50,
            "enable_advanced_features": True
        })
    
    def save_settings(self):
        """Save settings to config file"""
        config_file = self.path_resolver.config_dir / "settings.json"
        
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    @property
    def deployment_mode(self) -> DeploymentMode:
        """Get deployment mode"""
        return self._deployment_mode
    
    def register_language(self, config: LanguageConfig):
        """Register a language configuration"""
        self._language_configs[config.code] = config
    
    def get_language_config(self, code: str) -> Optional[LanguageConfig]:
        """Get language configuration"""
        return self._language_configs.get(code)
    
    def get_supported_languages(self) -> List[LanguageConfig]:
        """Get all supported languages"""
        return list(self._language_configs.values())
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self._settings.get(section, {}).get(key, default)
    
    def set_setting(self, section: str, key: str, value: Any):
        """Set a setting value"""
        if section not in self._settings:
            self._settings[section] = {}
        self._settings[section][key] = value
    
    def validate_environment(self) -> bool:
        """Validate the runtime environment"""
        return self.environment_validator.is_valid()
    
    def get_validation_report(self) -> str:
        """Get environment validation report"""
        return self.environment_validator.get_report()
    
    def get_status(self) -> Dict[str, Any]:
        """Get runtime configuration status"""
        return {
            "deployment_mode": self._deployment_mode.value,
            "base_dir": str(self.path_resolver.base_dir),
            "paths": self.path_resolver.to_dict(),
            "languages": {k: v.to_dict() for k, v in self._language_configs.items()},
            "models": self.model_registry.get_status(),
            "settings": self._settings,
            "environment_valid": self.validate_environment()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.get_status()


def get_runtime_config() -> RuntimeConfig:
    """Get the singleton runtime configuration instance"""
    return RuntimeConfig()


def get_path_resolver() -> PathResolver:
    """Get the path resolver from runtime config"""
    return get_runtime_config().path_resolver


def get_model_registry() -> ModelRegistry:
    """Get the model registry from runtime config"""
    return get_runtime_config().model_registry


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """Get a setting value"""
    return get_runtime_config().get_setting(section, key, default)
