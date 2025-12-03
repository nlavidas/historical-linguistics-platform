"""
HLP Annotation Base Engine - Abstract Base Class for Annotation Engines

This module provides the abstract base class and common interfaces
for all annotation engines in the platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Tuple, Union,
    Iterator, Callable, Type, Generic, TypeVar
)
from datetime import datetime
from enum import Enum, auto
from contextlib import contextmanager

from hlp_core.models import (
    Corpus, Document, Sentence, Token,
    MorphologicalFeatures, SyntacticRelation,
    Language, AnnotationStatus, PartOfSpeech, DependencyRelation,
    SemanticRole, NamedEntity, InformationStructure
)

logger = logging.getLogger(__name__)


class AnnotationCapability(Enum):
    """Capabilities that annotation engines can provide"""
    TOKENIZATION = "tokenization"
    SENTENCE_SPLITTING = "sentence_splitting"
    POS_TAGGING = "pos_tagging"
    LEMMATIZATION = "lemmatization"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"
    DEPENDENCY_PARSING = "dependency_parsing"
    CONSTITUENCY_PARSING = "constituency_parsing"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    SEMANTIC_ROLE_LABELING = "semantic_role_labeling"
    COREFERENCE_RESOLUTION = "coreference_resolution"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    EMBEDDINGS = "embeddings"
    TRANSLATION = "translation"
    GENERATION = "generation"


class EngineStatus(Enum):
    """Status of an annotation engine"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ProcessingMode(Enum):
    """Processing modes for annotation"""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class AnnotationConfig:
    """Base configuration for annotation engines"""
    language: str = "grc"
    
    batch_size: int = 32
    max_sequence_length: int = 512
    
    use_gpu: bool = False
    device: str = "cpu"
    num_threads: int = 4
    
    timeout: float = 300.0
    retry_count: int = 3
    retry_delay: float = 1.0
    
    cache_enabled: bool = True
    cache_size: int = 10000
    
    verbose: bool = False
    log_level: str = "INFO"
    
    custom_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "language": self.language,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "use_gpu": self.use_gpu,
            "device": self.device,
            "num_threads": self.num_threads,
            "timeout": self.timeout,
            "cache_enabled": self.cache_enabled,
            "custom_options": self.custom_options
        }


@dataclass
class AnnotationResult:
    """Result of an annotation operation"""
    success: bool
    
    sentences: List[Sentence] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    
    entities: List[NamedEntity] = field(default_factory=list)
    semantic_roles: List[SemanticRole] = field(default_factory=list)
    
    embeddings: Optional[Any] = None
    
    raw_output: Optional[Any] = None
    
    processing_time_ms: float = 0.0
    tokens_processed: int = 0
    sentences_processed: int = 0
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    engine_name: Optional[str] = None
    engine_version: Optional[str] = None
    
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "sentences_processed": self.sentences_processed,
            "tokens_processed": self.tokens_processed,
            "processing_time_ms": self.processing_time_ms,
            "engine_name": self.engine_name,
            "confidence": self.confidence,
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class EngineMetrics:
    """Metrics for annotation engine performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    total_tokens_processed: int = 0
    total_sentences_processed: int = 0
    
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float("inf")
    max_processing_time_ms: float = 0.0
    
    cache_hits: int = 0
    cache_misses: int = 0
    
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    
    def update(self, result: AnnotationResult):
        """Update metrics with result"""
        self.total_requests += 1
        self.last_request_time = datetime.now()
        
        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if result.errors:
                self.last_error_time = datetime.now()
                self.last_error_message = result.errors[0]
        
        self.total_tokens_processed += result.tokens_processed
        self.total_sentences_processed += result.sentences_processed
        
        self.total_processing_time_ms += result.processing_time_ms
        self.min_processing_time_ms = min(self.min_processing_time_ms, result.processing_time_ms)
        self.max_processing_time_ms = max(self.max_processing_time_ms, result.processing_time_ms)
        
        if self.total_requests > 0:
            self.average_processing_time_ms = self.total_processing_time_ms / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "total_tokens_processed": self.total_tokens_processed,
            "total_sentences_processed": self.total_sentences_processed,
            "average_processing_time_ms": self.average_processing_time_ms,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }


class AnnotationCache:
    """LRU cache for annotation results"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[AnnotationResult, datetime]] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
    
    def _make_key(self, text: str, capabilities: List[AnnotationCapability], language: str) -> str:
        """Create cache key"""
        import hashlib
        cap_str = ",".join(sorted(c.value for c in capabilities))
        key_str = f"{language}:{cap_str}:{text}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self, 
        text: str, 
        capabilities: List[AnnotationCapability],
        language: str
    ) -> Optional[AnnotationResult]:
        """Get cached result"""
        key = self._make_key(text, capabilities, language)
        
        with self._lock:
            if key in self._cache:
                result, _ = self._cache[key]
                self._access_order.remove(key)
                self._access_order.append(key)
                return result
        
        return None
    
    def put(
        self,
        text: str,
        capabilities: List[AnnotationCapability],
        language: str,
        result: AnnotationResult
    ):
        """Put result in cache"""
        key = self._make_key(text, capabilities, language)
        
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = (result, datetime.now())
            self._access_order.append(key)
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    @property
    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)


class AnnotationEngine(ABC):
    """Abstract base class for annotation engines"""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        self.config = config or AnnotationConfig()
        self._status = EngineStatus.UNINITIALIZED
        self._metrics = EngineMetrics()
        self._cache = AnnotationCache(max_size=self.config.cache_size) if self.config.cache_enabled else None
        self._lock = threading.Lock()
        self._model = None
        self._initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Engine version"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AnnotationCapability]:
        """List of capabilities this engine provides"""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List of supported language codes"""
        pass
    
    @property
    def status(self) -> EngineStatus:
        """Get engine status"""
        return self._status
    
    @property
    def metrics(self) -> EngineMetrics:
        """Get engine metrics"""
        return self._metrics
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self._status == EngineStatus.READY
    
    def supports_capability(self, capability: AnnotationCapability) -> bool:
        """Check if engine supports a capability"""
        return capability in self.capabilities
    
    def supports_language(self, language: str) -> bool:
        """Check if engine supports a language"""
        return language in self.supported_languages
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the engine"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown the engine"""
        pass
    
    @abstractmethod
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Internal text processing method"""
        pass
    
    def annotate_text(
        self,
        text: str,
        capabilities: Optional[List[AnnotationCapability]] = None
    ) -> AnnotationResult:
        """Annotate text with specified capabilities"""
        if not self.is_ready:
            if not self.initialize():
                return AnnotationResult(
                    success=False,
                    errors=["Engine initialization failed"]
                )
        
        if capabilities is None:
            capabilities = self.capabilities
        
        unsupported = [c for c in capabilities if not self.supports_capability(c)]
        if unsupported:
            return AnnotationResult(
                success=False,
                errors=[f"Unsupported capabilities: {[c.value for c in unsupported]}"]
            )
        
        if self._cache:
            cached = self._cache.get(text, capabilities, self.config.language)
            if cached:
                self._metrics.cache_hits += 1
                return cached
            self._metrics.cache_misses += 1
        
        start_time = time.time()
        
        try:
            with self._lock:
                self._status = EngineStatus.PROCESSING
            
            result = self._process_text(text, capabilities)
            result.engine_name = self.name
            result.engine_version = self.version
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            if self._cache and result.success:
                self._cache.put(text, capabilities, self.config.language, result)
            
            self._metrics.update(result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Annotation error in {self.name}")
            result = AnnotationResult(
                success=False,
                errors=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000
            )
            self._metrics.update(result)
            return result
        
        finally:
            with self._lock:
                if self._status == EngineStatus.PROCESSING:
                    self._status = EngineStatus.READY
    
    def annotate_sentence(
        self,
        sentence: Sentence,
        capabilities: Optional[List[AnnotationCapability]] = None
    ) -> AnnotationResult:
        """Annotate a sentence object"""
        text = sentence.text or " ".join(t.form for t in sentence.tokens)
        result = self.annotate_text(text, capabilities)
        
        if result.success and result.tokens:
            if len(result.tokens) == len(sentence.tokens):
                for orig_token, new_token in zip(sentence.tokens, result.tokens):
                    if new_token.morphology:
                        orig_token.morphology = new_token.morphology
                    if new_token.syntax:
                        orig_token.syntax = new_token.syntax
                    if new_token.lemma:
                        orig_token.lemma = new_token.lemma
        
        return result
    
    def annotate_document(
        self,
        document: Document,
        capabilities: Optional[List[AnnotationCapability]] = None
    ) -> AnnotationResult:
        """Annotate a document"""
        total_result = AnnotationResult(success=True, engine_name=self.name)
        start_time = time.time()
        
        for sentence in document.sentences:
            result = self.annotate_sentence(sentence, capabilities)
            
            total_result.sentences_processed += 1
            total_result.tokens_processed += len(sentence.tokens)
            
            if not result.success:
                total_result.warnings.extend(result.errors)
            
            total_result.entities.extend(result.entities)
            total_result.semantic_roles.extend(result.semantic_roles)
        
        total_result.processing_time_ms = (time.time() - start_time) * 1000
        return total_result
    
    def annotate_corpus(
        self,
        corpus: Corpus,
        capabilities: Optional[List[AnnotationCapability]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> AnnotationResult:
        """Annotate an entire corpus"""
        total_result = AnnotationResult(success=True, engine_name=self.name)
        start_time = time.time()
        
        total_sentences = sum(len(d.sentences) for d in corpus.documents)
        processed = 0
        
        for document in corpus.documents:
            result = self.annotate_document(document, capabilities)
            
            total_result.sentences_processed += result.sentences_processed
            total_result.tokens_processed += result.tokens_processed
            total_result.warnings.extend(result.warnings)
            total_result.entities.extend(result.entities)
            total_result.semantic_roles.extend(result.semantic_roles)
            
            processed += len(document.sentences)
            if progress_callback:
                progress_callback(processed, total_sentences)
        
        total_result.processing_time_ms = (time.time() - start_time) * 1000
        return total_result
    
    def annotate_batch(
        self,
        texts: List[str],
        capabilities: Optional[List[AnnotationCapability]] = None
    ) -> List[AnnotationResult]:
        """Annotate a batch of texts"""
        results = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = [self.annotate_text(text, capabilities) for text in batch]
            results.extend(batch_results)
        
        return results
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self._status.value,
            "capabilities": [c.value for c in self.capabilities],
            "supported_languages": self.supported_languages,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict(),
            "cache_size": self._cache.size if self._cache else 0
        }
    
    def clear_cache(self):
        """Clear the annotation cache"""
        if self._cache:
            self._cache.clear()
    
    def reset_metrics(self):
        """Reset engine metrics"""
        self._metrics = EngineMetrics()
    
    @contextmanager
    def session(self):
        """Context manager for engine session"""
        try:
            if not self.is_ready:
                self.initialize()
            yield self
        finally:
            pass
    
    def __enter__(self):
        """Enter context manager"""
        if not self.is_ready:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        pass


class DummyEngine(AnnotationEngine):
    """Dummy engine for testing"""
    
    @property
    def name(self) -> str:
        return "DummyEngine"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        return [
            AnnotationCapability.TOKENIZATION,
            AnnotationCapability.POS_TAGGING
        ]
    
    @property
    def supported_languages(self) -> List[str]:
        return ["grc", "la", "en"]
    
    def initialize(self) -> bool:
        self._status = EngineStatus.READY
        self._initialized = True
        return True
    
    def shutdown(self):
        self._status = EngineStatus.SHUTDOWN
        self._initialized = False
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        tokens = []
        for idx, word in enumerate(text.split(), start=1):
            token = Token(
                id=idx,
                form=word,
                lemma=word.lower(),
                morphology=MorphologicalFeatures(pos=PartOfSpeech.X)
            )
            tokens.append(token)
        
        return AnnotationResult(
            success=True,
            tokens=tokens,
            tokens_processed=len(tokens)
        )


def create_engine(
    engine_type: str,
    config: Optional[AnnotationConfig] = None
) -> AnnotationEngine:
    """Factory function to create annotation engines"""
    engine_map = {
        "dummy": DummyEngine,
    }
    
    engine_class = engine_map.get(engine_type.lower())
    if engine_class is None:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    return engine_class(config)
