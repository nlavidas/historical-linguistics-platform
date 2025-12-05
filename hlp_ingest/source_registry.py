"""
HLP Ingest Source Registry - Source Registry Management

This module provides a centralized registry for managing text sources
and their configurations.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from enum import Enum

from hlp_core.models import Language, Period, Corpus

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of text sources"""
    PERSEUS = "perseus"
    FIRST1K_GREEK = "first1k_greek"
    GUTENBERG = "gutenberg"
    PROIEL = "proiel"
    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    GITHUB = "github"
    URL = "url"
    DATABASE = "database"
    CUSTOM = "custom"


class SourceStatus(Enum):
    """Status of a source"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class SourceCredentials:
    """Credentials for accessing a source"""
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without sensitive data)"""
        return {
            "has_api_key": self.api_key is not None,
            "has_username": self.username is not None,
            "has_token": self.token is not None
        }


@dataclass
class SourceDefinition:
    """Definition of a text source"""
    id: str
    name: str
    
    source_type: SourceType
    
    languages: List[Language] = field(default_factory=list)
    
    periods: List[Period] = field(default_factory=list)
    
    base_url: Optional[str] = None
    
    local_path: Optional[str] = None
    
    credentials: Optional[SourceCredentials] = None
    
    config: Dict[str, Any] = field(default_factory=dict)
    
    status: SourceStatus = SourceStatus.ACTIVE
    
    description: Optional[str] = None
    
    priority: int = 0
    
    rate_limit: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if source is available"""
        return self.status == SourceStatus.ACTIVE
    
    def supports_language(self, language: Language) -> bool:
        """Check if source supports a language"""
        if not self.languages:
            return True
        return language in self.languages
    
    def supports_period(self, period: Period) -> bool:
        """Check if source supports a period"""
        if not self.periods:
            return True
        return period in self.periods
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type.value,
            "languages": [l.value for l in self.languages],
            "periods": [p.value for p in self.periods],
            "base_url": self.base_url,
            "local_path": self.local_path,
            "status": self.status.value,
            "description": self.description,
            "priority": self.priority,
            "rate_limit": self.rate_limit
        }


class SourceRegistry:
    """Registry for managing text sources"""
    
    _instance: Optional[SourceRegistry] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sources: Dict[str, SourceDefinition] = {}
        self._handlers: Dict[SourceType, Callable] = {}
        self._initialized = True
        
        self._register_default_sources()
    
    def _register_default_sources(self):
        """Register default sources"""
        self.register(SourceDefinition(
            id="perseus",
            name="Perseus Digital Library",
            source_type=SourceType.PERSEUS,
            languages=[Language.ANCIENT_GREEK, Language.LATIN],
            periods=[
                Period.ARCHAIC, Period.CLASSICAL, Period.HELLENISTIC,
                Period.ROMAN, Period.LATE_ANTIQUE
            ],
            base_url="https://scaife.perseus.org",
            description="Perseus Digital Library - Greek and Latin texts",
            priority=10
        ))
        
        self.register(SourceDefinition(
            id="first1k_greek",
            name="First1KGreek",
            source_type=SourceType.FIRST1K_GREEK,
            languages=[Language.ANCIENT_GREEK],
            periods=[
                Period.ARCHAIC, Period.CLASSICAL, Period.HELLENISTIC,
                Period.ROMAN, Period.LATE_ANTIQUE, Period.BYZANTINE
            ],
            base_url="https://github.com/OpenGreekAndLatin/First1KGreek",
            description="First Thousand Years of Greek - Open Greek and Latin",
            priority=9
        ))
        
        self.register(SourceDefinition(
            id="gutenberg",
            name="Project Gutenberg",
            source_type=SourceType.GUTENBERG,
            languages=[
                Language.ANCIENT_GREEK, Language.LATIN,
                Language.ENGLISH, Language.GERMAN, Language.FRENCH
            ],
            base_url="https://gutendex.com",
            description="Project Gutenberg - Free eBooks",
            priority=5
        ))
        
        self.register(SourceDefinition(
            id="proiel",
            name="PROIEL Treebank",
            source_type=SourceType.PROIEL,
            languages=[
                Language.ANCIENT_GREEK, Language.LATIN,
                Language.GOTHIC, Language.OLD_CHURCH_SLAVONIC,
                Language.ARMENIAN
            ],
            periods=[
                Period.CLASSICAL, Period.HELLENISTIC, Period.ROMAN,
                Period.LATE_ANTIQUE, Period.MEDIEVAL
            ],
            base_url="https://github.com/proiel/proiel-treebank",
            description="PROIEL Treebank - Syntactically annotated texts",
            priority=10
        ))
    
    def register(self, source: SourceDefinition) -> bool:
        """Register a source"""
        if source.id in self._sources:
            logger.warning(f"Source {source.id} already registered, updating")
        
        self._sources[source.id] = source
        logger.info(f"Registered source: {source.id}")
        return True
    
    def unregister(self, source_id: str) -> bool:
        """Unregister a source"""
        if source_id in self._sources:
            del self._sources[source_id]
            logger.info(f"Unregistered source: {source_id}")
            return True
        return False
    
    def get(self, source_id: str) -> Optional[SourceDefinition]:
        """Get a source by ID"""
        return self._sources.get(source_id)
    
    def list_all(self) -> List[SourceDefinition]:
        """List all registered sources"""
        return list(self._sources.values())
    
    def list_by_type(self, source_type: SourceType) -> List[SourceDefinition]:
        """List sources by type"""
        return [s for s in self._sources.values() if s.source_type == source_type]
    
    def list_by_language(self, language: Language) -> List[SourceDefinition]:
        """List sources supporting a language"""
        return [s for s in self._sources.values() if s.supports_language(language)]
    
    def list_by_period(self, period: Period) -> List[SourceDefinition]:
        """List sources supporting a period"""
        return [s for s in self._sources.values() if s.supports_period(period)]
    
    def list_active(self) -> List[SourceDefinition]:
        """List active sources"""
        return [s for s in self._sources.values() if s.is_available()]
    
    def get_best_source(
        self,
        language: Optional[Language] = None,
        period: Optional[Period] = None
    ) -> Optional[SourceDefinition]:
        """Get best source for criteria"""
        candidates = self.list_active()
        
        if language:
            candidates = [s for s in candidates if s.supports_language(language)]
        
        if period:
            candidates = [s for s in candidates if s.supports_period(period)]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda s: s.priority)
    
    def register_handler(
        self,
        source_type: SourceType,
        handler: Callable
    ):
        """Register a handler for a source type"""
        self._handlers[source_type] = handler
        logger.info(f"Registered handler for {source_type.value}")
    
    def get_handler(self, source_type: SourceType) -> Optional[Callable]:
        """Get handler for a source type"""
        return self._handlers.get(source_type)
    
    def set_status(self, source_id: str, status: SourceStatus) -> bool:
        """Set source status"""
        source = self.get(source_id)
        if source:
            source.status = status
            return True
        return False
    
    def update_config(
        self,
        source_id: str,
        config: Dict[str, Any]
    ) -> bool:
        """Update source configuration"""
        source = self.get(source_id)
        if source:
            source.config.update(config)
            return True
        return False
    
    def set_credentials(
        self,
        source_id: str,
        credentials: SourceCredentials
    ) -> bool:
        """Set source credentials"""
        source = self.get(source_id)
        if source:
            source.credentials = credentials
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary"""
        return {
            "sources": {
                source_id: source.to_dict()
                for source_id, source in self._sources.items()
            },
            "handlers": list(self._handlers.keys())
        }
    
    def clear(self):
        """Clear all sources"""
        self._sources.clear()
        self._handlers.clear()
        logger.info("Cleared source registry")
    
    def reset(self):
        """Reset to default sources"""
        self.clear()
        self._register_default_sources()
        logger.info("Reset source registry to defaults")


_registry: Optional[SourceRegistry] = None


def get_registry() -> SourceRegistry:
    """Get the global source registry"""
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
    return _registry


def register_source(source: SourceDefinition) -> bool:
    """Register a source in the global registry"""
    return get_registry().register(source)


def get_source(source_id: str) -> Optional[SourceDefinition]:
    """Get a source from the global registry"""
    return get_registry().get(source_id)


def list_sources(
    source_type: Optional[SourceType] = None,
    language: Optional[Language] = None,
    period: Optional[Period] = None,
    active_only: bool = True
) -> List[SourceDefinition]:
    """List sources from the global registry"""
    registry = get_registry()
    
    if source_type:
        sources = registry.list_by_type(source_type)
    elif language:
        sources = registry.list_by_language(language)
    elif period:
        sources = registry.list_by_period(period)
    elif active_only:
        sources = registry.list_active()
    else:
        sources = registry.list_all()
    
    return sources
