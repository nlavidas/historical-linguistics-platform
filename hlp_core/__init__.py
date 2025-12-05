"""
HLP Core - Historical Linguistics Platform Core Module

This package provides the foundational data models, database management,
configuration, and logging infrastructure for the entire platform.

Modules:
    models: Core domain objects (Corpus, Document, Sentence, Token, etc.)
    db: Database abstraction layer with schema management
    config_runtime: Runtime configuration and environment management
    logging_monitoring: Structured logging and metrics collection

University of Athens - Nikolaos Lavidas
PROIEL/Syntacticus-style Diachronic Linguistics Platform
"""

from hlp_core.models import (
    Corpus,
    Document,
    Sentence,
    Token,
    MorphologicalFeatures,
    SyntacticRelation,
    ValencyPattern,
    ValencyFrame,
    Lexeme,
    LemmaSense,
    DiachronicStage,
    SourceMetadata,
    AnnotationLayer,
    SemanticRole,
    NamedEntity,
    InformationStructure,
    TreeNode,
    DependencyTree,
    PROIELNode,
    CoNLLUToken,
)

from hlp_core.db import (
    DatabaseManager,
    ConnectionPool,
    SchemaManager,
    QueryBuilder,
    TransactionManager,
)

from hlp_core.config_runtime import (
    RuntimeConfig,
    EnvironmentValidator,
    PathResolver,
    ModelRegistry,
)

from hlp_core.logging_monitoring import (
    PlatformLogger,
    MetricsCollector,
    PerformanceMonitor,
    AlertManager,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"
__institution__ = "University of Athens"

__all__ = [
    "Corpus",
    "Document", 
    "Sentence",
    "Token",
    "MorphologicalFeatures",
    "SyntacticRelation",
    "ValencyPattern",
    "ValencyFrame",
    "Lexeme",
    "LemmaSense",
    "DiachronicStage",
    "SourceMetadata",
    "AnnotationLayer",
    "SemanticRole",
    "NamedEntity",
    "InformationStructure",
    "TreeNode",
    "DependencyTree",
    "PROIELNode",
    "CoNLLUToken",
    "DatabaseManager",
    "ConnectionPool",
    "SchemaManager",
    "QueryBuilder",
    "TransactionManager",
    "RuntimeConfig",
    "EnvironmentValidator",
    "PathResolver",
    "ModelRegistry",
    "PlatformLogger",
    "MetricsCollector",
    "PerformanceMonitor",
    "AlertManager",
]
