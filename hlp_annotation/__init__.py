"""
HLP Annotation - Multi-AI Annotation Engine Package

This package provides a unified interface for multiple NLP annotation
engines including Stanza, spaCy, HuggingFace Transformers, and Ollama.

Modules:
    base_engine: Abstract base class for annotation engines
    stanza_engine: Stanza-based annotation pipeline
    spacy_engine: spaCy-based annotation pipeline
    hf_transformers_engine: HuggingFace Transformers integration
    ollama_engine: Ollama local LLM wrapper
    ensemble: Multi-engine ensemble annotation

University of Athens - Nikolaos Lavidas
"""

from hlp_annotation.base_engine import (
    AnnotationEngine,
    AnnotationCapability,
    AnnotationResult,
    AnnotationConfig,
    EngineStatus,
)

from hlp_annotation.stanza_engine import (
    StanzaEngine,
    StanzaConfig,
)

from hlp_annotation.spacy_engine import (
    SpacyEngine,
    SpacyConfig,
)

from hlp_annotation.hf_transformers_engine import (
    HuggingFaceEngine,
    HuggingFaceConfig,
)

from hlp_annotation.ollama_engine import (
    OllamaEngine,
    OllamaConfig,
)

from hlp_annotation.ensemble import (
    EnsembleEngine,
    EnsembleConfig,
    VotingStrategy,
    EngineWeight,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "AnnotationEngine",
    "AnnotationCapability",
    "AnnotationResult",
    "AnnotationConfig",
    "EngineStatus",
    "StanzaEngine",
    "StanzaConfig",
    "SpacyEngine",
    "SpacyConfig",
    "HuggingFaceEngine",
    "HuggingFaceConfig",
    "OllamaEngine",
    "OllamaConfig",
    "EnsembleEngine",
    "EnsembleConfig",
    "VotingStrategy",
    "EngineWeight",
]
