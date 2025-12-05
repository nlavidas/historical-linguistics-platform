"""
HLP Agents - Multi-Agent Orchestration for Linguistic Analysis

This module provides multi-agent orchestration for annotation,
analysis, and corpus building using open-source AI engines.

Supported AI Engines (all open-source, community-driven):
- Stanza (Stanford NLP)
- spaCy (Explosion AI)
- HuggingFace Transformers
- Ollama (local LLMs)
- NLTK
- Flair
- UDPipe
- CLTK (Classical Language Toolkit)
- Trankit (Multilingual NLP)
- Gensim (Topic modeling)
- FastText (Word embeddings)
- Sentence Transformers (Sentence embeddings)
- Polyglot (Multilingual NLP)
- LatinCy (Latin NLP)
- GreCy (Greek NLP)

Agent Types:
- AnnotationAgent: POS tagging, dependency parsing, NER
- ValencyAgent: Valency pattern extraction and analysis
- DiachronicAgent: Diachronic change detection
- QualityAgent: Annotation quality assurance
- CollectionAgent: Text collection and preprocessing
- CitationAgent: Citation management and verification

University of Athens - Nikolaos Lavidas
"""

from hlp_agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentStatus,
    AgentTask,
    AgentResult,
)

from hlp_agents.annotation_agent import (
    AnnotationAgent,
    MultiEngineAnnotationAgent,
)

from hlp_agents.valency_agent import (
    ValencyAgent,
    ValencyExtractionAgent,
)

from hlp_agents.diachronic_agent import (
    DiachronicAgent,
    ChangeDetectionAgent,
)

from hlp_agents.orchestrator import (
    AgentOrchestrator,
    Pipeline,
    PipelineStep,
)

from hlp_agents.opensource_engines import (
    OpenSourceEngine,
    OpenSourceEngineRegistry,
    MultiEngineProcessor,
    CLTKEngine,
    UDPipeEngine,
    TrankitEngine,
    FlairEngine,
    NLTKEngine,
    GensimEngine,
    SentenceTransformersEngine,
    PolyglotEngine,
    FastTextEngine,
    LatinCyEngine,
    GreCyEngine,
    HuggingFaceMultilingualEngine,
    EngineCategory,
    LanguageSupport,
    get_best_engines_for_task,
)

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'AgentStatus',
    'AgentTask',
    'AgentResult',
    'AnnotationAgent',
    'MultiEngineAnnotationAgent',
    'ValencyAgent',
    'ValencyExtractionAgent',
    'DiachronicAgent',
    'ChangeDetectionAgent',
    'AgentOrchestrator',
    'Pipeline',
    'PipelineStep',
    'OpenSourceEngine',
    'OpenSourceEngineRegistry',
    'MultiEngineProcessor',
    'CLTKEngine',
    'UDPipeEngine',
    'TrankitEngine',
    'FlairEngine',
    'NLTKEngine',
    'GensimEngine',
    'SentenceTransformersEngine',
    'PolyglotEngine',
    'FastTextEngine',
    'LatinCyEngine',
    'GreCyEngine',
    'HuggingFaceMultilingualEngine',
    'EngineCategory',
    'LanguageSupport',
    'get_best_engines_for_task',
]
