"""
HLP Semantics - Semantic Analysis Package

This package provides comprehensive support for semantic analysis,
including semantic role labeling, named entity recognition, and
information structure annotation.

Modules:
    srl: Semantic role labeling
    ner: Named entity recognition
    information_structure: Topic/focus annotation

University of Athens - Nikolaos Lavidas
"""

from hlp_semantics.srl import (
    SemanticRole,
    SemanticFrame,
    SRLAnnotator,
    SRLConfig,
    SRLResult,
    annotate_semantic_roles,
    extract_predicate_arguments,
)

from hlp_semantics.ner import (
    EntityType,
    NamedEntity,
    NERAnnotator,
    NERConfig,
    NERResult,
    annotate_entities,
    extract_entities,
)

from hlp_semantics.information_structure import (
    InformationStatus,
    TopicType,
    FocusType,
    InformationUnit,
    ISAnnotator,
    ISConfig,
    ISResult,
    annotate_information_structure,
    identify_topic,
    identify_focus,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "SemanticRole",
    "SemanticFrame",
    "SRLAnnotator",
    "SRLConfig",
    "SRLResult",
    "annotate_semantic_roles",
    "extract_predicate_arguments",
    "EntityType",
    "NamedEntity",
    "NERAnnotator",
    "NERConfig",
    "NERResult",
    "annotate_entities",
    "extract_entities",
    "InformationStatus",
    "TopicType",
    "FocusType",
    "InformationUnit",
    "ISAnnotator",
    "ISConfig",
    "ISResult",
    "annotate_information_structure",
    "identify_topic",
    "identify_focus",
]
