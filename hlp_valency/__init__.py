"""
HLP Valency - Valency Extraction and Analysis Package

This package provides comprehensive support for extracting, analyzing,
and managing valency patterns from annotated corpora, with special
focus on diachronic analysis of Indo-European languages.

Modules:
    pattern_extractor: Extract valency frames from annotated text
    pattern_normalization: Normalize and canonicalize patterns
    lexicon_builder: Build valency lexicons
    reports: Generate valency reports

University of Athens - Nikolaos Lavidas
"""

from hlp_valency.pattern_extractor import (
    ValencyExtractor,
    ExtractionConfig,
    ExtractionResult,
    extract_valency_frames,
    extract_from_sentence,
    extract_from_document,
    extract_from_corpus,
)

from hlp_valency.pattern_normalization import (
    PatternNormalizer,
    NormalizationConfig,
    normalize_pattern,
    normalize_frame,
    canonicalize_arguments,
)

from hlp_valency.lexicon_builder import (
    LexiconBuilder,
    ValencyLexicon,
    LexiconEntry,
    build_lexicon,
    merge_lexicons,
    export_lexicon,
)

from hlp_valency.reports import (
    ValencyReporter,
    ReportConfig,
    generate_valency_report,
    generate_diachronic_report,
    generate_comparison_report,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "ValencyExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "extract_valency_frames",
    "extract_from_sentence",
    "extract_from_document",
    "extract_from_corpus",
    "PatternNormalizer",
    "NormalizationConfig",
    "normalize_pattern",
    "normalize_frame",
    "canonicalize_arguments",
    "LexiconBuilder",
    "ValencyLexicon",
    "LexiconEntry",
    "build_lexicon",
    "merge_lexicons",
    "export_lexicon",
    "ValencyReporter",
    "ReportConfig",
    "generate_valency_report",
    "generate_diachronic_report",
    "generate_comparison_report",
]
