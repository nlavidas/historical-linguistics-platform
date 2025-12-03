"""
HLP Ingest - Text Acquisition Package

This package provides comprehensive support for acquiring texts from
various digital humanities sources including Perseus, First1KGreek,
Project Gutenberg, and PROIEL treebanks.

Modules:
    perseus: Perseus Digital Library harvesting
    first1k_greek: First1KGreek corpus routines
    gutenberg: Project Gutenberg text fetching
    proiel_treebanks: PROIEL treebank import
    source_registry: Source registry management

University of Athens - Nikolaos Lavidas
"""

from hlp_ingest.perseus import (
    PerseusClient,
    PerseusConfig,
    PerseusText,
    fetch_perseus_text,
    search_perseus,
    list_perseus_works,
)

from hlp_ingest.first1k_greek import (
    First1KGreekClient,
    First1KGreekConfig,
    fetch_first1k_text,
    list_first1k_works,
    download_first1k_corpus,
)

from hlp_ingest.gutenberg import (
    GutenbergClient,
    GutenbergConfig,
    GutenbergText,
    fetch_gutenberg_text,
    search_gutenberg,
    list_gutenberg_languages,
)

from hlp_ingest.proiel_treebanks import (
    PROIELTreebankClient,
    PROIELTreebankConfig,
    import_proiel_treebank,
    list_proiel_treebanks,
    download_proiel_treebank,
)

from hlp_ingest.source_registry import (
    SourceRegistry,
    SourceDefinition,
    SourceType,
    register_source,
    get_source,
    list_sources,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "PerseusClient",
    "PerseusConfig",
    "PerseusText",
    "fetch_perseus_text",
    "search_perseus",
    "list_perseus_works",
    "First1KGreekClient",
    "First1KGreekConfig",
    "fetch_first1k_text",
    "list_first1k_works",
    "download_first1k_corpus",
    "GutenbergClient",
    "GutenbergConfig",
    "GutenbergText",
    "fetch_gutenberg_text",
    "search_gutenberg",
    "list_gutenberg_languages",
    "PROIELTreebankClient",
    "PROIELTreebankConfig",
    "import_proiel_treebank",
    "list_proiel_treebanks",
    "download_proiel_treebank",
    "SourceRegistry",
    "SourceDefinition",
    "SourceType",
    "register_source",
    "get_source",
    "list_sources",
]
