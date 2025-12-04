"""
HLP Search - Full-text search and retrieval for historical linguistics texts

This module provides comprehensive search capabilities for:
- Full-text search across all collected texts
- Filters by language, period, genre, author
- Fuzzy matching for historical spelling variants
- Lemma-based search for morphologically rich languages
- Parallel text search for translations

University of Athens - Nikolaos Lavidas
"""

from hlp_search.text_search import (
    TextSearchEngine,
    SearchQuery,
    SearchResult,
    SearchFilter,
)

from hlp_search.translation_tracker import (
    TranslationTracker,
    TranslationType,
    TranslationPair,
    TranslationChain,
)

from hlp_search.influential_texts import (
    InfluentialTextsRegistry,
    TextInfluence,
    InfluenceType,
)

__all__ = [
    'TextSearchEngine',
    'SearchQuery',
    'SearchResult',
    'SearchFilter',
    'TranslationTracker',
    'TranslationType',
    'TranslationPair',
    'TranslationChain',
    'InfluentialTextsRegistry',
    'TextInfluence',
    'InfluenceType',
]
