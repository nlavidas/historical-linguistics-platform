"""
HLP Prompts - Devin-style prompt interface for linguistic queries

This module provides a natural language interface for:
- Corpus queries and searches
- Linguistic analysis requests
- Valency pattern extraction
- Diachronic comparisons
- Translation tracking

University of Athens - Nikolaos Lavidas
"""

from hlp_prompts.prompt_engine import (
    PromptEngine,
    PromptResult,
    PromptType,
)

from hlp_prompts.linguistic_prompts import (
    LinguisticPromptLibrary,
    PromptTemplate,
)

__all__ = [
    'PromptEngine',
    'PromptResult',
    'PromptType',
    'LinguisticPromptLibrary',
    'PromptTemplate',
]
