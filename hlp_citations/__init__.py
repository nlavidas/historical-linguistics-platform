"""
HLP Citations - Academic Citation Manager

This module provides citation management for academic papers
produced using the Historical Linguistics Platform.

Features:
- BibTeX generation
- Citation verification
- Reference formatting (APA, MLA, Chicago, etc.)
- DOI lookup and validation
- Integration with Zotero, Mendeley
- PROIEL and Syntacticus citation standards

University of Athens - Nikolaos Lavidas
"""

from hlp_citations.citation_manager import (
    CitationManager,
    Citation,
    CitationStyle,
    BibTeXGenerator,
    ReferenceFormatter,
)

from hlp_citations.corpus_citations import (
    CorpusCitationGenerator,
    PROIELCitation,
    SyntacticusCitation,
    PerseusDigitalLibraryCitation,
)

from hlp_citations.doi_resolver import (
    DOIResolver,
    DOIValidator,
)

__all__ = [
    'CitationManager',
    'Citation',
    'CitationStyle',
    'BibTeXGenerator',
    'ReferenceFormatter',
    'CorpusCitationGenerator',
    'PROIELCitation',
    'SyntacticusCitation',
    'PerseusDigitalLibraryCitation',
    'DOIResolver',
    'DOIValidator',
]
