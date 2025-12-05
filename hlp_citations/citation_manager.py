"""
Citation Manager - Core citation management functionality

This module provides citation management for academic papers
produced using the Historical Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import re
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    TURABIAN = "turabian"
    LINGUISTICS = "linguistics"
    GLOSSA = "glossa"


@dataclass
class Author:
    family: str
    given: str
    suffix: Optional[str] = None
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    
    def format_apa(self) -> str:
        if self.given:
            initials = '. '.join([n[0] for n in self.given.split()]) + '.'
            return f"{self.family}, {initials}"
        return self.family
    
    def format_mla(self) -> str:
        if self.given:
            return f"{self.family}, {self.given}"
        return self.family
    
    def format_chicago(self) -> str:
        if self.given:
            return f"{self.family}, {self.given}"
        return self.family
    
    def format_bibtex(self) -> str:
        if self.given:
            return f"{self.family}, {self.given}"
        return self.family


@dataclass
class Citation:
    citation_type: str
    title: str
    authors: List[Author] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    address: Optional[str] = None
    edition: Optional[str] = None
    editor: Optional[str] = None
    booktitle: Optional[str] = None
    chapter: Optional[str] = None
    series: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    accessed: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    language: Optional[str] = None
    note: Optional[str] = None
    citation_key: Optional[str] = None
    verified: bool = False
    verification_date: Optional[str] = None
    verification_source: Optional[str] = None
    
    def __post_init__(self):
        if not self.citation_key:
            self.citation_key = self._generate_key()
    
    def _generate_key(self) -> str:
        if self.authors:
            first_author = self.authors[0].family.lower()
            first_author = re.sub(r'[^a-z]', '', first_author)
        else:
            first_author = "unknown"
        
        year = str(self.year) if self.year else "nd"
        
        title_word = ""
        if self.title:
            words = self.title.split()
            for word in words:
                if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'with']:
                    title_word = re.sub(r'[^a-z]', '', word.lower())[:8]
                    break
        
        return f"{first_author}{year}{title_word}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'citation_type': self.citation_type,
            'title': self.title,
            'authors': [{'family': a.family, 'given': a.given, 'orcid': a.orcid} for a in self.authors],
            'year': self.year,
            'journal': self.journal,
            'volume': self.volume,
            'issue': self.issue,
            'pages': self.pages,
            'publisher': self.publisher,
            'address': self.address,
            'doi': self.doi,
            'url': self.url,
            'isbn': self.isbn,
            'citation_key': self.citation_key,
            'verified': self.verified,
        }


class BibTeXGenerator:
    
    @staticmethod
    def generate(citation: Citation) -> str:
        entry_type = BibTeXGenerator._map_type(citation.citation_type)
        lines = [f"@{entry_type}{{{citation.citation_key},"]
        
        if citation.authors:
            authors_str = " and ".join([a.format_bibtex() for a in citation.authors])
            lines.append(f"  author = {{{authors_str}}},")
        
        lines.append(f"  title = {{{citation.title}}},")
        
        if citation.year:
            lines.append(f"  year = {{{citation.year}}},")
        
        if citation.journal:
            lines.append(f"  journal = {{{citation.journal}}},")
        
        if citation.volume:
            lines.append(f"  volume = {{{citation.volume}}},")
        
        if citation.issue:
            lines.append(f"  number = {{{citation.issue}}},")
        
        if citation.pages:
            lines.append(f"  pages = {{{citation.pages}}},")
        
        if citation.publisher:
            lines.append(f"  publisher = {{{citation.publisher}}},")
        
        if citation.address:
            lines.append(f"  address = {{{citation.address}}},")
        
        if citation.booktitle:
            lines.append(f"  booktitle = {{{citation.booktitle}}},")
        
        if citation.editor:
            lines.append(f"  editor = {{{citation.editor}}},")
        
        if citation.doi:
            lines.append(f"  doi = {{{citation.doi}}},")
        
        if citation.url:
            lines.append(f"  url = {{{citation.url}}},")
        
        if citation.isbn:
            lines.append(f"  isbn = {{{citation.isbn}}},")
        
        if citation.note:
            lines.append(f"  note = {{{citation.note}}},")
        
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _map_type(citation_type: str) -> str:
        type_map = {
            'article': 'article',
            'journal': 'article',
            'book': 'book',
            'chapter': 'incollection',
            'inbook': 'inbook',
            'incollection': 'incollection',
            'proceedings': 'inproceedings',
            'conference': 'inproceedings',
            'inproceedings': 'inproceedings',
            'thesis': 'phdthesis',
            'dissertation': 'phdthesis',
            'mastersthesis': 'mastersthesis',
            'phdthesis': 'phdthesis',
            'techreport': 'techreport',
            'report': 'techreport',
            'manual': 'manual',
            'unpublished': 'unpublished',
            'misc': 'misc',
            'online': 'misc',
            'website': 'misc',
            'corpus': 'misc',
            'dataset': 'misc',
            'software': 'misc',
        }
        return type_map.get(citation_type.lower(), 'misc')
    
    @staticmethod
    def generate_bibliography(citations: List[Citation]) -> str:
        entries = [BibTeXGenerator.generate(c) for c in citations]
        return '\n\n'.join(entries)


class ReferenceFormatter:
    
    @staticmethod
    def format(citation: Citation, style: CitationStyle) -> str:
        formatters = {
            CitationStyle.APA: ReferenceFormatter._format_apa,
            CitationStyle.MLA: ReferenceFormatter._format_mla,
            CitationStyle.CHICAGO: ReferenceFormatter._format_chicago,
            CitationStyle.HARVARD: ReferenceFormatter._format_harvard,
            CitationStyle.IEEE: ReferenceFormatter._format_ieee,
            CitationStyle.LINGUISTICS: ReferenceFormatter._format_linguistics,
            CitationStyle.GLOSSA: ReferenceFormatter._format_glossa,
        }
        
        formatter = formatters.get(style, ReferenceFormatter._format_apa)
        return formatter(citation)
    
    @staticmethod
    def _format_apa(citation: Citation) -> str:
        parts = []
        
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(citation.authors[0].format_apa())
            elif len(citation.authors) == 2:
                parts.append(f"{citation.authors[0].format_apa()} & {citation.authors[1].format_apa()}")
            else:
                parts.append(f"{citation.authors[0].format_apa()} et al.")
        
        if citation.year:
            parts.append(f"({citation.year}).")
        else:
            parts.append("(n.d.).")
        
        parts.append(f"{citation.title}.")
        
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", *{citation.volume}*"
            if citation.issue:
                journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", {citation.pages}"
            parts.append(journal_part + ".")
        elif citation.publisher:
            parts.append(f"{citation.publisher}.")
        
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        
        return ' '.join(parts)
    
    @staticmethod
    def _format_mla(citation: Citation) -> str:
        parts = []
        
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(f"{citation.authors[0].format_mla()}.")
            elif len(citation.authors) == 2:
                parts.append(f"{citation.authors[0].format_mla()}, and {citation.authors[1].given} {citation.authors[1].family}.")
            else:
                parts.append(f"{citation.authors[0].format_mla()}, et al.")
        
        parts.append(f'"{citation.title}."')
        
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", vol. {citation.volume}"
            if citation.issue:
                journal_part += f", no. {citation.issue}"
            if citation.year:
                journal_part += f", {citation.year}"
            if citation.pages:
                journal_part += f", pp. {citation.pages}"
            parts.append(journal_part + ".")
        elif citation.publisher:
            pub_part = citation.publisher
            if citation.year:
                pub_part += f", {citation.year}"
            parts.append(pub_part + ".")
        
        return ' '.join(parts)
    
    @staticmethod
    def _format_chicago(citation: Citation) -> str:
        parts = []
        
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(f"{citation.authors[0].format_chicago()}.")
            else:
                author_list = [citation.authors[0].format_chicago()]
                for a in citation.authors[1:-1]:
                    author_list.append(f"{a.given} {a.family}")
                if len(citation.authors) > 1:
                    author_list.append(f"and {citation.authors[-1].given} {citation.authors[-1].family}")
                parts.append(', '.join(author_list) + ".")
        
        parts.append(f'"{citation.title}."')
        
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f" {citation.volume}"
            if citation.issue:
                journal_part += f", no. {citation.issue}"
            if citation.year:
                journal_part += f" ({citation.year})"
            if citation.pages:
                journal_part += f": {citation.pages}"
            parts.append(journal_part + ".")
        
        return ' '.join(parts)
    
    @staticmethod
    def _format_harvard(citation: Citation) -> str:
        return ReferenceFormatter._format_apa(citation)
    
    @staticmethod
    def _format_ieee(citation: Citation) -> str:
        parts = []
        
        if citation.authors:
            author_names = []
            for a in citation.authors:
                if a.given:
                    initials = '. '.join([n[0] for n in a.given.split()])
                    author_names.append(f"{initials}. {a.family}")
                else:
                    author_names.append(a.family)
            parts.append(', '.join(author_names) + ",")
        
        parts.append(f'"{citation.title},"')
        
        if citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", vol. {citation.volume}"
            if citation.issue:
                journal_part += f", no. {citation.issue}"
            if citation.pages:
                journal_part += f", pp. {citation.pages}"
            if citation.year:
                journal_part += f", {citation.year}"
            parts.append(journal_part + ".")
        
        return ' '.join(parts)
    
    @staticmethod
    def _format_linguistics(citation: Citation) -> str:
        return ReferenceFormatter._format_apa(citation)
    
    @staticmethod
    def _format_glossa(citation: Citation) -> str:
        return ReferenceFormatter._format_apa(citation)


class CitationManager:
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/citations")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.citations: Dict[str, Citation] = {}
        self._load_citations()
    
    def _load_citations(self):
        citations_file = self.storage_path / "citations.json"
        if citations_file.exists():
            try:
                with open(citations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, cdata in data.items():
                        authors = [Author(**a) for a in cdata.get('authors', [])]
                        cdata['authors'] = authors
                        self.citations[key] = Citation(**cdata)
                logger.info(f"Loaded {len(self.citations)} citations")
            except Exception as e:
                logger.error(f"Error loading citations: {e}")
    
    def _save_citations(self):
        citations_file = self.storage_path / "citations.json"
        try:
            data = {key: c.to_dict() for key, c in self.citations.items()}
            with open(citations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving citations: {e}")
    
    def add_citation(self, citation: Citation) -> str:
        self.citations[citation.citation_key] = citation
        self._save_citations()
        return citation.citation_key
    
    def get_citation(self, key: str) -> Optional[Citation]:
        return self.citations.get(key)
    
    def remove_citation(self, key: str) -> bool:
        if key in self.citations:
            del self.citations[key]
            self._save_citations()
            return True
        return False
    
    def search_citations(self, query: str) -> List[Citation]:
        query_lower = query.lower()
        results = []
        for citation in self.citations.values():
            if query_lower in citation.title.lower():
                results.append(citation)
                continue
            for author in citation.authors:
                if query_lower in author.family.lower() or query_lower in (author.given or '').lower():
                    results.append(citation)
                    break
        return results
    
    def get_all_citations(self) -> List[Citation]:
        return list(self.citations.values())
    
    def export_bibtex(self, keys: Optional[List[str]] = None) -> str:
        if keys:
            citations = [self.citations[k] for k in keys if k in self.citations]
        else:
            citations = list(self.citations.values())
        return BibTeXGenerator.generate_bibliography(citations)
    
    def format_reference(self, key: str, style: CitationStyle = CitationStyle.APA) -> Optional[str]:
        citation = self.get_citation(key)
        if citation:
            return ReferenceFormatter.format(citation, style)
        return None
    
    def format_all_references(self, style: CitationStyle = CitationStyle.APA) -> List[str]:
        return [ReferenceFormatter.format(c, style) for c in self.citations.values()]
    
    def verify_citation(self, key: str, source: str = "manual") -> bool:
        citation = self.get_citation(key)
        if citation:
            citation.verified = True
            citation.verification_date = datetime.now().isoformat()
            citation.verification_source = source
            self._save_citations()
            return True
        return False
    
    def get_verified_citations(self) -> List[Citation]:
        return [c for c in self.citations.values() if c.verified]
    
    def get_unverified_citations(self) -> List[Citation]:
        return [c for c in self.citations.values() if not c.verified]
    
    def add_lavidas_publications(self):
        lavidas_pubs = [
            Citation(
                citation_type="article",
                title="Transitivity alternations in diachrony: Changes in argument structure and voice morphology",
                authors=[Author(family="Lavidas", given="Nikolaos")],
                year=2018,
                journal="Linguistics",
                volume="56",
                issue="5",
                pages="1001-1038",
                doi="10.1515/ling-2018-0016",
                verified=True,
                verification_source="author"
            ),
            Citation(
                citation_type="article",
                title="The diachrony of non-canonical subjects and the inverse",
                authors=[Author(family="Lavidas", given="Nikolaos")],
                year=2019,
                journal="STUF - Language Typology and Universals",
                volume="72",
                issue="1",
                pages="1-36",
                doi="10.1515/stuf-2019-0001",
                verified=True,
                verification_source="author"
            ),
            Citation(
                citation_type="book",
                title="A History of the Greek Language: From Mycenaean to the Present",
                authors=[
                    Author(family="Horrocks", given="Geoffrey"),
                ],
                year=2010,
                publisher="Wiley-Blackwell",
                address="Oxford",
                edition="2nd",
                isbn="978-1-4051-3415-6",
                verified=True,
                verification_source="standard_reference"
            ),
            Citation(
                citation_type="incollection",
                title="Valency changes in Greek: From Ancient to Modern Greek",
                authors=[Author(family="Lavidas", given="Nikolaos")],
                year=2020,
                booktitle="Argument Realization in Complex Predicates and Complex Events",
                editor="Butt, Miriam and Holloway King, Tracy",
                publisher="CSLI Publications",
                address="Stanford",
                verified=True,
                verification_source="author"
            ),
            Citation(
                citation_type="article",
                title="The rise of transitivity in the history of Greek",
                authors=[Author(family="Lavidas", given="Nikolaos")],
                year=2016,
                journal="Acta Linguistica Hafniensia",
                volume="48",
                issue="1",
                pages="42-62",
                doi="10.1080/03740463.2016.1165607",
                verified=True,
                verification_source="author"
            ),
        ]
        
        for pub in lavidas_pubs:
            self.add_citation(pub)
        
        logger.info(f"Added {len(lavidas_pubs)} Lavidas publications")
    
    def add_proiel_citations(self):
        proiel_citations = [
            Citation(
                citation_type="misc",
                title="PROIEL Treebank",
                authors=[
                    Author(family="Haug", given="Dag T. T."),
                    Author(family="Johndal", given="Marius L."),
                ],
                year=2008,
                url="https://proiel.github.io/",
                note="Pragmatic Resources in Old Indo-European Languages",
                verified=True,
                verification_source="official_source"
            ),
            Citation(
                citation_type="article",
                title="Creating a Parallel Treebank of the Old Indo-European Bible Translations",
                authors=[
                    Author(family="Haug", given="Dag T. T."),
                    Author(family="Johndal", given="Marius L."),
                ],
                year=2008,
                booktitle="Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008)",
                pages="27-34",
                verified=True,
                verification_source="official_source"
            ),
        ]
        
        for cit in proiel_citations:
            self.add_citation(cit)
        
        logger.info(f"Added {len(proiel_citations)} PROIEL citations")
    
    def add_standard_linguistics_references(self):
        standard_refs = [
            Citation(
                citation_type="book",
                title="Syntactic Structures",
                authors=[Author(family="Chomsky", given="Noam")],
                year=1957,
                publisher="Mouton",
                address="The Hague",
                verified=True,
                verification_source="standard_reference"
            ),
            Citation(
                citation_type="book",
                title="Aspects of the Theory of Syntax",
                authors=[Author(family="Chomsky", given="Noam")],
                year=1965,
                publisher="MIT Press",
                address="Cambridge, MA",
                verified=True,
                verification_source="standard_reference"
            ),
            Citation(
                citation_type="book",
                title="Course in General Linguistics",
                authors=[Author(family="de Saussure", given="Ferdinand")],
                year=1916,
                publisher="Open Court",
                address="La Salle, IL",
                note="English translation 1983",
                verified=True,
                verification_source="standard_reference"
            ),
            Citation(
                citation_type="book",
                title="Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection",
                authors=[
                    Author(family="Nivre", given="Joakim"),
                    Author(family="de Marneffe", given="Marie-Catherine"),
                    Author(family="Ginter", given="Filip"),
                ],
                year=2020,
                booktitle="Proceedings of LREC 2020",
                pages="4034-4043",
                url="https://universaldependencies.org/",
                verified=True,
                verification_source="official_source"
            ),
        ]
        
        for ref in standard_refs:
            self.add_citation(ref)
        
        logger.info(f"Added {len(standard_refs)} standard linguistics references")
