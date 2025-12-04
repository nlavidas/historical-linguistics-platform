"""
Corpus Citations - Citation generators for linguistic corpora

This module provides citation generation for specific corpora
used in historical linguistics research.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from hlp_citations.citation_manager import Citation, Author, CitationStyle, ReferenceFormatter

logger = logging.getLogger(__name__)


@dataclass
class PROIELCitation:
    treebank_name: str
    language: str
    version: Optional[str] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        return Citation(
            citation_type="misc",
            title=f"PROIEL Treebank: {self.treebank_name}",
            authors=[
                Author(family="Haug", given="Dag T. T."),
                Author(family="Johndal", given="Marius L."),
            ],
            year=2008,
            url="https://proiel.github.io/",
            note=f"Language: {self.language}. Version: {self.version or 'latest'}. Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}",
            verified=True,
            verification_source="proiel_official"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="inproceedings",
            title="Creating a Parallel Treebank of the Old Indo-European Bible Translations",
            authors=[
                Author(family="Haug", given="Dag T. T."),
                Author(family="Johndal", given="Marius L."),
            ],
            year=2008,
            booktitle="Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008)",
            pages="27-34",
            address="Marrakech, Morocco",
            verified=True,
            verification_source="proiel_official"
        )


@dataclass
class SyntacticusCitation:
    corpus_name: str
    language: str
    version: Optional[str] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        return Citation(
            citation_type="misc",
            title=f"Syntacticus: {self.corpus_name}",
            authors=[
                Author(family="Eckhoff", given="Hanne Martine"),
                Author(family="Bech", given="Kristin"),
                Author(family="Bouma", given="Gerlof"),
                Author(family="Eide", given="Kristine"),
                Author(family="Haug", given="Dag T. T."),
                Author(family="Haugen", given="Odd Einar"),
                Author(family="Johndal", given="Marius"),
            ],
            year=2018,
            url="https://syntacticus.org/",
            note=f"Language: {self.language}. Version: {self.version or 'latest'}. Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}",
            verified=True,
            verification_source="syntacticus_official"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="inproceedings",
            title="The PROIEL Treebank Family: A Standard for Early Attestations of Indo-European Languages",
            authors=[
                Author(family="Eckhoff", given="Hanne Martine"),
                Author(family="Bech", given="Kristin"),
                Author(family="Bouma", given="Gerlof"),
                Author(family="Eide", given="Kristine"),
                Author(family="Haug", given="Dag T. T."),
                Author(family="Haugen", given="Odd Einar"),
                Author(family="Johndal", given="Marius"),
            ],
            year=2018,
            booktitle="Language Resources and Evaluation",
            volume="52",
            pages="29-65",
            doi="10.1007/s10579-017-9388-5",
            verified=True,
            verification_source="syntacticus_official"
        )


@dataclass
class PerseusDigitalLibraryCitation:
    work_title: str
    author: str
    language: str
    urn: Optional[str] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        return Citation(
            citation_type="misc",
            title=f"{self.work_title}",
            authors=[Author(family=self.author, given="")],
            url=f"https://www.perseus.tufts.edu/hopper/text?doc={self.urn}" if self.urn else "https://www.perseus.tufts.edu/",
            note=f"Perseus Digital Library. Language: {self.language}. Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}",
            verified=True,
            verification_source="perseus_digital_library"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="misc",
            title="Perseus Digital Library",
            authors=[Author(family="Crane", given="Gregory R.")],
            year=1987,
            publisher="Tufts University",
            address="Medford, MA",
            url="https://www.perseus.tufts.edu/",
            note="Ongoing project since 1987",
            verified=True,
            verification_source="perseus_official"
        )


@dataclass
class First1KGreekCitation:
    work_title: str
    author: str
    urn: Optional[str] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        return Citation(
            citation_type="misc",
            title=f"{self.work_title}",
            authors=[Author(family=self.author, given="")],
            url="https://opengreekandlatin.github.io/First1KGreek/",
            note=f"First1KGreek Project. URN: {self.urn or 'N/A'}. Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}",
            verified=True,
            verification_source="first1kgreek"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="misc",
            title="First Thousand Years of Greek",
            authors=[
                Author(family="Babeu", given="Alison"),
                Author(family="Crane", given="Gregory R."),
            ],
            year=2011,
            url="https://opengreekandlatin.github.io/First1KGreek/",
            note="Open Greek and Latin Project",
            verified=True,
            verification_source="first1kgreek_official"
        )


@dataclass
class UniversalDependenciesCitation:
    treebank_name: str
    language: str
    version: Optional[str] = None
    contributors: Optional[List[str]] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        authors = []
        if self.contributors:
            for contrib in self.contributors[:3]:
                parts = contrib.split()
                if len(parts) >= 2:
                    authors.append(Author(family=parts[-1], given=' '.join(parts[:-1])))
                else:
                    authors.append(Author(family=contrib, given=""))
        
        return Citation(
            citation_type="misc",
            title=f"Universal Dependencies: {self.treebank_name}",
            authors=authors if authors else [Author(family="Universal Dependencies Consortium", given="")],
            url=f"https://universaldependencies.org/treebanks/{self.treebank_name.lower().replace(' ', '_')}/",
            note=f"Language: {self.language}. Version: {self.version or 'latest'}. Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}",
            verified=True,
            verification_source="universal_dependencies"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="inproceedings",
            title="Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection",
            authors=[
                Author(family="Nivre", given="Joakim"),
                Author(family="de Marneffe", given="Marie-Catherine"),
                Author(family="Ginter", given="Filip"),
                Author(family="Hajic", given="Jan"),
                Author(family="Manning", given="Christopher D."),
                Author(family="Pyysalo", given="Sampo"),
                Author(family="Schuster", given="Sebastian"),
                Author(family="Tyers", given="Francis"),
                Author(family="Zeman", given="Daniel"),
            ],
            year=2020,
            booktitle="Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020)",
            pages="4034-4043",
            address="Marseille, France",
            url="https://universaldependencies.org/",
            verified=True,
            verification_source="ud_official"
        )


@dataclass
class GutenbergCitation:
    work_title: str
    author: str
    ebook_number: int
    language: str
    original_publication_year: Optional[int] = None
    translator: Optional[str] = None
    accessed_date: Optional[str] = None
    
    def to_citation(self) -> Citation:
        authors = [Author(family=self.author, given="")]
        note_parts = [f"Project Gutenberg EBook #{self.ebook_number}"]
        
        if self.translator:
            note_parts.append(f"Translated by {self.translator}")
        
        if self.original_publication_year:
            note_parts.append(f"Original publication: {self.original_publication_year}")
        
        note_parts.append(f"Accessed: {self.accessed_date or datetime.now().strftime('%Y-%m-%d')}")
        
        return Citation(
            citation_type="misc",
            title=self.work_title,
            authors=authors,
            year=self.original_publication_year,
            url=f"https://www.gutenberg.org/ebooks/{self.ebook_number}",
            note=". ".join(note_parts),
            language=self.language,
            verified=True,
            verification_source="project_gutenberg"
        )
    
    @staticmethod
    def get_standard_citation() -> Citation:
        return Citation(
            citation_type="misc",
            title="Project Gutenberg",
            authors=[Author(family="Hart", given="Michael S.")],
            year=1971,
            url="https://www.gutenberg.org/",
            note="Free eBooks since 1971",
            verified=True,
            verification_source="gutenberg_official"
        )


class CorpusCitationGenerator:
    
    def __init__(self):
        self.standard_citations: Dict[str, Citation] = {}
        self._load_standard_citations()
    
    def _load_standard_citations(self):
        self.standard_citations['proiel'] = PROIELCitation.get_standard_citation()
        self.standard_citations['syntacticus'] = SyntacticusCitation.get_standard_citation()
        self.standard_citations['perseus'] = PerseusDigitalLibraryCitation.get_standard_citation()
        self.standard_citations['first1kgreek'] = First1KGreekCitation.get_standard_citation()
        self.standard_citations['universal_dependencies'] = UniversalDependenciesCitation.get_standard_citation()
        self.standard_citations['gutenberg'] = GutenbergCitation.get_standard_citation()
    
    def get_standard_citation(self, corpus_type: str) -> Optional[Citation]:
        return self.standard_citations.get(corpus_type.lower())
    
    def generate_proiel_citation(
        self,
        treebank_name: str,
        language: str,
        version: Optional[str] = None
    ) -> Citation:
        cit = PROIELCitation(
            treebank_name=treebank_name,
            language=language,
            version=version,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_syntacticus_citation(
        self,
        corpus_name: str,
        language: str,
        version: Optional[str] = None
    ) -> Citation:
        cit = SyntacticusCitation(
            corpus_name=corpus_name,
            language=language,
            version=version,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_perseus_citation(
        self,
        work_title: str,
        author: str,
        language: str,
        urn: Optional[str] = None
    ) -> Citation:
        cit = PerseusDigitalLibraryCitation(
            work_title=work_title,
            author=author,
            language=language,
            urn=urn,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_first1kgreek_citation(
        self,
        work_title: str,
        author: str,
        urn: Optional[str] = None
    ) -> Citation:
        cit = First1KGreekCitation(
            work_title=work_title,
            author=author,
            urn=urn,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_ud_citation(
        self,
        treebank_name: str,
        language: str,
        version: Optional[str] = None,
        contributors: Optional[List[str]] = None
    ) -> Citation:
        cit = UniversalDependenciesCitation(
            treebank_name=treebank_name,
            language=language,
            version=version,
            contributors=contributors,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_gutenberg_citation(
        self,
        work_title: str,
        author: str,
        ebook_number: int,
        language: str,
        original_publication_year: Optional[int] = None,
        translator: Optional[str] = None
    ) -> Citation:
        cit = GutenbergCitation(
            work_title=work_title,
            author=author,
            ebook_number=ebook_number,
            language=language,
            original_publication_year=original_publication_year,
            translator=translator,
            accessed_date=datetime.now().strftime('%Y-%m-%d')
        )
        return cit.to_citation()
    
    def generate_hlp_citation(self) -> Citation:
        return Citation(
            citation_type="software",
            title="Historical Linguistics Platform (HLP)",
            authors=[
                Author(
                    family="Lavidas",
                    given="Nikolaos",
                    affiliation="National and Kapodistrian University of Athens"
                ),
            ],
            year=2024,
            url="https://github.com/nlavidas/historical-linguistics-platform",
            note="Diachronic corpus analysis platform for historical linguistics research. Funded by HFRI.",
            verified=True,
            verification_source="author"
        )
    
    def get_all_standard_citations(self) -> List[Citation]:
        return list(self.standard_citations.values())
    
    def format_corpus_acknowledgment(
        self,
        corpora_used: List[str],
        style: CitationStyle = CitationStyle.APA
    ) -> str:
        acknowledgment_parts = []
        
        for corpus in corpora_used:
            citation = self.get_standard_citation(corpus)
            if citation:
                formatted = ReferenceFormatter.format(citation, style)
                acknowledgment_parts.append(formatted)
        
        if acknowledgment_parts:
            return "This research uses the following corpora and resources:\n\n" + "\n\n".join(acknowledgment_parts)
        
        return ""
