"""
HLP Ingest PROIEL Treebanks - PROIEL Treebank Import

This module provides utilities for importing and processing PROIEL
treebank data, including the official PROIEL corpus and related
treebanks.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import xml.etree.ElementTree as ET

from hlp_core.models import (
    Language, Period, Document, Corpus, Sentence, Token,
    MorphologyAnnotation, SyntaxAnnotation, Case, Tense, Mood, Voice
)
from hlp_io.proiel_xml import PROIELReader, PROIELConfig

logger = logging.getLogger(__name__)


PROIEL_GITHUB_URL = "https://github.com/proiel/proiel-treebank"
PROIEL_RAW_URL = "https://raw.githubusercontent.com/proiel/proiel-treebank/master"
PROIEL_RELEASES_URL = "https://github.com/proiel/proiel-treebank/releases"

SYNTACTICUS_URL = "https://syntacticus.org"
SYNTACTICUS_API_URL = "https://syntacticus.org/api"


PROIEL_TREEBANKS = {
    "greek-nt": {
        "name": "Greek New Testament",
        "language": Language.ANCIENT_GREEK,
        "period": Period.ROMAN,
        "file": "greek-nt.xml",
        "description": "The Greek New Testament (Nestle-Aland 27th edition)"
    },
    "latin-nt": {
        "name": "Latin New Testament (Vulgate)",
        "language": Language.LATIN,
        "period": Period.LATE_LATIN,
        "file": "latin-nt.xml",
        "description": "The Latin Vulgate New Testament"
    },
    "greek-herodotus": {
        "name": "Herodotus' Histories",
        "language": Language.ANCIENT_GREEK,
        "period": Period.CLASSICAL,
        "file": "greek-herodotus.xml",
        "description": "Herodotus' Histories (selections)"
    },
    "latin-caesar": {
        "name": "Caesar's Gallic War",
        "language": Language.LATIN,
        "period": Period.CLASSICAL_LATIN,
        "file": "latin-caesar.xml",
        "description": "Caesar's De Bello Gallico"
    },
    "latin-cicero": {
        "name": "Cicero's Letters",
        "language": Language.LATIN,
        "period": Period.CLASSICAL_LATIN,
        "file": "latin-cicero.xml",
        "description": "Cicero's Letters to Atticus"
    },
    "gothic-nt": {
        "name": "Gothic New Testament",
        "language": Language.GOTHIC,
        "period": Period.LATE_ANTIQUE,
        "file": "gothic-nt.xml",
        "description": "Wulfila's Gothic Bible translation"
    },
    "armenian-nt": {
        "name": "Armenian New Testament",
        "language": Language.ARMENIAN,
        "period": Period.LATE_ANTIQUE,
        "file": "armenian-nt.xml",
        "description": "Classical Armenian New Testament"
    },
    "old-church-slavonic": {
        "name": "Old Church Slavonic",
        "language": Language.OLD_CHURCH_SLAVONIC,
        "period": Period.MEDIEVAL,
        "file": "old-church-slavonic.xml",
        "description": "Old Church Slavonic texts"
    }
}


@dataclass
class PROIELTreebankConfig:
    """Configuration for PROIEL treebank client"""
    github_token: Optional[str] = None
    
    cache_dir: Optional[str] = None
    
    rate_limit: float = 0.5
    
    timeout: float = 60.0
    
    max_retries: int = 3
    
    user_agent: str = "HLP-Platform/1.0 (Historical Linguistics Platform)"
    
    validate_xml: bool = True


@dataclass
class PROIELTreebank:
    """Represents a PROIEL treebank"""
    id: str
    name: str
    
    language: Language
    
    period: Period
    
    sentences: List[Sentence] = field(default_factory=list)
    
    documents: List[Document] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_corpus(self) -> Corpus:
        """Convert to HLP Corpus"""
        return Corpus(
            id=self.id,
            name=self.name,
            language=self.language,
            documents=self.documents,
            metadata={
                "source": "proiel",
                "period": self.period.value,
                **self.metadata
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics"""
        if self.statistics:
            return self.statistics
        
        total_tokens = 0
        total_sentences = len(self.sentences)
        pos_counts = {}
        case_counts = {}
        
        for sentence in self.sentences:
            for token in sentence.tokens:
                total_tokens += 1
                
                if token.pos:
                    pos_counts[token.pos] = pos_counts.get(token.pos, 0) + 1
                
                if token.morphology and token.morphology.case:
                    case_str = token.morphology.case.value
                    case_counts[case_str] = case_counts.get(case_str, 0) + 1
        
        self.statistics = {
            "total_sentences": total_sentences,
            "total_tokens": total_tokens,
            "pos_distribution": pos_counts,
            "case_distribution": case_counts,
            "avg_sentence_length": total_tokens / total_sentences if total_sentences > 0 else 0
        }
        
        return self.statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language.value,
            "period": self.period.value,
            "sentence_count": len(self.sentences),
            "document_count": len(self.documents),
            "statistics": self.get_statistics()
        }


class PROIELTreebankClient:
    """Client for PROIEL treebanks"""
    
    def __init__(self, config: Optional[PROIELTreebankConfig] = None):
        self.config = config or PROIELTreebankConfig()
        self._last_request_time = 0.0
        self._session = None
        self._proiel_reader = PROIELReader()
    
    def _get_session(self):
        """Get or create HTTP session"""
        if self._session is None:
            import requests
            self._session = requests.Session()
            headers = {"User-Agent": self.config.user_agent}
            if self.config.github_token:
                headers["Authorization"] = f"token {self.config.github_token}"
            self._session.headers.update(headers)
        return self._session
    
    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Make HTTP request with retries"""
        session = self._get_session()
        
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                response = session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {url}")
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def list_treebanks(self) -> List[Dict[str, Any]]:
        """List available PROIEL treebanks"""
        return [
            {
                "id": tb_id,
                "name": tb_info["name"],
                "language": tb_info["language"].value,
                "period": tb_info["period"].value,
                "description": tb_info["description"]
            }
            for tb_id, tb_info in PROIEL_TREEBANKS.items()
        ]
    
    def import_treebank(
        self,
        treebank_id: str,
        from_url: bool = True,
        local_path: Optional[str] = None
    ) -> Optional[PROIELTreebank]:
        """Import a PROIEL treebank"""
        if treebank_id not in PROIEL_TREEBANKS:
            logger.error(f"Unknown treebank: {treebank_id}")
            return None
        
        tb_info = PROIEL_TREEBANKS[treebank_id]
        
        if from_url:
            xml_content = self._fetch_treebank_xml(tb_info["file"])
        elif local_path:
            xml_content = self._load_local_xml(local_path)
        else:
            logger.error("Must specify either from_url=True or local_path")
            return None
        
        if not xml_content:
            return None
        
        return self._parse_treebank(treebank_id, tb_info, xml_content)
    
    def _fetch_treebank_xml(self, filename: str) -> Optional[str]:
        """Fetch treebank XML from GitHub"""
        url = f"{PROIEL_RAW_URL}/{filename}"
        return self._make_request(url)
    
    def _load_local_xml(self, path: str) -> Optional[str]:
        """Load treebank XML from local file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load local XML: {e}")
            return None
    
    def _parse_treebank(
        self,
        treebank_id: str,
        tb_info: Dict[str, Any],
        xml_content: str
    ) -> PROIELTreebank:
        """Parse treebank XML"""
        treebank = PROIELTreebank(
            id=treebank_id,
            name=tb_info["name"],
            language=tb_info["language"],
            period=tb_info["period"],
            metadata={"description": tb_info["description"]}
        )
        
        try:
            root = ET.fromstring(xml_content)
            
            for source_elem in root.findall('.//source'):
                doc = self._parse_source(source_elem, tb_info["language"])
                if doc:
                    treebank.documents.append(doc)
                    treebank.sentences.extend(doc.sentences)
            
            if not treebank.sentences:
                for sent_elem in root.findall('.//sentence'):
                    sentence = self._parse_sentence(sent_elem, tb_info["language"])
                    if sentence:
                        treebank.sentences.append(sentence)
            
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        
        return treebank
    
    def _parse_source(
        self,
        source_elem: ET.Element,
        language: Language
    ) -> Optional[Document]:
        """Parse a source element"""
        source_id = source_elem.get("id", "")
        title = source_elem.get("title", "Unknown")
        author = source_elem.get("author", "Unknown")
        
        sentences = []
        
        for div_elem in source_elem.findall('.//div'):
            for sent_elem in div_elem.findall('.//sentence'):
                sentence = self._parse_sentence(sent_elem, language)
                if sentence:
                    sentences.append(sentence)
        
        if not sentences:
            for sent_elem in source_elem.findall('.//sentence'):
                sentence = self._parse_sentence(sent_elem, language)
                if sentence:
                    sentences.append(sentence)
        
        if not sentences:
            return None
        
        text = " ".join(s.text for s in sentences if s.text)
        
        return Document(
            id=source_id,
            title=title,
            author=author,
            language=language,
            text=text,
            sentences=sentences
        )
    
    def _parse_sentence(
        self,
        sent_elem: ET.Element,
        language: Language
    ) -> Optional[Sentence]:
        """Parse a sentence element"""
        sent_id = sent_elem.get("id", "")
        
        tokens = []
        
        for token_elem in sent_elem.findall('.//token'):
            token = self._parse_token(token_elem)
            if token:
                tokens.append(token)
        
        if not tokens:
            return None
        
        text = " ".join(t.form for t in tokens if t.form)
        
        return Sentence(
            id=sent_id,
            text=text,
            tokens=tokens,
            language=language
        )
    
    def _parse_token(self, token_elem: ET.Element) -> Optional[Token]:
        """Parse a token element"""
        token_id = token_elem.get("id", "")
        form = token_elem.get("form", "")
        lemma = token_elem.get("lemma", "")
        pos = token_elem.get("part-of-speech", "")
        
        if not form:
            return None
        
        morphology = self._parse_morphology(token_elem)
        syntax = self._parse_syntax(token_elem)
        
        return Token(
            id=token_id,
            form=form,
            lemma=lemma,
            pos=pos,
            morphology=morphology,
            syntax=syntax
        )
    
    def _parse_morphology(self, token_elem: ET.Element) -> Optional[MorphologyAnnotation]:
        """Parse morphology from token element"""
        morph_str = token_elem.get("morphology", "")
        
        if not morph_str or len(morph_str) < 8:
            return None
        
        case_map = {
            "n": Case.NOMINATIVE,
            "g": Case.GENITIVE,
            "d": Case.DATIVE,
            "a": Case.ACCUSATIVE,
            "v": Case.VOCATIVE,
            "b": Case.ABLATIVE,
            "i": Case.INSTRUMENTAL,
            "l": Case.LOCATIVE,
        }
        
        tense_map = {
            "p": Tense.PRESENT,
            "i": Tense.IMPERFECT,
            "f": Tense.FUTURE,
            "a": Tense.AORIST,
            "r": Tense.PERFECT,
            "l": Tense.PLUPERFECT,
            "t": Tense.FUTURE_PERFECT,
        }
        
        mood_map = {
            "i": Mood.INDICATIVE,
            "s": Mood.SUBJUNCTIVE,
            "o": Mood.OPTATIVE,
            "m": Mood.IMPERATIVE,
            "n": Mood.INFINITIVE,
            "p": Mood.PARTICIPLE,
            "g": Mood.GERUND,
            "d": Mood.GERUNDIVE,
            "u": Mood.SUPINE,
        }
        
        voice_map = {
            "a": Voice.ACTIVE,
            "p": Voice.PASSIVE,
            "m": Voice.MIDDLE,
            "e": Voice.MEDIO_PASSIVE,
        }
        
        case = case_map.get(morph_str[6].lower()) if len(morph_str) > 6 else None
        tense = tense_map.get(morph_str[2].lower()) if len(morph_str) > 2 else None
        mood = mood_map.get(morph_str[3].lower()) if len(morph_str) > 3 else None
        voice = voice_map.get(morph_str[4].lower()) if len(morph_str) > 4 else None
        
        return MorphologyAnnotation(
            case=case,
            tense=tense,
            mood=mood,
            voice=voice,
            proiel_morph=morph_str
        )
    
    def _parse_syntax(self, token_elem: ET.Element) -> Optional[SyntaxAnnotation]:
        """Parse syntax from token element"""
        head = token_elem.get("head-id", "")
        relation = token_elem.get("relation", "")
        
        if not head and not relation:
            return None
        
        return SyntaxAnnotation(
            head=head,
            deprel=relation
        )
    
    def download_treebank(
        self,
        treebank_id: str,
        output_dir: Union[str, Path]
    ) -> bool:
        """Download treebank to local directory"""
        if treebank_id not in PROIEL_TREEBANKS:
            logger.error(f"Unknown treebank: {treebank_id}")
            return False
        
        tb_info = PROIEL_TREEBANKS[treebank_id]
        
        xml_content = self._fetch_treebank_xml(tb_info["file"])
        if not xml_content:
            return False
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / tb_info["file"]
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
            logger.info(f"Downloaded treebank to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save treebank: {e}")
            return False
    
    def import_from_syntacticus(
        self,
        corpus_id: str
    ) -> Optional[PROIELTreebank]:
        """Import treebank from Syntacticus"""
        url = f"{SYNTACTICUS_API_URL}/corpora/{corpus_id}"
        
        content = self._make_request(url)
        if not content:
            return None
        
        logger.info(f"Syntacticus import for {corpus_id} not yet implemented")
        return None


def import_proiel_treebank(
    treebank_id: str,
    config: Optional[PROIELTreebankConfig] = None
) -> Optional[PROIELTreebank]:
    """Import a PROIEL treebank"""
    client = PROIELTreebankClient(config)
    return client.import_treebank(treebank_id)


def list_proiel_treebanks() -> List[Dict[str, Any]]:
    """List available PROIEL treebanks"""
    client = PROIELTreebankClient()
    return client.list_treebanks()


def download_proiel_treebank(
    treebank_id: str,
    output_dir: Union[str, Path],
    config: Optional[PROIELTreebankConfig] = None
) -> bool:
    """Download a PROIEL treebank"""
    client = PROIELTreebankClient(config)
    return client.download_treebank(treebank_id, output_dir)
