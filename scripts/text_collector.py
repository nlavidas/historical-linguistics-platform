#!/usr/bin/env python3
"""
Text Collection and Preprocessing Pipeline
Comprehensive system for collecting, preprocessing, and managing historical linguistic texts
Focus: Greek (Ancient, Koine, Byzantine, Modern) with English comparative corpus

Features:
- Multi-source text collection (Perseus, PROIEL, Wikisource, Project Gutenberg, etc.)
- Intralingual translations (different Greek periods)
- Interlingual translations (Greek-English parallel texts)
- Influential and representative text selection
- 24/7 automated collection daemon
- Preprocessing pipeline with normalization, tokenization, sentence splitting
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
import sqlite3
import requests
import threading
import queue
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "database_path": "corpus_platform.db",
    "raw_texts_dir": "raw_texts",
    "processed_texts_dir": "processed_texts",
    "parallel_texts_dir": "parallel_texts",
    "max_workers": 4,
    "request_delay": 1.0,
    "max_retries": 3,
    "chunk_size": 10000,
    "user_agent": "DiachronicLinguisticsPlatform/2.0 (Research; +https://github.com/nlavidas/historical-linguistics-platform)"
}

# Greek text sources
GREEK_SOURCES = {
    "perseus": {
        "name": "Perseus Digital Library",
        "base_url": "https://www.perseus.tufts.edu",
        "api_url": "https://www.perseus.tufts.edu/hopper/xmlchunk",
        "languages": ["grc", "la"],
        "periods": ["archaic", "classical", "hellenistic", "roman"],
        "text_types": ["prose", "poetry", "drama", "philosophy", "history", "oratory"]
    },
    "proiel": {
        "name": "PROIEL Treebank",
        "base_url": "https://proiel.github.io",
        "github_url": "https://github.com/proiel/proiel-treebank",
        "languages": ["grc", "la", "got", "cu", "xcl"],
        "annotated": True
    },
    "first1k": {
        "name": "First1KGreek",
        "base_url": "https://opengreekandlatin.github.io/First1KGreek",
        "github_url": "https://github.com/OpenGreekAndLatin/First1KGreek",
        "languages": ["grc"],
        "format": "TEI XML"
    },
    "diorisis": {
        "name": "Diorisis Ancient Greek Corpus",
        "base_url": "https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256",
        "languages": ["grc"],
        "size": "10M words"
    },
    "tlg": {
        "name": "Thesaurus Linguae Graecae",
        "note": "Subscription required",
        "languages": ["grc"],
        "comprehensive": True
    },
    "wikisource_grc": {
        "name": "Wikisource Ancient Greek",
        "base_url": "https://el.wikisource.org",
        "api_url": "https://el.wikisource.org/w/api.php",
        "languages": ["grc", "el"]
    },
    "sacred_texts": {
        "name": "Sacred Texts Archive",
        "base_url": "https://www.sacred-texts.com",
        "categories": ["bible", "classics", "gnostic"]
    }
}

# English comparative sources
ENGLISH_SOURCES = {
    "gutenberg": {
        "name": "Project Gutenberg",
        "base_url": "https://www.gutenberg.org",
        "api_url": "https://gutendex.com/books",
        "languages": ["en"],
        "focus": ["translations", "classics", "philosophy"]
    },
    "wikisource_en": {
        "name": "Wikisource English",
        "base_url": "https://en.wikisource.org",
        "api_url": "https://en.wikisource.org/w/api.php",
        "languages": ["en"]
    },
    "internet_archive": {
        "name": "Internet Archive",
        "base_url": "https://archive.org",
        "api_url": "https://archive.org/advancedsearch.php",
        "languages": ["en", "grc", "la"]
    }
}

# Influential Greek texts to prioritize
INFLUENTIAL_TEXTS = {
    "homer": {
        "author": "Homer",
        "works": ["Iliad", "Odyssey"],
        "period": "archaic",
        "importance": "foundational",
        "translations": ["English", "Latin", "German", "French"]
    },
    "hesiod": {
        "author": "Hesiod",
        "works": ["Theogony", "Works and Days"],
        "period": "archaic",
        "importance": "foundational"
    },
    "herodotus": {
        "author": "Herodotus",
        "works": ["Histories"],
        "period": "classical",
        "importance": "foundational",
        "genre": "history"
    },
    "thucydides": {
        "author": "Thucydides",
        "works": ["History of the Peloponnesian War"],
        "period": "classical",
        "importance": "foundational",
        "genre": "history"
    },
    "plato": {
        "author": "Plato",
        "works": ["Republic", "Symposium", "Phaedo", "Apology", "Timaeus", "Laws"],
        "period": "classical",
        "importance": "foundational",
        "genre": "philosophy"
    },
    "aristotle": {
        "author": "Aristotle",
        "works": ["Nicomachean Ethics", "Politics", "Poetics", "Metaphysics", "Physics"],
        "period": "classical",
        "importance": "foundational",
        "genre": "philosophy"
    },
    "sophocles": {
        "author": "Sophocles",
        "works": ["Oedipus Rex", "Antigone", "Electra"],
        "period": "classical",
        "importance": "foundational",
        "genre": "drama"
    },
    "euripides": {
        "author": "Euripides",
        "works": ["Medea", "Bacchae", "Hippolytus"],
        "period": "classical",
        "importance": "foundational",
        "genre": "drama"
    },
    "aeschylus": {
        "author": "Aeschylus",
        "works": ["Oresteia", "Prometheus Bound", "Seven Against Thebes"],
        "period": "classical",
        "importance": "foundational",
        "genre": "drama"
    },
    "aristophanes": {
        "author": "Aristophanes",
        "works": ["Clouds", "Birds", "Frogs", "Lysistrata"],
        "period": "classical",
        "importance": "foundational",
        "genre": "comedy"
    },
    "demosthenes": {
        "author": "Demosthenes",
        "works": ["Philippics", "Olynthiacs", "On the Crown"],
        "period": "classical",
        "importance": "foundational",
        "genre": "oratory"
    },
    "xenophon": {
        "author": "Xenophon",
        "works": ["Anabasis", "Memorabilia", "Cyropaedia"],
        "period": "classical",
        "importance": "high",
        "genre": "history"
    },
    "lysias": {
        "author": "Lysias",
        "works": ["Against Eratosthenes", "Funeral Oration"],
        "period": "classical",
        "importance": "high",
        "genre": "oratory"
    },
    "isocrates": {
        "author": "Isocrates",
        "works": ["Panegyricus", "Antidosis"],
        "period": "classical",
        "importance": "high",
        "genre": "oratory"
    },
    "new_testament": {
        "author": "Various",
        "works": ["Matthew", "Mark", "Luke", "John", "Acts", "Romans", "Corinthians"],
        "period": "hellenistic",
        "importance": "foundational",
        "genre": "religious",
        "language_variety": "Koine"
    },
    "septuagint": {
        "author": "Various",
        "works": ["Genesis", "Exodus", "Psalms", "Isaiah"],
        "period": "hellenistic",
        "importance": "foundational",
        "genre": "religious",
        "language_variety": "Koine"
    },
    "plutarch": {
        "author": "Plutarch",
        "works": ["Parallel Lives", "Moralia"],
        "period": "roman",
        "importance": "high",
        "genre": "biography"
    },
    "epictetus": {
        "author": "Epictetus",
        "works": ["Discourses", "Enchiridion"],
        "period": "roman",
        "importance": "high",
        "genre": "philosophy"
    },
    "marcus_aurelius": {
        "author": "Marcus Aurelius",
        "works": ["Meditations"],
        "period": "roman",
        "importance": "high",
        "genre": "philosophy"
    },
    "lucian": {
        "author": "Lucian",
        "works": ["True History", "Dialogues of the Dead"],
        "period": "roman",
        "importance": "high",
        "genre": "satire"
    }
}

# Representative text categories
TEXT_CATEGORIES = {
    "literary": {
        "epic": ["Homer", "Hesiod", "Apollonius"],
        "lyric": ["Sappho", "Pindar", "Bacchylides"],
        "drama": ["Aeschylus", "Sophocles", "Euripides", "Aristophanes", "Menander"],
        "prose": ["Herodotus", "Thucydides", "Xenophon", "Plato", "Aristotle"]
    },
    "technical": {
        "philosophy": ["Plato", "Aristotle", "Stoics", "Epicureans", "Neoplatonists"],
        "science": ["Hippocrates", "Galen", "Euclid", "Archimedes", "Ptolemy"],
        "rhetoric": ["Demosthenes", "Isocrates", "Lysias", "Aristotle Rhetoric"]
    },
    "religious": {
        "jewish": ["Septuagint", "Philo", "Josephus"],
        "christian": ["New Testament", "Apostolic Fathers", "Church Fathers"],
        "pagan": ["Orphic Hymns", "Magical Papyri"]
    },
    "documentary": {
        "inscriptions": ["Attic inscriptions", "Delphic inscriptions"],
        "papyri": ["Documentary papyri", "Literary papyri"]
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TextMetadata:
    """Metadata for collected texts"""
    id: str
    title: str
    author: str
    language: str
    period: str = ""
    genre: str = ""
    source: str = ""
    source_url: str = ""
    date_collected: str = ""
    date_composed: str = ""
    word_count: int = 0
    character_count: int = 0
    is_translation: bool = False
    original_language: str = ""
    parallel_text_id: str = ""
    checksum: str = ""
    processing_status: str = "pending"
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TextMetadata':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CollectionTask:
    """Task for text collection"""
    task_id: str
    source: str
    url: str
    text_type: str
    priority: int = 5
    retries: int = 0
    status: str = "pending"
    error: str = ""
    created_at: str = ""
    completed_at: str = ""


@dataclass
class ParallelText:
    """Parallel text pair for translation studies"""
    id: str
    source_text_id: str
    target_text_id: str
    source_language: str
    target_language: str
    alignment_type: str = "document"  # document, paragraph, sentence
    alignment_data: Dict = field(default_factory=dict)
    quality_score: float = 0.0


# ============================================================================
# TEXT NORMALIZER
# ============================================================================

class TextNormalizer:
    """Normalize and clean text for processing"""
    
    # Greek character mappings
    GREEK_NORMALIZATION = {
        # Polytonic to monotonic (optional)
        '\u1F00': '\u03B1',  # ἀ -> α
        '\u1F01': '\u03B1',  # ἁ -> α
        '\u1F02': '\u03B1',  # ἂ -> α
        '\u1F03': '\u03B1',  # ἃ -> α
        '\u1F04': '\u03B1',  # ἄ -> α
        '\u1F05': '\u03B1',  # ἅ -> α
        '\u1F06': '\u03B1',  # ἆ -> α
        '\u1F07': '\u03B1',  # ἇ -> α
        # Add more as needed
    }
    
    # Unicode normalization categories
    COMBINING_MARKS = re.compile(r'[\u0300-\u036f]')
    
    def __init__(self, preserve_diacritics: bool = True):
        self.preserve_diacritics = preserve_diacritics
    
    def normalize(self, text: str, language: str = "grc") -> str:
        """Normalize text based on language"""
        if not text:
            return ""
        
        # Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        text = self._remove_control_chars(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Language-specific normalization
        if language in ["grc", "el"]:
            text = self._normalize_greek(text)
        elif language == "la":
            text = self._normalize_latin(text)
        elif language == "en":
            text = self._normalize_english(text)
        
        return text.strip()
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs"""
        return ''.join(c for c in text if c == '\n' or c == '\t' or not unicodedata.category(c).startswith('C'))
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line endings
        text = re.sub(r'\r\n|\r', '\n', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def _normalize_greek(self, text: str) -> str:
        """Normalize Greek text"""
        # Normalize Greek quotation marks
        text = text.replace('«', '"').replace('»', '"')
        
        # Normalize Greek punctuation
        text = text.replace('·', ';')  # ano teleia to semicolon
        text = text.replace('᾽', "'")  # koronis to apostrophe
        
        # Optionally remove diacritics
        if not self.preserve_diacritics:
            text = self._remove_greek_diacritics(text)
        
        return text
    
    def _remove_greek_diacritics(self, text: str) -> str:
        """Remove Greek diacritical marks"""
        # NFD decomposition then remove combining marks
        text = unicodedata.normalize('NFD', text)
        text = self.COMBINING_MARKS.sub('', text)
        return unicodedata.normalize('NFC', text)
    
    def _normalize_latin(self, text: str) -> str:
        """Normalize Latin text"""
        # Normalize u/v and i/j (optional, configurable)
        # text = text.replace('v', 'u').replace('j', 'i')
        return text
    
    def _normalize_english(self, text: str) -> str:
        """Normalize English text"""
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text
    
    def clean_xml_text(self, text: str) -> str:
        """Clean text extracted from XML"""
        # Remove XML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        return text
    
    def segment_sentences(self, text: str, language: str = "grc") -> List[str]:
        """Segment text into sentences"""
        if language in ["grc", "la"]:
            # Greek/Latin sentence boundaries
            pattern = r'(?<=[.;:!?·])\s+'
        else:
            # English sentence boundaries
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def segment_paragraphs(self, text: str) -> List[str]:
        """Segment text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]


# ============================================================================
# TEXT COLLECTORS (Abstract Base)
# ============================================================================

class TextCollector(ABC):
    """Abstract base class for text collectors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.get('user_agent', CONFIG['user_agent'])
        })
        self.normalizer = TextNormalizer()
    
    @abstractmethod
    def collect(self, **kwargs) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect texts from source"""
        pass
    
    @abstractmethod
    def get_text_list(self, **kwargs) -> List[Dict]:
        """Get list of available texts"""
        pass
    
    def _make_request(self, url: str, params: Dict = None, 
                     retries: int = 3, delay: float = 1.0) -> Optional[requests.Response]:
        """Make HTTP request with retries"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        return None
    
    def _generate_id(self, *args) -> str:
        """Generate unique ID from arguments"""
        content = '|'.join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, text: str) -> str:
        """Calculate text checksum"""
        return hashlib.sha256(text.encode()).hexdigest()


# ============================================================================
# PERSEUS COLLECTOR
# ============================================================================

class PerseusCollector(TextCollector):
    """Collect texts from Perseus Digital Library"""
    
    BASE_URL = "https://www.perseus.tufts.edu"
    CATALOG_URL = "https://www.perseus.tufts.edu/hopper/collection"
    
    # Known Perseus Greek texts
    GREEK_TEXTS = {
        "Homer": {
            "Iliad": "Perseus:text:1999.01.0133",
            "Odyssey": "Perseus:text:1999.01.0135"
        },
        "Hesiod": {
            "Theogony": "Perseus:text:1999.01.0129",
            "Works and Days": "Perseus:text:1999.01.0131"
        },
        "Herodotus": {
            "Histories": "Perseus:text:1999.01.0125"
        },
        "Thucydides": {
            "History": "Perseus:text:1999.01.0247"
        },
        "Plato": {
            "Republic": "Perseus:text:1999.01.0167",
            "Apology": "Perseus:text:1999.01.0169",
            "Symposium": "Perseus:text:1999.01.0173"
        },
        "Aristotle": {
            "Nicomachean Ethics": "Perseus:text:1999.01.0053",
            "Politics": "Perseus:text:1999.01.0057"
        },
        "Sophocles": {
            "Oedipus Tyrannus": "Perseus:text:1999.01.0191",
            "Antigone": "Perseus:text:1999.01.0185"
        },
        "Euripides": {
            "Medea": "Perseus:text:1999.01.0113",
            "Bacchae": "Perseus:text:1999.01.0091"
        },
        "Aeschylus": {
            "Agamemnon": "Perseus:text:1999.01.0003",
            "Prometheus Bound": "Perseus:text:1999.01.0009"
        },
        "Aristophanes": {
            "Clouds": "Perseus:text:1999.01.0027",
            "Birds": "Perseus:text:1999.01.0025"
        },
        "Demosthenes": {
            "Philippic 1": "Perseus:text:1999.01.0069",
            "On the Crown": "Perseus:text:1999.01.0071"
        },
        "Xenophon": {
            "Anabasis": "Perseus:text:1999.01.0201",
            "Memorabilia": "Perseus:text:1999.01.0207"
        },
        "Lysias": {
            "Against Eratosthenes": "Perseus:text:1999.01.0153"
        },
        "Isocrates": {
            "Panegyricus": "Perseus:text:1999.01.0143"
        },
        "Plutarch": {
            "Theseus": "Perseus:text:2008.01.0063",
            "Romulus": "Perseus:text:2008.01.0064"
        }
    }
    
    def get_text_list(self, **kwargs) -> List[Dict]:
        """Get list of available Greek texts"""
        texts = []
        for author, works in self.GREEK_TEXTS.items():
            for title, urn in works.items():
                texts.append({
                    "author": author,
                    "title": title,
                    "urn": urn,
                    "language": "grc",
                    "source": "perseus"
                })
        return texts
    
    def collect(self, author: str = None, title: str = None, 
               urn: str = None, **kwargs) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect texts from Perseus"""
        
        if urn:
            yield from self._collect_by_urn(urn)
        elif author and title:
            if author in self.GREEK_TEXTS and title in self.GREEK_TEXTS[author]:
                urn = self.GREEK_TEXTS[author][title]
                yield from self._collect_by_urn(urn, author, title)
        else:
            # Collect all known texts
            for author, works in self.GREEK_TEXTS.items():
                for title, urn in works.items():
                    yield from self._collect_by_urn(urn, author, title)
                    time.sleep(self.config.get('request_delay', 1.0))
    
    def _collect_by_urn(self, urn: str, author: str = "", 
                       title: str = "") -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect text by Perseus URN"""
        url = f"{self.BASE_URL}/hopper/xmlchunk"
        params = {"doc": urn}
        
        response = self._make_request(url, params)
        if not response:
            logger.error(f"Failed to fetch {urn}")
            return
        
        try:
            # Parse XML response
            text = self._extract_text_from_xml(response.text)
            text = self.normalizer.normalize(text, "grc")
            
            if not text:
                logger.warning(f"No text extracted from {urn}")
                return
            
            metadata = TextMetadata(
                id=self._generate_id("perseus", urn),
                title=title or urn,
                author=author or "Unknown",
                language="grc",
                source="perseus",
                source_url=f"{self.BASE_URL}/hopper/text?doc={urn}",
                date_collected=datetime.now().isoformat(),
                word_count=len(text.split()),
                character_count=len(text),
                checksum=self._calculate_checksum(text)
            )
            
            yield text, metadata
            
        except Exception as e:
            logger.error(f"Error processing {urn}: {e}")
    
    def _extract_text_from_xml(self, xml_content: str) -> str:
        """Extract text from Perseus XML"""
        try:
            # Clean XML
            xml_content = re.sub(r'<\?xml[^>]*\?>', '', xml_content)
            xml_content = re.sub(r'<!DOCTYPE[^>]*>', '', xml_content)
            
            root = ET.fromstring(xml_content)
            
            # Extract text from various elements
            text_parts = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text)
                if elem.tail:
                    text_parts.append(elem.tail)
            
            return ' '.join(text_parts)
            
        except ET.ParseError:
            # Fallback: strip tags
            return self.normalizer.clean_xml_text(xml_content)


# ============================================================================
# WIKISOURCE COLLECTOR
# ============================================================================

class WikisourceCollector(TextCollector):
    """Collect texts from Wikisource"""
    
    APIS = {
        "grc": "https://el.wikisource.org/w/api.php",
        "en": "https://en.wikisource.org/w/api.php"
    }
    
    # Categories for Greek texts
    GREEK_CATEGORIES = [
        "Αρχαία_ελληνική_λογοτεχνία",
        "Αρχαία_ελληνική_φιλοσοφία",
        "Αρχαία_ελληνική_ποίηση",
        "Βυζαντινή_λογοτεχνία"
    ]
    
    def get_text_list(self, language: str = "grc", **kwargs) -> List[Dict]:
        """Get list of available texts"""
        texts = []
        api_url = self.APIS.get(language, self.APIS["grc"])
        
        for category in self.GREEK_CATEGORIES:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": 500,
                "format": "json"
            }
            
            response = self._make_request(api_url, params)
            if response:
                data = response.json()
                for page in data.get("query", {}).get("categorymembers", []):
                    texts.append({
                        "title": page["title"],
                        "pageid": page["pageid"],
                        "language": language,
                        "category": category,
                        "source": "wikisource"
                    })
        
        return texts
    
    def collect(self, language: str = "grc", title: str = None, 
               pageid: int = None, **kwargs) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect texts from Wikisource"""
        api_url = self.APIS.get(language, self.APIS["grc"])
        
        if title or pageid:
            yield from self._collect_page(api_url, language, title, pageid)
        else:
            # Collect from categories
            for category in self.GREEK_CATEGORIES:
                yield from self._collect_category(api_url, language, category)
    
    def _collect_page(self, api_url: str, language: str, 
                     title: str = None, pageid: int = None) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect single page"""
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": True,
            "format": "json"
        }
        
        if title:
            params["titles"] = title
        elif pageid:
            params["pageids"] = pageid
        else:
            return
        
        response = self._make_request(api_url, params)
        if not response:
            return
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_data in pages.items():
            if page_id == "-1":
                continue
            
            text = page_data.get("extract", "")
            if not text:
                continue
            
            text = self.normalizer.normalize(text, language)
            
            metadata = TextMetadata(
                id=self._generate_id("wikisource", language, page_id),
                title=page_data.get("title", ""),
                author="Unknown",
                language=language,
                source="wikisource",
                source_url=page_data.get("fullurl", ""),
                date_collected=datetime.now().isoformat(),
                word_count=len(text.split()),
                character_count=len(text),
                checksum=self._calculate_checksum(text)
            )
            
            yield text, metadata
    
    def _collect_category(self, api_url: str, language: str, 
                         category: str) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect all pages in category"""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 50,
            "format": "json"
        }
        
        response = self._make_request(api_url, params)
        if not response:
            return
        
        data = response.json()
        for page in data.get("query", {}).get("categorymembers", []):
            yield from self._collect_page(api_url, language, title=page["title"])
            time.sleep(self.config.get('request_delay', 1.0))


# ============================================================================
# GUTENBERG COLLECTOR (English translations)
# ============================================================================

class GutenbergCollector(TextCollector):
    """Collect texts from Project Gutenberg"""
    
    API_URL = "https://gutendex.com/books"
    
    # Search terms for Greek-related English texts
    SEARCH_TERMS = [
        "Homer translation",
        "Plato translation",
        "Aristotle translation",
        "Greek tragedy",
        "Herodotus",
        "Thucydides",
        "Xenophon",
        "Plutarch",
        "Greek philosophy",
        "Ancient Greece"
    ]
    
    def get_text_list(self, search: str = None, **kwargs) -> List[Dict]:
        """Get list of available texts"""
        texts = []
        
        search_terms = [search] if search else self.SEARCH_TERMS
        
        for term in search_terms:
            params = {
                "search": term,
                "languages": "en"
            }
            
            response = self._make_request(self.API_URL, params)
            if response:
                data = response.json()
                for book in data.get("results", []):
                    texts.append({
                        "id": book["id"],
                        "title": book["title"],
                        "authors": [a["name"] for a in book.get("authors", [])],
                        "language": "en",
                        "source": "gutenberg",
                        "formats": book.get("formats", {})
                    })
        
        return texts
    
    def collect(self, book_id: int = None, search: str = None, 
               **kwargs) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect texts from Gutenberg"""
        
        if book_id:
            yield from self._collect_book(book_id)
        else:
            texts = self.get_text_list(search)
            for text_info in texts[:50]:  # Limit to 50 texts
                yield from self._collect_book(text_info["id"])
                time.sleep(self.config.get('request_delay', 1.0))
    
    def _collect_book(self, book_id: int) -> Generator[Tuple[str, TextMetadata], None, None]:
        """Collect single book"""
        # Get book metadata
        response = self._make_request(f"{self.API_URL}/{book_id}")
        if not response:
            return
        
        book = response.json()
        
        # Get text content
        formats = book.get("formats", {})
        text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
        
        if not text_url:
            logger.warning(f"No plain text format for book {book_id}")
            return
        
        text_response = self._make_request(text_url)
        if not text_response:
            return
        
        text = text_response.text
        text = self.normalizer.normalize(text, "en")
        
        # Remove Gutenberg header/footer
        text = self._remove_gutenberg_boilerplate(text)
        
        if not text:
            return
        
        authors = [a["name"] for a in book.get("authors", [])]
        
        metadata = TextMetadata(
            id=self._generate_id("gutenberg", book_id),
            title=book.get("title", ""),
            author=", ".join(authors) if authors else "Unknown",
            language="en",
            source="gutenberg",
            source_url=f"https://www.gutenberg.org/ebooks/{book_id}",
            date_collected=datetime.now().isoformat(),
            word_count=len(text.split()),
            character_count=len(text),
            checksum=self._calculate_checksum(text),
            is_translation=True,
            original_language="grc"
        )
        
        yield text, metadata
    
    def _remove_gutenberg_boilerplate(self, text: str) -> str:
        """Remove Project Gutenberg header and footer"""
        # Find start marker
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT"
        ]
        
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[idx:]
                # Find end of header line
                newline_idx = text.find('\n')
                if newline_idx > 0:
                    text = text[newline_idx + 1:]
                break
        
        # Find end marker
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of Project Gutenberg"
        ]
        
        for marker in end_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx]
                break
        
        return text.strip()


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class CollectorDatabase:
    """Database for text collection management"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize collection tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Raw texts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_texts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                language TEXT NOT NULL,
                period TEXT,
                genre TEXT,
                source TEXT,
                source_url TEXT,
                content TEXT,
                word_count INTEGER,
                character_count INTEGER,
                checksum TEXT,
                is_translation INTEGER DEFAULT 0,
                original_language TEXT,
                parallel_text_id TEXT,
                processing_status TEXT DEFAULT 'pending',
                metadata TEXT,
                date_collected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                date_processed TIMESTAMP
            )
        """)
        
        # Collection tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_tasks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                url TEXT,
                text_type TEXT,
                priority INTEGER DEFAULT 5,
                retries INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Parallel texts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parallel_texts (
                id TEXT PRIMARY KEY,
                source_text_id TEXT NOT NULL,
                target_text_id TEXT NOT NULL,
                source_language TEXT,
                target_language TEXT,
                alignment_type TEXT DEFAULT 'document',
                alignment_data TEXT,
                quality_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_text_id) REFERENCES raw_texts(id),
                FOREIGN KEY (target_text_id) REFERENCES raw_texts(id)
            )
        """)
        
        # Collection statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                texts_collected INTEGER,
                words_collected INTEGER,
                errors INTEGER,
                duration_seconds REAL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_texts_language ON raw_texts(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_texts_source ON raw_texts(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_texts_status ON raw_texts(processing_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON collection_tasks(status)")
        
        conn.commit()
        conn.close()
    
    def save_text(self, text: str, metadata: TextMetadata) -> bool:
        """Save collected text"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO raw_texts 
                (id, title, author, language, period, genre, source, source_url,
                 content, word_count, character_count, checksum, is_translation,
                 original_language, parallel_text_id, processing_status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.id, metadata.title, metadata.author, metadata.language,
                metadata.period, metadata.genre, metadata.source, metadata.source_url,
                text, metadata.word_count, metadata.character_count, metadata.checksum,
                1 if metadata.is_translation else 0, metadata.original_language,
                metadata.parallel_text_id, metadata.processing_status,
                json.dumps(metadata.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving text: {e}")
            return False
    
    def get_text(self, text_id: str) -> Optional[Tuple[str, TextMetadata]]:
        """Get text by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM raw_texts WHERE id = ?", (text_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        metadata = TextMetadata(
            id=row['id'],
            title=row['title'],
            author=row['author'],
            language=row['language'],
            period=row['period'] or "",
            genre=row['genre'] or "",
            source=row['source'] or "",
            source_url=row['source_url'] or "",
            word_count=row['word_count'] or 0,
            character_count=row['character_count'] or 0,
            checksum=row['checksum'] or "",
            is_translation=bool(row['is_translation']),
            original_language=row['original_language'] or "",
            parallel_text_id=row['parallel_text_id'] or "",
            processing_status=row['processing_status'] or "pending"
        )
        
        return row['content'], metadata
    
    def get_texts_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Get texts by processing status"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, author, language, source, word_count, processing_status
            FROM raw_texts 
            WHERE processing_status = ?
            LIMIT ?
        """, (status, limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def update_status(self, text_id: str, status: str):
        """Update text processing status"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE raw_texts 
            SET processing_status = ?, date_processed = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, text_id))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total texts
        cursor.execute("SELECT COUNT(*) FROM raw_texts")
        stats['total_texts'] = cursor.fetchone()[0]
        
        # Total words
        cursor.execute("SELECT SUM(word_count) FROM raw_texts")
        stats['total_words'] = cursor.fetchone()[0] or 0
        
        # By language
        cursor.execute("""
            SELECT language, COUNT(*) as count, SUM(word_count) as words
            FROM raw_texts GROUP BY language
        """)
        stats['by_language'] = {
            row['language']: {'texts': row['count'], 'words': row['words'] or 0}
            for row in cursor.fetchall()
        }
        
        # By source
        cursor.execute("""
            SELECT source, COUNT(*) as count, SUM(word_count) as words
            FROM raw_texts GROUP BY source
        """)
        stats['by_source'] = {
            row['source']: {'texts': row['count'], 'words': row['words'] or 0}
            for row in cursor.fetchall()
        }
        
        # By status
        cursor.execute("""
            SELECT processing_status, COUNT(*) as count
            FROM raw_texts GROUP BY processing_status
        """)
        stats['by_status'] = {row['processing_status']: row['count'] for row in cursor.fetchall()}
        
        # Parallel texts
        cursor.execute("SELECT COUNT(*) FROM parallel_texts")
        stats['parallel_texts'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def save_parallel_text(self, parallel: ParallelText) -> bool:
        """Save parallel text pair"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO parallel_texts 
                (id, source_text_id, target_text_id, source_language, target_language,
                 alignment_type, alignment_data, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parallel.id, parallel.source_text_id, parallel.target_text_id,
                parallel.source_language, parallel.target_language,
                parallel.alignment_type, json.dumps(parallel.alignment_data),
                parallel.quality_score
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving parallel text: {e}")
            return False


# ============================================================================
# COLLECTION DAEMON
# ============================================================================

class CollectionDaemon:
    """24/7 text collection daemon"""
    
    def __init__(self, db: CollectorDatabase, config: Dict = None):
        self.db = db
        self.config = config or CONFIG
        self.running = False
        self.task_queue = queue.Queue()
        self.collectors = {
            'perseus': PerseusCollector(config),
            'wikisource': WikisourceCollector(config),
            'gutenberg': GutenbergCollector(config)
        }
        self.stats = {
            'texts_collected': 0,
            'words_collected': 0,
            'errors': 0,
            'start_time': None
        }
    
    def start(self):
        """Start collection daemon"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("Starting collection daemon...")
        
        # Start worker threads
        workers = []
        for i in range(self.config.get('max_workers', 4)):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            workers.append(worker)
        
        # Start scheduler thread
        scheduler = threading.Thread(target=self._scheduler, daemon=True)
        scheduler.start()
        
        logger.info(f"Collection daemon started with {len(workers)} workers")
    
    def stop(self):
        """Stop collection daemon"""
        self.running = False
        logger.info("Stopping collection daemon...")
    
    def _scheduler(self):
        """Schedule collection tasks"""
        while self.running:
            try:
                # Schedule Perseus collection
                self._schedule_perseus_tasks()
                
                # Schedule Wikisource collection
                self._schedule_wikisource_tasks()
                
                # Schedule Gutenberg collection
                self._schedule_gutenberg_tasks()
                
                # Wait before next scheduling cycle
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _schedule_perseus_tasks(self):
        """Schedule Perseus collection tasks"""
        collector = self.collectors['perseus']
        texts = collector.get_text_list()
        
        for text_info in texts:
            task = CollectionTask(
                task_id=f"perseus_{text_info['urn']}",
                source='perseus',
                url=text_info['urn'],
                text_type='greek_classical',
                priority=1 if text_info['author'] in ['Homer', 'Plato', 'Aristotle'] else 5
            )
            self.task_queue.put(task)
    
    def _schedule_wikisource_tasks(self):
        """Schedule Wikisource collection tasks"""
        collector = self.collectors['wikisource']
        texts = collector.get_text_list(language='grc')
        
        for text_info in texts[:100]:  # Limit
            task = CollectionTask(
                task_id=f"wikisource_{text_info['pageid']}",
                source='wikisource',
                url=str(text_info['pageid']),
                text_type='greek',
                priority=5
            )
            self.task_queue.put(task)
    
    def _schedule_gutenberg_tasks(self):
        """Schedule Gutenberg collection tasks"""
        collector = self.collectors['gutenberg']
        texts = collector.get_text_list()
        
        for text_info in texts[:50]:  # Limit
            task = CollectionTask(
                task_id=f"gutenberg_{text_info['id']}",
                source='gutenberg',
                url=str(text_info['id']),
                text_type='english_translation',
                priority=3
            )
            self.task_queue.put(task)
    
    def _worker(self):
        """Worker thread for processing tasks"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=10)
                self._process_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.stats['errors'] += 1
    
    def _process_task(self, task: CollectionTask):
        """Process collection task"""
        collector = self.collectors.get(task.source)
        if not collector:
            logger.warning(f"Unknown source: {task.source}")
            return
        
        try:
            if task.source == 'perseus':
                for text, metadata in collector.collect(urn=task.url):
                    self._save_collected_text(text, metadata)
            
            elif task.source == 'wikisource':
                for text, metadata in collector.collect(pageid=int(task.url)):
                    self._save_collected_text(text, metadata)
            
            elif task.source == 'gutenberg':
                for text, metadata in collector.collect(book_id=int(task.url)):
                    self._save_collected_text(text, metadata)
            
            task.status = 'completed'
            task.completed_at = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Task error {task.task_id}: {e}")
            task.status = 'failed'
            task.error = str(e)
            task.retries += 1
            
            if task.retries < self.config.get('max_retries', 3):
                self.task_queue.put(task)
    
    def _save_collected_text(self, text: str, metadata: TextMetadata):
        """Save collected text to database"""
        if self.db.save_text(text, metadata):
            self.stats['texts_collected'] += 1
            self.stats['words_collected'] += metadata.word_count
            logger.info(f"Collected: {metadata.title} ({metadata.word_count} words)")
    
    def get_stats(self) -> Dict:
        """Get daemon statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime'] = str(datetime.now() - stats['start_time'])
        stats['queue_size'] = self.task_queue.qsize()
        stats['db_stats'] = self.db.get_statistics()
        return stats


# ============================================================================
# MAIN COLLECTION MANAGER
# ============================================================================

class TextCollectionManager:
    """Main interface for text collection"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db = CollectorDatabase(db_path)
        self.daemon = None
        self.collectors = {
            'perseus': PerseusCollector(),
            'wikisource': WikisourceCollector(),
            'gutenberg': GutenbergCollector()
        }
    
    def collect_influential_texts(self):
        """Collect all influential Greek texts"""
        logger.info("Starting collection of influential texts...")
        
        collector = self.collectors['perseus']
        
        for author, info in INFLUENTIAL_TEXTS.items():
            logger.info(f"Collecting {info['author']}...")
            
            for text, metadata in collector.collect(author=info['author']):
                metadata.period = info.get('period', '')
                metadata.genre = info.get('genre', '')
                self.db.save_text(text, metadata)
                logger.info(f"  Saved: {metadata.title}")
    
    def collect_parallel_texts(self, source_lang: str = "grc", target_lang: str = "en"):
        """Collect parallel Greek-English texts"""
        logger.info(f"Collecting parallel texts ({source_lang} -> {target_lang})...")
        
        # Collect Greek originals
        greek_collector = self.collectors['perseus']
        english_collector = self.collectors['gutenberg']
        
        # Known parallel texts
        parallel_works = [
            {"greek_author": "Homer", "greek_title": "Iliad", "english_search": "Iliad Homer"},
            {"greek_author": "Homer", "greek_title": "Odyssey", "english_search": "Odyssey Homer"},
            {"greek_author": "Plato", "greek_title": "Republic", "english_search": "Republic Plato"},
        ]
        
        for work in parallel_works:
            # Collect Greek
            for greek_text, greek_meta in greek_collector.collect(
                author=work['greek_author'], 
                title=work['greek_title']
            ):
                self.db.save_text(greek_text, greek_meta)
                
                # Collect English translation
                for eng_text, eng_meta in english_collector.collect(
                    search=work['english_search']
                ):
                    eng_meta.is_translation = True
                    eng_meta.original_language = "grc"
                    eng_meta.parallel_text_id = greek_meta.id
                    self.db.save_text(eng_text, eng_meta)
                    
                    # Create parallel text record
                    parallel = ParallelText(
                        id=f"parallel_{greek_meta.id}_{eng_meta.id}",
                        source_text_id=greek_meta.id,
                        target_text_id=eng_meta.id,
                        source_language="grc",
                        target_language="en",
                        alignment_type="document"
                    )
                    self.db.save_parallel_text(parallel)
                    break  # Just first translation
    
    def start_daemon(self):
        """Start 24/7 collection daemon"""
        if self.daemon and self.daemon.running:
            logger.warning("Daemon already running")
            return
        
        self.daemon = CollectionDaemon(self.db)
        self.daemon.start()
    
    def stop_daemon(self):
        """Stop collection daemon"""
        if self.daemon:
            self.daemon.stop()
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        stats = self.db.get_statistics()
        if self.daemon:
            stats['daemon'] = self.daemon.get_stats()
        return stats
    
    def export_texts(self, output_dir: str, language: str = None, 
                    format: str = "txt") -> int:
        """Export collected texts to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        sql = "SELECT * FROM raw_texts"
        params = []
        if language:
            sql += " WHERE language = ?"
            params.append(language)
        
        cursor.execute(sql, params)
        
        count = 0
        for row in cursor.fetchall():
            filename = f"{row['id']}_{row['language']}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                if format == "json":
                    data = {
                        'id': row['id'],
                        'title': row['title'],
                        'author': row['author'],
                        'language': row['language'],
                        'content': row['content']
                    }
                    json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    f.write(f"# {row['title']}\n")
                    f.write(f"# Author: {row['author']}\n")
                    f.write(f"# Language: {row['language']}\n\n")
                    f.write(row['content'])
            
            count += 1
        
        conn.close()
        return count


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Collection Pipeline")
    parser.add_argument('command', choices=['collect', 'daemon', 'stats', 'export'],
                       help="Command to run")
    parser.add_argument('--source', choices=['perseus', 'wikisource', 'gutenberg', 'all'],
                       default='all', help="Text source")
    parser.add_argument('--language', default='grc', help="Language code")
    parser.add_argument('--output', default='exported_texts', help="Output directory")
    parser.add_argument('--influential', action='store_true', 
                       help="Collect influential texts only")
    parser.add_argument('--parallel', action='store_true',
                       help="Collect parallel texts")
    
    args = parser.parse_args()
    
    manager = TextCollectionManager()
    
    if args.command == 'collect':
        if args.influential:
            manager.collect_influential_texts()
        elif args.parallel:
            manager.collect_parallel_texts()
        else:
            # Collect from specified source
            if args.source in manager.collectors:
                collector = manager.collectors[args.source]
                for text, metadata in collector.collect():
                    manager.db.save_text(text, metadata)
                    print(f"Collected: {metadata.title}")
    
    elif args.command == 'daemon':
        print("Starting collection daemon (Ctrl+C to stop)...")
        manager.start_daemon()
        try:
            while True:
                time.sleep(60)
                stats = manager.get_statistics()
                print(f"Texts: {stats.get('total_texts', 0)}, Words: {stats.get('total_words', 0)}")
        except KeyboardInterrupt:
            manager.stop_daemon()
            print("\nDaemon stopped.")
    
    elif args.command == 'stats':
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'export':
        count = manager.export_texts(args.output, args.language)
        print(f"Exported {count} texts to {args.output}")


if __name__ == "__main__":
    main()
