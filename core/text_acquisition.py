"""
Text Acquisition Engine - Comprehensive Greek Text Collection
Finds and downloads texts from all major sources
"""

import os
import re
import json
import sqlite3
import logging
import requests
import time
import hashlib
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TEXT SOURCE CONFIGURATION
# =============================================================================

PERSEUS_CONFIG = {
    "base_url": "https://www.perseus.tufts.edu/hopper/",
    "catalog_url": "https://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:Greco-Roman",
    "text_url": "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:",
    "xml_url": "https://www.perseus.tufts.edu/hopper/xmlchunk?doc=Perseus:text:",
    "greek_authors": [
        "Homer", "Hesiod", "Pindar", "Aeschylus", "Sophocles", "Euripides",
        "Aristophanes", "Herodotus", "Thucydides", "Xenophon", "Plato",
        "Aristotle", "Demosthenes", "Lysias", "Isocrates", "Plutarch",
        "Lucian", "Pausanias", "Strabo", "Diodorus", "Polybius",
        "Josephus", "Philo", "Epictetus", "Marcus Aurelius"
    ]
}

FIRST1KGREEK_CONFIG = {
    "base_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/",
    "catalog_url": "https://api.github.com/repos/OpenGreekAndLatin/First1KGreek/git/trees/master?recursive=1",
    "file_pattern": r".*\.xml$"
}

PROIEL_CONFIG = {
    "base_url": "https://raw.githubusercontent.com/proiel/proiel-treebank/master/",
    "files": [
        "greek-nt.xml",
        "greek-herodotus.xml", 
        "greek-sphrantzes.xml"
    ]
}

TLG_OPEN_CONFIG = {
    "description": "Thesaurus Linguae Graecae - Open texts",
    "sources": [
        "https://stephanus.tlg.uci.edu/",  # Reference only - requires subscription
    ]
}

DIORISIS_CONFIG = {
    "base_url": "https://figshare.com/ndownloader/files/",
    "description": "Diorisis Ancient Greek Corpus - 10M words",
    "files": {
        "corpus": "12654183"  # Diorisis corpus file ID
    }
}

OPENITI_GREEK = {
    "base_url": "https://raw.githubusercontent.com/OpenITI/",
    "description": "Open Islamicate Texts Initiative - includes Greek translations"
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextMetadata:
    """Metadata for a text"""
    id: str
    title: str
    author: str
    language: str = "grc"
    period: str = "unknown"
    century: str = ""
    genre: str = ""
    source: str = ""
    source_url: str = ""
    word_count: int = 0
    sentence_count: int = 0
    
@dataclass
class RawText:
    """Raw text with metadata"""
    metadata: TextMetadata
    content: str
    xml_content: str = ""
    tei_header: Dict = field(default_factory=dict)

@dataclass 
class ProcessedSentence:
    """A processed sentence"""
    id: str
    text: str
    tokens: List[Dict]
    
# =============================================================================
# PERIOD CLASSIFICATION
# =============================================================================

AUTHOR_PERIODS = {
    # Archaic (800-500 BCE)
    "Homer": ("archaic", "8th BCE"),
    "Hesiod": ("archaic", "7th BCE"),
    "Sappho": ("archaic", "6th BCE"),
    "Pindar": ("archaic", "5th BCE"),
    
    # Classical (500-323 BCE)
    "Aeschylus": ("classical", "5th BCE"),
    "Sophocles": ("classical", "5th BCE"),
    "Euripides": ("classical", "5th BCE"),
    "Aristophanes": ("classical", "5th BCE"),
    "Herodotus": ("classical", "5th BCE"),
    "Thucydides": ("classical", "5th BCE"),
    "Xenophon": ("classical", "4th BCE"),
    "Plato": ("classical", "4th BCE"),
    "Aristotle": ("classical", "4th BCE"),
    "Demosthenes": ("classical", "4th BCE"),
    "Lysias": ("classical", "4th BCE"),
    "Isocrates": ("classical", "4th BCE"),
    
    # Hellenistic (323-31 BCE)
    "Polybius": ("hellenistic", "2nd BCE"),
    "Callimachus": ("hellenistic", "3rd BCE"),
    "Apollonius": ("hellenistic", "3rd BCE"),
    "Septuagint": ("hellenistic", "3rd BCE"),
    
    # Roman/Koine (31 BCE - 300 CE)
    "Plutarch": ("koine", "1st CE"),
    "Epictetus": ("koine", "1st CE"),
    "Lucian": ("koine", "2nd CE"),
    "Pausanias": ("koine", "2nd CE"),
    "Marcus Aurelius": ("koine", "2nd CE"),
    "Galen": ("koine", "2nd CE"),
    "New Testament": ("koine", "1st CE"),
    "Josephus": ("koine", "1st CE"),
    "Philo": ("koine", "1st CE"),
    
    # Late Antique (300-600 CE)
    "Eusebius": ("late_antique", "4th CE"),
    "John Chrysostom": ("late_antique", "4th CE"),
    "Basil": ("late_antique", "4th CE"),
    "Gregory": ("late_antique", "4th CE"),
    
    # Byzantine (600-1453 CE)
    "Anna Comnena": ("byzantine", "12th CE"),
    "Michael Psellus": ("byzantine", "11th CE"),
    "Maximus Planudes": ("byzantine", "13th CE"),
    "George Sphrantzes": ("byzantine", "15th CE"),
    "Chronicle of Morea": ("medieval", "14th CE"),
    "Digenes Akritas": ("medieval", "12th CE"),
}

def classify_period(author: str, title: str = "") -> Tuple[str, str]:
    """Classify text period based on author or title"""
    # Check author
    for known_author, (period, century) in AUTHOR_PERIODS.items():
        if known_author.lower() in author.lower():
            return period, century
    
    # Check title keywords
    title_lower = title.lower()
    if any(x in title_lower for x in ["new testament", "gospel", "acts", "epistle"]):
        return "koine", "1st CE"
    if any(x in title_lower for x in ["septuagint", "lxx"]):
        return "hellenistic", "3rd BCE"
    if any(x in title_lower for x in ["chronicle", "morea"]):
        return "medieval", "14th CE"
    if any(x in title_lower for x in ["iliad", "odyssey"]):
        return "archaic", "8th BCE"
        
    return "unknown", ""

# =============================================================================
# PERSEUS DIGITAL LIBRARY
# =============================================================================

class PerseusCollector:
    """Collect texts from Perseus Digital Library"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "perseus"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_greek_texts_catalog(self) -> List[Dict]:
        """Get catalog of Greek texts from Perseus"""
        texts = []
        
        # Known Perseus Greek text URNs
        greek_texts = [
            # Homer
            {"urn": "1999.01.0133", "author": "Homer", "title": "Iliad", "genre": "epic"},
            {"urn": "1999.01.0135", "author": "Homer", "title": "Odyssey", "genre": "epic"},
            
            # Tragedy
            {"urn": "1999.01.0003", "author": "Aeschylus", "title": "Agamemnon", "genre": "tragedy"},
            {"urn": "1999.01.0005", "author": "Aeschylus", "title": "Libation Bearers", "genre": "tragedy"},
            {"urn": "1999.01.0007", "author": "Aeschylus", "title": "Eumenides", "genre": "tragedy"},
            {"urn": "1999.01.0009", "author": "Aeschylus", "title": "Persians", "genre": "tragedy"},
            {"urn": "1999.01.0011", "author": "Aeschylus", "title": "Prometheus Bound", "genre": "tragedy"},
            {"urn": "1999.01.0013", "author": "Aeschylus", "title": "Seven Against Thebes", "genre": "tragedy"},
            {"urn": "1999.01.0015", "author": "Aeschylus", "title": "Suppliants", "genre": "tragedy"},
            
            {"urn": "1999.01.0183", "author": "Sophocles", "title": "Ajax", "genre": "tragedy"},
            {"urn": "1999.01.0185", "author": "Sophocles", "title": "Antigone", "genre": "tragedy"},
            {"urn": "1999.01.0187", "author": "Sophocles", "title": "Electra", "genre": "tragedy"},
            {"urn": "1999.01.0189", "author": "Sophocles", "title": "Oedipus at Colonus", "genre": "tragedy"},
            {"urn": "1999.01.0191", "author": "Sophocles", "title": "Oedipus Tyrannus", "genre": "tragedy"},
            {"urn": "1999.01.0193", "author": "Sophocles", "title": "Philoctetes", "genre": "tragedy"},
            {"urn": "1999.01.0195", "author": "Sophocles", "title": "Trachiniae", "genre": "tragedy"},
            
            {"urn": "1999.01.0083", "author": "Euripides", "title": "Alcestis", "genre": "tragedy"},
            {"urn": "1999.01.0085", "author": "Euripides", "title": "Andromache", "genre": "tragedy"},
            {"urn": "1999.01.0087", "author": "Euripides", "title": "Bacchae", "genre": "tragedy"},
            {"urn": "1999.01.0089", "author": "Euripides", "title": "Cyclops", "genre": "satyr"},
            {"urn": "1999.01.0091", "author": "Euripides", "title": "Electra", "genre": "tragedy"},
            {"urn": "1999.01.0093", "author": "Euripides", "title": "Hecuba", "genre": "tragedy"},
            {"urn": "1999.01.0095", "author": "Euripides", "title": "Helen", "genre": "tragedy"},
            {"urn": "1999.01.0097", "author": "Euripides", "title": "Heracleidae", "genre": "tragedy"},
            {"urn": "1999.01.0099", "author": "Euripides", "title": "Heracles", "genre": "tragedy"},
            {"urn": "1999.01.0101", "author": "Euripides", "title": "Hippolytus", "genre": "tragedy"},
            {"urn": "1999.01.0103", "author": "Euripides", "title": "Ion", "genre": "tragedy"},
            {"urn": "1999.01.0105", "author": "Euripides", "title": "Iphigenia at Aulis", "genre": "tragedy"},
            {"urn": "1999.01.0107", "author": "Euripides", "title": "Iphigenia in Tauris", "genre": "tragedy"},
            {"urn": "1999.01.0109", "author": "Euripides", "title": "Medea", "genre": "tragedy"},
            {"urn": "1999.01.0111", "author": "Euripides", "title": "Orestes", "genre": "tragedy"},
            {"urn": "1999.01.0113", "author": "Euripides", "title": "Phoenissae", "genre": "tragedy"},
            {"urn": "1999.01.0115", "author": "Euripides", "title": "Rhesus", "genre": "tragedy"},
            {"urn": "1999.01.0117", "author": "Euripides", "title": "Suppliants", "genre": "tragedy"},
            {"urn": "1999.01.0119", "author": "Euripides", "title": "Trojan Women", "genre": "tragedy"},
            
            # Comedy
            {"urn": "1999.01.0023", "author": "Aristophanes", "title": "Acharnians", "genre": "comedy"},
            {"urn": "1999.01.0025", "author": "Aristophanes", "title": "Birds", "genre": "comedy"},
            {"urn": "1999.01.0027", "author": "Aristophanes", "title": "Clouds", "genre": "comedy"},
            {"urn": "1999.01.0029", "author": "Aristophanes", "title": "Ecclesiazusae", "genre": "comedy"},
            {"urn": "1999.01.0031", "author": "Aristophanes", "title": "Frogs", "genre": "comedy"},
            {"urn": "1999.01.0033", "author": "Aristophanes", "title": "Knights", "genre": "comedy"},
            {"urn": "1999.01.0035", "author": "Aristophanes", "title": "Lysistrata", "genre": "comedy"},
            {"urn": "1999.01.0037", "author": "Aristophanes", "title": "Peace", "genre": "comedy"},
            {"urn": "1999.01.0039", "author": "Aristophanes", "title": "Plutus", "genre": "comedy"},
            {"urn": "1999.01.0041", "author": "Aristophanes", "title": "Thesmophoriazusae", "genre": "comedy"},
            {"urn": "1999.01.0043", "author": "Aristophanes", "title": "Wasps", "genre": "comedy"},
            
            # History
            {"urn": "1999.01.0125", "author": "Herodotus", "title": "Histories", "genre": "history"},
            {"urn": "1999.01.0199", "author": "Thucydides", "title": "History of the Peloponnesian War", "genre": "history"},
            {"urn": "1999.01.0201", "author": "Xenophon", "title": "Anabasis", "genre": "history"},
            {"urn": "1999.01.0205", "author": "Xenophon", "title": "Hellenica", "genre": "history"},
            {"urn": "1999.01.0165", "author": "Polybius", "title": "Histories", "genre": "history"},
            
            # Philosophy
            {"urn": "1999.01.0167", "author": "Plato", "title": "Republic", "genre": "philosophy"},
            {"urn": "1999.01.0169", "author": "Plato", "title": "Apology", "genre": "philosophy"},
            {"urn": "1999.01.0171", "author": "Plato", "title": "Symposium", "genre": "philosophy"},
            {"urn": "1999.01.0173", "author": "Plato", "title": "Phaedo", "genre": "philosophy"},
            {"urn": "1999.01.0175", "author": "Plato", "title": "Phaedrus", "genre": "philosophy"},
            {"urn": "1999.01.0177", "author": "Plato", "title": "Laws", "genre": "philosophy"},
            
            {"urn": "1999.01.0045", "author": "Aristotle", "title": "Nicomachean Ethics", "genre": "philosophy"},
            {"urn": "1999.01.0057", "author": "Aristotle", "title": "Politics", "genre": "philosophy"},
            {"urn": "1999.01.0059", "author": "Aristotle", "title": "Poetics", "genre": "philosophy"},
            {"urn": "1999.01.0061", "author": "Aristotle", "title": "Rhetoric", "genre": "philosophy"},
            {"urn": "1999.01.0047", "author": "Aristotle", "title": "Metaphysics", "genre": "philosophy"},
            
            # Oratory
            {"urn": "1999.01.0069", "author": "Demosthenes", "title": "Olynthiacs", "genre": "oratory"},
            {"urn": "1999.01.0071", "author": "Demosthenes", "title": "Philippics", "genre": "oratory"},
            {"urn": "1999.01.0073", "author": "Demosthenes", "title": "On the Crown", "genre": "oratory"},
            {"urn": "1999.01.0153", "author": "Lysias", "title": "Orations", "genre": "oratory"},
            {"urn": "1999.01.0143", "author": "Isocrates", "title": "Orations", "genre": "oratory"},
            
            # Later Greek
            {"urn": "1999.01.0161", "author": "Plutarch", "title": "Parallel Lives", "genre": "biography"},
            {"urn": "1999.01.0149", "author": "Lucian", "title": "Works", "genre": "satire"},
            {"urn": "1999.01.0159", "author": "Pausanias", "title": "Description of Greece", "genre": "geography"},
        ]
        
        return greek_texts
    
    def download_text(self, urn: str) -> Optional[str]:
        """Download text from Perseus by URN"""
        cache_file = self.cache_dir / f"{urn.replace('.', '_')}.txt"
        
        # Check cache
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
        
        try:
            # Try XML endpoint
            url = f"{PERSEUS_CONFIG['xml_url']}{urn}"
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Parse XML and extract text
                text = self._extract_text_from_xml(response.text)
                if text:
                    cache_file.write_text(text, encoding='utf-8')
                    return text
                    
        except Exception as e:
            logger.warning(f"Failed to download {urn}: {e}")
            
        return None
    
    def _extract_text_from_xml(self, xml_content: str) -> str:
        """Extract plain text from Perseus XML"""
        try:
            # Remove XML declaration and parse
            xml_content = re.sub(r'<\?xml[^>]+\?>', '', xml_content)
            root = ET.fromstring(f"<root>{xml_content}</root>")
            
            # Extract all text content
            text_parts = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())
                    
            return ' '.join(text_parts)
            
        except ET.ParseError:
            # Fallback: use BeautifulSoup
            soup = BeautifulSoup(xml_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
    
    def collect_all(self) -> List[RawText]:
        """Collect all available Greek texts"""
        texts = []
        catalog = self.get_greek_texts_catalog()
        
        for entry in catalog:
            logger.info(f"Collecting {entry['author']} - {entry['title']}...")
            
            content = self.download_text(entry['urn'])
            if content:
                period, century = classify_period(entry['author'], entry['title'])
                
                metadata = TextMetadata(
                    id=f"perseus_{entry['urn']}",
                    title=entry['title'],
                    author=entry['author'],
                    language="grc",
                    period=period,
                    century=century,
                    genre=entry.get('genre', ''),
                    source="Perseus Digital Library",
                    source_url=f"{PERSEUS_CONFIG['text_url']}{entry['urn']}"
                )
                
                texts.append(RawText(metadata=metadata, content=content))
                logger.info(f"  Collected {len(content)} characters")
                
            time.sleep(1)  # Rate limiting
            
        return texts

# =============================================================================
# FIRST1KGREEK COLLECTOR
# =============================================================================

class First1KGreekCollector:
    """Collect texts from First1KGreek GitHub repository"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "first1k"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_file_list(self) -> List[str]:
        """Get list of XML files from repository"""
        try:
            response = requests.get(FIRST1KGREEK_CONFIG['catalog_url'], timeout=60)
            if response.status_code == 200:
                data = response.json()
                files = [
                    item['path'] for item in data.get('tree', [])
                    if item['path'].endswith('.xml') and 'data/' in item['path']
                ]
                return files[:200]  # Limit to first 200 files
        except Exception as e:
            logger.error(f"Failed to get First1KGreek file list: {e}")
        return []
    
    def download_file(self, file_path: str) -> Optional[str]:
        """Download a single XML file"""
        cache_file = self.cache_dir / file_path.replace('/', '_')
        
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
            
        try:
            url = f"{FIRST1KGREEK_CONFIG['base_url']}{file_path}"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                cache_file.write_text(response.text, encoding='utf-8')
                return response.text
        except Exception as e:
            logger.warning(f"Failed to download {file_path}: {e}")
            
        return None
    
    def parse_tei_xml(self, xml_content: str, file_path: str) -> Optional[RawText]:
        """Parse TEI XML and extract text with metadata"""
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            
            # Extract metadata from TEI header
            title = ""
            author = ""
            
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)
                
            author_elem = soup.find('author')
            if author_elem:
                author = author_elem.get_text(strip=True)
            
            # Extract text content
            body = soup.find('body') or soup.find('text')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            if not text or len(text) < 100:
                return None
                
            period, century = classify_period(author, title)
            
            # Generate ID from file path
            text_id = f"first1k_{hashlib.md5(file_path.encode()).hexdigest()[:12]}"
            
            metadata = TextMetadata(
                id=text_id,
                title=title or file_path.split('/')[-1],
                author=author or "Unknown",
                language="grc",
                period=period,
                century=century,
                source="First1KGreek",
                source_url=f"https://github.com/OpenGreekAndLatin/First1KGreek/blob/master/{file_path}"
            )
            
            return RawText(metadata=metadata, content=text, xml_content=xml_content)
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def collect_all(self) -> List[RawText]:
        """Collect all texts from First1KGreek"""
        texts = []
        files = self.get_file_list()
        
        logger.info(f"Found {len(files)} files in First1KGreek")
        
        for file_path in files:
            content = self.download_file(file_path)
            if content:
                raw_text = self.parse_tei_xml(content, file_path)
                if raw_text:
                    texts.append(raw_text)
                    logger.info(f"  Collected: {raw_text.metadata.title[:50]}...")
                    
            time.sleep(0.5)  # Rate limiting
            
        return texts

# =============================================================================
# PROIEL TREEBANK COLLECTOR
# =============================================================================

class PROIELCollector:
    """Collect texts from PROIEL Treebank"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "proiel"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_treebank(self, filename: str) -> Optional[str]:
        """Download PROIEL treebank file"""
        cache_file = self.cache_dir / filename
        
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
            
        try:
            url = f"{PROIEL_CONFIG['base_url']}{filename}"
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                cache_file.write_text(response.text, encoding='utf-8')
                return response.text
        except Exception as e:
            logger.error(f"Failed to download PROIEL {filename}: {e}")
            
        return None
    
    def parse_proiel_xml(self, xml_content: str, filename: str) -> List[Dict]:
        """Parse PROIEL XML format and extract annotated sentences"""
        sentences = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # PROIEL XML structure: source > div > sentence > token
            for source in root.findall('.//source'):
                source_id = source.get('id', '')
                
                for sentence in source.findall('.//sentence'):
                    sent_id = sentence.get('id', '')
                    tokens = []
                    
                    for token in sentence.findall('.//token'):
                        token_data = {
                            'id': token.get('id', ''),
                            'form': token.get('form', ''),
                            'lemma': token.get('lemma', ''),
                            'pos': token.get('part-of-speech', ''),
                            'morph': token.get('morphology', ''),
                            'head': token.get('head-id', ''),
                            'relation': token.get('relation', ''),
                            'presentation_before': token.get('presentation-before', ''),
                            'presentation_after': token.get('presentation-after', '')
                        }
                        tokens.append(token_data)
                    
                    if tokens:
                        sentences.append({
                            'id': f"{source_id}_{sent_id}",
                            'source': filename,
                            'tokens': tokens
                        })
                        
        except ET.ParseError as e:
            logger.error(f"Failed to parse PROIEL XML {filename}: {e}")
            
        return sentences
    
    def collect_all(self) -> Dict[str, List[Dict]]:
        """Collect all PROIEL treebank data"""
        all_data = {}
        
        for filename in PROIEL_CONFIG['files']:
            logger.info(f"Collecting PROIEL {filename}...")
            content = self.download_treebank(filename)
            
            if content:
                sentences = self.parse_proiel_xml(content, filename)
                all_data[filename] = sentences
                logger.info(f"  Collected {len(sentences)} sentences")
                
        return all_data

# =============================================================================
# NEW TESTAMENT GREEK
# =============================================================================

class NewTestamentCollector:
    """Collect Greek New Testament texts"""
    
    SBLGNT_URL = "https://raw.githubusercontent.com/morphgnt/sblgnt/master/sblgnt/"
    
    NT_BOOKS = [
        ("01-matthew.txt", "Matthew"),
        ("02-mark.txt", "Mark"),
        ("03-luke.txt", "Luke"),
        ("04-john.txt", "John"),
        ("05-acts.txt", "Acts"),
        ("06-romans.txt", "Romans"),
        ("07-1corinthians.txt", "1 Corinthians"),
        ("08-2corinthians.txt", "2 Corinthians"),
        ("09-galatians.txt", "Galatians"),
        ("10-ephesians.txt", "Ephesians"),
        ("11-philippians.txt", "Philippians"),
        ("12-colossians.txt", "Colossians"),
        ("13-1thessalonians.txt", "1 Thessalonians"),
        ("14-2thessalonians.txt", "2 Thessalonians"),
        ("15-1timothy.txt", "1 Timothy"),
        ("16-2timothy.txt", "2 Timothy"),
        ("17-titus.txt", "Titus"),
        ("18-philemon.txt", "Philemon"),
        ("19-hebrews.txt", "Hebrews"),
        ("20-james.txt", "James"),
        ("21-1peter.txt", "1 Peter"),
        ("22-2peter.txt", "2 Peter"),
        ("23-1john.txt", "1 John"),
        ("24-2john.txt", "2 John"),
        ("25-3john.txt", "3 John"),
        ("26-jude.txt", "Jude"),
        ("27-revelation.txt", "Revelation")
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "nt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_book(self, filename: str) -> Optional[str]:
        """Download NT book"""
        cache_file = self.cache_dir / filename
        
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
            
        try:
            url = f"{self.SBLGNT_URL}{filename}"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                cache_file.write_text(response.text, encoding='utf-8')
                return response.text
        except Exception as e:
            logger.warning(f"Failed to download NT {filename}: {e}")
            
        return None
    
    def parse_morphgnt(self, content: str) -> List[Dict]:
        """Parse MorphGNT format"""
        tokens = []
        
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 7:
                token = {
                    'ref': parts[0],  # Book.Chapter.Verse
                    'pos': parts[1],
                    'parse': parts[2],
                    'text': parts[3],
                    'word': parts[4],
                    'normalized': parts[5],
                    'lemma': parts[6]
                }
                tokens.append(token)
                
        return tokens
    
    def collect_all(self) -> List[RawText]:
        """Collect all NT books"""
        texts = []
        
        for filename, book_name in self.NT_BOOKS:
            logger.info(f"Collecting NT {book_name}...")
            content = self.download_book(filename)
            
            if content:
                tokens = self.parse_morphgnt(content)
                
                # Reconstruct text
                text = ' '.join(t['text'] for t in tokens)
                
                metadata = TextMetadata(
                    id=f"nt_{filename.replace('.txt', '')}",
                    title=f"New Testament - {book_name}",
                    author="New Testament",
                    language="grc",
                    period="koine",
                    century="1st CE",
                    genre="religious",
                    source="MorphGNT/SBLGNT",
                    source_url=f"{self.SBLGNT_URL}{filename}",
                    word_count=len(tokens)
                )
                
                texts.append(RawText(
                    metadata=metadata,
                    content=text,
                    tei_header={'tokens': tokens}
                ))
                
        return texts

# =============================================================================
# MASTER COLLECTOR
# =============================================================================

class MasterTextCollector:
    """Master collector that coordinates all sources"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.perseus = PerseusCollector(data_dir)
        self.first1k = First1KGreekCollector(data_dir)
        self.proiel = PROIELCollector(data_dir)
        self.nt = NewTestamentCollector(data_dir)
        
    def collect_all_sources(self) -> Dict[str, List]:
        """Collect from all sources"""
        results = {
            'perseus': [],
            'first1k': [],
            'proiel': {},
            'nt': []
        }
        
        logger.info("=" * 60)
        logger.info("STARTING MASTER TEXT COLLECTION")
        logger.info("=" * 60)
        
        # Collect from each source
        try:
            logger.info("\n--- Perseus Digital Library ---")
            results['perseus'] = self.perseus.collect_all()
            logger.info(f"Perseus: {len(results['perseus'])} texts")
        except Exception as e:
            logger.error(f"Perseus collection failed: {e}")
            
        try:
            logger.info("\n--- First1KGreek ---")
            results['first1k'] = self.first1k.collect_all()
            logger.info(f"First1KGreek: {len(results['first1k'])} texts")
        except Exception as e:
            logger.error(f"First1KGreek collection failed: {e}")
            
        try:
            logger.info("\n--- PROIEL Treebank ---")
            results['proiel'] = self.proiel.collect_all()
            total_sent = sum(len(v) for v in results['proiel'].values())
            logger.info(f"PROIEL: {total_sent} sentences")
        except Exception as e:
            logger.error(f"PROIEL collection failed: {e}")
            
        try:
            logger.info("\n--- New Testament (MorphGNT) ---")
            results['nt'] = self.nt.collect_all()
            logger.info(f"NT: {len(results['nt'])} books")
        except Exception as e:
            logger.error(f"NT collection failed: {e}")
            
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def get_statistics(self, results: Dict) -> Dict:
        """Get collection statistics"""
        stats = {
            'sources': {},
            'total_texts': 0,
            'total_characters': 0,
            'by_period': {},
            'by_genre': {}
        }
        
        # Process Perseus
        for text in results.get('perseus', []):
            stats['total_texts'] += 1
            stats['total_characters'] += len(text.content)
            
            period = text.metadata.period
            stats['by_period'][period] = stats['by_period'].get(period, 0) + 1
            
            genre = text.metadata.genre
            if genre:
                stats['by_genre'][genre] = stats['by_genre'].get(genre, 0) + 1
                
        stats['sources']['perseus'] = len(results.get('perseus', []))
        
        # Process First1K
        for text in results.get('first1k', []):
            stats['total_texts'] += 1
            stats['total_characters'] += len(text.content)
            
            period = text.metadata.period
            stats['by_period'][period] = stats['by_period'].get(period, 0) + 1
            
        stats['sources']['first1k'] = len(results.get('first1k', []))
        
        # Process NT
        for text in results.get('nt', []):
            stats['total_texts'] += 1
            stats['total_characters'] += len(text.content)
            
        stats['sources']['nt'] = len(results.get('nt', []))
        
        # Process PROIEL
        proiel_sentences = sum(len(v) for v in results.get('proiel', {}).values())
        stats['sources']['proiel_sentences'] = proiel_sentences
        
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data"
    
    collector = MasterTextCollector(data_dir)
    results = collector.collect_all_sources()
    stats = collector.get_statistics(results)
    
    print("\n" + "=" * 60)
    print("COLLECTION STATISTICS")
    print("=" * 60)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print("\nBy Source:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")
    print("\nBy Period:")
    for period, count in stats['by_period'].items():
        print(f"  {period}: {count}")
