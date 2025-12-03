"""
PROIEL Text Collector - Based on 'Texts for PROEIL.xlsx'
Collects REAL texts from verified sources for PROIEL-style annotation

Sources from Excel:
- Gutenberg Project
- Perseus Digital Library
- METS (Middle English Text Series)
- Internet Archive
- Wikisource
- Oxford Text Archive
- Michigan Library
- Latin Library
- Various academic repositories
"""

import os
import re
import json
import sqlite3
import logging
import requests
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TEXT SOURCES FROM EXCEL 'Texts for PROEIL.xlsx'
# =============================================================================

PROIEL_TEXT_SOURCES = {
    # ========== ANCIENT GREEK ==========
    "homer_iliad": {
        "id": "athdgc.001.001.grc",
        "title": "Iliad",
        "author": "Homer",
        "period": "archaic",
        "century": "8th BCE",
        "language": "grc",
        "genre": "epic",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133",
            "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml"
        ],
        "status": "priority"
    },
    "homer_odyssey": {
        "id": "athdgc.001.002.grc",
        "title": "Odyssey",
        "author": "Homer",
        "period": "archaic",
        "century": "8th BCE",
        "language": "grc",
        "genre": "epic",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0135",
            "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0012/tlg002/tlg0012.tlg002.perseus-grc2.xml"
        ],
        "status": "priority"
    },
    "hesiod_theogony": {
        "id": "athdgc.002.001.grc",
        "title": "Theogony",
        "author": "Hesiod",
        "period": "archaic",
        "century": "7th BCE",
        "language": "grc",
        "genre": "epic",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0129"
        ],
        "status": "priority"
    },
    "hesiod_works_days": {
        "id": "athdgc.002.002.grc",
        "title": "Works and Days",
        "author": "Hesiod",
        "period": "archaic",
        "century": "7th BCE",
        "language": "grc",
        "genre": "didactic",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0131"
        ],
        "status": "priority"
    },
    "herodotus_histories": {
        "id": "athdgc.003.001.grc",
        "title": "Histories",
        "author": "Herodotus",
        "period": "classical",
        "century": "5th BCE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0125",
            "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0016/tlg001/tlg0016.tlg001.perseus-grc1.xml"
        ],
        "status": "priority"
    },
    "thucydides_history": {
        "id": "athdgc.004.001.grc",
        "title": "History of the Peloponnesian War",
        "author": "Thucydides",
        "period": "classical",
        "century": "5th BCE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0199"
        ],
        "status": "priority"
    },
    "plato_republic": {
        "id": "athdgc.005.001.grc",
        "title": "Republic",
        "author": "Plato",
        "period": "classical",
        "century": "4th BCE",
        "language": "grc",
        "genre": "philosophy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0167"
        ],
        "status": "priority"
    },
    "plato_symposium": {
        "id": "athdgc.005.002.grc",
        "title": "Symposium",
        "author": "Plato",
        "period": "classical",
        "century": "4th BCE",
        "language": "grc",
        "genre": "philosophy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0173"
        ],
        "status": "priority"
    },
    "aristotle_nicomachean": {
        "id": "athdgc.006.001.grc",
        "title": "Nicomachean Ethics",
        "author": "Aristotle",
        "period": "classical",
        "century": "4th BCE",
        "language": "grc",
        "genre": "philosophy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0053"
        ],
        "status": "priority"
    },
    "xenophon_anabasis": {
        "id": "athdgc.007.001.grc",
        "title": "Anabasis",
        "author": "Xenophon",
        "period": "classical",
        "century": "4th BCE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0201"
        ],
        "status": "priority"
    },
    "sophocles_antigone": {
        "id": "athdgc.008.001.grc",
        "title": "Antigone",
        "author": "Sophocles",
        "period": "classical",
        "century": "5th BCE",
        "language": "grc",
        "genre": "tragedy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0185"
        ],
        "status": "priority"
    },
    "euripides_medea": {
        "id": "athdgc.009.001.grc",
        "title": "Medea",
        "author": "Euripides",
        "period": "classical",
        "century": "5th BCE",
        "language": "grc",
        "genre": "tragedy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0113"
        ],
        "status": "priority"
    },
    "aristophanes_clouds": {
        "id": "athdgc.010.001.grc",
        "title": "Clouds",
        "author": "Aristophanes",
        "period": "classical",
        "century": "5th BCE",
        "language": "grc",
        "genre": "comedy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0027"
        ],
        "status": "priority"
    },
    "demosthenes_philippics": {
        "id": "athdgc.011.001.grc",
        "title": "Philippics",
        "author": "Demosthenes",
        "period": "classical",
        "century": "4th BCE",
        "language": "grc",
        "genre": "oratory",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0069"
        ],
        "status": "priority"
    },
    "lysias_speeches": {
        "id": "athdgc.012.001.grc",
        "title": "Speeches",
        "author": "Lysias",
        "period": "classical",
        "century": "5th-4th BCE",
        "language": "grc",
        "genre": "oratory",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0153"
        ],
        "status": "priority"
    },
    
    # ========== HELLENISTIC/KOINE GREEK ==========
    "polybius_histories": {
        "id": "athdgc.020.001.grc",
        "title": "Histories",
        "author": "Polybius",
        "period": "hellenistic",
        "century": "2nd BCE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0233"
        ],
        "status": "priority"
    },
    "septuagint_genesis": {
        "id": "athdgc.021.001.grc",
        "title": "Genesis (LXX)",
        "author": "Anonymous",
        "period": "hellenistic",
        "century": "3rd BCE",
        "language": "grc",
        "genre": "religious",
        "sources": [
            "https://www.ellopos.net/elencyclop/septuagint/genesis.asp"
        ],
        "status": "priority"
    },
    "new_testament_matthew": {
        "id": "athdgc.022.001.grc",
        "title": "Gospel of Matthew",
        "author": "Matthew",
        "period": "koine",
        "century": "1st CE",
        "language": "grc",
        "genre": "religious",
        "sources": [
            "https://raw.githubusercontent.com/morphgnt/sblgnt/master/01-matthew.txt",
            "https://www.ellopos.net/elpenor/greek-texts/new-testament/matthew.asp"
        ],
        "status": "priority"
    },
    "new_testament_john": {
        "id": "athdgc.022.004.grc",
        "title": "Gospel of John",
        "author": "John",
        "period": "koine",
        "century": "1st CE",
        "language": "grc",
        "genre": "religious",
        "sources": [
            "https://raw.githubusercontent.com/morphgnt/sblgnt/master/04-john.txt"
        ],
        "status": "priority"
    },
    "josephus_jewish_war": {
        "id": "athdgc.023.001.grc",
        "title": "Jewish War",
        "author": "Josephus",
        "period": "koine",
        "century": "1st CE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0147"
        ],
        "status": "priority"
    },
    "plutarch_lives": {
        "id": "athdgc.024.001.grc",
        "title": "Parallel Lives",
        "author": "Plutarch",
        "period": "koine",
        "century": "1st-2nd CE",
        "language": "grc",
        "genre": "biography",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:2008.01.0006"
        ],
        "status": "priority"
    },
    "epictetus_discourses": {
        "id": "athdgc.025.001.grc",
        "title": "Discourses",
        "author": "Epictetus",
        "period": "koine",
        "century": "1st-2nd CE",
        "language": "grc",
        "genre": "philosophy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0235"
        ],
        "status": "priority"
    },
    "marcus_aurelius_meditations": {
        "id": "athdgc.026.001.grc",
        "title": "Meditations",
        "author": "Marcus Aurelius",
        "period": "koine",
        "century": "2nd CE",
        "language": "grc",
        "genre": "philosophy",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:2008.01.0641"
        ],
        "status": "priority"
    },
    
    # ========== LATE ANTIQUE/PATRISTIC ==========
    "basil_hexaemeron": {
        "id": "athdgc.030.001.grc",
        "title": "Hexaemeron",
        "author": "Basil of Caesarea",
        "period": "late_antique",
        "century": "4th CE",
        "language": "grc",
        "genre": "religious",
        "sources": [
            "https://www.documentacatholicaomnia.eu/04z/z_0329-0379__Basilius_Caesariensis__Homiliae_In_Hexaemeron__GR.pdf.html"
        ],
        "status": "secondary"
    },
    "john_chrysostom_homilies": {
        "id": "athdgc.031.001.grc",
        "title": "Homilies on Matthew",
        "author": "John Chrysostom",
        "period": "late_antique",
        "century": "4th CE",
        "language": "grc",
        "genre": "religious",
        "sources": [],
        "status": "secondary"
    },
    
    # ========== BYZANTINE ==========
    "procopius_wars": {
        "id": "athdgc.040.001.grc",
        "title": "Wars",
        "author": "Procopius",
        "period": "byzantine",
        "century": "6th CE",
        "language": "grc",
        "genre": "history",
        "sources": [
            "https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:2007.01.0025"
        ],
        "status": "priority"
    },
    "anna_comnena_alexiad": {
        "id": "athdgc.041.001.grc",
        "title": "Alexiad",
        "author": "Anna Comnena",
        "period": "byzantine",
        "century": "12th CE",
        "language": "grc",
        "genre": "history",
        "sources": [],
        "status": "priority"
    },
    "psellus_chronographia": {
        "id": "athdgc.042.001.grc",
        "title": "Chronographia",
        "author": "Michael Psellus",
        "period": "byzantine",
        "century": "11th CE",
        "language": "grc",
        "genre": "history",
        "sources": [],
        "status": "priority"
    },
    "digenes_akritas": {
        "id": "athdgc.043.001.grc",
        "title": "Digenes Akritas",
        "author": "Anonymous",
        "period": "byzantine",
        "century": "12th CE",
        "language": "grc",
        "genre": "epic",
        "sources": [],
        "status": "priority"
    },
    
    # ========== MEDIEVAL ENGLISH (from Excel) ==========
    "beowulf": {
        "id": "athdgc.001.001.ang",
        "title": "Beowulf",
        "author": "Anonymous",
        "period": "medieval",
        "century": "8th-11th CE",
        "language": "ang",
        "genre": "epic",
        "sources": [
            "https://www.gutenberg.org/cache/epub/16328/pg16328.txt"
        ],
        "status": "uploaded"
    },
    "sir_gawain": {
        "id": "athdgc.002.010.enm",
        "title": "Sir Gawain and the Green Knight",
        "author": "Anonymous",
        "period": "medieval",
        "century": "14th CE",
        "language": "enm",
        "genre": "romance",
        "sources": [
            "https://www.gutenberg.org/cache/epub/14568/pg14568.txt",
            "https://www.gersum.org/texts/show?occurrences=Gaw"
        ],
        "status": "uploaded"
    },
    "sir_orfeo": {
        "id": "athdgc.002.011.enm",
        "title": "Sir Orfeo",
        "author": "Anonymous",
        "period": "medieval",
        "century": "13th-14th CE",
        "language": "enm",
        "genre": "romance",
        "sources": [
            "https://metseditions.org/texts/A4GxkPrUzD59hz0LcavGPc7152ZE6Wz"
        ],
        "status": "uploaded"
    },
    "lay_le_freine": {
        "id": "athdgc.002.007.ang",
        "title": "Lay le Freine",
        "author": "Anonymous",
        "period": "medieval",
        "century": "13th-14th CE",
        "language": "enm",
        "genre": "romance",
        "sources": [],
        "status": "uploaded"
    },
    "canterbury_tales": {
        "id": "athdgc.016.001.enm",
        "title": "Canterbury Tales",
        "author": "Geoffrey Chaucer",
        "period": "medieval",
        "century": "14th CE",
        "language": "enm",
        "genre": "poetry",
        "sources": [
            "https://www.gutenberg.org/cache/epub/22120/pg22120.txt"
        ],
        "status": "uploaded"
    },
    "piers_plowman": {
        "id": "athdgc.032.001.enm",
        "title": "The Vision of Piers Plowman",
        "author": "William Langland",
        "period": "medieval",
        "century": "14th CE",
        "language": "enm",
        "genre": "allegory",
        "sources": [
            "https://www.gutenberg.org/cache/epub/43660/pg43660.txt"
        ],
        "status": "uploaded"
    },
    "wycliffe_bible": {
        "id": "athdgc.001.001.001.enm",
        "title": "Wycliffe Bible",
        "author": "John Wycliffe",
        "period": "medieval",
        "century": "14th CE",
        "language": "enm",
        "genre": "religious",
        "sources": [
            "https://www.gutenberg.org/cache/epub/10625/pg10625.txt"
        ],
        "status": "uploaded"
    },
    "tyndale_bible": {
        "id": "athdgc.001.001.ene",
        "title": "Tyndale's New Testament",
        "author": "William Tyndale",
        "period": "early_modern",
        "century": "16th CE",
        "language": "ene",
        "genre": "religious",
        "sources": [
            "https://www.gutenberg.org/cache/epub/53646/pg53646.txt"
        ],
        "status": "uploaded"
    },
    
    # ========== LATIN (comparative) ==========
    "vulgate_genesis": {
        "id": "athdgc.050.001.lat",
        "title": "Vulgate Genesis",
        "author": "Jerome",
        "period": "late_antique",
        "century": "4th CE",
        "language": "lat",
        "genre": "religious",
        "sources": [
            "https://www.thelatinlibrary.com/bible/genesis.shtml"
        ],
        "status": "priority"
    },
    "ovid_metamorphoses": {
        "id": "athdgc.051.001.lat",
        "title": "Metamorphoses",
        "author": "Ovid",
        "period": "classical",
        "century": "1st CE",
        "language": "lat",
        "genre": "epic",
        "sources": [
            "https://www.thelatinlibrary.com/ovid/ovid.met1.shtml"
        ],
        "status": "priority"
    },
    "virgil_aeneid": {
        "id": "athdgc.052.001.lat",
        "title": "Aeneid",
        "author": "Virgil",
        "period": "classical",
        "century": "1st BCE",
        "language": "lat",
        "genre": "epic",
        "sources": [
            "https://www.thelatinlibrary.com/verg.html"
        ],
        "status": "priority"
    },
    
    # ========== MODERN GREEK ==========
    "erotokritos": {
        "id": "athdgc.060.001.ell",
        "title": "Erotokritos",
        "author": "Vitsentzos Kornaros",
        "period": "early_modern",
        "century": "17th CE",
        "language": "ell",
        "genre": "romance",
        "sources": [],
        "status": "priority"
    }
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CollectedText:
    """A collected text with full metadata"""
    id: str
    title: str
    author: str
    period: str
    century: str
    language: str
    genre: str
    content: str
    source_url: str
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    collection_date: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        if not self.collection_date:
            self.collection_date = datetime.now().isoformat()
        if not self.checksum and self.content:
            self.checksum = hashlib.md5(self.content.encode()).hexdigest()
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())
        if not self.char_count and self.content:
            self.char_count = len(self.content)


# =============================================================================
# TEXT COLLECTORS
# =============================================================================

class PerseusCollector:
    """Collect texts from Perseus Digital Library"""
    
    BASE_URL = "https://www.perseus.tufts.edu/hopper"
    GITHUB_BASE = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "perseus"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreekCorpusPlatform/1.0 (Academic Research)'
        })
    
    def collect_text(self, text_info: Dict) -> Optional[CollectedText]:
        """Collect a single text from Perseus"""
        text_id = text_info['id']
        cache_file = self.cache_dir / f"{text_id}.txt"
        
        # Check cache
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8')
            logger.info(f"Loaded from cache: {text_info['title']}")
        else:
            content = self._download_text(text_info)
            if content:
                cache_file.write_text(content, encoding='utf-8')
        
        if not content or len(content) < 100:
            return None
        
        return CollectedText(
            id=text_id,
            title=text_info['title'],
            author=text_info['author'],
            period=text_info['period'],
            century=text_info['century'],
            language=text_info['language'],
            genre=text_info['genre'],
            content=content,
            source_url=text_info['sources'][0] if text_info['sources'] else ""
        )
    
    def _download_text(self, text_info: Dict) -> Optional[str]:
        """Download text from Perseus sources"""
        for source_url in text_info.get('sources', []):
            try:
                if 'github' in source_url.lower():
                    content = self._download_github_xml(source_url)
                elif 'perseus' in source_url.lower():
                    content = self._download_perseus_html(source_url)
                else:
                    content = self._download_generic(source_url)
                
                if content and len(content) > 100:
                    logger.info(f"Downloaded: {text_info['title']} ({len(content)} chars)")
                    return content
                    
            except Exception as e:
                logger.warning(f"Failed to download {source_url}: {e}")
                continue
        
        return None
    
    def _download_github_xml(self, url: str) -> Optional[str]:
        """Download and parse XML from GitHub"""
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                return None
            
            # Parse TEI XML
            root = ET.fromstring(response.content)
            
            # Extract text from body
            namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Try different paths
            text_parts = []
            
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    # Skip metadata elements
                    if any(skip in elem.tag.lower() for skip in ['header', 'note', 'bibl']):
                        continue
                    text_parts.append(elem.text.strip())
            
            return ' '.join(text_parts)
            
        except Exception as e:
            logger.warning(f"GitHub XML parse error: {e}")
            return None
    
    def _download_perseus_html(self, url: str) -> Optional[str]:
        """Download and parse HTML from Perseus"""
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main text div
            text_div = soup.find('div', class_='text_container')
            if not text_div:
                text_div = soup.find('div', id='text_container')
            if not text_div:
                text_div = soup.find('div', class_='text')
            
            if text_div:
                # Remove notes and apparatus
                for note in text_div.find_all(['note', 'span'], class_=['note', 'apparatus']):
                    note.decompose()
                
                return text_div.get_text(separator=' ', strip=True)
            
            return None
            
        except Exception as e:
            logger.warning(f"Perseus HTML parse error: {e}")
            return None
    
    def _download_generic(self, url: str) -> Optional[str]:
        """Generic download for other sources"""
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                return None
            
            # Try to detect content type
            content_type = response.headers.get('content-type', '')
            
            if 'xml' in content_type:
                soup = BeautifulSoup(response.content, 'xml')
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            return soup.get_text(separator=' ', strip=True)
            
        except Exception as e:
            logger.warning(f"Generic download error: {e}")
            return None


class GutenbergCollector:
    """Collect texts from Project Gutenberg"""
    
    BASE_URL = "https://www.gutenberg.org"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "gutenberg"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def collect_text(self, text_info: Dict) -> Optional[CollectedText]:
        """Collect a single text from Gutenberg"""
        text_id = text_info['id']
        cache_file = self.cache_dir / f"{text_id}.txt"
        
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8')
            logger.info(f"Loaded from cache: {text_info['title']}")
        else:
            content = self._download_text(text_info)
            if content:
                cache_file.write_text(content, encoding='utf-8')
        
        if not content or len(content) < 100:
            return None
        
        return CollectedText(
            id=text_id,
            title=text_info['title'],
            author=text_info['author'],
            period=text_info['period'],
            century=text_info['century'],
            language=text_info['language'],
            genre=text_info['genre'],
            content=content,
            source_url=text_info['sources'][0] if text_info['sources'] else ""
        )
    
    def _download_text(self, text_info: Dict) -> Optional[str]:
        """Download text from Gutenberg"""
        for source_url in text_info.get('sources', []):
            if 'gutenberg' not in source_url.lower():
                continue
            
            try:
                response = self.session.get(source_url, timeout=60)
                if response.status_code == 200:
                    content = response.text
                    
                    # Clean Gutenberg header/footer
                    content = self._clean_gutenberg_text(content)
                    
                    if content and len(content) > 100:
                        logger.info(f"Downloaded from Gutenberg: {text_info['title']}")
                        return content
                        
            except Exception as e:
                logger.warning(f"Gutenberg download error: {e}")
        
        return None
    
    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Gutenberg header and footer"""
        # Find start marker
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT"
        ]
        
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                # Find end of line
                idx = text.find('\n', idx)
                if idx != -1:
                    text = text[idx+1:]
                break
        
        # Find end marker
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg"
        ]
        
        for marker in end_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx]
                break
        
        return text.strip()


class MorphGNTCollector:
    """Collect New Testament texts from MorphGNT"""
    
    BASE_URL = "https://raw.githubusercontent.com/morphgnt/sblgnt/master"
    
    BOOKS = {
        "01-matthew": "Gospel of Matthew",
        "02-mark": "Gospel of Mark",
        "03-luke": "Gospel of Luke",
        "04-john": "Gospel of John",
        "05-acts": "Acts of the Apostles",
        "06-romans": "Romans",
        "07-1corinthians": "1 Corinthians",
        "08-2corinthians": "2 Corinthians",
        "09-galatians": "Galatians",
        "10-ephesians": "Ephesians",
        "11-philippians": "Philippians",
        "12-colossians": "Colossians",
        "13-1thessalonians": "1 Thessalonians",
        "14-2thessalonians": "2 Thessalonians",
        "15-1timothy": "1 Timothy",
        "16-2timothy": "2 Timothy",
        "17-titus": "Titus",
        "18-philemon": "Philemon",
        "19-hebrews": "Hebrews",
        "20-james": "James",
        "21-1peter": "1 Peter",
        "22-2peter": "2 Peter",
        "23-1john": "1 John",
        "24-2john": "2 John",
        "25-3john": "3 John",
        "26-jude": "Jude",
        "27-revelation": "Revelation"
    }
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "morphgnt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def collect_all(self) -> List[CollectedText]:
        """Collect all NT books"""
        texts = []
        
        for book_file, book_name in self.BOOKS.items():
            text = self.collect_book(book_file, book_name)
            if text:
                texts.append(text)
            time.sleep(0.5)  # Rate limiting
        
        return texts
    
    def collect_book(self, book_file: str, book_name: str) -> Optional[CollectedText]:
        """Collect a single NT book"""
        cache_file = self.cache_dir / f"{book_file}.txt"
        
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8')
        else:
            url = f"{self.BASE_URL}/{book_file}.txt"
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code != 200:
                    return None
                
                # Parse MorphGNT format
                content = self._parse_morphgnt(response.text)
                cache_file.write_text(content, encoding='utf-8')
                
            except Exception as e:
                logger.warning(f"MorphGNT download error for {book_name}: {e}")
                return None
        
        if not content:
            return None
        
        return CollectedText(
            id=f"nt_{book_file}",
            title=book_name,
            author="New Testament",
            period="koine",
            century="1st CE",
            language="grc",
            genre="religious",
            content=content,
            source_url=f"{self.BASE_URL}/{book_file}.txt"
        )
    
    def _parse_morphgnt(self, raw_text: str) -> str:
        """Parse MorphGNT format to plain text"""
        words = []
        current_verse = ""
        
        for line in raw_text.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                # Format: book chapter verse pos parse word normalized lemma
                word = parts[5] if len(parts) > 5 else parts[-1]
                words.append(word)
        
        return ' '.join(words)


class LatinLibraryCollector:
    """Collect texts from The Latin Library"""
    
    BASE_URL = "https://www.thelatinlibrary.com"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "latinlibrary"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def collect_text(self, text_info: Dict) -> Optional[CollectedText]:
        """Collect a Latin text"""
        text_id = text_info['id']
        cache_file = self.cache_dir / f"{text_id}.txt"
        
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8')
        else:
            content = self._download_text(text_info)
            if content:
                cache_file.write_text(content, encoding='utf-8')
        
        if not content or len(content) < 100:
            return None
        
        return CollectedText(
            id=text_id,
            title=text_info['title'],
            author=text_info['author'],
            period=text_info['period'],
            century=text_info['century'],
            language=text_info['language'],
            genre=text_info['genre'],
            content=content,
            source_url=text_info['sources'][0] if text_info['sources'] else ""
        )
    
    def _download_text(self, text_info: Dict) -> Optional[str]:
        """Download from Latin Library"""
        for source_url in text_info.get('sources', []):
            if 'latinlibrary' not in source_url.lower():
                continue
            
            try:
                response = self.session.get(source_url, timeout=60)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove navigation
                for nav in soup.find_all(['table', 'hr']):
                    nav.decompose()
                
                # Get body text
                body = soup.find('body')
                if body:
                    return body.get_text(separator=' ', strip=True)
                    
            except Exception as e:
                logger.warning(f"Latin Library error: {e}")
        
        return None


# =============================================================================
# MASTER COLLECTOR
# =============================================================================

class PROIELTextCollector:
    """Master collector for all PROIEL texts"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize collectors
        self.perseus = PerseusCollector(self.cache_dir)
        self.gutenberg = GutenbergCollector(self.cache_dir)
        self.morphgnt = MorphGNTCollector(self.cache_dir)
        self.latin = LatinLibraryCollector(self.cache_dir)
        
        # Database
        self.db_path = self.data_dir / "collected_texts.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the collection database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collected_texts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                century TEXT,
                language TEXT,
                genre TEXT,
                content TEXT,
                source_url TEXT,
                word_count INTEGER,
                char_count INTEGER,
                sentence_count INTEGER,
                collection_date TEXT,
                checksum TEXT,
                status TEXT DEFAULT 'collected'
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_period ON collected_texts(period)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_language ON collected_texts(language)
        """)
        
        conn.commit()
        conn.close()
    
    def collect_all(self) -> Dict[str, int]:
        """Collect all texts from all sources"""
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'by_period': {},
            'by_language': {}
        }
        
        logger.info("=" * 60)
        logger.info("STARTING PROIEL TEXT COLLECTION")
        logger.info("=" * 60)
        
        # Collect from defined sources
        for text_id, text_info in PROIEL_TEXT_SOURCES.items():
            stats['total'] += 1
            
            try:
                text = self._collect_single_text(text_info)
                
                if text:
                    self._store_text(text)
                    stats['success'] += 1
                    
                    # Update period stats
                    period = text.period
                    stats['by_period'][period] = stats['by_period'].get(period, 0) + 1
                    
                    # Update language stats
                    lang = text.language
                    stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
                    
                    logger.info(f"✓ Collected: {text.title} ({text.word_count} words)")
                else:
                    stats['failed'] += 1
                    logger.warning(f"✗ Failed: {text_info['title']}")
                    
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Error collecting {text_info['title']}: {e}")
            
            # Rate limiting
            time.sleep(1)
        
        # Collect all NT books
        logger.info("\nCollecting New Testament...")
        nt_texts = self.morphgnt.collect_all()
        for text in nt_texts:
            self._store_text(text)
            stats['success'] += 1
            stats['by_period']['koine'] = stats['by_period'].get('koine', 0) + 1
            stats['by_language']['grc'] = stats['by_language'].get('grc', 0) + 1
        
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
        logger.info("=" * 60)
        
        return stats
    
    def _collect_single_text(self, text_info: Dict) -> Optional[CollectedText]:
        """Collect a single text using appropriate collector"""
        sources = text_info.get('sources', [])
        
        if not sources:
            return None
        
        # Determine which collector to use
        for source in sources:
            source_lower = source.lower()
            
            if 'perseus' in source_lower or 'github.com/perseusdl' in source_lower:
                return self.perseus.collect_text(text_info)
            elif 'gutenberg' in source_lower:
                return self.gutenberg.collect_text(text_info)
            elif 'latinlibrary' in source_lower:
                return self.latin.collect_text(text_info)
            elif 'morphgnt' in source_lower:
                # Handled separately
                pass
        
        # Try Perseus as default for Greek
        if text_info.get('language') == 'grc':
            return self.perseus.collect_text(text_info)
        
        return None
    
    def _store_text(self, text: CollectedText):
        """Store collected text in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO collected_texts
            (id, title, author, period, century, language, genre, content,
             source_url, word_count, char_count, sentence_count, collection_date, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text.id, text.title, text.author, text.period, text.century,
            text.language, text.genre, text.content, text.source_url,
            text.word_count, text.char_count, text.sentence_count,
            text.collection_date, text.checksum
        ))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM collected_texts")
        stats['total_texts'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(word_count) FROM collected_texts")
        result = cursor.fetchone()[0]
        stats['total_words'] = result if result else 0
        
        cursor.execute("SELECT SUM(char_count) FROM collected_texts")
        result = cursor.fetchone()[0]
        stats['total_chars'] = result if result else 0
        
        cursor.execute("""
            SELECT period, COUNT(*), SUM(word_count) 
            FROM collected_texts 
            GROUP BY period
        """)
        stats['by_period'] = {
            row[0]: {'texts': row[1], 'words': row[2] or 0}
            for row in cursor.fetchall()
        }
        
        cursor.execute("""
            SELECT language, COUNT(*), SUM(word_count)
            FROM collected_texts
            GROUP BY language
        """)
        stats['by_language'] = {
            row[0]: {'texts': row[1], 'words': row[2] or 0}
            for row in cursor.fetchall()
        }
        
        conn.close()
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data"
    
    collector = PROIELTextCollector(data_dir)
    stats = collector.collect_all()
    
    print("\n" + "=" * 60)
    print("COLLECTION STATISTICS")
    print("=" * 60)
    
    final_stats = collector.get_statistics()
    print(f"Total texts: {final_stats['total_texts']}")
    print(f"Total words: {final_stats['total_words']:,}")
    print(f"Total characters: {final_stats['total_chars']:,}")
    
    print("\nBy Period:")
    for period, data in final_stats['by_period'].items():
        print(f"  {period}: {data['texts']} texts, {data['words']:,} words")
    
    print("\nBy Language:")
    for lang, data in final_stats['by_language'].items():
        print(f"  {lang}: {data['texts']} texts, {data['words']:,} words")
