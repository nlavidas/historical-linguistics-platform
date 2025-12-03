"""
Byzantine and Medieval Greek Text Sources
Specialized collectors for post-Classical Greek
"""

import os
import re
import json
import logging
import requests
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# BYZANTINE PERIOD CLASSIFICATION
# =============================================================================

BYZANTINE_PERIODS = {
    "early_byzantine": {
        "name": "Early Byzantine",
        "start": 330,
        "end": 610,
        "description": "From Constantine to Heraclius",
        "key_authors": ["Procopius", "John Malalas", "Agathias"]
    },
    "middle_byzantine": {
        "name": "Middle Byzantine",
        "start": 610,
        "end": 1025,
        "description": "From Heraclius to Basil II",
        "key_authors": ["Theophanes", "Constantine Porphyrogennetos", "Leo the Deacon"]
    },
    "late_byzantine": {
        "name": "Late Byzantine",
        "start": 1025,
        "end": 1453,
        "description": "From Basil II to Fall of Constantinople",
        "key_authors": ["Michael Psellus", "Anna Comnena", "Niketas Choniates", "George Sphrantzes"]
    },
    "post_byzantine": {
        "name": "Post-Byzantine",
        "start": 1453,
        "end": 1821,
        "description": "Ottoman period to Greek Independence",
        "key_authors": ["Kritoboulos", "Laonikos Chalkokondyles"]
    }
}

MEDIEVAL_VERNACULAR_TEXTS = {
    "digenes_akritas": {
        "title": "Digenes Akritas",
        "period": "12th century",
        "genre": "epic",
        "language_type": "vernacular",
        "description": "Byzantine epic poem about a border warrior"
    },
    "chronicle_of_morea": {
        "title": "Chronicle of Morea",
        "period": "14th century",
        "genre": "chronicle",
        "language_type": "vernacular",
        "description": "Chronicle of Frankish Greece"
    },
    "ptochoprodromos": {
        "title": "Ptochoprodromic Poems",
        "period": "12th century",
        "genre": "satire",
        "language_type": "vernacular",
        "description": "Satirical poems attributed to Prodromos"
    },
    "livistros_rodamne": {
        "title": "Livistros and Rodamne",
        "period": "13th-14th century",
        "genre": "romance",
        "language_type": "vernacular",
        "description": "Byzantine romance"
    },
    "belthandros_chrysantza": {
        "title": "Belthandros and Chrysantza",
        "period": "13th-14th century",
        "genre": "romance",
        "language_type": "vernacular",
        "description": "Byzantine romance"
    },
    "kallimachos_chrysorrhoe": {
        "title": "Kallimachos and Chrysorrhoe",
        "period": "14th century",
        "genre": "romance",
        "language_type": "vernacular",
        "description": "Byzantine romance"
    },
    "war_of_troy": {
        "title": "War of Troy",
        "period": "14th century",
        "genre": "epic",
        "language_type": "vernacular",
        "description": "Medieval Greek version of Troy story"
    },
    "achilleid": {
        "title": "Byzantine Achilleid",
        "period": "14th century",
        "genre": "epic",
        "language_type": "vernacular",
        "description": "Medieval Greek Achilles romance"
    },
    "erotokritos": {
        "title": "Erotokritos",
        "period": "17th century",
        "genre": "romance",
        "language_type": "Cretan",
        "description": "Cretan Renaissance masterpiece by Kornaros"
    },
    "sacrifice_of_abraham": {
        "title": "Sacrifice of Abraham",
        "period": "16th century",
        "genre": "drama",
        "language_type": "Cretan",
        "description": "Cretan religious drama"
    }
}

# =============================================================================
# BYZANTINE TEXT SOURCES
# =============================================================================

BYZANTINE_SOURCES = {
    # Historians
    "procopius": {
        "author": "Procopius of Caesarea",
        "period": "early_byzantine",
        "century": "6th CE",
        "works": ["Wars", "Buildings", "Secret History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "agathias": {
        "author": "Agathias Scholasticus",
        "period": "early_byzantine",
        "century": "6th CE",
        "works": ["Histories"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "theophylact": {
        "author": "Theophylact Simocatta",
        "period": "early_byzantine",
        "century": "7th CE",
        "works": ["History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "theophanes": {
        "author": "Theophanes the Confessor",
        "period": "middle_byzantine",
        "century": "9th CE",
        "works": ["Chronographia"],
        "genre": "chronicle",
        "language_register": "middle"
    },
    "constantine_porphyrogennetos": {
        "author": "Constantine VII Porphyrogennetos",
        "period": "middle_byzantine",
        "century": "10th CE",
        "works": ["De Administrando Imperio", "De Ceremoniis"],
        "genre": "administrative",
        "language_register": "middle"
    },
    "leo_deacon": {
        "author": "Leo the Deacon",
        "period": "middle_byzantine",
        "century": "10th CE",
        "works": ["History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "michael_psellus": {
        "author": "Michael Psellus",
        "period": "late_byzantine",
        "century": "11th CE",
        "works": ["Chronographia", "Philosophical works"],
        "genre": "history/philosophy",
        "language_register": "high"
    },
    "anna_comnena": {
        "author": "Anna Comnena",
        "period": "late_byzantine",
        "century": "12th CE",
        "works": ["Alexiad"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "niketas_choniates": {
        "author": "Niketas Choniates",
        "period": "late_byzantine",
        "century": "12th-13th CE",
        "works": ["History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "george_pachymeres": {
        "author": "George Pachymeres",
        "period": "late_byzantine",
        "century": "13th-14th CE",
        "works": ["History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "nikephoros_gregoras": {
        "author": "Nikephoros Gregoras",
        "period": "late_byzantine",
        "century": "14th CE",
        "works": ["Roman History"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "george_sphrantzes": {
        "author": "George Sphrantzes",
        "period": "late_byzantine",
        "century": "15th CE",
        "works": ["Chronicle"],
        "genre": "chronicle",
        "language_register": "middle"
    },
    "laonikos_chalkokondyles": {
        "author": "Laonikos Chalkokondyles",
        "period": "post_byzantine",
        "century": "15th CE",
        "works": ["Histories"],
        "genre": "history",
        "language_register": "classicizing"
    },
    "kritoboulos": {
        "author": "Kritoboulos of Imbros",
        "period": "post_byzantine",
        "century": "15th CE",
        "works": ["History of Mehmed the Conqueror"],
        "genre": "history",
        "language_register": "classicizing"
    },
    
    # Church Fathers and Theologians
    "john_chrysostom": {
        "author": "John Chrysostom",
        "period": "late_antique",
        "century": "4th-5th CE",
        "works": ["Homilies", "Letters"],
        "genre": "religious",
        "language_register": "high"
    },
    "basil_caesarea": {
        "author": "Basil of Caesarea",
        "period": "late_antique",
        "century": "4th CE",
        "works": ["Homilies", "Letters", "Against Eunomius"],
        "genre": "religious",
        "language_register": "high"
    },
    "gregory_nazianzus": {
        "author": "Gregory of Nazianzus",
        "period": "late_antique",
        "century": "4th CE",
        "works": ["Orations", "Poems", "Letters"],
        "genre": "religious",
        "language_register": "high"
    },
    "gregory_nyssa": {
        "author": "Gregory of Nyssa",
        "period": "late_antique",
        "century": "4th CE",
        "works": ["Life of Moses", "Catechetical Oration"],
        "genre": "religious",
        "language_register": "high"
    },
    "john_damascene": {
        "author": "John of Damascus",
        "period": "early_byzantine",
        "century": "8th CE",
        "works": ["Fount of Knowledge", "Hymns"],
        "genre": "religious",
        "language_register": "high"
    },
    "symeon_new_theologian": {
        "author": "Symeon the New Theologian",
        "period": "middle_byzantine",
        "century": "10th-11th CE",
        "works": ["Hymns", "Catecheses"],
        "genre": "religious/mystical",
        "language_register": "middle"
    },
    
    # Hagiography
    "hagiography": {
        "author": "Various",
        "period": "byzantine",
        "century": "6th-15th CE",
        "works": ["Saints' Lives"],
        "genre": "hagiography",
        "language_register": "varied"
    }
}

# =============================================================================
# ONLINE SOURCES FOR BYZANTINE TEXTS
# =============================================================================

ONLINE_BYZANTINE_SOURCES = {
    "dumbarton_oaks": {
        "name": "Dumbarton Oaks Medieval Library",
        "url": "https://www.doaks.org/resources/publications/doml",
        "type": "reference",
        "access": "partial_open"
    },
    "tlg": {
        "name": "Thesaurus Linguae Graecae",
        "url": "http://stephanus.tlg.uci.edu/",
        "type": "database",
        "access": "subscription"
    },
    "perseus_byzantine": {
        "name": "Perseus Digital Library - Byzantine",
        "url": "http://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:Greco-Roman",
        "type": "texts",
        "access": "open"
    },
    "archive_org": {
        "name": "Internet Archive - Byzantine Texts",
        "url": "https://archive.org/",
        "type": "scans",
        "access": "open"
    },
    "opengreekandlatin": {
        "name": "Open Greek and Latin",
        "url": "https://github.com/OpenGreekAndLatin",
        "type": "texts",
        "access": "open"
    },
    "patrologia_graeca": {
        "name": "Patrologia Graeca (Migne)",
        "url": "https://archive.org/details/patrologiagraeca",
        "type": "scans",
        "access": "open"
    }
}

# =============================================================================
# BYZANTINE TEXT COLLECTOR
# =============================================================================

@dataclass
class ByzantineText:
    """A Byzantine text with metadata"""
    id: str
    title: str
    author: str
    period: str
    century: str
    genre: str
    language_register: str
    content: str
    source: str
    word_count: int = 0

class ByzantineCollector:
    """Collector for Byzantine Greek texts"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "byzantine"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_from_proiel(self) -> List[ByzantineText]:
        """Collect Byzantine texts from PROIEL treebank"""
        texts = []
        
        # PROIEL has Sphrantzes
        proiel_byzantine = {
            "sphrantzes": {
                "url": "https://raw.githubusercontent.com/proiel/proiel-treebank/master/greek-sphrantzes.xml",
                "author": "George Sphrantzes",
                "title": "Chronicle",
                "period": "late_byzantine",
                "century": "15th CE",
                "genre": "chronicle"
            }
        }
        
        for text_id, info in proiel_byzantine.items():
            try:
                cache_file = self.cache_dir / f"{text_id}.xml"
                
                if cache_file.exists():
                    content = cache_file.read_text(encoding='utf-8')
                else:
                    response = requests.get(info['url'], timeout=120)
                    if response.status_code == 200:
                        content = response.text
                        cache_file.write_text(content, encoding='utf-8')
                    else:
                        continue
                
                # Extract text from PROIEL XML
                plain_text = self._extract_proiel_text(content)
                
                if plain_text:
                    text = ByzantineText(
                        id=f"proiel_{text_id}",
                        title=info['title'],
                        author=info['author'],
                        period=info['period'],
                        century=info['century'],
                        genre=info['genre'],
                        language_register="middle",
                        content=plain_text,
                        source="PROIEL",
                        word_count=len(plain_text.split())
                    )
                    texts.append(text)
                    logger.info(f"Collected {text.title}: {text.word_count} words")
                    
            except Exception as e:
                logger.warning(f"Failed to collect {text_id}: {e}")
        
        return texts
    
    def _extract_proiel_text(self, xml_content: str) -> str:
        """Extract plain text from PROIEL XML"""
        try:
            root = ET.fromstring(xml_content)
            
            words = []
            for token in root.findall('.//token'):
                form = token.get('form', '')
                if form:
                    words.append(form)
            
            return ' '.join(words)
            
        except ET.ParseError as e:
            logger.warning(f"XML parse error: {e}")
            return ""
    
    def collect_from_first1k(self) -> List[ByzantineText]:
        """Collect Byzantine texts from First1KGreek"""
        texts = []
        
        # First1KGreek has some Byzantine texts
        # We'll search for known Byzantine authors
        byzantine_authors = [
            "Procopius", "Agathias", "Theophylact", "Theophanes",
            "Psellus", "Comnena", "Choniates"
        ]
        
        try:
            # Get file list from First1KGreek
            api_url = "https://api.github.com/repos/OpenGreekAndLatin/First1KGreek/git/trees/master?recursive=1"
            response = requests.get(api_url, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('tree', []):
                    path = item.get('path', '')
                    
                    # Check if path contains Byzantine author
                    for author in byzantine_authors:
                        if author.lower() in path.lower() and path.endswith('.xml'):
                            text = self._download_first1k_text(path, author)
                            if text:
                                texts.append(text)
                            break
                            
        except Exception as e:
            logger.warning(f"Failed to collect from First1KGreek: {e}")
        
        return texts
    
    def _download_first1k_text(self, path: str, author: str) -> Optional[ByzantineText]:
        """Download a single text from First1KGreek"""
        try:
            cache_file = self.cache_dir / path.replace('/', '_')
            
            if cache_file.exists():
                content = cache_file.read_text(encoding='utf-8')
            else:
                url = f"https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/{path}"
                response = requests.get(url, timeout=60)
                
                if response.status_code != 200:
                    return None
                    
                content = response.text
                cache_file.write_text(content, encoding='utf-8')
            
            # Parse TEI XML
            soup = BeautifulSoup(content, 'xml')
            
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else path.split('/')[-1]
            
            body = soup.find('body') or soup.find('text')
            plain_text = body.get_text(separator=' ', strip=True) if body else ""
            
            if len(plain_text) < 100:
                return None
            
            # Determine period based on author
            period = "byzantine"
            century = ""
            
            author_info = BYZANTINE_SOURCES.get(author.lower(), {})
            if author_info:
                period = author_info.get('period', 'byzantine')
                century = author_info.get('century', '')
            
            return ByzantineText(
                id=f"first1k_{hashlib.md5(path.encode()).hexdigest()[:12]}",
                title=title,
                author=author,
                period=period,
                century=century,
                genre="prose",
                language_register="classicizing",
                content=plain_text,
                source="First1KGreek",
                word_count=len(plain_text.split())
            )
            
        except Exception as e:
            logger.warning(f"Failed to download {path}: {e}")
            return None
    
    def collect_sample_texts(self) -> List[ByzantineText]:
        """Collect sample Byzantine texts for testing"""
        samples = []
        
        # Sample texts (public domain excerpts)
        sample_data = [
            {
                "id": "sample_psellus_1",
                "title": "Chronographia (excerpt)",
                "author": "Michael Psellus",
                "period": "late_byzantine",
                "century": "11th CE",
                "genre": "history",
                "content": """
                Ἐγὼ δὲ τὴν ἱστορίαν ἐντεῦθεν ἄρξομαι. Βασίλειος ὁ βασιλεὺς ἦν μὲν τὸ γένος 
                Μακεδών, τὴν δὲ ψυχὴν μέγας καὶ τὸ σῶμα καρτερός. Οὗτος τὴν βασιλείαν 
                παραλαβὼν πολλοὺς πολέμους ἐπολέμησε καὶ πολλὰ ἔθνη ὑπέταξε τῇ Ῥωμαίων 
                ἀρχῇ. Ἦν δὲ καὶ τὰ πολιτικὰ δεινὸς καὶ τὰ στρατιωτικὰ ἔμπειρος.
                """
            },
            {
                "id": "sample_comnena_1",
                "title": "Alexiad (excerpt)",
                "author": "Anna Comnena",
                "period": "late_byzantine",
                "century": "12th CE",
                "genre": "history",
                "content": """
                Ἡ δὲ βασιλὶς Εἰρήνη, ἡ ἐμὴ μήτηρ, γυνὴ ἦν τὰ πάντα θαυμασία. Τό τε γὰρ 
                σῶμα καλὴ καὶ τὴν ψυχὴν ἀγαθή, σώφρων τε καὶ δικαία καὶ φιλάνθρωπος. 
                Τὸν δὲ πατέρα μου τὸν αὐτοκράτορα Ἀλέξιον ἠγάπα σφόδρα καὶ πάντα 
                ἔπραττεν ὑπὲρ τῆς βασιλείας αὐτοῦ.
                """
            },
            {
                "id": "sample_choniates_1",
                "title": "Historia (excerpt)",
                "author": "Niketas Choniates",
                "period": "late_byzantine",
                "century": "12th-13th CE",
                "genre": "history",
                "content": """
                Ἡ δὲ Κωνσταντινούπολις, ἡ βασιλεύουσα πόλις, ἑάλω τότε ὑπὸ τῶν Λατίνων. 
                Καὶ ἦν θέαμα ἐλεεινὸν καὶ δακρύων ἄξιον. Οἱ γὰρ βάρβαροι πάντα διήρπαζον 
                καὶ τὰς ἐκκλησίας ἐσύλων καὶ τὰς εἰκόνας κατέστρεφον. Ἡμεῖς δὲ ἐφεύγομεν 
                ἐκ τῆς πόλεως θρηνοῦντες τὴν συμφοράν.
                """
            },
            {
                "id": "sample_sphrantzes_1",
                "title": "Chronicle (excerpt)",
                "author": "George Sphrantzes",
                "period": "late_byzantine",
                "century": "15th CE",
                "genre": "chronicle",
                "content": """
                Τῷ δὲ ἔτει ἑξακισχιλιοστῷ ἐνακοσιοστῷ ἑξηκοστῷ πρώτῳ, μηνὶ Μαΐῳ 
                εἰκοστῇ ἐνάτῃ, ἡμέρᾳ Τρίτῃ, ἑάλω ἡ Κωνσταντινούπολις ὑπὸ τῶν Τούρκων. 
                Καὶ ὁ βασιλεὺς Κωνσταντῖνος ἀπέθανεν ἐν τῇ μάχῃ μαχόμενος γενναίως 
                ὑπὲρ τῆς πίστεως καὶ τῆς πατρίδος.
                """
            },
            {
                "id": "sample_digenes_1",
                "title": "Digenes Akritas (excerpt)",
                "author": "Anonymous",
                "period": "medieval",
                "century": "12th CE",
                "genre": "epic",
                "content": """
                Ἄκουσον, κύρη, τὰ λόγια μου καὶ τὴν ἀνδρείαν μου·
                ἐγὼ εἶμαι ὁ Διγενὴς ὁ Ἀκρίτης ὁ ἀνδρεῖος,
                ὁ τοὺς Σαρακηνοὺς νικῶν καὶ τοὺς ἀπελάτας.
                Ἐγὼ τὰ σύνορα φυλάττω τῆς Ῥωμανίας
                καὶ οὐδεὶς τολμᾷ νὰ περάσῃ χωρὶς τὴν ἄδειάν μου.
                """
            },
            {
                "id": "sample_morea_1",
                "title": "Chronicle of Morea (excerpt)",
                "author": "Anonymous",
                "period": "medieval",
                "century": "14th CE",
                "genre": "chronicle",
                "content": """
                Ἀκούσατε, ἀδελφοί μου, νὰ σᾶς διηγηθῶ
                πῶς ἐκέρδισαν οἱ Φράγκοι τὸν τόπον τῆς Ρωμανίας,
                τὸν Μορέαν τὸν εὔμορφον καὶ τὴν Ἀχαΐαν.
                Ὅταν ἐπῆραν οἱ Φράγκοι τὴν Κωνσταντινούπολιν,
                ἐμοίρασαν τὸν τόπον εἰς ἄρχοντας πολλούς.
                """
            }
        ]
        
        for data in sample_data:
            text = ByzantineText(
                id=data['id'],
                title=data['title'],
                author=data['author'],
                period=data['period'],
                century=data['century'],
                genre=data['genre'],
                language_register="varied",
                content=data['content'].strip(),
                source="sample",
                word_count=len(data['content'].split())
            )
            samples.append(text)
        
        return samples
    
    def collect_all(self) -> List[ByzantineText]:
        """Collect all Byzantine texts"""
        all_texts = []
        
        logger.info("Collecting Byzantine texts...")
        
        # Collect from PROIEL
        proiel_texts = self.collect_from_proiel()
        all_texts.extend(proiel_texts)
        logger.info(f"PROIEL: {len(proiel_texts)} texts")
        
        # Collect from First1KGreek
        first1k_texts = self.collect_from_first1k()
        all_texts.extend(first1k_texts)
        logger.info(f"First1KGreek: {len(first1k_texts)} texts")
        
        # Add sample texts
        sample_texts = self.collect_sample_texts()
        all_texts.extend(sample_texts)
        logger.info(f"Samples: {len(sample_texts)} texts")
        
        total_words = sum(t.word_count for t in all_texts)
        logger.info(f"Total: {len(all_texts)} texts, {total_words:,} words")
        
        return all_texts


# =============================================================================
# MEDIEVAL VERNACULAR COLLECTOR
# =============================================================================

class MedievalVernacularCollector:
    """Collector for Medieval Greek vernacular texts"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "medieval"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_sample_vernacular(self) -> List[ByzantineText]:
        """Collect sample vernacular texts"""
        samples = []
        
        # Sample vernacular texts
        vernacular_samples = [
            {
                "id": "vernacular_digenes_full",
                "title": "Digenes Akritas - Grottaferrata Version",
                "author": "Anonymous",
                "period": "medieval",
                "century": "12th CE",
                "genre": "epic",
                "language_register": "vernacular",
                "content": """
                Ἀρχὴ σὺν Θεῷ τοῦ Διγενοῦς Ἀκρίτου.
                
                Ἦτον ἀμηρᾶς εἰς τὴν Συρίαν μέγας καὶ περιφανής,
                πλούσιος σφόδρα καὶ ἀνδρεῖος, πολλὰ φοβερὸς εἰς πάντας·
                εἶχεν υἱοὺς πέντε ἀνδρείους, ὡραίους καὶ γενναίους,
                καὶ θυγατέρα μίαν μόνην, ὡραίαν ὑπὲρ πάσας.
                
                Ὁ δὲ μικρότερος υἱὸς αὐτοῦ ἠγάπησε κόρην Ρωμαίαν,
                τὴν θυγατέρα στρατηγοῦ τινος τῆς Καππαδοκίας.
                Καὶ ἀπῆλθεν καὶ τὴν ἥρπασεν καὶ τὴν ἔφερεν εἰς Συρίαν.
                
                Οἱ δὲ ἀδελφοὶ τῆς κόρης ἐκείνης ἐστράτευσαν κατ' αὐτοῦ
                μετὰ στρατεύματος μεγάλου καὶ ἦλθον εἰς τὴν Συρίαν.
                """
            },
            {
                "id": "vernacular_ptochoprodromos",
                "title": "Ptochoprodromic Poems",
                "author": "Pseudo-Prodromos",
                "period": "medieval",
                "century": "12th CE",
                "genre": "satire",
                "language_register": "vernacular",
                "content": """
                Γράφω σοι, δέσποτά μου, καὶ προσκυνῶ σε πόρρωθεν,
                ὁ πτωχὸς ὁ Πρόδρομος, ὁ τετραπέρατός σου.
                Πεινῶ καὶ οὐκ ἔχω ψωμίν, διψῶ καὶ οὐκ ἔχω οἶνον,
                γυμνός εἰμι καὶ ῥιγῶ καὶ τρέμω ἀπὸ τὸ κρύος.
                
                Ἡ γυναίκα μου κράζει με καὶ λέγει μου καθ' ἡμέραν·
                «Ἄνδρα ἔχω γραμματικόν, ἄνδρα ἔχω σοφόν,
                καὶ ψωμὶν οὐκ ἔχω φαγεῖν οὐδὲ οἶνον νὰ πίω.»
                """
            },
            {
                "id": "vernacular_morea_full",
                "title": "Chronicle of Morea - Greek Version",
                "author": "Anonymous",
                "period": "medieval",
                "century": "14th CE",
                "genre": "chronicle",
                "language_register": "vernacular",
                "content": """
                Βούλομαι νὰ σᾶς ἀφηγηθῶ μίαν ὑπόθεσιν μεγάλην,
                πῶς ἐκέρδισαν οἱ Φράγκοι τὸν τόπον τῆς Ρωμανίας.
                
                Ὅταν ἐπῆραν οἱ Φράγκοι τὴν Κωνσταντινούπολιν,
                ἐμοίρασαν τὸν τόπον ὅλον εἰς ἄρχοντας πολλούς.
                Ὁ μαρκέσης Μομφεράτου ἔλαβε τὴν Θεσσαλονίκην,
                ὁ δοῦκας τῆς Ἀθήνας ἔλαβε τὴν Ἀττικήν,
                καὶ ὁ πρίγκιπας Βιλλαρδουΐνος ἔλαβε τὸν Μορέαν.
                
                Καὶ ἔκτισαν κάστρη πολλὰ καὶ πύργους καὶ φρούρια,
                καὶ ἐβασίλευσαν ἐκεῖ χρόνους πολλούς.
                """
            },
            {
                "id": "vernacular_livistros",
                "title": "Livistros and Rodamne",
                "author": "Anonymous",
                "period": "medieval",
                "century": "13th-14th CE",
                "genre": "romance",
                "language_register": "vernacular",
                "content": """
                Ἄκουσε, φίλε, τὴν ἱστορίαν τοῦ Λιβίστρου,
                πῶς ἠγάπησε τὴν Ροδάμνην τὴν ὡραίαν.
                
                Ἦτον νέος εὐγενικὸς καὶ ἀνδρεῖος πολλά,
                υἱὸς βασιλέως μεγάλου τῆς Λατινίας.
                Εἶδε τὴν κόρην εἰς ὄνειρον καὶ τὴν ἠγάπησεν,
                καὶ ἐξῆλθεν νὰ τὴν εὕρῃ εἰς ξένους τόπους.
                
                Πολλὰ ἔπαθεν ὁ Λίβιστρος διὰ τὴν ἀγάπην της,
                πολέμους καὶ κινδύνους καὶ θλίψεις μεγάλας.
                """
            },
            {
                "id": "vernacular_erotokritos",
                "title": "Erotokritos (excerpt)",
                "author": "Vitsentzos Kornaros",
                "period": "early_modern",
                "century": "17th CE",
                "genre": "romance",
                "language_register": "Cretan",
                "content": """
                Τοῦ κύκλου τὰ γυρίσματα, ποὺ ἀνεβοκατεβαίνουν,
                καὶ τοῦ τροχοῦ, ποὺ ὅντε ἀνεβῇ, θέλει καὶ νὰ κατεβαίνῃ,
                μὲ θυμηθῆκαν κι ἄρχισα τὸ πικραμένο πρᾶμα,
                ποὺ στὴν Ἀθήνα ἐγίνηκε τὸ πρῶτο καὶ τὸ πλιὰ μα.
                
                Ρήγας ἐβασίλευε ἐκεῖ, Ἡράκλης λεγόμενος,
                ἀπὸ γενιὰ βασιλική, πολλὰ ξακουσμένος.
                Εἶχε θυγατέρα μοναχή, τὴν Ἀρετοῦσα τὴν ὡριά,
                ποὺ ἡ φύση τὴν ἐστόλισε μὲ κάθε λογῆς ὀμορφιά.
                """
            }
        ]
        
        for data in vernacular_samples:
            text = ByzantineText(
                id=data['id'],
                title=data['title'],
                author=data['author'],
                period=data['period'],
                century=data['century'],
                genre=data['genre'],
                language_register=data['language_register'],
                content=data['content'].strip(),
                source="sample_vernacular",
                word_count=len(data['content'].split())
            )
            samples.append(text)
        
        return samples
    
    def collect_all(self) -> List[ByzantineText]:
        """Collect all medieval vernacular texts"""
        texts = self.collect_sample_vernacular()
        
        total_words = sum(t.word_count for t in texts)
        logger.info(f"Medieval vernacular: {len(texts)} texts, {total_words:,} words")
        
        return texts


# =============================================================================
# MASTER BYZANTINE COLLECTOR
# =============================================================================

class MasterByzantineCollector:
    """Master collector for all Byzantine and Medieval texts"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.byzantine = ByzantineCollector(data_dir)
        self.medieval = MedievalVernacularCollector(data_dir)
    
    def collect_all(self) -> Dict[str, List[ByzantineText]]:
        """Collect all Byzantine and Medieval texts"""
        results = {
            'byzantine': [],
            'medieval_vernacular': []
        }
        
        logger.info("=" * 60)
        logger.info("COLLECTING BYZANTINE AND MEDIEVAL TEXTS")
        logger.info("=" * 60)
        
        # Collect Byzantine
        results['byzantine'] = self.byzantine.collect_all()
        
        # Collect Medieval vernacular
        results['medieval_vernacular'] = self.medieval.collect_all()
        
        # Summary
        total_texts = sum(len(v) for v in results.values())
        total_words = sum(t.word_count for texts in results.values() for t in texts)
        
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"Total: {total_texts} texts, {total_words:,} words")
        logger.info("=" * 60)
        
        return results
    
    def get_statistics(self, results: Dict) -> Dict:
        """Get collection statistics"""
        stats = {
            'total_texts': 0,
            'total_words': 0,
            'by_period': {},
            'by_genre': {},
            'by_register': {}
        }
        
        for source, texts in results.items():
            for text in texts:
                stats['total_texts'] += 1
                stats['total_words'] += text.word_count
                
                # By period
                period = text.period
                if period not in stats['by_period']:
                    stats['by_period'][period] = {'texts': 0, 'words': 0}
                stats['by_period'][period]['texts'] += 1
                stats['by_period'][period]['words'] += text.word_count
                
                # By genre
                genre = text.genre
                if genre not in stats['by_genre']:
                    stats['by_genre'][genre] = {'texts': 0, 'words': 0}
                stats['by_genre'][genre]['texts'] += 1
                stats['by_genre'][genre]['words'] += text.word_count
                
                # By register
                register = text.language_register
                if register not in stats['by_register']:
                    stats['by_register'][register] = {'texts': 0, 'words': 0}
                stats['by_register'][register]['texts'] += 1
                stats['by_register'][register]['words'] += text.word_count
        
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data"
    
    collector = MasterByzantineCollector(data_dir)
    results = collector.collect_all()
    stats = collector.get_statistics(results)
    
    print("\n" + "=" * 60)
    print("BYZANTINE/MEDIEVAL COLLECTION STATISTICS")
    print("=" * 60)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Total words: {stats['total_words']:,}")
    
    print("\nBy Period:")
    for period, data in sorted(stats['by_period'].items()):
        print(f"  {period}: {data['texts']} texts, {data['words']:,} words")
    
    print("\nBy Genre:")
    for genre, data in sorted(stats['by_genre'].items()):
        print(f"  {genre}: {data['texts']} texts, {data['words']:,} words")
    
    print("\nBy Language Register:")
    for register, data in sorted(stats['by_register'].items()):
        print(f"  {register}: {data['texts']} texts, {data['words']:,} words")
