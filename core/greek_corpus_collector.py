"""
Professional Greek Corpus Collector - Full Diachronic Coverage
Archaic, Classical, Hellenistic, Koine, Byzantine, Medieval, Early Modern Greek
With OCR support and UD/Penn-style annotation
Focus on PROIEL, Perseus, First1KGreek, and open-source Byzantine/Medieval sources
"""

import os
import re
import json
import sqlite3
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Token:
    """Token with UD and Penn-style annotation"""
    id: int
    form: str
    lemma: str
    pos_ud: str
    pos_penn: str
    morph: str
    head: int
    deprel: str
    misc: str = ""

@dataclass
class TextMetadata:
    """Metadata for a Greek text"""
    id: str
    title: str
    author: str
    period: str
    sub_period: str
    century: str
    genre: str
    source: str
    language: str
    script: str = "polytonic"

# =============================================================================
# GREEK PERIODS - DETAILED DIACHRONIC CLASSIFICATION
# =============================================================================

GREEK_PERIODS = {
    # Ancient Greek
    "archaic": {
        "start": -800, "end": -500, 
        "label": "Archaic Greek",
        "sub_periods": ["early_archaic", "late_archaic"]
    },
    "classical": {
        "start": -500, "end": -323,
        "label": "Classical Greek", 
        "sub_periods": ["early_classical", "high_classical", "late_classical"]
    },
    "hellenistic": {
        "start": -323, "end": -31,
        "label": "Hellenistic Greek",
        "sub_periods": ["early_hellenistic", "late_hellenistic"]
    },
    "koine": {
        "start": -31, "end": 330,
        "label": "Koine Greek",
        "sub_periods": ["roman_koine", "late_koine"]
    },
    # Byzantine Greek
    "early_byzantine": {
        "start": 330, "end": 610,
        "label": "Early Byzantine Greek",
        "sub_periods": ["constantinian", "justinianic"]
    },
    "middle_byzantine": {
        "start": 610, "end": 1081,
        "label": "Middle Byzantine Greek",
        "sub_periods": ["dark_ages", "macedonian_renaissance"]
    },
    "late_byzantine": {
        "start": 1081, "end": 1453,
        "label": "Late Byzantine Greek",
        "sub_periods": ["komnenian", "palaiologan"]
    },
    # Medieval Greek (Post-Byzantine)
    "early_medieval": {
        "start": 1453, "end": 1600,
        "label": "Early Post-Byzantine Greek",
        "sub_periods": ["ottoman_conquest", "venetian_territories"]
    },
    "middle_medieval": {
        "start": 1600, "end": 1750,
        "label": "Middle Post-Byzantine Greek",
        "sub_periods": ["ottoman_period", "cretan_renaissance"]
    },
    "late_medieval": {
        "start": 1750, "end": 1830,
        "label": "Late Post-Byzantine / Pre-Modern Greek",
        "sub_periods": ["enlightenment", "pre_independence"]
    },
    # Modern Greek
    "early_modern": {
        "start": 1830, "end": 1900,
        "label": "Early Modern Greek",
        "sub_periods": ["katharevousa_period", "demotic_emergence"]
    },
    "modern": {
        "start": 1900, "end": 2025,
        "label": "Modern Greek",
        "sub_periods": ["20th_century", "contemporary"]
    }
}

# =============================================================================
# POS TAG MAPPINGS (UD to Penn)
# =============================================================================

UD_TO_PENN = {
    "NOUN": "NN", "PROPN": "NNP", "VERB": "VB", "AUX": "VB",
    "ADJ": "JJ", "ADV": "RB", "PRON": "PRP", "DET": "DT",
    "ADP": "IN", "CONJ": "CC", "CCONJ": "CC", "SCONJ": "IN",
    "NUM": "CD", "PART": "RP", "INTJ": "UH", "PUNCT": ".",
    "SYM": "SYM", "X": "XX", "AUX": "MD"
}

# =============================================================================
# CORPUS SOURCES - COMPREHENSIVE
# =============================================================================

CORPUS_SOURCES = {
    # ========== ANCIENT GREEK ==========
    "ud_greek_proiel": {
        "name": "UD Ancient Greek PROIEL",
        "description": "New Testament, Herodotus, Sphrantzes Chronicle",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master",
        "files": ["grc_proiel-ud-train.conllu", "grc_proiel-ud-dev.conllu", "grc_proiel-ud-test.conllu"],
        "period": "mixed", "language": "grc", "annotated": True
    },
    "ud_greek_perseus": {
        "name": "UD Ancient Greek Perseus",
        "description": "Classical authors: Homer, Sophocles, Plato, etc.",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master",
        "files": ["grc_perseus-ud-train.conllu", "grc_perseus-ud-dev.conllu", "grc_perseus-ud-test.conllu"],
        "period": "classical", "language": "grc", "annotated": True
    },
    
    # ========== BYZANTINE GREEK ==========
    "proiel_sphrantzes": {
        "name": "PROIEL Sphrantzes Chronicle",
        "description": "George Sphrantzes - Late Byzantine chronicle (15th c.)",
        "raw_url": "https://raw.githubusercontent.com/proiel/proiel-treebank/master",
        "files": ["sphrantzes.xml"],
        "period": "late_byzantine", "language": "grc", "annotated": True,
        "format": "proiel-xml"
    },
    
    # ========== MEDIEVAL GREEK SOURCES ==========
    "opengreeklatin_first1k": {
        "name": "First 1000 Years of Greek",
        "description": "Open Greek and Latin - extensive Greek corpus",
        "base_url": "https://github.com/OpenGreekAndLatin/First1KGreek",
        "raw_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master",
        "period": "mixed", "language": "grc", "annotated": False,
        "format": "tei-xml"
    },
    
    # ========== COMPARISON CORPORA ==========
    "ud_latin_proiel": {
        "name": "UD Latin PROIEL",
        "description": "Latin comparison corpus",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master",
        "files": ["la_proiel-ud-train.conllu", "la_proiel-ud-dev.conllu", "la_proiel-ud-test.conllu"],
        "period": "classical", "language": "la", "annotated": True
    },
    "ud_latin_llct": {
        "name": "UD Latin LLCT",
        "description": "Late Latin Charter Treebank - Medieval Latin",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/master",
        "files": ["la_llct-ud-train.conllu", "la_llct-ud-dev.conllu", "la_llct-ud-test.conllu"],
        "period": "medieval", "language": "la", "annotated": True
    },
    "ud_spanish": {
        "name": "UD Spanish AnCora",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master",
        "files": ["es_ancora-ud-train.conllu", "es_ancora-ud-dev.conllu", "es_ancora-ud-test.conllu"],
        "period": "modern", "language": "es", "annotated": True
    },
    "ud_french": {
        "name": "UD French GSD",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master",
        "files": ["fr_gsd-ud-train.conllu", "fr_gsd-ud-dev.conllu", "fr_gsd-ud-test.conllu"],
        "period": "modern", "language": "fr", "annotated": True
    },
    "ud_italian": {
        "name": "UD Italian ISDT",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master",
        "files": ["it_isdt-ud-train.conllu", "it_isdt-ud-dev.conllu", "it_isdt-ud-test.conllu"],
        "period": "modern", "language": "it", "annotated": True
    },
    "ud_portuguese": {
        "name": "UD Portuguese Bosque",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master",
        "files": ["pt_bosque-ud-train.conllu", "pt_bosque-ud-dev.conllu", "pt_bosque-ud-test.conllu"],
        "period": "modern", "language": "pt", "annotated": True
    },
    "ud_romanian": {
        "name": "UD Romanian RRT",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master",
        "files": ["ro_rrt-ud-train.conllu", "ro_rrt-ud-dev.conllu", "ro_rrt-ud-test.conllu"],
        "period": "modern", "language": "ro", "annotated": True
    },
    "ud_english": {
        "name": "UD English EWT",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master",
        "files": ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"],
        "period": "modern", "language": "en", "annotated": True
    },
    "ud_old_french": {
        "name": "UD Old French SRCMF",
        "description": "Medieval French for comparison",
        "raw_url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/master",
        "files": ["fro_srcmf-ud-train.conllu", "fro_srcmf-ud-dev.conllu", "fro_srcmf-ud-test.conllu"],
        "period": "medieval", "language": "fro", "annotated": True
    }
}

# =============================================================================
# BYZANTINE AND MEDIEVAL GREEK TEXTS CATALOG
# =============================================================================

BYZANTINE_MEDIEVAL_TEXTS = {
    # Early Byzantine (330-610)
    "john_chrysostom": {
        "author": "John Chrysostom",
        "title": "Homilies",
        "period": "early_byzantine",
        "century": "4th-5th",
        "genre": "religious",
        "tlg_id": "2062"
    },
    "procopius": {
        "author": "Procopius of Caesarea",
        "title": "History of the Wars",
        "period": "early_byzantine",
        "century": "6th",
        "genre": "historiography"
    },
    
    # Middle Byzantine (610-1081)
    "theophanes": {
        "author": "Theophanes the Confessor",
        "title": "Chronographia",
        "period": "middle_byzantine",
        "century": "9th",
        "genre": "chronicle"
    },
    "constantine_porphyrogennetos": {
        "author": "Constantine VII Porphyrogennetos",
        "title": "De Administrando Imperio",
        "period": "middle_byzantine",
        "century": "10th",
        "genre": "political treatise"
    },
    "digenis_akritas": {
        "author": "Anonymous",
        "title": "Digenis Akritas",
        "period": "middle_byzantine",
        "century": "10th-12th",
        "genre": "epic poetry"
    },
    
    # Late Byzantine (1081-1453)
    "anna_komnene": {
        "author": "Anna Komnene",
        "title": "Alexiad",
        "period": "late_byzantine",
        "century": "12th",
        "genre": "historiography"
    },
    "ptochoprodromos": {
        "author": "Ptochoprodromos",
        "title": "Ptochoprodromic Poems",
        "period": "late_byzantine",
        "century": "12th",
        "genre": "vernacular poetry"
    },
    "chronicle_morea": {
        "author": "Anonymous",
        "title": "Chronicle of Morea",
        "period": "late_byzantine",
        "century": "14th",
        "genre": "chronicle"
    },
    "sphrantzes": {
        "author": "George Sphrantzes",
        "title": "Chronicon Minus",
        "period": "late_byzantine",
        "century": "15th",
        "genre": "chronicle"
    },
    
    # Early Post-Byzantine / Medieval (1453-1600)
    "erotokritos": {
        "author": "Vitsentzos Kornaros",
        "title": "Erotokritos",
        "period": "early_medieval",
        "century": "17th",
        "genre": "romance poetry"
    },
    "sacrifice_abraham": {
        "author": "Vitsentzos Kornaros",
        "title": "The Sacrifice of Abraham",
        "period": "early_medieval",
        "century": "17th",
        "genre": "religious drama"
    },
    
    # Middle Post-Byzantine (1600-1750)
    "cretan_theater": {
        "author": "Various",
        "title": "Cretan Renaissance Theater",
        "period": "middle_medieval",
        "century": "16th-17th",
        "genre": "drama"
    },
    
    # Late Post-Byzantine / Pre-Modern (1750-1830)
    "rigas_feraios": {
        "author": "Rigas Feraios",
        "title": "Revolutionary Writings",
        "period": "late_medieval",
        "century": "18th",
        "genre": "political"
    },
    "adamantios_korais": {
        "author": "Adamantios Korais",
        "title": "Philological Works",
        "period": "late_medieval",
        "century": "18th-19th",
        "genre": "philology"
    }
}

# =============================================================================
# TEXT CLEANING FUNCTIONS
# =============================================================================

def remove_emojis(text: str) -> str:
    """Remove all emojis and special Unicode symbols"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def normalize_greek(text: str) -> str:
    """Normalize Greek text - preserve polytonic accents"""
    return unicodedata.normalize('NFC', text)

def clean_text(text: str) -> str:
    """Full text cleaning pipeline"""
    if not text:
        return ""
    text = remove_emojis(text)
    text = normalize_greek(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =============================================================================
# CONLL-U PARSER
# =============================================================================

class CoNLLUParser:
    """Parse CoNLL-U format files (Universal Dependencies standard)"""
    
    @staticmethod
    def parse_file(filepath: str) -> List[List[Token]]:
        """Parse CoNLL-U file into sentences of tokens"""
        sentences = []
        current_sentence = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                    
                if line.startswith('#'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 10:
                    try:
                        token_id = parts[0]
                        if '-' in token_id or '.' in token_id:
                            continue
                            
                        token = Token(
                            id=int(token_id),
                            form=clean_text(parts[1]),
                            lemma=clean_text(parts[2]),
                            pos_ud=parts[3],
                            pos_penn=UD_TO_PENN.get(parts[3], "XX"),
                            morph=parts[5],
                            head=int(parts[6]) if parts[6] != '_' else 0,
                            deprel=parts[7],
                            misc=parts[9] if len(parts) > 9 else ""
                        )
                        current_sentence.append(token)
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping malformed line: {e}")
                        
        if current_sentence:
            sentences.append(current_sentence)
            
        return sentences
    
    @staticmethod
    def to_conllu(sentences: List[List[Token]]) -> str:
        """Convert sentences back to CoNLL-U format"""
        lines = []
        for sent_idx, sentence in enumerate(sentences):
            lines.append(f"# sent_id = {sent_idx + 1}")
            lines.append(f"# text = {' '.join(t.form for t in sentence)}")
            for token in sentence:
                line = f"{token.id}\t{token.form}\t{token.lemma}\t{token.pos_ud}\t_\t{token.morph}\t{token.head}\t{token.deprel}\t_\t{token.misc}"
                lines.append(line)
            lines.append("")
        return '\n'.join(lines)

# =============================================================================
# OCR ENGINE FOR GREEK MANUSCRIPTS
# =============================================================================

class GreekOCR:
    """OCR engine supporting multiple backends for Greek text"""
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        self.easyocr_available = self._check_easyocr()
        
    def _check_tesseract(self) -> bool:
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                   capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def _check_easyocr(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def ocr_image(self, image_path: str, method: str = 'auto') -> Optional[str]:
        """Perform OCR on an image file"""
        if method == 'auto':
            if self.tesseract_available:
                method = 'tesseract'
            elif self.easyocr_available:
                method = 'easyocr'
            else:
                logger.error("No OCR engine available")
                return None
                
        if method == 'tesseract':
            return self._tesseract_ocr(image_path)
        elif method == 'easyocr':
            return self._easyocr_ocr(image_path)
        return None
        
    def _tesseract_ocr(self, image_path: str) -> Optional[str]:
        """Use Tesseract for Greek OCR"""
        try:
            # Try polytonic Greek first, fall back to modern Greek
            result = subprocess.run(
                ['tesseract', image_path, 'stdout', '-l', 'grc+ell', '--psm', '6'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return clean_text(result.stdout)
            else:
                logger.warning(f"Tesseract error: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return None
            
    def _easyocr_ocr(self, image_path: str) -> Optional[str]:
        """Use EasyOCR for Greek OCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['el', 'en'], gpu=False)
            results = reader.readtext(image_path)
            text = ' '.join([r[1] for r in results])
            return clean_text(text)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return None
            
    def ocr_pdf(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """OCR a PDF file page by page"""
        try:
            from pdf2image import convert_from_path
            
            if output_dir is None:
                output_dir = Path(pdf_path).parent / "ocr_output"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            pages = convert_from_path(pdf_path, dpi=300)
            texts = []
            
            for i, page in enumerate(pages):
                img_path = Path(output_dir) / f"page_{i+1}.png"
                page.save(str(img_path), 'PNG')
                text = self.ocr_image(str(img_path))
                if text:
                    texts.append(text)
                logger.info(f"OCR completed for page {i+1}/{len(pages)}")
                
            return texts
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return []

# =============================================================================
# MAIN CORPUS COLLECTOR
# =============================================================================

class GreekCorpusCollector:
    """
    Professional Greek Corpus Collector
    Collects, preprocesses, and stores Greek texts from multiple sources
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Detect environment
            if os.path.exists("/root/corpus_platform"):
                data_dir = "/root/corpus_platform/data"
            else:
                data_dir = "data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "greek_corpus.db"
        self.ocr = GreekOCR()
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Texts table
        c.execute('''CREATE TABLE IF NOT EXISTS texts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            period TEXT,
            sub_period TEXT,
            century TEXT,
            genre TEXT,
            source TEXT,
            language TEXT DEFAULT 'grc',
            script TEXT DEFAULT 'polytonic',
            token_count INTEGER DEFAULT 0,
            sentence_count INTEGER DEFAULT 0,
            annotated BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Sentences table
        c.execute('''CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id TEXT NOT NULL,
            sentence_num INTEGER,
            raw_text TEXT,
            token_count INTEGER,
            FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
        )''')
        
        # Tokens table with full annotation
        c.execute('''CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id INTEGER NOT NULL,
            token_num INTEGER,
            form TEXT NOT NULL,
            lemma TEXT,
            pos_ud TEXT,
            pos_penn TEXT,
            morph TEXT,
            head INTEGER,
            deprel TEXT,
            misc TEXT,
            FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
        )''')
        
        # Indexes for efficient querying
        c.execute('CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_tokens_pos_ud ON tokens(pos_ud)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_tokens_form ON tokens(form)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_texts_period ON texts(period)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_texts_language ON texts(language)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_texts_author ON texts(author)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_sentences_text ON sentences(text_id)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
        
    def download_file(self, url: str, local_path: Path) -> bool:
        """Download file from URL with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading: {url}")
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                    
                logger.info(f"Saved: {local_path}")
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Download failed after {max_retries} attempts: {url}")
                    return False
        return False
        
    def store_sentences(self, text_id: str, title: str, author: str,
                       period: str, source: str, sentences: List[List[Token]],
                       language: str = "grc", sub_period: str = None,
                       century: str = None, genre: str = None,
                       annotated: bool = True):
        """Store sentences and tokens in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Calculate totals
        total_tokens = sum(len(s) for s in sentences)
        
        # Delete existing data for this text
        c.execute('SELECT id FROM sentences WHERE text_id = ?', (text_id,))
        sentence_ids = [row[0] for row in c.fetchall()]
        if sentence_ids:
            c.execute(f'DELETE FROM tokens WHERE sentence_id IN ({",".join("?" * len(sentence_ids))})', sentence_ids)
        c.execute('DELETE FROM sentences WHERE text_id = ?', (text_id,))
        c.execute('DELETE FROM texts WHERE id = ?', (text_id,))
        
        # Insert text record
        c.execute('''INSERT INTO texts 
                    (id, title, author, period, sub_period, century, genre, source, 
                     language, token_count, sentence_count, annotated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (text_id, title, author, period, sub_period, century, genre, source,
                  language, total_tokens, len(sentences), annotated))
        
        # Insert sentences and tokens
        for sent_num, sentence in enumerate(sentences):
            raw_text = ' '.join(t.form for t in sentence)
            c.execute('''INSERT INTO sentences (text_id, sentence_num, raw_text, token_count)
                        VALUES (?, ?, ?, ?)''',
                     (text_id, sent_num, raw_text, len(sentence)))
            sentence_id = c.lastrowid
            
            for token in sentence:
                c.execute('''INSERT INTO tokens 
                            (sentence_id, token_num, form, lemma, pos_ud, pos_penn, 
                             morph, head, deprel, misc)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (sentence_id, token.id, token.form, token.lemma, token.pos_ud,
                          token.pos_penn, token.morph, token.head, token.deprel, token.misc))
                          
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(sentences)} sentences, {total_tokens} tokens for {text_id}")
        
    def collect_ud_corpus(self, corpus_id: str, corpus_info: dict) -> int:
        """Collect a Universal Dependencies corpus"""
        logger.info(f"Collecting {corpus_info['name']}...")
        total_sentences = 0
        
        for filename in corpus_info.get("files", []):
            url = f"{corpus_info['raw_url']}/{filename}"
            local_path = self.data_dir / corpus_id / filename
            
            if self.download_file(url, local_path):
                sentences = CoNLLUParser.parse_file(str(local_path))
                total_sentences += len(sentences)
                
                text_id = f"{corpus_id}_{filename.replace('.conllu', '')}"
                self.store_sentences(
                    text_id=text_id,
                    title=corpus_info['name'],
                    author="Various",
                    period=corpus_info.get('period', 'unknown'),
                    source=corpus_id,
                    sentences=sentences,
                    language=corpus_info.get('language', 'grc'),
                    annotated=True
                )
                
        logger.info(f"Collected {total_sentences} sentences from {corpus_info['name']}")
        return total_sentences
        
    def collect_all_ud_corpora(self) -> Dict[str, int]:
        """Collect all configured UD corpora"""
        results = {}
        
        for corpus_id, corpus_info in CORPUS_SOURCES.items():
            if corpus_info.get('files') and corpus_info.get('raw_url'):
                try:
                    count = self.collect_ud_corpus(corpus_id, corpus_info)
                    results[corpus_id] = count
                except Exception as e:
                    logger.error(f"Failed to collect {corpus_id}: {e}")
                    results[corpus_id] = 0
                    
        return results
        
    def collect_all(self) -> Dict[str, int]:
        """Collect all corpora"""
        logger.info("=" * 60)
        logger.info("STARTING CORPUS COLLECTION")
        logger.info("=" * 60)
        
        results = self.collect_all_ud_corpora()
        
        # Summary
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        
        total = 0
        for corpus, count in sorted(results.items()):
            logger.info(f"  {corpus}: {count} sentences")
            total += count
            
        logger.info(f"  TOTAL: {total} sentences")
        
        return results
        
    def get_statistics(self) -> Dict:
        """Get comprehensive corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = {}
        
        # Total counts
        c.execute("SELECT COUNT(*) FROM texts")
        stats["total_texts"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM sentences")
        stats["total_sentences"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM tokens")
        stats["total_tokens"] = c.fetchone()[0]
        
        # By language
        c.execute("""SELECT language, COUNT(*), SUM(token_count), SUM(sentence_count) 
                    FROM texts GROUP BY language ORDER BY SUM(token_count) DESC""")
        stats["by_language"] = {
            row[0]: {"texts": row[1], "tokens": row[2] or 0, "sentences": row[3] or 0} 
            for row in c.fetchall()
        }
        
        # By period
        c.execute("""SELECT period, COUNT(*), SUM(token_count), SUM(sentence_count)
                    FROM texts GROUP BY period ORDER BY period""")
        stats["by_period"] = {
            row[0]: {"texts": row[1], "tokens": row[2] or 0, "sentences": row[3] or 0}
            for row in c.fetchall()
        }
        
        # Greek POS distribution
        c.execute('''SELECT t.pos_ud, COUNT(*) FROM tokens t
                    JOIN sentences s ON t.sentence_id = s.id
                    JOIN texts tx ON s.text_id = tx.id
                    WHERE tx.language = 'grc'
                    GROUP BY t.pos_ud ORDER BY COUNT(*) DESC LIMIT 20''')
        stats["greek_pos_distribution"] = {row[0]: row[1] for row in c.fetchall()}
        
        # Top lemmas in Greek
        c.execute('''SELECT t.lemma, COUNT(*) FROM tokens t
                    JOIN sentences s ON t.sentence_id = s.id
                    JOIN texts tx ON s.text_id = tx.id
                    WHERE tx.language = 'grc' AND t.lemma IS NOT NULL AND t.lemma != '_'
                    GROUP BY t.lemma ORDER BY COUNT(*) DESC LIMIT 50''')
        stats["top_greek_lemmas"] = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
        return stats
        
    def search_lemma(self, lemma: str, language: str = None) -> List[Dict]:
        """Search for occurrences of a lemma"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if language:
            c.execute('''SELECT t.form, t.lemma, t.pos_ud, t.morph, s.raw_text, tx.title, tx.period
                        FROM tokens t
                        JOIN sentences s ON t.sentence_id = s.id
                        JOIN texts tx ON s.text_id = tx.id
                        WHERE t.lemma = ? AND tx.language = ?
                        LIMIT 100''', (lemma, language))
        else:
            c.execute('''SELECT t.form, t.lemma, t.pos_ud, t.morph, s.raw_text, tx.title, tx.period
                        FROM tokens t
                        JOIN sentences s ON t.sentence_id = s.id
                        JOIN texts tx ON s.text_id = tx.id
                        WHERE t.lemma = ?
                        LIMIT 100''', (lemma,))
                        
        results = []
        for row in c.fetchall():
            results.append({
                "form": row[0],
                "lemma": row[1],
                "pos": row[2],
                "morph": row[3],
                "context": row[4],
                "source": row[5],
                "period": row[6]
            })
            
        conn.close()
        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GREEK DIACHRONIC CORPUS COLLECTOR")
    print("Professional Corpus Collection with UD/Penn Annotation")
    print("=" * 70)
    
    collector = GreekCorpusCollector()
    
    # Collect all corpora
    results = collector.collect_all()
    
    # Get and display statistics
    stats = collector.get_statistics()
    
    print("\n" + "=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total tokens: {stats['total_tokens']}")
    
    print("\nBy Language:")
    for lang, data in stats.get('by_language', {}).items():
        print(f"  {lang}: {data['texts']} texts, {data['tokens']:,} tokens, {data['sentences']:,} sentences")
        
    print("\nBy Period:")
    for period, data in stats.get('by_period', {}).items():
        print(f"  {period}: {data['texts']} texts, {data['tokens']:,} tokens")
        
    print("\nGreek POS Distribution (top 10):")
    for pos, count in list(stats.get('greek_pos_distribution', {}).items())[:10]:
        print(f"  {pos}: {count:,}")
        
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
