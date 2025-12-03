"""
Autonomous Corpus Engine - Super Powerful 24/7 Operation
WAL mode for concurrent access, multiple sources, OCR pipeline
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import unicodedata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/corpus_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "data_dir": "/root/corpus_platform/data",
    "db_path": "/root/corpus_platform/data/greek_corpus.db",
    "max_workers": 4,
    "request_timeout": 120,
    "retry_attempts": 3,
    "collection_interval_hours": 1,
    "ocr_enabled": True
}

# =============================================================================
# ALL CORPUS SOURCES - COMPREHENSIVE
# =============================================================================

ALL_SOURCES = {
    # ========== ANCIENT GREEK (with annotation) ==========
    "ud_grc_proiel": {
        "name": "UD Ancient Greek PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master",
        "files": ["grc_proiel-ud-train.conllu", "grc_proiel-ud-dev.conllu", "grc_proiel-ud-test.conllu"],
        "language": "grc", "period": "mixed", "priority": 1
    },
    "ud_grc_perseus": {
        "name": "UD Ancient Greek Perseus",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master",
        "files": ["grc_perseus-ud-train.conllu", "grc_perseus-ud-dev.conllu", "grc_perseus-ud-test.conllu"],
        "language": "grc", "period": "classical", "priority": 1
    },
    
    # ========== LATIN (comparison) ==========
    "ud_la_proiel": {
        "name": "UD Latin PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master",
        "files": ["la_proiel-ud-train.conllu", "la_proiel-ud-dev.conllu", "la_proiel-ud-test.conllu"],
        "language": "la", "period": "classical", "priority": 2
    },
    "ud_la_llct": {
        "name": "UD Latin LLCT (Medieval)",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/master",
        "files": ["la_llct-ud-train.conllu", "la_llct-ud-dev.conllu", "la_llct-ud-test.conllu"],
        "language": "la", "period": "medieval", "priority": 2
    },
    "ud_la_ittb": {
        "name": "UD Latin ITTB (Thomas Aquinas)",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master",
        "files": ["la_ittb-ud-train.conllu", "la_ittb-ud-dev.conllu", "la_ittb-ud-test.conllu"],
        "language": "la", "period": "medieval", "priority": 2
    },
    
    # ========== ROMANCE LANGUAGES ==========
    "ud_es_ancora": {
        "name": "UD Spanish AnCora",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master",
        "files": ["es_ancora-ud-train.conllu", "es_ancora-ud-dev.conllu", "es_ancora-ud-test.conllu"],
        "language": "es", "period": "modern", "priority": 2
    },
    "ud_fr_gsd": {
        "name": "UD French GSD",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master",
        "files": ["fr_gsd-ud-train.conllu", "fr_gsd-ud-dev.conllu", "fr_gsd-ud-test.conllu"],
        "language": "fr", "period": "modern", "priority": 2
    },
    "ud_it_isdt": {
        "name": "UD Italian ISDT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master",
        "files": ["it_isdt-ud-train.conllu", "it_isdt-ud-dev.conllu", "it_isdt-ud-test.conllu"],
        "language": "it", "period": "modern", "priority": 2
    },
    "ud_pt_bosque": {
        "name": "UD Portuguese Bosque",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master",
        "files": ["pt_bosque-ud-train.conllu", "pt_bosque-ud-dev.conllu", "pt_bosque-ud-test.conllu"],
        "language": "pt", "period": "modern", "priority": 2
    },
    "ud_ro_rrt": {
        "name": "UD Romanian RRT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master",
        "files": ["ro_rrt-ud-train.conllu", "ro_rrt-ud-dev.conllu", "ro_rrt-ud-test.conllu"],
        "language": "ro", "period": "modern", "priority": 2
    },
    "ud_ca_ancora": {
        "name": "UD Catalan AnCora",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Catalan-AnCora/master",
        "files": ["ca_ancora-ud-train.conllu", "ca_ancora-ud-dev.conllu", "ca_ancora-ud-test.conllu"],
        "language": "ca", "period": "modern", "priority": 3
    },
    
    # ========== OLD/MEDIEVAL ROMANCE ==========
    "ud_fro_srcmf": {
        "name": "UD Old French SRCMF",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/master",
        "files": ["fro_srcmf-ud-train.conllu", "fro_srcmf-ud-dev.conllu", "fro_srcmf-ud-test.conllu"],
        "language": "fro", "period": "medieval", "priority": 2
    },
    
    # ========== ENGLISH ==========
    "ud_en_ewt": {
        "name": "UD English EWT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master",
        "files": ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"],
        "language": "en", "period": "modern", "priority": 2
    },
    
    # ========== OTHER INDO-EUROPEAN (PROIEL family) ==========
    "ud_got_proiel": {
        "name": "UD Gothic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/master",
        "files": ["got_proiel-ud-train.conllu", "got_proiel-ud-dev.conllu", "got_proiel-ud-test.conllu"],
        "language": "got", "period": "ancient", "priority": 3
    },
    "ud_cu_proiel": {
        "name": "UD Old Church Slavonic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/master",
        "files": ["cu_proiel-ud-train.conllu", "cu_proiel-ud-dev.conllu", "cu_proiel-ud-test.conllu"],
        "language": "cu", "period": "medieval", "priority": 3
    },
    
    # ========== MODERN GREEK ==========
    "ud_el_gdt": {
        "name": "UD Modern Greek GDT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master",
        "files": ["el_gdt-ud-train.conllu", "el_gdt-ud-dev.conllu", "el_gdt-ud-test.conllu"],
        "language": "el", "period": "modern", "priority": 1
    },
    
    # ========== MORE LATIN ==========
    "ud_la_perseus": {
        "name": "UD Latin Perseus",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-Perseus/master",
        "files": ["la_perseus-ud-train.conllu", "la_perseus-ud-dev.conllu", "la_perseus-ud-test.conllu"],
        "language": "la", "period": "classical", "priority": 2
    },
    "ud_la_udante": {
        "name": "UD Latin UDante (Dante)",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-UDante/master",
        "files": ["la_udante-ud-train.conllu", "la_udante-ud-dev.conllu", "la_udante-ud-test.conllu"],
        "language": "la", "period": "medieval", "priority": 3
    },
    
    # ========== MORE ROMANCE ==========
    "ud_gl_ctg": {
        "name": "UD Galician CTG",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Galician-CTG/master",
        "files": ["gl_ctg-ud-train.conllu", "gl_ctg-ud-dev.conllu", "gl_ctg-ud-test.conllu"],
        "language": "gl", "period": "modern", "priority": 3
    },
    "ud_oc_torotxa": {
        "name": "UD Occitan Torotxa",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Occitan-Torotxa/master",
        "files": ["oc_torotxa-ud-test.conllu"],
        "language": "oc", "period": "modern", "priority": 3
    },
    
    # ========== OLD EAST SLAVIC ==========
    "ud_orv_rnc": {
        "name": "UD Old Russian RNC",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_East_Slavic-RNC/master",
        "files": ["orv_rnc-ud-train.conllu", "orv_rnc-ud-dev.conllu", "orv_rnc-ud-test.conllu"],
        "language": "orv", "period": "medieval", "priority": 3
    },
    
    # ========== GERMANIC ==========
    "ud_de_gsd": {
        "name": "UD German GSD",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master",
        "files": ["de_gsd-ud-train.conllu", "de_gsd-ud-dev.conllu", "de_gsd-ud-test.conllu"],
        "language": "de", "period": "modern", "priority": 3
    },
    
    # ========== ARMENIAN ==========
    "ud_hy_armtdp": {
        "name": "UD Armenian ArmTDP",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Armenian-ArmTDP/master",
        "files": ["hy_armtdp-ud-train.conllu", "hy_armtdp-ud-dev.conllu", "hy_armtdp-ud-test.conllu"],
        "language": "hy", "period": "modern", "priority": 3
    },
    
    # ========== COPTIC (for Byzantine studies) ==========
    "ud_cop_scriptorium": {
        "name": "UD Coptic Scriptorium",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Coptic-Scriptorium/master",
        "files": ["cop_scriptorium-ud-train.conllu", "cop_scriptorium-ud-dev.conllu", "cop_scriptorium-ud-test.conllu"],
        "language": "cop", "period": "ancient", "priority": 3
    },
    
    # ========== HEBREW (for Biblical studies) ==========
    "ud_hbo_ptnk": {
        "name": "UD Ancient Hebrew PTNK",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Hebrew-PTNK/master",
        "files": ["hbo_ptnk-ud-train.conllu", "hbo_ptnk-ud-dev.conllu", "hbo_ptnk-ud-test.conllu"],
        "language": "hbo", "period": "ancient", "priority": 3
    },
    
    # ========== ARABIC (for Byzantine/Medieval contact) ==========
    "ud_ar_padt": {
        "name": "UD Arabic PADT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master",
        "files": ["ar_padt-ud-train.conllu", "ar_padt-ud-dev.conllu", "ar_padt-ud-test.conllu"],
        "language": "ar", "period": "classical", "priority": 3
    }
}

# =============================================================================
# POS MAPPINGS (UD to Penn Treebank style)
# =============================================================================

UD_TO_PENN = {
    "NOUN": "NN", "PROPN": "NNP", "VERB": "VB", "AUX": "MD",
    "ADJ": "JJ", "ADV": "RB", "PRON": "PRP", "DET": "DT",
    "ADP": "IN", "CONJ": "CC", "CCONJ": "CC", "SCONJ": "IN",
    "NUM": "CD", "PART": "RP", "INTJ": "UH", "PUNCT": ".",
    "SYM": "SYM", "X": "XX"
}

# =============================================================================
# TEXT CLEANING
# =============================================================================

def remove_emojis(text: str) -> str:
    """Remove all emojis and special Unicode symbols"""
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55"
        "\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def clean_text(text: str) -> str:
    """Full text cleaning pipeline"""
    if not text:
        return ""
    text = remove_emojis(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =============================================================================
# TOKEN DATACLASS
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

# =============================================================================
# DATABASE WITH WAL MODE (NO LOCKING)
# =============================================================================

class CorpusDatabase:
    """SQLite database with WAL mode for concurrent access"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with WAL mode"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        return conn
        
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS texts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            period TEXT,
            century TEXT,
            genre TEXT,
            source TEXT,
            language TEXT DEFAULT 'grc',
            token_count INTEGER DEFAULT 0,
            sentence_count INTEGER DEFAULT 0,
            file_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id TEXT NOT NULL,
            sentence_num INTEGER,
            raw_text TEXT,
            token_count INTEGER,
            FOREIGN KEY (text_id) REFERENCES texts(id) ON DELETE CASCADE
        )''')
        
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
        
        c.execute('''CREATE TABLE IF NOT EXISTS collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            status TEXT,
            sentences_added INTEGER,
            tokens_added INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Indexes for fast queries
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma)',
            'CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos_ud)',
            'CREATE INDEX IF NOT EXISTS idx_tokens_form ON tokens(form)',
            'CREATE INDEX IF NOT EXISTS idx_texts_period ON texts(period)',
            'CREATE INDEX IF NOT EXISTS idx_texts_language ON texts(language)',
            'CREATE INDEX IF NOT EXISTS idx_sentences_text ON sentences(text_id)'
        ]
        for idx in indexes:
            c.execute(idx)
            
        conn.commit()
        conn.close()
        logger.info(f"Database initialized with WAL mode: {self.db_path}")
        
    def store_text(self, text_id: str, title: str, author: str, period: str, 
                   source: str, sentences: List[List[Token]], language: str, 
                   file_hash: str = None) -> Tuple[int, int]:
        """Store text with sentences and tokens"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Check if already exists with same hash (skip if unchanged)
        c.execute('SELECT file_hash FROM texts WHERE id = ?', (text_id,))
        row = c.fetchone()
        if row and row[0] == file_hash:
            conn.close()
            return 0, 0  # Already up to date
            
        total_tokens = sum(len(s) for s in sentences)
        
        # Delete old data
        c.execute('SELECT id FROM sentences WHERE text_id = ?', (text_id,))
        sent_ids = [r[0] for r in c.fetchall()]
        if sent_ids:
            placeholders = ','.join('?' * len(sent_ids))
            c.execute(f'DELETE FROM tokens WHERE sentence_id IN ({placeholders})', sent_ids)
        c.execute('DELETE FROM sentences WHERE text_id = ?', (text_id,))
        c.execute('DELETE FROM texts WHERE id = ?', (text_id,))
        
        # Insert new text
        c.execute('''INSERT INTO texts (id, title, author, period, source, language, 
                     token_count, sentence_count, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (text_id, title, author, period, source, language, 
                  total_tokens, len(sentences), file_hash))
        
        # Insert sentences and tokens
        for sent_num, sentence in enumerate(sentences):
            raw_text = ' '.join(t.form for t in sentence)
            c.execute('''INSERT INTO sentences (text_id, sentence_num, raw_text, token_count) 
                        VALUES (?, ?, ?, ?)''',
                     (text_id, sent_num, raw_text, len(sentence)))
            sid = c.lastrowid
            
            for token in sentence:
                c.execute('''INSERT INTO tokens 
                            (sentence_id, token_num, form, lemma, pos_ud, pos_penn, 
                             morph, head, deprel, misc)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (sid, token.id, token.form, token.lemma, token.pos_ud, 
                          token.pos_penn, token.morph, token.head, token.deprel, token.misc))
                          
        conn.commit()
        conn.close()
        return len(sentences), total_tokens
        
    def log_collection(self, source: str, status: str, sentences: int, tokens: int):
        """Log collection activity"""
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO collection_log (source, status, sentences_added, tokens_added) 
                    VALUES (?, ?, ?, ?)''', (source, status, sentences, tokens))
        conn.commit()
        conn.close()
        
    def get_statistics(self) -> Dict:
        """Get comprehensive corpus statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM texts")
        stats["total_texts"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM sentences")
        stats["total_sentences"] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM tokens")
        stats["total_tokens"] = c.fetchone()[0]
        
        c.execute("""SELECT language, COUNT(*), SUM(token_count) 
                    FROM texts GROUP BY language ORDER BY SUM(token_count) DESC""")
        stats["by_language"] = {r[0]: {"texts": r[1], "tokens": r[2] or 0} for r in c.fetchall()}
        
        c.execute("""SELECT period, COUNT(*), SUM(token_count) 
                    FROM texts GROUP BY period ORDER BY period""")
        stats["by_period"] = {r[0]: {"texts": r[1], "tokens": r[2] or 0} for r in c.fetchall()}
        
        conn.close()
        return stats

# =============================================================================
# CONLLU PARSER
# =============================================================================

class CoNLLUParser:
    """Parse CoNLL-U format (Universal Dependencies standard)"""
    
    @staticmethod
    def parse_content(content: str) -> List[List[Token]]:
        """Parse CoNLL-U content string into sentences"""
        sentences = []
        current = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
                
            if line.startswith('#'):
                continue
                
            parts = line.split('\t')
            if len(parts) >= 10:
                try:
                    tid = parts[0]
                    # Skip multi-word tokens and empty nodes
                    if '-' in tid or '.' in tid:
                        continue
                        
                    token = Token(
                        id=int(tid),
                        form=clean_text(parts[1]),
                        lemma=clean_text(parts[2]),
                        pos_ud=parts[3],
                        pos_penn=UD_TO_PENN.get(parts[3], "XX"),
                        morph=parts[5],
                        head=int(parts[6]) if parts[6] != '_' else 0,
                        deprel=parts[7],
                        misc=parts[9] if len(parts) > 9 else ""
                    )
                    current.append(token)
                except (ValueError, IndexError):
                    pass
                    
        if current:
            sentences.append(current)
            
        return sentences

# =============================================================================
# AUTONOMOUS ENGINE
# =============================================================================

class AutonomousEngine:
    """Main autonomous corpus collection engine"""
    
    def __init__(self):
        Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
        self.db = CorpusDatabase(CONFIG["db_path"])
        
    def download_file(self, url: str) -> Optional[str]:
        """Download file with retry logic"""
        for attempt in range(CONFIG["retry_attempts"]):
            try:
                response = requests.get(url, timeout=CONFIG["request_timeout"])
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2 ** attempt)
        return None
        
    def collect_source(self, source_id: str, source_info: dict) -> Tuple[int, int]:
        """Collect a single corpus source"""
        logger.info(f"Collecting {source_info['name']}...")
        total_sentences = 0
        total_tokens = 0
        
        for filename in source_info.get("files", []):
            url = f"{source_info['url']}/{filename}"
            content = self.download_file(url)
            
            if content:
                file_hash = hashlib.md5(content.encode()).hexdigest()
                sentences = CoNLLUParser.parse_content(content)
                
                text_id = f"{source_id}_{filename.replace('.conllu', '')}"
                sent_count, tok_count = self.db.store_text(
                    text_id=text_id,
                    title=source_info['name'],
                    author="Various",
                    period=source_info.get('period', 'unknown'),
                    source=source_id,
                    sentences=sentences,
                    language=source_info.get('language', 'unknown'),
                    file_hash=file_hash
                )
                total_sentences += sent_count
                total_tokens += tok_count
                
                if sent_count > 0:
                    logger.info(f"  {filename}: {sent_count} sentences, {tok_count} tokens")
                    
        self.db.log_collection(source_id, "success", total_sentences, total_tokens)
        logger.info(f"Completed {source_info['name']}: {total_sentences} sentences, {total_tokens} tokens")
        return total_sentences, total_tokens
        
    def collect_all(self, priority_filter: int = None) -> Dict:
        """Collect all configured corpora"""
        logger.info("=" * 60)
        logger.info("STARTING AUTONOMOUS COLLECTION")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        results = {}
        sources = ALL_SOURCES
        
        if priority_filter:
            sources = {k: v for k, v in sources.items() if v.get('priority', 99) <= priority_filter}
            
        # Sort by priority
        sorted_sources = sorted(sources.items(), key=lambda x: x[1].get('priority', 99))
        
        for source_id, source_info in sorted_sources:
            try:
                sent, tok = self.collect_source(source_id, source_info)
                results[source_id] = {"sentences": sent, "tokens": tok, "status": "success"}
            except Exception as e:
                logger.error(f"Failed to collect {source_id}: {e}")
                results[source_id] = {"sentences": 0, "tokens": 0, "status": f"error: {e}"}
                
        # Summary
        total_sent = sum(r["sentences"] for r in results.values())
        total_tok = sum(r["tokens"] for r in results.values())
        
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info(f"New data: {total_sent} sentences, {total_tok} tokens")
        logger.info("=" * 60)
        
        stats = self.db.get_statistics()
        logger.info(f"Database totals: {stats['total_texts']} texts, "
                   f"{stats['total_sentences']:,} sentences, {stats['total_tokens']:,} tokens")
        
        return results
        
    def run_forever(self):
        """Run collection loop forever"""
        logger.info("Starting autonomous engine in continuous mode...")
        
        while True:
            try:
                self.collect_all()
                stats = self.db.get_statistics()
                logger.info(f"Cycle complete. DB: {stats['total_tokens']:,} tokens")
            except Exception as e:
                logger.error(f"Collection cycle error: {e}")
                
            # Wait for next cycle
            wait_hours = CONFIG["collection_interval_hours"]
            logger.info(f"Sleeping for {wait_hours} hour(s)...")
            time.sleep(wait_hours * 3600)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    engine = AutonomousEngine()
    
    # Run one collection cycle
    engine.collect_all()
    
    # Print stats
    stats = engine.db.get_statistics()
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Total sentences: {stats['total_sentences']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print("\nBy Language:")
    for lang, data in stats.get('by_language', {}).items():
        print(f"  {lang}: {data['texts']} texts, {data['tokens']:,} tokens")
    print("\nBy Period:")
    for period, data in stats.get('by_period', {}).items():
        print(f"  {period}: {data['texts']} texts, {data['tokens']:,} tokens")
