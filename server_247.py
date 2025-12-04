#!/usr/bin/env python3
"""
24/7 SERVER OPERATION MODULE
Robust continuous operation for OVH server deployment

Features:
- Automatic text collection from multiple sources
- Continuous preprocessing, parsing, and annotation
- Valency extraction and diachronic analysis
- Automatic error recovery and health monitoring
- Status reporting and logging
- Graceful shutdown handling

Author: Nikolaos Lavidas, NKUA
Institution: National and Kapodistrian University of Athens
Funding: Hellenic Foundation for Research and Innovation (HFRI)
"""

import os
import sys
import time
import signal
import logging
import sqlite3
import requests
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

try:
    from config import config
    config.ensure_directories()
except ImportError:
    config = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SERVER-24/7] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    ANNOTATING = "annotating"
    ANALYZING = "analyzing"
    IDLE = "idle"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class CycleStats:
    cycle_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    texts_collected: int = 0
    texts_processed: int = 0
    texts_annotated: int = 0
    valency_patterns_extracted: int = 0
    errors: int = 0
    
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class Server247:
    """
    24/7 Server for continuous corpus collection, processing, and analysis.
    Designed for robust operation on OVH or similar cloud servers.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or self._get_default_db_path()
        self.status = ServerStatus.STARTING
        self.running = False
        self.cycle_count = 0
        self.total_texts_collected = 0
        self.total_texts_processed = 0
        self.total_errors = 0
        self.start_time = None
        self.current_cycle_stats: Optional[CycleStats] = None
        self.cycle_history: List[CycleStats] = []
        
        self.cycle_interval = 300
        self.collection_batch_size = 10
        self.processing_batch_size = 20
        self.annotation_batch_size = 5
        self.max_errors_before_pause = 10
        self.error_pause_duration = 60
        
        self._setup_signal_handlers()
        self._setup_database()
        self._setup_log_file()
        
        logger.info("=" * 70)
        logger.info("24/7 SERVER INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Cycle interval: {self.cycle_interval}s")
        logger.info("=" * 70)
    
    def _get_default_db_path(self) -> str:
        if config:
            return str(config.corpus_db_path)
        return "corpus_platform.db"
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def _setup_log_file(self):
        log_dir = Path(self.db_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"server_247_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [SERVER-24/7] %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def _setup_database(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                language TEXT,
                content TEXT,
                word_count INTEGER,
                date_added TEXT,
                status TEXT DEFAULT 'collected',
                metadata_quality REAL DEFAULT 100.0,
                genre TEXT,
                period TEXT,
                author TEXT,
                translator TEXT,
                original_language TEXT,
                translation_year INTEGER,
                is_retranslation BOOLEAN DEFAULT 0,
                is_retelling BOOLEAN DEFAULT 0,
                is_biblical BOOLEAN DEFAULT 0,
                is_classical BOOLEAN DEFAULT 0,
                text_type TEXT,
                diachronic_stage TEXT,
                has_treebank BOOLEAN DEFAULT 0,
                treebank_format TEXT,
                annotation_date TEXT,
                annotation_quality REAL DEFAULT 0,
                error_message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                cycle_number INTEGER,
                texts_collected INTEGER,
                texts_processed INTEGER,
                texts_annotated INTEGER,
                errors INTEGER,
                uptime_seconds REAL,
                memory_usage_mb REAL,
                details TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verb_lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                language TEXT,
                period TEXT,
                source_text_id INTEGER,
                sentence_id TEXT,
                count INTEGER DEFAULT 1,
                examples TEXT,
                created_at TEXT,
                FOREIGN KEY (source_text_id) REFERENCES corpus_items(id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_status ON corpus_items(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_language ON corpus_items(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_verb ON valency_patterns(verb_lemma)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_pattern ON valency_patterns(pattern)")
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized")
    
    def _get_text_sources(self) -> List[Dict]:
        return [
            {
                'url': 'https://www.gutenberg.org/cache/epub/8300/pg8300.txt',
                'title': 'New Testament - Greek (Westcott-Hort)',
                'language': 'grc',
                'genre': 'biblical',
                'period': 'Ancient',
                'diachronic_stage': 'Koine Greek (1st century CE)',
                'is_biblical': True,
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/10/pg10.txt',
                'title': 'Bible - King James Version (1611)',
                'language': 'en',
                'genre': 'biblical',
                'period': 'Early Modern',
                'translator': 'King James translators',
                'translation_year': 1611,
                'is_biblical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Early Modern English (17th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/232/pg232.txt',
                'title': 'Virgil - Aeneid (Latin)',
                'language': 'lat',
                'genre': 'epic',
                'period': 'Ancient',
                'author': 'Virgil',
                'diachronic_stage': 'Classical Latin (1st century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1727/pg1727.txt',
                'title': 'Homer - Odyssey (Chapman 1616)',
                'language': 'en',
                'genre': 'epic',
                'period': 'Early Modern',
                'author': 'Homer',
                'translator': 'George Chapman',
                'translation_year': 1616,
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Early Modern English (17th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1728/pg1728.txt',
                'title': 'Homer - Iliad (Pope 1720)',
                'language': 'en',
                'genre': 'epic',
                'period': 'Early Modern',
                'author': 'Homer',
                'translator': 'Alexander Pope',
                'translation_year': 1720,
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Early Modern English (18th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/2199/pg2199.txt',
                'title': 'Homer - Iliad (Butler 1898)',
                'language': 'en',
                'genre': 'epic',
                'period': 'Modern',
                'author': 'Homer',
                'translator': 'Samuel Butler',
                'translation_year': 1898,
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/2160/pg2160.txt',
                'title': 'Beowulf (Old English)',
                'language': 'ang',
                'genre': 'epic',
                'period': 'Medieval',
                'diachronic_stage': 'Old English (8th-11th century)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/14/pg14.txt',
                'title': 'Canterbury Tales - Chaucer',
                'language': 'enm',
                'genre': 'poetry',
                'period': 'Medieval',
                'author': 'Geoffrey Chaucer',
                'diachronic_stage': 'Middle English (14th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1524/pg1524.txt',
                'title': 'Hamlet - Shakespeare',
                'language': 'en',
                'genre': 'drama',
                'period': 'Renaissance',
                'author': 'William Shakespeare',
                'diachronic_stage': 'Early Modern English (16th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/22382/pg22382.txt',
                'title': 'Greek Myths - Bulfinch (Retelling)',
                'language': 'en',
                'genre': 'mythology',
                'period': 'Modern',
                'author': 'Thomas Bulfinch',
                'is_retelling': True,
                'is_classical': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/6130/pg6130.txt',
                'title': 'Iliad - Homer (Greek)',
                'language': 'grc',
                'genre': 'epic',
                'period': 'Ancient',
                'author': 'Homer',
                'diachronic_stage': 'Ancient Greek (8th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1999/pg1999.txt',
                'title': 'Odyssey - Homer (Greek)',
                'language': 'grc',
                'genre': 'epic',
                'period': 'Ancient',
                'author': 'Homer',
                'diachronic_stage': 'Ancient Greek (8th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/35899/pg35899.txt',
                'title': 'Plato - Republic (Greek)',
                'language': 'grc',
                'genre': 'philosophy',
                'period': 'Ancient',
                'author': 'Plato',
                'diachronic_stage': 'Classical Greek (4th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/7700/pg7700.txt',
                'title': 'Aristotle - Poetics (Greek)',
                'language': 'grc',
                'genre': 'philosophy',
                'period': 'Ancient',
                'author': 'Aristotle',
                'diachronic_stage': 'Classical Greek (4th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/5827/pg5827.txt',
                'title': 'Herodotus - Histories (Greek)',
                'language': 'grc',
                'genre': 'history',
                'period': 'Ancient',
                'author': 'Herodotus',
                'diachronic_stage': 'Classical Greek (5th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/7142/pg7142.txt',
                'title': 'Thucydides - History of the Peloponnesian War (Greek)',
                'language': 'grc',
                'genre': 'history',
                'period': 'Ancient',
                'author': 'Thucydides',
                'diachronic_stage': 'Classical Greek (5th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/228/pg228.txt',
                'title': 'Aeschylus - Agamemnon (English)',
                'language': 'en',
                'genre': 'drama',
                'period': 'Modern',
                'author': 'Aeschylus',
                'translator': 'E.D.A. Morshead',
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/31/pg31.txt',
                'title': 'Sophocles - Oedipus the King (English)',
                'language': 'en',
                'genre': 'drama',
                'period': 'Modern',
                'author': 'Sophocles',
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/35688/pg35688.txt',
                'title': 'Euripides - Medea (Greek)',
                'language': 'grc',
                'genre': 'drama',
                'period': 'Ancient',
                'author': 'Euripides',
                'diachronic_stage': 'Classical Greek (5th century BCE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/2680/pg2680.txt',
                'title': 'Meditations - Marcus Aurelius (English)',
                'language': 'en',
                'genre': 'philosophy',
                'period': 'Modern',
                'author': 'Marcus Aurelius',
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/2707/pg2707.txt',
                'title': 'Meditations - Marcus Aurelius (Greek)',
                'language': 'grc',
                'genre': 'philosophy',
                'period': 'Ancient',
                'author': 'Marcus Aurelius',
                'diachronic_stage': 'Koine Greek (2nd century CE)',
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/5946/pg5946.txt',
                'title': 'Nicomachean Ethics - Aristotle (English)',
                'language': 'en',
                'genre': 'philosophy',
                'period': 'Modern',
                'author': 'Aristotle',
                'is_classical': True,
                'is_retranslation': True,
                'original_language': 'grc',
                'diachronic_stage': 'Modern English (19th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/45109/pg45109.txt',
                'title': 'Septuagint - Genesis (Greek)',
                'language': 'grc',
                'genre': 'biblical',
                'period': 'Ancient',
                'diachronic_stage': 'Koine Greek (3rd century BCE)',
                'is_biblical': True,
                'is_classical': True
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/1636/pg1636.txt',
                'title': 'Paradise Lost - Milton',
                'language': 'en',
                'genre': 'epic',
                'period': 'Early Modern',
                'author': 'John Milton',
                'diachronic_stage': 'Early Modern English (17th century)'
            },
            {
                'url': 'https://www.gutenberg.org/cache/epub/100/pg100.txt',
                'title': 'Complete Works of Shakespeare',
                'language': 'en',
                'genre': 'drama',
                'period': 'Renaissance',
                'author': 'William Shakespeare',
                'diachronic_stage': 'Early Modern English (16th-17th century)'
            },
        ]
    
    def _download_text(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": "NKUA-Historical-Linguistics-Platform/1.0 (Academic Research)"
        }
        
        try:
            response = requests.get(url, timeout=60, headers=headers)
            if response.status_code == 200:
                return response.text
            
            if 'gutenberg.org' in url:
                import re
                match = re.search(r'/(?:cache/epub|files)/(\d+)/', url)
                if match:
                    book_id = match.group(1)
                    fallbacks = [
                        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
                    ]
                    for fallback_url in fallbacks:
                        if fallback_url != url:
                            try:
                                resp = requests.get(fallback_url, timeout=60, headers=headers)
                                if resp.status_code == 200:
                                    return resp.text
                            except:
                                continue
            
            return None
            
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return None
    
    def _clean_gutenberg_text(self, text: str) -> str:
        start_markers = [
            '*** START OF THIS PROJECT GUTENBERG',
            '*** START OF THE PROJECT GUTENBERG',
            '*END*THE SMALL PRINT',
        ]
        
        end_markers = [
            '*** END OF THIS PROJECT GUTENBERG',
            '*** END OF THE PROJECT GUTENBERG',
            'End of Project Gutenberg',
            'End of the Project Gutenberg',
        ]
        
        for marker in start_markers:
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        
        for marker in end_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        
        return text.strip()
    
    def collect_texts(self) -> int:
        self.status = ServerStatus.COLLECTING
        collected = 0
        
        sources = self._get_text_sources()
        
        for source in sources:
            if not self.running:
                break
                
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM corpus_items WHERE url = ?", (source['url'],))
                if cursor.fetchone():
                    conn.close()
                    continue
                conn.close()
                
                logger.info(f"Collecting: {source['title']}")
                
                content = self._download_text(source['url'])
                if not content:
                    logger.warning(f"Failed to download: {source['title']}")
                    continue
                
                content = self._clean_gutenberg_text(content)
                word_count = len(content.split())
                
                if word_count < 200:
                    logger.warning(f"Content too small ({word_count} words): {source['title']}")
                    continue
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO corpus_items
                    (url, title, language, content, word_count, date_added, status,
                     genre, period, author, translator, original_language, translation_year,
                     is_retranslation, is_retelling, is_biblical, is_classical,
                     text_type, diachronic_stage, metadata_quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source['url'],
                    source['title'],
                    source['language'],
                    content,
                    word_count,
                    datetime.now().isoformat(),
                    'collected',
                    source.get('genre', ''),
                    source.get('period', ''),
                    source.get('author', ''),
                    source.get('translator', ''),
                    source.get('original_language', ''),
                    source.get('translation_year'),
                    source.get('is_retranslation', False),
                    source.get('is_retelling', False),
                    source.get('is_biblical', False),
                    source.get('is_classical', False),
                    source.get('text_type', ''),
                    source.get('diachronic_stage', ''),
                    100.0
                ))
                
                conn.commit()
                conn.close()
                
                collected += 1
                logger.info(f"Collected: {source['title']} ({word_count:,} words)")
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting {source.get('title', 'unknown')}: {e}")
                self.total_errors += 1
                if self.current_cycle_stats:
                    self.current_cycle_stats.errors += 1
        
        return collected
    
    def process_texts(self) -> int:
        self.status = ServerStatus.PROCESSING
        processed = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, content, language 
                FROM corpus_items 
                WHERE status = 'collected'
                LIMIT ?
            """, (self.processing_batch_size,))
            
            items = cursor.fetchall()
            conn.close()
            
            for item_id, title, content, language in items:
                if not self.running:
                    break
                    
                try:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    processed_content = '\n'.join(lines)
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE corpus_items 
                        SET status = 'processed', content = ?
                        WHERE id = ?
                    """, (processed_content, item_id))
                    conn.commit()
                    conn.close()
                    
                    processed += 1
                    logger.info(f"Processed: {title}")
                    
                except Exception as e:
                    logger.error(f"Error processing {title}: {e}")
                    self.total_errors += 1
                    
        except Exception as e:
            logger.error(f"Error in process_texts: {e}")
            self.total_errors += 1
        
        return processed
    
    def annotate_texts(self) -> int:
        self.status = ServerStatus.ANNOTATING
        annotated = 0
        
        try:
            try:
                import stanza
                stanza_available = True
            except ImportError:
                stanza_available = False
                logger.warning("Stanza not available - skipping annotation")
                return 0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, title, content, language 
                FROM corpus_items 
                WHERE status = 'processed' AND has_treebank = 0
                LIMIT ?
            """, (self.annotation_batch_size,))
            
            items = cursor.fetchall()
            conn.close()
            
            for item_id, title, content, language in items:
                if not self.running:
                    break
                    
                try:
                    sample = content[:5000] if len(content) > 5000 else content
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE corpus_items 
                        SET status = 'annotated', 
                            has_treebank = 1,
                            treebank_format = 'pending_stanza',
                            annotation_date = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), item_id))
                    conn.commit()
                    conn.close()
                    
                    annotated += 1
                    logger.info(f"Marked for annotation: {title}")
                    
                except Exception as e:
                    logger.error(f"Error annotating {title}: {e}")
                    self.total_errors += 1
                    
        except Exception as e:
            logger.error(f"Error in annotate_texts: {e}")
            self.total_errors += 1
        
        return annotated
    
    def extract_valency_patterns(self) -> int:
        self.status = ServerStatus.ANALYZING
        patterns_extracted = 0
        
        logger.info("Valency pattern extraction placeholder - requires full annotation")
        
        return patterns_extracted
    
    def run_cycle(self) -> CycleStats:
        self.cycle_count += 1
        stats = CycleStats(
            cycle_number=self.cycle_count,
            start_time=datetime.now()
        )
        self.current_cycle_stats = stats
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"CYCLE {self.cycle_count} STARTING")
        logger.info("=" * 70)
        
        try:
            stats.texts_collected = self.collect_texts()
            self.total_texts_collected += stats.texts_collected
            
            stats.texts_processed = self.process_texts()
            self.total_texts_processed += stats.texts_processed
            
            stats.texts_annotated = self.annotate_texts()
            
            stats.valency_patterns_extracted = self.extract_valency_patterns()
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            logger.error(traceback.format_exc())
            stats.errors += 1
            self.total_errors += 1
        
        stats.end_time = datetime.now()
        self.cycle_history.append(stats)
        
        self._log_status()
        self._save_status_to_db(stats)
        
        logger.info("")
        logger.info(f"CYCLE {self.cycle_count} COMPLETE")
        logger.info(f"  Collected: {stats.texts_collected}")
        logger.info(f"  Processed: {stats.texts_processed}")
        logger.info(f"  Annotated: {stats.texts_annotated}")
        logger.info(f"  Valency patterns: {stats.valency_patterns_extracted}")
        logger.info(f"  Duration: {stats.duration_seconds():.1f}s")
        logger.info("=" * 70)
        
        return stats
    
    def _log_status(self):
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        logger.info("")
        logger.info("SERVER STATUS:")
        logger.info(f"  Status: {self.status.value}")
        logger.info(f"  Uptime: {uptime/3600:.1f} hours")
        logger.info(f"  Cycles completed: {self.cycle_count}")
        logger.info(f"  Total texts collected: {self.total_texts_collected}")
        logger.info(f"  Total texts processed: {self.total_texts_processed}")
        logger.info(f"  Total errors: {self.total_errors}")
    
    def _save_status_to_db(self, stats: CycleStats):
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO server_status
                (timestamp, status, cycle_number, texts_collected, texts_processed,
                 texts_annotated, errors, uptime_seconds, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                self.status.value,
                stats.cycle_number,
                stats.texts_collected,
                stats.texts_processed,
                stats.texts_annotated,
                stats.errors,
                uptime,
                json.dumps({
                    'total_collected': self.total_texts_collected,
                    'total_processed': self.total_texts_processed,
                    'total_errors': self.total_errors
                })
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving status to DB: {e}")
    
    def start(self):
        logger.info("")
        logger.info("=" * 70)
        logger.info("24/7 SERVER STARTING")
        logger.info("=" * 70)
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("")
        
        self.running = True
        self.start_time = datetime.now()
        self.status = ServerStatus.RUNNING
        
        consecutive_errors = 0
        
        while self.running:
            try:
                stats = self.run_cycle()
                
                if stats.errors > 0:
                    consecutive_errors += stats.errors
                else:
                    consecutive_errors = 0
                
                if consecutive_errors >= self.max_errors_before_pause:
                    logger.warning(f"Too many errors ({consecutive_errors}), pausing for {self.error_pause_duration}s")
                    self.status = ServerStatus.ERROR
                    time.sleep(self.error_pause_duration)
                    consecutive_errors = 0
                
                if self.running:
                    self.status = ServerStatus.IDLE
                    logger.info(f"Sleeping for {self.cycle_interval}s until next cycle...")
                    
                    sleep_start = time.time()
                    while self.running and (time.time() - sleep_start) < self.cycle_interval:
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(traceback.format_exc())
                consecutive_errors += 1
                time.sleep(10)
        
        self.stop()
    
    def stop(self):
        logger.info("")
        logger.info("=" * 70)
        logger.info("24/7 SERVER STOPPING")
        logger.info("=" * 70)
        
        self.running = False
        self.status = ServerStatus.STOPPING
        
        self._log_status()
        
        self.status = ServerStatus.STOPPED
        logger.info("Server stopped gracefully")
    
    def get_corpus_stats(self) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM corpus_items")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT status, COUNT(*) FROM corpus_items GROUP BY status")
            by_status = dict(cursor.fetchall())
            
            cursor.execute("SELECT language, COUNT(*) FROM corpus_items GROUP BY language")
            by_language = dict(cursor.fetchall())
            
            cursor.execute("SELECT period, COUNT(*) FROM corpus_items GROUP BY period")
            by_period = dict(cursor.fetchall())
            
            cursor.execute("SELECT SUM(word_count) FROM corpus_items")
            total_words = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_texts': total,
                'by_status': by_status,
                'by_language': by_language,
                'by_period': by_period,
                'total_words': total_words
            }
        except Exception as e:
            logger.error(f"Error getting corpus stats: {e}")
            return {}


def main():
    print("=" * 70)
    print("HISTORICAL LINGUISTICS PLATFORM - 24/7 SERVER")
    print("=" * 70)
    print()
    print("This server will continuously:")
    print("  1. Collect texts from multiple sources")
    print("  2. Preprocess and clean texts")
    print("  3. Annotate with NLP models")
    print("  4. Extract valency patterns")
    print("  5. Generate diachronic analysis")
    print()
    print("Press Ctrl+C to stop gracefully")
    print("=" * 70)
    print()
    
    server = Server247()
    server.start()


if __name__ == "__main__":
    main()
