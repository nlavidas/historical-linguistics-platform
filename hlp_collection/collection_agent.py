"""
Collection Agent - 24/7 Text Collection Orchestrator

This module provides a comprehensive collection agent that orchestrates
text collection from multiple sources with OCR capabilities.

Features:
- Multi-source collection (Perseus, First1KGreek, Internet Archive, Byzantine, Gutenberg)
- OCR for digitized manuscripts
- Rate limiting and polite crawling
- Persistent storage with SQLite
- 24/7 continuous operation
- Progress tracking and statistics

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import threading
import time
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import queue

logger = logging.getLogger(__name__)


class CollectionSource(Enum):
    PERSEUS = "perseus"
    FIRST1KGREEK = "first1kgreek"
    INTERNET_ARCHIVE = "internet_archive"
    BYZANTINE = "byzantine"
    GUTENBERG = "gutenberg"
    OCR = "ocr"
    CUSTOM = "custom"


class CollectionStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class CollectionConfig:
    sources: List[CollectionSource] = field(default_factory=lambda: [
        CollectionSource.PERSEUS,
        CollectionSource.FIRST1KGREEK,
        CollectionSource.GUTENBERG,
        CollectionSource.BYZANTINE,
    ])
    languages: List[str] = field(default_factory=lambda: ['grc', 'lat', 'en'])
    max_texts_per_source: int = 100
    collection_interval_hours: float = 6.0
    rate_limit_seconds: float = 2.0
    enable_ocr: bool = True
    ocr_engine: str = "tesseract"
    database_path: str = "data/collected_texts.db"
    log_dir: str = "logs/collection"
    retry_failed: bool = True
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sources': [s.value for s in self.sources],
            'languages': self.languages,
            'max_texts_per_source': self.max_texts_per_source,
            'collection_interval_hours': self.collection_interval_hours,
            'rate_limit_seconds': self.rate_limit_seconds,
            'enable_ocr': self.enable_ocr,
            'ocr_engine': self.ocr_engine,
            'database_path': self.database_path,
        }


@dataclass
class CollectionJob:
    job_id: str
    source: CollectionSource
    language: str
    status: str = "pending"
    texts_collected: int = 0
    errors: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'source': self.source.value,
            'language': self.language,
            'status': self.status,
            'texts_collected': self.texts_collected,
            'errors': self.errors,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
        }


class CollectionDatabase:
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS collected_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT,
                    author TEXT,
                    language TEXT,
                    content TEXT,
                    url TEXT,
                    period TEXT,
                    genre TEXT,
                    word_count INTEGER,
                    metadata TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, source_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS collection_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    language TEXT,
                    status TEXT DEFAULT 'pending',
                    texts_collected INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS collection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    texts_collected INTEGER DEFAULT 0,
                    bytes_collected INTEGER DEFAULT 0,
                    sources_queried INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    ocr_pages INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_texts_source ON collected_texts(source)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_texts_language ON collected_texts(language)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_texts_period ON collected_texts(period)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON collection_jobs(status)')
            
            conn.commit()
    
    def save_text(self, text) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO collected_texts
                    (source, source_id, title, author, language, content, url, period, genre, word_count, metadata, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    text.source,
                    text.source_id,
                    text.title,
                    text.author,
                    text.language,
                    text.content,
                    text.url,
                    text.period,
                    text.genre,
                    text.word_count,
                    json.dumps(text.metadata),
                    text.collected_at.isoformat(),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving text: {e}")
            return False
    
    def get_text_count(self, source: Optional[str] = None, language: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM collected_texts WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        if language:
            query += " AND language = ?"
            params.append(language)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
    def get_texts(
        self,
        source: Optional[str] = None,
        language: Optional[str] = None,
        period: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM collected_texts WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        if language:
            query += " AND language = ?"
            params.append(language)
        if period:
            query += " AND period = ?"
            params.append(period)
        
        query += " ORDER BY collected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def text_exists(self, source: str, source_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM collected_texts WHERE source = ? AND source_id = ?",
                (source, source_id)
            )
            return cursor.fetchone() is not None
    
    def save_job(self, job: CollectionJob) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO collection_jobs
                    (job_id, source, language, status, texts_collected, errors, started_at, completed_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job.job_id,
                    job.source.value,
                    job.language,
                    job.status,
                    job.texts_collected,
                    job.errors,
                    job.started_at.isoformat() if job.started_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    json.dumps(job.metadata),
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving job: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total_texts = conn.execute("SELECT COUNT(*) FROM collected_texts").fetchone()[0]
            total_words = conn.execute("SELECT SUM(word_count) FROM collected_texts").fetchone()[0] or 0
            
            sources = conn.execute(
                "SELECT source, COUNT(*) as count FROM collected_texts GROUP BY source"
            ).fetchall()
            
            languages = conn.execute(
                "SELECT language, COUNT(*) as count FROM collected_texts GROUP BY language"
            ).fetchall()
            
            periods = conn.execute(
                "SELECT period, COUNT(*) as count FROM collected_texts GROUP BY period"
            ).fetchall()
            
            return {
                'total_texts': total_texts,
                'total_words': total_words,
                'by_source': {s[0]: s[1] for s in sources},
                'by_language': {l[0]: l[1] for l in languages},
                'by_period': {p[0]: p[1] for p in periods},
            }
    
    def update_daily_stats(self, texts: int = 0, bytes_: int = 0, sources: int = 0, errors: int = 0, ocr: int = 0):
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO collection_stats (date, texts_collected, bytes_collected, sources_queried, errors, ocr_pages)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    texts_collected = texts_collected + excluded.texts_collected,
                    bytes_collected = bytes_collected + excluded.bytes_collected,
                    sources_queried = sources_queried + excluded.sources_queried,
                    errors = errors + excluded.errors,
                    ocr_pages = ocr_pages + excluded.ocr_pages
            ''', (today, texts, bytes_, sources, errors, ocr))
            conn.commit()


class CollectionAgent:
    
    def __init__(self, config: Optional[CollectionConfig] = None):
        self.config = config or CollectionConfig()
        self.db = CollectionDatabase(self.config.database_path)
        
        self.status = CollectionStatus.IDLE
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._job_queue: queue.Queue = queue.Queue()
        self._callbacks: List[Callable] = []
        
        self._collectors = {}
        self._ocr_engine = None
        
        self.stats = {
            'texts_collected': 0,
            'errors': 0,
            'last_collection': None,
            'uptime_start': None,
        }
        
        self._init_collectors()
    
    def _init_collectors(self):
        from hlp_collection.sources import (
            PerseusCollector,
            First1KGreekCollector,
            InternetArchiveCollector,
            ByzantineTextCollector,
            GutenbergCollector,
        )
        
        self._collectors = {
            CollectionSource.PERSEUS: PerseusCollector(rate_limit=self.config.rate_limit_seconds),
            CollectionSource.FIRST1KGREEK: First1KGreekCollector(rate_limit=self.config.rate_limit_seconds),
            CollectionSource.INTERNET_ARCHIVE: InternetArchiveCollector(rate_limit=self.config.rate_limit_seconds),
            CollectionSource.BYZANTINE: ByzantineTextCollector(rate_limit=self.config.rate_limit_seconds),
            CollectionSource.GUTENBERG: GutenbergCollector(rate_limit=self.config.rate_limit_seconds),
        }
        
        if self.config.enable_ocr:
            try:
                from hlp_collection.ocr_engine import OCRFactory
                self._ocr_engine = OCRFactory.create(self.config.ocr_engine)
            except Exception as e:
                logger.warning(f"OCR engine not available: {e}")
    
    def start(self):
        if self._running:
            logger.warning("Collection agent already running")
            return
        
        self._running = True
        self._paused = False
        self.status = CollectionStatus.RUNNING
        self.stats['uptime_start'] = datetime.now()
        
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        
        logger.info("Collection agent started")
    
    def stop(self):
        self._running = False
        self.status = CollectionStatus.STOPPED
        
        if self._thread:
            self._thread.join(timeout=10)
        
        logger.info("Collection agent stopped")
    
    def pause(self):
        self._paused = True
        self.status = CollectionStatus.PAUSED
        logger.info("Collection agent paused")
    
    def resume(self):
        self._paused = False
        self.status = CollectionStatus.RUNNING
        logger.info("Collection agent resumed")
    
    def _collection_loop(self):
        while self._running:
            if self._paused:
                time.sleep(1)
                continue
            
            try:
                self._run_collection_cycle()
            except Exception as e:
                logger.error(f"Error in collection cycle: {e}")
                self.stats['errors'] += 1
            
            interval_seconds = self.config.collection_interval_hours * 3600
            sleep_end = time.time() + interval_seconds
            
            while time.time() < sleep_end and self._running:
                if self._paused:
                    break
                time.sleep(60)
    
    def _run_collection_cycle(self):
        logger.info("Starting collection cycle")
        
        for source in self.config.sources:
            if not self._running or self._paused:
                break
            
            job = CollectionJob(
                job_id=f"{source.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source=source,
                language=','.join(self.config.languages),
                status='running',
                started_at=datetime.now(),
            )
            
            self.db.save_job(job)
            
            try:
                collector = self._collectors.get(source)
                if not collector:
                    logger.warning(f"No collector for source: {source}")
                    continue
                
                for language in self.config.languages:
                    if not self._running or self._paused:
                        break
                    
                    logger.info(f"Collecting from {source.value} for language {language}")
                    
                    for text in collector.collect(
                        language=language,
                        max_texts=self.config.max_texts_per_source
                    ):
                        if not self._running or self._paused:
                            break
                        
                        if not self.db.text_exists(text.source, text.source_id):
                            if self.db.save_text(text):
                                job.texts_collected += 1
                                self.stats['texts_collected'] += 1
                                self.db.update_daily_stats(texts=1, bytes_=len(text.content))
                                
                                for callback in self._callbacks:
                                    try:
                                        callback('text_collected', text)
                                    except Exception:
                                        pass
                
                job.status = 'completed'
                job.completed_at = datetime.now()
                
            except Exception as e:
                logger.error(f"Error collecting from {source.value}: {e}")
                job.status = 'failed'
                job.errors += 1
                job.metadata['error'] = str(e)
            
            self.db.save_job(job)
        
        self.stats['last_collection'] = datetime.now()
        logger.info(f"Collection cycle completed. Total texts: {self.stats['texts_collected']}")
    
    def collect_now(self, source: Optional[CollectionSource] = None, language: Optional[str] = None):
        if source:
            sources = [source]
        else:
            sources = self.config.sources
        
        languages = [language] if language else self.config.languages
        
        for src in sources:
            collector = self._collectors.get(src)
            if not collector:
                continue
            
            for lang in languages:
                for text in collector.collect(language=lang, max_texts=self.config.max_texts_per_source):
                    if not self.db.text_exists(text.source, text.source_id):
                        self.db.save_text(text)
                        self.stats['texts_collected'] += 1
                        yield text
    
    def process_ocr(self, image_path: str, language: str = "grc"):
        if not self._ocr_engine:
            logger.error("OCR engine not initialized")
            return None
        
        from hlp_collection.ocr_engine import OCRConfig
        
        self._ocr_engine.config = OCRConfig.for_language(language)
        
        if not self._ocr_engine._initialized:
            self._ocr_engine.initialize()
        
        result = self._ocr_engine.process_image(image_path)
        
        if result and result.text:
            self.db.update_daily_stats(ocr=1)
        
        return result
    
    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'running': self._running,
            'paused': self._paused,
            'stats': self.stats,
            'db_stats': self.db.get_stats(),
            'config': self.config.to_dict(),
        }
    
    def get_collected_texts(
        self,
        source: Optional[str] = None,
        language: Optional[str] = None,
        period: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        return self.db.get_texts(
            source=source,
            language=language,
            period=period,
            limit=limit,
            offset=offset
        )


def create_collection_agent(
    sources: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    database_path: str = "data/collected_texts.db"
) -> CollectionAgent:
    
    source_map = {
        'perseus': CollectionSource.PERSEUS,
        'first1kgreek': CollectionSource.FIRST1KGREEK,
        'internet_archive': CollectionSource.INTERNET_ARCHIVE,
        'byzantine': CollectionSource.BYZANTINE,
        'gutenberg': CollectionSource.GUTENBERG,
    }
    
    config = CollectionConfig(
        sources=[source_map.get(s, CollectionSource.PERSEUS) for s in (sources or ['perseus', 'first1kgreek', 'gutenberg', 'byzantine'])],
        languages=languages or ['grc', 'lat', 'en'],
        database_path=database_path,
    )
    
    return CollectionAgent(config)
