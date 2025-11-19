"""
═══════════════════════════════════════════════════════════════════════════
UNIFIED AI CORPUS PLATFORM
Automatic Scraping, Parsing, and Annotation System
═══════════════════════════════════════════════════════════════════════════

Super-powerful AI corpus platform that combines:
1. Automatic text scraping from multiple sources (24/7)
2. Automatic parsing and preprocessing
3. Automatic annotation with multiple AI models
4. Real-time monitoring and management
5. Multi-format export and analysis

Author: Nikolaos Lavidas, NKUA
Institution: National and Kapodistrian University of Athens (NKUA)
Funding: Hellenic Foundation for Research and Innovation (HFRI)
Version: 1.0.0
Date: November 9, 2025
═══════════════════════════════════════════════════════════════════════════
"""

# Load local models configuration (use Z:\models\ - no re-downloads)
try:
    import local_models_config
except ImportError:
    pass  # Fall back to default model locations if config not available

import asyncio
import logging
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corpus_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of corpus items through the pipeline"""
    DISCOVERED = "discovered"      # Found by scraper
    SCRAPED = "scraped"            # Downloaded successfully
    PARSED = "parsed"              # Preprocessed
    ANNOTATED = "annotated"        # Fully annotated
    FAILED = "failed"              # Processing failed
    QUEUED = "queued"              # Waiting in queue


class SourceType(Enum):
    """Types of text sources"""
    PERSEUS = "perseus"
    GITHUB = "github"
    ARCHIVE_ORG = "archive_org"
    FIRST1K_GREEK = "first1k_greek"
    PROJECT_GUTENBERG = "gutenberg"
    CUSTOM_URL = "custom_url"
    LOCAL_FILE = "local_file"
    API = "api"
    RSS = "rss"


class AnnotationType(Enum):
    """Types of annotation available"""
    STANZA = "stanza"              # Stanford Stanza
    MULTI_AI = "multi_ai"          # Multiple AI models ensemble
    PROIEL = "proiel"              # PROIEL format
    TREEBANK = "treebank"          # Treebank conversion
    VALENCY = "valency"            # Valency extraction


@dataclass
class CorpusItem:
    """Represents a text in the corpus pipeline"""
    id: Optional[int]
    url: str
    source_type: str
    language: Optional[str]
    status: str
    priority: int
    content_hash: Optional[str]
    raw_path: Optional[str]
    parsed_path: Optional[str]
    annotated_path: Optional[str]
    metadata: str  # JSON string
    created_at: str
    scraped_at: Optional[str]
    parsed_at: Optional[str]
    annotated_at: Optional[str]
    error_message: Optional[str]


class UnifiedCorpusDatabase:
    """Unified database for entire corpus pipeline"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main corpus items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                language TEXT,
                status TEXT DEFAULT 'discovered',
                priority INTEGER DEFAULT 5,
                content_hash TEXT,
                raw_path TEXT,
                parsed_path TEXT,
                annotated_path TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                scraped_at TEXT,
                parsed_at TEXT,
                annotated_at TEXT,
                error_message TEXT
            )
        """)
        
        # Annotations table (supports multiple annotation types per item)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                annotation_type TEXT NOT NULL,
                annotation_data TEXT,
                file_path TEXT,
                created_at TEXT NOT NULL,
                processing_time REAL,
                model_version TEXT,
                FOREIGN KEY (item_id) REFERENCES corpus_items (id)
            )
        """)
        
        # Processing pipeline status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_details TEXT,
                FOREIGN KEY (item_id) REFERENCES corpus_items (id)
            )
        """)
        
        # Sources configuration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT UNIQUE NOT NULL,
                source_type TEXT NOT NULL,
                base_url TEXT,
                enabled INTEGER DEFAULT 1,
                check_interval INTEGER DEFAULT 3600,
                last_check TEXT,
                config TEXT
            )
        """)
        
        # Statistics and metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_items INTEGER,
                scraped_items INTEGER,
                parsed_items INTEGER,
                annotated_items INTEGER,
                failed_items INTEGER,
                processing_rate REAL,
                avg_scrape_time REAL,
                avg_parse_time REAL,
                avg_annotation_time REAL
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON corpus_items(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON corpus_items(source_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_language ON corpus_items(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON corpus_items(priority DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_item_annotations ON annotations(item_id)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def add_item(self, url: str, source_type: str, metadata: Dict = None, 
                 priority: int = 5, language: str = None) -> int:
        """Add new corpus item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata or {})
        created_at = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO corpus_items 
                (url, source_type, language, priority, metadata, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'discovered')
            """, (url, source_type, language, priority, metadata_json, created_at))
            
            item_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Added corpus item {item_id}: {url}")
            return item_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Item already exists: {url}")
            cursor.execute("SELECT id FROM corpus_items WHERE url = ?", (url,))
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def update_status(self, item_id: int, status: str, error_message: str = None):
        """Update item status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE corpus_items 
            SET status = ?, error_message = ?
            WHERE id = ?
        """, (status, error_message, item_id))
        
        conn.commit()
        conn.close()
    
    def get_items_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Get items with specific status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM corpus_items 
            WHERE status = ?
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (status, limit))
        
        items = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return items
    
    def add_annotation(self, item_id: int, annotation_type: str, 
                      annotation_data: Dict, file_path: str = None,
                      processing_time: float = None, model_version: str = None):
        """Add annotation for an item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO annotations 
            (item_id, annotation_type, annotation_data, file_path, 
             created_at, processing_time, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (item_id, annotation_type, json.dumps(annotation_data), 
              file_path, datetime.now().isoformat(), processing_time, model_version))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get current platform statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM corpus_items 
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Count by source
        cursor.execute("""
            SELECT source_type, COUNT(*) as count 
            FROM corpus_items 
            GROUP BY source_type
        """)
        source_counts = dict(cursor.fetchall())
        
        # Count by language
        cursor.execute("""
            SELECT language, COUNT(*) as count 
            FROM corpus_items 
            WHERE language IS NOT NULL
            GROUP BY language
        """)
        language_counts = dict(cursor.fetchall())
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM corpus_items")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_items': total,
            'status_counts': status_counts,
            'source_counts': source_counts,
            'language_counts': language_counts,
            'timestamp': datetime.now().isoformat()
        }


class AutomaticScraper:
    """Automatic text scraping component"""
    
    def __init__(self, db: UnifiedCorpusDatabase):
        self.db = db
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def scrape_url(self, url: str, item_id: int) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Scrape content from URL
        Returns: (success, content, error_message)
        """
        try:
            await self.init_session()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Save to file
                    content_hash = hashlib.sha256(content.encode()).hexdigest()
                    file_path = Path(f"data/raw/{item_id}_{content_hash[:8]}.txt")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"Scraped item {item_id}: {len(content)} chars")
                    return True, str(file_path), None
                else:
                    error = f"HTTP {response.status}"
                    logger.error(f"Scraping failed for item {item_id}: {error}")
                    return False, None, error
                    
        except Exception as e:
            error = str(e)
            logger.error(f"Scraping error for item {item_id}: {error}")
            return False, None, error
    
    async def process_queue(self, batch_size: int = 10):
        """Process scraping queue"""
        items = self.db.get_items_by_status('discovered', limit=batch_size)
        
        if not items:
            return 0
        
        logger.info(f"Processing {len(items)} items for scraping")
        
        tasks = []
        for item in items:
            self.db.update_status(item['id'], 'queued')
            tasks.append(self.scrape_item(item))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return len([r for r in results if r and not isinstance(r, Exception)])
    
    async def scrape_item(self, item: Dict) -> bool:
        """Scrape single item"""
        success, file_path, error = await self.scrape_url(item['url'], item['id'])
        
        if success:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE corpus_items 
                SET status = 'scraped', raw_path = ?, scraped_at = ?,
                    content_hash = ?
                WHERE id = ?
            """, (file_path, datetime.now().isoformat(), 
                  hashlib.sha256(file_path.encode()).hexdigest()[:16], 
                  item['id']))
            conn.commit()
            conn.close()
        else:
            self.db.update_status(item['id'], 'failed', error)
        
        return success


class AutomaticParser:
    """Automatic parsing and preprocessing component"""
    
    def __init__(self, db: UnifiedCorpusDatabase):
        self.db = db
    
    def parse_text(self, file_path: str, item_id: int, language: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parse and preprocess text
        Returns: (success, parsed_file_path, error_message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic preprocessing
            # Remove extra whitespace
            lines = [line.strip() for line in content.split('\n')]
            lines = [line for line in lines if line]
            parsed_content = '\n'.join(lines)
            
            # Detect language if not provided
            if not language:
                language = self.detect_language(parsed_content)
            
            # Save parsed version
            parsed_path = Path(f"data/parsed/{item_id}_parsed.txt")
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(parsed_path, 'w', encoding='utf-8') as f:
                f.write(parsed_content)
            
            logger.info(f"Parsed item {item_id}: {len(lines)} lines")
            return True, str(parsed_path), None
            
        except Exception as e:
            error = str(e)
            logger.error(f"Parsing error for item {item_id}: {error}")
            return False, None, error
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Check for Greek characters
        if any(ord(char) >= 0x0370 and ord(char) <= 0x03FF for char in text[:1000]):
            return "grc"  # Ancient Greek
        return "en"  # Default to English
    
    def process_queue(self, batch_size: int = 20) -> int:
        """Process parsing queue"""
        items = self.db.get_items_by_status('scraped', limit=batch_size)
        
        if not items:
            return 0
        
        logger.info(f"Processing {len(items)} items for parsing")
        
        processed = 0
        for item in items:
            self.db.update_status(item['id'], 'queued')
            success, parsed_path, error = self.parse_text(
                item['raw_path'], item['id'], item.get('language')
            )
            
            if success:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE corpus_items 
                    SET status = 'parsed', parsed_path = ?, parsed_at = ?
                    WHERE id = ?
                """, (parsed_path, datetime.now().isoformat(), item['id']))
                conn.commit()
                conn.close()
                processed += 1
            else:
                self.db.update_status(item['id'], 'failed', error)
        
        return processed


class AutomaticAnnotator:
    """Automatic annotation component with multiple AI models"""
    
    def __init__(self, db: UnifiedCorpusDatabase):
        self.db = db
        self.stanza_available = False
        self.init_annotators()
    
    def init_annotators(self):
        """Initialize available annotators"""
        try:
            import stanza
            self.stanza_available = True
            logger.info("Stanza annotator available")
        except ImportError:
            logger.warning("Stanza not available - install with: pip install stanza")
    
    def annotate_text(self, file_path: str, item_id: int, 
                     language: str = "en") -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Annotate text with available models
        Returns: (success, annotation_data, error_message)
        """
        if not self.stanza_available:
            return False, None, "No annotators available"
        
        try:
            import stanza
            
            # Load appropriate model
            nlp = stanza.Pipeline(
                lang=language, 
                processors='tokenize,pos,lemma,depparse',
                verbose=False
            )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Annotate
            start_time = datetime.now()
            doc = nlp(text)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to JSON
            annotation_data = {
                'sentences': [],
                'metadata': {
                    'language': language,
                    'model': 'stanza',
                    'processing_time': processing_time
                }
            }
            
            for sent in doc.sentences:
                sent_data = {
                    'text': sent.text,
                    'tokens': []
                }
                for token in sent.tokens:
                    for word in token.words:
                        sent_data['tokens'].append({
                            'text': word.text,
                            'lemma': word.lemma,
                            'upos': word.upos,
                            'xpos': word.xpos,
                            'feats': word.feats,
                            'head': word.head,
                            'deprel': word.deprel
                        })
                annotation_data['sentences'].append(sent_data)
            
            # Save annotation
            annotation_path = Path(f"data/annotated/{item_id}_annotated.json")
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Annotated item {item_id}: {len(doc.sentences)} sentences")
            return True, annotation_data, None
            
        except Exception as e:
            error = str(e)
            logger.error(f"Annotation error for item {item_id}: {error}")
            return False, None, error
    
    def process_queue(self, batch_size: int = 5) -> int:
        """Process annotation queue"""
        items = self.db.get_items_by_status('parsed', limit=batch_size)
        
        if not items:
            return 0
        
        logger.info(f"Processing {len(items)} items for annotation")
        
        processed = 0
        for item in items:
            self.db.update_status(item['id'], 'queued')
            
            language = item.get('language') or 'en'
            success, annotation_data, error = self.annotate_text(
                item['parsed_path'], item['id'], language
            )
            
            if success:
                # Update database
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE corpus_items 
                    SET status = 'annotated', annotated_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), item['id']))
                conn.commit()
                conn.close()
                
                # Store annotation
                self.db.add_annotation(
                    item['id'], 
                    'stanza',
                    annotation_data,
                    f"data/annotated/{item['id']}_annotated.json",
                    annotation_data['metadata']['processing_time'],
                    'stanza-1.4'
                )
                processed += 1
            else:
                self.db.update_status(item['id'], 'failed', error)
        
        return processed


class UnifiedCorpusPlatform:
    """Main unified corpus platform orchestrator"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db = UnifiedCorpusDatabase(db_path)
        self.scraper = AutomaticScraper(self.db)
        self.parser = AutomaticParser(self.db)
        self.annotator = AutomaticAnnotator(self.db)
        self.running = False
    
    async def run_pipeline(self, cycles: int = None, cycle_delay: int = 10):
        """
        Run the complete pipeline
        
        Args:
            cycles: Number of cycles to run (None = infinite)
            cycle_delay: Seconds between cycles
        """
        self.running = True
        cycle_count = 0
        
        logger.info("=" * 70)
        logger.info("UNIFIED CORPUS PLATFORM STARTED")
        logger.info("=" * 70)
        
        try:
            while self.running and (cycles is None or cycle_count < cycles):
                cycle_count += 1
                logger.info(f"\n{'=' * 70}")
                logger.info(f"PIPELINE CYCLE {cycle_count}")
                logger.info(f"{'=' * 70}")
                
                # Phase 1: Scraping
                logger.info("\n[Phase 1/3] Scraping new texts...")
                scraped = await self.scraper.process_queue(batch_size=10)
                logger.info(f"✓ Scraped: {scraped} items")
                
                # Phase 2: Parsing
                logger.info("\n[Phase 2/3] Parsing texts...")
                parsed = self.parser.process_queue(batch_size=20)
                logger.info(f"✓ Parsed: {parsed} items")
                
                # Phase 3: Annotation
                logger.info("\n[Phase 3/3] Annotating texts...")
                annotated = self.annotator.process_queue(batch_size=5)
                logger.info(f"✓ Annotated: {annotated} items")
                
                # Show statistics
                stats = self.db.get_statistics()
                logger.info(f"\n{'=' * 70}")
                logger.info("PLATFORM STATISTICS")
                logger.info(f"{'=' * 70}")
                logger.info(f"Total items: {stats['total_items']}")
                logger.info(f"Status breakdown: {stats['status_counts']}")
                logger.info(f"Sources: {stats['source_counts']}")
                logger.info(f"Languages: {stats['language_counts']}")
                
                # Check if work is done
                if scraped == 0 and parsed == 0 and annotated == 0:
                    logger.info("\n✓ No pending items - pipeline idle")
                    if cycles is not None:
                        break
                
                if self.running and (cycles is None or cycle_count < cycles):
                    logger.info(f"\nWaiting {cycle_delay}s before next cycle...")
                    await asyncio.sleep(cycle_delay)
                    
        except KeyboardInterrupt:
            logger.info("\n\nReceived interrupt signal - shutting down...")
            self.running = False
        finally:
            await self.scraper.close_session()
            logger.info("\n" + "=" * 70)
            logger.info("PLATFORM STOPPED")
            logger.info("=" * 70)
    
    def add_source_urls(self, urls: List[str], source_type: str = "custom_url",
                       priority: int = 5, language: str = None):
        """Add multiple URLs to process"""
        added = 0
        for url in urls:
            try:
                self.db.add_item(url, source_type, priority=priority, language=language)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add {url}: {e}")
        
        logger.info(f"Added {added} items to corpus")
        return added
    
    def show_status(self):
        """Show current platform status"""
        stats = self.db.get_statistics()
        
        print("\n" + "=" * 70)
        print("UNIFIED CORPUS PLATFORM STATUS")
        print("=" * 70)
        print(f"\nTotal Items: {stats['total_items']}")
        print(f"\nStatus Breakdown:")
        for status, count in stats['status_counts'].items():
            print(f"  {status:15s}: {count:5d}")
        print(f"\nSource Types:")
        for source, count in stats['source_counts'].items():
            print(f"  {source:15s}: {count:5d}")
        print(f"\nLanguages:")
        for lang, count in stats['language_counts'].items():
            print(f"  {lang:15s}: {count:5d}")
        print("\n" + "=" * 70 + "\n")


# CLI Interface
async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified AI Corpus Platform - Automatic Scraping, Parsing, and Annotation"
    )
    parser.add_argument('--db', default='corpus_platform.db', help='Database file path')
    parser.add_argument('--add-urls', nargs='+', help='Add URLs to process')
    parser.add_argument('--source-type', default='custom_url', help='Source type for URLs')
    parser.add_argument('--language', help='Language code (grc, en, etc.)')
    parser.add_argument('--priority', type=int, default=5, help='Priority (1-10)')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    parser.add_argument('--cycles', type=int, help='Number of processing cycles (default: infinite)')
    parser.add_argument('--delay', type=int, default=10, help='Seconds between cycles')
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = UnifiedCorpusPlatform(args.db)
    
    # Add URLs if provided
    if args.add_urls:
        platform.add_source_urls(
            args.add_urls,
            source_type=args.source_type,
            priority=args.priority,
            language=args.language
        )
    
    # Show status if requested
    if args.status:
        platform.show_status()
        return
    
    # Run pipeline
    await platform.run_pipeline(cycles=args.cycles, cycle_delay=args.delay)


if __name__ == "__main__":
    asyncio.run(main())
