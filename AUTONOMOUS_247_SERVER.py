#!/usr/bin/env python3
"""
AUTONOMOUS 24/7 SERVER SYSTEM
Runs continuously, collects data, processes corpus, never crashes

Features:
- Continuous data collection from multiple sources
- Automatic error recovery
- Periodic valency extraction
- Database optimization
- Health monitoring
- Log rotation
- Graceful shutdown handling

Usage:
    python3 AUTONOMOUS_247_SERVER.py

Or as systemd service:
    systemctl start corpus-collector
"""

import os
import sys
import time
import signal
import sqlite3
import json
import logging
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading
import queue

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration"""
    # Paths
    DATA_DIR = Path("/root/corpus_platform/data")
    DB_PATH = DATA_DIR / "corpus_platform.db"
    CACHE_DIR = DATA_DIR / "cache"
    LOG_DIR = DATA_DIR / "logs"
    
    # Timing (seconds)
    COLLECTION_INTERVAL = 3600  # 1 hour between collection cycles
    VALENCY_INTERVAL = 7200     # 2 hours between valency extraction
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes between health checks
    DB_OPTIMIZE_INTERVAL = 86400  # 24 hours between DB optimization
    
    # Limits
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 120
    MAX_LOG_SIZE_MB = 100
    
    # Sources
    UD_TREEBANKS = {
        "grc_proiel": {
            "name": "Ancient Greek PROIEL",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master",
            "period": "classical",
            "language": "grc"
        },
        "grc_perseus": {
            "name": "Ancient Greek Perseus", 
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master",
            "period": "classical",
            "language": "grc"
        },
        "el_gdt": {
            "name": "Modern Greek GDT",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master",
            "period": "modern",
            "language": "el"
        },
        "la_proiel": {
            "name": "Latin PROIEL",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master",
            "period": "classical", 
            "language": "la"
        },
        "la_ittb": {
            "name": "Latin ITTB",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master",
            "period": "medieval",
            "language": "la"
        },
        "la_llct": {
            "name": "Latin LLCT",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/master",
            "period": "late_antique",
            "language": "la"
        },
        "got_proiel": {
            "name": "Gothic PROIEL",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/master",
            "period": "medieval",
            "language": "got"
        },
        "cu_proiel": {
            "name": "Old Church Slavonic PROIEL",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/master",
            "period": "medieval",
            "language": "cu"
        },
        "orv_rnc": {
            "name": "Old Russian RNC",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Russian-RNC/master",
            "period": "medieval",
            "language": "orv"
        },
        "sa_vedic": {
            "name": "Sanskrit Vedic",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-Vedic/master",
            "period": "archaic",
            "language": "sa"
        },
        "xcl_caval": {
            "name": "Classical Armenian CAVaL",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Classical_Armenian-CAVaL/master",
            "period": "classical",
            "language": "xcl"
        },
        "hit_hittb": {
            "name": "Hittite HitTB",
            "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Hittite-HitTB/master",
            "period": "archaic",
            "language": "hit"
        },
    }

# Create directories
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Setup logging with rotation"""
    log_file = Config.LOG_DIR / f"autonomous_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Rotate old logs
    for old_log in Config.LOG_DIR.glob("autonomous_*.log"):
        if old_log != log_file:
            size_mb = old_log.stat().st_size / (1024 * 1024)
            age_days = (datetime.now() - datetime.fromtimestamp(old_log.stat().st_mtime)).days
            if size_mb > Config.MAX_LOG_SIZE_MB or age_days > 7:
                old_log.unlink()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Thread-safe database manager"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Documents table
            c.execute("""CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                language TEXT DEFAULT 'grc',
                source TEXT,
                source_url TEXT,
                content_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Sentences table
            c.execute("""CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                sentence_index INTEGER,
                text TEXT,
                sent_id TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )""")
            
            # Tokens table
            c.execute("""CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER,
                token_index INTEGER,
                form TEXT,
                lemma TEXT,
                upos TEXT,
                xpos TEXT,
                feats TEXT,
                head INTEGER,
                deprel TEXT,
                deps TEXT,
                misc TEXT,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )""")
            
            # Valency frames table
            c.execute("""CREATE TABLE IF NOT EXISTS valency_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                period TEXT,
                language TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lemma, pattern, period, language)
            )""")
            
            # Collection log table
            c.execute("""CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                status TEXT,
                documents_added INTEGER DEFAULT 0,
                sentences_added INTEGER DEFAULT 0,
                tokens_added INTEGER DEFAULT 0,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # System stats table
            c.execute("""CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE,
                stat_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            
            # Indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(document_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tok_sent ON tokens(sentence_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tok_lemma ON tokens(lemma)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tok_upos ON tokens(upos)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_doc_lang ON documents(language)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_doc_period ON documents(period)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_val_lemma ON valency_frames(lemma)")
            
            conn.commit()
            conn.close()
            logger.info("Database schema initialized")
    
    def execute(self, query: str, params: tuple = ()) -> Optional[int]:
        """Execute a query and return lastrowid"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.execute(query, params)
                conn.commit()
                return c.lastrowid
            except sqlite3.IntegrityError:
                return None
            finally:
                conn.close()
    
    def executemany(self, query: str, params_list: List[tuple]):
        """Execute many queries"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.executemany(query, params_list)
                conn.commit()
            finally:
                conn.close()
    
    def fetchall(self, query: str, params: tuple = ()) -> List[tuple]:
        """Fetch all results"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.execute(query, params)
                return c.fetchall()
            finally:
                conn.close()
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[tuple]:
        """Fetch one result"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            try:
                c.execute(query, params)
                return c.fetchone()
            finally:
                conn.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        stats['documents'] = self.fetchone("SELECT COUNT(*) FROM documents")[0]
        stats['sentences'] = self.fetchone("SELECT COUNT(*) FROM sentences")[0]
        stats['tokens'] = self.fetchone("SELECT COUNT(*) FROM tokens")[0]
        stats['valency_frames'] = self.fetchone("SELECT COUNT(*) FROM valency_frames")[0]
        
        # By language
        rows = self.fetchall("SELECT language, COUNT(*) FROM documents GROUP BY language")
        stats['by_language'] = {r[0]: r[1] for r in rows}
        
        # By period
        rows = self.fetchall("SELECT period, COUNT(*) FROM documents GROUP BY period")
        stats['by_period'] = {r[0]: r[1] for r in rows}
        
        return stats
    
    def optimize(self):
        """Optimize database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.close()
            logger.info("Database optimized")

# =============================================================================
# CONLL-U PARSER
# =============================================================================

def parse_conllu(content: str) -> List[Dict]:
    """Parse CoNLL-U content into sentences"""
    sentences = []
    current = {"id": "", "text": "", "tokens": []}
    
    for line in content.split('\n'):
        line = line.strip()
        
        if not line:
            if current["tokens"]:
                sentences.append(current)
                current = {"id": "", "text": "", "tokens": []}
        elif line.startswith('# sent_id'):
            current["id"] = line.split('=', 1)[-1].strip()
        elif line.startswith('# text'):
            current["text"] = line.split('=', 1)[-1].strip()
        elif not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 10:
                tok_id = parts[0]
                # Skip multiword tokens and empty nodes
                if '-' in tok_id or '.' in tok_id:
                    continue
                
                current["tokens"].append({
                    "id": tok_id,
                    "form": parts[1],
                    "lemma": parts[2] if parts[2] != '_' else parts[1],
                    "upos": parts[3] if parts[3] != '_' else '',
                    "xpos": parts[4] if parts[4] != '_' else '',
                    "feats": parts[5] if parts[5] != '_' else '',
                    "head": parts[6] if parts[6] != '_' else '0',
                    "deprel": parts[7] if parts[7] != '_' else '',
                    "deps": parts[8] if parts[8] != '_' else '',
                    "misc": parts[9] if parts[9] != '_' else ''
                })
    
    # Don't forget last sentence
    if current["tokens"]:
        sentences.append(current)
    
    return sentences

# =============================================================================
# DATA COLLECTOR
# =============================================================================

class DataCollector:
    """Collects data from various sources"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """Initialize requests session"""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'CorpusPlatform/1.0 (Academic Research)'
            })
        except ImportError:
            logger.warning("requests not available, using urllib")
            self.session = None
    
    def download(self, url: str, cache_path: Path) -> str:
        """Download file with caching"""
        # Check cache first
        if cache_path.exists():
            # Check if cache is fresh (less than 24 hours old)
            age = time.time() - cache_path.stat().st_mtime
            if age < 86400:
                return cache_path.read_text(encoding='utf-8')
        
        # Download
        content = ""
        for attempt in range(Config.MAX_RETRIES):
            try:
                if self.session:
                    response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                    if response.status_code == 200:
                        content = response.text
                        break
                else:
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=Config.REQUEST_TIMEOUT) as resp:
                        content = resp.read().decode('utf-8')
                        break
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Cache if successful
        if content:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content, encoding='utf-8')
        
        return content
    
    def collect_ud_treebank(self, tb_id: str, tb_info: Dict) -> Tuple[int, int, int]:
        """Collect a single UD treebank"""
        docs_added = 0
        sents_added = 0
        toks_added = 0
        
        for split in ["train", "dev", "test"]:
            filename = f"{tb_id}-ud-{split}.conllu"
            url = f"{tb_info['url']}/{filename}"
            cache_path = Config.CACHE_DIR / f"{tb_id}_{split}.conllu"
            
            logger.info(f"  Collecting {tb_id} {split}...")
            
            content = self.download(url, cache_path)
            if not content:
                logger.warning(f"  No content for {tb_id} {split}")
                continue
            
            # Check if already imported (by content hash)
            content_hash = hashlib.md5(content.encode()).hexdigest()
            existing = self.db.fetchone(
                "SELECT id FROM documents WHERE content_hash = ?", 
                (content_hash,)
            )
            if existing:
                logger.info(f"  {tb_id} {split} already imported, skipping")
                continue
            
            # Parse
            sentences = parse_conllu(content)
            if not sentences:
                logger.warning(f"  No sentences parsed from {tb_id} {split}")
                continue
            
            # Insert document
            doc_id = self.db.execute(
                """INSERT INTO documents (title, author, period, language, source, source_url, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (f"{tb_info['name']} - {split}", "Universal Dependencies",
                 tb_info['period'], tb_info['language'], f"UD_{tb_id}", url, content_hash)
            )
            
            if not doc_id:
                continue
            
            docs_added += 1
            
            # Insert sentences and tokens in batches
            for sent_idx, sent in enumerate(sentences):
                sent_id = self.db.execute(
                    "INSERT INTO sentences (document_id, sentence_index, text, sent_id) VALUES (?, ?, ?, ?)",
                    (doc_id, sent_idx, sent["text"], sent["id"])
                )
                
                if sent_id:
                    sents_added += 1
                    
                    for tok in sent["tokens"]:
                        self.db.execute(
                            """INSERT INTO tokens (sentence_id, token_index, form, lemma, upos, xpos, feats, head, deprel, deps, misc)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (sent_id, int(tok["id"]), tok["form"], tok["lemma"],
                             tok["upos"], tok["xpos"], tok["feats"],
                             int(tok["head"]) if tok["head"].isdigit() else 0,
                             tok["deprel"], tok["deps"], tok["misc"])
                        )
                        toks_added += 1
            
            logger.info(f"  Added {len(sentences)} sentences from {tb_id} {split}")
        
        return docs_added, sents_added, toks_added
    
    def collect_all(self) -> Dict:
        """Collect from all sources"""
        start_time = datetime.now()
        total_docs = 0
        total_sents = 0
        total_toks = 0
        
        logger.info("=" * 60)
        logger.info("Starting collection cycle")
        logger.info("=" * 60)
        
        for tb_id, tb_info in Config.UD_TREEBANKS.items():
            try:
                logger.info(f"Collecting {tb_info['name']}...")
                docs, sents, toks = self.collect_ud_treebank(tb_id, tb_info)
                total_docs += docs
                total_sents += sents
                total_toks += toks
                
                # Log progress
                self.db.execute(
                    """INSERT INTO collection_log (source, status, documents_added, sentences_added, tokens_added, started_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (tb_id, 'success', docs, sents, toks, start_time.isoformat())
                )
                
            except Exception as e:
                logger.error(f"Error collecting {tb_id}: {e}")
                self.db.execute(
                    """INSERT INTO collection_log (source, status, error_message, started_at)
                       VALUES (?, ?, ?, ?)""",
                    (tb_id, 'error', str(e), start_time.isoformat())
                )
        
        logger.info(f"Collection complete: {total_docs} docs, {total_sents} sents, {total_toks} toks")
        
        return {
            'documents': total_docs,
            'sentences': total_sents,
            'tokens': total_toks,
            'duration': (datetime.now() - start_time).total_seconds()
        }

# =============================================================================
# VALENCY EXTRACTOR
# =============================================================================

class ValencyExtractor:
    """Extract valency frames from corpus"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def extract(self) -> int:
        """Extract valency frames from all verbs"""
        logger.info("Extracting valency frames...")
        
        # Get verb-dependent pairs with period and language
        rows = self.db.fetchall("""
            SELECT t1.lemma, t2.deprel, d.period, d.language, t1.sentence_id
            FROM tokens t1
            JOIN tokens t2 ON t1.sentence_id = t2.sentence_id AND t2.head = t1.token_index
            JOIN sentences s ON t1.sentence_id = s.id
            JOIN documents d ON s.document_id = d.id
            WHERE t1.upos = 'VERB' AND t1.lemma != '' AND t1.lemma != '_'
        """)
        
        # Aggregate frames
        frames = defaultdict(lambda: {'count': 0, 'examples': set()})
        
        for lemma, deprel, period, language, sent_id in rows:
            key = (lemma, deprel, period or 'unknown', language or 'unknown')
            frames[key]['count'] += 1
            if len(frames[key]['examples']) < 3:
                frames[key]['examples'].add(str(sent_id))
        
        # Build full patterns per verb
        verb_patterns = defaultdict(lambda: defaultdict(set))
        for (lemma, deprel, period, language), data in frames.items():
            verb_patterns[(lemma, period, language)].add(deprel)
        
        # Insert frames
        count = 0
        for (lemma, period, language), deprels in verb_patterns.items():
            pattern = "+".join(sorted(deprels))
            
            # Get frequency (sum of all deprel counts for this verb)
            freq = sum(
                frames[(lemma, d, period, language)]['count'] 
                for d in deprels 
                if (lemma, d, period, language) in frames
            )
            
            result = self.db.execute(
                """INSERT OR REPLACE INTO valency_frames (lemma, pattern, frequency, period, language)
                   VALUES (?, ?, ?, ?, ?)""",
                (lemma, pattern, freq, period, language)
            )
            if result:
                count += 1
        
        logger.info(f"Extracted {count} valency frames")
        return count

# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """Monitor system health"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.start_time = datetime.now()
    
    def check(self) -> Dict:
        """Run health check"""
        health = {
            'status': 'healthy',
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        # Database stats
        try:
            stats = self.db.get_stats()
            health['database'] = stats
        except Exception as e:
            health['status'] = 'degraded'
            health['database_error'] = str(e)
        
        # Disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(Config.DATA_DIR)
            health['disk'] = {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'percent_used': (used / total) * 100
            }
            if health['disk']['percent_used'] > 90:
                health['status'] = 'warning'
        except:
            pass
        
        # Update stats in DB
        self.db.execute(
            "INSERT OR REPLACE INTO system_stats (stat_name, stat_value) VALUES (?, ?)",
            ('last_health_check', json.dumps(health))
        )
        
        return health
    
    def log_status(self):
        """Log current status"""
        health = self.check()
        
        logger.info("-" * 40)
        logger.info(f"Status: {health['status']}")
        logger.info(f"Uptime: {health['uptime_hours']:.1f} hours")
        
        if 'database' in health:
            db = health['database']
            logger.info(f"Documents: {db.get('documents', 0):,}")
            logger.info(f"Sentences: {db.get('sentences', 0):,}")
            logger.info(f"Tokens: {db.get('tokens', 0):,}")
            logger.info(f"Valency Frames: {db.get('valency_frames', 0):,}")
        
        if 'disk' in health:
            logger.info(f"Disk: {health['disk']['free_gb']:.1f} GB free ({health['disk']['percent_used']:.1f}% used)")
        
        logger.info("-" * 40)

# =============================================================================
# MAIN AUTONOMOUS SYSTEM
# =============================================================================

class AutonomousSystem:
    """Main 24/7 autonomous system"""
    
    def __init__(self):
        self.running = True
        self.db = DatabaseManager(Config.DB_PATH)
        self.collector = DataCollector(self.db)
        self.valency = ValencyExtractor(self.db)
        self.health = HealthMonitor(self.db)
        
        # Timestamps for scheduling
        self.last_collection = datetime.min
        self.last_valency = datetime.min
        self.last_health = datetime.min
        self.last_optimize = datetime.min
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("Shutdown signal received, finishing current task...")
        self.running = False
    
    def run(self):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("AUTONOMOUS 24/7 SYSTEM STARTING")
        logger.info("=" * 60)
        logger.info(f"Data directory: {Config.DATA_DIR}")
        logger.info(f"Database: {Config.DB_PATH}")
        logger.info(f"Collection interval: {Config.COLLECTION_INTERVAL}s")
        logger.info(f"Valency interval: {Config.VALENCY_INTERVAL}s")
        logger.info("=" * 60)
        
        # Initial collection
        self._run_collection()
        self._run_valency()
        self.health.log_status()
        
        # Main loop
        while self.running:
            try:
                now = datetime.now()
                
                # Collection cycle
                if (now - self.last_collection).total_seconds() >= Config.COLLECTION_INTERVAL:
                    self._run_collection()
                
                # Valency extraction
                if (now - self.last_valency).total_seconds() >= Config.VALENCY_INTERVAL:
                    self._run_valency()
                
                # Health check
                if (now - self.last_health).total_seconds() >= Config.HEALTH_CHECK_INTERVAL:
                    self.health.log_status()
                    self.last_health = now
                
                # Database optimization
                if (now - self.last_optimize).total_seconds() >= Config.DB_OPTIMIZE_INTERVAL:
                    self.db.optimize()
                    self.last_optimize = now
                
                # Sleep before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(300)  # Wait 5 minutes before retrying
        
        logger.info("System shutdown complete")
    
    def _run_collection(self):
        """Run collection with error handling"""
        try:
            self.collector.collect_all()
            self.last_collection = datetime.now()
        except Exception as e:
            logger.error(f"Collection error: {e}")
            logger.error(traceback.format_exc())
    
    def _run_valency(self):
        """Run valency extraction with error handling"""
        try:
            self.valency.extract()
            self.last_valency = datetime.now()
        except Exception as e:
            logger.error(f"Valency extraction error: {e}")
            logger.error(traceback.format_exc())

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     AUTONOMOUS 24/7 CORPUS COLLECTION SYSTEM              ║
    ║     Indo-European Historical Linguistics Platform         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    system = AutonomousSystem()
    system.run()
