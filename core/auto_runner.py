#!/usr/bin/env python3
"""
AUTO RUNNER - Executes the complete corpus collection pipeline
This script ACTUALLY RUNS the collection, not just defines it

Run this on the server:
    python3 core/auto_runner.py

Or with systemd service for continuous operation
"""

import os
import sys
import json
import time
import logging
import sqlite3
import requests
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/corpus_auto_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "data_dir": "/root/corpus_platform/data",
    "db_path": "/root/corpus_platform/data/corpus_platform.db",
    "run_interval_hours": 1,
    "max_retries": 3,
    "request_timeout": 120,
    "enable_ud_collection": True,
    "enable_perseus_collection": True,
    "enable_gutenberg_collection": True,
    "enable_morphgnt_collection": True
}

# =============================================================================
# UNIVERSAL DEPENDENCIES SOURCES
# =============================================================================

UD_SOURCES = {
    # Ancient Greek
    "grc_proiel": {
        "name": "Ancient Greek PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PROIEL/master",
        "files": ["grc_proiel-ud-train.conllu", "grc_proiel-ud-dev.conllu", "grc_proiel-ud-test.conllu"],
        "language": "grc",
        "period": "mixed"
    },
    "grc_perseus": {
        "name": "Ancient Greek Perseus",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master",
        "files": ["grc_perseus-ud-train.conllu", "grc_perseus-ud-dev.conllu", "grc_perseus-ud-test.conllu"],
        "language": "grc",
        "period": "classical"
    },
    
    # Modern Greek
    "el_gdt": {
        "name": "Modern Greek GDT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Greek-GDT/master",
        "files": ["el_gdt-ud-train.conllu", "el_gdt-ud-dev.conllu", "el_gdt-ud-test.conllu"],
        "language": "el",
        "period": "modern"
    },
    
    # Latin
    "la_proiel": {
        "name": "Latin PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-PROIEL/master",
        "files": ["la_proiel-ud-train.conllu", "la_proiel-ud-dev.conllu", "la_proiel-ud-test.conllu"],
        "language": "la",
        "period": "classical"
    },
    "la_llct": {
        "name": "Latin LLCT (Medieval)",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/master",
        "files": ["la_llct-ud-train.conllu", "la_llct-ud-dev.conllu", "la_llct-ud-test.conllu"],
        "language": "la",
        "period": "medieval"
    },
    "la_ittb": {
        "name": "Latin ITTB (Thomas Aquinas)",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master",
        "files": ["la_ittb-ud-train.conllu", "la_ittb-ud-dev.conllu", "la_ittb-ud-test.conllu"],
        "language": "la",
        "period": "medieval"
    },
    
    # Gothic (PROIEL family)
    "got_proiel": {
        "name": "Gothic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Gothic-PROIEL/master",
        "files": ["got_proiel-ud-train.conllu", "got_proiel-ud-dev.conllu", "got_proiel-ud-test.conllu"],
        "language": "got",
        "period": "ancient"
    },
    
    # Old Church Slavonic (PROIEL family)
    "cu_proiel": {
        "name": "Old Church Slavonic PROIEL",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_Church_Slavonic-PROIEL/master",
        "files": ["cu_proiel-ud-train.conllu", "cu_proiel-ud-dev.conllu", "cu_proiel-ud-test.conllu"],
        "language": "cu",
        "period": "medieval"
    },
    
    # Old French
    "fro_srcmf": {
        "name": "Old French SRCMF",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/master",
        "files": ["fro_srcmf-ud-train.conllu", "fro_srcmf-ud-dev.conllu", "fro_srcmf-ud-test.conllu"],
        "language": "fro",
        "period": "medieval"
    },
    
    # Old English
    "ang_ycoe": {
        "name": "Old English YCOE",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Old_English-YCOE/master",
        "files": ["ang_ycoe-ud-train.conllu", "ang_ycoe-ud-dev.conllu", "ang_ycoe-ud-test.conllu"],
        "language": "ang",
        "period": "medieval"
    },
    
    # English
    "en_ewt": {
        "name": "English EWT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master",
        "files": ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"],
        "language": "en",
        "period": "modern"
    },
    
    # Romance languages
    "es_ancora": {
        "name": "Spanish AnCora",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master",
        "files": ["es_ancora-ud-train.conllu", "es_ancora-ud-dev.conllu", "es_ancora-ud-test.conllu"],
        "language": "es",
        "period": "modern"
    },
    "fr_gsd": {
        "name": "French GSD",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master",
        "files": ["fr_gsd-ud-train.conllu", "fr_gsd-ud-dev.conllu", "fr_gsd-ud-test.conllu"],
        "language": "fr",
        "period": "modern"
    },
    "it_isdt": {
        "name": "Italian ISDT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master",
        "files": ["it_isdt-ud-train.conllu", "it_isdt-ud-dev.conllu", "it_isdt-ud-test.conllu"],
        "language": "it",
        "period": "modern"
    },
    "pt_bosque": {
        "name": "Portuguese Bosque",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-Bosque/master",
        "files": ["pt_bosque-ud-train.conllu", "pt_bosque-ud-dev.conllu", "pt_bosque-ud-test.conllu"],
        "language": "pt",
        "period": "modern"
    },
    "ro_rrt": {
        "name": "Romanian RRT",
        "url": "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master",
        "files": ["ro_rrt-ud-train.conllu", "ro_rrt-ud-dev.conllu", "ro_rrt-ud-test.conllu"],
        "language": "ro",
        "period": "modern"
    }
}

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

class CorpusDatabase:
    """Database operations for corpus storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                century TEXT,
                genre TEXT,
                source TEXT,
                language TEXT DEFAULT 'grc',
                sentence_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                annotation_status TEXT DEFAULT 'pending',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sentences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                sentence_index INTEGER,
                text TEXT NOT NULL,
                tokens TEXT,
                annotation_status TEXT DEFAULT 'pending',
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT,
                token_index INTEGER,
                form TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                morphology TEXT,
                head INTEGER,
                relation TEXT,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
        """)
        
        # Collection log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                documents_added INTEGER,
                sentences_added INTEGER,
                tokens_added INTEGER,
                status TEXT,
                details TEXT
            )
        """)
        
        # Valency lexicon
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_lexicon (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                arguments TEXT,
                frequency INTEGER DEFAULT 1,
                period TEXT,
                examples TEXT,
                semantic_class TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_period ON documents(period)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_lang ON documents(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tok_sent ON tokens(sentence_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tok_lemma ON tokens(lemma)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def store_conllu_document(self, doc_id: str, source_info: Dict, sentences: List[Dict]) -> Dict:
        """Store a CoNLL-U document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {'sentences': 0, 'tokens': 0}
        
        try:
            # Check if document exists
            cursor.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
            if cursor.fetchone():
                logger.info(f"Document already exists: {doc_id}")
                conn.close()
                return stats
            
            # Insert document
            cursor.execute("""
                INSERT INTO documents (id, title, author, period, genre, source, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                source_info.get('name', doc_id),
                source_info.get('author', 'Various'),
                source_info.get('period', 'unknown'),
                source_info.get('genre', 'mixed'),
                source_info.get('url', ''),
                source_info.get('language', 'grc')
            ))
            
            # Insert sentences and tokens
            for sent_idx, sentence in enumerate(sentences):
                sent_id = f"{doc_id}_s{sent_idx:05d}"
                sent_text = ' '.join(t['form'] for t in sentence['tokens'])
                
                cursor.execute("""
                    INSERT INTO sentences (id, document_id, sentence_index, text, tokens, annotation_status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    sent_id, doc_id, sent_idx, sent_text,
                    json.dumps([t['form'] for t in sentence['tokens']]),
                    'annotated'
                ))
                
                for tok_idx, token in enumerate(sentence['tokens']):
                    cursor.execute("""
                        INSERT INTO tokens (sentence_id, token_index, form, lemma, pos, morphology, head, relation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sent_id, tok_idx,
                        token.get('form', ''),
                        token.get('lemma', ''),
                        token.get('upos', ''),
                        token.get('feats', ''),
                        token.get('head', 0),
                        token.get('deprel', '')
                    ))
                    stats['tokens'] += 1
                
                stats['sentences'] += 1
            
            # Update document counts
            cursor.execute("""
                UPDATE documents SET sentence_count = ?, token_count = ?
                WHERE id = ?
            """, (stats['sentences'], stats['tokens'], doc_id))
            
            conn.commit()
            logger.info(f"Stored {doc_id}: {stats['sentences']} sentences, {stats['tokens']} tokens")
            
        except Exception as e:
            logger.error(f"Error storing document {doc_id}: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        return stats
    
    def log_collection(self, source: str, docs: int, sents: int, toks: int, status: str, details: str = ""):
        """Log a collection run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO collection_log (source, documents_added, sentences_added, tokens_added, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (source, docs, sents, toks, status, details))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(sentence_count) FROM documents")
        result = cursor.fetchone()[0]
        stats['sentences'] = result if result else 0
        
        cursor.execute("SELECT SUM(token_count) FROM documents")
        result = cursor.fetchone()[0]
        stats['tokens'] = result if result else 0
        
        cursor.execute("""
            SELECT period, COUNT(*), SUM(token_count)
            FROM documents WHERE period IS NOT NULL
            GROUP BY period
        """)
        stats['by_period'] = {
            row[0]: {'docs': row[1], 'tokens': row[2] or 0}
            for row in cursor.fetchall()
        }
        
        cursor.execute("""
            SELECT language, COUNT(*), SUM(token_count)
            FROM documents WHERE language IS NOT NULL
            GROUP BY language
        """)
        stats['by_language'] = {
            row[0]: {'docs': row[1], 'tokens': row[2] or 0}
            for row in cursor.fetchall()
        }
        
        conn.close()
        return stats


# =============================================================================
# CONLL-U PARSER
# =============================================================================

def parse_conllu(content: str) -> List[Dict]:
    """Parse CoNLL-U format content"""
    sentences = []
    current_tokens = []
    current_meta = {}
    
    for line in content.split('\n'):
        line = line.strip()
        
        if not line:
            if current_tokens:
                sentences.append({
                    'meta': current_meta,
                    'tokens': current_tokens
                })
            current_tokens = []
            current_meta = {}
            continue
        
        if line.startswith('#'):
            if '=' in line:
                key, value = line[1:].split('=', 1)
                current_meta[key.strip()] = value.strip()
            continue
        
        parts = line.split('\t')
        if len(parts) >= 10:
            # Skip multi-word tokens and empty nodes
            if '-' in parts[0] or '.' in parts[0]:
                continue
            
            try:
                head = int(parts[6]) if parts[6].isdigit() else 0
            except:
                head = 0
            
            token = {
                'id': parts[0],
                'form': parts[1],
                'lemma': parts[2],
                'upos': parts[3],
                'xpos': parts[4],
                'feats': parts[5],
                'head': head,
                'deprel': parts[7],
                'deps': parts[8],
                'misc': parts[9]
            }
            current_tokens.append(token)
    
    if current_tokens:
        sentences.append({
            'meta': current_meta,
            'tokens': current_tokens
        })
    
    return sentences


# =============================================================================
# UD COLLECTOR
# =============================================================================

class UDCollector:
    """Collect from Universal Dependencies"""
    
    def __init__(self, db: CorpusDatabase, cache_dir: Path):
        self.db = db
        self.cache_dir = cache_dir / "ud"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreekCorpusPlatform/1.0 (Academic Research)'
        })
    
    def collect_all(self) -> Dict:
        """Collect all UD sources"""
        total_stats = {
            'sources': 0,
            'documents': 0,
            'sentences': 0,
            'tokens': 0,
            'errors': []
        }
        
        for source_id, source_info in UD_SOURCES.items():
            logger.info(f"\nCollecting: {source_info['name']}")
            
            try:
                stats = self.collect_source(source_id, source_info)
                total_stats['sources'] += 1
                total_stats['documents'] += 1
                total_stats['sentences'] += stats['sentences']
                total_stats['tokens'] += stats['tokens']
                
            except Exception as e:
                logger.error(f"Error collecting {source_id}: {e}")
                total_stats['errors'].append(f"{source_id}: {str(e)}")
            
            time.sleep(1)  # Rate limiting
        
        return total_stats
    
    def collect_source(self, source_id: str, source_info: Dict) -> Dict:
        """Collect a single UD source"""
        all_sentences = []
        
        for filename in source_info['files']:
            url = f"{source_info['url']}/{filename}"
            cache_file = self.cache_dir / f"{source_id}_{filename}"
            
            # Check cache
            if cache_file.exists():
                content = cache_file.read_text(encoding='utf-8')
                logger.info(f"  Loaded from cache: {filename}")
            else:
                # Download
                try:
                    response = self.session.get(url, timeout=CONFIG['request_timeout'])
                    if response.status_code != 200:
                        logger.warning(f"  Failed to download {filename}: {response.status_code}")
                        continue
                    
                    content = response.text
                    cache_file.write_text(content, encoding='utf-8')
                    logger.info(f"  Downloaded: {filename}")
                    
                except Exception as e:
                    logger.warning(f"  Download error for {filename}: {e}")
                    continue
            
            # Parse
            sentences = parse_conllu(content)
            all_sentences.extend(sentences)
        
        # Store in database
        if all_sentences:
            stats = self.db.store_conllu_document(source_id, source_info, all_sentences)
            return stats
        
        return {'sentences': 0, 'tokens': 0}


# =============================================================================
# MAIN AUTO RUNNER
# =============================================================================

class AutoRunner:
    """Main auto runner that executes everything"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        
        # Create directories
        self.data_dir = Path(self.config['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db = CorpusDatabase(self.config['db_path'])
        
        # Initialize collectors
        self.ud_collector = UDCollector(self.db, self.data_dir / "cache")
    
    def run_once(self) -> Dict:
        """Run a single collection cycle"""
        start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info("STARTING AUTO RUNNER COLLECTION CYCLE")
        logger.info(f"Time: {start_time.isoformat()}")
        logger.info("=" * 70)
        
        results = {
            'start_time': start_time.isoformat(),
            'ud_collection': {},
            'final_stats': {},
            'errors': []
        }
        
        try:
            # Collect from Universal Dependencies
            if self.config.get('enable_ud_collection', True):
                logger.info("\n" + "=" * 50)
                logger.info("COLLECTING FROM UNIVERSAL DEPENDENCIES")
                logger.info("=" * 50)
                
                results['ud_collection'] = self.ud_collector.collect_all()
                
                self.db.log_collection(
                    'universal_dependencies',
                    results['ud_collection']['documents'],
                    results['ud_collection']['sentences'],
                    results['ud_collection']['tokens'],
                    'success',
                    json.dumps(results['ud_collection'])
                )
            
            # Get final statistics
            results['final_stats'] = self.db.get_statistics()
            
        except Exception as e:
            logger.error(f"Collection error: {e}")
            traceback.print_exc()
            results['errors'].append(str(e))
            
            self.db.log_collection(
                'auto_runner',
                0, 0, 0,
                'error',
                str(e)
            )
        
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("COLLECTION CYCLE COMPLETE")
        logger.info(f"Duration: {results['duration_seconds']:.1f} seconds")
        logger.info(f"Documents: {results['final_stats'].get('documents', 0)}")
        logger.info(f"Sentences: {results['final_stats'].get('sentences', 0):,}")
        logger.info(f"Tokens: {results['final_stats'].get('tokens', 0):,}")
        logger.info("=" * 70)
        
        return results
    
    def run_continuous(self):
        """Run continuously with intervals"""
        logger.info("Starting continuous collection mode")
        logger.info(f"Interval: {self.config['run_interval_hours']} hours")
        
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                traceback.print_exc()
            
            # Wait for next cycle
            wait_seconds = self.config['run_interval_hours'] * 3600
            logger.info(f"Waiting {self.config['run_interval_hours']} hours until next cycle...")
            time.sleep(wait_seconds)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Greek Corpus Auto Runner')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                       help='Run mode: once or continuous')
    parser.add_argument('--data-dir', type=str, default='/root/corpus_platform/data',
                       help='Data directory')
    parser.add_argument('--interval', type=int, default=1,
                       help='Collection interval in hours (for continuous mode)')
    
    args = parser.parse_args()
    
    # Update config
    config = CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['db_path'] = f"{args.data_dir}/corpus_platform.db"
    config['run_interval_hours'] = args.interval
    
    # Create and run
    runner = AutoRunner(config)
    
    if args.mode == 'once':
        results = runner.run_once()
        
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        
        stats = results['final_stats']
        print(f"Documents: {stats.get('documents', 0)}")
        print(f"Sentences: {stats.get('sentences', 0):,}")
        print(f"Tokens: {stats.get('tokens', 0):,}")
        
        print("\nBy Period:")
        for period, data in stats.get('by_period', {}).items():
            print(f"  {period}: {data['docs']} docs, {data['tokens']:,} tokens")
        
        print("\nBy Language:")
        for lang, data in stats.get('by_language', {}).items():
            print(f"  {lang}: {data['docs']} docs, {data['tokens']:,} tokens")
        
    else:
        runner.run_continuous()


if __name__ == "__main__":
    main()
