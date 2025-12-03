"""
Database Integration Module
Connects collected texts to the main platform database
Ensures data flows from collection → preprocessing → annotation → UI
"""

import os
import re
import json
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_CONFIG = {
    "main_db": "corpus_platform.db",
    "collected_db": "collected_texts.db",
    "preprocessed_db": "preprocessed.db",
    "annotations_db": "annotations.db",
    "wal_mode": True,
    "timeout": 30
}

# =============================================================================
# GREEK PERIODS MAPPING
# =============================================================================

PERIOD_MAPPING = {
    # Standard periods
    "archaic": {"name": "Archaic Greek", "start": -800, "end": -500},
    "classical": {"name": "Classical Greek", "start": -500, "end": -323},
    "hellenistic": {"name": "Hellenistic Greek", "start": -323, "end": -31},
    "koine": {"name": "Koine Greek", "start": -31, "end": 330},
    "late_antique": {"name": "Late Antique Greek", "start": 330, "end": 600},
    "byzantine": {"name": "Byzantine Greek", "start": 600, "end": 1453},
    "medieval": {"name": "Medieval Greek", "start": 1100, "end": 1453},
    "early_modern": {"name": "Early Modern Greek", "start": 1453, "end": 1830},
    "modern": {"name": "Modern Greek", "start": 1830, "end": 2025},
    
    # Alternative names
    "ancient": {"name": "Ancient Greek", "start": -800, "end": -31},
    "post_classical": {"name": "Post-Classical", "start": -323, "end": 330},
    "mixed": {"name": "Mixed Periods", "start": -800, "end": 2025}
}

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseIntegration:
    """Integrates all databases and ensures data consistency"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database paths
        self.main_db_path = self.data_dir / DB_CONFIG['main_db']
        self.collected_db_path = self.data_dir / DB_CONFIG['collected_db']
        self.preprocessed_db_path = self.data_dir / DB_CONFIG['preprocessed_db']
        
        # Initialize all databases
        self._init_main_database()
        self._init_preprocessed_database()
    
    def _get_connection(self, db_path: Path) -> sqlite3.Connection:
        """Get database connection with WAL mode"""
        conn = sqlite3.connect(db_path, timeout=DB_CONFIG['timeout'])
        conn.row_factory = sqlite3.Row
        
        if DB_CONFIG['wal_mode']:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        
        return conn
    
    def _init_main_database(self):
        """Initialize the main platform database"""
        conn = self._get_connection(self.main_db_path)
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sentences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                sentence_index INTEGER,
                text TEXT NOT NULL,
                translation TEXT,
                tokens TEXT,
                semantic_roles TEXT,
                annotation_status TEXT DEFAULT 'pending',
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id TEXT NOT NULL,
                token_index INTEGER,
                form TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                morphology TEXT,
                head INTEGER,
                relation TEXT,
                semantic_role TEXT,
                gloss TEXT,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
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
                semantic_class TEXT,
                UNIQUE(lemma, pattern, period)
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
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_period ON documents(period)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_language ON documents(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_token_sent ON tokens(sentence_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_lemma ON valency_lexicon(lemma)")
        
        conn.commit()
        conn.close()
        logger.info(f"Main database initialized: {self.main_db_path}")
    
    def _init_preprocessed_database(self):
        """Initialize the preprocessed texts database"""
        conn = self._get_connection(self.preprocessed_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preprocessed_texts (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                original_text TEXT,
                normalized_text TEXT,
                sentences TEXT,
                tokens TEXT,
                word_count INTEGER,
                sentence_count INTEGER,
                preprocessing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def import_collected_texts(self) -> Dict[str, int]:
        """Import texts from collected_texts.db to main database"""
        if not self.collected_db_path.exists():
            logger.warning("No collected texts database found")
            return {'imported': 0, 'skipped': 0, 'errors': 0}
        
        stats = {'imported': 0, 'skipped': 0, 'errors': 0}
        
        # Connect to both databases
        collected_conn = self._get_connection(self.collected_db_path)
        main_conn = self._get_connection(self.main_db_path)
        
        collected_cursor = collected_conn.cursor()
        main_cursor = main_conn.cursor()
        
        # Get all collected texts
        collected_cursor.execute("""
            SELECT id, title, author, period, century, language, genre,
                   content, source_url, word_count, char_count, collection_date
            FROM collected_texts
        """)
        
        for row in collected_cursor.fetchall():
            try:
                # Check if already imported
                main_cursor.execute("SELECT id FROM documents WHERE id = ?", (row['id'],))
                if main_cursor.fetchone():
                    stats['skipped'] += 1
                    continue
                
                # Preprocess the text
                sentences, tokens = self._preprocess_text(row['content'])
                
                # Insert document
                main_cursor.execute("""
                    INSERT INTO documents 
                    (id, title, author, period, century, genre, source, language,
                     sentence_count, token_count, annotation_status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['id'], row['title'], row['author'], row['period'],
                    row['century'], row['genre'], row['source_url'], row['language'],
                    len(sentences), sum(len(s['tokens']) for s in sentences),
                    'imported', json.dumps({
                        'collection_date': row['collection_date'],
                        'word_count': row['word_count'],
                        'char_count': row['char_count']
                    })
                ))
                
                # Insert sentences and tokens
                for sent_idx, sent_data in enumerate(sentences):
                    sent_id = f"{row['id']}_s{sent_idx:04d}"
                    
                    main_cursor.execute("""
                        INSERT INTO sentences
                        (id, document_id, sentence_index, text, tokens, annotation_status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        sent_id, row['id'], sent_idx, sent_data['text'],
                        json.dumps(sent_data['tokens']), 'pending'
                    ))
                    
                    # Insert tokens
                    for tok_idx, token in enumerate(sent_data['tokens']):
                        main_cursor.execute("""
                            INSERT INTO tokens
                            (sentence_id, token_index, form, lemma, pos)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            sent_id, tok_idx, token['form'],
                            token.get('lemma', ''), token.get('pos', '')
                        ))
                
                stats['imported'] += 1
                
                if stats['imported'] % 10 == 0:
                    main_conn.commit()
                    logger.info(f"Imported {stats['imported']} documents...")
                    
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error importing {row['id']}: {e}")
        
        main_conn.commit()
        
        # Log the import
        main_cursor.execute("""
            INSERT INTO collection_log (source, documents_added, sentences_added, tokens_added, status, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'collected_texts_import',
            stats['imported'],
            0,  # Will be calculated
            0,  # Will be calculated
            'completed',
            json.dumps(stats)
        ))
        main_conn.commit()
        
        collected_conn.close()
        main_conn.close()
        
        logger.info(f"Import complete: {stats}")
        return stats
    
    def _preprocess_text(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Basic preprocessing of text into sentences and tokens"""
        if not text:
            return [], []
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Split into sentences (basic)
        sentence_endings = re.compile(r'([.;·!?]+)\s+')
        parts = sentence_endings.split(text)
        
        sentences = []
        current_sent = ""
        
        for i, part in enumerate(parts):
            if sentence_endings.match(part):
                current_sent += part
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""
            else:
                current_sent += part
        
        if current_sent.strip():
            sentences.append(current_sent.strip())
        
        # Tokenize each sentence
        result = []
        for sent_text in sentences:
            tokens = self._tokenize(sent_text)
            result.append({
                'text': sent_text,
                'tokens': tokens
            })
        
        return result, []
    
    def _tokenize(self, text: str) -> List[Dict]:
        """Basic tokenization"""
        # Simple word tokenization
        words = re.findall(r'\b[\w\u0370-\u03FF\u1F00-\u1FFF]+\b', text)
        
        tokens = []
        for word in words:
            tokens.append({
                'form': word,
                'lemma': '',
                'pos': ''
            })
        
        return tokens
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics from main database"""
        conn = self._get_connection(self.main_db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Document counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(sentence_count) FROM documents")
        result = cursor.fetchone()[0]
        stats['sentences'] = result if result else 0
        
        cursor.execute("SELECT SUM(token_count) FROM documents")
        result = cursor.fetchone()[0]
        stats['tokens'] = result if result else 0
        
        # By period
        cursor.execute("""
            SELECT period, COUNT(*), SUM(sentence_count), SUM(token_count)
            FROM documents
            WHERE period IS NOT NULL
            GROUP BY period
        """)
        stats['by_period'] = {
            row[0]: {
                'documents': row[1],
                'sentences': row[2] or 0,
                'tokens': row[3] or 0
            }
            for row in cursor.fetchall()
        }
        
        # By language
        cursor.execute("""
            SELECT language, COUNT(*), SUM(token_count)
            FROM documents
            WHERE language IS NOT NULL
            GROUP BY language
        """)
        stats['by_language'] = {
            row[0]: {'documents': row[1], 'tokens': row[2] or 0}
            for row in cursor.fetchall()
        }
        
        # Valency lexicon
        cursor.execute("SELECT COUNT(DISTINCT lemma) FROM valency_lexicon")
        stats['valency_verbs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM valency_lexicon")
        stats['valency_patterns'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def sync_to_platform_db(self):
        """Ensure main database is synced with platform_app expectations"""
        conn = self._get_connection(self.main_db_path)
        cursor = conn.cursor()
        
        # Update document counts
        cursor.execute("""
            UPDATE documents SET
                sentence_count = (SELECT COUNT(*) FROM sentences WHERE document_id = documents.id),
                token_count = (SELECT COUNT(*) FROM tokens t 
                              JOIN sentences s ON t.sentence_id = s.id 
                              WHERE s.document_id = documents.id)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database synced with platform")


# =============================================================================
# CONLL-U IMPORTER
# =============================================================================

class CoNLLUImporter:
    """Import CoNLL-U format files from Universal Dependencies"""
    
    def __init__(self, db_integration: DatabaseIntegration):
        self.db = db_integration
    
    def import_file(self, filepath: Path, source_info: Dict) -> Dict[str, int]:
        """Import a CoNLL-U file"""
        stats = {'sentences': 0, 'tokens': 0}
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return stats
        
        conn = self.db._get_connection(self.db.main_db_path)
        cursor = conn.cursor()
        
        # Create document entry
        doc_id = source_info.get('id', filepath.stem)
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (id, title, author, period, genre, source, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            source_info.get('title', filepath.stem),
            source_info.get('author', 'Unknown'),
            source_info.get('period', 'unknown'),
            source_info.get('genre', 'unknown'),
            source_info.get('source', str(filepath)),
            source_info.get('language', 'grc')
        ))
        
        # Parse CoNLL-U
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sentences = self._parse_conllu(content)
        
        for sent_idx, sentence in enumerate(sentences):
            sent_id = f"{doc_id}_s{sent_idx:04d}"
            sent_text = ' '.join(t['form'] for t in sentence['tokens'])
            
            cursor.execute("""
                INSERT INTO sentences
                (id, document_id, sentence_index, text, tokens, annotation_status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sent_id, doc_id, sent_idx, sent_text,
                json.dumps(sentence['tokens']), 'annotated'
            ))
            
            for tok_idx, token in enumerate(sentence['tokens']):
                cursor.execute("""
                    INSERT INTO tokens
                    (sentence_id, token_index, form, lemma, pos, morphology, head, relation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sent_id, tok_idx,
                    token['form'], token['lemma'], token['upos'],
                    token['feats'], token['head'], token['deprel']
                ))
                stats['tokens'] += 1
            
            stats['sentences'] += 1
        
        # Update document counts
        cursor.execute("""
            UPDATE documents SET sentence_count = ?, token_count = ?
            WHERE id = ?
        """, (stats['sentences'], stats['tokens'], doc_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Imported {filepath.name}: {stats['sentences']} sentences, {stats['tokens']} tokens")
        return stats
    
    def _parse_conllu(self, content: str) -> List[Dict]:
        """Parse CoNLL-U format"""
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
                # Metadata
                if '=' in line:
                    key, value = line[1:].split('=', 1)
                    current_meta[key.strip()] = value.strip()
                continue
            
            parts = line.split('\t')
            if len(parts) >= 10:
                # Skip multi-word tokens
                if '-' in parts[0] or '.' in parts[0]:
                    continue
                
                token = {
                    'id': parts[0],
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3],
                    'xpos': parts[4],
                    'feats': parts[5],
                    'head': int(parts[6]) if parts[6].isdigit() else 0,
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
# AUTO-SYNC SERVICE
# =============================================================================

class AutoSyncService:
    """Automatically sync collected texts to main database"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.db = DatabaseIntegration(data_dir)
        self.conllu = CoNLLUImporter(self.db)
    
    def run_full_sync(self) -> Dict:
        """Run complete synchronization"""
        results = {
            'collected_import': {},
            'conllu_import': {},
            'final_stats': {}
        }
        
        logger.info("=" * 60)
        logger.info("STARTING FULL DATABASE SYNC")
        logger.info("=" * 60)
        
        # Import collected texts
        logger.info("\n1. Importing collected texts...")
        results['collected_import'] = self.db.import_collected_texts()
        
        # Import CoNLL-U files
        logger.info("\n2. Importing CoNLL-U files...")
        conllu_dir = self.data_dir / "conllu"
        if conllu_dir.exists():
            for conllu_file in conllu_dir.glob("*.conllu"):
                source_info = self._get_source_info(conllu_file)
                stats = self.conllu.import_file(conllu_file, source_info)
                results['conllu_import'][conllu_file.name] = stats
        
        # Sync counts
        logger.info("\n3. Syncing database...")
        self.db.sync_to_platform_db()
        
        # Get final stats
        results['final_stats'] = self.db.get_statistics()
        
        logger.info("\n" + "=" * 60)
        logger.info("SYNC COMPLETE")
        logger.info(f"Documents: {results['final_stats']['documents']}")
        logger.info(f"Sentences: {results['final_stats']['sentences']}")
        logger.info(f"Tokens: {results['final_stats']['tokens']}")
        logger.info("=" * 60)
        
        return results
    
    def _get_source_info(self, filepath: Path) -> Dict:
        """Extract source info from filename"""
        name = filepath.stem
        
        # Try to parse UD naming convention
        parts = name.split('-')
        
        info = {
            'id': name,
            'title': name,
            'source': str(filepath)
        }
        
        if len(parts) >= 2:
            lang_code = parts[0]
            info['language'] = lang_code
            
            # Map language codes
            lang_map = {
                'grc': 'Ancient Greek',
                'el': 'Modern Greek',
                'la': 'Latin',
                'got': 'Gothic',
                'cu': 'Old Church Slavonic',
                'ang': 'Old English',
                'enm': 'Middle English',
                'en': 'English',
                'fro': 'Old French',
                'fr': 'French'
            }
            
            if lang_code in lang_map:
                info['title'] = f"{lang_map[lang_code]} - {parts[1] if len(parts) > 1 else 'Unknown'}"
        
        return info


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data"
    
    sync = AutoSyncService(data_dir)
    results = sync.run_full_sync()
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    stats = results['final_stats']
    print(f"Documents: {stats['documents']}")
    print(f"Sentences: {stats['sentences']:,}")
    print(f"Tokens: {stats['tokens']:,}")
    
    print("\nBy Period:")
    for period, data in stats.get('by_period', {}).items():
        print(f"  {period}: {data['documents']} docs, {data['tokens']:,} tokens")
    
    print("\nBy Language:")
    for lang, data in stats.get('by_language', {}).items():
        print(f"  {lang}: {data['documents']} docs, {data['tokens']:,} tokens")
