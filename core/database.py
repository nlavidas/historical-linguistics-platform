"""
Database Management Module
Comprehensive database operations for the Diachronic Linguistics Platform
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Token:
    """Linguistic token with full annotation"""
    id: int
    form: str
    lemma: str = ""
    pos: str = ""
    xpos: str = ""
    morphology: Dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = ""
    deps: str = ""
    misc: Dict[str, str] = field(default_factory=dict)
    etymology: Optional[Dict] = None
    semantic_role: str = ""
    
    def to_conllu(self) -> str:
        feats = "|".join(f"{k}={v}" for k, v in self.morphology.items()) if self.morphology else "_"
        misc = "|".join(f"{k}={v}" for k, v in self.misc.items()) if self.misc else "_"
        return f"{self.id}\t{self.form}\t{self.lemma}\t{self.pos}\t{self.xpos or '_'}\t{feats}\t{self.head}\t{self.deprel}\t{self.deps or '_'}\t{misc}"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Token':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_conllu(cls, line: str) -> Optional['Token']:
        parts = line.strip().split('\t')
        if len(parts) < 10:
            return None
        
        try:
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                return None
            
            morphology = {}
            if parts[5] != '_':
                for feat in parts[5].split('|'):
                    if '=' in feat:
                        k, v = feat.split('=', 1)
                        morphology[k] = v
            
            misc = {}
            if parts[9] != '_':
                for item in parts[9].split('|'):
                    if '=' in item:
                        k, v = item.split('=', 1)
                        misc[k] = v
            
            return cls(
                id=int(token_id),
                form=parts[1],
                lemma=parts[2],
                pos=parts[3],
                xpos=parts[4] if parts[4] != '_' else '',
                morphology=morphology,
                head=int(parts[6]) if parts[6] != '_' else 0,
                deprel=parts[7] if parts[7] != '_' else '',
                deps=parts[8] if parts[8] != '_' else '',
                misc=misc
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing CoNLL-U line: {e}")
            return None


@dataclass
class Sentence:
    """Sentence with tokens and metadata"""
    id: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    valency_patterns: List[Dict] = field(default_factory=list)
    
    def to_conllu(self) -> str:
        lines = [f"# sent_id = {self.id}", f"# text = {self.text}"]
        for k, v in self.metadata.items():
            lines.append(f"# {k} = {v}")
        for token in self.tokens:
            lines.append(token.to_conllu())
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'tokens': [t.to_dict() for t in self.tokens],
            'metadata': self.metadata,
            'valency_patterns': self.valency_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Sentence':
        tokens = [Token.from_dict(t) for t in data.get('tokens', [])]
        return cls(
            id=data.get('id', ''),
            text=data.get('text', ''),
            tokens=tokens,
            metadata=data.get('metadata', {}),
            valency_patterns=data.get('valency_patterns', [])
        )
    
    def get_dependency_tree(self) -> Dict:
        tree = {'root': [], 'children': {}}
        for token in self.tokens:
            if token.head == 0:
                tree['root'].append(token.id)
            else:
                if token.head not in tree['children']:
                    tree['children'][token.head] = []
                tree['children'][token.head].append(token.id)
        return tree
    
    @property
    def token_count(self) -> int:
        return len(self.tokens)
    
    def get_token_by_id(self, token_id: int) -> Optional[Token]:
        for token in self.tokens:
            if token.id == token_id:
                return token
        return None


@dataclass
class Document:
    """Document with sentences and metadata"""
    id: str
    title: str
    author: str = "Unknown"
    language: str = "grc"
    period: str = ""
    date: str = ""
    genre: str = ""
    source: str = ""
    sentences: List[Sentence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_status: str = "pending"
    
    @property
    def token_count(self) -> int:
        return sum(s.token_count for s in self.sentences)
    
    @property
    def sentence_count(self) -> int:
        return len(self.sentences)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'period': self.period,
            'date': self.date,
            'genre': self.genre,
            'source': self.source,
            'sentences': [s.to_dict() for s in self.sentences],
            'metadata': self.metadata,
            'annotation_status': self.annotation_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        sentences = [Sentence.from_dict(s) for s in data.get('sentences', [])]
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            author=data.get('author', 'Unknown'),
            language=data.get('language', 'grc'),
            period=data.get('period', ''),
            date=data.get('date', ''),
            genre=data.get('genre', ''),
            source=data.get('source', ''),
            sentences=sentences,
            metadata=data.get('metadata', {}),
            annotation_status=data.get('annotation_status', 'pending')
        )
    
    def to_conllu(self) -> str:
        lines = [f"# newdoc id = {self.id}"]
        for sentence in self.sentences:
            lines.append(sentence.to_conllu())
            lines.append('')
        return '\n'.join(lines)


@dataclass
class ValencyFrame:
    """Verbal valency frame"""
    verb_lemma: str
    language: str
    pattern: str
    arguments: List[Dict[str, str]] = field(default_factory=list)
    frequency: int = 1
    examples: List[str] = field(default_factory=list)
    period: str = ""
    source: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ValencyFrame':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EtymologyEntry:
    """Etymology entry"""
    lemma: str
    language: str
    pos: str = ""
    proto_form: str = ""
    cognates: List[Dict[str, str]] = field(default_factory=list)
    semantic_development: List[str] = field(default_factory=list)
    borrowing_info: Optional[Dict] = None
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Thread-safe database manager"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = "corpus_platform.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        if self._initialized:
            return
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
    
    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT DEFAULT 'Unknown',
                    language TEXT NOT NULL,
                    period TEXT,
                    date_composed TEXT,
                    genre TEXT,
                    source TEXT,
                    content TEXT,
                    metadata TEXT,
                    annotation_status TEXT DEFAULT 'pending',
                    token_count INTEGER DEFAULT 0,
                    sentence_count INTEGER DEFAULT 0,
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
                    tokens TEXT,
                    metadata TEXT,
                    valency_patterns TEXT,
                    token_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # Tokens table for detailed queries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sentence_id TEXT NOT NULL,
                    token_index INTEGER NOT NULL,
                    form TEXT NOT NULL,
                    lemma TEXT,
                    pos TEXT,
                    xpos TEXT,
                    morphology TEXT,
                    head INTEGER,
                    deprel TEXT,
                    deps TEXT,
                    misc TEXT,
                    FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
                )
            """)
            
            # Valency lexicon
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS valency_lexicon (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    verb_lemma TEXT NOT NULL,
                    language TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    arguments TEXT,
                    frequency INTEGER DEFAULT 1,
                    examples TEXT,
                    period TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(verb_lemma, language, pattern, period)
                )
            """)
            
            # Etymology entries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etymology (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lemma TEXT NOT NULL,
                    language TEXT NOT NULL,
                    pos TEXT,
                    proto_form TEXT,
                    cognates TEXT,
                    semantic_development TEXT,
                    borrowing_info TEXT,
                    references TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lemma, language, pos)
                )
            """)
            
            # Research projects
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Project-document associations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_documents (
                    project_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (project_id, document_id),
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # Analysis cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT NOT NULL UNIQUE,
                    language TEXT,
                    analysis_types TEXT,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT
                )
            """)
            
            # System alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    severity TEXT NOT NULL,
                    category TEXT,
                    message TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TIMESTAMP,
                    resolved_by TEXT
                )
            """)
            
            # Improvement suggestions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvement_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    suggestion TEXT NOT NULL,
                    priority TEXT DEFAULT 'medium',
                    impact_estimate TEXT,
                    implementation_status TEXT DEFAULT 'pending',
                    implemented_at TIMESTAMP
                )
            """)
            
            # User activity log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    details TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_period ON documents(period)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentences_document ON sentences(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tokens_sentence ON tokens(sentence_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_verb ON valency_lexicon(verb_lemma)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_language ON valency_lexicon(language)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_etymology_lemma ON etymology(lemma)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_cache_hash ON analysis_cache(text_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    def save_document(self, doc: Document) -> bool:
        """Save document to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert/update document
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, title, author, language, period, date_composed, genre, source, 
                     metadata, annotation_status, token_count, sentence_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    doc.id, doc.title, doc.author, doc.language, doc.period,
                    doc.date, doc.genre, doc.source, json.dumps(doc.metadata),
                    doc.annotation_status, doc.token_count, doc.sentence_count
                ))
                
                # Delete existing sentences and tokens
                cursor.execute("DELETE FROM sentences WHERE document_id = ?", (doc.id,))
                
                # Insert sentences
                for idx, sentence in enumerate(doc.sentences):
                    cursor.execute("""
                        INSERT INTO sentences 
                        (id, document_id, sentence_index, text, tokens, metadata, 
                         valency_patterns, token_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sentence.id, doc.id, idx, sentence.text,
                        json.dumps([t.to_dict() for t in sentence.tokens]),
                        json.dumps(sentence.metadata),
                        json.dumps(sentence.valency_patterns),
                        sentence.token_count
                    ))
                    
                    # Insert tokens
                    for token in sentence.tokens:
                        cursor.execute("""
                            INSERT INTO tokens 
                            (sentence_id, token_index, form, lemma, pos, xpos, 
                             morphology, head, deprel, deps, misc)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sentence.id, token.id, token.form, token.lemma,
                            token.pos, token.xpos, json.dumps(token.morphology),
                            token.head, token.deprel, token.deps, json.dumps(token.misc)
                        ))
                
                conn.commit()
                self._log_activity('save_document', 'document', doc.id)
                return True
                
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Get sentences
                cursor.execute("""
                    SELECT * FROM sentences 
                    WHERE document_id = ? 
                    ORDER BY sentence_index
                """, (doc_id,))
                
                sentences = []
                for sent_row in cursor.fetchall():
                    tokens_data = json.loads(sent_row['tokens']) if sent_row['tokens'] else []
                    tokens = [Token.from_dict(t) for t in tokens_data]
                    
                    sentence = Sentence(
                        id=sent_row['id'],
                        text=sent_row['text'],
                        tokens=tokens,
                        metadata=json.loads(sent_row['metadata']) if sent_row['metadata'] else {},
                        valency_patterns=json.loads(sent_row['valency_patterns']) if sent_row['valency_patterns'] else []
                    )
                    sentences.append(sentence)
                
                return Document(
                    id=row['id'],
                    title=row['title'],
                    author=row['author'],
                    language=row['language'],
                    period=row['period'],
                    date=row['date_composed'],
                    genre=row['genre'],
                    source=row['source'],
                    sentences=sentences,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    annotation_status=row['annotation_status']
                )
                
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                self._log_activity('delete_document', 'document', doc_id)
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def search_documents(self, query: str = "", filters: Dict = None, 
                        limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """Search documents with filters"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                sql = """
                    SELECT d.*, COUNT(s.id) as sentence_count_actual
                    FROM documents d
                    LEFT JOIN sentences s ON d.id = s.document_id
                    WHERE 1=1
                """
                count_sql = "SELECT COUNT(DISTINCT d.id) FROM documents d WHERE 1=1"
                params = []
                count_params = []
                
                if query:
                    sql += " AND (d.title LIKE ? OR d.author LIKE ? OR d.content LIKE ?)"
                    count_sql += " AND (d.title LIKE ? OR d.author LIKE ? OR d.content LIKE ?)"
                    search_param = f"%{query}%"
                    params.extend([search_param] * 3)
                    count_params.extend([search_param] * 3)
                
                if filters:
                    if filters.get('language'):
                        sql += " AND d.language = ?"
                        count_sql += " AND d.language = ?"
                        params.append(filters['language'])
                        count_params.append(filters['language'])
                    
                    if filters.get('period'):
                        sql += " AND d.period = ?"
                        count_sql += " AND d.period = ?"
                        params.append(filters['period'])
                        count_params.append(filters['period'])
                    
                    if filters.get('genre'):
                        sql += " AND d.genre = ?"
                        count_sql += " AND d.genre = ?"
                        params.append(filters['genre'])
                        count_params.append(filters['genre'])
                    
                    if filters.get('author'):
                        sql += " AND d.author LIKE ?"
                        count_sql += " AND d.author LIKE ?"
                        params.append(f"%{filters['author']}%")
                        count_params.append(f"%{filters['author']}%")
                    
                    if filters.get('annotated_only'):
                        sql += " AND d.annotation_status = 'complete'"
                        count_sql += " AND d.annotation_status = 'complete'"
                
                sql += " GROUP BY d.id ORDER BY d.title LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute(count_sql, count_params)
                total = cursor.fetchone()[0]
                
                return results, total
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return [], 0
    
    # ========================================================================
    # CORPUS STATISTICS
    # ========================================================================
    
    def get_corpus_statistics(self) -> Dict:
        """Get comprehensive corpus statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Document counts
                cursor.execute("SELECT COUNT(*) FROM documents")
                stats['total_documents'] = cursor.fetchone()[0]
                
                # Sentence counts
                cursor.execute("SELECT COUNT(*) FROM sentences")
                stats['total_sentences'] = cursor.fetchone()[0]
                
                # Token counts
                cursor.execute("SELECT COUNT(*) FROM tokens")
                stats['total_tokens'] = cursor.fetchone()[0]
                
                # Unique lemmas
                cursor.execute("SELECT COUNT(DISTINCT lemma) FROM tokens WHERE lemma IS NOT NULL AND lemma != ''")
                stats['unique_lemmas'] = cursor.fetchone()[0]
                
                # Language distribution
                cursor.execute("""
                    SELECT language, COUNT(*) as count, SUM(token_count) as tokens
                    FROM documents 
                    GROUP BY language 
                    ORDER BY count DESC
                """)
                stats['languages'] = {
                    row['language']: {'documents': row['count'], 'tokens': row['tokens'] or 0}
                    for row in cursor.fetchall()
                }
                
                # Period distribution
                cursor.execute("""
                    SELECT period, COUNT(*) as count 
                    FROM documents 
                    WHERE period IS NOT NULL AND period != ''
                    GROUP BY period
                """)
                stats['periods'] = {row['period']: row['count'] for row in cursor.fetchall()}
                
                # Genre distribution
                cursor.execute("""
                    SELECT genre, COUNT(*) as count 
                    FROM documents 
                    WHERE genre IS NOT NULL AND genre != ''
                    GROUP BY genre
                    ORDER BY count DESC
                    LIMIT 20
                """)
                stats['genres'] = {row['genre']: row['count'] for row in cursor.fetchall()}
                
                # Annotation status
                cursor.execute("""
                    SELECT annotation_status, COUNT(*) as count 
                    FROM documents 
                    GROUP BY annotation_status
                """)
                stats['annotation_status'] = {row['annotation_status']: row['count'] for row in cursor.fetchall()}
                
                # Valency entries
                cursor.execute("SELECT COUNT(*) FROM valency_lexicon")
                stats['valency_entries'] = cursor.fetchone()[0]
                
                # Etymology entries
                cursor.execute("SELECT COUNT(*) FROM etymology")
                stats['etymology_entries'] = cursor.fetchone()[0]
                
                # Top authors
                cursor.execute("""
                    SELECT author, COUNT(*) as count 
                    FROM documents 
                    WHERE author IS NOT NULL AND author != 'Unknown'
                    GROUP BY author
                    ORDER BY count DESC
                    LIMIT 20
                """)
                stats['top_authors'] = {row['author']: row['count'] for row in cursor.fetchall()}
                
                # POS distribution
                cursor.execute("""
                    SELECT pos, COUNT(*) as count 
                    FROM tokens 
                    WHERE pos IS NOT NULL AND pos != ''
                    GROUP BY pos
                    ORDER BY count DESC
                """)
                stats['pos_distribution'] = {row['pos']: row['count'] for row in cursor.fetchall()}
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting corpus statistics: {e}")
            return {}
    
    # ========================================================================
    # VALENCY OPERATIONS
    # ========================================================================
    
    def save_valency_frame(self, frame: ValencyFrame) -> bool:
        """Save valency frame"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO valency_lexicon 
                    (verb_lemma, language, pattern, arguments, frequency, examples, period, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    frame.verb_lemma, frame.language, frame.pattern,
                    json.dumps(frame.arguments), frame.frequency,
                    json.dumps(frame.examples), frame.period, frame.source
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving valency frame: {e}")
            return False
    
    def search_valency(self, lemma: str = "", language: str = None, 
                      pattern: str = None, limit: int = 100) -> List[Dict]:
        """Search valency lexicon"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM valency_lexicon WHERE 1=1"
                params = []
                
                if lemma:
                    sql += " AND verb_lemma LIKE ?"
                    params.append(f"%{lemma}%")
                
                if language:
                    sql += " AND language = ?"
                    params.append(language)
                
                if pattern:
                    sql += " AND pattern = ?"
                    params.append(pattern)
                
                sql += " ORDER BY frequency DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    result = dict(row)
                    result['arguments'] = json.loads(result['arguments']) if result['arguments'] else []
                    result['examples'] = json.loads(result['examples']) if result['examples'] else []
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching valency: {e}")
            return []
    
    def get_valency_statistics(self) -> Dict:
        """Get valency lexicon statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total entries
                cursor.execute("SELECT COUNT(*) FROM valency_lexicon")
                stats['total_entries'] = cursor.fetchone()[0]
                
                # Unique verbs
                cursor.execute("SELECT COUNT(DISTINCT verb_lemma) FROM valency_lexicon")
                stats['unique_verbs'] = cursor.fetchone()[0]
                
                # Pattern distribution
                cursor.execute("""
                    SELECT pattern, COUNT(*) as count 
                    FROM valency_lexicon 
                    GROUP BY pattern 
                    ORDER BY count DESC
                """)
                stats['patterns'] = {row['pattern']: row['count'] for row in cursor.fetchall()}
                
                # Language distribution
                cursor.execute("""
                    SELECT language, COUNT(DISTINCT verb_lemma) as verbs, COUNT(*) as patterns
                    FROM valency_lexicon 
                    GROUP BY language
                """)
                stats['languages'] = {
                    row['language']: {'verbs': row['verbs'], 'patterns': row['patterns']}
                    for row in cursor.fetchall()
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting valency statistics: {e}")
            return {}
    
    # ========================================================================
    # ETYMOLOGY OPERATIONS
    # ========================================================================
    
    def save_etymology(self, entry: EtymologyEntry) -> bool:
        """Save etymology entry"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO etymology 
                    (lemma, language, pos, proto_form, cognates, semantic_development, 
                     borrowing_info, references)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.lemma, entry.language, entry.pos, entry.proto_form,
                    json.dumps(entry.cognates), json.dumps(entry.semantic_development),
                    json.dumps(entry.borrowing_info) if entry.borrowing_info else None,
                    json.dumps(entry.references)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving etymology: {e}")
            return False
    
    def search_etymology(self, lemma: str = "", language: str = None, limit: int = 50) -> List[Dict]:
        """Search etymology database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM etymology WHERE 1=1"
                params = []
                
                if lemma:
                    sql += " AND lemma LIKE ?"
                    params.append(f"%{lemma}%")
                
                if language:
                    sql += " AND language = ?"
                    params.append(language)
                
                sql += " LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    result = dict(row)
                    result['cognates'] = json.loads(result['cognates']) if result['cognates'] else []
                    result['semantic_development'] = json.loads(result['semantic_development']) if result['semantic_development'] else []
                    result['borrowing_info'] = json.loads(result['borrowing_info']) if result['borrowing_info'] else None
                    result['references'] = json.loads(result['references']) if result['references'] else []
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching etymology: {e}")
            return []
    
    # ========================================================================
    # MONITORING AND LOGGING
    # ========================================================================
    
    def log_metric(self, name: str, value: float, metadata: Dict = None):
        """Log performance metric"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO performance_metrics (metric_name, metric_value, metadata)
                    VALUES (?, ?, ?)
                """, (name, value, json.dumps(metadata) if metadata else None))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging metric: {e}")
    
    def create_alert(self, severity: str, category: str, message: str):
        """Create system alert"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (severity, category, message)
                    VALUES (?, ?, ?)
                """, (severity, category, message))
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def get_recent_alerts(self, limit: int = 20, include_resolved: bool = False) -> List[Dict]:
        """Get recent alerts"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM alerts"
                if not include_resolved:
                    sql += " WHERE resolved = 0"
                sql += " ORDER BY timestamp DESC LIMIT ?"
                
                cursor.execute(sql, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: int, resolved_by: str = None) -> bool:
        """Resolve an alert"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE alerts 
                    SET resolved = 1, resolved_at = CURRENT_TIMESTAMP, resolved_by = ?
                    WHERE id = ?
                """, (resolved_by, alert_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def _log_activity(self, action: str, entity_type: str = None, 
                     entity_id: str = None, details: Dict = None):
        """Log user activity"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO activity_log (action, entity_type, entity_id, details)
                    VALUES (?, ?, ?, ?)
                """, (action, entity_type, entity_id, json.dumps(details) if details else None))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
    
    def get_activity_log(self, limit: int = 100, entity_type: str = None) -> List[Dict]:
        """Get activity log"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                sql = "SELECT * FROM activity_log"
                params = []
                
                if entity_type:
                    sql += " WHERE entity_type = ?"
                    params.append(entity_type)
                
                sql += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return []
    
    # ========================================================================
    # CACHE OPERATIONS
    # ========================================================================
    
    def get_cached_analysis(self, text_hash: str) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT result FROM analysis_cache WHERE text_hash = ?
                """, (text_hash,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row['result'])
                return None
        except Exception as e:
            logger.error(f"Error getting cached analysis: {e}")
            return None
    
    def cache_analysis(self, text_hash: str, language: str, 
                      analysis_types: List[str], result: Dict):
        """Cache analysis result"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (text_hash, language, analysis_types, result)
                    VALUES (?, ?, ?, ?)
                """, (text_hash, language, json.dumps(analysis_types), json.dumps(result)))
                conn.commit()
        except Exception as e:
            logger.error(f"Error caching analysis: {e}")
    
    def clear_cache(self, older_than_days: int = 7) -> int:
        """Clear old cache entries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM analysis_cache 
                    WHERE created_at < datetime('now', ?)
                """, (f'-{older_than_days} days',))
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
