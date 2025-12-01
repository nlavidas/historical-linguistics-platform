#!/usr/bin/env python3
"""
Corpus Manager for Greek Diachronic Platform
Manages text collection, storage, and retrieval

Features:
- Multi-source collection (Perseus, PROIEL, First1KGreek, TLG)
- Period classification
- Genre tagging
- Author management
- Version control for texts
- Export in multiple formats
"""

import os
import re
import json
import sqlite3
import logging
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GREEK PERIODS
# ============================================================================

GREEK_PERIODS = {
    "mycenaean": {
        "name": "Mycenaean Greek",
        "start": -1600,
        "end": -1100,
        "description": "Linear B tablets, earliest Greek"
    },
    "archaic": {
        "name": "Archaic Greek",
        "start": -800,
        "end": -500,
        "description": "Homer, Hesiod, early lyric poetry"
    },
    "classical": {
        "name": "Classical Greek",
        "start": -500,
        "end": -323,
        "description": "Attic prose, drama, philosophy"
    },
    "hellenistic": {
        "name": "Hellenistic Greek",
        "start": -323,
        "end": -31,
        "description": "Koine development, Septuagint"
    },
    "roman": {
        "name": "Roman Period Greek",
        "start": -31,
        "end": 300,
        "description": "New Testament, Second Sophistic"
    },
    "late_antique": {
        "name": "Late Antique Greek",
        "start": 300,
        "end": 600,
        "description": "Church Fathers, early Byzantine"
    },
    "byzantine": {
        "name": "Byzantine Greek",
        "start": 600,
        "end": 1453,
        "description": "Medieval learned Greek"
    },
    "early_modern": {
        "name": "Early Modern Greek",
        "start": 1453,
        "end": 1830,
        "description": "Post-Byzantine, Cretan Renaissance"
    }
}

# ============================================================================
# GENRES
# ============================================================================

GENRES = {
    "epic": "Epic poetry",
    "lyric": "Lyric poetry",
    "drama_tragedy": "Tragedy",
    "drama_comedy": "Comedy",
    "history": "Historical prose",
    "philosophy": "Philosophical prose",
    "oratory": "Oratory and rhetoric",
    "scientific": "Scientific and technical",
    "religious": "Religious texts",
    "hagiography": "Saints' lives",
    "chronicle": "Chronicles",
    "romance": "Romance literature",
    "epistolary": "Letters",
    "legal": "Legal documents",
    "inscriptions": "Inscriptions",
    "papyri": "Papyri"
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Author:
    """Author information"""
    id: str
    name: str
    name_greek: str = ""
    dates: str = ""
    period: str = ""
    genres: List[str] = field(default_factory=list)
    description: str = ""
    tlg_id: str = ""
    wikipedia: str = ""


@dataclass
class Work:
    """Work/text information"""
    id: str
    title: str
    title_greek: str = ""
    author_id: str = ""
    period: str = ""
    genre: str = ""
    date_composed: str = ""
    language_register: str = ""
    description: str = ""
    tlg_id: str = ""
    perseus_id: str = ""


@dataclass
class TextVersion:
    """Version of a text"""
    id: str
    work_id: str
    version_name: str  # e.g., "OCT", "Teubner", "digital"
    source: str
    source_url: str = ""
    content: str = ""
    token_count: int = 0
    sentence_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""


# ============================================================================
# CORPUS DATABASE
# ============================================================================

class CorpusDatabase:
    """Database for corpus management"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Authors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_greek TEXT,
                dates TEXT,
                period TEXT,
                genres TEXT,
                description TEXT,
                tlg_id TEXT,
                wikipedia TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Works table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS works (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                title_greek TEXT,
                author_id TEXT,
                period TEXT,
                genre TEXT,
                date_composed TEXT,
                language_register TEXT,
                description TEXT,
                tlg_id TEXT,
                perseus_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES authors(id)
            )
        """)
        
        # Text versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_versions (
                id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL,
                version_name TEXT,
                source TEXT,
                source_url TEXT,
                content TEXT,
                token_count INTEGER DEFAULT 0,
                sentence_count INTEGER DEFAULT 0,
                checksum TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (work_id) REFERENCES works(id)
            )
        """)
        
        # Sentences table (for processed texts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                sentence_idx INTEGER,
                text TEXT,
                tokens TEXT,
                lemmas TEXT,
                pos_tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (version_id) REFERENCES text_versions(id)
            )
        """)
        
        # Collection log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                work_id TEXT,
                action TEXT,
                status TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    # Author methods
    def add_author(self, author: Author) -> bool:
        """Add author to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO authors
                (id, name, name_greek, dates, period, genres, description, tlg_id, wikipedia)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                author.id, author.name, author.name_greek, author.dates,
                author.period, json.dumps(author.genres), author.description,
                author.tlg_id, author.wikipedia
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding author: {e}")
            return False
    
    def get_author(self, author_id: str) -> Optional[Dict]:
        """Get author by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM authors WHERE id = ?", (author_id,))
        row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None
    
    def get_authors(self, period: str = None) -> List[Dict]:
        """Get all authors, optionally filtered by period"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if period:
            cursor.execute("SELECT * FROM authors WHERE period = ?", (period,))
        else:
            cursor.execute("SELECT * FROM authors ORDER BY name")
        
        authors = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return authors
    
    # Work methods
    def add_work(self, work: Work) -> bool:
        """Add work to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO works
                (id, title, title_greek, author_id, period, genre,
                 date_composed, language_register, description, tlg_id, perseus_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                work.id, work.title, work.title_greek, work.author_id,
                work.period, work.genre, work.date_composed,
                work.language_register, work.description, work.tlg_id, work.perseus_id
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding work: {e}")
            return False
    
    def get_work(self, work_id: str) -> Optional[Dict]:
        """Get work by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM works WHERE id = ?", (work_id,))
        row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None
    
    def get_works(self, author_id: str = None, period: str = None, 
                  genre: str = None) -> List[Dict]:
        """Get works with optional filters"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM works WHERE 1=1"
        params = []
        
        if author_id:
            query += " AND author_id = ?"
            params.append(author_id)
        if period:
            query += " AND period = ?"
            params.append(period)
        if genre:
            query += " AND genre = ?"
            params.append(genre)
        
        query += " ORDER BY title"
        
        cursor.execute(query, params)
        works = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return works
    
    # Text version methods
    def add_text_version(self, version: TextVersion) -> bool:
        """Add text version"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate checksum
            checksum = hashlib.md5(version.content.encode()).hexdigest()
            
            cursor.execute("""
                INSERT OR REPLACE INTO text_versions
                (id, work_id, version_name, source, source_url, content,
                 token_count, sentence_count, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.id, version.work_id, version.version_name,
                version.source, version.source_url, version.content,
                version.token_count, version.sentence_count, checksum
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding text version: {e}")
            return False
    
    def get_text_version(self, version_id: str) -> Optional[Dict]:
        """Get text version by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM text_versions WHERE id = ?", (version_id,))
        row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None
    
    def get_text_versions(self, work_id: str) -> List[Dict]:
        """Get all versions of a work"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM text_versions WHERE work_id = ? ORDER BY created_at DESC
        """, (work_id,))
        
        versions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return versions
    
    # Statistics
    def get_statistics(self) -> Dict:
        """Get corpus statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM authors")
        stats["authors"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM works")
        stats["works"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM text_versions")
        stats["text_versions"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(token_count) FROM text_versions")
        result = cursor.fetchone()[0]
        stats["total_tokens"] = result if result else 0
        
        cursor.execute("""
            SELECT period, COUNT(*) FROM works 
            WHERE period IS NOT NULL GROUP BY period
        """)
        stats["works_by_period"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT genre, COUNT(*) FROM works 
            WHERE genre IS NOT NULL GROUP BY genre
        """)
        stats["works_by_genre"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats
    
    # Logging
    def log_collection(self, source: str, work_id: str, action: str, 
                       status: str, message: str):
        """Log collection action"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO collection_log (source, work_id, action, status, message)
                VALUES (?, ?, ?, ?, ?)
            """, (source, work_id, action, status, message))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging collection: {e}")


# ============================================================================
# TEXT COLLECTORS
# ============================================================================

class PerseusCollector:
    """Collector for Perseus Digital Library"""
    
    BASE_URL = "https://www.perseus.tufts.edu/hopper"
    
    def __init__(self, db: CorpusDatabase):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreekCorpusPlatform/3.0 (Academic Research)'
        })
    
    def get_text(self, perseus_id: str) -> Optional[str]:
        """Get text from Perseus"""
        url = f"{self.BASE_URL}/text?doc=Perseus:text:{perseus_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                # Parse HTML and extract text
                # This is simplified - real implementation would parse properly
                text = self._extract_text(response.text)
                self.db.log_collection("perseus", perseus_id, "fetch", "success", 
                                       f"Retrieved {len(text)} characters")
                return text
            else:
                self.db.log_collection("perseus", perseus_id, "fetch", "error",
                                       f"HTTP {response.status_code}")
                return None
        except Exception as e:
            self.db.log_collection("perseus", perseus_id, "fetch", "error", str(e))
            return None
    
    def _extract_text(self, html: str) -> str:
        """Extract Greek text from Perseus HTML"""
        # Simple extraction - would need proper HTML parsing
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class First1KGreekCollector:
    """Collector for Open Greek and Latin First1KGreek"""
    
    BASE_URL = "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data"
    
    def __init__(self, db: CorpusDatabase):
        self.db = db
        self.session = requests.Session()
    
    def get_text(self, urn: str) -> Optional[str]:
        """Get text by CTS URN"""
        # Convert URN to file path
        # e.g., urn:cts:greekLit:tlg0012.tlg001 -> tlg0012/tlg001
        parts = urn.split(':')
        if len(parts) >= 4:
            work_parts = parts[3].split('.')
            if len(work_parts) >= 2:
                path = f"{work_parts[0]}/{work_parts[1]}"
                url = f"{self.BASE_URL}/{path}/{work_parts[0]}.{work_parts[1]}.perseus-grc1.xml"
                
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        text = self._extract_from_xml(response.text)
                        self.db.log_collection("first1k", urn, "fetch", "success",
                                               f"Retrieved {len(text)} characters")
                        return text
                except Exception as e:
                    self.db.log_collection("first1k", urn, "fetch", "error", str(e))
        
        return None
    
    def _extract_from_xml(self, xml_content: str) -> str:
        """Extract text from TEI XML"""
        try:
            root = ET.fromstring(xml_content)
            # Find text body
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            body = root.find('.//tei:body', ns)
            if body is not None:
                return ''.join(body.itertext())
            return ''.join(root.itertext())
        except Exception:
            # Fallback: strip XML tags
            return re.sub(r'<[^>]+>', '', xml_content)


class PROIELCollector:
    """Collector for PROIEL Treebank"""
    
    BASE_URL = "https://raw.githubusercontent.com/proiel/proiel-treebank/master/releases"
    
    def __init__(self, db: CorpusDatabase):
        self.db = db
        self.session = requests.Session()
    
    def get_treebank(self, filename: str) -> Optional[Dict]:
        """Get PROIEL treebank file"""
        url = f"{self.BASE_URL}/{filename}"
        
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                data = self._parse_proiel_xml(response.text)
                self.db.log_collection("proiel", filename, "fetch", "success",
                                       f"Retrieved {len(data.get('sentences', []))} sentences")
                return data
        except Exception as e:
            self.db.log_collection("proiel", filename, "fetch", "error", str(e))
        
        return None
    
    def _parse_proiel_xml(self, xml_content: str) -> Dict:
        """Parse PROIEL XML format"""
        result = {
            'sentences': [],
            'tokens': []
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            for sentence in root.findall('.//sentence'):
                sent_data = {
                    'id': sentence.get('id'),
                    'tokens': []
                }
                
                for token in sentence.findall('.//token'):
                    token_data = {
                        'id': token.get('id'),
                        'form': token.get('form'),
                        'lemma': token.get('lemma'),
                        'pos': token.get('part-of-speech'),
                        'morph': token.get('morphology'),
                        'head': token.get('head-id'),
                        'relation': token.get('relation')
                    }
                    sent_data['tokens'].append(token_data)
                
                result['sentences'].append(sent_data)
        
        except Exception as e:
            logger.error(f"Error parsing PROIEL XML: {e}")
        
        return result


# ============================================================================
# CORPUS MANAGER
# ============================================================================

class CorpusManager:
    """Main corpus manager"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db = CorpusDatabase(db_path)
        
        # Collectors
        self.perseus = PerseusCollector(self.db)
        self.first1k = First1KGreekCollector(self.db)
        self.proiel = PROIELCollector(self.db)
    
    def add_author(self, name: str, name_greek: str = "", dates: str = "",
                   period: str = "", genres: List[str] = None,
                   description: str = "") -> str:
        """Add author and return ID"""
        author_id = hashlib.md5(name.encode()).hexdigest()[:12]
        
        author = Author(
            id=author_id,
            name=name,
            name_greek=name_greek,
            dates=dates,
            period=period,
            genres=genres or [],
            description=description
        )
        
        self.db.add_author(author)
        return author_id
    
    def add_work(self, title: str, author_id: str, period: str = "",
                 genre: str = "", title_greek: str = "",
                 description: str = "") -> str:
        """Add work and return ID"""
        work_id = hashlib.md5(f"{author_id}:{title}".encode()).hexdigest()[:12]
        
        work = Work(
            id=work_id,
            title=title,
            title_greek=title_greek,
            author_id=author_id,
            period=period,
            genre=genre,
            description=description
        )
        
        self.db.add_work(work)
        return work_id
    
    def add_text(self, work_id: str, content: str, source: str,
                 version_name: str = "default", source_url: str = "") -> str:
        """Add text version and return ID"""
        version_id = hashlib.md5(
            f"{work_id}:{version_name}:{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        # Count tokens and sentences
        tokens = content.split()
        sentences = re.split(r'[.;·!?]+', content)
        
        version = TextVersion(
            id=version_id,
            work_id=work_id,
            version_name=version_name,
            source=source,
            source_url=source_url,
            content=content,
            token_count=len(tokens),
            sentence_count=len([s for s in sentences if s.strip()])
        )
        
        self.db.add_text_version(version)
        return version_id
    
    def collect_from_perseus(self, perseus_id: str, work_id: str) -> bool:
        """Collect text from Perseus"""
        text = self.perseus.get_text(perseus_id)
        if text:
            self.add_text(work_id, text, "perseus", "perseus_digital",
                         f"https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:{perseus_id}")
            return True
        return False
    
    def collect_from_first1k(self, urn: str, work_id: str) -> bool:
        """Collect text from First1KGreek"""
        text = self.first1k.get_text(urn)
        if text:
            self.add_text(work_id, text, "first1k", "ogl_first1k",
                         f"https://github.com/OpenGreekAndLatin/First1KGreek")
            return True
        return False
    
    def search_texts(self, query: str, period: str = None,
                     genre: str = None) -> List[Dict]:
        """Search texts"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        sql = """
            SELECT tv.id, tv.content, w.title, w.period, w.genre, a.name as author
            FROM text_versions tv
            JOIN works w ON tv.work_id = w.id
            LEFT JOIN authors a ON w.author_id = a.id
            WHERE tv.content LIKE ?
        """
        params = [f"%{query}%"]
        
        if period:
            sql += " AND w.period = ?"
            params.append(period)
        
        if genre:
            sql += " AND w.genre = ?"
            params.append(genre)
        
        sql += " LIMIT 100"
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            # Find context around match
            content = row['content']
            idx = content.lower().find(query.lower())
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                context = content[start:end]
            else:
                context = content[:100]
            
            results.append({
                'version_id': row['id'],
                'title': row['title'],
                'author': row['author'],
                'period': row['period'],
                'genre': row['genre'],
                'context': context
            })
        
        conn.close()
        return results
    
    def export_corpus(self, output_dir: str, format: str = "txt"):
        """Export corpus to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        works = self.db.get_works()
        
        for work in works:
            versions = self.db.get_text_versions(work['id'])
            
            for version in versions:
                if not version['content']:
                    continue
                
                # Create filename
                filename = f"{work['id']}_{version['version_name']}"
                
                if format == "txt":
                    filepath = output_path / f"{filename}.txt"
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"# {work['title']}\n")
                        f.write(f"# Period: {work['period']}\n")
                        f.write(f"# Genre: {work['genre']}\n\n")
                        f.write(version['content'])
                
                elif format == "json":
                    filepath = output_path / f"{filename}.json"
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump({
                            'work': work,
                            'version': {
                                'id': version['id'],
                                'name': version['version_name'],
                                'source': version['source'],
                                'token_count': version['token_count']
                            },
                            'content': version['content']
                        }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(works)} works to {output_dir}")
    
    def get_statistics(self) -> Dict:
        """Get corpus statistics"""
        return self.db.get_statistics()


# ============================================================================
# SEED DATA
# ============================================================================

def seed_canonical_authors(manager: CorpusManager):
    """Seed database with canonical Greek authors"""
    
    authors = [
        # Archaic
        ("Homer", "Ὅμηρος", "8th c. BCE", "archaic", ["epic"]),
        ("Hesiod", "Ἡσίοδος", "8th-7th c. BCE", "archaic", ["epic", "didactic"]),
        ("Sappho", "Σαπφώ", "7th-6th c. BCE", "archaic", ["lyric"]),
        ("Pindar", "Πίνδαρος", "518-438 BCE", "archaic", ["lyric"]),
        
        # Classical
        ("Aeschylus", "Αἰσχύλος", "525-456 BCE", "classical", ["drama_tragedy"]),
        ("Sophocles", "Σοφοκλῆς", "496-406 BCE", "classical", ["drama_tragedy"]),
        ("Euripides", "Εὐριπίδης", "480-406 BCE", "classical", ["drama_tragedy"]),
        ("Aristophanes", "Ἀριστοφάνης", "446-386 BCE", "classical", ["drama_comedy"]),
        ("Herodotus", "Ἡρόδοτος", "484-425 BCE", "classical", ["history"]),
        ("Thucydides", "Θουκυδίδης", "460-400 BCE", "classical", ["history"]),
        ("Xenophon", "Ξενοφῶν", "430-354 BCE", "classical", ["history", "philosophy"]),
        ("Plato", "Πλάτων", "428-348 BCE", "classical", ["philosophy"]),
        ("Aristotle", "Ἀριστοτέλης", "384-322 BCE", "classical", ["philosophy", "scientific"]),
        ("Demosthenes", "Δημοσθένης", "384-322 BCE", "classical", ["oratory"]),
        ("Lysias", "Λυσίας", "445-380 BCE", "classical", ["oratory"]),
        
        # Hellenistic
        ("Polybius", "Πολύβιος", "200-118 BCE", "hellenistic", ["history"]),
        ("Apollonius Rhodius", "Ἀπολλώνιος Ῥόδιος", "3rd c. BCE", "hellenistic", ["epic"]),
        
        # Roman
        ("Plutarch", "Πλούταρχος", "46-120 CE", "roman", ["philosophy", "history"]),
        ("Lucian", "Λουκιανός", "125-180 CE", "roman", ["satire"]),
        ("Epictetus", "Ἐπίκτητος", "50-135 CE", "roman", ["philosophy"]),
        ("Marcus Aurelius", "Μάρκος Αὐρήλιος", "121-180 CE", "roman", ["philosophy"]),
        
        # Late Antique
        ("John Chrysostom", "Ἰωάννης Χρυσόστομος", "347-407 CE", "late_antique", ["religious"]),
        ("Basil of Caesarea", "Βασίλειος Καισαρείας", "330-379 CE", "late_antique", ["religious"]),
        ("Gregory of Nazianzus", "Γρηγόριος Ναζιανζηνός", "329-390 CE", "late_antique", ["religious"]),
        
        # Byzantine
        ("Anna Comnena", "Ἄννα Κομνηνή", "1083-1153 CE", "byzantine", ["history"]),
        ("Michael Psellus", "Μιχαὴλ Ψελλός", "1018-1078 CE", "byzantine", ["history", "philosophy"]),
        ("Maximus Planudes", "Μάξιμος Πλανούδης", "1260-1305 CE", "byzantine", ["translation"]),
        
        # Early Modern
        ("Vitsentzos Kornaros", "Βιτσέντζος Κορνάρος", "1553-1613 CE", "early_modern", ["romance"]),
    ]
    
    for name, greek, dates, period, genres in authors:
        manager.add_author(name, greek, dates, period, genres)
    
    logger.info(f"Seeded {len(authors)} canonical authors")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Corpus Manager")
    parser.add_argument('command', choices=['stats', 'seed', 'search', 'export', 'collect'],
                       help="Command to run")
    parser.add_argument('--query', '-q', help="Search query")
    parser.add_argument('--period', '-p', help="Period filter")
    parser.add_argument('--genre', '-g', help="Genre filter")
    parser.add_argument('--output', '-o', help="Output directory")
    parser.add_argument('--format', '-f', default="txt", help="Export format")
    
    args = parser.parse_args()
    
    manager = CorpusManager()
    
    if args.command == 'stats':
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'seed':
        seed_canonical_authors(manager)
        print("Database seeded with canonical authors")
    
    elif args.command == 'search':
        if args.query:
            results = manager.search_texts(args.query, args.period, args.genre)
            for r in results:
                print(f"{r['title']} ({r['author']}) - {r['period']}")
                print(f"  ...{r['context']}...")
                print()
        else:
            print("Please provide --query")
    
    elif args.command == 'export':
        output = args.output or "corpus_export"
        manager.export_corpus(output, args.format)
        print(f"Exported to {output}")
    
    elif args.command == 'collect':
        print("Collection requires specific source and work IDs")


if __name__ == "__main__":
    main()
