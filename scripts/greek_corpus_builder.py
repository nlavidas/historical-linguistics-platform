#!/usr/bin/env python3
"""
Complete Greek Corpus Builder
Comprehensive system for building the definitive Greek corpus
Based on the work of Nikolaos Lavidas and University of Athens

Features:
- Complete Greek text collection (Homer to Byzantine)
- PROIEL-style annotation
- Morphological analysis
- Syntactic parsing
- Valency extraction
- Metadata tracking
- Quality monitoring
"""

import os
import sys
import re
import json
import time
import logging
import sqlite3
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('greek_corpus_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - GREEK CORPUS
# ============================================================================

GREEK_CORPUS_CONFIG = {
    "name": "Complete Greek Corpus",
    "version": "1.0.0",
    "institution": "University of Athens",
    "principal_investigator": "Nikolaos Lavidas",
    "database_path": "greek_corpus.db",
    "annotation_standard": "PROIEL",
    "languages": ["grc", "grc-koine", "grc-byzantine"],
    
    # Text sources
    "sources": {
        "perseus": {
            "name": "Perseus Digital Library",
            "url": "http://www.perseus.tufts.edu",
            "priority": 1
        },
        "proiel": {
            "name": "PROIEL Treebank",
            "url": "https://proiel.github.io",
            "priority": 1
        },
        "first1k": {
            "name": "First1KGreek",
            "url": "https://opengreekandlatin.github.io/First1KGreek",
            "priority": 2
        },
        "tlg": {
            "name": "Thesaurus Linguae Graecae",
            "url": "http://stephanus.tlg.uci.edu",
            "priority": 1
        },
        "diorisis": {
            "name": "Diorisis Ancient Greek Corpus",
            "url": "https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256",
            "priority": 2
        }
    },
    
    # Periods
    "periods": {
        "archaic": {"start": -800, "end": -500, "name": "Archaic Greek"},
        "classical": {"start": -500, "end": -323, "name": "Classical Greek"},
        "hellenistic": {"start": -323, "end": -31, "name": "Hellenistic Greek"},
        "koine": {"start": -300, "end": 300, "name": "Koine Greek"},
        "roman": {"start": -31, "end": 284, "name": "Roman Period"},
        "late_antique": {"start": 284, "end": 600, "name": "Late Antique"},
        "byzantine": {"start": 330, "end": 1453, "name": "Byzantine Greek"},
        "modern": {"start": 1453, "end": 2000, "name": "Modern Greek"}
    },
    
    # Genres
    "genres": [
        "epic", "lyric", "tragedy", "comedy", "history", "philosophy",
        "oratory", "rhetoric", "biography", "geography", "science",
        "medicine", "mathematics", "astronomy", "theology", "patristic",
        "hagiography", "chronicle", "romance", "epistle", "apocalyptic"
    ],
    
    # Core authors
    "core_authors": {
        # Archaic
        "Homer": {"period": "archaic", "works": ["Iliad", "Odyssey"]},
        "Hesiod": {"period": "archaic", "works": ["Theogony", "Works and Days"]},
        
        # Classical - Tragedy
        "Aeschylus": {"period": "classical", "works": ["Oresteia", "Prometheus Bound", "Seven Against Thebes"]},
        "Sophocles": {"period": "classical", "works": ["Oedipus Rex", "Antigone", "Electra"]},
        "Euripides": {"period": "classical", "works": ["Medea", "Bacchae", "Hippolytus"]},
        
        # Classical - Comedy
        "Aristophanes": {"period": "classical", "works": ["Clouds", "Birds", "Frogs"]},
        "Menander": {"period": "hellenistic", "works": ["Dyskolos", "Samia"]},
        
        # Classical - History
        "Herodotus": {"period": "classical", "works": ["Histories"]},
        "Thucydides": {"period": "classical", "works": ["History of the Peloponnesian War"]},
        "Xenophon": {"period": "classical", "works": ["Anabasis", "Hellenica", "Memorabilia"]},
        
        # Classical - Philosophy
        "Plato": {"period": "classical", "works": ["Republic", "Symposium", "Phaedo", "Apology"]},
        "Aristotle": {"period": "classical", "works": ["Nicomachean Ethics", "Politics", "Poetics", "Metaphysics"]},
        
        # Classical - Oratory
        "Demosthenes": {"period": "classical", "works": ["Philippics", "On the Crown"]},
        "Lysias": {"period": "classical", "works": ["Against Eratosthenes"]},
        "Isocrates": {"period": "classical", "works": ["Panegyricus"]},
        
        # Hellenistic
        "Polybius": {"period": "hellenistic", "works": ["Histories"]},
        "Apollonius": {"period": "hellenistic", "works": ["Argonautica"]},
        "Callimachus": {"period": "hellenistic", "works": ["Hymns", "Aetia"]},
        "Theocritus": {"period": "hellenistic", "works": ["Idylls"]},
        
        # Koine - New Testament
        "New Testament": {"period": "koine", "works": ["Gospels", "Acts", "Epistles", "Revelation"]},
        "Septuagint": {"period": "hellenistic", "works": ["Pentateuch", "Prophets", "Writings"]},
        
        # Roman Period
        "Plutarch": {"period": "roman", "works": ["Parallel Lives", "Moralia"]},
        "Epictetus": {"period": "roman", "works": ["Discourses", "Enchiridion"]},
        "Marcus Aurelius": {"period": "roman", "works": ["Meditations"]},
        "Lucian": {"period": "roman", "works": ["True History", "Dialogues"]},
        "Dio Chrysostom": {"period": "roman", "works": ["Orations"]},
        "Pausanias": {"period": "roman", "works": ["Description of Greece"]},
        "Galen": {"period": "roman", "works": ["Medical writings"]},
        
        # Late Antique - Patristic
        "Clement of Alexandria": {"period": "late_antique", "works": ["Stromata"]},
        "Origen": {"period": "late_antique", "works": ["Contra Celsum", "Hexapla"]},
        "Eusebius": {"period": "late_antique", "works": ["Ecclesiastical History"]},
        "Athanasius": {"period": "late_antique", "works": ["On the Incarnation"]},
        "Basil of Caesarea": {"period": "late_antique", "works": ["Hexaemeron"]},
        "Gregory of Nazianzus": {"period": "late_antique", "works": ["Orations"]},
        "Gregory of Nyssa": {"period": "late_antique", "works": ["Life of Moses"]},
        "John Chrysostom": {"period": "late_antique", "works": ["Homilies"]},
        
        # Byzantine
        "Procopius": {"period": "byzantine", "works": ["Wars", "Secret History"]},
        "Anna Comnena": {"period": "byzantine", "works": ["Alexiad"]},
        "Michael Psellus": {"period": "byzantine", "works": ["Chronographia"]}
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GreekToken:
    """Token with full PROIEL-style annotation"""
    id: int
    form: str
    lemma: str = ""
    pos: str = ""
    morphology: Dict[str, str] = field(default_factory=dict)
    head: int = 0
    relation: str = ""
    gloss: str = ""
    semantic_role: str = ""
    information_status: str = ""
    
    def to_proiel(self) -> Dict:
        return {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "part-of-speech": self.pos,
            "morphology": self.morphology,
            "head-id": self.head,
            "relation": self.relation,
            "gloss": self.gloss,
            "semantic-role": self.semantic_role,
            "information-status": self.information_status
        }


@dataclass
class GreekSentence:
    """Sentence with PROIEL annotation"""
    id: str
    text: str
    tokens: List[GreekToken] = field(default_factory=list)
    translation: str = ""
    notes: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class GreekDocument:
    """Document in the Greek corpus"""
    id: str
    title: str
    author: str
    period: str
    genre: str
    source: str
    sentences: List[GreekSentence] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        return sum(len(s.tokens) for s in self.sentences)
    
    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


@dataclass
class CollectionMetadata:
    """Metadata for tracking collection progress"""
    source: str
    author: str
    work: str
    status: str  # pending, collecting, collected, processing, complete, error
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    token_count: int = 0
    sentence_count: int = 0
    error_message: str = ""
    quality_score: float = 0.0


# ============================================================================
# GREEK MORPHOLOGY
# ============================================================================

class GreekMorphology:
    """Greek morphological analysis"""
    
    # PROIEL morphological categories
    CATEGORIES = {
        "person": ["1", "2", "3"],
        "number": ["s", "d", "p"],  # singular, dual, plural
        "tense": ["p", "i", "r", "l", "t", "f", "a"],  # present, imperfect, perfect, pluperfect, future perfect, future, aorist
        "mood": ["i", "s", "o", "n", "m", "p"],  # indicative, subjunctive, optative, infinitive, imperative, participle
        "voice": ["a", "m", "p"],  # active, middle, passive
        "gender": ["m", "f", "n"],
        "case": ["n", "g", "d", "a", "v"],  # nominative, genitive, dative, accusative, vocative
        "degree": ["p", "c", "s"]  # positive, comparative, superlative
    }
    
    # POS tags (PROIEL style)
    POS_TAGS = {
        "A-": "adjective",
        "Df": "adverb",
        "S-": "article",
        "Ma": "cardinal numeral",
        "Nb": "common noun",
        "C-": "conjunction",
        "Pd": "demonstrative pronoun",
        "F-": "foreign word",
        "Px": "indefinite pronoun",
        "N-": "infinitive marker",
        "I-": "interjection",
        "Du": "interrogative adverb",
        "Pi": "interrogative pronoun",
        "Mo": "ordinal numeral",
        "Pp": "personal pronoun",
        "Pk": "personal reflexive pronoun",
        "Ps": "possessive pronoun",
        "Pt": "possessive reflexive pronoun",
        "R-": "preposition",
        "Ne": "proper noun",
        "Py": "quantifier",
        "Pc": "reciprocal pronoun",
        "Dq": "relative adverb",
        "Pr": "relative pronoun",
        "G-": "subjunction",
        "V-": "verb",
        "X-": "unassigned"
    }
    
    # Common Greek endings for POS detection
    ENDINGS = {
        "verb": [
            "ω", "εις", "ει", "ομεν", "ετε", "ουσι", "ουσιν",  # present active
            "ον", "ες", "ε", "ομεν", "ετε", "ον",  # imperfect
            "α", "ας", "ε", "αμεν", "ατε", "αν",  # aorist
            "μαι", "σαι", "ται", "μεθα", "σθε", "νται",  # middle/passive
            "ειν", "αι", "σθαι",  # infinitives
            "ων", "ουσα", "ον", "ας", "ασα", "αν"  # participles
        ],
        "noun": [
            "ος", "ου", "ῳ", "ον", "ε",  # 2nd declension masc
            "η", "ης", "ῃ", "ην",  # 1st declension fem
            "α", "ας", "ᾳ", "αν",  # 1st declension fem (alpha)
            "ον", "ου", "ῳ",  # 2nd declension neut
            "ις", "εως", "ει", "ιν",  # 3rd declension
            "ευς", "εως", "ει", "εα",  # 3rd declension -ευς
            "μα", "ματος", "ματι"  # 3rd declension -μα
        ],
        "adjective": [
            "ος", "η", "ον",  # 1st/2nd declension
            "υς", "εια", "υ",  # -υς adjectives
            "ης", "ες",  # 3rd declension
            "ων", "ον",  # comparative
            "τατος", "τατη", "τατον",  # superlative
            "τερος", "τερα", "τερον"  # comparative
        ],
        "adverb": ["ως", "ῶς"],
        "article": ["ὁ", "ἡ", "τό", "οἱ", "αἱ", "τά", "τοῦ", "τῆς", "τῷ", "τήν", "τόν"],
        "preposition": ["ἐν", "εἰς", "ἐκ", "ἐξ", "ἀπό", "πρός", "παρά", "μετά", "διά", "ὑπό", "περί", "κατά", "ἐπί", "ὑπέρ", "πρό", "ἀντί", "σύν"]
    }
    
    def __init__(self):
        self.cache = {}
    
    def analyze(self, form: str) -> Dict:
        """Analyze Greek word form"""
        if form in self.cache:
            return self.cache[form]
        
        result = {
            "form": form,
            "pos": self._detect_pos(form),
            "morphology": self._detect_morphology(form)
        }
        
        self.cache[form] = result
        return result
    
    def _detect_pos(self, form: str) -> str:
        """Detect part of speech"""
        form_lower = form.lower()
        
        # Check articles
        if form_lower in self.ENDINGS["article"]:
            return "S-"
        
        # Check prepositions
        if form_lower in self.ENDINGS["preposition"]:
            return "R-"
        
        # Check verb endings
        for ending in self.ENDINGS["verb"]:
            if form_lower.endswith(ending):
                return "V-"
        
        # Check noun endings
        for ending in self.ENDINGS["noun"]:
            if form_lower.endswith(ending):
                return "Nb"
        
        # Check adjective endings
        for ending in self.ENDINGS["adjective"]:
            if form_lower.endswith(ending):
                return "A-"
        
        # Check adverb endings
        for ending in self.ENDINGS["adverb"]:
            if form_lower.endswith(ending):
                return "Df"
        
        return "X-"
    
    def _detect_morphology(self, form: str) -> Dict:
        """Detect morphological features"""
        morph = {}
        form_lower = form.lower()
        
        # Detect case from endings
        if form_lower.endswith(("ος", "ης", "υς", "ις", "ευς")):
            morph["case"] = "n"  # nominative
        elif form_lower.endswith(("ου", "ης", "εως", "ους")):
            morph["case"] = "g"  # genitive
        elif form_lower.endswith(("ῳ", "ῃ", "ει", "ι")):
            morph["case"] = "d"  # dative
        elif form_lower.endswith(("ον", "ην", "αν", "α", "ιν", "εα", "υν")):
            morph["case"] = "a"  # accusative
        
        # Detect number
        if form_lower.endswith(("οι", "αι", "α", "ες", "εις", "ων", "ους", "ας")):
            morph["number"] = "p"  # plural
        else:
            morph["number"] = "s"  # singular
        
        # Detect gender from article or ending patterns
        if form_lower.startswith(("ὁ", "τοῦ", "τῷ", "τόν", "οἱ", "τῶν", "τοῖς", "τούς")):
            morph["gender"] = "m"
        elif form_lower.startswith(("ἡ", "τῆς", "τῇ", "τήν", "αἱ", "ταῖς", "τάς")):
            morph["gender"] = "f"
        elif form_lower.startswith(("τό", "τά")):
            morph["gender"] = "n"
        
        return morph


# ============================================================================
# GREEK VALENCY
# ============================================================================

class GreekValency:
    """Greek verbal valency analysis"""
    
    # Common Greek valency patterns
    PATTERNS = {
        "intransitive": {
            "pattern": "NOM",
            "description": "Subject only",
            "examples": ["βαίνω", "ἔρχομαι", "πίπτω"]
        },
        "transitive": {
            "pattern": "NOM+ACC",
            "description": "Subject + direct object",
            "examples": ["λέγω", "ποιέω", "ἔχω", "λαμβάνω"]
        },
        "ditransitive": {
            "pattern": "NOM+ACC+DAT",
            "description": "Subject + direct object + indirect object",
            "examples": ["δίδωμι", "λέγω"]
        },
        "genitive_object": {
            "pattern": "NOM+GEN",
            "description": "Subject + genitive object",
            "examples": ["ἀκούω", "ἅπτομαι", "μέμνημαι"]
        },
        "dative_object": {
            "pattern": "NOM+DAT",
            "description": "Subject + dative object",
            "examples": ["πείθομαι", "ἕπομαι", "βοηθέω"]
        },
        "double_accusative": {
            "pattern": "NOM+ACC+ACC",
            "description": "Subject + two accusative objects",
            "examples": ["διδάσκω", "ἐρωτάω"]
        },
        "accusative_genitive": {
            "pattern": "NOM+ACC+GEN",
            "description": "Subject + accusative + genitive",
            "examples": ["πληρόω", "ἀξιόω"]
        },
        "copular": {
            "pattern": "NOM+NOM",
            "description": "Subject + predicate nominative",
            "examples": ["εἰμί", "γίγνομαι"]
        }
    }
    
    # Semantic roles
    SEMANTIC_ROLES = {
        "A": "Agent",
        "P": "Patient",
        "R": "Recipient",
        "B": "Beneficiary",
        "I": "Instrument",
        "L": "Location",
        "S": "Source",
        "G": "Goal",
        "T": "Theme",
        "E": "Experiencer",
        "C": "Cause"
    }
    
    def __init__(self):
        self.lexicon = {}
    
    def extract_valency(self, sentence: GreekSentence) -> List[Dict]:
        """Extract valency patterns from sentence"""
        patterns = []
        
        for token in sentence.tokens:
            if token.pos == "V-":
                pattern = self._analyze_verb_frame(token, sentence)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_verb_frame(self, verb: GreekToken, sentence: GreekSentence) -> Optional[Dict]:
        """Analyze verbal argument frame"""
        arguments = []
        
        for token in sentence.tokens:
            if token.head == verb.id:
                arg = {
                    "form": token.form,
                    "lemma": token.lemma,
                    "relation": token.relation,
                    "case": token.morphology.get("case", ""),
                    "role": token.semantic_role
                }
                arguments.append(arg)
        
        if not arguments:
            return None
        
        # Determine pattern
        cases = [a["case"] for a in arguments if a["case"]]
        pattern = self._determine_pattern(cases)
        
        return {
            "verb_form": verb.form,
            "verb_lemma": verb.lemma,
            "pattern": pattern,
            "arguments": arguments,
            "sentence_id": sentence.id
        }
    
    def _determine_pattern(self, cases: List[str]) -> str:
        """Determine valency pattern from cases"""
        case_set = set(cases)
        
        if case_set == {"n"}:
            return "NOM"
        elif case_set == {"n", "a"}:
            return "NOM+ACC"
        elif case_set == {"n", "g"}:
            return "NOM+GEN"
        elif case_set == {"n", "d"}:
            return "NOM+DAT"
        elif case_set == {"n", "a", "d"}:
            return "NOM+ACC+DAT"
        elif case_set == {"n", "a", "g"}:
            return "NOM+ACC+GEN"
        else:
            return "+".join(sorted(cases)).upper()


# ============================================================================
# DATABASE
# ============================================================================

class GreekCorpusDatabase:
    """Database for Greek corpus"""
    
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
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                period TEXT,
                genre TEXT,
                source TEXT,
                sentence_count INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
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
                notes TEXT,
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
                gloss TEXT,
                semantic_role TEXT,
                information_status TEXT,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id)
            )
        """)
        
        # Valency lexicon
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_lexicon (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                examples TEXT,
                period TEXT,
                semantic_class TEXT,
                notes TEXT,
                UNIQUE(lemma, pattern)
            )
        """)
        
        # Collection metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                author TEXT,
                work TEXT,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                token_count INTEGER DEFAULT 0,
                sentence_count INTEGER DEFAULT 0,
                error_message TEXT,
                quality_score REAL DEFAULT 0.0
            )
        """)
        
        # Processing log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                target TEXT,
                status TEXT,
                details TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_author ON documents(author)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_period ON documents(period)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sents_doc ON sentences(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tokens_sent ON tokens(sentence_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_valency_lemma ON valency_lexicon(lemma)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def save_document(self, doc: GreekDocument) -> bool:
        """Save document to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Save document
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, author, period, genre, source, sentence_count, token_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.author, doc.period, doc.genre, doc.source,
                doc.sentence_count, doc.token_count, json.dumps(doc.metadata)
            ))
            
            # Save sentences and tokens
            for idx, sentence in enumerate(doc.sentences):
                cursor.execute("""
                    INSERT OR REPLACE INTO sentences 
                    (id, document_id, sentence_index, text, translation, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sentence.id, doc.id, idx, sentence.text,
                    sentence.translation, sentence.notes, json.dumps(sentence.metadata)
                ))
                
                # Save tokens
                for token in sentence.tokens:
                    cursor.execute("""
                        INSERT INTO tokens 
                        (sentence_id, token_index, form, lemma, pos, morphology, 
                         head, relation, gloss, semantic_role, information_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sentence.id, token.id, token.form, token.lemma, token.pos,
                        json.dumps(token.morphology), token.head, token.relation,
                        token.gloss, token.semantic_role, token.information_status
                    ))
            
            conn.commit()
            conn.close()
            
            self._log_action("save_document", doc.id, "success", f"Saved {doc.sentence_count} sentences")
            return True
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            self._log_action("save_document", doc.id if doc else "", "error", str(e))
            return False
    
    def get_statistics(self) -> Dict:
        """Get corpus statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats["total_documents"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(sentence_count) FROM documents")
        stats["total_sentences"] = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT SUM(token_count) FROM documents")
        stats["total_tokens"] = cursor.fetchone()[0] or 0
        
        # By period
        cursor.execute("""
            SELECT period, COUNT(*) as docs, SUM(token_count) as tokens
            FROM documents WHERE period IS NOT NULL
            GROUP BY period
        """)
        stats["by_period"] = {
            row["period"]: {"documents": row["docs"], "tokens": row["tokens"] or 0}
            for row in cursor.fetchall()
        }
        
        # By author
        cursor.execute("""
            SELECT author, COUNT(*) as docs, SUM(token_count) as tokens
            FROM documents WHERE author IS NOT NULL
            GROUP BY author ORDER BY tokens DESC LIMIT 20
        """)
        stats["top_authors"] = {
            row["author"]: {"documents": row["docs"], "tokens": row["tokens"] or 0}
            for row in cursor.fetchall()
        }
        
        # By genre
        cursor.execute("""
            SELECT genre, COUNT(*) as docs, SUM(token_count) as tokens
            FROM documents WHERE genre IS NOT NULL
            GROUP BY genre
        """)
        stats["by_genre"] = {
            row["genre"]: {"documents": row["docs"], "tokens": row["tokens"] or 0}
            for row in cursor.fetchall()
        }
        
        # Valency statistics
        cursor.execute("SELECT COUNT(DISTINCT lemma) FROM valency_lexicon")
        stats["valency_verbs"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT pattern, COUNT(*) FROM valency_lexicon GROUP BY pattern")
        stats["valency_patterns"] = {row["pattern"]: row[1] for row in cursor.fetchall()}
        
        # Collection status
        cursor.execute("""
            SELECT status, COUNT(*) FROM collection_metadata GROUP BY status
        """)
        stats["collection_status"] = {row["status"]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats
    
    def _log_action(self, action: str, target: str, status: str, details: str = ""):
        """Log processing action"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_log (action, target, status, details)
                VALUES (?, ?, ?, ?)
            """, (action, target, status, details))
            conn.commit()
            conn.close()
        except:
            pass


# ============================================================================
# METADATA MONITOR
# ============================================================================

class MetadataMonitor:
    """Monitor collection and processing metadata"""
    
    def __init__(self, db: GreekCorpusDatabase):
        self.db = db
        self.start_time = datetime.now()
    
    def get_current_status(self) -> Dict:
        """Get current processing status"""
        stats = self.db.get_statistics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "corpus_stats": stats,
            "collection_progress": self._get_collection_progress(),
            "recent_activity": self._get_recent_activity()
        }
    
    def _get_collection_progress(self) -> Dict:
        """Get collection progress"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source, 
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as complete,
                   SUM(token_count) as tokens
            FROM collection_metadata
            GROUP BY source
        """)
        
        progress = {}
        for row in cursor.fetchall():
            progress[row["source"]] = {
                "total": row["total"],
                "complete": row["complete"],
                "tokens": row["tokens"] or 0,
                "percent": (row["complete"] / row["total"] * 100) if row["total"] > 0 else 0
            }
        
        conn.close()
        return progress
    
    def _get_recent_activity(self, limit: int = 20) -> List[Dict]:
        """Get recent processing activity"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, action, target, status, details
            FROM processing_log
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        activity = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return activity


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Greek Corpus Builder")
    parser.add_argument('command', choices=['init', 'stats', 'monitor', 'collect', 'process'],
                       help="Command to run")
    parser.add_argument('--author', help="Author to collect")
    parser.add_argument('--period', help="Period to collect")
    parser.add_argument('--source', help="Source to use")
    
    args = parser.parse_args()
    
    db = GreekCorpusDatabase()
    
    if args.command == 'init':
        print("Database initialized")
        print(json.dumps(GREEK_CORPUS_CONFIG, indent=2, ensure_ascii=False))
    
    elif args.command == 'stats':
        stats = db.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.command == 'monitor':
        monitor = MetadataMonitor(db)
        status = monitor.get_current_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    elif args.command == 'collect':
        print(f"Would collect from: {args.source or 'all sources'}")
        print(f"Author: {args.author or 'all'}")
        print(f"Period: {args.period or 'all'}")
    
    elif args.command == 'process':
        print("Processing collected texts...")


if __name__ == "__main__":
    main()
