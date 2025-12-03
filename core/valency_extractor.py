"""
Valency Extractor - Automatic extraction of verb valency patterns from annotated corpora
Builds valency lexicon from PROIEL/UD annotated texts

Features:
- Extract argument structures from dependency trees
- Identify valency patterns (NOM, NOM+ACC, NOM+DAT, etc.)
- Track frequency by period
- Detect valency alternations
- Generate valency frames for SRL
"""

import os
import re
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CASE AND RELATION MAPPINGS
# =============================================================================

class Case(Enum):
    """Grammatical cases"""
    NOM = "nominative"
    GEN = "genitive"
    DAT = "dative"
    ACC = "accusative"
    VOC = "vocative"
    INS = "instrumental"
    LOC = "locative"
    ABL = "ablative"

class ArgumentType(Enum):
    """Argument types"""
    SUBJECT = "subject"
    DIRECT_OBJECT = "direct_object"
    INDIRECT_OBJECT = "indirect_object"
    OBLIQUE = "oblique"
    COMPLEMENT = "complement"
    PREDICATE = "predicate"

# UD to Case mapping
UD_CASE_MAP = {
    'Nom': Case.NOM,
    'Gen': Case.GEN,
    'Dat': Case.DAT,
    'Acc': Case.ACC,
    'Voc': Case.VOC,
    'Ins': Case.INS,
    'Loc': Case.LOC,
    'Abl': Case.ABL
}

# UD relations to argument types
UD_RELATION_MAP = {
    'nsubj': ArgumentType.SUBJECT,
    'nsubj:pass': ArgumentType.SUBJECT,
    'obj': ArgumentType.DIRECT_OBJECT,
    'iobj': ArgumentType.INDIRECT_OBJECT,
    'obl': ArgumentType.OBLIQUE,
    'obl:arg': ArgumentType.OBLIQUE,
    'xcomp': ArgumentType.COMPLEMENT,
    'ccomp': ArgumentType.COMPLEMENT,
    'cop': ArgumentType.PREDICATE
}

# PROIEL relations to argument types
PROIEL_RELATION_MAP = {
    'sub': ArgumentType.SUBJECT,
    'obj': ArgumentType.DIRECT_OBJECT,
    'obl': ArgumentType.OBLIQUE,
    'ag': ArgumentType.SUBJECT,  # Agent in passive
    'xobj': ArgumentType.COMPLEMENT,
    'comp': ArgumentType.COMPLEMENT,
    'pid': ArgumentType.PREDICATE
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Argument:
    """A verb argument"""
    arg_type: str
    case: str
    lemma: str
    form: str
    relation: str
    preposition: Optional[str] = None
    
    def to_pattern_string(self) -> str:
        """Convert to pattern string like 'ACC' or 'πρός+ACC'"""
        if self.preposition:
            return f"{self.preposition}+{self.case}"
        return self.case

@dataclass
class ValencyInstance:
    """A single instance of a verb with its arguments"""
    verb_lemma: str
    verb_form: str
    arguments: List[Argument]
    sentence_id: str
    sentence_text: str
    period: str = ""
    source: str = ""
    
    def get_pattern(self) -> str:
        """Get valency pattern string"""
        parts = ['NOM']  # Subject always implied
        
        for arg in sorted(self.arguments, key=lambda a: a.arg_type):
            if arg.arg_type == 'subject':
                continue
            parts.append(arg.to_pattern_string())
        
        return '+'.join(parts)

@dataclass
class ValencyFrame:
    """A valency frame for a verb sense"""
    lemma: str
    pattern: str
    arguments: List[Dict]
    frequency: int = 1
    periods: Dict[str, int] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    semantic_class: str = ""
    alternations: List[str] = field(default_factory=list)
    
    def add_instance(self, instance: ValencyInstance):
        """Add an instance to this frame"""
        self.frequency += 1
        
        if instance.period:
            self.periods[instance.period] = self.periods.get(instance.period, 0) + 1
        
        if len(self.examples) < 5:
            self.examples.append(instance.sentence_text[:200])

@dataclass
class ValencyEntry:
    """Complete valency entry for a verb lemma"""
    lemma: str
    frames: Dict[str, ValencyFrame] = field(default_factory=dict)
    total_frequency: int = 0
    
    def add_frame(self, pattern: str, frame: ValencyFrame):
        """Add or merge a frame"""
        if pattern in self.frames:
            self.frames[pattern].frequency += frame.frequency
            for period, count in frame.periods.items():
                self.frames[pattern].periods[period] = \
                    self.frames[pattern].periods.get(period, 0) + count
            self.frames[pattern].examples.extend(frame.examples[:3])
        else:
            self.frames[pattern] = frame
        
        self.total_frequency += frame.frequency
    
    def get_primary_pattern(self) -> str:
        """Get most frequent pattern"""
        if not self.frames:
            return "NOM"
        return max(self.frames.items(), key=lambda x: x[1].frequency)[0]

# =============================================================================
# VALENCY EXTRACTOR
# =============================================================================

class ValencyExtractor:
    """Extract valency patterns from annotated sentences"""
    
    def __init__(self):
        self.entries: Dict[str, ValencyEntry] = {}
        self.instances: List[ValencyInstance] = []
    
    def extract_from_sentence(self, tokens: List[Dict], sentence_id: str = "",
                             sentence_text: str = "", period: str = "",
                             source: str = "") -> List[ValencyInstance]:
        """Extract valency instances from a sentence"""
        instances = []
        
        # Find all verbs
        verbs = [t for t in tokens if self._is_verb(t)]
        
        for verb in verbs:
            # Get arguments for this verb
            arguments = self._get_arguments(verb, tokens)
            
            if arguments:
                instance = ValencyInstance(
                    verb_lemma=verb.get('lemma', verb.get('form', '')),
                    verb_form=verb.get('form', ''),
                    arguments=arguments,
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    period=period,
                    source=source
                )
                instances.append(instance)
                self._add_instance(instance)
        
        return instances
    
    def _is_verb(self, token: Dict) -> bool:
        """Check if token is a verb"""
        pos = token.get('upos', token.get('pos', ''))
        return pos in ('VERB', 'AUX', 'V-', 'V')
    
    def _get_arguments(self, verb: Dict, tokens: List[Dict]) -> List[Argument]:
        """Get arguments for a verb"""
        arguments = []
        verb_id = self._get_token_id(verb)
        
        for token in tokens:
            head = token.get('head', 0)
            
            # Check if this token is a dependent of the verb
            if head == verb_id:
                relation = token.get('deprel', token.get('relation', ''))
                
                # Check if it's an argument relation
                arg_type = self._get_argument_type(relation)
                if arg_type:
                    case = self._get_case(token)
                    prep = self._get_preposition(token, tokens)
                    
                    arg = Argument(
                        arg_type=arg_type,
                        case=case,
                        lemma=token.get('lemma', ''),
                        form=token.get('form', ''),
                        relation=relation,
                        preposition=prep
                    )
                    arguments.append(arg)
        
        return arguments
    
    def _get_token_id(self, token: Dict) -> int:
        """Get token ID"""
        token_id = token.get('id', token.get('token_index', 0))
        if isinstance(token_id, str):
            try:
                return int(token_id)
            except:
                return 0
        return token_id
    
    def _get_argument_type(self, relation: str) -> Optional[str]:
        """Map relation to argument type"""
        # UD relations
        if relation in UD_RELATION_MAP:
            return UD_RELATION_MAP[relation].value
        
        # PROIEL relations
        if relation in PROIEL_RELATION_MAP:
            return PROIEL_RELATION_MAP[relation].value
        
        # Check for argument-like relations
        if relation in ('nsubj', 'obj', 'iobj', 'obl', 'sub', 'ag'):
            return relation
        
        return None
    
    def _get_case(self, token: Dict) -> str:
        """Get case from token features"""
        # Check feats field
        feats = token.get('feats', token.get('morphology', ''))
        
        if isinstance(feats, str) and feats != '_':
            # Parse UD-style features
            for feat in feats.split('|'):
                if feat.startswith('Case='):
                    case_val = feat.split('=')[1]
                    return case_val.upper()[:3]
        
        # Check xpos for PROIEL-style morphology
        xpos = token.get('xpos', '')
        if len(xpos) >= 8:
            # PROIEL morphology position 7 is case
            case_char = xpos[7] if len(xpos) > 7 else ''
            case_map = {'n': 'NOM', 'g': 'GEN', 'd': 'DAT', 'a': 'ACC', 'v': 'VOC'}
            return case_map.get(case_char, 'UNK')
        
        return 'UNK'
    
    def _get_preposition(self, token: Dict, tokens: List[Dict]) -> Optional[str]:
        """Get governing preposition if any"""
        token_id = self._get_token_id(token)
        
        for t in tokens:
            if t.get('head') == token_id:
                rel = t.get('deprel', t.get('relation', ''))
                pos = t.get('upos', t.get('pos', ''))
                
                if rel == 'case' or pos in ('ADP', 'R-'):
                    return t.get('lemma', t.get('form', ''))
        
        return None
    
    def _add_instance(self, instance: ValencyInstance):
        """Add instance to entries"""
        self.instances.append(instance)
        
        lemma = instance.verb_lemma
        pattern = instance.get_pattern()
        
        if lemma not in self.entries:
            self.entries[lemma] = ValencyEntry(lemma=lemma)
        
        if pattern not in self.entries[lemma].frames:
            self.entries[lemma].frames[pattern] = ValencyFrame(
                lemma=lemma,
                pattern=pattern,
                arguments=[asdict(a) for a in instance.arguments]
            )
        
        self.entries[lemma].frames[pattern].add_instance(instance)
        self.entries[lemma].total_frequency += 1
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics"""
        return {
            'total_verbs': len(self.entries),
            'total_instances': len(self.instances),
            'total_patterns': sum(len(e.frames) for e in self.entries.values()),
            'top_verbs': sorted(
                [(l, e.total_frequency) for l, e in self.entries.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'pattern_distribution': self._get_pattern_distribution()
        }
    
    def _get_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of patterns"""
        dist = defaultdict(int)
        for entry in self.entries.values():
            for pattern, frame in entry.frames.items():
                dist[pattern] += frame.frequency
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:20])


# =============================================================================
# VALENCY DATABASE
# =============================================================================

class ValencyDatabase:
    """Database for storing valency lexicon"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Valency entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL UNIQUE,
                total_frequency INTEGER DEFAULT 0,
                primary_pattern TEXT,
                semantic_classes TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Valency frames table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                pattern TEXT NOT NULL,
                arguments TEXT,
                frequency INTEGER DEFAULT 1,
                periods TEXT,
                examples TEXT,
                semantic_class TEXT,
                alternations TEXT,
                FOREIGN KEY (entry_id) REFERENCES valency_entries(id),
                UNIQUE(entry_id, pattern)
            )
        """)
        
        # Valency instances table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS valency_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                frame_id INTEGER,
                verb_form TEXT,
                sentence_id TEXT,
                sentence_text TEXT,
                period TEXT,
                source TEXT,
                arguments TEXT,
                FOREIGN KEY (entry_id) REFERENCES valency_entries(id),
                FOREIGN KEY (frame_id) REFERENCES valency_frames(id)
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_lemma ON valency_entries(lemma)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_pattern ON valency_frames(pattern)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_instance_period ON valency_instances(period)")
        
        conn.commit()
        conn.close()
    
    def store_entry(self, entry: ValencyEntry):
        """Store a valency entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert or update entry
            cursor.execute("""
                INSERT OR REPLACE INTO valency_entries
                (lemma, total_frequency, primary_pattern, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                entry.lemma,
                entry.total_frequency,
                entry.get_primary_pattern(),
                json.dumps({'frame_count': len(entry.frames)})
            ))
            
            entry_id = cursor.lastrowid
            
            # Insert frames
            for pattern, frame in entry.frames.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO valency_frames
                    (entry_id, pattern, arguments, frequency, periods, examples, semantic_class, alternations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    pattern,
                    json.dumps(frame.arguments),
                    frame.frequency,
                    json.dumps(frame.periods),
                    json.dumps(frame.examples[:5]),
                    frame.semantic_class,
                    json.dumps(frame.alternations)
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing entry {entry.lemma}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def store_extractor_results(self, extractor: ValencyExtractor):
        """Store all results from an extractor"""
        for lemma, entry in extractor.entries.items():
            self.store_entry(entry)
        
        logger.info(f"Stored {len(extractor.entries)} valency entries")
    
    def get_entry(self, lemma: str) -> Optional[Dict]:
        """Get valency entry for a lemma"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM valency_entries WHERE lemma = ?
        """, (lemma,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        entry = dict(row)
        entry_id = entry['id']
        
        # Get frames
        cursor.execute("""
            SELECT * FROM valency_frames WHERE entry_id = ?
        """, (entry_id,))
        
        entry['frames'] = []
        for frame_row in cursor.fetchall():
            frame = dict(frame_row)
            frame['arguments'] = json.loads(frame['arguments']) if frame['arguments'] else []
            frame['periods'] = json.loads(frame['periods']) if frame['periods'] else {}
            frame['examples'] = json.loads(frame['examples']) if frame['examples'] else []
            frame['alternations'] = json.loads(frame['alternations']) if frame['alternations'] else []
            entry['frames'].append(frame)
        
        conn.close()
        return entry
    
    def search_by_pattern(self, pattern: str) -> List[Dict]:
        """Search entries by pattern"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT e.lemma, e.total_frequency, f.pattern, f.frequency
            FROM valency_entries e
            JOIN valency_frames f ON e.id = f.entry_id
            WHERE f.pattern LIKE ?
            ORDER BY f.frequency DESC
            LIMIT 100
        """, (f"%{pattern}%",))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM valency_entries")
        stats['entry_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM valency_frames")
        stats['frame_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT pattern) FROM valency_frames")
        stats['pattern_count'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT pattern, SUM(frequency) as total
            FROM valency_frames
            GROUP BY pattern
            ORDER BY total DESC
            LIMIT 10
        """)
        stats['top_patterns'] = [(row[0], row[1]) for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT lemma, total_frequency
            FROM valency_entries
            ORDER BY total_frequency DESC
            LIMIT 10
        """)
        stats['top_verbs'] = [(row[0], row[1]) for row in cursor.fetchall()]
        
        conn.close()
        return stats


# =============================================================================
# CORPUS PROCESSOR
# =============================================================================

class CorpusValencyProcessor:
    """Process corpus to extract valency"""
    
    def __init__(self, corpus_db_path: str, valency_db_path: str):
        self.corpus_db_path = corpus_db_path
        self.valency_db = ValencyDatabase(valency_db_path)
        self.extractor = ValencyExtractor()
    
    def process_corpus(self) -> Dict:
        """Process entire corpus"""
        conn = sqlite3.connect(self.corpus_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        stats = {
            'documents_processed': 0,
            'sentences_processed': 0,
            'instances_extracted': 0
        }
        
        # Get all documents
        cursor.execute("""
            SELECT id, title, period, language FROM documents
            WHERE language IN ('grc', 'el', 'la')
        """)
        
        documents = cursor.fetchall()
        
        for doc in documents:
            doc_id = doc['id']
            period = doc['period'] or 'unknown'
            
            # Get sentences for this document
            cursor.execute("""
                SELECT id, text, tokens FROM sentences
                WHERE document_id = ?
            """, (doc_id,))
            
            for sent_row in cursor.fetchall():
                sent_id = sent_row['id']
                sent_text = sent_row['text']
                
                # Get tokens
                cursor.execute("""
                    SELECT * FROM tokens WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_id,))
                
                tokens = [dict(row) for row in cursor.fetchall()]
                
                if tokens:
                    instances = self.extractor.extract_from_sentence(
                        tokens=tokens,
                        sentence_id=sent_id,
                        sentence_text=sent_text,
                        period=period,
                        source=doc_id
                    )
                    stats['instances_extracted'] += len(instances)
                
                stats['sentences_processed'] += 1
            
            stats['documents_processed'] += 1
            
            if stats['documents_processed'] % 10 == 0:
                logger.info(f"Processed {stats['documents_processed']} documents, "
                           f"{stats['instances_extracted']} instances")
        
        conn.close()
        
        # Store results
        self.valency_db.store_extractor_results(self.extractor)
        
        stats['extractor_stats'] = self.extractor.get_statistics()
        stats['database_stats'] = self.valency_db.get_statistics()
        
        return stats


# =============================================================================
# PREDEFINED GREEK VALENCY LEXICON
# =============================================================================

GREEK_VALENCY_LEXICON = {
    # Transfer verbs
    "δίδωμι": {
        "gloss": "give",
        "frames": [
            {"pattern": "NOM+ACC+DAT", "args": ["giver", "gift", "recipient"], "class": "transfer"},
            {"pattern": "NOM+ACC", "args": ["giver", "gift"], "class": "transfer"}
        ]
    },
    "λαμβάνω": {
        "gloss": "take, receive",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["receiver", "thing"], "class": "transfer"},
            {"pattern": "NOM+ACC+παρά+GEN", "args": ["receiver", "thing", "source"], "class": "transfer"}
        ]
    },
    "φέρω": {
        "gloss": "carry, bring",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["carrier", "thing"], "class": "motion"},
            {"pattern": "NOM+ACC+DAT", "args": ["carrier", "thing", "goal"], "class": "transfer"}
        ]
    },
    
    # Communication verbs
    "λέγω": {
        "gloss": "say, speak",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["speaker", "content"], "class": "communication"},
            {"pattern": "NOM+ὅτι-clause", "args": ["speaker", "content"], "class": "communication"},
            {"pattern": "NOM+ACC+DAT", "args": ["speaker", "content", "addressee"], "class": "communication"}
        ]
    },
    "φημί": {
        "gloss": "say, affirm",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["speaker", "content"], "class": "communication"},
            {"pattern": "NOM+ACI", "args": ["speaker", "content"], "class": "communication"}
        ]
    },
    "ἐρωτάω": {
        "gloss": "ask",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["asker", "question"], "class": "communication"},
            {"pattern": "NOM+ACC+ACC", "args": ["asker", "person", "question"], "class": "communication"}
        ]
    },
    "κελεύω": {
        "gloss": "order, command",
        "frames": [
            {"pattern": "NOM+ACC+INF", "args": ["commander", "commanded", "action"], "class": "communication"}
        ]
    },
    
    # Motion verbs
    "ἔρχομαι": {
        "gloss": "come, go",
        "frames": [
            {"pattern": "NOM", "args": ["mover"], "class": "motion"},
            {"pattern": "NOM+εἰς+ACC", "args": ["mover", "goal"], "class": "motion"},
            {"pattern": "NOM+ἐκ+GEN", "args": ["mover", "source"], "class": "motion"}
        ]
    },
    "ἄγω": {
        "gloss": "lead, bring",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["leader", "led"], "class": "motion"},
            {"pattern": "NOM+ACC+εἰς+ACC", "args": ["leader", "led", "goal"], "class": "motion"}
        ]
    },
    "πέμπω": {
        "gloss": "send",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["sender", "sent"], "class": "motion"},
            {"pattern": "NOM+ACC+εἰς+ACC", "args": ["sender", "sent", "goal"], "class": "motion"},
            {"pattern": "NOM+ACC+πρός+ACC", "args": ["sender", "sent", "recipient"], "class": "motion"}
        ]
    },
    "βαίνω": {
        "gloss": "go, walk",
        "frames": [
            {"pattern": "NOM", "args": ["mover"], "class": "motion"},
            {"pattern": "NOM+εἰς+ACC", "args": ["mover", "goal"], "class": "motion"}
        ]
    },
    
    # Perception verbs
    "ὁράω": {
        "gloss": "see",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["perceiver", "perceived"], "class": "perception"},
            {"pattern": "NOM+ACC+PART", "args": ["perceiver", "perceived", "state"], "class": "perception"}
        ]
    },
    "ἀκούω": {
        "gloss": "hear",
        "frames": [
            {"pattern": "NOM+GEN", "args": ["hearer", "source"], "class": "perception"},
            {"pattern": "NOM+ACC", "args": ["hearer", "content"], "class": "perception"},
            {"pattern": "NOM+GEN+GEN", "args": ["hearer", "source", "content"], "class": "perception"}
        ]
    },
    "γιγνώσκω": {
        "gloss": "know, recognize",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["knower", "known"], "class": "cognition"},
            {"pattern": "NOM+ὅτι-clause", "args": ["knower", "content"], "class": "cognition"}
        ]
    },
    
    # Causative/change verbs
    "ποιέω": {
        "gloss": "make, do",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["agent", "product"], "class": "creation"},
            {"pattern": "NOM+ACC+ACC", "args": ["agent", "patient", "result"], "class": "causative"}
        ]
    },
    "τίθημι": {
        "gloss": "put, place",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["agent", "theme"], "class": "placement"},
            {"pattern": "NOM+ACC+εἰς+ACC", "args": ["agent", "theme", "location"], "class": "placement"}
        ]
    },
    "καθίστημι": {
        "gloss": "set up, establish",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["agent", "theme"], "class": "causative"},
            {"pattern": "NOM+ACC+ACC", "args": ["agent", "patient", "result"], "class": "causative"}
        ]
    },
    
    # Psych verbs
    "φοβέομαι": {
        "gloss": "fear",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["experiencer", "stimulus"], "class": "psych"},
            {"pattern": "NOM+μή-clause", "args": ["experiencer", "content"], "class": "psych"}
        ]
    },
    "βούλομαι": {
        "gloss": "want, wish",
        "frames": [
            {"pattern": "NOM+INF", "args": ["experiencer", "content"], "class": "psych"},
            {"pattern": "NOM+ACC", "args": ["experiencer", "desired"], "class": "psych"}
        ]
    },
    "ἐθέλω": {
        "gloss": "want, be willing",
        "frames": [
            {"pattern": "NOM+INF", "args": ["experiencer", "content"], "class": "psych"}
        ]
    },
    
    # Social verbs
    "πείθω": {
        "gloss": "persuade",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["persuader", "persuaded"], "class": "social"},
            {"pattern": "NOM+ACC+INF", "args": ["persuader", "persuaded", "action"], "class": "social"}
        ]
    },
    "πείθομαι": {
        "gloss": "obey, trust",
        "frames": [
            {"pattern": "NOM+DAT", "args": ["obeyer", "authority"], "class": "social"}
        ]
    },
    "κωλύω": {
        "gloss": "prevent, hinder",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["preventer", "prevented"], "class": "social"},
            {"pattern": "NOM+ACC+INF", "args": ["preventer", "prevented", "action"], "class": "social"},
            {"pattern": "NOM+ACC+GEN", "args": ["preventer", "prevented", "source"], "class": "social"}
        ]
    },
    
    # Existence/state verbs
    "εἰμί": {
        "gloss": "be",
        "frames": [
            {"pattern": "NOM", "args": ["subject"], "class": "existence"},
            {"pattern": "NOM+NOM", "args": ["subject", "predicate"], "class": "copula"},
            {"pattern": "NOM+DAT", "args": ["possessed", "possessor"], "class": "possession"}
        ]
    },
    "γίγνομαι": {
        "gloss": "become, happen",
        "frames": [
            {"pattern": "NOM", "args": ["theme"], "class": "change"},
            {"pattern": "NOM+NOM", "args": ["theme", "result"], "class": "change"}
        ]
    },
    "ἔχω": {
        "gloss": "have, hold",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["possessor", "possessed"], "class": "possession"},
            {"pattern": "NOM+ADV", "args": ["subject", "state"], "class": "state"}
        ]
    },
    
    # Teaching/learning
    "διδάσκω": {
        "gloss": "teach",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["teacher", "student"], "class": "transfer"},
            {"pattern": "NOM+ACC+ACC", "args": ["teacher", "student", "subject"], "class": "transfer"},
            {"pattern": "NOM+ACC+INF", "args": ["teacher", "student", "skill"], "class": "transfer"}
        ]
    },
    "μανθάνω": {
        "gloss": "learn",
        "frames": [
            {"pattern": "NOM+ACC", "args": ["learner", "content"], "class": "cognition"},
            {"pattern": "NOM+INF", "args": ["learner", "skill"], "class": "cognition"},
            {"pattern": "NOM+παρά+GEN", "args": ["learner", "teacher"], "class": "cognition"}
        ]
    }
}


def populate_predefined_lexicon(db_path: str):
    """Populate database with predefined lexicon"""
    db = ValencyDatabase(db_path)
    
    for lemma, data in GREEK_VALENCY_LEXICON.items():
        entry = ValencyEntry(lemma=lemma)
        
        for frame_data in data['frames']:
            pattern = frame_data['pattern']
            frame = ValencyFrame(
                lemma=lemma,
                pattern=pattern,
                arguments=[{'role': arg} for arg in frame_data['args']],
                semantic_class=frame_data.get('class', ''),
                frequency=100  # Base frequency for predefined
            )
            entry.frames[pattern] = frame
            entry.total_frequency += 100
        
        db.store_entry(entry)
    
    logger.info(f"Populated {len(GREEK_VALENCY_LEXICON)} predefined entries")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/root/corpus_platform/data"
    
    valency_db_path = f"{data_dir}/valency_lexicon.db"
    corpus_db_path = f"{data_dir}/corpus_platform.db"
    
    # Populate predefined lexicon
    print("Populating predefined valency lexicon...")
    populate_predefined_lexicon(valency_db_path)
    
    # Process corpus if exists
    if os.path.exists(corpus_db_path):
        print("\nProcessing corpus for valency extraction...")
        processor = CorpusValencyProcessor(corpus_db_path, valency_db_path)
        stats = processor.process_corpus()
        
        print("\n" + "=" * 60)
        print("VALENCY EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Sentences processed: {stats['sentences_processed']}")
        print(f"Instances extracted: {stats['instances_extracted']}")
        
        if 'database_stats' in stats:
            db_stats = stats['database_stats']
            print(f"\nDatabase statistics:")
            print(f"  Entries: {db_stats['entry_count']}")
            print(f"  Frames: {db_stats['frame_count']}")
            print(f"  Patterns: {db_stats['pattern_count']}")
    else:
        print(f"\nCorpus database not found: {corpus_db_path}")
        print("Run auto_runner.py first to collect texts.")
    
    # Show sample entries
    db = ValencyDatabase(valency_db_path)
    print("\nSample entries:")
    for lemma in ['δίδωμι', 'λέγω', 'ἔρχομαι']:
        entry = db.get_entry(lemma)
        if entry:
            print(f"\n  {lemma}:")
            for frame in entry['frames']:
                print(f"    {frame['pattern']}: {frame['frequency']} instances")
