"""
Semantic Role Labeling System
PropBank/FrameNet style annotation for Greek
Following Jurafsky & Martin Chapter 21
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SEMANTIC ROLE DEFINITIONS
# =============================================================================

class SemanticRole(Enum):
    """PropBank-style semantic roles"""
    
    # Core arguments
    ARG0 = "ARG0"  # Agent, Experiencer, Causer
    ARG1 = "ARG1"  # Patient, Theme, Undergoer
    ARG2 = "ARG2"  # Instrument, Beneficiary, Attribute
    ARG3 = "ARG3"  # Starting point, Source
    ARG4 = "ARG4"  # Ending point, Goal
    ARG5 = "ARG5"  # Direction
    
    # Adjunct arguments (modifiers)
    ARGM_LOC = "ARGM-LOC"    # Location
    ARGM_TMP = "ARGM-TMP"    # Temporal
    ARGM_MNR = "ARGM-MNR"    # Manner
    ARGM_CAU = "ARGM-CAU"    # Cause
    ARGM_PRP = "ARGM-PRP"    # Purpose
    ARGM_DIR = "ARGM-DIR"    # Direction
    ARGM_EXT = "ARGM-EXT"    # Extent
    ARGM_DIS = "ARGM-DIS"    # Discourse marker
    ARGM_ADV = "ARGM-ADV"    # General adverbial
    ARGM_NEG = "ARGM-NEG"    # Negation
    ARGM_MOD = "ARGM-MOD"    # Modal
    ARGM_REC = "ARGM-REC"    # Reciprocal
    ARGM_PRD = "ARGM-PRD"    # Secondary predication
    ARGM_COM = "ARGM-COM"    # Comitative
    ARGM_GOL = "ARGM-GOL"    # Goal
    ARGM_LVB = "ARGM-LVB"    # Light verb
    
    # Reference arguments
    R_ARG0 = "R-ARG0"  # Reference to ARG0
    R_ARG1 = "R-ARG1"  # Reference to ARG1
    R_ARG2 = "R-ARG2"  # Reference to ARG2
    
    # Continuation arguments
    C_ARG0 = "C-ARG0"  # Continuation of ARG0
    C_ARG1 = "C-ARG1"  # Continuation of ARG1


# Role descriptions
ROLE_DESCRIPTIONS = {
    "ARG0": {
        "name": "Agent",
        "description": "The volitional causer of an event",
        "typical_realization": "Nominative subject",
        "examples": ["ὁ στρατηγός (the general)", "ὁ βασιλεύς (the king)"]
    },
    "ARG1": {
        "name": "Patient/Theme",
        "description": "The entity affected by the action or undergoing change",
        "typical_realization": "Accusative object",
        "examples": ["τὴν πόλιν (the city)", "τὸν λόγον (the word)"]
    },
    "ARG2": {
        "name": "Instrument/Beneficiary/Attribute",
        "description": "Secondary participant, instrument, or attribute",
        "typical_realization": "Dative or instrumental",
        "examples": ["τῷ ξίφει (with the sword)", "τῷ παιδί (for the child)"]
    },
    "ARG3": {
        "name": "Starting Point/Source",
        "description": "Origin or source of motion/transfer",
        "typical_realization": "Genitive or ἐκ/ἀπό + genitive",
        "examples": ["ἐκ τῆς πόλεως (from the city)"]
    },
    "ARG4": {
        "name": "Ending Point/Goal",
        "description": "Destination or goal of motion/transfer",
        "typical_realization": "εἰς + accusative or dative",
        "examples": ["εἰς τὴν πόλιν (to the city)"]
    },
    "ARGM-LOC": {
        "name": "Location",
        "description": "Where the event takes place",
        "typical_realization": "ἐν + dative, locative",
        "examples": ["ἐν τῇ ἀγορᾷ (in the agora)"]
    },
    "ARGM-TMP": {
        "name": "Temporal",
        "description": "When the event takes place",
        "typical_realization": "Temporal expressions, genitive of time",
        "examples": ["τῇ ὑστεραίᾳ (on the next day)"]
    },
    "ARGM-MNR": {
        "name": "Manner",
        "description": "How the action is performed",
        "typical_realization": "Adverbs, dative of manner",
        "examples": ["ταχέως (quickly)", "σπουδῇ (with haste)"]
    },
    "ARGM-CAU": {
        "name": "Cause",
        "description": "The reason for the event",
        "typical_realization": "διά + accusative, causal clauses",
        "examples": ["διὰ τὸν φόβον (because of fear)"]
    },
    "ARGM-PRP": {
        "name": "Purpose",
        "description": "The purpose of the action",
        "typical_realization": "ἵνα clause, infinitive of purpose",
        "examples": ["ἵνα μάθῃ (in order to learn)"]
    },
    "ARGM-NEG": {
        "name": "Negation",
        "description": "Negation marker",
        "typical_realization": "οὐ, μή",
        "examples": ["οὐκ (not)"]
    }
}


# =============================================================================
# VERB FRAME DEFINITIONS
# =============================================================================

@dataclass
class VerbFrame:
    """Definition of a verb's argument structure"""
    
    lemma: str
    sense: str = ""
    
    # Core arguments
    arg0: str = ""  # Description of ARG0
    arg1: str = ""  # Description of ARG1
    arg2: str = ""  # Description of ARG2
    arg3: str = ""  # Description of ARG3
    arg4: str = ""  # Description of ARG4
    
    # Typical patterns
    patterns: List[str] = field(default_factory=list)
    
    # Examples
    examples: List[Dict] = field(default_factory=list)
    
    # Semantic class
    semantic_class: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Greek verb frames (sample)
GREEK_VERB_FRAMES = {
    "δίδωμι": VerbFrame(
        lemma="δίδωμι",
        sense="give",
        arg0="giver",
        arg1="thing given",
        arg2="recipient",
        patterns=["NOM + ACC + DAT"],
        semantic_class="transfer",
        examples=[
            {"text": "ὁ πατὴρ δίδωσι τῷ παιδὶ τὸ βιβλίον", 
             "gloss": "The father gives the book to the child"}
        ]
    ),
    "λέγω": VerbFrame(
        lemma="λέγω",
        sense="say, speak",
        arg0="speaker",
        arg1="content/utterance",
        arg2="addressee",
        patterns=["NOM + ACC", "NOM + ὅτι clause", "NOM + ACC + DAT"],
        semantic_class="communication",
        examples=[
            {"text": "ὁ ῥήτωρ λέγει τὸν λόγον", 
             "gloss": "The orator speaks the speech"}
        ]
    ),
    "ἄγω": VerbFrame(
        lemma="ἄγω",
        sense="lead, bring",
        arg0="leader/bringer",
        arg1="entity led/brought",
        arg3="source",
        arg4="goal",
        patterns=["NOM + ACC", "NOM + ACC + εἰς ACC", "NOM + ACC + ἐκ GEN + εἰς ACC"],
        semantic_class="motion_caused",
        examples=[
            {"text": "ὁ στρατηγὸς ἄγει τοὺς στρατιώτας εἰς τὴν πόλιν",
             "gloss": "The general leads the soldiers to the city"}
        ]
    ),
    "ἔρχομαι": VerbFrame(
        lemma="ἔρχομαι",
        sense="come, go",
        arg0="mover",
        arg3="source",
        arg4="goal",
        patterns=["NOM", "NOM + εἰς ACC", "NOM + ἐκ GEN"],
        semantic_class="motion",
        examples=[
            {"text": "ὁ ἄγγελος ἔρχεται εἰς τὴν πόλιν",
             "gloss": "The messenger comes to the city"}
        ]
    ),
    "ὁράω": VerbFrame(
        lemma="ὁράω",
        sense="see",
        arg0="perceiver",
        arg1="thing perceived",
        patterns=["NOM + ACC", "NOM + ὅτι clause", "NOM + ACC + participle"],
        semantic_class="perception",
        examples=[
            {"text": "ὁ παῖς ὁρᾷ τὸν πατέρα",
             "gloss": "The child sees the father"}
        ]
    ),
    "ἀκούω": VerbFrame(
        lemma="ἀκούω",
        sense="hear",
        arg0="hearer",
        arg1="thing heard",
        arg2="source of sound",
        patterns=["NOM + GEN", "NOM + ACC", "NOM + GEN + participle"],
        semantic_class="perception",
        examples=[
            {"text": "ὁ παῖς ἀκούει τοῦ διδασκάλου",
             "gloss": "The child hears/listens to the teacher"}
        ]
    ),
    "γράφω": VerbFrame(
        lemma="γράφω",
        sense="write",
        arg0="writer",
        arg1="thing written",
        arg2="recipient",
        patterns=["NOM + ACC", "NOM + ACC + DAT"],
        semantic_class="creation",
        examples=[
            {"text": "ὁ ποιητὴς γράφει τὸ ποίημα",
             "gloss": "The poet writes the poem"}
        ]
    ),
    "πέμπω": VerbFrame(
        lemma="πέμπω",
        sense="send",
        arg0="sender",
        arg1="thing/person sent",
        arg2="recipient",
        arg4="destination",
        patterns=["NOM + ACC", "NOM + ACC + DAT", "NOM + ACC + εἰς ACC"],
        semantic_class="transfer",
        examples=[
            {"text": "ὁ βασιλεὺς πέμπει τὸν ἄγγελον εἰς τὴν πόλιν",
             "gloss": "The king sends the messenger to the city"}
        ]
    ),
    "ποιέω": VerbFrame(
        lemma="ποιέω",
        sense="make, do",
        arg0="maker/doer",
        arg1="thing made/done",
        arg2="result/beneficiary",
        patterns=["NOM + ACC", "NOM + ACC + ACC (double accusative)"],
        semantic_class="creation",
        examples=[
            {"text": "ὁ τέκτων ποιεῖ τὴν οἰκίαν",
             "gloss": "The carpenter makes the house"}
        ]
    ),
    "λαμβάνω": VerbFrame(
        lemma="λαμβάνω",
        sense="take, receive",
        arg0="taker/receiver",
        arg1="thing taken/received",
        arg2="source",
        patterns=["NOM + ACC", "NOM + ACC + παρά GEN"],
        semantic_class="transfer",
        examples=[
            {"text": "ὁ στρατιώτης λαμβάνει τὸ ξίφος",
             "gloss": "The soldier takes the sword"}
        ]
    ),
    "φέρω": VerbFrame(
        lemma="φέρω",
        sense="carry, bear",
        arg0="carrier",
        arg1="thing carried",
        arg3="source",
        arg4="goal",
        patterns=["NOM + ACC", "NOM + ACC + εἰς ACC"],
        semantic_class="motion_caused",
        examples=[
            {"text": "ὁ δοῦλος φέρει τὸν οἶνον",
             "gloss": "The slave carries the wine"}
        ]
    ),
    "εἰμί": VerbFrame(
        lemma="εἰμί",
        sense="be",
        arg1="subject/theme",
        arg2="predicate/attribute",
        patterns=["NOM + NOM", "NOM + ADJ", "NOM + ἐν DAT"],
        semantic_class="stative",
        examples=[
            {"text": "ὁ ἀνὴρ σοφός ἐστιν",
             "gloss": "The man is wise"}
        ]
    ),
    "γίγνομαι": VerbFrame(
        lemma="γίγνομαι",
        sense="become, happen",
        arg1="entity undergoing change",
        arg2="result state",
        patterns=["NOM", "NOM + NOM", "NOM + ADJ"],
        semantic_class="change_of_state",
        examples=[
            {"text": "ὁ παῖς ἀνὴρ γίγνεται",
             "gloss": "The boy becomes a man"}
        ]
    ),
    "βούλομαι": VerbFrame(
        lemma="βούλομαι",
        sense="want, wish",
        arg0="wanter",
        arg1="thing wanted (infinitive clause)",
        patterns=["NOM + INF", "NOM + ACC"],
        semantic_class="desire",
        examples=[
            {"text": "βούλομαι μαθεῖν",
             "gloss": "I want to learn"}
        ]
    ),
    "δύναμαι": VerbFrame(
        lemma="δύναμαι",
        sense="be able, can",
        arg0="able entity",
        arg1="ability content (infinitive)",
        patterns=["NOM + INF"],
        semantic_class="modal",
        examples=[
            {"text": "δύναμαι λέγειν",
             "gloss": "I am able to speak"}
        ]
    ),
    "κελεύω": VerbFrame(
        lemma="κελεύω",
        sense="order, command",
        arg0="commander",
        arg1="commanded action (infinitive)",
        arg2="person commanded",
        patterns=["NOM + ACC + INF"],
        semantic_class="communication",
        examples=[
            {"text": "ὁ στρατηγὸς κελεύει τοὺς στρατιώτας μάχεσθαι",
             "gloss": "The general orders the soldiers to fight"}
        ]
    ),
    "πείθω": VerbFrame(
        lemma="πείθω",
        sense="persuade (active), obey (middle)",
        arg0="persuader",
        arg1="person persuaded / thing obeyed",
        arg2="content of persuasion",
        patterns=["NOM + ACC + INF (active)", "NOM + DAT (middle)"],
        semantic_class="communication",
        examples=[
            {"text": "πείθω τὸν φίλον μένειν",
             "gloss": "I persuade my friend to stay"}
        ]
    ),
    "μανθάνω": VerbFrame(
        lemma="μανθάνω",
        sense="learn",
        arg0="learner",
        arg1="thing learned",
        arg2="teacher/source",
        patterns=["NOM + ACC", "NOM + INF", "NOM + παρά GEN"],
        semantic_class="cognition",
        examples=[
            {"text": "ὁ παῖς μανθάνει τὴν τέχνην",
             "gloss": "The child learns the skill"}
        ]
    ),
    "διδάσκω": VerbFrame(
        lemma="διδάσκω",
        sense="teach",
        arg0="teacher",
        arg1="thing taught",
        arg2="student",
        patterns=["NOM + ACC + ACC (double accusative)", "NOM + ACC + INF"],
        semantic_class="transfer",
        examples=[
            {"text": "ὁ διδάσκαλος διδάσκει τοὺς παῖδας τὴν γραμματικήν",
             "gloss": "The teacher teaches the children grammar"}
        ]
    ),
    "νομίζω": VerbFrame(
        lemma="νομίζω",
        sense="think, believe",
        arg0="thinker",
        arg1="content of thought",
        patterns=["NOM + ACC + INF", "NOM + ὅτι clause"],
        semantic_class="cognition",
        examples=[
            {"text": "νομίζω τοῦτο ἀληθὲς εἶναι",
             "gloss": "I think this to be true"}
        ]
    ),
    "οἶδα": VerbFrame(
        lemma="οἶδα",
        sense="know",
        arg0="knower",
        arg1="thing known",
        patterns=["NOM + ACC", "NOM + ὅτι clause", "NOM + INF"],
        semantic_class="cognition",
        examples=[
            {"text": "οἶδα τὴν ἀλήθειαν",
             "gloss": "I know the truth"}
        ]
    ),
}


# =============================================================================
# SEMANTIC ROLE ANNOTATION
# =============================================================================

@dataclass
class SRLAnnotation:
    """Semantic role annotation for a predicate"""
    
    predicate_id: int
    predicate_lemma: str
    predicate_sense: str = ""
    
    # Arguments: role -> (token_ids, text)
    arguments: Dict[str, Tuple[List[int], str]] = field(default_factory=dict)
    
    # Frame reference
    frame_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'predicate_id': self.predicate_id,
            'predicate_lemma': self.predicate_lemma,
            'predicate_sense': self.predicate_sense,
            'arguments': {
                role: {'token_ids': ids, 'text': text}
                for role, (ids, text) in self.arguments.items()
            },
            'frame_id': self.frame_id
        }
    
    def add_argument(self, role: str, token_ids: List[int], text: str):
        """Add an argument"""
        self.arguments[role] = (token_ids, text)
    
    def get_argument(self, role: str) -> Optional[Tuple[List[int], str]]:
        """Get an argument by role"""
        return self.arguments.get(role)


@dataclass
class SRLSentence:
    """A sentence with SRL annotations"""
    
    id: str
    text: str
    tokens: List[Dict]
    
    # Predicate-argument structures
    predicates: List[SRLAnnotation] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'tokens': self.tokens,
            'predicates': [p.to_dict() for p in self.predicates]
        }
    
    def to_conll_srl(self) -> str:
        """Convert to CoNLL SRL format"""
        lines = []
        
        # Header
        lines.append(f"# sent_id = {self.id}")
        lines.append(f"# text = {self.text}")
        
        # Build role columns
        n_predicates = len(self.predicates)
        role_columns = [['*'] * len(self.tokens) for _ in range(n_predicates)]
        
        for pred_idx, pred in enumerate(self.predicates):
            # Mark predicate
            role_columns[pred_idx][pred.predicate_id - 1] = f"({pred.predicate_lemma})"
            
            # Mark arguments
            for role, (token_ids, _) in pred.arguments.items():
                if token_ids:
                    start_id = min(token_ids)
                    end_id = max(token_ids)
                    
                    if start_id == end_id:
                        role_columns[pred_idx][start_id - 1] = f"({role}*)"
                    else:
                        role_columns[pred_idx][start_id - 1] = f"({role}*"
                        role_columns[pred_idx][end_id - 1] = "*)"
        
        # Output tokens
        for i, token in enumerate(self.tokens):
            cols = [
                str(token.get('id', i + 1)),
                token.get('form', ''),
                token.get('lemma', '_'),
                token.get('pos', '_'),
            ]
            
            # Add role columns
            for pred_idx in range(n_predicates):
                cols.append(role_columns[pred_idx][i])
            
            lines.append('\t'.join(cols))
        
        return '\n'.join(lines)


# =============================================================================
# SRL PREDICTOR (RULE-BASED)
# =============================================================================

class SRLPredictor:
    """Rule-based SRL predictor for Greek"""
    
    def __init__(self):
        self.verb_frames = GREEK_VERB_FRAMES
        
    def predict(self, tokens: List[Dict]) -> List[SRLAnnotation]:
        """Predict SRL for a sentence"""
        annotations = []
        
        # Find predicates (verbs)
        predicates = [t for t in tokens if t.get('pos', '').startswith('V') or t.get('pos') == 'VERB']
        
        for pred in predicates:
            pred_id = pred.get('id', 0)
            pred_lemma = pred.get('lemma', pred.get('form', ''))
            
            annotation = SRLAnnotation(
                predicate_id=pred_id,
                predicate_lemma=pred_lemma
            )
            
            # Get frame if available
            if pred_lemma in self.verb_frames:
                frame = self.verb_frames[pred_lemma]
                annotation.predicate_sense = frame.sense
                annotation.frame_id = pred_lemma
            
            # Find arguments based on dependency relations
            for token in tokens:
                if token.get('head', 0) == pred_id:
                    role = self._map_relation_to_role(token, pred)
                    if role:
                        # Get span (for now, just the token)
                        span_tokens = self._get_span(token, tokens)
                        span_text = ' '.join(t.get('form', '') for t in span_tokens)
                        span_ids = [t.get('id', 0) for t in span_tokens]
                        
                        annotation.add_argument(role, span_ids, span_text)
            
            annotations.append(annotation)
        
        return annotations
    
    def _map_relation_to_role(self, token: Dict, predicate: Dict) -> Optional[str]:
        """Map dependency relation to semantic role"""
        relation = token.get('relation', '')
        pos = token.get('pos', '')
        case = self._get_case(token)
        
        # Subject -> ARG0 (usually agent)
        if relation in ['nsubj', 'sub']:
            return 'ARG0'
        
        # Direct object -> ARG1 (usually patient/theme)
        if relation in ['obj', 'dobj']:
            return 'ARG1'
        
        # Indirect object -> ARG2 (usually recipient/beneficiary)
        if relation in ['iobj', 'obl:arg']:
            return 'ARG2'
        
        # Oblique arguments
        if relation.startswith('obl'):
            # Check preposition or case for more specific role
            if 'εἰς' in token.get('form', '') or case == 'Acc':
                return 'ARG4'  # Goal
            if 'ἐκ' in token.get('form', '') or 'ἀπό' in token.get('form', ''):
                return 'ARG3'  # Source
            if case == 'Dat':
                return 'ARG2'  # Beneficiary/Instrument
            return 'ARGM-LOC'  # Default to location
        
        # Adverbial modifiers
        if relation == 'advmod':
            # Check for negation
            if token.get('lemma', '') in ['οὐ', 'μή', 'οὐκ', 'οὐχ']:
                return 'ARGM-NEG'
            return 'ARGM-MNR'
        
        # Temporal modifiers
        if relation == 'obl:tmod':
            return 'ARGM-TMP'
        
        # Clausal complements
        if relation in ['ccomp', 'xcomp']:
            return 'ARG1'
        
        # Adverbial clauses
        if relation == 'advcl':
            return 'ARGM-ADV'
        
        return None
    
    def _get_case(self, token: Dict) -> str:
        """Extract case from morphology"""
        morph = token.get('morph', '')
        if 'Case=' in morph:
            match = re.search(r'Case=(\w+)', morph)
            if match:
                return match.group(1)
        return ''
    
    def _get_span(self, head_token: Dict, all_tokens: List[Dict]) -> List[Dict]:
        """Get all tokens in a constituent headed by head_token"""
        head_id = head_token.get('id', 0)
        span = [head_token]
        
        # Find all dependents (recursively)
        def find_dependents(token_id):
            deps = []
            for t in all_tokens:
                if t.get('head', 0) == token_id:
                    deps.append(t)
                    deps.extend(find_dependents(t.get('id', 0)))
            return deps
        
        span.extend(find_dependents(head_id))
        
        # Sort by ID
        span.sort(key=lambda t: t.get('id', 0))
        
        return span


# =============================================================================
# SRL DATABASE
# =============================================================================

class SRLDatabase:
    """Database for storing SRL annotations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Verb frames table
        c.execute('''CREATE TABLE IF NOT EXISTS verb_frames (
            lemma TEXT PRIMARY KEY,
            sense TEXT,
            arg0 TEXT,
            arg1 TEXT,
            arg2 TEXT,
            arg3 TEXT,
            arg4 TEXT,
            patterns TEXT,
            semantic_class TEXT,
            examples TEXT
        )''')
        
        # SRL annotations table
        c.execute('''CREATE TABLE IF NOT EXISTS srl_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id TEXT NOT NULL,
            predicate_id INTEGER,
            predicate_lemma TEXT,
            predicate_sense TEXT,
            arguments TEXT,
            frame_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_srl_sentence ON srl_annotations(sentence_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_srl_lemma ON srl_annotations(predicate_lemma)')
        
        conn.commit()
        conn.close()
        
        # Load default frames
        self._load_default_frames()
    
    def _load_default_frames(self):
        """Load default verb frames"""
        conn = self.get_connection()
        c = conn.cursor()
        
        for lemma, frame in GREEK_VERB_FRAMES.items():
            c.execute('''INSERT OR REPLACE INTO verb_frames 
                        (lemma, sense, arg0, arg1, arg2, arg3, arg4, patterns, semantic_class, examples)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (lemma, frame.sense, frame.arg0, frame.arg1, frame.arg2,
                      frame.arg3, frame.arg4, json.dumps(frame.patterns),
                      frame.semantic_class, json.dumps(frame.examples)))
        
        conn.commit()
        conn.close()
    
    def store_annotation(self, sentence_id: str, annotation: SRLAnnotation):
        """Store an SRL annotation"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('''INSERT INTO srl_annotations 
                    (sentence_id, predicate_id, predicate_lemma, predicate_sense, arguments, frame_id)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (sentence_id, annotation.predicate_id, annotation.predicate_lemma,
                  annotation.predicate_sense, json.dumps(annotation.to_dict()['arguments']),
                  annotation.frame_id))
        
        conn.commit()
        conn.close()
    
    def get_frame(self, lemma: str) -> Optional[VerbFrame]:
        """Get verb frame by lemma"""
        conn = self.get_connection()
        c = conn.cursor()
        
        c.execute('SELECT * FROM verb_frames WHERE lemma = ?', (lemma,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return None
        
        frame = VerbFrame(
            lemma=row[0],
            sense=row[1] or '',
            arg0=row[2] or '',
            arg1=row[3] or '',
            arg2=row[4] or '',
            arg3=row[5] or '',
            arg4=row[6] or '',
            patterns=json.loads(row[7]) if row[7] else [],
            semantic_class=row[8] or '',
            examples=json.loads(row[9]) if row[9] else []
        )
        
        conn.close()
        return frame
    
    def get_statistics(self) -> Dict:
        """Get SRL statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM verb_frames")
        stats['frame_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM srl_annotations")
        stats['annotation_count'] = c.fetchone()[0]
        
        c.execute("SELECT predicate_lemma, COUNT(*) FROM srl_annotations GROUP BY predicate_lemma ORDER BY COUNT(*) DESC LIMIT 20")
        stats['top_predicates'] = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT semantic_class, COUNT(*) FROM verb_frames WHERE semantic_class != '' GROUP BY semantic_class")
        stats['by_semantic_class'] = {r[0]: r[1] for r in c.fetchall()}
        
        conn.close()
        return stats


# =============================================================================
# VISUALIZATION
# =============================================================================

class SRLVisualizer:
    """Visualize SRL annotations"""
    
    ROLE_COLORS = {
        'ARG0': '#ef4444',      # Red
        'ARG1': '#f97316',      # Orange
        'ARG2': '#eab308',      # Yellow
        'ARG3': '#22c55e',      # Green
        'ARG4': '#06b6d4',      # Cyan
        'ARGM-LOC': '#3b82f6',  # Blue
        'ARGM-TMP': '#8b5cf6',  # Purple
        'ARGM-MNR': '#ec4899',  # Pink
        'ARGM-CAU': '#6366f1',  # Indigo
        'ARGM-PRP': '#14b8a6',  # Teal
        'ARGM-NEG': '#f43f5e',  # Rose
    }
    
    @classmethod
    def generate_html(cls, sentence: SRLSentence) -> str:
        """Generate HTML visualization"""
        html_parts = [
            '<div class="srl-container" style="font-family: sans-serif; padding: 20px;">',
            f'<div class="sentence-text" style="font-size: 1.3em; margin-bottom: 20px; font-family: \'Gentium Plus\', serif;">{sentence.text}</div>',
        ]
        
        for pred in sentence.predicates:
            html_parts.append(f'''
                <div class="predicate-block" style="margin-bottom: 20px; padding: 15px; 
                     background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <div class="predicate-header" style="font-weight: bold; margin-bottom: 10px;">
                        Predicate: <span style="color: #3b82f6;">{pred.predicate_lemma}</span>
                        {f'({pred.predicate_sense})' if pred.predicate_sense else ''}
                    </div>
                    <div class="arguments" style="display: flex; flex-wrap: wrap; gap: 10px;">
            ''')
            
            for role, (token_ids, text) in pred.arguments.items():
                color = cls.ROLE_COLORS.get(role, '#94a3b8')
                role_name = ROLE_DESCRIPTIONS.get(role, {}).get('name', role)
                
                html_parts.append(f'''
                    <div class="argument" style="padding: 8px 12px; border-radius: 6px; 
                         background: {color}20; border: 1px solid {color};">
                        <div class="role" style="font-size: 0.8em; font-weight: bold; color: {color};">
                            {role} ({role_name})
                        </div>
                        <div class="text" style="font-size: 1.1em; margin-top: 4px;">
                            {text}
                        </div>
                    </div>
                ''')
            
            html_parts.append('</div></div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    @classmethod
    def generate_svg(cls, sentence: SRLSentence, width: int = 1000, height: int = 300) -> str:
        """Generate SVG visualization"""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]
        
        # Token positions
        n_tokens = len(sentence.tokens)
        token_width = (width - 100) / n_tokens
        token_y = height - 80
        
        # Draw tokens
        for i, token in enumerate(sentence.tokens):
            x = 50 + i * token_width + token_width / 2
            
            svg_parts.append(
                f'<text x="{x}" y="{token_y}" text-anchor="middle" '
                f'font-size="14" fill="#1e293b">{token.get("form", "")}</text>'
            )
            svg_parts.append(
                f'<text x="{x}" y="{token_y + 15}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{token.get("id", i+1)}</text>'
            )
        
        # Draw predicate-argument structure
        pred_y = 50
        for pred_idx, pred in enumerate(sentence.predicates):
            pred_x = 50 + (pred.predicate_id - 1) * token_width + token_width / 2
            
            # Predicate marker
            svg_parts.append(
                f'<circle cx="{pred_x}" cy="{pred_y}" r="20" fill="#3b82f6"/>'
            )
            svg_parts.append(
                f'<text x="{pred_x}" y="{pred_y + 5}" text-anchor="middle" '
                f'font-size="10" fill="white" font-weight="bold">PRED</text>'
            )
            
            # Draw argument arcs
            for role, (token_ids, _) in pred.arguments.items():
                if token_ids:
                    arg_x = 50 + (min(token_ids) - 1) * token_width + token_width / 2
                    color = cls.ROLE_COLORS.get(role, '#94a3b8')
                    
                    # Arc
                    mid_y = pred_y + 30
                    svg_parts.append(
                        f'<path d="M {pred_x} {pred_y + 20} Q {(pred_x + arg_x)/2} {mid_y + 30} {arg_x} {token_y - 20}" '
                        f'fill="none" stroke="{color}" stroke-width="2"/>'
                    )
                    
                    # Role label
                    label_x = (pred_x + arg_x) / 2
                    label_y = mid_y + 20
                    svg_parts.append(
                        f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                        f'font-size="10" fill="{color}" font-weight="bold">{role}</text>'
                    )
            
            pred_y += 60
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test SRL
    test_tokens = [
        {'id': 1, 'form': 'ὁ', 'lemma': 'ὁ', 'pos': 'DET', 'head': 2, 'relation': 'det'},
        {'id': 2, 'form': 'στρατηγός', 'lemma': 'στρατηγός', 'pos': 'NOUN', 'head': 3, 'relation': 'nsubj', 'morph': 'Case=Nom'},
        {'id': 3, 'form': 'ἄγει', 'lemma': 'ἄγω', 'pos': 'VERB', 'head': 0, 'relation': 'root'},
        {'id': 4, 'form': 'τούς', 'lemma': 'ὁ', 'pos': 'DET', 'head': 5, 'relation': 'det'},
        {'id': 5, 'form': 'στρατιώτας', 'lemma': 'στρατιώτης', 'pos': 'NOUN', 'head': 3, 'relation': 'obj', 'morph': 'Case=Acc'},
        {'id': 6, 'form': 'εἰς', 'lemma': 'εἰς', 'pos': 'ADP', 'head': 8, 'relation': 'case'},
        {'id': 7, 'form': 'τήν', 'lemma': 'ὁ', 'pos': 'DET', 'head': 8, 'relation': 'det'},
        {'id': 8, 'form': 'πόλιν', 'lemma': 'πόλις', 'pos': 'NOUN', 'head': 3, 'relation': 'obl', 'morph': 'Case=Acc'},
    ]
    
    # Predict SRL
    predictor = SRLPredictor()
    annotations = predictor.predict(test_tokens)
    
    print("=" * 60)
    print("SRL PREDICTION TEST")
    print("=" * 60)
    
    for ann in annotations:
        print(f"\nPredicate: {ann.predicate_lemma} ({ann.predicate_sense})")
        for role, (ids, text) in ann.arguments.items():
            print(f"  {role}: {text} (tokens: {ids})")
    
    # Create sentence
    sentence = SRLSentence(
        id='test_1',
        text='ὁ στρατηγὸς ἄγει τοὺς στρατιώτας εἰς τὴν πόλιν',
        tokens=test_tokens,
        predicates=annotations
    )
    
    # Generate visualization
    html = SRLVisualizer.generate_html(sentence)
    print(f"\nGenerated HTML visualization ({len(html)} characters)")
    
    # CoNLL format
    conll = sentence.to_conll_srl()
    print(f"\nCoNLL SRL format:")
    print(conll)
