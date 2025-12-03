#!/usr/bin/env python3
"""
ANNOTATION TOOLS - Complete Linguistic Annotation System
Multi-layer annotation for PROIEL-style corpora

Features:
1. Morphological annotation
2. Syntactic annotation (dependency parsing)
3. Semantic role labeling
4. Named entity recognition
5. Coreference resolution
6. Discourse annotation
7. Information structure
8. Annotation validation
9. Inter-annotator agreement
10. Export to multiple formats

Based on: UAM CorpusTool / ANNIS principles
"""

import os
import re
import json
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum, auto
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ANNOTATION LAYERS
# =============================================================================

class AnnotationLayer(Enum):
    """Types of annotation layers"""
    TOKENIZATION = auto()
    MORPHOLOGY = auto()
    POS = auto()
    LEMMA = auto()
    SYNTAX = auto()
    SEMANTICS = auto()
    NER = auto()
    COREFERENCE = auto()
    DISCOURSE = auto()
    INFORMATION_STRUCTURE = auto()
    PRAGMATICS = auto()

@dataclass
class Annotation:
    """Base annotation class"""
    id: str
    layer: AnnotationLayer
    target_id: str  # ID of annotated element
    value: Any
    annotator: str = ""
    timestamp: str = ""
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.layer}_{self.target_id}_{self.value}".encode()
            ).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class Span:
    """A span of text/tokens"""
    start: int
    end: int
    text: str = ""
    tokens: List[str] = field(default_factory=list)

# =============================================================================
# MORPHOLOGICAL ANNOTATION
# =============================================================================

class MorphologicalFeatures:
    """Standard morphological features (UD-style)"""
    
    # Feature categories
    CATEGORIES = {
        'POS': ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 
                'PART', 'NUM', 'INTJ', 'PUNCT', 'X'],
        'Case': ['Nom', 'Gen', 'Dat', 'Acc', 'Voc', 'Loc', 'Ins', 'Abl'],
        'Number': ['Sing', 'Plur', 'Dual'],
        'Gender': ['Masc', 'Fem', 'Neut', 'Com'],
        'Person': ['1', '2', '3'],
        'Tense': ['Pres', 'Past', 'Fut', 'Pqp', 'Imp', 'Perf', 'Aor'],
        'Aspect': ['Imp', 'Perf', 'Prosp'],
        'Mood': ['Ind', 'Sub', 'Imp', 'Opt', 'Cnd', 'Pot'],
        'Voice': ['Act', 'Pass', 'Mid', 'Cau'],
        'VerbForm': ['Fin', 'Inf', 'Part', 'Ger', 'Sup', 'Conv'],
        'Degree': ['Pos', 'Cmp', 'Sup', 'Abs'],
        'Definite': ['Def', 'Ind', 'Spec'],
        'PronType': ['Prs', 'Rcp', 'Art', 'Int', 'Rel', 'Dem', 'Ind', 'Tot', 'Neg'],
    }
    
    # PROIEL-style morphology codes
    PROIEL_CODES = {
        # Position 1: Part of speech
        'pos': {
            'A-': 'ADJ', 'C-': 'CONJ', 'Df': 'ADV', 'Dq': 'ADV',
            'Du': 'ADV', 'F-': 'INTJ', 'G-': 'PART', 'I-': 'INTJ',
            'Ma': 'NUM', 'Mo': 'NUM', 'N-': 'NOUN', 'Nb': 'NOUN',
            'Ne': 'NOUN', 'Pc': 'PRON', 'Pd': 'PRON', 'Pi': 'PRON',
            'Pk': 'PRON', 'Pp': 'PRON', 'Pr': 'PRON', 'Ps': 'PRON',
            'Pt': 'PRON', 'Px': 'PRON', 'R-': 'ADP', 'S-': 'ADP',
            'V-': 'VERB', 'X-': 'X',
        },
        # Position 2: Person
        'person': {'1': '1', '2': '2', '3': '3', '-': None},
        # Position 3: Number
        'number': {'s': 'Sing', 'p': 'Plur', 'd': 'Dual', '-': None},
        # Position 4: Tense
        'tense': {
            'p': 'Pres', 'i': 'Imp', 'f': 'Fut', 'a': 'Aor',
            'r': 'Perf', 'l': 'Pqp', 't': 'FutPerf', '-': None
        },
        # Position 5: Mood
        'mood': {
            'i': 'Ind', 's': 'Sub', 'm': 'Imp', 'o': 'Opt',
            'n': 'Inf', 'p': 'Part', 'd': 'Ger', 'g': 'Gerv',
            'u': 'Sup', '-': None
        },
        # Position 6: Voice
        'voice': {'a': 'Act', 'p': 'Pass', 'm': 'Mid', 'e': 'Mid', '-': None},
        # Position 7: Gender
        'gender': {'m': 'Masc', 'f': 'Fem', 'n': 'Neut', '-': None},
        # Position 8: Case
        'case': {
            'n': 'Nom', 'g': 'Gen', 'd': 'Dat', 'a': 'Acc',
            'v': 'Voc', 'l': 'Loc', 'b': 'Abl', 'i': 'Ins', '-': None
        },
        # Position 9: Degree
        'degree': {'p': 'Pos', 'c': 'Cmp', 's': 'Sup', '-': None},
    }
    
    @classmethod
    def parse_proiel(cls, morph_code: str) -> Dict[str, str]:
        """Parse PROIEL morphology code"""
        features = {}
        
        if len(morph_code) < 2:
            return features
        
        # POS (positions 1-2)
        pos_code = morph_code[:2]
        if pos_code in cls.PROIEL_CODES['pos']:
            features['POS'] = cls.PROIEL_CODES['pos'][pos_code]
        
        # Other features
        if len(morph_code) >= 3:
            person = morph_code[2]
            if person in cls.PROIEL_CODES['person'] and cls.PROIEL_CODES['person'][person]:
                features['Person'] = cls.PROIEL_CODES['person'][person]
        
        if len(morph_code) >= 4:
            number = morph_code[3]
            if number in cls.PROIEL_CODES['number'] and cls.PROIEL_CODES['number'][number]:
                features['Number'] = cls.PROIEL_CODES['number'][number]
        
        if len(morph_code) >= 5:
            tense = morph_code[4]
            if tense in cls.PROIEL_CODES['tense'] and cls.PROIEL_CODES['tense'][tense]:
                features['Tense'] = cls.PROIEL_CODES['tense'][tense]
        
        if len(morph_code) >= 6:
            mood = morph_code[5]
            if mood in cls.PROIEL_CODES['mood'] and cls.PROIEL_CODES['mood'][mood]:
                features['Mood'] = cls.PROIEL_CODES['mood'][mood]
        
        if len(morph_code) >= 7:
            voice = morph_code[6]
            if voice in cls.PROIEL_CODES['voice'] and cls.PROIEL_CODES['voice'][voice]:
                features['Voice'] = cls.PROIEL_CODES['voice'][voice]
        
        if len(morph_code) >= 8:
            gender = morph_code[7]
            if gender in cls.PROIEL_CODES['gender'] and cls.PROIEL_CODES['gender'][gender]:
                features['Gender'] = cls.PROIEL_CODES['gender'][gender]
        
        if len(morph_code) >= 9:
            case = morph_code[8]
            if case in cls.PROIEL_CODES['case'] and cls.PROIEL_CODES['case'][case]:
                features['Case'] = cls.PROIEL_CODES['case'][case]
        
        return features
    
    @classmethod
    def to_ud_string(cls, features: Dict[str, str]) -> str:
        """Convert features dict to UD string format"""
        if not features:
            return '_'
        
        # Sort features alphabetically
        sorted_feats = sorted(features.items())
        return '|'.join(f"{k}={v}" for k, v in sorted_feats if k != 'POS')
    
    @classmethod
    def from_ud_string(cls, feat_string: str) -> Dict[str, str]:
        """Parse UD feature string"""
        if feat_string == '_' or not feat_string:
            return {}
        
        features = {}
        for feat in feat_string.split('|'):
            if '=' in feat:
                key, value = feat.split('=', 1)
                features[key] = value
        
        return features


@dataclass
class MorphAnnotation(Annotation):
    """Morphological annotation"""
    lemma: str = ""
    pos: str = ""
    features: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.layer = AnnotationLayer.MORPHOLOGY
        super().__post_init__()


# =============================================================================
# SYNTACTIC ANNOTATION
# =============================================================================

class DependencyRelations:
    """Standard dependency relations (UD-style)"""
    
    RELATIONS = {
        # Core arguments
        'nsubj': 'nominal subject',
        'obj': 'object',
        'iobj': 'indirect object',
        'csubj': 'clausal subject',
        'ccomp': 'clausal complement',
        'xcomp': 'open clausal complement',
        
        # Non-core dependents
        'obl': 'oblique nominal',
        'vocative': 'vocative',
        'expl': 'expletive',
        'dislocated': 'dislocated elements',
        'advcl': 'adverbial clause modifier',
        'advmod': 'adverbial modifier',
        'discourse': 'discourse element',
        'aux': 'auxiliary',
        'cop': 'copula',
        'mark': 'marker',
        
        # Nominal dependents
        'nmod': 'nominal modifier',
        'appos': 'appositional modifier',
        'nummod': 'numeric modifier',
        'amod': 'adjectival modifier',
        'det': 'determiner',
        'clf': 'classifier',
        'case': 'case marking',
        
        # Coordination
        'conj': 'conjunct',
        'cc': 'coordinating conjunction',
        
        # MWE
        'fixed': 'fixed multiword expression',
        'flat': 'flat multiword expression',
        'compound': 'compound',
        
        # Loose
        'list': 'list',
        'parataxis': 'parataxis',
        
        # Special
        'orphan': 'orphan',
        'goeswith': 'goes with',
        'reparandum': 'overridden disfluency',
        'punct': 'punctuation',
        'root': 'root',
        'dep': 'unspecified dependency',
    }
    
    # PROIEL-specific relations
    PROIEL_RELATIONS = {
        'sub': 'nsubj',
        'obj': 'obj',
        'obl': 'obl',
        'ag': 'obl:agent',
        'atr': 'amod',
        'apos': 'appos',
        'aux': 'aux',
        'comp': 'ccomp',
        'expl': 'expl',
        'narg': 'obl',
        'nonsub': 'obl',
        'part': 'compound:prt',
        'per': 'obl',
        'pid': 'xcomp',
        'piv': 'obl',
        'pred': 'root',
        'rel': 'acl:relcl',
        'sub': 'nsubj',
        'voc': 'vocative',
        'xadv': 'advcl',
        'xobj': 'xcomp',
        'xsub': 'csubj',
    }


@dataclass
class SyntaxAnnotation(Annotation):
    """Syntactic (dependency) annotation"""
    head: int = 0
    deprel: str = ""
    deps: str = ""  # Enhanced dependencies
    
    def __post_init__(self):
        self.layer = AnnotationLayer.SYNTAX
        super().__post_init__()


# =============================================================================
# SEMANTIC ROLE LABELING
# =============================================================================

class SemanticRoles:
    """Semantic role labels (PropBank/FrameNet style)"""
    
    # Core arguments
    CORE_ROLES = {
        'ARG0': 'Agent, Experiencer, or Causer',
        'ARG1': 'Patient, Theme, or Stimulus',
        'ARG2': 'Instrument, Beneficiary, or Attribute',
        'ARG3': 'Starting point, Beneficiary, or Attribute',
        'ARG4': 'Ending point',
        'ARG5': 'Direction',
    }
    
    # Modifier arguments
    MODIFIER_ROLES = {
        'ARGM-LOC': 'Location',
        'ARGM-TMP': 'Time',
        'ARGM-MNR': 'Manner',
        'ARGM-CAU': 'Cause',
        'ARGM-PRP': 'Purpose',
        'ARGM-DIR': 'Direction',
        'ARGM-EXT': 'Extent',
        'ARGM-DIS': 'Discourse marker',
        'ARGM-ADV': 'General adverbial',
        'ARGM-NEG': 'Negation',
        'ARGM-MOD': 'Modal',
        'ARGM-PRD': 'Secondary predication',
        'ARGM-GOL': 'Goal',
        'ARGM-REC': 'Reciprocal',
        'ARGM-COM': 'Comitative',
    }
    
    ALL_ROLES = {**CORE_ROLES, **MODIFIER_ROLES}


@dataclass
class SRLAnnotation(Annotation):
    """Semantic role annotation"""
    predicate_id: str = ""
    role: str = ""
    span: Span = None
    
    def __post_init__(self):
        self.layer = AnnotationLayer.SEMANTICS
        super().__post_init__()


# =============================================================================
# NAMED ENTITY RECOGNITION
# =============================================================================

class NamedEntityTypes:
    """Named entity types"""
    
    TYPES = {
        'PER': 'Person',
        'ORG': 'Organization',
        'LOC': 'Location',
        'GPE': 'Geo-Political Entity',
        'FAC': 'Facility',
        'EVT': 'Event',
        'WOA': 'Work of Art',
        'LAW': 'Law',
        'LANG': 'Language',
        'DATE': 'Date',
        'TIME': 'Time',
        'MONEY': 'Money',
        'QUANTITY': 'Quantity',
        'ORDINAL': 'Ordinal',
        'CARDINAL': 'Cardinal',
        'PERCENT': 'Percent',
        'NORP': 'Nationality/Religious/Political group',
        'PRODUCT': 'Product',
    }
    
    # Historical/Classical specific
    CLASSICAL_TYPES = {
        'DEITY': 'Deity/God',
        'MYTH': 'Mythological figure',
        'HERO': 'Heroic figure',
        'ETHNIC': 'Ethnic group',
        'TRIBE': 'Tribe',
        'DYNASTY': 'Dynasty',
        'BATTLE': 'Battle',
        'TREATY': 'Treaty',
        'FESTIVAL': 'Festival',
        'TEMPLE': 'Temple',
        'ORACLE': 'Oracle',
    }
    
    ALL_TYPES = {**TYPES, **CLASSICAL_TYPES}


@dataclass
class NERAnnotation(Annotation):
    """Named entity annotation"""
    entity_type: str = ""
    span: Span = None
    normalized: str = ""  # Normalized form
    wikidata_id: str = ""  # Link to Wikidata
    
    def __post_init__(self):
        self.layer = AnnotationLayer.NER
        super().__post_init__()


# =============================================================================
# COREFERENCE ANNOTATION
# =============================================================================

@dataclass
class Mention:
    """A mention in coreference"""
    id: str
    span: Span
    head_token: str
    mention_type: str  # 'proper', 'nominal', 'pronominal'
    
@dataclass
class CoreferenceChain:
    """A coreference chain"""
    id: str
    mentions: List[Mention]
    entity_type: str = ""
    
@dataclass
class CorefAnnotation(Annotation):
    """Coreference annotation"""
    chain_id: str = ""
    mention: Mention = None
    
    def __post_init__(self):
        self.layer = AnnotationLayer.COREFERENCE
        super().__post_init__()


# =============================================================================
# INFORMATION STRUCTURE
# =============================================================================

class InformationStatus:
    """Information structure categories"""
    
    STATUS = {
        'new': 'New information',
        'given': 'Given/old information',
        'accessible': 'Accessible information',
        'inferrable': 'Inferrable information',
        'bridging': 'Bridging reference',
    }
    
    FOCUS_TYPES = {
        'information': 'Information focus',
        'contrastive': 'Contrastive focus',
        'exhaustive': 'Exhaustive focus',
    }
    
    TOPIC_TYPES = {
        'aboutness': 'Aboutness topic',
        'frame': 'Frame-setting topic',
        'contrastive': 'Contrastive topic',
    }


@dataclass
class InfoStructAnnotation(Annotation):
    """Information structure annotation"""
    info_status: str = ""
    focus_type: str = ""
    topic_type: str = ""
    
    def __post_init__(self):
        self.layer = AnnotationLayer.INFORMATION_STRUCTURE
        super().__post_init__()


# =============================================================================
# ANNOTATION STORE
# =============================================================================

class AnnotationStore:
    """Store and manage annotations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize annotation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                target_id TEXT NOT NULL,
                value TEXT,
                annotator TEXT,
                timestamp TEXT,
                confidence REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Morphology annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS morph_annotations (
                id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                features TEXT,
                annotator TEXT,
                timestamp TEXT,
                FOREIGN KEY (id) REFERENCES annotations(id)
            )
        """)
        
        # Syntax annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS syntax_annotations (
                id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                head INTEGER,
                deprel TEXT,
                deps TEXT,
                annotator TEXT,
                timestamp TEXT,
                FOREIGN KEY (id) REFERENCES annotations(id)
            )
        """)
        
        # SRL annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS srl_annotations (
                id TEXT PRIMARY KEY,
                predicate_id TEXT NOT NULL,
                role TEXT,
                span_start INTEGER,
                span_end INTEGER,
                span_text TEXT,
                annotator TEXT,
                timestamp TEXT,
                FOREIGN KEY (id) REFERENCES annotations(id)
            )
        """)
        
        # NER annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ner_annotations (
                id TEXT PRIMARY KEY,
                entity_type TEXT,
                span_start INTEGER,
                span_end INTEGER,
                span_text TEXT,
                normalized TEXT,
                wikidata_id TEXT,
                annotator TEXT,
                timestamp TEXT,
                FOREIGN KEY (id) REFERENCES annotations(id)
            )
        """)
        
        # Coreference annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coref_annotations (
                id TEXT PRIMARY KEY,
                chain_id TEXT NOT NULL,
                mention_id TEXT,
                span_start INTEGER,
                span_end INTEGER,
                head_token TEXT,
                mention_type TEXT,
                annotator TEXT,
                timestamp TEXT,
                FOREIGN KEY (id) REFERENCES annotations(id)
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_layer ON annotations(layer)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_target ON annotations(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_morph_token ON morph_annotations(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_syntax_token ON syntax_annotations(token_id)")
        
        conn.commit()
        conn.close()
    
    def add_annotation(self, annotation: Annotation):
        """Add an annotation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert base annotation
        cursor.execute("""
            INSERT OR REPLACE INTO annotations
            (id, layer, target_id, value, annotator, timestamp, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            annotation.id,
            annotation.layer.name,
            annotation.target_id,
            json.dumps(annotation.value) if isinstance(annotation.value, (dict, list)) else str(annotation.value),
            annotation.annotator,
            annotation.timestamp,
            annotation.confidence,
            json.dumps(annotation.metadata)
        ))
        
        # Insert layer-specific data
        if isinstance(annotation, MorphAnnotation):
            cursor.execute("""
                INSERT OR REPLACE INTO morph_annotations
                (id, token_id, lemma, pos, features, annotator, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                annotation.id,
                annotation.target_id,
                annotation.lemma,
                annotation.pos,
                json.dumps(annotation.features),
                annotation.annotator,
                annotation.timestamp
            ))
        
        elif isinstance(annotation, SyntaxAnnotation):
            cursor.execute("""
                INSERT OR REPLACE INTO syntax_annotations
                (id, token_id, head, deprel, deps, annotator, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                annotation.id,
                annotation.target_id,
                annotation.head,
                annotation.deprel,
                annotation.deps,
                annotation.annotator,
                annotation.timestamp
            ))
        
        elif isinstance(annotation, NERAnnotation):
            cursor.execute("""
                INSERT OR REPLACE INTO ner_annotations
                (id, entity_type, span_start, span_end, span_text, normalized, wikidata_id, annotator, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                annotation.id,
                annotation.entity_type,
                annotation.span.start if annotation.span else 0,
                annotation.span.end if annotation.span else 0,
                annotation.span.text if annotation.span else '',
                annotation.normalized,
                annotation.wikidata_id,
                annotation.annotator,
                annotation.timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def get_annotations(self, target_id: str, layer: AnnotationLayer = None) -> List[Dict]:
        """Get annotations for a target"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if layer:
            cursor.execute("""
                SELECT * FROM annotations
                WHERE target_id = ? AND layer = ?
            """, (target_id, layer.name))
        else:
            cursor.execute("""
                SELECT * FROM annotations
                WHERE target_id = ?
            """, (target_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM annotations")
        stats['total_annotations'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT layer, COUNT(*) FROM annotations
            GROUP BY layer
        """)
        stats['by_layer'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT annotator, COUNT(*) FROM annotations
            WHERE annotator IS NOT NULL AND annotator != ''
            GROUP BY annotator
        """)
        stats['by_annotator'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats


# =============================================================================
# INTER-ANNOTATOR AGREEMENT
# =============================================================================

class InterAnnotatorAgreement:
    """Calculate inter-annotator agreement"""
    
    def __init__(self, store: AnnotationStore):
        self.store = store
    
    def cohen_kappa(self, annotator1: str, annotator2: str, 
                    layer: AnnotationLayer) -> float:
        """Calculate Cohen's Kappa"""
        conn = sqlite3.connect(self.store.db_path)
        cursor = conn.cursor()
        
        # Get annotations from both annotators
        cursor.execute("""
            SELECT target_id, value FROM annotations
            WHERE annotator = ? AND layer = ?
        """, (annotator1, layer.name))
        ann1 = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT target_id, value FROM annotations
            WHERE annotator = ? AND layer = ?
        """, (annotator2, layer.name))
        ann2 = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Find common targets
        common = set(ann1.keys()) & set(ann2.keys())
        
        if not common:
            return 0.0
        
        # Calculate agreement
        agree = sum(1 for t in common if ann1[t] == ann2[t])
        total = len(common)
        
        po = agree / total  # Observed agreement
        
        # Expected agreement
        values1 = Counter(ann1[t] for t in common)
        values2 = Counter(ann2[t] for t in common)
        all_values = set(values1.keys()) | set(values2.keys())
        
        pe = sum(
            (values1.get(v, 0) / total) * (values2.get(v, 0) / total)
            for v in all_values
        )
        
        # Kappa
        if pe == 1:
            return 1.0
        
        return (po - pe) / (1 - pe)
    
    def fleiss_kappa(self, annotators: List[str], 
                     layer: AnnotationLayer) -> float:
        """Calculate Fleiss' Kappa for multiple annotators"""
        conn = sqlite3.connect(self.store.db_path)
        cursor = conn.cursor()
        
        # Get all annotations
        annotations = defaultdict(dict)
        for annotator in annotators:
            cursor.execute("""
                SELECT target_id, value FROM annotations
                WHERE annotator = ? AND layer = ?
            """, (annotator, layer.name))
            
            for row in cursor.fetchall():
                annotations[row[0]][annotator] = row[1]
        
        conn.close()
        
        # Filter to items annotated by all
        complete = {
            t: anns for t, anns in annotations.items()
            if len(anns) == len(annotators)
        }
        
        if not complete:
            return 0.0
        
        n = len(annotators)
        N = len(complete)
        
        # Get all categories
        categories = set()
        for anns in complete.values():
            categories.update(anns.values())
        
        # Calculate P_i for each item
        P_i_sum = 0
        for anns in complete.values():
            counts = Counter(anns.values())
            P_i = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
            P_i_sum += P_i
        
        P_bar = P_i_sum / N
        
        # Calculate P_j for each category
        p_j = {}
        for cat in categories:
            count = sum(
                1 for anns in complete.values()
                for v in anns.values() if v == cat
            )
            p_j[cat] = count / (N * n)
        
        P_e = sum(p ** 2 for p in p_j.values())
        
        # Kappa
        if P_e == 1:
            return 1.0
        
        return (P_bar - P_e) / (1 - P_e)


# =============================================================================
# ANNOTATION VALIDATOR
# =============================================================================

class AnnotationValidator:
    """Validate annotations"""
    
    def __init__(self):
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
    
    def validate_morphology(self, annotation: MorphAnnotation) -> bool:
        """Validate morphological annotation"""
        valid = True
        
        # Check POS
        if annotation.pos and annotation.pos not in MorphologicalFeatures.CATEGORIES['POS']:
            self.errors.append({
                'type': 'invalid_pos',
                'annotation_id': annotation.id,
                'value': annotation.pos,
                'message': f"Invalid POS tag: {annotation.pos}"
            })
            valid = False
        
        # Check features
        for feat, value in annotation.features.items():
            if feat in MorphologicalFeatures.CATEGORIES:
                if value not in MorphologicalFeatures.CATEGORIES[feat]:
                    self.warnings.append({
                        'type': 'invalid_feature_value',
                        'annotation_id': annotation.id,
                        'feature': feat,
                        'value': value,
                        'message': f"Invalid value '{value}' for feature '{feat}'"
                    })
        
        return valid
    
    def validate_syntax(self, annotation: SyntaxAnnotation, 
                       sentence_length: int) -> bool:
        """Validate syntactic annotation"""
        valid = True
        
        # Check head
        if annotation.head < 0 or annotation.head > sentence_length:
            self.errors.append({
                'type': 'invalid_head',
                'annotation_id': annotation.id,
                'value': annotation.head,
                'message': f"Invalid head index: {annotation.head}"
            })
            valid = False
        
        # Check deprel
        if annotation.deprel and annotation.deprel not in DependencyRelations.RELATIONS:
            self.warnings.append({
                'type': 'unknown_deprel',
                'annotation_id': annotation.id,
                'value': annotation.deprel,
                'message': f"Unknown dependency relation: {annotation.deprel}"
            })
        
        return valid
    
    def validate_ner(self, annotation: NERAnnotation) -> bool:
        """Validate NER annotation"""
        valid = True
        
        if annotation.entity_type not in NamedEntityTypes.ALL_TYPES:
            self.warnings.append({
                'type': 'unknown_entity_type',
                'annotation_id': annotation.id,
                'value': annotation.entity_type,
                'message': f"Unknown entity type: {annotation.entity_type}"
            })
        
        return valid
    
    def get_report(self) -> Dict:
        """Get validation report"""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }


# =============================================================================
# EXPORT FORMATS
# =============================================================================

class AnnotationExporter:
    """Export annotations to various formats"""
    
    def __init__(self, store: AnnotationStore, corpus_db: str):
        self.store = store
        self.corpus_db = corpus_db
    
    def export_conllu(self, output_path: str, document_ids: List[str] = None):
        """Export to CoNLL-U format"""
        conn = sqlite3.connect(self.corpus_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Get sentences
            if document_ids:
                placeholders = ','.join('?' * len(document_ids))
                cursor.execute(f"""
                    SELECT * FROM sentences
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, sentence_index
                """, document_ids)
            else:
                cursor.execute("""
                    SELECT * FROM sentences
                    ORDER BY document_id, sentence_index
                """)
            
            for sent_row in cursor.fetchall():
                sent_id = sent_row['id']
                
                f.write(f"# sent_id = {sent_id}\n")
                f.write(f"# text = {sent_row['text']}\n")
                
                # Get tokens
                cursor.execute("""
                    SELECT * FROM tokens
                    WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_id,))
                
                for tok_row in cursor.fetchall():
                    tok_id = tok_row['id']
                    
                    # Get annotations
                    morph_anns = self.store.get_annotations(
                        str(tok_id), AnnotationLayer.MORPHOLOGY
                    )
                    syntax_anns = self.store.get_annotations(
                        str(tok_id), AnnotationLayer.SYNTAX
                    )
                    
                    # Build CoNLL-U line
                    idx = tok_row['token_index']
                    form = tok_row['form']
                    lemma = tok_row['lemma'] or '_'
                    upos = tok_row['upos'] or '_'
                    xpos = tok_row['xpos'] or '_'
                    feats = tok_row['feats'] or '_'
                    head = tok_row['head'] if tok_row['head'] is not None else '_'
                    deprel = tok_row['deprel'] or '_'
                    deps = '_'
                    misc = '_'
                    
                    f.write(f"{idx}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}\n")
                
                f.write("\n")
        
        conn.close()
        logger.info(f"Exported to {output_path}")
    
    def export_xml(self, output_path: str, document_ids: List[str] = None):
        """Export to XML format (PAULA-style)"""
        root = ET.Element('corpus')
        
        conn = sqlite3.connect(self.corpus_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get documents
        if document_ids:
            placeholders = ','.join('?' * len(document_ids))
            cursor.execute(f"""
                SELECT * FROM documents
                WHERE id IN ({placeholders})
            """, document_ids)
        else:
            cursor.execute("SELECT * FROM documents")
        
        for doc_row in cursor.fetchall():
            doc_elem = ET.SubElement(root, 'document')
            doc_elem.set('id', str(doc_row['id']))
            doc_elem.set('title', doc_row['title'] or '')
            
            # Get sentences
            cursor.execute("""
                SELECT * FROM sentences
                WHERE document_id = ?
                ORDER BY sentence_index
            """, (doc_row['id'],))
            
            for sent_row in cursor.fetchall():
                sent_elem = ET.SubElement(doc_elem, 'sentence')
                sent_elem.set('id', str(sent_row['id']))
                sent_elem.text = sent_row['text']
                
                # Get tokens
                cursor.execute("""
                    SELECT * FROM tokens
                    WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_row['id'],))
                
                for tok_row in cursor.fetchall():
                    tok_elem = ET.SubElement(sent_elem, 'token')
                    tok_elem.set('id', str(tok_row['id']))
                    tok_elem.set('form', tok_row['form'])
                    
                    if tok_row['lemma']:
                        tok_elem.set('lemma', tok_row['lemma'])
                    if tok_row['upos']:
                        tok_elem.set('pos', tok_row['upos'])
                    if tok_row['feats']:
                        tok_elem.set('feats', tok_row['feats'])
        
        conn.close()
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        logger.info(f"Exported to {output_path}")
    
    def export_json(self, output_path: str, document_ids: List[str] = None):
        """Export to JSON format"""
        conn = sqlite3.connect(self.corpus_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        data = {'documents': []}
        
        if document_ids:
            placeholders = ','.join('?' * len(document_ids))
            cursor.execute(f"""
                SELECT * FROM documents
                WHERE id IN ({placeholders})
            """, document_ids)
        else:
            cursor.execute("SELECT * FROM documents")
        
        for doc_row in cursor.fetchall():
            doc_data = {
                'id': doc_row['id'],
                'title': doc_row['title'],
                'sentences': []
            }
            
            cursor.execute("""
                SELECT * FROM sentences
                WHERE document_id = ?
                ORDER BY sentence_index
            """, (doc_row['id'],))
            
            for sent_row in cursor.fetchall():
                sent_data = {
                    'id': sent_row['id'],
                    'text': sent_row['text'],
                    'tokens': []
                }
                
                cursor.execute("""
                    SELECT * FROM tokens
                    WHERE sentence_id = ?
                    ORDER BY token_index
                """, (sent_row['id'],))
                
                for tok_row in cursor.fetchall():
                    tok_data = {
                        'id': tok_row['id'],
                        'form': tok_row['form'],
                        'lemma': tok_row['lemma'],
                        'pos': tok_row['upos'],
                        'feats': tok_row['feats'],
                        'head': tok_row['head'],
                        'deprel': tok_row['deprel']
                    }
                    sent_data['tokens'].append(tok_data)
                
                doc_data['sentences'].append(sent_data)
            
            data['documents'].append(doc_data)
        
        conn.close()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data/annotations.db"
    
    print("=" * 70)
    print("ANNOTATION TOOLS - Multi-layer Linguistic Annotation")
    print("=" * 70)
    
    # Initialize store
    store = AnnotationStore(db_path)
    
    # Test morphological annotation
    print("\n📝 Testing morphological annotation:")
    morph = MorphAnnotation(
        id="",
        layer=AnnotationLayer.MORPHOLOGY,
        target_id="token_1",
        value="V-3SPIA---",
        lemma="λέγω",
        pos="VERB",
        features={'Person': '3', 'Number': 'Sing', 'Tense': 'Pres', 'Mood': 'Ind', 'Voice': 'Act'}
    )
    store.add_annotation(morph)
    print(f"  Added: {morph.id}")
    
    # Test PROIEL parsing
    print("\n🔍 Testing PROIEL morphology parsing:")
    test_codes = ['V-3SPIA---', 'N--S---MN-', 'A--S---MN-']
    for code in test_codes:
        features = MorphologicalFeatures.parse_proiel(code)
        print(f"  {code}: {features}")
    
    # Get statistics
    print("\n📊 Annotation statistics:")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Annotation Tools ready!")
