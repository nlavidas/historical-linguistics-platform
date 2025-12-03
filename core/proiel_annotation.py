"""
PROIEL-Style Annotation System
Full implementation of PROIEL/Syntacticus annotation standards
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
# PROIEL POS TAGS
# =============================================================================

class PROIELPos(Enum):
    """PROIEL Part-of-Speech tags"""
    # Nominals
    Nb = "Nb"  # Common noun
    Ne = "Ne"  # Proper noun
    Nb_ABBR = "Nb"  # Abbreviated noun
    
    # Adjectives
    A_ = "A-"  # Adjective
    
    # Pronouns
    Pp = "Pp"  # Personal pronoun
    Pk = "Pk"  # Reflexive pronoun
    Ps = "Ps"  # Possessive pronoun
    Pd = "Pd"  # Demonstrative pronoun
    Pi = "Pi"  # Interrogative pronoun
    Pr = "Pr"  # Relative pronoun
    Px = "Px"  # Indefinite pronoun
    Pc = "Pc"  # Reciprocal pronoun
    
    # Determiners
    S_ = "S-"  # Article/determiner
    
    # Verbs
    V_ = "V-"  # Verb (finite or non-finite)
    
    # Adverbs
    Df = "Df"  # Adverb
    Dq = "Dq"  # Relative adverb
    Du = "Du"  # Interrogative adverb
    
    # Prepositions
    R_ = "R-"  # Preposition
    
    # Conjunctions
    C_ = "C-"  # Conjunction
    G_ = "G-"  # Subjunction
    
    # Interjections
    I_ = "I-"  # Interjection
    
    # Numerals
    Ma = "Ma"  # Cardinal numeral
    Mo = "Mo"  # Ordinal numeral
    
    # Particles
    N_ = "N-"  # Negative particle
    
    # Punctuation
    X_ = "X-"  # Punctuation
    
    # Foreign/Unknown
    F_ = "F-"  # Foreign word


# =============================================================================
# PROIEL MORPHOLOGY
# =============================================================================

@dataclass
class PROIELMorphology:
    """PROIEL morphological features"""
    
    # Person (1, 2, 3)
    person: str = "-"
    
    # Number (s=singular, d=dual, p=plural)
    number: str = "-"
    
    # Tense (p=present, i=imperfect, f=future, a=aorist, r=perfect, l=pluperfect, t=future perfect)
    tense: str = "-"
    
    # Mood (i=indicative, s=subjunctive, o=optative, m=imperative, n=infinitive, p=participle, d=gerund, g=gerundive, u=supine)
    mood: str = "-"
    
    # Voice (a=active, m=middle, p=passive, e=medio-passive)
    voice: str = "-"
    
    # Gender (m=masculine, f=feminine, n=neuter)
    gender: str = "-"
    
    # Case (n=nominative, g=genitive, d=dative, a=accusative, v=vocative, b=ablative, l=locative, i=instrumental)
    case: str = "-"
    
    # Degree (p=positive, c=comparative, s=superlative)
    degree: str = "-"
    
    # Strength (w=weak, s=strong) - for adjectives
    strength: str = "-"
    
    # Inflection (n=non-inflecting, i=inflecting)
    inflection: str = "-"
    
    def to_tag(self) -> str:
        """Convert to PROIEL morphology tag string"""
        return f"{self.person}{self.number}{self.tense}{self.mood}{self.voice}{self.gender}{self.case}{self.degree}{self.strength}{self.inflection}"
    
    @classmethod
    def from_tag(cls, tag: str) -> 'PROIELMorphology':
        """Parse PROIEL morphology tag string"""
        if len(tag) < 10:
            tag = tag + "-" * (10 - len(tag))
        
        return cls(
            person=tag[0],
            number=tag[1],
            tense=tag[2],
            mood=tag[3],
            voice=tag[4],
            gender=tag[5],
            case=tag[6],
            degree=tag[7],
            strength=tag[8] if len(tag) > 8 else "-",
            inflection=tag[9] if len(tag) > 9 else "-"
        )
    
    def to_ud_feats(self) -> str:
        """Convert to Universal Dependencies FEATS format"""
        feats = []
        
        # Person
        if self.person in '123':
            feats.append(f"Person={self.person}")
        
        # Number
        num_map = {'s': 'Sing', 'p': 'Plur', 'd': 'Dual'}
        if self.number in num_map:
            feats.append(f"Number={num_map[self.number]}")
        
        # Tense
        tense_map = {
            'p': 'Pres', 'i': 'Imp', 'f': 'Fut', 
            'a': 'Past', 'r': 'Perf', 'l': 'Pqp', 't': 'FutPerf'
        }
        if self.tense in tense_map:
            feats.append(f"Tense={tense_map[self.tense]}")
        
        # Mood
        mood_map = {
            'i': 'Ind', 's': 'Sub', 'o': 'Opt', 
            'm': 'Imp', 'n': 'Inf', 'p': 'Part'
        }
        if self.mood in mood_map:
            feats.append(f"Mood={mood_map[self.mood]}")
        
        # Voice
        voice_map = {'a': 'Act', 'm': 'Mid', 'p': 'Pass', 'e': 'Mid'}
        if self.voice in voice_map:
            feats.append(f"Voice={voice_map[self.voice]}")
        
        # Gender
        gender_map = {'m': 'Masc', 'f': 'Fem', 'n': 'Neut'}
        if self.gender in gender_map:
            feats.append(f"Gender={gender_map[self.gender]}")
        
        # Case
        case_map = {
            'n': 'Nom', 'g': 'Gen', 'd': 'Dat', 
            'a': 'Acc', 'v': 'Voc', 'b': 'Abl', 
            'l': 'Loc', 'i': 'Ins'
        }
        if self.case in case_map:
            feats.append(f"Case={case_map[self.case]}")
        
        # Degree
        degree_map = {'p': 'Pos', 'c': 'Cmp', 's': 'Sup'}
        if self.degree in degree_map:
            feats.append(f"Degree={degree_map[self.degree]}")
        
        return '|'.join(feats) if feats else '_'


# =============================================================================
# PROIEL DEPENDENCY RELATIONS
# =============================================================================

class PROIELRelation(Enum):
    """PROIEL dependency relations"""
    
    # Core arguments
    pred = "pred"       # Predicate (root)
    sub = "sub"         # Subject
    obj = "obj"         # Object
    obl = "obl"         # Oblique
    
    # Agent
    ag = "ag"           # Agent (in passive)
    
    # Modifiers
    atr = "atr"         # Attribute
    adv = "adv"         # Adverbial
    
    # Apposition
    apos = "apos"       # Apposition
    
    # Auxiliary elements
    aux = "aux"         # Auxiliary
    
    # Complements
    comp = "comp"       # Complement
    
    # Expletive
    expl = "expl"       # Expletive
    
    # Non-arguments
    narg = "narg"       # Non-argument
    
    # Non-subject ex-argument
    nonsub = "nonsub"   # Non-subject ex-argument
    
    # Parenthetical
    parpred = "parpred" # Parenthetical predication
    
    # Peripheral
    per = "per"         # Peripheral
    
    # Predicate identity
    pid = "pid"         # Predicate identity
    
    # Vocative
    voc = "voc"         # Vocative
    
    # External arguments (open complements)
    xadv = "xadv"       # External adverbial (open adverbial complement)
    xobj = "xobj"       # External object (open objective complement)
    xsub = "xsub"       # External subject
    
    # Coordination
    part = "part"       # Partitive


# Mapping PROIEL relations to UD relations
PROIEL_TO_UD_RELATIONS = {
    'pred': 'root',
    'sub': 'nsubj',
    'obj': 'obj',
    'obl': 'obl',
    'ag': 'obl:agent',
    'atr': 'amod',
    'adv': 'advmod',
    'apos': 'appos',
    'aux': 'aux',
    'comp': 'ccomp',
    'expl': 'expl',
    'narg': 'dep',
    'nonsub': 'dep',
    'parpred': 'parataxis',
    'per': 'dep',
    'pid': 'cop',
    'voc': 'vocative',
    'xadv': 'advcl',
    'xobj': 'xcomp',
    'xsub': 'nsubj',
    'part': 'conj',
}


# =============================================================================
# PROIEL TOKEN
# =============================================================================

@dataclass
class PROIELToken:
    """A token in PROIEL format"""
    
    # Required fields
    id: int
    form: str
    
    # Lemma and morphology
    lemma: str = ""
    pos: str = "X-"
    morphology: str = "----------"
    
    # Syntax
    head_id: int = 0
    relation: str = ""
    
    # Information structure
    information_status: str = ""  # new, acc-gen, acc-inf, old, etc.
    
    # Presentation
    presentation_before: str = ""
    presentation_after: str = ""
    
    # Flags
    foreign: bool = False
    empty_token_sort: str = ""  # For empty tokens: P, C, V
    
    # Semantic annotation
    semantic_role: str = ""
    
    # Gloss
    gloss: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        # Parse morphology
        morph = PROIELMorphology.from_tag(self.morphology)
        ud_feats = morph.to_ud_feats()
        
        # Map relation
        ud_rel = PROIEL_TO_UD_RELATIONS.get(self.relation, 'dep')
        
        # Map POS to UPOS
        upos = self._map_pos_to_upos()
        
        return '\t'.join([
            str(self.id),
            self.form,
            self.lemma or '_',
            upos,
            self.pos,  # XPOS
            ud_feats,
            str(self.head_id),
            ud_rel,
            '_',  # DEPS
            self._format_misc()
        ])
    
    def _map_pos_to_upos(self) -> str:
        """Map PROIEL POS to Universal POS"""
        pos_map = {
            'Nb': 'NOUN', 'Ne': 'PROPN',
            'A-': 'ADJ',
            'Pp': 'PRON', 'Pk': 'PRON', 'Ps': 'DET', 'Pd': 'DET',
            'Pi': 'PRON', 'Pr': 'PRON', 'Px': 'PRON', 'Pc': 'PRON',
            'S-': 'DET',
            'V-': 'VERB',
            'Df': 'ADV', 'Dq': 'ADV', 'Du': 'ADV',
            'R-': 'ADP',
            'C-': 'CCONJ', 'G-': 'SCONJ',
            'I-': 'INTJ',
            'Ma': 'NUM', 'Mo': 'ADJ',
            'N-': 'PART',
            'X-': 'PUNCT',
            'F-': 'X',
        }
        return pos_map.get(self.pos, 'X')
    
    def _format_misc(self) -> str:
        """Format MISC field"""
        misc_parts = []
        
        if self.gloss:
            misc_parts.append(f"Gloss={self.gloss}")
        if self.information_status:
            misc_parts.append(f"InfStat={self.information_status}")
        if self.semantic_role:
            misc_parts.append(f"SemRole={self.semantic_role}")
        if self.presentation_before:
            misc_parts.append(f"SpaceBefore={self.presentation_before}")
        if self.presentation_after:
            misc_parts.append(f"SpaceAfter={self.presentation_after}")
        
        return '|'.join(misc_parts) if misc_parts else '_'


# =============================================================================
# PROIEL SENTENCE
# =============================================================================

@dataclass
class PROIELSentence:
    """A sentence in PROIEL format"""
    
    id: str
    tokens: List[PROIELToken] = field(default_factory=list)
    
    # Metadata
    citation: str = ""
    presentation: str = ""
    
    # Status
    status: str = "unannotated"  # unannotated, annotated, reviewed
    annotator: str = ""
    reviewer: str = ""
    
    def get_text(self) -> str:
        """Reconstruct sentence text"""
        parts = []
        for token in self.tokens:
            if token.presentation_before:
                parts.append(token.presentation_before)
            parts.append(token.form)
            if token.presentation_after:
                parts.append(token.presentation_after)
            else:
                parts.append(' ')
        return ''.join(parts).strip()
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        lines = [
            f"# sent_id = {self.id}",
            f"# text = {self.get_text()}"
        ]
        
        if self.citation:
            lines.append(f"# citation = {self.citation}")
        
        for token in self.tokens:
            lines.append(token.to_conllu())
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.get_text(),
            'citation': self.citation,
            'status': self.status,
            'tokens': [t.to_dict() for t in self.tokens]
        }
    
    def validate(self) -> List[str]:
        """Validate sentence annotation"""
        errors = []
        
        # Check for root
        roots = [t for t in self.tokens if t.head_id == 0]
        if len(roots) == 0:
            errors.append("No root token found")
        elif len(roots) > 1:
            errors.append(f"Multiple roots found: {[t.id for t in roots]}")
        
        # Check for cycles
        if self._has_cycle():
            errors.append("Dependency cycle detected")
        
        # Check head references
        token_ids = {t.id for t in self.tokens}
        for token in self.tokens:
            if token.head_id != 0 and token.head_id not in token_ids:
                errors.append(f"Token {token.id} has invalid head {token.head_id}")
        
        # Check projectivity (optional - non-projective is allowed)
        
        return errors
    
    def _has_cycle(self) -> bool:
        """Check for dependency cycles"""
        visited = set()
        
        for token in self.tokens:
            current = token.id
            path = set()
            
            while current != 0:
                if current in path:
                    return True
                path.add(current)
                
                # Find head
                head_token = next((t for t in self.tokens if t.id == current), None)
                if head_token is None:
                    break
                current = head_token.head_id
        
        return False


# =============================================================================
# PROIEL SOURCE (DOCUMENT)
# =============================================================================

@dataclass
class PROIELSource:
    """A source document in PROIEL format"""
    
    id: str
    title: str
    author: str = ""
    
    # Classification
    language: str = "grc"
    period: str = ""
    genre: str = ""
    
    # Sentences
    sentences: List[PROIELSentence] = field(default_factory=list)
    
    # Metadata
    citation_part: str = ""
    edition: str = ""
    editor: str = ""
    
    def get_statistics(self) -> Dict:
        """Get source statistics"""
        total_tokens = sum(len(s.tokens) for s in self.sentences)
        
        # POS distribution
        pos_dist = {}
        for sent in self.sentences:
            for token in sent.tokens:
                pos_dist[token.pos] = pos_dist.get(token.pos, 0) + 1
        
        # Relation distribution
        rel_dist = {}
        for sent in self.sentences:
            for token in sent.tokens:
                rel_dist[token.relation] = rel_dist.get(token.relation, 0) + 1
        
        return {
            'sentence_count': len(self.sentences),
            'token_count': total_tokens,
            'pos_distribution': pos_dist,
            'relation_distribution': rel_dist
        }
    
    def to_xml(self) -> str:
        """Convert to PROIEL XML format"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<proiel>',
            f'  <source id="{self.id}" language="{self.language}">',
            f'    <title>{self._escape_xml(self.title)}</title>',
        ]
        
        if self.author:
            lines.append(f'    <author>{self._escape_xml(self.author)}</author>')
        
        for sent in self.sentences:
            lines.append(f'    <sentence id="{sent.id}" status="{sent.status}">')
            
            for token in sent.tokens:
                attrs = [
                    f'id="{token.id}"',
                    f'form="{self._escape_xml(token.form)}"',
                ]
                
                if token.lemma:
                    attrs.append(f'lemma="{self._escape_xml(token.lemma)}"')
                if token.pos:
                    attrs.append(f'part-of-speech="{token.pos}"')
                if token.morphology and token.morphology != "----------":
                    attrs.append(f'morphology="{token.morphology}"')
                if token.head_id:
                    attrs.append(f'head-id="{token.head_id}"')
                if token.relation:
                    attrs.append(f'relation="{token.relation}"')
                if token.gloss:
                    attrs.append(f'gloss="{self._escape_xml(token.gloss)}"')
                
                lines.append(f'      <token {" ".join(attrs)}/>')
            
            lines.append('    </sentence>')
        
        lines.extend([
            '  </source>',
            '</proiel>'
        ])
        
        return '\n'.join(lines)
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        parts = []
        
        # Document metadata
        parts.append(f"# newdoc id = {self.id}")
        parts.append(f"# title = {self.title}")
        if self.author:
            parts.append(f"# author = {self.author}")
        parts.append("")
        
        # Sentences
        for sent in self.sentences:
            parts.append(sent.to_conllu())
            parts.append("")
        
        return '\n'.join(parts)
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))


# =============================================================================
# ANNOTATION EDITOR
# =============================================================================

class AnnotationEditor:
    """Editor for PROIEL annotations"""
    
    def __init__(self, sentence: PROIELSentence):
        self.sentence = sentence
        self.history: List[Dict] = []
        
    def set_lemma(self, token_id: int, lemma: str):
        """Set lemma for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.lemma = lemma
    
    def set_pos(self, token_id: int, pos: str):
        """Set POS tag for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.pos = pos
    
    def set_morphology(self, token_id: int, morphology: str):
        """Set morphology for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.morphology = morphology
    
    def set_head(self, token_id: int, head_id: int):
        """Set head for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.head_id = head_id
    
    def set_relation(self, token_id: int, relation: str):
        """Set dependency relation for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.relation = relation
    
    def set_gloss(self, token_id: int, gloss: str):
        """Set gloss for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.gloss = gloss
    
    def set_semantic_role(self, token_id: int, role: str):
        """Set semantic role for a token"""
        token = self._get_token(token_id)
        if token:
            self._save_state()
            token.semantic_role = role
    
    def undo(self) -> bool:
        """Undo last change"""
        if self.history:
            state = self.history.pop()
            self._restore_state(state)
            return True
        return False
    
    def _get_token(self, token_id: int) -> Optional[PROIELToken]:
        """Get token by ID"""
        for token in self.sentence.tokens:
            if token.id == token_id:
                return token
        return None
    
    def _save_state(self):
        """Save current state for undo"""
        state = {
            'tokens': [asdict(t) for t in self.sentence.tokens]
        }
        self.history.append(state)
        
        # Limit history size
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def _restore_state(self, state: Dict):
        """Restore state from history"""
        self.sentence.tokens = [
            PROIELToken(**t) for t in state['tokens']
        ]


# =============================================================================
# ANNOTATION DATABASE
# =============================================================================

class AnnotationDatabase:
    """Database for storing PROIEL annotations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Sources table
        c.execute('''CREATE TABLE IF NOT EXISTS proiel_sources (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            language TEXT DEFAULT 'grc',
            period TEXT,
            genre TEXT,
            edition TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Sentences table
        c.execute('''CREATE TABLE IF NOT EXISTS proiel_sentences (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            sentence_num INTEGER,
            citation TEXT,
            status TEXT DEFAULT 'unannotated',
            annotator TEXT,
            reviewer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES proiel_sources(id)
        )''')
        
        # Tokens table
        c.execute('''CREATE TABLE IF NOT EXISTS proiel_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id TEXT NOT NULL,
            token_num INTEGER NOT NULL,
            form TEXT NOT NULL,
            lemma TEXT,
            pos TEXT,
            morphology TEXT,
            head_id INTEGER,
            relation TEXT,
            information_status TEXT,
            semantic_role TEXT,
            gloss TEXT,
            presentation_before TEXT,
            presentation_after TEXT,
            FOREIGN KEY (sentence_id) REFERENCES proiel_sentences(id)
        )''')
        
        # Indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_proiel_tokens_lemma ON proiel_tokens(lemma)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_proiel_tokens_pos ON proiel_tokens(pos)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_proiel_tokens_relation ON proiel_tokens(relation)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_proiel_sentences_source ON proiel_sentences(source_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_proiel_sentences_status ON proiel_sentences(status)')
        
        conn.commit()
        conn.close()
    
    def store_source(self, source: PROIELSource):
        """Store a PROIEL source"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Insert source
        c.execute('''INSERT OR REPLACE INTO proiel_sources 
                    (id, title, author, language, period, genre, edition, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                 (source.id, source.title, source.author, source.language,
                  source.period, source.genre, source.edition))
        
        # Insert sentences
        for i, sent in enumerate(source.sentences):
            c.execute('''INSERT OR REPLACE INTO proiel_sentences
                        (id, source_id, sentence_num, citation, status, annotator, reviewer, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                     (sent.id, source.id, i, sent.citation, sent.status,
                      sent.annotator, sent.reviewer))
            
            # Delete old tokens
            c.execute('DELETE FROM proiel_tokens WHERE sentence_id = ?', (sent.id,))
            
            # Insert tokens
            for token in sent.tokens:
                c.execute('''INSERT INTO proiel_tokens
                            (sentence_id, token_num, form, lemma, pos, morphology,
                             head_id, relation, information_status, semantic_role,
                             gloss, presentation_before, presentation_after)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (sent.id, token.id, token.form, token.lemma, token.pos,
                          token.morphology, token.head_id, token.relation,
                          token.information_status, token.semantic_role,
                          token.gloss, token.presentation_before, token.presentation_after))
        
        conn.commit()
        conn.close()
    
    def get_source(self, source_id: str) -> Optional[PROIELSource]:
        """Get a PROIEL source by ID"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Get source
        c.execute('SELECT * FROM proiel_sources WHERE id = ?', (source_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return None
        
        source = PROIELSource(
            id=row['id'],
            title=row['title'],
            author=row['author'] or '',
            language=row['language'],
            period=row['period'] or '',
            genre=row['genre'] or '',
            edition=row['edition'] or ''
        )
        
        # Get sentences
        c.execute('''SELECT * FROM proiel_sentences 
                    WHERE source_id = ? ORDER BY sentence_num''', (source_id,))
        
        for sent_row in c.fetchall():
            sent = PROIELSentence(
                id=sent_row['id'],
                citation=sent_row['citation'] or '',
                status=sent_row['status'],
                annotator=sent_row['annotator'] or '',
                reviewer=sent_row['reviewer'] or ''
            )
            
            # Get tokens
            c.execute('''SELECT * FROM proiel_tokens 
                        WHERE sentence_id = ? ORDER BY token_num''', (sent.id,))
            
            for tok_row in c.fetchall():
                token = PROIELToken(
                    id=tok_row['token_num'],
                    form=tok_row['form'],
                    lemma=tok_row['lemma'] or '',
                    pos=tok_row['pos'] or 'X-',
                    morphology=tok_row['morphology'] or '----------',
                    head_id=tok_row['head_id'] or 0,
                    relation=tok_row['relation'] or '',
                    information_status=tok_row['information_status'] or '',
                    semantic_role=tok_row['semantic_role'] or '',
                    gloss=tok_row['gloss'] or '',
                    presentation_before=tok_row['presentation_before'] or '',
                    presentation_after=tok_row['presentation_after'] or ''
                )
                sent.tokens.append(token)
            
            source.sentences.append(sent)
        
        conn.close()
        return source
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM proiel_sources")
        stats['source_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM proiel_sentences")
        stats['sentence_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM proiel_tokens")
        stats['token_count'] = c.fetchone()[0]
        
        c.execute("SELECT status, COUNT(*) FROM proiel_sentences GROUP BY status")
        stats['by_status'] = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT pos, COUNT(*) FROM proiel_tokens GROUP BY pos ORDER BY COUNT(*) DESC")
        stats['pos_distribution'] = {r[0]: r[1] for r in c.fetchall()}
        
        c.execute("SELECT relation, COUNT(*) FROM proiel_tokens WHERE relation != '' GROUP BY relation ORDER BY COUNT(*) DESC")
        stats['relation_distribution'] = {r[0]: r[1] for r in c.fetchall()}
        
        conn.close()
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test PROIEL annotation
    
    # Create a sample sentence
    tokens = [
        PROIELToken(id=1, form='Ἐν', lemma='ἐν', pos='R-', morphology='----------', head_id=2, relation='adv'),
        PROIELToken(id=2, form='ἀρχῇ', lemma='ἀρχή', pos='Nb', morphology='-s---fd---', head_id=3, relation='obl'),
        PROIELToken(id=3, form='ἦν', lemma='εἰμί', pos='V-', morphology='3sii-a----', head_id=0, relation='pred'),
        PROIELToken(id=4, form='ὁ', lemma='ὁ', pos='S-', morphology='-s---mn---', head_id=5, relation='atr'),
        PROIELToken(id=5, form='λόγος', lemma='λόγος', pos='Nb', morphology='-s---mn---', head_id=3, relation='sub'),
    ]
    
    sentence = PROIELSentence(
        id='john_1_1',
        tokens=tokens,
        citation='John 1:1',
        status='annotated'
    )
    
    # Validate
    errors = sentence.validate()
    print(f"Validation errors: {errors}")
    
    # Convert to CoNLL-U
    conllu = sentence.to_conllu()
    print("\nCoNLL-U format:")
    print(conllu)
    
    # Create source
    source = PROIELSource(
        id='john',
        title='Gospel of John',
        author='John',
        language='grc',
        period='koine',
        genre='religious',
        sentences=[sentence]
    )
    
    # Convert to XML
    xml = source.to_xml()
    print("\nPROIEL XML format:")
    print(xml[:500] + "...")
    
    # Statistics
    stats = source.get_statistics()
    print(f"\nStatistics: {stats}")
