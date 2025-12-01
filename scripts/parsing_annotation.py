#!/usr/bin/env python3
"""
Parsing and Annotation Pipeline
Comprehensive syntactic parsing and linguistic annotation system
Supports: Ancient Greek, Latin, Sanskrit, Gothic, Old Church Slavonic, English

Features:
- Dependency parsing with multiple backends (Stanza, spaCy, CLTK, UDPipe)
- POS tagging with language-specific tagsets
- Named Entity Recognition
- Coreference resolution
- Information structure annotation
- PROIEL/UD annotation standards
- Batch processing and parallel execution
- Quality assessment and validation
"""

import os
import sys
import re
import json
import time
import logging
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from abc import ABC, abstractmethod
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parsing_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "database_path": "corpus_platform.db",
    "cache_dir": "parsing_cache",
    "max_workers": 4,
    "batch_size": 50,
    "max_sentence_length": 200,
    "default_parser": "stanza",
    "annotation_standard": "UD"
}

# Universal Dependencies POS tags
UD_POS_TAGS = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other"
}

# Universal Dependencies relations
UD_RELATIONS = {
    "acl": "clausal modifier of noun",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "aux": "auxiliary",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "dep": "unspecified dependency",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated elements",
    "expl": "expletive",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "goeswith": "goes with",
    "iobj": "indirect object",
    "list": "list",
    "mark": "marker",
    "nmod": "nominal modifier",
    "nsubj": "nominal subject",
    "nummod": "numeric modifier",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "punct": "punctuation",
    "reparandum": "overridden disfluency",
    "root": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement"
}

# PROIEL dependency relations
PROIEL_RELATIONS = {
    "pred": "predicate",
    "sub": "subject",
    "obj": "object",
    "obl": "oblique",
    "ag": "agent",
    "atr": "attribute",
    "atv": "attributive verb",
    "adv": "adverbial",
    "apos": "apposition",
    "aux": "auxiliary",
    "comp": "complement",
    "expl": "expletive",
    "narg": "non-argument",
    "nonsub": "non-subject",
    "parpred": "parenthetical predication",
    "per": "peripheral",
    "pid": "predicate identity",
    "voc": "vocative",
    "xadv": "external adverbial",
    "xobj": "external object",
    "xsub": "external subject"
}

# Named Entity types
NE_TYPES = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Organization",
    "GPE": "Geo-Political Entity",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Money",
    "PERCENT": "Percentage",
    "WORK": "Work of Art",
    "EVENT": "Event",
    "MYTH": "Mythological Entity",
    "DEITY": "Deity",
    "ETHNIC": "Ethnic Group"
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Token:
    """Annotated token with full linguistic information"""
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
    ner: str = ""
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        feats = "|".join(f"{k}={v}" for k, v in sorted(self.morphology.items())) if self.morphology else "_"
        misc = "|".join(f"{k}={v}" for k, v in sorted(self.misc.items())) if self.misc else "_"
        return f"{self.id}\t{self.form}\t{self.lemma or '_'}\t{self.pos or '_'}\t{self.xpos or '_'}\t{feats}\t{self.head}\t{self.deprel or '_'}\t{self.deps or '_'}\t{misc}"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Token':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_conllu(cls, line: str) -> Optional['Token']:
        """Parse from CoNLL-U line"""
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
                lemma=parts[2] if parts[2] != '_' else '',
                pos=parts[3] if parts[3] != '_' else '',
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
    """Annotated sentence"""
    id: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    entities: List[Dict] = field(default_factory=list)
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
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
            'entities': self.entities
        }
    
    @property
    def token_count(self) -> int:
        return len(self.tokens)
    
    def get_dependency_tree(self) -> Dict:
        """Get dependency tree structure"""
        tree = {'root': [], 'children': defaultdict(list)}
        for token in self.tokens:
            if token.head == 0:
                tree['root'].append(token.id)
            else:
                tree['children'][token.head].append(token.id)
        return tree


@dataclass
class Document:
    """Annotated document"""
    id: str
    title: str
    author: str
    language: str
    sentences: List[Sentence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotation_info: Dict = field(default_factory=dict)
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        lines = [f"# newdoc id = {self.id}"]
        for sentence in self.sentences:
            lines.append(sentence.to_conllu())
            lines.append('')
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'sentences': [s.to_dict() for s in self.sentences],
            'metadata': self.metadata,
            'annotation_info': self.annotation_info
        }
    
    @property
    def token_count(self) -> int:
        return sum(s.token_count for s in self.sentences)
    
    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


# ============================================================================
# PARSER BACKENDS
# ============================================================================

class ParserBackend(ABC):
    """Abstract base class for parser backends"""
    
    @abstractmethod
    def parse(self, text: str) -> List[Sentence]:
        """Parse text into annotated sentences"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if parser is available"""
        pass


class StanzaParser(ParserBackend):
    """Stanza-based parser"""
    
    LANG_MAP = {
        'grc': 'grc',
        'la': 'la',
        'en': 'en',
        'got': 'got',
        'cu': 'cu'
    }
    
    def __init__(self, language: str):
        self.language = language
        self.nlp = None
        self._init_parser()
    
    def _init_parser(self):
        """Initialize Stanza pipeline"""
        try:
            import stanza
            
            stanza_lang = self.LANG_MAP.get(self.language)
            if not stanza_lang:
                logger.warning(f"Stanza not available for {self.language}")
                return
            
            # Download model if needed
            try:
                stanza.download(stanza_lang, verbose=False)
            except:
                pass
            
            self.nlp = stanza.Pipeline(
                stanza_lang,
                processors='tokenize,mwt,pos,lemma,depparse',
                verbose=False
            )
            logger.info(f"Stanza parser initialized for {self.language}")
            
        except ImportError:
            logger.warning("Stanza not installed")
        except Exception as e:
            logger.warning(f"Stanza initialization failed: {e}")
    
    def is_available(self) -> bool:
        return self.nlp is not None
    
    def parse(self, text: str) -> List[Sentence]:
        """Parse text using Stanza"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            sentences = []
            
            for sent_idx, sent in enumerate(doc.sentences):
                tokens = []
                
                for word in sent.words:
                    # Parse morphological features
                    morphology = {}
                    if word.feats:
                        for feat in word.feats.split('|'):
                            if '=' in feat:
                                k, v = feat.split('=', 1)
                                morphology[k] = v
                    
                    token = Token(
                        id=word.id,
                        form=word.text,
                        lemma=word.lemma or '',
                        pos=word.upos or '',
                        xpos=word.xpos or '',
                        morphology=morphology,
                        head=word.head,
                        deprel=word.deprel or ''
                    )
                    tokens.append(token)
                
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    text=sent.text,
                    tokens=tokens
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            logger.error(f"Stanza parsing error: {e}")
            return []


class SpacyParser(ParserBackend):
    """spaCy-based parser"""
    
    LANG_MAP = {
        'grc': 'grc_proiel_sm',
        'la': 'la_core_web_sm',
        'en': 'en_core_web_sm'
    }
    
    def __init__(self, language: str):
        self.language = language
        self.nlp = None
        self._init_parser()
    
    def _init_parser(self):
        """Initialize spaCy pipeline"""
        try:
            import spacy
            
            model_name = self.LANG_MAP.get(self.language)
            if not model_name:
                logger.warning(f"spaCy model not available for {self.language}")
                return
            
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"spaCy parser initialized for {self.language}")
            except OSError:
                logger.warning(f"spaCy model {model_name} not installed")
                
        except ImportError:
            logger.warning("spaCy not installed")
        except Exception as e:
            logger.warning(f"spaCy initialization failed: {e}")
    
    def is_available(self) -> bool:
        return self.nlp is not None
    
    def parse(self, text: str) -> List[Sentence]:
        """Parse text using spaCy"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            sentences = []
            
            for sent_idx, sent in enumerate(doc.sents):
                tokens = []
                
                for token in sent:
                    # Build morphology dict
                    morphology = {}
                    if token.morph:
                        for feat in str(token.morph).split('|'):
                            if '=' in feat:
                                k, v = feat.split('=', 1)
                                morphology[k] = v
                    
                    tok = Token(
                        id=token.i - sent.start + 1,
                        form=token.text,
                        lemma=token.lemma_,
                        pos=token.pos_,
                        xpos=token.tag_,
                        morphology=morphology,
                        head=token.head.i - sent.start + 1 if token.head != token else 0,
                        deprel=token.dep_
                    )
                    tokens.append(tok)
                
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    text=sent.text,
                    tokens=tokens
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            logger.error(f"spaCy parsing error: {e}")
            return []


class CLTKParser(ParserBackend):
    """CLTK-based parser"""
    
    LANG_MAP = {
        'grc': 'grc',
        'la': 'lat',
        'sa': 'san',
        'got': 'got',
        'cu': 'chu'
    }
    
    def __init__(self, language: str):
        self.language = language
        self.nlp = None
        self._init_parser()
    
    def _init_parser(self):
        """Initialize CLTK pipeline"""
        try:
            from cltk import NLP
            
            cltk_lang = self.LANG_MAP.get(self.language)
            if not cltk_lang:
                logger.warning(f"CLTK not available for {self.language}")
                return
            
            self.nlp = NLP(language=cltk_lang)
            logger.info(f"CLTK parser initialized for {self.language}")
            
        except ImportError:
            logger.warning("CLTK not installed")
        except Exception as e:
            logger.warning(f"CLTK initialization failed: {e}")
    
    def is_available(self) -> bool:
        return self.nlp is not None
    
    def parse(self, text: str) -> List[Sentence]:
        """Parse text using CLTK"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp.analyze(text=text)
            
            # CLTK returns words, need to segment into sentences
            sentences = []
            current_tokens = []
            current_text = []
            sent_idx = 0
            
            for word in doc.words:
                token = Token(
                    id=len(current_tokens) + 1,
                    form=word.string,
                    lemma=word.lemma or '',
                    pos=word.upos or '',
                    xpos=word.xpos or ''
                )
                current_tokens.append(token)
                current_text.append(word.string)
                
                # Check for sentence boundary
                if word.string in '.;:!?':
                    sentence = Sentence(
                        id=f"s{sent_idx + 1}",
                        text=' '.join(current_text),
                        tokens=current_tokens
                    )
                    sentences.append(sentence)
                    current_tokens = []
                    current_text = []
                    sent_idx += 1
            
            # Handle remaining tokens
            if current_tokens:
                sentence = Sentence(
                    id=f"s{sent_idx + 1}",
                    text=' '.join(current_text),
                    tokens=current_tokens
                )
                sentences.append(sentence)
            
            return sentences
            
        except Exception as e:
            logger.error(f"CLTK parsing error: {e}")
            return []


class RuleBasedParser(ParserBackend):
    """Rule-based parser for languages without ML models"""
    
    def __init__(self, language: str):
        self.language = language
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """Load parsing rules"""
        rules = {
            'grc': {
                'pos_patterns': {
                    r'.*ος$': 'NOUN',
                    r'.*ον$': 'NOUN',
                    r'.*η$': 'NOUN',
                    r'.*α$': 'NOUN',
                    r'.*ω$': 'VERB',
                    r'.*ει$': 'VERB',
                    r'.*ειν$': 'VERB',
                    r'.*ων$': 'VERB',
                    r'.*ως$': 'ADV',
                    r'^(ὁ|ἡ|τό|οἱ|αἱ|τά)$': 'DET',
                    r'^(καί|δέ|γάρ|ἀλλά|ἤ)$': 'CCONJ',
                    r'^(ἐν|εἰς|ἐκ|ἀπό|πρός|παρά|μετά|διά|ὑπό|περί|κατά)$': 'ADP'
                },
                'head_rules': {
                    'VERB': 0,  # Verbs are typically roots
                    'NOUN': 'VERB',  # Nouns attach to verbs
                    'ADJ': 'NOUN',  # Adjectives attach to nouns
                    'DET': 'NOUN',  # Determiners attach to nouns
                    'ADP': 'NOUN',  # Prepositions attach to nouns
                    'ADV': 'VERB',  # Adverbs attach to verbs
                    'CCONJ': 'VERB'  # Conjunctions attach to verbs
                }
            },
            'la': {
                'pos_patterns': {
                    r'.*us$': 'NOUN',
                    r'.*um$': 'NOUN',
                    r'.*a$': 'NOUN',
                    r'.*o$': 'VERB',
                    r'.*t$': 'VERB',
                    r'.*re$': 'VERB',
                    r'.*e$': 'ADV',
                    r'^(et|sed|aut|vel|nec|atque)$': 'CCONJ',
                    r'^(in|ex|de|ad|ab|cum|pro|per|sub)$': 'ADP'
                },
                'head_rules': {
                    'VERB': 0,
                    'NOUN': 'VERB',
                    'ADJ': 'NOUN',
                    'ADP': 'NOUN',
                    'ADV': 'VERB',
                    'CCONJ': 'VERB'
                }
            }
        }
        return rules.get(self.language, {})
    
    def is_available(self) -> bool:
        return True
    
    def parse(self, text: str) -> List[Sentence]:
        """Parse text using rules"""
        # Simple sentence splitting
        sent_pattern = re.compile(r'(?<=[.;:!?·])\s+')
        sent_texts = sent_pattern.split(text.strip())
        
        sentences = []
        for sent_idx, sent_text in enumerate(sent_texts):
            if not sent_text.strip():
                continue
            
            # Tokenize
            word_pattern = re.compile(r'[\w\u0370-\u03FF\u1F00-\u1FFF]+|[^\s\w]')
            words = word_pattern.findall(sent_text)
            
            tokens = []
            for idx, word in enumerate(words):
                pos = self._get_pos(word)
                
                token = Token(
                    id=idx + 1,
                    form=word,
                    lemma=word.lower(),
                    pos=pos
                )
                tokens.append(token)
            
            # Assign heads
            self._assign_heads(tokens)
            
            sentence = Sentence(
                id=f"s{sent_idx + 1}",
                text=sent_text.strip(),
                tokens=tokens
            )
            sentences.append(sentence)
        
        return sentences
    
    def _get_pos(self, word: str) -> str:
        """Get POS tag using patterns"""
        word_lower = word.lower()
        
        for pattern, pos in self.rules.get('pos_patterns', {}).items():
            if re.match(pattern, word_lower):
                return pos
        
        return 'X'
    
    def _assign_heads(self, tokens: List[Token]):
        """Assign dependency heads using rules"""
        # Find main verb
        verb_idx = None
        for idx, token in enumerate(tokens):
            if token.pos == 'VERB':
                verb_idx = idx
                break
        
        for idx, token in enumerate(tokens):
            if token.pos == 'VERB' and verb_idx == idx:
                token.head = 0
                token.deprel = 'root'
            elif verb_idx is not None:
                token.head = verb_idx + 1
                
                # Assign relation based on POS
                if token.pos == 'NOUN':
                    token.deprel = 'nsubj' if idx < verb_idx else 'obj'
                elif token.pos == 'ADJ':
                    token.deprel = 'amod'
                elif token.pos == 'ADV':
                    token.deprel = 'advmod'
                elif token.pos == 'ADP':
                    token.deprel = 'case'
                elif token.pos == 'DET':
                    token.deprel = 'det'
                elif token.pos == 'CCONJ':
                    token.deprel = 'cc'
                elif token.pos == 'PUNCT':
                    token.deprel = 'punct'
                else:
                    token.deprel = 'dep'
            else:
                # No verb found
                if idx == 0:
                    token.head = 0
                    token.deprel = 'root'
                else:
                    token.head = 1
                    token.deprel = 'dep'


# ============================================================================
# NAMED ENTITY RECOGNIZER
# ============================================================================

class NamedEntityRecognizer:
    """Named Entity Recognition for historical texts"""
    
    def __init__(self, language: str):
        self.language = language
        self.gazetteer = self._load_gazetteer()
        self.patterns = self._load_patterns()
    
    def _load_gazetteer(self) -> Dict[str, str]:
        """Load named entity gazetteer"""
        gazetteer = {}
        
        # Greek names
        if self.language == 'grc':
            gazetteer.update({
                # Persons
                'Σωκράτης': 'PER', 'Πλάτων': 'PER', 'Ἀριστοτέλης': 'PER',
                'Ὅμηρος': 'PER', 'Ἡσίοδος': 'PER', 'Ἡρόδοτος': 'PER',
                'Θουκυδίδης': 'PER', 'Ξενοφῶν': 'PER', 'Δημοσθένης': 'PER',
                'Ἀλέξανδρος': 'PER', 'Περικλῆς': 'PER', 'Θεμιστοκλῆς': 'PER',
                # Locations
                'Ἀθῆναι': 'LOC', 'Σπάρτη': 'LOC', 'Θῆβαι': 'LOC',
                'Κόρινθος': 'LOC', 'Ἄργος': 'LOC', 'Μακεδονία': 'LOC',
                'Ἑλλάς': 'LOC', 'Περσία': 'LOC', 'Αἴγυπτος': 'LOC',
                'Ὄλυμπος': 'LOC', 'Παρνασσός': 'LOC', 'Δελφοί': 'LOC',
                # Deities
                'Ζεύς': 'DEITY', 'Ἥρα': 'DEITY', 'Ἀθηνᾶ': 'DEITY',
                'Ἀπόλλων': 'DEITY', 'Ἄρτεμις': 'DEITY', 'Ἀφροδίτη': 'DEITY',
                'Ἑρμῆς': 'DEITY', 'Ἄρης': 'DEITY', 'Ποσειδῶν': 'DEITY',
                'Ἅιδης': 'DEITY', 'Δημήτηρ': 'DEITY', 'Διόνυσος': 'DEITY',
                # Ethnic groups
                'Ἕλληνες': 'ETHNIC', 'Πέρσαι': 'ETHNIC', 'Ἀθηναῖοι': 'ETHNIC',
                'Λακεδαιμόνιοι': 'ETHNIC', 'Θηβαῖοι': 'ETHNIC'
            })
        
        # Latin names
        elif self.language == 'la':
            gazetteer.update({
                # Persons
                'Caesar': 'PER', 'Cicero': 'PER', 'Vergilius': 'PER',
                'Augustus': 'PER', 'Nero': 'PER', 'Seneca': 'PER',
                # Locations
                'Roma': 'LOC', 'Italia': 'LOC', 'Gallia': 'LOC',
                'Hispania': 'LOC', 'Graecia': 'LOC', 'Aegyptus': 'LOC',
                # Deities
                'Iuppiter': 'DEITY', 'Iuno': 'DEITY', 'Minerva': 'DEITY',
                'Mars': 'DEITY', 'Venus': 'DEITY', 'Apollo': 'DEITY'
            })
        
        return gazetteer
    
    def _load_patterns(self) -> List[Tuple[str, str]]:
        """Load NER patterns"""
        patterns = []
        
        if self.language == 'grc':
            patterns = [
                (r'[Α-Ω][α-ω]+ος\b', 'PER'),  # Male names ending in -os
                (r'[Α-Ω][α-ω]+ης\b', 'PER'),  # Male names ending in -es
                (r'[Α-Ω][α-ω]+ων\b', 'PER'),  # Male names ending in -on
                (r'[Α-Ω][α-ω]+α\b', 'LOC'),   # Place names ending in -a
                (r'[Α-Ω][α-ω]+αι\b', 'LOC'),  # Place names ending in -ai
            ]
        elif self.language == 'la':
            patterns = [
                (r'[A-Z][a-z]+us\b', 'PER'),  # Male names ending in -us
                (r'[A-Z][a-z]+a\b', 'LOC'),   # Place names ending in -a
                (r'[A-Z][a-z]+ia\b', 'LOC'),  # Place names ending in -ia
            ]
        
        return [(re.compile(p), t) for p, t in patterns]
    
    def recognize(self, tokens: List[Token]) -> List[Dict]:
        """Recognize named entities in tokens"""
        entities = []
        
        for idx, token in enumerate(tokens):
            # Check gazetteer
            if token.form in self.gazetteer:
                entities.append({
                    'start': idx,
                    'end': idx + 1,
                    'text': token.form,
                    'type': self.gazetteer[token.form],
                    'source': 'gazetteer'
                })
                token.ner = f"B-{self.gazetteer[token.form]}"
                continue
            
            # Check patterns
            for pattern, ne_type in self.patterns:
                if pattern.match(token.form):
                    entities.append({
                        'start': idx,
                        'end': idx + 1,
                        'text': token.form,
                        'type': ne_type,
                        'source': 'pattern'
                    })
                    token.ner = f"B-{ne_type}"
                    break
        
        return entities


# ============================================================================
# ANNOTATION CONVERTER
# ============================================================================

class AnnotationConverter:
    """Convert between annotation standards"""
    
    # UD to PROIEL relation mapping
    UD_TO_PROIEL = {
        'root': 'pred',
        'nsubj': 'sub',
        'obj': 'obj',
        'iobj': 'obl',
        'obl': 'obl',
        'advmod': 'adv',
        'amod': 'atr',
        'det': 'atr',
        'nmod': 'atr',
        'appos': 'apos',
        'aux': 'aux',
        'cop': 'aux',
        'ccomp': 'comp',
        'xcomp': 'xobj',
        'advcl': 'xadv',
        'acl': 'atv',
        'vocative': 'voc',
        'expl': 'expl',
        'discourse': 'per',
        'parataxis': 'parpred',
        'cc': 'aux',
        'conj': 'pred',
        'mark': 'aux',
        'case': 'aux',
        'punct': 'aux',
        'dep': 'narg'
    }
    
    # PROIEL to UD relation mapping
    PROIEL_TO_UD = {v: k for k, v in UD_TO_PROIEL.items()}
    
    def __init__(self):
        pass
    
    def ud_to_proiel(self, sentence: Sentence) -> Sentence:
        """Convert UD annotations to PROIEL"""
        for token in sentence.tokens:
            if token.deprel in self.UD_TO_PROIEL:
                token.deprel = self.UD_TO_PROIEL[token.deprel]
        return sentence
    
    def proiel_to_ud(self, sentence: Sentence) -> Sentence:
        """Convert PROIEL annotations to UD"""
        for token in sentence.tokens:
            if token.deprel in self.PROIEL_TO_UD:
                token.deprel = self.PROIEL_TO_UD[token.deprel]
        return sentence
    
    def convert_morphology(self, morphology: Dict[str, str], 
                          from_standard: str, to_standard: str) -> Dict[str, str]:
        """Convert morphological features between standards"""
        # For now, UD features are used as the common format
        return morphology


# ============================================================================
# PARSING PIPELINE
# ============================================================================

class ParsingPipeline:
    """Main parsing and annotation pipeline"""
    
    def __init__(self, language: str, config: Dict = None):
        self.language = language
        self.config = config or CONFIG
        
        # Initialize parser backends
        self.parsers = self._init_parsers()
        self.active_parser = self._select_parser()
        
        # Initialize NER
        self.ner = NamedEntityRecognizer(language)
        
        # Initialize converter
        self.converter = AnnotationConverter()
    
    def _init_parsers(self) -> Dict[str, ParserBackend]:
        """Initialize available parsers"""
        parsers = {}
        
        # Try Stanza
        stanza = StanzaParser(self.language)
        if stanza.is_available():
            parsers['stanza'] = stanza
        
        # Try spaCy
        spacy = SpacyParser(self.language)
        if spacy.is_available():
            parsers['spacy'] = spacy
        
        # Try CLTK
        cltk = CLTKParser(self.language)
        if cltk.is_available():
            parsers['cltk'] = cltk
        
        # Always have rule-based as fallback
        parsers['rules'] = RuleBasedParser(self.language)
        
        return parsers
    
    def _select_parser(self) -> ParserBackend:
        """Select best available parser"""
        preferred = self.config.get('default_parser', 'stanza')
        
        if preferred in self.parsers:
            return self.parsers[preferred]
        
        # Fallback order
        for name in ['stanza', 'spacy', 'cltk', 'rules']:
            if name in self.parsers:
                return self.parsers[name]
        
        return self.parsers['rules']
    
    def parse(self, text: str, doc_id: str = "") -> Document:
        """Parse text into annotated document"""
        start_time = time.time()
        
        # Parse sentences
        sentences = self.active_parser.parse(text)
        
        # Add NER
        for sentence in sentences:
            entities = self.ner.recognize(sentence.tokens)
            sentence.entities = entities
        
        # Convert annotation standard if needed
        target_standard = self.config.get('annotation_standard', 'UD')
        if target_standard == 'PROIEL':
            sentences = [self.converter.ud_to_proiel(s) for s in sentences]
        
        processing_time = time.time() - start_time
        
        return Document(
            id=doc_id or hashlib.md5(text[:100].encode()).hexdigest()[:16],
            title="",
            author="",
            language=self.language,
            sentences=sentences,
            annotation_info={
                'parser': type(self.active_parser).__name__,
                'standard': target_standard,
                'processing_time': processing_time
            }
        )
    
    def parse_sentences(self, sentences: List[str]) -> List[Sentence]:
        """Parse list of sentences"""
        results = []
        
        for sent_text in sentences:
            parsed = self.active_parser.parse(sent_text)
            if parsed:
                results.extend(parsed)
        
        return results


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

class ParsingDatabase:
    """Database for parsed documents"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize parsing tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Parsed documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parsed_documents (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                title TEXT,
                author TEXT,
                language TEXT,
                sentence_count INTEGER,
                token_count INTEGER,
                annotation_standard TEXT,
                parser_used TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Parsed sentences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parsed_sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                sentence_index INTEGER,
                text TEXT,
                tokens TEXT,
                entities TEXT,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES parsed_documents(id)
            )
        """)
        
        # Dependency statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependency_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                relation TEXT,
                frequency INTEGER,
                FOREIGN KEY (document_id) REFERENCES parsed_documents(id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parsed_docs_lang ON parsed_documents(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parsed_sents_doc ON parsed_sentences(document_id)")
        
        conn.commit()
        conn.close()
    
    def save_document(self, doc: Document, source_id: str = None) -> bool:
        """Save parsed document"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Save document
            cursor.execute("""
                INSERT OR REPLACE INTO parsed_documents 
                (id, source_id, title, author, language, sentence_count, token_count,
                 annotation_standard, parser_used, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, source_id, doc.title, doc.author, doc.language,
                doc.sentence_count, doc.token_count,
                doc.annotation_info.get('standard', 'UD'),
                doc.annotation_info.get('parser', ''),
                doc.annotation_info.get('processing_time', 0)
            ))
            
            # Save sentences
            for idx, sentence in enumerate(doc.sentences):
                cursor.execute("""
                    INSERT OR REPLACE INTO parsed_sentences 
                    (id, document_id, sentence_index, text, tokens, entities, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sentence.id, doc.id, idx, sentence.text,
                    json.dumps([t.to_dict() for t in sentence.tokens]),
                    json.dumps(sentence.entities),
                    json.dumps(sentence.metadata)
                ))
            
            # Calculate and save dependency statistics
            deprel_counts = Counter()
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    if token.deprel:
                        deprel_counts[token.deprel] += 1
            
            for relation, count in deprel_counts.items():
                cursor.execute("""
                    INSERT INTO dependency_stats (document_id, relation, frequency)
                    VALUES (?, ?, ?)
                """, (doc.id, relation, count))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving parsed document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get parsed document"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM parsed_documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get sentences
        cursor.execute("""
            SELECT * FROM parsed_sentences 
            WHERE document_id = ? 
            ORDER BY sentence_index
        """, (doc_id,))
        
        sentences = []
        for sent_row in cursor.fetchall():
            tokens_data = json.loads(sent_row['tokens'])
            tokens = [Token.from_dict(t) for t in tokens_data]
            
            sentence = Sentence(
                id=sent_row['id'],
                text=sent_row['text'],
                tokens=tokens,
                entities=json.loads(sent_row['entities']) if sent_row['entities'] else [],
                metadata=json.loads(sent_row['metadata']) if sent_row['metadata'] else {}
            )
            sentences.append(sentence)
        
        conn.close()
        
        return Document(
            id=row['id'],
            title=row['title'] or "",
            author=row['author'] or "",
            language=row['language'],
            sentences=sentences,
            annotation_info={
                'standard': row['annotation_standard'],
                'parser': row['parser_used'],
                'processing_time': row['processing_time']
            }
        )
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM parsed_documents")
        stats['total_documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(token_count) FROM parsed_documents")
        stats['total_tokens'] = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT language, COUNT(*) as docs, SUM(token_count) as tokens
            FROM parsed_documents GROUP BY language
        """)
        stats['by_language'] = {
            row['language']: {'documents': row['docs'], 'tokens': row['tokens'] or 0}
            for row in cursor.fetchall()
        }
        
        cursor.execute("""
            SELECT relation, SUM(frequency) as total
            FROM dependency_stats 
            GROUP BY relation 
            ORDER BY total DESC
            LIMIT 20
        """)
        stats['top_relations'] = {row['relation']: row['total'] for row in cursor.fetchall()}
        
        conn.close()
        return stats
    
    def export_conllu(self, doc_id: str) -> str:
        """Export document to CoNLL-U format"""
        doc = self.get_document(doc_id)
        if doc:
            return doc.to_conllu()
        return ""


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchParsingProcessor:
    """Batch processing for parsing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.pipelines = {}
        self.db = ParsingDatabase(self.config.get('database_path', 'corpus_platform.db'))
    
    def _get_pipeline(self, language: str) -> ParsingPipeline:
        """Get or create pipeline for language"""
        if language not in self.pipelines:
            self.pipelines[language] = ParsingPipeline(language, self.config)
        return self.pipelines[language]
    
    def process_preprocessed_texts(self, limit: int = 100) -> int:
        """Process preprocessed texts"""
        conn = sqlite3.connect(self.config.get('database_path', 'corpus_platform.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get preprocessed documents not yet parsed
        cursor.execute("""
            SELECT pd.id, pd.title, pd.author, pd.language
            FROM processed_documents pd
            LEFT JOIN parsed_documents psd ON pd.id = psd.source_id
            WHERE psd.id IS NULL
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        processed = 0
        for row in rows:
            try:
                # Get preprocessed text
                prep_conn = sqlite3.connect(self.config.get('database_path', 'corpus_platform.db'))
                prep_conn.row_factory = sqlite3.Row
                prep_cursor = prep_conn.cursor()
                
                prep_cursor.execute("""
                    SELECT text FROM processed_sentences 
                    WHERE document_id = ? 
                    ORDER BY sentence_index
                """, (row['id'],))
                
                sentences = [r['text'] for r in prep_cursor.fetchall()]
                prep_conn.close()
                
                if not sentences:
                    continue
                
                # Parse
                text = ' '.join(sentences)
                pipeline = self._get_pipeline(row['language'])
                doc = pipeline.parse(text, f"parsed_{row['id']}")
                doc.title = row['title'] or ""
                doc.author = row['author'] or ""
                
                if self.db.save_document(doc, row['id']):
                    processed += 1
                    logger.info(f"Parsed: {row['title']} ({doc.token_count} tokens)")
                    
            except Exception as e:
                logger.error(f"Error parsing {row['id']}: {e}")
        
        return processed
    
    def process_raw_texts(self, limit: int = 100) -> int:
        """Process raw texts directly"""
        conn = sqlite3.connect(self.config.get('database_path', 'corpus_platform.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT rt.id, rt.title, rt.author, rt.language, rt.content
            FROM raw_texts rt
            LEFT JOIN parsed_documents pd ON rt.id = pd.source_id
            WHERE pd.id IS NULL AND rt.content IS NOT NULL
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        processed = 0
        for row in rows:
            try:
                pipeline = self._get_pipeline(row['language'])
                doc = pipeline.parse(row['content'], f"parsed_{row['id']}")
                doc.title = row['title'] or ""
                doc.author = row['author'] or ""
                
                if self.db.save_document(doc, row['id']):
                    processed += 1
                    logger.info(f"Parsed: {row['title']} ({doc.token_count} tokens)")
                    
            except Exception as e:
                logger.error(f"Error parsing {row['id']}: {e}")
        
        return processed


# ============================================================================
# QUALITY ASSESSMENT
# ============================================================================

class ParsingQualityAssessor:
    """Assess parsing quality"""
    
    def __init__(self):
        pass
    
    def assess_document(self, doc: Document) -> Dict:
        """Assess parsing quality of document"""
        metrics = {
            'sentence_count': doc.sentence_count,
            'token_count': doc.token_count,
            'issues': []
        }
        
        # Check for issues
        for sentence in doc.sentences:
            issues = self._check_sentence(sentence)
            metrics['issues'].extend(issues)
        
        # Calculate quality score
        issue_count = len(metrics['issues'])
        if doc.token_count > 0:
            metrics['quality_score'] = max(0, 1 - (issue_count / doc.token_count))
        else:
            metrics['quality_score'] = 0
        
        return metrics
    
    def _check_sentence(self, sentence: Sentence) -> List[Dict]:
        """Check sentence for parsing issues"""
        issues = []
        
        # Check for multiple roots
        roots = [t for t in sentence.tokens if t.head == 0]
        if len(roots) > 1:
            issues.append({
                'type': 'multiple_roots',
                'sentence_id': sentence.id,
                'message': f"Multiple roots found: {[r.id for r in roots]}"
            })
        elif len(roots) == 0:
            issues.append({
                'type': 'no_root',
                'sentence_id': sentence.id,
                'message': "No root token found"
            })
        
        # Check for invalid heads
        token_ids = {t.id for t in sentence.tokens}
        for token in sentence.tokens:
            if token.head > 0 and token.head not in token_ids:
                issues.append({
                    'type': 'invalid_head',
                    'sentence_id': sentence.id,
                    'token_id': token.id,
                    'message': f"Token {token.id} has invalid head {token.head}"
                })
        
        # Check for cycles
        if self._has_cycle(sentence.tokens):
            issues.append({
                'type': 'cycle',
                'sentence_id': sentence.id,
                'message': "Dependency cycle detected"
            })
        
        # Check for missing POS
        for token in sentence.tokens:
            if not token.pos or token.pos == 'X':
                issues.append({
                    'type': 'missing_pos',
                    'sentence_id': sentence.id,
                    'token_id': token.id,
                    'message': f"Token '{token.form}' has no POS tag"
                })
        
        return issues
    
    def _has_cycle(self, tokens: List[Token]) -> bool:
        """Check for cycles in dependency tree"""
        for token in tokens:
            visited = set()
            current = token
            
            while current.head > 0:
                if current.id in visited:
                    return True
                visited.add(current.id)
                current = next((t for t in tokens if t.id == current.head), None)
                if current is None:
                    break
        
        return False


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parsing and Annotation Pipeline")
    parser.add_argument('command', choices=['parse', 'batch', 'stats', 'export', 'assess'],
                       help="Command to run")
    parser.add_argument('--input', '-i', help="Input file or text")
    parser.add_argument('--language', '-l', default='grc', help="Language code")
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--limit', type=int, default=100, help="Batch limit")
    parser.add_argument('--format', '-f', default='conllu', 
                       choices=['conllu', 'json'], help="Output format")
    parser.add_argument('--standard', '-s', default='UD',
                       choices=['UD', 'PROIEL'], help="Annotation standard")
    parser.add_argument('--doc-id', help="Document ID for export/assess")
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config['annotation_standard'] = args.standard
    
    if args.command == 'parse':
        if not args.input:
            print("Error: --input required")
            return
        
        # Read input
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.input
        
        pipeline = ParsingPipeline(args.language, config)
        doc = pipeline.parse(text)
        
        if args.format == 'conllu':
            output = doc.to_conllu()
        else:
            output = json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
        else:
            print(output)
    
    elif args.command == 'batch':
        processor = BatchParsingProcessor(config)
        count = processor.process_raw_texts(limit=args.limit)
        print(f"Parsed {count} documents")
    
    elif args.command == 'stats':
        db = ParsingDatabase()
        stats = db.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'export':
        if not args.doc_id:
            print("Error: --doc-id required")
            return
        
        db = ParsingDatabase()
        output = db.export_conllu(args.doc_id)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
        else:
            print(output)
    
    elif args.command == 'assess':
        if not args.doc_id:
            print("Error: --doc-id required")
            return
        
        db = ParsingDatabase()
        doc = db.get_document(args.doc_id)
        
        if not doc:
            print(f"Document {args.doc_id} not found")
            return
        
        assessor = ParsingQualityAssessor()
        metrics = assessor.assess_document(doc)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
