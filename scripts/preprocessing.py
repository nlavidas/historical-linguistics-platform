#!/usr/bin/env python3
"""
Text Preprocessing and Lemmatization Pipeline
Comprehensive preprocessing for historical linguistic texts
Supports: Ancient Greek, Latin, Sanskrit, Gothic, Old Church Slavonic, English

Features:
- Unicode normalization and cleaning
- Sentence and word tokenization
- Lemmatization with multiple backends (CLTK, Stanza, spaCy)
- Morphological analysis
- Named entity recognition
- Stopword removal
- N-gram extraction
- Text statistics
"""

import os
import sys
import re
import json
import time
import logging
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from abc import ABC, abstractmethod
import pickle
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "database_path": "corpus_platform.db",
    "cache_dir": "preprocessing_cache",
    "max_workers": 4,
    "batch_size": 100,
    "min_word_length": 1,
    "max_word_length": 50,
    "sentence_min_tokens": 3,
    "sentence_max_tokens": 200
}

# Language-specific configurations
LANGUAGE_CONFIG = {
    "grc": {
        "name": "Ancient Greek",
        "script": "greek",
        "word_pattern": r'[\u0370-\u03FF\u1F00-\u1FFF]+',
        "sentence_delimiters": r'[.;:·!?]',
        "preserve_diacritics": True,
        "stopwords": [
            "ὁ", "ἡ", "τό", "τοῦ", "τῆς", "τῷ", "τήν", "τόν",
            "οἱ", "αἱ", "τά", "τῶν", "τοῖς", "ταῖς", "τούς", "τάς",
            "καί", "δέ", "τε", "γάρ", "μέν", "οὖν", "ἀλλά", "ἤ",
            "εἰ", "ὅτι", "ὡς", "ἐν", "εἰς", "ἐκ", "ἀπό", "πρός",
            "παρά", "μετά", "διά", "ὑπό", "περί", "κατά", "ἐπί",
            "οὐ", "οὐκ", "οὐχ", "μή", "αὐτός", "αὐτή", "αὐτό",
            "οὗτος", "αὕτη", "τοῦτο", "ἐκεῖνος", "ὅς", "ἥ", "ὅ",
            "τις", "τι", "ἄν", "εἶναι", "ἐστί", "ἦν"
        ],
        "punctuation": ".,;:·!?()[]{}\"'«»—–-"
    },
    "la": {
        "name": "Latin",
        "script": "latin",
        "word_pattern": r'[a-zA-ZāēīōūȳĀĒĪŌŪȲæœÆŒ]+',
        "sentence_delimiters": r'[.;:!?]',
        "preserve_diacritics": True,
        "stopwords": [
            "et", "in", "est", "non", "cum", "ad", "ut", "sed",
            "qui", "quae", "quod", "de", "ex", "ab", "per", "pro",
            "si", "aut", "vel", "nec", "neque", "atque", "ac",
            "hic", "haec", "hoc", "ille", "illa", "illud",
            "is", "ea", "id", "ipse", "ipsa", "ipsum",
            "ego", "tu", "nos", "vos", "se", "sui", "sibi",
            "sum", "esse", "fui", "eram", "ero"
        ],
        "punctuation": ".,;:!?()[]{}\"'-"
    },
    "sa": {
        "name": "Sanskrit",
        "script": "devanagari",
        "word_pattern": r'[\u0900-\u097F]+',
        "sentence_delimiters": r'[।॥]',
        "preserve_diacritics": True,
        "stopwords": ["च", "वा", "तु", "हि", "एव", "अपि", "न", "यत्", "तत्"],
        "punctuation": "।॥,;:!?"
    },
    "got": {
        "name": "Gothic",
        "script": "gothic",
        "word_pattern": r'[a-zA-Z𐌰-𐍊]+',
        "sentence_delimiters": r'[.;:!?]',
        "preserve_diacritics": True,
        "stopwords": ["jah", "in", "sa", "so", "þata", "ak", "ni", "du"],
        "punctuation": ".,;:!?()[]{}\"'-"
    },
    "cu": {
        "name": "Old Church Slavonic",
        "script": "cyrillic",
        "word_pattern": r'[\u0400-\u04FF\u0500-\u052F]+',
        "sentence_delimiters": r'[.;:!?]',
        "preserve_diacritics": True,
        "stopwords": ["и", "въ", "же", "не", "на", "отъ", "съ"],
        "punctuation": ".,;:!?()[]{}\"'-"
    },
    "en": {
        "name": "English",
        "script": "latin",
        "word_pattern": r'[a-zA-Z]+',
        "sentence_delimiters": r'[.!?]',
        "preserve_diacritics": False,
        "stopwords": [
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is", "was",
            "are", "were", "been", "be", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may",
            "might", "must", "shall", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they",
            "what", "which", "who", "whom", "whose", "where", "when",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also"
        ],
        "punctuation": ".,;:!?()[]{}\"'-"
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Token:
    """Preprocessed token"""
    id: int
    form: str
    lemma: str = ""
    pos: str = ""
    morphology: Dict[str, str] = field(default_factory=dict)
    is_stopword: bool = False
    is_punctuation: bool = False
    normalized_form: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Sentence:
    """Preprocessed sentence"""
    id: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    language: str = ""
    
    @property
    def token_count(self) -> int:
        return len(self.tokens)
    
    @property
    def word_count(self) -> int:
        return len([t for t in self.tokens if not t.is_punctuation])
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'tokens': [t.to_dict() for t in self.tokens],
            'language': self.language
        }


@dataclass
class ProcessedDocument:
    """Fully preprocessed document"""
    id: str
    title: str
    author: str
    language: str
    sentences: List[Sentence] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    processing_time: float = 0.0
    
    @property
    def sentence_count(self) -> int:
        return len(self.sentences)
    
    @property
    def token_count(self) -> int:
        return sum(s.token_count for s in self.sentences)
    
    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sentences)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'author': self.author,
            'language': self.language,
            'sentences': [s.to_dict() for s in self.sentences],
            'statistics': self.statistics,
            'processing_time': self.processing_time
        }


# ============================================================================
# TOKENIZERS
# ============================================================================

class Tokenizer:
    """Base tokenizer for historical languages"""
    
    def __init__(self, language: str):
        self.language = language
        self.config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['en'])
        self.word_pattern = re.compile(self.config['word_pattern'])
        self.sentence_pattern = re.compile(
            f"(?<=[{self.config['sentence_delimiters'][1:-1]}])\\s+"
        )
        self.stopwords = set(self.config.get('stopwords', []))
        self.punctuation = set(self.config.get('punctuation', ''))
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence delimiters
        sentences = self.sentence_pattern.split(text)
        
        # Filter and clean
        result = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 1:
                result.append(sent)
        
        return result
    
    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words"""
        # Find all word tokens
        words = self.word_pattern.findall(text)
        
        # Also extract punctuation as separate tokens
        all_tokens = []
        remaining = text
        
        for word in words:
            idx = remaining.find(word)
            if idx > 0:
                # Check for punctuation before word
                prefix = remaining[:idx]
                for char in prefix:
                    if char in self.punctuation:
                        all_tokens.append(char)
            
            all_tokens.append(word)
            remaining = remaining[idx + len(word):]
        
        # Check for trailing punctuation
        for char in remaining:
            if char in self.punctuation:
                all_tokens.append(char)
        
        return all_tokens
    
    def is_stopword(self, word: str) -> bool:
        """Check if word is a stopword"""
        return word.lower() in self.stopwords
    
    def is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation"""
        return all(c in self.punctuation for c in token)
    
    def normalize(self, word: str) -> str:
        """Normalize word form"""
        # Unicode NFC normalization
        word = unicodedata.normalize('NFC', word)
        
        # Lowercase
        word = word.lower()
        
        # Remove diacritics if configured
        if not self.config.get('preserve_diacritics', True):
            word = self._remove_diacritics(word)
        
        return word
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks"""
        nfkd = unicodedata.normalize('NFD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))


class GreekTokenizer(Tokenizer):
    """Specialized tokenizer for Ancient Greek"""
    
    # Greek-specific patterns
    ELISION_PATTERN = re.compile(r"([α-ωἀ-ῷ]+)'")
    CRASIS_PATTERN = re.compile(r"κ[ἀἁᾀᾁ]")
    
    def __init__(self):
        super().__init__('grc')
    
    def tokenize_words(self, text: str) -> List[str]:
        """Greek-specific word tokenization"""
        # Handle elision
        text = self.ELISION_PATTERN.sub(r'\1 ', text)
        
        # Standard tokenization
        return super().tokenize_words(text)
    
    def normalize(self, word: str) -> str:
        """Greek-specific normalization"""
        word = super().normalize(word)
        
        # Normalize sigma variants
        word = word.replace('ς', 'σ')
        
        # Normalize lunate sigma
        word = word.replace('ϲ', 'σ')
        
        return word


class LatinTokenizer(Tokenizer):
    """Specialized tokenizer for Latin"""
    
    # Latin-specific patterns
    ENCLITIC_PATTERN = re.compile(r"(\w+)(que|ve|ne|ce)$", re.IGNORECASE)
    
    def __init__(self):
        super().__init__('la')
    
    def tokenize_words(self, text: str) -> List[str]:
        """Latin-specific word tokenization"""
        tokens = super().tokenize_words(text)
        
        # Handle enclitics (optional)
        # expanded = []
        # for token in tokens:
        #     match = self.ENCLITIC_PATTERN.match(token)
        #     if match:
        #         expanded.extend([match.group(1), match.group(2)])
        #     else:
        #         expanded.append(token)
        # return expanded
        
        return tokens
    
    def normalize(self, word: str) -> str:
        """Latin-specific normalization"""
        word = super().normalize(word)
        
        # Optionally normalize u/v and i/j
        # word = word.replace('v', 'u').replace('j', 'i')
        
        return word


# ============================================================================
# LEMMATIZERS
# ============================================================================

class Lemmatizer(ABC):
    """Abstract base class for lemmatizers"""
    
    @abstractmethod
    def lemmatize(self, word: str, pos: str = None) -> str:
        """Get lemma for word"""
        pass
    
    @abstractmethod
    def lemmatize_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Lemmatize list of tokens"""
        pass


class RuleBasedLemmatizer(Lemmatizer):
    """Rule-based lemmatizer for historical languages"""
    
    def __init__(self, language: str):
        self.language = language
        self.rules = self._load_rules()
        self.lexicon = self._load_lexicon()
    
    def _load_rules(self) -> Dict:
        """Load language-specific lemmatization rules"""
        rules = {
            'grc': {
                'noun_endings': {
                    'ος': '', 'ον': '', 'ου': '', 'ῳ': '', 'ε': '',
                    'οι': '', 'ων': '', 'οις': '', 'ους': '',
                    'η': '', 'ης': '', 'ῃ': '', 'ην': '',
                    'αι': '', 'ῶν': '', 'αις': '', 'ας': '',
                    'α': '', 'ας': '', 'ᾳ': '', 'αν': '',
                    'ις': '', 'ιος': '', 'ι': '', 'ιν': '',
                    'εις': '', 'εων': '', 'εσι': '',
                    'υς': '', 'υος': '', 'υι': '', 'υν': '',
                    'ευς': '', 'εως': '', 'ει': '', 'εα': ''
                },
                'verb_endings': {
                    'ω': '', 'εις': '', 'ει': '', 'ομεν': '', 'ετε': '', 'ουσι': '',
                    'ον': '', 'ες': '', 'ε': '', 'ομεν': '', 'ετε': '', 'ον': '',
                    'α': '', 'ας': '', 'ε': '', 'αμεν': '', 'ατε': '', 'αν': '',
                    'σα': '', 'σας': '', 'σε': '', 'σαμεν': '', 'σατε': '', 'σαν': '',
                    'κα': '', 'κας': '', 'κε': '', 'καμεν': '', 'κατε': '', 'κασι': '',
                    'μαι': '', 'σαι': '', 'ται': '', 'μεθα': '', 'σθε': '', 'νται': '',
                    'μην': '', 'σο': '', 'το': '', 'μεθα': '', 'σθε': '', 'ντο': '',
                    'ειν': '', 'σαι': '', 'κεναι': '', 'σθαι': '',
                    'ων': '', 'ουσα': '', 'ον': '', 'ας': '', 'ασα': '', 'αν': '',
                    'σων': '', 'σουσα': '', 'σον': '', 'κως': '', 'κυια': '', 'κος': ''
                }
            },
            'la': {
                'noun_endings': {
                    'us': '', 'i': '', 'o': '', 'um': '', 'e': '',
                    'a': '', 'ae': '', 'am': '', 'arum': '', 'is': '', 'as': '',
                    'is': '', 'i': '', 'em': '', 'e': '', 'ium': '', 'ibus': '', 'es': '',
                    'us': '', 'ui': '', 'u': '', 'uum': '', 'ibus': '', 'ua': '',
                    'es': '', 'ei': '', 'em': '', 'e': '', 'erum': '', 'ebus': ''
                },
                'verb_endings': {
                    'o': '', 's': '', 't': '', 'mus': '', 'tis': '', 'nt': '',
                    'bam': '', 'bas': '', 'bat': '', 'bamus': '', 'batis': '', 'bant': '',
                    'bo': '', 'bis': '', 'bit': '', 'bimus': '', 'bitis': '', 'bunt': '',
                    'i': '', 'isti': '', 'it': '', 'imus': '', 'istis': '', 'erunt': '',
                    'eram': '', 'eras': '', 'erat': '', 'eramus': '', 'eratis': '', 'erant': '',
                    'ero': '', 'eris': '', 'erit': '', 'erimus': '', 'eritis': '', 'erint': '',
                    're': '', 'sse': '', 'isse': '',
                    'ns': '', 'ntis': '', 'nti': '', 'ntem': '', 'nte': '',
                    'tus': '', 'ta': '', 'tum': '', 'ti': '', 'tae': ''
                }
            }
        }
        return rules.get(self.language, {})
    
    def _load_lexicon(self) -> Dict[str, str]:
        """Load lemma lexicon"""
        # In production, load from file
        lexicon = {}
        
        # Greek common forms
        if self.language == 'grc':
            lexicon.update({
                'ἐστί': 'εἰμί', 'ἐστιν': 'εἰμί', 'ἦν': 'εἰμί', 'εἶναι': 'εἰμί',
                'ἔχει': 'ἔχω', 'ἔχειν': 'ἔχω', 'εἶχε': 'ἔχω',
                'λέγει': 'λέγω', 'λέγειν': 'λέγω', 'εἶπε': 'λέγω',
                'ποιεῖ': 'ποιέω', 'ποιεῖν': 'ποιέω', 'ἐποίησε': 'ποιέω'
            })
        
        # Latin common forms
        elif self.language == 'la':
            lexicon.update({
                'est': 'sum', 'sunt': 'sum', 'erat': 'sum', 'esse': 'sum',
                'habet': 'habeo', 'habere': 'habeo', 'habuit': 'habeo',
                'dicit': 'dico', 'dicere': 'dico', 'dixit': 'dico',
                'facit': 'facio', 'facere': 'facio', 'fecit': 'facio'
            })
        
        return lexicon
    
    def lemmatize(self, word: str, pos: str = None) -> str:
        """Get lemma for word"""
        word_lower = word.lower()
        
        # Check lexicon first
        if word_lower in self.lexicon:
            return self.lexicon[word_lower]
        
        # Apply rules
        if pos in ['NOUN', 'ADJ', 'PROPN'] or pos is None:
            for ending, replacement in sorted(
                self.rules.get('noun_endings', {}).items(),
                key=lambda x: len(x[0]), reverse=True
            ):
                if word_lower.endswith(ending):
                    return word_lower[:-len(ending)] + replacement
        
        if pos in ['VERB'] or pos is None:
            for ending, replacement in sorted(
                self.rules.get('verb_endings', {}).items(),
                key=lambda x: len(x[0]), reverse=True
            ):
                if word_lower.endswith(ending):
                    return word_lower[:-len(ending)] + replacement
        
        return word_lower
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Lemmatize list of tokens"""
        return [(token, self.lemmatize(token)) for token in tokens]


class CLTKLemmatizer(Lemmatizer):
    """CLTK-based lemmatizer"""
    
    def __init__(self, language: str):
        self.language = language
        self.lemmatizer = None
        self._init_cltk()
    
    def _init_cltk(self):
        """Initialize CLTK lemmatizer"""
        try:
            from cltk.lemmatize.processes import LemmatizationProcess
            from cltk import NLP
            
            lang_map = {
                'grc': 'grc',
                'la': 'lat',
                'sa': 'san',
                'got': 'got',
                'cu': 'chu'
            }
            
            cltk_lang = lang_map.get(self.language)
            if cltk_lang:
                self.nlp = NLP(language=cltk_lang)
                logger.info(f"CLTK initialized for {self.language}")
            else:
                logger.warning(f"CLTK not available for {self.language}")
                
        except ImportError:
            logger.warning("CLTK not installed")
        except Exception as e:
            logger.warning(f"CLTK initialization failed: {e}")
    
    def lemmatize(self, word: str, pos: str = None) -> str:
        """Get lemma using CLTK"""
        if self.nlp:
            try:
                doc = self.nlp.analyze(text=word)
                if doc.words:
                    return doc.words[0].lemma or word
            except Exception as e:
                logger.debug(f"CLTK lemmatization failed for '{word}': {e}")
        return word
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Lemmatize list of tokens using CLTK"""
        if self.nlp:
            try:
                text = ' '.join(tokens)
                doc = self.nlp.analyze(text=text)
                return [(w.string, w.lemma or w.string) for w in doc.words]
            except Exception as e:
                logger.debug(f"CLTK batch lemmatization failed: {e}")
        
        return [(t, t) for t in tokens]


class StanzaLemmatizer(Lemmatizer):
    """Stanza-based lemmatizer"""
    
    def __init__(self, language: str):
        self.language = language
        self.nlp = None
        self._init_stanza()
    
    def _init_stanza(self):
        """Initialize Stanza pipeline"""
        try:
            import stanza
            
            lang_map = {
                'grc': 'grc',
                'la': 'la',
                'en': 'en'
            }
            
            stanza_lang = lang_map.get(self.language)
            if stanza_lang:
                # Download model if needed
                try:
                    stanza.download(stanza_lang, verbose=False)
                except:
                    pass
                
                self.nlp = stanza.Pipeline(
                    stanza_lang,
                    processors='tokenize,mwt,pos,lemma',
                    verbose=False
                )
                logger.info(f"Stanza initialized for {self.language}")
            else:
                logger.warning(f"Stanza not available for {self.language}")
                
        except ImportError:
            logger.warning("Stanza not installed")
        except Exception as e:
            logger.warning(f"Stanza initialization failed: {e}")
    
    def lemmatize(self, word: str, pos: str = None) -> str:
        """Get lemma using Stanza"""
        if self.nlp:
            try:
                doc = self.nlp(word)
                if doc.sentences and doc.sentences[0].words:
                    return doc.sentences[0].words[0].lemma or word
            except Exception as e:
                logger.debug(f"Stanza lemmatization failed for '{word}': {e}")
        return word
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Lemmatize list of tokens using Stanza"""
        if self.nlp:
            try:
                text = ' '.join(tokens)
                doc = self.nlp(text)
                result = []
                for sent in doc.sentences:
                    for word in sent.words:
                        result.append((word.text, word.lemma or word.text))
                return result
            except Exception as e:
                logger.debug(f"Stanza batch lemmatization failed: {e}")
        
        return [(t, t) for t in tokens]


# ============================================================================
# MORPHOLOGICAL ANALYZER
# ============================================================================

class MorphologicalAnalyzer:
    """Morphological analysis for historical languages"""
    
    def __init__(self, language: str):
        self.language = language
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """Load morphological analysis rules"""
        rules = {
            'grc': {
                'noun': {
                    'ος': {'Case': 'Nom', 'Number': 'Sing', 'Gender': 'Masc'},
                    'ου': {'Case': 'Gen', 'Number': 'Sing'},
                    'ῳ': {'Case': 'Dat', 'Number': 'Sing'},
                    'ον': {'Case': 'Acc', 'Number': 'Sing'},
                    'ε': {'Case': 'Voc', 'Number': 'Sing'},
                    'οι': {'Case': 'Nom', 'Number': 'Plur', 'Gender': 'Masc'},
                    'ων': {'Case': 'Gen', 'Number': 'Plur'},
                    'οις': {'Case': 'Dat', 'Number': 'Plur'},
                    'ους': {'Case': 'Acc', 'Number': 'Plur'},
                    'η': {'Case': 'Nom', 'Number': 'Sing', 'Gender': 'Fem'},
                    'ης': {'Case': 'Gen', 'Number': 'Sing', 'Gender': 'Fem'},
                    'ῃ': {'Case': 'Dat', 'Number': 'Sing', 'Gender': 'Fem'},
                    'ην': {'Case': 'Acc', 'Number': 'Sing', 'Gender': 'Fem'},
                    'αι': {'Case': 'Nom', 'Number': 'Plur', 'Gender': 'Fem'},
                    'ῶν': {'Case': 'Gen', 'Number': 'Plur'},
                    'αις': {'Case': 'Dat', 'Number': 'Plur', 'Gender': 'Fem'},
                    'ας': {'Case': 'Acc', 'Number': 'Plur', 'Gender': 'Fem'}
                },
                'verb': {
                    'ω': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '1', 'Number': 'Sing'},
                    'εις': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '2', 'Number': 'Sing'},
                    'ει': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '3', 'Number': 'Sing'},
                    'ομεν': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '1', 'Number': 'Plur'},
                    'ετε': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '2', 'Number': 'Plur'},
                    'ουσι': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '3', 'Number': 'Plur'},
                    'ειν': {'Mood': 'Inf', 'Tense': 'Pres', 'Voice': 'Act'},
                    'ων': {'Mood': 'Part', 'Tense': 'Pres', 'Voice': 'Act', 'Case': 'Nom', 'Gender': 'Masc'},
                    'ουσα': {'Mood': 'Part', 'Tense': 'Pres', 'Voice': 'Act', 'Case': 'Nom', 'Gender': 'Fem'},
                    'ον': {'Mood': 'Part', 'Tense': 'Pres', 'Voice': 'Act', 'Case': 'Nom', 'Gender': 'Neut'}
                }
            },
            'la': {
                'noun': {
                    'us': {'Case': 'Nom', 'Number': 'Sing', 'Gender': 'Masc'},
                    'i': {'Case': 'Gen', 'Number': 'Sing'},
                    'o': {'Case': 'Dat', 'Number': 'Sing'},
                    'um': {'Case': 'Acc', 'Number': 'Sing'},
                    'e': {'Case': 'Voc', 'Number': 'Sing'},
                    'a': {'Case': 'Nom', 'Number': 'Sing', 'Gender': 'Fem'},
                    'ae': {'Case': 'Gen', 'Number': 'Sing', 'Gender': 'Fem'},
                    'am': {'Case': 'Acc', 'Number': 'Sing', 'Gender': 'Fem'},
                    'arum': {'Case': 'Gen', 'Number': 'Plur', 'Gender': 'Fem'},
                    'is': {'Case': 'Dat', 'Number': 'Plur'},
                    'os': {'Case': 'Acc', 'Number': 'Plur', 'Gender': 'Masc'},
                    'as': {'Case': 'Acc', 'Number': 'Plur', 'Gender': 'Fem'}
                },
                'verb': {
                    'o': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '1', 'Number': 'Sing'},
                    's': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '2', 'Number': 'Sing'},
                    't': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '3', 'Number': 'Sing'},
                    'mus': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '1', 'Number': 'Plur'},
                    'tis': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '2', 'Number': 'Plur'},
                    'nt': {'Mood': 'Ind', 'Tense': 'Pres', 'Voice': 'Act', 'Person': '3', 'Number': 'Plur'},
                    're': {'Mood': 'Inf', 'Tense': 'Pres', 'Voice': 'Act'},
                    'ns': {'Mood': 'Part', 'Tense': 'Pres', 'Voice': 'Act'}
                }
            }
        }
        return rules.get(self.language, {})
    
    def analyze(self, word: str, pos: str = None) -> Dict[str, str]:
        """Analyze morphology of word"""
        word_lower = word.lower()
        features = {}
        
        # Try noun endings
        for ending, feats in sorted(
            self.rules.get('noun', {}).items(),
            key=lambda x: len(x[0]), reverse=True
        ):
            if word_lower.endswith(ending):
                features.update(feats)
                break
        
        # If no noun match, try verb endings
        if not features:
            for ending, feats in sorted(
                self.rules.get('verb', {}).items(),
                key=lambda x: len(x[0]), reverse=True
            ):
                if word_lower.endswith(ending):
                    features.update(feats)
                    break
        
        return features


# ============================================================================
# TEXT STATISTICS
# ============================================================================

class TextStatistics:
    """Calculate text statistics"""
    
    def __init__(self):
        pass
    
    def calculate(self, tokens: List[Token]) -> Dict:
        """Calculate comprehensive statistics"""
        if not tokens:
            return {}
        
        # Filter out punctuation
        words = [t for t in tokens if not t.is_punctuation]
        
        # Basic counts
        stats = {
            'token_count': len(tokens),
            'word_count': len(words),
            'punctuation_count': len(tokens) - len(words),
            'unique_forms': len(set(t.form for t in words)),
            'unique_lemmas': len(set(t.lemma for t in words if t.lemma))
        }
        
        # Type-token ratio
        if stats['word_count'] > 0:
            stats['type_token_ratio'] = stats['unique_forms'] / stats['word_count']
        else:
            stats['type_token_ratio'] = 0
        
        # Word length statistics
        word_lengths = [len(t.form) for t in words]
        if word_lengths:
            stats['avg_word_length'] = sum(word_lengths) / len(word_lengths)
            stats['max_word_length'] = max(word_lengths)
            stats['min_word_length'] = min(word_lengths)
        
        # Stopword ratio
        stopwords = [t for t in words if t.is_stopword]
        stats['stopword_count'] = len(stopwords)
        stats['stopword_ratio'] = len(stopwords) / len(words) if words else 0
        
        # POS distribution
        pos_counts = Counter(t.pos for t in words if t.pos)
        stats['pos_distribution'] = dict(pos_counts)
        
        # Frequency distribution
        form_counts = Counter(t.form.lower() for t in words)
        stats['hapax_legomena'] = sum(1 for count in form_counts.values() if count == 1)
        stats['hapax_ratio'] = stats['hapax_legomena'] / stats['unique_forms'] if stats['unique_forms'] > 0 else 0
        
        # Most common words
        stats['most_common_words'] = form_counts.most_common(20)
        
        # Most common lemmas
        lemma_counts = Counter(t.lemma for t in words if t.lemma)
        stats['most_common_lemmas'] = lemma_counts.most_common(20)
        
        return stats
    
    def calculate_ngrams(self, tokens: List[Token], n: int = 2) -> Dict[Tuple, int]:
        """Calculate n-gram frequencies"""
        words = [t.form.lower() for t in tokens if not t.is_punctuation]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        
        return dict(Counter(ngrams).most_common(100))


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """Main preprocessing pipeline"""
    
    def __init__(self, language: str, config: Dict = None):
        self.language = language
        self.config = config or CONFIG
        
        # Initialize components
        if language == 'grc':
            self.tokenizer = GreekTokenizer()
        elif language == 'la':
            self.tokenizer = LatinTokenizer()
        else:
            self.tokenizer = Tokenizer(language)
        
        # Try to use advanced lemmatizers, fall back to rule-based
        self.lemmatizer = self._init_lemmatizer()
        self.morph_analyzer = MorphologicalAnalyzer(language)
        self.stats_calculator = TextStatistics()
    
    def _init_lemmatizer(self) -> Lemmatizer:
        """Initialize best available lemmatizer"""
        # Try Stanza first
        try:
            lemmatizer = StanzaLemmatizer(self.language)
            if lemmatizer.nlp:
                return lemmatizer
        except:
            pass
        
        # Try CLTK
        try:
            lemmatizer = CLTKLemmatizer(self.language)
            if hasattr(lemmatizer, 'nlp') and lemmatizer.nlp:
                return lemmatizer
        except:
            pass
        
        # Fall back to rule-based
        return RuleBasedLemmatizer(self.language)
    
    def process_text(self, text: str, text_id: str = "") -> ProcessedDocument:
        """Process raw text"""
        start_time = time.time()
        
        # Tokenize into sentences
        sentence_texts = self.tokenizer.tokenize_sentences(text)
        
        sentences = []
        for sent_idx, sent_text in enumerate(sentence_texts):
            sentence = self._process_sentence(sent_text, f"{text_id}_s{sent_idx}")
            
            # Filter by token count
            if (self.config.get('sentence_min_tokens', 3) <= sentence.token_count <= 
                self.config.get('sentence_max_tokens', 200)):
                sentences.append(sentence)
        
        # Calculate statistics
        all_tokens = [t for s in sentences for t in s.tokens]
        statistics = self.stats_calculator.calculate(all_tokens)
        
        processing_time = time.time() - start_time
        
        return ProcessedDocument(
            id=text_id,
            title="",
            author="",
            language=self.language,
            sentences=sentences,
            statistics=statistics,
            processing_time=processing_time
        )
    
    def _process_sentence(self, text: str, sent_id: str) -> Sentence:
        """Process single sentence"""
        # Tokenize
        word_tokens = self.tokenizer.tokenize_words(text)
        
        # Lemmatize
        lemmatized = self.lemmatizer.lemmatize_tokens(word_tokens)
        
        # Build token objects
        tokens = []
        for idx, (form, lemma) in enumerate(lemmatized):
            is_punct = self.tokenizer.is_punctuation(form)
            is_stop = self.tokenizer.is_stopword(form) if not is_punct else False
            
            # Morphological analysis
            morphology = {}
            if not is_punct:
                morphology = self.morph_analyzer.analyze(form)
            
            token = Token(
                id=idx + 1,
                form=form,
                lemma=lemma,
                pos=self._infer_pos(form, morphology),
                morphology=morphology,
                is_stopword=is_stop,
                is_punctuation=is_punct,
                normalized_form=self.tokenizer.normalize(form)
            )
            tokens.append(token)
        
        return Sentence(
            id=sent_id,
            text=text,
            tokens=tokens,
            language=self.language
        )
    
    def _infer_pos(self, form: str, morphology: Dict) -> str:
        """Infer POS from morphology"""
        if 'Mood' in morphology or 'Tense' in morphology:
            return 'VERB'
        elif 'Case' in morphology:
            if 'Gender' in morphology:
                return 'NOUN'
            return 'NOUN'
        return 'X'


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

class PreprocessingDatabase:
    """Database for preprocessed texts"""
    
    def __init__(self, db_path: str = "corpus_platform.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize preprocessing tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Processed documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_documents (
                id TEXT PRIMARY KEY,
                raw_text_id TEXT,
                title TEXT,
                author TEXT,
                language TEXT,
                sentence_count INTEGER,
                token_count INTEGER,
                word_count INTEGER,
                statistics TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_text_id) REFERENCES raw_texts(id)
            )
        """)
        
        # Processed sentences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_sentences (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                sentence_index INTEGER,
                text TEXT,
                tokens TEXT,
                token_count INTEGER,
                FOREIGN KEY (document_id) REFERENCES processed_documents(id)
            )
        """)
        
        # Lemma index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lemma_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL,
                language TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                document_ids TEXT,
                UNIQUE(lemma, language)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proc_docs_language ON processed_documents(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proc_sents_doc ON processed_sentences(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lemma_index_lemma ON lemma_index(lemma)")
        
        conn.commit()
        conn.close()
    
    def save_document(self, doc: ProcessedDocument, raw_text_id: str = None) -> bool:
        """Save processed document"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Save document
            cursor.execute("""
                INSERT OR REPLACE INTO processed_documents 
                (id, raw_text_id, title, author, language, sentence_count, 
                 token_count, word_count, statistics, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, raw_text_id, doc.title, doc.author, doc.language,
                doc.sentence_count, doc.token_count, doc.word_count,
                json.dumps(doc.statistics), doc.processing_time
            ))
            
            # Save sentences
            for idx, sentence in enumerate(doc.sentences):
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_sentences 
                    (id, document_id, sentence_index, text, tokens, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    sentence.id, doc.id, idx, sentence.text,
                    json.dumps([t.to_dict() for t in sentence.tokens]),
                    sentence.token_count
                ))
            
            # Update lemma index
            lemma_counts = Counter()
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    if token.lemma and not token.is_punctuation:
                        lemma_counts[token.lemma] += 1
            
            for lemma, count in lemma_counts.items():
                cursor.execute("""
                    INSERT INTO lemma_index (lemma, language, frequency, document_ids)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(lemma, language) DO UPDATE SET
                    frequency = frequency + ?,
                    document_ids = document_ids || ',' || ?
                """, (lemma, doc.language, count, doc.id, count, doc.id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Get processed document"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM processed_documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get sentences
        cursor.execute("""
            SELECT * FROM processed_sentences 
            WHERE document_id = ? 
            ORDER BY sentence_index
        """, (doc_id,))
        
        sentences = []
        for sent_row in cursor.fetchall():
            tokens_data = json.loads(sent_row['tokens'])
            tokens = [Token(**t) for t in tokens_data]
            
            sentence = Sentence(
                id=sent_row['id'],
                text=sent_row['text'],
                tokens=tokens,
                language=row['language']
            )
            sentences.append(sentence)
        
        conn.close()
        
        return ProcessedDocument(
            id=row['id'],
            title=row['title'] or "",
            author=row['author'] or "",
            language=row['language'],
            sentences=sentences,
            statistics=json.loads(row['statistics']) if row['statistics'] else {},
            processing_time=row['processing_time'] or 0.0
        )
    
    def search_lemma(self, lemma: str, language: str = None) -> List[Dict]:
        """Search lemma index"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sql = "SELECT * FROM lemma_index WHERE lemma LIKE ?"
        params = [f"%{lemma}%"]
        
        if language:
            sql += " AND language = ?"
            params.append(language)
        
        sql += " ORDER BY frequency DESC LIMIT 100"
        
        cursor.execute(sql, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """Get preprocessing statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM processed_documents")
        stats['total_documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(token_count) FROM processed_documents")
        stats['total_tokens'] = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM lemma_index")
        stats['unique_lemmas'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT language, COUNT(*) as docs, SUM(token_count) as tokens
            FROM processed_documents GROUP BY language
        """)
        stats['by_language'] = {
            row['language']: {'documents': row['docs'], 'tokens': row['tokens'] or 0}
            for row in cursor.fetchall()
        }
        
        conn.close()
        return stats


# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Batch processing for large corpora"""
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.pipelines = {}
        self.db = PreprocessingDatabase(self.config.get('database_path', 'corpus_platform.db'))
    
    def _get_pipeline(self, language: str) -> PreprocessingPipeline:
        """Get or create pipeline for language"""
        if language not in self.pipelines:
            self.pipelines[language] = PreprocessingPipeline(language, self.config)
        return self.pipelines[language]
    
    def process_raw_texts(self, status: str = 'pending', limit: int = 100) -> int:
        """Process raw texts from database"""
        # Get raw texts
        conn = sqlite3.connect(self.config.get('database_path', 'corpus_platform.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, author, language, content 
            FROM raw_texts 
            WHERE processing_status = ?
            LIMIT ?
        """, (status, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        processed = 0
        for row in rows:
            try:
                pipeline = self._get_pipeline(row['language'])
                doc = pipeline.process_text(row['content'], row['id'])
                doc.title = row['title'] or ""
                doc.author = row['author'] or ""
                
                if self.db.save_document(doc, row['id']):
                    # Update raw text status
                    self._update_raw_status(row['id'], 'processed')
                    processed += 1
                    logger.info(f"Processed: {row['title']} ({doc.token_count} tokens)")
                    
            except Exception as e:
                logger.error(f"Error processing {row['id']}: {e}")
                self._update_raw_status(row['id'], 'error')
        
        return processed
    
    def _update_raw_status(self, text_id: str, status: str):
        """Update raw text processing status"""
        conn = sqlite3.connect(self.config.get('database_path', 'corpus_platform.db'))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE raw_texts 
            SET processing_status = ?, date_processed = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, text_id))
        
        conn.commit()
        conn.close()
    
    def process_file(self, filepath: str, language: str) -> Optional[ProcessedDocument]:
        """Process single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            pipeline = self._get_pipeline(language)
            doc = pipeline.process_text(text, Path(filepath).stem)
            
            return doc
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            return None
    
    def process_directory(self, dirpath: str, language: str, 
                         pattern: str = "*.txt") -> int:
        """Process all files in directory"""
        processed = 0
        
        for filepath in Path(dirpath).glob(pattern):
            doc = self.process_file(str(filepath), language)
            if doc:
                self.db.save_document(doc)
                processed += 1
                logger.info(f"Processed: {filepath.name}")
        
        return processed


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument('command', choices=['process', 'batch', 'stats', 'search'],
                       help="Command to run")
    parser.add_argument('--input', '-i', help="Input file or directory")
    parser.add_argument('--language', '-l', default='grc', help="Language code")
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--limit', type=int, default=100, help="Batch limit")
    parser.add_argument('--query', '-q', help="Search query")
    
    args = parser.parse_args()
    
    if args.command == 'process':
        if not args.input:
            print("Error: --input required")
            return
        
        pipeline = PreprocessingPipeline(args.language)
        
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc = pipeline.process_text(text, Path(args.input).stem)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
            else:
                print(json.dumps(doc.statistics, indent=2))
        
        else:
            processor = BatchProcessor()
            count = processor.process_directory(args.input, args.language)
            print(f"Processed {count} files")
    
    elif args.command == 'batch':
        processor = BatchProcessor()
        count = processor.process_raw_texts(limit=args.limit)
        print(f"Processed {count} texts")
    
    elif args.command == 'stats':
        db = PreprocessingDatabase()
        stats = db.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required")
            return
        
        db = PreprocessingDatabase()
        results = db.search_lemma(args.query, args.language)
        
        for result in results[:20]:
            print(f"{result['lemma']}: {result['frequency']} occurrences")


if __name__ == "__main__":
    main()
