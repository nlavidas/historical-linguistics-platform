"""
Advanced Greek NLP Pipeline
Comprehensive linguistic processing for Ancient, Byzantine, Medieval, and Modern Greek

Features:
- Unicode normalization (NFC, polytonic handling)
- Greek-specific tokenization
- Sentence boundary detection
- Morphological analysis heuristics
- POS tagging rules
- Lemmatization patterns
- Dependency parsing preparation
"""

import os
import re
import json
import sqlite3
import logging
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GREEK CHARACTER SETS
# =============================================================================

# Basic Greek alphabet
GREEK_LOWER = 'αβγδεζηθικλμνξοπρσςτυφχψω'
GREEK_UPPER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'

# Extended Greek with diacritics (polytonic)
GREEK_EXTENDED = (
    'ἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎἏ'  # Alpha with breathing/accents
    'ἐἑἒἓἔἕἘἙἚἛἜἝ'      # Epsilon
    'ἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮἯ'  # Eta
    'ἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾἿ'  # Iota
    'ὀὁὂὃὄὅὈὉὊὋὌὍ'      # Omicron
    'ὐὑὒὓὔὕὖὗὙὛὝὟ'      # Upsilon
    'ὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯ'  # Omega
    'ὰάὲέὴήὶίὸόὺύὼώ'    # Grave/acute
    'ᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏ'  # Alpha with iota subscript
    'ᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟ'  # Eta with iota subscript
    'ᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯ'  # Omega with iota subscript
    'ᾰᾱᾲᾳᾴᾶᾷᾸᾹᾺΆᾼ'      # Alpha variants
    'ῂῃῄῆῇῈΈῊΉῌ'        # Eta variants
    'ῐῑῒΐῖῗῘῙῚΊ'        # Iota variants
    'ῠῡῢΰῤῥῦῧῨῩῪΎῬ'    # Upsilon/Rho variants
    'ῲῳῴῶῷῸΌῺΏῼ'        # Omega variants
)

# Punctuation
GREEK_PUNCTUATION = '·;,.!?:\'\"()[]{}«»—–-'
SENTENCE_ENDINGS = '.;·!?'

# All Greek characters
ALL_GREEK = set(GREEK_LOWER + GREEK_UPPER + GREEK_EXTENDED)

# =============================================================================
# GREEK MORPHOLOGY PATTERNS
# =============================================================================

# Verb endings by tense/mood
VERB_ENDINGS = {
    # Present Active Indicative
    'present_active': {
        '1sg': ['ω', 'μι'],
        '2sg': ['εις', 'ς'],
        '3sg': ['ει', 'σι'],
        '1pl': ['ομεν', 'μεν'],
        '2pl': ['ετε', 'τε'],
        '3pl': ['ουσι', 'ουσιν', 'ασι', 'ασιν']
    },
    # Imperfect Active
    'imperfect_active': {
        '1sg': ['ον'],
        '2sg': ['ες'],
        '3sg': ['ε', 'εν'],
        '1pl': ['ομεν'],
        '2pl': ['ετε'],
        '3pl': ['ον']
    },
    # Aorist Active
    'aorist_active': {
        '1sg': ['α', 'σα'],
        '2sg': ['ας', 'σας', 'ες'],
        '3sg': ['ε', 'εν', 'σε', 'σεν'],
        '1pl': ['αμεν', 'σαμεν'],
        '2pl': ['ατε', 'σατε'],
        '3pl': ['αν', 'σαν']
    },
    # Perfect Active
    'perfect_active': {
        '1sg': ['κα'],
        '2sg': ['κας'],
        '3sg': ['κε', 'κεν'],
        '1pl': ['καμεν'],
        '2pl': ['κατε'],
        '3pl': ['κασι', 'κασιν']
    },
    # Present Middle/Passive
    'present_mp': {
        '1sg': ['ομαι', 'μαι'],
        '2sg': ['ῃ', 'ει', 'σαι'],
        '3sg': ['εται', 'ται'],
        '1pl': ['ομεθα', 'μεθα'],
        '2pl': ['εσθε', 'σθε'],
        '3pl': ['ονται', 'νται']
    },
    # Infinitives
    'infinitive': {
        'present_active': ['ειν', 'ναι', 'σθαι'],
        'aorist_active': ['αι', 'σαι'],
        'perfect_active': ['κέναι', 'εναι'],
        'present_mp': ['εσθαι', 'σθαι'],
        'aorist_mp': ['ῆναι', 'θῆναι']
    },
    # Participles
    'participle': {
        'present_active': ['ων', 'ουσα', 'ον', 'ντ'],
        'aorist_active': ['ας', 'ασα', 'αν', 'σας', 'σασα', 'σαν'],
        'perfect_active': ['κως', 'κυια', 'κος', 'ώς', 'υῖα', 'ός'],
        'present_mp': ['ομενος', 'ομενη', 'ομενον', 'μενος', 'μενη', 'μενον'],
        'aorist_passive': ['θεις', 'θεισα', 'θεν', 'είς', 'εῖσα', 'έν']
    }
}

# Noun endings by declension
NOUN_ENDINGS = {
    # First declension (feminine)
    'first_fem': {
        'nom_sg': ['α', 'η', 'ᾱ'],
        'gen_sg': ['ας', 'ης', 'ᾱς'],
        'dat_sg': ['ᾳ', 'ῃ'],
        'acc_sg': ['αν', 'ην', 'ᾱν'],
        'voc_sg': ['α', 'η'],
        'nom_pl': ['αι'],
        'gen_pl': ['ων', 'ῶν'],
        'dat_pl': ['αις'],
        'acc_pl': ['ας', 'ᾱς']
    },
    # First declension (masculine)
    'first_masc': {
        'nom_sg': ['ας', 'ης'],
        'gen_sg': ['ου'],
        'dat_sg': ['ᾳ', 'ῃ'],
        'acc_sg': ['αν', 'ην'],
        'voc_sg': ['α', 'η'],
        'nom_pl': ['αι'],
        'gen_pl': ['ων', 'ῶν'],
        'dat_pl': ['αις'],
        'acc_pl': ['ας']
    },
    # Second declension
    'second': {
        'nom_sg': ['ος', 'ον'],
        'gen_sg': ['ου'],
        'dat_sg': ['ῳ'],
        'acc_sg': ['ον'],
        'voc_sg': ['ε', 'ον'],
        'nom_pl': ['οι', 'α'],
        'gen_pl': ['ων'],
        'dat_pl': ['οις'],
        'acc_pl': ['ους', 'α']
    },
    # Third declension
    'third': {
        'nom_sg': ['ς', '', 'ξ', 'ψ', 'ρ', 'ν'],
        'gen_sg': ['ος', 'ως', 'εως', 'ους'],
        'dat_sg': ['ι', 'ει'],
        'acc_sg': ['α', 'ν', 'ιν'],
        'voc_sg': ['', 'ς'],
        'nom_pl': ['ες', 'εις', 'α'],
        'gen_pl': ['ων', 'εων'],
        'dat_pl': ['σι', 'σιν', 'εσι', 'εσιν'],
        'acc_pl': ['ας', 'εις', 'α']
    }
}

# Article forms
ARTICLE_FORMS = {
    'masc': {
        'nom_sg': 'ὁ', 'gen_sg': 'τοῦ', 'dat_sg': 'τῷ', 'acc_sg': 'τόν',
        'nom_pl': 'οἱ', 'gen_pl': 'τῶν', 'dat_pl': 'τοῖς', 'acc_pl': 'τούς'
    },
    'fem': {
        'nom_sg': 'ἡ', 'gen_sg': 'τῆς', 'dat_sg': 'τῇ', 'acc_sg': 'τήν',
        'nom_pl': 'αἱ', 'gen_pl': 'τῶν', 'dat_pl': 'ταῖς', 'acc_pl': 'τάς'
    },
    'neut': {
        'nom_sg': 'τό', 'gen_sg': 'τοῦ', 'dat_sg': 'τῷ', 'acc_sg': 'τό',
        'nom_pl': 'τά', 'gen_pl': 'τῶν', 'dat_pl': 'τοῖς', 'acc_pl': 'τά'
    }
}

# Prepositions with their typical cases
PREPOSITIONS = {
    # Genitive only
    'ἀντί': ['gen'], 'ἀπό': ['gen'], 'ἐκ': ['gen'], 'ἐξ': ['gen'],
    'πρό': ['gen'], 'ὑπέρ': ['gen', 'acc'],
    
    # Dative only
    'ἐν': ['dat'], 'σύν': ['dat'],
    
    # Accusative only
    'εἰς': ['acc'], 'ἀνά': ['acc'], 'ὡς': ['acc'],
    
    # Multiple cases
    'διά': ['gen', 'acc'], 'κατά': ['gen', 'acc'], 'μετά': ['gen', 'acc'],
    'παρά': ['gen', 'dat', 'acc'], 'περί': ['gen', 'dat', 'acc'],
    'πρός': ['gen', 'dat', 'acc'], 'ὑπό': ['gen', 'dat', 'acc'],
    'ἐπί': ['gen', 'dat', 'acc'], 'ἀμφί': ['gen', 'dat', 'acc']
}

# Common particles
PARTICLES = {
    'ἄν', 'γάρ', 'γε', 'δέ', 'δή', 'καί', 'μέν', 'μή', 'οὐ', 'οὐκ', 'οὐχ',
    'οὖν', 'τε', 'ἀλλά', 'ἀλλ', 'εἰ', 'ἤ', 'ὅτι', 'ὡς', 'ὥστε', 'ἵνα',
    'ὅπως', 'ἐάν', 'ἄρα', 'τοίνυν', 'μέντοι', 'καίτοι', 'ἀτάρ'
}

# Conjunctions
CONJUNCTIONS = {
    'καί', 'τε', 'ἤ', 'ἀλλά', 'ἀλλ', 'δέ', 'γάρ', 'οὖν', 'ὅτι', 'ὡς',
    'ἐπεί', 'ἐπειδή', 'ὅτε', 'ὅταν', 'εἰ', 'ἐάν', 'ἵνα', 'ὅπως', 'ὥστε',
    'πρίν', 'ἕως', 'μέχρι', 'ἄχρι', 'οὔτε', 'μήτε', 'εἴτε'
}

# =============================================================================
# NORMALIZER
# =============================================================================

class GreekNormalizer:
    """Normalize Greek text for consistent processing"""
    
    # Sigma normalization
    FINAL_SIGMA = 'ς'
    MEDIAL_SIGMA = 'σ'
    LUNATE_SIGMA = 'ϲ'
    
    # Common variant mappings
    VARIANT_MAP = {
        'ϲ': 'σ',  # Lunate sigma
        'ϛ': 'στ',  # Stigma
        'ϝ': 'ϝ',  # Digamma (keep)
        'ϟ': 'ϟ',  # Koppa (keep)
        'ϡ': 'ϡ',  # Sampi (keep)
        'ʹ': '',   # Numeral sign
        '͵': '',   # Lower numeral sign
        '\u0374': '',  # Greek numeral sign
        '\u0375': '',  # Greek lower numeral sign
    }
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Full normalization pipeline"""
        if not text:
            return ""
        
        # Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # Apply variant mappings
        for old, new in cls.VARIANT_MAP.items():
            text = text.replace(old, new)
        
        # Normalize sigmas
        text = cls._normalize_sigmas(text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @classmethod
    def _normalize_sigmas(cls, text: str) -> str:
        """Normalize sigma variants"""
        result = []
        chars = list(text)
        
        for i, char in enumerate(chars):
            if char in ('σ', 'ς', 'ϲ'):
                # Check if word-final
                is_final = (i == len(chars) - 1 or 
                           not chars[i + 1].isalpha() or
                           chars[i + 1] not in ALL_GREEK)
                
                result.append(cls.FINAL_SIGMA if is_final else cls.MEDIAL_SIGMA)
            else:
                result.append(char)
        
        return ''.join(result)
    
    @classmethod
    def strip_diacritics(cls, text: str) -> str:
        """Remove all diacritics (for comparison)"""
        # Decompose
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Recompose
        return unicodedata.normalize('NFC', text)
    
    @classmethod
    def to_monotonic(cls, text: str) -> str:
        """Convert polytonic to monotonic Greek"""
        # This is a simplified conversion
        # Full conversion would require more complex mapping
        
        monotonic_map = {
            # Alpha
            'ἀ': 'α', 'ἁ': 'α', 'ἂ': 'ά', 'ἃ': 'ά', 'ἄ': 'ά', 'ἅ': 'ά',
            'ἆ': 'ά', 'ἇ': 'ά', 'ὰ': 'ά', 'ᾀ': 'α', 'ᾁ': 'α', 'ᾂ': 'ά',
            'ᾃ': 'ά', 'ᾄ': 'ά', 'ᾅ': 'ά', 'ᾆ': 'ά', 'ᾇ': 'ά', 'ᾲ': 'ά',
            'ᾳ': 'α', 'ᾴ': 'ά', 'ᾶ': 'ά', 'ᾷ': 'ά',
            # Epsilon
            'ἐ': 'ε', 'ἑ': 'ε', 'ἒ': 'έ', 'ἓ': 'έ', 'ἔ': 'έ', 'ἕ': 'έ',
            'ὲ': 'έ',
            # Eta
            'ἠ': 'η', 'ἡ': 'η', 'ἢ': 'ή', 'ἣ': 'ή', 'ἤ': 'ή', 'ἥ': 'ή',
            'ἦ': 'ή', 'ἧ': 'ή', 'ὴ': 'ή', 'ᾐ': 'η', 'ᾑ': 'η', 'ᾒ': 'ή',
            'ᾓ': 'ή', 'ᾔ': 'ή', 'ᾕ': 'ή', 'ᾖ': 'ή', 'ᾗ': 'ή', 'ῂ': 'ή',
            'ῃ': 'η', 'ῄ': 'ή', 'ῆ': 'ή', 'ῇ': 'ή',
            # Iota
            'ἰ': 'ι', 'ἱ': 'ι', 'ἲ': 'ί', 'ἳ': 'ί', 'ἴ': 'ί', 'ἵ': 'ί',
            'ἶ': 'ί', 'ἷ': 'ί', 'ὶ': 'ί', 'ῐ': 'ι', 'ῑ': 'ι', 'ῒ': 'ί',
            'ΐ': 'ί', 'ῖ': 'ί', 'ῗ': 'ί',
            # Omicron
            'ὀ': 'ο', 'ὁ': 'ο', 'ὂ': 'ό', 'ὃ': 'ό', 'ὄ': 'ό', 'ὅ': 'ό',
            'ὸ': 'ό',
            # Upsilon
            'ὐ': 'υ', 'ὑ': 'υ', 'ὒ': 'ύ', 'ὓ': 'ύ', 'ὔ': 'ύ', 'ὕ': 'ύ',
            'ὖ': 'ύ', 'ὗ': 'ύ', 'ὺ': 'ύ', 'ῠ': 'υ', 'ῡ': 'υ', 'ῢ': 'ύ',
            'ΰ': 'ύ', 'ῦ': 'ύ', 'ῧ': 'ύ',
            # Omega
            'ὠ': 'ω', 'ὡ': 'ω', 'ὢ': 'ώ', 'ὣ': 'ώ', 'ὤ': 'ώ', 'ὥ': 'ώ',
            'ὦ': 'ώ', 'ὧ': 'ώ', 'ὼ': 'ώ', 'ᾠ': 'ω', 'ᾡ': 'ω', 'ᾢ': 'ώ',
            'ᾣ': 'ώ', 'ᾤ': 'ώ', 'ᾥ': 'ώ', 'ᾦ': 'ώ', 'ᾧ': 'ώ', 'ῲ': 'ώ',
            'ῳ': 'ω', 'ῴ': 'ώ', 'ῶ': 'ώ', 'ῷ': 'ώ',
            # Rho
            'ῤ': 'ρ', 'ῥ': 'ρ',
        }
        
        result = []
        for char in text:
            result.append(monotonic_map.get(char, char))
        
        return ''.join(result)


# =============================================================================
# TOKENIZER
# =============================================================================

class GreekTokenizer:
    """Tokenize Greek text"""
    
    # Token patterns
    WORD_PATTERN = re.compile(
        r"[\w\u0370-\u03FF\u1F00-\u1FFF]+"
    )
    
    PUNCT_PATTERN = re.compile(
        r"[·;,.!?:'\"\(\)\[\]\{\}«»—–\-]+"
    )
    
    @classmethod
    def tokenize(cls, text: str) -> List[Dict]:
        """Tokenize text into words and punctuation"""
        if not text:
            return []
        
        tokens = []
        pos = 0
        
        while pos < len(text):
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            
            if pos >= len(text):
                break
            
            # Try word match
            word_match = cls.WORD_PATTERN.match(text, pos)
            if word_match:
                tokens.append({
                    'form': word_match.group(),
                    'start': pos,
                    'end': word_match.end(),
                    'type': 'word'
                })
                pos = word_match.end()
                continue
            
            # Try punctuation match
            punct_match = cls.PUNCT_PATTERN.match(text, pos)
            if punct_match:
                tokens.append({
                    'form': punct_match.group(),
                    'start': pos,
                    'end': punct_match.end(),
                    'type': 'punct'
                })
                pos = punct_match.end()
                continue
            
            # Single character fallback
            tokens.append({
                'form': text[pos],
                'start': pos,
                'end': pos + 1,
                'type': 'other'
            })
            pos += 1
        
        return tokens
    
    @classmethod
    def tokenize_simple(cls, text: str) -> List[str]:
        """Simple tokenization returning just word forms"""
        tokens = cls.tokenize(text)
        return [t['form'] for t in tokens if t['type'] == 'word']


# =============================================================================
# SENTENCE SPLITTER
# =============================================================================

class GreekSentenceSplitter:
    """Split Greek text into sentences"""
    
    # Sentence ending patterns
    SENTENCE_END = re.compile(r'([.;·!?]+)\s+')
    
    # Abbreviations that don't end sentences
    ABBREVIATIONS = {'κτλ', 'π.χ', 'δηλ', 'κ.α', 'κ.λπ', 'κ.τ.λ'}
    
    @classmethod
    def split(cls, text: str) -> List[str]:
        """Split text into sentences"""
        if not text:
            return []
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        sentences = []
        current = ""
        
        parts = cls.SENTENCE_END.split(text)
        
        for i, part in enumerate(parts):
            if cls.SENTENCE_END.match(part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    @classmethod
    def split_with_positions(cls, text: str) -> List[Dict]:
        """Split with position information"""
        sentences = []
        current_pos = 0
        
        for sent_text in cls.split(text):
            start = text.find(sent_text, current_pos)
            if start == -1:
                start = current_pos
            
            sentences.append({
                'text': sent_text,
                'start': start,
                'end': start + len(sent_text)
            })
            
            current_pos = start + len(sent_text)
        
        return sentences


# =============================================================================
# MORPHOLOGICAL ANALYZER
# =============================================================================

class GreekMorphAnalyzer:
    """Heuristic morphological analysis for Greek"""
    
    def __init__(self):
        self.normalizer = GreekNormalizer()
        
        # Build lookup sets
        self._build_lookups()
    
    def _build_lookups(self):
        """Build lookup tables for analysis"""
        # Article forms
        self.articles = set()
        for gender_forms in ARTICLE_FORMS.values():
            self.articles.update(gender_forms.values())
        
        # Prepositions
        self.prepositions = set(PREPOSITIONS.keys())
        
        # Particles and conjunctions
        self.particles = PARTICLES
        self.conjunctions = CONJUNCTIONS
        
        # Build ending patterns
        self.verb_endings = self._flatten_endings(VERB_ENDINGS)
        self.noun_endings = self._flatten_endings(NOUN_ENDINGS)
    
    def _flatten_endings(self, endings_dict: Dict) -> Set[str]:
        """Flatten nested endings dict to set"""
        result = set()
        
        def recurse(d):
            for v in d.values():
                if isinstance(v, dict):
                    recurse(v)
                elif isinstance(v, list):
                    result.update(v)
                elif isinstance(v, str):
                    result.add(v)
        
        recurse(endings_dict)
        return result
    
    def analyze(self, form: str) -> Dict:
        """Analyze a single word form"""
        if not form:
            return {'form': form, 'pos': 'X', 'features': {}}
        
        # Normalize
        normalized = self.normalizer.normalize(form)
        lower = normalized.lower()
        stripped = self.normalizer.strip_diacritics(lower)
        
        result = {
            'form': form,
            'normalized': normalized,
            'pos': 'X',
            'features': {},
            'possible_lemmas': []
        }
        
        # Check closed classes first
        if lower in self.articles or stripped in {'ο', 'η', 'το', 'οι', 'αι', 'τα', 'του', 'της', 'τω', 'τον', 'την', 'των', 'τοις', 'ταις', 'τους', 'τας'}:
            result['pos'] = 'DET'
            result['features']['PronType'] = 'Art'
            return result
        
        if lower in self.prepositions:
            result['pos'] = 'ADP'
            result['features']['cases'] = PREPOSITIONS.get(lower, [])
            return result
        
        if lower in self.particles:
            result['pos'] = 'PART'
            return result
        
        if lower in self.conjunctions:
            result['pos'] = 'CCONJ' if lower in {'καί', 'τε', 'ἤ', 'οὔτε', 'μήτε'} else 'SCONJ'
            return result
        
        # Check for verb patterns
        verb_analysis = self._analyze_verb(lower, stripped)
        if verb_analysis:
            result['pos'] = 'VERB'
            result['features'].update(verb_analysis)
            return result
        
        # Check for noun patterns
        noun_analysis = self._analyze_noun(lower, stripped)
        if noun_analysis:
            result['pos'] = 'NOUN'
            result['features'].update(noun_analysis)
            return result
        
        # Default: assume noun
        result['pos'] = 'NOUN'
        return result
    
    def _analyze_verb(self, form: str, stripped: str) -> Optional[Dict]:
        """Try to analyze as verb"""
        features = {}
        
        # Check common verb endings
        for ending in sorted(self.verb_endings, key=len, reverse=True):
            if form.endswith(ending) or stripped.endswith(ending):
                features['VerbForm'] = 'Fin'
                
                # Determine tense/mood from ending
                if ending in ['ω', 'εις', 'ει', 'ομεν', 'ετε', 'ουσι', 'ουσιν']:
                    features['Tense'] = 'Pres'
                    features['Mood'] = 'Ind'
                    features['Voice'] = 'Act'
                elif ending in ['ον', 'ες', 'ομεν', 'ετε']:
                    features['Tense'] = 'Imp'
                    features['Mood'] = 'Ind'
                elif ending in ['α', 'ας', 'ε', 'αμεν', 'ατε', 'αν', 'σα', 'σας', 'σε', 'σαμεν', 'σατε', 'σαν']:
                    features['Tense'] = 'Aor'
                    features['Mood'] = 'Ind'
                elif ending in ['κα', 'κας', 'κε', 'καμεν', 'κατε', 'κασι']:
                    features['Tense'] = 'Perf'
                    features['Mood'] = 'Ind'
                elif ending in ['ομαι', 'ῃ', 'εται', 'ομεθα', 'εσθε', 'ονται']:
                    features['Voice'] = 'Mid' if 'Pass' not in features else 'Pass'
                elif ending in ['ειν', 'ναι', 'σθαι', 'αι', 'σαι']:
                    features['VerbForm'] = 'Inf'
                elif ending in ['ων', 'ουσα', 'ον', 'ας', 'ασα', 'αν', 'μενος', 'μενη', 'μενον']:
                    features['VerbForm'] = 'Part'
                
                return features
        
        return None
    
    def _analyze_noun(self, form: str, stripped: str) -> Optional[Dict]:
        """Try to analyze as noun"""
        features = {}
        
        # Check noun endings
        for ending in sorted(self.noun_endings, key=len, reverse=True):
            if form.endswith(ending) or stripped.endswith(ending):
                # Determine case/number from ending
                if ending in ['ος', 'η', 'α', 'ον', 'ης', 'ας']:
                    features['Case'] = 'Nom'
                    features['Number'] = 'Sing'
                elif ending in ['ου', 'ης', 'ας', 'ος', 'ως', 'εως']:
                    features['Case'] = 'Gen'
                    features['Number'] = 'Sing'
                elif ending in ['ῳ', 'ῃ', 'ᾳ', 'ι', 'ει']:
                    features['Case'] = 'Dat'
                    features['Number'] = 'Sing'
                elif ending in ['ον', 'ην', 'αν', 'α', 'ν', 'ιν']:
                    features['Case'] = 'Acc'
                    features['Number'] = 'Sing'
                elif ending in ['οι', 'αι', 'ες', 'εις', 'α']:
                    features['Case'] = 'Nom'
                    features['Number'] = 'Plur'
                elif ending in ['ων', 'εων']:
                    features['Case'] = 'Gen'
                    features['Number'] = 'Plur'
                elif ending in ['οις', 'αις', 'σι', 'σιν', 'εσι', 'εσιν']:
                    features['Case'] = 'Dat'
                    features['Number'] = 'Plur'
                elif ending in ['ους', 'ας', 'εις', 'α']:
                    features['Case'] = 'Acc'
                    features['Number'] = 'Plur'
                
                return features
        
        return None
    
    def analyze_sentence(self, tokens: List[str]) -> List[Dict]:
        """Analyze all tokens in a sentence"""
        return [self.analyze(token) for token in tokens]


# =============================================================================
# LEMMATIZER
# =============================================================================

class GreekLemmatizer:
    """Basic lemmatization for Greek"""
    
    # Common irregular forms
    IRREGULAR_LEMMAS = {
        # εἰμί (to be)
        'εἰμί': 'εἰμί', 'εἶ': 'εἰμί', 'ἐστί': 'εἰμί', 'ἐστίν': 'εἰμί',
        'ἐσμέν': 'εἰμί', 'ἐστέ': 'εἰμί', 'εἰσί': 'εἰμί', 'εἰσίν': 'εἰμί',
        'ἦν': 'εἰμί', 'ἦσθα': 'εἰμί', 'ἦμεν': 'εἰμί', 'ἦτε': 'εἰμί', 'ἦσαν': 'εἰμί',
        'ἔσομαι': 'εἰμί', 'ἔσῃ': 'εἰμί', 'ἔσται': 'εἰμί',
        'ὤν': 'εἰμί', 'οὖσα': 'εἰμί', 'ὄν': 'εἰμί',
        
        # εἶμι (to go)
        'εἶμι': 'εἶμι', 'εἶ': 'εἶμι', 'εἶσι': 'εἶμι',
        'ἴμεν': 'εἶμι', 'ἴτε': 'εἶμι', 'ἴασι': 'εἶμι',
        'ᾖα': 'εἶμι', 'ᾔεις': 'εἶμι', 'ᾔει': 'εἶμι',
        'ἰών': 'εἶμι', 'ἰοῦσα': 'εἶμι', 'ἰόν': 'εἶμι',
        
        # οἶδα (to know)
        'οἶδα': 'οἶδα', 'οἶσθα': 'οἶδα', 'οἶδε': 'οἶδα', 'οἶδεν': 'οἶδα',
        'ἴσμεν': 'οἶδα', 'ἴστε': 'οἶδα', 'ἴσασι': 'οἶδα', 'ἴσασιν': 'οἶδα',
        'ᾔδη': 'οἶδα', 'ᾔδεις': 'οἶδα', 'ᾔδει': 'οἶδα',
        'εἰδώς': 'οἶδα', 'εἰδυῖα': 'οἶδα', 'εἰδός': 'οἶδα',
        
        # φημί (to say)
        'φημί': 'φημί', 'φῄς': 'φημί', 'φησί': 'φημί', 'φησίν': 'φημί',
        'φαμέν': 'φημί', 'φατέ': 'φημί', 'φασί': 'φημί', 'φασίν': 'φημί',
        'ἔφην': 'φημί', 'ἔφησθα': 'φημί', 'ἔφη': 'φημί',
        
        # Common articles
        'ὁ': 'ὁ', 'ἡ': 'ὁ', 'τό': 'ὁ', 'τοῦ': 'ὁ', 'τῆς': 'ὁ',
        'τῷ': 'ὁ', 'τῇ': 'ὁ', 'τόν': 'ὁ', 'τήν': 'ὁ',
        'οἱ': 'ὁ', 'αἱ': 'ὁ', 'τά': 'ὁ', 'τῶν': 'ὁ',
        'τοῖς': 'ὁ', 'ταῖς': 'ὁ', 'τούς': 'ὁ', 'τάς': 'ὁ',
        
        # Pronouns
        'ἐγώ': 'ἐγώ', 'ἐμοῦ': 'ἐγώ', 'μου': 'ἐγώ', 'ἐμοί': 'ἐγώ', 'μοι': 'ἐγώ',
        'ἐμέ': 'ἐγώ', 'με': 'ἐγώ',
        'σύ': 'σύ', 'σοῦ': 'σύ', 'σου': 'σύ', 'σοί': 'σύ', 'σοι': 'σύ',
        'σέ': 'σύ', 'σε': 'σύ',
        'αὐτός': 'αὐτός', 'αὐτή': 'αὐτός', 'αὐτό': 'αὐτός',
        'αὐτοῦ': 'αὐτός', 'αὐτῆς': 'αὐτός', 'αὐτῷ': 'αὐτός', 'αὐτῇ': 'αὐτός',
        'αὐτόν': 'αὐτός', 'αὐτήν': 'αὐτός',
        
        # Relative pronouns
        'ὅς': 'ὅς', 'ἥ': 'ὅς', 'ὅ': 'ὅς', 'οὗ': 'ὅς', 'ἧς': 'ὅς',
        'ᾧ': 'ὅς', 'ᾗ': 'ὅς', 'ὅν': 'ὅς', 'ἥν': 'ὅς',
        'οἵ': 'ὅς', 'αἵ': 'ὅς', 'ἅ': 'ὅς', 'ὧν': 'ὅς',
        'οἷς': 'ὅς', 'αἷς': 'ὅς', 'οὕς': 'ὅς', 'ἅς': 'ὅς',
        
        # Interrogative/indefinite
        'τίς': 'τίς', 'τί': 'τίς', 'τίνος': 'τίς', 'τίνι': 'τίς', 'τίνα': 'τίς',
        'τις': 'τις', 'τι': 'τις', 'τινός': 'τις', 'τινί': 'τις', 'τινά': 'τις',
        
        # Demonstratives
        'οὗτος': 'οὗτος', 'αὕτη': 'οὗτος', 'τοῦτο': 'οὗτος',
        'τούτου': 'οὗτος', 'ταύτης': 'οὗτος', 'τούτῳ': 'οὗτος', 'ταύτῃ': 'οὗτος',
        'τοῦτον': 'οὗτος', 'ταύτην': 'οὗτος',
        'ἐκεῖνος': 'ἐκεῖνος', 'ἐκείνη': 'ἐκεῖνος', 'ἐκεῖνο': 'ἐκεῖνος',
        'ὅδε': 'ὅδε', 'ἥδε': 'ὅδε', 'τόδε': 'ὅδε',
    }
    
    def __init__(self):
        self.normalizer = GreekNormalizer()
    
    def lemmatize(self, form: str) -> str:
        """Get lemma for a word form"""
        if not form:
            return form
        
        # Normalize
        normalized = self.normalizer.normalize(form)
        lower = normalized.lower()
        
        # Check irregular forms
        if lower in self.IRREGULAR_LEMMAS:
            return self.IRREGULAR_LEMMAS[lower]
        
        # Try to strip endings
        lemma = self._strip_endings(lower)
        
        return lemma if lemma else form
    
    def _strip_endings(self, form: str) -> str:
        """Try to strip inflectional endings"""
        # This is a simplified approach
        # Full lemmatization would require dictionary lookup
        
        # Try verb endings
        verb_endings = ['ομαι', 'εται', 'ονται', 'ομεν', 'ετε', 'ουσι', 'ουσιν',
                       'ειν', 'σθαι', 'ων', 'ουσα', 'ον', 'μενος', 'μενη', 'μενον']
        
        for ending in sorted(verb_endings, key=len, reverse=True):
            if form.endswith(ending) and len(form) > len(ending) + 2:
                return form[:-len(ending)] + 'ω'
        
        # Try noun endings
        noun_endings = ['ος', 'ου', 'ῳ', 'ον', 'οι', 'ων', 'οις', 'ους',
                       'η', 'ης', 'ῃ', 'ην', 'αι', 'ας', 'αις',
                       'α', 'ας', 'ᾳ', 'αν']
        
        for ending in sorted(noun_endings, key=len, reverse=True):
            if form.endswith(ending) and len(form) > len(ending) + 2:
                stem = form[:-len(ending)]
                # Return nominative singular form
                if ending in ['ου', 'ῳ', 'ον', 'οι', 'ων', 'οις', 'ους']:
                    return stem + 'ος'
                elif ending in ['ης', 'ῃ', 'ην', 'αι', 'ας', 'αις']:
                    return stem + 'η'
                elif ending in ['ας', 'ᾳ', 'αν']:
                    return stem + 'α'
        
        return form


# =============================================================================
# COMPLETE NLP PIPELINE
# =============================================================================

class GreekNLPPipeline:
    """Complete NLP pipeline for Greek text"""
    
    def __init__(self):
        self.normalizer = GreekNormalizer()
        self.tokenizer = GreekTokenizer()
        self.sentence_splitter = GreekSentenceSplitter()
        self.morph_analyzer = GreekMorphAnalyzer()
        self.lemmatizer = GreekLemmatizer()
    
    def process(self, text: str) -> Dict:
        """Process text through complete pipeline"""
        if not text:
            return {'sentences': [], 'tokens': 0, 'words': 0}
        
        # Normalize
        normalized = self.normalizer.normalize(text)
        
        # Split into sentences
        sentences = self.sentence_splitter.split(normalized)
        
        result = {
            'original': text,
            'normalized': normalized,
            'sentences': [],
            'total_tokens': 0,
            'total_words': 0
        }
        
        for sent_idx, sent_text in enumerate(sentences):
            # Tokenize
            tokens = self.tokenizer.tokenize(sent_text)
            
            # Analyze each token
            analyzed_tokens = []
            for token in tokens:
                if token['type'] == 'word':
                    analysis = self.morph_analyzer.analyze(token['form'])
                    analysis['lemma'] = self.lemmatizer.lemmatize(token['form'])
                    analyzed_tokens.append(analysis)
                    result['total_words'] += 1
                else:
                    analyzed_tokens.append({
                        'form': token['form'],
                        'pos': 'PUNCT' if token['type'] == 'punct' else 'X',
                        'lemma': token['form']
                    })
                
                result['total_tokens'] += 1
            
            result['sentences'].append({
                'index': sent_idx,
                'text': sent_text,
                'tokens': analyzed_tokens
            })
        
        return result
    
    def process_to_conllu(self, text: str, doc_id: str = "doc") -> str:
        """Process text and output CoNLL-U format"""
        processed = self.process(text)
        
        lines = []
        
        for sent in processed['sentences']:
            # Sentence metadata
            lines.append(f"# sent_id = {doc_id}-{sent['index']:04d}")
            lines.append(f"# text = {sent['text']}")
            
            # Tokens
            for tok_idx, token in enumerate(sent['tokens'], 1):
                form = token.get('form', '_')
                lemma = token.get('lemma', '_')
                upos = token.get('pos', 'X')
                xpos = '_'
                
                # Build features string
                feats = token.get('features', {})
                feats_str = '|'.join(f"{k}={v}" for k, v in feats.items()) if feats else '_'
                
                head = token.get('head', 0)
                deprel = token.get('relation', '_')
                
                line = f"{tok_idx}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats_str}\t{head}\t{deprel}\t_\t_"
                lines.append(line)
            
            lines.append("")  # Empty line between sentences
        
        return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test the pipeline
    pipeline = GreekNLPPipeline()
    
    # Test texts
    test_texts = [
        "ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.",
        "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην.",
        "ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον, ὃς μάλα πολλὰ πλάγχθη.",
        "γνῶθι σεαυτόν.",
        "πάντα ῥεῖ καὶ οὐδὲν μένει."
    ]
    
    print("=" * 60)
    print("GREEK NLP PIPELINE TEST")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nInput: {text}")
        result = pipeline.process(text)
        
        print(f"Sentences: {len(result['sentences'])}")
        print(f"Words: {result['total_words']}")
        
        for sent in result['sentences']:
            print(f"\n  Sentence {sent['index']}: {sent['text']}")
            for token in sent['tokens']:
                if token['pos'] != 'PUNCT':
                    print(f"    {token['form']:15} → {token.get('lemma', '_'):15} [{token['pos']}]")
        
        print("-" * 40)
    
    # Output CoNLL-U
    print("\n" + "=" * 60)
    print("CONLL-U OUTPUT")
    print("=" * 60)
    
    conllu = pipeline.process_to_conllu(test_texts[0], "john_1_1")
    print(conllu)
