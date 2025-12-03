"""
Text Preprocessing Pipeline - PROIEL/Syntacticus Style
Tokenization, Normalization, Sentence Splitting, Morphological Analysis
"""

import os
import re
import json
import sqlite3
import logging
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GREEK UNICODE RANGES AND CHARACTERS
# =============================================================================

GREEK_RANGES = {
    'basic': (0x0370, 0x03FF),      # Greek and Coptic
    'extended': (0x1F00, 0x1FFF),   # Greek Extended
    'ancient_numbers': (0x10140, 0x1018F),  # Ancient Greek Numbers
}

# Greek punctuation
GREEK_PUNCTUATION = {
    '·': 'MIDDLE_DOT',      # ano teleia (semicolon function)
    ';': 'QUESTION_MARK',   # erotimatiko (Greek question mark)
    ',': 'COMMA',
    '.': 'PERIOD',
    '·': 'COLON',
    '—': 'DASH',
    '«': 'LEFT_QUOTE',
    '»': 'RIGHT_QUOTE',
    '(': 'LEFT_PAREN',
    ')': 'RIGHT_PAREN',
    '[': 'LEFT_BRACKET',
    ']': 'RIGHT_BRACKET',
}

# Sentence-ending punctuation
SENTENCE_ENDERS = {'.', ';', '·', '!', '?'}

# Diacritics
GREEK_DIACRITICS = {
    'acute': '\u0301',      # ́
    'grave': '\u0300',      # ̀
    'circumflex': '\u0342', # ͂
    'smooth': '\u0313',     # ̓
    'rough': '\u0314',      # ̔
    'iota_sub': '\u0345',   # ͅ
    'diaeresis': '\u0308',  # ̈
    'macron': '\u0304',     # ̄
    'breve': '\u0306',      # ̆
}

# =============================================================================
# NORMALIZATION
# =============================================================================

class GreekNormalizer:
    """Normalize Greek text to standard forms"""
    
    # Unicode normalization mappings
    LUNATE_SIGMA = 'ϲ'
    FINAL_SIGMA = 'ς'
    MEDIAL_SIGMA = 'σ'
    
    # Beta code to Unicode (for legacy texts)
    BETA_TO_UNICODE = {
        'a': 'α', 'b': 'β', 'g': 'γ', 'd': 'δ', 'e': 'ε',
        'z': 'ζ', 'h': 'η', 'q': 'θ', 'i': 'ι', 'k': 'κ',
        'l': 'λ', 'm': 'μ', 'n': 'ν', 'c': 'ξ', 'o': 'ο',
        'p': 'π', 'r': 'ρ', 's': 'σ', 't': 'τ', 'u': 'υ',
        'f': 'φ', 'x': 'χ', 'y': 'ψ', 'w': 'ω',
        'A': 'Α', 'B': 'Β', 'G': 'Γ', 'D': 'Δ', 'E': 'Ε',
        'Z': 'Ζ', 'H': 'Η', 'Q': 'Θ', 'I': 'Ι', 'K': 'Κ',
        'L': 'Λ', 'M': 'Μ', 'N': 'Ν', 'C': 'Ξ', 'O': 'Ο',
        'P': 'Π', 'R': 'Ρ', 'S': 'Σ', 'T': 'Τ', 'U': 'Υ',
        'F': 'Φ', 'X': 'Χ', 'Y': 'Ψ', 'W': 'Ω',
    }
    
    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """Apply Unicode NFC normalization"""
        return unicodedata.normalize('NFC', text)
    
    @classmethod
    def normalize_sigma(cls, text: str) -> str:
        """Normalize sigma variants"""
        # Convert lunate sigma to standard
        text = text.replace(cls.LUNATE_SIGMA, cls.MEDIAL_SIGMA)
        text = text.replace('Ϲ', 'Σ')  # Capital lunate
        
        # Fix final sigma
        result = []
        for i, char in enumerate(text):
            if char == cls.MEDIAL_SIGMA:
                # Check if word-final
                next_char = text[i+1] if i+1 < len(text) else ' '
                if not next_char.isalpha():
                    result.append(cls.FINAL_SIGMA)
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    @classmethod
    def remove_editorial_marks(cls, text: str) -> str:
        """Remove editorial brackets and marks"""
        # Remove square brackets (editorial additions)
        text = re.sub(r'\[([^\]]*)\]', r'\1', text)
        # Remove angle brackets (editorial deletions)
        text = re.sub(r'<([^>]*)>', r'\1', text)
        # Remove curly brackets
        text = re.sub(r'\{([^\}]*)\}', r'\1', text)
        # Remove daggers (cruces)
        text = text.replace('†', '').replace('‡', '')
        # Remove asterisks
        text = text.replace('*', '')
        return text
    
    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @classmethod
    def remove_line_numbers(cls, text: str) -> str:
        """Remove line/verse numbers"""
        # Remove patterns like [123] or (123) at start of lines
        text = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', text, flags=re.MULTILINE)
        # Remove inline numbers
        text = re.sub(r'\s+\d+\s+', ' ', text)
        return text
    
    @classmethod
    def beta_to_unicode(cls, text: str) -> str:
        """Convert Beta Code to Unicode Greek"""
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            if char in cls.BETA_TO_UNICODE:
                result.append(cls.BETA_TO_UNICODE[char])
            else:
                result.append(char)
            i += 1
        return ''.join(result)
    
    @classmethod
    def full_normalize(cls, text: str) -> str:
        """Apply full normalization pipeline"""
        text = cls.normalize_unicode(text)
        text = cls.normalize_sigma(text)
        text = cls.remove_editorial_marks(text)
        text = cls.remove_line_numbers(text)
        text = cls.normalize_whitespace(text)
        return text

# =============================================================================
# TOKENIZATION
# =============================================================================

class GreekTokenizer:
    """Tokenize Greek text"""
    
    # Token types
    TOKEN_TYPES = {
        'WORD': 'word',
        'PUNCT': 'punctuation',
        'NUM': 'number',
        'UNKNOWN': 'unknown'
    }
    
    # Greek word pattern
    GREEK_WORD_PATTERN = re.compile(
        r'[\u0370-\u03FF\u1F00-\u1FFF]+'  # Greek characters with diacritics
    )
    
    # Number patterns (Greek numerals)
    GREEK_NUMERAL_PATTERN = re.compile(
        r"[αβγδεϛζηθικλμνξοπϟρστυφχψωϡ]'|"  # Greek numerals with keraia
        r'\d+'  # Arabic numerals
    )
    
    @classmethod
    def tokenize(cls, text: str) -> List[Dict]:
        """Tokenize text into tokens with metadata"""
        tokens = []
        position = 0
        token_id = 0
        
        # Normalize first
        text = GreekNormalizer.full_normalize(text)
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                position += 1
                continue
            
            # Check for Greek word
            match = cls.GREEK_WORD_PATTERN.match(text[i:])
            if match:
                word = match.group()
                tokens.append({
                    'id': token_id,
                    'form': word,
                    'type': cls.TOKEN_TYPES['WORD'],
                    'start': position,
                    'end': position + len(word)
                })
                token_id += 1
                i += len(word)
                position += len(word)
                continue
            
            # Check for punctuation
            if char in GREEK_PUNCTUATION or char in '.,;:!?()[]{}«»—-':
                tokens.append({
                    'id': token_id,
                    'form': char,
                    'type': cls.TOKEN_TYPES['PUNCT'],
                    'start': position,
                    'end': position + 1
                })
                token_id += 1
                i += 1
                position += 1
                continue
            
            # Check for number
            num_match = cls.GREEK_NUMERAL_PATTERN.match(text[i:])
            if num_match:
                num = num_match.group()
                tokens.append({
                    'id': token_id,
                    'form': num,
                    'type': cls.TOKEN_TYPES['NUM'],
                    'start': position,
                    'end': position + len(num)
                })
                token_id += 1
                i += len(num)
                position += len(num)
                continue
            
            # Unknown character - skip
            i += 1
            position += 1
        
        return tokens
    
    @classmethod
    def tokenize_to_words(cls, text: str) -> List[str]:
        """Simple tokenization returning just word forms"""
        tokens = cls.tokenize(text)
        return [t['form'] for t in tokens if t['type'] == cls.TOKEN_TYPES['WORD']]

# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

class GreekSentenceSplitter:
    """Split Greek text into sentences"""
    
    # Abbreviations that don't end sentences
    ABBREVIATIONS = {
        'κτλ', 'π.χ', 'δηλ', 'βλ', 'σελ', 'αρ', 'κεφ',
        'cf', 'etc', 'e.g', 'i.e', 'vs'
    }
    
    @classmethod
    def split(cls, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        current = []
        
        tokens = GreekTokenizer.tokenize(text)
        
        for i, token in enumerate(tokens):
            current.append(token['form'])
            
            # Check for sentence end
            if token['form'] in SENTENCE_ENDERS:
                # Check if it's an abbreviation
                if i > 0:
                    prev_word = tokens[i-1]['form'].lower() if tokens[i-1]['type'] == 'word' else ''
                    if prev_word in cls.ABBREVIATIONS:
                        continue
                
                # Check if followed by lowercase (not sentence end)
                if i + 1 < len(tokens):
                    next_token = tokens[i+1]
                    if next_token['type'] == 'word':
                        first_char = next_token['form'][0]
                        # If next word starts with lowercase, might not be sentence end
                        # But in Greek, this is less reliable due to different conventions
                        pass
                
                # End sentence
                sentence = ' '.join(current)
                sentence = re.sub(r'\s+([.,;:!?])', r'\1', sentence)  # Fix punctuation spacing
                sentences.append(sentence.strip())
                current = []
        
        # Add remaining tokens as final sentence
        if current:
            sentence = ' '.join(current)
            sentence = re.sub(r'\s+([.,;:!?])', r'\1', sentence)
            sentences.append(sentence.strip())
        
        return [s for s in sentences if s]  # Remove empty sentences
    
    @classmethod
    def split_with_ids(cls, text: str, text_id: str = "") -> List[Dict]:
        """Split into sentences with IDs and metadata"""
        sentences = cls.split(text)
        
        result = []
        for i, sent in enumerate(sentences):
            tokens = GreekTokenizer.tokenize(sent)
            word_tokens = [t for t in tokens if t['type'] == 'word']
            
            result.append({
                'id': f"{text_id}_s{i+1:04d}" if text_id else f"s{i+1:04d}",
                'text': sent,
                'tokens': tokens,
                'word_count': len(word_tokens),
                'token_count': len(tokens)
            })
        
        return result

# =============================================================================
# MORPHOLOGICAL ANALYSIS
# =============================================================================

@dataclass
class MorphAnalysis:
    """Morphological analysis result"""
    lemma: str
    pos: str
    person: str = ""
    number: str = ""
    tense: str = ""
    mood: str = ""
    voice: str = ""
    gender: str = ""
    case: str = ""
    degree: str = ""
    
    def to_proiel_tag(self) -> str:
        """Convert to PROIEL-style morphology tag"""
        # PROIEL format: pos-person-number-tense-mood-voice-gender-case-degree
        parts = [
            self.pos or '-',
            self.person or '-',
            self.number or '-',
            self.tense or '-',
            self.mood or '-',
            self.voice or '-',
            self.gender or '-',
            self.case or '-',
            self.degree or '-'
        ]
        return ''.join(parts)
    
    def to_ud_feats(self) -> str:
        """Convert to Universal Dependencies FEATS format"""
        feats = []
        
        if self.person:
            feats.append(f"Person={self.person}")
        if self.number:
            num_map = {'s': 'Sing', 'p': 'Plur', 'd': 'Dual'}
            feats.append(f"Number={num_map.get(self.number, self.number)}")
        if self.tense:
            tense_map = {'p': 'Pres', 'i': 'Past', 'f': 'Fut', 'a': 'Past', 'r': 'Past', 'l': 'Past'}
            feats.append(f"Tense={tense_map.get(self.tense, self.tense)}")
        if self.mood:
            mood_map = {'i': 'Ind', 's': 'Sub', 'o': 'Opt', 'm': 'Imp', 'n': 'Inf', 'p': 'Part'}
            feats.append(f"Mood={mood_map.get(self.mood, self.mood)}")
        if self.voice:
            voice_map = {'a': 'Act', 'm': 'Mid', 'p': 'Pass', 'e': 'Mid'}
            feats.append(f"Voice={voice_map.get(self.voice, self.voice)}")
        if self.gender:
            gender_map = {'m': 'Masc', 'f': 'Fem', 'n': 'Neut'}
            feats.append(f"Gender={gender_map.get(self.gender, self.gender)}")
        if self.case:
            case_map = {'n': 'Nom', 'g': 'Gen', 'd': 'Dat', 'a': 'Acc', 'v': 'Voc'}
            feats.append(f"Case={case_map.get(self.case, self.case)}")
        if self.degree:
            degree_map = {'p': 'Pos', 'c': 'Cmp', 's': 'Sup'}
            feats.append(f"Degree={degree_map.get(self.degree, self.degree)}")
        
        return '|'.join(feats) if feats else '_'


class GreekMorphAnalyzer:
    """Morphological analyzer for Greek"""
    
    # POS tag mappings
    POS_TAGS = {
        'noun': 'NOUN',
        'verb': 'VERB',
        'adj': 'ADJ',
        'adv': 'ADV',
        'pron': 'PRON',
        'det': 'DET',
        'prep': 'ADP',
        'conj': 'CCONJ',
        'part': 'PART',
        'intj': 'INTJ',
        'num': 'NUM',
        'punct': 'PUNCT',
    }
    
    # Common Greek function words with analyses
    FUNCTION_WORDS = {
        'ὁ': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='s', case='n'),
        'ἡ': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='s', case='n'),
        'τό': MorphAnalysis(lemma='ὁ', pos='DET', gender='n', number='s', case='n'),
        'τοῦ': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='s', case='g'),
        'τῆς': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='s', case='g'),
        'τῷ': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='s', case='d'),
        'τῇ': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='s', case='d'),
        'τόν': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='s', case='a'),
        'τήν': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='s', case='a'),
        'οἱ': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='p', case='n'),
        'αἱ': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='p', case='n'),
        'τά': MorphAnalysis(lemma='ὁ', pos='DET', gender='n', number='p', case='n'),
        'τῶν': MorphAnalysis(lemma='ὁ', pos='DET', number='p', case='g'),
        'τοῖς': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='p', case='d'),
        'ταῖς': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='p', case='d'),
        'τούς': MorphAnalysis(lemma='ὁ', pos='DET', gender='m', number='p', case='a'),
        'τάς': MorphAnalysis(lemma='ὁ', pos='DET', gender='f', number='p', case='a'),
        
        # Pronouns
        'ἐγώ': MorphAnalysis(lemma='ἐγώ', pos='PRON', person='1', number='s', case='n'),
        'σύ': MorphAnalysis(lemma='σύ', pos='PRON', person='2', number='s', case='n'),
        'αὐτός': MorphAnalysis(lemma='αὐτός', pos='PRON', person='3', gender='m', number='s', case='n'),
        'αὐτή': MorphAnalysis(lemma='αὐτός', pos='PRON', person='3', gender='f', number='s', case='n'),
        'αὐτό': MorphAnalysis(lemma='αὐτός', pos='PRON', person='3', gender='n', number='s', case='n'),
        
        # Prepositions
        'ἐν': MorphAnalysis(lemma='ἐν', pos='ADP'),
        'εἰς': MorphAnalysis(lemma='εἰς', pos='ADP'),
        'ἐκ': MorphAnalysis(lemma='ἐκ', pos='ADP'),
        'ἐξ': MorphAnalysis(lemma='ἐκ', pos='ADP'),
        'ἀπό': MorphAnalysis(lemma='ἀπό', pos='ADP'),
        'πρός': MorphAnalysis(lemma='πρός', pos='ADP'),
        'ὑπό': MorphAnalysis(lemma='ὑπό', pos='ADP'),
        'περί': MorphAnalysis(lemma='περί', pos='ADP'),
        'κατά': MorphAnalysis(lemma='κατά', pos='ADP'),
        'μετά': MorphAnalysis(lemma='μετά', pos='ADP'),
        'διά': MorphAnalysis(lemma='διά', pos='ADP'),
        'παρά': MorphAnalysis(lemma='παρά', pos='ADP'),
        'ἐπί': MorphAnalysis(lemma='ἐπί', pos='ADP'),
        'σύν': MorphAnalysis(lemma='σύν', pos='ADP'),
        
        # Conjunctions
        'καί': MorphAnalysis(lemma='καί', pos='CCONJ'),
        'δέ': MorphAnalysis(lemma='δέ', pos='CCONJ'),
        'ἀλλά': MorphAnalysis(lemma='ἀλλά', pos='CCONJ'),
        'ἤ': MorphAnalysis(lemma='ἤ', pos='CCONJ'),
        'γάρ': MorphAnalysis(lemma='γάρ', pos='CCONJ'),
        'οὖν': MorphAnalysis(lemma='οὖν', pos='CCONJ'),
        'τε': MorphAnalysis(lemma='τε', pos='CCONJ'),
        'ὅτι': MorphAnalysis(lemma='ὅτι', pos='SCONJ'),
        'ἵνα': MorphAnalysis(lemma='ἵνα', pos='SCONJ'),
        'ὡς': MorphAnalysis(lemma='ὡς', pos='SCONJ'),
        'εἰ': MorphAnalysis(lemma='εἰ', pos='SCONJ'),
        'ἐάν': MorphAnalysis(lemma='ἐάν', pos='SCONJ'),
        
        # Particles
        'μέν': MorphAnalysis(lemma='μέν', pos='PART'),
        'οὐ': MorphAnalysis(lemma='οὐ', pos='PART'),
        'οὐκ': MorphAnalysis(lemma='οὐ', pos='PART'),
        'οὐχ': MorphAnalysis(lemma='οὐ', pos='PART'),
        'μή': MorphAnalysis(lemma='μή', pos='PART'),
        'ἄν': MorphAnalysis(lemma='ἄν', pos='PART'),
        
        # Common verbs (εἰμί)
        'εἰμί': MorphAnalysis(lemma='εἰμί', pos='AUX', person='1', number='s', tense='p', mood='i', voice='a'),
        'εἶ': MorphAnalysis(lemma='εἰμί', pos='AUX', person='2', number='s', tense='p', mood='i', voice='a'),
        'ἐστί': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='s', tense='p', mood='i', voice='a'),
        'ἐστίν': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='s', tense='p', mood='i', voice='a'),
        'ἐσμέν': MorphAnalysis(lemma='εἰμί', pos='AUX', person='1', number='p', tense='p', mood='i', voice='a'),
        'ἐστέ': MorphAnalysis(lemma='εἰμί', pos='AUX', person='2', number='p', tense='p', mood='i', voice='a'),
        'εἰσί': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='p', tense='p', mood='i', voice='a'),
        'εἰσίν': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='p', tense='p', mood='i', voice='a'),
        'ἦν': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='s', tense='i', mood='i', voice='a'),
        'ἦσαν': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='p', tense='i', mood='i', voice='a'),
        'ἔσται': MorphAnalysis(lemma='εἰμί', pos='AUX', person='3', number='s', tense='f', mood='i', voice='a'),
        'εἶναι': MorphAnalysis(lemma='εἰμί', pos='AUX', tense='p', mood='n', voice='a'),
        'ὤν': MorphAnalysis(lemma='εἰμί', pos='AUX', gender='m', number='s', case='n', tense='p', mood='p', voice='a'),
        'οὖσα': MorphAnalysis(lemma='εἰμί', pos='AUX', gender='f', number='s', case='n', tense='p', mood='p', voice='a'),
        'ὄν': MorphAnalysis(lemma='εἰμί', pos='AUX', gender='n', number='s', case='n', tense='p', mood='p', voice='a'),
    }
    
    @classmethod
    def analyze(cls, form: str) -> Optional[MorphAnalysis]:
        """Analyze a single word form"""
        # Normalize form
        form_normalized = GreekNormalizer.normalize_unicode(form)
        
        # Check function words
        if form_normalized in cls.FUNCTION_WORDS:
            return cls.FUNCTION_WORDS[form_normalized]
        
        # Basic heuristics for unknown words
        return cls._heuristic_analysis(form_normalized)
    
    @classmethod
    def _heuristic_analysis(cls, form: str) -> MorphAnalysis:
        """Apply heuristic rules for morphological analysis"""
        # Default analysis
        analysis = MorphAnalysis(lemma=form, pos='X')
        
        # Verb endings
        verb_endings = {
            'ω': ('1', 's', 'p', 'i', 'a'),
            'εις': ('2', 's', 'p', 'i', 'a'),
            'ει': ('3', 's', 'p', 'i', 'a'),
            'ομεν': ('1', 'p', 'p', 'i', 'a'),
            'ετε': ('2', 'p', 'p', 'i', 'a'),
            'ουσι': ('3', 'p', 'p', 'i', 'a'),
            'ουσιν': ('3', 'p', 'p', 'i', 'a'),
        }
        
        for ending, (person, number, tense, mood, voice) in verb_endings.items():
            if form.endswith(ending):
                return MorphAnalysis(
                    lemma=form[:-len(ending)] + 'ω',
                    pos='VERB',
                    person=person,
                    number=number,
                    tense=tense,
                    mood=mood,
                    voice=voice
                )
        
        # Noun endings (simplified)
        noun_endings = {
            'ος': ('m', 's', 'n'),
            'ου': ('m', 's', 'g'),
            'ῳ': ('m', 's', 'd'),
            'ον': ('m', 's', 'a'),
            'οι': ('m', 'p', 'n'),
            'ων': ('', 'p', 'g'),
            'οις': ('m', 'p', 'd'),
            'ους': ('m', 'p', 'a'),
            'α': ('f', 's', 'n'),
            'ης': ('f', 's', 'g'),
            'ῃ': ('f', 's', 'd'),
            'αν': ('f', 's', 'a'),
            'αι': ('f', 'p', 'n'),
            'ας': ('f', 'p', 'a'),
        }
        
        for ending, (gender, number, case) in noun_endings.items():
            if form.endswith(ending) and len(form) > len(ending):
                return MorphAnalysis(
                    lemma=form,
                    pos='NOUN',
                    gender=gender,
                    number=number,
                    case=case
                )
        
        return analysis


# =============================================================================
# PROIEL-STYLE TOKEN
# =============================================================================

@dataclass
class PROIELToken:
    """Token in PROIEL format"""
    id: int
    form: str
    lemma: str
    pos: str
    morph: str
    head: int = 0
    relation: str = ""
    
    # Additional fields
    presentation_before: str = ""
    presentation_after: str = ""
    foreign: bool = False
    empty: bool = False
    
    def to_conllu(self) -> str:
        """Convert to CoNLL-U format"""
        # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        return '\t'.join([
            str(self.id),
            self.form,
            self.lemma,
            self.pos,
            '_',  # XPOS
            self.morph if self.morph else '_',
            str(self.head),
            self.relation if self.relation else '_',
            '_',  # DEPS
            '_'   # MISC
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

class PreprocessingPipeline:
    """Complete preprocessing pipeline"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.normalizer = GreekNormalizer
        self.tokenizer = GreekTokenizer
        self.sentence_splitter = GreekSentenceSplitter
        self.morph_analyzer = GreekMorphAnalyzer
        
    def process_text(self, text: str, text_id: str = "") -> List[Dict]:
        """Process a single text through the full pipeline"""
        # Normalize
        normalized = self.normalizer.full_normalize(text)
        
        # Split into sentences
        sentences = self.sentence_splitter.split_with_ids(normalized, text_id)
        
        # Process each sentence
        processed_sentences = []
        for sent in sentences:
            processed_tokens = []
            
            for i, token in enumerate(sent['tokens']):
                if token['type'] == 'word':
                    # Morphological analysis
                    analysis = self.morph_analyzer.analyze(token['form'])
                    
                    proiel_token = PROIELToken(
                        id=i + 1,
                        form=token['form'],
                        lemma=analysis.lemma if analysis else token['form'],
                        pos=analysis.pos if analysis else 'X',
                        morph=analysis.to_ud_feats() if analysis else '_'
                    )
                    processed_tokens.append(proiel_token.to_dict())
                    
                elif token['type'] == 'punctuation':
                    proiel_token = PROIELToken(
                        id=i + 1,
                        form=token['form'],
                        lemma=token['form'],
                        pos='PUNCT',
                        morph='_'
                    )
                    processed_tokens.append(proiel_token.to_dict())
            
            processed_sentences.append({
                'id': sent['id'],
                'text': sent['text'],
                'tokens': processed_tokens,
                'word_count': sent['word_count']
            })
        
        return processed_sentences
    
    def process_to_conllu(self, text: str, text_id: str = "") -> str:
        """Process text and output in CoNLL-U format"""
        sentences = self.process_text(text, text_id)
        
        lines = []
        for sent in sentences:
            # Add sentence metadata
            lines.append(f"# sent_id = {sent['id']}")
            lines.append(f"# text = {sent['text']}")
            
            # Add tokens
            for token in sent['tokens']:
                proiel_token = PROIELToken(**token)
                lines.append(proiel_token.to_conllu())
            
            lines.append("")  # Empty line between sentences
        
        return '\n'.join(lines)
    
    def process_file(self, input_path: str, output_format: str = 'json') -> str:
        """Process a file and save output"""
        input_path = Path(input_path)
        
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Generate text ID from filename
        text_id = input_path.stem
        
        # Process
        if output_format == 'conllu':
            output = self.process_to_conllu(text, text_id)
            output_path = self.output_dir / f"{text_id}.conllu"
        else:
            sentences = self.process_text(text, text_id)
            output = json.dumps(sentences, ensure_ascii=False, indent=2)
            output_path = self.output_dir / f"{text_id}.json"
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        
        return str(output_path)
    
    def get_statistics(self, processed_sentences: List[Dict]) -> Dict:
        """Get statistics for processed text"""
        stats = {
            'sentence_count': len(processed_sentences),
            'token_count': sum(len(s['tokens']) for s in processed_sentences),
            'word_count': sum(s['word_count'] for s in processed_sentences),
            'pos_distribution': Counter(),
            'lemma_count': 0
        }
        
        lemmas = set()
        for sent in processed_sentences:
            for token in sent['tokens']:
                stats['pos_distribution'][token['pos']] += 1
                lemmas.add(token['lemma'])
        
        stats['lemma_count'] = len(lemmas)
        stats['pos_distribution'] = dict(stats['pos_distribution'])
        
        return stats


# =============================================================================
# DATABASE STORAGE
# =============================================================================

class PreprocessedDatabase:
    """Store preprocessed texts in SQLite"""
    
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
        
        # Texts table
        c.execute('''CREATE TABLE IF NOT EXISTS preprocessed_texts (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            period TEXT,
            century TEXT,
            genre TEXT,
            source TEXT,
            language TEXT DEFAULT 'grc',
            sentence_count INTEGER,
            token_count INTEGER,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Sentences table
        c.execute('''CREATE TABLE IF NOT EXISTS preprocessed_sentences (
            id TEXT PRIMARY KEY,
            text_id TEXT,
            sentence_num INTEGER,
            text TEXT,
            token_count INTEGER,
            word_count INTEGER,
            FOREIGN KEY (text_id) REFERENCES preprocessed_texts(id)
        )''')
        
        # Tokens table
        c.execute('''CREATE TABLE IF NOT EXISTS preprocessed_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id TEXT,
            token_num INTEGER,
            form TEXT,
            lemma TEXT,
            pos TEXT,
            morph TEXT,
            head INTEGER,
            relation TEXT,
            FOREIGN KEY (sentence_id) REFERENCES preprocessed_sentences(id)
        )''')
        
        # Indexes
        c.execute('CREATE INDEX IF NOT EXISTS idx_prep_tokens_lemma ON preprocessed_tokens(lemma)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_prep_tokens_pos ON preprocessed_tokens(pos)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_prep_sentences_text ON preprocessed_sentences(text_id)')
        
        conn.commit()
        conn.close()
    
    def store_processed_text(self, text_id: str, metadata: Dict, sentences: List[Dict]):
        """Store a processed text"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Calculate totals
        total_tokens = sum(len(s['tokens']) for s in sentences)
        total_words = sum(s['word_count'] for s in sentences)
        
        # Insert text
        c.execute('''INSERT OR REPLACE INTO preprocessed_texts 
                    (id, title, author, period, century, genre, source, language,
                     sentence_count, token_count, word_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (text_id, metadata.get('title', ''), metadata.get('author', ''),
                  metadata.get('period', ''), metadata.get('century', ''),
                  metadata.get('genre', ''), metadata.get('source', ''),
                  metadata.get('language', 'grc'),
                  len(sentences), total_tokens, total_words))
        
        # Insert sentences and tokens
        for i, sent in enumerate(sentences):
            sent_id = sent['id']
            
            c.execute('''INSERT OR REPLACE INTO preprocessed_sentences
                        (id, text_id, sentence_num, text, token_count, word_count)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (sent_id, text_id, i, sent['text'], 
                      len(sent['tokens']), sent['word_count']))
            
            # Delete old tokens
            c.execute('DELETE FROM preprocessed_tokens WHERE sentence_id = ?', (sent_id,))
            
            # Insert tokens
            for j, token in enumerate(sent['tokens']):
                c.execute('''INSERT INTO preprocessed_tokens
                            (sentence_id, token_num, form, lemma, pos, morph, head, relation)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (sent_id, j, token['form'], token['lemma'],
                          token['pos'], token['morph'], 
                          token.get('head', 0), token.get('relation', '')))
        
        conn.commit()
        conn.close()
        
        return total_tokens
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        c = conn.cursor()
        
        stats = {}
        
        c.execute("SELECT COUNT(*) FROM preprocessed_texts")
        stats['text_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM preprocessed_sentences")
        stats['sentence_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM preprocessed_tokens")
        stats['token_count'] = c.fetchone()[0]
        
        c.execute("SELECT COUNT(DISTINCT lemma) FROM preprocessed_tokens")
        stats['lemma_count'] = c.fetchone()[0]
        
        c.execute("""SELECT period, COUNT(*), SUM(token_count) 
                    FROM preprocessed_texts GROUP BY period""")
        stats['by_period'] = {r[0]: {'texts': r[1], 'tokens': r[2]} for r in c.fetchall()}
        
        c.execute("""SELECT pos, COUNT(*) FROM preprocessed_tokens GROUP BY pos ORDER BY COUNT(*) DESC""")
        stats['pos_distribution'] = {r[0]: r[1] for r in c.fetchall()}
        
        conn.close()
        return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Test preprocessing
    test_text = """
    Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος.
    οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν. πάντα δι' αὐτοῦ ἐγένετο, καὶ χωρὶς αὐτοῦ 
    ἐγένετο οὐδὲ ἕν ὃ γέγονεν.
    """
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus_platform/data"
    
    pipeline = PreprocessingPipeline(data_dir)
    
    # Process test text
    sentences = pipeline.process_text(test_text, "john_1")
    
    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)
    
    for sent in sentences:
        print(f"\nSentence: {sent['text'][:50]}...")
        print(f"Tokens: {len(sent['tokens'])}")
        for token in sent['tokens'][:5]:
            print(f"  {token['form']:15} {token['lemma']:15} {token['pos']:6} {token['morph']}")
    
    # Get statistics
    stats = pipeline.get_statistics(sentences)
    print(f"\nStatistics:")
    print(f"  Sentences: {stats['sentence_count']}")
    print(f"  Tokens: {stats['token_count']}")
    print(f"  Words: {stats['word_count']}")
    print(f"  Unique lemmas: {stats['lemma_count']}")
