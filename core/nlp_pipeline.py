#!/usr/bin/env python3
"""
NLP Pipeline for Historical Greek
Complete processing pipeline for Greek texts across all periods

Components:
1. Tokenization
2. Sentence splitting
3. POS tagging
4. Lemmatization
5. Morphological analysis
6. Dependency parsing
7. Semantic role labeling
8. Named entity recognition

Supports: Ancient, Koine, Byzantine, Medieval, Early Modern Greek
"""

import os
import re
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GREEK UNICODE HANDLING
# ============================================================================

class GreekUnicode:
    """Greek Unicode normalization and handling"""
    
    # Greek character ranges
    GREEK_BASIC = range(0x0370, 0x0400)  # Greek and Coptic
    GREEK_EXTENDED = range(0x1F00, 0x2000)  # Greek Extended
    
    # Diacritics
    ACUTE = '\u0301'
    GRAVE = '\u0300'
    CIRCUMFLEX = '\u0342'
    SMOOTH = '\u0313'
    ROUGH = '\u0314'
    IOTA_SUB = '\u0345'
    DIAERESIS = '\u0308'
    MACRON = '\u0304'
    BREVE = '\u0306'
    
    # Punctuation
    ANO_TELEIA = '\u0387'  # Greek semicolon (raised dot)
    EROTIMATIKO = '\u037E'  # Greek question mark
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize Greek text to NFC form"""
        return unicodedata.normalize('NFC', text)
    
    @classmethod
    def decompose(cls, text: str) -> str:
        """Decompose Greek text to NFD form"""
        return unicodedata.normalize('NFD', text)
    
    @classmethod
    def strip_diacritics(cls, text: str) -> str:
        """Remove all diacritics from Greek text"""
        decomposed = cls.decompose(text)
        return ''.join(c for c in decomposed if not unicodedata.combining(c))
    
    @classmethod
    def is_greek(cls, char: str) -> bool:
        """Check if character is Greek"""
        if len(char) != 1:
            return False
        code = ord(char)
        return (code in cls.GREEK_BASIC or 
                code in cls.GREEK_EXTENDED or
                unicodedata.name(char, '').startswith('GREEK'))
    
    @classmethod
    def is_greek_word(cls, word: str) -> bool:
        """Check if word is Greek"""
        return any(cls.is_greek(c) for c in word)
    
    @classmethod
    def get_diacritics(cls, char: str) -> List[str]:
        """Get list of diacritics on a character"""
        decomposed = cls.decompose(char)
        return [c for c in decomposed if unicodedata.combining(c)]
    
    @classmethod
    def standardize_sigma(cls, text: str) -> str:
        """Standardize final sigma"""
        # Replace word-final sigma with final sigma
        result = []
        words = text.split()
        for word in words:
            if word and word[-1] == 'σ':
                word = word[:-1] + 'ς'
            result.append(word)
        return ' '.join(result)


# ============================================================================
# TOKENIZER
# ============================================================================

class GreekTokenizer:
    """Tokenizer for Greek texts"""
    
    def __init__(self):
        # Punctuation patterns
        self.punct_pattern = re.compile(r'([.,;:!?\'"«»\[\](){}·;])')
        
        # Elision pattern (apostrophe between words)
        self.elision_pattern = re.compile(r"(\w+)'(\w+)")
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+')
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Greek text"""
        # Normalize
        text = GreekUnicode.normalize(text)
        
        # Handle elision
        text = self.elision_pattern.sub(r'\1 \2', text)
        
        # Separate punctuation
        text = self.punct_pattern.sub(r' \1 ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[List[str]]:
        """Tokenize text into sentences, then tokens"""
        sentences = self.split_sentences(text)
        return [self.tokenize(sent) for sent in sentences]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Greek sentence-ending punctuation
        sent_end = re.compile(r'([.;·!?])\s+')
        
        # Split
        parts = sent_end.split(text)
        
        # Reconstruct sentences
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences


# ============================================================================
# POS TAGGER
# ============================================================================

class GreekPOSTagger:
    """POS tagger for Greek"""
    
    # PROIEL-style POS tags
    POS_TAGS = {
        'A-': 'Adjective',
        'C-': 'Conjunction',
        'Df': 'Adverb',
        'Dq': 'Relative adverb',
        'Du': 'Interrogative adverb',
        'F-': 'Foreign word',
        'G-': 'Subjunction',
        'I-': 'Interjection',
        'Ma': 'Cardinal numeral',
        'Mo': 'Ordinal numeral',
        'Nb': 'Common noun',
        'Ne': 'Proper noun',
        'Pc': 'Reciprocal pronoun',
        'Pd': 'Demonstrative pronoun',
        'Pi': 'Interrogative pronoun',
        'Pk': 'Reflexive pronoun',
        'Pp': 'Personal pronoun',
        'Pr': 'Relative pronoun',
        'Ps': 'Possessive pronoun',
        'Pt': 'Determiner',
        'Px': 'Indefinite pronoun',
        'Py': 'Quantifier',
        'R-': 'Preposition',
        'S-': 'Article',
        'V-': 'Verb',
        'X-': 'Unassigned'
    }
    
    # Morphological features
    PERSON = {'1': 'first', '2': 'second', '3': 'third'}
    NUMBER = {'s': 'singular', 'd': 'dual', 'p': 'plural'}
    TENSE = {'p': 'present', 'i': 'imperfect', 'r': 'perfect', 
             'l': 'pluperfect', 'f': 'future', 'a': 'aorist', 't': 'future perfect'}
    MOOD = {'i': 'indicative', 's': 'subjunctive', 'o': 'optative',
            'm': 'imperative', 'n': 'infinitive', 'p': 'participle'}
    VOICE = {'a': 'active', 'm': 'middle', 'p': 'passive'}
    GENDER = {'m': 'masculine', 'f': 'feminine', 'n': 'neuter'}
    CASE = {'n': 'nominative', 'g': 'genitive', 'd': 'dative', 
            'a': 'accusative', 'v': 'vocative'}
    DEGREE = {'p': 'positive', 'c': 'comparative', 's': 'superlative'}
    
    def __init__(self):
        self.model = None
        self.lexicon = {}
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load POS lexicon"""
        # Common Greek words with POS
        self.lexicon = {
            'ὁ': 'S-',
            'ἡ': 'S-',
            'τό': 'S-',
            'τοῦ': 'S-',
            'τῆς': 'S-',
            'τῷ': 'S-',
            'τῇ': 'S-',
            'τόν': 'S-',
            'τήν': 'S-',
            'οἱ': 'S-',
            'αἱ': 'S-',
            'τά': 'S-',
            'καί': 'C-',
            'δέ': 'C-',
            'γάρ': 'C-',
            'ἀλλά': 'C-',
            'ἤ': 'C-',
            'τε': 'C-',
            'μέν': 'C-',
            'οὖν': 'C-',
            'ἐν': 'R-',
            'εἰς': 'R-',
            'ἐκ': 'R-',
            'ἐξ': 'R-',
            'ἀπό': 'R-',
            'πρός': 'R-',
            'ὑπό': 'R-',
            'περί': 'R-',
            'διά': 'R-',
            'κατά': 'R-',
            'μετά': 'R-',
            'παρά': 'R-',
            'ἐπί': 'R-',
            'οὐ': 'Df',
            'οὐκ': 'Df',
            'οὐχ': 'Df',
            'μή': 'Df',
            'ὡς': 'G-',
            'ὅτι': 'G-',
            'εἰ': 'G-',
            'ἵνα': 'G-',
            'ὅτε': 'G-',
            'ἐάν': 'G-',
            'ἄν': 'G-',
            'ἐγώ': 'Pp',
            'σύ': 'Pp',
            'ἡμεῖς': 'Pp',
            'ὑμεῖς': 'Pp',
            'αὐτός': 'Pp',
            'αὐτή': 'Pp',
            'αὐτό': 'Pp',
            'οὗτος': 'Pd',
            'αὕτη': 'Pd',
            'τοῦτο': 'Pd',
            'ἐκεῖνος': 'Pd',
            'ὅς': 'Pr',
            'ἥ': 'Pr',
            'ὅ': 'Pr',
            'τίς': 'Pi',
            'τί': 'Pi',
            'εἰμί': 'V-',
            'ἐστί': 'V-',
            'ἐστίν': 'V-',
            'εἶναι': 'V-',
            'ἦν': 'V-',
            'λέγω': 'V-',
            'λέγει': 'V-',
            'εἶπεν': 'V-',
            'ἔχω': 'V-',
            'ἔχει': 'V-',
            'γίγνομαι': 'V-',
            'γίνομαι': 'V-',
            'ποιέω': 'V-',
            'ποιεῖ': 'V-',
        }
    
    def tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Tag tokens with POS"""
        tagged = []
        
        for token in tokens:
            # Normalize
            normalized = GreekUnicode.normalize(token.lower())
            
            # Lookup in lexicon
            if normalized in self.lexicon:
                tag = self.lexicon[normalized]
            elif GreekUnicode.strip_diacritics(normalized) in self.lexicon:
                tag = self.lexicon[GreekUnicode.strip_diacritics(normalized)]
            else:
                # Heuristic tagging
                tag = self._heuristic_tag(token)
            
            tagged.append((token, tag))
        
        return tagged
    
    def _heuristic_tag(self, token: str) -> str:
        """Heuristic POS tagging"""
        # Check if punctuation
        if re.match(r'^[.,;:!?·;«»\[\](){}]+$', token):
            return 'X-'
        
        # Check if number
        if re.match(r'^\d+$', token):
            return 'Ma'
        
        # Check common endings
        lower = token.lower()
        
        # Verb endings
        if lower.endswith(('ω', 'εις', 'ει', 'ομεν', 'ετε', 'ουσι', 'ουσιν')):
            return 'V-'
        if lower.endswith(('ον', 'ες', 'ε', 'ομεν', 'ετε', 'ον')):
            return 'V-'
        if lower.endswith(('μαι', 'σαι', 'ται', 'μεθα', 'σθε', 'νται')):
            return 'V-'
        
        # Noun endings
        if lower.endswith(('ος', 'ου', 'ῳ', 'ον', 'οι', 'ων', 'οις', 'ους')):
            return 'Nb'
        if lower.endswith(('η', 'ης', 'ῃ', 'ην', 'αι', 'ων', 'αις', 'ας')):
            return 'Nb'
        if lower.endswith(('α', 'ας', 'ᾳ', 'αν')):
            return 'Nb'
        
        # Adjective endings
        if lower.endswith(('ος', 'α', 'ον', 'η', 'ες')):
            return 'A-'
        
        # Default
        return 'X-'
    
    def parse_morphology(self, tag: str) -> Dict[str, str]:
        """Parse morphological tag"""
        if len(tag) < 2:
            return {"pos": tag}
        
        result = {"pos": tag[:2]}
        
        # Extended morphology (PROIEL format: 10 positions)
        if len(tag) >= 3:
            result["person"] = self.PERSON.get(tag[2], None)
        if len(tag) >= 4:
            result["number"] = self.NUMBER.get(tag[3], None)
        if len(tag) >= 5:
            result["tense"] = self.TENSE.get(tag[4], None)
        if len(tag) >= 6:
            result["mood"] = self.MOOD.get(tag[5], None)
        if len(tag) >= 7:
            result["voice"] = self.VOICE.get(tag[6], None)
        if len(tag) >= 8:
            result["gender"] = self.GENDER.get(tag[7], None)
        if len(tag) >= 9:
            result["case"] = self.CASE.get(tag[8], None)
        if len(tag) >= 10:
            result["degree"] = self.DEGREE.get(tag[9], None)
        
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


# ============================================================================
# LEMMATIZER
# ============================================================================

class GreekLemmatizer:
    """Lemmatizer for Greek"""
    
    def __init__(self):
        self.lexicon = {}
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load lemma lexicon"""
        # Common Greek forms -> lemmas
        self.lexicon = {
            # Article
            'ὁ': 'ὁ', 'ἡ': 'ὁ', 'τό': 'ὁ',
            'τοῦ': 'ὁ', 'τῆς': 'ὁ', 'τοῦ': 'ὁ',
            'τῷ': 'ὁ', 'τῇ': 'ὁ', 'τῷ': 'ὁ',
            'τόν': 'ὁ', 'τήν': 'ὁ', 'τό': 'ὁ',
            'οἱ': 'ὁ', 'αἱ': 'ὁ', 'τά': 'ὁ',
            'τῶν': 'ὁ', 'τοῖς': 'ὁ', 'ταῖς': 'ὁ',
            'τούς': 'ὁ', 'τάς': 'ὁ',
            
            # εἰμί (to be)
            'εἰμί': 'εἰμί', 'εἶ': 'εἰμί', 'ἐστί': 'εἰμί', 'ἐστίν': 'εἰμί',
            'ἐσμέν': 'εἰμί', 'ἐστέ': 'εἰμί', 'εἰσί': 'εἰμί', 'εἰσίν': 'εἰμί',
            'ἦν': 'εἰμί', 'ἦσθα': 'εἰμί', 'ἦμεν': 'εἰμί', 'ἦτε': 'εἰμί', 'ἦσαν': 'εἰμί',
            'ἔσομαι': 'εἰμί', 'ἔσῃ': 'εἰμί', 'ἔσται': 'εἰμί',
            'εἶναι': 'εἰμί', 'ὤν': 'εἰμί', 'οὖσα': 'εἰμί', 'ὄν': 'εἰμί',
            
            # λέγω (to say)
            'λέγω': 'λέγω', 'λέγεις': 'λέγω', 'λέγει': 'λέγω',
            'λέγομεν': 'λέγω', 'λέγετε': 'λέγω', 'λέγουσι': 'λέγω',
            'ἔλεγον': 'λέγω', 'ἔλεγες': 'λέγω', 'ἔλεγε': 'λέγω',
            'εἶπον': 'λέγω', 'εἶπες': 'λέγω', 'εἶπε': 'λέγω', 'εἶπεν': 'λέγω',
            'λέξω': 'λέγω', 'εἴρηκα': 'λέγω', 'εἴρημαι': 'λέγω',
            
            # ἔχω (to have)
            'ἔχω': 'ἔχω', 'ἔχεις': 'ἔχω', 'ἔχει': 'ἔχω',
            'ἔχομεν': 'ἔχω', 'ἔχετε': 'ἔχω', 'ἔχουσι': 'ἔχω',
            'εἶχον': 'ἔχω', 'ἕξω': 'ἔχω', 'ἔσχον': 'ἔχω', 'ἔσχηκα': 'ἔχω',
            
            # ποιέω (to do/make)
            'ποιέω': 'ποιέω', 'ποιεῖς': 'ποιέω', 'ποιεῖ': 'ποιέω',
            'ποιοῦμεν': 'ποιέω', 'ποιεῖτε': 'ποιέω', 'ποιοῦσι': 'ποιέω',
            'ἐποίουν': 'ποιέω', 'ἐποίησα': 'ποιέω', 'πεποίηκα': 'ποιέω',
            
            # γίγνομαι (to become)
            'γίγνομαι': 'γίγνομαι', 'γίνομαι': 'γίγνομαι',
            'γίγνεται': 'γίγνομαι', 'γίνεται': 'γίγνομαι',
            'ἐγένετο': 'γίγνομαι', 'γέγονα': 'γίγνομαι', 'γεγένημαι': 'γίγνομαι',
            
            # Common nouns
            'ἄνθρωπος': 'ἄνθρωπος', 'ἀνθρώπου': 'ἄνθρωπος', 'ἀνθρώπῳ': 'ἄνθρωπος',
            'ἄνθρωπον': 'ἄνθρωπος', 'ἄνθρωποι': 'ἄνθρωπος', 'ἀνθρώπων': 'ἄνθρωπος',
            'λόγος': 'λόγος', 'λόγου': 'λόγος', 'λόγῳ': 'λόγος',
            'λόγον': 'λόγος', 'λόγοι': 'λόγος', 'λόγων': 'λόγος',
            'θεός': 'θεός', 'θεοῦ': 'θεός', 'θεῷ': 'θεός',
            'θεόν': 'θεός', 'θεοί': 'θεός', 'θεῶν': 'θεός',
        }
    
    def lemmatize(self, token: str) -> str:
        """Get lemma for token"""
        normalized = GreekUnicode.normalize(token.lower())
        
        # Direct lookup
        if normalized in self.lexicon:
            return self.lexicon[normalized]
        
        # Try without diacritics
        stripped = GreekUnicode.strip_diacritics(normalized)
        for form, lemma in self.lexicon.items():
            if GreekUnicode.strip_diacritics(form) == stripped:
                return lemma
        
        # Return original if not found
        return token
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Lemmatize list of tokens"""
        return [(token, self.lemmatize(token)) for token in tokens]


# ============================================================================
# DEPENDENCY PARSER
# ============================================================================

class GreekDependencyParser:
    """Dependency parser for Greek using PROIEL relations"""
    
    # PROIEL dependency relations
    RELATIONS = {
        'pred': 'Predicate',
        'sub': 'Subject',
        'obj': 'Object',
        'obl': 'Oblique',
        'adv': 'Adverbial',
        'atr': 'Attribute',
        'apos': 'Apposition',
        'aux': 'Auxiliary',
        'comp': 'Complement',
        'expl': 'Expletive',
        'narg': 'Non-argument',
        'nonsub': 'Non-subject',
        'parpred': 'Parenthetical',
        'per': 'Peripheral',
        'pid': 'Predicate identity',
        'voc': 'Vocative',
        'xadv': 'External adverbial',
        'xobj': 'External object',
        'xsub': 'External subject'
    }
    
    def __init__(self):
        self.model = None
    
    def parse(self, tokens: List[str], pos_tags: List[str]) -> List[Dict]:
        """Parse sentence to dependency tree"""
        # Simple rule-based parsing
        tree = []
        
        # Find main verb (predicate)
        verb_idx = None
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos.startswith('V'):
                verb_idx = i
                break
        
        # Build tree
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            node = {
                'id': i + 1,
                'form': token,
                'pos': pos,
                'head': 0,
                'relation': 'root'
            }
            
            if verb_idx is not None:
                if i == verb_idx:
                    node['head'] = 0
                    node['relation'] = 'pred'
                elif pos.startswith('S'):  # Article
                    # Attach to following noun
                    node['head'] = i + 2 if i + 1 < len(tokens) else verb_idx + 1
                    node['relation'] = 'atr'
                elif pos.startswith('N'):  # Noun
                    # Check case for relation
                    node['head'] = verb_idx + 1
                    if i < verb_idx:
                        node['relation'] = 'sub'
                    else:
                        node['relation'] = 'obj'
                elif pos.startswith('R'):  # Preposition
                    node['head'] = verb_idx + 1
                    node['relation'] = 'obl'
                elif pos.startswith('A'):  # Adjective
                    # Attach to nearest noun
                    node['head'] = i  # Previous word
                    node['relation'] = 'atr'
                elif pos.startswith('Df'):  # Adverb
                    node['head'] = verb_idx + 1
                    node['relation'] = 'adv'
                elif pos.startswith('C') or pos.startswith('G'):  # Conjunction
                    node['head'] = verb_idx + 1
                    node['relation'] = 'aux'
                else:
                    node['head'] = verb_idx + 1 if verb_idx is not None else 0
                    node['relation'] = 'narg'
            
            tree.append(node)
        
        return tree
    
    def to_conllu(self, tree: List[Dict]) -> str:
        """Convert tree to CoNLL-U format"""
        lines = []
        for node in tree:
            line = f"{node['id']}\t{node['form']}\t_\t{node['pos']}\t_\t_\t{node['head']}\t{node['relation']}\t_\t_"
            lines.append(line)
        return '\n'.join(lines)


# ============================================================================
# SEMANTIC ROLE LABELER
# ============================================================================

class GreekSRLabeler:
    """Semantic Role Labeler for Greek"""
    
    # PropBank-style roles
    ROLES = {
        'ARG0': 'Agent',
        'ARG1': 'Patient/Theme',
        'ARG2': 'Instrument/Beneficiary',
        'ARG3': 'Starting point',
        'ARG4': 'Ending point',
        'ARGM-LOC': 'Location',
        'ARGM-TMP': 'Temporal',
        'ARGM-MNR': 'Manner',
        'ARGM-CAU': 'Cause',
        'ARGM-PRP': 'Purpose',
        'ARGM-DIR': 'Direction',
        'ARGM-EXT': 'Extent',
        'ARGM-NEG': 'Negation',
        'ARGM-MOD': 'Modal',
        'ARGM-ADV': 'Adverbial'
    }
    
    def __init__(self):
        self.verb_frames = {}
        self._load_frames()
    
    def _load_frames(self):
        """Load verb frames"""
        # Common Greek verb frames
        self.verb_frames = {
            'δίδωμι': {
                'ARG0': 'giver',
                'ARG1': 'thing given',
                'ARG2': 'recipient'
            },
            'λέγω': {
                'ARG0': 'speaker',
                'ARG1': 'utterance',
                'ARG2': 'hearer'
            },
            'ποιέω': {
                'ARG0': 'agent',
                'ARG1': 'thing made'
            },
            'ἔχω': {
                'ARG0': 'possessor',
                'ARG1': 'possession'
            },
            'ἔρχομαι': {
                'ARG0': 'entity in motion',
                'ARG1': 'destination',
                'ARG2': 'source'
            },
            'ὁράω': {
                'ARG0': 'perceiver',
                'ARG1': 'thing seen'
            },
            'ἀκούω': {
                'ARG0': 'hearer',
                'ARG1': 'sound/utterance',
                'ARG2': 'source'
            }
        }
    
    def label(self, tokens: List[str], pos_tags: List[str], 
              lemmas: List[str], tree: List[Dict]) -> List[Dict]:
        """Label semantic roles"""
        labels = []
        
        # Find predicate
        pred_idx = None
        pred_lemma = None
        for i, (pos, lemma) in enumerate(zip(pos_tags, lemmas)):
            if pos.startswith('V'):
                pred_idx = i
                pred_lemma = lemma
                break
        
        if pred_idx is None:
            return labels
        
        # Get frame for predicate
        frame = self.verb_frames.get(pred_lemma, {})
        
        # Label arguments
        for i, node in enumerate(tree):
            label = {
                'token_id': i,
                'token': tokens[i],
                'role': None,
                'description': None
            }
            
            if node['relation'] == 'sub':
                label['role'] = 'ARG0'
                label['description'] = frame.get('ARG0', 'Agent')
            elif node['relation'] == 'obj':
                label['role'] = 'ARG1'
                label['description'] = frame.get('ARG1', 'Patient')
            elif node['relation'] == 'obl':
                # Check preposition for role
                if i > 0 and pos_tags[i-1].startswith('R'):
                    prep = tokens[i-1].lower()
                    if prep in ('εἰς', 'πρός'):
                        label['role'] = 'ARG4'
                        label['description'] = 'Goal'
                    elif prep in ('ἐκ', 'ἀπό'):
                        label['role'] = 'ARG3'
                        label['description'] = 'Source'
                    elif prep in ('ἐν',):
                        label['role'] = 'ARGM-LOC'
                        label['description'] = 'Location'
                    elif prep in ('διά',):
                        label['role'] = 'ARGM-CAU'
                        label['description'] = 'Cause'
                else:
                    label['role'] = 'ARG2'
                    label['description'] = frame.get('ARG2', 'Oblique')
            elif node['relation'] == 'adv':
                label['role'] = 'ARGM-ADV'
                label['description'] = 'Adverbial'
            
            if label['role']:
                labels.append(label)
        
        return labels


# ============================================================================
# NAMED ENTITY RECOGNIZER
# ============================================================================

class GreekNER:
    """Named Entity Recognizer for Greek"""
    
    ENTITY_TYPES = {
        'PER': 'Person',
        'LOC': 'Location',
        'ORG': 'Organization',
        'GPE': 'Geo-political entity',
        'DATE': 'Date',
        'TIME': 'Time',
        'MISC': 'Miscellaneous'
    }
    
    def __init__(self):
        self.gazetteer = {}
        self._load_gazetteer()
    
    def _load_gazetteer(self):
        """Load named entity gazetteer"""
        # Common Greek names and places
        self.gazetteer = {
            # Persons
            'Σωκράτης': 'PER',
            'Πλάτων': 'PER',
            'Ἀριστοτέλης': 'PER',
            'Ὅμηρος': 'PER',
            'Ἡρόδοτος': 'PER',
            'Θουκυδίδης': 'PER',
            'Ξενοφῶν': 'PER',
            'Δημοσθένης': 'PER',
            'Ἀλέξανδρος': 'PER',
            'Περικλῆς': 'PER',
            'Ἰησοῦς': 'PER',
            'Παῦλος': 'PER',
            'Πέτρος': 'PER',
            
            # Locations
            'Ἀθῆναι': 'LOC',
            'Ἀθήνη': 'LOC',
            'Σπάρτη': 'LOC',
            'Κόρινθος': 'LOC',
            'Θῆβαι': 'LOC',
            'Ἑλλάς': 'LOC',
            'Ἰταλία': 'LOC',
            'Αἴγυπτος': 'LOC',
            'Περσία': 'LOC',
            'Ῥώμη': 'LOC',
            'Ἱερουσαλήμ': 'LOC',
            'Γαλιλαία': 'LOC',
            
            # Geo-political
            'Ἕλληνες': 'GPE',
            'Πέρσαι': 'GPE',
            'Ῥωμαῖοι': 'GPE',
            'Ἰουδαῖοι': 'GPE',
        }
    
    def recognize(self, tokens: List[str], pos_tags: List[str]) -> List[Dict]:
        """Recognize named entities"""
        entities = []
        
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            normalized = GreekUnicode.normalize(token)
            
            # Check gazetteer
            if normalized in self.gazetteer:
                entities.append({
                    'start': i,
                    'end': i + 1,
                    'text': token,
                    'type': self.gazetteer[normalized],
                    'confidence': 1.0
                })
            # Check if proper noun
            elif pos.startswith('Ne'):
                entities.append({
                    'start': i,
                    'end': i + 1,
                    'text': token,
                    'type': 'PER',  # Default to person
                    'confidence': 0.7
                })
        
        return entities


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

class GreekNLPPipeline:
    """Complete NLP pipeline for Greek"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        
        # Initialize components
        self.tokenizer = GreekTokenizer()
        self.pos_tagger = GreekPOSTagger()
        self.lemmatizer = GreekLemmatizer()
        self.parser = GreekDependencyParser()
        self.srl = GreekSRLabeler()
        self.ner = GreekNER()
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                sentence_idx INTEGER,
                text TEXT,
                tokens TEXT,
                pos_tags TEXT,
                lemmas TEXT,
                parse_tree TEXT,
                srl_labels TEXT,
                entities TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process(self, text: str) -> Dict:
        """Process text through full pipeline"""
        result = {
            'text': text,
            'sentences': []
        }
        
        # Split into sentences
        sentences = self.tokenizer.split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            sent_result = self.process_sentence(sentence)
            sent_result['sentence_idx'] = sent_idx
            result['sentences'].append(sent_result)
        
        return result
    
    def process_sentence(self, sentence: str) -> Dict:
        """Process single sentence"""
        # Tokenize
        tokens = self.tokenizer.tokenize(sentence)
        
        # POS tag
        tagged = self.pos_tagger.tag(tokens)
        pos_tags = [tag for _, tag in tagged]
        
        # Lemmatize
        lemmatized = self.lemmatizer.lemmatize_tokens(tokens)
        lemmas = [lemma for _, lemma in lemmatized]
        
        # Parse
        tree = self.parser.parse(tokens, pos_tags)
        
        # SRL
        srl_labels = self.srl.label(tokens, pos_tags, lemmas, tree)
        
        # NER
        entities = self.ner.recognize(tokens, pos_tags)
        
        return {
            'text': sentence,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'lemmas': lemmas,
            'parse_tree': tree,
            'srl_labels': srl_labels,
            'entities': entities
        }
    
    def save_result(self, document_id: str, result: Dict):
        """Save processing result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sent in result['sentences']:
            cursor.execute("""
                INSERT INTO processed_sentences
                (document_id, sentence_idx, text, tokens, pos_tags, lemmas,
                 parse_tree, srl_labels, entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                sent['sentence_idx'],
                sent['text'],
                json.dumps(sent['tokens']),
                json.dumps(sent['pos_tags']),
                json.dumps(sent['lemmas']),
                json.dumps(sent['parse_tree']),
                json.dumps(sent['srl_labels']),
                json.dumps(sent['entities'])
            ))
        
        conn.commit()
        conn.close()
    
    def process_document(self, document_id: str, text: str) -> Dict:
        """Process and save document"""
        result = self.process(text)
        self.save_result(document_id, result)
        return result


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Greek NLP Pipeline")
    parser.add_argument('command', choices=['process', 'tokenize', 'tag', 'parse'],
                       help="Command to run")
    parser.add_argument('--text', '-t', help="Text to process")
    parser.add_argument('--file', '-f', help="File to process")
    parser.add_argument('--output', '-o', help="Output file")
    
    args = parser.parse_args()
    
    pipeline = GreekNLPPipeline()
    
    # Get text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = "ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν."
    
    if args.command == 'process':
        result = pipeline.process(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.command == 'tokenize':
        tokens = pipeline.tokenizer.tokenize(text)
        print(' | '.join(tokens))
    
    elif args.command == 'tag':
        tokens = pipeline.tokenizer.tokenize(text)
        tagged = pipeline.pos_tagger.tag(tokens)
        for token, tag in tagged:
            print(f"{token}\t{tag}")
    
    elif args.command == 'parse':
        result = pipeline.process_sentence(text)
        print(pipeline.parser.to_conllu(result['parse_tree']))


if __name__ == "__main__":
    main()
