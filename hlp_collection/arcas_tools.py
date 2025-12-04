"""
ARCAS Tools Integration - Digital Humanities Tools for Historical Linguistics

ARCAS (Advanced Research Computing for Arts and Sciences) provides
tools for digital humanities research including:
- Text analysis and annotation
- Corpus linguistics tools
- Named entity recognition for historical texts
- Lemmatization for ancient languages
- Morphological analysis
- Treebank tools

This module integrates ARCAS-style tools for diachronic linguistics.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LemmaEntry:
    form: str
    lemma: str
    pos: str
    morphology: Dict[str, str] = field(default_factory=dict)
    frequency: int = 1
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'form': self.form,
            'lemma': self.lemma,
            'pos': self.pos,
            'morphology': self.morphology,
            'frequency': self.frequency,
            'sources': self.sources,
        }


@dataclass
class MorphAnalysis:
    form: str
    lemma: str
    pos: str
    person: Optional[str] = None
    number: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    voice: Optional[str] = None
    gender: Optional[str] = None
    case: Optional[str] = None
    degree: Optional[str] = None
    
    def to_proiel_string(self) -> str:
        parts = []
        if self.person:
            parts.append(self.person[0])
        else:
            parts.append('-')
        if self.number:
            parts.append(self.number[0])
        else:
            parts.append('-')
        if self.tense:
            parts.append(self.tense[0])
        else:
            parts.append('-')
        if self.mood:
            parts.append(self.mood[0])
        else:
            parts.append('-')
        if self.voice:
            parts.append(self.voice[0])
        else:
            parts.append('-')
        if self.gender:
            parts.append(self.gender[0])
        else:
            parts.append('-')
        if self.case:
            parts.append(self.case[0])
        else:
            parts.append('-')
        if self.degree:
            parts.append(self.degree[0])
        else:
            parts.append('-')
        
        return ''.join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'form': self.form,
            'lemma': self.lemma,
            'pos': self.pos,
            'person': self.person,
            'number': self.number,
            'tense': self.tense,
            'mood': self.mood,
            'voice': self.voice,
            'gender': self.gender,
            'case': self.case,
            'degree': self.degree,
            'proiel_morph': self.to_proiel_string(),
        }


class GreekLemmatizer:
    
    COMMON_LEMMAS = {
        'ὁ': 'ὁ',
        'ἡ': 'ὁ',
        'τό': 'ὁ',
        'τοῦ': 'ὁ',
        'τῆς': 'ὁ',
        'τῷ': 'ὁ',
        'τήν': 'ὁ',
        'τόν': 'ὁ',
        'οἱ': 'ὁ',
        'αἱ': 'ὁ',
        'τά': 'ὁ',
        'τῶν': 'ὁ',
        'τοῖς': 'ὁ',
        'ταῖς': 'ὁ',
        'τούς': 'ὁ',
        'τάς': 'ὁ',
        'καί': 'καί',
        'δέ': 'δέ',
        'μέν': 'μέν',
        'γάρ': 'γάρ',
        'ἀλλά': 'ἀλλά',
        'οὐ': 'οὐ',
        'οὐκ': 'οὐ',
        'οὐχ': 'οὐ',
        'μή': 'μή',
        'εἰ': 'εἰ',
        'ἐάν': 'ἐάν',
        'ὅτι': 'ὅτι',
        'ὡς': 'ὡς',
        'ἐν': 'ἐν',
        'εἰς': 'εἰς',
        'ἐκ': 'ἐκ',
        'ἐξ': 'ἐκ',
        'ἀπό': 'ἀπό',
        'πρός': 'πρός',
        'ὑπό': 'ὑπό',
        'περί': 'περί',
        'διά': 'διά',
        'κατά': 'κατά',
        'μετά': 'μετά',
        'παρά': 'παρά',
        'ἐπί': 'ἐπί',
        'εἰμί': 'εἰμί',
        'ἐστί': 'εἰμί',
        'ἐστίν': 'εἰμί',
        'ἦν': 'εἰμί',
        'εἶναι': 'εἰμί',
        'ὤν': 'εἰμί',
        'οὖσα': 'εἰμί',
        'ὄν': 'εἰμί',
        'λέγω': 'λέγω',
        'λέγει': 'λέγω',
        'λέγειν': 'λέγω',
        'εἶπον': 'λέγω',
        'εἶπε': 'λέγω',
        'εἶπεν': 'λέγω',
        'ἔφη': 'φημί',
        'φησί': 'φημί',
        'φησίν': 'φημί',
        'ποιέω': 'ποιέω',
        'ποιεῖ': 'ποιέω',
        'ποιεῖν': 'ποιέω',
        'ἐποίησε': 'ποιέω',
        'ἐποίησεν': 'ποιέω',
        'γίγνομαι': 'γίγνομαι',
        'γίνομαι': 'γίγνομαι',
        'γίνεται': 'γίγνομαι',
        'ἐγένετο': 'γίγνομαι',
        'ἔχω': 'ἔχω',
        'ἔχει': 'ἔχω',
        'ἔχειν': 'ἔχω',
        'εἶχε': 'ἔχω',
        'εἶχεν': 'ἔχω',
        'αὐτός': 'αὐτός',
        'αὐτή': 'αὐτός',
        'αὐτό': 'αὐτός',
        'αὐτοῦ': 'αὐτός',
        'αὐτῆς': 'αὐτός',
        'αὐτῷ': 'αὐτός',
        'αὐτῇ': 'αὐτός',
        'αὐτόν': 'αὐτός',
        'αὐτήν': 'αὐτός',
        'αὐτοί': 'αὐτός',
        'αὐταί': 'αὐτός',
        'αὐτά': 'αὐτός',
        'αὐτῶν': 'αὐτός',
        'αὐτοῖς': 'αὐτός',
        'αὐταῖς': 'αὐτός',
        'αὐτούς': 'αὐτός',
        'αὐτάς': 'αὐτός',
        'οὗτος': 'οὗτος',
        'αὕτη': 'οὗτος',
        'τοῦτο': 'οὗτος',
        'τούτου': 'οὗτος',
        'ταύτης': 'οὗτος',
        'τούτῳ': 'οὗτος',
        'ταύτῃ': 'οὗτος',
        'τοῦτον': 'οὗτος',
        'ταύτην': 'οὗτος',
        'οὗτοι': 'οὗτος',
        'αὗται': 'οὗτος',
        'ταῦτα': 'οὗτος',
        'τούτων': 'οὗτος',
        'τούτοις': 'οὗτος',
        'ταύταις': 'οὗτος',
        'τούτους': 'οὗτος',
        'ταύτας': 'οὗτος',
        'ἐκεῖνος': 'ἐκεῖνος',
        'ἐκείνη': 'ἐκεῖνος',
        'ἐκεῖνο': 'ἐκεῖνος',
        'ὅς': 'ὅς',
        'ἥ': 'ὅς',
        'ὅ': 'ὅς',
        'οὗ': 'ὅς',
        'ἧς': 'ὅς',
        'ᾧ': 'ὅς',
        'ᾗ': 'ὅς',
        'ὅν': 'ὅς',
        'ἥν': 'ὅς',
        'οἵ': 'ὅς',
        'αἵ': 'ὅς',
        'ἅ': 'ὅς',
        'ὧν': 'ὅς',
        'οἷς': 'ὅς',
        'αἷς': 'ὅς',
        'οὕς': 'ὅς',
        'ἅς': 'ὅς',
        'τίς': 'τίς',
        'τί': 'τίς',
        'τίνος': 'τίς',
        'τίνι': 'τίς',
        'τίνα': 'τίς',
        'τίνες': 'τίς',
        'τίνων': 'τίς',
        'τίσι': 'τίς',
        'τίνας': 'τίς',
        'τις': 'τις',
        'τι': 'τις',
        'τινός': 'τις',
        'τινί': 'τις',
        'τινά': 'τις',
        'τινές': 'τις',
        'τινῶν': 'τις',
        'τισί': 'τις',
        'τινάς': 'τις',
        'πᾶς': 'πᾶς',
        'πᾶσα': 'πᾶς',
        'πᾶν': 'πᾶς',
        'παντός': 'πᾶς',
        'πάσης': 'πᾶς',
        'παντί': 'πᾶς',
        'πάσῃ': 'πᾶς',
        'πάντα': 'πᾶς',
        'πᾶσαν': 'πᾶς',
        'πάντες': 'πᾶς',
        'πᾶσαι': 'πᾶς',
        'πάντων': 'πᾶς',
        'πασῶν': 'πᾶς',
        'πᾶσι': 'πᾶς',
        'πάσαις': 'πᾶς',
        'πάντας': 'πᾶς',
        'πάσας': 'πᾶς',
        'ἄνθρωπος': 'ἄνθρωπος',
        'ἀνθρώπου': 'ἄνθρωπος',
        'ἀνθρώπῳ': 'ἄνθρωπος',
        'ἄνθρωπον': 'ἄνθρωπος',
        'ἄνθρωποι': 'ἄνθρωπος',
        'ἀνθρώπων': 'ἄνθρωπος',
        'ἀνθρώποις': 'ἄνθρωπος',
        'ἀνθρώπους': 'ἄνθρωπος',
        'θεός': 'θεός',
        'θεοῦ': 'θεός',
        'θεῷ': 'θεός',
        'θεόν': 'θεός',
        'θεοί': 'θεός',
        'θεῶν': 'θεός',
        'θεοῖς': 'θεός',
        'θεούς': 'θεός',
        'λόγος': 'λόγος',
        'λόγου': 'λόγος',
        'λόγῳ': 'λόγος',
        'λόγον': 'λόγος',
        'λόγοι': 'λόγος',
        'λόγων': 'λόγος',
        'λόγοις': 'λόγος',
        'λόγους': 'λόγος',
        'πόλις': 'πόλις',
        'πόλεως': 'πόλις',
        'πόλει': 'πόλις',
        'πόλιν': 'πόλις',
        'πόλεις': 'πόλις',
        'πόλεων': 'πόλις',
        'πόλεσι': 'πόλις',
        'βασιλεύς': 'βασιλεύς',
        'βασιλέως': 'βασιλεύς',
        'βασιλεῖ': 'βασιλεύς',
        'βασιλέα': 'βασιλεύς',
        'βασιλεῖς': 'βασιλεύς',
        'βασιλέων': 'βασιλεύς',
        'βασιλεῦσι': 'βασιλεύς',
    }
    
    def __init__(self):
        self.custom_lemmas: Dict[str, str] = {}
        self.cache: Dict[str, str] = {}
    
    def lemmatize(self, form: str) -> str:
        if form in self.cache:
            return self.cache[form]
        
        normalized = self._normalize(form)
        
        if normalized in self.custom_lemmas:
            lemma = self.custom_lemmas[normalized]
        elif normalized in self.COMMON_LEMMAS:
            lemma = self.COMMON_LEMMAS[normalized]
        else:
            lemma = self._guess_lemma(normalized)
        
        self.cache[form] = lemma
        return lemma
    
    def _normalize(self, form: str) -> str:
        return form.strip()
    
    def _guess_lemma(self, form: str) -> str:
        verb_endings = [
            ('ομαι', 'ομαι'),
            ('εται', 'ομαι'),
            ('ονται', 'ομαι'),
            ('ει', 'ω'),
            ('εις', 'ω'),
            ('ουσι', 'ω'),
            ('ουσιν', 'ω'),
            ('ομεν', 'ω'),
            ('ετε', 'ω'),
            ('ων', 'ω'),
            ('ουσα', 'ω'),
            ('ον', 'ω'),
            ('σα', 'ω'),
            ('σας', 'ω'),
            ('σαν', 'ω'),
            ('σαντ', 'ω'),
        ]
        
        noun_endings = [
            ('ου', 'ος'),
            ('ῳ', 'ος'),
            ('ον', 'ος'),
            ('οι', 'ος'),
            ('ων', 'ος'),
            ('οις', 'ος'),
            ('ους', 'ος'),
            ('ης', 'η'),
            ('ῃ', 'η'),
            ('ην', 'η'),
            ('αι', 'η'),
            ('ας', 'α'),
            ('αν', 'α'),
            ('εως', 'ις'),
            ('ει', 'ις'),
            ('ιν', 'ις'),
            ('εις', 'ις'),
            ('εων', 'ις'),
            ('εσι', 'ις'),
        ]
        
        for ending, replacement in verb_endings:
            if form.endswith(ending):
                return form[:-len(ending)] + replacement
        
        for ending, replacement in noun_endings:
            if form.endswith(ending):
                return form[:-len(ending)] + replacement
        
        return form
    
    def add_lemma(self, form: str, lemma: str):
        self.custom_lemmas[form] = lemma
        if form in self.cache:
            del self.cache[form]
    
    def batch_lemmatize(self, forms: List[str]) -> List[str]:
        return [self.lemmatize(form) for form in forms]


class LatinLemmatizer:
    
    COMMON_LEMMAS = {
        'et': 'et',
        'in': 'in',
        'est': 'sum',
        'sunt': 'sum',
        'esse': 'sum',
        'erat': 'sum',
        'erant': 'sum',
        'fuit': 'sum',
        'non': 'non',
        'cum': 'cum',
        'ad': 'ad',
        'ex': 'ex',
        'de': 'de',
        'per': 'per',
        'pro': 'pro',
        'ab': 'ab',
        'sed': 'sed',
        'si': 'si',
        'ut': 'ut',
        'aut': 'aut',
        'vel': 'vel',
        'atque': 'atque',
        'ac': 'atque',
        'neque': 'neque',
        'nec': 'neque',
        'quod': 'qui',
        'qui': 'qui',
        'quae': 'qui',
        'quem': 'qui',
        'quam': 'qui',
        'quo': 'qui',
        'qua': 'qui',
        'quibus': 'qui',
        'quorum': 'qui',
        'quarum': 'qui',
        'hic': 'hic',
        'haec': 'hic',
        'hoc': 'hic',
        'huius': 'hic',
        'huic': 'hic',
        'hunc': 'hic',
        'hanc': 'hic',
        'hi': 'hic',
        'hae': 'hic',
        'hos': 'hic',
        'has': 'hic',
        'horum': 'hic',
        'harum': 'hic',
        'his': 'hic',
        'ille': 'ille',
        'illa': 'ille',
        'illud': 'ille',
        'illius': 'ille',
        'illi': 'ille',
        'illum': 'ille',
        'illam': 'ille',
        'illo': 'ille',
        'is': 'is',
        'ea': 'is',
        'id': 'is',
        'eius': 'is',
        'ei': 'is',
        'eum': 'is',
        'eam': 'is',
        'eo': 'is',
        'ii': 'is',
        'eae': 'is',
        'eos': 'is',
        'eas': 'is',
        'eorum': 'is',
        'earum': 'is',
        'iis': 'is',
        'eis': 'is',
        'omnis': 'omnis',
        'omne': 'omnis',
        'omnem': 'omnis',
        'omni': 'omnis',
        'omnes': 'omnis',
        'omnia': 'omnis',
        'omnium': 'omnis',
        'omnibus': 'omnis',
        'ipse': 'ipse',
        'ipsa': 'ipse',
        'ipsum': 'ipse',
        'ipsius': 'ipse',
        'ipsi': 'ipse',
        'ipso': 'ipse',
        'rex': 'rex',
        'regis': 'rex',
        'regi': 'rex',
        'regem': 'rex',
        'rege': 'rex',
        'reges': 'rex',
        'regum': 'rex',
        'regibus': 'rex',
        'deus': 'deus',
        'dei': 'deus',
        'deo': 'deus',
        'deum': 'deus',
        'di': 'deus',
        'dii': 'deus',
        'deorum': 'deus',
        'dis': 'deus',
        'diis': 'deus',
        'deos': 'deus',
        'homo': 'homo',
        'hominis': 'homo',
        'homini': 'homo',
        'hominem': 'homo',
        'homine': 'homo',
        'homines': 'homo',
        'hominum': 'homo',
        'hominibus': 'homo',
        'res': 'res',
        'rei': 'res',
        'rem': 'res',
        're': 'res',
        'rerum': 'res',
        'rebus': 'res',
    }
    
    def __init__(self):
        self.custom_lemmas: Dict[str, str] = {}
        self.cache: Dict[str, str] = {}
    
    def lemmatize(self, form: str) -> str:
        form_lower = form.lower()
        
        if form_lower in self.cache:
            return self.cache[form_lower]
        
        if form_lower in self.custom_lemmas:
            lemma = self.custom_lemmas[form_lower]
        elif form_lower in self.COMMON_LEMMAS:
            lemma = self.COMMON_LEMMAS[form_lower]
        else:
            lemma = self._guess_lemma(form_lower)
        
        self.cache[form_lower] = lemma
        return lemma
    
    def _guess_lemma(self, form: str) -> str:
        noun_endings = [
            ('arum', 'a'),
            ('orum', 'us'),
            ('ibus', 'us'),
            ('ae', 'a'),
            ('am', 'a'),
            ('as', 'a'),
            ('is', 'a'),
            ('i', 'us'),
            ('o', 'us'),
            ('um', 'us'),
            ('os', 'us'),
        ]
        
        verb_endings = [
            ('amus', 'are'),
            ('atis', 'are'),
            ('ant', 'are'),
            ('abam', 'are'),
            ('abas', 'are'),
            ('abat', 'are'),
            ('abamus', 'are'),
            ('abatis', 'are'),
            ('abant', 'are'),
            ('avi', 'are'),
            ('avit', 'are'),
            ('averunt', 'are'),
            ('emus', 'ere'),
            ('etis', 'ere'),
            ('ent', 'ere'),
            ('ebam', 'ere'),
            ('ebat', 'ere'),
            ('imus', 'ire'),
            ('itis', 'ire'),
            ('iunt', 'ire'),
            ('at', 'are'),
            ('et', 'ere'),
            ('it', 'ire'),
        ]
        
        for ending, replacement in verb_endings:
            if form.endswith(ending):
                stem = form[:-len(ending)]
                if len(stem) >= 2:
                    return stem + replacement
        
        for ending, replacement in noun_endings:
            if form.endswith(ending):
                stem = form[:-len(ending)]
                if len(stem) >= 2:
                    return stem + replacement
        
        return form
    
    def add_lemma(self, form: str, lemma: str):
        self.custom_lemmas[form.lower()] = lemma
        if form.lower() in self.cache:
            del self.cache[form.lower()]
    
    def batch_lemmatize(self, forms: List[str]) -> List[str]:
        return [self.lemmatize(form) for form in forms]


class MorphologicalAnalyzer:
    
    GREEK_POS_TAGS = {
        'N': 'noun',
        'V': 'verb',
        'A': 'adjective',
        'D': 'adverb',
        'P': 'preposition',
        'C': 'conjunction',
        'R': 'pronoun',
        'M': 'numeral',
        'I': 'interjection',
        'X': 'particle',
        'G': 'article',
    }
    
    LATIN_POS_TAGS = {
        'N': 'noun',
        'V': 'verb',
        'A': 'adjective',
        'D': 'adverb',
        'P': 'preposition',
        'C': 'conjunction',
        'R': 'pronoun',
        'M': 'numeral',
        'I': 'interjection',
    }
    
    def __init__(self, language: str = "grc"):
        self.language = language
        self.greek_lemmatizer = GreekLemmatizer()
        self.latin_lemmatizer = LatinLemmatizer()
    
    def analyze(self, form: str) -> MorphAnalysis:
        if self.language in ['grc', 'greek', 'ancient_greek']:
            return self._analyze_greek(form)
        elif self.language in ['lat', 'latin']:
            return self._analyze_latin(form)
        else:
            return MorphAnalysis(form=form, lemma=form, pos='X')
    
    def _analyze_greek(self, form: str) -> MorphAnalysis:
        lemma = self.greek_lemmatizer.lemmatize(form)
        
        pos = self._guess_greek_pos(form, lemma)
        
        analysis = MorphAnalysis(form=form, lemma=lemma, pos=pos)
        
        if pos == 'V':
            self._analyze_greek_verb(form, analysis)
        elif pos in ['N', 'A', 'R']:
            self._analyze_greek_nominal(form, analysis)
        
        return analysis
    
    def _guess_greek_pos(self, form: str, lemma: str) -> str:
        articles = {'ὁ', 'ἡ', 'τό', 'τοῦ', 'τῆς', 'τῷ', 'τήν', 'τόν', 'οἱ', 'αἱ', 'τά', 'τῶν', 'τοῖς', 'ταῖς', 'τούς', 'τάς'}
        if form in articles:
            return 'G'
        
        prepositions = {'ἐν', 'εἰς', 'ἐκ', 'ἐξ', 'ἀπό', 'πρός', 'ὑπό', 'περί', 'διά', 'κατά', 'μετά', 'παρά', 'ἐπί', 'ὑπέρ', 'ἀντί', 'σύν'}
        if form in prepositions:
            return 'P'
        
        conjunctions = {'καί', 'δέ', 'μέν', 'γάρ', 'ἀλλά', 'ἤ', 'τε', 'οὖν', 'ὅτι', 'ὡς', 'εἰ', 'ἐάν', 'ἵνα', 'ὅπως'}
        if form in conjunctions:
            return 'C'
        
        particles = {'οὐ', 'οὐκ', 'οὐχ', 'μή', 'ἄν', 'γε', 'δή', 'μέν', 'οὖν', 'ἄρα', 'τοι', 'που', 'πω', 'νῦν'}
        if form in particles:
            return 'X'
        
        verb_endings = ['ω', 'εις', 'ει', 'ομεν', 'ετε', 'ουσι', 'ουσιν', 'ομαι', 'εται', 'ονται', 'ειν', 'σθαι']
        for ending in verb_endings:
            if form.endswith(ending):
                return 'V'
        
        return 'N'
    
    def _analyze_greek_verb(self, form: str, analysis: MorphAnalysis):
        if form.endswith('ω'):
            analysis.person = '1'
            analysis.number = 'singular'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('εις'):
            analysis.person = '2'
            analysis.number = 'singular'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('ει'):
            analysis.person = '3'
            analysis.number = 'singular'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('ομεν'):
            analysis.person = '1'
            analysis.number = 'plural'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('ετε'):
            analysis.person = '2'
            analysis.number = 'plural'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('ουσι') or form.endswith('ουσιν'):
            analysis.person = '3'
            analysis.number = 'plural'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'active'
        elif form.endswith('ομαι'):
            analysis.person = '1'
            analysis.number = 'singular'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'middle'
        elif form.endswith('εται'):
            analysis.person = '3'
            analysis.number = 'singular'
            analysis.tense = 'present'
            analysis.mood = 'indicative'
            analysis.voice = 'middle'
        elif form.endswith('ειν'):
            analysis.tense = 'present'
            analysis.mood = 'infinitive'
            analysis.voice = 'active'
        elif form.endswith('σθαι'):
            analysis.tense = 'present'
            analysis.mood = 'infinitive'
            analysis.voice = 'middle'
    
    def _analyze_greek_nominal(self, form: str, analysis: MorphAnalysis):
        if form.endswith('ος'):
            analysis.gender = 'masculine'
            analysis.case = 'nominative'
            analysis.number = 'singular'
        elif form.endswith('ου'):
            analysis.gender = 'masculine'
            analysis.case = 'genitive'
            analysis.number = 'singular'
        elif form.endswith('ῳ'):
            analysis.gender = 'masculine'
            analysis.case = 'dative'
            analysis.number = 'singular'
        elif form.endswith('ον'):
            analysis.case = 'accusative'
            analysis.number = 'singular'
        elif form.endswith('οι'):
            analysis.gender = 'masculine'
            analysis.case = 'nominative'
            analysis.number = 'plural'
        elif form.endswith('ων'):
            analysis.case = 'genitive'
            analysis.number = 'plural'
        elif form.endswith('οις'):
            analysis.case = 'dative'
            analysis.number = 'plural'
        elif form.endswith('ους'):
            analysis.case = 'accusative'
            analysis.number = 'plural'
        elif form.endswith('η') or form.endswith('α'):
            analysis.gender = 'feminine'
            analysis.case = 'nominative'
            analysis.number = 'singular'
        elif form.endswith('ης') or form.endswith('ας'):
            analysis.gender = 'feminine'
            analysis.case = 'genitive'
            analysis.number = 'singular'
    
    def _analyze_latin(self, form: str) -> MorphAnalysis:
        lemma = self.latin_lemmatizer.lemmatize(form)
        
        pos = self._guess_latin_pos(form, lemma)
        
        analysis = MorphAnalysis(form=form, lemma=lemma, pos=pos)
        
        return analysis
    
    def _guess_latin_pos(self, form: str, lemma: str) -> str:
        prepositions = {'in', 'ad', 'ex', 'de', 'per', 'pro', 'ab', 'cum', 'sub', 'inter', 'ante', 'post', 'super', 'trans', 'contra', 'sine', 'propter'}
        if form.lower() in prepositions:
            return 'P'
        
        conjunctions = {'et', 'sed', 'aut', 'vel', 'atque', 'ac', 'neque', 'nec', 'si', 'ut', 'cum', 'dum', 'quod', 'quia', 'nam', 'enim', 'igitur', 'ergo', 'autem', 'tamen'}
        if form.lower() in conjunctions:
            return 'C'
        
        verb_endings = ['o', 'as', 'at', 'amus', 'atis', 'ant', 'eo', 'es', 'et', 'emus', 'etis', 'ent', 'io', 'is', 'it', 'imus', 'itis', 'iunt', 'or', 'aris', 'atur', 'amur', 'amini', 'antur', 'are', 'ere', 'ire', 'ari', 'eri', 'iri']
        for ending in verb_endings:
            if form.lower().endswith(ending):
                return 'V'
        
        return 'N'
    
    def batch_analyze(self, forms: List[str]) -> List[MorphAnalysis]:
        return [self.analyze(form) for form in forms]


class CorpusStatistics:
    
    def __init__(self):
        self.token_count = 0
        self.type_count = 0
        self.sentence_count = 0
        self.word_frequencies: Counter = Counter()
        self.lemma_frequencies: Counter = Counter()
        self.pos_distribution: Counter = Counter()
        self.bigram_frequencies: Counter = Counter()
        self.trigram_frequencies: Counter = Counter()
    
    def update(self, tokens: List[str], lemmas: Optional[List[str]] = None, pos_tags: Optional[List[str]] = None):
        self.token_count += len(tokens)
        self.word_frequencies.update(tokens)
        self.type_count = len(self.word_frequencies)
        
        if lemmas:
            self.lemma_frequencies.update(lemmas)
        
        if pos_tags:
            self.pos_distribution.update(pos_tags)
        
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            self.bigram_frequencies[bigram] += 1
        
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.trigram_frequencies[trigram] += 1
    
    def get_ttr(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.type_count / self.token_count
    
    def get_hapax_legomena(self) -> List[str]:
        return [word for word, count in self.word_frequencies.items() if count == 1]
    
    def get_dis_legomena(self) -> List[str]:
        return [word for word, count in self.word_frequencies.items() if count == 2]
    
    def get_top_words(self, n: int = 100) -> List[Tuple[str, int]]:
        return self.word_frequencies.most_common(n)
    
    def get_top_lemmas(self, n: int = 100) -> List[Tuple[str, int]]:
        return self.lemma_frequencies.most_common(n)
    
    def get_top_bigrams(self, n: int = 50) -> List[Tuple[Tuple[str, str], int]]:
        return self.bigram_frequencies.most_common(n)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'token_count': self.token_count,
            'type_count': self.type_count,
            'sentence_count': self.sentence_count,
            'ttr': self.get_ttr(),
            'hapax_count': len(self.get_hapax_legomena()),
            'dis_count': len(self.get_dis_legomena()),
            'pos_distribution': dict(self.pos_distribution),
            'top_words': self.get_top_words(20),
            'top_lemmas': self.get_top_lemmas(20),
        }


class ARCASToolkit:
    
    def __init__(self, language: str = "grc"):
        self.language = language
        self.greek_lemmatizer = GreekLemmatizer()
        self.latin_lemmatizer = LatinLemmatizer()
        self.morph_analyzer = MorphologicalAnalyzer(language)
        self.stats = CorpusStatistics()
    
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s\u0370-\u03FF\u1F00-\u1FFF]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        if self.language in ['grc', 'greek', 'ancient_greek']:
            return self.greek_lemmatizer.batch_lemmatize(tokens)
        elif self.language in ['lat', 'latin']:
            return self.latin_lemmatizer.batch_lemmatize(tokens)
        else:
            return tokens
    
    def analyze_morphology(self, tokens: List[str]) -> List[MorphAnalysis]:
        return self.morph_analyzer.batch_analyze(tokens)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        tokens = self.tokenize(text)
        lemmas = self.lemmatize(tokens)
        analyses = self.analyze_morphology(tokens)
        
        pos_tags = [a.pos for a in analyses]
        
        self.stats.update(tokens, lemmas, pos_tags)
        
        return {
            'tokens': tokens,
            'lemmas': lemmas,
            'analyses': [a.to_dict() for a in analyses],
            'token_count': len(tokens),
            'type_count': len(set(tokens)),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.stats.to_dict()
    
    def export_conllu(self, tokens: List[str], analyses: List[MorphAnalysis]) -> str:
        lines = []
        for i, (token, analysis) in enumerate(zip(tokens, analyses), 1):
            feats = []
            if analysis.case:
                feats.append(f"Case={analysis.case.capitalize()}")
            if analysis.gender:
                feats.append(f"Gender={analysis.gender.capitalize()}")
            if analysis.number:
                feats.append(f"Number={analysis.number.capitalize()}")
            if analysis.person:
                feats.append(f"Person={analysis.person}")
            if analysis.tense:
                feats.append(f"Tense={analysis.tense.capitalize()}")
            if analysis.mood:
                feats.append(f"Mood={analysis.mood.capitalize()}")
            if analysis.voice:
                feats.append(f"Voice={analysis.voice.capitalize()}")
            
            feat_str = '|'.join(feats) if feats else '_'
            
            line = f"{i}\t{token}\t{analysis.lemma}\t{analysis.pos}\t_\t{feat_str}\t_\t_\t_\t_"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def export_proiel_xml(self, tokens: List[str], analyses: List[MorphAnalysis], sentence_id: str = "s1") -> str:
        xml_lines = [f'<sentence id="{sentence_id}">']
        
        for i, (token, analysis) in enumerate(zip(tokens, analyses), 1):
            morph = analysis.to_proiel_string()
            xml_lines.append(
                f'  <token id="{sentence_id}_{i}" form="{token}" lemma="{analysis.lemma}" '
                f'part-of-speech="{analysis.pos}" morphology="{morph}"/>'
            )
        
        xml_lines.append('</sentence>')
        return '\n'.join(xml_lines)
