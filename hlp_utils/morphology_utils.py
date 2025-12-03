"""
HLP Utils Morphology - Morphology Utilities

This module provides utilities for parsing and formatting
morphological annotations in various formats.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


PROIEL_POS_MAP = {
    "A-": "ADJ",
    "C-": "CCONJ",
    "Df": "ADV",
    "Dq": "ADV",
    "Du": "ADV",
    "F-": "X",
    "G-": "SCONJ",
    "I-": "INTJ",
    "Ma": "NUM",
    "Mo": "NUM",
    "N-": "NOUN",
    "Nb": "NOUN",
    "Ne": "PROPN",
    "Pc": "PRON",
    "Pd": "PRON",
    "Pi": "PRON",
    "Pk": "PRON",
    "Pp": "PRON",
    "Pr": "PRON",
    "Ps": "PRON",
    "Pt": "PRON",
    "Px": "PRON",
    "Py": "PRON",
    "R-": "ADP",
    "S-": "ADP",
    "V-": "VERB",
    "X-": "X",
}

PROIEL_PERSON_MAP = {
    "1": "1",
    "2": "2",
    "3": "3",
    "-": None,
}

PROIEL_NUMBER_MAP = {
    "s": "Sing",
    "d": "Dual",
    "p": "Plur",
    "-": None,
}

PROIEL_TENSE_MAP = {
    "p": "Pres",
    "i": "Imp",
    "r": "Perf",
    "l": "Pqp",
    "t": "FutPerf",
    "f": "Fut",
    "a": "Aor",
    "-": None,
}

PROIEL_MOOD_MAP = {
    "i": "Ind",
    "s": "Sub",
    "o": "Opt",
    "m": "Imp",
    "n": "Inf",
    "p": "Part",
    "d": "Ger",
    "g": "Gdv",
    "u": "Sup",
    "-": None,
}

PROIEL_VOICE_MAP = {
    "a": "Act",
    "m": "Mid",
    "p": "Pass",
    "e": "Mid",
    "-": None,
}

PROIEL_GENDER_MAP = {
    "m": "Masc",
    "f": "Fem",
    "n": "Neut",
    "o": "Masc,Fem",
    "p": "Masc,Neut",
    "r": "Fem,Neut",
    "q": "Masc,Fem,Neut",
    "-": None,
}

PROIEL_CASE_MAP = {
    "n": "Nom",
    "g": "Gen",
    "d": "Dat",
    "a": "Acc",
    "v": "Voc",
    "b": "Abl",
    "l": "Loc",
    "i": "Ins",
    "-": None,
}

PROIEL_DEGREE_MAP = {
    "p": "Pos",
    "c": "Cmp",
    "s": "Sup",
    "-": None,
}


@dataclass
class MorphologyFeatures:
    """Morphological features"""
    person: Optional[str] = None
    number: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    voice: Optional[str] = None
    gender: Optional[str] = None
    case: Optional[str] = None
    degree: Optional[str] = None
    
    aspect: Optional[str] = None
    definiteness: Optional[str] = None
    polarity: Optional[str] = None
    
    additional: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        result = {}
        
        if self.person:
            result["Person"] = self.person
        if self.number:
            result["Number"] = self.number
        if self.tense:
            result["Tense"] = self.tense
        if self.mood:
            result["Mood"] = self.mood
        if self.voice:
            result["Voice"] = self.voice
        if self.gender:
            result["Gender"] = self.gender
        if self.case:
            result["Case"] = self.case
        if self.degree:
            result["Degree"] = self.degree
        if self.aspect:
            result["Aspect"] = self.aspect
        if self.definiteness:
            result["Definite"] = self.definiteness
        if self.polarity:
            result["Polarity"] = self.polarity
        
        result.update(self.additional)
        
        return result
    
    def to_ud_string(self) -> str:
        """Convert to UD feature string"""
        features = self.to_dict()
        
        if not features:
            return "_"
        
        return "|".join(f"{k}={v}" for k, v in sorted(features.items()))


class MorphologyParser:
    """Parser for morphological annotations"""
    
    def __init__(self):
        pass
    
    def parse_proiel(self, morph_string: str) -> MorphologyFeatures:
        """Parse PROIEL morphology string"""
        features = MorphologyFeatures()
        
        if not morph_string or morph_string == "-" * 8:
            return features
        
        morph_string = morph_string.ljust(8, "-")
        
        features.person = PROIEL_PERSON_MAP.get(morph_string[0])
        features.number = PROIEL_NUMBER_MAP.get(morph_string[1])
        features.tense = PROIEL_TENSE_MAP.get(morph_string[2])
        features.mood = PROIEL_MOOD_MAP.get(morph_string[3])
        features.voice = PROIEL_VOICE_MAP.get(morph_string[4])
        features.gender = PROIEL_GENDER_MAP.get(morph_string[5])
        features.case = PROIEL_CASE_MAP.get(morph_string[6])
        features.degree = PROIEL_DEGREE_MAP.get(morph_string[7])
        
        return features
    
    def parse_ud(self, feature_string: str) -> MorphologyFeatures:
        """Parse UD feature string"""
        features = MorphologyFeatures()
        
        if not feature_string or feature_string == "_":
            return features
        
        for pair in feature_string.split("|"):
            if "=" not in pair:
                continue
            
            key, value = pair.split("=", 1)
            
            if key == "Person":
                features.person = value
            elif key == "Number":
                features.number = value
            elif key == "Tense":
                features.tense = value
            elif key == "Mood":
                features.mood = value
            elif key == "Voice":
                features.voice = value
            elif key == "Gender":
                features.gender = value
            elif key == "Case":
                features.case = value
            elif key == "Degree":
                features.degree = value
            elif key == "Aspect":
                features.aspect = value
            elif key == "Definite":
                features.definiteness = value
            elif key == "Polarity":
                features.polarity = value
            else:
                features.additional[key] = value
        
        return features


class MorphologyFormatter:
    """Formatter for morphological annotations"""
    
    def __init__(self):
        self._proiel_person_reverse = {v: k for k, v in PROIEL_PERSON_MAP.items() if v}
        self._proiel_number_reverse = {v: k for k, v in PROIEL_NUMBER_MAP.items() if v}
        self._proiel_tense_reverse = {v: k for k, v in PROIEL_TENSE_MAP.items() if v}
        self._proiel_mood_reverse = {v: k for k, v in PROIEL_MOOD_MAP.items() if v}
        self._proiel_voice_reverse = {v: k for k, v in PROIEL_VOICE_MAP.items() if v}
        self._proiel_gender_reverse = {v: k for k, v in PROIEL_GENDER_MAP.items() if v}
        self._proiel_case_reverse = {v: k for k, v in PROIEL_CASE_MAP.items() if v}
        self._proiel_degree_reverse = {v: k for k, v in PROIEL_DEGREE_MAP.items() if v}
    
    def format_proiel(self, features: MorphologyFeatures) -> str:
        """Format features as PROIEL string"""
        result = []
        
        result.append(self._proiel_person_reverse.get(features.person, "-"))
        result.append(self._proiel_number_reverse.get(features.number, "-"))
        result.append(self._proiel_tense_reverse.get(features.tense, "-"))
        result.append(self._proiel_mood_reverse.get(features.mood, "-"))
        result.append(self._proiel_voice_reverse.get(features.voice, "-"))
        result.append(self._proiel_gender_reverse.get(features.gender, "-"))
        result.append(self._proiel_case_reverse.get(features.case, "-"))
        result.append(self._proiel_degree_reverse.get(features.degree, "-"))
        
        return "".join(result)
    
    def format_ud(self, features: MorphologyFeatures) -> str:
        """Format features as UD string"""
        return features.to_ud_string()


def parse_proiel_morphology(morph_string: str) -> Dict[str, str]:
    """Parse PROIEL morphology string"""
    parser = MorphologyParser()
    features = parser.parse_proiel(morph_string)
    return features.to_dict()


def parse_ud_features(feature_string: str) -> Dict[str, str]:
    """Parse UD feature string"""
    parser = MorphologyParser()
    features = parser.parse_ud(feature_string)
    return features.to_dict()


def format_proiel_morphology(features: Dict[str, str]) -> str:
    """Format features as PROIEL string"""
    morph = MorphologyFeatures(
        person=features.get("Person"),
        number=features.get("Number"),
        tense=features.get("Tense"),
        mood=features.get("Mood"),
        voice=features.get("Voice"),
        gender=features.get("Gender"),
        case=features.get("Case"),
        degree=features.get("Degree")
    )
    formatter = MorphologyFormatter()
    return formatter.format_proiel(morph)


def format_ud_features(features: Dict[str, str]) -> str:
    """Format features as UD string"""
    if not features:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(features.items()))
