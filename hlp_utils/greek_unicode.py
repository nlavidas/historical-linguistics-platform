"""
HLP Utils Greek Unicode - Greek Unicode Normalization

This module provides comprehensive support for Greek Unicode
normalization, including accent handling, breathing marks,
and Beta Code conversion.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import unicodedata
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AccentType(Enum):
    """Types of Greek accents"""
    ACUTE = "acute"
    GRAVE = "grave"
    CIRCUMFLEX = "circumflex"
    NONE = "none"


class BreathingType(Enum):
    """Types of Greek breathing marks"""
    SMOOTH = "smooth"
    ROUGH = "rough"
    NONE = "none"


GREEK_LOWER_START = 0x0370
GREEK_LOWER_END = 0x03FF
GREEK_EXTENDED_START = 0x1F00
GREEK_EXTENDED_END = 0x1FFF

COMBINING_ACUTE = "\u0301"
COMBINING_GRAVE = "\u0300"
COMBINING_CIRCUMFLEX = "\u0342"
COMBINING_SMOOTH = "\u0313"
COMBINING_ROUGH = "\u0314"
COMBINING_IOTA_SUBSCRIPT = "\u0345"
COMBINING_DIAERESIS = "\u0308"
COMBINING_MACRON = "\u0304"
COMBINING_BREVE = "\u0306"

ACCENT_CHARS = {
    COMBINING_ACUTE,
    COMBINING_GRAVE,
    COMBINING_CIRCUMFLEX,
    "\u00B4",
    "\u0384",
    "\u1FFD",
    "\u1FEF",
    "\u1FC0",
    "\u1FC1",
}

BREATHING_CHARS = {
    COMBINING_SMOOTH,
    COMBINING_ROUGH,
    "\u1FBD",
    "\u1FBF",
    "\u1FFE",
}

BETA_CODE_MAP = {
    "a": "\u03B1", "b": "\u03B2", "g": "\u03B3", "d": "\u03B4",
    "e": "\u03B5", "z": "\u03B6", "h": "\u03B7", "q": "\u03B8",
    "i": "\u03B9", "k": "\u03BA", "l": "\u03BB", "m": "\u03BC",
    "n": "\u03BD", "c": "\u03BE", "o": "\u03BF", "p": "\u03C0",
    "r": "\u03C1", "s": "\u03C3", "t": "\u03C4", "u": "\u03C5",
    "f": "\u03C6", "x": "\u03C7", "y": "\u03C8", "w": "\u03C9",
    "A": "\u0391", "B": "\u0392", "G": "\u0393", "D": "\u0394",
    "E": "\u0395", "Z": "\u0396", "H": "\u0397", "Q": "\u0398",
    "I": "\u0399", "K": "\u039A", "L": "\u039B", "M": "\u039C",
    "N": "\u039D", "C": "\u039E", "O": "\u039F", "P": "\u03A0",
    "R": "\u03A1", "S": "\u03A3", "T": "\u03A4", "U": "\u03A5",
    "F": "\u03A6", "X": "\u03A7", "Y": "\u03A8", "W": "\u03A9",
}

BETA_CODE_REVERSE = {v: k for k, v in BETA_CODE_MAP.items()}

BETA_CODE_DIACRITICS = {
    "/": COMBINING_ACUTE,
    "\\": COMBINING_GRAVE,
    "=": COMBINING_CIRCUMFLEX,
    ")": COMBINING_SMOOTH,
    "(": COMBINING_ROUGH,
    "|": COMBINING_IOTA_SUBSCRIPT,
    "+": COMBINING_DIAERESIS,
}

FINAL_SIGMA = "\u03C2"
MEDIAL_SIGMA = "\u03C3"
LUNATE_SIGMA = "\u03F2"


@dataclass
class NormalizationConfig:
    """Configuration for Greek normalization"""
    normalize_unicode: bool = True
    
    strip_accents: bool = False
    
    strip_breathing: bool = False
    
    normalize_sigma: bool = True
    
    normalize_iota_subscript: bool = False
    
    lowercase: bool = False
    
    preserve_punctuation: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class GreekNormalizer:
    """Normalizer for Greek text"""
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
    
    def normalize(self, text: str) -> str:
        """Normalize Greek text"""
        if not text:
            return text
        
        result = text
        
        if self.config.normalize_unicode:
            result = unicodedata.normalize("NFC", result)
        
        if self.config.strip_accents:
            result = strip_accents(result)
        
        if self.config.strip_breathing:
            result = strip_breathing(result)
        
        if self.config.normalize_sigma:
            result = normalize_sigma(result)
        
        if self.config.normalize_iota_subscript:
            result = self._normalize_iota_subscript(result)
        
        if self.config.lowercase:
            result = result.lower()
        
        return result
    
    def _normalize_iota_subscript(self, text: str) -> str:
        """Normalize iota subscript to iota adscript"""
        result = []
        
        for char in text:
            decomposed = unicodedata.normalize("NFD", char)
            
            if COMBINING_IOTA_SUBSCRIPT in decomposed:
                decomposed = decomposed.replace(COMBINING_IOTA_SUBSCRIPT, "")
                result.append(unicodedata.normalize("NFC", decomposed))
                result.append("\u03B9")
            else:
                result.append(char)
        
        return "".join(result)


def normalize_greek(
    text: str,
    strip_accents_flag: bool = False,
    strip_breathing_flag: bool = False,
    normalize_sigma_flag: bool = True
) -> str:
    """Normalize Greek text"""
    config = NormalizationConfig(
        strip_accents=strip_accents_flag,
        strip_breathing=strip_breathing_flag,
        normalize_sigma=normalize_sigma_flag
    )
    normalizer = GreekNormalizer(config)
    return normalizer.normalize(text)


def strip_accents(text: str) -> str:
    """Remove accents from Greek text"""
    if not text:
        return text
    
    decomposed = unicodedata.normalize("NFD", text)
    
    result = []
    for char in decomposed:
        if char not in ACCENT_CHARS and unicodedata.category(char) != "Mn":
            result.append(char)
        elif char in BREATHING_CHARS or char == COMBINING_IOTA_SUBSCRIPT:
            result.append(char)
    
    return unicodedata.normalize("NFC", "".join(result))


def strip_breathing(text: str) -> str:
    """Remove breathing marks from Greek text"""
    if not text:
        return text
    
    decomposed = unicodedata.normalize("NFD", text)
    
    result = []
    for char in decomposed:
        if char not in BREATHING_CHARS:
            result.append(char)
    
    return unicodedata.normalize("NFC", "".join(result))


def normalize_sigma(text: str) -> str:
    """Normalize sigma forms (final vs medial)"""
    if not text:
        return text
    
    result = []
    chars = list(text)
    
    for i, char in enumerate(chars):
        if char in (MEDIAL_SIGMA, FINAL_SIGMA, LUNATE_SIGMA):
            is_final = True
            
            for j in range(i + 1, len(chars)):
                next_char = chars[j]
                if is_greek(next_char):
                    is_final = False
                    break
                elif next_char.isalpha():
                    break
            
            result.append(FINAL_SIGMA if is_final else MEDIAL_SIGMA)
        else:
            result.append(char)
    
    return "".join(result)


def to_beta_code(text: str) -> str:
    """Convert Greek Unicode to Beta Code"""
    if not text:
        return text
    
    decomposed = unicodedata.normalize("NFD", text)
    
    result = []
    current_base = ""
    current_diacritics = []
    
    for char in decomposed:
        if char in BETA_CODE_REVERSE:
            if current_base:
                result.append(current_base)
                result.extend(current_diacritics)
            current_base = BETA_CODE_REVERSE[char]
            current_diacritics = []
        
        elif char == COMBINING_ACUTE:
            current_diacritics.append("/")
        elif char == COMBINING_GRAVE:
            current_diacritics.append("\\")
        elif char == COMBINING_CIRCUMFLEX:
            current_diacritics.append("=")
        elif char == COMBINING_SMOOTH:
            current_diacritics.append(")")
        elif char == COMBINING_ROUGH:
            current_diacritics.append("(")
        elif char == COMBINING_IOTA_SUBSCRIPT:
            current_diacritics.append("|")
        elif char == COMBINING_DIAERESIS:
            current_diacritics.append("+")
        
        else:
            if current_base:
                result.append(current_base)
                result.extend(current_diacritics)
                current_base = ""
                current_diacritics = []
            result.append(char)
    
    if current_base:
        result.append(current_base)
        result.extend(current_diacritics)
    
    return "".join(result)


def from_beta_code(text: str) -> str:
    """Convert Beta Code to Greek Unicode"""
    if not text:
        return text
    
    result = []
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if char in BETA_CODE_MAP:
            base = BETA_CODE_MAP[char]
            diacritics = []
            
            j = i + 1
            while j < len(text) and text[j] in BETA_CODE_DIACRITICS:
                diacritics.append(BETA_CODE_DIACRITICS[text[j]])
                j += 1
            
            combined = base + "".join(diacritics)
            result.append(unicodedata.normalize("NFC", combined))
            i = j
        
        else:
            result.append(char)
            i += 1
    
    return "".join(result)


def is_greek(char: str) -> bool:
    """Check if a character is Greek"""
    if not char:
        return False
    
    code = ord(char[0])
    
    if GREEK_LOWER_START <= code <= GREEK_LOWER_END:
        return True
    
    if GREEK_EXTENDED_START <= code <= GREEK_EXTENDED_END:
        return True
    
    return False


def get_accent_type(char: str) -> AccentType:
    """Get the accent type of a Greek character"""
    if not char:
        return AccentType.NONE
    
    decomposed = unicodedata.normalize("NFD", char)
    
    if COMBINING_ACUTE in decomposed:
        return AccentType.ACUTE
    elif COMBINING_GRAVE in decomposed:
        return AccentType.GRAVE
    elif COMBINING_CIRCUMFLEX in decomposed:
        return AccentType.CIRCUMFLEX
    
    return AccentType.NONE


def get_breathing_type(char: str) -> BreathingType:
    """Get the breathing type of a Greek character"""
    if not char:
        return BreathingType.NONE
    
    decomposed = unicodedata.normalize("NFD", char)
    
    if COMBINING_SMOOTH in decomposed:
        return BreathingType.SMOOTH
    elif COMBINING_ROUGH in decomposed:
        return BreathingType.ROUGH
    
    return BreathingType.NONE


def has_iota_subscript(char: str) -> bool:
    """Check if a character has iota subscript"""
    if not char:
        return False
    
    decomposed = unicodedata.normalize("NFD", char)
    return COMBINING_IOTA_SUBSCRIPT in decomposed


def decompose_greek_char(char: str) -> Dict[str, Any]:
    """Decompose a Greek character into its components"""
    if not char:
        return {"base": "", "accents": [], "breathing": None, "iota_subscript": False}
    
    decomposed = unicodedata.normalize("NFD", char)
    
    base = ""
    accents = []
    breathing = None
    iota_subscript = False
    
    for c in decomposed:
        if is_greek(c) and unicodedata.category(c) != "Mn":
            base = c
        elif c == COMBINING_ACUTE:
            accents.append("acute")
        elif c == COMBINING_GRAVE:
            accents.append("grave")
        elif c == COMBINING_CIRCUMFLEX:
            accents.append("circumflex")
        elif c == COMBINING_SMOOTH:
            breathing = "smooth"
        elif c == COMBINING_ROUGH:
            breathing = "rough"
        elif c == COMBINING_IOTA_SUBSCRIPT:
            iota_subscript = True
    
    return {
        "base": base,
        "accents": accents,
        "breathing": breathing,
        "iota_subscript": iota_subscript
    }
