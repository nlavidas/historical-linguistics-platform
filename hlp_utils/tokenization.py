"""
HLP Utils Tokenization - Text Tokenization Utilities

This module provides comprehensive tokenization support for
ancient Greek, Latin, and other historical languages.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Types of tokens"""
    WORD = "word"
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    WHITESPACE = "whitespace"
    SYMBOL = "symbol"
    UNKNOWN = "unknown"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    language: str = "grc"
    
    preserve_whitespace: bool = False
    
    split_punctuation: bool = True
    
    split_clitics: bool = True
    
    normalize_unicode: bool = True
    
    lowercase: bool = False
    
    sentence_boundary_chars: str = ".;:"
    
    word_boundary_pattern: str = r"\s+"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenSpan:
    """Represents a token with position information"""
    text: str
    
    token_type: TokenType
    
    start: int
    
    end: int
    
    normalized: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "type": self.token_type.value,
            "start": self.start,
            "end": self.end,
            "normalized": self.normalized
        }


GREEK_PUNCTUATION = {
    ".", ",", ";", ":", "·",
    "\u0387",
    "\u037E",
    "\u00B7",
    "(", ")", "[", "]", "{", "}",
    "\"", "'", "\u2019", "\u201C", "\u201D",
}

LATIN_PUNCTUATION = {
    ".", ",", ";", ":", "!", "?",
    "(", ")", "[", "]", "{", "}",
    "\"", "'", "-", "\u2014",
}

GREEK_SENTENCE_BOUNDARIES = {".", ";", ":"}
LATIN_SENTENCE_BOUNDARIES = {".", "!", "?"}

GREEK_CLITICS = {
    "τε", "δέ", "γε", "περ", "τοι", "μέν", "γάρ",
    "τ'", "δ'", "γ'",
}

LATIN_CLITICS = {
    "que", "ve", "ne", "cum",
}


class Tokenizer:
    """Tokenizer for ancient texts"""
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self._word_pattern = re.compile(
            self.config.word_boundary_pattern
        )
        
        if self.config.language in ["grc", "greek", "ancient_greek"]:
            punct_chars = "".join(re.escape(c) for c in GREEK_PUNCTUATION)
        else:
            punct_chars = "".join(re.escape(c) for c in LATIN_PUNCTUATION)
        
        self._punct_pattern = re.compile(f"([{punct_chars}])")
        
        self._number_pattern = re.compile(r"\d+")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        if self.config.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize("NFC", text)
        
        if self.config.split_punctuation:
            text = self._punct_pattern.sub(r" \1 ", text)
        
        tokens = self._word_pattern.split(text)
        
        tokens = [t for t in tokens if t.strip()]
        
        if self.config.split_clitics:
            tokens = self._split_clitics(tokens)
        
        if self.config.lowercase:
            tokens = [t.lower() for t in tokens]
        
        return tokens
    
    def tokenize_with_spans(self, text: str) -> List[TokenSpan]:
        """Tokenize text with position information"""
        if not text:
            return []
        
        spans = []
        current_pos = 0
        
        tokens = self.tokenize(text)
        
        for token in tokens:
            start = text.find(token, current_pos)
            if start == -1:
                start = current_pos
            
            end = start + len(token)
            
            token_type = self._classify_token(token)
            
            spans.append(TokenSpan(
                text=token,
                token_type=token_type,
                start=start,
                end=end
            ))
            
            current_pos = end
        
        return spans
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not text:
            return []
        
        if self.config.language in ["grc", "greek", "ancient_greek"]:
            boundaries = GREEK_SENTENCE_BOUNDARIES
        else:
            boundaries = LATIN_SENTENCE_BOUNDARIES
        
        pattern = f"([{''.join(re.escape(b) for b in boundaries)}])"
        
        parts = re.split(pattern, text)
        
        sentences = []
        current = ""
        
        for part in parts:
            if part in boundaries:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def _split_clitics(self, tokens: List[str]) -> List[str]:
        """Split clitics from tokens"""
        if self.config.language in ["grc", "greek", "ancient_greek"]:
            clitics = GREEK_CLITICS
        else:
            clitics = LATIN_CLITICS
        
        result = []
        
        for token in tokens:
            split = False
            
            for clitic in clitics:
                if token.endswith(clitic) and len(token) > len(clitic):
                    base = token[:-len(clitic)]
                    result.append(base)
                    result.append(clitic)
                    split = True
                    break
            
            if not split:
                result.append(token)
        
        return result
    
    def _classify_token(self, token: str) -> TokenType:
        """Classify a token by type"""
        if not token:
            return TokenType.UNKNOWN
        
        if token.isspace():
            return TokenType.WHITESPACE
        
        if self._number_pattern.fullmatch(token):
            return TokenType.NUMBER
        
        if len(token) == 1:
            if token in GREEK_PUNCTUATION or token in LATIN_PUNCTUATION:
                return TokenType.PUNCTUATION
        
        if token.isalpha():
            return TokenType.WORD
        
        return TokenType.UNKNOWN


def tokenize_text(
    text: str,
    language: str = "grc",
    split_punctuation: bool = True
) -> List[str]:
    """Tokenize text"""
    config = TokenizerConfig(
        language=language,
        split_punctuation=split_punctuation
    )
    tokenizer = Tokenizer(config)
    return tokenizer.tokenize(text)


def tokenize_greek(text: str) -> List[str]:
    """Tokenize Greek text"""
    return tokenize_text(text, language="grc")


def tokenize_latin(text: str) -> List[str]:
    """Tokenize Latin text"""
    return tokenize_text(text, language="la")


def sentence_split(text: str, language: str = "grc") -> List[str]:
    """Split text into sentences"""
    config = TokenizerConfig(language=language)
    tokenizer = Tokenizer(config)
    return tokenizer.sentence_tokenize(text)


def word_tokenize(text: str, language: str = "grc") -> List[str]:
    """Tokenize text into words only"""
    config = TokenizerConfig(
        language=language,
        split_punctuation=True
    )
    tokenizer = Tokenizer(config)
    tokens = tokenizer.tokenize(text)
    
    return [t for t in tokens if t.isalpha()]
