"""
HLP Valency Pattern Normalization - Normalize and Canonicalize Valency Patterns

This module provides utilities for normalizing valency patterns to
canonical forms, enabling comparison across different annotation
schemes and languages.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from enum import Enum

from hlp_valency.pattern_extractor import (
    ExtractedFrame, Argument, ArgumentType, FrameType
)
from hlp_core.models import (
    Case, ValencyFrame, ValencyPattern, Language
)

logger = logging.getLogger(__name__)


class NormalizationLevel(Enum):
    """Levels of pattern normalization"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    SEMANTIC = "semantic"


class CaseNormalization(Enum):
    """Case normalization strategies"""
    PRESERVE = "preserve"
    COLLAPSE_OBLIQUE = "collapse_oblique"
    ABSTRACT = "abstract"
    SEMANTIC = "semantic"


CASE_HIERARCHY = {
    Case.NOMINATIVE: 1,
    Case.ACCUSATIVE: 2,
    Case.GENITIVE: 3,
    Case.DATIVE: 4,
    Case.ABLATIVE: 5,
    Case.INSTRUMENTAL: 6,
    Case.LOCATIVE: 7,
    Case.VOCATIVE: 8,
}

OBLIQUE_CASES = {
    Case.GENITIVE, Case.DATIVE, Case.ABLATIVE,
    Case.INSTRUMENTAL, Case.LOCATIVE
}

ARGUMENT_HIERARCHY = {
    ArgumentType.SUBJECT: 1,
    ArgumentType.DIRECT_OBJECT: 2,
    ArgumentType.INDIRECT_OBJECT: 3,
    ArgumentType.OBLIQUE: 4,
    ArgumentType.COMPLEMENT: 5,
    ArgumentType.PREDICATIVE: 6,
    ArgumentType.ADVERBIAL: 7,
}

SEMANTIC_ROLE_MAPPING = {
    (ArgumentType.SUBJECT, Case.NOMINATIVE): "AGENT",
    (ArgumentType.SUBJECT, Case.DATIVE): "EXPERIENCER",
    (ArgumentType.SUBJECT, Case.GENITIVE): "POSSESSOR",
    (ArgumentType.DIRECT_OBJECT, Case.ACCUSATIVE): "PATIENT",
    (ArgumentType.DIRECT_OBJECT, Case.GENITIVE): "PARTITIVE",
    (ArgumentType.INDIRECT_OBJECT, Case.DATIVE): "RECIPIENT",
    (ArgumentType.OBLIQUE, Case.GENITIVE): "SOURCE",
    (ArgumentType.OBLIQUE, Case.DATIVE): "BENEFICIARY",
    (ArgumentType.OBLIQUE, Case.ABLATIVE): "SOURCE",
    (ArgumentType.OBLIQUE, Case.INSTRUMENTAL): "INSTRUMENT",
    (ArgumentType.OBLIQUE, Case.LOCATIVE): "LOCATION",
}


@dataclass
class NormalizationConfig:
    """Configuration for pattern normalization"""
    level: NormalizationLevel = NormalizationLevel.STANDARD
    
    case_normalization: CaseNormalization = CaseNormalization.PRESERVE
    
    normalize_lemmas: bool = True
    lowercase_lemmas: bool = True
    
    remove_diacritics: bool = False
    
    collapse_reflexives: bool = True
    
    merge_similar_patterns: bool = False
    similarity_threshold: float = 0.8
    
    include_prepositions: bool = True
    
    include_clausal_marker: bool = True
    
    sort_arguments: bool = True
    
    language: str = "grc"
    
    custom_lemma_mappings: Dict[str, str] = field(default_factory=dict)
    
    custom_case_mappings: Dict[Case, Case] = field(default_factory=dict)


@dataclass
class NormalizedPattern:
    """Represents a normalized valency pattern"""
    verb_lemma: str
    
    arguments: List[NormalizedArgument]
    
    pattern_string: str
    
    canonical_form: str
    
    frame_type: FrameType
    
    original_patterns: List[str] = field(default_factory=list)
    
    frequency: int = 1
    
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "verb_lemma": self.verb_lemma,
            "arguments": [a.to_dict() for a in self.arguments],
            "pattern_string": self.pattern_string,
            "canonical_form": self.canonical_form,
            "frame_type": self.frame_type.value,
            "original_patterns": self.original_patterns,
            "frequency": self.frequency,
            "confidence": self.confidence
        }


@dataclass
class NormalizedArgument:
    """Represents a normalized argument"""
    arg_type: ArgumentType
    
    case: Optional[Case] = None
    
    preposition: Optional[str] = None
    
    semantic_role: Optional[str] = None
    
    is_clausal: bool = False
    
    is_optional: bool = False
    
    def to_string(self) -> str:
        """Convert to string representation"""
        parts = []
        
        if self.semantic_role:
            parts.append(self.semantic_role)
        else:
            parts.append(self.arg_type.value.upper())
        
        if self.case:
            parts.append(f"[{self.case.value}]")
        
        if self.preposition:
            parts.append(f"({self.preposition})")
        
        if self.is_clausal:
            parts.append("{CL}")
        
        if self.is_optional:
            return f"({' '.join(parts)})"
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "arg_type": self.arg_type.value,
            "case": self.case.value if self.case else None,
            "preposition": self.preposition,
            "semantic_role": self.semantic_role,
            "is_clausal": self.is_clausal,
            "is_optional": self.is_optional
        }


class PatternNormalizer:
    """Normalizes valency patterns to canonical forms"""
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
    
    def normalize_frame(self, frame: ExtractedFrame) -> NormalizedPattern:
        """Normalize an extracted frame to canonical form"""
        verb_lemma = self._normalize_lemma(frame.verb_lemma)
        
        normalized_args = []
        for arg in frame.arguments:
            norm_arg = self._normalize_argument(arg)
            if norm_arg:
                normalized_args.append(norm_arg)
        
        if self.config.sort_arguments:
            normalized_args = self._sort_arguments(normalized_args)
        
        pattern_string = self._build_pattern_string(verb_lemma, normalized_args)
        canonical_form = self._build_canonical_form(verb_lemma, normalized_args)
        
        return NormalizedPattern(
            verb_lemma=verb_lemma,
            arguments=normalized_args,
            pattern_string=pattern_string,
            canonical_form=canonical_form,
            frame_type=frame.frame_type,
            original_patterns=[frame.get_pattern_string()],
            confidence=frame.confidence
        )
    
    def normalize_frames(self, frames: List[ExtractedFrame]) -> List[NormalizedPattern]:
        """Normalize multiple frames"""
        normalized = [self.normalize_frame(f) for f in frames]
        
        if self.config.merge_similar_patterns:
            normalized = self._merge_similar(normalized)
        
        return normalized
    
    def _normalize_lemma(self, lemma: str) -> str:
        """Normalize a lemma"""
        if not lemma:
            return ""
        
        if lemma in self.config.custom_lemma_mappings:
            lemma = self.config.custom_lemma_mappings[lemma]
        
        if self.config.lowercase_lemmas:
            lemma = lemma.lower()
        
        if self.config.remove_diacritics:
            lemma = self._remove_diacritics(lemma)
        
        return lemma
    
    def _normalize_argument(self, arg: Argument) -> Optional[NormalizedArgument]:
        """Normalize an argument"""
        arg_type = arg.arg_type
        
        if self.config.collapse_reflexives and arg.is_reflexive:
            return None
        
        case = arg.case
        if case and self.config.case_normalization != CaseNormalization.PRESERVE:
            case = self._normalize_case(case)
        
        preposition = None
        if self.config.include_prepositions and arg.preposition:
            preposition = self._normalize_lemma(arg.preposition)
        
        semantic_role = None
        if self.config.level == NormalizationLevel.SEMANTIC:
            semantic_role = self._infer_semantic_role(arg_type, case, preposition)
        
        return NormalizedArgument(
            arg_type=arg_type,
            case=case,
            preposition=preposition,
            semantic_role=semantic_role,
            is_clausal=arg.is_clausal and self.config.include_clausal_marker
        )
    
    def _normalize_case(self, case: Case) -> Optional[Case]:
        """Normalize case according to configuration"""
        if case in self.config.custom_case_mappings:
            return self.config.custom_case_mappings[case]
        
        if self.config.case_normalization == CaseNormalization.COLLAPSE_OBLIQUE:
            if case in OBLIQUE_CASES:
                return Case.GENITIVE
        
        elif self.config.case_normalization == CaseNormalization.ABSTRACT:
            if case == Case.NOMINATIVE:
                return Case.NOMINATIVE
            elif case == Case.ACCUSATIVE:
                return Case.ACCUSATIVE
            else:
                return Case.GENITIVE
        
        return case
    
    def _infer_semantic_role(
        self,
        arg_type: ArgumentType,
        case: Optional[Case],
        preposition: Optional[str]
    ) -> Optional[str]:
        """Infer semantic role from argument type and case"""
        if case:
            role = SEMANTIC_ROLE_MAPPING.get((arg_type, case))
            if role:
                return role
        
        if arg_type == ArgumentType.SUBJECT:
            return "AGENT"
        elif arg_type == ArgumentType.DIRECT_OBJECT:
            return "PATIENT"
        elif arg_type == ArgumentType.INDIRECT_OBJECT:
            return "RECIPIENT"
        elif arg_type == ArgumentType.COMPLEMENT:
            return "THEME"
        
        return None
    
    def _sort_arguments(self, arguments: List[NormalizedArgument]) -> List[NormalizedArgument]:
        """Sort arguments by hierarchy"""
        return sorted(
            arguments,
            key=lambda a: ARGUMENT_HIERARCHY.get(a.arg_type, 99)
        )
    
    def _build_pattern_string(
        self,
        verb_lemma: str,
        arguments: List[NormalizedArgument]
    ) -> str:
        """Build pattern string representation"""
        arg_strings = [arg.to_string() for arg in arguments]
        return f"{verb_lemma}({', '.join(arg_strings)})"
    
    def _build_canonical_form(
        self,
        verb_lemma: str,
        arguments: List[NormalizedArgument]
    ) -> str:
        """Build canonical form for comparison"""
        parts = [verb_lemma]
        
        for arg in arguments:
            part = arg.arg_type.value[0].upper()
            if arg.case:
                part += arg.case.value[0].upper()
            if arg.is_clausal:
                part += "c"
            parts.append(part)
        
        return "_".join(parts)
    
    def _merge_similar(self, patterns: List[NormalizedPattern]) -> List[NormalizedPattern]:
        """Merge similar patterns"""
        if not patterns:
            return []
        
        merged = {}
        
        for pattern in patterns:
            key = pattern.canonical_form
            
            if key in merged:
                merged[key].frequency += 1
                merged[key].original_patterns.extend(pattern.original_patterns)
            else:
                merged[key] = pattern
        
        return list(merged.values())
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text"""
        import unicodedata
        
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    def compute_similarity(self, p1: NormalizedPattern, p2: NormalizedPattern) -> float:
        """Compute similarity between two patterns"""
        if p1.verb_lemma != p2.verb_lemma:
            return 0.0
        
        args1 = set((a.arg_type, a.case) for a in p1.arguments)
        args2 = set((a.arg_type, a.case) for a in p2.arguments)
        
        intersection = len(args1 & args2)
        union = len(args1 | args2)
        
        if union == 0:
            return 1.0 if p1.verb_lemma == p2.verb_lemma else 0.0
        
        return intersection / union


def normalize_pattern(
    frame: ExtractedFrame,
    config: Optional[NormalizationConfig] = None
) -> NormalizedPattern:
    """Normalize a single frame"""
    normalizer = PatternNormalizer(config)
    return normalizer.normalize_frame(frame)


def normalize_frame(
    frame: ExtractedFrame,
    config: Optional[NormalizationConfig] = None
) -> NormalizedPattern:
    """Normalize a single frame (alias)"""
    return normalize_pattern(frame, config)


def canonicalize_arguments(
    arguments: List[Argument],
    config: Optional[NormalizationConfig] = None
) -> List[NormalizedArgument]:
    """Canonicalize a list of arguments"""
    normalizer = PatternNormalizer(config)
    normalized = []
    
    for arg in arguments:
        norm_arg = normalizer._normalize_argument(arg)
        if norm_arg:
            normalized.append(norm_arg)
    
    if config and config.sort_arguments:
        normalized = normalizer._sort_arguments(normalized)
    
    return normalized


def compute_pattern_similarity(
    p1: NormalizedPattern,
    p2: NormalizedPattern,
    config: Optional[NormalizationConfig] = None
) -> float:
    """Compute similarity between two patterns"""
    normalizer = PatternNormalizer(config)
    return normalizer.compute_similarity(p1, p2)
