"""
HLP Semantics SRL - Semantic Role Labeling

This module provides comprehensive support for semantic role labeling
in ancient and historical texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum

from hlp_core.models import (
    Language, Sentence, Token, Document,
    MorphologyAnnotation, SyntaxAnnotation, Case
)

logger = logging.getLogger(__name__)


class SemanticRole(Enum):
    """Semantic roles based on PropBank/VerbNet/FrameNet"""
    AGENT = "Agent"
    PATIENT = "Patient"
    THEME = "Theme"
    EXPERIENCER = "Experiencer"
    BENEFICIARY = "Beneficiary"
    RECIPIENT = "Recipient"
    INSTRUMENT = "Instrument"
    LOCATION = "Location"
    SOURCE = "Source"
    GOAL = "Goal"
    PATH = "Path"
    MANNER = "Manner"
    CAUSE = "Cause"
    PURPOSE = "Purpose"
    TIME = "Time"
    EXTENT = "Extent"
    RESULT = "Result"
    STIMULUS = "Stimulus"
    ATTRIBUTE = "Attribute"
    COMITATIVE = "Comitative"
    POSSESSOR = "Possessor"
    TOPIC = "Topic"
    CONTENT = "Content"
    PREDICATE = "Predicate"
    
    ARG0 = "ARG0"
    ARG1 = "ARG1"
    ARG2 = "ARG2"
    ARG3 = "ARG3"
    ARG4 = "ARG4"
    ARGM_LOC = "ARGM-LOC"
    ARGM_TMP = "ARGM-TMP"
    ARGM_MNR = "ARGM-MNR"
    ARGM_CAU = "ARGM-CAU"
    ARGM_PRP = "ARGM-PRP"
    ARGM_DIR = "ARGM-DIR"
    ARGM_EXT = "ARGM-EXT"
    ARGM_DIS = "ARGM-DIS"
    ARGM_ADV = "ARGM-ADV"
    ARGM_MOD = "ARGM-MOD"
    ARGM_NEG = "ARGM-NEG"


CASE_TO_ROLE_MAPPING = {
    Case.NOMINATIVE: [SemanticRole.AGENT, SemanticRole.EXPERIENCER, SemanticRole.THEME],
    Case.ACCUSATIVE: [SemanticRole.PATIENT, SemanticRole.THEME, SemanticRole.GOAL],
    Case.GENITIVE: [SemanticRole.SOURCE, SemanticRole.POSSESSOR, SemanticRole.CAUSE],
    Case.DATIVE: [SemanticRole.RECIPIENT, SemanticRole.BENEFICIARY, SemanticRole.EXPERIENCER],
    Case.ABLATIVE: [SemanticRole.SOURCE, SemanticRole.INSTRUMENT, SemanticRole.CAUSE],
    Case.INSTRUMENTAL: [SemanticRole.INSTRUMENT, SemanticRole.MANNER, SemanticRole.COMITATIVE],
    Case.LOCATIVE: [SemanticRole.LOCATION, SemanticRole.TIME],
}

DEPREL_TO_ROLE_MAPPING = {
    "nsubj": SemanticRole.AGENT,
    "nsubj:pass": SemanticRole.PATIENT,
    "obj": SemanticRole.PATIENT,
    "iobj": SemanticRole.RECIPIENT,
    "obl": SemanticRole.LOCATION,
    "obl:agent": SemanticRole.AGENT,
    "obl:loc": SemanticRole.LOCATION,
    "obl:tmod": SemanticRole.TIME,
    "advmod": SemanticRole.MANNER,
    "xcomp": SemanticRole.PREDICATE,
    "ccomp": SemanticRole.CONTENT,
}

PROIEL_RELATION_TO_ROLE = {
    "sub": SemanticRole.AGENT,
    "obj": SemanticRole.PATIENT,
    "obl": SemanticRole.LOCATION,
    "dat": SemanticRole.RECIPIENT,
    "ag": SemanticRole.AGENT,
    "atr": SemanticRole.ATTRIBUTE,
    "adv": SemanticRole.MANNER,
    "xobj": SemanticRole.PREDICATE,
    "comp": SemanticRole.CONTENT,
}


@dataclass
class SemanticArgument:
    """Represents a semantic argument"""
    role: SemanticRole
    
    tokens: List[Token]
    
    head_token: Optional[Token] = None
    
    text: str = ""
    
    confidence: float = 1.0
    
    source: str = "rule"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role.value,
            "text": self.text,
            "token_ids": [t.id for t in self.tokens],
            "head_token_id": self.head_token.id if self.head_token else None,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class SemanticFrame:
    """Represents a semantic frame (predicate + arguments)"""
    predicate: Token
    
    predicate_lemma: str
    
    arguments: List[SemanticArgument]
    
    frame_type: Optional[str] = None
    
    sentence_id: Optional[str] = None
    
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_argument(self, role: SemanticRole) -> Optional[SemanticArgument]:
        """Get argument by role"""
        for arg in self.arguments:
            if arg.role == role:
                return arg
        return None
    
    def has_role(self, role: SemanticRole) -> bool:
        """Check if frame has a specific role"""
        return any(arg.role == role for arg in self.arguments)
    
    def get_roles(self) -> List[SemanticRole]:
        """Get all roles in frame"""
        return [arg.role for arg in self.arguments]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "predicate_id": self.predicate.id,
            "predicate_lemma": self.predicate_lemma,
            "predicate_form": self.predicate.form,
            "frame_type": self.frame_type,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "sentence_id": self.sentence_id,
            "confidence": self.confidence
        }


@dataclass
class SRLConfig:
    """Configuration for SRL annotator"""
    use_case_heuristics: bool = True
    
    use_deprel_mapping: bool = True
    
    use_proiel_relations: bool = True
    
    use_ml_model: bool = False
    
    model_name: Optional[str] = None
    
    min_confidence: float = 0.5
    
    include_adjuncts: bool = True
    
    language: str = "grc"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SRLResult:
    """Result of SRL annotation"""
    frames: List[SemanticFrame]
    
    sentence_id: Optional[str] = None
    
    total_predicates: int = 0
    
    total_arguments: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "frames": [f.to_dict() for f in self.frames],
            "sentence_id": self.sentence_id,
            "total_predicates": self.total_predicates,
            "total_arguments": self.total_arguments,
            "error_count": len(self.errors)
        }


class SRLAnnotator:
    """Annotator for semantic role labeling"""
    
    def __init__(self, config: Optional[SRLConfig] = None):
        self.config = config or SRLConfig()
        self._model = None
    
    def annotate_sentence(self, sentence: Sentence) -> SRLResult:
        """Annotate a sentence with semantic roles"""
        frames = []
        errors = []
        
        predicates = self._find_predicates(sentence)
        
        for predicate in predicates:
            try:
                frame = self._build_frame(predicate, sentence)
                if frame:
                    frames.append(frame)
            except Exception as e:
                errors.append(f"Error processing predicate {predicate.form}: {str(e)}")
        
        total_arguments = sum(len(f.arguments) for f in frames)
        
        return SRLResult(
            frames=frames,
            sentence_id=sentence.id,
            total_predicates=len(predicates),
            total_arguments=total_arguments,
            errors=errors
        )
    
    def annotate_document(self, document: Document) -> List[SRLResult]:
        """Annotate all sentences in a document"""
        results = []
        
        for sentence in document.sentences:
            result = self.annotate_sentence(sentence)
            results.append(result)
        
        return results
    
    def _find_predicates(self, sentence: Sentence) -> List[Token]:
        """Find predicates (verbs) in sentence"""
        predicates = []
        
        verb_pos = {"V", "VERB", "AUX"}
        
        for token in sentence.tokens:
            if token.pos in verb_pos or (token.xpos and token.xpos.startswith("V")):
                predicates.append(token)
        
        return predicates
    
    def _build_frame(self, predicate: Token, sentence: Sentence) -> Optional[SemanticFrame]:
        """Build semantic frame for a predicate"""
        arguments = []
        
        token_map = {t.id: t for t in sentence.tokens}
        
        for token in sentence.tokens:
            if token.id == predicate.id:
                continue
            
            if token.syntax and token.syntax.head == predicate.id:
                arg = self._classify_argument(token, predicate, sentence)
                if arg:
                    arguments.append(arg)
        
        if not arguments:
            for token in sentence.tokens:
                if token.id == predicate.id:
                    continue
                
                arg = self._classify_by_case(token, predicate)
                if arg:
                    arguments.append(arg)
        
        return SemanticFrame(
            predicate=predicate,
            predicate_lemma=predicate.lemma or predicate.form,
            arguments=arguments,
            sentence_id=sentence.id
        )
    
    def _classify_argument(
        self,
        token: Token,
        predicate: Token,
        sentence: Sentence
    ) -> Optional[SemanticArgument]:
        """Classify an argument based on syntactic relation"""
        if not token.syntax:
            return None
        
        deprel = token.syntax.deprel
        
        role = None
        
        if self.config.use_deprel_mapping and deprel in DEPREL_TO_ROLE_MAPPING:
            role = DEPREL_TO_ROLE_MAPPING[deprel]
        
        elif self.config.use_proiel_relations and deprel in PROIEL_RELATION_TO_ROLE:
            role = PROIEL_RELATION_TO_ROLE[deprel]
        
        if role is None and self.config.use_case_heuristics:
            role = self._infer_role_from_case(token, deprel)
        
        if role is None:
            return None
        
        tokens = self._get_subtree(token, sentence)
        text = " ".join(t.form for t in tokens)
        
        return SemanticArgument(
            role=role,
            tokens=tokens,
            head_token=token,
            text=text,
            source="syntax"
        )
    
    def _classify_by_case(
        self,
        token: Token,
        predicate: Token
    ) -> Optional[SemanticArgument]:
        """Classify argument by case alone"""
        if not self.config.use_case_heuristics:
            return None
        
        if not token.morphology or not token.morphology.case:
            return None
        
        case = token.morphology.case
        
        if case not in CASE_TO_ROLE_MAPPING:
            return None
        
        possible_roles = CASE_TO_ROLE_MAPPING[case]
        role = possible_roles[0]
        
        return SemanticArgument(
            role=role,
            tokens=[token],
            head_token=token,
            text=token.form,
            confidence=0.6,
            source="case_heuristic"
        )
    
    def _infer_role_from_case(
        self,
        token: Token,
        deprel: str
    ) -> Optional[SemanticRole]:
        """Infer semantic role from case"""
        if not token.morphology or not token.morphology.case:
            return None
        
        case = token.morphology.case
        
        if case == Case.NOMINATIVE:
            if deprel in ["nsubj:pass", "sub:pass"]:
                return SemanticRole.PATIENT
            return SemanticRole.AGENT
        
        elif case == Case.ACCUSATIVE:
            return SemanticRole.PATIENT
        
        elif case == Case.DATIVE:
            return SemanticRole.RECIPIENT
        
        elif case == Case.GENITIVE:
            return SemanticRole.SOURCE
        
        elif case == Case.ABLATIVE:
            return SemanticRole.SOURCE
        
        elif case == Case.INSTRUMENTAL:
            return SemanticRole.INSTRUMENT
        
        elif case == Case.LOCATIVE:
            return SemanticRole.LOCATION
        
        return None
    
    def _get_subtree(self, head: Token, sentence: Sentence) -> List[Token]:
        """Get all tokens in subtree rooted at head"""
        tokens = [head]
        
        for token in sentence.tokens:
            if token.syntax and token.syntax.head == head.id:
                tokens.extend(self._get_subtree(token, sentence))
        
        tokens.sort(key=lambda t: int(t.id) if t.id.isdigit() else 0)
        
        return tokens


def annotate_semantic_roles(
    sentence: Sentence,
    config: Optional[SRLConfig] = None
) -> SRLResult:
    """Annotate semantic roles in a sentence"""
    annotator = SRLAnnotator(config)
    return annotator.annotate_sentence(sentence)


def extract_predicate_arguments(
    sentence: Sentence,
    config: Optional[SRLConfig] = None
) -> List[SemanticFrame]:
    """Extract predicate-argument structures"""
    result = annotate_semantic_roles(sentence, config)
    return result.frames
