"""
HLP Semantics Information Structure - Topic/Focus Annotation

This module provides comprehensive support for information structure
annotation in ancient and historical texts, following PROIEL guidelines.

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


class InformationStatus(Enum):
    """Information status categories (PROIEL-style)"""
    OLD = "old"
    ACC_GEN = "acc_gen"
    ACC_SIT = "acc_sit"
    ACC_INF = "acc_inf"
    NEW = "new"
    NON_SPEC = "non_spec"
    QUANT = "quant"
    KIND = "kind"


class TopicType(Enum):
    """Types of topics"""
    ABOUTNESS_TOPIC = "aboutness_topic"
    CONTRASTIVE_TOPIC = "contrastive_topic"
    FRAME_SETTING_TOPIC = "frame_setting_topic"
    FAMILIAR_TOPIC = "familiar_topic"
    SHIFTED_TOPIC = "shifted_topic"
    RESUMED_TOPIC = "resumed_topic"
    HANGING_TOPIC = "hanging_topic"


class FocusType(Enum):
    """Types of focus"""
    NEW_INFORMATION_FOCUS = "new_information_focus"
    CONTRASTIVE_FOCUS = "contrastive_focus"
    EXHAUSTIVE_FOCUS = "exhaustive_focus"
    SELECTIVE_FOCUS = "selective_focus"
    CORRECTIVE_FOCUS = "corrective_focus"
    VERUM_FOCUS = "verum_focus"
    MIRATIVE_FOCUS = "mirative_focus"


class DiscourseRelation(Enum):
    """Discourse relations"""
    ELABORATION = "elaboration"
    CONTRAST = "contrast"
    CAUSE = "cause"
    RESULT = "result"
    CONDITION = "condition"
    CONCESSION = "concession"
    TEMPORAL = "temporal"
    BACKGROUND = "background"
    EXPLANATION = "explanation"
    CONTINUATION = "continuation"


@dataclass
class InformationUnit:
    """Represents an information structure unit"""
    tokens: List[Token]
    
    text: str
    
    info_status: Optional[InformationStatus] = None
    
    topic_type: Optional[TopicType] = None
    
    focus_type: Optional[FocusType] = None
    
    is_topic: bool = False
    
    is_focus: bool = False
    
    is_background: bool = False
    
    is_given: bool = False
    
    is_new: bool = False
    
    is_contrastive: bool = False
    
    position: str = "medial"
    
    confidence: float = 1.0
    
    source: str = "rule"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "token_ids": [t.id for t in self.tokens],
            "info_status": self.info_status.value if self.info_status else None,
            "topic_type": self.topic_type.value if self.topic_type else None,
            "focus_type": self.focus_type.value if self.focus_type else None,
            "is_topic": self.is_topic,
            "is_focus": self.is_focus,
            "is_background": self.is_background,
            "is_given": self.is_given,
            "is_new": self.is_new,
            "is_contrastive": self.is_contrastive,
            "position": self.position,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class ISAnnotation:
    """Information structure annotation for a sentence"""
    sentence_id: str
    
    units: List[InformationUnit]
    
    topic_units: List[InformationUnit] = field(default_factory=list)
    
    focus_units: List[InformationUnit] = field(default_factory=list)
    
    background_units: List[InformationUnit] = field(default_factory=list)
    
    discourse_relation: Optional[DiscourseRelation] = None
    
    sentence_type: str = "declarative"
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sentence_id": self.sentence_id,
            "units": [u.to_dict() for u in self.units],
            "topic_count": len(self.topic_units),
            "focus_count": len(self.focus_units),
            "background_count": len(self.background_units),
            "discourse_relation": self.discourse_relation.value if self.discourse_relation else None,
            "sentence_type": self.sentence_type
        }


@dataclass
class ISConfig:
    """Configuration for IS annotator"""
    use_word_order: bool = True
    
    use_morphology: bool = True
    
    use_syntax: bool = True
    
    use_discourse: bool = True
    
    use_ml_model: bool = False
    
    model_name: Optional[str] = None
    
    language: str = "grc"
    
    word_order: str = "free"
    
    topic_position: str = "initial"
    
    focus_position: str = "preverbal"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ISResult:
    """Result of IS annotation"""
    annotations: List[ISAnnotation]
    
    document_id: Optional[str] = None
    
    total_topics: int = 0
    
    total_foci: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "annotations": [a.to_dict() for a in self.annotations],
            "document_id": self.document_id,
            "total_topics": self.total_topics,
            "total_foci": self.total_foci,
            "error_count": len(self.errors)
        }


class ISAnnotator:
    """Annotator for information structure"""
    
    def __init__(self, config: Optional[ISConfig] = None):
        self.config = config or ISConfig()
        self._model = None
        self._discourse_context: List[Set[str]] = []
    
    def annotate_sentence(
        self,
        sentence: Sentence,
        previous_sentences: Optional[List[Sentence]] = None
    ) -> ISAnnotation:
        """Annotate a sentence with information structure"""
        if previous_sentences:
            self._update_discourse_context(previous_sentences)
        
        units = []
        
        verb_position = self._find_verb_position(sentence)
        
        for i, token in enumerate(sentence.tokens):
            unit = self._analyze_token(token, i, verb_position, sentence)
            if unit:
                units.append(unit)
        
        topic_units = [u for u in units if u.is_topic]
        focus_units = [u for u in units if u.is_focus]
        background_units = [u for u in units if u.is_background]
        
        sentence_type = self._determine_sentence_type(sentence)
        
        return ISAnnotation(
            sentence_id=sentence.id,
            units=units,
            topic_units=topic_units,
            focus_units=focus_units,
            background_units=background_units,
            sentence_type=sentence_type
        )
    
    def annotate_document(self, document: Document) -> ISResult:
        """Annotate all sentences in a document"""
        annotations = []
        total_topics = 0
        total_foci = 0
        
        self._discourse_context = []
        
        for i, sentence in enumerate(document.sentences):
            previous = document.sentences[:i] if i > 0 else None
            
            annotation = self.annotate_sentence(sentence, previous)
            annotations.append(annotation)
            
            total_topics += len(annotation.topic_units)
            total_foci += len(annotation.focus_units)
            
            self._add_to_discourse_context(sentence)
        
        return ISResult(
            annotations=annotations,
            document_id=document.id,
            total_topics=total_topics,
            total_foci=total_foci
        )
    
    def _analyze_token(
        self,
        token: Token,
        position: int,
        verb_position: int,
        sentence: Sentence
    ) -> Optional[InformationUnit]:
        """Analyze a token for information structure"""
        if token.pos in ["PUNCT", ".", ",", ";", ":"]:
            return None
        
        is_topic = False
        is_focus = False
        is_background = False
        is_given = False
        is_new = False
        is_contrastive = False
        
        topic_type = None
        focus_type = None
        info_status = None
        
        if self.config.use_word_order:
            pos_label = self._get_position_label(position, len(sentence.tokens))
            
            if pos_label == "initial" and self._is_nominal(token):
                is_topic = True
                topic_type = TopicType.ABOUTNESS_TOPIC
            
            if verb_position > 0 and position == verb_position - 1:
                if self._is_nominal(token):
                    is_focus = True
                    focus_type = FocusType.NEW_INFORMATION_FOCUS
        
        if self.config.use_morphology and token.morphology:
            if token.morphology.case == Case.NOMINATIVE:
                if not is_topic and not is_focus:
                    is_topic = True
                    topic_type = TopicType.ABOUTNESS_TOPIC
        
        if self.config.use_syntax and token.syntax:
            deprel = token.syntax.deprel
            
            if deprel in ["nsubj", "sub"]:
                is_topic = True
                topic_type = TopicType.ABOUTNESS_TOPIC
            
            elif deprel in ["obj", "obl"] and position < verb_position:
                is_focus = True
                focus_type = FocusType.NEW_INFORMATION_FOCUS
        
        if self.config.use_discourse:
            lemma = token.lemma or token.form
            
            if self._is_in_discourse_context(lemma):
                is_given = True
                info_status = InformationStatus.OLD
            else:
                is_new = True
                info_status = InformationStatus.NEW
        
        if not is_topic and not is_focus:
            is_background = True
        
        return InformationUnit(
            tokens=[token],
            text=token.form,
            info_status=info_status,
            topic_type=topic_type,
            focus_type=focus_type,
            is_topic=is_topic,
            is_focus=is_focus,
            is_background=is_background,
            is_given=is_given,
            is_new=is_new,
            is_contrastive=is_contrastive,
            position=self._get_position_label(position, len(sentence.tokens))
        )
    
    def _find_verb_position(self, sentence: Sentence) -> int:
        """Find position of main verb"""
        verb_pos = {"V", "VERB", "AUX"}
        
        for i, token in enumerate(sentence.tokens):
            if token.pos in verb_pos:
                return i
        
        return len(sentence.tokens) // 2
    
    def _get_position_label(self, position: int, total: int) -> str:
        """Get position label"""
        if position == 0:
            return "initial"
        elif position == total - 1:
            return "final"
        elif position < total // 3:
            return "early"
        elif position > 2 * total // 3:
            return "late"
        else:
            return "medial"
    
    def _is_nominal(self, token: Token) -> bool:
        """Check if token is nominal"""
        nominal_pos = {"N", "NOUN", "PROPN", "PRON", "DET", "Ne", "Nb", "Pc", "Pd", "Pi", "Pp", "Pr", "Ps", "Pt", "Px"}
        return token.pos in nominal_pos
    
    def _determine_sentence_type(self, sentence: Sentence) -> str:
        """Determine sentence type"""
        text = sentence.text.strip()
        
        if text.endswith("?") or text.endswith(";"):
            return "interrogative"
        elif text.endswith("!"):
            return "exclamative"
        
        for token in sentence.tokens:
            if token.morphology and token.morphology.mood:
                mood = token.morphology.mood
                if mood in ["IMP", "imperative"]:
                    return "imperative"
                elif mood in ["OPT", "optative"]:
                    return "optative"
        
        return "declarative"
    
    def _update_discourse_context(self, sentences: List[Sentence]):
        """Update discourse context from previous sentences"""
        for sentence in sentences[-5:]:
            self._add_to_discourse_context(sentence)
    
    def _add_to_discourse_context(self, sentence: Sentence):
        """Add sentence to discourse context"""
        lemmas = set()
        
        for token in sentence.tokens:
            if self._is_nominal(token):
                lemma = token.lemma or token.form
                lemmas.add(lemma.lower())
        
        self._discourse_context.append(lemmas)
        
        if len(self._discourse_context) > 10:
            self._discourse_context = self._discourse_context[-10:]
    
    def _is_in_discourse_context(self, lemma: str) -> bool:
        """Check if lemma is in discourse context"""
        lemma_lower = lemma.lower()
        
        for context in self._discourse_context:
            if lemma_lower in context:
                return True
        
        return False


def annotate_information_structure(
    sentence: Sentence,
    config: Optional[ISConfig] = None
) -> ISAnnotation:
    """Annotate information structure in a sentence"""
    annotator = ISAnnotator(config)
    return annotator.annotate_sentence(sentence)


def identify_topic(
    sentence: Sentence,
    config: Optional[ISConfig] = None
) -> List[InformationUnit]:
    """Identify topic(s) in a sentence"""
    annotation = annotate_information_structure(sentence, config)
    return annotation.topic_units


def identify_focus(
    sentence: Sentence,
    config: Optional[ISConfig] = None
) -> List[InformationUnit]:
    """Identify focus in a sentence"""
    annotation = annotate_information_structure(sentence, config)
    return annotation.focus_units
