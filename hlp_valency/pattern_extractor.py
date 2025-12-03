"""
HLP Valency Pattern Extractor - Extract Valency Frames from Annotated Text

This module provides comprehensive valency frame extraction from
syntactically annotated corpora, supporting both Universal Dependencies
and PROIEL annotation schemes.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from enum import Enum

from hlp_core.models import (
    Corpus, Document, Sentence, Token,
    MorphologicalFeatures, SyntacticRelation,
    PartOfSpeech, DependencyRelation, Case, Number, Gender,
    ValencyFrame, ValencyPattern,
    Language, Period
)

logger = logging.getLogger(__name__)


class ArgumentType(Enum):
    """Types of verbal arguments"""
    SUBJECT = "subject"
    DIRECT_OBJECT = "direct_object"
    INDIRECT_OBJECT = "indirect_object"
    OBLIQUE = "oblique"
    COMPLEMENT = "complement"
    ADVERBIAL = "adverbial"
    PREDICATIVE = "predicative"
    AGENT = "agent"
    PATIENT = "patient"
    EXPERIENCER = "experiencer"
    BENEFICIARY = "beneficiary"
    INSTRUMENT = "instrument"
    LOCATION = "location"
    SOURCE = "source"
    GOAL = "goal"
    MANNER = "manner"
    CAUSE = "cause"
    PURPOSE = "purpose"


class FrameType(Enum):
    """Types of valency frames"""
    INTRANSITIVE = "intransitive"
    TRANSITIVE = "transitive"
    DITRANSITIVE = "ditransitive"
    COPULAR = "copular"
    IMPERSONAL = "impersonal"
    REFLEXIVE = "reflexive"
    PASSIVE = "passive"
    CAUSATIVE = "causative"
    COMPLEX = "complex"


SUBJECT_RELATIONS = {
    DependencyRelation.NSUBJ,
    DependencyRelation.NSUBJ_PASS,
    DependencyRelation.CSUBJ,
    DependencyRelation.CSUBJ_PASS,
    DependencyRelation.SUB,
    DependencyRelation.XSUB,
}

OBJECT_RELATIONS = {
    DependencyRelation.OBJ,
    DependencyRelation.IOBJ,
    DependencyRelation.OBJ_PROIEL,
    DependencyRelation.XOBJ,
}

OBLIQUE_RELATIONS = {
    DependencyRelation.OBL,
    DependencyRelation.OBL_PROIEL,
    DependencyRelation.NARG,
}

COMPLEMENT_RELATIONS = {
    DependencyRelation.CCOMP,
    DependencyRelation.XCOMP,
    DependencyRelation.COMP,
}

ADVERBIAL_RELATIONS = {
    DependencyRelation.ADVMOD,
    DependencyRelation.ADVCL,
    DependencyRelation.ADV,
    DependencyRelation.XADV,
}

PROIEL_ARGUMENT_RELATIONS = {
    "SUB": ArgumentType.SUBJECT,
    "OBJ": ArgumentType.DIRECT_OBJECT,
    "OBL": ArgumentType.OBLIQUE,
    "NARG": ArgumentType.OBLIQUE,
    "XOBJ": ArgumentType.COMPLEMENT,
    "COMP": ArgumentType.COMPLEMENT,
    "ADV": ArgumentType.ADVERBIAL,
    "XADV": ArgumentType.ADVERBIAL,
    "AG": ArgumentType.AGENT,
}


@dataclass
class Argument:
    """Represents a verbal argument"""
    arg_type: ArgumentType
    token_ids: List[int]
    head_token_id: int
    
    case: Optional[Case] = None
    preposition: Optional[str] = None
    
    phrase_type: Optional[str] = None
    
    semantic_role: Optional[str] = None
    
    is_clausal: bool = False
    is_pronominal: bool = False
    is_reflexive: bool = False
    is_null: bool = False
    
    text: Optional[str] = None
    lemma: Optional[str] = None
    
    def to_pattern_string(self) -> str:
        """Convert to pattern string representation"""
        parts = [self.arg_type.value]
        
        if self.case:
            parts.append(f"[{self.case.value}]")
        
        if self.preposition:
            parts.append(f"({self.preposition})")
        
        if self.is_clausal:
            parts.append("{clause}")
        
        return "".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "arg_type": self.arg_type.value,
            "token_ids": self.token_ids,
            "head_token_id": self.head_token_id,
            "case": self.case.value if self.case else None,
            "preposition": self.preposition,
            "phrase_type": self.phrase_type,
            "semantic_role": self.semantic_role,
            "is_clausal": self.is_clausal,
            "is_pronominal": self.is_pronominal,
            "is_reflexive": self.is_reflexive,
            "is_null": self.is_null,
            "text": self.text,
            "lemma": self.lemma
        }


@dataclass
class ExtractedFrame:
    """Represents an extracted valency frame"""
    verb_token_id: int
    verb_lemma: str
    verb_form: str
    
    arguments: List[Argument]
    
    frame_type: FrameType = FrameType.INTRANSITIVE
    
    voice: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    
    is_auxiliary: bool = False
    is_copular: bool = False
    is_modal: bool = False
    
    auxiliaries: List[int] = field(default_factory=list)
    
    sentence_id: Optional[str] = None
    document_id: Optional[str] = None
    
    source_text: Optional[str] = None
    
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_pattern_string(self) -> str:
        """Get canonical pattern string"""
        arg_strings = sorted([arg.to_pattern_string() for arg in self.arguments])
        return f"{self.verb_lemma}({', '.join(arg_strings)})"
    
    def get_argument_structure(self) -> str:
        """Get argument structure string"""
        args = []
        for arg in self.arguments:
            if arg.arg_type == ArgumentType.SUBJECT:
                args.append("S")
            elif arg.arg_type == ArgumentType.DIRECT_OBJECT:
                args.append("O")
            elif arg.arg_type == ArgumentType.INDIRECT_OBJECT:
                args.append("IO")
            elif arg.arg_type == ArgumentType.OBLIQUE:
                args.append("OBL")
            elif arg.arg_type == ArgumentType.COMPLEMENT:
                args.append("COMP")
        return "-".join(sorted(args))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "verb_token_id": self.verb_token_id,
            "verb_lemma": self.verb_lemma,
            "verb_form": self.verb_form,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "frame_type": self.frame_type.value,
            "voice": self.voice,
            "tense": self.tense,
            "mood": self.mood,
            "is_auxiliary": self.is_auxiliary,
            "is_copular": self.is_copular,
            "is_modal": self.is_modal,
            "sentence_id": self.sentence_id,
            "document_id": self.document_id,
            "pattern_string": self.get_pattern_string(),
            "argument_structure": self.get_argument_structure(),
            "confidence": self.confidence
        }


@dataclass
class ExtractionConfig:
    """Configuration for valency extraction"""
    include_auxiliaries: bool = False
    include_copulars: bool = True
    include_modals: bool = True
    
    extract_null_arguments: bool = False
    
    include_adjuncts: bool = False
    
    use_proiel_relations: bool = True
    use_ud_relations: bool = True
    
    min_confidence: float = 0.0
    
    language: str = "grc"
    
    auxiliary_lemmas: Set[str] = field(default_factory=lambda: {
        "sum", "esse", "fui",
        "habeo", "habere",
        "possum", "posse",
        "volo", "velle",
        "debeo", "debere",
        "εἰμί", "ἔχω", "γίγνομαι",
    })
    
    copular_lemmas: Set[str] = field(default_factory=lambda: {
        "sum", "esse",
        "εἰμί", "γίγνομαι", "ὑπάρχω",
    })
    
    modal_lemmas: Set[str] = field(default_factory=lambda: {
        "possum", "posse",
        "volo", "velle",
        "debeo", "debere",
        "δύναμαι", "βούλομαι", "δεῖ", "χρή",
    })
    
    reflexive_pronouns: Set[str] = field(default_factory=lambda: {
        "se", "sui", "sibi",
        "ἑαυτοῦ", "ἑαυτόν", "ἑαυτῷ",
        "αὑτοῦ", "αὑτόν", "αὑτῷ",
    })


@dataclass
class ExtractionResult:
    """Result of valency extraction"""
    frames: List[ExtractedFrame]
    
    total_verbs: int = 0
    total_frames: int = 0
    
    frame_type_counts: Dict[str, int] = field(default_factory=dict)
    argument_type_counts: Dict[str, int] = field(default_factory=dict)
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_verbs": self.total_verbs,
            "total_frames": self.total_frames,
            "frame_type_counts": self.frame_type_counts,
            "argument_type_counts": self.argument_type_counts,
            "frames": [f.to_dict() for f in self.frames],
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms
        }


class ValencyExtractor:
    """Extracts valency frames from annotated text"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
    
    def extract_from_corpus(self, corpus: Corpus) -> ExtractionResult:
        """Extract valency frames from entire corpus"""
        import time
        start_time = time.time()
        
        all_frames = []
        total_verbs = 0
        frame_type_counts = defaultdict(int)
        argument_type_counts = defaultdict(int)
        errors = []
        warnings = []
        
        for document in corpus.documents:
            doc_result = self.extract_from_document(document)
            all_frames.extend(doc_result.frames)
            total_verbs += doc_result.total_verbs
            
            for ft, count in doc_result.frame_type_counts.items():
                frame_type_counts[ft] += count
            for at, count in doc_result.argument_type_counts.items():
                argument_type_counts[at] += count
            
            errors.extend(doc_result.errors)
            warnings.extend(doc_result.warnings)
        
        return ExtractionResult(
            frames=all_frames,
            total_verbs=total_verbs,
            total_frames=len(all_frames),
            frame_type_counts=dict(frame_type_counts),
            argument_type_counts=dict(argument_type_counts),
            errors=errors,
            warnings=warnings,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def extract_from_document(self, document: Document) -> ExtractionResult:
        """Extract valency frames from document"""
        all_frames = []
        total_verbs = 0
        frame_type_counts = defaultdict(int)
        argument_type_counts = defaultdict(int)
        errors = []
        
        for sentence in document.sentences:
            try:
                sent_result = self.extract_from_sentence(sentence)
                
                for frame in sent_result.frames:
                    frame.document_id = document.id
                
                all_frames.extend(sent_result.frames)
                total_verbs += sent_result.total_verbs
                
                for ft, count in sent_result.frame_type_counts.items():
                    frame_type_counts[ft] += count
                for at, count in sent_result.argument_type_counts.items():
                    argument_type_counts[at] += count
                    
            except Exception as e:
                errors.append(f"Error in sentence {sentence.id}: {str(e)}")
        
        return ExtractionResult(
            frames=all_frames,
            total_verbs=total_verbs,
            total_frames=len(all_frames),
            frame_type_counts=dict(frame_type_counts),
            argument_type_counts=dict(argument_type_counts),
            errors=errors
        )
    
    def extract_from_sentence(self, sentence: Sentence) -> ExtractionResult:
        """Extract valency frames from sentence"""
        frames = []
        total_verbs = 0
        frame_type_counts = defaultdict(int)
        argument_type_counts = defaultdict(int)
        
        token_map = {token.id: token for token in sentence.tokens}
        
        verbs = self._find_verbs(sentence)
        total_verbs = len(verbs)
        
        for verb_token in verbs:
            if self._should_skip_verb(verb_token):
                continue
            
            frame = self._extract_frame(verb_token, sentence, token_map)
            
            if frame:
                frame.sentence_id = sentence.id
                frame.source_text = sentence.text
                frames.append(frame)
                
                frame_type_counts[frame.frame_type.value] += 1
                for arg in frame.arguments:
                    argument_type_counts[arg.arg_type.value] += 1
        
        return ExtractionResult(
            frames=frames,
            total_verbs=total_verbs,
            total_frames=len(frames),
            frame_type_counts=dict(frame_type_counts),
            argument_type_counts=dict(argument_type_counts)
        )
    
    def _find_verbs(self, sentence: Sentence) -> List[Token]:
        """Find all verbs in sentence"""
        verbs = []
        
        for token in sentence.tokens:
            if token.morphology and token.morphology.pos == PartOfSpeech.VERB:
                verbs.append(token)
            elif token.morphology and token.morphology.pos == PartOfSpeech.AUX:
                if self.config.include_auxiliaries:
                    verbs.append(token)
        
        return verbs
    
    def _should_skip_verb(self, token: Token) -> bool:
        """Check if verb should be skipped"""
        lemma = token.lemma or token.form
        
        if not self.config.include_auxiliaries:
            if lemma in self.config.auxiliary_lemmas:
                return True
        
        if not self.config.include_copulars:
            if lemma in self.config.copular_lemmas:
                return True
        
        if not self.config.include_modals:
            if lemma in self.config.modal_lemmas:
                return True
        
        return False
    
    def _extract_frame(
        self,
        verb_token: Token,
        sentence: Sentence,
        token_map: Dict[int, Token]
    ) -> Optional[ExtractedFrame]:
        """Extract valency frame for a verb"""
        arguments = []
        
        dependents = self._get_dependents(verb_token.id, sentence)
        
        for dep_token in dependents:
            argument = self._classify_argument(dep_token, verb_token, sentence, token_map)
            if argument:
                arguments.append(argument)
        
        frame_type = self._determine_frame_type(arguments, verb_token)
        
        voice = None
        tense = None
        mood = None
        
        if verb_token.morphology:
            if verb_token.morphology.voice:
                voice = verb_token.morphology.voice.value
            if verb_token.morphology.tense:
                tense = verb_token.morphology.tense.value
            if verb_token.morphology.mood:
                mood = verb_token.morphology.mood.value
        
        lemma = verb_token.lemma or verb_token.form
        
        is_copular = lemma in self.config.copular_lemmas
        is_modal = lemma in self.config.modal_lemmas
        is_auxiliary = lemma in self.config.auxiliary_lemmas
        
        return ExtractedFrame(
            verb_token_id=verb_token.id,
            verb_lemma=lemma,
            verb_form=verb_token.form,
            arguments=arguments,
            frame_type=frame_type,
            voice=voice,
            tense=tense,
            mood=mood,
            is_auxiliary=is_auxiliary,
            is_copular=is_copular,
            is_modal=is_modal
        )
    
    def _get_dependents(self, head_id: int, sentence: Sentence) -> List[Token]:
        """Get all tokens that depend on the given head"""
        dependents = []
        
        for token in sentence.tokens:
            if token.syntax and token.syntax.head_id == head_id:
                dependents.append(token)
        
        return dependents
    
    def _classify_argument(
        self,
        dep_token: Token,
        verb_token: Token,
        sentence: Sentence,
        token_map: Dict[int, Token]
    ) -> Optional[Argument]:
        """Classify a dependent as an argument type"""
        if not dep_token.syntax:
            return None
        
        relation = dep_token.syntax.relation
        proiel_relation = dep_token.syntax.proiel_relation
        
        arg_type = None
        
        if self.config.use_proiel_relations and proiel_relation:
            arg_type = PROIEL_ARGUMENT_RELATIONS.get(proiel_relation)
        
        if arg_type is None and self.config.use_ud_relations and relation:
            if relation in SUBJECT_RELATIONS:
                arg_type = ArgumentType.SUBJECT
            elif relation in OBJECT_RELATIONS:
                if relation == DependencyRelation.IOBJ:
                    arg_type = ArgumentType.INDIRECT_OBJECT
                else:
                    arg_type = ArgumentType.DIRECT_OBJECT
            elif relation in OBLIQUE_RELATIONS:
                arg_type = ArgumentType.OBLIQUE
            elif relation in COMPLEMENT_RELATIONS:
                arg_type = ArgumentType.COMPLEMENT
            elif relation in ADVERBIAL_RELATIONS:
                if self.config.include_adjuncts:
                    arg_type = ArgumentType.ADVERBIAL
        
        if arg_type is None:
            return None
        
        case = None
        if dep_token.morphology and dep_token.morphology.case:
            case = dep_token.morphology.case
        
        preposition = self._find_preposition(dep_token, sentence, token_map)
        
        is_clausal = relation in COMPLEMENT_RELATIONS
        
        is_pronominal = dep_token.morphology and dep_token.morphology.pos == PartOfSpeech.PRON
        
        lemma = dep_token.lemma or dep_token.form
        is_reflexive = lemma in self.config.reflexive_pronouns
        
        subtree_ids = self._get_subtree(dep_token.id, sentence)
        
        return Argument(
            arg_type=arg_type,
            token_ids=subtree_ids,
            head_token_id=dep_token.id,
            case=case,
            preposition=preposition,
            is_clausal=is_clausal,
            is_pronominal=is_pronominal,
            is_reflexive=is_reflexive,
            text=dep_token.form,
            lemma=lemma
        )
    
    def _find_preposition(
        self,
        token: Token,
        sentence: Sentence,
        token_map: Dict[int, Token]
    ) -> Optional[str]:
        """Find preposition governing this token"""
        for t in sentence.tokens:
            if t.syntax and t.syntax.head_id == token.id:
                if t.syntax.relation == DependencyRelation.CASE:
                    return t.lemma or t.form
                if t.morphology and t.morphology.pos == PartOfSpeech.ADP:
                    return t.lemma or t.form
        
        return None
    
    def _get_subtree(self, head_id: int, sentence: Sentence) -> List[int]:
        """Get all token IDs in subtree rooted at head"""
        subtree = [head_id]
        
        for token in sentence.tokens:
            if token.syntax and token.syntax.head_id == head_id:
                subtree.extend(self._get_subtree(token.id, sentence))
        
        return sorted(subtree)
    
    def _determine_frame_type(
        self,
        arguments: List[Argument],
        verb_token: Token
    ) -> FrameType:
        """Determine the frame type based on arguments"""
        has_subject = any(a.arg_type == ArgumentType.SUBJECT for a in arguments)
        has_direct_obj = any(a.arg_type == ArgumentType.DIRECT_OBJECT for a in arguments)
        has_indirect_obj = any(a.arg_type == ArgumentType.INDIRECT_OBJECT for a in arguments)
        has_complement = any(a.arg_type == ArgumentType.COMPLEMENT for a in arguments)
        has_oblique = any(a.arg_type == ArgumentType.OBLIQUE for a in arguments)
        
        lemma = verb_token.lemma or verb_token.form
        if lemma in self.config.copular_lemmas:
            return FrameType.COPULAR
        
        voice = None
        if verb_token.morphology and verb_token.morphology.voice:
            voice = verb_token.morphology.voice.value
        
        if voice == "passive":
            return FrameType.PASSIVE
        
        has_reflexive = any(a.is_reflexive for a in arguments)
        if has_reflexive:
            return FrameType.REFLEXIVE
        
        if not has_subject:
            return FrameType.IMPERSONAL
        
        if has_direct_obj and has_indirect_obj:
            return FrameType.DITRANSITIVE
        
        if has_direct_obj or has_complement:
            return FrameType.TRANSITIVE
        
        if has_oblique and not has_direct_obj:
            return FrameType.COMPLEX
        
        return FrameType.INTRANSITIVE


def extract_valency_frames(
    corpus: Corpus,
    config: Optional[ExtractionConfig] = None
) -> ExtractionResult:
    """Extract valency frames from corpus"""
    extractor = ValencyExtractor(config)
    return extractor.extract_from_corpus(corpus)


def extract_from_sentence(
    sentence: Sentence,
    config: Optional[ExtractionConfig] = None
) -> ExtractionResult:
    """Extract valency frames from sentence"""
    extractor = ValencyExtractor(config)
    return extractor.extract_from_sentence(sentence)


def extract_from_document(
    document: Document,
    config: Optional[ExtractionConfig] = None
) -> ExtractionResult:
    """Extract valency frames from document"""
    extractor = ValencyExtractor(config)
    return extractor.extract_from_document(document)


def extract_from_corpus(
    corpus: Corpus,
    config: Optional[ExtractionConfig] = None
) -> ExtractionResult:
    """Extract valency frames from corpus"""
    return extract_valency_frames(corpus, config)
