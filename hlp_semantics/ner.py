"""
HLP Semantics NER - Named Entity Recognition

This module provides comprehensive support for named entity recognition
in ancient and historical texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum

from hlp_core.models import Language, Sentence, Token, Document

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Named entity types"""
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    GPE = "GPE"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FACILITY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    NORP = "NORP"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    
    DEITY = "DEITY"
    MYTHOLOGICAL = "MYTHOLOGICAL"
    ETHNIC = "ETHNIC"
    TITLE = "TITLE"
    OFFICE = "OFFICE"
    FESTIVAL = "FESTIVAL"
    LITERARY_WORK = "LITERARY_WORK"
    PLACE_ANCIENT = "PLACE_ANCIENT"
    INSTITUTION = "INSTITUTION"


GREEK_DEITY_PATTERNS = [
    r"Ζε[υύ]ς", r"Δι[οό]ς", r"Δι[ίι]",
    r"Ἥρα", r"Ἥρης", r"Ἥρᾳ",
    r"Ἀθην[αᾶ]", r"Ἀθηνᾶς", r"Ἀθηνᾷ",
    r"Ἀπόλλων", r"Ἀπόλλωνος",
    r"Ἄρτεμις", r"Ἀρτέμιδος",
    r"Ἄρης", r"Ἄρεος", r"Ἄρει",
    r"Ἀφροδίτη", r"Ἀφροδίτης",
    r"Ἑρμῆς", r"Ἑρμοῦ",
    r"Ποσειδῶν", r"Ποσειδῶνος",
    r"Δήμητρα", r"Δήμητρος",
    r"Ἥφαιστος", r"Ἡφαίστου",
    r"Διόνυσος", r"Διονύσου",
]

LATIN_DEITY_PATTERNS = [
    r"Iuppiter", r"Iovis", r"Iovi",
    r"Iuno", r"Iunonis",
    r"Minerva", r"Minervae",
    r"Apollo", r"Apollinis",
    r"Diana", r"Dianae",
    r"Mars", r"Martis",
    r"Venus", r"Veneris",
    r"Mercurius", r"Mercurii",
    r"Neptunus", r"Neptuni",
    r"Ceres", r"Cereris",
    r"Vulcanus", r"Vulcani",
    r"Bacchus", r"Bacchi",
]

GREEK_PLACE_PATTERNS = [
    r"Ἀθῆναι", r"Ἀθηνῶν", r"Ἀθήνησι",
    r"Σπάρτη", r"Σπάρτης", r"Λακεδαίμων",
    r"Θῆβαι", r"Θηβῶν",
    r"Κόρινθος", r"Κορίνθου",
    r"Ἄργος", r"Ἄργους",
    r"Μυκῆναι", r"Μυκηνῶν",
    r"Τροία", r"Τροίας", r"Ἴλιον",
    r"Ὄλυμπος", r"Ὀλύμπου",
    r"Δελφοί", r"Δελφῶν",
    r"Ἑλλάς", r"Ἑλλάδος",
]

LATIN_PLACE_PATTERNS = [
    r"Roma", r"Romae",
    r"Carthago", r"Carthaginis",
    r"Athenae", r"Athenarum",
    r"Alexandria", r"Alexandriae",
    r"Gallia", r"Galliae",
    r"Hispania", r"Hispaniae",
    r"Graecia", r"Graeciae",
    r"Italia", r"Italiae",
    r"Aegyptus", r"Aegypti",
]


@dataclass
class NamedEntity:
    """Represents a named entity"""
    entity_type: EntityType
    
    text: str
    
    tokens: List[Token]
    
    start_token_idx: int
    
    end_token_idx: int
    
    normalized_form: Optional[str] = None
    
    confidence: float = 1.0
    
    source: str = "rule"
    
    wikidata_id: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entity_type": self.entity_type.value,
            "text": self.text,
            "token_ids": [t.id for t in self.tokens],
            "start_token_idx": self.start_token_idx,
            "end_token_idx": self.end_token_idx,
            "normalized_form": self.normalized_form,
            "confidence": self.confidence,
            "source": self.source,
            "wikidata_id": self.wikidata_id
        }


@dataclass
class NERConfig:
    """Configuration for NER annotator"""
    use_pattern_matching: bool = True
    
    use_ml_model: bool = False
    
    model_name: Optional[str] = None
    
    use_gazetteer: bool = True
    
    gazetteer_path: Optional[str] = None
    
    min_confidence: float = 0.5
    
    merge_adjacent: bool = True
    
    language: str = "grc"
    
    entity_types: Optional[List[EntityType]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NERResult:
    """Result of NER annotation"""
    entities: List[NamedEntity]
    
    sentence_id: Optional[str] = None
    
    total_entities: int = 0
    
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[NamedEntity]:
        """Get entities of a specific type"""
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "sentence_id": self.sentence_id,
            "total_entities": self.total_entities,
            "entity_type_counts": self.entity_type_counts,
            "error_count": len(self.errors)
        }


class NERAnnotator:
    """Annotator for named entity recognition"""
    
    def __init__(self, config: Optional[NERConfig] = None):
        self.config = config or NERConfig()
        self._patterns: Dict[EntityType, List[re.Pattern]] = {}
        self._gazetteer: Dict[str, EntityType] = {}
        self._model = None
        
        self._compile_patterns()
        self._load_gazetteer()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        if self.config.language in ["grc", "greek", "ancient_greek"]:
            self._patterns[EntityType.DEITY] = [
                re.compile(p, re.UNICODE) for p in GREEK_DEITY_PATTERNS
            ]
            self._patterns[EntityType.LOCATION] = [
                re.compile(p, re.UNICODE) for p in GREEK_PLACE_PATTERNS
            ]
        
        elif self.config.language in ["la", "lat", "latin"]:
            self._patterns[EntityType.DEITY] = [
                re.compile(p, re.UNICODE) for p in LATIN_DEITY_PATTERNS
            ]
            self._patterns[EntityType.LOCATION] = [
                re.compile(p, re.UNICODE) for p in LATIN_PLACE_PATTERNS
            ]
        
        self._patterns[EntityType.PERSON] = [
            re.compile(r"[A-Z][a-z]+(?:us|os|es|is|as|ius|eus|aeus)$"),
        ]
    
    def _load_gazetteer(self):
        """Load gazetteer if configured"""
        if not self.config.use_gazetteer:
            return
        
        self._gazetteer = {
            "Zeus": EntityType.DEITY,
            "Athena": EntityType.DEITY,
            "Apollo": EntityType.DEITY,
            "Hermes": EntityType.DEITY,
            "Athens": EntityType.LOCATION,
            "Sparta": EntityType.LOCATION,
            "Rome": EntityType.LOCATION,
            "Troy": EntityType.LOCATION,
            "Homer": EntityType.PERSON,
            "Plato": EntityType.PERSON,
            "Aristotle": EntityType.PERSON,
            "Cicero": EntityType.PERSON,
            "Caesar": EntityType.PERSON,
        }
    
    def annotate_sentence(self, sentence: Sentence) -> NERResult:
        """Annotate a sentence with named entities"""
        entities = []
        errors = []
        
        if self.config.use_pattern_matching:
            pattern_entities = self._find_by_patterns(sentence)
            entities.extend(pattern_entities)
        
        if self.config.use_gazetteer:
            gazetteer_entities = self._find_by_gazetteer(sentence)
            entities.extend(gazetteer_entities)
        
        capitalized_entities = self._find_capitalized(sentence)
        entities.extend(capitalized_entities)
        
        if self.config.merge_adjacent:
            entities = self._merge_adjacent(entities)
        
        entities = self._deduplicate(entities)
        
        type_counts = {}
        for entity in entities:
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return NERResult(
            entities=entities,
            sentence_id=sentence.id,
            total_entities=len(entities),
            entity_type_counts=type_counts,
            errors=errors
        )
    
    def annotate_document(self, document: Document) -> List[NERResult]:
        """Annotate all sentences in a document"""
        results = []
        
        for sentence in document.sentences:
            result = self.annotate_sentence(sentence)
            results.append(result)
        
        return results
    
    def _find_by_patterns(self, sentence: Sentence) -> List[NamedEntity]:
        """Find entities using regex patterns"""
        entities = []
        
        for entity_type, patterns in self._patterns.items():
            for i, token in enumerate(sentence.tokens):
                for pattern in patterns:
                    if pattern.match(token.form):
                        entities.append(NamedEntity(
                            entity_type=entity_type,
                            text=token.form,
                            tokens=[token],
                            start_token_idx=i,
                            end_token_idx=i,
                            confidence=0.9,
                            source="pattern"
                        ))
                        break
        
        return entities
    
    def _find_by_gazetteer(self, sentence: Sentence) -> List[NamedEntity]:
        """Find entities using gazetteer"""
        entities = []
        
        for i, token in enumerate(sentence.tokens):
            form = token.form
            lemma = token.lemma or form
            
            entity_type = self._gazetteer.get(form) or self._gazetteer.get(lemma)
            
            if entity_type:
                entities.append(NamedEntity(
                    entity_type=entity_type,
                    text=form,
                    tokens=[token],
                    start_token_idx=i,
                    end_token_idx=i,
                    normalized_form=lemma,
                    confidence=0.95,
                    source="gazetteer"
                ))
        
        return entities
    
    def _find_capitalized(self, sentence: Sentence) -> List[NamedEntity]:
        """Find potential entities by capitalization"""
        entities = []
        
        for i, token in enumerate(sentence.tokens):
            if i == 0:
                continue
            
            form = token.form
            
            if form and form[0].isupper() and len(form) > 1:
                if token.pos in ["PROPN", "NNP", "Ne"]:
                    entity_type = EntityType.PERSON
                else:
                    entity_type = EntityType.PERSON
                
                entities.append(NamedEntity(
                    entity_type=entity_type,
                    text=form,
                    tokens=[token],
                    start_token_idx=i,
                    end_token_idx=i,
                    confidence=0.6,
                    source="capitalization"
                ))
        
        return entities
    
    def _merge_adjacent(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Merge adjacent entities of same type"""
        if not entities:
            return entities
        
        entities = sorted(entities, key=lambda e: e.start_token_idx)
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            if (entity.entity_type == current.entity_type and
                entity.start_token_idx == current.end_token_idx + 1):
                current = NamedEntity(
                    entity_type=current.entity_type,
                    text=f"{current.text} {entity.text}",
                    tokens=current.tokens + entity.tokens,
                    start_token_idx=current.start_token_idx,
                    end_token_idx=entity.end_token_idx,
                    confidence=min(current.confidence, entity.confidence),
                    source=current.source
                )
            else:
                merged.append(current)
                current = entity
        
        merged.append(current)
        
        return merged
    
    def _deduplicate(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.start_token_idx, entity.end_token_idx, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique


def annotate_entities(
    sentence: Sentence,
    config: Optional[NERConfig] = None
) -> NERResult:
    """Annotate named entities in a sentence"""
    annotator = NERAnnotator(config)
    return annotator.annotate_sentence(sentence)


def extract_entities(
    sentence: Sentence,
    entity_types: Optional[List[EntityType]] = None,
    config: Optional[NERConfig] = None
) -> List[NamedEntity]:
    """Extract named entities from a sentence"""
    result = annotate_entities(sentence, config)
    
    if entity_types:
        return [e for e in result.entities if e.entity_type in entity_types]
    
    return result.entities
