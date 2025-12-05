"""
HLP Annotation Ensemble - Multi-Engine Ensemble Annotation

This module provides ensemble annotation capabilities by combining
multiple annotation engines and using various voting/aggregation
strategies to produce high-quality annotations.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import Counter
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from hlp_annotation.base_engine import (
    AnnotationEngine, AnnotationCapability, AnnotationResult,
    AnnotationConfig, EngineStatus, EngineMetrics
)
from hlp_core.models import (
    Token, Sentence, Document, Corpus,
    MorphologicalFeatures, SyntacticRelation,
    PartOfSpeech, DependencyRelation,
    NamedEntity, NamedEntityType, SemanticRole
)

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Voting strategies for ensemble"""
    MAJORITY = "majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    UNANIMOUS = "unanimous"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    FIRST_AVAILABLE = "first_available"
    BEST_CONFIDENCE = "best_confidence"
    CASCADING = "cascading"
    UNION = "union"
    INTERSECTION = "intersection"


class AggregationMethod(Enum):
    """Aggregation methods for combining results"""
    VOTE = "vote"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class EngineWeight:
    """Weight configuration for an engine"""
    engine_name: str
    weight: float = 1.0
    
    priority: int = 0
    
    capability_weights: Dict[str, float] = field(default_factory=dict)
    
    min_confidence: float = 0.0
    
    enabled: bool = True
    
    def get_weight_for_capability(self, capability: AnnotationCapability) -> float:
        """Get weight for specific capability"""
        return self.capability_weights.get(capability.value, self.weight)


@dataclass
class EnsembleConfig(AnnotationConfig):
    """Configuration for ensemble engine"""
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED_MAJORITY
    
    aggregation_method: AggregationMethod = AggregationMethod.VOTE
    
    engine_weights: List[EngineWeight] = field(default_factory=list)
    
    min_agreement: float = 0.5
    
    fallback_strategy: VotingStrategy = VotingStrategy.FIRST_AVAILABLE
    
    parallel_execution: bool = True
    max_workers: int = 4
    
    timeout_per_engine: float = 60.0
    
    require_all_engines: bool = False
    
    min_engines_required: int = 1
    
    combine_entities: bool = True
    entity_overlap_threshold: float = 0.5
    
    track_disagreements: bool = True
    
    def get_engine_weight(self, engine_name: str) -> EngineWeight:
        """Get weight for engine"""
        for ew in self.engine_weights:
            if ew.engine_name == engine_name:
                return ew
        return EngineWeight(engine_name=engine_name)


@dataclass
class DisagreementRecord:
    """Record of disagreement between engines"""
    token_id: int
    token_form: str
    feature: str
    engine_values: Dict[str, Any]
    final_value: Any
    confidence: float


@dataclass
class EnsembleResult(AnnotationResult):
    """Extended result for ensemble annotation"""
    engine_results: Dict[str, AnnotationResult] = field(default_factory=dict)
    
    disagreements: List[DisagreementRecord] = field(default_factory=list)
    
    agreement_scores: Dict[str, float] = field(default_factory=dict)
    
    engines_used: List[str] = field(default_factory=list)
    engines_failed: List[str] = field(default_factory=list)


class EnsembleEngine(AnnotationEngine):
    """Ensemble annotation engine combining multiple engines"""
    
    def __init__(
        self,
        engines: Optional[List[AnnotationEngine]] = None,
        config: Optional[EnsembleConfig] = None
    ):
        super().__init__(config or EnsembleConfig())
        self._engines: List[AnnotationEngine] = engines or []
        self._engine_map: Dict[str, AnnotationEngine] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        
        for engine in self._engines:
            self._engine_map[engine.name] = engine
    
    @property
    def name(self) -> str:
        return "EnsembleEngine"
    
    @property
    def version(self) -> str:
        engine_versions = [f"{e.name}:{e.version}" for e in self._engines]
        return f"Ensemble ({len(self._engines)} engines)"
    
    @property
    def capabilities(self) -> List[AnnotationCapability]:
        all_caps = set()
        for engine in self._engines:
            all_caps.update(engine.capabilities)
        return list(all_caps)
    
    @property
    def supported_languages(self) -> List[str]:
        all_langs = set()
        for engine in self._engines:
            all_langs.update(engine.supported_languages)
        return list(all_langs)
    
    @property
    def engines(self) -> List[AnnotationEngine]:
        """Get list of engines"""
        return self._engines.copy()
    
    def add_engine(self, engine: AnnotationEngine, weight: float = 1.0):
        """Add engine to ensemble"""
        self._engines.append(engine)
        self._engine_map[engine.name] = engine
        
        if isinstance(self.config, EnsembleConfig):
            self.config.engine_weights.append(
                EngineWeight(engine_name=engine.name, weight=weight)
            )
    
    def remove_engine(self, engine_name: str) -> bool:
        """Remove engine from ensemble"""
        if engine_name in self._engine_map:
            engine = self._engine_map.pop(engine_name)
            self._engines.remove(engine)
            
            if isinstance(self.config, EnsembleConfig):
                self.config.engine_weights = [
                    ew for ew in self.config.engine_weights
                    if ew.engine_name != engine_name
                ]
            return True
        return False
    
    def set_engine_weight(self, engine_name: str, weight: float):
        """Set weight for engine"""
        if isinstance(self.config, EnsembleConfig):
            for ew in self.config.engine_weights:
                if ew.engine_name == engine_name:
                    ew.weight = weight
                    return
            self.config.engine_weights.append(
                EngineWeight(engine_name=engine_name, weight=weight)
            )
    
    def initialize(self) -> bool:
        """Initialize all engines"""
        if self._initialized:
            return True
        
        self._status = EngineStatus.INITIALIZING
        
        success_count = 0
        for engine in self._engines:
            try:
                if engine.initialize():
                    success_count += 1
                else:
                    logger.warning(f"Failed to initialize engine: {engine.name}")
            except Exception as e:
                logger.error(f"Error initializing engine {engine.name}: {e}")
        
        config = self.config
        min_required = config.min_engines_required if isinstance(config, EnsembleConfig) else 1
        
        if success_count >= min_required:
            if isinstance(config, EnsembleConfig) and config.parallel_execution:
                self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
            
            self._status = EngineStatus.READY
            self._initialized = True
            logger.info(f"Ensemble initialized with {success_count}/{len(self._engines)} engines")
            return True
        else:
            self._status = EngineStatus.ERROR
            logger.error(f"Not enough engines initialized: {success_count}/{min_required} required")
            return False
    
    def shutdown(self):
        """Shutdown all engines"""
        for engine in self._engines:
            try:
                engine.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down engine {engine.name}: {e}")
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        self._initialized = False
        self._status = EngineStatus.SHUTDOWN
        logger.info("Ensemble engine shutdown")
    
    def _process_text(
        self,
        text: str,
        capabilities: List[AnnotationCapability]
    ) -> AnnotationResult:
        """Process text with ensemble"""
        config = self.config
        if not isinstance(config, EnsembleConfig):
            config = EnsembleConfig()
        
        engine_results: Dict[str, AnnotationResult] = {}
        engines_used: List[str] = []
        engines_failed: List[str] = []
        
        if config.parallel_execution and self._executor:
            engine_results, engines_used, engines_failed = self._process_parallel(
                text, capabilities, config
            )
        else:
            engine_results, engines_used, engines_failed = self._process_sequential(
                text, capabilities, config
            )
        
        if not engine_results:
            return EnsembleResult(
                success=False,
                errors=["No engines produced results"],
                engines_failed=engines_failed
            )
        
        combined_result = self._combine_results(
            engine_results, capabilities, config
        )
        
        combined_result.engine_results = engine_results
        combined_result.engines_used = engines_used
        combined_result.engines_failed = engines_failed
        
        return combined_result
    
    def _process_parallel(
        self,
        text: str,
        capabilities: List[AnnotationCapability],
        config: EnsembleConfig
    ) -> Tuple[Dict[str, AnnotationResult], List[str], List[str]]:
        """Process with engines in parallel"""
        engine_results: Dict[str, AnnotationResult] = {}
        engines_used: List[str] = []
        engines_failed: List[str] = []
        
        futures = {}
        for engine in self._engines:
            if not engine.is_ready:
                continue
            
            engine_weight = config.get_engine_weight(engine.name)
            if not engine_weight.enabled:
                continue
            
            engine_caps = [c for c in capabilities if engine.supports_capability(c)]
            if not engine_caps:
                continue
            
            future = self._executor.submit(
                engine.annotate_text, text, engine_caps
            )
            futures[future] = engine.name
        
        for future in as_completed(futures, timeout=config.timeout_per_engine * len(futures)):
            engine_name = futures[future]
            try:
                result = future.result(timeout=config.timeout_per_engine)
                if result.success:
                    engine_results[engine_name] = result
                    engines_used.append(engine_name)
                else:
                    engines_failed.append(engine_name)
            except Exception as e:
                logger.error(f"Engine {engine_name} failed: {e}")
                engines_failed.append(engine_name)
        
        return engine_results, engines_used, engines_failed
    
    def _process_sequential(
        self,
        text: str,
        capabilities: List[AnnotationCapability],
        config: EnsembleConfig
    ) -> Tuple[Dict[str, AnnotationResult], List[str], List[str]]:
        """Process with engines sequentially"""
        engine_results: Dict[str, AnnotationResult] = {}
        engines_used: List[str] = []
        engines_failed: List[str] = []
        
        for engine in self._engines:
            if not engine.is_ready:
                continue
            
            engine_weight = config.get_engine_weight(engine.name)
            if not engine_weight.enabled:
                continue
            
            engine_caps = [c for c in capabilities if engine.supports_capability(c)]
            if not engine_caps:
                continue
            
            try:
                result = engine.annotate_text(text, engine_caps)
                if result.success:
                    engine_results[engine.name] = result
                    engines_used.append(engine.name)
                else:
                    engines_failed.append(engine.name)
            except Exception as e:
                logger.error(f"Engine {engine.name} failed: {e}")
                engines_failed.append(engine.name)
        
        return engine_results, engines_used, engines_failed
    
    def _combine_results(
        self,
        engine_results: Dict[str, AnnotationResult],
        capabilities: List[AnnotationCapability],
        config: EnsembleConfig
    ) -> EnsembleResult:
        """Combine results from multiple engines"""
        combined = EnsembleResult(success=True)
        
        all_tokens = self._combine_tokens(engine_results, config)
        combined.tokens = all_tokens
        combined.tokens_processed = len(all_tokens)
        
        if AnnotationCapability.NAMED_ENTITY_RECOGNITION in capabilities:
            combined.entities = self._combine_entities(engine_results, config)
        
        if AnnotationCapability.SEMANTIC_ROLE_LABELING in capabilities:
            combined.semantic_roles = self._combine_semantic_roles(engine_results, config)
        
        combined.agreement_scores = self._calculate_agreement(engine_results)
        
        if config.track_disagreements:
            combined.disagreements = self._track_disagreements(engine_results, all_tokens)
        
        confidences = [r.confidence for r in engine_results.values() if r.confidence > 0]
        if confidences:
            combined.confidence = sum(confidences) / len(confidences)
        
        return combined
    
    def _combine_tokens(
        self,
        engine_results: Dict[str, AnnotationResult],
        config: EnsembleConfig
    ) -> List[Token]:
        """Combine token annotations from multiple engines"""
        if not engine_results:
            return []
        
        first_result = next(iter(engine_results.values()))
        if not first_result.tokens:
            return []
        
        num_tokens = len(first_result.tokens)
        combined_tokens = []
        
        for idx in range(num_tokens):
            token_annotations = {}
            for engine_name, result in engine_results.items():
                if idx < len(result.tokens):
                    token_annotations[engine_name] = result.tokens[idx]
            
            if not token_annotations:
                continue
            
            combined_token = self._combine_single_token(
                token_annotations, config, idx + 1
            )
            combined_tokens.append(combined_token)
        
        return combined_tokens
    
    def _combine_single_token(
        self,
        token_annotations: Dict[str, Token],
        config: EnsembleConfig,
        token_id: int
    ) -> Token:
        """Combine annotations for a single token"""
        first_token = next(iter(token_annotations.values()))
        
        combined = Token(
            id=token_id,
            form=first_token.form
        )
        
        lemmas = {}
        for engine_name, token in token_annotations.items():
            if token.lemma:
                weight = config.get_engine_weight(engine_name).weight
                if token.lemma not in lemmas:
                    lemmas[token.lemma] = 0
                lemmas[token.lemma] += weight
        
        if lemmas:
            combined.lemma = max(lemmas, key=lemmas.get)
        
        combined.morphology = self._combine_morphology(
            {name: t.morphology for name, t in token_annotations.items() if t.morphology},
            config
        )
        
        combined.syntax = self._combine_syntax(
            {name: t.syntax for name, t in token_annotations.items() if t.syntax},
            config
        )
        
        return combined
    
    def _combine_morphology(
        self,
        morph_annotations: Dict[str, MorphologicalFeatures],
        config: EnsembleConfig
    ) -> Optional[MorphologicalFeatures]:
        """Combine morphological annotations"""
        if not morph_annotations:
            return None
        
        combined = MorphologicalFeatures()
        
        pos_votes = {}
        for engine_name, morph in morph_annotations.items():
            if morph.pos:
                weight = config.get_engine_weight(engine_name).weight
                if morph.pos not in pos_votes:
                    pos_votes[morph.pos] = 0
                pos_votes[morph.pos] += weight
        
        if pos_votes:
            combined.pos = max(pos_votes, key=pos_votes.get)
        
        for attr in ['case', 'number', 'gender', 'person', 'tense', 'mood', 'voice', 'degree']:
            votes = {}
            for engine_name, morph in morph_annotations.items():
                value = getattr(morph, attr, None)
                if value:
                    weight = config.get_engine_weight(engine_name).weight
                    if value not in votes:
                        votes[value] = 0
                    votes[value] += weight
            
            if votes:
                setattr(combined, attr, max(votes, key=votes.get))
        
        return combined
    
    def _combine_syntax(
        self,
        syntax_annotations: Dict[str, SyntacticRelation],
        config: EnsembleConfig
    ) -> Optional[SyntacticRelation]:
        """Combine syntactic annotations"""
        if not syntax_annotations:
            return None
        
        head_votes = {}
        rel_votes = {}
        
        for engine_name, syntax in syntax_annotations.items():
            weight = config.get_engine_weight(engine_name).weight
            
            if syntax.head_id not in head_votes:
                head_votes[syntax.head_id] = 0
            head_votes[syntax.head_id] += weight
            
            if syntax.relation:
                if syntax.relation not in rel_votes:
                    rel_votes[syntax.relation] = 0
                rel_votes[syntax.relation] += weight
        
        combined = SyntacticRelation(
            head_id=max(head_votes, key=head_votes.get) if head_votes else 0,
            relation=max(rel_votes, key=rel_votes.get) if rel_votes else DependencyRelation.DEP
        )
        
        return combined
    
    def _combine_entities(
        self,
        engine_results: Dict[str, AnnotationResult],
        config: EnsembleConfig
    ) -> List[NamedEntity]:
        """Combine named entity annotations"""
        all_entities = []
        
        for engine_name, result in engine_results.items():
            for entity in result.entities:
                entity.annotator = engine_name
                all_entities.append(entity)
        
        if not config.combine_entities:
            return all_entities
        
        merged = []
        used = set()
        
        for i, entity1 in enumerate(all_entities):
            if i in used:
                continue
            
            overlapping = [entity1]
            used.add(i)
            
            for j, entity2 in enumerate(all_entities):
                if j in used:
                    continue
                
                overlap = self._calculate_entity_overlap(entity1, entity2)
                if overlap >= config.entity_overlap_threshold:
                    overlapping.append(entity2)
                    used.add(j)
            
            merged_entity = self._merge_entities(overlapping, config)
            merged.append(merged_entity)
        
        return merged
    
    def _calculate_entity_overlap(self, e1: NamedEntity, e2: NamedEntity) -> float:
        """Calculate overlap between two entities"""
        if e1.span_start is None or e1.span_end is None:
            return 1.0 if e1.text == e2.text else 0.0
        if e2.span_start is None or e2.span_end is None:
            return 1.0 if e1.text == e2.text else 0.0
        
        start = max(e1.span_start, e2.span_start)
        end = min(e1.span_end, e2.span_end)
        
        if start >= end:
            return 0.0
        
        intersection = end - start
        union = max(e1.span_end, e2.span_end) - min(e1.span_start, e2.span_start)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_entities(
        self,
        entities: List[NamedEntity],
        config: EnsembleConfig
    ) -> NamedEntity:
        """Merge overlapping entities"""
        if len(entities) == 1:
            return entities[0]
        
        type_votes = {}
        for entity in entities:
            weight = config.get_engine_weight(entity.annotator or "").weight
            if entity.entity_type not in type_votes:
                type_votes[entity.entity_type] = 0
            type_votes[entity.entity_type] += weight
        
        best_type = max(type_votes, key=type_votes.get)
        
        texts = [e.text for e in entities]
        best_text = max(set(texts), key=texts.count)
        
        starts = [e.span_start for e in entities if e.span_start is not None]
        ends = [e.span_end for e in entities if e.span_end is not None]
        
        confidences = [e.confidence for e in entities if e.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        
        return NamedEntity(
            entity_type=best_type,
            text=best_text,
            span_start=min(starts) if starts else None,
            span_end=max(ends) if ends else None,
            confidence=avg_confidence
        )
    
    def _combine_semantic_roles(
        self,
        engine_results: Dict[str, AnnotationResult],
        config: EnsembleConfig
    ) -> List[SemanticRole]:
        """Combine semantic role annotations"""
        all_roles = []
        for result in engine_results.values():
            all_roles.extend(result.semantic_roles)
        return all_roles
    
    def _calculate_agreement(
        self,
        engine_results: Dict[str, AnnotationResult]
    ) -> Dict[str, float]:
        """Calculate agreement scores between engines"""
        agreement = {}
        
        engine_names = list(engine_results.keys())
        if len(engine_names) < 2:
            return agreement
        
        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i+1:]:
                pair_key = f"{name1}_{name2}"
                
                tokens1 = engine_results[name1].tokens
                tokens2 = engine_results[name2].tokens
                
                if not tokens1 or not tokens2:
                    continue
                
                matches = 0
                total = min(len(tokens1), len(tokens2))
                
                for t1, t2 in zip(tokens1, tokens2):
                    if t1.morphology and t2.morphology:
                        if t1.morphology.pos == t2.morphology.pos:
                            matches += 1
                
                agreement[pair_key] = matches / total if total > 0 else 0.0
        
        return agreement
    
    def _track_disagreements(
        self,
        engine_results: Dict[str, AnnotationResult],
        combined_tokens: List[Token]
    ) -> List[DisagreementRecord]:
        """Track disagreements between engines"""
        disagreements = []
        
        for idx, combined_token in enumerate(combined_tokens):
            engine_pos = {}
            for engine_name, result in engine_results.items():
                if idx < len(result.tokens):
                    token = result.tokens[idx]
                    if token.morphology and token.morphology.pos:
                        engine_pos[engine_name] = token.morphology.pos.value
            
            if len(set(engine_pos.values())) > 1:
                disagreements.append(DisagreementRecord(
                    token_id=combined_token.id,
                    token_form=combined_token.form,
                    feature="pos",
                    engine_values=engine_pos,
                    final_value=combined_token.morphology.pos.value if combined_token.morphology and combined_token.morphology.pos else None,
                    confidence=1.0 / len(set(engine_pos.values()))
                ))
        
        return disagreements
    
    def get_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all engines"""
        status = {}
        for engine in self._engines:
            status[engine.name] = engine.get_status_info()
        return status
    
    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get ensemble metrics"""
        return {
            "num_engines": len(self._engines),
            "engines_ready": sum(1 for e in self._engines if e.is_ready),
            "capabilities": [c.value for c in self.capabilities],
            "engine_metrics": {e.name: e.metrics.to_dict() for e in self._engines}
        }


def create_ensemble(
    engines: List[AnnotationEngine],
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED_MAJORITY,
    parallel: bool = True
) -> EnsembleEngine:
    """Factory function to create ensemble engine"""
    config = EnsembleConfig(
        voting_strategy=voting_strategy,
        parallel_execution=parallel
    )
    return EnsembleEngine(engines, config)
