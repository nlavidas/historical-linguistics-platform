"""
Annotation Agent - Multi-engine annotation for linguistic analysis

This module provides annotation agents that use multiple AI engines
for POS tagging, dependency parsing, NER, and morphological analysis.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from hlp_agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    AIEngine,
    EngineRegistry,
    StanzaEngine,
    SpaCyEngine,
    HuggingFaceEngine,
    OllamaEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class AnnotationConfig:
    engines: List[str] = field(default_factory=lambda: ['stanza'])
    language: str = "en"
    tasks: List[str] = field(default_factory=lambda: ['pos', 'lemma', 'depparse', 'ner'])
    ensemble_method: str = "voting"
    confidence_threshold: float = 0.8
    fallback_engine: str = "stanza"


class AnnotationAgent(BaseAgent):
    
    def __init__(self, config: AgentConfig, annotation_config: Optional[AnnotationConfig] = None):
        super().__init__(config)
        self.annotation_config = annotation_config or AnnotationConfig()
        self.engines: Dict[str, AIEngine] = {}
        self.primary_engine: Optional[AIEngine] = None
    
    def initialize(self) -> bool:
        for engine_name in self.annotation_config.engines:
            engine = EngineRegistry.get_engine(engine_name)
            if engine:
                if engine.load(self.annotation_config.language):
                    self.engines[engine_name] = engine
                    if not self.primary_engine:
                        self.primary_engine = engine
                    logger.info(f"Loaded engine: {engine_name}")
                else:
                    logger.warning(f"Failed to load engine: {engine_name}")
        
        if not self.engines:
            logger.error("No annotation engines loaded")
            return False
        
        return True
    
    def process_task(self, task: AgentTask) -> AgentResult:
        if task.task_type == "annotate":
            return self._annotate_text(task)
        elif task.task_type == "annotate_batch":
            return self._annotate_batch(task)
        elif task.task_type == "pos_tag":
            return self._pos_tag(task)
        elif task.task_type == "parse":
            return self._dependency_parse(task)
        elif task.task_type == "ner":
            return self._named_entity_recognition(task)
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Unknown task type: {task.task_type}"
            )
    
    def _annotate_text(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get('text', '') if isinstance(task.input_data, dict) else str(task.input_data)
        
        if not text:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No text provided"
            )
        
        results = {}
        
        for engine_name, engine in self.engines.items():
            try:
                result = engine.process(text, "full")
                results[engine_name] = result
            except Exception as e:
                logger.error(f"Engine {engine_name} failed: {e}")
                results[engine_name] = {'error': str(e)}
        
        if self.annotation_config.ensemble_method == "voting" and len(results) > 1:
            combined = self._ensemble_voting(results)
        else:
            primary_name = list(self.engines.keys())[0]
            combined = results.get(primary_name, {})
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'text': text,
                'annotation': combined,
                'engine_results': results,
                'engines_used': list(self.engines.keys()),
            }
        )
    
    def _annotate_batch(self, task: AgentTask) -> AgentResult:
        texts = task.input_data.get('texts', []) if isinstance(task.input_data, dict) else []
        
        if not texts:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No texts provided"
            )
        
        annotations = []
        for text in texts:
            sub_task = AgentTask.create("annotate", {'text': text})
            result = self._annotate_text(sub_task)
            if result.success:
                annotations.append(result.output_data)
            else:
                annotations.append({'error': result.error_message, 'text': text})
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'annotations': annotations,
                'count': len(annotations),
            }
        )
    
    def _pos_tag(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get('text', '') if isinstance(task.input_data, dict) else str(task.input_data)
        
        if not self.primary_engine:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No engine available"
            )
        
        result = self.primary_engine.process(text, "full")
        
        pos_tags = []
        for sent in result.get('sentences', []):
            for token in sent.get('tokens', []):
                pos_tags.append({
                    'word': token.get('text', ''),
                    'lemma': token.get('lemma', ''),
                    'pos': token.get('upos', token.get('tag', '')),
                    'xpos': token.get('xpos', token.get('tag', '')),
                })
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'text': text,
                'pos_tags': pos_tags,
            }
        )
    
    def _dependency_parse(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get('text', '') if isinstance(task.input_data, dict) else str(task.input_data)
        
        if not self.primary_engine:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No engine available"
            )
        
        result = self.primary_engine.process(text, "full")
        
        parse_trees = []
        for sent in result.get('sentences', []):
            tree = {
                'text': sent.get('text', ''),
                'tokens': []
            }
            for token in sent.get('tokens', []):
                tree['tokens'].append({
                    'id': token.get('id', 0),
                    'word': token.get('text', ''),
                    'lemma': token.get('lemma', ''),
                    'pos': token.get('upos', ''),
                    'head': token.get('head', 0),
                    'deprel': token.get('deprel', token.get('dep', '')),
                })
            parse_trees.append(tree)
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'text': text,
                'parse_trees': parse_trees,
            }
        )
    
    def _named_entity_recognition(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get('text', '') if isinstance(task.input_data, dict) else str(task.input_data)
        
        if not self.primary_engine:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No engine available"
            )
        
        result = self.primary_engine.process(text, "full")
        
        entities = result.get('entities', [])
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'text': text,
                'entities': entities,
            }
        )
    
    def _ensemble_voting(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        combined = {
            'sentences': [],
            'entities': [],
        }
        
        all_sentences = []
        for engine_name, result in results.items():
            if 'error' not in result:
                all_sentences.append(result.get('sentences', []))
        
        if all_sentences:
            combined['sentences'] = all_sentences[0]
        
        entity_votes: Dict[str, Dict] = {}
        for engine_name, result in results.items():
            if 'error' not in result:
                for entity in result.get('entities', []):
                    key = f"{entity.get('text', '')}_{entity.get('type', '')}"
                    if key not in entity_votes:
                        entity_votes[key] = {
                            'entity': entity,
                            'votes': 0
                        }
                    entity_votes[key]['votes'] += 1
        
        threshold = len(results) / 2
        for key, vote_data in entity_votes.items():
            if vote_data['votes'] >= threshold:
                combined['entities'].append(vote_data['entity'])
        
        return combined
    
    def cleanup(self):
        for engine in self.engines.values():
            engine.unload()
        self.engines.clear()


class MultiEngineAnnotationAgent(AnnotationAgent):
    
    def __init__(self, config: AgentConfig):
        annotation_config = AnnotationConfig(
            engines=['stanza', 'spacy', 'huggingface'],
            ensemble_method='voting'
        )
        super().__init__(config, annotation_config)
    
    def annotate_with_all_engines(self, text: str) -> Dict[str, Any]:
        task = AgentTask.create("annotate", {'text': text})
        result = self.process_task(task)
        return result.output_data if result.success else {'error': result.error_message}
    
    def compare_engines(self, text: str) -> Dict[str, Any]:
        comparison = {
            'text': text,
            'engines': {},
            'agreement': {},
        }
        
        for engine_name, engine in self.engines.items():
            try:
                result = engine.process(text, "full")
                comparison['engines'][engine_name] = result
            except Exception as e:
                comparison['engines'][engine_name] = {'error': str(e)}
        
        pos_agreement = self._calculate_pos_agreement(comparison['engines'])
        comparison['agreement']['pos'] = pos_agreement
        
        ner_agreement = self._calculate_ner_agreement(comparison['engines'])
        comparison['agreement']['ner'] = ner_agreement
        
        return comparison
    
    def _calculate_pos_agreement(self, engine_results: Dict[str, Dict]) -> float:
        all_pos_sequences = []
        
        for engine_name, result in engine_results.items():
            if 'error' not in result:
                pos_seq = []
                for sent in result.get('sentences', []):
                    for token in sent.get('tokens', []):
                        pos_seq.append(token.get('upos', token.get('tag', '')))
                all_pos_sequences.append(pos_seq)
        
        if len(all_pos_sequences) < 2:
            return 1.0
        
        agreements = 0
        total = 0
        
        min_len = min(len(seq) for seq in all_pos_sequences)
        for i in range(min_len):
            tags = [seq[i] for seq in all_pos_sequences]
            if len(set(tags)) == 1:
                agreements += 1
            total += 1
        
        return agreements / total if total > 0 else 0.0
    
    def _calculate_ner_agreement(self, engine_results: Dict[str, Dict]) -> float:
        all_entities = []
        
        for engine_name, result in engine_results.items():
            if 'error' not in result:
                entities = set()
                for ent in result.get('entities', []):
                    entities.add(f"{ent.get('text', '')}:{ent.get('type', '')}")
                all_entities.append(entities)
        
        if len(all_entities) < 2:
            return 1.0
        
        intersection = all_entities[0]
        union = all_entities[0]
        
        for entities in all_entities[1:]:
            intersection = intersection.intersection(entities)
            union = union.union(entities)
        
        return len(intersection) / len(union) if union else 1.0
