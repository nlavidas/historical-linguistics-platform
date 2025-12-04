"""
Diachronic Agent - Diachronic change detection and analysis

This module provides agents for detecting and analyzing
linguistic changes across historical periods.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

from hlp_agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodData:
    name: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    texts: List[Dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    vocabulary: Dict[str, int] = field(default_factory=dict)
    pos_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'text_count': len(self.texts),
            'token_count': self.token_count,
            'vocabulary_size': len(self.vocabulary),
        }


@dataclass
class ChangeEvent:
    feature: str
    change_type: str
    from_period: str
    to_period: str
    from_value: Any
    to_value: Any
    magnitude: float = 0.0
    significance: str = "low"
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature': self.feature,
            'change_type': self.change_type,
            'from_period': self.from_period,
            'to_period': self.to_period,
            'from_value': self.from_value,
            'to_value': self.to_value,
            'magnitude': self.magnitude,
            'significance': self.significance,
            'examples': self.examples[:5],
        }


class DiachronicAgent(BaseAgent):
    
    GREEK_PERIODS = {
        'Mycenaean': (-1600, -1100),
        'Archaic': (-800, -480),
        'Classical': (-480, -323),
        'Hellenistic': (-323, -31),
        'Roman': (-31, 330),
        'Byzantine': (330, 1453),
        'Post-Byzantine': (1453, 1830),
        'Modern': (1830, 2025),
    }
    
    ENGLISH_PERIODS = {
        'Old English': (450, 1100),
        'Middle English': (1100, 1500),
        'Early Modern English': (1500, 1700),
        'Late Modern English': (1700, 1900),
        'Present-Day English': (1900, 2025),
    }
    
    LATIN_PERIODS = {
        'Archaic Latin': (-700, -100),
        'Classical Latin': (-100, 14),
        'Silver Latin': (14, 200),
        'Late Latin': (200, 600),
        'Medieval Latin': (600, 1500),
        'Renaissance Latin': (1500, 1700),
        'New Latin': (1700, 2025),
    }
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.periods: Dict[str, PeriodData] = {}
        self.changes: List[ChangeEvent] = []
        self.language = config.custom_settings.get('language', 'grc')
    
    def initialize(self) -> bool:
        period_definitions = self._get_period_definitions()
        
        for name, (start, end) in period_definitions.items():
            self.periods[name] = PeriodData(
                name=name,
                start_year=start,
                end_year=end
            )
        
        logger.info(f"DiachronicAgent initialized with {len(self.periods)} periods for {self.language}")
        return True
    
    def _get_period_definitions(self) -> Dict[str, Tuple[int, int]]:
        if self.language in ['grc', 'el', 'greek']:
            return self.GREEK_PERIODS
        elif self.language in ['en', 'ang', 'enm', 'english']:
            return self.ENGLISH_PERIODS
        elif self.language in ['la', 'lat', 'latin']:
            return self.LATIN_PERIODS
        else:
            return self.GREEK_PERIODS
    
    def process_task(self, task: AgentTask) -> AgentResult:
        if task.task_type == "add_text":
            return self._add_text(task)
        elif task.task_type == "analyze_change":
            return self._analyze_change(task)
        elif task.task_type == "compare_periods":
            return self._compare_periods(task)
        elif task.task_type == "detect_changes":
            return self._detect_changes(task)
        elif task.task_type == "timeline":
            return self._generate_timeline(task)
        elif task.task_type == "feature_evolution":
            return self._track_feature_evolution(task)
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Unknown task type: {task.task_type}"
            )
    
    def _add_text(self, task: AgentTask) -> AgentResult:
        text_data = task.input_data if isinstance(task.input_data, dict) else {}
        
        period_name = text_data.get('period', '')
        tokens = text_data.get('tokens', [])
        text = text_data.get('text', '')
        
        if not period_name:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No period specified"
            )
        
        if period_name not in self.periods:
            self.periods[period_name] = PeriodData(name=period_name)
        
        period = self.periods[period_name]
        period.texts.append(text_data)
        
        for token in tokens:
            word = token.get('text', token.get('word', ''))
            lemma = token.get('lemma', word)
            pos = token.get('upos', token.get('pos', 'X'))
            
            period.token_count += 1
            period.vocabulary[lemma] = period.vocabulary.get(lemma, 0) + 1
            period.pos_distribution[pos] = period.pos_distribution.get(pos, 0) + 1
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'period': period_name,
                'texts_in_period': len(period.texts),
                'tokens_in_period': period.token_count,
            }
        )
    
    def _analyze_change(self, task: AgentTask) -> AgentResult:
        feature = task.input_data.get('feature', '') if isinstance(task.input_data, dict) else ''
        
        if not feature:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No feature specified"
            )
        
        sorted_periods = sorted(
            self.periods.values(),
            key=lambda p: p.start_year or 0
        )
        
        feature_values = []
        for period in sorted_periods:
            if period.token_count > 0:
                if feature == 'vocabulary_size':
                    value = len(period.vocabulary)
                elif feature == 'type_token_ratio':
                    value = len(period.vocabulary) / period.token_count if period.token_count > 0 else 0
                elif feature.startswith('pos_'):
                    pos = feature.replace('pos_', '').upper()
                    value = period.pos_distribution.get(pos, 0) / period.token_count if period.token_count > 0 else 0
                else:
                    value = 0
                
                feature_values.append({
                    'period': period.name,
                    'value': value,
                    'token_count': period.token_count,
                })
        
        changes = []
        for i in range(len(feature_values) - 1):
            v1 = feature_values[i]['value']
            v2 = feature_values[i + 1]['value']
            
            if v1 != 0:
                change_pct = ((v2 - v1) / v1) * 100
            else:
                change_pct = 100 if v2 > 0 else 0
            
            if abs(change_pct) > 10:
                changes.append({
                    'from_period': feature_values[i]['period'],
                    'to_period': feature_values[i + 1]['period'],
                    'from_value': v1,
                    'to_value': v2,
                    'change_percent': round(change_pct, 2),
                })
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'feature': feature,
                'values_by_period': feature_values,
                'significant_changes': changes,
            }
        )
    
    def _compare_periods(self, task: AgentTask) -> AgentResult:
        period1_name = task.input_data.get('period1', '') if isinstance(task.input_data, dict) else ''
        period2_name = task.input_data.get('period2', '') if isinstance(task.input_data, dict) else ''
        
        if not period1_name or not period2_name:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="Two periods must be specified"
            )
        
        period1 = self.periods.get(period1_name)
        period2 = self.periods.get(period2_name)
        
        if not period1 or not period2:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="One or both periods not found"
            )
        
        vocab1 = set(period1.vocabulary.keys())
        vocab2 = set(period2.vocabulary.keys())
        
        shared_vocab = vocab1.intersection(vocab2)
        new_vocab = vocab2 - vocab1
        lost_vocab = vocab1 - vocab2
        
        pos_comparison = {}
        all_pos = set(period1.pos_distribution.keys()).union(set(period2.pos_distribution.keys()))
        
        for pos in all_pos:
            p1_freq = period1.pos_distribution.get(pos, 0) / period1.token_count if period1.token_count > 0 else 0
            p2_freq = period2.pos_distribution.get(pos, 0) / period2.token_count if period2.token_count > 0 else 0
            
            pos_comparison[pos] = {
                'period1_freq': round(p1_freq * 100, 2),
                'period2_freq': round(p2_freq * 100, 2),
                'change': round((p2_freq - p1_freq) * 100, 2),
            }
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'period1': period1.to_dict(),
                'period2': period2.to_dict(),
                'vocabulary_comparison': {
                    'shared_count': len(shared_vocab),
                    'new_in_period2': len(new_vocab),
                    'lost_from_period1': len(lost_vocab),
                    'jaccard_similarity': len(shared_vocab) / len(vocab1.union(vocab2)) if vocab1.union(vocab2) else 0,
                },
                'pos_comparison': pos_comparison,
            }
        )
    
    def _detect_changes(self, task: AgentTask) -> AgentResult:
        threshold = task.input_data.get('threshold', 0.1) if isinstance(task.input_data, dict) else 0.1
        
        sorted_periods = sorted(
            [p for p in self.periods.values() if p.token_count > 0],
            key=lambda p: p.start_year or 0
        )
        
        detected_changes = []
        
        for i in range(len(sorted_periods) - 1):
            p1, p2 = sorted_periods[i], sorted_periods[i + 1]
            
            vocab1 = set(p1.vocabulary.keys())
            vocab2 = set(p2.vocabulary.keys())
            
            new_words = vocab2 - vocab1
            lost_words = vocab1 - vocab2
            
            if len(new_words) / len(vocab1) > threshold if vocab1 else False:
                detected_changes.append(ChangeEvent(
                    feature='vocabulary',
                    change_type='lexical_innovation',
                    from_period=p1.name,
                    to_period=p2.name,
                    from_value=len(vocab1),
                    to_value=len(vocab2),
                    magnitude=len(new_words) / len(vocab1) if vocab1 else 0,
                    significance='high' if len(new_words) / len(vocab1) > 0.3 else 'medium',
                    examples=list(new_words)[:10],
                ))
            
            all_pos = set(p1.pos_distribution.keys()).union(set(p2.pos_distribution.keys()))
            for pos in all_pos:
                freq1 = p1.pos_distribution.get(pos, 0) / p1.token_count if p1.token_count > 0 else 0
                freq2 = p2.pos_distribution.get(pos, 0) / p2.token_count if p2.token_count > 0 else 0
                
                change = abs(freq2 - freq1)
                if change > threshold:
                    detected_changes.append(ChangeEvent(
                        feature=f'pos_{pos}',
                        change_type='frequency_shift',
                        from_period=p1.name,
                        to_period=p2.name,
                        from_value=round(freq1 * 100, 2),
                        to_value=round(freq2 * 100, 2),
                        magnitude=change,
                        significance='high' if change > 0.2 else 'medium',
                    ))
        
        self.changes.extend(detected_changes)
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'changes_detected': len(detected_changes),
                'changes': [c.to_dict() for c in detected_changes],
            }
        )
    
    def _generate_timeline(self, task: AgentTask) -> AgentResult:
        feature = task.input_data.get('feature', 'vocabulary_size') if isinstance(task.input_data, dict) else 'vocabulary_size'
        
        sorted_periods = sorted(
            self.periods.values(),
            key=lambda p: p.start_year or 0
        )
        
        timeline = []
        for period in sorted_periods:
            if feature == 'vocabulary_size':
                value = len(period.vocabulary)
            elif feature == 'token_count':
                value = period.token_count
            elif feature == 'text_count':
                value = len(period.texts)
            elif feature == 'type_token_ratio':
                value = len(period.vocabulary) / period.token_count if period.token_count > 0 else 0
            else:
                value = 0
            
            timeline.append({
                'period': period.name,
                'start_year': period.start_year,
                'end_year': period.end_year,
                'value': value,
            })
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'feature': feature,
                'timeline': timeline,
            }
        )
    
    def _track_feature_evolution(self, task: AgentTask) -> AgentResult:
        word = task.input_data.get('word', '') if isinstance(task.input_data, dict) else ''
        
        if not word:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No word specified"
            )
        
        sorted_periods = sorted(
            self.periods.values(),
            key=lambda p: p.start_year or 0
        )
        
        evolution = []
        for period in sorted_periods:
            freq = period.vocabulary.get(word, 0)
            relative_freq = freq / period.token_count if period.token_count > 0 else 0
            
            evolution.append({
                'period': period.name,
                'frequency': freq,
                'relative_frequency': round(relative_freq * 10000, 4),
                'present': freq > 0,
            })
        
        first_appearance = None
        last_appearance = None
        for entry in evolution:
            if entry['present']:
                if first_appearance is None:
                    first_appearance = entry['period']
                last_appearance = entry['period']
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'word': word,
                'evolution': evolution,
                'first_appearance': first_appearance,
                'last_appearance': last_appearance,
                'continuous': all(e['present'] for e in evolution if evolution.index(e) >= evolution.index({'period': first_appearance, 'present': True}) and evolution.index(e) <= evolution.index({'period': last_appearance, 'present': True})) if first_appearance and last_appearance else False,
            }
        )
    
    def cleanup(self):
        self.periods.clear()
        self.changes.clear()


class ChangeDetectionAgent(DiachronicAgent):
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.change_patterns = []
    
    def detect_grammaticalization(self, word: str) -> Dict[str, Any]:
        task = AgentTask.create("feature_evolution", {'word': word})
        result = self.process_task(task)
        
        if not result.success:
            return {'error': result.error_message}
        
        evolution = result.output_data.get('evolution', [])
        
        freq_trend = []
        for i, entry in enumerate(evolution):
            if entry['present']:
                freq_trend.append(entry['relative_frequency'])
        
        if len(freq_trend) >= 2:
            increasing = all(freq_trend[i] <= freq_trend[i+1] for i in range(len(freq_trend)-1))
            if increasing and freq_trend[-1] > freq_trend[0] * 2:
                return {
                    'word': word,
                    'grammaticalization_detected': True,
                    'frequency_increase': freq_trend[-1] / freq_trend[0] if freq_trend[0] > 0 else 0,
                    'evolution': evolution,
                }
        
        return {
            'word': word,
            'grammaticalization_detected': False,
            'evolution': evolution,
        }
    
    def detect_semantic_shift(self, word: str) -> Dict[str, Any]:
        return {
            'word': word,
            'note': 'Semantic shift detection requires context analysis - not yet implemented',
        }
