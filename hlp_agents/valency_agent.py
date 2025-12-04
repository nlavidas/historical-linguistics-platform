"""
Valency Agent - Valency pattern extraction and analysis

This module provides agents for extracting and analyzing
valency patterns from annotated texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from hlp_agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    EngineRegistry,
)

logger = logging.getLogger(__name__)


@dataclass
class ValencyPattern:
    verb_lemma: str
    pattern: str
    arguments: List[Dict[str, str]]
    frequency: int = 1
    examples: List[str] = field(default_factory=list)
    language: str = ""
    period: str = ""
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'verb_lemma': self.verb_lemma,
            'pattern': self.pattern,
            'arguments': self.arguments,
            'frequency': self.frequency,
            'examples': self.examples[:5],
            'language': self.language,
            'period': self.period,
            'source': self.source,
        }


@dataclass
class ValencyFrame:
    verb_lemma: str
    frames: List[ValencyPattern]
    total_occurrences: int = 0
    dominant_pattern: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'verb_lemma': self.verb_lemma,
            'frames': [f.to_dict() for f in self.frames],
            'total_occurrences': self.total_occurrences,
            'dominant_pattern': self.dominant_pattern,
        }


class ValencyAgent(BaseAgent):
    
    CORE_ARGUMENTS = {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl'}
    OBLIQUE_RELATIONS = {'obl', 'obl:agent', 'obl:arg', 'obl:loc', 'obl:time'}
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.patterns: Dict[str, List[ValencyPattern]] = defaultdict(list)
        self.annotation_engine = None
    
    def initialize(self) -> bool:
        engine = EngineRegistry.get_engine('stanza')
        if engine:
            language = self.config.custom_settings.get('language', 'en')
            if engine.load(language):
                self.annotation_engine = engine
                logger.info(f"ValencyAgent initialized with Stanza for {language}")
                return True
        
        engine = EngineRegistry.get_engine('spacy')
        if engine:
            language = self.config.custom_settings.get('language', 'en')
            if engine.load(language):
                self.annotation_engine = engine
                logger.info(f"ValencyAgent initialized with spaCy for {language}")
                return True
        
        logger.warning("ValencyAgent running without annotation engine")
        return True
    
    def process_task(self, task: AgentTask) -> AgentResult:
        if task.task_type == "extract_valency":
            return self._extract_valency(task)
        elif task.task_type == "extract_from_parsed":
            return self._extract_from_parsed(task)
        elif task.task_type == "analyze_verb":
            return self._analyze_verb(task)
        elif task.task_type == "compare_patterns":
            return self._compare_patterns(task)
        elif task.task_type == "build_lexicon":
            return self._build_lexicon(task)
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message=f"Unknown task type: {task.task_type}"
            )
    
    def _extract_valency(self, task: AgentTask) -> AgentResult:
        text = task.input_data.get('text', '') if isinstance(task.input_data, dict) else str(task.input_data)
        language = task.input_data.get('language', 'en') if isinstance(task.input_data, dict) else 'en'
        period = task.input_data.get('period', '') if isinstance(task.input_data, dict) else ''
        
        if not text:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No text provided"
            )
        
        if self.annotation_engine:
            parsed = self.annotation_engine.process(text, "full")
        else:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No annotation engine available"
            )
        
        patterns = self._extract_patterns_from_parse(parsed, language, period)
        
        for pattern in patterns:
            self.patterns[pattern.verb_lemma].append(pattern)
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'patterns': [p.to_dict() for p in patterns],
                'verb_count': len(set(p.verb_lemma for p in patterns)),
                'pattern_count': len(patterns),
            }
        )
    
    def _extract_from_parsed(self, task: AgentTask) -> AgentResult:
        parsed_data = task.input_data.get('parsed', {}) if isinstance(task.input_data, dict) else {}
        language = task.input_data.get('language', 'en') if isinstance(task.input_data, dict) else 'en'
        period = task.input_data.get('period', '') if isinstance(task.input_data, dict) else ''
        
        if not parsed_data:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No parsed data provided"
            )
        
        patterns = self._extract_patterns_from_parse(parsed_data, language, period)
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'patterns': [p.to_dict() for p in patterns],
                'verb_count': len(set(p.verb_lemma for p in patterns)),
                'pattern_count': len(patterns),
            }
        )
    
    def _extract_patterns_from_parse(
        self,
        parsed: Dict[str, Any],
        language: str = "",
        period: str = ""
    ) -> List[ValencyPattern]:
        patterns = []
        
        for sentence in parsed.get('sentences', []):
            tokens = sentence.get('tokens', [])
            sentence_text = sentence.get('text', '')
            
            token_map = {t.get('id', i+1): t for i, t in enumerate(tokens)}
            
            verbs = [t for t in tokens if t.get('upos', t.get('pos', '')) == 'VERB']
            
            for verb in verbs:
                verb_id = verb.get('id', 0)
                verb_lemma = verb.get('lemma', verb.get('text', ''))
                
                arguments = []
                
                for token in tokens:
                    head = token.get('head', 0)
                    deprel = token.get('deprel', token.get('dep', ''))
                    
                    if head == verb_id and deprel in self.CORE_ARGUMENTS:
                        arg = {
                            'role': deprel,
                            'text': token.get('text', ''),
                            'lemma': token.get('lemma', ''),
                            'pos': token.get('upos', token.get('pos', '')),
                        }
                        
                        if deprel in self.OBLIQUE_RELATIONS:
                            case = self._extract_case(token, token_map)
                            if case:
                                arg['case'] = case
                            
                            prep = self._find_preposition(token, tokens)
                            if prep:
                                arg['preposition'] = prep
                        
                        arguments.append(arg)
                
                if arguments:
                    pattern_str = self._create_pattern_string(arguments)
                    
                    pattern = ValencyPattern(
                        verb_lemma=verb_lemma,
                        pattern=pattern_str,
                        arguments=arguments,
                        examples=[sentence_text],
                        language=language,
                        period=period,
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _create_pattern_string(self, arguments: List[Dict[str, str]]) -> str:
        parts = []
        
        role_order = ['nsubj', 'csubj', 'obj', 'iobj', 'ccomp', 'xcomp', 'obl']
        
        sorted_args = sorted(
            arguments,
            key=lambda a: role_order.index(a['role']) if a['role'] in role_order else 99
        )
        
        for arg in sorted_args:
            role = arg['role']
            pos = arg.get('pos', 'X')
            
            if role == 'nsubj':
                parts.append(f"NP-subj")
            elif role == 'obj':
                parts.append(f"NP-obj")
            elif role == 'iobj':
                parts.append(f"NP-iobj")
            elif role == 'ccomp':
                parts.append(f"CP")
            elif role == 'xcomp':
                parts.append(f"VP-inf")
            elif role.startswith('obl'):
                prep = arg.get('preposition', '')
                case = arg.get('case', '')
                if prep:
                    parts.append(f"PP[{prep}]")
                elif case:
                    parts.append(f"NP[{case}]")
                else:
                    parts.append(f"PP")
        
        return " + ".join(parts) if parts else "intransitive"
    
    def _extract_case(self, token: Dict, token_map: Dict) -> Optional[str]:
        feats = token.get('feats', '')
        if feats:
            if 'Case=' in feats:
                match = re.search(r'Case=(\w+)', feats)
                if match:
                    return match.group(1).lower()
        return None
    
    def _find_preposition(self, token: Dict, tokens: List[Dict]) -> Optional[str]:
        token_id = token.get('id', 0)
        
        for t in tokens:
            if t.get('head', 0) == token_id and t.get('deprel', t.get('dep', '')) == 'case':
                return t.get('lemma', t.get('text', ''))
        
        return None
    
    def _analyze_verb(self, task: AgentTask) -> AgentResult:
        verb_lemma = task.input_data.get('verb', '') if isinstance(task.input_data, dict) else str(task.input_data)
        
        if not verb_lemma:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No verb provided"
            )
        
        verb_patterns = self.patterns.get(verb_lemma, [])
        
        if not verb_patterns:
            return AgentResult(
                task_id=task.task_id,
                success=True,
                output_data={
                    'verb': verb_lemma,
                    'found': False,
                    'message': f"No patterns found for verb '{verb_lemma}'"
                }
            )
        
        pattern_counts: Dict[str, int] = defaultdict(int)
        pattern_examples: Dict[str, List[str]] = defaultdict(list)
        
        for p in verb_patterns:
            pattern_counts[p.pattern] += 1
            pattern_examples[p.pattern].extend(p.examples)
        
        total = sum(pattern_counts.values())
        dominant = max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
        
        frame = ValencyFrame(
            verb_lemma=verb_lemma,
            frames=verb_patterns,
            total_occurrences=total,
            dominant_pattern=dominant
        )
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'verb': verb_lemma,
                'found': True,
                'frame': frame.to_dict(),
                'pattern_distribution': dict(pattern_counts),
                'examples': {k: v[:3] for k, v in pattern_examples.items()},
            }
        )
    
    def _compare_patterns(self, task: AgentTask) -> AgentResult:
        verb = task.input_data.get('verb', '') if isinstance(task.input_data, dict) else ''
        periods = task.input_data.get('periods', []) if isinstance(task.input_data, dict) else []
        
        if not verb:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                error_message="No verb provided"
            )
        
        verb_patterns = self.patterns.get(verb, [])
        
        period_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for p in verb_patterns:
            period = p.period or "unknown"
            period_patterns[period][p.pattern] += 1
        
        comparison = {
            'verb': verb,
            'periods': {},
            'changes': [],
        }
        
        for period, patterns in period_patterns.items():
            total = sum(patterns.values())
            comparison['periods'][period] = {
                'patterns': dict(patterns),
                'total': total,
                'dominant': max(patterns.keys(), key=lambda k: patterns[k]) if patterns else None,
            }
        
        sorted_periods = sorted(period_patterns.keys())
        for i in range(len(sorted_periods) - 1):
            p1, p2 = sorted_periods[i], sorted_periods[i+1]
            patterns1 = set(period_patterns[p1].keys())
            patterns2 = set(period_patterns[p2].keys())
            
            new_patterns = patterns2 - patterns1
            lost_patterns = patterns1 - patterns2
            
            if new_patterns or lost_patterns:
                comparison['changes'].append({
                    'from_period': p1,
                    'to_period': p2,
                    'new_patterns': list(new_patterns),
                    'lost_patterns': list(lost_patterns),
                })
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data=comparison
        )
    
    def _build_lexicon(self, task: AgentTask) -> AgentResult:
        min_frequency = task.input_data.get('min_frequency', 1) if isinstance(task.input_data, dict) else 1
        
        lexicon = {}
        
        for verb_lemma, patterns in self.patterns.items():
            pattern_counts: Dict[str, int] = defaultdict(int)
            pattern_args: Dict[str, List[Dict]] = defaultdict(list)
            
            for p in patterns:
                pattern_counts[p.pattern] += 1
                if p.arguments not in pattern_args[p.pattern]:
                    pattern_args[p.pattern].append(p.arguments)
            
            total = sum(pattern_counts.values())
            
            if total >= min_frequency:
                lexicon[verb_lemma] = {
                    'total_occurrences': total,
                    'patterns': [
                        {
                            'pattern': pat,
                            'frequency': count,
                            'percentage': round(count / total * 100, 1),
                            'argument_structures': pattern_args[pat][:3],
                        }
                        for pat, count in sorted(pattern_counts.items(), key=lambda x: -x[1])
                    ]
                }
        
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output_data={
                'lexicon': lexicon,
                'verb_count': len(lexicon),
                'total_patterns': sum(len(v['patterns']) for v in lexicon.values()),
            }
        )
    
    def cleanup(self):
        if self.annotation_engine:
            self.annotation_engine.unload()
        self.patterns.clear()


class ValencyExtractionAgent(ValencyAgent):
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.greek_case_patterns = {
            'nominative': 'Nom',
            'genitive': 'Gen',
            'dative': 'Dat',
            'accusative': 'Acc',
            'vocative': 'Voc',
        }
    
    def extract_greek_valency(self, text: str, period: str = "") -> Dict[str, Any]:
        task = AgentTask.create("extract_valency", {
            'text': text,
            'language': 'grc',
            'period': period,
        })
        result = self.process_task(task)
        return result.output_data if result.success else {'error': result.error_message}
    
    def extract_latin_valency(self, text: str, period: str = "") -> Dict[str, Any]:
        task = AgentTask.create("extract_valency", {
            'text': text,
            'language': 'la',
            'period': period,
        })
        result = self.process_task(task)
        return result.output_data if result.success else {'error': result.error_message}
