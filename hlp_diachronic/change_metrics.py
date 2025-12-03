"""
HLP Diachronic Change Metrics - Calculate Diachronic Change Statistics

This module provides comprehensive support for calculating and analyzing
linguistic changes across historical periods.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from enum import Enum

from hlp_diachronic.periodization import (
    PeriodDefinition, PeriodSystem, GreekPeriodSystem, LatinPeriodSystem
)
from hlp_core.models import Period, Language
from hlp_valency.lexicon_builder import ValencyLexicon, LexiconEntry, PatternEntry
from hlp_valency.pattern_extractor import FrameType

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of linguistic change"""
    FREQUENCY_INCREASE = "frequency_increase"
    FREQUENCY_DECREASE = "frequency_decrease"
    INNOVATION = "innovation"
    LOSS = "loss"
    PATTERN_SHIFT = "pattern_shift"
    CASE_CHANGE = "case_change"
    ARGUMENT_ADDITION = "argument_addition"
    ARGUMENT_LOSS = "argument_loss"
    SEMANTIC_SHIFT = "semantic_shift"
    GRAMMATICALIZATION = "grammaticalization"
    LEXICALIZATION = "lexicalization"
    ANALOGICAL_CHANGE = "analogical_change"
    BORROWING = "borrowing"
    CALQUE = "calque"


class ChangeSignificance(Enum):
    """Significance levels for changes"""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class ChangeTrend(Enum):
    """Trend directions"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"
    EMERGING = "emerging"
    DECLINING = "declining"


@dataclass
class ChangeMetric:
    """A metric measuring linguistic change"""
    name: str
    
    value: float
    
    change_type: ChangeType
    
    period_start: Period
    period_end: Period
    
    significance: ChangeSignificance = ChangeSignificance.MINOR
    
    confidence: float = 1.0
    
    baseline_value: Optional[float] = None
    final_value: Optional[float] = None
    
    sample_size_start: int = 0
    sample_size_end: int = 0
    
    p_value: Optional[float] = None
    
    description: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_percent_change(self) -> Optional[float]:
        """Calculate percent change"""
        if self.baseline_value is None or self.baseline_value == 0:
            return None
        if self.final_value is None:
            return None
        return ((self.final_value - self.baseline_value) / self.baseline_value) * 100
    
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if change is statistically significant"""
        if self.p_value is not None:
            return self.p_value < threshold
        return self.significance in [ChangeSignificance.MAJOR, ChangeSignificance.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "change_type": self.change_type.value,
            "period_start": self.period_start.value,
            "period_end": self.period_end.value,
            "significance": self.significance.value,
            "confidence": self.confidence,
            "baseline_value": self.baseline_value,
            "final_value": self.final_value,
            "percent_change": self.get_percent_change(),
            "sample_size_start": self.sample_size_start,
            "sample_size_end": self.sample_size_end,
            "p_value": self.p_value,
            "description": self.description
        }


@dataclass
class DiachronicChange:
    """Represents a specific diachronic change"""
    lemma: str
    
    change_type: ChangeType
    
    period_start: Period
    period_end: Period
    
    description: str
    
    metrics: List[ChangeMetric] = field(default_factory=list)
    
    evidence: List[str] = field(default_factory=list)
    
    related_changes: List[str] = field(default_factory=list)
    
    trend: ChangeTrend = ChangeTrend.STABLE
    
    confidence: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: ChangeMetric):
        """Add a metric to this change"""
        self.metrics.append(metric)
    
    def get_overall_significance(self) -> ChangeSignificance:
        """Get overall significance based on metrics"""
        if not self.metrics:
            return ChangeSignificance.NEGLIGIBLE
        
        significance_order = [
            ChangeSignificance.NEGLIGIBLE,
            ChangeSignificance.MINOR,
            ChangeSignificance.MODERATE,
            ChangeSignificance.MAJOR,
            ChangeSignificance.CRITICAL
        ]
        
        max_sig = ChangeSignificance.NEGLIGIBLE
        for metric in self.metrics:
            if significance_order.index(metric.significance) > significance_order.index(max_sig):
                max_sig = metric.significance
        
        return max_sig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "lemma": self.lemma,
            "change_type": self.change_type.value,
            "period_start": self.period_start.value,
            "period_end": self.period_end.value,
            "description": self.description,
            "metrics": [m.to_dict() for m in self.metrics],
            "evidence": self.evidence,
            "related_changes": self.related_changes,
            "trend": self.trend.value,
            "confidence": self.confidence,
            "overall_significance": self.get_overall_significance().value
        }


@dataclass
class PeriodData:
    """Data for a specific period"""
    period: Period
    
    frequency: int = 0
    
    normalized_frequency: float = 0.0
    
    pattern_distribution: Dict[str, int] = field(default_factory=dict)
    
    frame_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    total_tokens: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChangeCalculator:
    """Calculates diachronic change metrics"""
    
    def __init__(
        self,
        period_system: Optional[PeriodSystem] = None,
        language: Language = Language.ANCIENT_GREEK
    ):
        if period_system:
            self.period_system = period_system
        elif language == Language.ANCIENT_GREEK:
            self.period_system = GreekPeriodSystem()
        elif language == Language.LATIN:
            self.period_system = LatinPeriodSystem()
        else:
            self.period_system = GreekPeriodSystem()
        
        self.language = language
    
    def calculate_frequency_change(
        self,
        entry: LexiconEntry,
        period_start: Period,
        period_end: Period
    ) -> ChangeMetric:
        """Calculate frequency change between periods"""
        start_freq = 0
        end_freq = 0
        
        for pattern in entry.patterns.values():
            start_freq += pattern.period_distribution.get(period_start.value, 0)
            end_freq += pattern.period_distribution.get(period_end.value, 0)
        
        if start_freq == 0 and end_freq == 0:
            change_value = 0.0
            change_type = ChangeType.FREQUENCY_DECREASE
        elif start_freq == 0:
            change_value = float('inf')
            change_type = ChangeType.INNOVATION
        elif end_freq == 0:
            change_value = -1.0
            change_type = ChangeType.LOSS
        else:
            change_value = (end_freq - start_freq) / start_freq
            change_type = ChangeType.FREQUENCY_INCREASE if change_value > 0 else ChangeType.FREQUENCY_DECREASE
        
        significance = self._calculate_significance(change_value, start_freq, end_freq)
        
        return ChangeMetric(
            name=f"frequency_change_{entry.lemma}",
            value=change_value,
            change_type=change_type,
            period_start=period_start,
            period_end=period_end,
            significance=significance,
            baseline_value=float(start_freq),
            final_value=float(end_freq),
            sample_size_start=start_freq,
            sample_size_end=end_freq,
            description=f"Frequency change for {entry.lemma} from {period_start.value} to {period_end.value}"
        )
    
    def calculate_pattern_change(
        self,
        entry: LexiconEntry,
        period_start: Period,
        period_end: Period
    ) -> List[ChangeMetric]:
        """Calculate pattern distribution changes"""
        metrics = []
        
        start_patterns = defaultdict(int)
        end_patterns = defaultdict(int)
        
        for pattern_key, pattern in entry.patterns.items():
            start_patterns[pattern_key] = pattern.period_distribution.get(period_start.value, 0)
            end_patterns[pattern_key] = pattern.period_distribution.get(period_end.value, 0)
        
        all_patterns = set(start_patterns.keys()) | set(end_patterns.keys())
        
        for pattern_key in all_patterns:
            start_freq = start_patterns.get(pattern_key, 0)
            end_freq = end_patterns.get(pattern_key, 0)
            
            if start_freq == 0 and end_freq > 0:
                change_type = ChangeType.INNOVATION
                change_value = 1.0
            elif start_freq > 0 and end_freq == 0:
                change_type = ChangeType.LOSS
                change_value = -1.0
            elif start_freq > 0:
                change_value = (end_freq - start_freq) / start_freq
                change_type = ChangeType.PATTERN_SHIFT
            else:
                continue
            
            significance = self._calculate_significance(change_value, start_freq, end_freq)
            
            metrics.append(ChangeMetric(
                name=f"pattern_change_{entry.lemma}_{pattern_key}",
                value=change_value,
                change_type=change_type,
                period_start=period_start,
                period_end=period_end,
                significance=significance,
                baseline_value=float(start_freq),
                final_value=float(end_freq),
                description=f"Pattern change for {pattern_key}"
            ))
        
        return metrics
    
    def calculate_frame_type_change(
        self,
        lexicon: ValencyLexicon,
        period_start: Period,
        period_end: Period
    ) -> Dict[FrameType, ChangeMetric]:
        """Calculate frame type distribution changes"""
        start_dist = defaultdict(int)
        end_dist = defaultdict(int)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                ft = pattern.frame_type
                start_dist[ft] += pattern.period_distribution.get(period_start.value, 0)
                end_dist[ft] += pattern.period_distribution.get(period_end.value, 0)
        
        metrics = {}
        
        for ft in FrameType:
            start_freq = start_dist.get(ft, 0)
            end_freq = end_dist.get(ft, 0)
            
            if start_freq == 0 and end_freq == 0:
                continue
            
            if start_freq == 0:
                change_value = float('inf')
            else:
                change_value = (end_freq - start_freq) / start_freq
            
            change_type = ChangeType.FREQUENCY_INCREASE if change_value > 0 else ChangeType.FREQUENCY_DECREASE
            significance = self._calculate_significance(change_value, start_freq, end_freq)
            
            metrics[ft] = ChangeMetric(
                name=f"frame_type_change_{ft.value}",
                value=change_value,
                change_type=change_type,
                period_start=period_start,
                period_end=period_end,
                significance=significance,
                baseline_value=float(start_freq),
                final_value=float(end_freq),
                description=f"Frame type {ft.value} change"
            )
        
        return metrics
    
    def detect_innovations(
        self,
        lexicon: ValencyLexicon,
        period: Period
    ) -> List[DiachronicChange]:
        """Detect innovations in a period"""
        innovations = []
        
        ordered_periods = self.period_system.get_ordered_periods()
        period_idx = None
        
        for i, p in enumerate(ordered_periods):
            if p.period_enum == period:
                period_idx = i
                break
        
        if period_idx is None or period_idx == 0:
            return innovations
        
        prev_period = ordered_periods[period_idx - 1].period_enum
        
        for entry in lexicon:
            for pattern_key, pattern in entry.patterns.items():
                prev_freq = pattern.period_distribution.get(prev_period.value, 0)
                curr_freq = pattern.period_distribution.get(period.value, 0)
                
                if prev_freq == 0 and curr_freq > 0:
                    change = DiachronicChange(
                        lemma=entry.lemma,
                        change_type=ChangeType.INNOVATION,
                        period_start=prev_period,
                        period_end=period,
                        description=f"New pattern {pattern.pattern_string} emerges in {period.value}",
                        trend=ChangeTrend.EMERGING,
                        evidence=[f"First attested in {period.value} with {curr_freq} occurrences"]
                    )
                    
                    change.add_metric(ChangeMetric(
                        name=f"innovation_{entry.lemma}_{pattern_key}",
                        value=1.0,
                        change_type=ChangeType.INNOVATION,
                        period_start=prev_period,
                        period_end=period,
                        significance=ChangeSignificance.MAJOR,
                        final_value=float(curr_freq)
                    ))
                    
                    innovations.append(change)
        
        return innovations
    
    def detect_losses(
        self,
        lexicon: ValencyLexicon,
        period: Period
    ) -> List[DiachronicChange]:
        """Detect losses in a period"""
        losses = []
        
        ordered_periods = self.period_system.get_ordered_periods()
        period_idx = None
        
        for i, p in enumerate(ordered_periods):
            if p.period_enum == period:
                period_idx = i
                break
        
        if period_idx is None or period_idx == 0:
            return losses
        
        prev_period = ordered_periods[period_idx - 1].period_enum
        
        for entry in lexicon:
            for pattern_key, pattern in entry.patterns.items():
                prev_freq = pattern.period_distribution.get(prev_period.value, 0)
                curr_freq = pattern.period_distribution.get(period.value, 0)
                
                if prev_freq > 0 and curr_freq == 0:
                    change = DiachronicChange(
                        lemma=entry.lemma,
                        change_type=ChangeType.LOSS,
                        period_start=prev_period,
                        period_end=period,
                        description=f"Pattern {pattern.pattern_string} lost in {period.value}",
                        trend=ChangeTrend.DECLINING,
                        evidence=[f"Last attested in {prev_period.value} with {prev_freq} occurrences"]
                    )
                    
                    change.add_metric(ChangeMetric(
                        name=f"loss_{entry.lemma}_{pattern_key}",
                        value=-1.0,
                        change_type=ChangeType.LOSS,
                        period_start=prev_period,
                        period_end=period,
                        significance=ChangeSignificance.MAJOR,
                        baseline_value=float(prev_freq)
                    ))
                    
                    losses.append(change)
        
        return losses
    
    def calculate_trend(
        self,
        entry: LexiconEntry,
        periods: List[Period]
    ) -> ChangeTrend:
        """Calculate overall trend across periods"""
        frequencies = []
        
        for period in periods:
            freq = 0
            for pattern in entry.patterns.values():
                freq += pattern.period_distribution.get(period.value, 0)
            frequencies.append(freq)
        
        if len(frequencies) < 2:
            return ChangeTrend.STABLE
        
        if frequencies[0] == 0 and frequencies[-1] > 0:
            return ChangeTrend.EMERGING
        
        if frequencies[0] > 0 and frequencies[-1] == 0:
            return ChangeTrend.DECLINING
        
        increases = 0
        decreases = 0
        
        for i in range(1, len(frequencies)):
            if frequencies[i] > frequencies[i-1]:
                increases += 1
            elif frequencies[i] < frequencies[i-1]:
                decreases += 1
        
        if increases > decreases * 2:
            return ChangeTrend.INCREASING
        elif decreases > increases * 2:
            return ChangeTrend.DECREASING
        elif increases > 0 and decreases > 0:
            return ChangeTrend.FLUCTUATING
        else:
            return ChangeTrend.STABLE
    
    def _calculate_significance(
        self,
        change_value: float,
        start_freq: int,
        end_freq: int
    ) -> ChangeSignificance:
        """Calculate significance of a change"""
        if math.isinf(change_value):
            return ChangeSignificance.MAJOR
        
        abs_change = abs(change_value)
        
        if abs_change < 0.1:
            return ChangeSignificance.NEGLIGIBLE
        elif abs_change < 0.25:
            return ChangeSignificance.MINOR
        elif abs_change < 0.5:
            return ChangeSignificance.MODERATE
        elif abs_change < 1.0:
            return ChangeSignificance.MAJOR
        else:
            return ChangeSignificance.CRITICAL
    
    def get_period_summary(
        self,
        lexicon: ValencyLexicon,
        period: Period
    ) -> Dict[str, Any]:
        """Get summary statistics for a period"""
        total_freq = 0
        pattern_count = 0
        frame_type_dist = defaultdict(int)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                freq = pattern.period_distribution.get(period.value, 0)
                if freq > 0:
                    total_freq += freq
                    pattern_count += 1
                    frame_type_dist[pattern.frame_type.value] += freq
        
        return {
            "period": period.value,
            "total_frequency": total_freq,
            "pattern_count": pattern_count,
            "frame_type_distribution": dict(frame_type_dist)
        }


def calculate_frequency_change(
    entry: LexiconEntry,
    period_start: Period,
    period_end: Period,
    language: Language = Language.ANCIENT_GREEK
) -> ChangeMetric:
    """Calculate frequency change for an entry"""
    calculator = ChangeCalculator(language=language)
    return calculator.calculate_frequency_change(entry, period_start, period_end)


def calculate_pattern_change(
    entry: LexiconEntry,
    period_start: Period,
    period_end: Period,
    language: Language = Language.ANCIENT_GREEK
) -> List[ChangeMetric]:
    """Calculate pattern changes for an entry"""
    calculator = ChangeCalculator(language=language)
    return calculator.calculate_pattern_change(entry, period_start, period_end)


def calculate_construction_change(
    lexicon: ValencyLexicon,
    period_start: Period,
    period_end: Period,
    language: Language = Language.ANCIENT_GREEK
) -> Dict[FrameType, ChangeMetric]:
    """Calculate construction/frame type changes"""
    calculator = ChangeCalculator(language=language)
    return calculator.calculate_frame_type_change(lexicon, period_start, period_end)


def detect_innovations(
    lexicon: ValencyLexicon,
    period: Period,
    language: Language = Language.ANCIENT_GREEK
) -> List[DiachronicChange]:
    """Detect innovations in a period"""
    calculator = ChangeCalculator(language=language)
    return calculator.detect_innovations(lexicon, period)


def detect_losses(
    lexicon: ValencyLexicon,
    period: Period,
    language: Language = Language.ANCIENT_GREEK
) -> List[DiachronicChange]:
    """Detect losses in a period"""
    calculator = ChangeCalculator(language=language)
    return calculator.detect_losses(lexicon, period)
