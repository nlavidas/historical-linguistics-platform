"""
HLP Diachronic Visualization Data - Generate Data for Visualizations

This module provides utilities for generating data structures suitable
for various visualization libraries and formats.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from enum import Enum

from hlp_diachronic.periodization import (
    PeriodDefinition, PeriodSystem, GreekPeriodSystem, LatinPeriodSystem
)
from hlp_diachronic.change_metrics import (
    ChangeMetric, DiachronicChange, ChangeType, ChangeTrend
)
from hlp_core.models import Period, Language
from hlp_valency.lexicon_builder import ValencyLexicon, LexiconEntry
from hlp_valency.pattern_extractor import FrameType

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts"""
    LINE = "line"
    BAR = "bar"
    STACKED_BAR = "stacked_bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    NETWORK = "network"
    TIMELINE = "timeline"
    TREEMAP = "treemap"


class ColorScheme(Enum):
    """Color schemes for visualizations"""
    DEFAULT = "default"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    QUALITATIVE = "qualitative"


DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

PERIOD_COLORS = {
    Period.MYCENAEAN: "#8B4513",
    Period.ARCHAIC: "#DAA520",
    Period.CLASSICAL: "#4169E1",
    Period.HELLENISTIC: "#32CD32",
    Period.ROMAN: "#DC143C",
    Period.LATE_ANTIQUE: "#9932CC",
    Period.BYZANTINE: "#FF8C00",
    Period.MODERN: "#20B2AA",
}


@dataclass
class DataPoint:
    """A single data point"""
    x: Any
    y: Any
    
    label: Optional[str] = None
    
    color: Optional[str] = None
    
    size: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"x": self.x, "y": self.y}
        if self.label:
            result["label"] = self.label
        if self.color:
            result["color"] = self.color
        if self.size:
            result["size"] = self.size
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class DataSeries:
    """A series of data points"""
    name: str
    
    data: List[DataPoint]
    
    color: Optional[str] = None
    
    series_type: str = "line"
    
    visible: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "data": [d.to_dict() for d in self.data],
            "color": self.color,
            "type": self.series_type,
            "visible": self.visible
        }


@dataclass
class TimeSeriesData:
    """Time series data for temporal visualizations"""
    title: str
    
    series: List[DataSeries]
    
    x_axis_label: str = "Period"
    y_axis_label: str = "Frequency"
    
    x_axis_type: str = "category"
    
    periods: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "series": [s.to_dict() for s in self.series],
            "xAxis": {
                "label": self.x_axis_label,
                "type": self.x_axis_type,
                "categories": self.periods
            },
            "yAxis": {
                "label": self.y_axis_label
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_plotly(self) -> Dict[str, Any]:
        """Convert to Plotly format"""
        traces = []
        for series in self.series:
            trace = {
                "name": series.name,
                "x": [d.x for d in series.data],
                "y": [d.y for d in series.data],
                "type": series.series_type,
                "visible": series.visible
            }
            if series.color:
                trace["marker"] = {"color": series.color}
            traces.append(trace)
        
        layout = {
            "title": self.title,
            "xaxis": {"title": self.x_axis_label},
            "yaxis": {"title": self.y_axis_label}
        }
        
        return {"data": traces, "layout": layout}
    
    def to_chartjs(self) -> Dict[str, Any]:
        """Convert to Chart.js format"""
        datasets = []
        for series in self.series:
            dataset = {
                "label": series.name,
                "data": [d.y for d in series.data],
                "hidden": not series.visible
            }
            if series.color:
                dataset["borderColor"] = series.color
                dataset["backgroundColor"] = series.color
            datasets.append(dataset)
        
        return {
            "type": "line",
            "data": {
                "labels": self.periods,
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": self.title}
                },
                "scales": {
                    "x": {"title": {"display": True, "text": self.x_axis_label}},
                    "y": {"title": {"display": True, "text": self.y_axis_label}}
                }
            }
        }


@dataclass
class DistributionData:
    """Distribution data for pie/bar charts"""
    title: str
    
    categories: List[str]
    values: List[float]
    
    colors: List[str] = field(default_factory=list)
    
    chart_type: ChartType = ChartType.PIE
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "categories": self.categories,
            "values": self.values,
            "colors": self.colors or DEFAULT_COLORS[:len(self.categories)],
            "chartType": self.chart_type.value
        }
    
    def to_plotly(self) -> Dict[str, Any]:
        """Convert to Plotly format"""
        if self.chart_type == ChartType.PIE:
            trace = {
                "type": "pie",
                "labels": self.categories,
                "values": self.values,
                "marker": {"colors": self.colors or DEFAULT_COLORS[:len(self.categories)]}
            }
        else:
            trace = {
                "type": "bar",
                "x": self.categories,
                "y": self.values,
                "marker": {"color": self.colors or DEFAULT_COLORS[:len(self.categories)]}
            }
        
        return {
            "data": [trace],
            "layout": {"title": self.title}
        }


@dataclass
class NetworkNode:
    """A node in a network"""
    id: str
    label: str
    
    group: Optional[str] = None
    
    size: float = 1.0
    
    color: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "size": self.size,
            "color": self.color
        }


@dataclass
class NetworkEdge:
    """An edge in a network"""
    source: str
    target: str
    
    weight: float = 1.0
    
    label: Optional[str] = None
    
    color: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "label": self.label,
            "color": self.color
        }


@dataclass
class NetworkData:
    """Network/graph data"""
    title: str
    
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    
    directed: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "directed": self.directed
        }
    
    def to_d3(self) -> Dict[str, Any]:
        """Convert to D3.js format"""
        return {
            "nodes": [{"id": n.id, "group": n.group or 1, "label": n.label} for n in self.nodes],
            "links": [{"source": e.source, "target": e.target, "value": e.weight} for e in self.edges]
        }
    
    def to_cytoscape(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js format"""
        elements = []
        
        for node in self.nodes:
            elements.append({
                "data": {"id": node.id, "label": node.label},
                "group": "nodes"
            })
        
        for edge in self.edges:
            elements.append({
                "data": {"source": edge.source, "target": edge.target, "weight": edge.weight},
                "group": "edges"
            })
        
        return {"elements": elements}


@dataclass
class VisualizationData:
    """Container for visualization data"""
    title: str
    
    chart_type: ChartType
    
    data: Union[TimeSeriesData, DistributionData, NetworkData, Dict[str, Any]]
    
    description: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data_dict = self.data.to_dict() if hasattr(self.data, 'to_dict') else self.data
        return {
            "title": self.title,
            "chartType": self.chart_type.value,
            "data": data_dict,
            "description": self.description
        }


class VisualizationGenerator:
    """Generates visualization data from linguistic data"""
    
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
    
    def generate_frequency_timeline(
        self,
        lexicon: ValencyLexicon,
        lemmas: Optional[List[str]] = None,
        top_n: int = 10
    ) -> TimeSeriesData:
        """Generate frequency timeline for lemmas"""
        periods = [p.period_enum.value for p in self.period_system.get_ordered_periods()]
        
        if lemmas is None:
            sorted_entries = sorted(
                lexicon.entries.values(),
                key=lambda e: e.total_frequency,
                reverse=True
            )
            lemmas = [e.lemma for e in sorted_entries[:top_n]]
        
        series = []
        for i, lemma in enumerate(lemmas):
            entry = lexicon.get_entry(lemma)
            if not entry:
                continue
            
            data_points = []
            for period in periods:
                freq = 0
                for pattern in entry.patterns.values():
                    freq += pattern.period_distribution.get(period, 0)
                data_points.append(DataPoint(x=period, y=freq))
            
            series.append(DataSeries(
                name=lemma,
                data=data_points,
                color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ))
        
        return TimeSeriesData(
            title="Frequency Timeline",
            series=series,
            periods=periods,
            y_axis_label="Frequency"
        )
    
    def generate_pattern_distribution(
        self,
        entry: LexiconEntry,
        period: Optional[Period] = None
    ) -> DistributionData:
        """Generate pattern distribution for an entry"""
        categories = []
        values = []
        
        for pattern_key, pattern in entry.patterns.items():
            if period:
                freq = pattern.period_distribution.get(period.value, 0)
            else:
                freq = pattern.frequency
            
            if freq > 0:
                categories.append(pattern.pattern_string)
                values.append(freq)
        
        title = f"Pattern Distribution: {entry.lemma}"
        if period:
            title += f" ({period.value})"
        
        return DistributionData(
            title=title,
            categories=categories,
            values=values,
            chart_type=ChartType.PIE
        )
    
    def generate_frame_type_distribution(
        self,
        lexicon: ValencyLexicon,
        period: Optional[Period] = None
    ) -> DistributionData:
        """Generate frame type distribution"""
        frame_counts = defaultdict(int)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                if period:
                    freq = pattern.period_distribution.get(period.value, 0)
                else:
                    freq = pattern.frequency
                
                frame_counts[pattern.frame_type.value] += freq
        
        categories = list(frame_counts.keys())
        values = [frame_counts[c] for c in categories]
        
        title = "Frame Type Distribution"
        if period:
            title += f" ({period.value})"
        
        return DistributionData(
            title=title,
            categories=categories,
            values=values,
            chart_type=ChartType.BAR
        )
    
    def generate_diachronic_heatmap(
        self,
        lexicon: ValencyLexicon,
        lemmas: Optional[List[str]] = None,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Generate heatmap data for diachronic analysis"""
        periods = [p.period_enum.value for p in self.period_system.get_ordered_periods()]
        
        if lemmas is None:
            sorted_entries = sorted(
                lexicon.entries.values(),
                key=lambda e: e.total_frequency,
                reverse=True
            )
            lemmas = [e.lemma for e in sorted_entries[:top_n]]
        
        z_data = []
        for lemma in lemmas:
            entry = lexicon.get_entry(lemma)
            if not entry:
                z_data.append([0] * len(periods))
                continue
            
            row = []
            for period in periods:
                freq = 0
                for pattern in entry.patterns.values():
                    freq += pattern.period_distribution.get(period, 0)
                row.append(freq)
            z_data.append(row)
        
        return {
            "type": "heatmap",
            "title": "Diachronic Frequency Heatmap",
            "x": periods,
            "y": lemmas,
            "z": z_data,
            "colorscale": "Viridis"
        }
    
    def generate_change_network(
        self,
        changes: List[DiachronicChange]
    ) -> NetworkData:
        """Generate network visualization of changes"""
        nodes = []
        edges = []
        node_ids = set()
        
        for change in changes:
            if change.lemma not in node_ids:
                nodes.append(NetworkNode(
                    id=change.lemma,
                    label=change.lemma,
                    group=change.change_type.value,
                    size=len(change.metrics)
                ))
                node_ids.add(change.lemma)
            
            for related in change.related_changes:
                if related not in node_ids:
                    nodes.append(NetworkNode(
                        id=related,
                        label=related,
                        group="related"
                    ))
                    node_ids.add(related)
                
                edges.append(NetworkEdge(
                    source=change.lemma,
                    target=related,
                    label=change.change_type.value
                ))
        
        return NetworkData(
            title="Diachronic Change Network",
            nodes=nodes,
            edges=edges
        )
    
    def generate_sankey_diagram(
        self,
        lexicon: ValencyLexicon,
        lemma: str,
        periods: Optional[List[Period]] = None
    ) -> Dict[str, Any]:
        """Generate Sankey diagram for pattern flow"""
        entry = lexicon.get_entry(lemma)
        if not entry:
            return {}
        
        if periods is None:
            periods = [p.period_enum for p in self.period_system.get_ordered_periods()]
        
        nodes = []
        links = []
        node_map = {}
        
        for i, period in enumerate(periods):
            for pattern_key, pattern in entry.patterns.items():
                freq = pattern.period_distribution.get(period.value, 0)
                if freq > 0:
                    node_id = f"{period.value}_{pattern_key}"
                    node_map[node_id] = len(nodes)
                    nodes.append({
                        "name": f"{pattern.pattern_string} ({period.value})"
                    })
        
        for i in range(len(periods) - 1):
            curr_period = periods[i]
            next_period = periods[i + 1]
            
            for pattern_key, pattern in entry.patterns.items():
                curr_freq = pattern.period_distribution.get(curr_period.value, 0)
                next_freq = pattern.period_distribution.get(next_period.value, 0)
                
                if curr_freq > 0 and next_freq > 0:
                    source_id = f"{curr_period.value}_{pattern_key}"
                    target_id = f"{next_period.value}_{pattern_key}"
                    
                    if source_id in node_map and target_id in node_map:
                        links.append({
                            "source": node_map[source_id],
                            "target": node_map[target_id],
                            "value": min(curr_freq, next_freq)
                        })
        
        return {
            "type": "sankey",
            "title": f"Pattern Flow: {lemma}",
            "nodes": nodes,
            "links": links
        }


def generate_timeline_data(
    lexicon: ValencyLexicon,
    lemmas: Optional[List[str]] = None,
    language: Language = Language.ANCIENT_GREEK
) -> TimeSeriesData:
    """Generate timeline data"""
    generator = VisualizationGenerator(language=language)
    return generator.generate_frequency_timeline(lexicon, lemmas)


def generate_frequency_chart_data(
    lexicon: ValencyLexicon,
    period: Optional[Period] = None,
    language: Language = Language.ANCIENT_GREEK
) -> DistributionData:
    """Generate frequency chart data"""
    generator = VisualizationGenerator(language=language)
    return generator.generate_frame_type_distribution(lexicon, period)


def generate_distribution_data(
    entry: LexiconEntry,
    period: Optional[Period] = None,
    language: Language = Language.ANCIENT_GREEK
) -> DistributionData:
    """Generate distribution data"""
    generator = VisualizationGenerator(language=language)
    return generator.generate_pattern_distribution(entry, period)


def generate_network_data(
    changes: List[DiachronicChange],
    language: Language = Language.ANCIENT_GREEK
) -> NetworkData:
    """Generate network data"""
    generator = VisualizationGenerator(language=language)
    return generator.generate_change_network(changes)
