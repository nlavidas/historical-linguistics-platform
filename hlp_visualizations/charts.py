"""
Chart Generator - Create charts for linguistic data visualization

This module generates JSON data for charts that can be rendered
by JavaScript charting libraries (Chart.js, D3.js, etc.)

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import Counter
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    SCATTER = "scatter"
    RADAR = "radar"
    POLAR = "polar"
    BUBBLE = "bubble"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SANKEY = "sankey"


@dataclass
class ChartDataset:
    label: str
    data: List[Any]
    backgroundColor: Optional[List[str]] = None
    borderColor: Optional[str] = None
    borderWidth: int = 1
    fill: bool = False
    tension: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'label': self.label,
            'data': self.data,
            'borderWidth': self.borderWidth,
            'fill': self.fill,
            'tension': self.tension,
        }
        if self.backgroundColor:
            result['backgroundColor'] = self.backgroundColor
        if self.borderColor:
            result['borderColor'] = self.borderColor
        return result


@dataclass
class ChartData:
    chart_type: ChartType
    title: str
    labels: List[str]
    datasets: List[ChartDataset]
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_chartjs_config(self) -> Dict[str, Any]:
        return {
            'type': self.chart_type.value,
            'data': {
                'labels': self.labels,
                'datasets': [ds.to_dict() for ds in self.datasets],
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': self.title,
                    },
                    'legend': {
                        'display': True,
                        'position': 'top',
                    },
                },
                **self.options,
            },
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_chartjs_config(), indent=2)


class ChartGenerator:
    
    COLORS = [
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(199, 199, 199, 0.8)',
        'rgba(83, 102, 255, 0.8)',
        'rgba(255, 99, 255, 0.8)',
        'rgba(99, 255, 132, 0.8)',
    ]
    
    BORDER_COLORS = [
        'rgba(54, 162, 235, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(199, 199, 199, 1)',
        'rgba(83, 102, 255, 1)',
        'rgba(255, 99, 255, 1)',
        'rgba(99, 255, 132, 1)',
    ]
    
    def __init__(self, db_path: str = "data/corpus_platform.db"):
        self.db_path = Path(db_path)
    
    def generate_language_distribution(self) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT language, COUNT(*) as count
                    FROM corpus_items
                    WHERE language IS NOT NULL
                    GROUP BY language
                    ORDER BY count DESC
                """)
                
                for row in cursor:
                    lang_code = row[0]
                    lang_name = self._get_language_name(lang_code)
                    labels.append(lang_name)
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating language distribution: {e}")
        
        return ChartData(
            chart_type=ChartType.PIE,
            title='Corpus Language Distribution',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Texts',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                )
            ],
        )
    
    def generate_period_distribution(self) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT period, COUNT(*) as count
                    FROM corpus_items
                    WHERE period IS NOT NULL
                    GROUP BY period
                    ORDER BY count DESC
                """)
                
                for row in cursor:
                    labels.append(row[0])
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating period distribution: {e}")
        
        return ChartData(
            chart_type=ChartType.BAR,
            title='Texts by Historical Period',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Number of Texts',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
        )
    
    def generate_genre_distribution(self) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT genre, COUNT(*) as count
                    FROM corpus_items
                    WHERE genre IS NOT NULL AND genre != ''
                    GROUP BY genre
                    ORDER BY count DESC
                """)
                
                for row in cursor:
                    labels.append(row[0].title())
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating genre distribution: {e}")
        
        return ChartData(
            chart_type=ChartType.DOUGHNUT,
            title='Texts by Genre',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Texts',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                )
            ],
        )
    
    def generate_word_count_histogram(self, bins: int = 10) -> ChartData:
        word_counts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT word_count
                    FROM corpus_items
                    WHERE word_count IS NOT NULL AND word_count > 0
                """)
                
                word_counts = [row[0] for row in cursor]
        except Exception as e:
            logger.error(f"Error generating word count histogram: {e}")
        
        if not word_counts:
            return ChartData(
                chart_type=ChartType.BAR,
                title='Word Count Distribution',
                labels=[],
                datasets=[],
            )
        
        min_count = min(word_counts)
        max_count = max(word_counts)
        bin_size = (max_count - min_count) / bins
        
        labels = []
        data = [0] * bins
        
        for i in range(bins):
            bin_start = min_count + i * bin_size
            bin_end = min_count + (i + 1) * bin_size
            labels.append(f"{int(bin_start/1000)}k-{int(bin_end/1000)}k")
        
        for count in word_counts:
            bin_index = min(int((count - min_count) / bin_size), bins - 1)
            data[bin_index] += 1
        
        return ChartData(
            chart_type=ChartType.BAR,
            title='Word Count Distribution',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Number of Texts',
                    data=data,
                    backgroundColor=[self.COLORS[0]] * bins,
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
        )
    
    def generate_collection_timeline(self) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DATE(date_added) as date, COUNT(*) as count
                    FROM corpus_items
                    WHERE date_added IS NOT NULL
                    GROUP BY DATE(date_added)
                    ORDER BY date
                """)
                
                cumulative = 0
                for row in cursor:
                    labels.append(row[0])
                    cumulative += row[1]
                    data.append(cumulative)
        except Exception as e:
            logger.error(f"Error generating collection timeline: {e}")
        
        return ChartData(
            chart_type=ChartType.LINE,
            title='Corpus Growth Over Time',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Total Texts',
                    data=data,
                    borderColor=self.BORDER_COLORS[0],
                    backgroundColor=self.COLORS[0],
                    fill=True,
                    tension=0.4,
                )
            ],
        )
    
    def generate_diachronic_stage_chart(self) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT diachronic_stage, COUNT(*) as count
                    FROM corpus_items
                    WHERE diachronic_stage IS NOT NULL AND diachronic_stage != ''
                    GROUP BY diachronic_stage
                    ORDER BY count DESC
                """)
                
                for row in cursor:
                    labels.append(row[0])
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating diachronic stage chart: {e}")
        
        return ChartData(
            chart_type=ChartType.BAR,
            title='Texts by Diachronic Stage',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Number of Texts',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
            options={
                'indexAxis': 'y',
            },
        )
    
    def generate_author_chart(self, limit: int = 15) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT author, COUNT(*) as count
                    FROM corpus_items
                    WHERE author IS NOT NULL AND author != ''
                    GROUP BY author
                    ORDER BY count DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor:
                    labels.append(row[0])
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating author chart: {e}")
        
        return ChartData(
            chart_type=ChartType.BAR,
            title=f'Top {limit} Authors by Text Count',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Number of Texts',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
            options={
                'indexAxis': 'y',
            },
        )
    
    def generate_translation_status_chart(self) -> ChartData:
        labels = ['Original', 'Translation', 'Retelling']
        data = [0, 0, 0]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                data[0] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items
                    WHERE is_retranslation = 0 AND is_retelling = 0
                """).fetchone()[0]
                
                data[1] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items
                    WHERE is_retranslation = 1
                """).fetchone()[0]
                
                data[2] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items
                    WHERE is_retelling = 1
                """).fetchone()[0]
        except Exception as e:
            logger.error(f"Error generating translation status chart: {e}")
        
        return ChartData(
            chart_type=ChartType.PIE,
            title='Original vs Translation vs Retelling',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Texts',
                    data=data,
                    backgroundColor=self.COLORS[:3],
                )
            ],
        )
    
    def generate_annotation_status_chart(self) -> ChartData:
        labels = ['Collected', 'Processed', 'Annotated', 'Has Treebank']
        data = [0, 0, 0, 0]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                data[0] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items WHERE status = 'collected'
                """).fetchone()[0]
                
                data[1] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items WHERE status = 'processed'
                """).fetchone()[0]
                
                data[2] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items WHERE status = 'annotated'
                """).fetchone()[0]
                
                data[3] = conn.execute("""
                    SELECT COUNT(*) FROM corpus_items WHERE has_treebank = 1
                """).fetchone()[0]
        except Exception as e:
            logger.error(f"Error generating annotation status chart: {e}")
        
        return ChartData(
            chart_type=ChartType.BAR,
            title='Text Processing Status',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Number of Texts',
                    data=data,
                    backgroundColor=self.COLORS[:4],
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
        )
    
    def generate_valency_pattern_chart(self, limit: int = 20) -> ChartData:
        labels = []
        data = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern, SUM(count) as total
                    FROM valency_patterns
                    GROUP BY pattern
                    ORDER BY total DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor:
                    labels.append(row[0])
                    data.append(row[1])
        except Exception as e:
            logger.error(f"Error generating valency pattern chart: {e}")
        
        return ChartData(
            chart_type=ChartType.BAR,
            title=f'Top {limit} Valency Patterns',
            labels=labels,
            datasets=[
                ChartDataset(
                    label='Occurrences',
                    data=data,
                    backgroundColor=self.COLORS[:len(data)],
                    borderColor=self.BORDER_COLORS[0],
                )
            ],
            options={
                'indexAxis': 'y',
            },
        )
    
    def generate_all_charts(self) -> Dict[str, ChartData]:
        return {
            'language_distribution': self.generate_language_distribution(),
            'period_distribution': self.generate_period_distribution(),
            'genre_distribution': self.generate_genre_distribution(),
            'word_count_histogram': self.generate_word_count_histogram(),
            'collection_timeline': self.generate_collection_timeline(),
            'diachronic_stage': self.generate_diachronic_stage_chart(),
            'author_chart': self.generate_author_chart(),
            'translation_status': self.generate_translation_status_chart(),
            'annotation_status': self.generate_annotation_status_chart(),
            'valency_patterns': self.generate_valency_pattern_chart(),
        }
    
    def _get_language_name(self, code: str) -> str:
        language_names = {
            'grc': 'Ancient Greek',
            'el': 'Modern Greek',
            'lat': 'Latin',
            'en': 'English',
            'ang': 'Old English',
            'enm': 'Middle English',
            'de': 'German',
            'fr': 'French',
            'it': 'Italian',
            'es': 'Spanish',
            'ru': 'Russian',
            'cu': 'Old Church Slavonic',
            'got': 'Gothic',
            'xcl': 'Classical Armenian',
            'sa': 'Sanskrit',
            'heb': 'Hebrew',
            'ara': 'Arabic',
            'cop': 'Coptic',
            'syc': 'Syriac',
        }
        return language_names.get(code, code.upper())
