"""
Visualization Engine - PROIEL/Syntacticus Style
Dependency Trees, Morphological Analysis, Diachronic Charts
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# COLOR SCHEMES
# =============================================================================

# PROIEL-style colors for POS tags
POS_COLORS = {
    'NOUN': '#4299e1',      # Blue
    'VERB': '#48bb78',      # Green
    'ADJ': '#ed8936',       # Orange
    'ADV': '#9f7aea',       # Purple
    'PRON': '#f56565',      # Red
    'DET': '#38b2ac',       # Teal
    'ADP': '#ed64a6',       # Pink
    'CCONJ': '#667eea',     # Indigo
    'SCONJ': '#667eea',     # Indigo
    'PART': '#a0aec0',      # Gray
    'INTJ': '#fbd38d',      # Yellow
    'NUM': '#fc8181',       # Light red
    'PUNCT': '#cbd5e0',     # Light gray
    'AUX': '#68d391',       # Light green
    'X': '#e2e8f0',         # Very light gray
}

# Period colors
PERIOD_COLORS = {
    'archaic': '#fef3c7',       # Amber
    'classical': '#dbeafe',     # Blue
    'hellenistic': '#d1fae5',   # Green
    'koine': '#ede9fe',         # Purple
    'late_antique': '#fce7f3',  # Pink
    'byzantine': '#fee2e2',     # Red
    'medieval': '#fef3c7',      # Amber
    'early_modern': '#e0e7ff',  # Indigo
    'modern': '#f0fdf4',        # Light green
}

# Dependency relation colors
RELATION_COLORS = {
    'nsubj': '#ef4444',     # Red - Subject
    'obj': '#f97316',       # Orange - Object
    'iobj': '#eab308',      # Yellow - Indirect object
    'obl': '#22c55e',       # Green - Oblique
    'advmod': '#06b6d4',    # Cyan - Adverbial
    'amod': '#3b82f6',      # Blue - Adjectival
    'nmod': '#8b5cf6',      # Purple - Nominal modifier
    'det': '#ec4899',       # Pink - Determiner
    'case': '#6366f1',      # Indigo - Case marker
    'conj': '#14b8a6',      # Teal - Conjunction
    'cc': '#f43f5e',        # Rose - Coordinating conjunction
    'punct': '#94a3b8',     # Gray - Punctuation
    'root': '#1f2937',      # Dark - Root
    'aux': '#84cc16',       # Lime - Auxiliary
    'cop': '#84cc16',       # Lime - Copula
    'mark': '#a855f7',      # Fuchsia - Marker
    'xcomp': '#0ea5e9',     # Sky - Open complement
    'ccomp': '#0ea5e9',     # Sky - Clausal complement
    'advcl': '#10b981',     # Emerald - Adverbial clause
    'acl': '#10b981',       # Emerald - Adnominal clause
}

# =============================================================================
# SVG DEPENDENCY TREE GENERATOR
# =============================================================================

class DependencyTreeSVG:
    """Generate SVG visualization of dependency trees"""
    
    def __init__(self, width: int = 1200, token_spacing: int = 100, 
                 level_height: int = 60, margin: int = 50):
        self.width = width
        self.token_spacing = token_spacing
        self.level_height = level_height
        self.margin = margin
        
    def generate(self, tokens: List[Dict], title: str = "") -> str:
        """Generate SVG for a sentence's dependency tree"""
        n_tokens = len(tokens)
        
        # Calculate dimensions
        content_width = n_tokens * self.token_spacing
        actual_width = max(self.width, content_width + 2 * self.margin)
        
        # Calculate max depth for height
        max_depth = self._calculate_max_depth(tokens)
        height = (max_depth + 3) * self.level_height + 2 * self.margin
        
        # Start SVG
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {actual_width} {height}" '
            f'width="{actual_width}" height="{height}">',
            '<defs>',
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>',
            '  </marker>',
            '</defs>',
            f'<rect width="{actual_width}" height="{height}" fill="white"/>',
        ]
        
        # Add title
        if title:
            svg_parts.append(
                f'<text x="{actual_width/2}" y="30" text-anchor="middle" '
                f'font-size="16" font-weight="bold" fill="#1e293b">{self._escape_xml(title)}</text>'
            )
        
        # Calculate token positions
        start_x = self.margin + (actual_width - content_width) / 2
        token_y = height - self.margin - 80
        
        positions = {}
        for i, token in enumerate(tokens):
            x = start_x + i * self.token_spacing
            positions[token.get('id', i+1)] = (x, token_y)
        
        # Draw arcs (dependencies)
        arc_y_base = token_y - 30
        for token in tokens:
            token_id = token.get('id', 0)
            head_id = token.get('head', 0)
            relation = token.get('relation', '')
            
            if head_id > 0 and head_id in positions and token_id in positions:
                x1, _ = positions[token_id]
                x2, _ = positions[head_id]
                
                # Calculate arc height based on distance
                distance = abs(x2 - x1)
                arc_height = min(distance * 0.4, max_depth * self.level_height * 0.8)
                
                # Get color for relation
                color = RELATION_COLORS.get(relation, '#64748b')
                
                # Draw arc
                if x1 < x2:
                    # Arc going right
                    svg_parts.append(
                        f'<path d="M {x1} {arc_y_base} Q {(x1+x2)/2} {arc_y_base - arc_height} {x2} {arc_y_base}" '
                        f'fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrowhead)"/>'
                    )
                else:
                    # Arc going left
                    svg_parts.append(
                        f'<path d="M {x1} {arc_y_base} Q {(x1+x2)/2} {arc_y_base - arc_height} {x2} {arc_y_base}" '
                        f'fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrowhead)"/>'
                    )
                
                # Add relation label
                label_x = (x1 + x2) / 2
                label_y = arc_y_base - arc_height / 2 - 5
                svg_parts.append(
                    f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                    f'font-size="10" fill="{color}" font-weight="500">{relation}</text>'
                )
        
        # Draw tokens
        for i, token in enumerate(tokens):
            token_id = token.get('id', i+1)
            x, y = positions.get(token_id, (start_x + i * self.token_spacing, token_y))
            
            form = token.get('form', '')
            lemma = token.get('lemma', '')
            pos = token.get('pos', 'X')
            
            # Get color for POS
            color = POS_COLORS.get(pos, '#e2e8f0')
            
            # Token box
            box_width = 80
            box_height = 60
            svg_parts.append(
                f'<rect x="{x - box_width/2}" y="{y}" width="{box_width}" height="{box_height}" '
                f'rx="5" fill="{color}" stroke="#64748b" stroke-width="1"/>'
            )
            
            # Form (word)
            svg_parts.append(
                f'<text x="{x}" y="{y + 20}" text-anchor="middle" '
                f'font-size="14" font-weight="bold" fill="#1e293b">{self._escape_xml(form)}</text>'
            )
            
            # Lemma
            svg_parts.append(
                f'<text x="{x}" y="{y + 35}" text-anchor="middle" '
                f'font-size="10" fill="#475569">{self._escape_xml(lemma)}</text>'
            )
            
            # POS tag
            svg_parts.append(
                f'<text x="{x}" y="{y + 50}" text-anchor="middle" '
                f'font-size="10" font-weight="500" fill="#1e293b">{pos}</text>'
            )
            
            # Token ID
            svg_parts.append(
                f'<text x="{x}" y="{y - 5}" text-anchor="middle" '
                f'font-size="9" fill="#94a3b8">{token_id}</text>'
            )
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _calculate_max_depth(self, tokens: List[Dict]) -> int:
        """Calculate maximum dependency depth"""
        # Build adjacency list
        children = {}
        root = None
        
        for token in tokens:
            token_id = token.get('id', 0)
            head_id = token.get('head', 0)
            
            if head_id == 0:
                root = token_id
            else:
                if head_id not in children:
                    children[head_id] = []
                children[head_id].append(token_id)
        
        # Calculate depth with BFS
        if root is None:
            return 1
            
        max_depth = 0
        queue = [(root, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            
            for child in children.get(node, []):
                queue.append((child, depth + 1))
        
        return max_depth
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))


# =============================================================================
# HTML INTERLINEAR GLOSSING
# =============================================================================

class InterlinearGloss:
    """Generate interlinear glossing HTML"""
    
    @classmethod
    def generate(cls, tokens: List[Dict], include_morph: bool = True) -> str:
        """Generate interlinear gloss HTML"""
        html_parts = [
            '<div class="interlinear-container" style="font-family: \'Gentium Plus\', serif;">',
        ]
        
        # Word line
        html_parts.append('<div class="word-line" style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 10px;">')
        for token in tokens:
            if token.get('pos') == 'PUNCT':
                continue
                
            form = token.get('form', '')
            pos = token.get('pos', 'X')
            color = POS_COLORS.get(pos, '#e2e8f0')
            
            html_parts.append(f'''
                <div class="word-unit" style="text-align: center;">
                    <div class="form" style="font-size: 1.4em; font-weight: bold; color: #1e293b; 
                         border-bottom: 3px solid {color}; padding-bottom: 5px;">
                        {form}
                    </div>
            ''')
            
            # Lemma line
            lemma = token.get('lemma', '')
            html_parts.append(f'''
                    <div class="lemma" style="font-size: 0.9em; color: #64748b; margin-top: 3px;">
                        {lemma}
                    </div>
            ''')
            
            # POS line
            html_parts.append(f'''
                    <div class="pos" style="font-size: 0.8em; font-weight: 500; color: {color}; margin-top: 2px;">
                        {pos}
                    </div>
            ''')
            
            # Morphology line
            if include_morph:
                morph = token.get('morph', '_')
                if morph and morph != '_':
                    html_parts.append(f'''
                        <div class="morph" style="font-size: 0.7em; color: #94a3b8; margin-top: 2px;">
                            {morph}
                        </div>
                    ''')
            
            # Gloss line (if available)
            gloss = token.get('gloss', '')
            if gloss:
                html_parts.append(f'''
                    <div class="gloss" style="font-size: 0.8em; color: #475569; margin-top: 3px; font-style: italic;">
                        '{gloss}'
                    </div>
                ''')
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)


# =============================================================================
# MORPHOLOGICAL ANALYSIS TABLE
# =============================================================================

class MorphologyTable:
    """Generate morphological analysis tables"""
    
    @classmethod
    def generate_html(cls, tokens: List[Dict]) -> str:
        """Generate HTML table for morphological analysis"""
        html_parts = [
            '<table class="morph-table" style="width: 100%; border-collapse: collapse; font-family: sans-serif;">',
            '<thead>',
            '<tr style="background: #f1f5f9;">',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">ID</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">Form</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">Lemma</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">POS</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">Morphology</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">Head</th>',
            '<th style="padding: 10px; border: 1px solid #e2e8f0; text-align: left;">Relation</th>',
            '</tr>',
            '</thead>',
            '<tbody>',
        ]
        
        for token in tokens:
            pos = token.get('pos', 'X')
            color = POS_COLORS.get(pos, '#e2e8f0')
            
            html_parts.append(f'''
                <tr>
                    <td style="padding: 8px; border: 1px solid #e2e8f0;">{token.get('id', '')}</td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0; font-weight: bold;">{token.get('form', '')}</td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0; font-style: italic;">{token.get('lemma', '')}</td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0;">
                        <span style="background: {color}; padding: 2px 8px; border-radius: 4px;">{pos}</span>
                    </td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0; font-size: 0.9em;">{token.get('morph', '_')}</td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0;">{token.get('head', 0)}</td>
                    <td style="padding: 8px; border: 1px solid #e2e8f0;">{token.get('relation', '')}</td>
                </tr>
            ''')
        
        html_parts.append('</tbody>')
        html_parts.append('</table>')
        
        return '\n'.join(html_parts)


# =============================================================================
# DIACHRONIC VISUALIZATION
# =============================================================================

class DiachronicCharts:
    """Generate diachronic analysis visualizations"""
    
    PERIODS_ORDER = [
        'archaic', 'classical', 'hellenistic', 'koine', 
        'late_antique', 'byzantine', 'medieval', 'early_modern', 'modern'
    ]
    
    PERIOD_LABELS = {
        'archaic': 'Archaic\n(800-500 BCE)',
        'classical': 'Classical\n(500-323 BCE)',
        'hellenistic': 'Hellenistic\n(323-31 BCE)',
        'koine': 'Koine\n(31 BCE-300 CE)',
        'late_antique': 'Late Antique\n(300-600 CE)',
        'byzantine': 'Byzantine\n(600-1453 CE)',
        'medieval': 'Medieval\n(1100-1453 CE)',
        'early_modern': 'Early Modern\n(1453-1830 CE)',
        'modern': 'Modern\n(1830-present)',
    }
    
    @classmethod
    def generate_period_distribution_svg(cls, data: Dict[str, int], 
                                         width: int = 800, height: int = 400) -> str:
        """Generate SVG bar chart for period distribution"""
        margin = {'top': 40, 'right': 30, 'bottom': 80, 'left': 60}
        chart_width = width - margin['left'] - margin['right']
        chart_height = height - margin['top'] - margin['bottom']
        
        # Filter and order periods
        periods = [p for p in cls.PERIODS_ORDER if p in data]
        values = [data[p] for p in periods]
        
        if not values:
            return '<svg><text>No data available</text></svg>'
        
        max_val = max(values)
        bar_width = chart_width / len(periods) * 0.8
        bar_gap = chart_width / len(periods) * 0.2
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            
            # Title
            f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">Token Distribution by Period</text>',
            
            # Y-axis
            f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" stroke="#e2e8f0" stroke-width="1"/>',
        ]
        
        # Y-axis labels
        for i in range(5):
            y = margin['top'] + chart_height * (1 - i/4)
            val = int(max_val * i / 4)
            svg_parts.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" font-size="10" fill="#64748b">{val:,}</text>')
            svg_parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>')
        
        # Bars
        for i, (period, value) in enumerate(zip(periods, values)):
            x = margin['left'] + i * (bar_width + bar_gap) + bar_gap/2
            bar_height = (value / max_val) * chart_height if max_val > 0 else 0
            y = margin['top'] + chart_height - bar_height
            
            color = PERIOD_COLORS.get(period, '#e2e8f0')
            
            # Bar
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                f'fill="{color}" stroke="#64748b" stroke-width="1" rx="2"/>'
            )
            
            # Value label
            svg_parts.append(
                f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" '
                f'font-size="10" fill="#1e293b">{value:,}</text>'
            )
            
            # Period label
            label = period.replace('_', '\n')
            svg_parts.append(
                f'<text x="{x + bar_width/2}" y="{height - margin["bottom"] + 15}" '
                f'text-anchor="middle" font-size="9" fill="#64748b">{label}</text>'
            )
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    @classmethod
    def generate_pos_distribution_svg(cls, data: Dict[str, int],
                                      width: int = 600, height: int = 400) -> str:
        """Generate SVG pie chart for POS distribution"""
        cx, cy = width / 2, height / 2
        radius = min(width, height) / 2 - 60
        
        total = sum(data.values())
        if total == 0:
            return '<svg><text>No data available</text></svg>'
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">POS Distribution</text>',
        ]
        
        # Sort by value
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        
        # Draw pie slices
        start_angle = 0
        for pos, count in sorted_data:
            if count == 0:
                continue
                
            angle = (count / total) * 360
            end_angle = start_angle + angle
            
            # Calculate arc
            start_rad = start_angle * 3.14159 / 180
            end_rad = end_angle * 3.14159 / 180
            
            x1 = cx + radius * math.cos(start_rad)
            y1 = cy + radius * math.sin(start_rad)
            x2 = cx + radius * math.cos(end_rad)
            y2 = cy + radius * math.sin(end_rad)
            
            large_arc = 1 if angle > 180 else 0
            
            color = POS_COLORS.get(pos, '#e2e8f0')
            
            # Pie slice
            svg_parts.append(
                f'<path d="M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z" '
                f'fill="{color}" stroke="white" stroke-width="2"/>'
            )
            
            # Label
            mid_angle = (start_angle + end_angle) / 2
            mid_rad = mid_angle * 3.14159 / 180
            label_radius = radius * 0.7
            label_x = cx + label_radius * math.cos(mid_rad)
            label_y = cy + label_radius * math.sin(mid_rad)
            
            if angle > 15:  # Only show label if slice is big enough
                pct = count / total * 100
                svg_parts.append(
                    f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                    f'font-size="10" font-weight="bold" fill="#1e293b">{pos}</text>'
                )
                svg_parts.append(
                    f'<text x="{label_x}" y="{label_y + 12}" text-anchor="middle" '
                    f'font-size="9" fill="#64748b">{pct:.1f}%</text>'
                )
            
            start_angle = end_angle
        
        # Legend
        legend_x = width - 100
        legend_y = 50
        for i, (pos, count) in enumerate(sorted_data[:10]):
            color = POS_COLORS.get(pos, '#e2e8f0')
            y = legend_y + i * 20
            svg_parts.append(f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{color}"/>')
            svg_parts.append(f'<text x="{legend_x + 18}" y="{y + 10}" font-size="10" fill="#1e293b">{pos}</text>')
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


# Import math for pie chart
import math


# =============================================================================
# SENTENCE ALIGNMENT VISUALIZATION
# =============================================================================

class SentenceAlignment:
    """Visualize parallel sentence alignments"""
    
    @classmethod
    def generate_html(cls, sentences: List[Dict], languages: List[str]) -> str:
        """Generate HTML for aligned sentences"""
        html_parts = [
            '<div class="alignment-container" style="font-family: sans-serif;">',
        ]
        
        for i, sent_group in enumerate(sentences):
            html_parts.append(f'''
                <div class="sentence-group" style="margin-bottom: 20px; padding: 15px; 
                     background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <div class="sentence-id" style="font-size: 0.8em; color: #64748b; margin-bottom: 10px;">
                        Sentence {i + 1}
                    </div>
            ''')
            
            for lang in languages:
                if lang in sent_group:
                    text = sent_group[lang]
                    lang_label = lang.upper()
                    
                    html_parts.append(f'''
                        <div class="lang-row" style="margin-bottom: 8px;">
                            <span class="lang-label" style="display: inline-block; width: 40px; 
                                   font-weight: bold; color: #3b82f6;">{lang_label}:</span>
                            <span class="lang-text" style="font-size: 1.1em;">{text}</span>
                        </div>
                    ''')
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)


# =============================================================================
# CORPUS STATISTICS DASHBOARD
# =============================================================================

class StatsDashboard:
    """Generate statistics dashboard HTML"""
    
    @classmethod
    def generate_html(cls, stats: Dict) -> str:
        """Generate HTML dashboard for corpus statistics"""
        html_parts = [
            '<div class="stats-dashboard" style="font-family: sans-serif;">',
            
            # Main metrics
            '<div class="metrics-row" style="display: flex; gap: 20px; margin-bottom: 30px;">',
        ]
        
        # Metric cards
        metrics = [
            ('Texts', stats.get('text_count', 0), '#667eea'),
            ('Sentences', stats.get('sentence_count', 0), '#f093fb'),
            ('Tokens', stats.get('token_count', 0), '#4facfe'),
            ('Lemmas', stats.get('lemma_count', 0), '#43e97b'),
        ]
        
        for label, value, color in metrics:
            html_parts.append(f'''
                <div class="metric-card" style="flex: 1; padding: 20px; border-radius: 10px; 
                     background: linear-gradient(135deg, {color} 0%, {color}99 100%); color: white; text-align: center;">
                    <div class="metric-value" style="font-size: 2em; font-weight: bold;">{value:,}</div>
                    <div class="metric-label" style="font-size: 0.9em; opacity: 0.9;">{label}</div>
                </div>
            ''')
        
        html_parts.append('</div>')
        
        # Period breakdown
        if 'by_period' in stats:
            html_parts.append('''
                <div class="period-section" style="margin-bottom: 30px;">
                    <h3 style="color: #1e293b; margin-bottom: 15px;">By Period</h3>
                    <div class="period-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
            ''')
            
            for period, data in stats['by_period'].items():
                color = PERIOD_COLORS.get(period, '#e2e8f0')
                texts = data.get('texts', 0)
                tokens = data.get('tokens', 0)
                
                html_parts.append(f'''
                    <div class="period-card" style="padding: 15px; border-radius: 8px; 
                         background: {color}; border: 1px solid #e2e8f0;">
                        <div class="period-name" style="font-weight: bold; color: #1e293b; margin-bottom: 5px;">
                            {period.replace('_', ' ').title()}
                        </div>
                        <div class="period-stats" style="font-size: 0.9em; color: #64748b;">
                            {texts:,} texts · {tokens:,} tokens
                        </div>
                    </div>
                ''')
            
            html_parts.append('</div></div>')
        
        # POS distribution
        if 'pos_distribution' in stats:
            html_parts.append('''
                <div class="pos-section" style="margin-bottom: 30px;">
                    <h3 style="color: #1e293b; margin-bottom: 15px;">POS Distribution</h3>
                    <div class="pos-bars" style="display: flex; flex-direction: column; gap: 8px;">
            ''')
            
            total_pos = sum(stats['pos_distribution'].values())
            sorted_pos = sorted(stats['pos_distribution'].items(), key=lambda x: x[1], reverse=True)
            
            for pos, count in sorted_pos[:10]:
                pct = (count / total_pos * 100) if total_pos > 0 else 0
                color = POS_COLORS.get(pos, '#e2e8f0')
                
                html_parts.append(f'''
                    <div class="pos-bar-row" style="display: flex; align-items: center; gap: 10px;">
                        <div class="pos-label" style="width: 60px; font-weight: 500;">{pos}</div>
                        <div class="pos-bar-container" style="flex: 1; height: 24px; background: #f1f5f9; border-radius: 4px; overflow: hidden;">
                            <div class="pos-bar" style="width: {pct}%; height: 100%; background: {color};"></div>
                        </div>
                        <div class="pos-count" style="width: 80px; text-align: right; font-size: 0.9em; color: #64748b;">
                            {count:,} ({pct:.1f}%)
                        </div>
                    </div>
                ''')
            
            html_parts.append('</div></div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_sentence_visualization(tokens: List[Dict], output_path: str, 
                                  format: str = 'svg') -> str:
    """Export sentence visualization to file"""
    output_path = Path(output_path)
    
    if format == 'svg':
        generator = DependencyTreeSVG()
        content = generator.generate(tokens)
        output_path = output_path.with_suffix('.svg')
    elif format == 'html':
        content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Sentence Visualization</title>
            <style>
                body {{ font-family: 'Gentium Plus', serif; padding: 20px; }}
            </style>
        </head>
        <body>
            <h2>Dependency Tree</h2>
            {DependencyTreeSVG().generate(tokens)}
            
            <h2>Interlinear Gloss</h2>
            {InterlinearGloss.generate(tokens)}
            
            <h2>Morphological Analysis</h2>
            {MorphologyTable.generate_html(tokens)}
        </body>
        </html>
        '''
        output_path = output_path.with_suffix('.html')
    else:
        raise ValueError(f"Unknown format: {format}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')
    
    return str(output_path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test visualization
    test_tokens = [
        {'id': 1, 'form': 'Ἐν', 'lemma': 'ἐν', 'pos': 'ADP', 'morph': '_', 'head': 2, 'relation': 'case'},
        {'id': 2, 'form': 'ἀρχῇ', 'lemma': 'ἀρχή', 'pos': 'NOUN', 'morph': 'Case=Dat|Gender=Fem|Number=Sing', 'head': 3, 'relation': 'obl'},
        {'id': 3, 'form': 'ἦν', 'lemma': 'εἰμί', 'pos': 'AUX', 'morph': 'Mood=Ind|Number=Sing|Person=3|Tense=Past|Voice=Act', 'head': 0, 'relation': 'root'},
        {'id': 4, 'form': 'ὁ', 'lemma': 'ὁ', 'pos': 'DET', 'morph': 'Case=Nom|Gender=Masc|Number=Sing', 'head': 5, 'relation': 'det'},
        {'id': 5, 'form': 'λόγος', 'lemma': 'λόγος', 'pos': 'NOUN', 'morph': 'Case=Nom|Gender=Masc|Number=Sing', 'head': 3, 'relation': 'nsubj'},
    ]
    
    # Generate SVG
    svg_gen = DependencyTreeSVG()
    svg = svg_gen.generate(test_tokens, "John 1:1 - ἐν ἀρχῇ ἦν ὁ λόγος")
    
    print("Generated SVG visualization")
    print(f"SVG length: {len(svg)} characters")
    
    # Generate interlinear
    interlinear = InterlinearGloss.generate(test_tokens)
    print(f"\nInterlinear HTML length: {len(interlinear)} characters")
    
    # Generate morphology table
    morph_table = MorphologyTable.generate_html(test_tokens)
    print(f"Morphology table HTML length: {len(morph_table)} characters")
