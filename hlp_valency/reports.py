"""
HLP Valency Reports - Generate Valency Analysis Reports

This module provides utilities for generating comprehensive reports
on valency patterns, including diachronic analysis and comparisons.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from enum import Enum

from hlp_valency.pattern_extractor import (
    ExtractedFrame, Argument, ArgumentType, FrameType, ExtractionResult
)
from hlp_valency.pattern_normalization import (
    NormalizedPattern, NormalizedArgument
)
from hlp_valency.lexicon_builder import (
    ValencyLexicon, LexiconEntry, PatternEntry
)
from hlp_core.models import Language, Period

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TSV = "tsv"
    LATEX = "latex"


class ReportType(Enum):
    """Types of reports"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    DIACHRONIC = "diachronic"
    COMPARISON = "comparison"
    STATISTICAL = "statistical"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType = ReportType.SUMMARY
    output_format: ReportFormat = ReportFormat.JSON
    
    include_examples: bool = True
    max_examples: int = 5
    
    include_statistics: bool = True
    
    include_visualizations: bool = False
    
    include_period_analysis: bool = True
    
    min_frequency: int = 1
    
    top_n_patterns: int = 20
    
    language: str = "en"
    
    title: Optional[str] = None
    author: Optional[str] = None
    
    custom_sections: List[str] = field(default_factory=list)


@dataclass
class ReportSection:
    """A section of a report"""
    title: str
    content: Any
    section_type: str = "text"
    
    subsections: List[ReportSection] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "subsections": [s.to_dict() for s in self.subsections]
        }


@dataclass
class ValencyReport:
    """A complete valency report"""
    title: str
    report_type: ReportType
    
    sections: List[ReportSection] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.now)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ReportSection):
        """Add a section to the report"""
        self.sections.append(section)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "report_type": self.report_type.value,
            "generated_at": self.generated_at.isoformat(),
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        lines = [f"# {self.title}", ""]
        lines.append(f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        for section in self.sections:
            lines.extend(self._section_to_markdown(section, level=2))
        
        return "\n".join(lines)
    
    def _section_to_markdown(self, section: ReportSection, level: int = 2) -> List[str]:
        """Convert section to Markdown"""
        lines = []
        header = "#" * level
        lines.append(f"{header} {section.title}")
        lines.append("")
        
        if section.section_type == "text":
            lines.append(str(section.content))
        elif section.section_type == "table":
            lines.extend(self._table_to_markdown(section.content))
        elif section.section_type == "list":
            for item in section.content:
                lines.append(f"- {item}")
        elif section.section_type == "statistics":
            for key, value in section.content.items():
                lines.append(f"- **{key}**: {value}")
        
        lines.append("")
        
        for subsection in section.subsections:
            lines.extend(self._section_to_markdown(subsection, level + 1))
        
        return lines
    
    def _table_to_markdown(self, table_data: Dict[str, Any]) -> List[str]:
        """Convert table data to Markdown"""
        lines = []
        
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return lines
    
    def to_html(self) -> str:
        """Convert to HTML format"""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".section { margin-bottom: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.title}</h1>",
            f"<p><em>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</em></p>",
        ]
        
        for section in self.sections:
            lines.extend(self._section_to_html(section, level=2))
        
        lines.extend(["</body>", "</html>"])
        return "\n".join(lines)
    
    def _section_to_html(self, section: ReportSection, level: int = 2) -> List[str]:
        """Convert section to HTML"""
        lines = [f'<div class="section">']
        lines.append(f"<h{level}>{section.title}</h{level}>")
        
        if section.section_type == "text":
            lines.append(f"<p>{section.content}</p>")
        elif section.section_type == "table":
            lines.extend(self._table_to_html(section.content))
        elif section.section_type == "list":
            lines.append("<ul>")
            for item in section.content:
                lines.append(f"<li>{item}</li>")
            lines.append("</ul>")
        elif section.section_type == "statistics":
            lines.append("<dl>")
            for key, value in section.content.items():
                lines.append(f"<dt>{key}</dt><dd>{value}</dd>")
            lines.append("</dl>")
        
        for subsection in section.subsections:
            lines.extend(self._section_to_html(subsection, level + 1))
        
        lines.append("</div>")
        return lines
    
    def _table_to_html(self, table_data: Dict[str, Any]) -> List[str]:
        """Convert table data to HTML"""
        lines = ["<table>"]
        
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])
        
        if headers:
            lines.append("<thead><tr>")
            for header in headers:
                lines.append(f"<th>{header}</th>")
            lines.append("</tr></thead>")
        
        lines.append("<tbody>")
        for row in rows:
            lines.append("<tr>")
            for cell in row:
                lines.append(f"<td>{cell}</td>")
            lines.append("</tr>")
        lines.append("</tbody>")
        
        lines.append("</table>")
        return lines


class ValencyReporter:
    """Generates valency analysis reports"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def generate_summary_report(
        self,
        lexicon: ValencyLexicon,
        title: Optional[str] = None
    ) -> ValencyReport:
        """Generate summary report for lexicon"""
        report = ValencyReport(
            title=title or f"Valency Summary: {lexicon.name}",
            report_type=ReportType.SUMMARY
        )
        
        report.add_section(self._create_overview_section(lexicon))
        
        if self.config.include_statistics:
            report.add_section(self._create_statistics_section(lexicon))
        
        report.add_section(self._create_top_patterns_section(lexicon))
        
        report.add_section(self._create_frame_type_section(lexicon))
        
        return report
    
    def generate_detailed_report(
        self,
        lexicon: ValencyLexicon,
        title: Optional[str] = None
    ) -> ValencyReport:
        """Generate detailed report for lexicon"""
        report = ValencyReport(
            title=title or f"Valency Analysis: {lexicon.name}",
            report_type=ReportType.DETAILED
        )
        
        report.add_section(self._create_overview_section(lexicon))
        
        if self.config.include_statistics:
            report.add_section(self._create_statistics_section(lexicon))
        
        report.add_section(self._create_frame_type_section(lexicon))
        
        report.add_section(self._create_argument_analysis_section(lexicon))
        
        report.add_section(self._create_entries_section(lexicon))
        
        return report
    
    def generate_diachronic_report(
        self,
        lexicon: ValencyLexicon,
        periods: Optional[List[Period]] = None,
        title: Optional[str] = None
    ) -> ValencyReport:
        """Generate diachronic analysis report"""
        report = ValencyReport(
            title=title or f"Diachronic Valency Analysis: {lexicon.name}",
            report_type=ReportType.DIACHRONIC
        )
        
        report.add_section(self._create_overview_section(lexicon))
        
        report.add_section(self._create_period_distribution_section(lexicon, periods))
        
        report.add_section(self._create_diachronic_changes_section(lexicon, periods))
        
        return report
    
    def generate_comparison_report(
        self,
        lexicons: List[ValencyLexicon],
        title: Optional[str] = None
    ) -> ValencyReport:
        """Generate comparison report for multiple lexicons"""
        report = ValencyReport(
            title=title or "Valency Comparison Report",
            report_type=ReportType.COMPARISON
        )
        
        report.add_section(self._create_comparison_overview_section(lexicons))
        
        report.add_section(self._create_shared_entries_section(lexicons))
        
        report.add_section(self._create_pattern_comparison_section(lexicons))
        
        return report
    
    def _create_overview_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create overview section"""
        stats = lexicon.get_statistics()
        
        content = {
            "Lexicon Name": lexicon.name,
            "Language": lexicon.language.value,
            "Version": lexicon.version,
            "Total Entries": stats["entry_count"],
            "Total Patterns": stats["total_patterns"],
            "Total Instances": stats["total_instances"],
            "Source Corpora": ", ".join(lexicon.source_corpora) or "N/A"
        }
        
        return ReportSection(
            title="Overview",
            content=content,
            section_type="statistics"
        )
    
    def _create_statistics_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create statistics section"""
        stats = lexicon.get_statistics()
        
        content = {
            "Average Patterns per Entry": f"{stats['avg_patterns_per_entry']:.2f}",
            "Average Frequency per Pattern": f"{stats['avg_frequency_per_pattern']:.2f}",
        }
        
        for ft, count in stats["frame_type_distribution"].items():
            content[f"Frame Type: {ft}"] = count
        
        return ReportSection(
            title="Statistics",
            content=content,
            section_type="statistics"
        )
    
    def _create_top_patterns_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create top patterns section"""
        all_patterns = []
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                all_patterns.append({
                    "lemma": entry.lemma,
                    "pattern": pattern.pattern_string,
                    "frequency": pattern.frequency,
                    "frame_type": pattern.frame_type.value
                })
        
        all_patterns.sort(key=lambda x: x["frequency"], reverse=True)
        top_patterns = all_patterns[:self.config.top_n_patterns]
        
        table_data = {
            "headers": ["Rank", "Lemma", "Pattern", "Frequency", "Frame Type"],
            "rows": [
                [i + 1, p["lemma"], p["pattern"], p["frequency"], p["frame_type"]]
                for i, p in enumerate(top_patterns)
            ]
        }
        
        return ReportSection(
            title=f"Top {self.config.top_n_patterns} Patterns",
            content=table_data,
            section_type="table"
        )
    
    def _create_frame_type_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create frame type analysis section"""
        frame_type_counts = defaultdict(int)
        frame_type_examples = defaultdict(list)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                frame_type_counts[pattern.frame_type.value] += pattern.frequency
                
                if len(frame_type_examples[pattern.frame_type.value]) < self.config.max_examples:
                    frame_type_examples[pattern.frame_type.value].append(
                        f"{entry.lemma}: {pattern.pattern_string}"
                    )
        
        subsections = []
        for ft, count in sorted(frame_type_counts.items(), key=lambda x: x[1], reverse=True):
            examples = frame_type_examples[ft]
            
            subsection = ReportSection(
                title=ft.replace("_", " ").title(),
                content=f"Total occurrences: {count}",
                section_type="text"
            )
            
            if self.config.include_examples and examples:
                subsection.subsections.append(ReportSection(
                    title="Examples",
                    content=examples,
                    section_type="list"
                ))
            
            subsections.append(subsection)
        
        return ReportSection(
            title="Frame Type Analysis",
            content="Distribution of valency frame types",
            section_type="text",
            subsections=subsections
        )
    
    def _create_argument_analysis_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create argument analysis section"""
        arg_type_counts = defaultdict(int)
        case_counts = defaultdict(int)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                for arg in pattern.arguments:
                    arg_type_counts[arg.arg_type.value] += pattern.frequency
                    if arg.case:
                        case_counts[arg.case.value] += pattern.frequency
        
        content = {
            "headers": ["Argument Type", "Frequency"],
            "rows": [
                [at, count]
                for at, count in sorted(arg_type_counts.items(), key=lambda x: x[1], reverse=True)
            ]
        }
        
        case_section = ReportSection(
            title="Case Distribution",
            content={
                "headers": ["Case", "Frequency"],
                "rows": [
                    [case, count]
                    for case, count in sorted(case_counts.items(), key=lambda x: x[1], reverse=True)
                ]
            },
            section_type="table"
        )
        
        return ReportSection(
            title="Argument Analysis",
            content=content,
            section_type="table",
            subsections=[case_section]
        )
    
    def _create_entries_section(self, lexicon: ValencyLexicon) -> ReportSection:
        """Create entries section"""
        sorted_entries = sorted(
            lexicon.entries.values(),
            key=lambda e: e.total_frequency,
            reverse=True
        )
        
        subsections = []
        for entry in sorted_entries[:50]:
            patterns_content = []
            for pattern in sorted(entry.patterns.values(), key=lambda p: p.frequency, reverse=True):
                patterns_content.append(
                    f"{pattern.pattern_string} (freq: {pattern.frequency})"
                )
            
            subsections.append(ReportSection(
                title=f"{entry.lemma} (total: {entry.total_frequency})",
                content=patterns_content,
                section_type="list"
            ))
        
        return ReportSection(
            title="Lexicon Entries",
            content=f"Showing top 50 entries by frequency",
            section_type="text",
            subsections=subsections
        )
    
    def _create_period_distribution_section(
        self,
        lexicon: ValencyLexicon,
        periods: Optional[List[Period]]
    ) -> ReportSection:
        """Create period distribution section"""
        period_counts = defaultdict(int)
        
        for entry in lexicon:
            for pattern in entry.patterns.values():
                for period, count in pattern.period_distribution.items():
                    period_counts[period] += count
        
        content = {
            "headers": ["Period", "Frequency"],
            "rows": [
                [period, count]
                for period, count in sorted(period_counts.items())
            ]
        }
        
        return ReportSection(
            title="Period Distribution",
            content=content,
            section_type="table"
        )
    
    def _create_diachronic_changes_section(
        self,
        lexicon: ValencyLexicon,
        periods: Optional[List[Period]]
    ) -> ReportSection:
        """Create diachronic changes section"""
        content = "Analysis of valency changes across periods"
        
        return ReportSection(
            title="Diachronic Changes",
            content=content,
            section_type="text"
        )
    
    def _create_comparison_overview_section(
        self,
        lexicons: List[ValencyLexicon]
    ) -> ReportSection:
        """Create comparison overview section"""
        rows = []
        for lex in lexicons:
            stats = lex.get_statistics()
            rows.append([
                lex.name,
                lex.language.value,
                stats["entry_count"],
                stats["total_patterns"],
                stats["total_instances"]
            ])
        
        content = {
            "headers": ["Lexicon", "Language", "Entries", "Patterns", "Instances"],
            "rows": rows
        }
        
        return ReportSection(
            title="Comparison Overview",
            content=content,
            section_type="table"
        )
    
    def _create_shared_entries_section(
        self,
        lexicons: List[ValencyLexicon]
    ) -> ReportSection:
        """Create shared entries section"""
        all_lemmas = [set(lex.entries.keys()) for lex in lexicons]
        
        shared = set.intersection(*all_lemmas) if all_lemmas else set()
        
        content = {
            "Shared Entries": len(shared),
            "Shared Lemmas": ", ".join(sorted(list(shared)[:20])) + ("..." if len(shared) > 20 else "")
        }
        
        return ReportSection(
            title="Shared Entries",
            content=content,
            section_type="statistics"
        )
    
    def _create_pattern_comparison_section(
        self,
        lexicons: List[ValencyLexicon]
    ) -> ReportSection:
        """Create pattern comparison section"""
        content = "Comparison of valency patterns across lexicons"
        
        return ReportSection(
            title="Pattern Comparison",
            content=content,
            section_type="text"
        )
    
    def export_report(
        self,
        report: ValencyReport,
        file_path: Union[str, Path],
        format: Optional[ReportFormat] = None
    ):
        """Export report to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        format = format or self.config.output_format
        
        if format == ReportFormat.JSON:
            content = report.to_json()
        elif format == ReportFormat.HTML:
            content = report.to_html()
        elif format == ReportFormat.MARKDOWN:
            content = report.to_markdown()
        else:
            content = report.to_json()
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def generate_valency_report(
    lexicon: ValencyLexicon,
    config: Optional[ReportConfig] = None
) -> ValencyReport:
    """Generate valency report"""
    reporter = ValencyReporter(config)
    return reporter.generate_summary_report(lexicon)


def generate_diachronic_report(
    lexicon: ValencyLexicon,
    periods: Optional[List[Period]] = None,
    config: Optional[ReportConfig] = None
) -> ValencyReport:
    """Generate diachronic report"""
    reporter = ValencyReporter(config)
    return reporter.generate_diachronic_report(lexicon, periods)


def generate_comparison_report(
    lexicons: List[ValencyLexicon],
    config: Optional[ReportConfig] = None
) -> ValencyReport:
    """Generate comparison report"""
    reporter = ValencyReporter(config)
    return reporter.generate_comparison_report(lexicons)
