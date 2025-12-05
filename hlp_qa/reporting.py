"""
HLP QA Reporting - QA Metrics Reporting

This module provides comprehensive reporting for quality assurance
metrics and audit results.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from hlp_qa.validators import ValidationResult, ValidationLevel
from hlp_qa.annotation_audit import AuditResult, AuditLevel, AuditIssue

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TSV = "tsv"
    LATEX = "latex"


class QualityGrade(Enum):
    """Quality grades"""
    EXCELLENT = "A"
    GOOD = "B"
    ACCEPTABLE = "C"
    POOR = "D"
    FAILING = "F"


@dataclass
class QAMetrics:
    """Quality assurance metrics"""
    total_items: int = 0
    
    valid_items: int = 0
    
    invalid_items: int = 0
    
    error_count: int = 0
    
    warning_count: int = 0
    
    info_count: int = 0
    
    completeness_score: float = 1.0
    
    accuracy_score: float = 1.0
    
    consistency_score: float = 1.0
    
    overall_score: float = 1.0
    
    grade: QualityGrade = QualityGrade.EXCELLENT
    
    metrics_by_category: Dict[str, float] = field(default_factory=dict)
    
    def calculate_overall(self):
        """Calculate overall score and grade"""
        self.overall_score = (
            self.completeness_score * 0.3 +
            self.accuracy_score * 0.4 +
            self.consistency_score * 0.3
        )
        
        if self.overall_score >= 0.95:
            self.grade = QualityGrade.EXCELLENT
        elif self.overall_score >= 0.85:
            self.grade = QualityGrade.GOOD
        elif self.overall_score >= 0.70:
            self.grade = QualityGrade.ACCEPTABLE
        elif self.overall_score >= 0.50:
            self.grade = QualityGrade.POOR
        else:
            self.grade = QualityGrade.FAILING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_items": self.total_items,
            "valid_items": self.valid_items,
            "invalid_items": self.invalid_items,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "consistency_score": self.consistency_score,
            "overall_score": self.overall_score,
            "grade": self.grade.value,
            "metrics_by_category": self.metrics_by_category
        }


@dataclass
class QAReportSection:
    """Section of a QA report"""
    title: str
    content: str
    
    subsections: List[QAReportSection] = field(default_factory=list)
    
    metrics: Optional[Dict[str, Any]] = None
    
    issues: List[AuditIssue] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "metrics": self.metrics,
            "issue_count": len(self.issues)
        }


@dataclass
class QAReport:
    """Quality assurance report"""
    title: str
    
    generated_at: datetime = field(default_factory=datetime.now)
    
    metrics: QAMetrics = field(default_factory=QAMetrics)
    
    sections: List[QAReportSection] = field(default_factory=list)
    
    validation_results: List[ValidationResult] = field(default_factory=list)
    
    audit_results: List[AuditResult] = field(default_factory=list)
    
    summary: str = ""
    
    recommendations: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "metrics": self.metrics.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class QAReporter:
    """Reporter for QA metrics"""
    
    def __init__(self):
        self._templates: Dict[ReportFormat, str] = {}
    
    def generate_report(
        self,
        validation_results: Optional[List[ValidationResult]] = None,
        audit_results: Optional[List[AuditResult]] = None,
        title: str = "Quality Assurance Report"
    ) -> QAReport:
        """Generate a QA report"""
        report = QAReport(title=title)
        
        if validation_results:
            report.validation_results = validation_results
            self._add_validation_section(report, validation_results)
        
        if audit_results:
            report.audit_results = audit_results
            self._add_audit_section(report, audit_results)
        
        self._calculate_metrics(report)
        self._generate_summary(report)
        self._generate_recommendations(report)
        
        return report
    
    def _add_validation_section(
        self,
        report: QAReport,
        results: List[ValidationResult]
    ):
        """Add validation section to report"""
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        total_items = sum(r.validated_items for r in results)
        
        content = f"Validated {total_items} items across {len(results)} files.\n"
        content += f"Found {total_errors} errors and {total_warnings} warnings."
        
        section = QAReportSection(
            title="Validation Results",
            content=content,
            metrics={
                "total_items": total_items,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "files_validated": len(results)
            }
        )
        
        for result in results:
            if result.errors:
                subsection = QAReportSection(
                    title=f"Errors ({len(result.errors)})",
                    content="\n".join(
                        f"- {e.code}: {e.message}" for e in result.errors[:20]
                    )
                )
                section.subsections.append(subsection)
        
        report.sections.append(section)
        report.metrics.error_count += total_errors
        report.metrics.warning_count += total_warnings
    
    def _add_audit_section(
        self,
        report: QAReport,
        results: List[AuditResult]
    ):
        """Add audit section to report"""
        total_issues = sum(len(r.issues) for r in results)
        total_items = sum(r.items_audited for r in results)
        avg_score = sum(r.quality_score for r in results) / len(results) if results else 1.0
        
        content = f"Audited {total_items} items across {len(results)} documents.\n"
        content += f"Found {total_issues} issues. Average quality score: {avg_score:.2%}"
        
        section = QAReportSection(
            title="Audit Results",
            content=content,
            metrics={
                "total_items": total_items,
                "total_issues": total_issues,
                "average_score": avg_score,
                "documents_audited": len(results)
            }
        )
        
        category_counts: Dict[str, int] = {}
        for result in results:
            for cat, count in result.issues_by_category.items():
                category_counts[cat] = category_counts.get(cat, 0) + count
        
        if category_counts:
            cat_content = "\n".join(
                f"- {cat}: {count} issues"
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
            )
            section.subsections.append(QAReportSection(
                title="Issues by Category",
                content=cat_content
            ))
        
        report.sections.append(section)
        report.metrics.total_items += total_items
    
    def _calculate_metrics(self, report: QAReport):
        """Calculate overall metrics"""
        if report.validation_results:
            valid_count = sum(1 for r in report.validation_results if r.valid)
            total = len(report.validation_results)
            report.metrics.valid_items = valid_count
            report.metrics.invalid_items = total - valid_count
        
        if report.audit_results:
            scores = [r.quality_score for r in report.audit_results]
            report.metrics.accuracy_score = sum(scores) / len(scores) if scores else 1.0
            
            completeness_issues = sum(
                r.issues_by_category.get("completeness", 0)
                for r in report.audit_results
            )
            total_items = report.metrics.total_items or 1
            report.metrics.completeness_score = max(
                0.0, 1.0 - (completeness_issues / total_items)
            )
            
            consistency_issues = sum(
                r.issues_by_category.get("consistency", 0)
                for r in report.audit_results
            )
            report.metrics.consistency_score = max(
                0.0, 1.0 - (consistency_issues / total_items)
            )
        
        report.metrics.calculate_overall()
    
    def _generate_summary(self, report: QAReport):
        """Generate report summary"""
        metrics = report.metrics
        
        summary_parts = []
        
        summary_parts.append(
            f"Overall quality grade: {metrics.grade.value} "
            f"({metrics.overall_score:.1%})"
        )
        
        if metrics.error_count > 0:
            summary_parts.append(f"Found {metrics.error_count} validation errors.")
        
        if metrics.warning_count > 0:
            summary_parts.append(f"Found {metrics.warning_count} warnings.")
        
        if metrics.completeness_score < 0.9:
            summary_parts.append(
                f"Completeness score ({metrics.completeness_score:.1%}) "
                "indicates missing annotations."
            )
        
        if metrics.consistency_score < 0.9:
            summary_parts.append(
                f"Consistency score ({metrics.consistency_score:.1%}) "
                "indicates annotation inconsistencies."
            )
        
        report.summary = " ".join(summary_parts)
    
    def _generate_recommendations(self, report: QAReport):
        """Generate recommendations"""
        metrics = report.metrics
        recommendations = []
        
        if metrics.error_count > 0:
            recommendations.append(
                "Fix validation errors before proceeding with analysis."
            )
        
        if metrics.completeness_score < 0.8:
            recommendations.append(
                "Review and complete missing annotations, especially "
                "morphology and syntax for content words."
            )
        
        if metrics.consistency_score < 0.8:
            recommendations.append(
                "Review annotation guidelines and ensure consistent "
                "application across the corpus."
            )
        
        if metrics.accuracy_score < 0.8:
            recommendations.append(
                "Consider manual review of flagged issues to improve "
                "annotation accuracy."
            )
        
        if metrics.grade in [QualityGrade.POOR, QualityGrade.FAILING]:
            recommendations.append(
                "Significant quality issues detected. Consider re-annotation "
                "or additional training for annotators."
            )
        
        if not recommendations:
            recommendations.append(
                "Quality metrics are satisfactory. Continue with current "
                "annotation practices."
            )
        
        report.recommendations = recommendations
    
    def export_report(
        self,
        report: QAReport,
        output_path: Union[str, Path],
        format: ReportFormat = ReportFormat.JSON
    ):
        """Export report to file"""
        output_path = Path(output_path)
        
        if format == ReportFormat.JSON:
            self._export_json(report, output_path)
        elif format == ReportFormat.HTML:
            self._export_html(report, output_path)
        elif format == ReportFormat.MARKDOWN:
            self._export_markdown(report, output_path)
        elif format == ReportFormat.TSV:
            self._export_tsv(report, output_path)
        elif format == ReportFormat.LATEX:
            self._export_latex(report, output_path)
    
    def _export_json(self, report: QAReport, path: Path):
        """Export as JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _export_html(self, report: QAReport, path: Path):
        """Export as HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .metrics {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .grade {{ font-size: 2em; font-weight: bold; }}
        .grade-A {{ color: #28a745; }}
        .grade-B {{ color: #5cb85c; }}
        .grade-C {{ color: #f0ad4e; }}
        .grade-D {{ color: #d9534f; }}
        .grade-F {{ color: #c9302c; }}
        .section {{ margin: 20px 0; }}
        .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 5px; }}
        ul {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="metrics">
        <h2>Quality Metrics</h2>
        <p class="grade grade-{report.metrics.grade.value}">
            Grade: {report.metrics.grade.value}
        </p>
        <p>Overall Score: {report.metrics.overall_score:.1%}</p>
        <p>Completeness: {report.metrics.completeness_score:.1%}</p>
        <p>Accuracy: {report.metrics.accuracy_score:.1%}</p>
        <p>Consistency: {report.metrics.consistency_score:.1%}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <p>{report.summary}</p>
    </div>
"""
        
        for section in report.sections:
            html += f"""
    <div class="section">
        <h2>{section.title}</h2>
        <p>{section.content}</p>
    </div>
"""
        
        html += """
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
        for rec in report.recommendations:
            html += f"            <li>{rec}</li>\n"
        
        html += """        </ul>
    </div>
</body>
</html>
"""
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    
    def _export_markdown(self, report: QAReport, path: Path):
        """Export as Markdown"""
        md = f"# {report.title}\n\n"
        md += f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += "## Quality Metrics\n\n"
        md += f"- **Grade**: {report.metrics.grade.value}\n"
        md += f"- **Overall Score**: {report.metrics.overall_score:.1%}\n"
        md += f"- **Completeness**: {report.metrics.completeness_score:.1%}\n"
        md += f"- **Accuracy**: {report.metrics.accuracy_score:.1%}\n"
        md += f"- **Consistency**: {report.metrics.consistency_score:.1%}\n\n"
        
        md += "## Summary\n\n"
        md += f"{report.summary}\n\n"
        
        for section in report.sections:
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"
        
        md += "## Recommendations\n\n"
        for rec in report.recommendations:
            md += f"- {rec}\n"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
    
    def _export_tsv(self, report: QAReport, path: Path):
        """Export metrics as TSV"""
        lines = ["Metric\tValue"]
        
        metrics = report.metrics
        lines.append(f"Grade\t{metrics.grade.value}")
        lines.append(f"Overall Score\t{metrics.overall_score:.4f}")
        lines.append(f"Completeness\t{metrics.completeness_score:.4f}")
        lines.append(f"Accuracy\t{metrics.accuracy_score:.4f}")
        lines.append(f"Consistency\t{metrics.consistency_score:.4f}")
        lines.append(f"Total Items\t{metrics.total_items}")
        lines.append(f"Errors\t{metrics.error_count}")
        lines.append(f"Warnings\t{metrics.warning_count}")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _export_latex(self, report: QAReport, path: Path):
        """Export as LaTeX"""
        latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{xcolor}

\begin{document}

"""
        latex += f"\\title{{{report.title}}}\n"
        latex += "\\maketitle\n\n"
        
        latex += "\\section{Quality Metrics}\n\n"
        latex += "\\begin{tabular}{lr}\n\\toprule\n"
        latex += "Metric & Value \\\\\n\\midrule\n"
        latex += f"Grade & {report.metrics.grade.value} \\\\\n"
        latex += f"Overall Score & {report.metrics.overall_score:.1\\%} \\\\\n"
        latex += f"Completeness & {report.metrics.completeness_score:.1\\%} \\\\\n"
        latex += f"Accuracy & {report.metrics.accuracy_score:.1\\%} \\\\\n"
        latex += f"Consistency & {report.metrics.consistency_score:.1\\%} \\\\\n"
        latex += "\\bottomrule\n\\end{tabular}\n\n"
        
        latex += "\\section{Summary}\n\n"
        latex += f"{report.summary}\n\n"
        
        latex += "\\section{Recommendations}\n\n"
        latex += "\\begin{itemize}\n"
        for rec in report.recommendations:
            latex += f"\\item {rec}\n"
        latex += "\\end{itemize}\n\n"
        
        latex += "\\end{document}\n"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex)


def generate_qa_report(
    validation_results: Optional[List[ValidationResult]] = None,
    audit_results: Optional[List[AuditResult]] = None,
    title: str = "Quality Assurance Report"
) -> QAReport:
    """Generate a QA report"""
    reporter = QAReporter()
    return reporter.generate_report(validation_results, audit_results, title)


def export_qa_report(
    report: QAReport,
    output_path: Union[str, Path],
    format: ReportFormat = ReportFormat.JSON
):
    """Export a QA report"""
    reporter = QAReporter()
    reporter.export_report(report, output_path, format)
