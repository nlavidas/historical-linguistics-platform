"""
HLP QA - Quality Assurance Package

This package provides comprehensive support for quality assurance,
including schema validation, annotation auditing, and QA reporting.

Modules:
    validators: Schema and format validation
    annotation_audit: Annotation quality checks
    reporting: QA metrics reporting

University of Athens - Nikolaos Lavidas
"""

from hlp_qa.validators import (
    ValidationLevel,
    ValidationResult,
    ValidationError,
    SchemaValidator,
    PROIELValidator,
    CoNLLUValidator,
    validate_proiel_xml,
    validate_conllu,
    validate_document,
)

from hlp_qa.annotation_audit import (
    AuditLevel,
    AuditResult,
    AuditIssue,
    AnnotationAuditor,
    audit_morphology,
    audit_syntax,
    audit_valency,
    run_full_audit,
)

from hlp_qa.reporting import (
    QAReport,
    QAMetrics,
    QAReporter,
    generate_qa_report,
    export_qa_report,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "ValidationError",
    "SchemaValidator",
    "PROIELValidator",
    "CoNLLUValidator",
    "validate_proiel_xml",
    "validate_conllu",
    "validate_document",
    "AuditLevel",
    "AuditResult",
    "AuditIssue",
    "AnnotationAuditor",
    "audit_morphology",
    "audit_syntax",
    "audit_valency",
    "run_full_audit",
    "QAReport",
    "QAMetrics",
    "QAReporter",
    "generate_qa_report",
    "export_qa_report",
]
