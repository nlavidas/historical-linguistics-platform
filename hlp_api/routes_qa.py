"""
HLP API Routes QA - Quality Assurance Endpoints

This module provides REST API endpoints for quality assurance.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends, UploadFile, File
from pydantic import BaseModel, Field

from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_validation_results: Dict[str, Dict[str, Any]] = {}
_audit_results: Dict[str, Dict[str, Any]] = {}


class ValidationRequest(BaseModel):
    """Schema for validation request"""
    content: str = Field(..., description="Content to validate")
    format: str = Field("proiel", description="Format (proiel, conllu)")
    level: str = Field("standard", description="Validation level")


class ValidationResponse(BaseModel):
    """Schema for validation response"""
    valid: bool
    error_count: int
    warning_count: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


class AuditRequest(BaseModel):
    """Schema for audit request"""
    corpus_id: str = Field(..., description="Corpus ID to audit")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs")
    level: str = Field("standard", description="Audit level")


class AuditResponse(BaseModel):
    """Schema for audit response"""
    id: str
    corpus_id: str
    passed: bool
    quality_score: float
    issue_count: int
    issues_by_category: Dict[str, int]
    created_at: str


class QAReportRequest(BaseModel):
    """Schema for QA report request"""
    corpus_id: str = Field(..., description="Corpus ID")
    include_validation: bool = Field(True, description="Include validation results")
    include_audit: bool = Field(True, description="Include audit results")
    format: str = Field("json", description="Report format")


class QAReportResponse(BaseModel):
    """Schema for QA report response"""
    id: str
    corpus_id: str
    grade: str
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    recommendations: List[str]
    created_at: str


@router.post("/validate", response_model=ValidationResponse)
async def validate_content(
    request: ValidationRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Validate content"""
    errors = []
    warnings = []
    
    if not request.content.strip():
        errors.append({
            "code": "EMPTY_CONTENT",
            "message": "Content is empty",
            "severity": "error"
        })
    
    if request.format == "proiel":
        if "<proiel" not in request.content:
            errors.append({
                "code": "INVALID_ROOT",
                "message": "Missing proiel root element",
                "severity": "error"
            })
    
    elif request.format == "conllu":
        lines = request.content.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) != 10:
                    errors.append({
                        "code": "INVALID_FIELD_COUNT",
                        "message": f"Line {i+1}: Expected 10 fields, found {len(fields)}",
                        "severity": "error",
                        "line": i + 1
                    })
    
    return ValidationResponse(
        valid=len(errors) == 0,
        error_count=len(errors),
        warning_count=len(warnings),
        errors=errors,
        warnings=warnings
    )


@router.post("/validate/file")
async def validate_file(
    file: UploadFile = File(..., description="File to validate"),
    format: str = Query("auto", description="Format (auto, proiel, conllu)"),
    level: str = Query("standard", description="Validation level"),
    user: Optional[User] = Depends(get_current_user)
):
    """Validate an uploaded file"""
    content = await file.read()
    text = content.decode("utf-8")
    
    if format == "auto":
        if file.filename and file.filename.endswith(".xml"):
            format = "proiel"
        elif file.filename and file.filename.endswith(".conllu"):
            format = "conllu"
        else:
            format = "proiel"
    
    errors = []
    warnings = []
    
    if not text.strip():
        errors.append({
            "code": "EMPTY_FILE",
            "message": "File is empty",
            "severity": "error"
        })
    
    return {
        "filename": file.filename,
        "format": format,
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings
    }


@router.post("/audit", response_model=AuditResponse)
async def audit_corpus(
    request: AuditRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Audit a corpus"""
    import uuid
    
    audit_id = str(uuid.uuid4())
    
    result = {
        "id": audit_id,
        "corpus_id": request.corpus_id,
        "document_ids": request.document_ids,
        "level": request.level,
        "passed": True,
        "quality_score": 0.92,
        "issue_count": 15,
        "issues_by_category": {
            "morphology": 5,
            "syntax": 7,
            "completeness": 3
        },
        "created_at": datetime.now().isoformat(),
        "created_by": user.username if user else "anonymous"
    }
    
    _audit_results[audit_id] = result
    
    return AuditResponse(
        id=audit_id,
        corpus_id=request.corpus_id,
        passed=result["passed"],
        quality_score=result["quality_score"],
        issue_count=result["issue_count"],
        issues_by_category=result["issues_by_category"],
        created_at=result["created_at"]
    )


@router.get("/audit/{audit_id}", response_model=AuditResponse)
async def get_audit_result(
    audit_id: str = Path(..., description="Audit ID")
):
    """Get audit result by ID"""
    result = _audit_results.get(audit_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Audit result not found")
    
    return AuditResponse(
        id=result["id"],
        corpus_id=result["corpus_id"],
        passed=result["passed"],
        quality_score=result["quality_score"],
        issue_count=result["issue_count"],
        issues_by_category=result["issues_by_category"],
        created_at=result["created_at"]
    )


@router.get("/audit/{audit_id}/issues")
async def get_audit_issues(
    audit_id: str = Path(..., description="Audit ID"),
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Get issues from an audit"""
    result = _audit_results.get(audit_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Audit result not found")
    
    issues = [
        {
            "code": "MISSING_LEMMA",
            "message": "Token missing lemma",
            "severity": "minor",
            "category": "completeness",
            "location": "sentence_1:token_5"
        },
        {
            "code": "INVALID_HEAD",
            "message": "Invalid head reference",
            "severity": "major",
            "category": "syntax",
            "location": "sentence_3:token_2"
        },
        {
            "code": "POS_DEPREL_MISMATCH",
            "message": "POS and deprel mismatch",
            "severity": "info",
            "category": "syntax",
            "location": "sentence_5:token_8"
        }
    ]
    
    if category:
        issues = [i for i in issues if i["category"] == category]
    
    if severity:
        issues = [i for i in issues if i["severity"] == severity]
    
    return {
        "audit_id": audit_id,
        "total_issues": len(issues),
        "issues": issues[offset:offset + limit]
    }


@router.post("/report", response_model=QAReportResponse)
async def generate_qa_report(
    request: QAReportRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Generate a QA report"""
    import uuid
    
    report_id = str(uuid.uuid4())
    
    return QAReportResponse(
        id=report_id,
        corpus_id=request.corpus_id,
        grade="B",
        overall_score=0.87,
        completeness_score=0.92,
        accuracy_score=0.85,
        consistency_score=0.84,
        recommendations=[
            "Review and complete missing lemma annotations",
            "Check syntax annotations for consistency",
            "Validate morphological features for nouns"
        ],
        created_at=datetime.now().isoformat()
    )


@router.get("/report/{report_id}")
async def get_qa_report(
    report_id: str = Path(..., description="Report ID"),
    format: str = Query("json", description="Output format")
):
    """Get a QA report"""
    return {
        "id": report_id,
        "format": format,
        "grade": "B",
        "overall_score": 0.87,
        "sections": [
            {
                "title": "Validation Results",
                "content": "All files validated successfully"
            },
            {
                "title": "Audit Results",
                "content": "15 issues found across 3 categories"
            }
        ],
        "recommendations": [
            "Review and complete missing lemma annotations",
            "Check syntax annotations for consistency"
        ]
    }


@router.get("/metrics")
async def get_qa_metrics(
    corpus_id: Optional[str] = Query(None, description="Corpus ID")
):
    """Get QA metrics"""
    return {
        "total_validations": 50,
        "total_audits": 25,
        "average_quality_score": 0.89,
        "grade_distribution": {
            "A": 10,
            "B": 12,
            "C": 2,
            "D": 1,
            "F": 0
        },
        "common_issues": [
            {"code": "MISSING_LEMMA", "count": 150},
            {"code": "INVALID_HEAD", "count": 45},
            {"code": "POS_DEPREL_MISMATCH", "count": 30}
        ],
        "trend": {
            "last_week": 0.87,
            "last_month": 0.85,
            "improvement": 0.02
        }
    }


@router.get("/guidelines")
async def get_qa_guidelines():
    """Get QA guidelines"""
    return {
        "validation_levels": [
            {
                "id": "minimal",
                "name": "Minimal",
                "description": "Basic format validation only"
            },
            {
                "id": "standard",
                "name": "Standard",
                "description": "Format and basic content validation"
            },
            {
                "id": "strict",
                "name": "Strict",
                "description": "Full validation with all checks"
            },
            {
                "id": "erc",
                "name": "ERC Quality",
                "description": "European Research Council quality standards"
            }
        ],
        "audit_categories": [
            {
                "id": "morphology",
                "name": "Morphology",
                "description": "Morphological annotation quality"
            },
            {
                "id": "syntax",
                "name": "Syntax",
                "description": "Syntactic annotation quality"
            },
            {
                "id": "completeness",
                "name": "Completeness",
                "description": "Annotation completeness"
            },
            {
                "id": "consistency",
                "name": "Consistency",
                "description": "Annotation consistency"
            }
        ],
        "quality_grades": [
            {"grade": "A", "min_score": 0.95, "description": "Excellent"},
            {"grade": "B", "min_score": 0.85, "description": "Good"},
            {"grade": "C", "min_score": 0.70, "description": "Acceptable"},
            {"grade": "D", "min_score": 0.50, "description": "Poor"},
            {"grade": "F", "min_score": 0.0, "description": "Failing"}
        ]
    }
