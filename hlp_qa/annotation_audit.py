"""
HLP QA Annotation Audit - Annotation Quality Checks

This module provides comprehensive auditing for annotation quality,
including morphology, syntax, and valency annotation checks.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from collections import Counter

from hlp_core.models import (
    Language, Document, Corpus, Sentence, Token,
    MorphologyAnnotation, SyntaxAnnotation, Case
)

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    THOROUGH = "thorough"
    ERC = "erc"


class IssueSeverity(Enum):
    """Severity of audit issues"""
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of audit issues"""
    MORPHOLOGY = "morphology"
    SYNTAX = "syntax"
    VALENCY = "valency"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"


@dataclass
class AuditIssue:
    """Represents an audit issue"""
    code: str
    message: str
    severity: IssueSeverity
    category: IssueCategory
    
    location: Optional[str] = None
    
    token_id: Optional[str] = None
    
    sentence_id: Optional[str] = None
    
    expected: Optional[str] = None
    
    actual: Optional[str] = None
    
    suggestion: Optional[str] = None
    
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "location": self.location,
            "token_id": self.token_id,
            "sentence_id": self.sentence_id,
            "expected": self.expected,
            "actual": self.actual,
            "suggestion": self.suggestion,
            "confidence": self.confidence
        }


@dataclass
class AuditResult:
    """Result of an audit"""
    passed: bool
    
    issues: List[AuditIssue] = field(default_factory=list)
    
    audit_level: AuditLevel = AuditLevel.STANDARD
    
    items_audited: int = 0
    
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    
    quality_score: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: AuditIssue):
        """Add an issue"""
        self.issues.append(issue)
        
        cat = issue.category.value
        self.issues_by_category[cat] = self.issues_by_category.get(cat, 0) + 1
        
        sev = issue.severity.value
        self.issues_by_severity[sev] = self.issues_by_severity.get(sev, 0) + 1
        
        if issue.severity in [IssueSeverity.MAJOR, IssueSeverity.CRITICAL]:
            self.passed = False
    
    def calculate_quality_score(self):
        """Calculate quality score"""
        if self.items_audited == 0:
            self.quality_score = 1.0
            return
        
        penalty = 0.0
        
        for issue in self.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                penalty += 0.1
            elif issue.severity == IssueSeverity.MAJOR:
                penalty += 0.05
            elif issue.severity == IssueSeverity.MINOR:
                penalty += 0.01
            else:
                penalty += 0.001
        
        self.quality_score = max(0.0, 1.0 - (penalty / self.items_audited))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "audit_level": self.audit_level.value,
            "items_audited": self.items_audited,
            "issues_by_category": self.issues_by_category,
            "issues_by_severity": self.issues_by_severity,
            "quality_score": self.quality_score,
            "issues": [i.to_dict() for i in self.issues[:100]]
        }


CASE_AGREEMENT_RULES = {
    ("nsubj", "VERB"): [Case.NOMINATIVE],
    ("obj", "VERB"): [Case.ACCUSATIVE, Case.GENITIVE],
    ("iobj", "VERB"): [Case.DATIVE],
    ("obl:agent", "VERB"): [Case.GENITIVE, Case.ABLATIVE],
}

POS_DEPREL_COMPATIBILITY = {
    "NOUN": ["nsubj", "obj", "iobj", "obl", "nmod", "appos", "conj", "root"],
    "VERB": ["root", "ccomp", "xcomp", "advcl", "acl", "conj", "parataxis"],
    "ADJ": ["amod", "xcomp", "conj", "root"],
    "ADV": ["advmod", "conj"],
    "ADP": ["case", "mark"],
    "DET": ["det"],
    "PRON": ["nsubj", "obj", "iobj", "obl", "nmod", "expl"],
    "PROPN": ["nsubj", "obj", "iobj", "obl", "nmod", "appos", "flat", "conj"],
}


class AnnotationAuditor:
    """Auditor for annotation quality"""
    
    def __init__(self, level: AuditLevel = AuditLevel.STANDARD):
        self.level = level
        self._lemma_cache: Dict[str, Set[str]] = {}
        self._pos_cache: Dict[str, Counter] = {}
    
    def audit_document(self, document: Document) -> AuditResult:
        """Audit a document"""
        result = AuditResult(
            passed=True,
            audit_level=self.level
        )
        
        for sentence in document.sentences:
            self._audit_sentence(sentence, result)
        
        result.items_audited = sum(
            len(s.tokens) for s in document.sentences
        )
        
        result.calculate_quality_score()
        
        return result
    
    def audit_corpus(self, corpus: Corpus) -> AuditResult:
        """Audit a corpus"""
        result = AuditResult(
            passed=True,
            audit_level=self.level
        )
        
        for document in corpus.documents:
            doc_result = self.audit_document(document)
            result.issues.extend(doc_result.issues)
            result.items_audited += doc_result.items_audited
        
        for issue in result.issues:
            cat = issue.category.value
            result.issues_by_category[cat] = result.issues_by_category.get(cat, 0) + 1
            
            sev = issue.severity.value
            result.issues_by_severity[sev] = result.issues_by_severity.get(sev, 0) + 1
            
            if issue.severity in [IssueSeverity.MAJOR, IssueSeverity.CRITICAL]:
                result.passed = False
        
        result.calculate_quality_score()
        
        return result
    
    def _audit_sentence(self, sentence: Sentence, result: AuditResult):
        """Audit a sentence"""
        for token in sentence.tokens:
            self._audit_token(token, sentence, result)
        
        self._audit_sentence_structure(sentence, result)
        
        if self.level in [AuditLevel.THOROUGH, AuditLevel.ERC]:
            self._audit_consistency(sentence, result)
    
    def _audit_token(
        self,
        token: Token,
        sentence: Sentence,
        result: AuditResult
    ):
        """Audit a token"""
        self._audit_morphology(token, sentence, result)
        self._audit_syntax(token, sentence, result)
        
        if self.level in [AuditLevel.THOROUGH, AuditLevel.ERC]:
            self._audit_completeness(token, sentence, result)
    
    def _audit_morphology(
        self,
        token: Token,
        sentence: Sentence,
        result: AuditResult
    ):
        """Audit morphology annotation"""
        if not token.morphology:
            if self.level in [AuditLevel.THOROUGH, AuditLevel.ERC]:
                if token.pos in ["NOUN", "VERB", "ADJ", "PRON"]:
                    result.add_issue(AuditIssue(
                        code="MISSING_MORPHOLOGY",
                        message=f"Token '{token.form}' missing morphology",
                        severity=IssueSeverity.MINOR,
                        category=IssueCategory.COMPLETENESS,
                        token_id=token.id,
                        sentence_id=sentence.id
                    ))
            return
        
        morph = token.morphology
        
        if token.pos == "NOUN" and not morph.case:
            result.add_issue(AuditIssue(
                code="NOUN_MISSING_CASE",
                message=f"Noun '{token.form}' missing case annotation",
                severity=IssueSeverity.MINOR,
                category=IssueCategory.MORPHOLOGY,
                token_id=token.id,
                sentence_id=sentence.id
            ))
        
        if token.pos == "VERB":
            if not morph.tense and not morph.mood:
                result.add_issue(AuditIssue(
                    code="VERB_MISSING_TAM",
                    message=f"Verb '{token.form}' missing tense/mood",
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.MORPHOLOGY,
                    token_id=token.id,
                    sentence_id=sentence.id
                ))
    
    def _audit_syntax(
        self,
        token: Token,
        sentence: Sentence,
        result: AuditResult
    ):
        """Audit syntax annotation"""
        if not token.syntax:
            if self.level in [AuditLevel.THOROUGH, AuditLevel.ERC]:
                result.add_issue(AuditIssue(
                    code="MISSING_SYNTAX",
                    message=f"Token '{token.form}' missing syntax annotation",
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.COMPLETENESS,
                    token_id=token.id,
                    sentence_id=sentence.id
                ))
            return
        
        syntax = token.syntax
        
        if syntax.head and syntax.head != "0":
            token_map = {t.id: t for t in sentence.tokens}
            if syntax.head not in token_map:
                result.add_issue(AuditIssue(
                    code="INVALID_HEAD",
                    message=f"Token '{token.form}' has invalid head '{syntax.head}'",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.SYNTAX,
                    token_id=token.id,
                    sentence_id=sentence.id
                ))
        
        if token.pos and syntax.deprel:
            compatible = POS_DEPREL_COMPATIBILITY.get(token.pos, [])
            base_deprel = syntax.deprel.split(":")[0]
            
            if compatible and base_deprel not in compatible:
                result.add_issue(AuditIssue(
                    code="POS_DEPREL_MISMATCH",
                    message=f"POS '{token.pos}' unusual with deprel '{syntax.deprel}'",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.SYNTAX,
                    token_id=token.id,
                    sentence_id=sentence.id,
                    actual=f"{token.pos} + {syntax.deprel}"
                ))
    
    def _audit_sentence_structure(self, sentence: Sentence, result: AuditResult):
        """Audit sentence structure"""
        root_count = 0
        
        for token in sentence.tokens:
            if token.syntax and token.syntax.head == "0":
                root_count += 1
        
        if root_count == 0:
            result.add_issue(AuditIssue(
                code="NO_ROOT",
                message=f"Sentence {sentence.id} has no root",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.SYNTAX,
                sentence_id=sentence.id
            ))
        elif root_count > 1:
            result.add_issue(AuditIssue(
                code="MULTIPLE_ROOTS",
                message=f"Sentence {sentence.id} has {root_count} roots",
                severity=IssueSeverity.MINOR,
                category=IssueCategory.SYNTAX,
                sentence_id=sentence.id
            ))
        
        if self.level in [AuditLevel.THOROUGH, AuditLevel.ERC]:
            self._check_projectivity(sentence, result)
    
    def _check_projectivity(self, sentence: Sentence, result: AuditResult):
        """Check for non-projective dependencies"""
        edges = []
        
        for token in sentence.tokens:
            if token.syntax and token.syntax.head:
                try:
                    dep_idx = int(token.id)
                    head_idx = int(token.syntax.head)
                    if head_idx != 0:
                        edges.append((min(dep_idx, head_idx), max(dep_idx, head_idx)))
                except ValueError:
                    continue
        
        for i, (s1, e1) in enumerate(edges):
            for s2, e2 in edges[i+1:]:
                if s1 < s2 < e1 < e2 or s2 < s1 < e2 < e1:
                    result.add_issue(AuditIssue(
                        code="NON_PROJECTIVE",
                        message=f"Non-projective dependency in sentence {sentence.id}",
                        severity=IssueSeverity.INFO,
                        category=IssueCategory.SYNTAX,
                        sentence_id=sentence.id
                    ))
                    return
    
    def _audit_consistency(self, sentence: Sentence, result: AuditResult):
        """Audit annotation consistency"""
        lemma_pos = {}
        
        for token in sentence.tokens:
            if token.lemma and token.pos:
                key = token.lemma.lower()
                if key not in lemma_pos:
                    lemma_pos[key] = set()
                lemma_pos[key].add(token.pos)
        
        for lemma, pos_set in lemma_pos.items():
            if len(pos_set) > 1:
                result.add_issue(AuditIssue(
                    code="INCONSISTENT_POS",
                    message=f"Lemma '{lemma}' has multiple POS: {pos_set}",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.CONSISTENCY,
                    sentence_id=sentence.id
                ))
    
    def _audit_completeness(
        self,
        token: Token,
        sentence: Sentence,
        result: AuditResult
    ):
        """Audit annotation completeness"""
        if not token.lemma:
            result.add_issue(AuditIssue(
                code="MISSING_LEMMA",
                message=f"Token '{token.form}' missing lemma",
                severity=IssueSeverity.MINOR,
                category=IssueCategory.COMPLETENESS,
                token_id=token.id,
                sentence_id=sentence.id
            ))
        
        if not token.pos:
            result.add_issue(AuditIssue(
                code="MISSING_POS",
                message=f"Token '{token.form}' missing POS tag",
                severity=IssueSeverity.MINOR,
                category=IssueCategory.COMPLETENESS,
                token_id=token.id,
                sentence_id=sentence.id
            ))


def audit_morphology(
    document: Document,
    level: AuditLevel = AuditLevel.STANDARD
) -> AuditResult:
    """Audit morphology annotations"""
    auditor = AnnotationAuditor(level)
    result = auditor.audit_document(document)
    
    result.issues = [
        i for i in result.issues
        if i.category == IssueCategory.MORPHOLOGY
    ]
    
    return result


def audit_syntax(
    document: Document,
    level: AuditLevel = AuditLevel.STANDARD
) -> AuditResult:
    """Audit syntax annotations"""
    auditor = AnnotationAuditor(level)
    result = auditor.audit_document(document)
    
    result.issues = [
        i for i in result.issues
        if i.category == IssueCategory.SYNTAX
    ]
    
    return result


def audit_valency(
    document: Document,
    level: AuditLevel = AuditLevel.STANDARD
) -> AuditResult:
    """Audit valency annotations"""
    auditor = AnnotationAuditor(level)
    result = auditor.audit_document(document)
    
    result.issues = [
        i for i in result.issues
        if i.category == IssueCategory.VALENCY
    ]
    
    return result


def run_full_audit(
    document: Document,
    level: AuditLevel = AuditLevel.STANDARD
) -> AuditResult:
    """Run full audit on a document"""
    auditor = AnnotationAuditor(level)
    return auditor.audit_document(document)
