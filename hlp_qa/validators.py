"""
HLP QA Validators - Schema and Format Validation

This module provides comprehensive validation for various formats
including PROIEL XML, CoNLL-U, and internal data structures.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path
from enum import Enum

from hlp_core.models import (
    Language, Document, Corpus, Sentence, Token,
    MorphologyAnnotation, SyntaxAnnotation
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PROIEL = "proiel"
    ERC = "erc"


class ValidationSeverity(Enum):
    """Severity of validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Represents a validation error"""
    code: str
    message: str
    severity: ValidationSeverity
    
    location: Optional[str] = None
    
    line_number: Optional[int] = None
    
    element: Optional[str] = None
    
    expected: Optional[str] = None
    
    actual: Optional[str] = None
    
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location,
            "line_number": self.line_number,
            "element": self.element,
            "expected": self.expected,
            "actual": self.actual,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    
    errors: List[ValidationError] = field(default_factory=list)
    
    warnings: List[ValidationError] = field(default_factory=list)
    
    info: List[ValidationError] = field(default_factory=list)
    
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    validated_items: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ValidationError):
        """Add an error"""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.valid = False
        elif error.severity == ValidationSeverity.CRITICAL:
            self.errors.append(error)
            self.valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "validation_level": self.validation_level.value,
            "validated_items": self.validated_items,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings]
        }


PROIEL_POS_TAGS = {
    "A-", "C-", "Df", "Dq", "Du", "F-", "G-", "I-",
    "Ma", "Mo", "N-", "Nb", "Ne", "Pc", "Pd", "Pi",
    "Pk", "Pp", "Pr", "Ps", "Pt", "Px", "Py", "R-",
    "S-", "V-", "X-"
}

PROIEL_RELATIONS = {
    "adnom", "adv", "ag", "apos", "arg", "atr", "aux",
    "comp", "expl", "narg", "nonsub", "obj", "obl",
    "parpred", "part", "per", "pid", "pred", "rel",
    "sub", "voc", "xadv", "xobj", "xsub"
}

UD_POS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X"
}

UD_RELATIONS = {
    "acl", "advcl", "advmod", "amod", "appos", "aux",
    "case", "cc", "ccomp", "clf", "compound", "conj",
    "cop", "csubj", "dep", "det", "discourse", "dislocated",
    "expl", "fixed", "flat", "goeswith", "iobj", "list",
    "mark", "nmod", "nsubj", "nummod", "obj", "obl",
    "orphan", "parataxis", "punct", "reparandum", "root",
    "vocative", "xcomp"
}


class SchemaValidator:
    """Base validator class"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
    
    def validate(self, data: Any) -> ValidationResult:
        """Validate data"""
        raise NotImplementedError


class PROIELValidator(SchemaValidator):
    """Validator for PROIEL XML format"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.PROIEL):
        super().__init__(level)
        self._schema = None
    
    def validate(self, xml_content: Union[str, Path]) -> ValidationResult:
        """Validate PROIEL XML"""
        result = ValidationResult(
            valid=True,
            validation_level=self.level
        )
        
        if isinstance(xml_content, Path):
            try:
                with open(xml_content, "r", encoding="utf-8") as f:
                    xml_content = f.read()
            except Exception as e:
                result.add_error(ValidationError(
                    code="FILE_READ_ERROR",
                    message=f"Failed to read file: {str(e)}",
                    severity=ValidationSeverity.CRITICAL
                ))
                return result
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            result.add_error(ValidationError(
                code="XML_PARSE_ERROR",
                message=f"XML parsing failed: {str(e)}",
                severity=ValidationSeverity.CRITICAL,
                line_number=e.position[0] if hasattr(e, 'position') else None
            ))
            return result
        
        self._validate_root(root, result)
        self._validate_sources(root, result)
        self._validate_sentences(root, result)
        self._validate_tokens(root, result)
        
        return result
    
    def _validate_root(self, root: ET.Element, result: ValidationResult):
        """Validate root element"""
        if root.tag != "proiel":
            result.add_error(ValidationError(
                code="INVALID_ROOT",
                message=f"Expected root element 'proiel', found '{root.tag}'",
                severity=ValidationSeverity.ERROR,
                expected="proiel",
                actual=root.tag
            ))
        
        version = root.get("export-time")
        if not version and self.level in [ValidationLevel.STRICT, ValidationLevel.PROIEL]:
            result.add_error(ValidationError(
                code="MISSING_EXPORT_TIME",
                message="Missing export-time attribute on root element",
                severity=ValidationSeverity.WARNING
            ))
    
    def _validate_sources(self, root: ET.Element, result: ValidationResult):
        """Validate source elements"""
        sources = root.findall(".//source")
        
        if not sources:
            result.add_error(ValidationError(
                code="NO_SOURCES",
                message="No source elements found",
                severity=ValidationSeverity.ERROR
            ))
            return
        
        for source in sources:
            source_id = source.get("id")
            if not source_id:
                result.add_error(ValidationError(
                    code="MISSING_SOURCE_ID",
                    message="Source element missing id attribute",
                    severity=ValidationSeverity.ERROR
                ))
            
            language = source.get("language")
            if not language:
                result.add_error(ValidationError(
                    code="MISSING_LANGUAGE",
                    message=f"Source {source_id} missing language attribute",
                    severity=ValidationSeverity.WARNING
                ))
            
            result.validated_items += 1
    
    def _validate_sentences(self, root: ET.Element, result: ValidationResult):
        """Validate sentence elements"""
        sentences = root.findall(".//sentence")
        
        sentence_ids = set()
        
        for sentence in sentences:
            sent_id = sentence.get("id")
            
            if not sent_id:
                result.add_error(ValidationError(
                    code="MISSING_SENTENCE_ID",
                    message="Sentence element missing id attribute",
                    severity=ValidationSeverity.ERROR
                ))
            elif sent_id in sentence_ids:
                result.add_error(ValidationError(
                    code="DUPLICATE_SENTENCE_ID",
                    message=f"Duplicate sentence id: {sent_id}",
                    severity=ValidationSeverity.ERROR
                ))
            else:
                sentence_ids.add(sent_id)
            
            tokens = sentence.findall(".//token")
            if not tokens:
                result.add_error(ValidationError(
                    code="EMPTY_SENTENCE",
                    message=f"Sentence {sent_id} has no tokens",
                    severity=ValidationSeverity.WARNING,
                    location=sent_id
                ))
            
            result.validated_items += 1
    
    def _validate_tokens(self, root: ET.Element, result: ValidationResult):
        """Validate token elements"""
        tokens = root.findall(".//token")
        
        token_ids = set()
        
        for token in tokens:
            token_id = token.get("id")
            
            if not token_id:
                result.add_error(ValidationError(
                    code="MISSING_TOKEN_ID",
                    message="Token element missing id attribute",
                    severity=ValidationSeverity.ERROR
                ))
            elif token_id in token_ids:
                result.add_error(ValidationError(
                    code="DUPLICATE_TOKEN_ID",
                    message=f"Duplicate token id: {token_id}",
                    severity=ValidationSeverity.ERROR
                ))
            else:
                token_ids.add(token_id)
            
            form = token.get("form")
            if not form and not token.get("empty-token-sort"):
                result.add_error(ValidationError(
                    code="MISSING_FORM",
                    message=f"Token {token_id} missing form attribute",
                    severity=ValidationSeverity.ERROR,
                    location=token_id
                ))
            
            pos = token.get("part-of-speech")
            if pos and self.level in [ValidationLevel.STRICT, ValidationLevel.PROIEL]:
                if pos not in PROIEL_POS_TAGS:
                    result.add_error(ValidationError(
                        code="INVALID_POS",
                        message=f"Invalid POS tag '{pos}' for token {token_id}",
                        severity=ValidationSeverity.WARNING,
                        location=token_id,
                        actual=pos
                    ))
            
            relation = token.get("relation")
            if relation and self.level in [ValidationLevel.STRICT, ValidationLevel.PROIEL]:
                if relation not in PROIEL_RELATIONS:
                    result.add_error(ValidationError(
                        code="INVALID_RELATION",
                        message=f"Invalid relation '{relation}' for token {token_id}",
                        severity=ValidationSeverity.WARNING,
                        location=token_id,
                        actual=relation
                    ))
            
            head_id = token.get("head-id")
            if head_id and head_id not in token_ids and head_id != "0":
                pass
            
            result.validated_items += 1


class CoNLLUValidator(SchemaValidator):
    """Validator for CoNLL-U format"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(level)
    
    def validate(self, content: Union[str, Path]) -> ValidationResult:
        """Validate CoNLL-U content"""
        result = ValidationResult(
            valid=True,
            validation_level=self.level
        )
        
        if isinstance(content, Path):
            try:
                with open(content, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                result.add_error(ValidationError(
                    code="FILE_READ_ERROR",
                    message=f"Failed to read file: {str(e)}",
                    severity=ValidationSeverity.CRITICAL
                ))
                return result
        
        lines = content.split("\n")
        
        sentence_count = 0
        token_count = 0
        current_sentence_tokens = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                if current_sentence_tokens:
                    self._validate_sentence_tokens(
                        current_sentence_tokens, sentence_count, result
                    )
                    sentence_count += 1
                    current_sentence_tokens = []
                continue
            
            if line.startswith("#"):
                continue
            
            fields = line.split("\t")
            
            if len(fields) != 10:
                result.add_error(ValidationError(
                    code="INVALID_FIELD_COUNT",
                    message=f"Line {line_num}: Expected 10 fields, found {len(fields)}",
                    severity=ValidationSeverity.ERROR,
                    line_number=line_num,
                    expected="10",
                    actual=str(len(fields))
                ))
                continue
            
            token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
            
            if not re.match(r"^\d+(-\d+)?(\.\d+)?$", token_id):
                result.add_error(ValidationError(
                    code="INVALID_TOKEN_ID",
                    message=f"Line {line_num}: Invalid token ID format '{token_id}'",
                    severity=ValidationSeverity.ERROR,
                    line_number=line_num
                ))
            
            if upos != "_" and upos not in UD_POS_TAGS:
                result.add_error(ValidationError(
                    code="INVALID_UPOS",
                    message=f"Line {line_num}: Invalid UPOS tag '{upos}'",
                    severity=ValidationSeverity.WARNING,
                    line_number=line_num,
                    actual=upos
                ))
            
            if head != "_" and not head.isdigit():
                result.add_error(ValidationError(
                    code="INVALID_HEAD",
                    message=f"Line {line_num}: Invalid head '{head}'",
                    severity=ValidationSeverity.ERROR,
                    line_number=line_num
                ))
            
            base_deprel = deprel.split(":")[0] if ":" in deprel else deprel
            if deprel != "_" and base_deprel not in UD_RELATIONS:
                result.add_error(ValidationError(
                    code="INVALID_DEPREL",
                    message=f"Line {line_num}: Invalid deprel '{deprel}'",
                    severity=ValidationSeverity.WARNING,
                    line_number=line_num,
                    actual=deprel
                ))
            
            current_sentence_tokens.append({
                "id": token_id,
                "head": head,
                "line": line_num
            })
            
            token_count += 1
            result.validated_items += 1
        
        if current_sentence_tokens:
            self._validate_sentence_tokens(
                current_sentence_tokens, sentence_count, result
            )
        
        result.metadata["sentence_count"] = sentence_count
        result.metadata["token_count"] = token_count
        
        return result
    
    def _validate_sentence_tokens(
        self,
        tokens: List[Dict[str, Any]],
        sentence_num: int,
        result: ValidationResult
    ):
        """Validate tokens within a sentence"""
        token_ids = set()
        
        for token in tokens:
            tid = token["id"]
            
            if "-" in tid:
                continue
            
            if tid in token_ids:
                result.add_error(ValidationError(
                    code="DUPLICATE_TOKEN_ID",
                    message=f"Sentence {sentence_num}: Duplicate token ID '{tid}'",
                    severity=ValidationSeverity.ERROR,
                    line_number=token["line"]
                ))
            else:
                token_ids.add(tid)
        
        has_root = False
        for token in tokens:
            if token["head"] == "0":
                if has_root:
                    result.add_error(ValidationError(
                        code="MULTIPLE_ROOTS",
                        message=f"Sentence {sentence_num}: Multiple root tokens",
                        severity=ValidationSeverity.WARNING
                    ))
                has_root = True
        
        if not has_root and tokens:
            result.add_error(ValidationError(
                code="NO_ROOT",
                message=f"Sentence {sentence_num}: No root token found",
                severity=ValidationSeverity.WARNING
            ))


class DocumentValidator(SchemaValidator):
    """Validator for internal Document objects"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(level)
    
    def validate(self, document: Document) -> ValidationResult:
        """Validate a Document object"""
        result = ValidationResult(
            valid=True,
            validation_level=self.level
        )
        
        if not document.id:
            result.add_error(ValidationError(
                code="MISSING_DOCUMENT_ID",
                message="Document missing id",
                severity=ValidationSeverity.ERROR
            ))
        
        if not document.text and not document.sentences:
            result.add_error(ValidationError(
                code="EMPTY_DOCUMENT",
                message="Document has no text or sentences",
                severity=ValidationSeverity.WARNING
            ))
        
        for i, sentence in enumerate(document.sentences):
            self._validate_sentence(sentence, i, result)
        
        result.validated_items = len(document.sentences)
        
        return result
    
    def _validate_sentence(
        self,
        sentence: Sentence,
        index: int,
        result: ValidationResult
    ):
        """Validate a sentence"""
        if not sentence.id:
            result.add_error(ValidationError(
                code="MISSING_SENTENCE_ID",
                message=f"Sentence at index {index} missing id",
                severity=ValidationSeverity.WARNING
            ))
        
        if not sentence.tokens:
            result.add_error(ValidationError(
                code="EMPTY_SENTENCE",
                message=f"Sentence {sentence.id or index} has no tokens",
                severity=ValidationSeverity.WARNING
            ))
            return
        
        for j, token in enumerate(sentence.tokens):
            self._validate_token(token, j, sentence.id or str(index), result)
    
    def _validate_token(
        self,
        token: Token,
        index: int,
        sentence_id: str,
        result: ValidationResult
    ):
        """Validate a token"""
        if not token.form:
            result.add_error(ValidationError(
                code="MISSING_TOKEN_FORM",
                message=f"Token at index {index} in sentence {sentence_id} missing form",
                severity=ValidationSeverity.ERROR
            ))
        
        if self.level in [ValidationLevel.STRICT, ValidationLevel.ERC]:
            if not token.lemma:
                result.add_error(ValidationError(
                    code="MISSING_LEMMA",
                    message=f"Token '{token.form}' in sentence {sentence_id} missing lemma",
                    severity=ValidationSeverity.WARNING
                ))
            
            if not token.pos:
                result.add_error(ValidationError(
                    code="MISSING_POS",
                    message=f"Token '{token.form}' in sentence {sentence_id} missing POS",
                    severity=ValidationSeverity.WARNING
                ))


def validate_proiel_xml(
    content: Union[str, Path],
    level: ValidationLevel = ValidationLevel.PROIEL
) -> ValidationResult:
    """Validate PROIEL XML content"""
    validator = PROIELValidator(level)
    return validator.validate(content)


def validate_conllu(
    content: Union[str, Path],
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate CoNLL-U content"""
    validator = CoNLLUValidator(level)
    return validator.validate(content)


def validate_document(
    document: Document,
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate a Document object"""
    validator = DocumentValidator(level)
    return validator.validate(document)
