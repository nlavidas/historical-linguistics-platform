"""
HLP Pipelines Valency - Valency Extraction Pipeline

This module provides a comprehensive pipeline for extracting
valency patterns from annotated texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from hlp_core.models import Language, Period, Document, Corpus, Sentence
from hlp_valency.pattern_extractor import (
    ValencyExtractor, ExtractionConfig, ExtractionResult,
    extract_valency_frames, ExtractedFrame
)
from hlp_valency.pattern_normalization import (
    PatternNormalizer, NormalizationConfig, NormalizedPattern
)
from hlp_valency.lexicon_builder import (
    LexiconBuilder, ValencyLexicon
)
from hlp_valency.reports import (
    ValencyReporter, ReportConfig, ValencyReport
)

logger = logging.getLogger(__name__)


class ValencyStatus(Enum):
    """Status of a valency job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValencyPriority(Enum):
    """Priority levels for valency jobs"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class ValencyConfig:
    """Configuration for valency pipeline"""
    extraction_config: Optional[ExtractionConfig] = None
    
    normalization_config: Optional[NormalizationConfig] = None
    
    build_lexicon: bool = True
    
    generate_reports: bool = True
    
    report_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    
    output_dir: Optional[str] = None
    
    min_frequency: int = 1
    
    include_auxiliaries: bool = False
    
    include_copulars: bool = True
    
    track_periods: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValencyJob:
    """Represents a valency extraction job"""
    id: str
    
    corpus_id: str
    
    document_ids: List[str]
    
    language: Language
    
    status: ValencyStatus = ValencyStatus.PENDING
    
    priority: ValencyPriority = ValencyPriority.NORMAL
    
    created_at: datetime = field(default_factory=datetime.now)
    
    started_at: Optional[datetime] = None
    
    completed_at: Optional[datetime] = None
    
    progress: float = 0.0
    
    frames_extracted: int = 0
    
    patterns_normalized: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark job as started"""
        self.status = ValencyStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self):
        """Mark job as completed"""
        self.status = ValencyStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
    
    def mark_failed(self, error: str):
        """Mark job as failed"""
        self.status = ValencyStatus.FAILED
        self.completed_at = datetime.now()
        self.errors.append(error)
    
    def update_progress(self, completed: int, total: int):
        """Update progress"""
        if total > 0:
            self.progress = completed / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "corpus_id": self.corpus_id,
            "document_count": len(self.document_ids),
            "language": self.language.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "frames_extracted": self.frames_extracted,
            "patterns_normalized": self.patterns_normalized,
            "error_count": len(self.errors)
        }


@dataclass
class ValencyResult:
    """Result of a valency extraction operation"""
    job_id: str
    
    success: bool
    
    frames: List[ExtractedFrame] = field(default_factory=list)
    
    normalized_patterns: List[NormalizedPattern] = field(default_factory=list)
    
    lexicon: Optional[ValencyLexicon] = None
    
    reports: List[ValencyReport] = field(default_factory=list)
    
    frames_extracted: int = 0
    
    patterns_normalized: int = 0
    
    unique_verbs: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    duration_seconds: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "frames_extracted": self.frames_extracted,
            "patterns_normalized": self.patterns_normalized,
            "unique_verbs": self.unique_verbs,
            "has_lexicon": self.lexicon is not None,
            "report_count": len(self.reports),
            "error_count": len(self.errors),
            "duration_seconds": self.duration_seconds
        }


class ValencyPipeline:
    """Pipeline for extracting valency patterns"""
    
    def __init__(self, config: Optional[ValencyConfig] = None):
        self.config = config or ValencyConfig()
        self._jobs: Dict[str, ValencyJob] = {}
        self._extractor: Optional[ValencyExtractor] = None
        self._normalizer: Optional[PatternNormalizer] = None
        self._lexicon_builder: Optional[LexiconBuilder] = None
        self._reporter: Optional[ValencyReporter] = None
    
    def _get_extractor(self) -> ValencyExtractor:
        """Get or create extractor"""
        if self._extractor is None:
            self._extractor = ValencyExtractor(self.config.extraction_config)
        return self._extractor
    
    def _get_normalizer(self) -> PatternNormalizer:
        """Get or create normalizer"""
        if self._normalizer is None:
            self._normalizer = PatternNormalizer(self.config.normalization_config)
        return self._normalizer
    
    def _get_lexicon_builder(self, language: Language) -> LexiconBuilder:
        """Get or create lexicon builder"""
        if self._lexicon_builder is None:
            self._lexicon_builder = LexiconBuilder(
                language=language,
                normalization_config=self.config.normalization_config
            )
        return self._lexicon_builder
    
    def _get_reporter(self) -> ValencyReporter:
        """Get or create reporter"""
        if self._reporter is None:
            self._reporter = ValencyReporter()
        return self._reporter
    
    def create_job(
        self,
        corpus: Corpus,
        document_ids: Optional[List[str]] = None,
        priority: ValencyPriority = ValencyPriority.NORMAL
    ) -> ValencyJob:
        """Create a new valency job"""
        if document_ids is None:
            document_ids = [doc.id for doc in corpus.documents]
        
        job = ValencyJob(
            id=str(uuid.uuid4()),
            corpus_id=corpus.id,
            document_ids=document_ids,
            language=corpus.language or Language.ANCIENT_GREEK,
            priority=priority
        )
        
        self._jobs[job.id] = job
        logger.info(f"Created valency job {job.id} for {len(document_ids)} documents")
        
        return job
    
    def run_job(self, job: ValencyJob, corpus: Corpus) -> ValencyResult:
        """Run a valency extraction job"""
        start_time = time.time()
        job.mark_started()
        
        all_frames = []
        all_patterns = []
        errors = []
        
        extractor = self._get_extractor()
        normalizer = self._get_normalizer()
        
        doc_map = {doc.id: doc for doc in corpus.documents}
        total = len(job.document_ids)
        
        for i, doc_id in enumerate(job.document_ids):
            doc = doc_map.get(doc_id)
            if not doc:
                errors.append(f"Document not found: {doc_id}")
                continue
            
            try:
                extraction_result = extractor.extract_from_document(doc)
                all_frames.extend(extraction_result.frames)
                
                for frame in extraction_result.frames:
                    normalized = normalizer.normalize_frame(frame)
                    all_patterns.append(normalized)
                
            except Exception as e:
                errors.append(f"Error processing {doc_id}: {str(e)}")
                logger.error(f"Error processing {doc_id}: {e}")
            
            job.update_progress(i + 1, total)
            job.frames_extracted = len(all_frames)
            job.patterns_normalized = len(all_patterns)
        
        lexicon = None
        if self.config.build_lexicon and all_frames:
            lexicon_builder = self._get_lexicon_builder(job.language)
            lexicon = lexicon_builder.build_from_frames(
                all_frames,
                lexicon_name=f"{corpus.name}_valency"
            )
        
        reports = []
        if self.config.generate_reports and lexicon:
            reporter = self._get_reporter()
            
            summary_report = reporter.generate_summary_report(lexicon)
            reports.append(summary_report)
            
            if self.config.output_dir:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for fmt in self.config.report_formats:
                    from hlp_valency.reports import ReportFormat
                    report_format = ReportFormat(fmt)
                    reporter.export_report(
                        summary_report,
                        output_dir / f"valency_report.{fmt}",
                        report_format
                    )
        
        job.mark_completed()
        job.errors = errors
        
        duration = time.time() - start_time
        
        unique_verbs = len(set(f.verb_lemma for f in all_frames))
        
        return ValencyResult(
            job_id=job.id,
            success=len(errors) == 0,
            frames=all_frames,
            normalized_patterns=all_patterns,
            lexicon=lexicon,
            reports=reports,
            frames_extracted=len(all_frames),
            patterns_normalized=len(all_patterns),
            unique_verbs=unique_verbs,
            errors=errors,
            duration_seconds=duration
        )
    
    def extract_from_corpus(
        self,
        corpus: Corpus,
        document_ids: Optional[List[str]] = None
    ) -> ValencyResult:
        """Extract valency from a corpus"""
        job = self.create_job(corpus, document_ids)
        return self.run_job(job, corpus)
    
    def extract_from_sentences(
        self,
        sentences: List[Sentence],
        language: Language = Language.ANCIENT_GREEK
    ) -> ExtractionResult:
        """Extract valency from sentences"""
        extractor = self._get_extractor()
        
        all_frames = []
        
        for sentence in sentences:
            result = extractor.extract_from_sentence(sentence)
            all_frames.extend(result.frames)
        
        return ExtractionResult(
            frames=all_frames,
            total_sentences=len(sentences),
            total_verbs=len(all_frames)
        )
    
    def build_lexicon_from_frames(
        self,
        frames: List[ExtractedFrame],
        language: Language = Language.ANCIENT_GREEK,
        name: str = "valency_lexicon"
    ) -> ValencyLexicon:
        """Build lexicon from frames"""
        lexicon_builder = self._get_lexicon_builder(language)
        return lexicon_builder.build_from_frames(frames, name)
    
    def get_job(self, job_id: str) -> Optional[ValencyJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[ValencyStatus] = None
    ) -> List[ValencyJob]:
        """List jobs"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(job_id)
        if job and job.status in [ValencyStatus.PENDING, ValencyStatus.RUNNING]:
            job.status = ValencyStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False


def run_valency_pipeline(
    corpus: Corpus,
    config: Optional[ValencyConfig] = None
) -> ValencyResult:
    """Run valency pipeline"""
    pipeline = ValencyPipeline(config)
    return pipeline.extract_from_corpus(corpus)


def schedule_valency_extraction(
    corpus: Corpus,
    document_ids: Optional[List[str]] = None,
    priority: ValencyPriority = ValencyPriority.NORMAL,
    config: Optional[ValencyConfig] = None
) -> ValencyJob:
    """Schedule a valency extraction job"""
    pipeline = ValencyPipeline(config)
    return pipeline.create_job(corpus, document_ids, priority)
