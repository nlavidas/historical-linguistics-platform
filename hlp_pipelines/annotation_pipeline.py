"""
HLP Pipelines Annotation - Annotation Scheduling Pipeline

This module provides a comprehensive pipeline for scheduling and
running annotation tasks on texts.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from hlp_core.models import (
    Language, Document, Corpus, Sentence, Token,
    MorphologyAnnotation, SyntaxAnnotation
)
from hlp_annotation.base_engine import AnnotationEngine, AnnotationResult as EngineResult
from hlp_annotation.stanza_engine import StanzaEngine, StanzaConfig, create_stanza_engine
from hlp_annotation.spacy_engine import SpacyEngine, SpacyConfig, create_spacy_engine
from hlp_annotation.ensemble import EnsembleEngine, EnsembleConfig, create_ensemble

logger = logging.getLogger(__name__)


class AnnotationStatus(Enum):
    """Status of an annotation job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AnnotationLevel(Enum):
    """Levels of annotation"""
    TOKENIZATION = "tokenization"
    POS_TAGGING = "pos_tagging"
    LEMMATIZATION = "lemmatization"
    MORPHOLOGY = "morphology"
    DEPENDENCY = "dependency"
    NER = "ner"
    SRL = "srl"
    FULL = "full"


class AnnotationPriority(Enum):
    """Priority levels for annotation jobs"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class AnnotationConfig:
    """Configuration for annotation pipeline"""
    max_workers: int = 2
    
    batch_size: int = 100
    
    timeout: float = 600.0
    
    max_retries: int = 3
    
    engines: List[str] = field(default_factory=lambda: ["stanza"])
    
    use_ensemble: bool = False
    
    annotation_levels: List[AnnotationLevel] = field(
        default_factory=lambda: [AnnotationLevel.FULL]
    )
    
    use_gpu: bool = False
    
    cache_models: bool = True
    
    output_format: str = "proiel"
    
    validate_output: bool = True
    
    store_intermediate: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationJob:
    """Represents an annotation job"""
    id: str
    
    corpus_id: str
    
    document_ids: List[str]
    
    language: Language
    
    status: AnnotationStatus = AnnotationStatus.PENDING
    
    priority: AnnotationPriority = AnnotationPriority.NORMAL
    
    annotation_levels: List[AnnotationLevel] = field(default_factory=list)
    
    engines: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    started_at: Optional[datetime] = None
    
    completed_at: Optional[datetime] = None
    
    progress: float = 0.0
    
    sentences_processed: int = 0
    
    tokens_processed: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark job as started"""
        self.status = AnnotationStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self):
        """Mark job as completed"""
        self.status = AnnotationStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
    
    def mark_failed(self, error: str):
        """Mark job as failed"""
        self.status = AnnotationStatus.FAILED
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
            "annotation_levels": [l.value for l in self.annotation_levels],
            "engines": self.engines,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "sentences_processed": self.sentences_processed,
            "tokens_processed": self.tokens_processed,
            "error_count": len(self.errors)
        }


@dataclass
class AnnotationResult:
    """Result of an annotation operation"""
    job_id: str
    
    success: bool
    
    annotated_documents: List[Document] = field(default_factory=list)
    
    annotated_corpus: Optional[Corpus] = None
    
    sentences_processed: int = 0
    
    tokens_processed: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    duration_seconds: float = 0.0
    
    engine_stats: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "document_count": len(self.annotated_documents),
            "sentences_processed": self.sentences_processed,
            "tokens_processed": self.tokens_processed,
            "error_count": len(self.errors),
            "duration_seconds": self.duration_seconds,
            "engine_stats": self.engine_stats
        }


class AnnotationPipeline:
    """Pipeline for annotating texts"""
    
    def __init__(self, config: Optional[AnnotationConfig] = None):
        self.config = config or AnnotationConfig()
        self._jobs: Dict[str, AnnotationJob] = {}
        self._engines: Dict[str, AnnotationEngine] = {}
        self._ensemble: Optional[EnsembleEngine] = None
    
    def _get_engine(self, engine_name: str, language: Language) -> Optional[AnnotationEngine]:
        """Get or create annotation engine"""
        key = f"{engine_name}_{language.value}"
        
        if key not in self._engines:
            if engine_name == "stanza":
                self._engines[key] = create_stanza_engine(
                    language=language.value,
                    use_gpu=self.config.use_gpu
                )
            elif engine_name == "spacy":
                self._engines[key] = create_spacy_engine(
                    language=language.value,
                    use_gpu=self.config.use_gpu
                )
        
        engine = self._engines.get(key)
        
        if engine and not engine.is_initialized():
            engine.initialize()
        
        return engine
    
    def _get_ensemble(self, language: Language) -> EnsembleEngine:
        """Get or create ensemble engine"""
        if self._ensemble is None:
            engines = []
            for engine_name in self.config.engines:
                engine = self._get_engine(engine_name, language)
                if engine:
                    engines.append(engine)
            
            self._ensemble = create_ensemble(engines)
        
        return self._ensemble
    
    def create_job(
        self,
        corpus: Corpus,
        document_ids: Optional[List[str]] = None,
        priority: AnnotationPriority = AnnotationPriority.NORMAL
    ) -> AnnotationJob:
        """Create a new annotation job"""
        if document_ids is None:
            document_ids = [doc.id for doc in corpus.documents]
        
        job = AnnotationJob(
            id=str(uuid.uuid4()),
            corpus_id=corpus.id,
            document_ids=document_ids,
            language=corpus.language or Language.ANCIENT_GREEK,
            priority=priority,
            annotation_levels=self.config.annotation_levels,
            engines=self.config.engines
        )
        
        self._jobs[job.id] = job
        logger.info(f"Created annotation job {job.id} for {len(document_ids)} documents")
        
        return job
    
    def run_job(self, job: AnnotationJob, corpus: Corpus) -> AnnotationResult:
        """Run an annotation job"""
        start_time = time.time()
        job.mark_started()
        
        annotated_documents = []
        errors = []
        total_sentences = 0
        total_tokens = 0
        
        if self.config.use_ensemble:
            engine = self._get_ensemble(job.language)
        else:
            engine = self._get_engine(self.config.engines[0], job.language)
        
        if not engine:
            job.mark_failed("No annotation engine available")
            return AnnotationResult(
                job_id=job.id,
                success=False,
                errors=["No annotation engine available"]
            )
        
        doc_map = {doc.id: doc for doc in corpus.documents}
        total = len(job.document_ids)
        
        for i, doc_id in enumerate(job.document_ids):
            doc = doc_map.get(doc_id)
            if not doc:
                errors.append(f"Document not found: {doc_id}")
                continue
            
            try:
                annotated_doc = self._annotate_document(engine, doc, job.language)
                annotated_documents.append(annotated_doc)
                
                total_sentences += len(annotated_doc.sentences)
                total_tokens += sum(
                    len(s.tokens) for s in annotated_doc.sentences
                )
                
            except Exception as e:
                errors.append(f"Error annotating {doc_id}: {str(e)}")
                logger.error(f"Error annotating {doc_id}: {e}")
            
            job.update_progress(i + 1, total)
            job.sentences_processed = total_sentences
            job.tokens_processed = total_tokens
        
        job.mark_completed()
        job.errors = errors
        
        duration = time.time() - start_time
        
        annotated_corpus = Corpus(
            id=f"{corpus.id}_annotated",
            name=f"{corpus.name} (Annotated)",
            language=corpus.language,
            documents=annotated_documents,
            metadata={
                **corpus.metadata,
                "annotation_job_id": job.id,
                "engines": job.engines
            }
        )
        
        return AnnotationResult(
            job_id=job.id,
            success=len(errors) == 0,
            annotated_documents=annotated_documents,
            annotated_corpus=annotated_corpus,
            sentences_processed=total_sentences,
            tokens_processed=total_tokens,
            errors=errors,
            duration_seconds=duration,
            engine_stats=engine.get_metrics() if hasattr(engine, 'get_metrics') else {}
        )
    
    def _annotate_document(
        self,
        engine: AnnotationEngine,
        document: Document,
        language: Language
    ) -> Document:
        """Annotate a single document"""
        text = document.text
        
        result = engine.process(text)
        
        sentences = []
        for sent_data in result.sentences:
            tokens = []
            for tok_data in sent_data.get("tokens", []):
                token = Token(
                    id=tok_data.get("id", ""),
                    form=tok_data.get("form", ""),
                    lemma=tok_data.get("lemma", ""),
                    pos=tok_data.get("upos", ""),
                    xpos=tok_data.get("xpos", ""),
                    morphology=self._create_morphology(tok_data),
                    syntax=self._create_syntax(tok_data)
                )
                tokens.append(token)
            
            sentence = Sentence(
                id=sent_data.get("id", ""),
                text=sent_data.get("text", ""),
                tokens=tokens,
                language=language
            )
            sentences.append(sentence)
        
        return Document(
            id=document.id,
            title=document.title,
            author=document.author,
            language=language,
            text=text,
            sentences=sentences,
            metadata={
                **document.metadata,
                "annotated": True,
                "annotation_engine": engine.__class__.__name__
            }
        )
    
    def _create_morphology(self, tok_data: Dict[str, Any]) -> Optional[MorphologyAnnotation]:
        """Create morphology annotation from token data"""
        feats = tok_data.get("feats", {})
        if not feats:
            return None
        
        from hlp_core.models import Case, Number, Gender, Person, Tense, Mood, Voice
        
        case_map = {
            "Nom": Case.NOMINATIVE,
            "Gen": Case.GENITIVE,
            "Dat": Case.DATIVE,
            "Acc": Case.ACCUSATIVE,
            "Voc": Case.VOCATIVE,
            "Abl": Case.ABLATIVE,
        }
        
        return MorphologyAnnotation(
            case=case_map.get(feats.get("Case")),
            number=feats.get("Number"),
            gender=feats.get("Gender"),
            person=feats.get("Person"),
            tense=feats.get("Tense"),
            mood=feats.get("Mood"),
            voice=feats.get("Voice"),
            ud_feats=feats
        )
    
    def _create_syntax(self, tok_data: Dict[str, Any]) -> Optional[SyntaxAnnotation]:
        """Create syntax annotation from token data"""
        head = tok_data.get("head")
        deprel = tok_data.get("deprel")
        
        if head is None and not deprel:
            return None
        
        return SyntaxAnnotation(
            head=str(head) if head is not None else None,
            deprel=deprel
        )
    
    def annotate_corpus(
        self,
        corpus: Corpus,
        document_ids: Optional[List[str]] = None
    ) -> AnnotationResult:
        """Annotate a corpus"""
        job = self.create_job(corpus, document_ids)
        return self.run_job(job, corpus)
    
    def annotate_text(
        self,
        text: str,
        language: Language = Language.ANCIENT_GREEK
    ) -> EngineResult:
        """Annotate a single text"""
        if self.config.use_ensemble:
            engine = self._get_ensemble(language)
        else:
            engine = self._get_engine(self.config.engines[0], language)
        
        if not engine:
            raise RuntimeError("No annotation engine available")
        
        return engine.process(text)
    
    def get_job(self, job_id: str) -> Optional[AnnotationJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[AnnotationStatus] = None
    ) -> List[AnnotationJob]:
        """List jobs"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(job_id)
        if job and job.status in [AnnotationStatus.PENDING, AnnotationStatus.RUNNING]:
            job.status = AnnotationStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False
    
    def shutdown(self):
        """Shutdown all engines"""
        for engine in self._engines.values():
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
        
        if self._ensemble and hasattr(self._ensemble, 'shutdown'):
            self._ensemble.shutdown()


def run_annotation_pipeline(
    corpus: Corpus,
    config: Optional[AnnotationConfig] = None
) -> AnnotationResult:
    """Run annotation pipeline"""
    pipeline = AnnotationPipeline(config)
    try:
        return pipeline.annotate_corpus(corpus)
    finally:
        pipeline.shutdown()


def schedule_annotation(
    corpus: Corpus,
    document_ids: Optional[List[str]] = None,
    priority: AnnotationPriority = AnnotationPriority.NORMAL,
    config: Optional[AnnotationConfig] = None
) -> AnnotationJob:
    """Schedule an annotation job"""
    pipeline = AnnotationPipeline(config)
    return pipeline.create_job(corpus, document_ids, priority)
