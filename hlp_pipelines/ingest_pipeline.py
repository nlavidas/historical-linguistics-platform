"""
HLP Pipelines Ingest - Text Ingestion Workflow

This module provides a comprehensive pipeline for ingesting texts
from various sources into the platform.

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

from hlp_core.models import Language, Period, Document, Corpus
from hlp_ingest.source_registry import (
    SourceRegistry, SourceDefinition, SourceType, get_registry
)
from hlp_ingest.perseus import PerseusClient, PerseusConfig
from hlp_ingest.first1k_greek import First1KGreekClient, First1KGreekConfig
from hlp_ingest.gutenberg import GutenbergClient, GutenbergConfig
from hlp_ingest.proiel_treebanks import PROIELTreebankClient, PROIELTreebankConfig

logger = logging.getLogger(__name__)


class IngestStatus(Enum):
    """Status of an ingest job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class IngestPriority(Enum):
    """Priority levels for ingest jobs"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class IngestConfig:
    """Configuration for ingest pipeline"""
    max_workers: int = 4
    
    batch_size: int = 10
    
    rate_limit: float = 1.0
    
    timeout: float = 300.0
    
    max_retries: int = 3
    
    retry_delay: float = 5.0
    
    cache_dir: Optional[str] = None
    
    output_dir: Optional[str] = None
    
    validate_texts: bool = True
    
    deduplicate: bool = True
    
    store_raw: bool = True
    
    languages: List[Language] = field(default_factory=list)
    
    periods: List[Period] = field(default_factory=list)
    
    sources: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestJob:
    """Represents an ingest job"""
    id: str
    
    source_type: SourceType
    
    source_id: str
    
    target_ids: List[str]
    
    status: IngestStatus = IngestStatus.PENDING
    
    priority: IngestPriority = IngestPriority.NORMAL
    
    created_at: datetime = field(default_factory=datetime.now)
    
    started_at: Optional[datetime] = None
    
    completed_at: Optional[datetime] = None
    
    progress: float = 0.0
    
    documents_ingested: int = 0
    
    documents_failed: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self):
        """Mark job as started"""
        self.status = IngestStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self):
        """Mark job as completed"""
        self.status = IngestStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
    
    def mark_failed(self, error: str):
        """Mark job as failed"""
        self.status = IngestStatus.FAILED
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
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "target_count": len(self.target_ids),
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "documents_ingested": self.documents_ingested,
            "documents_failed": self.documents_failed,
            "error_count": len(self.errors)
        }


@dataclass
class IngestResult:
    """Result of an ingest operation"""
    job_id: str
    
    success: bool
    
    documents: List[Document] = field(default_factory=list)
    
    corpus: Optional[Corpus] = None
    
    documents_ingested: int = 0
    
    documents_failed: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    duration_seconds: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "document_count": len(self.documents),
            "documents_ingested": self.documents_ingested,
            "documents_failed": self.documents_failed,
            "error_count": len(self.errors),
            "duration_seconds": self.duration_seconds
        }


class IngestPipeline:
    """Pipeline for ingesting texts from various sources"""
    
    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()
        self._registry = get_registry()
        self._jobs: Dict[str, IngestJob] = {}
        self._clients: Dict[SourceType, Any] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def _get_client(self, source_type: SourceType) -> Any:
        """Get or create client for source type"""
        if source_type not in self._clients:
            if source_type == SourceType.PERSEUS:
                self._clients[source_type] = PerseusClient(PerseusConfig(
                    cache_dir=self.config.cache_dir,
                    rate_limit=self.config.rate_limit
                ))
            elif source_type == SourceType.FIRST1K_GREEK:
                self._clients[source_type] = First1KGreekClient(First1KGreekConfig(
                    cache_dir=self.config.cache_dir,
                    rate_limit=self.config.rate_limit
                ))
            elif source_type == SourceType.GUTENBERG:
                self._clients[source_type] = GutenbergClient(GutenbergConfig(
                    cache_dir=self.config.cache_dir,
                    rate_limit=self.config.rate_limit
                ))
            elif source_type == SourceType.PROIEL:
                self._clients[source_type] = PROIELTreebankClient(PROIELTreebankConfig(
                    cache_dir=self.config.cache_dir,
                    rate_limit=self.config.rate_limit
                ))
        
        return self._clients.get(source_type)
    
    def create_job(
        self,
        source_type: SourceType,
        source_id: str,
        target_ids: List[str],
        priority: IngestPriority = IngestPriority.NORMAL
    ) -> IngestJob:
        """Create a new ingest job"""
        job = IngestJob(
            id=str(uuid.uuid4()),
            source_type=source_type,
            source_id=source_id,
            target_ids=target_ids,
            priority=priority
        )
        
        self._jobs[job.id] = job
        logger.info(f"Created ingest job {job.id} for {len(target_ids)} targets")
        
        return job
    
    def run_job(self, job: IngestJob) -> IngestResult:
        """Run an ingest job"""
        start_time = time.time()
        job.mark_started()
        
        documents = []
        errors = []
        
        client = self._get_client(job.source_type)
        if not client:
            job.mark_failed(f"No client available for {job.source_type.value}")
            return IngestResult(
                job_id=job.id,
                success=False,
                errors=[f"No client available for {job.source_type.value}"]
            )
        
        total = len(job.target_ids)
        
        for i, target_id in enumerate(job.target_ids):
            try:
                doc = self._ingest_single(client, job.source_type, target_id)
                if doc:
                    documents.append(doc)
                    job.documents_ingested += 1
                else:
                    job.documents_failed += 1
                    errors.append(f"Failed to ingest {target_id}")
                    
            except Exception as e:
                job.documents_failed += 1
                errors.append(f"Error ingesting {target_id}: {str(e)}")
                logger.error(f"Error ingesting {target_id}: {e}")
            
            job.update_progress(i + 1, total)
        
        job.mark_completed()
        job.errors = errors
        
        duration = time.time() - start_time
        
        corpus = None
        if documents:
            corpus = Corpus(
                id=f"ingest_{job.id}",
                name=f"Ingested from {job.source_id}",
                documents=documents,
                metadata={
                    "source_type": job.source_type.value,
                    "source_id": job.source_id,
                    "job_id": job.id
                }
            )
        
        return IngestResult(
            job_id=job.id,
            success=job.documents_failed == 0,
            documents=documents,
            corpus=corpus,
            documents_ingested=job.documents_ingested,
            documents_failed=job.documents_failed,
            errors=errors,
            duration_seconds=duration
        )
    
    def _ingest_single(
        self,
        client: Any,
        source_type: SourceType,
        target_id: str
    ) -> Optional[Document]:
        """Ingest a single document"""
        if source_type == SourceType.PERSEUS:
            text = client.fetch_text(target_id)
            return text.to_document() if text else None
        
        elif source_type == SourceType.FIRST1K_GREEK:
            text = client.fetch_text(target_id)
            return text.to_document() if text else None
        
        elif source_type == SourceType.GUTENBERG:
            text = client.fetch_text(int(target_id))
            return text.to_document() if text else None
        
        elif source_type == SourceType.PROIEL:
            treebank = client.import_treebank(target_id)
            if treebank and treebank.documents:
                return treebank.documents[0]
            return None
        
        return None
    
    def run_parallel(
        self,
        jobs: List[IngestJob],
        max_workers: Optional[int] = None
    ) -> List[IngestResult]:
        """Run multiple jobs in parallel"""
        max_workers = max_workers or self.config.max_workers
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_job, job): job
                for job in jobs
            }
            
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Job {job.id} failed: {e}")
                    results.append(IngestResult(
                        job_id=job.id,
                        success=False,
                        errors=[str(e)]
                    ))
        
        return results
    
    def ingest_from_source(
        self,
        source_id: str,
        target_ids: Optional[List[str]] = None,
        max_documents: Optional[int] = None
    ) -> IngestResult:
        """Ingest from a registered source"""
        source = self._registry.get(source_id)
        if not source:
            return IngestResult(
                job_id="",
                success=False,
                errors=[f"Unknown source: {source_id}"]
            )
        
        if target_ids is None:
            target_ids = self._discover_targets(source, max_documents)
        
        job = self.create_job(
            source_type=source.source_type,
            source_id=source_id,
            target_ids=target_ids
        )
        
        return self.run_job(job)
    
    def _discover_targets(
        self,
        source: SourceDefinition,
        max_documents: Optional[int] = None
    ) -> List[str]:
        """Discover available targets from a source"""
        client = self._get_client(source.source_type)
        if not client:
            return []
        
        targets = []
        
        if source.source_type == SourceType.PERSEUS:
            works = client.list_works()
            targets = [w.urn for w in works]
        
        elif source.source_type == SourceType.FIRST1K_GREEK:
            works = client.list_works()
            targets = [w.file_path for w in works]
        
        elif source.source_type == SourceType.GUTENBERG:
            for lang in self.config.languages or [Language.ANCIENT_GREEK, Language.LATIN]:
                works = client.search(language=lang, max_results=100)
                targets.extend([str(w.gutenberg_id) for w in works])
        
        elif source.source_type == SourceType.PROIEL:
            treebanks = client.list_treebanks()
            targets = [tb["id"] for tb in treebanks]
        
        if max_documents:
            targets = targets[:max_documents]
        
        return targets
    
    def get_job(self, job_id: str) -> Optional[IngestJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[IngestStatus] = None
    ) -> List[IngestJob]:
        """List jobs"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(job_id)
        if job and job.status in [IngestStatus.PENDING, IngestStatus.RUNNING]:
            job.status = IngestStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False


def run_ingest_pipeline(
    source_id: str,
    target_ids: Optional[List[str]] = None,
    config: Optional[IngestConfig] = None
) -> IngestResult:
    """Run ingest pipeline"""
    pipeline = IngestPipeline(config)
    return pipeline.ingest_from_source(source_id, target_ids)


def schedule_ingest(
    source_id: str,
    target_ids: List[str],
    priority: IngestPriority = IngestPriority.NORMAL,
    config: Optional[IngestConfig] = None
) -> IngestJob:
    """Schedule an ingest job"""
    pipeline = IngestPipeline(config)
    source = get_registry().get(source_id)
    
    if not source:
        raise ValueError(f"Unknown source: {source_id}")
    
    return pipeline.create_job(
        source_type=source.source_type,
        source_id=source_id,
        target_ids=target_ids,
        priority=priority
    )
