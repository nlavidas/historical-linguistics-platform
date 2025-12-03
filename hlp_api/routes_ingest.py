"""
HLP API Routes Ingest - Text Ingestion Endpoints

This module provides REST API endpoints for text ingestion.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_ingest_jobs: Dict[str, Dict[str, Any]] = {}


class IngestSourceConfig(BaseModel):
    """Schema for ingest source configuration"""
    source_type: str = Field(..., description="Source type (perseus, first1k, gutenberg, proiel)")
    source_id: Optional[str] = Field(None, description="Specific source ID")
    target_ids: Optional[List[str]] = Field(None, description="Target text IDs")
    language: Optional[str] = Field(None, description="Language filter")
    period: Optional[str] = Field(None, description="Period filter")


class IngestJobCreate(BaseModel):
    """Schema for creating an ingest job"""
    corpus_id: str = Field(..., description="Target corpus ID")
    sources: List[IngestSourceConfig] = Field(..., description="Sources to ingest from")
    priority: str = Field("normal", description="Job priority")
    validate_texts: bool = Field(True, description="Validate ingested texts")


class IngestJobResponse(BaseModel):
    """Schema for ingest job response"""
    id: str
    corpus_id: str
    status: str
    progress: float
    documents_ingested: int
    documents_failed: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class SourceInfo(BaseModel):
    """Schema for source information"""
    id: str
    name: str
    source_type: str
    languages: List[str]
    description: str
    status: str


@router.get("/sources", response_model=List[SourceInfo])
async def list_sources():
    """List available text sources"""
    return [
        SourceInfo(
            id="perseus",
            name="Perseus Digital Library",
            source_type="perseus",
            languages=["grc", "la"],
            description="Classical Greek and Latin texts from Perseus",
            status="active"
        ),
        SourceInfo(
            id="first1k_greek",
            name="First1KGreek",
            source_type="first1k",
            languages=["grc"],
            description="First Thousand Years of Greek corpus",
            status="active"
        ),
        SourceInfo(
            id="gutenberg",
            name="Project Gutenberg",
            source_type="gutenberg",
            languages=["grc", "la", "en", "de", "fr"],
            description="Public domain texts from Gutenberg",
            status="active"
        ),
        SourceInfo(
            id="proiel",
            name="PROIEL Treebanks",
            source_type="proiel",
            languages=["grc", "la", "got", "cu", "xcl"],
            description="Annotated treebanks from PROIEL project",
            status="active"
        )
    ]


@router.get("/sources/{source_id}", response_model=SourceInfo)
async def get_source(
    source_id: str = Path(..., description="Source ID")
):
    """Get source information"""
    sources = {
        "perseus": SourceInfo(
            id="perseus",
            name="Perseus Digital Library",
            source_type="perseus",
            languages=["grc", "la"],
            description="Classical Greek and Latin texts from Perseus",
            status="active"
        ),
        "first1k_greek": SourceInfo(
            id="first1k_greek",
            name="First1KGreek",
            source_type="first1k",
            languages=["grc"],
            description="First Thousand Years of Greek corpus",
            status="active"
        )
    }
    
    source = sources.get(source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return source


@router.get("/sources/{source_id}/texts")
async def list_source_texts(
    source_id: str = Path(..., description="Source ID"),
    language: Optional[str] = Query(None, description="Language filter"),
    author: Optional[str] = Query(None, description="Author filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List available texts from a source"""
    texts = [
        {
            "id": "urn:cts:greekLit:tlg0012.tlg001",
            "title": "Iliad",
            "author": "Homer",
            "language": "grc",
            "period": "Archaic"
        },
        {
            "id": "urn:cts:greekLit:tlg0012.tlg002",
            "title": "Odyssey",
            "author": "Homer",
            "language": "grc",
            "period": "Archaic"
        },
        {
            "id": "urn:cts:greekLit:tlg0059.tlg001",
            "title": "Republic",
            "author": "Plato",
            "language": "grc",
            "period": "Classical"
        },
        {
            "id": "urn:cts:latinLit:phi0448.phi001",
            "title": "De Bello Gallico",
            "author": "Caesar",
            "language": "la",
            "period": "Classical"
        }
    ]
    
    if language:
        texts = [t for t in texts if t["language"] == language]
    
    if author:
        texts = [t for t in texts if author.lower() in t["author"].lower()]
    
    return {
        "source_id": source_id,
        "total": len(texts),
        "texts": texts[offset:offset + limit]
    }


@router.post("/jobs", response_model=IngestJobResponse)
async def create_ingest_job(
    job_data: IngestJobCreate,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_current_user)
):
    """Create an ingest job"""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    job = {
        "id": job_id,
        "corpus_id": job_data.corpus_id,
        "sources": [s.dict() for s in job_data.sources],
        "priority": job_data.priority,
        "validate_texts": job_data.validate_texts,
        "status": "pending",
        "progress": 0.0,
        "documents_ingested": 0,
        "documents_failed": 0,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "created_by": user.username if user else "anonymous"
    }
    
    _ingest_jobs[job_id] = job
    
    return IngestJobResponse(
        id=job_id,
        corpus_id=job_data.corpus_id,
        status="pending",
        progress=0.0,
        documents_ingested=0,
        documents_failed=0,
        created_at=job["created_at"],
        started_at=None,
        completed_at=None
    )


@router.get("/jobs", response_model=List[IngestJobResponse])
async def list_ingest_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """List ingest jobs"""
    jobs = list(_ingest_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    jobs = sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]
    
    return [
        IngestJobResponse(
            id=j["id"],
            corpus_id=j["corpus_id"],
            status=j["status"],
            progress=j["progress"],
            documents_ingested=j["documents_ingested"],
            documents_failed=j["documents_failed"],
            created_at=j["created_at"],
            started_at=j.get("started_at"),
            completed_at=j.get("completed_at")
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=IngestJobResponse)
async def get_ingest_job(
    job_id: str = Path(..., description="Job ID")
):
    """Get an ingest job by ID"""
    job = _ingest_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return IngestJobResponse(
        id=job["id"],
        corpus_id=job["corpus_id"],
        status=job["status"],
        progress=job["progress"],
        documents_ingested=job["documents_ingested"],
        documents_failed=job["documents_failed"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_ingest_job(
    job_id: str = Path(..., description="Job ID"),
    user: User = Depends(require_auth)
):
    """Cancel an ingest job"""
    job = _ingest_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    
    return {"message": "Job cancelled", "id": job_id}


@router.post("/fetch")
async def fetch_text(
    source_type: str = Body(..., description="Source type"),
    text_id: str = Body(..., description="Text ID"),
    user: Optional[User] = Depends(get_current_user)
):
    """Fetch a single text from a source"""
    return {
        "source_type": source_type,
        "text_id": text_id,
        "status": "fetched",
        "title": "Sample Text",
        "author": "Unknown",
        "language": "grc",
        "text_length": 5000,
        "fetched_at": datetime.now().isoformat()
    }


@router.get("/statistics")
async def get_ingest_statistics():
    """Get ingestion statistics"""
    total_jobs = len(_ingest_jobs)
    completed_jobs = sum(1 for j in _ingest_jobs.values() if j["status"] == "completed")
    failed_jobs = sum(1 for j in _ingest_jobs.values() if j["status"] == "failed")
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "pending_jobs": total_jobs - completed_jobs - failed_jobs,
        "total_documents_ingested": sum(
            j["documents_ingested"] for j in _ingest_jobs.values()
        ),
        "total_documents_failed": sum(
            j["documents_failed"] for j in _ingest_jobs.values()
        ),
        "sources_used": {
            "perseus": 10,
            "first1k": 5,
            "gutenberg": 3,
            "proiel": 8
        }
    }
