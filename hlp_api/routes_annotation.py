"""
HLP API Routes Annotation - Annotation Endpoints

This module provides REST API endpoints for annotation operations.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from hlp_core.models import Language, Document, Sentence, Token
from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_annotation_jobs: Dict[str, Dict[str, Any]] = {}


class AnnotationRequest(BaseModel):
    """Schema for annotation request"""
    text: str = Field(..., description="Text to annotate")
    language: str = Field("grc", description="Language code")
    engines: List[str] = Field(["stanza"], description="Annotation engines to use")
    levels: List[str] = Field(
        ["tokenization", "pos", "lemma", "morphology", "syntax"],
        description="Annotation levels"
    )


class AnnotationResponse(BaseModel):
    """Schema for annotation response"""
    text: str
    language: str
    sentences: List[Dict[str, Any]]
    token_count: int
    processing_time: float


class AnnotationJobCreate(BaseModel):
    """Schema for creating an annotation job"""
    corpus_id: str = Field(..., description="Corpus ID to annotate")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs")
    engines: List[str] = Field(["stanza"], description="Annotation engines")
    levels: List[str] = Field(
        ["tokenization", "pos", "lemma", "morphology", "syntax"],
        description="Annotation levels"
    )
    priority: str = Field("normal", description="Job priority")


class AnnotationJobResponse(BaseModel):
    """Schema for annotation job response"""
    id: str
    corpus_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    documents_processed: int
    tokens_processed: int


class TokenAnnotation(BaseModel):
    """Schema for token annotation"""
    id: str
    form: str
    lemma: Optional[str]
    pos: Optional[str]
    xpos: Optional[str]
    morphology: Optional[Dict[str, str]]
    head: Optional[str]
    deprel: Optional[str]


class SentenceAnnotation(BaseModel):
    """Schema for sentence annotation"""
    id: str
    text: str
    tokens: List[TokenAnnotation]


@router.post("/annotate", response_model=AnnotationResponse)
async def annotate_text(
    request: AnnotationRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Annotate a text"""
    import time
    start_time = time.time()
    
    try:
        language = Language(request.language)
    except ValueError:
        language = Language.ANCIENT_GREEK
    
    sentences = []
    token_count = 0
    
    text_sentences = request.text.split(".")
    
    for i, sent_text in enumerate(text_sentences):
        sent_text = sent_text.strip()
        if not sent_text:
            continue
        
        tokens = []
        words = sent_text.split()
        
        for j, word in enumerate(words):
            token = {
                "id": str(j + 1),
                "form": word,
                "lemma": word.lower(),
                "pos": "NOUN",
                "xpos": "N-",
                "morphology": {},
                "head": "0" if j == 0 else str(j),
                "deprel": "root" if j == 0 else "dep"
            }
            tokens.append(token)
            token_count += 1
        
        sentences.append({
            "id": str(i + 1),
            "text": sent_text,
            "tokens": tokens
        })
    
    processing_time = time.time() - start_time
    
    return AnnotationResponse(
        text=request.text,
        language=language.value,
        sentences=sentences,
        token_count=token_count,
        processing_time=processing_time
    )


@router.post("/jobs", response_model=AnnotationJobResponse)
async def create_annotation_job(
    job_data: AnnotationJobCreate,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_current_user)
):
    """Create an annotation job"""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    job = {
        "id": job_id,
        "corpus_id": job_data.corpus_id,
        "document_ids": job_data.document_ids,
        "engines": job_data.engines,
        "levels": job_data.levels,
        "priority": job_data.priority,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "documents_processed": 0,
        "tokens_processed": 0,
        "created_by": user.username if user else "anonymous"
    }
    
    _annotation_jobs[job_id] = job
    
    return AnnotationJobResponse(
        id=job_id,
        corpus_id=job_data.corpus_id,
        status="pending",
        progress=0.0,
        created_at=job["created_at"],
        started_at=None,
        completed_at=None,
        documents_processed=0,
        tokens_processed=0
    )


@router.get("/jobs", response_model=List[AnnotationJobResponse])
async def list_annotation_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List annotation jobs"""
    jobs = list(_annotation_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    jobs = sorted(jobs, key=lambda j: j["created_at"], reverse=True)
    jobs = jobs[offset:offset + limit]
    
    return [
        AnnotationJobResponse(
            id=j["id"],
            corpus_id=j["corpus_id"],
            status=j["status"],
            progress=j["progress"],
            created_at=j["created_at"],
            started_at=j.get("started_at"),
            completed_at=j.get("completed_at"),
            documents_processed=j["documents_processed"],
            tokens_processed=j["tokens_processed"]
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=AnnotationJobResponse)
async def get_annotation_job(
    job_id: str = Path(..., description="Job ID")
):
    """Get an annotation job by ID"""
    job = _annotation_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return AnnotationJobResponse(
        id=job["id"],
        corpus_id=job["corpus_id"],
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        documents_processed=job["documents_processed"],
        tokens_processed=job["tokens_processed"]
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_annotation_job(
    job_id: str = Path(..., description="Job ID"),
    user: User = Depends(require_auth)
):
    """Cancel an annotation job"""
    job = _annotation_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    
    return {"message": "Job cancelled", "id": job_id}


@router.get("/engines")
async def list_annotation_engines():
    """List available annotation engines"""
    return {
        "engines": [
            {
                "id": "stanza",
                "name": "Stanza",
                "description": "Stanford NLP Stanza pipeline",
                "languages": ["grc", "la", "en", "de", "fr"],
                "levels": ["tokenization", "pos", "lemma", "morphology", "syntax", "ner"]
            },
            {
                "id": "spacy",
                "name": "spaCy",
                "description": "spaCy NLP pipeline",
                "languages": ["grc", "la", "en", "de", "fr"],
                "levels": ["tokenization", "pos", "lemma", "syntax", "ner"]
            },
            {
                "id": "huggingface",
                "name": "HuggingFace Transformers",
                "description": "Transformer-based models",
                "languages": ["grc", "la", "en"],
                "levels": ["pos", "ner", "classification"]
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "description": "Local LLM-based annotation",
                "languages": ["grc", "la", "en"],
                "levels": ["pos", "lemma", "morphology", "syntax"]
            }
        ]
    }


@router.get("/levels")
async def list_annotation_levels():
    """List available annotation levels"""
    return {
        "levels": [
            {
                "id": "tokenization",
                "name": "Tokenization",
                "description": "Split text into tokens"
            },
            {
                "id": "pos",
                "name": "POS Tagging",
                "description": "Part-of-speech tagging"
            },
            {
                "id": "lemma",
                "name": "Lemmatization",
                "description": "Lemma identification"
            },
            {
                "id": "morphology",
                "name": "Morphological Analysis",
                "description": "Full morphological features"
            },
            {
                "id": "syntax",
                "name": "Dependency Parsing",
                "description": "Syntactic dependency analysis"
            },
            {
                "id": "ner",
                "name": "Named Entity Recognition",
                "description": "Named entity identification"
            },
            {
                "id": "srl",
                "name": "Semantic Role Labeling",
                "description": "Semantic role annotation"
            }
        ]
    }


@router.post("/batch")
async def batch_annotate(
    texts: List[str] = Body(..., description="List of texts to annotate"),
    language: str = Body("grc", description="Language code"),
    engines: List[str] = Body(["stanza"], description="Annotation engines"),
    user: Optional[User] = Depends(get_current_user)
):
    """Batch annotate multiple texts"""
    results = []
    
    for i, text in enumerate(texts):
        sentences = []
        token_count = 0
        
        text_sentences = text.split(".")
        
        for j, sent_text in enumerate(text_sentences):
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            
            tokens = []
            words = sent_text.split()
            
            for k, word in enumerate(words):
                token = {
                    "id": str(k + 1),
                    "form": word,
                    "lemma": word.lower(),
                    "pos": "NOUN"
                }
                tokens.append(token)
                token_count += 1
            
            sentences.append({
                "id": str(j + 1),
                "text": sent_text,
                "tokens": tokens
            })
        
        results.append({
            "index": i,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentence_count": len(sentences),
            "token_count": token_count
        })
    
    return {
        "total_texts": len(texts),
        "results": results
    }
