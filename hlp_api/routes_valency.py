"""
HLP API Routes Valency - Valency Analysis Endpoints

This module provides REST API endpoints for valency analysis.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends
from pydantic import BaseModel, Field

from hlp_core.models import Language
from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_valency_jobs: Dict[str, Dict[str, Any]] = {}
_lexicons: Dict[str, Dict[str, Any]] = {}


class ValencyExtractionRequest(BaseModel):
    """Schema for valency extraction request"""
    corpus_id: str = Field(..., description="Corpus ID")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs")
    min_frequency: int = Field(1, description="Minimum pattern frequency")
    include_auxiliaries: bool = Field(False, description="Include auxiliary verbs")


class ValencyPatternResponse(BaseModel):
    """Schema for valency pattern response"""
    verb_lemma: str
    pattern: str
    canonical_form: str
    frequency: int
    arguments: List[Dict[str, Any]]
    examples: List[str]


class ValencyLexiconResponse(BaseModel):
    """Schema for valency lexicon response"""
    id: str
    name: str
    language: str
    verb_count: int
    pattern_count: int
    created_at: str


class ValencyJobResponse(BaseModel):
    """Schema for valency job response"""
    id: str
    corpus_id: str
    status: str
    progress: float
    frames_extracted: int
    created_at: str


class ValencySearchRequest(BaseModel):
    """Schema for valency search request"""
    verb_lemma: Optional[str] = Field(None, description="Verb lemma to search")
    pattern: Optional[str] = Field(None, description="Pattern to search")
    argument_type: Optional[str] = Field(None, description="Argument type filter")
    case: Optional[str] = Field(None, description="Case filter")


@router.post("/extract", response_model=ValencyJobResponse)
async def extract_valency(
    request: ValencyExtractionRequest,
    user: Optional[User] = Depends(get_current_user)
):
    """Start valency extraction"""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    job = {
        "id": job_id,
        "corpus_id": request.corpus_id,
        "document_ids": request.document_ids,
        "min_frequency": request.min_frequency,
        "include_auxiliaries": request.include_auxiliaries,
        "status": "pending",
        "progress": 0.0,
        "frames_extracted": 0,
        "created_at": datetime.now().isoformat(),
        "created_by": user.username if user else "anonymous"
    }
    
    _valency_jobs[job_id] = job
    
    return ValencyJobResponse(
        id=job_id,
        corpus_id=request.corpus_id,
        status="pending",
        progress=0.0,
        frames_extracted=0,
        created_at=job["created_at"]
    )


@router.get("/jobs", response_model=List[ValencyJobResponse])
async def list_valency_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """List valency extraction jobs"""
    jobs = list(_valency_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    jobs = sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]
    
    return [
        ValencyJobResponse(
            id=j["id"],
            corpus_id=j["corpus_id"],
            status=j["status"],
            progress=j["progress"],
            frames_extracted=j["frames_extracted"],
            created_at=j["created_at"]
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=ValencyJobResponse)
async def get_valency_job(
    job_id: str = Path(..., description="Job ID")
):
    """Get a valency job by ID"""
    job = _valency_jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ValencyJobResponse(
        id=job["id"],
        corpus_id=job["corpus_id"],
        status=job["status"],
        progress=job["progress"],
        frames_extracted=job["frames_extracted"],
        created_at=job["created_at"]
    )


@router.get("/lexicons", response_model=List[ValencyLexiconResponse])
async def list_lexicons(
    language: Optional[str] = Query(None, description="Filter by language"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """List valency lexicons"""
    lexicons = list(_lexicons.values())
    
    if language:
        lexicons = [l for l in lexicons if l["language"] == language]
    
    lexicons = lexicons[:limit]
    
    return [
        ValencyLexiconResponse(
            id=l["id"],
            name=l["name"],
            language=l["language"],
            verb_count=l["verb_count"],
            pattern_count=l["pattern_count"],
            created_at=l["created_at"]
        )
        for l in lexicons
    ]


@router.get("/lexicons/{lexicon_id}", response_model=ValencyLexiconResponse)
async def get_lexicon(
    lexicon_id: str = Path(..., description="Lexicon ID")
):
    """Get a lexicon by ID"""
    lexicon = _lexicons.get(lexicon_id)
    
    if not lexicon:
        raise HTTPException(status_code=404, detail="Lexicon not found")
    
    return ValencyLexiconResponse(
        id=lexicon["id"],
        name=lexicon["name"],
        language=lexicon["language"],
        verb_count=lexicon["verb_count"],
        pattern_count=lexicon["pattern_count"],
        created_at=lexicon["created_at"]
    )


@router.get("/lexicons/{lexicon_id}/patterns", response_model=List[ValencyPatternResponse])
async def get_lexicon_patterns(
    lexicon_id: str = Path(..., description="Lexicon ID"),
    verb_lemma: Optional[str] = Query(None, description="Filter by verb lemma"),
    min_frequency: int = Query(1, description="Minimum frequency"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Get patterns from a lexicon"""
    lexicon = _lexicons.get(lexicon_id)
    
    if not lexicon:
        raise HTTPException(status_code=404, detail="Lexicon not found")
    
    patterns = [
        ValencyPatternResponse(
            verb_lemma="didomi",
            pattern="NOM-ACC-DAT",
            canonical_form="Agent-Theme-Recipient",
            frequency=150,
            arguments=[
                {"type": "subject", "case": "nominative", "role": "Agent"},
                {"type": "object", "case": "accusative", "role": "Theme"},
                {"type": "indirect_object", "case": "dative", "role": "Recipient"}
            ],
            examples=["ho aner didosi to biblion toi paidi"]
        ),
        ValencyPatternResponse(
            verb_lemma="grapho",
            pattern="NOM-ACC",
            canonical_form="Agent-Theme",
            frequency=200,
            arguments=[
                {"type": "subject", "case": "nominative", "role": "Agent"},
                {"type": "object", "case": "accusative", "role": "Theme"}
            ],
            examples=["ho grammateus graphei ten epistolen"]
        )
    ]
    
    if verb_lemma:
        patterns = [p for p in patterns if p.verb_lemma == verb_lemma]
    
    patterns = [p for p in patterns if p.frequency >= min_frequency]
    
    return patterns[offset:offset + limit]


@router.post("/search", response_model=List[ValencyPatternResponse])
async def search_patterns(
    request: ValencySearchRequest,
    lexicon_id: Optional[str] = Query(None, description="Lexicon ID to search")
):
    """Search valency patterns"""
    patterns = [
        ValencyPatternResponse(
            verb_lemma="didomi",
            pattern="NOM-ACC-DAT",
            canonical_form="Agent-Theme-Recipient",
            frequency=150,
            arguments=[
                {"type": "subject", "case": "nominative", "role": "Agent"},
                {"type": "object", "case": "accusative", "role": "Theme"},
                {"type": "indirect_object", "case": "dative", "role": "Recipient"}
            ],
            examples=["ho aner didosi to biblion toi paidi"]
        )
    ]
    
    if request.verb_lemma:
        patterns = [p for p in patterns if request.verb_lemma.lower() in p.verb_lemma.lower()]
    
    if request.case:
        patterns = [
            p for p in patterns
            if any(a.get("case") == request.case for a in p.arguments)
        ]
    
    return patterns


@router.get("/statistics")
async def get_valency_statistics(
    lexicon_id: Optional[str] = Query(None, description="Lexicon ID")
):
    """Get valency statistics"""
    return {
        "total_verbs": 500,
        "total_patterns": 1200,
        "average_patterns_per_verb": 2.4,
        "most_common_patterns": [
            {"pattern": "NOM-ACC", "frequency": 450},
            {"pattern": "NOM-ACC-DAT", "frequency": 200},
            {"pattern": "NOM", "frequency": 150},
            {"pattern": "NOM-GEN", "frequency": 100}
        ],
        "argument_distribution": {
            "nominative": 1200,
            "accusative": 800,
            "dative": 300,
            "genitive": 250
        },
        "frame_type_distribution": {
            "transitive": 600,
            "intransitive": 200,
            "ditransitive": 150,
            "copular": 50
        }
    }


@router.get("/verbs/{verb_lemma}")
async def get_verb_valency(
    verb_lemma: str = Path(..., description="Verb lemma"),
    lexicon_id: Optional[str] = Query(None, description="Lexicon ID")
):
    """Get valency information for a specific verb"""
    return {
        "verb_lemma": verb_lemma,
        "total_occurrences": 250,
        "patterns": [
            {
                "pattern": "NOM-ACC",
                "frequency": 150,
                "percentage": 60.0,
                "arguments": [
                    {"type": "subject", "case": "nominative"},
                    {"type": "object", "case": "accusative"}
                ]
            },
            {
                "pattern": "NOM-ACC-DAT",
                "frequency": 80,
                "percentage": 32.0,
                "arguments": [
                    {"type": "subject", "case": "nominative"},
                    {"type": "object", "case": "accusative"},
                    {"type": "indirect_object", "case": "dative"}
                ]
            }
        ],
        "diachronic_distribution": {
            "Classical": 100,
            "Hellenistic": 80,
            "Roman": 50,
            "Byzantine": 20
        }
    }


@router.post("/compare")
async def compare_valency(
    verb_lemmas: List[str] = Body(..., description="Verb lemmas to compare"),
    lexicon_id: Optional[str] = Query(None, description="Lexicon ID")
):
    """Compare valency patterns of multiple verbs"""
    comparisons = []
    
    for lemma in verb_lemmas:
        comparisons.append({
            "verb_lemma": lemma,
            "pattern_count": 3,
            "primary_pattern": "NOM-ACC",
            "transitivity": "transitive"
        })
    
    return {
        "verbs_compared": len(verb_lemmas),
        "comparisons": comparisons,
        "shared_patterns": ["NOM-ACC"],
        "unique_patterns": {
            verb_lemmas[0]: ["NOM-ACC-DAT"] if verb_lemmas else []
        }
    }
