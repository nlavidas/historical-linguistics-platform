"""
HLP API Routes Corpus - Corpus Management Endpoints

This module provides REST API endpoints for corpus management.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends, UploadFile, File
from pydantic import BaseModel, Field

from hlp_core.models import Language, Period, Document, Corpus, Sentence
from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_corpora: Dict[str, Corpus] = {}
_documents: Dict[str, Document] = {}


class CorpusCreate(BaseModel):
    """Schema for creating a corpus"""
    name: str = Field(..., description="Corpus name")
    description: Optional[str] = Field(None, description="Corpus description")
    language: str = Field("grc", description="Language code")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CorpusUpdate(BaseModel):
    """Schema for updating a corpus"""
    name: Optional[str] = Field(None, description="Corpus name")
    description: Optional[str] = Field(None, description="Corpus description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CorpusResponse(BaseModel):
    """Schema for corpus response"""
    id: str
    name: str
    description: Optional[str]
    language: str
    document_count: int
    created_at: str
    metadata: Dict[str, Any]


class DocumentCreate(BaseModel):
    """Schema for creating a document"""
    title: str = Field(..., description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    text: str = Field(..., description="Document text")
    language: str = Field("grc", description="Language code")
    period: Optional[str] = Field(None, description="Historical period")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: str
    title: str
    author: Optional[str]
    language: str
    text_length: int
    sentence_count: int
    token_count: int
    period: Optional[str]
    metadata: Dict[str, Any]


class SentenceResponse(BaseModel):
    """Schema for sentence response"""
    id: str
    text: str
    token_count: int


@router.get("/", response_model=List[CorpusResponse])
async def list_corpora(
    language: Optional[str] = Query(None, description="Filter by language"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List all corpora"""
    corpora = list(_corpora.values())
    
    if language:
        corpora = [c for c in corpora if c.language and c.language.value == language]
    
    corpora = corpora[offset:offset + limit]
    
    return [
        CorpusResponse(
            id=c.id,
            name=c.name,
            description=c.metadata.get("description"),
            language=c.language.value if c.language else "unknown",
            document_count=len(c.documents),
            created_at=c.metadata.get("created_at", datetime.now().isoformat()),
            metadata=c.metadata
        )
        for c in corpora
    ]


@router.post("/", response_model=CorpusResponse)
async def create_corpus(
    corpus_data: CorpusCreate,
    user: Optional[User] = Depends(get_current_user)
):
    """Create a new corpus"""
    import uuid
    
    corpus_id = str(uuid.uuid4())
    
    try:
        language = Language(corpus_data.language)
    except ValueError:
        language = Language.ANCIENT_GREEK
    
    corpus = Corpus(
        id=corpus_id,
        name=corpus_data.name,
        language=language,
        documents=[],
        metadata={
            "description": corpus_data.description,
            "created_at": datetime.now().isoformat(),
            "created_by": user.username if user else "anonymous",
            **(corpus_data.metadata or {})
        }
    )
    
    _corpora[corpus_id] = corpus
    
    return CorpusResponse(
        id=corpus.id,
        name=corpus.name,
        description=corpus_data.description,
        language=language.value,
        document_count=0,
        created_at=corpus.metadata["created_at"],
        metadata=corpus.metadata
    )


@router.get("/{corpus_id}", response_model=CorpusResponse)
async def get_corpus(
    corpus_id: str = Path(..., description="Corpus ID")
):
    """Get a corpus by ID"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    return CorpusResponse(
        id=corpus.id,
        name=corpus.name,
        description=corpus.metadata.get("description"),
        language=corpus.language.value if corpus.language else "unknown",
        document_count=len(corpus.documents),
        created_at=corpus.metadata.get("created_at", ""),
        metadata=corpus.metadata
    )


@router.put("/{corpus_id}", response_model=CorpusResponse)
async def update_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    corpus_data: CorpusUpdate = Body(...),
    user: User = Depends(require_auth)
):
    """Update a corpus"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    if corpus_data.name:
        corpus.name = corpus_data.name
    
    if corpus_data.description:
        corpus.metadata["description"] = corpus_data.description
    
    if corpus_data.metadata:
        corpus.metadata.update(corpus_data.metadata)
    
    corpus.metadata["updated_at"] = datetime.now().isoformat()
    corpus.metadata["updated_by"] = user.username
    
    return CorpusResponse(
        id=corpus.id,
        name=corpus.name,
        description=corpus.metadata.get("description"),
        language=corpus.language.value if corpus.language else "unknown",
        document_count=len(corpus.documents),
        created_at=corpus.metadata.get("created_at", ""),
        metadata=corpus.metadata
    )


@router.delete("/{corpus_id}")
async def delete_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    user: User = Depends(require_auth)
):
    """Delete a corpus"""
    if corpus_id not in _corpora:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    del _corpora[corpus_id]
    
    return {"message": "Corpus deleted", "id": corpus_id}


@router.get("/{corpus_id}/documents", response_model=List[DocumentResponse])
async def list_corpus_documents(
    corpus_id: str = Path(..., description="Corpus ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List documents in a corpus"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    documents = corpus.documents[offset:offset + limit]
    
    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            author=doc.author,
            language=doc.language.value if doc.language else "unknown",
            text_length=len(doc.text) if doc.text else 0,
            sentence_count=len(doc.sentences),
            token_count=sum(len(s.tokens) for s in doc.sentences),
            period=doc.metadata.get("period"),
            metadata=doc.metadata
        )
        for doc in documents
    ]


@router.post("/{corpus_id}/documents", response_model=DocumentResponse)
async def add_document_to_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_data: DocumentCreate = Body(...),
    user: Optional[User] = Depends(get_current_user)
):
    """Add a document to a corpus"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    import uuid
    
    doc_id = str(uuid.uuid4())
    
    try:
        language = Language(document_data.language)
    except ValueError:
        language = corpus.language or Language.ANCIENT_GREEK
    
    document = Document(
        id=doc_id,
        title=document_data.title,
        author=document_data.author,
        language=language,
        text=document_data.text,
        sentences=[],
        metadata={
            "period": document_data.period,
            "created_at": datetime.now().isoformat(),
            "created_by": user.username if user else "anonymous",
            **(document_data.metadata or {})
        }
    )
    
    corpus.documents.append(document)
    _documents[doc_id] = document
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        author=document.author,
        language=language.value,
        text_length=len(document.text),
        sentence_count=0,
        token_count=0,
        period=document_data.period,
        metadata=document.metadata
    )


@router.get("/{corpus_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_id: str = Path(..., description="Document ID")
):
    """Get a document by ID"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    document = None
    for doc in corpus.documents:
        if doc.id == document_id:
            document = doc
            break
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document.id,
        title=document.title,
        author=document.author,
        language=document.language.value if document.language else "unknown",
        text_length=len(document.text) if document.text else 0,
        sentence_count=len(document.sentences),
        token_count=sum(len(s.tokens) for s in document.sentences),
        period=document.metadata.get("period"),
        metadata=document.metadata
    )


@router.get("/{corpus_id}/documents/{document_id}/sentences", response_model=List[SentenceResponse])
async def get_document_sentences(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_id: str = Path(..., description="Document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Get sentences from a document"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    document = None
    for doc in corpus.documents:
        if doc.id == document_id:
            document = doc
            break
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    sentences = document.sentences[offset:offset + limit]
    
    return [
        SentenceResponse(
            id=sent.id,
            text=sent.text,
            token_count=len(sent.tokens)
        )
        for sent in sentences
    ]


@router.get("/{corpus_id}/statistics")
async def get_corpus_statistics(
    corpus_id: str = Path(..., description="Corpus ID")
):
    """Get corpus statistics"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    total_documents = len(corpus.documents)
    total_sentences = sum(len(doc.sentences) for doc in corpus.documents)
    total_tokens = sum(
        sum(len(s.tokens) for s in doc.sentences)
        for doc in corpus.documents
    )
    total_characters = sum(
        len(doc.text) if doc.text else 0
        for doc in corpus.documents
    )
    
    return {
        "corpus_id": corpus_id,
        "total_documents": total_documents,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "average_tokens_per_sentence": total_tokens / total_sentences if total_sentences > 0 else 0,
        "average_sentences_per_document": total_sentences / total_documents if total_documents > 0 else 0
    }


@router.post("/{corpus_id}/upload")
async def upload_document(
    corpus_id: str = Path(..., description="Corpus ID"),
    file: UploadFile = File(..., description="Document file"),
    user: Optional[User] = Depends(get_current_user)
):
    """Upload a document file"""
    corpus = _corpora.get(corpus_id)
    
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    content = await file.read()
    text = content.decode("utf-8")
    
    import uuid
    doc_id = str(uuid.uuid4())
    
    document = Document(
        id=doc_id,
        title=file.filename or "Uploaded Document",
        language=corpus.language or Language.ANCIENT_GREEK,
        text=text,
        sentences=[],
        metadata={
            "filename": file.filename,
            "content_type": file.content_type,
            "created_at": datetime.now().isoformat(),
            "created_by": user.username if user else "anonymous"
        }
    )
    
    corpus.documents.append(document)
    _documents[doc_id] = document
    
    return {
        "message": "Document uploaded",
        "document_id": doc_id,
        "filename": file.filename,
        "text_length": len(text)
    }
