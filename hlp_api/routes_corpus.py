"""
HLP API Routes Corpus - Corpus Management Endpoints

This module provides REST API endpoints for corpus management.
Uses DatabaseManager for persistent storage.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends, UploadFile, File
from pydantic import BaseModel, Field

from hlp_core.models import Language, Period, Document, Corpus, Sentence
from hlp_core.db import DatabaseManager, DatabaseConfig, DatabaseType, get_default_db_manager
from hlp_api.auth import get_current_user, require_auth, User

logger = logging.getLogger(__name__)

router = APIRouter()

_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = get_default_db_manager()
    return _db_manager


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
    """List all corpora from database"""
    db = get_db()
    
    sql = "SELECT id, name, description, language, metadata, created_at FROM corpora"
    params = []
    
    if language:
        sql += " WHERE language = ?"
        params.append(language)
    
    sql += f" LIMIT {limit} OFFSET {offset}"
    
    rows = db.fetch_all(sql, params if params else None)
    
    results = []
    for row in rows:
        row_dict = dict(row) if hasattr(row, 'keys') else {
            'id': row[0], 'name': row[1], 'description': row[2],
            'language': row[3], 'metadata': row[4], 'created_at': row[5]
        }
        
        doc_count = db.count("documents", {"corpus_id": row_dict['id']})
        
        metadata = {}
        if row_dict.get('metadata'):
            try:
                metadata = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
            except:
                metadata = {}
        
        results.append(CorpusResponse(
            id=row_dict['id'],
            name=row_dict['name'],
            description=row_dict.get('description'),
            language=row_dict.get('language', 'unknown'),
            document_count=doc_count,
            created_at=row_dict.get('created_at', datetime.now().isoformat()),
            metadata=metadata
        ))
    
    return results


@router.post("/", response_model=CorpusResponse)
async def create_corpus(
    corpus_data: CorpusCreate,
    user: Optional[User] = Depends(get_current_user)
):
    """Create a new corpus in database"""
    db = get_db()
    
    corpus_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    metadata = {
        "created_by": user.username if user else "anonymous",
        **(corpus_data.metadata or {})
    }
    
    db.insert("corpora", {
        "id": corpus_id,
        "name": corpus_data.name,
        "description": corpus_data.description,
        "language": corpus_data.language,
        "metadata": json.dumps(metadata),
        "created_at": created_at,
        "updated_at": created_at
    })
    
    return CorpusResponse(
        id=corpus_id,
        name=corpus_data.name,
        description=corpus_data.description,
        language=corpus_data.language,
        document_count=0,
        created_at=created_at,
        metadata=metadata
    )


@router.get("/{corpus_id}", response_model=CorpusResponse)
async def get_corpus(
    corpus_id: str = Path(..., description="Corpus ID")
):
    """Get a corpus by ID from database"""
    db = get_db()
    
    row = db.fetch_one(
        "SELECT id, name, description, language, metadata, created_at FROM corpora WHERE id = ?",
        [corpus_id]
    )
    
    if not row:
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    row_dict = dict(row) if hasattr(row, 'keys') else {
        'id': row[0], 'name': row[1], 'description': row[2],
        'language': row[3], 'metadata': row[4], 'created_at': row[5]
    }
    
    doc_count = db.count("documents", {"corpus_id": corpus_id})
    
    metadata = {}
    if row_dict.get('metadata'):
        try:
            metadata = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
        except:
            metadata = {}
    
    return CorpusResponse(
        id=row_dict['id'],
        name=row_dict['name'],
        description=row_dict.get('description'),
        language=row_dict.get('language', 'unknown'),
        document_count=doc_count,
        created_at=row_dict.get('created_at', ''),
        metadata=metadata
    )


@router.put("/{corpus_id}", response_model=CorpusResponse)
async def update_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    corpus_data: CorpusUpdate = Body(...),
    user: User = Depends(require_auth)
):
    """Update a corpus in database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    update_data = {"updated_at": datetime.now().isoformat()}
    
    if corpus_data.name:
        update_data["name"] = corpus_data.name
    
    if corpus_data.description:
        update_data["description"] = corpus_data.description
    
    if corpus_data.metadata:
        row = db.fetch_one("SELECT metadata FROM corpora WHERE id = ?", [corpus_id])
        existing_metadata = {}
        if row and row[0]:
            try:
                existing_metadata = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            except:
                pass
        existing_metadata.update(corpus_data.metadata)
        existing_metadata["updated_by"] = user.username
        update_data["metadata"] = json.dumps(existing_metadata)
    
    db.update("corpora", update_data, {"id": corpus_id})
    
    return await get_corpus(corpus_id)


@router.delete("/{corpus_id}")
async def delete_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    user: User = Depends(require_auth)
):
    """Delete a corpus from database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    db.delete("documents", {"corpus_id": corpus_id})
    db.delete("corpora", {"id": corpus_id})
    
    return {"message": "Corpus deleted", "id": corpus_id}


@router.get("/{corpus_id}/documents", response_model=List[DocumentResponse])
async def list_corpus_documents(
    corpus_id: str = Path(..., description="Corpus ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List documents in a corpus from database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    rows = db.fetch_all(
        f"SELECT id, title, author, language, period, sentence_count, token_count, metadata FROM documents WHERE corpus_id = ? LIMIT {limit} OFFSET {offset}",
        [corpus_id]
    )
    
    results = []
    for row in rows:
        row_dict = dict(row) if hasattr(row, 'keys') else {
            'id': row[0], 'title': row[1], 'author': row[2], 'language': row[3],
            'period': row[4], 'sentence_count': row[5], 'token_count': row[6], 'metadata': row[7]
        }
        
        metadata = {}
        if row_dict.get('metadata'):
            try:
                metadata = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
            except:
                metadata = {}
        
        results.append(DocumentResponse(
            id=row_dict['id'],
            title=row_dict['title'],
            author=row_dict.get('author'),
            language=row_dict.get('language', 'unknown'),
            text_length=metadata.get('text_length', 0),
            sentence_count=row_dict.get('sentence_count', 0) or 0,
            token_count=row_dict.get('token_count', 0) or 0,
            period=row_dict.get('period'),
            metadata=metadata
        ))
    
    return results


@router.post("/{corpus_id}/documents", response_model=DocumentResponse)
async def add_document_to_corpus(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_data: DocumentCreate = Body(...),
    user: Optional[User] = Depends(get_current_user)
):
    """Add a document to a corpus in database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    doc_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    metadata = {
        "text_length": len(document_data.text),
        "created_by": user.username if user else "anonymous",
        **(document_data.metadata or {})
    }
    
    db.insert("documents", {
        "id": doc_id,
        "corpus_id": corpus_id,
        "title": document_data.title,
        "author": document_data.author,
        "language": document_data.language,
        "period": document_data.period,
        "sentence_count": 0,
        "token_count": 0,
        "metadata": json.dumps(metadata),
        "created_at": created_at,
        "updated_at": created_at
    })
    
    return DocumentResponse(
        id=doc_id,
        title=document_data.title,
        author=document_data.author,
        language=document_data.language,
        text_length=len(document_data.text),
        sentence_count=0,
        token_count=0,
        period=document_data.period,
        metadata=metadata
    )


@router.get("/{corpus_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_id: str = Path(..., description="Document ID")
):
    """Get a document by ID from database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    row = db.fetch_one(
        "SELECT id, title, author, language, period, sentence_count, token_count, metadata FROM documents WHERE id = ? AND corpus_id = ?",
        [document_id, corpus_id]
    )
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    
    row_dict = dict(row) if hasattr(row, 'keys') else {
        'id': row[0], 'title': row[1], 'author': row[2], 'language': row[3],
        'period': row[4], 'sentence_count': row[5], 'token_count': row[6], 'metadata': row[7]
    }
    
    metadata = {}
    if row_dict.get('metadata'):
        try:
            metadata = json.loads(row_dict['metadata']) if isinstance(row_dict['metadata'], str) else row_dict['metadata']
        except:
            metadata = {}
    
    return DocumentResponse(
        id=row_dict['id'],
        title=row_dict['title'],
        author=row_dict.get('author'),
        language=row_dict.get('language', 'unknown'),
        text_length=metadata.get('text_length', 0),
        sentence_count=row_dict.get('sentence_count', 0) or 0,
        token_count=row_dict.get('token_count', 0) or 0,
        period=row_dict.get('period'),
        metadata=metadata
    )


@router.get("/{corpus_id}/documents/{document_id}/sentences", response_model=List[SentenceResponse])
async def get_document_sentences(
    corpus_id: str = Path(..., description="Corpus ID"),
    document_id: str = Path(..., description="Document ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Get sentences from a document from database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    if not db.exists("documents", {"id": document_id, "corpus_id": corpus_id}):
        raise HTTPException(status_code=404, detail="Document not found")
    
    rows = db.fetch_all(
        f"SELECT id, text FROM sentences WHERE document_id = ? LIMIT {limit} OFFSET {offset}",
        [document_id]
    )
    
    results = []
    for row in rows:
        row_dict = dict(row) if hasattr(row, 'keys') else {'id': row[0], 'text': row[1]}
        
        token_count = db.count("tokens", {"sentence_id": row_dict['id']})
        
        results.append(SentenceResponse(
            id=row_dict['id'],
            text=row_dict.get('text', ''),
            token_count=token_count
        ))
    
    return results


@router.get("/{corpus_id}/statistics")
async def get_corpus_statistics(
    corpus_id: str = Path(..., description="Corpus ID")
):
    """Get corpus statistics from database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    total_documents = db.count("documents", {"corpus_id": corpus_id})
    
    sentence_row = db.fetch_one(
        "SELECT COUNT(*) FROM sentences s JOIN documents d ON s.document_id = d.id WHERE d.corpus_id = ?",
        [corpus_id]
    )
    total_sentences = sentence_row[0] if sentence_row else 0
    
    token_row = db.fetch_one(
        "SELECT COUNT(*) FROM tokens t JOIN sentences s ON t.sentence_id = s.id JOIN documents d ON s.document_id = d.id WHERE d.corpus_id = ?",
        [corpus_id]
    )
    total_tokens = token_row[0] if token_row else 0
    
    return {
        "corpus_id": corpus_id,
        "total_documents": total_documents,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "average_tokens_per_sentence": total_tokens / total_sentences if total_sentences > 0 else 0,
        "average_sentences_per_document": total_sentences / total_documents if total_documents > 0 else 0
    }


@router.post("/{corpus_id}/upload")
async def upload_document(
    corpus_id: str = Path(..., description="Corpus ID"),
    file: UploadFile = File(..., description="Document file"),
    user: Optional[User] = Depends(get_current_user)
):
    """Upload a document file to database"""
    db = get_db()
    
    if not db.exists("corpora", {"id": corpus_id}):
        raise HTTPException(status_code=404, detail="Corpus not found")
    
    content = await file.read()
    text = content.decode("utf-8")
    
    doc_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    corpus_row = db.fetch_one("SELECT language FROM corpora WHERE id = ?", [corpus_id])
    language = corpus_row[0] if corpus_row else "grc"
    
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "text_length": len(text),
        "created_by": user.username if user else "anonymous"
    }
    
    db.insert("documents", {
        "id": doc_id,
        "corpus_id": corpus_id,
        "title": file.filename or "Uploaded Document",
        "language": language,
        "sentence_count": 0,
        "token_count": 0,
        "metadata": json.dumps(metadata),
        "created_at": created_at,
        "updated_at": created_at
    })
    
    return {
        "message": "Document uploaded",
        "document_id": doc_id,
        "filename": file.filename,
        "text_length": len(text)
    }
