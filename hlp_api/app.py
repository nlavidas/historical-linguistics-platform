"""
HLP API App - FastAPI Application

This module provides the main FastAPI application for the
Historical Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

_app: Optional[FastAPI] = None


@dataclass
class APIConfig:
    """Configuration for the API"""
    title: str = "Historical Linguistics Platform API"
    
    description: str = "API for diachronic linguistics analysis"
    
    version: str = "1.0.0"
    
    debug: bool = False
    
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    cors_allow_credentials: bool = True
    
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    api_prefix: str = "/api/v1"
    
    docs_url: str = "/docs"
    
    redoc_url: str = "/redoc"
    
    openapi_url: str = "/openapi.json"
    
    enable_auth: bool = False
    
    rate_limit_enabled: bool = True
    
    rate_limit_requests: int = 100
    
    rate_limit_window: int = 60
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Historical Linguistics Platform API")
    
    app.state.initialized = True
    
    yield
    
    logger.info("Shutting down Historical Linguistics Platform API")
    
    app.state.initialized = False


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create FastAPI application"""
    global _app
    
    config = config or APIConfig()
    
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        debug=config.debug,
        docs_url=config.docs_url,
        redoc_url=config.redoc_url,
        openapi_url=config.openapi_url,
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=config.cors_allow_methods,
        allow_headers=config.cors_allow_headers,
    )
    
    app.state.config = config
    
    from hlp_api.routes_corpus import router as corpus_router
    from hlp_api.routes_annotation import router as annotation_router
    from hlp_api.routes_valency import router as valency_router
    from hlp_api.routes_diachronic import router as diachronic_router
    from hlp_api.routes_ingest import router as ingest_router
    from hlp_api.routes_qa import router as qa_router
    from hlp_api.routes_ui import router as ui_router
    
    app.include_router(corpus_router, prefix=f"{config.api_prefix}/corpus", tags=["Corpus"])
    app.include_router(annotation_router, prefix=f"{config.api_prefix}/annotation", tags=["Annotation"])
    app.include_router(valency_router, prefix=f"{config.api_prefix}/valency", tags=["Valency"])
    app.include_router(diachronic_router, prefix=f"{config.api_prefix}/diachronic", tags=["Diachronic"])
    app.include_router(ingest_router, prefix=f"{config.api_prefix}/ingest", tags=["Ingest"])
    app.include_router(qa_router, prefix=f"{config.api_prefix}/qa", tags=["QA"])
    app.include_router(ui_router, tags=["UI"])
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": config.title,
            "version": config.version,
            "status": "running",
            "docs": config.docs_url
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": config.version
        }
    
    @app.get("/info")
    async def info():
        """API information endpoint"""
        return {
            "title": config.title,
            "description": config.description,
            "version": config.version,
            "endpoints": {
                "corpus": f"{config.api_prefix}/corpus",
                "annotation": f"{config.api_prefix}/annotation",
                "valency": f"{config.api_prefix}/valency",
                "diachronic": f"{config.api_prefix}/diachronic",
                "ingest": f"{config.api_prefix}/ingest",
                "qa": f"{config.api_prefix}/qa"
            }
        }
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Internal server error",
                "status_code": 500
            }
        )
    
    _app = app
    
    return app


def get_app() -> Optional[FastAPI]:
    """Get the current FastAPI application"""
    return _app
