"""
HLP API - FastAPI Application Package

This package provides a comprehensive REST API for the Historical
Linguistics Platform, including endpoints for corpus management,
annotation, valency analysis, and more.

Modules:
    app: FastAPI application
    routes_corpus: Corpus endpoints
    routes_annotation: Annotation endpoints
    routes_valency: Valency endpoints
    auth: Authentication layer

University of Athens - Nikolaos Lavidas
"""

from hlp_api.app import (
    create_app,
    get_app,
    APIConfig,
)

from hlp_api.routes_corpus import router as corpus_router
from hlp_api.routes_annotation import router as annotation_router
from hlp_api.routes_valency import router as valency_router
from hlp_api.auth import (
    AuthConfig,
    authenticate_user,
    create_access_token,
    get_current_user,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "create_app",
    "get_app",
    "APIConfig",
    "corpus_router",
    "annotation_router",
    "valency_router",
    "AuthConfig",
    "authenticate_user",
    "create_access_token",
    "get_current_user",
]
