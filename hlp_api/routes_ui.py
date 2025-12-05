"""
HLP API UI Routes - Web Dashboard Interface

This module provides browser-based UI routes for the
Historical Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

router = APIRouter()

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/ui", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "dashboard"}
    )


@router.get("/ui/corpora", response_class=HTMLResponse)
async def ui_corpora(request: Request):
    """Corpora management page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "corpora"}
    )


@router.get("/ui/annotation", response_class=HTMLResponse)
async def ui_annotation(request: Request):
    """Annotation jobs page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "annotation"}
    )


@router.get("/ui/valency", response_class=HTMLResponse)
async def ui_valency(request: Request):
    """Valency analysis page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "valency"}
    )


@router.get("/ui/ingest", response_class=HTMLResponse)
async def ui_ingest(request: Request):
    """Text ingestion page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "ingest"}
    )


@router.get("/ui/diachronic", response_class=HTMLResponse)
async def ui_diachronic(request: Request):
    """Diachronic analysis page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "diachronic"}
    )


@router.get("/ui/qa", response_class=HTMLResponse)
async def ui_qa(request: Request):
    """Quality assurance page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "qa"}
    )


@router.get("/ui/system", response_class=HTMLResponse)
async def ui_system(request: Request):
    """System monitoring page"""
    return templates.TemplateResponse(
        "hlp_dashboard.html",
        {"request": request, "page": "system"}
    )
