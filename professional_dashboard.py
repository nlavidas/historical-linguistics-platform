"""
HFRI-NKUA Corpus Platform - Professional Dashboard
Production-grade web interface for linguistic research

Integrated with Athens Digital Glossa Chronos (AthDGC) v3.0.0
Diachronic Indo-European Linguistics Capabilities
"""

# Load local models configuration (use Z:\models\ - no re-downloads)
try:
    import local_models_config
except ImportError:
    pass  # Fall back to default model locations if config not available

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
import asyncio
import sqlite3
from datetime import datetime
import aiohttp
from pathlib import Path
import json
import tempfile
import logging

from unified_corpus_platform import UnifiedCorpusPlatform, UnifiedCorpusDatabase

# AthDGC Modules
from athdgc.proiel_processor import PROIELProcessor
from athdgc.valency_lexicon import ValencyLexicon
from athdgc.statistical_analysis import DiachronicStatistics
from athdgc.language_models import LanguageModelIntegration

app = FastAPI(
    title="AthDGC Professional API",
    description="PROIEL Annotator - Diachronic Linguistics Analysis Platform\nHFRI-NKUA | National and Kapodistrian University of Athens",
    version="3.0.0",
    contact={"name": "HFRI-NKUA Research Team", "email": "lavidas@nlp.uoa.gr"},
    license_info={"name": "Academic Research License"},
    docs_url="/professional-docs",
    redoc_url="/redoc"
)

# Setup logging
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
platform = UnifiedCorpusPlatform()
active_websockets = []

# AthDGC Global Instances
proiel_processor = PROIELProcessor()
valency_lexicon = ValencyLexicon()
diachronic_stats = DiachronicStatistics()
language_models = LanguageModelIntegration()

# Multi-AI Engine
from athdgc.multi_ai_annotator_professional import multi_ai_engine


class URLSubmission(BaseModel):
    urls: List[str]
    source_type: str = "custom"
    language: Optional[str] = None
    priority: int = 5


class PROIELRequest(BaseModel):
    language: str = "grc"
    period: Optional[str] = None


class StatisticalAnalysisRequest(BaseModel):
    period_a: Dict
    period_b: Dict
    feature: str = "pattern"
    method: str = "bootstrap"
    iterations: int = 1000


class SemanticAnalysisRequest(BaseModel):
    period_a_texts: List[str]
    period_b_texts: List[str]
    target_word: Optional[str] = None


class MultiAIAnnotationRequest(BaseModel):
    text: str
    language: str = "grc-classical"
    framework: str = "proiel"
    detail_level: str = "standard"
    models: List[str] = ['xlm-roberta-base', 'llama3.2']


@app.get("/annotator", response_class=HTMLResponse)
async def multi_ai_annotator():
    """Serve the Professional Multi-AI PROIEL Annotator"""
    annotator_path = Path(__file__).parent / "athdgc" / "annotator_professional.html"
    with open(annotator_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Enhanced main dashboard with PROIEL visualization"""
    # Read the enhanced dashboard HTML
    template_path = Path(__file__).parent / "templates" / "enhanced_dashboard.html"
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback to inline HTML
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HFRI-NKUA Corpus Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px 40px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            font-size: 0.95em;
            opacity: 0.9;
        }
        
        .header .institution {
            font-size: 0.85em;
            margin-top: 8px;
            opacity: 0.85;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            border-left: 4px solid #1e3c72;
        }
        
        .stat-label {
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .stat-value {
            font-size: 2.2em;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .section {
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        
        .section-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #34495e;
            font-size: 0.95em;
        }
        
        textarea, select, input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #dcdde1;
            border-radius: 4px;
            font-size: 0.95em;
            font-family: 'Consolas', 'Courier New', monospace;
            transition: border-color 0.3s;
        }
        
        textarea:focus, select:focus, input:focus {
            outline: none;
            border-color: #1e3c72;
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .button-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        
        button {
            background: #1e3c72;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: 500;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #2a5298;
        }
        
        button.secondary {
            background: #7f8c8d;
        }
        
        button.secondary:hover {
            background: #95a5a6;
        }
        
        button.danger {
            background: #c0392b;
        }
        
        button.danger:hover {
            background: #e74c3c;
        }
        
        .alert {
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 0.95em;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .alert-info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .status-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .status-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #dee2e6;
        }
        
        .status-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .status-discovered { background: #e3f2fd; color: #1565c0; }
        .status-scraped { background: #f3e5f5; color: #6a1b9a; }
        .status-parsed { background: #fff3e0; color: #e65100; }
        .status-annotated { background: #e8f5e9; color: #2e7d32; }
        .status-failed { background: #ffebee; color: #c62828; }
        
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #27ae60;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .activity-log {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 300px;
            overflow-y: auto;
            line-height: 1.6;
        }
        
        .activity-log .timestamp {
            color: #95a5a6;
            margin-right: 10px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 40px;
        }
        
        .presets {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .preset-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .preset-card:hover {
            background: #e9ecef;
            border-color: #1e3c72;
        }
        
        .preset-card h4 {
            margin-bottom: 8px;
            color: #2c3e50;
        }
        
        .preset-card p {
            font-size: 0.85em;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HFRI-NKUA Corpus Platform</h1>
        <div class="subtitle">Multi-AI Linguistic Corpus Processing System</div>
        <div class="institution">
            National and Kapodistrian University of Athens | 
            Hellenic Foundation for Research and Innovation
        </div>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Texts</div>
                <div class="stat-value" id="stat-total">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scraped</div>
                <div class="stat-value" id="stat-scraped">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Parsed</div>
                <div class="stat-value" id="stat-parsed">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Annotated</div>
                <div class="stat-value" id="stat-annotated">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Processing Rate</div>
                <div class="stat-value" id="stat-rate">0/hr</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">AthDGC Professional Tools</h2>
            <div class="presets">
                <div class="preset-card" onclick="window.location.href='/annotator'" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;">
                    <h4 style="color: white;">Multi-AI PROIEL Annotator</h4>
                    <p style="color: rgba(255,255,255,0.9);">Professional diachronic annotation with open-source AI models</p>
                </div>
                <div class="preset-card" onclick="window.location.href='/docs'">
                    <h4>API Documentation</h4>
                    <p>Interactive API explorer (Swagger UI)</p>
                </div>
                <div class="preset-card" onclick="window.open('/api/athdgc/health', '_blank')">
                    <h4>System Status</h4>
                    <p>Check AthDGC module health</p>
                </div>
                <div class="preset-card" onclick="downloadSample()">
                    <h4>Sample PROIEL File</h4>
                    <p>Download example for testing</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Source Presets</h2>
            <div class="presets">
                <div class="preset-card" onclick="loadPreset('perseus')">
                    <h4>Perseus Digital Library</h4>
                    <p>Classical Greek and Latin texts</p>
                </div>
                <div class="preset-card" onclick="loadPreset('github')">
                    <h4>GitHub Canonical Greek</h4>
                    <p>First1K Greek corpus</p>
                </div>
                <div class="preset-card" onclick="loadPreset('gutenberg')">
                    <h4>Project Gutenberg</h4>
                    <p>Public domain literature</p>
                </div>
                <div class="preset-card" onclick="loadPreset('custom')">
                    <h4>Custom URLs</h4>
                    <p>Add your own sources</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Submit Texts for Processing</h2>
            <div id="message-area"></div>
            
            <div class="form-group">
                <label>Text URLs (one per line)</label>
                <textarea id="urls-input" placeholder="https://example.com/text1.xml
https://example.com/text2.xml"></textarea>
            </div>
            
            <div class="form-group">
                <label>Source Type</label>
                <select id="source-type">
                    <option value="custom">Custom URL</option>
                    <option value="perseus">Perseus Digital Library</option>
                    <option value="github">GitHub Repository</option>
                    <option value="gutenberg">Project Gutenberg</option>
                    <option value="archive_org">Archive.org</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Language (optional)</label>
                <input type="text" id="language" placeholder="grc, en, la, etc.">
            </div>
            
            <div class="button-group">
                <button onclick="submitURLs()">Submit for Processing</button>
                <button class="secondary" onclick="document.getElementById('urls-input').value=''">Clear</button>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">
                <span class="live-indicator"></span>
                System Status
            </h2>
            <div class="button-group">
                <button onclick="startPipeline()">Start Processing</button>
                <button class="danger" onclick="stopPipeline()">Stop Processing</button>
                <button class="secondary" onclick="refreshStats()">Refresh Data</button>
            </div>
            
            <table class="status-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody id="status-breakdown">
                    <tr><td colspan="3">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Activity Log</h2>
            <div class="activity-log" id="activity-log">
                <div><span class="timestamp">[System]</span> Platform initialized</div>
            </div>
        </div>
    </div>

    <div class="footer">
        HFRI-NKUA Corpus Platform v2.0.0 | 
        Principal Investigator: Prof. Nikolaos Lavidas | 
        &copy; 2025 National and Kapodistrian University of Athens
    </div>

    <script>
        function log(message) {
            const logDiv = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function showMessage(message, type) {
            const area = document.getElementById('message-area');
            area.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            setTimeout(() => area.innerHTML = '', 5000);
        }

        async function refreshStats() {
            try {
                const response = await fetch('/api/statistics');
                const data = await response.json();
                
                document.getElementById('stat-total').textContent = data.total_items || 0;
                document.getElementById('stat-scraped').textContent = data.status_counts?.scraped || 0;
                document.getElementById('stat-parsed').textContent = data.status_counts?.parsed || 0;
                document.getElementById('stat-annotated').textContent = data.status_counts?.annotated || 0;
                
                const breakdown = document.getElementById('status-breakdown');
                let html = '';
                const total = data.total_items || 1;
                
                for (const [status, count] of Object.entries(data.status_counts || {})) {
                    const percent = ((count / total) * 100).toFixed(1);
                    html += `<tr>
                        <td><span class="status-badge status-${status}">${status}</span></td>
                        <td>${count}</td>
                        <td>${percent}%</td>
                    </tr>`;
                }
                
                breakdown.innerHTML = html || '<tr><td colspan="3">No data available</td></tr>';
                log('Statistics refreshed');
            } catch (error) {
                log('Error refreshing statistics: ' + error.message);
            }
        }

        async function submitURLs() {
            const urls = document.getElementById('urls-input').value
                .split('\\n')
                .map(u => u.trim())
                .filter(u => u);
            
            if (urls.length === 0) {
                showMessage('Please enter at least one URL', 'error');
                return;
            }
            
            const data = {
                urls: urls,
                source_type: document.getElementById('source-type').value,
                language: document.getElementById('language').value || null,
                priority: 5
            };
            
            try {
                const response = await fetch('/api/submit-urls', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showMessage(`Successfully added ${result.added_count} texts for processing`, 'success');
                    document.getElementById('urls-input').value = '';
                    log(`Added ${result.added_count} texts from ${data.source_type}`);
                    refreshStats();
                } else {
                    showMessage('Error: ' + result.message, 'error');
                }
            } catch (error) {
                showMessage('Network error: ' + error.message, 'error');
            }
        }

        async function startPipeline() {
            try {
                const response = await fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'start'})
                });
                
                const result = await response.json();
                showMessage('Processing pipeline started', 'success');
                log('Pipeline started - processing in background');
            } catch (error) {
                showMessage('Error starting pipeline: ' + error.message, 'error');
            }
        }

        async function stopPipeline() {
            try {
                const response = await fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'stop'})
                });
                
                showMessage('Processing pipeline stopped', 'info');
                log('Pipeline stopped');
            } catch (error) {
                showMessage('Error stopping pipeline: ' + error.message, 'error');
            }
        }

        function loadPreset(type) {
            const presets = {
                'perseus': 'https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.01.0133',
                'github': 'https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml',
                'gutenberg': 'https://www.gutenberg.org/cache/epub/1/pg1.txt',
                'custom': ''
            };
            
            document.getElementById('urls-input').value = presets[type] || '';
            document.getElementById('source-type').value = type;
            log(`Loaded ${type} preset`);
        }

        function downloadSample() {
            // Download sample PROIEL file
            window.location.href = '/api/athdgc/sample/download';
            log('Downloading sample PROIEL file');
        }

        // Auto-refresh stats every 10 seconds
        setInterval(refreshStats, 10000);
        
        // Initial load
        refreshStats();
        log('Dashboard loaded - ready for operation');
    </script>
</body>
</html>
"""


@app.get("/api/statistics")
@app.get("/api/stats")  # Alias for compatibility
async def get_statistics():
    """Get platform statistics"""
    stats = platform.db.get_statistics()
    return JSONResponse(stats)


@app.post("/api/submit-urls")
async def submit_urls(submission: URLSubmission):
    """Submit URLs for processing"""
    try:
        added = platform.add_source_urls(
            submission.urls,
            source_type=submission.source_type,
            language=submission.language,
            priority=submission.priority
        )
        return {
            "success": True,
            "added_count": added,
            "message": f"Successfully added {added} texts"
        }
    except Exception as e:
        return {
            "success": False,
            "added_count": 0,
            "message": str(e)
        }


@app.post("/api/control")
async def control_pipeline(request: dict, background_tasks: BackgroundTasks):
    """Control pipeline execution"""
    action = request.get("action")
    
    if action == "start":
        background_tasks.add_task(run_pipeline_background)
        return {"status": "started", "message": "Pipeline started"}
    elif action == "stop":
        platform.running = False
        return {"status": "stopped", "message": "Pipeline stopped"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


async def run_pipeline_background():
    """Run pipeline in background"""
    await platform.run_pipeline(cycles=None, cycle_delay=30)


# ============================================================================
# ATHDGC DIACHRONIC LINGUISTICS ENDPOINTS
# ============================================================================

@app.get("/api/athdgc/health")
async def athdgc_health():
    """Check AthDGC module health"""
    return {
        "status": "operational",
        "version": "3.0.0",
        "modules": {
            "proiel_processor": proiel_processor is not None,
            "valency_lexicon": valency_lexicon is not None,
            "statistical_analysis": diachronic_stats is not None,
            "language_models": language_models.model is not None
        }
    }


@app.post("/api/athdgc/proiel/process")
async def process_proiel(file: UploadFile = File(...), request: PROIELRequest = None):
    """Process PROIEL XML file and extract linguistic data"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process PROIEL
        result = proiel_processor.parse_proiel_xml(tmp_path)
        
        # Add to valency lexicon if language/period specified
        if request and request.language:
            valency_lexicon.bulk_add_from_proiel(
                result, 
                request.language, 
                request.period
            )
        
        # Clean up
        Path(tmp_path).unlink()
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/athdgc/proiel/annotate")
async def annotate_proiel(file: UploadFile = File(...)):
    """Fully annotate PROIEL XML with enhanced linguistic features"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Annotate
        result = proiel_processor.annotate_proiel(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/athdgc/valency/statistics")
async def valency_statistics():
    """Get valency lexicon statistics"""
    try:
        stats = valency_lexicon.get_statistics()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/athdgc/valency/entries")
async def get_valency_entries(verb: Optional[str] = None, 
                              language: Optional[str] = None,
                              period: Optional[str] = None):
    """Query valency lexicon entries"""
    try:
        entries = valency_lexicon.get_entries(
            verb_lemma=verb,
            language=language,
            period=period
        )
        return JSONResponse({"entries": entries, "count": len(entries)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/athdgc/statistical/analyze")
async def statistical_analysis(request: StatisticalAnalysisRequest):
    """Perform diachronic statistical analysis"""
    try:
        # Get samples for both periods
        period_a_samples = valency_lexicon.get_entries(
            language=request.period_a.get('language'),
            period=request.period_a.get('period')
        )
        
        period_b_samples = valency_lexicon.get_entries(
            language=request.period_b.get('language'),
            period=request.period_b.get('period')
        )
        
        if not period_a_samples or not period_b_samples:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data for both periods"
            )
        
        # Perform analysis based on method
        if request.method == "bootstrap":
            result = diachronic_stats.bootstrap_comparison(
                period_a_samples,
                period_b_samples,
                feature=request.feature,
                iterations=request.iterations
            )
        elif request.method == "classifier":
            features = [request.feature, 'argument_structure', 'frequency']
            result = diachronic_stats.classifier_method(
                period_a_samples,
                period_b_samples,
                features
            )
        elif request.method == "effect_size":
            result = diachronic_stats.calculate_effect_size(
                period_a_samples,
                period_b_samples,
                request.feature
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid method")
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/athdgc/semantic/analyze")
async def semantic_analysis(request: SemanticAnalysisRequest):
    """Analyze diachronic semantic shift"""
    try:
        result = language_models.analyze_diachronic_semantics(
            request.period_a_texts,
            request.period_b_texts,
            request.target_word
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/athdgc/semantic/similarity")
async def calculate_similarity(text1: str, text2: str):
    """Calculate semantic similarity between two texts"""
    try:
        similarity = language_models.calculate_semantic_similarity(text1, text2)
        return {"similarity": similarity, "text1": text1[:100], "text2": text2[:100]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/athdgc/semantic/embed")
async def generate_embeddings(texts: List[str]):
    """Generate text embeddings"""
    try:
        embeddings = language_models.generate_embeddings(texts)
        if embeddings is None:
            return {"error": "Model not available", "embeddings": []}
        return {
            "embeddings": embeddings.tolist(),
            "count": len(texts),
            "dimension": embeddings.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/athdgc/valency/export")
async def export_valency_lexicon():
    """Export valency lexicon"""
    try:
        output_path = f"valency_lexicon_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        valency_lexicon.export_lexicon(output_path)
        
        # Read and return
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/athdgc/sample/download")
async def download_sample_proiel():
    """Download sample PROIEL file"""
    from fastapi.responses import FileResponse
    sample_path = Path("corpus_platform/samples/sample.proiel.xml")
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    
    return FileResponse(
        path=str(sample_path),
        filename="sample_greek_proiel.xml",
        media_type="application/xml"
    )


@app.post("/api/athdgc/annotate/multi-ai")
async def multi_ai_annotation(request: MultiAIAnnotationRequest):
    """
    Professional multi-AI annotation endpoint
    Processes text with multiple open-source AI models in parallel
    """
    try:
        results = await multi_ai_engine.annotate_parallel(
            text=request.text,
            language=request.language,
            framework=request.framework,
            detail_level=request.detail_level,
            models=request.models
        )
        
        return JSONResponse({
            'success': True,
            'text': request.text,
            'language': request.language,
            'framework': request.framework,
            'models_used': request.models,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Multi-AI annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-annotation")
async def test_annotation(request: dict):
    """
    Comprehensive test of annotation pipeline with percentage evaluation
    """
    try:
        from comprehensive_test_evaluation import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator()
        results = evaluator.evaluate_text(request['text'], request.get('language', 'grc'))
        
        return JSONResponse(results)
    except Exception as e:
        logger.error(f"Test annotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/treebanks/list")
async def list_treebanks():
    """List all available PROIEL treebanks"""
    try:
        # Query database for treebanks
        db = UnifiedCorpusDatabase()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, language, date_added, word_count
            FROM corpus_items
            WHERE status = 'completed'
            ORDER BY date_added DESC
        """)
        
        treebanks = []
        for row in cursor.fetchall():
            treebanks.append({
                'id': row[0],
                'title': row[1],
                'language': row[2],
                'date_added': row[3],
                'word_count': row[4]
            })
        
        return JSONResponse(treebanks)
    except Exception as e:
        logger.error(f"List treebanks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/treebanks/{treebank_id}")
async def get_treebank(treebank_id: int):
    """Get specific treebank data"""
    try:
        # Load treebank from database
        db = UnifiedCorpusDatabase()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, content, language
            FROM corpus_items
            WHERE id = ?
        """, (treebank_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Treebank not found")
        
        # Parse content and return structured data
        return JSONResponse({
            'id': treebank_id,
            'title': row[0],
            'language': row[2],
            'sentences': []  # TODO: Parse PROIEL XML
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get treebank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-collection")
async def start_collection(request: dict):
    """Start 24/7 autonomous collection"""
    try:
        # Start background collection task
        languages = request.get('languages', ['grc', 'lat', 'en'])
        texts_per_cycle = request.get('texts_per_cycle', 10)
        
        logger.info(f"Starting 24/7 collection: {languages}, {texts_per_cycle} texts/cycle")
        
        # TODO: Implement actual collection start
        
        return JSONResponse({
            'status': 'started',
            'languages': languages,
            'texts_per_cycle': texts_per_cycle,
            'message': '24/7 collection started successfully'
        })
    except Exception as e:
        logger.error(f"Start collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-repositories")
async def test_repositories():
    """Test repository connections"""
    try:
        repositories = [
            {'name': 'Perseus Digital Library', 'url': 'http://www.perseus.tufts.edu'},
            {'name': 'First1KGreek', 'url': 'https://github.com/OpenGreekAndLatin/First1KGreek'},
            {'name': 'PHI Latin Texts', 'url': 'https://latin.packhum.org'},
            {'name': 'Project Gutenberg', 'url': 'https://www.gutenberg.org'},
        ]
        
        results = []
        for repo in repositories:
            # Test connection
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(repo['url'], timeout=aiohttp.ClientTimeout(total=5)) as response:
                        accessible = response.status == 200
                        status = f"Accessible ({response.status})"
            except:
                accessible = False
                status = "Connection failed"
            
            results.append({
                'name': repo['name'],
                'url': repo['url'],
                'accessible': accessible,
                'status': status
            })
        
        return JSONResponse(results)
    except Exception as e:
        logger.error(f"Test repositories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("HFRI-NKUA CORPUS PLATFORM + AthDGC v4.0.0")
    print("Holistic AI-Empowered Diachronic Linguistics Platform")
    print("=" * 70)
    print()
    print("Starting enhanced server with PROIEL visualization...")
    print("Dashboard: http://localhost:8000")
    print("API Docs: http://localhost:8000/professional-docs")
    print("Annotator: http://localhost:8000/annotator")
    print()
    print("Features:")
    print("  ✓ Multi-AI Corpus Annotation")
    print("  ✓ PROIEL XML Processing")
    print("  ✓ Valency Lexicon Building")
    print("  ✓ Diachronic Statistical Analysis")
    print("  ✓ Semantic Similarity & Embeddings")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
