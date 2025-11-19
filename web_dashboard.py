"""
Web Dashboard for Unified Corpus Platform
Real-time monitoring and control interface
"""

# Load local models configuration (use Z:\models\ - no re-downloads)
try:
    import local_models_config
except ImportError:
    pass  # Fall back to default model locations if config not available

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from pathlib import Path
import sqlite3

from unified_corpus_platform import UnifiedCorpusPlatform, UnifiedCorpusDatabase

# Initialize FastAPI
app = FastAPI(
    title="Unified Corpus Platform Dashboard",
    description="Monitor and control your AI corpus platform",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global platform
platform = UnifiedCorpusPlatform()
pipeline_running = False


# === Models ===

class AddURLsRequest(BaseModel):
    urls: List[str]
    source_type: str = "custom_url"
    language: Optional[str] = None
    priority: int = 5


class ControlRequest(BaseModel):
    action: str  # 'start' or 'stop'
    cycles: Optional[int] = None
    delay: int = 10


# === API Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Unified Corpus Platform</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }
        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
        }
        .section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .section h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }
        button:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        button.stop {
            background: #e74c3c;
        }
        button.stop:hover {
            background: #c0392b;
        }
        input, textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 10px;
        }
        textarea {
            resize: vertical;
            min-height: 100px;
            font-family: monospace;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px 5px 5px 0;
        }
        .status-discovered { background: #3498db; color: white; }
        .status-scraped { background: #9b59b6; color: white; }
        .status-parsed { background: #f39c12; color: white; }
        .status-annotated { background: #27ae60; color: white; }
        .status-failed { background: #e74c3c; color: white; }
        .log {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
        }
        .success {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Unified Corpus Platform</h1>
        <p class="subtitle">Automatic Scraping, Parsing & Annotation</p>
        
        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <div class="stat-label">Total Items</div>
                <div class="stat-value" id="total">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Scraped</div>
                <div class="stat-value" id="scraped">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Parsed</div>
                <div class="stat-value" id="parsed">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Annotated</div>
                <div class="stat-value" id="annotated">-</div>
            </div>
        </div>

        <div class="section">
            <h2>📥 Add URLs</h2>
            <textarea id="urls" placeholder="Enter URLs (one per line)&#10;https://example.com/text1.xml&#10;https://example.com/text2.xml"></textarea>
            <input type="text" id="language" placeholder="Language (grc, en, etc.) - optional">
            <div class="controls">
                <button onclick="addURLs()">Add URLs</button>
                <select id="source-type" style="padding: 12px; border-radius: 8px; border: 2px solid #ddd;">
                    <option value="custom_url">Custom URL</option>
                    <option value="perseus">Perseus</option>
                    <option value="github">GitHub</option>
                    <option value="gutenberg">Project Gutenberg</option>
                </select>
            </div>
            <div id="add-message"></div>
        </div>

        <div class="section">
            <h2>⚙️ Pipeline Control</h2>
            <div class="controls">
                <button onclick="startPipeline()">▶️ Start Pipeline</button>
                <button onclick="stopPipeline()" class="stop">⏹️ Stop Pipeline</button>
                <button onclick="refreshStats()">🔄 Refresh Stats</button>
            </div>
            <p style="margin-top: 15px; color: #666;">
                Pipeline status: <strong id="pipeline-status">Unknown</strong>
            </p>
        </div>

        <div class="section">
            <h2>📊 Status Breakdown</h2>
            <div id="status-breakdown"></div>
        </div>

        <div class="refresh-info">
            Dashboard auto-refreshes every 10 seconds
        </div>
    </div>

    <script>
        async function refreshStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total').textContent = data.total_items;
                const statusCounts = data.status_counts || {};
                document.getElementById('scraped').textContent = statusCounts.scraped || 0;
                document.getElementById('parsed').textContent = statusCounts.parsed || 0;
                document.getElementById('annotated').textContent = statusCounts.annotated || 0;
                
                // Status breakdown
                let breakdown = '';
                for (const [status, count] of Object.entries(statusCounts)) {
                    breakdown += `<span class="status-badge status-${status}">${status}: ${count}</span>`;
                }
                document.getElementById('status-breakdown').innerHTML = breakdown || 'No items yet';
                
            } catch (error) {
                console.error('Error refreshing stats:', error);
            }
        }

        async function addURLs() {
            const urlsText = document.getElementById('urls').value.trim();
            const language = document.getElementById('language').value.trim() || null;
            const sourceType = document.getElementById('source-type').value;
            const messageDiv = document.getElementById('add-message');
            
            if (!urlsText) {
                messageDiv.innerHTML = '<div class="error">Please enter at least one URL</div>';
                return;
            }
            
            const urls = urlsText.split('\\n').filter(u => u.trim());
            
            try {
                const response = await fetch('/api/add-urls', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        urls: urls,
                        source_type: sourceType,
                        language: language,
                        priority: 5
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    messageDiv.innerHTML = `<div class="success">${result.message}</div>`;
                    document.getElementById('urls').value = '';
                    refreshStats();
                } else {
                    messageDiv.innerHTML = `<div class="error">${result.message}</div>`;
                }
                
            } catch (error) {
                messageDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        async function startPipeline() {
            try {
                const response = await fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'start', cycles: null, delay: 10})
                });
                const result = await response.json();
                document.getElementById('pipeline-status').textContent = 'Running';
            } catch (error) {
                alert('Error starting pipeline: ' + error.message);
            }
        }

        async function stopPipeline() {
            try {
                const response = await fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'stop'})
                });
                const result = await response.json();
                document.getElementById('pipeline-status').textContent = 'Stopped';
            } catch (error) {
                alert('Error stopping pipeline: ' + error.message);
            }
        }

        // Auto-refresh every 10 seconds
        setInterval(refreshStats, 10000);
        
        // Initial load
        refreshStats();
    </script>
</body>
</html>
"""


@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    stats = platform.db.get_statistics()
    return JSONResponse(stats)


@app.post("/api/add-urls")
async def add_urls(request: AddURLsRequest):
    """Add URLs to process"""
    try:
        added = platform.add_source_urls(
            request.urls,
            source_type=request.source_type,
            language=request.language,
            priority=request.priority
        )
        return {
            "success": True,
            "added_count": added,
            "message": f"Successfully added {added} URL(s) to the corpus"
        }
    except Exception as e:
        return {
            "success": False,
            "added_count": 0,
            "message": f"Error: {str(e)}"
        }


@app.post("/api/control")
async def control_pipeline(request: ControlRequest, background_tasks: BackgroundTasks):
    """Control pipeline execution"""
    global pipeline_running
    
    if request.action == "start":
        if pipeline_running:
            raise HTTPException(status_code=400, detail="Pipeline already running")
        
        pipeline_running = True
        background_tasks.add_task(run_pipeline_background, request.cycles, request.delay)
        return {"status": "started", "message": "Pipeline started in background"}
    
    elif request.action == "stop":
        platform.running = False
        pipeline_running = False
        return {"status": "stopped", "message": "Pipeline stopped"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


async def run_pipeline_background(cycles: Optional[int], delay: int):
    """Run pipeline in background"""
    global pipeline_running
    try:
        await platform.run_pipeline(cycles=cycles, cycle_delay=delay)
    finally:
        pipeline_running = False


@app.get("/api/items")
async def get_items(status: Optional[str] = None, limit: int = 100):
    """Get corpus items"""
    if status:
        items = platform.db.get_items_by_status(status, limit)
    else:
        conn = sqlite3.connect(platform.db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM corpus_items LIMIT {limit}")
        items = [dict(row) for row in cursor.fetchall()]
        conn.close()
    
    return JSONResponse({"items": items, "count": len(items)})


if __name__ == "__main__":
    import uvicorn
    print("Starting Unified Corpus Platform Dashboard...")
    print("Dashboard will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000)
