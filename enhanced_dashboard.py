"""
Enhanced Professional Metadata Dashboard with Visualizations
Features:
- Real-time metrics
- Interactive visualizations
- Advanced filtering
- Export capabilities
"""

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import uvicorn
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Corpus Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_db_connection():
    """Create database connection with row factory"""
    db_path = Path("Z:/corpus_platform/corpus_metadata.db")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

# Dashboard Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Data API Endpoints

@app.get("/api/stats")
async def get_stats():
    """Get overview statistics"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_texts,
                SUM(word_count) as total_words,
                COUNT(DISTINCT author) as unique_authors,
                COUNT(DISTINCT language) as languages
            FROM texts 
            WHERE status = 'complete'
        """)
        stats = dict(cursor.fetchone())
        
        # Growth over time
        cursor.execute("""
            SELECT 
                DATE(processing_date) as date,
                COUNT(*) as texts_added,
                SUM(word_count) as words_added
            FROM texts
            WHERE status = 'complete'
            GROUP BY DATE(processing_date)
            ORDER BY date
        """)
        growth_data = [dict(row) for row in cursor.fetchall()]
        
        # Language distribution
        cursor.execute("""
            SELECT 
                language,
                COUNT(*) as text_count,
                SUM(word_count) as word_count
            FROM texts
            WHERE status = 'complete'
            GROUP BY language
            ORDER BY word_count DESC
        """)
        language_data = [dict(row) for row in cursor.fetchall()]
        
        # Period distribution
        cursor.execute("""
            SELECT 
                period,
                COUNT(*) as text_count,
                SUM(word_count) as word_count
            FROM texts
            WHERE status = 'complete' AND period IS NOT NULL
            GROUP BY period
            ORDER BY MIN(date_bce)
        """)
        period_data = [dict(row) for row in cursor.fetchall()]
        
        # Genre distribution
        cursor.execute("""
            SELECT 
                genre,
                COUNT(*) as text_count,
                SUM(word_count) as word_count
            FROM texts
            WHERE status = 'complete' AND genre IS NOT NULL
            GROUP BY genre
            ORDER BY word_count DESC
        """)
        genre_data = [dict(row) for row in cursor.fetchall()]
        
        # Author productivity
        cursor.execute("""
            SELECT 
                author,
                COUNT(*) as text_count,
                SUM(word_count) as word_count
            FROM texts
            WHERE status = 'complete' AND author IS NOT NULL
            GROUP BY author
            ORDER BY word_count DESC
            LIMIT 20
        """)
        author_data = [dict(row) for row in cursor.fetchall()]
        
        return {
            "overview": stats,
            "growth": growth_data,
            "languages": language_data,
            "periods": period_data,
            "genres": genre_data,
            "authors": author_data
        }

@app.get("/api/texts")
async def get_texts(
    page: int = 1, 
    per_page: int = 20,
    language: Optional[str] = None,
    period: Optional[str] = None,
    genre: Optional[str] = None,
    author: Optional[str] = None,
    search: Optional[str] = None
):
    """Get paginated list of texts with filtering"""
    offset = (page - 1) * per_page
    
    query = """
        SELECT * FROM texts
        WHERE status = 'complete'
    """
    params = []
    
    # Apply filters
    conditions = []
    if language:
        conditions.append("language = ?")
        params.append(language)
    if period:
        conditions.append("period = ?")
        params.append(period)
    if genre:
        conditions.append("genre = ?")
        params.append(genre)
    if author:
        conditions.append("author = ?")
        params.append(author)
    if search:
        conditions.append("(author LIKE ? OR work LIKE ? OR content LIKE ?)")
        search_term = f"%{search}%"
        params.extend([search_term, search_term, search_term])
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    # Get total count
    count_query = f"SELECT COUNT(*) as total FROM ({query})"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(count_query, params)
        total = cursor.fetchone()["total"]
        
        # Get paginated results
        query += " ORDER BY author, work LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": (total + per_page - 1) // per_page,
            "data": results
        }

# Visualization Endpoints

@app.get("/api/charts/language-distribution")
async def language_distribution_chart():
    """Generate language distribution chart"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                language,
                COUNT(*) as text_count,
                SUM(word_count) as word_count
            FROM texts
            WHERE status = 'complete'
            GROUP BY language
            ORDER BY word_count DESC
            LIMIT 10
        """)
        data = [dict(row) for row in cursor.fetchall()]
        
        # Create bar chart
        df = pd.DataFrame(data)
        fig = px.bar(
            df, 
            x='language', 
            y='word_count',
            title='Word Count by Language',
            labels={'word_count': 'Word Count', 'language': 'Language'},
            hover_data=['text_count']
        )
        
        return JSONResponse(content=fig.to_json())

# Export Endpoints

@app.get("/api/export/csv")
async def export_csv():
    """Export all texts as CSV"""
    with get_db_connection() as conn:
        df = pd.read_sql("""
            SELECT 
                author, work, language, period, date_bce, genre,
                word_count, sentence_count, file_size_kb, processing_date
            FROM texts 
            WHERE status = 'complete'
            ORDER BY author, work
        """, conn)
        
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=corpus_export.csv"
        return response

# Template Routes

@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Dashboard page with visualizations"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/browse")
async def browse_page(request: Request):
    """Browse texts page"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get filter options
        cursor.execute("SELECT DISTINCT language FROM texts WHERE language IS NOT NULL ORDER BY language")
        languages = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT DISTINCT period FROM texts WHERE period IS NOT NULL ORDER BY period")
        periods = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT DISTINCT genre FROM texts WHERE genre IS NOT NULL ORDER BY genre")
        genres = [row[0] for row in cursor.fetchall()]
        
        return templates.TemplateResponse(
            "browse.html",
            {
                "request": request,
                "languages": languages,
                "periods": periods,
                "genres": genres
            }
        )

# Health Check

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM texts")
            count = cursor.fetchone()["count"]
            
            return {
                "status": "healthy",
                "database": "connected",
                "text_count": count,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unavailable")

# Startup Event

@app.on_event("startup")
async def startup():
    """Initialize the application"""
    # Create necessary directories
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    # Create default templates if they don't exist
    if not Path("templates/dashboard.html").exists():
        create_default_templates()
    
    logger.info("Enhanced Dashboard started successfully")

def create_default_templates():
    """Create default HTML templates"""
    dashboard_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Corpus Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .stat-card {
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chart-container {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Corpus Platform</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link active" href="/dashboard">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/browse">Browse Texts</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/api/export/csv">Export Data</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <h1 class="mb-4">Corpus Dashboard</h1>
            
            <!-- Stats Overview -->
            <div class="row mb-4" id="stats-overview">
                <!-- Stats will be loaded here by JavaScript -->
            </div>
            
            <!-- Charts Row 1 -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Texts Added Over Time</h4>
                        <div id="growth-chart"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Word Count by Language</h4>
                        <div id="language-chart"></div>
                    </div>
                </div>
            </div>
            
            <!-- Charts Row 2 -->
            <div class="row">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Texts by Period</h4>
                        <div id="period-chart"></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h4>Texts by Genre</h4>
                        <div id="genre-chart"></div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="/static/dashboard.js"></script>
    </body>
    </html>
    """
    
    browse_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Browse Texts - Corpus Platform</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css">
        <style>
            .filter-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .dataTables_wrapper {
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Corpus Platform</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link" href="/dashboard">Dashboard</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" href="/browse">Browse Texts</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/api/export/csv">Export Data</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <h1 class="mb-4">Browse Texts</h1>
            
            <!-- Filters -->
            <div class="filter-section">
                <form id="filters-form" class="row g-3">
                    <div class="col-md-3">
                        <label for="language" class="form-label">Language</label>
                        <select id="language" class="form-select">
                            <option value="">All Languages</option>
                            {% for lang in languages %}
                                <option value="{{ lang }}">{{ lang }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label for="period" class="form-label">Period</label>
                        <select id="period" class="form-select">
                            <option value="">All Periods</option>
                            {% for period in periods %}
                                <option value="{{ period }}">{{ period }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label for="genre" class="form-label">Genre</label>
                        <select id="genre" class="form-select">
                            <option value="">All Genres</option>
                            {% for genre in genres %}
                                <option value="{{ genre }}">{{ genre }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="col-md-3 d-flex align-items-end">
                        <button type="submit" class="btn btn-primary w-100">Apply Filters</button>
                    </div>
                </form>
            </div>
            
            <!-- Texts Table -->
            <table id="texts-table" class="table table-striped" style="width:100%">
                <thead>
                    <tr>
                        <th>Author</th>
                        <th>Work</th>
                        <th>Language</th>
                        <th>Period</th>
                        <th>Genre</th>
                        <th>Words</th>
                        <th>Date Added</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Data will be loaded by DataTables -->
                </tbody>
            </table>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
        <script src="/static/browse.js"></script>
    </body>
    </html>
    """
    
    # Write templates to files
    Path("templates/dashboard.html").write_text(dashboard_template)
    Path("templates/browse.html").write_text(browse_template)
    
    # Create static directory and JavaScript files
    Path("static").mkdir(exist_ok=True)
    
    # dashboard.js
    dashboard_js = """
    // Load dashboard data
    async function loadDashboard() {
        try {
            // Load stats
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            // Update stats overview
            const stats = data.overview;
            document.getElementById('stats-overview').innerHTML = `
                <div class="col-md-3">
                    <div class="stat-card bg-primary text-white">
                        <h3>${stats.total_texts.toLocaleString()}</h3>
                        <p class="mb-0">Texts</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card bg-success text-white">
                        <h3>${stats.total_words.toLocaleString()}</h3>
                        <p class="mb-0">Words</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card bg-info text-white">
                        <h3>${stats.unique_authors.toLocaleString()}</h3>
                        <p class="mb-0">Authors</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card bg-warning text-dark">
                        <h3>${stats.languages.toLocaleString()}</h3>
                        <p class="mb-0">Languages</p>
                    </div>
                </div>
            `;
            
            // Create charts
            createGrowthChart(data.growth);
            createLanguageChart(data.languages);
            createPeriodChart(data.periods);
            createGenreChart(data.genres);
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }
    
    // Create growth chart
    function createGrowthChart(data) {
        const trace = {
            x: data.map(d => d.date),
            y: data.map(d => d.texts_added),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Texts Added',
            yaxis: 'y1'
        };
        
        const trace2 = {
            x: data.map(d => d.date),
            y: data.map(d => d.words_added),
            type: 'bar',
            name: 'Words Added',
            yaxis: 'y2'
        };
        
        const layout = {
            title: 'Corpus Growth Over Time',
            xaxis: { title: 'Date' },
            yaxis: { 
                title: 'Texts Added',
                side: 'left',
                showgrid: false
            },
            yaxis2: {
                title: 'Words Added',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            },
            showlegend: true,
            legend: { x: 1.1, y: 1 }
        };
        
        Plotly.newPlot('growth-chart', [trace, trace2], layout);
    }
    
    // Create language chart
    function createLanguageChart(data) {
        const trace = {
            labels: data.map(d => d.language),
            values: data.map(d => d.word_count),
            type: 'pie',
            textinfo: 'label+percent',
            hoverinfo: 'label+value+percent',
            textposition: 'inside',
            hole: 0.4
        };
        
        const layout = {
            showlegend: false
        };
        
        Plotly.newPlot('language-chart', [trace], layout);
    }
    
    // Create period chart
    function createPeriodChart(data) {
        const trace = {
            x: data.map(d => d.period),
            y: data.map(d => d.word_count),
            type: 'bar'
        };
        
        const layout = {
            xaxis: { title: 'Period' },
            yaxis: { title: 'Word Count' }
        };
        
        Plotly.newPlot('period-chart', [trace], layout);
    }
    
    // Create genre chart
    function createGenreChart(data) {
        const trace = {
            x: data.map(d => d.genre),
            y: data.map(d => d.word_count),
            type: 'bar'
        };
        
        const layout = {
            xaxis: { title: 'Genre' },
            yaxis: { title: 'Word Count' }
        };
        
        Plotly.newPlot('genre-chart', [trace], layout);
    }
    
    // Initialize dashboard when the page loads
    document.addEventListener('DOMContentLoaded', () => {
        loadDashboard();
        
        // Refresh data every 5 minutes
        setInterval(loadDashboard, 5 * 60 * 1000);
    });
    """
    
    # browse.js
    browse_js = """
    // Initialize DataTable
    let dataTable;
    
    // Initialize the page
    document.addEventListener('DOMContentLoaded', () => {
        initializeDataTable();
        setupEventListeners();
    });
    
    // Initialize DataTable with server-side processing
    function initializeDataTable() {
        dataTable = $('#texts-table').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: '/api/texts',
                data: function(d) {
                    // Add filter parameters
                    d.language = $('#language').val();
                    d.period = $('#period').val();
                    d.genre = $('#genre').val();
                    d.search = d.search.value;
                }
            },
            columns: [
                { data: 'author' },
                { data: 'work' },
                { data: 'language' },
                { data: 'period' },
                { data: 'genre' },
                { 
                    data: 'word_count',
                    render: function(data) {
                        return data ? data.toLocaleString() : '';
                    }
                },
                { 
                    data: 'processing_date',
                    render: function(data) {
                        return data ? new Date(data).toLocaleDateString() : '';
                    }
                }
            ],
            order: [[0, 'asc']],
            pageLength: 25,
            lengthMenu: [10, 25, 50, 100]
        });
    }
    
    // Set up event listeners
    function setupEventListeners() {
        // Form submission
        $('#filters-form').on('submit', function(e) {
            e.preventDefault();
            dataTable.ajax.reload();
            return false;
        });
        
        // Reset filters
        $('#reset-filters').on('click', function() {
            $('#filters-form')[0].reset();
            dataTable.ajax.reload();
        });
    }
    """
    
    # Write JavaScript files
    Path("static/dashboard.js").write_text(dashboard_js)
    Path("static/browse.js").write_text(browse_js)

if __name__ == "__main__":
    # Create necessary directories
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "enhanced_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
