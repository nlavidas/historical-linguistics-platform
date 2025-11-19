"""
METADATA DASHBOARD FOR CORPUS
Shows authors, works, word counts, and all metadata in beautiful interface
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
from pathlib import Path
from datetime import datetime
import uvicorn
from typing import Dict, List

app = FastAPI(title="HFRI Corpus Metadata Dashboard")

def get_metadata_db():
    """Connect to metadata database"""
    db_path = Path("Z:/corpus_platform/corpus_metadata.db")
    if not db_path.exists():
        # Create it if doesn't exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                author TEXT NOT NULL,
                work TEXT NOT NULL,
                language TEXT,
                period TEXT,
                date_bce INTEGER,
                genre TEXT,
                word_count INTEGER,
                sentence_count INTEGER,
                file_path TEXT,
                file_size_kb REAL,
                download_url TEXT,
                processing_date TIMESTAMP,
                status TEXT
            )
        """)
        conn.commit()
    else:
        conn = sqlite3.connect(str(db_path))
    return conn

@app.get("/", response_class=HTMLResponse)
async def metadata_dashboard():
    """Main metadata dashboard"""
    
    conn = get_metadata_db()
    cursor = conn.cursor()
    
    # Get all texts with metadata
    cursor.execute("""
        SELECT author, work, language, period, date_bce, genre, 
               word_count, sentence_count, file_size_kb, processing_date, status
        FROM texts
        ORDER BY author, date_bce
    """)
    texts = cursor.fetchall()
    
    # Get statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_texts,
            SUM(word_count) as total_words,
            SUM(sentence_count) as total_sentences,
            ROUND(AVG(word_count), 0) as avg_words_per_text
        FROM texts WHERE status='complete'
    """)
    stats = cursor.fetchone()
    
    # By language
    cursor.execute("""
        SELECT language, COUNT(*), SUM(word_count), SUM(sentence_count)
        FROM texts WHERE status='complete'
        GROUP BY language
    """)
    by_language = cursor.fetchall()
    
    # By author
    cursor.execute("""
        SELECT author, COUNT(*), SUM(word_count), SUM(sentence_count)
        FROM texts WHERE status='complete'
        GROUP BY author
        ORDER BY author
    """)
    by_author = cursor.fetchall()
    
    # By period
    cursor.execute("""
        SELECT period, COUNT(*), SUM(word_count)
        FROM texts WHERE status='complete'
        GROUP BY period
        ORDER BY MIN(date_bce)
    """)
    by_period = cursor.fetchall()
    
    # By genre
    cursor.execute("""
        SELECT genre, COUNT(*), SUM(word_count)
        FROM texts WHERE status='complete'
        GROUP BY genre
        ORDER BY COUNT(*) DESC
    """)
    by_genre = cursor.fetchall()
    
    conn.close()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HFRI Corpus Metadata Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }}
            
            h1 {{
                color: #2a5298;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            
            .subtitle {{
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }}
            
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .stat-label {{
                font-size: 1em;
                opacity: 0.9;
            }}
            
            .section {{
                margin: 30px 0;
            }}
            
            .section-title {{
                font-size: 1.8em;
                color: #2a5298;
                margin-bottom: 15px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            th {{
                background: linear-gradient(135deg, #2a5298 0%, #667eea 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-size: 1.1em;
            }}
            
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
            }}
            
            tr:hover {{
                background: #f5f5f5;
            }}
            
            .language-grc {{
                color: #d35400;
                font-weight: bold;
            }}
            
            .language-lat {{
                color: #27ae60;
                font-weight: bold;
            }}
            
            .status-complete {{
                background: #27ae60;
                color: white;
                padding: 4px 10px;
                border-radius: 15px;
                font-size: 0.9em;
            }}
            
            .status-failed {{
                background: #e74c3c;
                color: white;
                padding: 4px 10px;
                border-radius: 15px;
                font-size: 0.9em;
            }}
            
            .timestamp {{
                text-align: center;
                color: #999;
                margin-top: 30px;
                font-size: 0.9em;
            }}
            
            .refresh-btn {{
                display: block;
                width: 200px;
                margin: 20px auto;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                text-align: center;
                text-decoration: none;
            }}
            
            .refresh-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }}
        </style>
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function(){{
                location.reload();
            }}, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>📚 HFRI Corpus Metadata Dashboard</h1>
            <div class="subtitle">
                Prof. Nikolaos Lavidas - NKUA/HFRI<br>
                Live corpus statistics with full metadata tracking
            </div>
            
            <!-- Overall Statistics -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Texts</div>
                    <div class="stat-value">{stats[0] or 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Words</div>
                    <div class="stat-value">{stats[1] or 0:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Sentences</div>
                    <div class="stat-value">{stats[2] or 0:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Words/Text</div>
                    <div class="stat-value">{int(stats[3] or 0):,}</div>
                </div>
            </div>
            
            <!-- By Language -->
            <div class="section">
                <h2 class="section-title">📖 By Language</h2>
                <table>
                    <tr>
                        <th>Language</th>
                        <th>Texts</th>
                        <th>Words</th>
                        <th>Sentences</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td class="language-{row[0]}">{
                            "Ancient Greek (grc)" if row[0] == "grc" else "Latin (lat)"
                        }</td>
                        <td>{row[1]}</td>
                        <td>{row[2]:,}</td>
                        <td>{row[3]:,}</td>
                    </tr>
                    ''' for row in by_language)}
                </table>
            </div>
            
            <!-- By Author -->
            <div class="section">
                <h2 class="section-title">✍️ By Author</h2>
                <table>
                    <tr>
                        <th>Author</th>
                        <th>Works</th>
                        <th>Total Words</th>
                        <th>Total Sentences</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td><strong>{row[0]}</strong></td>
                        <td>{row[1]}</td>
                        <td>{row[2]:,}</td>
                        <td>{row[3]:,}</td>
                    </tr>
                    ''' for row in by_author)}
                </table>
            </div>
            
            <!-- By Period -->
            <div class="section">
                <h2 class="section-title">🕰️ By Historical Period</h2>
                <table>
                    <tr>
                        <th>Period</th>
                        <th>Texts</th>
                        <th>Total Words</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td><strong>{row[0]}</strong></td>
                        <td>{row[1]}</td>
                        <td>{row[2]:,}</td>
                    </tr>
                    ''' for row in by_period)}
                </table>
            </div>
            
            <!-- By Genre -->
            <div class="section">
                <h2 class="section-title">🎭 By Genre</h2>
                <table>
                    <tr>
                        <th>Genre</th>
                        <th>Texts</th>
                        <th>Total Words</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td><strong>{row[0]}</strong></td>
                        <td>{row[1]}</td>
                        <td>{row[2]:,}</td>
                    </tr>
                    ''' for row in by_genre)}
                </table>
            </div>
            
            <!-- All Texts Detail -->
            <div class="section">
                <h2 class="section-title">📋 All Texts - Complete Metadata</h2>
                <table>
                    <tr>
                        <th>Author</th>
                        <th>Work</th>
                        <th>Lang</th>
                        <th>Period</th>
                        <th>Date BCE</th>
                        <th>Genre</th>
                        <th>Words</th>
                        <th>Sentences</th>
                        <th>Size (KB)</th>
                        <th>Status</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td><strong>{row[0]}</strong></td>
                        <td>{row[1]}</td>
                        <td class="language-{row[2]}">{row[2]}</td>
                        <td>{row[3]}</td>
                        <td>{row[4]}</td>
                        <td>{row[5]}</td>
                        <td>{row[6]:,}</td>
                        <td>{row[7]:,}</td>
                        <td>{row[8]:.1f}</td>
                        <td><span class="status-{row[10]}">{row[10]}</span></td>
                    </tr>
                    ''' for row in texts)}
                </table>
            </div>
            
            <a href="/" class="refresh-btn">🔄 Refresh Data</a>
            
            <div class="timestamp">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Greece Time<br>
                Auto-refreshes every 30 seconds
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

@app.get("/api/stats")
async def get_stats():
    """API endpoint for statistics"""
    conn = get_metadata_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_texts,
            SUM(word_count) as total_words,
            SUM(sentence_count) as total_sentences
        FROM texts WHERE status='complete'
    """)
    stats = cursor.fetchone()
    
    conn.close()
    
    return {
        "total_texts": stats[0] or 0,
        "total_words": stats[1] or 0,
        "total_sentences": stats[2] or 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("="*70)
    print("🌐 METADATA DASHBOARD STARTING")
    print("="*70)
    print("Access at: http://localhost:8080")
    print("Features: Authors, word counts, all metadata")
    print("Auto-refresh: Every 30 seconds")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
