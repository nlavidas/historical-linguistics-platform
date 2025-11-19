"""
Professional Metadata Dashboard - HFRI Corpus Platform
Clean, academic interface without emojis or excessive colors
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import sqlite3
from pathlib import Path
from datetime import datetime
import uvicorn

app = FastAPI(title="HFRI Corpus Metadata")

def get_metadata_db():
    """Connect to metadata database"""
    db_path = Path("Z:/corpus_platform/corpus_metadata.db")
    if not db_path.exists():
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
async def professional_dashboard():
    """Professional academic dashboard"""
    
    conn = get_metadata_db()
    cursor = conn.cursor()
    
    # Get all texts
    cursor.execute("""
        SELECT author, work, language, period, date_bce, genre, 
               word_count, sentence_count, file_size_kb, processing_date, status
        FROM texts
        ORDER BY author, date_bce
    """)
    texts = cursor.fetchall()
    
    # Statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_texts,
            SUM(word_count) as total_words,
            SUM(sentence_count) as total_sentences,
            ROUND(AVG(word_count), 0) as avg_words
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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HFRI Corpus Platform - Metadata Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Times New Roman', Georgia, serif;
                background: #f5f5f5;
                padding: 30px;
                color: #000;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            header {{
                border-bottom: 3px solid #8B0000;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            
            h1 {{
                font-size: 28px;
                font-weight: bold;
                color: #8B0000;
                margin-bottom: 10px;
            }}
            
            .subtitle {{
                font-size: 14px;
                color: #666;
                font-style: italic;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin: 30px 0;
                border: 1px solid #ccc;
                padding: 20px;
                background: #fafafa;
            }}
            
            .stat-box {{
                text-align: center;
                padding: 15px;
                border-right: 1px solid #ddd;
            }}
            
            .stat-box:last-child {{
                border-right: none;
            }}
            
            .stat-value {{
                font-size: 32px;
                font-weight: bold;
                color: #8B0000;
                margin: 10px 0;
            }}
            
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .section {{
                margin: 40px 0;
            }}
            
            .section-title {{
                font-size: 18px;
                font-weight: bold;
                color: #8B0000;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #8B0000;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-size: 13px;
            }}
            
            th {{
                background: #8B0000;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: normal;
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }}
            
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            
            tr:nth-child(even) {{
                background: #fafafa;
            }}
            
            .lang-grc {{
                font-style: italic;
            }}
            
            .status-complete {{
                background: #8B0000;
                color: white;
                padding: 3px 8px;
                font-size: 10px;
                text-transform: uppercase;
            }}
            
            .status-processing {{
                background: #999;
                color: white;
                padding: 3px 8px;
                font-size: 10px;
                text-transform: uppercase;
            }}
            
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ccc;
                text-align: center;
                font-size: 11px;
                color: #666;
            }}
            
            .refresh-info {{
                text-align: right;
                font-size: 11px;
                color: #999;
                margin-top: 20px;
            }}
            
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    box-shadow: none;
                }}
            }}
        </style>
        <script>
            setTimeout(function(){{ location.reload(); }}, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>HFRI Corpus Platform: Metadata Dashboard</h1>
                <div class="subtitle">
                    Principal Investigator: Prof. Nikolaos Lavidas<br>
                    Institution: National and Kapodistrian University of Athens (NKUA)<br>
                    Funding: Hellenic Foundation for Research and Innovation (HFRI)
                </div>
            </header>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Total Texts</div>
                    <div class="stat-value">{stats[0] or 0}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Words</div>
                    <div class="stat-value">{stats[1] or 0:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Sentences</div>
                    <div class="stat-value">{stats[2] or 0:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average Words/Text</div>
                    <div class="stat-value">{int(stats[3] or 0):,}</div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Distribution by Language</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Language</th>
                            <th style="text-align: right;">Texts</th>
                            <th style="text-align: right;">Words</th>
                            <th style="text-align: right;">Sentences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td>{
                                "Ancient Greek" if row[0] == "grc" else "Latin"
                            } ({row[0]})</td>
                            <td style="text-align: right;">{row[1]}</td>
                            <td style="text-align: right;">{row[2]:,}</td>
                            <td style="text-align: right;">{row[3]:,}</td>
                        </tr>
                        ''' for row in by_language)}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Distribution by Author</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Author</th>
                            <th style="text-align: right;">Works</th>
                            <th style="text-align: right;">Total Words</th>
                            <th style="text-align: right;">Total Sentences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td><strong>{row[0]}</strong></td>
                            <td style="text-align: right;">{row[1]}</td>
                            <td style="text-align: right;">{row[2]:,}</td>
                            <td style="text-align: right;">{row[3]:,}</td>
                        </tr>
                        ''' for row in by_author)}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Distribution by Historical Period</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th style="text-align: right;">Texts</th>
                            <th style="text-align: right;">Total Words</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td>{row[0]}</td>
                            <td style="text-align: right;">{row[1]}</td>
                            <td style="text-align: right;">{row[2]:,}</td>
                        </tr>
                        ''' for row in by_period)}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Distribution by Genre</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Genre</th>
                            <th style="text-align: right;">Texts</th>
                            <th style="text-align: right;">Total Words</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td>{row[0]}</td>
                            <td style="text-align: right;">{row[1]}</td>
                            <td style="text-align: right;">{row[2]:,}</td>
                        </tr>
                        ''' for row in by_genre)}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Complete Corpus Inventory</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Author</th>
                            <th>Work</th>
                            <th>Language</th>
                            <th>Period</th>
                            <th>Date BCE</th>
                            <th>Genre</th>
                            <th style="text-align: right;">Words</th>
                            <th style="text-align: right;">Sentences</th>
                            <th style="text-align: right;">Size (KB)</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f'''
                        <tr>
                            <td><strong>{row[0]}</strong></td>
                            <td class="{'lang-grc' if row[2] == 'grc' else ''}">{row[1]}</td>
                            <td>{row[2]}</td>
                            <td>{row[3]}</td>
                            <td style="text-align: center;">{row[4]}</td>
                            <td>{row[5]}</td>
                            <td style="text-align: right;">{row[6]:,}</td>
                            <td style="text-align: right;">{row[7]:,}</td>
                            <td style="text-align: right;">{row[8]:.1f}</td>
                            <td><span class="status-{row[10]}">{row[10]}</span></td>
                        </tr>
                        ''' for row in texts)}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <strong>HFRI-NKUA AI Corpus Platform</strong><br>
                Autonomous Open-Access Text Scraping System for PROIEL Corpus Construction<br>
                Version 3.0 | {datetime.now().year}
            </div>
            
            <div class="refresh-info">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 30 seconds
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

@app.get("/api/stats")
async def api_stats():
    """JSON API for statistics"""
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
    print("PROFESSIONAL METADATA DASHBOARD - PROIEL RED")
    print("="*70)
    print("URL: http://localhost:9001")
    print("Style: Clean academic interface with PROIEL red")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="warning")
