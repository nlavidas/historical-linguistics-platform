#!/usr/bin/env python3
"""Windsurf Web Control Panel - Flask-based web app for the diachronic corpus platform.

This serves the Windsurf-style control panel as a live web app on the VM,
with buttons that trigger commands remotely. Includes real-time updates via
periodic refreshes and WebSocket support for dashboards.

Run:
    python windsurf_web_panel.py

Then open http://localhost:8000 (or VM IP) in browser.

Features:
- Live control panel with command execution
- Real-time dashboard updates
- Mobile-responsive UI
"""

import json
import logging
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytz
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

ROOT = Path(__file__).parent
DB_PATH = ROOT / "corpus_platform.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("windsurf_web_panel")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for real-time updates
current_stats = {}
current_reports = {}
last_update = None

# HTML Template for the control panel
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Diachronic Corpus – Windsurf Web Control Panel</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #e5e7eb;
      font-size: 18px;
      line-height: 1.6;
    }
    h1 {
      font-size: 2.2rem;
      margin-bottom: 0.25rem;
    }
    .subtitle {
      font-size: 0.95rem;
      color: #9ca3af;
      margin-bottom: 1.5rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 24px;
      align-items: flex-start;
    }
    @media (max-width: 768px) {
      .grid {
        grid-template-columns: 1fr;
        gap: 16px;
      }
      body {
        font-size: 16px;
        padding: 16px;
      }
      h1 {
        font-size: 1.8rem;
      }
      .card {
        padding: 16px 18px 18px;
      }
      button {
        font-size: 0.9rem;
        padding: 10px 14px;
      }
    }
    .card {
      background: #020617;
      border-radius: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.6);
      padding: 18px 20px 20px;
      border: 1px solid #1f2937;
    }
    .card h2 {
      font-size: 1.3rem;
      margin: 0 0 0.75rem;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      font-size: 1rem;
      margin: 3px 0;
    }
    .stat-label { color: #9ca3af; }
    .stat-value { font-weight: 600; }
    .big-number {
      font-size: 2.0rem;
      font-weight: 700;
    }
    button {
      background: #3b82f6;
      color: white;
      border: none;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      margin: 5px;
      width: calc(100% - 10px);
      text-align: left;
      transition: background 0.2s;
    }
    button:hover {
      background: #2563eb;
    }
    button:disabled {
      background: #374151;
      cursor: not-allowed;
    }
    a {
      color: #60a5fa;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
      margin-top: 0.5rem;
    }
    th, td {
      padding: 4px 6px;
      border-bottom: 1px solid #1f2937;
    }
    th {
      text-align: left;
      color: #9ca3af;
      font-weight: 500;
    }
    .footer {
      margin-top: 24px;
      font-size: 0.85rem;
      color: #6b7280;
    }
    .status {
      font-size: 0.9rem;
      margin-top: 8px;
    }
    .success { color: #4ade80; }
    .error { color: #f87171; }
    .info { color: #60a5fa; }
  </style>
</head>
<body>
  <h1>Diachronic Corpus – Windsurf Web Control Panel</h1>
  <div class="subtitle">Live control center for autonomous diachronic linguistics platform. Last updated: <span id="last-update">{{ last_update }}</span></div>

  <div class="grid">
    <div class="card">
      <h2>Corpus Overview</h2>
      <div class="stat-row"><span class="stat-label">Total Texts</span><span class="stat-value" id="total-texts">{{ stats.total_texts }}</span></div>
      <div class="stat-row"><span class="stat-label">Total Words</span><span class="stat-value" id="total-words">{{ stats.total_words }}</span></div>
      <div class="stat-row"><span class="stat-label">Avg. Annotation</span><span class="stat-value" id="avg-annotation">{{ "%.1f"|format(stats.avg_annotation) }}%</span></div>
      <table id="lang-table">
        <thead><tr><th>Language</th><th>Texts</th></tr></thead>
        <tbody>
          {% for lang, count in stats.languages.items() %}
          <tr><td>{{ lang }}</td><td>{{ count }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Autonomous Operations</h2>
      <p>Run these to start the platform's autonomous workflows:</p>
      <button onclick="runCommand('python3 run_professional_cycle.py')" id="btn-professional-cycle">🚀 Run Full Professional Cycle</button>
      <button onclick="runCommand('python3 autonomous_night_operation.py')" id="btn-night-supervisor">🌙 Start Night Supervisor</button>
      <button onclick="runCommand('python3 autonomous_247_collection.py --languages grc lat en --texts-per-cycle 10 --cycle-delay 300')" id="btn-collection">📚 Start 24/7 Collection</button>
      <button onclick="runCommand('python3 annotation_worker_247.py')" id="btn-annotation">🧠 Start Annotation Worker</button>
      <button onclick="runCommand('python3 professional_dashboard.py')" id="btn-dashboard">📊 Start HTTP Dashboard</button>
      <div class="status" id="command-status"></div>
    </div>

    <div class="card">
      <h2>Visualization & Reports</h2>
      <p>Generate and view dashboards:</p>
      <button onclick="runCommand('python3 generate_quality_control_dashboard.py')" id="btn-qc-dashboard">🎛️ Generate QC Dashboard</button>
      <button onclick="runCommand('python3 generate_annotation_visualizations.py')" id="btn-charts-dashboard">📈 Generate Charts Dashboard</button>
      <button onclick="runCommand('python3 diachronic_research_agent.py')" id="btn-research-agent">🤖 Generate Research Agent Report</button>
      <button onclick="runCommand('python3 windsurf_control_panel.py')" id="btn-static-panel">📄 Generate Static Control Panel</button>
      <h3>Recent QC Dashboards</h3>
      <ul id="qc-reports">
        {% for report in reports.visual_reports %}
        <li><a href="/static/visual_reports/{{ report }}" target="_blank">{{ report }}</a></li>
        {% endfor %}
      </ul>
      <h3>Recent Agent Reports</h3>
      <ul id="agent-reports">
        {% for report in reports.agent_reports %}
        <li><a href="/static/agent_reports/{{ report }}" target="_blank">{{ report }}</a></li>
        {% endfor %}
      </ul>
      <h3>Recent Night Reports</h3>
      <ul id="night-reports">
        {% for report in reports.night_reports %}
        <li><a href="/static/night_reports/{{ report }}" target="_blank">{{ report }}</a></li>
        {% endfor %}
      </ul>
    </div>

    <div class="card">
      <h2>Exports & Integrations</h2>
      <p>Export corpus to external tools:</p>
      <button onclick="runCommand('python3 export_for_corpusexplorer.py')" id="btn-corpusexplorer">📊 Export to CorpusExplorer</button>
      <button onclick="runCommand('python3 export_for_txm.py')" id="btn-txm">📖 Export to TXM (TEI)</button>
      <button onclick="runCommand('python3 export_for_huggingface.py')" id="btn-huggingface">🤗 Export to Hugging Face Format</button>
      <button onclick="runCommand('python3 export_for_obsidian.py')" id="btn-obsidian">📝 Export to Obsidian Markdown</button>
      <button onclick="runCommand('python3 run_multi_ai_on_latest.py')" id="btn-multi-ai">🤖 Run Multi-AI Annotation</button>
      <button onclick="runCommand('python3 evaluate_metadata_and_annotations.py')" id="btn-evaluation">📊 Run Evaluation</button>
    </div>

    <div class="card">
      <h2>Advanced Features</h2>
      <p>Contrastive linguistics and valency analysis:</p>
      <button onclick="runCommand('python3 advanced_valency_analysis.py')" id="btn-valency">🔬 Run Advanced Valency Analysis</button>
      <button onclick="runCommand('python3 api_discovery_collector.py --languages grc lat en --periods all')" id="btn-api-discovery">🌐 API Discovery (New Sources)</button>
      <button onclick="runCommand('python3 lightside_integration.py')" id="btn-lightside">💡 LightSide Integration</button>
      <button onclick="runCommand('python3 test_lightside.py')" id="btn-test-lightside">🧪 Test LightSide Tools</button>
      <p>Self-improving agent:</p>
      <button onclick="runCommand('python3 diachronic_research_agent.py --ml')" id="btn-ml-agent">🧠 ML-Powered Agent Suggestions</button>
    </div>

    <div class="card">
      <h2>Utilities</h2>
      <p>Maintenance and diagnostics:</p>
      <button onclick="runCommand('python3 preprocess_texts.py')" id="btn-preprocess">🔧 Run Text Preprocessing</button>
      <button onclick="runCommand('python3 check_results.py')" id="btn-check-results">✅ Check System Status</button>
      <button onclick="runCommand('python3 generate_daily_summary.py')" id="btn-daily-summary">📋 Generate Daily Summary</button>
      <button onclick="runCommand('python3 diagnose_and_fix.py')" id="btn-diagnose">🔍 Diagnose & Fix Issues</button>
      <button onclick="runCommand('python3 import_external_corpus.py')" id="btn-import">📥 Import External Corpus</button>
    </div>

    <div class="card">
      <h2>Platform Status</h2>
      <p>Current state and options:</p>
      <div class="stat-row"><span class="stat-label">VM Location</span><span class="stat-value">OVH Public Cloud</span></div>
      <div class="stat-row"><span class="stat-label">Focus Languages</span><span class="stat-value">Greek, Latin, English</span></div>
      <div class="stat-row"><span class="stat-label">Annotation Priority</span><span class="stat-value">Greek & Latin First</span></div>
      <div class="stat-row"><span class="stat-label">Railway Worker</span><span class="stat-value">Optional (Paused)</span></div>
      <div class="stat-row"><span class="stat-label">Night Cycle End</span><span class="stat-value">08:00 Greece Time</span></div>
    </div>
  </div>

  <div class="footer">
    This live control panel allows remote command execution and real-time updates.
    Built with open-source tools (Flask, SocketIO, Python, SQLite, Stanza, PROIEL).
  </div>

  <script>
    const socket = io();

    socket.on('update_stats', function(data) {
      document.getElementById('total-texts').textContent = data.total_texts;
      document.getElementById('total-words').textContent = data.total_words;
      document.getElementById('avg-annotation').textContent = data.avg_annotation.toFixed(1) + '%';
      document.getElementById('last-update').textContent = new Date().toLocaleString();
      
      const langTable = document.getElementById('lang-table').querySelector('tbody');
      langTable.innerHTML = '';
      Object.entries(data.languages).forEach(([lang, count]) => {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${lang}</td><td>${count}</td>`;
        langTable.appendChild(row);
      });
    });

    socket.on('update_reports', function(data) {
      ['qc-reports', 'agent-reports', 'night-reports'].forEach(id => {
        const list = document.getElementById(id);
        list.innerHTML = '';
        const reports = data[id.replace('-reports', '_reports')];
        reports.forEach(report => {
          const li = document.createElement('li');
          li.innerHTML = `<a href="/static/${id.replace('-reports', '_reports')}/${report}" target="_blank">${report}</a>`;
          list.appendChild(li);
        });
      });
    });

    socket.on('command_status', function(data) {
      const statusDiv = document.getElementById('command-status');
      statusDiv.textContent = data.message;
      statusDiv.className = 'status ' + data.type;
      
      // Re-enable button
      if (data.button_id) {
        document.getElementById(data.button_id).disabled = false;
      }
    });

    function runCommand(cmd) {
      const button = event.target;
      button.disabled = true;
      
      fetch('/run_command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({command: cmd, button_id: button.id})
      })
      .then(response => response.json())
      .then(data => {
        if (!data.success) {
          button.disabled = false;
          alert('Command failed: ' + data.error);
        }
      })
      .catch(error => {
        button.disabled = false;
        alert('Network error: ' + error);
      });
    }

    // Periodic stats update
    setInterval(() => {
      fetch('/get_stats')
        .then(response => response.json())
        .then(data => socket.emit('update_stats', data));
    }, 30000); // Every 30 seconds
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, stats=current_stats, reports=current_reports, last_update=last_update)

@app.route('/run_command', methods=['POST'])
def run_command():
    data = request.json
    cmd = data.get('command', '')
    button_id = data.get('button_id', '')

    if not cmd:
        return jsonify({'success': False, 'error': 'No command provided'})

    try:
        # Run command in background
        def run_async():
            socketio.emit('command_status', {'message': f'Running: {cmd}', 'type': 'info', 'button_id': button_id})
            result = subprocess.run(cmd.split(), cwd=str(ROOT), capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                socketio.emit('command_status', {'message': f'✓ Command completed: {cmd}', 'type': 'success', 'button_id': button_id})
            else:
                socketio.emit('command_status', {'message': f'✗ Command failed: {result.stderr}', 'type': 'error', 'button_id': button_id})

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stats')
def get_stats():
    return jsonify(get_corpus_stats())

@app.route('/static/<path:filename>')
def static_files(filename):
    return app.send_static_file(filename)

@socketio.on('connect')
def handle_connect():
    emit('update_stats', get_corpus_stats())
    emit('update_reports', get_recent_reports())

def update_data():
    """Background thread to update global data periodically."""
    global current_stats, current_reports, last_update
    while True:
        try:
            current_stats = get_corpus_stats()
            current_reports = get_recent_reports()
            last_update = datetime.now(pytz.timezone('Europe/Athens')).strftime('%Y-%m-%d %H:%M:%S %Z')
            socketio.emit('update_stats', current_stats)
            socketio.emit('update_reports', current_reports)
        except Exception as e:
            logger.error(f"Error updating data: {e}")
        time.sleep(30)  # Update every 30 seconds

def get_corpus_stats() -> Dict:
    """Get quick corpus statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(word_count), AVG(annotation_score) FROM corpus_items")
        total_texts, total_words, avg_annotation = cur.fetchone()
        cur.execute("SELECT language, COUNT(*) FROM corpus_items GROUP BY language")
        langs = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        return {
            "total_texts": total_texts or 0,
            "total_words": total_words or 0,
            "avg_annotation": avg_annotation or 0,
            "languages": langs,
        }
    except Exception as e:
        logger.warning("Could not get corpus stats: %s", e)
        return {"total_texts": 0, "total_words": 0, "avg_annotation": 0, "languages": {}}

def get_recent_reports() -> Dict[str, List[str]]:
    """Get list of recent reports."""
    reports = {}
    for subdir, glob_pattern in [
        ("visual_reports", "quality_control_dashboard_*.html"),
        ("agent_reports", "agent_report_*.md"),
        ("night_reports", "cycle_*.txt"),
    ]:
        dir_path = ROOT / "research_exports" / subdir
        if dir_path.exists():
            files = sorted(dir_path.glob(glob_pattern), reverse=True)[:3]
            reports[subdir] = [f.name for f in files]
        else:
            reports[subdir] = []
    return reports

if __name__ == '__main__':
    # Start background update thread
    update_thread = threading.Thread(target=update_data)
    update_thread.daemon = True
    update_thread.start()

    # Initialize data
    current_stats = get_corpus_stats()
    current_reports = get_recent_reports()
    last_update = datetime.now(pytz.timezone('Europe/Athens')).strftime('%Y-%m-%d %H:%M:%S %Z')

    logger.info("Starting Windsurf Web Control Panel on http://0.0.0.0:8000")
    socketio.run(app, host='0.0.0.0', port=8000, debug=False)
