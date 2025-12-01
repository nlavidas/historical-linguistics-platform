#!/usr/bin/env python3
"""
PERFECT BROWSER-STYLE MONITORING DASHBOARD
==========================================
Modern, responsive monitoring interface for the Historical Linguistics Platform
Features real-time updates, beautiful visualizations, and comprehensive control

Integrates with all community-driven AIs:
- Stanza (Stanford NLP)
- spaCy (Industrial NLP)
- Hugging Face Transformers
- Ollama (Local LLMs)
- NLTK, TextBlob, Polyglot
- UDPipe, Trankit, BERT
- GPT-J, GPT-Neo, LLaMA
- And many more...

Author: Nikolaos Lavidas
Institution: National and Kapodistrian University of Athens (NKUA)
Version: 2.0.0
Date: December 1, 2025
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time

# Web framework
from flask import Flask, render_template_string, jsonify, request, Response
from flask_socketio import SocketIO, emit

# Data processing
import pandas as pd
import numpy as np

# System monitoring
import psutil
import subprocess

# AI/ML imports
try:
    import stanza
    import spacy
    import transformers
    import torch
    from transformers import pipeline
    import nltk
    import textblob
    from textblob import TextBlob
    import polyglot
    from polyglot.detect import Detector
    import udpipe
    import trankit
    import ollama
    from ollama import Client as OllamaClient
except ImportError as e:
    logging.warning(f"Some AI libraries not available: {e}")

# Local imports
sys.path.append(str(Path(__file__).parent))
from unified_web_panel import panel, automation_jobs
from multi_ai_annotator import MultiAIAnnotator
from lightside_integration import LightSidePlatformIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perfect_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'perfect_monitoring_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
monitoring_data = {
    'system': {},
    'ai_models': {},
    'automation': [],
    'alerts': deque(maxlen=100),
    'performance': deque(maxlen=1000),
    'ai_usage': {}
}

@dataclass
class AIModel:
    """AI model information and status"""
    name: str
    type: str
    status: str
    description: str
    last_used: Optional[datetime] = None
    usage_count: int = 0
    performance_score: float = 0.0

class PerfectMonitor:
    """Ultimate monitoring system for all AIs and platform"""

    def __init__(self):
        self.ai_models = {}
        self.performance_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'disk_usage': 95.0,
            'response_time': 2.0,
            'error_rate': 5.0
        }
        self.init_ai_models()
        self.start_monitoring()

    def init_ai_models(self):
        """Initialize and detect all available AI models"""

        # Community-driven AIs
        ai_configs = [
            ('stanza', 'NLP', 'Stanford CoreNLP in Python'),
            ('spacy', 'NLP', 'Industrial-strength NLP'),
            ('transformers', 'LLM', 'Hugging Face Transformers'),
            ('torch', 'Deep Learning', 'PyTorch framework'),
            ('nltk', 'NLP', 'Natural Language Toolkit'),
            ('textblob', 'Sentiment', 'Simple NLP library'),
            ('polyglot', 'Multilingual', 'Polyglot NLP library'),
            ('udpipe', 'Parsing', 'UDPipe neural models'),
            ('trankit', 'Parsing', 'Trankit multilingual pipeline'),
            ('ollama', 'LLM', 'Local LLM server'),
        ]

        for module_name, model_type, description in ai_configs:
            try:
                __import__(module_name)
                status = 'available'
                logger.info(f"✓ {module_name} available")
            except ImportError:
                status = 'not_available'
                logger.warning(f"✗ {module_name} not available")

            self.ai_models[module_name] = AIModel(
                name=module_name,
                type=model_type,
                status=status,
                description=description
            )

        # Special handling for Ollama
        try:
            ollama_client = OllamaClient()
            models = ollama_client.list()
            if models:
                self.ai_models['ollama_models'] = AIModel(
                    name='Ollama Models',
                    type='LLM',
                    status='available',
                    description=f'{len(models["models"])} local LLMs available'
                )
        except:
            pass

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            return {
                'cpu_usage': cpu,
                'memory_usage': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_usage': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'network_sent_mb': round(network.bytes_sent / (1024**2), 2),
                'network_recv_mb': round(network.bytes_recv / (1024**2), 2),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            return {}

    def get_ai_status(self) -> Dict[str, Any]:
        """Get status of all AI models"""
        status = {}
        for name, model in self.ai_models.items():
            status[name] = {
                'name': model.name,
                'type': model.type,
                'status': model.status,
                'description': model.description,
                'last_used': model.last_used.isoformat() if model.last_used else None,
                'usage_count': model.usage_count,
                'performance_score': model.performance_score
            }
        return status

    def get_automation_status(self) -> List[Dict]:
        """Get automation job status"""
        return [job for job in automation_jobs]

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []

        if metrics.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'level': 'warning',
                'message': f'High CPU usage: {metrics["cpu_usage"]:.1f}%',
                'timestamp': datetime.now().isoformat()
            })

        if metrics.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
            alerts.append({
                'level': 'critical',
                'message': f'High memory usage: {metrics["memory_usage"]:.1f}%',
                'timestamp': datetime.now().isoformat()
            })

        if metrics.get('disk_usage', 0) > self.alert_thresholds['disk_usage']:
            alerts.append({
                'level': 'critical',
                'message': f'High disk usage: {metrics["disk_usage"]:.1f}%',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while True:
                try:
                    # Collect metrics
                    system_metrics = self.get_system_metrics()
                    ai_status = self.get_ai_status()
                    automation_status = self.get_automation_status()

                    # Check for alerts
                    alerts = self.check_alerts(system_metrics)

                    # Update global state
                    monitoring_data['system'] = system_metrics
                    monitoring_data['ai_models'] = ai_status
                    monitoring_data['automation'] = automation_status
                    monitoring_data['alerts'].extend(alerts)

                    # Store performance history
                    monitoring_data['performance'].append({
                        'timestamp': datetime.now().isoformat(),
                        'cpu': system_metrics.get('cpu_usage', 0),
                        'memory': system_metrics.get('memory_usage', 0),
                        'disk': system_metrics.get('disk_usage', 0)
                    })

                    # Emit to WebSocket clients
                    socketio.emit('metrics_update', {
                        'system': system_metrics,
                        'ai_status': ai_status,
                        'automation': automation_status,
                        'alerts': list(monitoring_data['alerts'])[-10:]  # Last 10 alerts
                    })

                    time.sleep(5)  # Update every 5 seconds

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(10)  # Wait longer on error

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Perfect monitoring started")

# Initialize the perfect monitor
monitor = PerfectMonitor()

# HTML Template for the Perfect Dashboard
PERFECT_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perfect AI Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .glow { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); }
        .ai-model-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .alert-critical { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }
        .alert-warning { background: linear-gradient(135deg, #ffd93d 0%, #ff8c00 100%); }
        .chart-container { position: relative; height: 300px; }
        .terminal { background: #1a202c; color: #e2e8f0; font-family: 'Courier New', monospace; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent mb-2">
                <i class="fas fa-brain mr-2"></i>Perfect AI Monitoring Dashboard
            </h1>
            <p class="text-gray-400">Real-time monitoring of all community-driven AIs and platform health</p>
            <div class="mt-4 flex justify-center space-x-4">
                <div class="bg-green-600 px-3 py-1 rounded-full text-sm flex items-center">
                    <i class="fas fa-circle text-xs mr-2"></i>System Online
                </div>
                <div class="bg-blue-600 px-3 py-1 rounded-full text-sm flex items-center">
                    <i class="fas fa-robot text-xs mr-2"></i>{{ ai_count }} AIs Active
                </div>
                <div class="bg-purple-600 px-3 py-1 rounded-full text-sm flex items-center">
                    <i class="fas fa-cog text-xs mr-2"></i>{{ automation_count }} Jobs Running
                </div>
            </div>
        </div>

        <!-- Alerts Section -->
        <div id="alerts-container" class="mb-8">
            <!-- Alerts will be dynamically inserted here -->
        </div>

        <!-- Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- System Metrics -->
            <div class="metric-card bg-gray-800 rounded-lg p-6 glow">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-blue-400">CPU Usage</h3>
                    <i class="fas fa-microchip text-2xl text-blue-400"></i>
                </div>
                <div class="text-3xl font-bold text-white mb-2" id="cpu-usage">--</div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                    <div class="bg-blue-600 h-2 rounded-full transition-all duration-500" id="cpu-bar" style="width: 0%"></div>
                </div>
            </div>

            <div class="metric-card bg-gray-800 rounded-lg p-6 glow">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-green-400">Memory</h3>
                    <i class="fas fa-memory text-2xl text-green-400"></i>
                </div>
                <div class="text-3xl font-bold text-white mb-2" id="memory-usage">--</div>
                <div class="text-sm text-gray-400" id="memory-details">-- / -- GB</div>
                <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
                    <div class="bg-green-600 h-2 rounded-full transition-all duration-500" id="memory-bar" style="width: 0%"></div>
                </div>
            </div>

            <div class="metric-card bg-gray-800 rounded-lg p-6 glow">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-purple-400">Disk Usage</h3>
                    <i class="fas fa-hdd text-2xl text-purple-400"></i>
                </div>
                <div class="text-3xl font-bold text-white mb-2" id="disk-usage">--</div>
                <div class="text-sm text-gray-400" id="disk-details">-- / -- GB</div>
                <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
                    <div class="bg-purple-600 h-2 rounded-full transition-all duration-500" id="disk-bar" style="width: 0%"></div>
                </div>
            </div>

            <div class="metric-card bg-gray-800 rounded-lg p-6 glow">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-semibold text-yellow-400">Network</h3>
                    <i class="fas fa-network-wired text-2xl text-yellow-400"></i>
                </div>
                <div class="text-lg font-bold text-white mb-1" id="network-sent">-- MB</div>
                <div class="text-sm text-gray-400 mb-2">Sent</div>
                <div class="text-lg font-bold text-white mb-1" id="network-recv">-- MB</div>
                <div class="text-sm text-gray-400">Received</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Performance Chart -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4 text-white">System Performance</h3>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>

            <!-- AI Models Status -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4 text-white">AI Models Status</h3>
                <div id="ai-models-container" class="space-y-3 max-h-64 overflow-y-auto">
                    <!-- AI models will be dynamically inserted here -->
                </div>
            </div>
        </div>

        <!-- Automation & Logs -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Automation Jobs -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4 text-white">Automation Jobs</h3>
                <div id="automation-container" class="space-y-2 max-h-64 overflow-y-auto">
                    <!-- Automation jobs will be dynamically inserted here -->
                </div>
            </div>

            <!-- Live Logs -->
            <div class="bg-gray-800 rounded-lg p-6">
                <h3 class="text-xl font-semibold mb-4 text-white">Live System Logs</h3>
                <div class="terminal rounded p-4 max-h-64 overflow-y-auto" id="logs-container">
                    <!-- Logs will be dynamically inserted here -->
                </div>
            </div>
        </div>

        <!-- Control Panel -->
        <div class="bg-gray-800 rounded-lg p-6">
            <h3 class="text-xl font-semibold mb-4 text-white">Control Panel</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button onclick="runAutomation('lightside')" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded transition-colors">
                    <i class="fas fa-brain mr-2"></i>Run LightSide Training
                </button>
                <button onclick="runAutomation('transformer')" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded transition-colors">
                    <i class="fas fa-cogs mr-2"></i>Run Transformer Annotation
                </button>
                <button onclick="runAutomation('export')" class="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded transition-colors">
                    <i class="fas fa-upload mr-2"></i>Run HF Export
                </button>
            </div>
            <div class="mt-4">
                <button onclick="clearAlerts()" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded transition-colors">
                    <i class="fas fa-trash mr-2"></i>Clear All Alerts
                </button>
                <button onclick="exportMetrics()" class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded ml-2 transition-colors">
                    <i class="fas fa-download mr-2"></i>Export Metrics
                </button>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let performanceChart;

        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(34, 197, 94)',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Disk %',
                        data: [],
                        borderColor: 'rgb(147, 51, 234)',
                        backgroundColor: 'rgba(147, 51, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }

        // Update metrics display
        function updateMetrics(data) {
            if (data.system) {
                // Update system metrics
                document.getElementById('cpu-usage').textContent = data.system.cpu_usage.toFixed(1) + '%';
                document.getElementById('cpu-bar').style.width = data.system.cpu_usage + '%';

                document.getElementById('memory-usage').textContent = data.system.memory_usage.toFixed(1) + '%';
                document.getElementById('memory-details').textContent = 
                    `${data.system.memory_used_gb} / ${data.system.memory_total_gb} GB`;
                document.getElementById('memory-bar').style.width = data.system.memory_usage + '%';

                document.getElementById('disk-usage').textContent = data.system.disk_usage.toFixed(1) + '%';
                document.getElementById('disk-details').textContent = 
                    `${data.system.disk_used_gb} / ${data.system.disk_total_gb} GB`;
                document.getElementById('disk-bar').style.width = data.system.disk_usage + '%';

                document.getElementById('network-sent').textContent = data.system.network_sent_mb + ' MB';
                document.getElementById('network-recv').textContent = data.system.network_recv_mb + ' MB';

                // Update performance chart
                const now = new Date().toLocaleTimeString();
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets.forEach(dataset => dataset.data.shift());
                }
                performanceChart.data.labels.push(now);
                performanceChart.data.datasets[0].data.push(data.system.cpu_usage);
                performanceChart.data.datasets[1].data.push(data.system.memory_usage);
                performanceChart.data.datasets[2].data.push(data.system.disk_usage);
                performanceChart.update();
            }

            if (data.ai_status) {
                updateAIModels(data.ai_status);
            }

            if (data.automation) {
                updateAutomation(data.automation);
            }

            if (data.alerts) {
                updateAlerts(data.alerts);
            }
        }

        // Update AI models display
        function updateAIModels(aiStatus) {
            const container = document.getElementById('ai-models-container');
            container.innerHTML = '';

            Object.values(aiStatus).forEach(model => {
                const statusColor = model.status === 'available' ? 'text-green-400' : 'text-red-400';
                const statusIcon = model.status === 'available' ? 'fas fa-check-circle' : 'fas fa-times-circle';

                const modelDiv = document.createElement('div');
                modelDiv.className = 'ai-model-card rounded p-3 flex justify-between items-center';
                modelDiv.innerHTML = `
                    <div>
                        <div class="font-semibold">${model.name}</div>
                        <div class="text-sm opacity-75">${model.type} - ${model.description}</div>
                    </div>
                    <div class="${statusColor} flex items-center">
                        <i class="${statusIcon} mr-2"></i>
                        ${model.status.replace('_', ' ')}
                    </div>
                `;
                container.appendChild(modelDiv);
            });
        }

        // Update automation jobs display
        function updateAutomation(jobs) {
            const container = document.getElementById('automation-container');
            container.innerHTML = '';

            if (jobs.length === 0) {
                container.innerHTML = '<div class="text-gray-400">No active automation jobs</div>';
                return;
            }

            jobs.slice(-10).forEach(job => {
                const statusColor = {
                    'running': 'text-blue-400',
                    'success': 'text-green-400',
                    'error': 'text-red-400'
                }[job.status] || 'text-gray-400';

                const jobDiv = document.createElement('div');
                jobDiv.className = 'bg-gray-700 rounded p-3';
                jobDiv.innerHTML = `
                    <div class="flex justify-between items-start">
                        <div>
                            <div class="font-semibold ${statusColor}">${job.status.toUpperCase()}</div>
                            <div class="text-sm text-gray-300">${job.message}</div>
                            <div class="text-xs text-gray-500">${job.category} • ${job.timestamp}</div>
                        </div>
                        <div class="text-xs ${statusColor}">
                            <i class="fas fa-clock mr-1"></i>${job.timestamp.split(' ')[1]}
                        </div>
                    </div>
                `;
                container.appendChild(jobDiv);
            });
        }

        // Update alerts display
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';

            if (alerts.length === 0) return;

            alerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert-${alert.level} text-white p-4 rounded-lg mb-4 flex items-center`;
                alertDiv.innerHTML = `
                    <i class="fas fa-exclamation-triangle mr-3"></i>
                    <div>
                        <div class="font-semibold">${alert.level.toUpperCase()}</div>
                        <div>${alert.message}</div>
                        <div class="text-sm opacity-75">${alert.timestamp}</div>
                    </div>
                `;
                container.appendChild(alertDiv);
            });
        }

        // Control panel functions
        function runAutomation(type) {
            fetch(`/api/automation/${type}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Automation started:', data);
                    showNotification('Automation job started successfully!', 'success');
                })
                .catch(error => {
                    console.error('Automation failed:', error);
                    showNotification('Failed to start automation job', 'error');
                });
        }

        function clearAlerts() {
            fetch('/api/alerts/clear', { method: 'POST' })
                .then(() => showNotification('Alerts cleared', 'success'))
                .catch(() => showNotification('Failed to clear alerts', 'error'));
        }

        function exportMetrics() {
            fetch('/api/metrics/export')
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `metrics_${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    window.URL.revokeObjectURL(url);
                    showNotification('Metrics exported successfully!', 'success');
                })
                .catch(() => showNotification('Failed to export metrics', 'error'));
        }

        function showNotification(message, type) {
            // Simple notification - you can enhance this
            console.log(`${type.toUpperCase()}: ${message}`);
            alert(message); // Replace with proper toast notification
        }

        // Socket.IO event handling
        socket.on('metrics_update', function(data) {
            updateMetrics(data);
        });

        socket.on('alert', function(alert) {
            updateAlerts([alert]);
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            // Initial data fetch
            fetch('/api/monitor/status')
                .then(response => response.json())
                .then(data => updateMetrics(data));
        });
    </script>
</body>
</html>
"""

# API Routes
@app.route('/perfect-monitor')
def perfect_monitor():
    """Serve the perfect monitoring dashboard"""
    ai_count = len([m for m in monitor.ai_models.values() if m.status == 'available'])
    automation_count = len([j for j in automation_jobs if j.get('status') == 'running'])

    return render_template_string(
        PERFECT_DASHBOARD_HTML,
        ai_count=ai_count,
        automation_count=automation_count
    )

@app.route('/api/monitor/status')
def get_monitor_status():
    """Get current monitoring status"""
    return jsonify({
        'system': monitoring_data['system'],
        'ai_status': monitoring_data['ai_models'],
        'automation': monitoring_data['automation'],
        'alerts': list(monitoring_data['alerts'])[-10:]
    })

@app.route('/api/automation/<job_type>', methods=['POST'])
def run_automation_job(job_type):
    """Run automation job"""
    try:
        if job_type == 'lightside':
            panel.run_lightside_training()
        elif job_type == 'transformer':
            panel.run_transformer_annotation()
        elif job_type == 'export':
            panel.run_hf_export()
        else:
            return jsonify({'error': 'Unknown job type'}), 400

        return jsonify({'status': 'success', 'message': f'{job_type} job started'})
    except Exception as e:
        logger.error(f"Automation job failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    monitoring_data['alerts'].clear()
    return jsonify({'status': 'success'})

@app.route('/api/metrics/export')
def export_metrics():
    """Export metrics as JSON"""
    return jsonify({
        'system_history': list(monitoring_data['performance']),
        'ai_usage': monitoring_data['ai_usage'],
        'alerts': list(monitoring_data['alerts']),
        'exported_at': datetime.now().isoformat()
    })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to monitoring dashboard")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from monitoring dashboard")

if __name__ == '__main__':
    logger.info("Starting Perfect AI Monitoring Dashboard...")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
