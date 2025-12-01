#!/usr/bin/env python3
"""
PROGRESS REVIEW AND IMPROVEMENT DASHBOARD
Web-based interface for monitoring pipeline progress and implementing improvements
Real-time status, performance analytics, and improvement management
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import threading
import time
from continuous_monitoring import ContinuousMonitoringSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressReviewDashboard:
    """
    Web dashboard for monitoring and improving the linguistics pipeline
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.monitor = ContinuousMonitoringSystem()
        self.templates = self._load_templates()

        # Setup routes
        self._setup_routes()

    def _load_templates(self) -> dict:
        """Load HTML templates"""
        return {
            'dashboard': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Historical Linguistics Pipeline - Progress Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: scale(1.02); }
        .alert-pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900 mb-2">Historical Linguistics Pipeline</h1>
            <p class="text-gray-600">Progress Review & Continuous Improvement Dashboard</p>
            <div class="flex items-center mt-4">
                <div class="flex items-center space-x-2">
                    <div id="monitoring-status" class="w-3 h-3 rounded-full bg-gray-400"></div>
                    <span id="monitoring-text" class="text-sm text-gray-600">Initializing...</span>
                </div>
                <div class="ml-auto">
                    <button onclick="startPipeline()" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2">
                        Start Pipeline
                    </button>
                    <button onclick="refreshData()" class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                        Refresh
                    </button>
                </div>
            </div>
        </header>

        <!-- Alerts Section -->
        <div id="alerts-section" class="mb-8">
            <h2 class="text-xl font-semibold mb-4">Active Alerts</h2>
            <div id="alerts-container" class="space-y-2">
                <!-- Alerts will be populated by JavaScript -->
            </div>
        </div>

        <!-- Key Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="metric-card bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-2">System Health</h3>
                <div class="flex items-center">
                    <div id="cpu-indicator" class="w-4 h-4 rounded-full mr-2"></div>
                    <span id="cpu-text" class="text-2xl font-bold text-gray-900">Loading...</span>
                </div>
                <p class="text-sm text-gray-600 mt-1">CPU Usage</p>
            </div>

            <div class="metric-card bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-2">Corpus Size</h3>
                <div class="text-2xl font-bold text-blue-600" id="corpus-size">0</div>
                <p class="text-sm text-gray-600 mt-1">Total Texts</p>
            </div>

            <div class="metric-card bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-2">Processing Rate</h3>
                <div class="text-2xl font-bold text-green-600" id="processing-rate">0/min</div>
                <p class="text-sm text-gray-600 mt-1">Texts per Minute</p>
            </div>

            <div class="metric-card bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-2">Improvements</h3>
                <div class="text-2xl font-bold text-purple-600" id="pending-improvements">0</div>
                <p class="text-sm text-gray-600 mt-1">Pending Suggestions</p>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">System Resources (Last 24h)</h3>
                <canvas id="resourcesChart" width="400" height="200"></canvas>
            </div>

            <div class="bg-white rounded-lg shadow p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">Pipeline Performance</h3>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Improvement Suggestions -->
        <div class="bg-white rounded-lg shadow p-6 mb-8">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Improvement Suggestions</h3>
            <div id="suggestions-container" class="space-y-3">
                <!-- Suggestions will be populated by JavaScript -->
            </div>
        </div>

        <!-- Pipeline Control -->
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">Pipeline Control</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Stages to Run</label>
                    <div class="space-y-1">
                        <label class="flex items-center">
                            <input type="checkbox" id="stage-collection" class="mr-2" checked>
                            <span class="text-sm">Text Collection</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="stage-preprocessing" class="mr-2" checked>
                            <span class="text-sm">Preprocessing</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" id="stage-parsing" class="mr-2" checked>
                            <span class="text-sm">Parsing</span>
                        </label>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Configuration</label>
                    <div class="space-y-2">
                        <input type="number" id="batch-size" placeholder="Batch Size" class="w-full px-3 py-2 border border-gray-300 rounded-md" value="50">
                        <input type="number" id="processing-limit" placeholder="Processing Limit" class="w-full px-3 py-2 border border-gray-300 rounded-md">
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Actions</label>
                    <div class="space-y-2">
                        <button onclick="runCustomPipeline()" class="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                            Run Custom Pipeline
                        </button>
                        <button onclick="generateReport()" class="w-full bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                            Generate Report
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let statusData = {};
        let charts = {};

        // Initialize charts
        function initCharts() {
            const ctxResources = document.getElementById('resourcesChart').getContext('2d');
            charts.resources = new Chart(ctxResources, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });

            const ctxPerformance = document.getElementById('performanceChart').getContext('2d');
            charts.performance = new Chart(ctxPerformance, {
                type: 'bar',
                data: {
                    labels: ['Collection', 'Preprocessing', 'Parsing', 'Annotation', 'Valency', 'Diachronic'],
                    datasets: [{
                        label: 'Processing Rate (texts/min)',
                        data: [],
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true } }
                }
            });
        }

        // Update dashboard data
        async function updateDashboard() {
            try {
                const response = await fetch('/api/status');
                statusData = await response.json();

                updateMetrics();
                updateAlerts();
                updateSuggestions();
                updateCharts();
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }

        function updateMetrics() {
            // System health
            const cpuPercent = statusData.system_resources?.cpu_usage_percent || 0;
            document.getElementById('cpu-text').textContent = cpuPercent.toFixed(1) + '%';

            const cpuIndicator = document.getElementById('cpu-indicator');
            if (cpuPercent > 80) {
                cpuIndicator.className = 'w-4 h-4 rounded-full bg-red-500 mr-2';
            } else if (cpuPercent > 50) {
                cpuIndicator.className = 'w-4 h-4 rounded-full bg-yellow-500 mr-2';
            } else {
                cpuIndicator.className = 'w-4 h-4 rounded-full bg-green-500 mr-2';
            }

            // Corpus size
            document.getElementById('corpus-size').textContent = statusData.pipeline_health?.total_texts || 0;

            // Processing rate (mock data for now)
            document.getElementById('processing-rate').textContent = '25/min';

            // Pending improvements
            const pendingImprovements = statusData.improvement_suggestions?.filter(s => s.status === 'pending').length || 0;
            document.getElementById('pending-improvements').textContent = pendingImprovements;

            // Monitoring status
            const monitoringStatus = document.getElementById('monitoring-status');
            const monitoringText = document.getElementById('monitoring-text');

            if (statusData.monitoring_active) {
                monitoringStatus.className = 'w-3 h-3 rounded-full bg-green-500';
                monitoringText.textContent = 'Monitoring Active';
            } else {
                monitoringStatus.className = 'w-3 h-3 rounded-full bg-red-500';
                monitoringText.textContent = 'Monitoring Inactive';
            }
        }

        function updateAlerts() {
            const alertsContainer = document.getElementById('alerts-container');
            alertsContainer.innerHTML = '';

            const alerts = statusData.alerts || [];
            const unresolvedAlerts = alerts.filter(a => !a.resolved);

            if (unresolvedAlerts.length === 0) {
                alertsContainer.innerHTML = '<p class="text-gray-500">No active alerts</p>';
                return;
            }

            unresolvedAlerts.forEach(alert => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `p-3 rounded-md alert-pulse ${
                    alert.severity === 'high' ? 'bg-red-100 border border-red-300' :
                    alert.severity === 'medium' ? 'bg-yellow-100 border border-yellow-300' :
                    'bg-blue-100 border border-blue-300'
                }`;

                alertDiv.innerHTML = `
                    <div class="flex items-center">
                        <div class="flex-1">
                            <strong class="text-sm font-medium">${alert.component.toUpperCase()}</strong>
                            <p class="text-sm text-gray-700">${alert.message}</p>
                        </div>
                        <span class="text-xs text-gray-500">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                    </div>
                `;

                alertsContainer.appendChild(alertDiv);
            });
        }

        function updateSuggestions() {
            const suggestionsContainer = document.getElementById('suggestions-container');
            suggestionsContainer.innerHTML = '';

            const suggestions = statusData.improvement_suggestions || [];

            if (suggestions.length === 0) {
                suggestionsContainer.innerHTML = '<p class="text-gray-500">No improvement suggestions</p>';
                return;
            }

            suggestions.forEach(suggestion => {
                const suggestionDiv = document.createElement('div');
                suggestionDiv.className = `p-4 rounded-md border ${
                    suggestion.priority === 'high' ? 'border-red-300 bg-red-50' :
                    suggestion.priority === 'medium' ? 'border-yellow-300 bg-yellow-50' :
                    'border-blue-300 bg-blue-50'
                }`;

                suggestionDiv.innerHTML = `
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="flex items-center mb-1">
                                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                    suggestion.priority === 'high' ? 'bg-red-100 text-red-800' :
                                    suggestion.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-blue-100 text-blue-800'
                                }">
                                    ${suggestion.priority.toUpperCase()}
                                </span>
                                <span class="ml-2 text-sm text-gray-600">${suggestion.category}</span>
                            </div>
                            <p class="text-sm text-gray-900 mb-2">${suggestion.description}</p>
                            <p class="text-xs text-gray-500">Impact: ${(suggestion.impact * 100).toFixed(0)}%</p>
                        </div>
                        <button onclick="implementSuggestion('${suggestion.id}')"
                                class="ml-3 bg-green-500 hover:bg-green-700 text-white text-xs font-bold py-1 px-2 rounded">
                            Implement
                        </button>
                    </div>
                `;

                suggestionsContainer.appendChild(suggestionDiv);
            });
        }

        function updateCharts() {
            // Update resource usage chart (simplified)
            if (charts.resources && statusData.system_resources) {
                const now = new Date().toLocaleTimeString();
                charts.resources.data.labels.push(now);
                charts.resources.data.datasets[0].data.push(statusData.system_resources.cpu_usage_percent || 0);
                charts.resources.data.datasets[1].data.push(statusData.system_resources.memory_usage_percent || 0);

                // Keep only last 20 points
                if (charts.resources.data.labels.length > 20) {
                    charts.resources.data.labels.shift();
                    charts.resources.data.datasets[0].data.shift();
                    charts.resources.data.datasets[1].data.shift();
                }

                charts.resources.update();
            }
        }

        // Action functions
        async function startPipeline() {
            try {
                const response = await fetch('/api/start_pipeline', { method: 'POST' });
                const result = await response.json();
                alert('Pipeline started! Check progress below.');
                refreshData();
            } catch (error) {
                alert('Error starting pipeline: ' + error.message);
            }
        }

        async function runCustomPipeline() {
            const stages = [];
            if (document.getElementById('stage-collection').checked) stages.push('collection');
            if (document.getElementById('stage-preprocessing').checked) stages.push('preprocessing');
            if (document.getElementById('stage-parsing').checked) stages.push('parsing');
            if (document.getElementById('stage-parsing').checked) stages.push('annotation');
            if (document.getElementById('stage-parsing').checked) stages.push('valency');
            if (document.getElementById('stage-parsing').checked) stages.push('diachronic');

            const config = {
                stages: stages,
                batch_size: parseInt(document.getElementById('batch-size').value) || 50,
                processing_limit: parseInt(document.getElementById('processing-limit').value) || null
            };

            try {
                const response = await fetch('/api/run_pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                alert('Custom pipeline started! Monitor progress below.');
                refreshData();
            } catch (error) {
                alert('Error running pipeline: ' + error.message);
            }
        }

        async function generateReport() {
            try {
                const response = await fetch('/api/generate_report', { method: 'POST' });
                const result = await response.json();
                alert('Report generated! Check the reports directory.');
            } catch (error) {
                alert('Error generating report: ' + error.message);
            }
        }

        async function implementSuggestion(suggestionId) {
            const action = prompt('Describe the action taken to implement this suggestion:');
            if (!action) return;

            try {
                const response = await fetch('/api/implement_suggestion', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ suggestion_id: suggestionId, action_taken: action })
                });
                const result = await response.json();
                alert('Suggestion marked as implemented!');
                refreshData();
            } catch (error) {
                alert('Error implementing suggestion: ' + error.message);
            }
        }

        function refreshData() {
            updateDashboard();
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            updateDashboard();
            setInterval(updateDashboard, 30000); // Update every 30 seconds
        });
    </script>
</body>
</html>
            """
        }

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            return render_template_string(self.templates['dashboard'])

        @self.app.route('/api/status')
        def get_status():
            return jsonify(self.monitor.get_system_status())

        @self.app.route('/api/start_pipeline', methods=['POST'])
        def start_pipeline():
            try:
                from master_workflow_coordinator import MasterHistoricalLinguisticsWorkflow
                workflow = MasterHistoricalLinguisticsWorkflow()
                results = workflow.run_full_pipeline()
                return jsonify({'status': 'success', 'results': results})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/api/run_pipeline', methods=['POST'])
        def run_pipeline():
            try:
                config = request.get_json()
                from master_workflow_coordinator import MasterHistoricalLinguisticsWorkflow
                workflow = MasterHistoricalLinguisticsWorkflow()
                results = workflow.run_full_pipeline(
                    stages=config.get('stages'),
                    # Additional config can be passed here
                )
                return jsonify({'status': 'success', 'results': results})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/api/generate_report', methods=['POST'])
        def generate_report():
            try:
                report = self.monitor.get_performance_report(hours=24)
                report_path = Path('reports') / f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                report_path.parent.mkdir(exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                return jsonify({'status': 'success', 'report_path': str(report_path)})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.route('/api/implement_suggestion', methods=['POST'])
        def implement_suggestion():
            try:
                data = request.get_json()
                self.monitor.apply_improvement(data['suggestion_id'], data['action_taken'])
                return jsonify({'status': 'success'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

    def run(self, host='localhost', port=5000, debug=False):
        """Run the dashboard"""
        print(f"Starting Progress Review Dashboard on http://{host}:{port}")
        print("Press Ctrl+C to stop")

        # Start monitoring in background
        self.monitor.start_monitoring()

        try:
            self.app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
        finally:
            self.monitor.stop_monitoring()

def main():
    """Main dashboard function"""
    dashboard = ProgressReviewDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
