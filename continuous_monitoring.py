#!/usr/bin/env python3
"""
CONTINUOUS MONITORING AND IMPROVEMENT SYSTEM
Real-time pipeline monitoring, performance analysis, and automated improvement suggestions
No demos, no placeholders - production system for ongoing research optimization
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import psutil
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import threading
import schedule
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousMonitoringSystem:
    """
    Real-time monitoring and improvement system for the linguistics pipeline
    Tracks performance, identifies bottlenecks, suggests optimizations
    """

    def __init__(self, db_path="corpus_efficient.db", monitoring_db="monitoring.db"):
        self.db_path = db_path
        self.monitoring_db = monitoring_db
        self.monitoring_active = False
        self.performance_history = defaultdict(list)
        self.alerts = []
        self.improvement_suggestions = []

        # Monitoring thresholds
        self.thresholds = {
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'disk_usage_percent': 90,
            'processing_rate_drop': 0.5,  # 50% drop triggers alert
            'error_rate_threshold': 0.1,  # 10% error rate triggers alert
        }

        # Initialize monitoring database
        self._init_monitoring_db()

        # Performance baselines (will be learned over time)
        self.baselines = self._load_baselines()

    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.monitoring_db)
        cursor = conn.cursor()

        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT,
                component TEXT,
                metric_name TEXT,
                metric_value REAL,
                context TEXT
            )
        ''')

        # Pipeline runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                stages_completed TEXT,
                total_runtime REAL,
                success_rate REAL,
                errors TEXT,
                improvements_applied TEXT
            )
        ''')

        # Improvement suggestions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS improvement_suggestions (
                suggestion_id TEXT PRIMARY KEY,
                timestamp TEXT,
                category TEXT,
                priority TEXT,
                description TEXT,
                implementation_status TEXT,
                impact_estimate REAL
            )
        ''')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT,
                severity TEXT,
                component TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    def _load_baselines(self) -> Dict[str, float]:
        """Load performance baselines from historical data"""
        baselines = {
            'text_collection_rate': 50.0,  # texts/minute
            'preprocessing_rate': 100.0,   # texts/minute
            'parsing_rate': 30.0,          # texts/minute
            'annotation_rate': 60.0,       # texts/minute
            'memory_usage_mb': 1000.0,     # MB
            'cpu_usage_percent': 50.0,
        }

        # Try to load from monitoring database
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            # Get average performance over last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()

            cursor.execute('''
                SELECT component, metric_name, AVG(metric_value)
                FROM performance_metrics
                WHERE timestamp > ?
                GROUP BY component, metric_name
            ''', (week_ago,))

            for component, metric_name, avg_value in cursor.fetchall():
                key = f"{component}_{metric_name}"
                baselines[key] = avg_value

            conn.close()

        except Exception as e:
            logger.warning(f"Could not load baselines from database: {e}")

        return baselines

    def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting continuous monitoring system")
        self.monitoring_active = True

        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()

        # Schedule periodic tasks
        schedule.every(5).minutes.do(self._analyze_performance)
        schedule.every(1).hours.do(self._generate_improvement_suggestions)
        schedule.every(24).hours.do(self._cleanup_old_data)

        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        logger.info("Stopping continuous monitoring system")
        self.monitoring_active = False

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_pipeline_health()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)

    def _scheduler_loop(self):
        """Scheduler loop for periodic tasks"""
        while self.monitoring_active:
            schedule.run_pending()
            time.sleep(60)

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now().isoformat()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._store_metric(timestamp, 'system', 'cpu_usage_percent', cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / 1024 / 1024
            self._store_metric(timestamp, 'system', 'memory_usage_percent', memory_percent)
            self._store_metric(timestamp, 'system', 'memory_usage_mb', memory_mb)

            # Disk usage for Z: drive
            disk = psutil.disk_usage('Z:')
            disk_percent = disk.percent
            self._store_metric(timestamp, 'system', 'disk_usage_percent', disk_percent)

            # Network I/O (if available)
            try:
                net = psutil.net_io_counters()
                if net:
                    bytes_sent_mb = net.bytes_sent / 1024 / 1024
                    bytes_recv_mb = net.bytes_recv / 1024 / 1024
                    self._store_metric(timestamp, 'system', 'network_sent_mb', bytes_sent_mb)
                    self._store_metric(timestamp, 'system', 'network_recv_mb', bytes_recv_mb)
            except:
                pass

            # Check thresholds and create alerts
            self._check_thresholds(timestamp, 'cpu', cpu_percent, self.thresholds['cpu_usage_percent'])
            self._check_thresholds(timestamp, 'memory', memory_percent, self.thresholds['memory_usage_percent'])
            self._check_thresholds(timestamp, 'disk', disk_percent, self.thresholds['disk_usage_percent'])

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _check_pipeline_health(self):
        """Check pipeline component health"""
        timestamp = datetime.now().isoformat()

        try:
            # Check if database is accessible
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent pipeline activity
            cursor.execute('''
                SELECT COUNT(*) as total_texts,
                       COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                       AVG(word_count) as avg_words,
                       MAX(date_added) as latest_addition
                FROM corpus_items
                WHERE date_added > datetime('now', '-1 hour')
            ''')

            result = cursor.fetchone()
            if result:
                total_texts, completed, avg_words, latest = result

                self._store_metric(timestamp, 'pipeline', 'texts_processed_hour', total_texts or 0)
                self._store_metric(timestamp, 'pipeline', 'completion_rate_hour', (completed or 0) / max(total_texts or 1, 1))
                if avg_words:
                    self._store_metric(timestamp, 'pipeline', 'avg_text_length', avg_words)

            conn.close()

        except Exception as e:
            logger.error(f"Error checking pipeline health: {e}")
            self._create_alert(timestamp, 'error', 'pipeline', f"Pipeline health check failed: {e}")

    def _store_metric(self, timestamp: str, component: str, metric_name: str, value: float, context: str = ""):
        """Store performance metric"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, component, metric_name, metric_value, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, component, metric_name, value, context))

            conn.commit()
            conn.close()

            # Keep in memory for quick access
            key = f"{component}_{metric_name}"
            self.performance_history[key].append((timestamp, value))

            # Keep only last 1000 values
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-1000:]

        except Exception as e:
            logger.error(f"Error storing metric {component}.{metric_name}: {e}")

    def _check_thresholds(self, timestamp: str, metric_type: str, current_value: float, threshold: float):
        """Check if metric exceeds threshold and create alert"""
        if current_value > threshold:
            severity = 'high' if current_value > threshold * 1.5 else 'medium'
            message = f"{metric_type.upper()} usage at {current_value:.1f}% (threshold: {threshold}%)"

            self._create_alert(timestamp, severity, 'system', message)

            # Suggest improvements based on threshold violations
            if metric_type == 'cpu':
                self._suggest_improvement('high', 'performance',
                                        f"High CPU usage detected ({current_value:.1f}%). Consider increasing batch_size or reducing parallel processing.")
            elif metric_type == 'memory':
                self._suggest_improvement('high', 'performance',
                                        f"High memory usage detected ({current_value:.1f}%). Consider reducing batch_size or implementing streaming processing.")
            elif metric_type == 'disk':
                self._suggest_improvement('high', 'storage',
                                        f"High disk usage detected ({current_value:.1f}%). Consider cleaning old logs or implementing data archiving.")

    def _create_alert(self, timestamp: str, severity: str, component: str, message: str):
        """Create an alert"""
        alert_id = f"{timestamp}_{component}_{severity}"

        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO alerts (alert_id, timestamp, severity, component, message)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_id, timestamp, severity, component, message))

            conn.commit()
            conn.close()

            # Add to in-memory alerts
            self.alerts.append({
                'id': alert_id,
                'timestamp': timestamp,
                'severity': severity,
                'component': component,
                'message': message,
                'resolved': False
            })

            logger.warning(f"ALERT [{severity.upper()}]: {component} - {message}")

        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    def _analyze_performance(self):
        """Analyze performance trends and identify issues"""
        logger.info("Analyzing performance trends...")

        try:
            # Get recent metrics (last hour)
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()

            cursor.execute('''
                SELECT component, metric_name, metric_value
                FROM performance_metrics
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (hour_ago,))

            recent_metrics = cursor.fetchall()
            conn.close()

            # Analyze trends
            component_metrics = defaultdict(list)
            for component, metric_name, value in recent_metrics:
                component_metrics[f"{component}_{metric_name}"].append(value)

            # Check for performance degradation
            for metric_key, values in component_metrics.items():
                if len(values) >= 10:  # Need some data points
                    recent_avg = np.mean(values[:5])  # Last 5 measurements
                    older_avg = np.mean(values[5:])   # Previous measurements

                    if older_avg > 0 and recent_avg < older_avg * self.thresholds['processing_rate_drop']:
                        degradation_pct = (1 - recent_avg / older_avg) * 100
                        self._create_alert(datetime.now().isoformat(), 'medium', 'performance',
                                         f"Performance degradation in {metric_key}: {degradation_pct:.1f}% drop")

                        # Suggest improvements
                        self._suggest_improvement('medium', 'performance',
                                                f"Performance degradation detected in {metric_key}. Consider checking resource usage and optimizing batch processing.")

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")

    def _generate_improvement_suggestions(self):
        """Generate improvement suggestions based on monitoring data"""
        logger.info("Generating improvement suggestions...")

        try:
            # Analyze component availability and usage
            available_tools = self._check_tool_availability()

            # Suggest based on performance patterns
            self._analyze_usage_patterns()

            # Suggest based on error patterns
            self._analyze_error_patterns()

            # Suggest optimizations based on resource usage
            self._suggest_resource_optimizations()

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")

    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check which NLP tools are available"""
        tools = {
            'nltk': False,
            'spacy': False,
            'stanza': False,
            'trankit': False,
            'udpipe': False,
            'polyglot': False,
            'textblob': False
        }

        for tool in tools.keys():
            try:
                __import__(tool)
                tools[tool] = True
            except ImportError:
                pass

        # Suggest improvements based on missing tools
        missing_tools = [tool for tool, available in tools.items() if not available]

        if missing_tools:
            self._suggest_improvement('low', 'tools',
                                    f"Consider installing missing tools for better performance: {', '.join(missing_tools)}")

        return tools

    def _analyze_usage_patterns(self):
        """Analyze usage patterns and suggest optimizations"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            # Get processing rates by component
            cursor.execute('''
                SELECT component, AVG(metric_value) as avg_rate
                FROM performance_metrics
                WHERE metric_name LIKE '%rate%' AND timestamp > datetime('now', '-24 hours')
                GROUP BY component
            ''')

            rates = dict(cursor.fetchall())
            conn.close()

            # Suggest optimizations based on rates
            if 'pipeline' in rates and rates['pipeline'] < 10:  # Very slow processing
                self._suggest_improvement('high', 'performance',
                                        "Very slow processing detected. Consider increasing batch_size, using GPU acceleration, or optimizing database queries.")

            if 'preprocessing' in rates and rates.get('preprocessing', 0) < rates.get('pipeline', 1000):
                self._suggest_improvement('medium', 'bottleneck',
                                        "Preprocessing may be bottleneck. Consider using faster tokenizers or parallel processing.")

        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")

    def _analyze_error_patterns(self):
        """Analyze error patterns and suggest fixes"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            # Get recent errors
            cursor.execute('''
                SELECT component, COUNT(*) as error_count
                FROM alerts
                WHERE severity = 'error' AND timestamp > datetime('now', '-24 hours')
                GROUP BY component
            ''')

            error_counts = dict(cursor.fetchall())
            conn.close()

            # Suggest fixes based on error patterns
            for component, count in error_counts.items():
                if count > 5:  # Multiple errors
                    self._suggest_improvement('high', 'reliability',
                                            f"High error rate in {component} ({count} errors). Check logs and consider implementing retry logic or error recovery.")

        except Exception as e:
            logger.error(f"Error analyzing error patterns: {e}")

    def _suggest_resource_optimizations(self):
        """Suggest resource usage optimizations"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            # Get average resource usage
            cursor.execute('''
                SELECT metric_name, AVG(metric_value) as avg_usage
                FROM performance_metrics
                WHERE component = 'system' AND timestamp > datetime('now', '-24 hours')
                GROUP BY metric_name
            ''')

            usages = dict(cursor.fetchall())
            conn.close()

            # Suggest optimizations
            cpu_avg = usages.get('cpu_usage_percent', 0)
            mem_avg = usages.get('memory_usage_percent', 0)

            if cpu_avg < 20 and mem_avg < 30:
                self._suggest_improvement('low', 'efficiency',
                                        "System resources underutilized. Consider increasing batch_size or enabling parallel processing for better throughput.")

            if mem_avg > 70:
                self._suggest_improvement('medium', 'memory',
                                        "High memory usage detected. Consider implementing streaming processing or reducing batch sizes.")

        except Exception as e:
            logger.error(f"Error suggesting resource optimizations: {e}")

    def _suggest_improvement(self, priority: str, category: str, description: str, impact_estimate: float = 0.5):
        """Create an improvement suggestion"""
        suggestion_id = f"{datetime.now().isoformat()}_{category}_{priority}"

        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO improvement_suggestions
                (suggestion_id, timestamp, category, priority, description, implementation_status, impact_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (suggestion_id, datetime.now().isoformat(), category, priority, description, 'pending', impact_estimate))

            conn.commit()
            conn.close()

            # Add to in-memory suggestions
            self.improvement_suggestions.append({
                'id': suggestion_id,
                'timestamp': datetime.now().isoformat(),
                'category': category,
                'priority': priority,
                'description': description,
                'status': 'pending',
                'impact': impact_estimate
            })

            logger.info(f"IMPROVEMENT SUGGESTION [{priority.upper()}]: {description}")

        except Exception as e:
            logger.error(f"Error creating improvement suggestion: {e}")

    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            # Keep only last 30 days of metrics
            month_ago = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (month_ago,))

            # Keep only last 90 days of alerts
            three_months_ago = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (three_months_ago,))

            conn.commit()
            conn.close()

            logger.info("Cleaned up old monitoring data")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'system_resources': {},
            'pipeline_health': {},
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'improvement_suggestions': self.improvement_suggestions[-5:],  # Last 5 suggestions
            'performance_baselines': self.baselines
        }

        try:
            # Current system resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('Z:')

            status['system_resources'] = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_used_mb': memory.used / 1024 / 1024,
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024
            }

            # Pipeline health
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('SELECT COUNT(*) FROM corpus_items')
                total_texts = cursor.fetchone()[0]

                cursor.execute("SELECT status, COUNT(*) FROM corpus_items GROUP BY status")
                status_counts = dict(cursor.fetchall())

                status['pipeline_health'] = {
                    'total_texts': total_texts,
                    'status_breakdown': status_counts,
                    'database_size_mb': os.path.getsize(self.db_path) / 1024 / 1024
                }

                conn.close()

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            status['error'] = str(e)

        return status

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for specified time period"""
        report = {
            'period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'metrics_summary': {},
            'trends': {},
            'recommendations': []
        }

        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            start_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            # Get metrics summary
            cursor.execute('''
                SELECT component, metric_name,
                       MIN(metric_value) as min_val,
                       MAX(metric_value) as max_val,
                       AVG(metric_value) as avg_val,
                       COUNT(*) as count
                FROM performance_metrics
                WHERE timestamp > ?
                GROUP BY component, metric_name
            ''', (start_time,))

            for row in cursor.fetchall():
                component, metric_name, min_val, max_val, avg_val, count = row
                key = f"{component}_{metric_name}"
                report['metrics_summary'][key] = {
                    'min': min_val,
                    'max': max_val,
                    'avg': avg_val,
                    'count': count
                }

            # Get recent alerts
            cursor.execute('''
                SELECT severity, component, message, COUNT(*) as count
                FROM alerts
                WHERE timestamp > ? AND resolved = 0
                GROUP BY severity, component, message
                ORDER BY count DESC
            ''', (start_time,))

            alerts_summary = []
            for row in cursor.fetchall():
                severity, component, message, count = row
                alerts_summary.append({
                    'severity': severity,
                    'component': component,
                    'message': message,
                    'count': count
                })

            report['alerts_summary'] = alerts_summary

            # Get improvement suggestions
            cursor.execute('''
                SELECT category, priority, description, impact_estimate
                FROM improvement_suggestions
                WHERE timestamp > ? AND implementation_status = 'pending'
                ORDER BY
                    CASE priority
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                    END,
                    impact_estimate DESC
            ''', (start_time,))

            suggestions = []
            for row in cursor.fetchall():
                category, priority, description, impact = row
                suggestions.append({
                    'category': category,
                    'priority': priority,
                    'description': description,
                    'impact_estimate': impact
                })

            report['improvement_suggestions'] = suggestions

            conn.close()

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            report['error'] = str(e)

        return report

    def apply_improvement(self, suggestion_id: str, action_taken: str):
        """Mark an improvement suggestion as implemented"""
        try:
            conn = sqlite3.connect(self.monitoring_db)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE improvement_suggestions
                SET implementation_status = 'implemented',
                    description = description || ' (Action: ' || ? || ')'
                WHERE suggestion_id = ?
            ''', (action_taken, suggestion_id))

            conn.commit()
            conn.close()

            # Update in-memory list
            for suggestion in self.improvement_suggestions:
                if suggestion['id'] == suggestion_id:
                    suggestion['status'] = 'implemented'
                    suggestion['action_taken'] = action_taken

            logger.info(f"Improvement {suggestion_id} marked as implemented: {action_taken}")

        except Exception as e:
            logger.error(f"Error applying improvement: {e}")

def main():
    """Main monitoring system"""
    monitor = ContinuousMonitoringSystem()

    print("Continuous Monitoring and Improvement System")
    print("=" * 50)
    print("Real-time pipeline monitoring and optimization")
    print("Press Ctrl+C to stop monitoring")
    print()

    # Start monitoring
    monitor.start_monitoring()

    try:
        # Keep running and show status periodically
        while True:
            status = monitor.get_system_status()

            print(f"\n--- System Status ({datetime.now().strftime('%H:%M:%S')}) ---")
            print(f"Monitoring Active: {status['monitoring_active']}")

            if 'system_resources' in status:
                res = status['system_resources']
                print(f"CPU: {res['cpu_usage_percent']:.1f}% | Memory: {res['memory_usage_percent']:.1f}% | Disk: {res['disk_usage_percent']:.1f}%")

            if 'pipeline_health' in status:
                ph = status['pipeline_health']
                print(f"Corpus: {ph['total_texts']} texts | DB Size: {ph['database_size_mb']:.1f}MB")

            if status.get('alerts'):
                print(f"Active Alerts: {len([a for a in status['alerts'] if not a['resolved']])}")

            if status.get('improvement_suggestions'):
                pending = len([s for s in status['improvement_suggestions'] if s['status'] == 'pending'])
                print(f"Pending Improvements: {pending}")

            time.sleep(60)  # Update every minute

    except KeyboardInterrupt:
        print("\nStopping monitoring system...")
        monitor.stop_monitoring()

    # Generate final report
    report = monitor.get_performance_report(hours=1)
    print("\nFinal Performance Report (last hour):")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
