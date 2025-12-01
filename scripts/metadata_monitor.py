#!/usr/bin/env python3
"""
Metadata Monitor - Real-time Collection and Processing Tracking
Comprehensive monitoring system for the Greek corpus platform

Features:
- Real-time collection status
- Processing pipeline monitoring
- Quality metrics tracking
- Error logging and alerting
- Progress visualization
- Review queue management
"""

import os
import sys
import json
import time
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MONITOR_CONFIG = {
    "database_path": "greek_corpus.db",
    "log_path": "monitor.log",
    "refresh_interval": 30,  # seconds
    "alert_thresholds": {
        "error_rate": 0.1,  # 10% error rate triggers alert
        "queue_size": 1000,  # pending items threshold
        "processing_time": 300  # seconds per document
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CollectionStatus:
    """Status of text collection"""
    source: str
    total_works: int = 0
    collected: int = 0
    pending: int = 0
    errors: int = 0
    tokens_collected: int = 0
    last_activity: Optional[datetime] = None
    current_work: str = ""
    
    @property
    def progress_percent(self) -> float:
        if self.total_works == 0:
            return 0.0
        return (self.collected / self.total_works) * 100


@dataclass
class ProcessingStatus:
    """Status of processing pipeline"""
    stage: str
    documents_processed: int = 0
    documents_pending: int = 0
    documents_error: int = 0
    tokens_processed: int = 0
    avg_processing_time: float = 0.0
    last_processed: Optional[datetime] = None
    current_document: str = ""


@dataclass
class QualityMetrics:
    """Quality metrics for the corpus"""
    total_documents: int = 0
    total_sentences: int = 0
    total_tokens: int = 0
    annotated_tokens: int = 0
    validated_documents: int = 0
    annotation_coverage: float = 0.0
    inter_annotator_agreement: float = 0.0
    error_rate: float = 0.0


@dataclass
class ReviewItem:
    """Item in review queue"""
    id: str
    document_id: str
    item_type: str  # annotation, error, quality
    description: str
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_review, resolved
    reviewer: str = ""
    resolution: str = ""


# ============================================================================
# DATABASE
# ============================================================================

class MonitorDatabase:
    """Database for monitoring data"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize monitoring tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Collection status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                total_works INTEGER DEFAULT 0,
                collected INTEGER DEFAULT 0,
                pending INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                tokens_collected INTEGER DEFAULT 0,
                last_activity TIMESTAMP,
                current_work TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Processing status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                documents_processed INTEGER DEFAULT 0,
                documents_pending INTEGER DEFAULT 0,
                documents_error INTEGER DEFAULT 0,
                tokens_processed INTEGER DEFAULT 0,
                avg_processing_time REAL DEFAULT 0,
                last_processed TIMESTAMP,
                current_document TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Activity log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT,
                details TEXT,
                status TEXT,
                duration REAL
            )
        """)
        
        # Review queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                item_type TEXT NOT NULL,
                description TEXT,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                reviewer TEXT,
                resolution TEXT,
                resolved_at TIMESTAMP
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TIMESTAMP
            )
        """)
        
        # Quality snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_documents INTEGER,
                total_sentences INTEGER,
                total_tokens INTEGER,
                annotated_tokens INTEGER,
                validated_documents INTEGER,
                annotation_coverage REAL,
                error_rate REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_activity(self, category: str, action: str, target: str = "",
                    details: str = "", status: str = "success", duration: float = 0):
        """Log an activity"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO activity_log (category, action, target, details, status, duration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (category, action, target, details, status, duration))
        
        conn.commit()
        conn.close()
    
    def add_review_item(self, item: ReviewItem):
        """Add item to review queue"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO review_queue 
            (id, document_id, item_type, description, priority, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item.id, item.document_id, item.item_type, 
              item.description, item.priority, item.status))
        
        conn.commit()
        conn.close()
    
    def get_review_queue(self, status: str = "pending", limit: int = 50) -> List[Dict]:
        """Get review queue items"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM review_queue 
            WHERE status = ?
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """, (status, limit))
        
        items = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return items
    
    def create_alert(self, severity: str, category: str, message: str, details: str = ""):
        """Create an alert"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (severity, category, message, details)
            VALUES (?, ?, ?, ?)
        """, (severity, category, message, details))
        
        conn.commit()
        conn.close()
        logger.warning(f"ALERT [{severity}] {category}: {message}")
    
    def get_recent_activity(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent activity"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT * FROM activity_log 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (cutoff.isoformat(), limit))
        
        activity = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return activity
    
    def save_quality_snapshot(self, metrics: QualityMetrics):
        """Save quality metrics snapshot"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO quality_snapshots 
            (total_documents, total_sentences, total_tokens, annotated_tokens,
             validated_documents, annotation_coverage, error_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (metrics.total_documents, metrics.total_sentences, metrics.total_tokens,
              metrics.annotated_tokens, metrics.validated_documents,
              metrics.annotation_coverage, metrics.error_rate))
        
        conn.commit()
        conn.close()


# ============================================================================
# METADATA MONITOR
# ============================================================================

class MetadataMonitor:
    """Main monitoring system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or MONITOR_CONFIG
        self.db = MonitorDatabase(self.config.get("database_path", "greek_corpus.db"))
        self.start_time = datetime.now()
        self._running = False
    
    def get_collection_status(self) -> Dict[str, CollectionStatus]:
        """Get status of all collection sources"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Get from collection_metadata table
        cursor.execute("""
            SELECT source,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as collected,
                   SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                   SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                   SUM(token_count) as tokens,
                   MAX(completed_at) as last_activity
            FROM collection_metadata
            GROUP BY source
        """)
        
        status = {}
        for row in cursor.fetchall():
            status[row["source"]] = CollectionStatus(
                source=row["source"],
                total_works=row["total"],
                collected=row["collected"],
                pending=row["pending"],
                errors=row["errors"],
                tokens_collected=row["tokens"] or 0,
                last_activity=row["last_activity"]
            )
        
        conn.close()
        return status
    
    def get_processing_status(self) -> Dict[str, ProcessingStatus]:
        """Get status of processing pipeline"""
        stages = ["preprocessing", "parsing", "annotation", "valency", "etymology"]
        status = {}
        
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        for stage in stages:
            # This would query actual processing tables
            # For now, return placeholder
            status[stage] = ProcessingStatus(stage=stage)
        
        conn.close()
        return status
    
    def get_quality_metrics(self) -> QualityMetrics:
        """Get current quality metrics"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        metrics = QualityMetrics()
        
        # Get document counts
        cursor.execute("SELECT COUNT(*) FROM documents")
        result = cursor.fetchone()
        metrics.total_documents = result[0] if result else 0
        
        cursor.execute("SELECT COUNT(*) FROM sentences")
        result = cursor.fetchone()
        metrics.total_sentences = result[0] if result else 0
        
        cursor.execute("SELECT COUNT(*) FROM tokens")
        result = cursor.fetchone()
        metrics.total_tokens = result[0] if result else 0
        
        # Calculate annotation coverage
        if metrics.total_tokens > 0:
            cursor.execute("SELECT COUNT(*) FROM tokens WHERE lemma IS NOT NULL AND lemma != ''")
            result = cursor.fetchone()
            metrics.annotated_tokens = result[0] if result else 0
            metrics.annotation_coverage = metrics.annotated_tokens / metrics.total_tokens
        
        conn.close()
        return metrics
    
    def get_current_status(self) -> Dict:
        """Get comprehensive current status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "collection": {k: asdict(v) for k, v in self.get_collection_status().items()},
            "processing": {k: asdict(v) for k, v in self.get_processing_status().items()},
            "quality": asdict(self.get_quality_metrics()),
            "review_queue": len(self.db.get_review_queue()),
            "recent_activity": self.db.get_recent_activity(hours=1, limit=10)
        }
    
    def check_alerts(self):
        """Check for alert conditions"""
        thresholds = self.config.get("alert_thresholds", {})
        
        # Check error rate
        metrics = self.get_quality_metrics()
        if metrics.error_rate > thresholds.get("error_rate", 0.1):
            self.db.create_alert(
                "warning", "quality",
                f"Error rate ({metrics.error_rate:.1%}) exceeds threshold"
            )
        
        # Check queue size
        queue_size = len(self.db.get_review_queue())
        if queue_size > thresholds.get("queue_size", 1000):
            self.db.create_alert(
                "warning", "queue",
                f"Review queue size ({queue_size}) exceeds threshold"
            )
    
    def add_to_review(self, document_id: str, item_type: str, 
                     description: str, priority: int = 1):
        """Add item to review queue"""
        import hashlib
        item_id = hashlib.md5(f"{document_id}:{item_type}:{description}".encode()).hexdigest()[:16]
        
        item = ReviewItem(
            id=item_id,
            document_id=document_id,
            item_type=item_type,
            description=description,
            priority=priority
        )
        
        self.db.add_review_item(item)
        logger.info(f"Added to review queue: {item_type} - {description[:50]}")
    
    def get_summary_report(self) -> str:
        """Generate summary report"""
        status = self.get_current_status()
        
        lines = [
            "=" * 60,
            "GREEK CORPUS PLATFORM - STATUS REPORT",
            f"Generated: {status['timestamp']}",
            f"Uptime: {status['uptime']}",
            "=" * 60,
            "",
            "COLLECTION STATUS:",
            "-" * 40
        ]
        
        for source, data in status["collection"].items():
            lines.append(f"  {source}:")
            lines.append(f"    Collected: {data['collected']}/{data['total_works']} ({data['collected']/max(data['total_works'],1)*100:.1f}%)")
            lines.append(f"    Tokens: {data['tokens_collected']:,}")
            lines.append(f"    Errors: {data['errors']}")
        
        lines.extend([
            "",
            "QUALITY METRICS:",
            "-" * 40,
            f"  Documents: {status['quality']['total_documents']:,}",
            f"  Sentences: {status['quality']['total_sentences']:,}",
            f"  Tokens: {status['quality']['total_tokens']:,}",
            f"  Annotation Coverage: {status['quality']['annotation_coverage']:.1%}",
            "",
            f"Review Queue: {status['review_queue']} items pending",
            "",
            "RECENT ACTIVITY:",
            "-" * 40
        ])
        
        for activity in status["recent_activity"][:5]:
            lines.append(f"  [{activity['timestamp']}] {activity['action']}: {activity['target']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ============================================================================
# REVIEW INTERFACE
# ============================================================================

class ReviewInterface:
    """Interface for reviewing items"""
    
    def __init__(self, monitor: MetadataMonitor):
        self.monitor = monitor
        self.db = monitor.db
    
    def get_next_item(self) -> Optional[Dict]:
        """Get next item to review"""
        items = self.db.get_review_queue(status="pending", limit=1)
        return items[0] if items else None
    
    def resolve_item(self, item_id: str, resolution: str, reviewer: str = ""):
        """Resolve a review item"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE review_queue 
            SET status = 'resolved', resolution = ?, reviewer = ?, resolved_at = ?
            WHERE id = ?
        """, (resolution, reviewer, datetime.now().isoformat(), item_id))
        
        conn.commit()
        conn.close()
        
        self.db.log_activity("review", "resolved", item_id, resolution)
    
    def skip_item(self, item_id: str, reason: str = ""):
        """Skip a review item"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE review_queue 
            SET priority = priority - 1
            WHERE id = ?
        """, (item_id,))
        
        conn.commit()
        conn.close()
        
        self.db.log_activity("review", "skipped", item_id, reason)
    
    def get_statistics(self) -> Dict:
        """Get review statistics"""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("""
            SELECT status, COUNT(*) FROM review_queue GROUP BY status
        """)
        stats["by_status"] = {row["status"]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT item_type, COUNT(*) FROM review_queue WHERE status = 'pending' GROUP BY item_type
        """)
        stats["pending_by_type"] = {row["item_type"]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT COUNT(*) FROM review_queue WHERE resolved_at > datetime('now', '-24 hours')
        """)
        stats["resolved_24h"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Metadata Monitor")
    parser.add_argument('command', choices=['status', 'report', 'review', 'alerts', 'activity'],
                       help="Command to run")
    parser.add_argument('--hours', type=int, default=24, help="Hours for activity")
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    args = parser.parse_args()
    
    monitor = MetadataMonitor()
    
    if args.command == 'status':
        status = monitor.get_current_status()
        if args.format == 'json':
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"Uptime: {status['uptime']}")
            print(f"Documents: {status['quality']['total_documents']}")
            print(f"Tokens: {status['quality']['total_tokens']}")
            print(f"Review Queue: {status['review_queue']} items")
    
    elif args.command == 'report':
        print(monitor.get_summary_report())
    
    elif args.command == 'review':
        interface = ReviewInterface(monitor)
        stats = interface.get_statistics()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'alerts':
        monitor.check_alerts()
        print("Alert check complete")
    
    elif args.command == 'activity':
        activity = monitor.db.get_recent_activity(hours=args.hours)
        if args.format == 'json':
            print(json.dumps(activity, indent=2))
        else:
            for item in activity:
                print(f"[{item['timestamp']}] {item['category']}/{item['action']}: {item['target']}")


if __name__ == "__main__":
    main()
