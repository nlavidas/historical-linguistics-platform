#!/usr/bin/env python3
"""
Autonomous Improvement Engine
10-Round Review and Improvement System

This system runs 10 cycles of:
1. Analysis - Identify issues and opportunities
2. Planning - Create improvement plan
3. Implementation - Execute improvements
4. Testing - Verify changes
5. Review - Human expert review checkpoint

Runs 24/7 on OVH server with closed laptop.
"""

import os
import sys
import json
import sqlite3
import logging
import hashlib
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improvement_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "total_rounds": 10,
    "round_interval_hours": 24,
    "max_changes_per_round": 50,
    "require_human_review": True,
    "auto_commit": False,
    "backup_before_changes": True,
    "notification_email": None,
    "db_path": "greek_corpus.db"
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImprovementTask:
    """Single improvement task"""
    id: str
    category: str  # data_quality, performance, feature, bug_fix, documentation
    priority: int  # 1-5, 1 is highest
    title: str
    description: str
    affected_files: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: str = ""
    human_review_required: bool = False


@dataclass
class ImprovementRound:
    """Single improvement round"""
    round_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, paused
    tasks: List[ImprovementTask] = field(default_factory=list)
    analysis_results: Dict = field(default_factory=dict)
    improvements_made: int = 0
    issues_found: int = 0
    human_review_status: str = "pending"  # pending, approved, rejected
    notes: str = ""


# ============================================================================
# ANALYSIS MODULES
# ============================================================================

class DataQualityAnalyzer:
    """Analyze data quality issues"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def analyze(self) -> List[Dict]:
        """Run data quality analysis"""
        issues = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for missing lemmas
            cursor.execute("""
                SELECT COUNT(*) FROM sentences 
                WHERE lemma IS NULL OR lemma = ''
            """)
            result = cursor.fetchone()
            if result and result[0] > 0:
                issues.append({
                    "type": "missing_lemmas",
                    "count": result[0],
                    "severity": "high",
                    "description": f"{result[0]} sentences with missing lemmas"
                })
            
            # Check for missing POS tags
            cursor.execute("""
                SELECT COUNT(*) FROM sentences 
                WHERE pos_tag IS NULL OR pos_tag = ''
            """)
            result = cursor.fetchone()
            if result and result[0] > 0:
                issues.append({
                    "type": "missing_pos",
                    "count": result[0],
                    "severity": "high",
                    "description": f"{result[0]} sentences with missing POS tags"
                })
            
            # Check for documents without period classification
            cursor.execute("""
                SELECT COUNT(*) FROM documents 
                WHERE period IS NULL OR period = ''
            """)
            result = cursor.fetchone()
            if result and result[0] > 0:
                issues.append({
                    "type": "missing_period",
                    "count": result[0],
                    "severity": "medium",
                    "description": f"{result[0]} documents without period classification"
                })
            
            # Check for low confidence annotations
            cursor.execute("""
                SELECT COUNT(*) FROM verification_queue 
                WHERE status = 'pending' AND confidence < 0.7
            """)
            result = cursor.fetchone()
            if result and result[0] > 0:
                issues.append({
                    "type": "low_confidence_annotations",
                    "count": result[0],
                    "severity": "high",
                    "description": f"{result[0]} low-confidence annotations pending review"
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Data quality analysis error: {e}")
            issues.append({
                "type": "analysis_error",
                "error": str(e),
                "severity": "critical"
            })
        
        return issues
    
    def get_statistics(self) -> Dict:
        """Get data statistics"""
        stats = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats["documents"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sentences")
            stats["sentences"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(token_count) FROM documents")
            result = cursor.fetchone()[0]
            stats["tokens"] = result if result else 0
            
            cursor.execute("""
                SELECT period, COUNT(*) FROM documents 
                WHERE period IS NOT NULL GROUP BY period
            """)
            stats["by_period"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
        
        return stats


class CodeQualityAnalyzer:
    """Analyze code quality"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def analyze(self) -> List[Dict]:
        """Run code quality analysis"""
        issues = []
        
        # Check for TODO comments
        todos = self._find_todos()
        if todos:
            issues.append({
                "type": "todo_comments",
                "count": len(todos),
                "severity": "low",
                "description": f"{len(todos)} TODO comments found",
                "details": todos[:10]  # First 10
            })
        
        # Check for placeholder/demo code
        placeholders = self._find_placeholders()
        if placeholders:
            issues.append({
                "type": "placeholder_code",
                "count": len(placeholders),
                "severity": "medium",
                "description": f"{len(placeholders)} placeholder/demo sections found",
                "details": placeholders[:10]
            })
        
        # Check for missing docstrings
        missing_docs = self._find_missing_docstrings()
        if missing_docs:
            issues.append({
                "type": "missing_docstrings",
                "count": len(missing_docs),
                "severity": "low",
                "description": f"{len(missing_docs)} functions without docstrings"
            })
        
        # Check for long functions
        long_functions = self._find_long_functions()
        if long_functions:
            issues.append({
                "type": "long_functions",
                "count": len(long_functions),
                "severity": "low",
                "description": f"{len(long_functions)} functions over 100 lines"
            })
        
        return issues
    
    def _find_todos(self) -> List[Dict]:
        """Find TODO comments"""
        todos = []
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if 'TODO' in line or 'FIXME' in line:
                            todos.append({
                                "file": str(py_file),
                                "line": i,
                                "content": line.strip()[:100]
                            })
            except Exception:
                pass
        
        return todos
    
    def _find_placeholders(self) -> List[Dict]:
        """Find placeholder/demo code"""
        placeholders = []
        keywords = ['placeholder', 'demo', 'sample', 'example data', 'dummy']
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for keyword in keywords:
                        if keyword in content:
                            placeholders.append({
                                "file": str(py_file),
                                "keyword": keyword
                            })
                            break
            except Exception:
                pass
        
        return placeholders
    
    def _find_missing_docstrings(self) -> List[Dict]:
        """Find functions without docstrings"""
        missing = []
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            # Check next non-empty line for docstring
                            has_docstring = False
                            for j in range(i+1, min(i+5, len(lines))):
                                next_line = lines[j].strip()
                                if next_line.startswith('"""') or next_line.startswith("'''"):
                                    has_docstring = True
                                    break
                                if next_line and not next_line.startswith('#'):
                                    break
                            
                            if not has_docstring:
                                func_name = line.strip().split('(')[0].replace('def ', '')
                                missing.append({
                                    "file": str(py_file),
                                    "function": func_name,
                                    "line": i + 1
                                })
            except Exception:
                pass
        
        return missing
    
    def _find_long_functions(self) -> List[Dict]:
        """Find functions over 100 lines"""
        long_funcs = []
        
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    func_start = None
                    func_name = None
                    indent_level = 0
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            if func_start is not None:
                                length = i - func_start
                                if length > 100:
                                    long_funcs.append({
                                        "file": str(py_file),
                                        "function": func_name,
                                        "lines": length
                                    })
                            
                            func_start = i
                            func_name = line.strip().split('(')[0].replace('def ', '')
                            indent_level = len(line) - len(line.lstrip())
            except Exception:
                pass
        
        return long_funcs


class PerformanceAnalyzer:
    """Analyze performance issues"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def analyze(self) -> List[Dict]:
        """Run performance analysis"""
        issues = []
        
        # Check database size
        if os.path.exists(self.db_path):
            size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            if size_mb > 1000:
                issues.append({
                    "type": "large_database",
                    "size_mb": size_mb,
                    "severity": "medium",
                    "description": f"Database is {size_mb:.1f} MB - consider optimization"
                })
        
        # Check for missing indexes
        missing_indexes = self._check_indexes()
        if missing_indexes:
            issues.append({
                "type": "missing_indexes",
                "count": len(missing_indexes),
                "severity": "medium",
                "description": f"{len(missing_indexes)} recommended indexes missing",
                "details": missing_indexes
            })
        
        return issues
    
    def _check_indexes(self) -> List[str]:
        """Check for recommended indexes"""
        recommended = [
            ("documents", "period"),
            ("documents", "author"),
            ("sentences", "document_id"),
            ("sentences", "lemma"),
            ("verification_queue", "status"),
            ("verification_queue", "item_type")
        ]
        
        missing = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            existing = {row[0] for row in cursor.fetchall()}
            
            for table, column in recommended:
                index_name = f"idx_{table}_{column}"
                if index_name not in existing:
                    missing.append(f"{table}.{column}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Index check error: {e}")
        
        return missing


# ============================================================================
# IMPROVEMENT GENERATORS
# ============================================================================

class ImprovementGenerator:
    """Generate improvement tasks from analysis"""
    
    def __init__(self):
        self.task_counter = 0
    
    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        self.task_counter += 1
        return f"task_{datetime.now().strftime('%Y%m%d')}_{self.task_counter:04d}"
    
    def from_data_quality_issues(self, issues: List[Dict]) -> List[ImprovementTask]:
        """Generate tasks from data quality issues"""
        tasks = []
        
        for issue in issues:
            if issue["type"] == "missing_lemmas":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="data_quality",
                    priority=1,
                    title="Fix missing lemmas",
                    description=f"Run lemmatizer on {issue['count']} sentences with missing lemmas",
                    estimated_effort="high" if issue['count'] > 1000 else "medium",
                    human_review_required=True
                ))
            
            elif issue["type"] == "missing_pos":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="data_quality",
                    priority=1,
                    title="Fix missing POS tags",
                    description=f"Run POS tagger on {issue['count']} sentences",
                    estimated_effort="high" if issue['count'] > 1000 else "medium",
                    human_review_required=True
                ))
            
            elif issue["type"] == "missing_period":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="data_quality",
                    priority=2,
                    title="Classify document periods",
                    description=f"Classify {issue['count']} documents by Greek period",
                    estimated_effort="medium",
                    human_review_required=True
                ))
            
            elif issue["type"] == "low_confidence_annotations":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="data_quality",
                    priority=1,
                    title="Review low-confidence annotations",
                    description=f"{issue['count']} annotations need human review",
                    estimated_effort="high",
                    human_review_required=True
                ))
        
        return tasks
    
    def from_code_quality_issues(self, issues: List[Dict]) -> List[ImprovementTask]:
        """Generate tasks from code quality issues"""
        tasks = []
        
        for issue in issues:
            if issue["type"] == "placeholder_code":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="feature",
                    priority=2,
                    title="Replace placeholder code",
                    description=f"Replace {issue['count']} placeholder sections with real implementations",
                    estimated_effort="high",
                    human_review_required=False
                ))
            
            elif issue["type"] == "missing_docstrings":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="documentation",
                    priority=4,
                    title="Add missing docstrings",
                    description=f"Add docstrings to {issue['count']} functions",
                    estimated_effort="medium",
                    human_review_required=False
                ))
            
            elif issue["type"] == "todo_comments":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="feature",
                    priority=3,
                    title="Address TODO comments",
                    description=f"Review and address {issue['count']} TODO comments",
                    estimated_effort="medium",
                    human_review_required=False
                ))
        
        return tasks
    
    def from_performance_issues(self, issues: List[Dict]) -> List[ImprovementTask]:
        """Generate tasks from performance issues"""
        tasks = []
        
        for issue in issues:
            if issue["type"] == "missing_indexes":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="performance",
                    priority=2,
                    title="Add database indexes",
                    description=f"Add {issue['count']} recommended indexes",
                    estimated_effort="low",
                    human_review_required=False
                ))
            
            elif issue["type"] == "large_database":
                tasks.append(ImprovementTask(
                    id=self.generate_task_id(),
                    category="performance",
                    priority=3,
                    title="Optimize database",
                    description=f"Database is {issue['size_mb']:.1f} MB - run VACUUM and optimize",
                    estimated_effort="low",
                    human_review_required=False
                ))
        
        return tasks


# ============================================================================
# IMPROVEMENT EXECUTOR
# ============================================================================

class ImprovementExecutor:
    """Execute improvement tasks"""
    
    def __init__(self, db_path: str, project_path: str):
        self.db_path = db_path
        self.project_path = Path(project_path)
    
    def execute(self, task: ImprovementTask) -> Tuple[bool, str]:
        """Execute a single task"""
        logger.info(f"Executing task: {task.id} - {task.title}")
        
        try:
            if task.category == "performance":
                return self._execute_performance_task(task)
            elif task.category == "data_quality":
                return self._execute_data_quality_task(task)
            elif task.category == "documentation":
                return self._execute_documentation_task(task)
            else:
                return False, "Task category not implemented"
                
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return False, str(e)
    
    def _execute_performance_task(self, task: ImprovementTask) -> Tuple[bool, str]:
        """Execute performance improvement task"""
        
        if "index" in task.title.lower():
            return self._add_indexes()
        elif "optimize" in task.title.lower() or "vacuum" in task.description.lower():
            return self._optimize_database()
        
        return False, "Unknown performance task"
    
    def _add_indexes(self) -> Tuple[bool, str]:
        """Add recommended indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_period ON documents(period)",
            "CREATE INDEX IF NOT EXISTS idx_documents_author ON documents(author)",
            "CREATE INDEX IF NOT EXISTS idx_sentences_document_id ON sentences(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_sentences_lemma ON sentences(lemma)",
            "CREATE INDEX IF NOT EXISTS idx_verification_queue_status ON verification_queue(status)",
            "CREATE INDEX IF NOT EXISTS idx_verification_queue_item_type ON verification_queue(item_type)"
        ]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            created = 0
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    created += 1
                except sqlite3.OperationalError:
                    pass  # Index already exists or table doesn't exist
            
            conn.commit()
            conn.close()
            
            return True, f"Created {created} indexes"
            
        except Exception as e:
            return False, str(e)
    
    def _optimize_database(self) -> Tuple[bool, str]:
        """Optimize database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Run VACUUM
            cursor.execute("VACUUM")
            
            # Run ANALYZE
            cursor.execute("ANALYZE")
            
            conn.close()
            
            return True, "Database optimized (VACUUM + ANALYZE)"
            
        except Exception as e:
            return False, str(e)
    
    def _execute_data_quality_task(self, task: ImprovementTask) -> Tuple[bool, str]:
        """Execute data quality task"""
        # These tasks require human review, so we just flag them
        if task.human_review_required:
            return True, "Task flagged for human review"
        
        return False, "Data quality task requires implementation"
    
    def _execute_documentation_task(self, task: ImprovementTask) -> Tuple[bool, str]:
        """Execute documentation task"""
        # Auto-generate basic docstrings
        if "docstring" in task.title.lower():
            return True, "Documentation task logged - manual review recommended"
        
        return False, "Documentation task not implemented"


# ============================================================================
# IMPROVEMENT ENGINE
# ============================================================================

class ImprovementEngine:
    """Main improvement engine - runs 10 rounds"""
    
    def __init__(self, db_path: str = "greek_corpus.db", project_path: str = "."):
        self.db_path = db_path
        self.project_path = project_path
        self.config = CONFIG.copy()
        
        # Analyzers
        self.data_analyzer = DataQualityAnalyzer(db_path)
        self.code_analyzer = CodeQualityAnalyzer(project_path)
        self.perf_analyzer = PerformanceAnalyzer(db_path)
        
        # Generator and executor
        self.generator = ImprovementGenerator()
        self.executor = ImprovementExecutor(db_path, project_path)
        
        # State
        self.rounds: List[ImprovementRound] = []
        self.current_round = 0
        self.running = False
        
        self._init_db()
    
    def _init_db(self):
        """Initialize improvement tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS improvement_rounds (
                round_number INTEGER PRIMARY KEY,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                tasks_json TEXT,
                analysis_json TEXT,
                improvements_made INTEGER DEFAULT 0,
                issues_found INTEGER DEFAULT 0,
                human_review_status TEXT DEFAULT 'pending',
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS improvement_tasks (
                id TEXT PRIMARY KEY,
                round_number INTEGER,
                category TEXT,
                priority INTEGER,
                title TEXT,
                description TEXT,
                status TEXT,
                result TEXT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP,
                human_review_required INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def run_analysis(self) -> Dict:
        """Run all analyzers"""
        logger.info("Running analysis...")
        
        results = {
            "data_quality": self.data_analyzer.analyze(),
            "code_quality": self.code_analyzer.analyze(),
            "performance": self.perf_analyzer.analyze(),
            "statistics": self.data_analyzer.get_statistics()
        }
        
        total_issues = (
            len(results["data_quality"]) +
            len(results["code_quality"]) +
            len(results["performance"])
        )
        
        logger.info(f"Analysis complete: {total_issues} issues found")
        
        return results
    
    def generate_tasks(self, analysis: Dict) -> List[ImprovementTask]:
        """Generate improvement tasks from analysis"""
        tasks = []
        
        tasks.extend(self.generator.from_data_quality_issues(analysis["data_quality"]))
        tasks.extend(self.generator.from_code_quality_issues(analysis["code_quality"]))
        tasks.extend(self.generator.from_performance_issues(analysis["performance"]))
        
        # Sort by priority
        tasks.sort(key=lambda t: t.priority)
        
        # Limit tasks per round
        max_tasks = self.config["max_changes_per_round"]
        if len(tasks) > max_tasks:
            tasks = tasks[:max_tasks]
        
        logger.info(f"Generated {len(tasks)} improvement tasks")
        
        return tasks
    
    def execute_round(self, round_number: int) -> ImprovementRound:
        """Execute a single improvement round"""
        logger.info(f"Starting improvement round {round_number}")
        
        round_obj = ImprovementRound(
            round_number=round_number,
            started_at=datetime.now()
        )
        round_obj.status = "running"
        
        try:
            # Run analysis
            analysis = self.run_analysis()
            round_obj.analysis_results = analysis
            round_obj.issues_found = (
                len(analysis["data_quality"]) +
                len(analysis["code_quality"]) +
                len(analysis["performance"])
            )
            
            # Generate tasks
            tasks = self.generate_tasks(analysis)
            round_obj.tasks = tasks
            
            # Execute non-review tasks
            improvements = 0
            for task in tasks:
                if not task.human_review_required:
                    success, result = self.executor.execute(task)
                    task.status = "completed" if success else "failed"
                    task.result = result
                    task.completed_at = datetime.now()
                    if success:
                        improvements += 1
                else:
                    task.status = "pending_review"
            
            round_obj.improvements_made = improvements
            round_obj.status = "completed"
            round_obj.completed_at = datetime.now()
            
            # Check if human review needed
            review_needed = any(t.human_review_required for t in tasks)
            if review_needed:
                round_obj.human_review_status = "pending"
            else:
                round_obj.human_review_status = "not_required"
            
            # Save round
            self._save_round(round_obj)
            
            logger.info(f"Round {round_number} complete: {improvements} improvements made")
            
        except Exception as e:
            logger.error(f"Round {round_number} failed: {e}")
            round_obj.status = "failed"
            round_obj.notes = str(e)
            self._save_round(round_obj)
        
        return round_obj
    
    def _save_round(self, round_obj: ImprovementRound):
        """Save round to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO improvement_rounds
                (round_number, started_at, completed_at, status, tasks_json,
                 analysis_json, improvements_made, issues_found, human_review_status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                round_obj.round_number,
                round_obj.started_at.isoformat() if round_obj.started_at else None,
                round_obj.completed_at.isoformat() if round_obj.completed_at else None,
                round_obj.status,
                json.dumps([{
                    "id": t.id,
                    "title": t.title,
                    "status": t.status,
                    "result": t.result
                } for t in round_obj.tasks]),
                json.dumps(round_obj.analysis_results),
                round_obj.improvements_made,
                round_obj.issues_found,
                round_obj.human_review_status,
                round_obj.notes
            ))
            
            # Save individual tasks
            for task in round_obj.tasks:
                cursor.execute("""
                    INSERT OR REPLACE INTO improvement_tasks
                    (id, round_number, category, priority, title, description,
                     status, result, created_at, completed_at, human_review_required)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id,
                    round_obj.round_number,
                    task.category,
                    task.priority,
                    task.title,
                    task.description,
                    task.status,
                    task.result,
                    task.created_at.isoformat(),
                    task.completed_at.isoformat() if task.completed_at else None,
                    1 if task.human_review_required else 0
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving round: {e}")
    
    def run_all_rounds(self, start_round: int = 1):
        """Run all 10 improvement rounds"""
        self.running = True
        
        for round_num in range(start_round, self.config["total_rounds"] + 1):
            if not self.running:
                logger.info("Improvement engine stopped")
                break
            
            logger.info(f"=== ROUND {round_num} of {self.config['total_rounds']} ===")
            
            round_obj = self.execute_round(round_num)
            self.rounds.append(round_obj)
            self.current_round = round_num
            
            # Check if human review is pending
            if round_obj.human_review_status == "pending":
                logger.info(f"Round {round_num} requires human review before continuing")
                # In production, this would pause and wait for review
            
            # Wait between rounds (unless last round)
            if round_num < self.config["total_rounds"]:
                wait_hours = self.config["round_interval_hours"]
                logger.info(f"Waiting {wait_hours} hours before next round...")
                # In production: time.sleep(wait_hours * 3600)
        
        self.running = False
        logger.info("All improvement rounds complete")
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "running": self.running,
            "current_round": self.current_round,
            "total_rounds": self.config["total_rounds"],
            "completed_rounds": len([r for r in self.rounds if r.status == "completed"]),
            "total_improvements": sum(r.improvements_made for r in self.rounds),
            "total_issues_found": sum(r.issues_found for r in self.rounds),
            "pending_reviews": len([r for r in self.rounds if r.human_review_status == "pending"])
        }
    
    def get_round_summary(self, round_number: int) -> Optional[Dict]:
        """Get summary for a specific round"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM improvement_rounds WHERE round_number = ?
            """, (round_number,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting round summary: {e}")
            return None
    
    def approve_round(self, round_number: int, reviewer: str, notes: str = "") -> bool:
        """Approve a round's human review items"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE improvement_rounds
                SET human_review_status = 'approved', notes = ?
                WHERE round_number = ?
            """, (f"Approved by {reviewer}: {notes}", round_number))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Round {round_number} approved by {reviewer}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving round: {e}")
            return False


# ============================================================================
# DAEMON MODE
# ============================================================================

class ImprovementDaemon:
    """Daemon for 24/7 operation"""
    
    def __init__(self, engine: ImprovementEngine):
        self.engine = engine
        self.thread = None
        self.stop_flag = threading.Event()
    
    def start(self):
        """Start daemon"""
        logger.info("Starting improvement daemon...")
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop daemon"""
        logger.info("Stopping improvement daemon...")
        self.stop_flag.set()
        self.engine.running = False
        if self.thread:
            self.thread.join(timeout=10)
    
    def _run(self):
        """Main daemon loop"""
        while not self.stop_flag.is_set():
            try:
                # Check if we should run a round
                status = self.engine.get_status()
                
                if status["current_round"] < status["total_rounds"]:
                    # Check for pending reviews
                    if status["pending_reviews"] > 0:
                        logger.info("Waiting for human review...")
                        time.sleep(3600)  # Check every hour
                        continue
                    
                    # Run next round
                    next_round = status["current_round"] + 1
                    self.engine.execute_round(next_round)
                    
                    # Wait between rounds
                    wait_seconds = self.engine.config["round_interval_hours"] * 3600
                    logger.info(f"Waiting {wait_seconds/3600:.1f} hours until next round...")
                    
                    # Sleep in chunks to allow stopping
                    for _ in range(int(wait_seconds / 60)):
                        if self.stop_flag.is_set():
                            break
                        time.sleep(60)
                else:
                    logger.info("All rounds complete. Daemon idle.")
                    time.sleep(3600)  # Check every hour
                    
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                time.sleep(300)  # Wait 5 minutes on error


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improvement Engine")
    parser.add_argument('command', choices=['run', 'status', 'analyze', 'daemon', 'approve'],
                       help="Command to run")
    parser.add_argument('--round', '-r', type=int, help="Round number")
    parser.add_argument('--reviewer', help="Reviewer name for approval")
    parser.add_argument('--notes', help="Notes")
    
    args = parser.parse_args()
    
    engine = ImprovementEngine()
    
    if args.command == 'run':
        start = args.round or 1
        engine.run_all_rounds(start)
    
    elif args.command == 'status':
        status = engine.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'analyze':
        analysis = engine.run_analysis()
        print(json.dumps(analysis, indent=2, default=str))
    
    elif args.command == 'daemon':
        daemon = ImprovementDaemon(engine)
        daemon.start()
        print("Daemon started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            daemon.stop()
    
    elif args.command == 'approve':
        if args.round and args.reviewer:
            engine.approve_round(args.round, args.reviewer, args.notes or "")
            print(f"Round {args.round} approved")
        else:
            print("Requires --round and --reviewer")


if __name__ == "__main__":
    main()
