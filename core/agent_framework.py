#!/usr/bin/env python3
"""
Agent Framework for Greek Diachronic Linguistics Platform
"Windsurf for Greek Linguistics"

Autonomous agents for:
1. Data collection and curation
2. Annotation and verification
3. Analysis and research
4. Quality assurance
5. Continuous improvement

All agents work through online APIs - no local AI downloads required.
"""

import os
import json
import sqlite3
import logging
import hashlib
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from queue import Queue, Empty
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS
# ============================================================================

class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_REVIEW = "waiting_review"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AgentTask:
    """Task for an agent"""
    id: str
    agent_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    from_agent: str
    to_agent: str
    message_type: str
    content: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    read: bool = False


@dataclass
class AgentConfig:
    """Agent configuration"""
    id: str
    name: str
    agent_type: str
    enabled: bool = True
    auto_start: bool = False
    max_concurrent_tasks: int = 5
    task_timeout: int = 300
    retry_delay: int = 60
    schedule: str = ""  # cron-like schedule
    parameters: Dict = field(default_factory=dict)


# ============================================================================
# BASE AGENT
# ============================================================================

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig, db_path: str = "greek_corpus.db"):
        self.config = config
        self.db_path = db_path
        self.status = AgentStatus.IDLE
        self.task_queue: Queue = Queue()
        self.message_queue: Queue = Queue()
        self.current_task: Optional[AgentTask] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_runtime": 0
        }
        
        self._init_db()
    
    def _init_db(self):
        """Initialize agent tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                task_type TEXT,
                priority INTEGER,
                payload TEXT,
                status TEXT,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                retries INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                from_agent TEXT,
                to_agent TEXT,
                message_type TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                read INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                level TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Agent type identifier"""
        pass
    
    @abstractmethod
    def process_task(self, task: AgentTask) -> Any:
        """Process a single task"""
        pass
    
    def start(self):
        """Start the agent"""
        if self.status == AgentStatus.RUNNING:
            return
        
        self.stop_event.clear()
        self.status = AgentStatus.RUNNING
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.log("info", f"Agent {self.config.name} started")
    
    def stop(self):
        """Stop the agent"""
        self.stop_event.set()
        self.status = AgentStatus.STOPPED
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        self.log("info", f"Agent {self.config.name} stopped")
    
    def pause(self):
        """Pause the agent"""
        self.status = AgentStatus.PAUSED
        self.log("info", f"Agent {self.config.name} paused")
    
    def resume(self):
        """Resume the agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            self.log("info", f"Agent {self.config.name} resumed")
    
    def submit_task(self, task: AgentTask):
        """Submit a task to the agent"""
        self.task_queue.put(task)
        self._save_task(task)
        self.log("info", f"Task {task.id} submitted")
    
    def send_message(self, to_agent: str, message_type: str, content: Dict):
        """Send message to another agent"""
        msg = AgentMessage(
            id=hashlib.md5(f"{self.config.id}:{to_agent}:{datetime.now()}".encode()).hexdigest()[:16],
            from_agent=self.config.id,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        self._save_message(msg)
        return msg
    
    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, from_agent, to_agent, message_type, content, timestamp
            FROM agent_messages
            WHERE to_agent = ? AND read = 0
            ORDER BY timestamp
        """, (self.config.id,))
        
        messages = []
        for row in cursor.fetchall():
            msg = AgentMessage(
                id=row[0],
                from_agent=row[1],
                to_agent=row[2],
                message_type=row[3],
                content=json.loads(row[4]),
                timestamp=datetime.fromisoformat(row[5])
            )
            messages.append(msg)
            
            # Mark as read
            cursor.execute(
                "UPDATE agent_messages SET read = 1 WHERE id = ?",
                (row[0],)
            )
        
        conn.commit()
        conn.close()
        return messages
    
    def _worker_loop(self):
        """Main worker loop"""
        while not self.stop_event.is_set():
            if self.status == AgentStatus.PAUSED:
                time.sleep(1)
                continue
            
            try:
                # Check for messages
                messages = self.receive_messages()
                for msg in messages:
                    self._handle_message(msg)
                
                # Get next task
                try:
                    task = self.task_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Process task
                self.current_task = task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                self._save_task(task)
                
                try:
                    start_time = time.time()
                    result = self.process_task(task)
                    elapsed = time.time() - start_time
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    self.stats["tasks_completed"] += 1
                    self.stats["total_runtime"] += elapsed
                    
                    self.log("info", f"Task {task.id} completed in {elapsed:.2f}s")
                    
                except Exception as e:
                    task.error = str(e)
                    task.retries += 1
                    
                    if task.retries < task.max_retries:
                        task.status = TaskStatus.PENDING
                        self.task_queue.put(task)
                        self.log("warning", f"Task {task.id} failed, retrying ({task.retries}/{task.max_retries})")
                    else:
                        task.status = TaskStatus.FAILED
                        self.stats["tasks_failed"] += 1
                        self.log("error", f"Task {task.id} failed permanently: {e}")
                
                self._save_task(task)
                self.current_task = None
                
            except Exception as e:
                self.log("error", f"Worker loop error: {e}")
                self.status = AgentStatus.ERROR
                time.sleep(5)
    
    def _handle_message(self, msg: AgentMessage):
        """Handle incoming message"""
        self.log("info", f"Received message from {msg.from_agent}: {msg.message_type}")
        # Override in subclasses for specific handling
    
    def _save_task(self, task: AgentTask):
        """Save task to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO agent_tasks
            (id, agent_id, task_type, priority, payload, status, result, error,
             created_at, started_at, completed_at, retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id,
            task.agent_id,
            task.task_type,
            task.priority.value,
            json.dumps(task.payload),
            task.status.value,
            json.dumps(task.result) if task.result else None,
            task.error,
            task.created_at.isoformat(),
            task.started_at.isoformat() if task.started_at else None,
            task.completed_at.isoformat() if task.completed_at else None,
            task.retries
        ))
        
        conn.commit()
        conn.close()
    
    def _save_message(self, msg: AgentMessage):
        """Save message to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_messages
            (id, from_agent, to_agent, message_type, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            msg.id,
            msg.from_agent,
            msg.to_agent,
            msg.message_type,
            json.dumps(msg.content),
            msg.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def log(self, level: str, message: str):
        """Log agent activity"""
        logger.log(
            getattr(logging, level.upper(), logging.INFO),
            f"[{self.config.name}] {message}"
        )
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_logs (agent_id, level, message)
                VALUES (?, ?, ?)
            """, (self.config.id, level, message))
            
            conn.commit()
            conn.close()
        except Exception:
            pass
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "agent_id": self.config.id,
            "name": self.config.name,
            "status": self.status.value,
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "total_runtime": self.stats["total_runtime"],
            "queue_size": self.task_queue.qsize()
        }


# ============================================================================
# DATA COLLECTION AGENT
# ============================================================================

class DataCollectionAgent(BaseAgent):
    """Agent for collecting Greek texts"""
    
    @property
    def agent_type(self) -> str:
        return "data_collection"
    
    def process_task(self, task: AgentTask) -> Any:
        """Process data collection task"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "collect_perseus":
            return self._collect_from_perseus(payload)
        elif task_type == "collect_first1k":
            return self._collect_from_first1k(payload)
        elif task_type == "collect_proiel":
            return self._collect_from_proiel(payload)
        elif task_type == "validate_text":
            return self._validate_text(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _collect_from_perseus(self, payload: Dict) -> Dict:
        """Collect from Perseus Digital Library"""
        import requests
        
        text_id = payload.get("text_id")
        url = f"https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:{text_id}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Extract text (simplified)
                import re
                text = re.sub(r'<[^>]+>', '', response.text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                return {
                    "source": "perseus",
                    "text_id": text_id,
                    "content_length": len(text),
                    "status": "success"
                }
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _collect_from_first1k(self, payload: Dict) -> Dict:
        """Collect from First1KGreek"""
        import requests
        
        urn = payload.get("urn")
        # Convert URN to GitHub raw URL
        parts = urn.split(':')
        if len(parts) >= 4:
            work_parts = parts[3].split('.')
            if len(work_parts) >= 2:
                url = f"https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/{work_parts[0]}/{work_parts[1]}/{work_parts[0]}.{work_parts[1]}.perseus-grc1.xml"
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        return {
                            "source": "first1k",
                            "urn": urn,
                            "content_length": len(response.text),
                            "status": "success"
                        }
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        
        return {"status": "error", "message": "Invalid URN"}
    
    def _collect_from_proiel(self, payload: Dict) -> Dict:
        """Collect from PROIEL treebank"""
        import requests
        
        filename = payload.get("filename")
        url = f"https://raw.githubusercontent.com/proiel/proiel-treebank/master/releases/{filename}"
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                return {
                    "source": "proiel",
                    "filename": filename,
                    "content_length": len(response.text),
                    "status": "success"
                }
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _validate_text(self, payload: Dict) -> Dict:
        """Validate collected text"""
        text = payload.get("text", "")
        
        # Check for Greek characters
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
        total_chars = len(text)
        greek_ratio = greek_chars / total_chars if total_chars > 0 else 0
        
        return {
            "total_chars": total_chars,
            "greek_chars": greek_chars,
            "greek_ratio": greek_ratio,
            "is_valid": greek_ratio > 0.3,
            "status": "success"
        }


# ============================================================================
# ANNOTATION AGENT
# ============================================================================

class AnnotationAgent(BaseAgent):
    """Agent for text annotation"""
    
    @property
    def agent_type(self) -> str:
        return "annotation"
    
    def process_task(self, task: AgentTask) -> Any:
        """Process annotation task"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "pos_tag":
            return self._pos_tag(payload)
        elif task_type == "lemmatize":
            return self._lemmatize(payload)
        elif task_type == "parse":
            return self._parse(payload)
        elif task_type == "ner":
            return self._ner(payload)
        elif task_type == "classify_period":
            return self._classify_period(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _pos_tag(self, payload: Dict) -> Dict:
        """POS tag text"""
        text = payload.get("text", "")
        
        # Simple rule-based POS tagging
        tokens = text.split()
        tagged = []
        
        for token in tokens:
            # Simplified tagging rules
            lower = token.lower()
            if lower in ('ὁ', 'ἡ', 'τό', 'τοῦ', 'τῆς', 'τῷ', 'τῇ'):
                tag = 'ART'
            elif lower in ('καί', 'δέ', 'γάρ', 'ἀλλά'):
                tag = 'CONJ'
            elif lower in ('ἐν', 'εἰς', 'ἐκ', 'ἀπό', 'πρός'):
                tag = 'PREP'
            elif lower.endswith(('ω', 'εις', 'ει', 'ομεν', 'ετε')):
                tag = 'VERB'
            elif lower.endswith(('ος', 'ου', 'ῳ', 'ον', 'η', 'ης')):
                tag = 'NOUN'
            else:
                tag = 'X'
            
            tagged.append({"token": token, "pos": tag})
        
        return {
            "tokens": len(tagged),
            "tagged": tagged,
            "status": "success"
        }
    
    def _lemmatize(self, payload: Dict) -> Dict:
        """Lemmatize text"""
        text = payload.get("text", "")
        
        # Simple lemmatization (would use proper lemmatizer in production)
        lemma_map = {
            'τοῦ': 'ὁ', 'τῆς': 'ὁ', 'τῷ': 'ὁ', 'τῇ': 'ὁ',
            'τόν': 'ὁ', 'τήν': 'ὁ', 'τούς': 'ὁ', 'τάς': 'ὁ',
            'ἐστί': 'εἰμί', 'ἐστίν': 'εἰμί', 'ἦν': 'εἰμί',
            'λέγει': 'λέγω', 'εἶπεν': 'λέγω',
        }
        
        tokens = text.split()
        lemmatized = []
        
        for token in tokens:
            lemma = lemma_map.get(token.lower(), token)
            lemmatized.append({"token": token, "lemma": lemma})
        
        return {
            "tokens": len(lemmatized),
            "lemmatized": lemmatized,
            "status": "success"
        }
    
    def _parse(self, payload: Dict) -> Dict:
        """Parse text"""
        text = payload.get("text", "")
        
        # Simplified parsing
        tokens = text.split()
        
        # Find verb as root
        root_idx = None
        for i, token in enumerate(tokens):
            if token.lower().endswith(('ω', 'εις', 'ει', 'ομεν')):
                root_idx = i
                break
        
        parsed = []
        for i, token in enumerate(tokens):
            node = {
                "id": i + 1,
                "token": token,
                "head": 0 if i == root_idx else (root_idx + 1 if root_idx else 0),
                "relation": "root" if i == root_idx else "dep"
            }
            parsed.append(node)
        
        return {
            "tokens": len(parsed),
            "tree": parsed,
            "status": "success"
        }
    
    def _ner(self, payload: Dict) -> Dict:
        """Named entity recognition"""
        text = payload.get("text", "")
        
        # Simple gazetteer-based NER
        entities = {
            'Σωκράτης': 'PERSON', 'Πλάτων': 'PERSON',
            'Ἀθῆναι': 'LOCATION', 'Σπάρτη': 'LOCATION',
            'Ἕλληνες': 'GPE', 'Πέρσαι': 'GPE'
        }
        
        found = []
        for entity, etype in entities.items():
            if entity in text:
                found.append({"text": entity, "type": etype})
        
        return {
            "entities": found,
            "count": len(found),
            "status": "success"
        }
    
    def _classify_period(self, payload: Dict) -> Dict:
        """Classify text by period"""
        text = payload.get("text", "")
        
        # Simple heuristics
        features = {
            "archaic": ['ἦμος', 'κεν', 'ἄρα'],
            "classical": ['μέν', 'δέ', 'γάρ', 'οὖν'],
            "hellenistic": ['ἵνα', 'ὅτι', 'ἐάν'],
            "byzantine": ['καθώς', 'ὥστε', 'διότι']
        }
        
        scores = {}
        for period, markers in features.items():
            score = sum(1 for m in markers if m in text)
            scores[period] = score
        
        best_period = max(scores, key=scores.get) if any(scores.values()) else "unknown"
        
        return {
            "period": best_period,
            "scores": scores,
            "confidence": scores.get(best_period, 0) / max(sum(scores.values()), 1),
            "status": "success"
        }


# ============================================================================
# ANALYSIS AGENT
# ============================================================================

class AnalysisAgent(BaseAgent):
    """Agent for linguistic analysis"""
    
    @property
    def agent_type(self) -> str:
        return "analysis"
    
    def process_task(self, task: AgentTask) -> Any:
        """Process analysis task"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "frequency_analysis":
            return self._frequency_analysis(payload)
        elif task_type == "collocation_analysis":
            return self._collocation_analysis(payload)
        elif task_type == "diachronic_comparison":
            return self._diachronic_comparison(payload)
        elif task_type == "stylometric_analysis":
            return self._stylometric_analysis(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _frequency_analysis(self, payload: Dict) -> Dict:
        """Word frequency analysis"""
        text = payload.get("text", "")
        
        from collections import Counter
        
        tokens = text.lower().split()
        freq = Counter(tokens)
        
        return {
            "total_tokens": len(tokens),
            "unique_tokens": len(freq),
            "top_20": freq.most_common(20),
            "hapax_legomena": len([w for w, c in freq.items() if c == 1]),
            "status": "success"
        }
    
    def _collocation_analysis(self, payload: Dict) -> Dict:
        """Collocation analysis"""
        text = payload.get("text", "")
        window = payload.get("window", 2)
        
        from collections import Counter
        
        tokens = text.lower().split()
        collocations = Counter()
        
        for i in range(len(tokens) - window):
            ngram = tuple(tokens[i:i+window+1])
            collocations[ngram] += 1
        
        return {
            "window_size": window,
            "total_collocations": len(collocations),
            "top_20": [{"collocation": ' '.join(c), "count": n} 
                      for c, n in collocations.most_common(20)],
            "status": "success"
        }
    
    def _diachronic_comparison(self, payload: Dict) -> Dict:
        """Compare texts across periods"""
        texts = payload.get("texts", [])
        
        results = []
        for text_info in texts:
            text = text_info.get("text", "")
            period = text_info.get("period", "unknown")
            
            tokens = text.lower().split()
            unique = set(tokens)
            
            results.append({
                "period": period,
                "token_count": len(tokens),
                "unique_count": len(unique),
                "type_token_ratio": len(unique) / len(tokens) if tokens else 0
            })
        
        return {
            "comparisons": results,
            "status": "success"
        }
    
    def _stylometric_analysis(self, payload: Dict) -> Dict:
        """Stylometric analysis"""
        text = payload.get("text", "")
        
        tokens = text.split()
        sentences = text.split('.')
        
        # Calculate metrics
        avg_word_length = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
        
        # Function words ratio
        function_words = {'ὁ', 'ἡ', 'τό', 'καί', 'δέ', 'γάρ', 'μέν', 'οὖν', 'ἐν', 'εἰς'}
        fw_count = sum(1 for t in tokens if t.lower() in function_words)
        fw_ratio = fw_count / len(tokens) if tokens else 0
        
        return {
            "token_count": len(tokens),
            "sentence_count": len(sentences),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "function_word_ratio": fw_ratio,
            "status": "success"
        }


# ============================================================================
# QUALITY ASSURANCE AGENT
# ============================================================================

class QualityAgent(BaseAgent):
    """Agent for quality assurance"""
    
    @property
    def agent_type(self) -> str:
        return "quality"
    
    def process_task(self, task: AgentTask) -> Any:
        """Process QA task"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "validate_annotation":
            return self._validate_annotation(payload)
        elif task_type == "check_consistency":
            return self._check_consistency(payload)
        elif task_type == "detect_errors":
            return self._detect_errors(payload)
        elif task_type == "generate_report":
            return self._generate_report(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _validate_annotation(self, payload: Dict) -> Dict:
        """Validate annotation quality"""
        annotations = payload.get("annotations", [])
        
        issues = []
        for ann in annotations:
            # Check for missing fields
            required = ['token', 'pos', 'lemma']
            for field in required:
                if field not in ann or not ann[field]:
                    issues.append({
                        "type": "missing_field",
                        "field": field,
                        "token": ann.get("token", "unknown")
                    })
            
            # Check for invalid POS tags
            valid_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PREP', 'CONJ', 'ART', 'PRON', 'X'}
            if ann.get("pos") and ann["pos"] not in valid_pos:
                issues.append({
                    "type": "invalid_pos",
                    "pos": ann["pos"],
                    "token": ann.get("token", "unknown")
                })
        
        return {
            "total_annotations": len(annotations),
            "issues_found": len(issues),
            "issues": issues[:50],  # Limit to 50
            "quality_score": 1 - (len(issues) / max(len(annotations), 1)),
            "status": "success"
        }
    
    def _check_consistency(self, payload: Dict) -> Dict:
        """Check annotation consistency"""
        annotations = payload.get("annotations", [])
        
        # Group by lemma
        lemma_pos = {}
        for ann in annotations:
            lemma = ann.get("lemma", "")
            pos = ann.get("pos", "")
            if lemma:
                if lemma not in lemma_pos:
                    lemma_pos[lemma] = set()
                lemma_pos[lemma].add(pos)
        
        # Find inconsistencies
        inconsistent = []
        for lemma, pos_set in lemma_pos.items():
            if len(pos_set) > 1:
                inconsistent.append({
                    "lemma": lemma,
                    "pos_tags": list(pos_set)
                })
        
        return {
            "total_lemmas": len(lemma_pos),
            "inconsistent_lemmas": len(inconsistent),
            "inconsistencies": inconsistent[:20],
            "consistency_score": 1 - (len(inconsistent) / max(len(lemma_pos), 1)),
            "status": "success"
        }
    
    def _detect_errors(self, payload: Dict) -> Dict:
        """Detect potential errors"""
        text = payload.get("text", "")
        
        errors = []
        
        # Check for encoding issues
        if '?' in text or '�' in text:
            errors.append({"type": "encoding_issue", "severity": "high"})
        
        # Check for mixed scripts
        has_greek = any('\u0370' <= c <= '\u03FF' for c in text)
        has_latin = any('a' <= c.lower() <= 'z' for c in text)
        if has_greek and has_latin:
            errors.append({"type": "mixed_scripts", "severity": "medium"})
        
        # Check for unusual characters
        unusual = [c for c in text if ord(c) > 0x2000 and ord(c) < 0x3000]
        if unusual:
            errors.append({"type": "unusual_characters", "severity": "low", "count": len(unusual)})
        
        return {
            "errors_found": len(errors),
            "errors": errors,
            "status": "success"
        }
    
    def _generate_report(self, payload: Dict) -> Dict:
        """Generate quality report"""
        data = payload.get("data", {})
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_items": data.get("total_items", 0),
                "validated": data.get("validated", 0),
                "issues": data.get("issues", 0)
            },
            "recommendations": [],
            "status": "success"
        }
        
        # Generate recommendations
        if data.get("issues", 0) > 0:
            report["recommendations"].append("Review and fix identified issues")
        
        if data.get("consistency_score", 1) < 0.9:
            report["recommendations"].append("Improve annotation consistency")
        
        return report


# ============================================================================
# AGENT MANAGER
# ============================================================================

class AgentManager:
    """Manages all agents"""
    
    def __init__(self, db_path: str = "greek_corpus.db"):
        self.db_path = db_path
        self.agents: Dict[str, BaseAgent] = {}
        self._init_default_agents()
    
    def _init_default_agents(self):
        """Initialize default agents"""
        # Data Collection Agent
        self.register_agent(DataCollectionAgent(
            AgentConfig(
                id="collector",
                name="Data Collector",
                agent_type="data_collection",
                auto_start=True
            ),
            self.db_path
        ))
        
        # Annotation Agent
        self.register_agent(AnnotationAgent(
            AgentConfig(
                id="annotator",
                name="Annotator",
                agent_type="annotation",
                auto_start=True
            ),
            self.db_path
        ))
        
        # Analysis Agent
        self.register_agent(AnalysisAgent(
            AgentConfig(
                id="analyzer",
                name="Analyzer",
                agent_type="analysis",
                auto_start=True
            ),
            self.db_path
        ))
        
        # Quality Agent
        self.register_agent(QualityAgent(
            AgentConfig(
                id="quality",
                name="Quality Assurance",
                agent_type="quality",
                auto_start=True
            ),
            self.db_path
        ))
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.config.id] = agent
        logger.info(f"Registered agent: {agent.config.name}")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def start_all(self):
        """Start all agents"""
        for agent in self.agents.values():
            if agent.config.auto_start:
                agent.start()
    
    def stop_all(self):
        """Stop all agents"""
        for agent in self.agents.values():
            agent.stop()
    
    def submit_task(self, agent_id: str, task_type: str, 
                    payload: Dict, priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Submit task to agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        task_id = hashlib.md5(
            f"{agent_id}:{task_type}:{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        task = AgentTask(
            id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            priority=priority,
            payload=payload
        )
        
        agent.submit_task(task)
        return task_id
    
    def get_all_stats(self) -> Dict:
        """Get stats for all agents"""
        return {
            agent_id: agent.get_stats()
            for agent_id, agent in self.agents.items()
        }
    
    def broadcast_message(self, from_agent: str, message_type: str, content: Dict):
        """Broadcast message to all agents"""
        for agent_id, agent in self.agents.items():
            if agent_id != from_agent:
                agent.send_message(agent_id, message_type, content)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Framework")
    parser.add_argument('command', choices=['start', 'stop', 'status', 'task'],
                       help="Command to run")
    parser.add_argument('--agent', '-a', help="Agent ID")
    parser.add_argument('--task', '-t', help="Task type")
    parser.add_argument('--payload', '-p', help="Task payload (JSON)")
    
    args = parser.parse_args()
    
    manager = AgentManager()
    
    if args.command == 'start':
        manager.start_all()
        print("All agents started")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all()
    
    elif args.command == 'stop':
        manager.stop_all()
        print("All agents stopped")
    
    elif args.command == 'status':
        stats = manager.get_all_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.command == 'task':
        if args.agent and args.task:
            payload = json.loads(args.payload) if args.payload else {}
            task_id = manager.submit_task(args.agent, args.task, payload)
            print(f"Task submitted: {task_id}")
        else:
            print("Requires --agent and --task")


if __name__ == "__main__":
    main()
