#!/usr/bin/env python3
"""
PROIEL WINDSURF - The "Windsurf" of Indo-European Historical Linguistics
A comprehensive platform for PROIEL-style corpus analysis

This is the central orchestration system that:
1. Manages multiple AI agents working in parallel
2. Provides real-time corpus analysis
3. Integrates all linguistic tools
4. Supports autonomous research workflows

Inspired by: Windsurf IDE's agentic capabilities
Applied to: Indo-European historical linguistics and PROIEL corpora
"""

import os
import sys
import json
import time
import queue
import sqlite3
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum, auto
import hashlib
import re
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration"""
    
    # Paths
    BASE_DIR = Path("/root/corpus_platform")
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = DATA_DIR / "cache"
    MODELS_DIR = DATA_DIR / "models"
    EXPORTS_DIR = DATA_DIR / "exports"
    
    # Database
    MAIN_DB = DATA_DIR / "corpus_platform.db"
    VALENCY_DB = DATA_DIR / "valency_lexicon.db"
    AGENTS_DB = DATA_DIR / "agents.db"
    
    # PROIEL Treebanks
    PROIEL_LANGUAGES = {
        'grc': 'Ancient Greek',
        'la': 'Latin', 
        'got': 'Gothic',
        'cu': 'Old Church Slavonic',
        'xcl': 'Classical Armenian',
        'orv': 'Old Russian',
        'ang': 'Old English',
        'non': 'Old Norse',
        'por': 'Portuguese',
        'spa': 'Spanish'
    }
    
    # Historical periods
    PERIODS = {
        'proto_ie': (-4000, -2000),
        'archaic': (-800, -500),
        'classical': (-500, -300),
        'hellenistic': (-300, 0),
        'koine': (0, 300),
        'late_antique': (300, 600),
        'medieval': (600, 1400),
        'early_modern': (1400, 1700),
        'modern': (1700, 2025)
    }
    
    # Agent configuration
    MAX_AGENTS = 10
    AGENT_TIMEOUT = 300  # seconds
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all directories exist"""
        for d in [cls.DATA_DIR, cls.CACHE_DIR, cls.MODELS_DIR, cls.EXPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TASK SYSTEM
# =============================================================================

class TaskStatus(Enum):
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """A task to be executed by an agent"""
    id: str
    name: str
    task_type: str
    params: Dict
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value > other.priority.value

@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    duration: float = 0.0
    metrics: Dict = field(default_factory=dict)


# =============================================================================
# AGENT SYSTEM
# =============================================================================

class AgentCapability(Enum):
    """Agent capabilities"""
    CORPUS_COLLECTION = auto()
    TEXT_PROCESSING = auto()
    MORPHOLOGICAL_ANALYSIS = auto()
    SYNTACTIC_PARSING = auto()
    VALENCY_EXTRACTION = auto()
    SEMANTIC_ANALYSIS = auto()
    DIACHRONIC_ANALYSIS = auto()
    STATISTICAL_ANALYSIS = auto()
    VISUALIZATION = auto()
    EXPORT = auto()

@dataclass
class AgentInfo:
    """Information about an agent"""
    id: str
    name: str
    capabilities: List[AgentCapability]
    status: str = "idle"
    current_task: Optional[str] = None
    tasks_completed: int = 0
    errors: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability]):
        self.info = AgentInfo(
            id=agent_id,
            name=name,
            capabilities=capabilities
        )
        self.running = False
        self.task_queue = queue.PriorityQueue()
        self._thread: Optional[threading.Thread] = None
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle a task type"""
        capability_map = {
            'collect': AgentCapability.CORPUS_COLLECTION,
            'process': AgentCapability.TEXT_PROCESSING,
            'morphology': AgentCapability.MORPHOLOGICAL_ANALYSIS,
            'parse': AgentCapability.SYNTACTIC_PARSING,
            'valency': AgentCapability.VALENCY_EXTRACTION,
            'semantic': AgentCapability.SEMANTIC_ANALYSIS,
            'diachronic': AgentCapability.DIACHRONIC_ANALYSIS,
            'statistics': AgentCapability.STATISTICAL_ANALYSIS,
            'visualize': AgentCapability.VISUALIZATION,
            'export': AgentCapability.EXPORT
        }
        
        required = capability_map.get(task_type.split('_')[0])
        return required in self.info.capabilities if required else False
    
    def start(self):
        """Start the agent"""
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Agent {self.info.name} started")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Agent {self.info.name} stopped")
    
    def submit_task(self, task: Task):
        """Submit a task to this agent"""
        self.task_queue.put(task)
    
    def _run_loop(self):
        """Main agent loop"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                self._execute_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Agent {self.info.name} error: {e}")
    
    def _execute_task(self, task: Task):
        """Execute a task"""
        self.info.status = "running"
        self.info.current_task = task.id
        self.info.last_active = datetime.now()
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            result = self.execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            self.info.tasks_completed += 1
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self.info.errors += 1
            logger.error(f"Task {task.id} failed: {e}")
        
        task.completed_at = datetime.now()
        self.info.status = "idle"
        self.info.current_task = None
    
    def execute(self, task: Task) -> Dict:
        """Execute task - override in subclasses"""
        raise NotImplementedError


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

class CorpusCollectorAgent(BaseAgent):
    """Agent for collecting corpus data"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="CorpusCollector",
            capabilities=[AgentCapability.CORPUS_COLLECTION]
        )
        self.session = None
    
    def execute(self, task: Task) -> Dict:
        """Execute collection task"""
        import requests
        
        if self.session is None:
            self.session = requests.Session()
        
        task_type = task.task_type
        params = task.params
        
        if task_type == "collect_ud_treebank":
            return self._collect_ud(params)
        elif task_type == "collect_proiel":
            return self._collect_proiel(params)
        elif task_type == "collect_perseus":
            return self._collect_perseus(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _collect_ud(self, params: Dict) -> Dict:
        """Collect Universal Dependencies treebank"""
        treebank = params.get('treebank', 'grc_proiel')
        base_url = "https://raw.githubusercontent.com/UniversalDependencies"
        
        results = {'sentences': 0, 'tokens': 0, 'files': []}
        
        for split in ['train', 'dev', 'test']:
            filename = f"{treebank}-ud-{split}.conllu"
            url = f"{base_url}/UD_{treebank.replace('_', '-').title()}/master/{filename}"
            
            try:
                response = self.session.get(url, timeout=120)
                if response.status_code == 200:
                    # Count sentences and tokens
                    content = response.text
                    sentences = content.count('\n\n')
                    tokens = len([l for l in content.split('\n') 
                                 if l and not l.startswith('#')])
                    
                    results['sentences'] += sentences
                    results['tokens'] += tokens
                    results['files'].append(filename)
            except Exception as e:
                logger.warning(f"Failed to collect {filename}: {e}")
        
        return results
    
    def _collect_proiel(self, params: Dict) -> Dict:
        """Collect from PROIEL treebanks"""
        # PROIEL XML format collection
        return {'status': 'not_implemented'}
    
    def _collect_perseus(self, params: Dict) -> Dict:
        """Collect from Perseus Digital Library"""
        return {'status': 'not_implemented'}


class MorphologyAgent(BaseAgent):
    """Agent for morphological analysis"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            name="MorphologyAnalyzer",
            capabilities=[AgentCapability.MORPHOLOGICAL_ANALYSIS]
        )
    
    def execute(self, task: Task) -> Dict:
        """Execute morphology task"""
        task_type = task.task_type
        params = task.params
        
        if task_type == "morphology_analyze":
            return self._analyze(params)
        elif task_type == "morphology_paradigm":
            return self._generate_paradigm(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _analyze(self, params: Dict) -> Dict:
        """Analyze morphology of a word"""
        word = params.get('word', '')
        language = params.get('language', 'grc')
        
        # Greek morphological patterns
        analysis = {
            'form': word,
            'language': language,
            'analyses': []
        }
        
        # Verb patterns
        verb_endings = {
            'ω': {'person': '1', 'number': 'sg', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
            'εις': {'person': '2', 'number': 'sg', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
            'ει': {'person': '3', 'number': 'sg', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
            'ομεν': {'person': '1', 'number': 'pl', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
            'ετε': {'person': '2', 'number': 'pl', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
            'ουσι': {'person': '3', 'number': 'pl', 'tense': 'pres', 'mood': 'ind', 'voice': 'act'},
        }
        
        for ending, feats in verb_endings.items():
            if word.endswith(ending):
                stem = word[:-len(ending)]
                analysis['analyses'].append({
                    'pos': 'VERB',
                    'stem': stem,
                    'features': feats
                })
        
        # Noun patterns
        noun_endings = {
            'ος': {'case': 'nom', 'number': 'sg', 'gender': 'masc'},
            'ου': {'case': 'gen', 'number': 'sg', 'gender': 'masc'},
            'ῳ': {'case': 'dat', 'number': 'sg', 'gender': 'masc'},
            'ον': {'case': 'acc', 'number': 'sg', 'gender': 'masc'},
            'οι': {'case': 'nom', 'number': 'pl', 'gender': 'masc'},
            'ων': {'case': 'gen', 'number': 'pl', 'gender': 'masc'},
        }
        
        for ending, feats in noun_endings.items():
            if word.endswith(ending):
                stem = word[:-len(ending)]
                analysis['analyses'].append({
                    'pos': 'NOUN',
                    'stem': stem,
                    'features': feats
                })
        
        return analysis
    
    def _generate_paradigm(self, params: Dict) -> Dict:
        """Generate full paradigm for a lemma"""
        lemma = params.get('lemma', '')
        pos = params.get('pos', 'NOUN')
        
        paradigm = {'lemma': lemma, 'pos': pos, 'forms': []}
        
        if pos == 'NOUN':
            # O-stem masculine paradigm
            stem = lemma[:-2] if lemma.endswith('ος') else lemma
            
            cases = [
                ('nom', 'sg', stem + 'ος'),
                ('gen', 'sg', stem + 'ου'),
                ('dat', 'sg', stem + 'ῳ'),
                ('acc', 'sg', stem + 'ον'),
                ('voc', 'sg', stem + 'ε'),
                ('nom', 'pl', stem + 'οι'),
                ('gen', 'pl', stem + 'ων'),
                ('dat', 'pl', stem + 'οις'),
                ('acc', 'pl', stem + 'ους'),
                ('voc', 'pl', stem + 'οι'),
            ]
            
            for case, number, form in cases:
                paradigm['forms'].append({
                    'case': case,
                    'number': number,
                    'form': form
                })
        
        return paradigm


class ValencyAgent(BaseAgent):
    """Agent for valency analysis"""
    
    def __init__(self, agent_id: str, db_path: str):
        super().__init__(
            agent_id=agent_id,
            name="ValencyAnalyzer",
            capabilities=[
                AgentCapability.VALENCY_EXTRACTION,
                AgentCapability.SEMANTIC_ANALYSIS
            ]
        )
        self.db_path = db_path
    
    def execute(self, task: Task) -> Dict:
        """Execute valency task"""
        task_type = task.task_type
        params = task.params
        
        if task_type == "valency_extract":
            return self._extract_valency(params)
        elif task_type == "valency_compare":
            return self._compare_valency(params)
        elif task_type == "valency_diachronic":
            return self._diachronic_valency(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _extract_valency(self, params: Dict) -> Dict:
        """Extract valency patterns from corpus"""
        lemma = params.get('lemma', '')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all instances of the verb
        cursor.execute("""
            SELECT t.sentence_id, t.token_index, s.text
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            WHERE t.lemma = ? AND t.upos = 'VERB'
        """, (lemma,))
        
        instances = cursor.fetchall()
        patterns = Counter()
        
        for sent_id, tok_idx, sent_text in instances:
            # Get dependents
            cursor.execute("""
                SELECT deprel, feats FROM tokens
                WHERE sentence_id = ? AND head = ?
            """, (sent_id, tok_idx))
            
            deps = cursor.fetchall()
            
            # Build pattern
            args = ['NOM']  # Subject implied
            for deprel, feats in deps:
                if deprel == 'obj':
                    args.append('ACC')
                elif deprel == 'iobj':
                    args.append('DAT')
                elif deprel == 'obl':
                    # Extract case
                    if feats and 'Case=' in feats:
                        case = re.search(r'Case=(\w+)', feats)
                        if case:
                            args.append(case.group(1).upper()[:3])
            
            pattern = '+'.join(sorted(set(args)))
            patterns[pattern] += 1
        
        conn.close()
        
        return {
            'lemma': lemma,
            'total_instances': len(instances),
            'patterns': dict(patterns.most_common())
        }
    
    def _compare_valency(self, params: Dict) -> Dict:
        """Compare valency of two verbs"""
        lemma1 = params.get('lemma1', '')
        lemma2 = params.get('lemma2', '')
        
        v1 = self._extract_valency({'lemma': lemma1})
        v2 = self._extract_valency({'lemma': lemma2})
        
        patterns1 = set(v1['patterns'].keys())
        patterns2 = set(v2['patterns'].keys())
        
        return {
            'lemma1': lemma1,
            'lemma2': lemma2,
            'shared_patterns': list(patterns1 & patterns2),
            'unique_to_1': list(patterns1 - patterns2),
            'unique_to_2': list(patterns2 - patterns1)
        }
    
    def _diachronic_valency(self, params: Dict) -> Dict:
        """Analyze valency changes across periods"""
        lemma = params.get('lemma', '')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get patterns by period
        cursor.execute("""
            SELECT d.period, t.sentence_id, t.token_index
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            JOIN documents d ON s.document_id = d.id
            WHERE t.lemma = ? AND t.upos = 'VERB'
        """, (lemma,))
        
        by_period = defaultdict(lambda: Counter())
        
        for period, sent_id, tok_idx in cursor.fetchall():
            cursor.execute("""
                SELECT deprel, feats FROM tokens
                WHERE sentence_id = ? AND head = ?
            """, (sent_id, tok_idx))
            
            deps = cursor.fetchall()
            args = ['NOM']
            
            for deprel, feats in deps:
                if deprel == 'obj':
                    args.append('ACC')
                elif deprel == 'iobj':
                    args.append('DAT')
            
            pattern = '+'.join(sorted(set(args)))
            by_period[period][pattern] += 1
        
        conn.close()
        
        return {
            'lemma': lemma,
            'by_period': {p: dict(c) for p, c in by_period.items()}
        }


class DiachronicAgent(BaseAgent):
    """Agent for diachronic analysis"""
    
    def __init__(self, agent_id: str, db_path: str):
        super().__init__(
            agent_id=agent_id,
            name="DiachronicAnalyzer",
            capabilities=[
                AgentCapability.DIACHRONIC_ANALYSIS,
                AgentCapability.STATISTICAL_ANALYSIS
            ]
        )
        self.db_path = db_path
    
    def execute(self, task: Task) -> Dict:
        """Execute diachronic task"""
        task_type = task.task_type
        params = task.params
        
        if task_type == "diachronic_frequency":
            return self._frequency_trend(params)
        elif task_type == "diachronic_change":
            return self._detect_change(params)
        elif task_type == "diachronic_compare":
            return self._compare_periods(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _frequency_trend(self, params: Dict) -> Dict:
        """Get frequency trend across periods"""
        lemma = params.get('lemma', '')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get frequencies by period
        cursor.execute("""
            SELECT d.period, COUNT(*) as freq
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            JOIN documents d ON s.document_id = d.id
            WHERE t.lemma = ?
            GROUP BY d.period
        """, (lemma,))
        
        raw_freqs = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get period sizes
        cursor.execute("""
            SELECT d.period, COUNT(*) as size
            FROM tokens t
            JOIN sentences s ON t.sentence_id = s.id
            JOIN documents d ON s.document_id = d.id
            GROUP BY d.period
        """)
        
        period_sizes = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Normalize
        normalized = {}
        for period, freq in raw_freqs.items():
            size = period_sizes.get(period, 1)
            normalized[period] = {
                'raw': freq,
                'per_million': (freq / size) * 1000000,
                'corpus_size': size
            }
        
        return {
            'lemma': lemma,
            'trend': normalized
        }
    
    def _detect_change(self, params: Dict) -> Dict:
        """Detect significant changes"""
        lemma = params.get('lemma', '')
        threshold = params.get('threshold', 2.0)
        
        trend = self._frequency_trend({'lemma': lemma})['trend']
        
        changes = []
        periods = sorted(trend.keys())
        
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i+1]
            f1 = trend[p1]['per_million']
            f2 = trend[p2]['per_million']
            
            if f1 > 0:
                ratio = f2 / f1
                if ratio >= threshold or ratio <= 1/threshold:
                    changes.append({
                        'from': p1,
                        'to': p2,
                        'ratio': ratio,
                        'direction': 'increase' if ratio > 1 else 'decrease'
                    })
        
        return {
            'lemma': lemma,
            'changes': changes
        }
    
    def _compare_periods(self, params: Dict) -> Dict:
        """Compare two periods"""
        period1 = params.get('period1', 'classical')
        period2 = params.get('period2', 'modern')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get top lemmas in each period
        results = {}
        
        for period in [period1, period2]:
            cursor.execute("""
                SELECT t.lemma, COUNT(*) as freq
                FROM tokens t
                JOIN sentences s ON t.sentence_id = s.id
                JOIN documents d ON s.document_id = d.id
                WHERE d.period = ?
                GROUP BY t.lemma
                ORDER BY freq DESC
                LIMIT 100
            """, (period,))
            
            results[period] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        # Find differences
        lemmas1 = set(results[period1].keys())
        lemmas2 = set(results[period2].keys())
        
        return {
            'period1': period1,
            'period2': period2,
            'shared': len(lemmas1 & lemmas2),
            'unique_to_1': len(lemmas1 - lemmas2),
            'unique_to_2': len(lemmas2 - lemmas1)
        }


class StatisticsAgent(BaseAgent):
    """Agent for statistical analysis"""
    
    def __init__(self, agent_id: str, db_path: str):
        super().__init__(
            agent_id=agent_id,
            name="StatisticsAnalyzer",
            capabilities=[AgentCapability.STATISTICAL_ANALYSIS]
        )
        self.db_path = db_path
    
    def execute(self, task: Task) -> Dict:
        """Execute statistics task"""
        task_type = task.task_type
        params = task.params
        
        if task_type == "statistics_corpus":
            return self._corpus_stats(params)
        elif task_type == "statistics_bootstrap":
            return self._bootstrap(params)
        elif task_type == "statistics_chi_square":
            return self._chi_square(params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _corpus_stats(self, params: Dict) -> Dict:
        """Get corpus statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sentences")
        stats['sentences'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tokens")
        stats['tokens'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT lemma) FROM tokens")
        stats['lemmas'] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT upos, COUNT(*) FROM tokens
            GROUP BY upos ORDER BY COUNT(*) DESC
        """)
        stats['pos_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats
    
    def _bootstrap(self, params: Dict) -> Dict:
        """Bootstrap confidence interval"""
        import random
        
        data = params.get('data', [])
        n_iterations = params.get('n_iterations', 1000)
        confidence = params.get('confidence', 0.95)
        
        if not data:
            return {'error': 'No data provided'}
        
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_iterations):
            sample = [random.choice(data) for _ in range(n)]
            bootstrap_means.append(sum(sample) / len(sample))
        
        bootstrap_means.sort()
        
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_iterations)
        upper_idx = int((1 - alpha / 2) * n_iterations)
        
        return {
            'mean': sum(data) / n,
            'ci_lower': bootstrap_means[lower_idx],
            'ci_upper': bootstrap_means[upper_idx],
            'n_iterations': n_iterations
        }
    
    def _chi_square(self, params: Dict) -> Dict:
        """Chi-square test"""
        observed = params.get('observed', [])
        expected = params.get('expected', [])
        
        if len(observed) != len(expected):
            return {'error': 'Observed and expected must have same length'}
        
        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
        df = len(observed) - 1
        
        return {
            'chi_square': chi2,
            'degrees_of_freedom': df
        }


# =============================================================================
# WINDSURF ORCHESTRATOR
# =============================================================================

class PROIELWindsurf:
    """
    The main orchestrator - the "Windsurf" of Indo-European linguistics
    Coordinates all agents and manages workflows
    """
    
    def __init__(self, db_path: str = None):
        Config.ensure_dirs()
        
        self.db_path = db_path or str(Config.MAIN_DB)
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks: List[str] = []
        
        # Agent pool
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()
        
        # Workflow management
        self.workflows: Dict[str, Dict] = {}
        
        # Running state
        self.running = False
        self._orchestrator_thread: Optional[threading.Thread] = None
        
        logger.info("PROIEL Windsurf initialized")
    
    def _init_agents(self):
        """Initialize the agent pool"""
        # Create specialized agents
        self.agents['collector'] = CorpusCollectorAgent('agent_collector')
        self.agents['morphology'] = MorphologyAgent('agent_morphology')
        self.agents['valency'] = ValencyAgent('agent_valency', self.db_path)
        self.agents['diachronic'] = DiachronicAgent('agent_diachronic', self.db_path)
        self.agents['statistics'] = StatisticsAgent('agent_statistics', self.db_path)
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        
        # Start all agents
        for agent in self.agents.values():
            agent.start()
        
        # Start orchestrator thread
        self._orchestrator_thread = threading.Thread(
            target=self._orchestrate_loop, daemon=True
        )
        self._orchestrator_thread.start()
        
        logger.info("PROIEL Windsurf started")
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        if self._orchestrator_thread:
            self._orchestrator_thread.join(timeout=5)
        
        logger.info("PROIEL Windsurf stopped")
    
    def submit_task(self, name: str, task_type: str, params: Dict,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    dependencies: List[str] = None) -> str:
        """Submit a task"""
        task_id = hashlib.md5(
            f"{name}_{task_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        task = Task(
            id=task_id,
            name=name,
            task_type=task_type,
            params=params,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        self.task_queue.put(task)
        
        logger.info(f"Task submitted: {task_id} ({name})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            'id': task.id,
            'name': task.name,
            'status': task.status.name,
            'result': task.result,
            'error': task.error
        }
    
    def _orchestrate_loop(self):
        """Main orchestration loop"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                self._dispatch_task(task)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Orchestration error: {e}")
    
    def _dispatch_task(self, task: Task):
        """Dispatch task to appropriate agent"""
        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                # Re-queue task
                task.status = TaskStatus.QUEUED
                self.task_queue.put(task)
                return
        
        # Find capable agent
        for agent in self.agents.values():
            if agent.can_handle(task.task_type) and agent.info.status == "idle":
                task.agent_id = agent.info.id
                agent.submit_task(task)
                return
        
        # No available agent, re-queue
        self.task_queue.put(task)
    
    # =========================================================================
    # HIGH-LEVEL WORKFLOWS
    # =========================================================================
    
    def analyze_verb(self, lemma: str) -> Dict:
        """Complete analysis of a verb"""
        results = {}
        
        # Valency analysis
        valency_task = self.submit_task(
            f"Valency analysis: {lemma}",
            "valency_extract",
            {'lemma': lemma},
            TaskPriority.HIGH
        )
        
        # Diachronic analysis
        diachronic_task = self.submit_task(
            f"Diachronic analysis: {lemma}",
            "diachronic_frequency",
            {'lemma': lemma},
            TaskPriority.HIGH
        )
        
        # Wait for results
        time.sleep(2)
        
        results['valency'] = self.get_task_status(valency_task)
        results['diachronic'] = self.get_task_status(diachronic_task)
        
        return results
    
    def corpus_overview(self) -> Dict:
        """Get complete corpus overview"""
        task_id = self.submit_task(
            "Corpus statistics",
            "statistics_corpus",
            {},
            TaskPriority.HIGH
        )
        
        time.sleep(1)
        return self.get_task_status(task_id)
    
    def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            'running': self.running,
            'agents': {
                name: {
                    'status': agent.info.status,
                    'tasks_completed': agent.info.tasks_completed,
                    'errors': agent.info.errors
                }
                for name, agent in self.agents.items()
            },
            'pending_tasks': self.task_queue.qsize(),
            'total_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks)
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PROIEL Windsurf')
    parser.add_argument('--db', type=str, default='/root/corpus_platform/data/corpus_platform.db')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROIEL WINDSURF - Indo-European Historical Linguistics Platform")
    print("=" * 70)
    
    windsurf = PROIELWindsurf(args.db)
    windsurf.start()
    
    if args.test:
        print("\n--- Running test ---")
        
        # Get corpus overview
        overview = windsurf.corpus_overview()
        print(f"Corpus overview: {overview}")
        
        # Analyze a verb
        analysis = windsurf.analyze_verb("λέγω")
        print(f"Verb analysis: {analysis}")
        
        time.sleep(3)
        print(f"\nStatus: {windsurf.get_status()}")
    else:
        print("\nWindsurf running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    windsurf.stop()
    print("\nWindsurf stopped.")
