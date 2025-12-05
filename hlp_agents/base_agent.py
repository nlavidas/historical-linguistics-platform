"""
Base Agent - Foundation for all HLP agents

This module provides the base agent class and common functionality
for all agents in the Historical Linguistics Platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import uuid
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentConfig:
    name: str
    agent_type: str
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    batch_size: int = 10
    parallel_tasks: int = 1
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'agent_type': self.agent_type,
            'enabled': self.enabled,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'batch_size': self.batch_size,
            'parallel_tasks': self.parallel_tasks,
            'log_level': self.log_level,
            'custom_settings': self.custom_settings,
        }


@dataclass
class AgentTask:
    task_id: str
    task_type: str
    input_data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def create(task_type: str, input_data: Any, priority: TaskPriority = TaskPriority.NORMAL) -> AgentTask:
        return AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            input_data=input_data,
            priority=priority
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority.name,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retries': self.retries,
            'metadata': self.metadata,
        }


@dataclass
class AgentResult:
    task_id: str
    success: bool
    output_data: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'success': self.success,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'metadata': self.metadata,
        }


class BaseAgent(ABC):
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.task_queue: Queue[AgentTask] = Queue()
        self.results: Dict[str, AgentResult] = {}
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time_ms': 0,
            'start_time': None,
        }
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, List[Callable]] = {
            'on_task_start': [],
            'on_task_complete': [],
            'on_task_error': [],
            'on_status_change': [],
        }
        
        logger.info(f"Agent {self.config.name} ({self.config.agent_type}) initialized")
    
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def process_task(self, task: AgentTask) -> AgentResult:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
    
    def start(self):
        if self._running:
            logger.warning(f"Agent {self.config.name} is already running")
            return
        
        self._set_status(AgentStatus.INITIALIZING)
        
        if not self.initialize():
            self._set_status(AgentStatus.FAILED)
            logger.error(f"Agent {self.config.name} failed to initialize")
            return
        
        self._running = True
        self.stats['start_time'] = datetime.now()
        self._set_status(AgentStatus.READY)
        
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"Agent {self.config.name} started")
    
    def stop(self):
        if not self._running:
            return
        
        self._running = False
        self._set_status(AgentStatus.STOPPED)
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        
        self.cleanup()
        logger.info(f"Agent {self.config.name} stopped")
    
    def pause(self):
        if self.status == AgentStatus.RUNNING:
            self._set_status(AgentStatus.PAUSED)
            logger.info(f"Agent {self.config.name} paused")
    
    def resume(self):
        if self.status == AgentStatus.PAUSED:
            self._set_status(AgentStatus.READY)
            logger.info(f"Agent {self.config.name} resumed")
    
    def submit_task(self, task: AgentTask) -> str:
        self.task_queue.put(task)
        logger.debug(f"Task {task.task_id} submitted to agent {self.config.name}")
        return task.task_id
    
    def get_result(self, task_id: str) -> Optional[AgentResult]:
        return self.results.get(task_id)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'name': self.config.name,
            'type': self.config.agent_type,
            'status': self.status.value,
            'queue_size': self.task_queue.qsize(),
            'stats': self.stats,
        }
    
    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _set_status(self, status: AgentStatus):
        old_status = self.status
        self.status = status
        self._trigger_callbacks('on_status_change', old_status, status)
    
    def _trigger_callbacks(self, event: str, *args):
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def _worker_loop(self):
        while self._running:
            if self.status == AgentStatus.PAUSED:
                time.sleep(0.5)
                continue
            
            try:
                task = self.task_queue.get(timeout=1)
            except Empty:
                continue
            
            self._set_status(AgentStatus.RUNNING)
            task.started_at = datetime.now()
            task.status = "running"
            
            self._trigger_callbacks('on_task_start', task)
            
            start_time = time.time()
            retries = 0
            result = None
            
            while retries <= self.config.max_retries:
                try:
                    result = self.process_task(task)
                    break
                except Exception as e:
                    retries += 1
                    task.retries = retries
                    logger.error(f"Task {task.task_id} failed (attempt {retries}): {e}")
                    
                    if retries > self.config.max_retries:
                        result = AgentResult(
                            task_id=task.task_id,
                            success=False,
                            error_message=str(e),
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        break
                    
                    time.sleep(1)
            
            if result:
                result.execution_time_ms = (time.time() - start_time) * 1000
                self.results[task.task_id] = result
                
                task.completed_at = datetime.now()
                task.status = "completed" if result.success else "failed"
                
                if result.success:
                    self.stats['tasks_completed'] += 1
                    self._trigger_callbacks('on_task_complete', task, result)
                else:
                    self.stats['tasks_failed'] += 1
                    self._trigger_callbacks('on_task_error', task, result)
                
                self.stats['total_execution_time_ms'] += result.execution_time_ms
            
            self._set_status(AgentStatus.READY)
            self.task_queue.task_done()


class AIEngine(ABC):
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.is_loaded = False
        self.models: Dict[str, Any] = {}
    
    @abstractmethod
    def load(self, language: str = "en") -> bool:
        pass
    
    @abstractmethod
    def process(self, text: str, task_type: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def unload(self):
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'is_loaded': self.is_loaded,
            'loaded_models': list(self.models.keys()),
        }


class StanzaEngine(AIEngine):
    
    def __init__(self):
        super().__init__("Stanza", "1.6")
        self.nlp = None
    
    def load(self, language: str = "en") -> bool:
        try:
            import stanza
            stanza.download(language, verbose=False)
            self.nlp = stanza.Pipeline(language, processors='tokenize,pos,lemma,depparse,ner')
            self.models[language] = self.nlp
            self.is_loaded = True
            logger.info(f"Stanza loaded for {language}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Stanza: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'Stanza not loaded'}
        
        doc = self.nlp(text)
        
        result = {
            'sentences': [],
            'entities': [],
        }
        
        for sent in doc.sentences:
            sentence_data = {
                'text': sent.text,
                'tokens': []
            }
            
            for word in sent.words:
                token_data = {
                    'id': word.id,
                    'text': word.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'feats': word.feats,
                    'head': word.head,
                    'deprel': word.deprel,
                }
                sentence_data['tokens'].append(token_data)
            
            result['sentences'].append(sentence_data)
        
        for sent in doc.sentences:
            for ent in sent.ents:
                result['entities'].append({
                    'text': ent.text,
                    'type': ent.type,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                })
        
        return result
    
    def unload(self):
        self.nlp = None
        self.models.clear()
        self.is_loaded = False


class SpaCyEngine(AIEngine):
    
    def __init__(self):
        super().__init__("spaCy", "3.7")
        self.nlp = None
    
    def load(self, language: str = "en") -> bool:
        try:
            import spacy
            
            model_map = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'fr': 'fr_core_news_sm',
                'es': 'es_core_news_sm',
                'it': 'it_core_news_sm',
                'pt': 'pt_core_news_sm',
                'nl': 'nl_core_news_sm',
                'el': 'el_core_news_sm',
                'grc': 'grc_proiel_sm',
                'la': 'la_core_web_sm',
            }
            
            model_name = model_map.get(language, 'en_core_web_sm')
            
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=True)
                self.nlp = spacy.load(model_name)
            
            self.models[language] = self.nlp
            self.is_loaded = True
            logger.info(f"spaCy loaded for {language} ({model_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")
            return False
    
    def process(self, text: str, task_type: str = "full") -> Dict[str, Any]:
        if not self.is_loaded or not self.nlp:
            return {'error': 'spaCy not loaded'}
        
        doc = self.nlp(text)
        
        result = {
            'sentences': [],
            'entities': [],
        }
        
        for sent in doc.sents:
            sentence_data = {
                'text': sent.text,
                'tokens': []
            }
            
            for token in sent:
                token_data = {
                    'id': token.i - sent.start + 1,
                    'text': token.text,
                    'lemma': token.lemma_,
                    'upos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'head': token.head.i - sent.start + 1 if token.head != token else 0,
                    'is_stop': token.is_stop,
                    'is_punct': token.is_punct,
                }
                sentence_data['tokens'].append(token_data)
            
            result['sentences'].append(sentence_data)
        
        for ent in doc.ents:
            result['entities'].append({
                'text': ent.text,
                'type': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
            })
        
        return result
    
    def unload(self):
        self.nlp = None
        self.models.clear()
        self.is_loaded = False


class HuggingFaceEngine(AIEngine):
    
    def __init__(self):
        super().__init__("HuggingFace", "4.35")
        self.pipelines: Dict[str, Any] = {}
    
    def load(self, language: str = "en") -> bool:
        try:
            from transformers import pipeline
            
            self.pipelines['ner'] = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            
            self.pipelines['pos'] = pipeline(
                "token-classification",
                model="vblagoje/bert-english-uncased-finetuned-pos"
            )
            
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            self.is_loaded = True
            logger.info(f"HuggingFace pipelines loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load HuggingFace: {e}")
            return False
    
    def process(self, text: str, task_type: str = "ner") -> Dict[str, Any]:
        if not self.is_loaded:
            return {'error': 'HuggingFace not loaded'}
        
        result = {}
        
        if task_type == "ner" and 'ner' in self.pipelines:
            entities = self.pipelines['ner'](text)
            result['entities'] = [
                {
                    'text': ent['word'],
                    'type': ent['entity_group'],
                    'score': ent['score'],
                    'start': ent['start'],
                    'end': ent['end'],
                }
                for ent in entities
            ]
        
        if task_type == "pos" and 'pos' in self.pipelines:
            pos_tags = self.pipelines['pos'](text)
            result['pos_tags'] = [
                {
                    'word': tag['word'],
                    'tag': tag['entity'],
                    'score': tag['score'],
                }
                for tag in pos_tags
            ]
        
        if task_type == "sentiment" and 'sentiment' in self.pipelines:
            sentiment = self.pipelines['sentiment'](text)
            result['sentiment'] = sentiment[0] if sentiment else None
        
        return result
    
    def unload(self):
        self.pipelines.clear()
        self.is_loaded = False


class OllamaEngine(AIEngine):
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__("Ollama", "0.1")
        self.base_url = base_url
        self.model_name = None
    
    def load(self, language: str = "en") -> bool:
        try:
            import requests
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    self.model_name = models[0]['name']
                    self.is_loaded = True
                    logger.info(f"Ollama connected, using model: {self.model_name}")
                    return True
                else:
                    logger.warning("Ollama running but no models available")
                    return False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def process(self, text: str, task_type: str = "analyze") -> Dict[str, Any]:
        if not self.is_loaded or not self.model_name:
            return {'error': 'Ollama not loaded'}
        
        try:
            import requests
            
            prompts = {
                'analyze': f"Analyze the following text linguistically. Identify parts of speech, named entities, and key grammatical structures:\n\n{text}",
                'pos': f"Tag each word in the following text with its part of speech:\n\n{text}",
                'ner': f"Identify all named entities (persons, places, organizations, dates) in the following text:\n\n{text}",
                'valency': f"Identify all verbs and their argument structures (subject, object, indirect object, etc.) in the following text:\n\n{text}",
                'translate': f"Translate the following text to English:\n\n{text}",
            }
            
            prompt = prompts.get(task_type, prompts['analyze'])
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result.get('response', ''),
                    'model': self.model_name,
                    'task_type': task_type,
                }
            
            return {'error': f"Ollama request failed: {response.status_code}"}
            
        except Exception as e:
            return {'error': str(e)}
    
    def unload(self):
        self.model_name = None
        self.is_loaded = False


class EngineRegistry:
    
    _engines: Dict[str, type] = {
        'stanza': StanzaEngine,
        'spacy': SpaCyEngine,
        'huggingface': HuggingFaceEngine,
        'ollama': OllamaEngine,
    }
    
    @classmethod
    def get_engine(cls, name: str) -> Optional[AIEngine]:
        engine_class = cls._engines.get(name.lower())
        if engine_class:
            return engine_class()
        return None
    
    @classmethod
    def list_engines(cls) -> List[str]:
        return list(cls._engines.keys())
    
    @classmethod
    def register_engine(cls, name: str, engine_class: type):
        cls._engines[name.lower()] = engine_class
