"""
Agent Orchestrator - Multi-agent pipeline orchestration

This module provides orchestration for running multiple agents
in coordinated pipelines for complex linguistic analysis tasks.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import uuid
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from queue import Queue

from hlp_agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentTask,
    AgentResult,
    AgentStatus,
    TaskPriority,
)

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStep:
    name: str
    agent_type: str
    task_type: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    timeout_seconds: int = 300
    retry_count: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'agent_type': self.agent_type,
            'task_type': self.task_type,
            'output_key': self.output_key,
            'required': self.required,
            'timeout_seconds': self.timeout_seconds,
        }


@dataclass
class Pipeline:
    pipeline_id: str
    name: str
    description: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @staticmethod
    def create(name: str, description: str = "") -> Pipeline:
        return Pipeline(
            pipeline_id=str(uuid.uuid4()),
            name=name,
            description=description
        )
    
    def add_step(self, step: PipelineStep) -> Pipeline:
        self.steps.append(step)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'steps': [s.to_dict() for s in self.steps],
            'current_step': self.current_step,
            'total_steps': len(self.steps),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'errors': self.errors,
        }


class AgentOrchestrator:
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.running_pipelines: Dict[str, threading.Thread] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            'on_pipeline_start': [],
            'on_pipeline_complete': [],
            'on_pipeline_error': [],
            'on_step_start': [],
            'on_step_complete': [],
            'on_step_error': [],
        }
        self._lock = threading.Lock()
        
        logger.info("AgentOrchestrator initialized")
    
    def register_agent(self, agent_type: str, agent: BaseAgent):
        with self._lock:
            self.agents[agent_type] = agent
            logger.info(f"Registered agent: {agent_type}")
    
    def unregister_agent(self, agent_type: str):
        with self._lock:
            if agent_type in self.agents:
                agent = self.agents.pop(agent_type)
                agent.stop()
                logger.info(f"Unregistered agent: {agent_type}")
    
    def get_agent(self, agent_type: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_type)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': agent_type,
                'name': agent.config.name,
                'status': agent.status.value,
            }
            for agent_type, agent in self.agents.items()
        ]
    
    def create_pipeline(self, name: str, description: str = "") -> Pipeline:
        pipeline = Pipeline.create(name, description)
        self.pipelines[pipeline.pipeline_id] = pipeline
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        return self.pipelines.get(pipeline_id)
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.pipelines.values()]
    
    def run_pipeline(self, pipeline_id: str, initial_input: Dict[str, Any] = None) -> bool:
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False
        
        if pipeline.status == PipelineStatus.RUNNING:
            logger.warning(f"Pipeline {pipeline_id} is already running")
            return False
        
        thread = threading.Thread(
            target=self._execute_pipeline,
            args=(pipeline, initial_input or {}),
            daemon=True
        )
        
        self.running_pipelines[pipeline_id] = thread
        thread.start()
        
        return True
    
    def _execute_pipeline(self, pipeline: Pipeline, initial_input: Dict[str, Any]):
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = datetime.now()
        pipeline.results = {'input': initial_input}
        
        self._trigger_callbacks('on_pipeline_start', pipeline)
        
        try:
            for i, step in enumerate(pipeline.steps):
                pipeline.current_step = i
                
                self._trigger_callbacks('on_step_start', pipeline, step)
                
                agent = self.agents.get(step.agent_type)
                if not agent:
                    error_msg = f"Agent type '{step.agent_type}' not found"
                    logger.error(error_msg)
                    
                    if step.required:
                        pipeline.errors.append(error_msg)
                        pipeline.status = PipelineStatus.FAILED
                        self._trigger_callbacks('on_pipeline_error', pipeline, error_msg)
                        return
                    else:
                        continue
                
                if agent.status not in [AgentStatus.READY, AgentStatus.RUNNING]:
                    agent.start()
                    time.sleep(0.5)
                
                task_input = self._prepare_step_input(step, pipeline.results)
                
                task = AgentTask.create(step.task_type, task_input)
                task_id = agent.submit_task(task)
                
                start_time = time.time()
                result = None
                
                while time.time() - start_time < step.timeout_seconds:
                    result = agent.get_result(task_id)
                    if result:
                        break
                    time.sleep(0.1)
                
                if not result:
                    error_msg = f"Step '{step.name}' timed out"
                    logger.error(error_msg)
                    
                    if step.required:
                        pipeline.errors.append(error_msg)
                        pipeline.status = PipelineStatus.FAILED
                        self._trigger_callbacks('on_step_error', pipeline, step, error_msg)
                        self._trigger_callbacks('on_pipeline_error', pipeline, error_msg)
                        return
                    else:
                        self._trigger_callbacks('on_step_error', pipeline, step, error_msg)
                        continue
                
                if not result.success:
                    error_msg = f"Step '{step.name}' failed: {result.error_message}"
                    logger.error(error_msg)
                    
                    if step.required:
                        pipeline.errors.append(error_msg)
                        pipeline.status = PipelineStatus.FAILED
                        self._trigger_callbacks('on_step_error', pipeline, step, error_msg)
                        self._trigger_callbacks('on_pipeline_error', pipeline, error_msg)
                        return
                    else:
                        self._trigger_callbacks('on_step_error', pipeline, step, error_msg)
                        continue
                
                output_key = step.output_key or step.name
                pipeline.results[output_key] = result.output_data
                
                self._trigger_callbacks('on_step_complete', pipeline, step, result)
                
                logger.info(f"Pipeline {pipeline.name}: Step '{step.name}' completed")
            
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.completed_at = datetime.now()
            
            self._trigger_callbacks('on_pipeline_complete', pipeline)
            
            logger.info(f"Pipeline {pipeline.name} completed successfully")
            
        except Exception as e:
            error_msg = f"Pipeline execution error: {str(e)}"
            logger.error(error_msg)
            pipeline.errors.append(error_msg)
            pipeline.status = PipelineStatus.FAILED
            self._trigger_callbacks('on_pipeline_error', pipeline, error_msg)
    
    def _prepare_step_input(self, step: PipelineStep, results: Dict[str, Any]) -> Dict[str, Any]:
        task_input = {}
        
        for target_key, source_path in step.input_mapping.items():
            value = self._get_nested_value(results, source_path)
            if value is not None:
                task_input[target_key] = value
        
        task_input.update(step.config)
        
        return task_input
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def pause_pipeline(self, pipeline_id: str) -> bool:
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline and pipeline.status == PipelineStatus.RUNNING:
            pipeline.status = PipelineStatus.PAUSED
            return True
        return False
    
    def resume_pipeline(self, pipeline_id: str) -> bool:
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline and pipeline.status == PipelineStatus.PAUSED:
            pipeline.status = PipelineStatus.RUNNING
            return True
        return False
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        pipeline = self.pipelines.get(pipeline_id)
        if pipeline:
            pipeline.status = PipelineStatus.CANCELLED
            if pipeline_id in self.running_pipelines:
                del self.running_pipelines[pipeline_id]
            return True
        return False
    
    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args):
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def shutdown(self):
        for pipeline_id in list(self.running_pipelines.keys()):
            self.cancel_pipeline(pipeline_id)
        
        for agent in self.agents.values():
            agent.stop()
        
        self.agents.clear()
        logger.info("AgentOrchestrator shutdown complete")


class PipelineBuilder:
    
    @staticmethod
    def create_annotation_pipeline(name: str = "Annotation Pipeline") -> Pipeline:
        pipeline = Pipeline.create(name, "Full annotation pipeline with multiple engines")
        
        pipeline.add_step(PipelineStep(
            name="tokenize",
            agent_type="annotation",
            task_type="annotate",
            input_mapping={'text': 'input.text'},
            output_key="annotation",
        ))
        
        pipeline.add_step(PipelineStep(
            name="extract_valency",
            agent_type="valency",
            task_type="extract_from_parsed",
            input_mapping={'parsed': 'annotation.annotation'},
            output_key="valency",
        ))
        
        return pipeline
    
    @staticmethod
    def create_diachronic_analysis_pipeline(name: str = "Diachronic Analysis") -> Pipeline:
        pipeline = Pipeline.create(name, "Diachronic change detection and analysis")
        
        pipeline.add_step(PipelineStep(
            name="annotate_texts",
            agent_type="annotation",
            task_type="annotate_batch",
            input_mapping={'texts': 'input.texts'},
            output_key="annotations",
        ))
        
        pipeline.add_step(PipelineStep(
            name="add_to_periods",
            agent_type="diachronic",
            task_type="add_text",
            input_mapping={
                'tokens': 'annotations.annotations',
                'period': 'input.period',
            },
            output_key="period_data",
        ))
        
        pipeline.add_step(PipelineStep(
            name="detect_changes",
            agent_type="diachronic",
            task_type="detect_changes",
            config={'threshold': 0.1},
            output_key="changes",
        ))
        
        return pipeline
    
    @staticmethod
    def create_valency_lexicon_pipeline(name: str = "Valency Lexicon Builder") -> Pipeline:
        pipeline = Pipeline.create(name, "Build valency lexicon from corpus")
        
        pipeline.add_step(PipelineStep(
            name="annotate",
            agent_type="annotation",
            task_type="annotate_batch",
            input_mapping={'texts': 'input.texts'},
            output_key="annotations",
        ))
        
        pipeline.add_step(PipelineStep(
            name="extract_patterns",
            agent_type="valency",
            task_type="extract_from_parsed",
            input_mapping={'parsed': 'annotations.annotations'},
            output_key="patterns",
        ))
        
        pipeline.add_step(PipelineStep(
            name="build_lexicon",
            agent_type="valency",
            task_type="build_lexicon",
            config={'min_frequency': 2},
            output_key="lexicon",
        ))
        
        return pipeline
    
    @staticmethod
    def create_full_corpus_pipeline(name: str = "Full Corpus Processing") -> Pipeline:
        pipeline = Pipeline.create(name, "Complete corpus processing pipeline")
        
        pipeline.add_step(PipelineStep(
            name="annotate",
            agent_type="annotation",
            task_type="annotate_batch",
            input_mapping={'texts': 'input.texts'},
            output_key="annotations",
        ))
        
        pipeline.add_step(PipelineStep(
            name="valency_extraction",
            agent_type="valency",
            task_type="extract_from_parsed",
            input_mapping={'parsed': 'annotations.annotations'},
            output_key="valency",
        ))
        
        pipeline.add_step(PipelineStep(
            name="diachronic_analysis",
            agent_type="diachronic",
            task_type="detect_changes",
            config={'threshold': 0.1},
            output_key="diachronic",
            required=False,
        ))
        
        pipeline.add_step(PipelineStep(
            name="build_lexicon",
            agent_type="valency",
            task_type="build_lexicon",
            config={'min_frequency': 1},
            output_key="lexicon",
        ))
        
        return pipeline


def create_default_orchestrator() -> AgentOrchestrator:
    from hlp_agents.annotation_agent import AnnotationAgent, AnnotationConfig
    from hlp_agents.valency_agent import ValencyAgent
    from hlp_agents.diachronic_agent import DiachronicAgent
    
    orchestrator = AgentOrchestrator()
    
    annotation_agent = AnnotationAgent(
        AgentConfig(name="AnnotationAgent", agent_type="annotation"),
        AnnotationConfig(engines=['stanza'])
    )
    orchestrator.register_agent("annotation", annotation_agent)
    
    valency_agent = ValencyAgent(
        AgentConfig(name="ValencyAgent", agent_type="valency")
    )
    orchestrator.register_agent("valency", valency_agent)
    
    diachronic_agent = DiachronicAgent(
        AgentConfig(
            name="DiachronicAgent",
            agent_type="diachronic",
            custom_settings={'language': 'grc'}
        )
    )
    orchestrator.register_agent("diachronic", diachronic_agent)
    
    return orchestrator
