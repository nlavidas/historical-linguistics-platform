"""
HLP Pipelines Cron Jobs - Scheduled Job Definitions

This module provides utilities for scheduling and managing
recurring jobs for the platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger(__name__)


class JobFrequency(Enum):
    """Predefined job frequencies"""
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class JobStatus(Enum):
    """Status of a scheduled job"""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    RUNNING = "running"
    ERROR = "error"


class JobType(Enum):
    """Types of jobs"""
    INGEST = "ingest"
    ANNOTATION = "annotation"
    VALENCY = "valency"
    BACKUP = "backup"
    CLEANUP = "cleanup"
    REPORT = "report"
    CUSTOM = "custom"


@dataclass
class JobSchedule:
    """Schedule definition for a job"""
    frequency: JobFrequency = JobFrequency.DAILY
    
    cron_expression: Optional[str] = None
    
    interval_seconds: Optional[int] = None
    
    hour: int = 0
    minute: int = 0
    
    day_of_week: Optional[int] = None
    
    day_of_month: Optional[int] = None
    
    timezone: str = "UTC"
    
    def get_next_run(self, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time"""
        now = from_time or datetime.now()
        
        if self.interval_seconds:
            return now + timedelta(seconds=self.interval_seconds)
        
        if self.frequency == JobFrequency.MINUTELY:
            return now + timedelta(minutes=1)
        
        elif self.frequency == JobFrequency.HOURLY:
            next_run = now.replace(minute=self.minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            return next_run
        
        elif self.frequency == JobFrequency.DAILY:
            next_run = now.replace(
                hour=self.hour, minute=self.minute,
                second=0, microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif self.frequency == JobFrequency.WEEKLY:
            next_run = now.replace(
                hour=self.hour, minute=self.minute,
                second=0, microsecond=0
            )
            days_ahead = (self.day_of_week or 0) - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run += timedelta(days=days_ahead)
            return next_run
        
        elif self.frequency == JobFrequency.MONTHLY:
            next_run = now.replace(
                day=self.day_of_month or 1,
                hour=self.hour, minute=self.minute,
                second=0, microsecond=0
            )
            if next_run <= now:
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            return next_run
        
        return now + timedelta(days=1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "frequency": self.frequency.value,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "hour": self.hour,
            "minute": self.minute,
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "timezone": self.timezone
        }


@dataclass
class JobExecution:
    """Record of a job execution"""
    id: str
    job_id: str
    
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    success: bool = False
    
    error_message: Optional[str] = None
    
    result: Optional[Any] = None
    
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class CronJob:
    """A scheduled job"""
    id: str
    name: str
    
    job_type: JobType
    
    schedule: JobSchedule
    
    handler: Optional[Callable] = None
    
    handler_args: Dict[str, Any] = field(default_factory=dict)
    
    status: JobStatus = JobStatus.ACTIVE
    
    created_at: datetime = field(default_factory=datetime.now)
    
    last_run: Optional[datetime] = None
    
    next_run: Optional[datetime] = None
    
    run_count: int = 0
    
    success_count: int = 0
    
    failure_count: int = 0
    
    max_retries: int = 3
    
    retry_delay: float = 60.0
    
    timeout: float = 3600.0
    
    executions: List[JobExecution] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_next_run(self):
        """Calculate next run time"""
        self.next_run = self.schedule.get_next_run(self.last_run)
    
    def should_run(self) -> bool:
        """Check if job should run now"""
        if self.status != JobStatus.ACTIVE:
            return False
        
        if self.next_run is None:
            self.calculate_next_run()
        
        return datetime.now() >= self.next_run
    
    def record_execution(self, execution: JobExecution):
        """Record an execution"""
        self.executions.append(execution)
        self.last_run = execution.started_at
        self.run_count += 1
        
        if execution.success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        self.calculate_next_run()
        
        if len(self.executions) > 100:
            self.executions = self.executions[-100:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "job_type": self.job_type.value,
            "schedule": self.schedule.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "recent_executions": [e.to_dict() for e in self.executions[-10:]]
        }


class CronScheduler:
    """Scheduler for managing cron jobs"""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._jobs: Dict[str, CronJob] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._handlers: Dict[JobType, Callable] = {}
    
    def register_handler(self, job_type: JobType, handler: Callable):
        """Register a handler for a job type"""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for {job_type.value}")
    
    def add_job(self, job: CronJob) -> str:
        """Add a job to the scheduler"""
        with self._lock:
            self._jobs[job.id] = job
            job.calculate_next_run()
        
        logger.info(f"Added job {job.id}: {job.name}")
        return job.id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the scheduler"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Removed job {job_id}")
                return True
        return False
    
    def get_job(self, job_id: str) -> Optional[CronJob]:
        """Get a job by ID"""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None
    ) -> List[CronJob]:
        """List all jobs"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        
        return sorted(jobs, key=lambda j: j.next_run or datetime.max)
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a job"""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.PAUSED
            return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a job"""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.ACTIVE
            job.calculate_next_run()
            return True
        return False
    
    def run_job_now(self, job_id: str) -> Optional[JobExecution]:
        """Run a job immediately"""
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        return self._execute_job(job)
    
    def _execute_job(self, job: CronJob) -> JobExecution:
        """Execute a job"""
        execution = JobExecution(
            id=str(uuid.uuid4()),
            job_id=job.id,
            started_at=datetime.now()
        )
        
        job.status = JobStatus.RUNNING
        
        try:
            handler = job.handler or self._handlers.get(job.job_type)
            
            if handler:
                result = handler(**job.handler_args)
                execution.result = result
                execution.success = True
            else:
                raise RuntimeError(f"No handler for job type {job.job_type.value}")
            
        except Exception as e:
            execution.success = False
            execution.error_message = str(e)
            logger.error(f"Job {job.id} failed: {e}")
        
        finally:
            execution.completed_at = datetime.now()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            job.status = JobStatus.ACTIVE if execution.success else JobStatus.ERROR
            job.record_execution(execution)
        
        return execution
    
    def start(self):
        """Start the scheduler"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")
    
    def _run_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                self._check_jobs()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_jobs(self):
        """Check and run due jobs"""
        with self._lock:
            jobs_to_run = [j for j in self._jobs.values() if j.should_run()]
        
        for job in jobs_to_run:
            try:
                self._execute_job(job)
            except Exception as e:
                logger.error(f"Error executing job {job.id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_jobs = len(self._jobs)
        active_jobs = len([j for j in self._jobs.values() if j.status == JobStatus.ACTIVE])
        paused_jobs = len([j for j in self._jobs.values() if j.status == JobStatus.PAUSED])
        
        total_runs = sum(j.run_count for j in self._jobs.values())
        total_successes = sum(j.success_count for j in self._jobs.values())
        total_failures = sum(j.failure_count for j in self._jobs.values())
        
        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "paused_jobs": paused_jobs,
            "total_runs": total_runs,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": total_successes / total_runs if total_runs > 0 else 0,
            "is_running": self._running
        }


_scheduler: Optional[CronScheduler] = None


def create_scheduler(check_interval: float = 60.0) -> CronScheduler:
    """Create or get the global scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = CronScheduler(check_interval)
    return _scheduler


def get_scheduler() -> Optional[CronScheduler]:
    """Get the global scheduler"""
    return _scheduler


def schedule_job(
    name: str,
    job_type: JobType,
    schedule: JobSchedule,
    handler: Optional[Callable] = None,
    handler_args: Optional[Dict[str, Any]] = None
) -> str:
    """Schedule a new job"""
    scheduler = create_scheduler()
    
    job = CronJob(
        id=str(uuid.uuid4()),
        name=name,
        job_type=job_type,
        schedule=schedule,
        handler=handler,
        handler_args=handler_args or {}
    )
    
    return scheduler.add_job(job)


def list_scheduled_jobs(
    status: Optional[JobStatus] = None,
    job_type: Optional[JobType] = None
) -> List[CronJob]:
    """List scheduled jobs"""
    scheduler = get_scheduler()
    if scheduler:
        return scheduler.list_jobs(status, job_type)
    return []
