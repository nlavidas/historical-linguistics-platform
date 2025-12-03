"""
HLP Pipelines - Workflow Orchestration Package

This package provides comprehensive support for orchestrating
text processing pipelines, including ingestion, annotation,
and valency extraction workflows.

Modules:
    ingest_pipeline: Text ingestion workflow
    annotation_pipeline: Annotation scheduling
    valency_pipeline: Valency extraction workflow
    cron_jobs: Scheduled job definitions

University of Athens - Nikolaos Lavidas
"""

from hlp_pipelines.ingest_pipeline import (
    IngestPipeline,
    IngestConfig,
    IngestJob,
    IngestResult,
    run_ingest_pipeline,
    schedule_ingest,
)

from hlp_pipelines.annotation_pipeline import (
    AnnotationPipeline,
    AnnotationConfig,
    AnnotationJob,
    AnnotationResult,
    run_annotation_pipeline,
    schedule_annotation,
)

from hlp_pipelines.valency_pipeline import (
    ValencyPipeline,
    ValencyConfig,
    ValencyJob,
    ValencyResult,
    run_valency_pipeline,
    schedule_valency_extraction,
)

from hlp_pipelines.cron_jobs import (
    CronScheduler,
    CronJob,
    JobSchedule,
    create_scheduler,
    schedule_job,
    list_scheduled_jobs,
)

__version__ = "1.0.0"
__author__ = "Nikolaos Lavidas"

__all__ = [
    "IngestPipeline",
    "IngestConfig",
    "IngestJob",
    "IngestResult",
    "run_ingest_pipeline",
    "schedule_ingest",
    "AnnotationPipeline",
    "AnnotationConfig",
    "AnnotationJob",
    "AnnotationResult",
    "run_annotation_pipeline",
    "schedule_annotation",
    "ValencyPipeline",
    "ValencyConfig",
    "ValencyJob",
    "ValencyResult",
    "run_valency_pipeline",
    "schedule_valency_extraction",
    "CronScheduler",
    "CronJob",
    "JobSchedule",
    "create_scheduler",
    "schedule_job",
    "list_scheduled_jobs",
]
