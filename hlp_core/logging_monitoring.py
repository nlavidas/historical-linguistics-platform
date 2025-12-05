"""
HLP Core Logging and Monitoring - Structured Logging and Metrics

This module provides comprehensive logging, metrics collection,
performance monitoring, and alerting functionality for the platform.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import threading
import queue
import traceback
import socket
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Any, Tuple, Union, 
    Callable, TypeVar, Generic
)
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import functools


class LogLevel(Enum):
    """Log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogRecord:
    """Structured log record"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    
    context: Dict[str, Any] = field(default_factory=dict)
    
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    duration_ms: Optional[float] = None
    
    hostname: str = field(default_factory=socket.gethostname)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "logger": self.logger_name,
            "message": self.message,
            "hostname": self.hostname
        }
        
        if self.module:
            result["module"] = self.module
        if self.function:
            result["function"] = self.function
        if self.line_number:
            result["line"] = self.line_number
        if self.exception_type:
            result["exception"] = {
                "type": self.exception_type,
                "message": self.exception_message,
                "traceback": self.exception_traceback
            }
        if self.context:
            result["context"] = self.context
        if self.request_id:
            result["request_id"] = self.request_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class Metric:
    """Metric data point"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    tags: Dict[str, str] = field(default_factory=dict)
    
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    source: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    context: Dict[str, Any] = field(default_factory=dict)
    
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging"""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        if self.include_context and hasattr(record, "context"):
            log_data["context"] = record.context
        
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Formatter for console output with colors"""
    
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m"
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console"""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"
        
        message = f"{timestamp} | {level_str} | {record.name} | {record.getMessage()}"
        
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler using a queue"""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._process_logs, daemon=True)
        self._worker.start()
    
    def emit(self, record: logging.LogRecord):
        """Add record to queue"""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass
    
    def _process_logs(self):
        """Process logs from queue"""
        while not self._shutdown.is_set():
            try:
                record = self._queue.get(timeout=0.1)
                self.target_handler.emit(record)
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def close(self):
        """Close handler"""
        self._shutdown.set()
        self._worker.join(timeout=5)
        self.target_handler.close()
        super().close()


class FileRotatingHandler(logging.Handler):
    """File handler with rotation based on size and time"""
    
    def __init__(
        self,
        log_dir: Path,
        base_name: str = "hlp",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 10,
        rotate_daily: bool = True
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.rotate_daily = rotate_daily
        
        self._current_date = datetime.now().date()
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._lock = threading.Lock()
        
        self._open_file()
    
    def _get_log_filename(self) -> Path:
        """Get current log filename"""
        date_str = self._current_date.strftime("%Y%m%d")
        return self.log_dir / f"{self.base_name}_{date_str}.log"
    
    def _open_file(self):
        """Open log file"""
        self._current_file = self._get_log_filename()
        self._file_handle = open(self._current_file, "a", encoding="utf-8")
    
    def _should_rotate(self) -> bool:
        """Check if rotation is needed"""
        if self.rotate_daily and datetime.now().date() != self._current_date:
            return True
        
        if self._current_file and self._current_file.exists():
            if self._current_file.stat().st_size >= self.max_bytes:
                return True
        
        return False
    
    def _rotate(self):
        """Rotate log file"""
        if self._file_handle:
            self._file_handle.close()
        
        if datetime.now().date() != self._current_date:
            self._current_date = datetime.now().date()
        else:
            timestamp = datetime.now().strftime("%H%M%S")
            rotated_name = self._current_file.with_suffix(f".{timestamp}.log")
            if self._current_file.exists():
                self._current_file.rename(rotated_name)
        
        self._cleanup_old_logs()
        self._open_file()
    
    def _cleanup_old_logs(self):
        """Remove old log files"""
        log_files = sorted(self.log_dir.glob(f"{self.base_name}_*.log"))
        while len(log_files) > self.backup_count:
            oldest = log_files.pop(0)
            oldest.unlink()
    
    def emit(self, record: logging.LogRecord):
        """Write log record to file"""
        with self._lock:
            try:
                if self._should_rotate():
                    self._rotate()
                
                msg = self.format(record)
                self._file_handle.write(msg + "\n")
                self._file_handle.flush()
            except Exception:
                self.handleError(record)
    
    def close(self):
        """Close handler"""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
        super().close()


class PlatformLogger:
    """Main platform logger with structured logging support"""
    
    _instances: Dict[str, "PlatformLogger"] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_dir: Optional[Path] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        self._logger.handlers.clear()
        
        self._context: Dict[str, Any] = {}
        self._request_id: Optional[str] = None
        
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            self._logger.addHandler(console_handler)
        
        if enable_file and log_dir:
            file_handler = FileRotatingHandler(log_dir, base_name=name.replace(".", "_"))
            if enable_json:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                ))
            self._logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_dir: Optional[Path] = None
    ) -> "PlatformLogger":
        """Get or create a logger instance"""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, level, log_dir)
            return cls._instances[name]
    
    def set_context(self, **kwargs):
        """Set context for all subsequent log messages"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear context"""
        self._context.clear()
    
    def set_request_id(self, request_id: str):
        """Set request ID for correlation"""
        self._request_id = request_id
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal log method"""
        extra = {
            "context": {**self._context, **kwargs.get("context", {})},
            "request_id": self._request_id
        }
        
        if "duration_ms" in kwargs:
            extra["duration_ms"] = kwargs["duration_ms"]
        
        self._logger.log(level.value, message, extra=extra, exc_info=kwargs.get("exc_info"))
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs["exc_info"] = True
        self._log(LogLevel.ERROR, message, **kwargs)
    
    @contextmanager
    def timed(self, operation: str, level: LogLevel = LogLevel.INFO):
        """Context manager for timing operations"""
        start_time = time.time()
        self._log(level, f"Starting: {operation}")
        
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self._log(level, f"Completed: {operation}", duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._log(LogLevel.ERROR, f"Failed: {operation} - {e}", duration_ms=duration_ms, exc_info=True)
            raise
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context"""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context


class MetricsCollector:
    """Collects and aggregates metrics"""
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._rates: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        self._tags: Dict[str, Dict[str, str]] = {}
        self._descriptions: Dict[str, str] = {}
        
        self._lock = threading.Lock()
        self._initialized = True
    
    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter"""
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value
            if tags:
                self._tags[key] = tags
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
            if tags:
                self._tags[key] = tags
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self._lock:
            key = self._make_key(name, tags)
            self._histograms[key].append(value)
            if tags:
                self._tags[key] = tags
            
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-5000:]
    
    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer value"""
        with self._lock:
            key = self._make_key(name, tags)
            self._timers[key].append(duration_ms)
            if tags:
                self._tags[key] = tags
            
            if len(self._timers[key]) > 10000:
                self._timers[key] = self._timers[key][-5000:]
    
    def rate(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Record a rate value"""
        with self._lock:
            key = self._make_key(name, tags)
            self._rates[key].append((datetime.now(), value))
            if tags:
                self._tags[key] = tags
            
            cutoff = datetime.now() - timedelta(hours=1)
            self._rates[key] = [(t, v) for t, v in self._rates[key] if t > cutoff]
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric"""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"
    
    def describe(self, name: str, description: str):
        """Add description for a metric"""
        self._descriptions[name] = description
    
    @contextmanager
    def timed_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.timer(name, duration_ms, tags)
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value"""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, tags)
        return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
        }
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics"""
        key = self._make_key(name, tags)
        values = self._timers.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min_ms": sorted_values[0],
            "max_ms": sorted_values[-1],
            "mean_ms": sum(values) / n,
            "p50_ms": sorted_values[n // 2],
            "p90_ms": sorted_values[int(n * 0.9)],
            "p95_ms": sorted_values[int(n * 0.95)],
            "p99_ms": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
        }
    
    def get_rate_per_minute(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get rate per minute"""
        key = self._make_key(name, tags)
        entries = self._rates.get(key, [])
        
        if not entries:
            return 0.0
        
        cutoff = datetime.now() - timedelta(minutes=1)
        recent = [v for t, v in entries if t > cutoff]
        return sum(recent)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms},
                "timers": {k: self.get_timer_stats(k) for k in self._timers},
                "rates_per_minute": {k: self.get_rate_per_minute(k) for k in self._rates}
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._rates.clear()


class PerformanceMonitor:
    """Monitors system and application performance"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or MetricsCollector()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._interval = 60
    
    def start(self, interval: int = 60):
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._interval = interval
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring"""
        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._collect_system_metrics()
                self._collect_process_metrics()
            except Exception:
                pass
            
            self._stop_event.wait(self._interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.gauge("system.cpu.percent", cpu_percent)
            
            memory = psutil.virtual_memory()
            self.metrics.gauge("system.memory.percent", memory.percent)
            self.metrics.gauge("system.memory.available_gb", memory.available / (1024**3))
            
            disk = psutil.disk_usage("/")
            self.metrics.gauge("system.disk.percent", disk.percent)
            self.metrics.gauge("system.disk.free_gb", disk.free / (1024**3))
            
            net_io = psutil.net_io_counters()
            self.metrics.gauge("system.network.bytes_sent", net_io.bytes_sent)
            self.metrics.gauge("system.network.bytes_recv", net_io.bytes_recv)
            
        except ImportError:
            pass
    
    def _collect_process_metrics(self):
        """Collect process-level metrics"""
        try:
            import psutil
            
            process = psutil.Process()
            
            self.metrics.gauge("process.cpu.percent", process.cpu_percent())
            
            memory_info = process.memory_info()
            self.metrics.gauge("process.memory.rss_mb", memory_info.rss / (1024**2))
            self.metrics.gauge("process.memory.vms_mb", memory_info.vms / (1024**2))
            
            self.metrics.gauge("process.threads", process.num_threads())
            
            try:
                fds = process.num_fds()
                self.metrics.gauge("process.file_descriptors", fds)
            except AttributeError:
                pass
            
        except ImportError:
            pass
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            import psutil
            
            stats["cpu_percent"] = psutil.cpu_percent()
            
            memory = psutil.virtual_memory()
            stats["memory_percent"] = memory.percent
            stats["memory_available_gb"] = round(memory.available / (1024**3), 2)
            
            disk = psutil.disk_usage("/")
            stats["disk_percent"] = disk.percent
            stats["disk_free_gb"] = round(disk.free / (1024**3), 2)
            
            process = psutil.Process()
            stats["process_memory_mb"] = round(process.memory_info().rss / (1024**2), 2)
            stats["process_threads"] = process.num_threads()
            
        except ImportError:
            stats["error"] = "psutil not available"
        
        return stats


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(
        self,
        logger: Optional[PlatformLogger] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        self.logger = logger or PlatformLogger.get_logger("alerts")
        self.metrics = metrics or MetricsCollector()
        
        self._alerts: Dict[str, Alert] = {}
        self._alert_rules: List[Dict[str, Any]] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
        self._alert_counter = 0
    
    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: Optional[str] = None
    ):
        """Add an alert rule"""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "message_template": message_template or f"{name}: {{metric_name}} {{condition}} {{threshold}} (current: {{value}})"
        }
        self._alert_rules.append(rule)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self._handlers.append(handler)
    
    def check_rules(self):
        """Check all alert rules"""
        for rule in self._alert_rules:
            try:
                self._check_rule(rule)
            except Exception as e:
                self.logger.error(f"Error checking rule {rule['name']}: {e}")
    
    def _check_rule(self, rule: Dict[str, Any]):
        """Check a single alert rule"""
        metric_name = rule["metric_name"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        value = self.metrics.get_gauge(metric_name)
        if value is None:
            value = self.metrics.get_counter(metric_name)
        if value is None:
            return
        
        triggered = False
        if condition == ">" and value > threshold:
            triggered = True
        elif condition == ">=" and value >= threshold:
            triggered = True
        elif condition == "<" and value < threshold:
            triggered = True
        elif condition == "<=" and value <= threshold:
            triggered = True
        elif condition == "==" and value == threshold:
            triggered = True
        elif condition == "!=" and value != threshold:
            triggered = True
        
        if triggered:
            message = rule["message_template"].format(
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                value=value
            )
            self.create_alert(
                severity=rule["severity"],
                title=rule["name"],
                message=message,
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold
            )
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert"""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            context=context or {}
        )
        
        with self._lock:
            self._alerts[alert_id] = alert
        
        self.logger.log(
            LogLevel.WARNING if severity in (AlertSeverity.INFO, AlertSeverity.WARNING) else LogLevel.ERROR,
            f"Alert [{severity.value}]: {title} - {message}"
        )
        
        self.metrics.counter("alerts.created", tags={"severity": severity.value})
        
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                self.metrics.counter("alerts.acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.metrics.counter("alerts.resolved")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self._lock:
            return [a for a in self._alerts.values() if not a.resolved]
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self._alerts.get(alert_id)
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        with self._lock:
            return [a for a in self._alerts.values() if a.severity == severity]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        with self._lock:
            active = [a for a in self._alerts.values() if not a.resolved]
            return {
                "total": len(self._alerts),
                "active": len(active),
                "by_severity": {
                    s.value: len([a for a in active if a.severity == s])
                    for s in AlertSeverity
                },
                "acknowledged": len([a for a in active if a.acknowledged]),
                "unacknowledged": len([a for a in active if not a.acknowledged])
            }
    
    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Remove old resolved alerts"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        with self._lock:
            to_remove = [
                aid for aid, alert in self._alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff
            ]
            for aid in to_remove:
                del self._alerts[aid]


def log_function_call(logger: Optional[PlatformLogger] = None, level: LogLevel = LogLevel.DEBUG):
    """Decorator to log function calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = PlatformLogger.get_logger(func.__module__)
            
            func_name = func.__qualname__
            logger._log(level, f"Calling {func_name}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger._log(level, f"Completed {func_name}", duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger._log(LogLevel.ERROR, f"Failed {func_name}: {e}", duration_ms=duration_ms, exc_info=True)
                raise
        
        return wrapper
    return decorator


def get_platform_logger(name: str = "hlp") -> PlatformLogger:
    """Get platform logger instance"""
    try:
        from hlp_core.config_runtime import get_path_resolver
        log_dir = get_path_resolver().logs_dir
    except ImportError:
        log_dir = Path("logs")
    
    return PlatformLogger.get_logger(name, log_dir=log_dir)


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()


def get_alert_manager() -> AlertManager:
    """Get alert manager instance"""
    return AlertManager()
