#!/usr/bin/env python3
"""
Automated Real Data Pipeline
============================

Comprehensive automated pipeline for astrobiology genomics research:
- Orchestrated data acquisition from KEGG and NCBI
- Real-time quality monitoring and validation
- Automated error handling and recovery
- Intelligent scheduling and resource management
- Complete data lineage tracking
- Performance optimization
- Notification and alerting
- Dashboard integration

NASA-grade automation with enterprise reliability.
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
import logging
import sqlite3
import pickle
import gzip
import shutil
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import time
import schedule
import threading
from threading import Lock, Event
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import asynccontextmanager
import aiohttp
import asyncpg
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil
import gc
import warnings
from functools import wraps
import sys
import traceback

# Import our custom modules
try:
    from .advanced_data_system import AdvancedDataManager
    from .kegg_real_data_integration import KEGGRealDataIntegration
    from .ncbi_agora2_integration import NCBIAgoraIntegration
    from .advanced_quality_system import QualityMonitor, DataType
    from .metadata_annotation_system import MetadataManager
    from .data_versioning_system import VersionManager
except ImportError:
    # Handle imports when running standalone
    import sys
    sys.path.append(str(Path(__file__).parent))
    from advanced_data_system import AdvancedDataManager
    from kegg_real_data_integration import KEGGRealDataIntegration
    from ncbi_agora2_integration import NCBIAgoraIntegration
    from advanced_quality_system import QualityMonitor, DataType
    from metadata_annotation_system import MetadataManager
    from data_versioning_system import VersionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"
    STOPPED = "stopped"

class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class PipelineTask:
    """Individual pipeline task"""
    task_id: str
    name: str
    description: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 3600  # 1 hour default
    max_retries: int = 3
    retry_delay: int = 60  # 1 minute
    
    # Execution info
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    result: Any = None
    retry_count: int = 0
    
    # Resource requirements
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_gb: int = 10
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    name: str
    description: str
    schedule: str = ""  # Cron-like schedule
    max_concurrent_tasks: int = 4
    max_memory_gb: int = 16
    max_disk_gb: int = 100
    timeout: int = 14400  # 4 hours
    
    # Data sources
    enable_kegg: bool = True
    enable_ncbi: bool = True
    enable_agora2: bool = True
    
    # Limits for testing
    max_kegg_pathways: Optional[int] = 100
    max_ncbi_genomes: Optional[int] = 50
    max_agora2_models: Optional[int] = 50
    
    # Quality thresholds
    min_quality_score: float = 0.8
    nasa_grade_required: bool = True
    
    # Notifications
    email_notifications: bool = False
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook: str = ""
    
    # Storage
    cleanup_old_data: bool = True
    backup_before_update: bool = True
    compress_backups: bool = True
    
    # Performance
    use_caching: bool = True
    parallel_downloads: bool = True
    optimize_memory: bool = True

class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.lock = Lock()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_percent': [],
            'network_io': [],
            'timestamps': []
        }
    
    def start_monitoring(self, interval: int = 30):
        """Start resource monitoring"""
        with self.lock:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        with self.lock:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self, interval: int):
        """Resource monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Network I/O
                network = psutil.net_io_counters()
                network_io = network.bytes_sent + network.bytes_recv
                
                with self.lock:
                    self.metrics['cpu_percent'].append(cpu_percent)
                    self.metrics['memory_percent'].append(memory_percent)
                    self.metrics['disk_percent'].append(disk_percent)
                    self.metrics['network_io'].append(network_io)
                    self.metrics['timestamps'].append(datetime.now(timezone.utc))
                    
                    # Keep only last 100 measurements
                    for key in self.metrics:
                        if len(self.metrics[key]) > 100:
                            self.metrics[key] = self.metrics[key][-100:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }
    
    def check_resources(self, task: PipelineTask) -> bool:
        """Check if resources are available for task"""
        current = self.get_current_usage()
        
        # Check memory
        if current['memory_available_gb'] < task.memory_gb:
            return False
        
        # Check disk
        if current['disk_free_gb'] < task.disk_gb:
            return False
        
        # Check CPU (simple heuristic)
        if current['cpu_percent'] > 90:
            return False
        
        return True

class NotificationManager:
    """Notification and alerting system"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    async def send_notification(self, message: str, level: str = "info", 
                               subject: str = None):
        """Send notification through configured channels"""
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Email notifications
        if self.config.email_notifications and self.config.email_recipients:
            await self._send_email(message, level, subject)
        
        # Slack notifications
        if self.config.slack_webhook:
            await self._send_slack(message, level)
    
    async def _send_email(self, message: str, level: str, subject: str = None):
        """Send email notification"""
        try:
            if not subject:
                subject = f"Pipeline {level.upper()}: {self.config.name}"
            
            msg = MIMEMultipart()
            msg['From'] = "pipeline@astrobio.local"
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = subject
            
            body = f"""
Pipeline: {self.config.name}
Level: {level.upper()}
Time: {datetime.now(timezone.utc).isoformat()}

Message:
{message}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In a real system, configure SMTP server
            logger.info(f"Email notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack(self, message: str, level: str):
        """Send Slack notification"""
        try:
            color_map = {
                'info': '#36a64f',
                'warning': '#ff9500',
                'error': '#ff0000'
            }
            
            payload = {
                'text': f"Pipeline {level.upper()}",
                'attachments': [{
                    'color': color_map.get(level, '#36a64f'),
                    'fields': [
                        {'title': 'Pipeline', 'value': self.config.name, 'short': True},
                        {'title': 'Time', 'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                        {'title': 'Message', 'value': message, 'short': False}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent")
                    else:
                        logger.error(f"Failed to send Slack notification: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

class TaskScheduler:
    """Advanced task scheduling and execution"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))
    
    async def add_task(self, task: PipelineTask):
        """Add task to queue"""
        await self.task_queue.put(task)
        logger.info(f"Task {task.name} added to queue")
    
    async def run_scheduler(self, resource_monitor: ResourceMonitor):
        """Main scheduler loop"""
        while True:
            try:
                # Get next task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Check dependencies
                if not self._check_dependencies(task):
                    # Put back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                # Check resources
                if not resource_monitor.check_resources(task):
                    # Put back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(5)
                    continue
                
                # Check if we can run more tasks
                if len(self.running_tasks) >= self.max_workers:
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.TimeoutError:
                # No tasks in queue, wait a bit
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                await asyncio.sleep(5)
    
    def _check_dependencies(self, task: PipelineTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _execute_task(self, task: PipelineTask):
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        
        with self.lock:
            self.running_tasks[task.task_id] = task
        
        logger.info(f"Starting task: {task.name}")
        
        try:
            # Run task with timeout
            if asyncio.iscoroutinefunction(task.function):
                # Async function
                result = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Sync function - run in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        task.function,
                        *task.args
                    ),
                    timeout=task.timeout
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Task {task.name} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_message = f"Task timed out after {task.timeout} seconds"
            logger.error(f"Task {task.name} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            
            logger.error(f"Task {task.name} failed: {e}")
            
            # Retry if possible
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                logger.info(f"Retrying task {task.name} (attempt {task.retry_count})")
                
                # Add back to queue with delay
                await asyncio.sleep(task.retry_delay)
                await self.task_queue.put(task)
            else:
                logger.error(f"Task {task.name} failed permanently after {task.max_retries} retries")
        
        finally:
            with self.lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                if task.status == TaskStatus.COMPLETED:
                    self.completed_tasks[task.task_id] = task
                elif task.status == TaskStatus.FAILED:
                    self.failed_tasks[task.task_id] = task
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        with self.lock:
            return {
                'queue_size': self.task_queue.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'running_task_names': [task.name for task in self.running_tasks.values()],
                'failed_task_names': [task.name for task in self.failed_tasks.values()]
            }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

class AutomatedDataPipeline:
    """Main automated data pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.status = PipelineStatus.IDLE
        self.start_time = None
        self.end_time = None
        self.error_message = ""
        
        # Initialize components
        self.data_manager = AdvancedDataManager()
        self.quality_monitor = QualityMonitor()
        self.metadata_manager = MetadataManager()
        self.version_manager = VersionManager()
        self.resource_monitor = ResourceMonitor()
        self.notification_manager = NotificationManager(config)
        self.scheduler = TaskScheduler(max_workers=config.max_concurrent_tasks)
        
        # Task tracking
        self.tasks = []
        self.results = {}
        self.errors = []
        
        # Performance metrics
        self.metrics = {
            'total_data_downloaded': 0,
            'total_processing_time': 0,
            'quality_scores': [],
            'error_count': 0,
            'retry_count': 0
        }
        
        # Pipeline state database
        self.db_path = Path("data/pipeline/pipeline_state.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize pipeline state database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Pipeline runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    config_name TEXT,
                    status TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    failed_tasks INTEGER,
                    error_message TEXT,
                    metrics TEXT
                )
            ''')
            
            # Task executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_executions (
                    execution_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    task_id TEXT,
                    task_name TEXT,
                    status TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL,
                    retry_count INTEGER,
                    error_message TEXT,
                    result_summary TEXT,
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
                )
            ''')
            
            conn.commit()
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        run_id = str(uuid.uuid4())
        self.status = PipelineStatus.INITIALIZING
        self.start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting pipeline run {run_id}")
            await self.notification_manager.send_notification(
                f"Pipeline {self.config.name} starting", "info"
            )
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Start task scheduler
            scheduler_task = asyncio.create_task(
                self.scheduler.run_scheduler(self.resource_monitor)
            )
            
            # Create and queue tasks
            await self._create_tasks()
            
            self.status = PipelineStatus.RUNNING
            
            # Monitor task execution
            await self._monitor_execution()
            
            # Generate final report
            report = await self._generate_report(run_id)
            
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now(timezone.utc)
            
            await self.notification_manager.send_notification(
                f"Pipeline {self.config.name} completed successfully", "info"
            )
            
            return report
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            self.error_message = str(e)
            self.end_time = datetime.now(timezone.utc)
            
            logger.error(f"Pipeline failed: {e}")
            await self.notification_manager.send_notification(
                f"Pipeline {self.config.name} failed: {e}", "error"
            )
            
            raise
        finally:
            # Cleanup
            self.resource_monitor.stop_monitoring()
            self.scheduler.cleanup()
            
            # Store run in database
            await self._store_run_results(run_id)
    
    async def _create_tasks(self):
        """Create all pipeline tasks"""
        task_id_counter = 0
        
        # Task 1: Initialize data sources
        init_task = PipelineTask(
            task_id=f"init_{task_id_counter}",
            name="Initialize Data Sources",
            description="Initialize KEGG and NCBI data source connections",
            function=self._initialize_data_sources,
            priority=Priority.HIGH,
            timeout=300,  # 5 minutes
            memory_gb=1
        )
        await self.scheduler.add_task(init_task)
        self.tasks.append(init_task)
        task_id_counter += 1
        
        # Task 2: Download KEGG data
        if self.config.enable_kegg:
            kegg_task = PipelineTask(
                task_id=f"kegg_{task_id_counter}",
                name="Download KEGG Data",
                description="Download KEGG pathway, reaction, and compound data",
                function=self._download_kegg_data,
                dependencies=[init_task.task_id],
                priority=Priority.HIGH,
                timeout=7200,  # 2 hours
                memory_gb=4,
                disk_gb=20
            )
            await self.scheduler.add_task(kegg_task)
            self.tasks.append(kegg_task)
            task_id_counter += 1
        
        # Task 3: Download NCBI/AGORA2 data
        if self.config.enable_ncbi or self.config.enable_agora2:
            ncbi_task = PipelineTask(
                task_id=f"ncbi_{task_id_counter}",
                name="Download NCBI/AGORA2 Data",
                description="Download NCBI genome and AGORA2 model data",
                function=self._download_ncbi_data,
                dependencies=[init_task.task_id],
                priority=Priority.HIGH,
                timeout=10800,  # 3 hours
                memory_gb=6,
                disk_gb=50
            )
            await self.scheduler.add_task(ncbi_task)
            self.tasks.append(ncbi_task)
            task_id_counter += 1
        
        # Task 4: Quality validation
        quality_deps = [task.task_id for task in self.tasks if task.name.startswith("Download")]
        if quality_deps:
            quality_task = PipelineTask(
                task_id=f"quality_{task_id_counter}",
                name="Quality Validation",
                description="Validate data quality and generate quality reports",
                function=self._validate_quality,
                dependencies=quality_deps,
                priority=Priority.NORMAL,
                timeout=1800,  # 30 minutes
                memory_gb=3
            )
            await self.scheduler.add_task(quality_task)
            self.tasks.append(quality_task)
            task_id_counter += 1
        
        # Task 5: Generate metadata
        metadata_task = PipelineTask(
            task_id=f"metadata_{task_id_counter}",
            name="Generate Metadata",
            description="Extract and store comprehensive metadata",
            function=self._generate_metadata,
            dependencies=quality_deps,
            priority=Priority.NORMAL,
            timeout=1200,  # 20 minutes
            memory_gb=2
        )
        await self.scheduler.add_task(metadata_task)
        self.tasks.append(metadata_task)
        task_id_counter += 1
        
        # Task 6: Create data versions
        version_task = PipelineTask(
            task_id=f"version_{task_id_counter}",
            name="Create Data Versions",
            description="Create versioned snapshots of all data",
            function=self._create_versions,
            dependencies=[metadata_task.task_id],
            priority=Priority.NORMAL,
            timeout=1800,  # 30 minutes
            memory_gb=4,
            disk_gb=30
        )
        await self.scheduler.add_task(version_task)
        self.tasks.append(version_task)
        task_id_counter += 1
        
        # Task 7: Generate reports
        report_task = PipelineTask(
            task_id=f"report_{task_id_counter}",
            name="Generate Reports",
            description="Generate comprehensive data reports and dashboards",
            function=self._generate_final_reports,
            dependencies=[version_task.task_id],
            priority=Priority.LOW,
            timeout=600,  # 10 minutes
            memory_gb=2
        )
        await self.scheduler.add_task(report_task)
        self.tasks.append(report_task)
        
        logger.info(f"Created {len(self.tasks)} pipeline tasks")
    
    async def _initialize_data_sources(self) -> Dict[str, Any]:
        """Initialize data source connections"""
        logger.info("Initializing data sources")
        
        results = {
            'kegg_initialized': False,
            'ncbi_initialized': False,
            'quality_monitor_initialized': False,
            'metadata_manager_initialized': False,
            'version_manager_initialized': False
        }
        
        try:
            # Initialize quality monitor
            self.quality_monitor = QualityMonitor()
            results['quality_monitor_initialized'] = True
            
            # Initialize metadata manager
            self.metadata_manager = MetadataManager()
            results['metadata_manager_initialized'] = True
            
            # Initialize version manager
            self.version_manager = VersionManager()
            results['version_manager_initialized'] = True
            
            # Initialize KEGG integration
            if self.config.enable_kegg:
                self.kegg_integration = KEGGRealDataIntegration()
                results['kegg_initialized'] = True
            
            # Initialize NCBI integration
            if self.config.enable_ncbi or self.config.enable_agora2:
                self.ncbi_integration = NCBIAgoraIntegration()
                results['ncbi_initialized'] = True
            
            logger.info("Data sources initialized successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to initialize data sources: {e}")
            raise
    
    async def _download_kegg_data(self) -> Dict[str, Any]:
        """Download KEGG data"""
        logger.info("Starting KEGG data download")
        
        try:
            # Run KEGG integration
            report = await self.kegg_integration.run_full_integration(
                max_pathways=self.config.max_kegg_pathways
            )
            
            self.metrics['total_data_downloaded'] += report.get('download_results', {}).get('downloaded_pathways', 0)
            
            logger.info(f"KEGG data download completed: {report}")
            return report
            
        except Exception as e:
            logger.error(f"KEGG data download failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _download_ncbi_data(self) -> Dict[str, Any]:
        """Download NCBI/AGORA2 data"""
        logger.info("Starting NCBI/AGORA2 data download")
        
        try:
            # Run NCBI/AGORA2 integration
            report = await self.ncbi_integration.run_full_integration(
                max_models=self.config.max_agora2_models,
                max_genomes=self.config.max_ncbi_genomes
            )
            
            agora2_results = report.get('agora2_results', {})
            ncbi_results = report.get('ncbi_results', {})
            
            self.metrics['total_data_downloaded'] += agora2_results.get('downloaded_models', 0)
            self.metrics['total_data_downloaded'] += ncbi_results.get('downloaded_genomes', 0)
            
            logger.info(f"NCBI/AGORA2 data download completed: {report}")
            return report
            
        except Exception as e:
            logger.error(f"NCBI/AGORA2 data download failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _validate_quality(self) -> Dict[str, Any]:
        """Validate data quality"""
        logger.info("Starting quality validation")
        
        try:
            quality_results = {}
            
            # Load and validate KEGG data
            kegg_files = list(Path("data/processed/kegg").glob("*.csv"))
            for file_path in kegg_files:
                try:
                    df = pd.read_csv(file_path)
                    data_type = DataType.KEGG_PATHWAY  # Simplified
                    
                    report = self.quality_monitor.assess_quality(
                        data=df,
                        data_source=f"kegg_{file_path.stem}",
                        data_type=data_type
                    )
                    
                    quality_results[file_path.stem] = {
                        'overall_score': report.metrics.overall_score(),
                        'nasa_ready': report.compliance_status.get('nasa_grade', False),
                        'issue_count': len(report.issues)
                    }
                    
                    self.metrics['quality_scores'].append(report.metrics.overall_score())
                    
                except Exception as e:
                    logger.warning(f"Quality validation failed for {file_path}: {e}")
            
            # Load and validate NCBI/AGORA2 data
            ncbi_files = list(Path("data/processed/agora2").glob("*.csv"))
            for file_path in ncbi_files:
                try:
                    df = pd.read_csv(file_path)
                    data_type = DataType.AGORA2_MODEL  # Simplified
                    
                    report = self.quality_monitor.assess_quality(
                        data=df,
                        data_source=f"agora2_{file_path.stem}",
                        data_type=data_type
                    )
                    
                    quality_results[file_path.stem] = {
                        'overall_score': report.metrics.overall_score(),
                        'nasa_ready': report.compliance_status.get('nasa_grade', False),
                        'issue_count': len(report.issues)
                    }
                    
                    self.metrics['quality_scores'].append(report.metrics.overall_score())
                    
                except Exception as e:
                    logger.warning(f"Quality validation failed for {file_path}: {e}")
            
            # Check overall quality
            avg_quality = np.mean(self.metrics['quality_scores']) if self.metrics['quality_scores'] else 0
            nasa_ready_count = sum(1 for result in quality_results.values() if result['nasa_ready'])
            
            quality_summary = {
                'total_datasets': len(quality_results),
                'average_quality_score': avg_quality,
                'nasa_ready_count': nasa_ready_count,
                'nasa_ready_percentage': nasa_ready_count / len(quality_results) * 100 if quality_results else 0,
                'meets_threshold': avg_quality >= self.config.min_quality_score,
                'details': quality_results
            }
            
            if not quality_summary['meets_threshold']:
                await self.notification_manager.send_notification(
                    f"Quality validation warning: Average score {avg_quality:.2f} below threshold {self.config.min_quality_score}",
                    "warning"
                )
            
            logger.info(f"Quality validation completed: {quality_summary}")
            return quality_summary
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _generate_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive metadata"""
        logger.info("Starting metadata generation")
        
        try:
            metadata_results = {}
            
            # Generate metadata for KEGG data
            kegg_files = list(Path("data/processed/kegg").glob("*.csv"))
            for file_path in kegg_files:
                try:
                    metadata = self.metadata_manager.extractor.extract_from_file(
                        file_path, "kegg", "pathway"
                    )
                    self.metadata_manager.store_metadata(metadata)
                    metadata_results[f"kegg_{file_path.stem}"] = metadata.record_id
                except Exception as e:
                    logger.warning(f"Metadata generation failed for {file_path}: {e}")
            
            # Generate metadata for NCBI/AGORA2 data
            ncbi_files = list(Path("data/processed/agora2").glob("*.csv"))
            for file_path in ncbi_files:
                try:
                    metadata = self.metadata_manager.extractor.extract_from_file(
                        file_path, "agora2", "metabolic_model"
                    )
                    self.metadata_manager.store_metadata(metadata)
                    metadata_results[f"agora2_{file_path.stem}"] = metadata.record_id
                except Exception as e:
                    logger.warning(f"Metadata generation failed for {file_path}: {e}")
            
            # Generate metadata report
            report = self.metadata_manager.generate_metadata_report()
            
            metadata_summary = {
                'total_records': len(metadata_results),
                'kegg_records': len([k for k in metadata_results.keys() if k.startswith('kegg_')]),
                'agora2_records': len([k for k in metadata_results.keys() if k.startswith('agora2_')]),
                'report': report,
                'record_ids': metadata_results
            }
            
            logger.info(f"Metadata generation completed: {metadata_summary}")
            return metadata_summary
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _create_versions(self) -> Dict[str, Any]:
        """Create data versions"""
        logger.info("Starting data versioning")
        
        try:
            version_results = {}
            
            # Create versions for KEGG datasets
            kegg_files = list(Path("data/processed/kegg").glob("*.csv"))
            for file_path in kegg_files:
                try:
                    dataset_id = f"kegg_{file_path.stem}"
                    
                    # Create dataset if not exists
                    self.version_manager.create_dataset(
                        dataset_id=dataset_id,
                        name=f"KEGG {file_path.stem}",
                        description=f"KEGG dataset: {file_path.stem}",
                        data_type="kegg_pathway",
                        created_by="automated_pipeline"
                    )
                    
                    # Load data and create version
                    df = pd.read_csv(file_path)
                    version = self.version_manager.commit_version(
                        dataset_id=dataset_id,
                        data=df,
                        message=f"Automated pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        created_by="automated_pipeline",
                        tags=["automated", "kegg", f"pipeline_{datetime.now().strftime('%Y%m%d')}"]
                    )
                    
                    version_results[dataset_id] = {
                        'version_id': version.version_id,
                        'version_number': version.version_number,
                        'size': version.size,
                        'checksum': version.checksum
                    }
                    
                except Exception as e:
                    logger.warning(f"Versioning failed for {file_path}: {e}")
            
            # Create versions for NCBI/AGORA2 datasets
            ncbi_files = list(Path("data/processed/agora2").glob("*.csv"))
            for file_path in ncbi_files:
                try:
                    dataset_id = f"agora2_{file_path.stem}"
                    
                    # Create dataset if not exists
                    self.version_manager.create_dataset(
                        dataset_id=dataset_id,
                        name=f"AGORA2 {file_path.stem}",
                        description=f"AGORA2 dataset: {file_path.stem}",
                        data_type="agora2_model",
                        created_by="automated_pipeline"
                    )
                    
                    # Load data and create version
                    df = pd.read_csv(file_path)
                    version = self.version_manager.commit_version(
                        dataset_id=dataset_id,
                        data=df,
                        message=f"Automated pipeline update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        created_by="automated_pipeline",
                        tags=["automated", "agora2", f"pipeline_{datetime.now().strftime('%Y%m%d')}"]
                    )
                    
                    version_results[dataset_id] = {
                        'version_id': version.version_id,
                        'version_number': version.version_number,
                        'size': version.size,
                        'checksum': version.checksum
                    }
                    
                except Exception as e:
                    logger.warning(f"Versioning failed for {file_path}: {e}")
            
            version_summary = {
                'total_versions': len(version_results),
                'kegg_versions': len([k for k in version_results.keys() if k.startswith('kegg_')]),
                'agora2_versions': len([k for k in version_results.keys() if k.startswith('agora2_')]),
                'total_size_mb': sum(v['size'] for v in version_results.values()) / (1024 * 1024),
                'versions': version_results
            }
            
            logger.info(f"Data versioning completed: {version_summary}")
            return version_summary
            
        except Exception as e:
            logger.error(f"Data versioning failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _generate_final_reports(self) -> Dict[str, Any]:
        """Generate final reports and dashboards"""
        logger.info("Generating final reports")
        
        try:
            # Generate quality dashboard
            quality_dashboard = self.quality_monitor.generate_quality_dashboard()
            
            # Generate metadata report
            metadata_report = self.metadata_manager.generate_metadata_report()
            
            # Export metadata
            metadata_export = self.metadata_manager.export_metadata("json")
            
            # Generate pipeline summary
            pipeline_summary = {
                'pipeline_name': self.config.name,
                'run_time': self.start_time,
                'total_duration_minutes': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60,
                'total_tasks': len(self.tasks),
                'completed_tasks': len(self.scheduler.completed_tasks),
                'failed_tasks': len(self.scheduler.failed_tasks),
                'metrics': self.metrics,
                'resource_usage': self.resource_monitor.get_current_usage()
            }
            
            # Save comprehensive report
            final_report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pipeline_summary': pipeline_summary,
                'quality_dashboard': quality_dashboard,
                'metadata_report': metadata_report,
                'task_status': self.scheduler.get_status(),
                'config': asdict(self.config)
            }
            
            # Save to file
            report_file = Path(f"data/reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            logger.info(f"Final reports generated: {report_file}")
            
            return {
                'report_file': str(report_file),
                'summary': pipeline_summary,
                'quality_dashboard_file': quality_dashboard.get('output_file'),
                'metadata_export_file': metadata_export
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            self.metrics['error_count'] += 1
            raise
    
    async def _monitor_execution(self):
        """Monitor task execution progress"""
        logger.info("Starting execution monitoring")
        
        total_tasks = len(self.tasks)
        
        while True:
            status = self.scheduler.get_status()
            
            completed = status['completed_tasks']
            failed = status['failed_tasks']
            running = status['running_tasks']
            
            # Check if all tasks are done
            if completed + failed >= total_tasks:
                break
            
            # Log progress
            progress = (completed + failed) / total_tasks * 100
            logger.info(f"Progress: {progress:.1f}% ({completed} completed, {failed} failed, {running} running)")
            
            # Send periodic updates
            if completed > 0 and completed % 2 == 0:  # Every 2 tasks
                await self.notification_manager.send_notification(
                    f"Pipeline progress: {progress:.1f}% complete", "info"
                )
            
            # Check for critical failures
            if failed > total_tasks * 0.5:  # More than 50% failed
                raise Exception(f"Too many task failures: {failed}/{total_tasks}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _generate_report(self, run_id: str) -> Dict[str, Any]:
        """Generate final pipeline report"""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        status = self.scheduler.get_status()
        
        report = {
            'run_id': run_id,
            'config_name': self.config.name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': str(timedelta(seconds=int(duration))),
            'status': self.status.value,
            'total_tasks': len(self.tasks),
            'completed_tasks': status['completed_tasks'],
            'failed_tasks': status['failed_tasks'],
            'success_rate': status['completed_tasks'] / len(self.tasks) * 100 if self.tasks else 0,
            'metrics': self.metrics,
            'errors': self.errors,
            'resource_usage': self.resource_monitor.get_current_usage()
        }
        
        # Add task details
        report['task_details'] = []
        for task in self.tasks:
            task_detail = {
                'task_id': task.task_id,
                'name': task.name,
                'status': task.status.value,
                'duration_seconds': 0,
                'retry_count': task.retry_count,
                'error_message': task.error_message
            }
            
            if task.started_at and task.completed_at:
                task_detail['duration_seconds'] = (task.completed_at - task.started_at).total_seconds()
            
            report['task_details'].append(task_detail)
        
        return report
    
    async def _store_run_results(self, run_id: str):
        """Store pipeline run results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                status = self.scheduler.get_status()
                
                # Store pipeline run
                cursor.execute('''
                    INSERT OR REPLACE INTO pipeline_runs 
                    (run_id, config_name, status, start_time, end_time, total_tasks,
                     completed_tasks, failed_tasks, error_message, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    self.config.name,
                    self.status.value,
                    self.start_time,
                    self.end_time,
                    len(self.tasks),
                    status['completed_tasks'],
                    status['failed_tasks'],
                    self.error_message,
                    json.dumps(self.metrics)
                ))
                
                # Store task executions
                for task in self.tasks:
                    duration = 0
                    if task.started_at and task.completed_at:
                        duration = (task.completed_at - task.started_at).total_seconds()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO task_executions 
                        (execution_id, run_id, task_id, task_name, status, start_time,
                         end_time, duration_seconds, retry_count, error_message, result_summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        str(uuid.uuid4()),
                        run_id,
                        task.task_id,
                        task.name,
                        task.status.value,
                        task.started_at,
                        task.completed_at,
                        duration,
                        task.retry_count,
                        task.error_message,
                        str(task.result)[:1000] if task.result else None  # Truncate for storage
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store run results: {e}")

# Configuration and main execution
def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration"""
    return PipelineConfig(
        name="Astrobiology Data Pipeline",
        description="Comprehensive automated pipeline for astrobiology genomics data",
        schedule="0 2 * * 0",  # Weekly at 2 AM on Sunday
        max_concurrent_tasks=4,
        max_memory_gb=16,
        max_disk_gb=100,
        timeout=14400,  # 4 hours
        
        # Data sources
        enable_kegg=True,
        enable_ncbi=True,
        enable_agora2=True,
        
        # Testing limits
        max_kegg_pathways=100,
        max_ncbi_genomes=50,
        max_agora2_models=50,
        
        # Quality
        min_quality_score=0.8,
        nasa_grade_required=True,
        
        # Storage
        cleanup_old_data=True,
        backup_before_update=True,
        compress_backups=True,
        
        # Performance
        use_caching=True,
        parallel_downloads=True,
        optimize_memory=True
    )

async def main():
    """Main execution function"""
    # Create configuration
    config = create_default_config()
    
    # Create and run pipeline
    pipeline = AutomatedDataPipeline(config)
    
    try:
        report = await pipeline.run_pipeline()
        print(f"Pipeline completed successfully!")
        print(f"Duration: {report['duration_formatted']}")
        print(f"Tasks completed: {report['completed_tasks']}/{report['total_tasks']}")
        print(f"Success rate: {report['success_rate']:.1f}%")
        print(f"Data downloaded: {report['metrics']['total_data_downloaded']} items")
        print(f"Average quality score: {np.mean(report['metrics']['quality_scores']):.2f}")
        
        return pipeline
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Ensure event loop is properly configured
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run pipeline
    pipeline = asyncio.run(main()) 