"""
Async Video Processing Queue
Solves: long-running video tasks blocking the Gradio UI thread.

Architecture:
- asyncio.Queue for job submission
- ThreadPoolExecutor for CPU-bound FFmpeg/MoviePy tasks
  (they release GIL during subprocess calls)
- Per-job progress callbacks streamed to Gradio via asyncio.Event
- Automatic cleanup of completed jobs after TTL
"""

import asyncio
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class VideoJob:
    job_id:      str
    fn:          Callable
    args:        tuple
    kwargs:      dict
    status:      JobStatus     = JobStatus.QUEUED
    progress:    float         = 0.0          # 0.0 – 1.0
    progress_msg: str          = "Queued"
    result:      Optional[Any] = None
    error:       Optional[str] = None
    created_at:  float         = field(default_factory=time.time)
    started_at:  Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.started_at:
            end = self.finished_at or time.time()
            return round(end - self.started_at, 2)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "job_id":      self.job_id,
            "status":      self.status.value,
            "progress":    self.progress,
            "message":     self.progress_msg,
            "result":      self.result,
            "error":       self.error,
            "elapsed_s":   self.elapsed,
        }


class VideoProcessingQueue:
    """
    Singleton async queue manager.
    
    Usage:
        queue = VideoProcessingQueue.get_instance()
        job_id = await queue.submit(trim_video, params)
        status = queue.get_status(job_id)
        result = await queue.wait_for(job_id, timeout=300)
    """

    _instance: Optional["VideoProcessingQueue"] = None

    @classmethod
    def get_instance(cls) -> "VideoProcessingQueue":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        max_workers: int = 2,     # Conservative: 2 parallel video jobs
        max_queue:   int = 20,
        job_ttl:     int = 3600,  # Clean completed jobs after 1 hour
    ):
        self._queue:    asyncio.Queue                = asyncio.Queue(maxsize=max_queue)
        self._jobs:     Dict[str, VideoJob]          = {}
        self._executor: ThreadPoolExecutor           = ThreadPoolExecutor(max_workers=max_workers)
        self._job_ttl   = job_ttl
        self._running   = False
        self._worker_task: Optional[asyncio.Task]   = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit(
        self,
        fn: Callable,
        *args,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs,
    ) -> str:
        """
        Enqueue a video processing job.
        Returns job_id immediately — caller polls or awaits completion.
        """
        job_id = uuid.uuid4().hex[:12]
        job    = VideoJob(job_id=job_id, fn=fn, args=args, kwargs=kwargs)
        self._jobs[job_id] = job

        await self._queue.put((job, progress_callback))
        logger.info(f"Job {job_id} queued: {fn.__name__}")

        if not self._running:
            await self._start_worker()

        return job_id

    def get_status(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    async def wait_for(self, job_id: str, timeout: float = 600) -> dict:
        """Block until job completes or timeout (seconds) elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self._jobs.get(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job.to_dict()
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    async def cancel(self, job_id: str) -> bool:
        """Mark a queued job as cancelled (running jobs cannot be interrupted)."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            return True
        return False

    def cleanup_old_jobs(self):
        """Remove completed jobs older than TTL."""
        now      = time.time()
        to_delete = [
            jid for jid, j in self._jobs.items()
            if j.finished_at and (now - j.finished_at) > self._job_ttl
        ]
        for jid in to_delete:
            del self._jobs[jid]
        if to_delete:
            logger.info(f"Cleaned {len(to_delete)} expired jobs")

    # ── Internal Worker ───────────────────────────────────────────────────────

    async def _start_worker(self):
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self):
        logger.info("Video queue worker started")
        try:
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.2)
                    # Auto-stop worker if queue is empty
                    if self._queue.empty():
                        self._running = False
                        break
                    continue

                job, progress_cb = item

                if job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                await self._run_job(job, progress_cb)
                self._queue.task_done()
                self.cleanup_old_jobs()
        except Exception as e:
            logger.error(f"Worker loop crashed: {e}")
            self._running = False

    async def _run_job(
        self,
        job: VideoJob,
        progress_cb: Optional[Callable],
    ):
        job.status     = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress   = 0.05
        job.progress_msg = "Starting..."

        if progress_cb:
            progress_cb(job.progress, job.progress_msg)

        loop = asyncio.get_running_loop()

        def _execute():
            return job.fn(*job.args, **job.kwargs)

        try:
            job.progress     = 0.1
            job.progress_msg = "Processing..."
            if progress_cb:
                progress_cb(0.1, "Processing...")

            # Run blocking operation in thread pool
            result = await loop.run_in_executor(self._executor, _execute)

            job.status       = JobStatus.COMPLETED
            job.result       = result
            job.progress     = 1.0
            job.progress_msg = "Done"
            job.finished_at  = time.time()

            if progress_cb:
                progress_cb(1.0, f"Completed in {job.elapsed}s")

            logger.info(f"Job {job.job_id} completed in {job.elapsed}s")

        except Exception as e:
            job.status       = JobStatus.FAILED
            job.error        = str(e)
            job.progress_msg = f"Failed: {e}"
            job.finished_at  = time.time()

            logger.error(
                f"Job {job.job_id} failed: {e}\n{traceback.format_exc()}"
            )
            if progress_cb:
                progress_cb(job.progress, f"❌ Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience async wrappers used by Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

async def submit_video_job(fn: Callable, params, timeout: int = 600) -> dict:
    """
    One-shot helper: submit job, wait for result, return dict.
    Used by Gradio event handlers that need a simple await-and-return.
    """
    queue = VideoProcessingQueue.get_instance()
    job_id = await queue.submit(fn, params)
    return await queue.wait_for(job_id, timeout=timeout)