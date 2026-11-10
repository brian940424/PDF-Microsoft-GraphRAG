"""Shared stop mechanism for long-running jobs (preprocessing, indexing, question answering, judging).

A job is registered under ``(project_id, kind)`` with a :class:`threading.Event`. Services accept
that event as ``stop_event`` and stop sending new requests once it is set; child processes started
for the job are terminated. The registry lives in the admin process, so a page reloaded while a job
runs can still find and stop it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .projects import ProjectError


PROCESS_STOP_GRACE_SECONDS = 10.0


class JobStopped(BaseException):
    """Raised inside a job when the user asked it to stop.

    It derives from BaseException (like KeyboardInterrupt) so the many ``except Exception``
    handlers that turn request failures into per-item errors do not swallow a stop request.
    """


class JobAlreadyRunning(ProjectError):
    pass


def raise_if_stopped(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise JobStopped("已停止")


@dataclass(eq=False)
class Job:
    project_id: str
    kind: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _registry: JobRegistry | None = field(default=None, repr=False)
    _handed_off: bool = field(default=False, repr=False)

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def release(self) -> None:
        if self._registry is not None:
            self._registry._remove(self)

    def run_in_thread(self, target: Callable[[], None]) -> threading.Thread:
        """Start `target` in a daemon thread that releases the job when it ends.

        The job then outlives the page that started it: a reloaded page can still stop it, and the
        registration is cleared only when the work itself finishes (including on errors).
        """
        self._handed_off = True

        def run() -> None:
            try:
                target()
            finally:
                self.release()

        thread = threading.Thread(target=run, daemon=True, name=f"job-{self.kind}-{self.project_id}")
        thread.start()
        return thread

    def release_unless_handed_off(self) -> None:
        if not self._handed_off:
            self.release()


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[tuple[str, str], Job] = {}

    def begin(self, project_id: str, kind: str) -> Job:
        """Register a new job; the same kind of job may run only once per project."""
        with self._lock:
            if (project_id, kind) in self._jobs:
                raise JobAlreadyRunning("此專案已有相同的工作正在執行，請等待完成或先停止")
            job = Job(project_id, kind, _registry=self)
            self._jobs[(project_id, kind)] = job
            return job

    @contextmanager
    def running(self, project_id: str, kind: str) -> Iterator[Job]:
        job = self.begin(project_id, kind)
        try:
            yield job
        finally:
            job.release()

    def get(self, project_id: str | None, kind: str) -> Job | None:
        if not project_id:
            return None
        with self._lock:
            return self._jobs.get((project_id, kind))

    def is_running(self, project_id: str | None, kind: str) -> bool:
        return self.get(project_id, kind) is not None

    def kinds(self, project_id: str | None) -> set[str]:
        with self._lock:
            return {kind for (owner, kind) in self._jobs if owner == project_id}

    def stop(self, project_id: str | None, kind: str) -> bool:
        job = self.get(project_id, kind)
        if job is None:
            return False
        job.stop()
        return True

    def _remove(self, job: Job) -> None:
        with self._lock:
            if self._jobs.get((job.project_id, job.kind)) is job:
                del self._jobs[(job.project_id, job.kind)]


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = PROCESS_STOP_GRACE_SECONDS) -> None:
    """SIGTERM the process group started with ``start_new_session=True``; SIGKILL it after the grace period."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        # Kill the group even when the leader exited, so no grandchild keeps running.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ProcessStopWatcher:
    """Terminates `process` when `stop_event` is set; use as a context manager around the wait."""

    def __init__(
        self,
        process: subprocess.Popen,
        stop_event: threading.Event | None,
        grace_seconds: float = PROCESS_STOP_GRACE_SECONDS,
    ) -> None:
        self.process = process
        self.stop_event = stop_event
        self.grace_seconds = grace_seconds
        self.stopped = False
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ProcessStopWatcher:
        if self.stop_event is not None:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()

    def _watch(self) -> None:
        assert self.stop_event is not None
        while not self._done.is_set():
            if self.stop_event.wait(0.2):
                if self.process.poll() is None:
                    self.stopped = True
                    terminate_process_group(self.process, self.grace_seconds)
                return
