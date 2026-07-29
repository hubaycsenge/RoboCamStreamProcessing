"""Frame queue and worker pool.

The IO thread must never block on inference — if it did, a slow model would stall
the socket, ZeroMQ's buffers would fill and the robot would see the link freeze
rather than degrade.  So receiving and processing are separated by a short
bounded queue.

The queue is deliberately tiny (default depth 2).  For a robot acting on what it
sees, a frame that has been waiting 400 ms is worse than useless, so when the
queue is full the oldest frame is evicted.  Every evicted frame still gets a
result sent back marked ``dropped``, which matters because the client uses
outstanding results to size its in-flight window; silently discarding a frame
would slowly starve the sender.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .processors import Processor
from .processors.base import Frame

log = logging.getLogger(__name__)


@dataclass
class ProcessedResult:
    """A finished frame on its way back to the IO thread."""

    session_id: str
    seq: int
    ok: bool
    reason: str
    frame: Frame
    data: Dict[str, Any]
    process_ms: float
    queue_ms: float


class FrameQueue:
    """Bounded, drop-on-full queue with visibility into what it discarded."""

    def __init__(self, max_depth: int = 2, drop_policy: str = "oldest") -> None:
        self.max_depth = max_depth
        self.drop_policy = drop_policy
        self._items: Deque[Tuple[int, Frame]] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.dropped_total = 0

    def put(self, frame: Frame) -> List[Frame]:
        """Enqueue a frame, returning any frames evicted to make room."""
        evicted: List[Frame] = []
        with self._cv:
            if self._closed:
                return [frame]
            if len(self._items) >= self.max_depth:
                if self.drop_policy == "newest":
                    # Refuse the arrival, keep the backlog ordered.
                    self.dropped_total += 1
                    return [frame]
                # "oldest": make room at the front.
                while len(self._items) >= self.max_depth:
                    _, old = self._items.popleft()
                    evicted.append(old)
                    self.dropped_total += 1
            self._items.append((time.monotonic_ns(), frame))
            self._cv.notify()
        return evicted

    def get(self, timeout: float = 0.1) -> Optional[Tuple[int, Frame]]:
        """Pop the oldest (enqueue_ns, frame), or None on timeout/close."""
        with self._cv:
            if not self._items and not self._closed:
                self._cv.wait(timeout)
            if self._items:
                return self._items.popleft()
            return None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def drain(self) -> List[Frame]:
        with self._cv:
            items = [f for _, f in self._items]
            self._items.clear()
            return items

    def __len__(self) -> int:
        with self._cv:
            return len(self._items)


class WorkerPool:
    """Threads that pull frames, run a processor and publish results.

    Each worker owns its own processor instance.  That is what a GPU model wants
    — one CUDA context and one set of weights per thread — and it means a
    stateful processor never sees interleaved frames from two threads.
    """

    def __init__(
        self,
        processor_factory: Callable[[], Processor],
        frame_queue: FrameQueue,
        result_queue: "queue.Queue[ProcessedResult]",
        workers: int = 1,
        name: str = "processor",
        on_result_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        self._factory = processor_factory
        self._frames = frame_queue
        self._results = result_queue
        self._n = workers
        self._name = name
        # Called after each result is queued so the IO thread can stop waiting
        # in poll() and send it immediately.
        self._notify = on_result_ready
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._setup_error: Optional[BaseException] = None

    def start(self, setup_timeout: float = 300.0) -> None:
        for i in range(self._n):
            t = threading.Thread(target=self._run, args=(i,), name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        # Wait for the first worker to finish setup so that model-loading errors
        # surface at startup rather than on the first frame.
        if not self._ready.wait(setup_timeout):
            raise RuntimeError(f"processor {self._name!r} did not finish setup within {setup_timeout}s")
        if self._setup_error is not None:
            raise RuntimeError(f"processor {self._name!r} failed to start") from self._setup_error

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._frames.close()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def _run(self, index: int) -> None:
        try:
            processor = self._factory()
            processor.setup()
        except BaseException as exc:  # noqa: BLE001 - reported to start()
            log.exception("worker %d: processor setup failed", index)
            if index == 0:
                self._setup_error = exc
                self._ready.set()
            return

        if index == 0:
            self._ready.set()
        log.info("worker %d ready (processor=%s)", index, processor.name)

        try:
            while not self._stop.is_set():
                item = self._frames.get(timeout=0.1)
                if item is None:
                    continue
                enqueued_ns, frame = item
                queue_ms = (time.monotonic_ns() - enqueued_ns) / 1e6

                t0 = time.perf_counter()
                try:
                    data = processor.process(frame) or {}
                    ok, reason = True, "ok"
                except Exception as exc:  # noqa: BLE001 - one bad frame is not fatal
                    log.exception("worker %d: processor raised on seq=%d", index, frame.seq)
                    data = {"error": f"{type(exc).__name__}: {exc}"}
                    ok, reason = False, "processor_failed"
                process_ms = (time.perf_counter() - t0) * 1000.0

                self._results.put(
                    ProcessedResult(
                        session_id=frame.session_id,
                        seq=frame.seq,
                        ok=ok,
                        reason=reason,
                        frame=frame,
                        data=data,
                        process_ms=process_ms,
                        queue_ms=queue_ms,
                    )
                )
                if self._notify is not None:
                    self._notify()
        finally:
            try:
                processor.close()
            except Exception:  # pragma: no cover - teardown best effort
                log.exception("worker %d: processor close failed", index)
            log.info("worker %d stopped", index)
