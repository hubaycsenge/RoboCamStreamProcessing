"""Letting worker threads interrupt the IO thread's poll().

Without this, a finished result waits for the current ``poll()`` to time out
before it can be sent — up to ``server.io_poll_ms`` of latency added to every
frame for no reason.  Measured on nipg36 that was ~20 ms out of a ~23 ms
server-side budget, i.e. almost all of it.

The mechanism is a ZeroMQ inproc PUSH/PULL pair: workers push a zero-byte
message, the IO loop has the PULL end registered in its poller and wakes at
once.  It stays inside ZeroMQ, so no extra file descriptors or platform-specific
self-pipe tricks.
"""

from __future__ import annotations

import threading

import zmq


class Waker:
    def __init__(self, ctx: zmq.Context, endpoint: str) -> None:
        self.ctx = ctx
        self.endpoint = endpoint
        self.rx = ctx.socket(zmq.PULL)
        self.rx.setsockopt(zmq.LINGER, 0)
        # inproc requires the bind to happen before any connect.
        self.rx.bind(endpoint)
        # ZeroMQ sockets are not thread-safe, so each worker gets its own.
        self._local = threading.local()
        self._senders: list[zmq.Socket] = []
        self._lock = threading.Lock()

    def wake(self) -> None:
        try:
            self._sender().send(b"", zmq.NOBLOCK)
        except zmq.Again:
            # Pipe already full: the IO thread has pending wakeups to process,
            # so it will get to the result anyway.
            pass
        except zmq.ZMQError:
            pass

    def drain(self) -> None:
        while True:
            try:
                self.rx.recv(zmq.NOBLOCK)
            except zmq.Again:
                return

    def close(self) -> None:
        with self._lock:
            for sock in self._senders:
                sock.close(linger=0)
            self._senders.clear()
        self.rx.close(linger=0)

    def _sender(self) -> zmq.Socket:
        sock = getattr(self._local, "sock", None)
        if sock is None:
            sock = self.ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 16)
            sock.connect(self.endpoint)
            self._local.sock = sock
            with self._lock:
                self._senders.append(sock)
        return sock
