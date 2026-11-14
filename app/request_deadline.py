"""Cancellable, wall-clock HTTP deadlines for the synchronous model clients.

A socket read timeout resets whenever data arrives. This transport instead
budgets the entire request, including validation, connection setup and reading
the complete body. Network I/O runs on one shared async loop, so timing out
cancels the actual request rather than abandoning a blocked worker thread.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
import math
import threading
import time
from typing import Callable

import httpx


class RequestDeadlineExceeded(httpx.TimeoutException):
    def __init__(self, seconds: float, request: httpx.Request):
        self.seconds = seconds
        super().__init__(f"Total request deadline of {seconds:g}s exceeded; request cancelled.", request=request)


class _NetworkLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="request-network", daemon=True)
        self.thread.start()


_network: _NetworkLoop | None = None
_network_lock = threading.Lock()


def _network_loop() -> asyncio.AbstractEventLoop:
    global _network
    with _network_lock:
        if _network is None:
            _network = _NetworkLoop()
        return _network.loop


class DeadlineTransport(httpx.BaseTransport):
    """Buffer non-streaming SDK responses within one cancellable deadline.

    The outer sync client handles redirects. Its redirect requests preserve
    extensions, so they share the original attempt's deadline. A new SDK retry
    builds a fresh request and gets a fresh deadline. Async clients and their
    connection pools are used exclusively on the network loop.
    """

    def __init__(self, seconds: float, validate: Callable[[str], None]):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("The total request deadline must be positive and finite.")
        self.seconds = seconds
        self.validate = validate
        self._client: httpx.AsyncClient | None = None
        self._lock = threading.Lock()
        self._pending: set[Future] = set()
        self._closed = False

    async def _send(self, request: httpx.Request, expires: float) -> httpx.Response:
        async with asyncio.timeout(max(0, expires - time.monotonic())):
            # DNS validation may block; cancellation must not let it block the
            # network loop. A late validation result never sends a request.
            await asyncio.to_thread(self.validate, str(request.url))
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=self.seconds, follow_redirects=False,
                    limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
                )
            outgoing = self._client.build_request(
                request.method, request.url, headers=request.headers,
                content=request.content, extensions=dict(request.extensions),
            )
            response = await self._client.send(outgoing, stream=True)
            try:
                # Preserve raw bytes: the outer sync client performs decoding.
                body = b"".join([chunk async for chunk in response.aiter_raw()])
                extensions = {k: v for k, v in response.extensions.items() if k != "network_stream"}
                return httpx.Response(response.status_code, headers=response.headers,
                                      stream=httpx.ByteStream(body), extensions=extensions)
            finally:
                await response.aclose()

    def _finished(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        expires = request.extensions.setdefault("gptkb_deadline", time.monotonic() + self.seconds)
        request.read()  # SDK requests already have their small JSON body buffered.
        with self._lock:
            if self._closed:
                raise RuntimeError("The request transport is closed.")
            future = asyncio.run_coroutine_threadsafe(self._send(request, expires), _network_loop())
            self._pending.add(future)
        future.add_done_callback(self._finished)
        try:
            return future.result(timeout=max(0, expires - time.monotonic()))
        except FutureTimeout:
            future.cancel()  # Propagates cancellation into HTTPX and closes its socket.
            raise RequestDeadlineExceeded(self.seconds, request) from None

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        if self._client is not None:
            asyncio.run_coroutine_threadsafe(self._close(), _network_loop()).result(timeout=5)


def deadline_reason(error: Exception) -> str | None:
    """Recover the deadline detail from the OpenAI SDK's timeout wrapper."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, RequestDeadlineExceeded):
            return str(error)
        error = error.__cause__
    return None
