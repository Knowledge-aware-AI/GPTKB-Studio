"""Keeps the log talking while a run waits.

A run can go quiet for minutes: a model request may take up to its timeout to
answer (or not answer at all), and an embedding request can stall the same
way. Every model and embedding request is registered here while it runs, and a
background thread
- warns once when a single request has had no answer for SLOW_SECONDS, and
- writes a status line whenever nothing has been logged for QUIET_SECONDS:
  the current step, the requests still waiting (per task and model, with the
  oldest one's age) and the triples still to disambiguate.
Busy runs log often enough that the heartbeat stays silent.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator

QUIET_SECONDS = 60.0
SLOW_SECONDS = 120.0
CHECK_SECONDS = 5.0


def format_age(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


class _Request:
    __slots__ = ("task", "model", "timeout", "started", "warned")

    def __init__(self, task: str, model: str, timeout: float | None) -> None:
        self.task, self.model, self.timeout = task, model, timeout
        self.started = time.monotonic()
        self.warned = False


class Heartbeat:
    def __init__(
        self,
        log: Callable[[str, str], None],
        describe: Callable[[], tuple[str | None, str | None]],
        stop_event: threading.Event,
        paused: Callable[[], bool] = lambda: False,
        quiet_seconds: float = QUIET_SECONDS,
        slow_seconds: float = SLOW_SECONDS,
    ) -> None:
        """`log(message, level)` writes to the run's log; `describe()` returns
        the current step and a short progress note (either may be None)."""
        self._log = log
        self._describe = describe
        self._stop = stop_event
        self._paused = paused
        self.quiet_seconds = quiet_seconds
        self.slow_seconds = slow_seconds
        self._lock = threading.Lock()
        self._requests: dict[int, _Request] = {}
        self._ids = itertools.count()
        self._last_log = time.monotonic()
        self._thread: threading.Thread | None = None

    # ---- called by the run -------------------------------------------------

    def logged(self) -> None:
        """Something was written to the log: the run is not quiet."""
        self._last_log = time.monotonic()

    @contextmanager
    def request(self, task: str, model: str, timeout: float | None = None) -> Iterator[None]:
        """Registers one request for as long as it runs."""
        key = next(self._ids)
        with self._lock:
            self._requests[key] = _Request(task, model, timeout)
        try:
            yield
        finally:
            with self._lock:
                self._requests.pop(key, None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()

    # ---- the background thread ---------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(CHECK_SECONDS):
            try:
                self.check()
            except Exception:
                pass  # the heartbeat must never break a run

    def check(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._paused():
            self._last_log = now  # a paused run is quiet on purpose
            return
        with self._lock:
            requests = list(self._requests.values())
            slow = [r for r in requests if not r.warned and now - r.started >= self.slow_seconds]
            for request in slow:
                request.warned = True
        for request in slow:
            retry = (f"; it will be cancelled and retried after {format_age(request.timeout)}"
                     if request.timeout else "")
            self._log(f"{request.task} request to {request.model} has had no answer for "
                      f"{format_age(now - request.started)}{retry}.", "warning")
        if now - self._last_log >= self.quiet_seconds:
            self._log(self.status_line(requests, now), "info")
            self._last_log = now

    def status_line(self, requests: list[_Request], now: float) -> str:
        step, progress = self._describe()
        head = f"Still working ({step})" if step else "Still working"
        groups: dict[tuple[str, str], list[_Request]] = defaultdict(list)
        for request in requests:
            groups[(request.task, request.model)].append(request)
        parts = []
        for (task, model), items in sorted(groups.items(), key=lambda g: min(r.started for r in g[1])):
            oldest = min(r.started for r in items)
            count = f"{len(items)} {task} request{'s' if len(items) != 1 else ''}"
            age = (f", the oldest for {format_age(now - oldest)}" if len(items) > 1
                   else f" for {format_age(now - oldest)}")
            timeout = items[0].timeout
            limit = f" (timeout after {format_age(timeout)})" if timeout else ""
            parts.append(f"{count} to {model} waiting for an answer{age}{limit}")
        if not parts:
            parts.append("no request is waiting for an answer")
        line = f"{head}: {'; '.join(parts)}"
        if progress:
            line += f"; {progress}"
        return line + "."
