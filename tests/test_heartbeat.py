"""Tests of the status lines a run writes while its log is quiet."""

from __future__ import annotations

import threading
import time
import unittest

from app.heartbeat import Heartbeat, format_age


class HeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.paused = False
        self.beat = Heartbeat(lambda message, level: (self.lines.append((level, message)), self.beat.logged()),
                              lambda: ("CD: 40 objects", "512 triples to disambiguate"),
                              threading.Event(), paused=lambda: self.paused)

    def test_quiet_run_gets_a_status_line(self):
        with self.beat.request("CD", "Qwen/Qwen3.8-27B", 300), self.beat.request("CD", "Qwen/Qwen3.8-27B", 300):
            start = time.monotonic()
            self.beat.check(start + 30)
            self.assertEqual(self.lines, [])  # not quiet long enough
            self.beat.check(start + 61)
        self.assertEqual(self.lines, [("info",
            "Still working (CD: 40 objects): 2 CD requests to Qwen/Qwen3.8-27B waiting for an answer, "
            "the oldest for 1m 01s (timeout after 5m 00s); 512 triples to disambiguate.")])

    def test_status_line_repeats_only_while_quiet(self):
        start = time.monotonic()
        self.beat.check(start + 61)
        self.beat.check(start + 62)  # the status line itself counts as logging
        self.assertEqual(len(self.lines), 1)
        self.assertIn("no request is waiting for an answer", self.lines[0][1])

    def test_slow_request_warns_once(self):
        with self.beat.request("Elicitation", "google/gemma-4-26B-A4B-it", 300):
            start = time.monotonic()
            self.beat.logged()
            self.beat.check(start + 121)
            self.beat.logged()
            self.beat.check(start + 150)
        warnings = [m for level, m in self.lines if level == "warning"]
        self.assertEqual(warnings, ["Elicitation request to google/gemma-4-26B-A4B-it has had no answer for 2m 01s; "
                                    "it will be cancelled and retried after 5m 00s."])

    def test_paused_run_stays_silent(self):
        self.paused = True
        with self.beat.request("CR", "m", 300):
            self.beat.check(time.monotonic() + 600)
        self.assertEqual(self.lines, [])

    def test_format_age(self):
        self.assertEqual([format_age(s) for s in (5, 65, 300)], ["5s", "1m 05s", "5m 00s"])


if __name__ == "__main__":
    unittest.main()
