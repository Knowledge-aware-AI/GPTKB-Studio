"""Tests that API keys never leave a run in what it emits (log, status)."""

from __future__ import annotations

import unittest
from unittest import mock

from app import server
from app.redact import redactor

KEY = "sk-user-0123456789abcdef"


class RedactorTest(unittest.TestCase):
    def test_keys_are_masked_everywhere(self):
        redact = redactor(KEY, "", None, "short")
        self.assertEqual(redact(f"Error 401: invalid key {KEY}."), "Error 401: invalid key ***.")
        self.assertEqual(redact({"message": KEY, "n": 3, "items": [KEY]}), {"message": "***", "n": 3, "items": ["***"]})
        self.assertEqual(redact("a short text"), "a short text")  # too short to be a key, left alone


class ListQueue(list):
    put = list.append


class RunEventsTest(unittest.TestCase):
    def test_events_of_a_run_carry_no_key(self):
        class EchoingRunner:
            """A run whose provider echoes the key in an error message."""

            def __init__(self, config, run_dir, emit, pause_event, stop_event):
                self.emit = emit

            def run(self):
                self.emit({"type": "log", "level": "error", "message": f"401 Incorrect API key provided: {KEY}"})

        events = ListQueue()
        config = {"pipeline": "gptkb2", "llm_url": "https://example.org/v1", "api_key": KEY, "model": "m",
                  "topic": "t", "seed_entity": "s", "embedding_api_key": "shared-embedding-key"}
        with mock.patch.object(server, "DisambiguatedKBCRunner", EchoingRunner):
            server._run_process_entry(config, "/tmp", events, None, None)
        self.assertEqual(events[0]["message"], "401 Incorrect API key provided: ***")


if __name__ == "__main__":
    unittest.main()
