"""Tests of the recognition of requests that do not fit a model's context window."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from app import server
from app.llm_errors import ContextWindowExceeded, context_window_error, cut_off_error


class BadRequest(Exception):
    status_code = 400


class ContextWindowTest(unittest.TestCase):
    def test_prompt_too_long_is_recognized(self):
        for message in (
            "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens",
            "Error code: 400 - {'error': {'code': 'context_length_exceeded'}}",
            "litellm.ContextWindowExceededError: prompt is too long",
        ):
            error = context_window_error(BadRequest(message), "m")
            self.assertIsInstance(error, ContextWindowExceeded, message)
            self.assertIn("context window of m", str(error))

    def test_other_failures_are_not(self):
        self.assertIsNone(context_window_error(BadRequest("invalid model name"), "m"))
        server_error = BadRequest("maximum context length")
        server_error.status_code = 500
        self.assertIsNone(context_window_error(server_error, "m"))

    def test_cut_off_answer(self):
        def response(reason):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content='[{"s": "a"'))],
                usage=SimpleNamespace(prompt_tokens=900, completion_tokens=31100),
            )
        self.assertIsNone(cut_off_error(response("stop"), "m"))
        error = cut_off_error(response("length"), "m")
        self.assertIn("900 prompt tokens + 31100 answer tokens", str(error))
        self.assertEqual(error.text, '[{"s": "a"')


class ZipTest(unittest.TestCase):
    def test_report_link_at_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "html").mkdir()
            (run / "html" / "index.html").write_text("report")
            names = zipfile.ZipFile(io.BytesIO(server._zip_directory(run))).namelist()
            self.assertIn("REPORT.html", names)
            self.assertIn("html/index.html", names)

    def test_no_link_without_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "runConfig.json").write_text("{}")
            names = zipfile.ZipFile(io.BytesIO(server._zip_directory(Path(tmp)))).namelist()
            self.assertEqual(names, ["runConfig.json"])


if __name__ == "__main__":
    unittest.main()
