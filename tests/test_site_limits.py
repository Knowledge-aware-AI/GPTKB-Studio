"""Tests of the per-site limits on the shared ScaDS.AI key and of read-only
(locked) folders, through the HTTP API with the run processes stubbed out."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import server

SHARED_KEY_RUN = {
    "llm_url": "https://llm.scads.ai/v1", "api_key": "", "model": "m", "topic": "t", "seed_entity": "s",
}
OWN_KEY_RUN = {**SHARED_KEY_RUN, "llm_url": "https://api.openai.com/v1", "api_key": "sk-own"}


def hanging_run(*args):
    """A run process that never ends by itself (a request hanging until its timeout)."""
    time.sleep(600)


class FakeProcess:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class SiteTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        runs = Path(self.tmp.name)
        self.launched: list[tuple[server.RunController, dict]] = []

        def launch(ctrl, run_dir, config_data):
            ctrl.process = FakeProcess()
            self.launched.append((ctrl, config_data))

        patches = [
            mock.patch.object(server, "RUNS_DIR", runs),
            mock.patch.object(server, "FOLDERS_PATH", runs / "_folders.json"),
            mock.patch.object(server, "DEFAULT_SCADS_API_KEY", "shared"),
            mock.patch.object(server, "_check_endpoint", lambda url: None),
            mock.patch.object(server, "_launch_runner", launch),
            mock.patch.object(server, "progress_counts", lambda path: {"queued": 1, "pending": 0}),
            mock.patch.dict(server.controllers, clear=True),
            mock.patch.object(server.EndpointRateLimiter, "check", lambda self, key: None),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)
        self.runs = runs
        self.client = TestClient(server.app)

    def start(self, payload, expected=200):
        response = self.client.post("/api/run", json=payload)
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def make_run(self, run_id, folder_id=None):
        run_dir = self.runs / run_id
        run_dir.mkdir()
        (run_dir / "runConfig.json").write_text(json.dumps({**SHARED_KEY_RUN, "pipeline": "gptkb2", "folder_id": folder_id}))
        return run_dir


class SharedKeyLimitTest(SiteTestCase):
    def test_no_limits_by_default(self):
        with mock.patch.object(server, "SHARED_KEY_RUN_SECONDS", None), \
                mock.patch.object(server, "MAX_SHARED_KEY_RUNS", None):
            self.start(SHARED_KEY_RUN)
        self.assertIsNone(self.launched[0][1]["max_runtime_seconds"])

    def test_shared_key_runs_are_limited(self):
        with mock.patch.object(server, "SHARED_KEY_RUN_SECONDS", 600):
            self.start(SHARED_KEY_RUN)
            self.start({**SHARED_KEY_RUN, "max_runtime_seconds": 120})
            self.start({**SHARED_KEY_RUN, "max_runtime_seconds": 99999})
            self.start(OWN_KEY_RUN)
        self.assertEqual([config["max_runtime_seconds"] for _, config in self.launched], [600, 120, 600, None])

    def test_limit_counts_over_resumes(self):
        run_dir = self.make_run("r1")
        with mock.patch.object(server, "SHARED_KEY_RUN_SECONDS", 600):
            server._add_shared_key_seconds(run_dir, 450.4)
            self.assertEqual(self.client.post("/api/runs/r1/resume", json=SHARED_KEY_RUN).status_code, 200)
            self.assertEqual(self.launched[-1][1]["max_runtime_seconds"], 149)
            self.launched[-1][0].process.alive = False

            server._add_shared_key_seconds(run_dir, 150)
            response = self.client.post("/api/runs/r1/resume", json=SHARED_KEY_RUN)
            self.assertEqual(response.status_code, 409)
            self.assertIn("own API key", response.json()["detail"])
            # with the user's own key the run can go on
            self.assertEqual(self.client.post("/api/runs/r1/resume", json=OWN_KEY_RUN).status_code, 200)

    def test_cap_on_runs_at_the_same_time(self):
        with mock.patch.object(server, "MAX_SHARED_KEY_RUNS", 2):
            self.start(SHARED_KEY_RUN)
            self.start(SHARED_KEY_RUN)
            self.assertIn("busy", self.start(SHARED_KEY_RUN, expected=503)["detail"])
            self.start(OWN_KEY_RUN)  # own keys are not capped
            self.launched[0][0].process.alive = False
            self.start(SHARED_KEY_RUN)

    def test_site_settings(self):
        with mock.patch.object(server, "_shared_key_minutes", 10), mock.patch.object(server, "MAX_SHARED_KEY_RUNS", 20):
            self.assertEqual(self.client.get("/api/site").json(), {"shared_key_minutes": 10, "max_shared_key_runs": 20, "label": None})


class HardStopTest(unittest.TestCase):
    def test_server_ends_a_run_past_its_time(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(server, "_run_process_entry", hanging_run), \
                mock.patch.object(server, "SHARED_KEY_TIME_SAVE_SECONDS", 0.5):
            run_dir = Path(tmp)
            ctrl = server.RunController("hang", run_dir)
            ctrl.shared_key = True
            ctrl.hard_stop_at = time.monotonic() + 1
            server._launch_runner(ctrl, run_dir, {})
            ctrl.listener_thread.join(timeout=15)
            self.assertFalse(ctrl.process.is_alive())
            self.assertEqual(ctrl.status, "stopped")
            self.assertGreaterEqual(server._shared_key_seconds_used(run_dir), 1)


class LockedFolderTest(SiteTestCase):
    def setUp(self):
        super().setUp()
        (self.runs / "_folders.json").write_text(json.dumps({"folders": [
            {"id": "show", "name": "Showcase", "locked": True},
            {"id": "mine", "name": "Mine"},
        ]}))
        self.make_run("shown", folder_id="show")
        self.make_run("free")

    def assertForbidden(self, response):
        self.assertEqual(response.status_code, 403, response.text)

    def test_folder_cannot_be_changed(self):
        self.assertForbidden(self.client.patch("/api/folders/show", json={"name": "x"}))
        self.assertForbidden(self.client.delete("/api/folders/show"))
        self.assertEqual(self.client.patch("/api/folders/mine", json={"name": "Ours"}).status_code, 200)

    def test_runs_in_it_cannot_be_changed(self):
        self.assertForbidden(self.client.delete("/api/runs/shown"))
        self.assertForbidden(self.client.patch("/api/runs/shown", json={"folder_id": None}))
        self.assertForbidden(self.client.post("/api/runs/shown/resume", json=OWN_KEY_RUN))
        self.assertEqual(self.client.post("/api/runs/bulk-delete", json={"run_ids": ["shown"]}).json()["deleted"], [])
        moved = self.client.post("/api/runs/bulk-move", json={"run_ids": ["shown"], "folder_id": "mine"}).json()
        self.assertEqual(moved["moved"], [])
        self.assertTrue((self.runs / "shown").exists())

    def test_nothing_can_be_put_into_it(self):
        self.assertForbidden(self.client.patch("/api/runs/free", json={"folder_id": "show"}))
        self.assertForbidden(self.client.post("/api/runs/bulk-move", json={"run_ids": ["free"], "folder_id": "show"}))
        self.assertForbidden(self.client.post("/api/run", json={**SHARED_KEY_RUN, "folder_id": "show"}))

    def test_other_runs_stay_free(self):
        self.assertEqual(self.client.patch("/api/runs/free", json={"folder_id": "mine"}).status_code, 200)
        self.assertEqual(self.client.delete("/api/runs/free").status_code, 200)


if __name__ == "__main__":
    unittest.main()
