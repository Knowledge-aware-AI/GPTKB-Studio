"""Disambiguation stages must be able to fill the configured worker pool."""

import sqlite3
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from app.disambig import store
from app.disambig.runner import DisambigConfig, DisambiguatedKBCRunner


class ConcurrencyTest(unittest.TestCase):
    def run_stage(self, stage, workers):
        # Keep the real stage code and executor. Replace just model inference,
        # candidate retrieval and housekeeping with deterministic offline fakes.
        runner = object.__new__(DisambiguatedKBCRunner)
        runner.config = DisambigConfig(llm_url='http://fake/v1', api_key='test', model='test',
                                       topic='test', seed_concept='seed', nthreads=workers)
        runner.db = sqlite3.connect(':memory:')
        self.addCleanup(runner.db.close)
        runner.db.executescript(store.SCHEMA)
        runner.db.execute("INSERT INTO concepts (id, seq, label, description) VALUES ('C0', 0, 'seed', 'Seed concept')")
        runner.db.execute("INSERT INTO predicates (id, seq, label, description) VALUES ('P0', 0, 'instanceOf', 'Membership')")
        for i in range(60):
            runner.db.execute(
                "INSERT INTO triples (id, seq, subject_id, subject_label, predicate_label, predicate_status, "
                "object_label, object_type, object_status) VALUES (?, ?, 'C0', 'seed', ?, ?, ?, ?, ?)",
                (f'T{i}', i, f'{i:05d}-relation', 'generated' if stage == 'pd' else 'finished',
                 f'concept-{i}', 'literal' if stage == 'pd' else 'concept',
                 'finished' if stage == 'pd' else 'cr_finished'),
            )
        runner.embedder = SimpleNamespace(embed=lambda labels: np.zeros((len(labels), 1024), dtype=np.float32))
        runner.concept_index = SimpleNamespace(search=lambda vectors, *args: [['C0'] for _ in vectors])
        runner.predicate_index = SimpleNamespace(search=lambda vectors, *args: [['P0'] for _ in vectors])
        runner.active_requests = 0
        runner.active_lock = threading.Lock()
        runner.dirty_concepts = set()
        runner._current = mock.Mock()
        runner._log = mock.Mock()
        runner._check_stop = mock.Mock(return_value=False)
        runner._emit_status = mock.Mock()
        runner._maybe_write_report = mock.Mock()

        barrier = threading.Barrier(workers, timeout=3)
        lock = threading.Lock()
        calls = active = peak = 0

        def infer(*args):
            nonlocal calls, active, peak
            with lock:
                ticket = calls
                calls += 1
                active += 1
                peak = max(peak, active)
            try:
                if ticket < workers:
                    barrier.wait()  # No result until the first pool is full.
                return 'F'  # None of the candidates: advance to description generation.
            finally:
                with lock:
                    active -= 1

        runner.llm = SimpleNamespace(chat=infer)
        with mock.patch('app.disambig.runner.cluster_labels', side_effect=lambda vectors, distance: np.arange(len(vectors))):
            self.assertTrue(getattr(runner, f'_stage_{stage}')())
        self.assertEqual(calls, 60)
        self.assertEqual(peak, workers)
        self.assertEqual(runner.active_requests, 0)
        self.assertEqual(runner.db.execute('SELECT SUM(attempts) FROM triples').fetchone()[0], 0)
        field, expected = ('predicate_status', 'pd_finished') if stage == 'pd' else ('object_status', 'cd_finished1')
        self.assertEqual(runner.db.execute(f'SELECT COUNT(*) FROM triples WHERE {field} = ?', (expected,)).fetchone()[0], 60)

    def test_both_stages_reach_fifty_concurrent_requests(self):
        for stage in ('pd', 'cd1'):
            with self.subTest(stage=stage):
                self.run_stage(stage, 50)

    def test_smaller_worker_limit_is_respected(self):
        for stage in ('pd', 'cd1'):
            with self.subTest(stage=stage):
                self.run_stage(stage, 7)
