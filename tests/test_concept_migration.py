"""Compatibility coverage for the former individual/class storage model."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
import threading
from unittest import mock

from app.concept_compat import normalize_config
from app.disambig import store


def legacy_database(path):
    db = sqlite3.connect(path)
    schema = (Path(__file__).parent / "fixtures" / "legacy_schema.sql").read_text(encoding="utf-8")
    db.executescript(schema)
    db.execute("INSERT INTO entities (id, seq, label, description, status) VALUES ('E0', 0, 'Alice', 'A physicist.', 'explored')")
    db.execute("INSERT INTO entities (id, seq, label, description, status, depth) VALUES ('E1', 1, 'Bob', 'Another physicist.', 'unexplored', 1)")
    db.execute("INSERT INTO concepts VALUES ('C0', 0, 'physicist', 'A scientist studying physics.', 'T0')")
    db.execute("INSERT INTO concept_aliases VALUES ('C0', 'physics researcher')")
    db.execute("INSERT INTO entity_aliases VALUES ('E0', 'Alice Smith')")
    db.execute("INSERT INTO predicates VALUES ('P0', 0, 'instanceOf', 'Class membership.', NULL)")
    db.execute("INSERT INTO predicates VALUES ('P1', 1, 'field', 'A field of work.', NULL)")
    for seq, predicate, obj, kind, status in [
        (0, 'instanceOf', 'physicist', 'class', 'finished'),
        (1, 'field', 'physics', 'literal', 'finished'),
        (2, 'colleague', 'Bob', 'ne', 'ner_finished'),
        (3, 'instanceOf', 'scientist', 'class', 'finished'),
    ]:
        db.execute("INSERT INTO triples (id, seq, subject_id, subject_label, predicate_label, predicate_id, "
                   "predicate_status, object_label, object_type, object_status) VALUES (?, ?, 'E0', 'Alice', ?, ?, 'finished', ?, ?, ?)",
                   (f'T{seq}', seq, predicate, 'P0' if predicate == 'instanceOf' else 'P1', obj, kind, status))
    db.execute("INSERT INTO instance_triples VALUES ('T0', 'E0', 'physicist', 'C0', 'finished', 0)")
    db.execute("INSERT INTO instance_triples VALUES ('T3', 'E0', 'scientist', NULL, 'cd_finished', 0)")
    db.execute("INSERT INTO decisions (triple_id, stage, target, context, candidates, answer, outcome, chosen_id) "
               "VALUES ('T2', 'ned1', 'Bob', '[Alice, colleague, Bob]', '[]', 'A', 'merged', 'E1')")
    db.execute("INSERT INTO decisions (triple_id, stage, target, context, candidates, answer, outcome, chosen_id) "
               "VALUES ('T0', 'cdg', 'physicist', '[Alice, instanceOf, physicist]', '[]', 'A scientist studying physics.', 'new', 'C0')")
    db.executemany("INSERT INTO meta VALUES (?, ?)", [('next_entity', '2'), ('next_concept', '1'), ('next_predicate', '2'), ('next_triple', '4')])
    db.commit()
    db.close()


class MigrationTest(unittest.TestCase):
    def test_unifies_classes_and_preserves_pending_work_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'knowledgeBase.db'
            legacy_database(path)
            counts = store.progress_counts(path)
            self.assertEqual(counts['queued'], 2)
            self.assertGreater(counts['pending'], 0)
            db = store.connect(path)
            self.assertEqual(db.execute("SELECT id, depth, status FROM concepts WHERE label = 'physicist'").fetchone(),
                             ('C0', 1, 'unexplored'))
            self.assertEqual(db.execute("SELECT object_id, object_type FROM triples WHERE id = 'T0'").fetchone(), ('C0', 'concept'))
            self.assertEqual(db.execute("SELECT object_type, object_status FROM triples WHERE id = 'T1'").fetchone(), ('unrecognized', 'generated'))
            self.assertEqual(db.execute("SELECT object_type, object_status FROM triples WHERE id = 'T2'").fetchone(), ('concept', 'cr_finished'))
            self.assertEqual(db.execute("SELECT object_type, object_status FROM triples WHERE id = 'T3'").fetchone(), ('concept', 'cr_finished'))
            self.assertEqual(set(db.execute("SELECT alias FROM concept_aliases")), {('Alice Smith',), ('physics researcher',)})
            self.assertEqual(db.execute("SELECT label FROM predicates WHERE id = 'P1'").fetchone()[0], 'field')
            self.assertEqual(store.get_meta(db, 'seed_id'), 'E0')
            self.assertGreaterEqual(int(store.get_meta(db, 'next_concept')), 3)
            self.assertEqual(set(db.execute("SELECT stage FROM decisions")), {('cd1',), ('legacy_class_description',)})
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertNotIn('entities', tables)
            self.assertNotIn('instance_triples', tables)
            before = db.execute("SELECT * FROM concepts ORDER BY seq").fetchall()
            db.close()
            db = store.connect(path)
            self.assertEqual(db.execute("SELECT * FROM concepts ORDER BY seq").fetchall(), before)
            db.close()

    def test_migration_rolls_back_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'knowledgeBase.db'
            legacy_database(path)
            with sqlite3.connect(path) as db:
                db.execute('DROP TABLE run')
                with self.assertRaises(sqlite3.OperationalError):
                    store._migrate_legacy(db)
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                self.assertIn('entities', tables)
                self.assertIn('concepts', tables)
                self.assertNotIn('legacy_classes', tables)
            db.close()

    def test_config_compatibility_prefers_explicit_new_fields(self):
        original = {'seed_entity': 'Einstein', 'max_entities': 20, 'ner_model': 'old', 'cr_model': 'new', 'ned_candidates': 7}
        self.assertEqual(normalize_config(original), {'seed_concept': 'Einstein', 'max_concepts': 20, 'cr_model': 'new', 'cd_candidates': 7})
        self.assertIn('seed_entity', original)

    def test_migrated_run_expands_classes_without_overwriting_old_predicates(self):
        from tests.test_disambig import FakeLLM, fake_embeddings, WORLD
        from app.disambig.runner import DisambigConfig, DisambiguatedKBCRunner, LLMClient
        from app.disambig.embeddings import EmbeddingClient
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            legacy_database(run_dir / 'knowledgeBase.db')
            config = DisambigConfig(llm_url='http://fake/v1', api_key='x', model='fake', topic='physics', seed_concept='Alice')
            fake = FakeLLM()
            with mock.patch('app.reasoning.capability_for', return_value=None), \
                 mock.patch.object(LLMClient, 'chat', lambda self, role, system, user, max_retries=5: fake(role, system, user)), \
                 mock.patch.object(EmbeddingClient, '_request', lambda self, texts, max_retries=6: fake_embeddings(texts)), \
                 mock.patch.dict(WORLD, {('physicist', None): [('subClassOf', 'scientist')]}):
                runner = DisambiguatedKBCRunner(config, run_dir, lambda event: None, threading.Event(), threading.Event())
                runner.run()
            with sqlite3.connect(run_dir / 'knowledgeBase.db') as db:
                self.assertEqual(db.execute("SELECT status FROM concepts WHERE id = 'C0'").fetchone()[0], 'explored')
                self.assertEqual(db.execute("SELECT label FROM predicates WHERE id = 'P1'").fetchone()[0], 'field')
                subclass_id = db.execute("SELECT id FROM predicates WHERE label = 'subClassOf'").fetchone()[0]
                self.assertNotEqual(subclass_id, 'P1')
                self.assertEqual(db.execute("SELECT predicate_id FROM triples WHERE predicate_label = 'subClassOf'").fetchone()[0], subclass_id)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished'").fetchone()[0], 0)
            db.close()
            self.assertIn('E0.html', (run_dir / 'html' / 'index.html').read_text(encoding='utf-8'))
