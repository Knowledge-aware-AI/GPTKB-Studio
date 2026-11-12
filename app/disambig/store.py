"""SQLite schema of a disambiguated GPTKB Studio run (knowledgeBase.db).

Concepts (individuals and general concepts) share canonical identifiers,
aliases, descriptions and exploration status. Relations have their own IDs;
objects are concepts or literals and triples move through the pipeline by status.
The same file is the run's working state (a run resumes from it) and its
downloadable export, so it also carries GPTKB Studio's `run` metadata table and
a log of every disambiguation decision for the HTML report.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Object (triple object) statuses, in pipeline order -- GPTKB 2.0's
# ObjectStatusType minus the "on..." statuses only its batch mode uses.
OBJ_GENERATED = "generated"
OBJ_CR_FINISHED = "cr_finished"
OBJ_CD1_FINISHED = "cd_finished1"
OBJ_DG_FINISHED = "dg_finished"
OBJ_FINISHED = "finished"

TYPE_UNRECOGNIZED = "unrecognized"
TYPE_CONCEPT = "concept"
TYPE_LITERAL = "literal"

PRED_GENERATED = "generated"
PRED_PD_FINISHED = "pd_finished"
PRED_FINISHED = "finished"

CONCEPT_UNEXPLORED = "unexplored"
CONCEPT_EXPLORED = "explored"
CONCEPT_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS run (
    run_id TEXT, pipeline TEXT, topic TEXT, seed_concept TEXT,
    model TEXT, cr_model TEXT, cd_model TEXT, embedding_model TEXT,
    llm_url TEXT, language TEXT, schema_mode TEXT, schema_text TEXT, prompt_used TEXT,
    facts_multiplier REAL, nthreads INTEGER, max_requests_per_minute INTEGER,
    max_concepts INTEGER, max_bfs_depth INTEGER, max_runtime_seconds INTEGER, max_cost REAL,
    started_at TEXT, ended_at TEXT, status TEXT, cost_usd REAL,
    reasoning_effort TEXT, reasoning_enabled INTEGER
);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unexplored',
    depth INTEGER NOT NULL DEFAULT 0,
    first_appeared TEXT,
    num_triples INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    UNIQUE (label, description)
);
CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts (status, seq);
CREATE INDEX IF NOT EXISTS idx_concepts_label ON concepts (label);

CREATE TABLE IF NOT EXISTS concept_aliases (
    concept_id TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (concept_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_concept_aliases_alias ON concept_aliases (alias);

CREATE TABLE IF NOT EXISTS predicates (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, label TEXT NOT NULL, description TEXT NOT NULL,
    first_appeared TEXT, UNIQUE (label, description)
);
CREATE TABLE IF NOT EXISTS predicate_aliases (
    predicate_id TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (predicate_id, alias)
);

CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    subject_label TEXT NOT NULL,
    predicate_label TEXT NOT NULL,
    predicate_id TEXT,
    predicate_status TEXT NOT NULL DEFAULT 'generated',
    object_label TEXT NOT NULL,
    object_id TEXT,
    object_type TEXT NOT NULL DEFAULT 'unrecognized',
    object_status TEXT NOT NULL DEFAULT 'generated',
    object_description TEXT,
    -- How the object got (or didn't get) its id: cache, cd1, cd2, new,
    -- unsure (CD1 option G) or failed (gave up after repeated errors).
    resolution TEXT,
    attempts INTEGER NOT NULL DEFAULT 0
);
-- GPTKB 2.0's triple_unique index: a resolved triple that would duplicate
-- an existing one is deleted instead (NULL ids never collide in SQLite).
CREATE UNIQUE INDEX IF NOT EXISTS triple_unique ON triples
    (subject_id, subject_label, predicate_id, predicate_label, object_id, object_label);
CREATE INDEX IF NOT EXISTS idx_triples_seq ON triples (seq);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples (subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples (object_id);
CREATE INDEX IF NOT EXISTS idx_triples_object_stage ON triples (object_type, object_status, seq);
CREATE INDEX IF NOT EXISTS idx_triples_predicate_stage ON triples (predicate_status, seq);
CREATE INDEX IF NOT EXISTS idx_triples_pred_obj ON triples (predicate_label, object_label, object_id);
CREATE INDEX IF NOT EXISTS idx_triples_object_label ON triples (object_label);

-- The candidates offered for each disambiguation decision (CD1, CD2, PD,
-- CD), in option order -- the model answers with an option letter.
CREATE TABLE IF NOT EXISTS candidates (
    triple_id TEXT NOT NULL, stage TEXT NOT NULL, position INTEGER NOT NULL, ref_id TEXT NOT NULL,
    PRIMARY KEY (triple_id, stage, position)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triple_id TEXT, stage TEXT, target TEXT, context TEXT,
    candidates TEXT, answer TEXT, outcome TEXT, chosen_id TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_chosen ON decisions (chosen_id);
CREATE INDEX IF NOT EXISTS idx_decisions_triple ON decisions (triple_id);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT, stage TEXT, error TEXT, detail TEXT, time TEXT
);

CREATE TABLE IF NOT EXISTS text_embeddings (text TEXT PRIMARY KEY, vec BLOB NOT NULL);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    try:
        _migrate_legacy(connection)
        connection.executescript(SCHEMA)
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


def _migrate_legacy(db: sqlite3.Connection) -> None:
    """Unify an original GPTKB Studio database transactionally on resume.

    Preserve existing IDs and decision traces. Former class nodes become
    expandable concepts; former literals are recognized again because the old
    recognizer rejected general concepts as literals.
    """
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "entities" not in tables:
        return
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("ALTER TABLE concepts RENAME TO legacy_classes")
        db.execute("ALTER TABLE concept_aliases RENAME TO legacy_class_aliases")
        db.execute("ALTER TABLE entities RENAME TO concepts")
        db.execute("ALTER TABLE entity_aliases RENAME TO concept_aliases")
        db.execute("ALTER TABLE concept_aliases RENAME COLUMN entity_id TO concept_id")
        for old, new in (("seed_entity", "seed_concept"), ("ner_model", "cr_model"),
                         ("ned_model", "cd_model"), ("max_entities", "max_concepts")):
            db.execute(f"ALTER TABLE run RENAME COLUMN {old} TO {new}")
        next_seq = db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM concepts").fetchone()[0]
        for class_id, label, description, first in db.execute(
            "SELECT id, label, description, first_appeared FROM legacy_classes ORDER BY seq"
        ).fetchall():
            existing = db.execute("SELECT id FROM concepts WHERE label = ? AND description = ?",
                                  (label, description)).fetchone()
            concept_id = existing[0] if existing else class_id
            if not existing:
                depth = db.execute(
                    "SELECT MIN(c.depth) + 1 FROM instance_triples it JOIN concepts c ON c.id = it.entity_id "
                    "WHERE it.concept_id = ?", (class_id,)
                ).fetchone()[0]
                db.execute("INSERT INTO concepts (id, seq, label, description, depth, first_appeared) "
                           "VALUES (?, ?, ?, ?, ?, ?)",
                           (concept_id, next_seq, label, description, depth or 1, first))
                next_seq += 1
            db.execute("INSERT OR IGNORE INTO concept_aliases SELECT ?, alias FROM legacy_class_aliases WHERE concept_id = ?",
                       (concept_id, class_id))
            db.execute("UPDATE triples SET object_id = ?, object_type = 'concept', object_status = 'finished', "
                       "resolution = 'migrated' WHERE id IN (SELECT triple_id FROM instance_triples WHERE concept_id = ?)",
                       (concept_id, class_id))
            if concept_id != class_id:
                db.execute("UPDATE candidates SET ref_id = ? WHERE ref_id = ?", (concept_id, class_id))
                db.execute("UPDATE decisions SET chosen_id = ? WHERE chosen_id = ?", (concept_id, class_id))
        db.execute("UPDATE triples SET object_type = 'concept', object_status = 'cr_finished', attempts = 0 "
                   "WHERE object_type = 'class'")
        db.execute("UPDATE triples SET object_type = 'unrecognized', object_status = 'generated', attempts = 0 "
                   "WHERE object_type = 'literal' AND COALESCE(resolution, '') != 'failed'")
        db.execute("UPDATE triples SET object_type = 'concept' WHERE object_type = 'ne'")
        for old, new in (("ner_finished", "cr_finished"), ("ned_finished1", "cd_finished1")):
            db.execute("UPDATE triples SET object_status = ? WHERE object_status = ?", (new, old))
        # Historical class decisions remain distinguishable from common CD.
        for table in ("decisions", "candidates"):
            for old, new in (("cdg", "legacy_class_description"), ("cd", "legacy_class_disambiguation"),
                             ("ner", "cr"), ("ned1", "cd1"), ("ned2", "cd2"), ("nedg", "cdg")):
                db.execute(f"UPDATE {table} SET stage = ? WHERE stage = ?", (new, old))
        for old, new in (("ned1", "cd1"), ("ned2", "cd2")):
            db.execute("UPDATE triples SET resolution = ? WHERE resolution = ?", (new, old))
        db.execute("DROP TABLE instance_triples")
        db.execute("DROP TABLE legacy_class_aliases")
        db.execute("DROP TABLE legacy_classes")
        # C IDs from the former class table and E IDs from the former individual
        # table remain valid, while all newly allocated IDs use C.
        largest_id = max((int(r[0][1:]) for r in db.execute("SELECT id FROM concepts")
                          if r[0].startswith("C") and r[0][1:].isdigit()), default=-1)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('next_concept', ?)", (str(max(next_seq, largest_id + 1)),))
        db.execute("DELETE FROM meta WHERE key = 'next_entity'")
        seed = db.execute("SELECT id FROM concepts ORDER BY depth, seq LIMIT 1").fetchone()
        db.execute("INSERT OR REPLACE INTO meta VALUES ('seed_id', ?)", (seed[0] if seed else "C0",))
        # P1 may already name a different relation in an old run. Keep every
        # predicate identity and find typing predicates by label instead.
        db.commit()
    except Exception:
        db.rollback()
        raise


def get_meta(connection: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))


def progress_counts(db_path: Path) -> dict[str, Any] | None:
    """Summary counts for the History table, read-only and without touching
    the schema -- None if the run has no database yet."""
    if not db_path.exists():
        return None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        def scalar(sql: str) -> int:
            return int(connection.execute(sql).fetchone()[0] or 0)

        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "entities" in tables:
            # History stays read-only. Include work that the resume migration
            # will introduce, so completed old runs can be extended to concepts.
            return {
                "processed": scalar("SELECT COUNT(*) FROM entities WHERE status != 'unexplored'"),
                "queued": scalar("SELECT COUNT(*) FROM entities WHERE status = 'unexplored'")
                          + scalar("SELECT COUNT(*) FROM concepts"),
                "concepts": scalar("SELECT COUNT(*) FROM entities") + scalar("SELECT COUNT(*) FROM concepts"),
                "triples": scalar("SELECT COUNT(*) FROM triples"),
                "literals": scalar("SELECT COUNT(*) FROM triples WHERE object_type = 'literal'"),
                "failed_subjects": scalar("SELECT COUNT(*) FROM failures"),
                "pending": scalar("SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished' "
                                  "OR (object_type = 'literal' AND COALESCE(resolution, '') != 'failed')")
                           + scalar("SELECT COUNT(*) FROM instance_triples WHERE status != 'finished'"),
                "cost_usd": (connection.execute("SELECT cost_usd FROM run").fetchone() or (None,))[0],
            }
        return {
            "processed": scalar("SELECT COUNT(*) FROM concepts WHERE status != 'unexplored'"),
            "queued": scalar("SELECT COUNT(*) FROM concepts WHERE status = 'unexplored'"),
            "concepts": scalar("SELECT COUNT(*) FROM concepts"),
            "triples": scalar("SELECT COUNT(*) FROM triples"),
            "literals": scalar("SELECT COUNT(*) FROM triples WHERE object_type = 'literal'"),
            "failed_subjects": scalar("SELECT COUNT(*) FROM failures"),
            "pending": scalar(
                "SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished'"
            ),
            "cost_usd": (connection.execute("SELECT cost_usd FROM run").fetchone() or (None,))[0],
        }
    except sqlite3.Error:
        return None
    finally:
        connection.close()
