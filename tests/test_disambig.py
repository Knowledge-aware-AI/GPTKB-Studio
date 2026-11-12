"""Offline tests of the GPTKB 2.0 disambiguation runner: a scripted fake LLM
and hashed character-trigram "embeddings" stand in for the real endpoints.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from app.disambig import prompts as P
from app.disambig.embeddings import EMBEDDING_DIM, EmbeddingClient, VectorIndex, cluster_labels
from app.disambig.runner import DisambigConfig, DisambiguatedKBCRunner, LLMClient
from app.llm_errors import ContextWindowExceeded
from app.runner import PROMPT_LANGUAGES, build_elicitation_prompt_body

# No real model endpoint is ever called: reasoning-capability probing
# (app/reasoning.py, tested in test_reasoning.py) is switched off here.
_no_probing = mock.patch("app.reasoning.capability_for", lambda llm_url, api_key, model: None)


def setUpModule():
    _no_probing.start()


def tearDownModule():
    _no_probing.stop()

# What the fake model "knows": subject label -> facts. Two different Munichs:
# the film (object of notableWork) and the city (object of hasCity/livesIn).
WORLD = {
    ("actor", None): [("subClassOf", "performer"), ("practices", "acting")],
    ("acting", None): [("subClassOf", "performing art"), ("practitioner", "actor"), ("website", "https://example.com/acting")],
    ("performer", None): [("subClassOf", "person")],
    ("Mathieu Amalric", None): [
        ("instanceOf", "actor"), ("notableWork", "Munich"), ("placeOfBirth", "Neuilly-sur-Seine"), ("birthYear", "1965"), ("field", "acting"), ("profession", "actor"),
    ],
    ("Munich", "film"): [("instanceOf", "film"), ("director", "Steven Spielberg"), ("releaseYear", "2005")],
    ("Munich", "city"): [("instanceOf", "city"), ("country", "Federal Republic of Germany"), ("population", "1.5 million")],
    ("Steven Spielberg", None): [
        ("instanceOf", "human"), ("notableWork", "Munich"), ("directed", "Jaws"), ("born in", "Cincinnati"),
    ],
    ("Neuilly-sur-Seine", None): [("instanceOf", "commune"), ("country", "France"), ("nearCity", "Paris")],
    ("France", None): [("instanceOf", "country"), ("capital", "Paris"), ("neighbour", "Germany")],
    ("Germany", None): [("instanceOf", "country"), ("hasCity", "Munich"), ("capital", "Berlin")],
    ("Paris", None): [("instanceOf", "city"), ("country", "France")],
}
# Synonyms the fake CD accepts as the same concept.
SAME = {frozenset({"Germany", "Federal Republic of Germany"})}
CITY_PREDICATES = {"hasCity", "capital", "nearCity", "placeOfBirth", "born in"}
FILM_PREDICATES = {"notableWork", "directed"}


def kind_for(predicate: str) -> str:
    if predicate in FILM_PREDICATES:
        return "film"
    if predicate in CITY_PREDICATES:
        return "city"
    if predicate in {"country", "neighbour"}:
        return "country"
    return "thing"


def same_concept(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) in SAME


def options(user: str) -> list[tuple[str, str]]:
    return re.findall(r'^[A-J]\. \w+: "([^"]*)" \nDescription: (.*)$', user, flags=re.M)


def none_letter(user: str) -> str:
    return re.search(r"^([A-L])\. None of above\.", user, flags=re.M).group(1)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()

    def count(self, kind: str) -> None:
        with self.lock:
            self.calls[kind] = self.calls.get(kind, 0) + 1

    def __call__(self, role: str, system: str, user: str, max_retries: int = 5) -> str:
        if not system:  # elicitation
            self.count("elicitation")
            label = re.search(r'Subject: "(.*)"', user).group(1)
            description = user.rsplit("Description of subject: ", 1)[1]
            kind = "film" if "film" in description else "city" if "city" in description else None
            facts = WORLD.get((label, kind)) or WORLD.get((label, None)) or []
            return json.dumps([{"s": label, "p": p, "o": o} for p, o in facts])
        if system == P.CR_SYSTEM:
            self.count("cr")
            phrase = re.search(r"Phrase: (.*) \| Statement", user).group(1)
            is_concept = bool(phrase) and not phrase[:1].isdigit() and not phrase.startswith("https://")
            return json.dumps({"phrase": phrase, "is_concept": is_concept})
        if system == P.CD1_SYSTEM:
            self.count("cd1")
            target = re.search(r"Target concept: (.*)", user).group(1)
            predicate = re.search(r"Context triple: \[.*?, (.*?), ", user).group(1)
            for index, (label, description) in enumerate(options(user)):
                if same_concept(label, target) and (kind_for(predicate) in description or kind_for(predicate) == "thing"):
                    return chr(65 + index)
            return none_letter(user)
        if system.startswith(P.CDG_SYSTEM):
            self.count("cdg")
            concept = re.search(r"Concept: (.*)", user).group(1)
            predicate = re.search(r"Triple: \[.*?, (.*?), ", user).group(1)
            return f"Description: {concept} is a {kind_for(predicate)}."
        if system == P.CD2_SYSTEM:
            self.count("cd2")
            target = re.search(r"Target concept: (.*)", user).group(1)
            target_description = re.search(r"Target concept description: (.*)", user).group(1)
            for index, (label, description) in enumerate(options(user)):
                if same_concept(label, target) and description.split(" is a ")[-1] == target_description.split(" is a ")[-1]:
                    return chr(65 + index)
            return none_letter(user)
        if system == P.PD_SYSTEM:
            self.count("pd")
            target = re.search(r"Target (?:predicate|class): (.*)", user).group(1)
            for index, (label, _) in enumerate(options(user)):
                if label.lower() == target.lower():
                    return chr(65 + index)
            return "F"
        if system.startswith(P.PDG_SYSTEM):
            self.count("pdg")
            return f"Indicates the relation {re.search(r'Predicate: (.*)', user).group(1)}."
        # Seed description (adapted CDG prompt).
        self.count("seed")
        return "Description: Mathieu Amalric is a French actor and film director."


def fake_embeddings(texts: list[str]) -> list[np.ndarray]:
    out = []
    for text in texts:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        padded = f"  {text.lower()}  "
        for i in range(len(padded) - 2):
            bucket = int(hashlib.md5(padded[i : i + 3].encode()).hexdigest()[:8], 16) % EMBEDDING_DIM
            vec[bucket] += 1.0
        out.append(vec / (np.linalg.norm(vec) or 1.0))
    return out


class RunnerTest(unittest.TestCase):
    def run_pipeline(self, **overrides):
        tmp = Path(tempfile.mkdtemp())
        run_dir = tmp / "run"
        run_dir.mkdir()
        (run_dir / "runConfig.json").write_text("{}")
        config = DisambigConfig(
            llm_url="http://fake/v1", api_key="x", model="fake", topic="French cinema",
            seed_concept=overrides.pop("seed_concept", "Mathieu Amalric"), nthreads=4, **overrides,
        )
        events = []
        fake = FakeLLM()
        originals = (LLMClient.chat, EmbeddingClient._request)
        LLMClient.chat = lambda self, role, system, user, max_retries=5: fake(role, system, user)
        EmbeddingClient._request = lambda self, texts, max_retries=6: fake_embeddings(texts)
        try:
            runner = DisambiguatedKBCRunner(config, run_dir, events.append, threading.Event(), threading.Event())
            runner.run()
        finally:
            LLMClient.chat, EmbeddingClient._request = originals
        return run_dir, events, fake

    def test_homonyms_and_synonyms(self):
        run_dir, events, fake = self.run_pipeline()
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        statuses = [e["status"] for e in events if e["type"] == "status"]
        self.assertEqual(statuses[-1], "finished")

        munichs = db.execute("SELECT id, description FROM concepts WHERE label = 'Munich'").fetchall()
        self.assertEqual(len(munichs), 2, munichs)
        self.assertEqual({"film" in d for _, d in munichs}, {True, False})

        film_id = next(i for i, d in munichs if "film" in d)
        city_id = next(i for i, d in munichs if "city" in d)
        work_objects = {r[0] for r in db.execute("SELECT object_id FROM triples WHERE predicate_label = 'notableWork'")}
        self.assertEqual(work_objects, {film_id})  # Amalric's and Spielberg's notableWork are the same film
        self.assertEqual(
            db.execute("SELECT object_id FROM triples WHERE predicate_label = 'hasCity'").fetchone()[0], city_id
        )

        # Germany and "Federal Republic of Germany" are one concept with an alias.
        germanies = db.execute("SELECT id FROM concepts WHERE label LIKE '%Germany%'").fetchall()
        self.assertEqual(len(germanies), 1, germanies)
        aliases = {r[0] for r in db.execute("SELECT alias FROM concept_aliases WHERE concept_id = ?", germanies[0])}
        self.assertEqual(aliases, {"Federal Republic of Germany"})

        # One Paris, referenced from three subjects.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'Paris'").fetchone()[0], 1)

        # Everything is resolved; literals stay literals; classes resolved.
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished' OR predicate_status != 'finished'").fetchone()[0],
            0,
        )
        self.assertEqual(db.execute("SELECT object_type FROM triples WHERE object_label = '1965'").fetchone()[0], "literal")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'city'").fetchone()[0], 1)
        # instanceOf is the seed predicate P0; there is no seed class, the
        # first class is created without asking the LLM about zero candidates.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'human'").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT label FROM concepts WHERE id = 'C0'").fetchone()[0], "Mathieu Amalric")
        self.assertEqual(db.execute("SELECT predicate_id FROM triples WHERE predicate_label = 'instanceOf' LIMIT 1").fetchone()[0], "P0")
        # Every concept reachable in the toy world was explored, and depth is tracked.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE status = 'unexplored'").fetchone()[0], 0)
        self.assertEqual(db.execute("SELECT depth FROM concepts WHERE id = ?", (film_id,)).fetchone()[0], 1)

        html = run_dir / "html"
        self.assertTrue((html / "index.html").exists())
        self.assertTrue((html / f"{film_id}.html").exists())
        self.assertIn("Munich", (html / "index.html").read_text())
        self.assertTrue((run_dir / "triples.jsonl").exists())
        self.assertEqual(db.execute("SELECT status FROM run").fetchone()[0], "finished")

    def test_general_concepts_share_identity_and_expand(self):
        run_dir, _, _ = self.run_pipeline()
        with sqlite3.connect(run_dir / "knowledgeBase.db") as db:
            actor = db.execute("SELECT id, status, depth FROM concepts WHERE label = 'actor'").fetchall()
            self.assertEqual(len(actor), 1)
            actor_id, status, depth = actor[0]
            self.assertEqual(status, "explored")
            self.assertEqual(depth, 1)
            actor_refs = db.execute("SELECT DISTINCT object_id FROM triples WHERE object_label = 'actor'").fetchall()
            self.assertEqual(actor_refs, [(actor_id,)])
            self.assertEqual(db.execute("SELECT status FROM concepts WHERE label = 'acting'").fetchone()[0], "explored")
            self.assertEqual(db.execute("SELECT status FROM concepts WHERE label = 'performer'").fetchone()[0], "explored")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM triples WHERE object_type NOT IN ('concept', 'literal')").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT object_type FROM triples WHERE object_label LIKE 'https://%'").fetchone()[0], "literal")
            self.assertEqual(db.execute("SELECT predicate_id FROM triples WHERE predicate_label = 'subClassOf' LIMIT 1").fetchone()[0], "P1")
        exported = [json.loads(line) for line in (run_dir / "triples.jsonl").read_text().splitlines()]
        self.assertTrue(any(t["p"] == "instanceOf" and t["o_id"] == actor_id and t["o_type"] == "concept" for t in exported))
        self.assertTrue((run_dir / "html" / f"{actor_id}.html").exists())

    def test_general_concept_synonyms_merge_but_subclasses_do_not(self):
        world = {
            ("Mathieu Amalric", None): [("studies", "machine learning"), ("knows", "Bob")],
            ("Bob", None): [("uses", "ML"), ("studies", "deep learning")],
            ("deep learning", None): [("subClassOf", "machine learning")],
        }
        with mock.patch.dict(WORLD, world, clear=True), mock.patch.dict(
            globals(), SAME={frozenset({"machine learning", "ML"})}
        ):
            run_dir, _, _ = self.run_pipeline()
        with sqlite3.connect(run_dir / "knowledgeBase.db") as db:
            ids = {row[0] for row in db.execute("SELECT object_id FROM triples WHERE object_label IN ('machine learning', 'ML')")}
            self.assertEqual(len(ids), 1)
            deep = db.execute("SELECT id FROM concepts WHERE label = 'deep learning'").fetchone()[0]
            self.assertNotIn(deep, ids)
            self.assertEqual(db.execute("SELECT object_id FROM triples WHERE subject_id = ? AND predicate_label = 'subClassOf'", (deep,)).fetchone()[0], next(iter(ids)))

    def test_general_concept_can_be_the_seed(self):
        with mock.patch.dict(WORLD, {('gravity', None): [('subClassOf', 'fundamental interaction')]}):
            run_dir, _, _ = self.run_pipeline(seed_concept='gravity')
        with sqlite3.connect(run_dir / 'knowledgeBase.db') as db:
            self.assertEqual(db.execute("SELECT label, status FROM concepts WHERE id = 'C0'").fetchone(), ('gravity', 'explored'))
            self.assertEqual(db.execute("SELECT status FROM concepts WHERE label = 'fundamental interaction'").fetchone()[0], 'explored')

    def test_similarity_floor_blocks_dissimilar_merges(self):
        # "Federal Republic of Germany" is not similar enough to "Germany" to
        # be offered as a candidate at a floor of 0.99, so it becomes its own concept.
        run_dir, events, _ = self.run_pipeline(cd_similarity_floor=0.99)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        labels = {r[0] for r in db.execute("SELECT label FROM concepts")}
        self.assertIn("Federal Republic of Germany", labels)
        self.assertIn("Germany", labels)
        # Identical labels are still merged: one Paris, one France.
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'Paris'").fetchone()[0], 1)
        self.assertEqual([e["status"] for e in events if e["type"] == "status"][-1], "finished")

    def test_more_than_five_candidates(self):
        run_dir, _, _ = self.run_pipeline(cd_candidates=7)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label = 'Munich'").fetchone()[0], 2)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM concepts WHERE label LIKE '%Germany%'").fetchone()[0], 1)
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished'").fetchone()[0], 0
        )

    def test_bfs_depth_limit(self):
        run_dir, _, _ = self.run_pipeline(max_bfs_depth=1)
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        explored_depths = {r[0] for r in db.execute("SELECT depth FROM concepts WHERE status = 'explored'")}
        self.assertEqual(explored_depths, {0, 1})
        self.assertGreater(db.execute("SELECT COUNT(*) FROM concepts WHERE depth = 2").fetchone()[0], 0)
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM triples WHERE object_status != 'finished'").fetchone()[0], 0
        )

    def run_with_elicitation(self, answer):
        """Runs the pipeline with every elicitation request answered by `answer(user)`."""
        fake = FakeLLM()
        calls = []

        def chat(self, role, system, user, max_retries=5):
            if role == "elicitation":
                calls.append(user)
                return answer(user, fake)
            return fake(role, system, user)

        tmp = Path(tempfile.mkdtemp())
        run_dir = tmp / "run"
        run_dir.mkdir()
        (run_dir / "runConfig.json").write_text("{}")
        config = DisambigConfig(llm_url="http://fake/v1", api_key="x", model="fake", topic="French cinema",
                                seed_concept="Mathieu Amalric", max_concepts=3)
        events = []
        originals = (LLMClient.chat, EmbeddingClient._request)
        LLMClient.chat = chat
        EmbeddingClient._request = lambda self, texts, max_retries=6: fake_embeddings(texts)
        try:
            DisambiguatedKBCRunner(config, run_dir, events.append, threading.Event(), threading.Event()).run()
        finally:
            LLMClient.chat, EmbeddingClient._request = originals
        return sqlite3.connect(run_dir / "knowledgeBase.db"), events, calls

    def test_context_window_gives_up_at_once(self):
        def too_long(user, fake):
            raise ContextWindowExceeded("the prompt does not fit the context window of fake")
        db, events, calls = self.run_with_elicitation(too_long)
        self.assertEqual(len(calls), 1)  # neither the request nor the concept is tried again
        self.assertEqual(db.execute("SELECT stage FROM failures").fetchall(), [("context_window",)])
        logs = [e["message"] for e in events if e["type"] == "log"]
        self.assertTrue(any("larger context window" in m for m in logs))

    def test_cut_off_answer_that_parses_is_kept(self):
        def cut_off(user, fake):
            raise ContextWindowExceeded("cut off", text=fake("elicitation", "", user) + "\nAnd also")
        db, _, calls = self.run_with_elicitation(cut_off)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM failures").fetchone()[0], 0)
        self.assertGreater(db.execute("SELECT COUNT(*) FROM triples").fetchone()[0], 0)

    def test_resume_is_idempotent(self):
        run_dir, _, _ = self.run_pipeline()
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        before = db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        db.close()
        config = DisambigConfig(llm_url="http://fake/v1", api_key="x", model="fake", topic="French cinema",
                                seed_concept="Mathieu Amalric")
        fake = FakeLLM()
        originals = (LLMClient.chat, EmbeddingClient._request)
        LLMClient.chat = lambda self, role, system, user, max_retries=5: fake(role, system, user)
        EmbeddingClient._request = lambda self, texts, max_retries=6: fake_embeddings(texts)
        try:
            DisambiguatedKBCRunner(config, run_dir, lambda e: None, threading.Event(), threading.Event()).run()
        finally:
            LLMClient.chat, EmbeddingClient._request = originals
        self.assertEqual(fake.calls, {})  # nothing left to do, no LLM calls
        db = sqlite3.connect(run_dir / "knowledgeBase.db")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM triples").fetchone()[0], before)


class UnitTest(unittest.TestCase):
    def test_parse_choice(self):
        self.assertEqual(P.parse_choice("B"), 2)
        self.assertEqual(P.parse_choice("B. Concept: Munich"), 2)
        self.assertIsNone(P.parse_choice("F"))
        self.assertIsNone(P.parse_choice("A or B"))  # ambiguous -> treated as none
        self.assertEqual(P.parse_choice("G", unsure=True), "unsure")
        self.assertEqual(P.parse_choice("G"), 7)  # out of range -> caller retries
        self.assertIsNone(P.parse_choice("G", "A-F"))
        self.assertEqual(P.parse_choice("<think>maybe A? or C</think>\nC"), 3)

    def test_parse_elicitation_formats(self):
        mini = P.parse_elicitation('```json\n[{"s": "X", "p": "a", "o": "Y"}]\n```')
        gptkb = P.parse_elicitation('{"facts": [{"subject": "X", "predicate": "a", "object": "Y"}]}')
        self.assertEqual(mini, gptkb)
        self.assertEqual(P.parse_elicitation("Sure! [] "), [])

    def test_parse_cr_and_description(self):
        self.assertTrue(P.parse_cr('{"phrase": "Paris", "is_concept": true}'))
        self.assertFalse(P.parse_cr('{"phrase": "1965", "is_concept": "false"}'))
        self.assertEqual(P.parse_description("Description: A city."), "A city.")
        for value in ('null', '1', '[]', '{}'):
            with self.assertRaises(P.ParsingError):
                P.parse_cr('{"is_concept": ' + value + '}')

    def test_all_prompt_languages_allow_concepts_and_both_typing_relations(self):
        for language, rules in PROMPT_LANGUAGES.items():
            with self.subTest(language=language):
                prompt = build_elicitation_prompt_body(language, 'physics', 1, 'open', '')
                self.assertIn(rules['rule_concept'], prompt)
                self.assertIn('instanceOf', prompt)
                self.assertIn('subClassOf', prompt)

    def test_language_adaptations(self):
        self.assertIn("istanzaDi", P.instance_of_labels("it"))
        self.assertIn("general classes", P.cr_system_prompt("en"))
        self.assertNotIn("english", P.cr_system_prompt("de"))
        self.assertNotIn("Latin script", P.cr_system_prompt("de"))

    def test_clustering_matches_threshold(self):
        vectors = np.stack(fake_embeddings(["Munich", "Munich ", "München", "Paris"]))
        labels = cluster_labels(vectors, 0.15)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[3])

    def test_parse_choice_verbose_fallback(self):
        # GPTKB 2.0's rule sees "A" (the article) and "D": ambiguous; the stated answer decides.
        self.assertEqual(P.parse_choice("The target is a character in the show, so the answer is D."), 4)
        self.assertEqual(P.parse_choice("F. \n\nThe target is a character, not an actor."), None)
        self.assertIsNone(P.parse_choice("A city is not D."))
        self.assertEqual(P.parse_choice("Answer: C"), 3)

    def test_vector_index_dedupes_aliases(self):
        index = VectorIndex()
        index.add(["E0", "E1", "E0"], np.stack(fake_embeddings(["Germany", "Berlin", "Federal Republic of Germany"])))
        self.assertEqual(index.search(np.stack(fake_embeddings(["Germany"])), 2)[0], ["E0", "E1"])
        self.assertEqual(index.search(np.stack(fake_embeddings(["Federal Republic of Germany"])), 1)[0], ["E0"])

    def test_vector_index_similarity_floor(self):
        index = VectorIndex()
        index.add(["E0", "E1"], np.stack(fake_embeddings(["Paris", "Berlin"])))
        scored = index.search_scored(np.stack(fake_embeddings(["Paris"])), 2)[0]
        self.assertEqual(scored[0][0], "E0")
        self.assertAlmostEqual(scored[0][1], 1.0, places=5)
        self.assertEqual(index.search(np.stack(fake_embeddings(["Paris"])), 2, min_similarity=0.9), [["E0"]])

    def test_vector_index(self):
        index = VectorIndex()
        index.add(["E0", "E1", "E2"], np.stack(fake_embeddings(["Paris", "Berlin", "Parisian"])))
        self.assertEqual(index.search(np.stack(fake_embeddings(["Paris"])), 2)[0], ["E0", "E2"])
        self.assertEqual(len(index.search(np.stack(fake_embeddings(["x"])), 10)[0]), 3)


if __name__ == "__main__":
    unittest.main()


class EndpointRoutingTest(unittest.TestCase):
    """Each role's request goes to its own endpoint, with that endpoint's key."""

    def test_roles_use_their_endpoints(self):
        from types import SimpleNamespace

        config = DisambigConfig(
            llm_url="https://api.openai.com/v1", api_key="sk-user", model="gpt-5.4-mini", topic="t", seed_concept="s",
            cr_model="google/gemma-4-26B-A4B-it", cr_llm_url="https://llm.scads.ai/v1", cr_api_key="scads-shared",
            cd_model="Qwen/Qwen3.8-27B", cd_llm_url="https://llm.scads.ai/v1", cd_api_key="scads-shared",
        )
        client = LLMClient(config, threading.Event(), lambda message, level: None)
        self.assertEqual(len(client.endpoints), 2)
        calls = []

        def fake_endpoint(endpoint):
            def create(**request):
                calls.append((endpoint.url, endpoint.api_key, request["model"], "temperature" in request))
                response = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="A"))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11),
                )
                return SimpleNamespace(parse=lambda: response, headers={})
            completions = SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

        for endpoint in client.endpoints.values():
            endpoint.client = fake_endpoint(endpoint)
        for role in ("elicitation", "cr", "cd"):
            client.chat(role, "system", "user")
        self.assertEqual(calls[0][:3], ("https://api.openai.com/v1", "sk-user", "gpt-5.4-mini"))
        self.assertFalse(calls[0][3])  # no temperature for OpenAI's GPT-5 models
        self.assertEqual(calls[1][:3], ("https://llm.scads.ai/v1", "scads-shared", "google/gemma-4-26B-A4B-it"))
        self.assertEqual(calls[2][:3], ("https://llm.scads.ai/v1", "scads-shared", "Qwen/Qwen3.8-27B"))
        self.assertTrue(calls[1][3])
        self.assertEqual(client.model_label("cd"), "Qwen/Qwen3.8-27B (llm.scads.ai)")
        self.assertTrue(client.cost_tracked())  # the OpenAI part is priced, ScaDS is free
        self.assertTrue(client.has_pricing())
