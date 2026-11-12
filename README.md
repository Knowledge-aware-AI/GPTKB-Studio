# GPTKB Studio

GPTKB Studio is a web application for building small, topic-specific knowledge
bases from what a large language model knows. You choose a topic, a seed
concept and a model; GPTKB Studio asks the model for facts about the seed as
subject–predicate–object triples, recognizes the concepts among the
objects, decides whether each one is an concept it already has or a new one,
and then asks about the new ones, round after round, until nothing new is left
or a budget runs out.

Concepts and relations are disambiguated during construction with the
pipeline of [GPTKB 2.0](https://github.com/Knowledge-aware-AI/GPTKB2_disambiguation_on_the_fly),
so the result has canonical identifiers rather than strings, and every merge
decision can be inspected.

A hosted instance runs at <https://gptkbstudio.scads.ai/>.

## Features

- **Any model, any OpenAI-compatible endpoint:** presets for ScaDS.AI, OpenAI
  and OpenRouter, or a custom URL (e.g. a self-hosted vLLM server; on a local network only with `MINIGPTKB_ALLOW_PRIVATE_ENDPOINTS=1`). One model
  does all three tasks (fact elicitation, concept recognition,
  disambiguation) unless CR and CD are given models of their own under
  *Advanced*.
- **Topic-scoped, multilingual elicitation:** 11 pre-packaged prompt languages, an editable
  prompt, open, example-driven or fixed-ontology schema.
- **Budgets:** caps on concepts, BFS depth, cost (USD) and runtime. Concept and
  depth caps stop the exploration but let pending disambiguation finish.
- **Live view:** metrics, the current pipeline step and the log stream into
  the browser; runs can be paused, stopped and resumed, also after a server
  restart.
- **Controllable merging:** a slider trades fewer wrong merges for more
  duplicates by restricting which existing concepts are offered as matches.
- **Results:** a browsable HTML report per run (statistics, an interactive
  graph, one page per concept with its disambiguation trace) and a SQLite
  database with concepts, relations, aliases, triples and every
  disambiguation decision.
- **Shared, rate-limited endpoints:** requests are paced per model below the
  limits the endpoint reports.

## How a run works

Each run is a separate worker process. In every round it runs the GPTKB 2.0
stages on whatever items are ready for them:

```
CDG -> CD2 -> CD1 -> Elicitation -> PDG -> PD -> CR
```

- **Elicitation** asks for the facts about unexplored concepts.
- **CR** classifies each triple's object as a concept or a literal.
- **CD1** matches an object against the most similar existing concepts, using
  its triple as context; **CDG** writes a description for objects without a
  match, and **CD2** matches again using that description, or creates a new
  concept.
- **PD/PDG** disambiguate relations and describe new ones.

Concepts include both particular individuals (such as Albert Einstein) and
general concepts (such as gravity, physicist and spacetime). Objects have two
semantic types: `concept` and `literal`. All concepts share one `concepts`
table, alias collection, candidate index and expansion queue. `instanceOf`
and `subClassOf` are distinct ordinary relations whose targets are expandable
concepts. Every concept should have at least one triple using either relation;
all 11 default elicitation prompts request this. This is a prompt instruction,
not a guarantee that a model will supply a correct typing fact.

The API and run configuration use `seed_concept`, `max_concepts`, `cr_model`
(concept recognition), and `cd_model` (concept disambiguation), with matching
`cr_*` and `cd_*` endpoint and candidate settings. Older field names are
accepted at the API/settings boundary. New exports and reports use concept
terminology throughout.

On resuming an original disambiguated run, its database is migrated in one
transaction. Existing node IDs and decision traces are retained; former
class nodes enter the expansion queue. Former literals are recognized again
because the old recognizer treated general concepts as literals. This may
require additional model calls. Review any saved custom elicitation prompt
to remove a named-entity-only restriction before resuming it.

Candidates are retrieved by label embedding (Qwen3-Embedding-4B). The state of
a run lives in `runs/<run id>/knowledgeBase.db`, which is also its export and
what a resumed run continues from.

The original crawler, which identifies concepts by their names as in the
miniGPTKB method it grew out of, is
kept in `app/runner.py`; runs made with it stay viewable and resumable, and new
ones can be started through the API with `"pipeline": "classic"`.

## Running it

Requirements: Python 3.11 or newer.

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

> **Embeddings need a ScaDS.AI key.** Disambiguation retrieves its candidates
> with the embedding model Qwen3-Embedding-4B, which GPTKB Studio calls on the
> ScaDS.AI inference service with the key in `SCADS_API_KEY`, whatever endpoint
> the language models use. Without that key, runs with disambiguation fail. To
> run GPTKB Studio without one, point the embeddings to another OpenAI-compatible
> embeddings endpoint (e.g. Qwen3-Embedding-4B served locally with vLLM or
> Ollama): the endpoint and model are set in `app/disambig/runner.py`
> (`SCADS_EMBEDDING_URL`, `DEFAULT_EMBEDDING_MODEL`) and the key in
> `app/server.py`. With a different embedding model, the similarity thresholds
> of the concept-merging slider and of label clustering may need adjusting.

Environment variables:

| Variable | Purpose |
|---|---|
| `SCADS_API_KEY` | Key for the ScaDS.AI inference service. Used for label embeddings, and for model calls on ScaDS.AI when the user leaves the API key field empty. |
| `MINIGPTKB_RUNS_DIR` | Where run folders are kept (default: `runs/`). |
| `MINIGPTKB_SHARED_KEY_MINUTES` | How many minutes one run may use the shared key, in total over all its sessions; the server ends it after that (default: no limit). |
| `MINIGPTKB_MAX_SHARED_KEY_RUNS` | How many runs may use the shared key at the same time (default: no limit). |
| `MINIGPTKB_SITE_LABEL` | A label shown in red under the app name, e.g. `Sandbox` on an internal test instance (default: none). |
| `MINIGPTKB_ALLOW_PRIVATE_ENDPOINTS` | Set to `1` to allow model endpoints on private, loopback or link-local addresses (e.g. a model served with vLLM or Ollama on the same machine). By default such endpoints are refused, so that users of a hosted instance cannot make the server call its own network. |

API keys that users enter in the form are passed to the run in memory only:
they are never written to disk, and they are sent only to the endpoint they
were entered for.

`deploy/` contains an example systemd unit and environment file for a hosted
instance.

A folder can be made read-only, e.g. for showcase runs on a public instance,
by adding `"locked": true` to its entry in `<runs dir>/_folders.json` on the
server. Nobody can then rename or delete it, delete, move, resume or stop its
runs, or put other runs into it; its runs can still be explored, watched and
downloaded.

## Repository layout

| Path | Contents |
|---|---|
| `app/server.py` | FastAPI server: REST API, run processes, event stream, run list and folders, per-site limits, downloads |
| `app/disambig/` | The disambiguation pipeline: runner (`runner.py`), prompts and answer parsers (`prompts.py`, `prompts/`), SQLite schema (`store.py`), embeddings and candidate retrieval (`embeddings.py`), pacing below rate limits (`throttle.py`), HTML report (`report.py`, `assets/`) |
| `app/runner.py` | The original string-identity crawler, and shared prompt building and pricing |
| `app/endpoints.py` | Refuses model endpoints on private, loopback or link-local addresses, for every request including redirects |
| `app/reasoning.py` | Finds out which reasoning controls (on/off, effort levels) a model has, by probing it |
| `app/heartbeat.py` | Status lines in the log while a run waits for model answers, and warnings about slow requests |
| `app/llm_errors.py` | Recognizes prompts or answers that exceed a model's context window |
| `app/redact.py` | Removes API keys from log lines and failure records |
| `app/usage.py` | Per-run counts of model calls and tokens (`usage.json`) |
| `app/static/` | The browser application (plain JavaScript, no build step) and the legal pages |
| `app/vendor/` | vis-network, for the graph view |
| `deploy/` | Example systemd unit and environment file for a hosted instance |
| `tests/` | Unit tests: `python -m unittest discover -s tests` |

## Citation

TBA

## License

GPTKB Studio is released under the
[Creative Commons Attribution 4.0 International](LICENSE) license.

The prompts in `app/disambig/prompts/` are adapted from the GPTKB 2.0 repository
and remain under the Apache License 2.0 (`app/disambig/prompts/LICENSE-GPTKB2`,
source commit in `app/disambig/prompts/SOURCE`). `app/vendor/vis-network.min.js`
is dual-licensed under the Apache License 2.0 and MIT.
