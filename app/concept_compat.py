"""Read-only translation of pre-concept API/settings field names.

New requests, persisted settings and responses use concept terminology.
Legacy spellings are accepted only at the input boundary.
"""

from typing import Any


def normalize_config(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    aliases = {"seed_entity": "seed_concept", "max_entities": "max_concepts"}
    for old, new in (("ner", "cr"), ("ned", "cd")):
        for suffix in ("model", "llm_url", "api_key", "candidates", "similarity_floor"):
            aliases[f"{old}_{suffix}"] = f"{new}_{suffix}"
    for old, new in aliases.items():
        if old in result:
            result.setdefault(new, result.pop(old))
    cached = result.get("cached_db_counts")
    if isinstance(cached, dict) and "entities" in cached:
        # Recompute counts: former classes become queued concepts on resume.
        result.pop("cached_db_counts")
    return result
