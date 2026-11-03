"""Keeps API keys out of everything a run shows or saves.

A run's log, status events and failure records are public (log panel, Run
Explorer, reports). Providers usually mask keys in their error messages, but
an endpoint may echo a key in full, so every text a run emits or records goes
through a redactor that replaces the run's keys with "***".
"""

from __future__ import annotations

from typing import Any, Callable

MASK = "***"
# Shorter values are not real keys, and replacing them could garble text.
MIN_SECRET_LENGTH = 8


def redactor(*secrets: str | None) -> Callable[[Any], Any]:
    """A function that replaces the given secrets in a string, or in the
    strings inside a dict or list, with MASK."""
    keys = sorted(
        {secret.strip() for secret in secrets if secret and len(secret.strip()) >= MIN_SECRET_LENGTH},
        key=len, reverse=True,
    )

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for key in keys:
                value = value.replace(key, MASK)
            return value
        if isinstance(value, dict):
            return {name: redact(item) for name, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact
