"""Model request failures that asking again cannot fix.

When a prompt is longer than the model's context window, the endpoint rejects
the request; when prompt and answer together reach the window (or the
endpoint's output limit), the answer comes back cut off, which for elicitation
means unparsable JSON. With the same prompt both happen again on every try, so
the runs neither retry such a request nor the item it belongs to, and they say
what happened instead of reporting a generic request or parsing failure.
"""

from __future__ import annotations

from typing import Any

# Phrases endpoints use when a prompt does not fit: OpenAI ("maximum context
# length", code context_length_exceeded), vLLM and OpenRouter ("This model's /
# endpoint's maximum context length is ..."), LiteLLM (ContextWindowExceededError),
# and other proxies ("prompt is too long").
_CONTEXT_PHRASES = (
    "context length", "context_length", "context window", "contextwindow", "maximum context",
    "prompt is too long", "input is too long", "too many input tokens", "reduce the length",
)


class ContextWindowExceeded(Exception):
    """The prompt, or prompt plus answer, does not fit the model. `text` is
    the cut-off answer, if one came back."""

    def __init__(self, message: str, text: str | None = None) -> None:
        super().__init__(message)
        self.text = text


def context_window_error(exc: Exception, model: str) -> ContextWindowExceeded | None:
    """The ContextWindowExceeded for a failed request, if that is why it failed."""
    status = getattr(exc, "status_code", None)
    if status not in (None, 400, 413, 422):
        return None
    text = f"{type(exc).__name__} {exc}".lower()
    if not any(phrase in text for phrase in _CONTEXT_PHRASES):
        return None
    return ContextWindowExceeded(
        f"the prompt does not fit the context window of {model} (endpoint said: {str(exc)[:240]})"
    )


def cut_off_error(response: Any, model: str) -> ContextWindowExceeded | None:
    """The ContextWindowExceeded for an answer that stopped at the length
    limit (finish_reason "length"), or None for a complete answer."""
    choices = getattr(response, "choices", None) or []
    if not choices or getattr(choices[0], "finish_reason", None) != "length":
        return None
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    sizes = (f" ({prompt} prompt tokens + {completion} answer tokens)"
             if prompt is not None and completion is not None else "")
    return ContextWindowExceeded(
        f"the answer of {model} was cut off: it reached the model's context window or output limit{sizes}",
        getattr(getattr(choices[0], "message", None), "content", None) or "",
    )
