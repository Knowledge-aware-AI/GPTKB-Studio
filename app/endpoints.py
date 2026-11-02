"""Keeps model requests away from the server's own network.

Users can enter any endpoint URL, and the server then sends requests to it
(model list, reasoning probe, the run itself). Without a check, a URL such as
http://127.0.0.1:8000 or http://169.254.169.254 would make the server call
services that are only reachable from inside its network. So an endpoint is
refused when its host resolves to a private, loopback, link-local or otherwise
non-public address. The check runs when a URL is entered and again before
every request (redirects included), so a public URL cannot redirect to, or
later resolve to, an internal address.

Setting MINIGPTKB_ALLOW_PRIVATE_ENDPOINTS=1 turns the check off, for a local
installation that uses a model served on the same machine or network.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
import time
from urllib.parse import urlsplit

import httpx
from openai import DefaultHttpxClient

ALLOW_PRIVATE = os.environ.get("MINIGPTKB_ALLOW_PRIVATE_ENDPOINTS", "").strip().lower() in ("1", "true", "yes")

# Resolutions are kept briefly so a run does not look up its endpoint for
# every request.
RESOLVE_TTL_SECONDS = 60.0
_resolved: dict[tuple[str, int], tuple[float, str | None]] = {}
_lock = threading.Lock()


class EndpointNotAllowed(ValueError):
    pass


def _problem(host: str, port: int) -> str | None:
    """Why the host may not be called, or None when all its addresses are public."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return f"The endpoint host {host!r} could not be resolved."
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global:
            return (f"The endpoint host {host!r} resolves to a non-public address ({address}); "
                    "only publicly reachable endpoints are allowed.")
    return None


def check_url(url: str) -> None:
    """Raises EndpointNotAllowed unless the URL is http(s) on a public host."""
    if ALLOW_PRIVATE:
        return
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise EndpointNotAllowed("The endpoint must be an http:// or https:// URL.")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise EndpointNotAllowed("The endpoint URL has an invalid port.") from None
    key = (parts.hostname.lower(), port)
    with _lock:
        cached = _resolved.get(key)
    if cached and time.monotonic() - cached[0] < RESOLVE_TTL_SECONDS:
        problem = cached[1]
    else:
        problem = _problem(*key)
        with _lock:
            _resolved[key] = (time.monotonic(), problem)
    if problem:
        raise EndpointNotAllowed(problem)


def _check_request(request: httpx.Request) -> None:
    check_url(str(request.url))


def http_client() -> httpx.Client:
    """An HTTP client for the OpenAI SDK that checks every request it sends."""
    return DefaultHttpxClient(event_hooks={"request": [_check_request]})
