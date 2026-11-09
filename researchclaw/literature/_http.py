"""Shared HTTP helpers for the literature source clients.

Every source client is a thin adapter over a plain HTTP JSON/XML endpoint, so
transport concerns — timeouts, 429 ``Retry-After``, transient 5xx — live here
instead of being re-implemented (and re-broken) once per client. The retry
policy mirrors ``openalex_client._request_with_retry``: exponential backoff
with jitter, ``Retry-After`` honoured, and an absurdly long ``Retry-After``
treated as a hard failure so a single call cannot stall a whole pipeline stage.

Callers keep their existing contract: they catch ``Exception`` and return an
empty list, so a failure degrades the way every other source does.

Why not every client uses this module
-------------------------------------
``openalex_client``, ``semantic_scholar`` and ``arxiv_client`` keep their own
transport on purpose:

- Semantic Scholar and arXiv own a **circuit breaker** whose state is driven
  by individual 429/error responses. ``fetch_json`` retries internally, so
  the breaker would only see the final failure and trip three times later
  than it should.
- OpenAlex's ``_request_with_retry`` *returns ``None``* instead of raising and
  logs redacted URLs; migrating it would change the caller contract.
- arXiv goes through the ``arxiv`` SDK, not raw HTTP at all.

New, stateless backends (OpenReview, Crossref, PubMed, CORE) use this module.
Migrating the three above is a separate refactor with its own test budget.

Public API
----------
- ``fetch_json(url, source=...)`` → decoded JSON payload
- ``fetch_bytes(url, source=...)`` → raw response body (XML endpoints)
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 45
_MAX_RETRY_AFTER = 300.0
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _retry_delay(exc: Exception, attempt: int, base_delay: float) -> float | None:
    """Seconds to wait before retrying, or ``None`` when the error is fatal."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code not in _RETRYABLE_STATUS:
            return None
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                wait = float(header)
            except ValueError:
                wait = base_delay * (2**attempt)
            if wait > _MAX_RETRY_AFTER:
                return None
            return wait
        return base_delay * (2**attempt)
    # URLError, socket timeouts, connection resets: all worth another try.
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        return base_delay * (2**attempt)
    return None


# A literature response is at most a few MB (1000 EPMC records with abstracts
# ~5 MB, a 10k-record PubMed efetch ~40 MB). Anything past this is a broken or
# hostile response, and reading it into memory is how a 60 s timeout turns into
# an OOM kill.
_MAX_BODY_BYTES = 64 * 1024 * 1024


def _read_limited(response: object, source: str) -> bytes:
    """Read a response body, refusing to buffer more than ``_MAX_BODY_BYTES``."""
    read = getattr(response, "read", None)
    if read is None:  # pragma: no cover - only for exotic file-like objects
        return b""
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = read(256 * 1024)
        except TypeError:  # stream whose read() takes no size
            chunk = read()
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_BODY_BYTES:
            raise ValueError(
                f"{source} response exceeded {_MAX_BODY_BYTES} bytes; truncated"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_bytes(
    url: str,
    *,
    source: str,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> bytes:
    """GET *url* and return the raw body, retrying transient failures.

    Parameters
    ----------
    url:
        Fully encoded request URL.
    source:
        Human readable source name, used in log lines only.
    headers:
        Extra request headers (``User-Agent``, ``Authorization`` ...).
    timeout:
        Per-attempt socket timeout in seconds.
    max_attempts:
        Total attempts, including the first one.
    base_delay:
        Backoff base; attempt *n* waits ``base_delay * 2**n`` plus jitter.

    Returns
    -------
    bytes
        Response body.

    Raises
    ------
    Exception
        The last transport error when every attempt failed or the error is
        not retryable (4xx other than 408/425/429, malformed payload).
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        request = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _read_limited(response, source)
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_exc = exc
            delay = _retry_delay(exc, attempt, base_delay)
            if delay is None or attempt == max(1, max_attempts) - 1:
                raise
            delay += random.uniform(0, 0.5)  # noqa: S311 - jitter only
            logger.warning(
                "%s request failed (%s). Retry %d/%d in %.1fs...",
                source,
                exc,
                attempt + 1,
                max(1, max_attempts) - 1,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises


def fetch_json(
    url: str,
    *,
    source: str,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> Any:
    """GET *url* and decode the body as JSON.

    See :func:`fetch_bytes` for the retry semantics and the raised exception.
    """
    raw = fetch_bytes(
        url,
        source=source,
        headers=headers,
        timeout=timeout,
        max_attempts=max_attempts,
        base_delay=base_delay,
    )
    return json.loads(raw)
