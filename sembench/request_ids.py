"""Deterministic request ids: the join key between a result row and an audit.

vLLM honours an ``X-Request-Id`` request header and turns it into the engine's
own request id by prefixing it, and the SemBlend vLLM connector writes that
engine id into every audit event. So the harness never has to *capture* an id
it can *set*: derive one from the five things that identify a request inside a
run, send it, and the join between a result row and the connector audit holds
by construction instead of by inference.

Derived, never random. The same manifest replayed under the same run id
re-derives the same ids, so an audit written on a worker joins to a result
written somewhere else with neither side keeping a table.

vLLM 0.29.0 sources for the behaviour this module relies on:

- ``vllm/entrypoints/serve/engine/serving.py:117-126`` -- ``_base_request_id``
  returns the ``X-Request-Id`` header when the caller sent one, otherwise the
  request body's ``request_id``, otherwise a random uuid. The header wins.
- ``vllm/entrypoints/openai/chat_completion/serving.py:281-283`` -- the engine
  request id is ``f"chatcmpl-{_base_request_id(raw_request, request.request_id)}"``.
  That prefixed string is what the connector sees and what the audit records,
  which is why rows carry the prefixed form.
- ``vllm/entrypoints/openai/chat_completion/protocol.py:389-396`` --
  ``ChatCompletionRequest.request_id`` is a real body field, so the id can be
  sent both ways and still survives a front end that strips unknown headers.
- ``vllm/entrypoints/openai/chat_completion/serving.py:305-308`` -- a request
  that expands into several engine inputs has ``_{i}`` appended per sub
  request, while the streamed response id stays the unsuffixed form. That
  mismatch is exactly why the id is set rather than read off ``chunk["id"]``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, replace
from typing import Any, TypeVar

# vLLM's chat endpoint prefixes whatever id it was given; the audit records the
# prefixed form, so that is the string a row has to carry to join on equality.
ENGINE_REQUEST_ID_PREFIX = "chatcmpl-"

RECIPIENT_ROLE = "recipient"

# How much of the digest ends up in the id. 16 hex characters is 64 bits: at
# benchmark scale the collision probability is not the thing that will go
# wrong, and a short id stays greppable in a log line.
_DIGEST_CHARS = 16
_MAX_TOKEN_CHARS = 24

# Where the id lands on a result row. ``sembench.schema`` grows the field
# alongside the audit join; a row schema with no slot for it is returned
# unchanged rather than refused, because the id is derived and therefore
# recoverable from (run_id, arm, item_id, role, stream_position) regardless.
ROW_REQUEST_ID_FIELDS = ("request_id", "engine_request_id")

_CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar(
    "sembench_current_request_id", default=None
)

RowT = TypeVar("RowT")


def donor_role(index: int) -> str:
    """Role token for the ``index``-th donor of one replay step."""
    return f"donor-{int(index)}"


def deterministic_request_id(
    *,
    run_id: str,
    arm: str,
    item_id: str,
    role: str,
    stream_position: int,
) -> str:
    """The id this harness sends as ``X-Request-Id`` for one request.

    The readable head (arm, stream position, role) is there so a human can
    find the request in an audit file; the digest tail is what actually makes
    the id unique, and it covers the run id and the item id as well, so two
    arms of the same run and two runs against the same audit file cannot
    collide.
    """
    digest = hashlib.sha256(
        "\x1f".join(
            (str(run_id), str(arm), str(item_id), str(role), str(int(stream_position)))
        ).encode("utf-8")
    ).hexdigest()
    return "-".join(
        (
            "sembench",
            _id_token(arm),
            f"{int(stream_position):06d}",
            _id_token(role),
            digest[:_DIGEST_CHARS],
        )
    )


def engine_request_id(request_id: str) -> str:
    """The id vLLM will actually use, which is the one the audit records."""
    return f"{ENGINE_REQUEST_ID_PREFIX}{request_id}"


def _id_token(value: str) -> str:
    """A header-safe, lowercase ASCII token: no encoding surprises in flight.

    HTTP header values must be latin-1 encodable, and an item id or arm label
    may be neither ASCII nor short, so everything outside ``[a-z0-9]`` becomes
    a dash and the result is clipped.
    """
    lowered = str(value).lower()
    cleaned = "".join(
        char if ("a" <= char <= "z" or "0" <= char <= "9") else "-" for char in lowered
    )
    trimmed = cleaned[:_MAX_TOKEN_CHARS].strip("-")
    return trimmed or "x"


@contextmanager
def sending_request_id(request_id: str | None) -> Iterator[str | None]:
    """Make ``request_id`` the id of the request issued inside this block.

    The chat call is a seam the runner's own tests replace wholesale, so the
    join key travels beside the call rather than through its signature: adding
    a parameter would mean every test double had to grow one too, and a double
    that did not would fail as a transport error rather than as a missing id.
    The context is per thread and is always restored, so a pooled worker
    thread cannot carry one request's id into the next request it serves.
    """
    token = _CURRENT_REQUEST_ID.set(request_id)
    try:
        yield request_id
    finally:
        _CURRENT_REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    """The id to send with the request being issued on this thread, if any."""
    return _CURRENT_REQUEST_ID.get()


def stamp_request_id(row: RowT, request_id: str) -> RowT:
    """Return a copy of ``row`` carrying the audit join key.

    A frozen row is never mutated: ``dataclasses.replace`` builds a new one.
    When the row schema has no field for the id the row is returned unchanged
    -- the alternative, raising, would take down a run over an artifact the
    join can re-derive.
    """
    names = {entry.name for entry in fields(row)}  # type: ignore[arg-type]
    for candidate in ROW_REQUEST_ID_FIELDS:
        if candidate in names:
            return replace(row, **{candidate: request_id})  # type: ignore[type-var]
    return row


def row_request_id_field(row_type: Any) -> str | None:
    """Which of ``ROW_REQUEST_ID_FIELDS`` this row schema actually carries."""
    names = {entry.name for entry in fields(row_type)}
    for candidate in ROW_REQUEST_ID_FIELDS:
        if candidate in names:
            return candidate
    return None
