"""Engine-reported per-request timings, read off a streamed completion.

Verified against vLLM 0.29.0 rather than assumed. With
``--enable-per-request-metrics`` the chat endpoint attaches a **top-level**
``metrics`` object to the **final usage chunk** of a stream --
``ChatCompletionStreamResponse.metrics``
(``vllm/entrypoints/openai/chat_completion/protocol.py:180``), populated at
``vllm/entrypoints/openai/chat_completion/serving.py:838-866``.

Three conditions gate it there, and all three are the caller's to meet:

- usage reporting must be on, i.e. ``stream_options.include_usage`` (the
  runner always sends it) or ``--enable-force-include-usage``. Without a final
  usage chunk there is nowhere for the object to ride;
- the request must ask for a single completion -- ``n > 1`` suppresses it;
- the server must have been started with the flag, which cannot be combined
  with ``--disable-log-stats``.

The object's fields are ``PerRequestMetrics``
(``vllm/entrypoints/generate/base/protocol.py:55-62``):
``time_to_first_token_ms``, ``generation_time_ms``, ``queue_time_ms``,
``mean_itl_ms``, ``tokens_per_second``, ``speculative_decoding``. The chunk is
serialized with ``exclude_none=True``, so a field whose timestamps were
unavailable is absent rather than null.

``tests/fixtures/vllm_0290_stream_chunk.json`` pins that shape. It is derived
from the source, not captured, and must be replaced by a chunk captured during
the E5 GPU run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# The two names vLLM 0.29.0 actually emits.
VLLM_0290_TTFT_KEY = "time_to_first_token_ms"
VLLM_0290_QUEUE_KEY = "queue_time_ms"

# TTFT is (first_token_ts - scheduled_ts) * 1000 and queue time is
# (scheduled_ts - queued_ts) * 1000
# (vllm/entrypoints/generate/base/serving.py:78-85), so the engine's TTFT
# excludes queue wait -- the only TTFT that stays readable once an arm runs
# under concurrency, and a number that must never be summed with the queue
# wait reported beside it.
#
# The names after the verified one are fallbacks for other engines and for
# vLLM's older timestamp-shaped record; a key without an `_ms` suffix is read
# as seconds.
ENGINE_TTFT_KEYS = (VLLM_0290_TTFT_KEY, "ttft_ms", "time_to_first_token", "ttft")
ENGINE_QUEUE_KEYS = (VLLM_0290_QUEUE_KEY, "time_in_queue_ms", "queue_time", "time_in_queue")

NO_ENGINE_TIMING: dict[str, float | None] = {"engine_ttft_ms": None, "queue_time_ms": None}


def float_or_none(value: Any) -> float | None:
    """``float(value)``, or None for anything that is not a number."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def engine_metrics_from_chunk(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The per-request metrics object carried by one SSE chunk, or None.

    Only a mapping counts. A gateway that puts something else under ``metrics``
    would otherwise be stored verbatim and read back as "the engine reported no
    timings", which is a different claim from "the engine reported timings this
    runner could not parse".
    """
    metrics = chunk.get("metrics")
    return metrics if isinstance(metrics, Mapping) else None


def _duration_ms(metrics: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """First present key as milliseconds; a key without a `_ms` suffix is seconds."""
    for key in keys:
        value = float_or_none(metrics.get(key))
        if value is None:
            continue
        return value if key.endswith("_ms") else value * 1000
    return None


def _elapsed_ms(metrics: Mapping[str, Any], start_key: str, end_key: str) -> float | None:
    """Milliseconds between two engine timestamps, or None if either is absent."""
    start = float_or_none(metrics.get(start_key))
    end = float_or_none(metrics.get(end_key))
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000


def engine_timing(response: dict[str, Any]) -> dict[str, float | None]:
    """Engine-reported TTFT and queue wait for one request, in milliseconds.

    Both stay None when the engine was not started with
    ``--enable-per-request-metrics``; the client-side ``ttft_ms`` is
    unaffected, but under concurrency it is dominated by queue time and a TTFT
    ratio taken from it is not the engine's.
    """
    metrics = response.get("metrics")
    if not isinstance(metrics, Mapping):
        return dict(NO_ENGINE_TIMING)
    queue_ms = _duration_ms(metrics, ENGINE_QUEUE_KEYS)
    if queue_ms is None:
        queue_ms = _elapsed_ms(metrics, "arrival_time", "first_scheduled_time")
    ttft_ms = _duration_ms(metrics, ENGINE_TTFT_KEYS)
    if ttft_ms is None:
        ttft_ms = _elapsed_ms(metrics, "first_scheduled_time", "first_token_time")
    return {"engine_ttft_ms": ttft_ms, "queue_time_ms": queue_ms}
