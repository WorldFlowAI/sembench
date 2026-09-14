"""Capture one real streamed chunk carrying per-request metrics, verbatim.

``sembench/engine_metrics.py`` parses the ``metrics`` object vLLM 0.29.0
attaches to the final usage chunk of a stream, and
``tests/fixtures/vllm_0290_stream_chunk.json`` pins that shape. That fixture is
**derived from source, not captured**: it was hand-built by reading the vLLM
serializers, so it proves the parser matches what the source says and nothing
about what a server emits. The phase-0 handoff makes replacing it a deliverable
of the E5 GPU smoke -- "E5 must capture one real streamed chunk and commit it
as the parser fixture" -- and this module is how the run produces it.

It is opt-in (``run-live-gateway --metrics-chunk-output PATH``) and writes
exactly once per run: the FIRST chunk of the FIRST request whose ``metrics``
object is a mapping. Once, because the file is evidence of the wire format, not
a log -- a second chunk would say the same thing and the first one is the one
that arrived before anything in the run could have gone sideways.

The file carries the raw SSE payload as it came off the socket
(``raw_sse_data``, the text after ``data: `` with nothing re-encoded) AND the
parsed object under ``chunk``, which is the key the fixture's readers already
use (``tests/test_harness_round3_runner.py:158-160``), so a captured file is a
drop-in replacement for the derived fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAPTURED_PROVENANCE = (
    "CAPTURED VERBATIM off the wire. raw_sse_data is the exact SSE payload the "
    "engine sent (the text after 'data: ', unmodified); chunk is that same "
    "payload parsed. This replaces the DERIVED fixture at "
    "tests/fixtures/vllm_0290_stream_chunk.json -- when it does, the assertion "
    "in tests/test_harness_round3_runner.py that the fixture says 'DERIVED FROM "
    "SOURCE, NOT CAPTURED' has to be inverted, because that is the whole point "
    "of the replacement."
)


@dataclass
class MetricsChunkCapture:
    """One-shot, thread-safe writer for the run's first metrics-bearing chunk.

    Thread-safe because the dispatcher runs requests on worker threads at any
    ``--concurrency``; without the lock two threads could both see "not yet
    written" and race on the same path.
    """

    path: str
    run_id: str = ""
    # ``LiveGatewayConfig.arm``: the cold/warm pairing arm, which on a phase-0
    # run is always "single". The PHASE-0 arm (a1_stock_pc, a4_conn_span) is
    # not a field of that config at all -- it reaches the runner as
    # ``--backend-id`` and as part of ``--run-id`` -- so it is read off
    # ``run_id`` rather than invented here.
    pairing_arm: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _written: bool = field(default=False, repr=False)

    @property
    def written(self) -> bool:
        return self._written

    def offer(
        self,
        *,
        raw: str,
        chunk: Mapping[str, Any],
        base_url: str = "",
        request_id: str | None = None,
    ) -> bool:
        """Write this chunk if nothing has been written yet. True when it wrote.

        A failure to write is swallowed after the flag is set: losing the
        fixture must not kill an arm that is otherwise measuring correctly, and
        retrying on the next chunk would quietly capture a later one while the
        file claims to be the first.
        """
        with self._lock:
            if self._written:
                return False
            self._written = True
        document = {
            "_provenance": CAPTURED_PROVENANCE,
            "_captured": {
                "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": self.run_id,
                "pairing_arm": self.pairing_arm,
                "base_url": base_url,
                "request_id": request_id,
                "sembench_symbol": "sembench.gateway_live._chat_completion",
            },
            "raw_sse_data": raw,
            "chunk": dict(chunk),
        }
        out = Path(self.path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as error:
            # Reported, not raised, and not retried: an unwritable fixture path
            # must not kill an arm that is otherwise measuring correctly, and a
            # retry on the next chunk would capture a LATER chunk into a file
            # that claims to be the first.
            print(f"metrics-chunk capture failed for {self.path}: {error}", file=sys.stderr)
            return False
        return True


def capture_for(
    path: str | None, *, run_id: str = "", pairing_arm: str = ""
) -> MetricsChunkCapture | None:
    """A capture for ``path``, or None when the run did not ask for one."""
    if not path:
        return None
    return MetricsChunkCapture(path=path, run_id=run_id, pairing_arm=pairing_arm)


def add_metrics_chunk_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--metrics-chunk-output`` on a runner subparser.

    Registered here rather than in ``sembench.cli`` so the flag's help text
    lives beside the code that honours it, and so the CLI module does not grow
    another block every time a runner gains an artifact.
    """
    parser.add_argument(
        "--metrics-chunk-output",
        default=None,
        metavar="PATH",
        help="Write the run's FIRST streamed chunk that carries a per-request metrics "
        "object to PATH, verbatim (the raw SSE payload plus the parsed chunk). This is "
        "the parser fixture tests/fixtures/vllm_0290_stream_chunk.json owes: the "
        "committed one is derived from vLLM source, not captured. Off by default",
    )
