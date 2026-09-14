"""Row-level reuse signals: what counts as semantic reuse, and what was audited.

These classify one row and are used by both the per-arm section-4 blocks and
the paired blocks. The distinction they exist to keep is the one the whole
suite turns on: ``cached_tokens`` is local prefix cache plus external transfer
and is never semantic-reuse evidence, while an absent measurement is None and
never zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from sembench.schema import PER_REQUEST_EXTERNAL_SOURCES, RequestMetrics

# A warm arm counts as a HIT only at >= this many SEMANTIC reuse tokens.
# Single-digit "hits" (e.g. a 3-token shared literal prefix) are numeric
# noise, not reuse.
REUSE_HIT_THRESHOLD_TOKENS = 64

# Mechanisms that are semantic reuse. "exact" is the prefix cache doing its
# ordinary job and is deliberately absent: on a prefix-caching-on arm a
# repeated document hits the local cache, and counting that as a semantic win
# is the single easiest way to publish a speedup the connector did not earn.
SEMANTIC_MECHANISMS = ("scatter", "head", "external")


def semantic_reuse_tokens(row: RequestMetrics) -> int | None:
    """Tokens this row reused SEMANTICALLY, or None when unmeasurable.

    Only two signals qualify, and neither can contain a local prefix hit:

    - ``external_confirmed_tokens`` — external KV connector mass (vLLM);
    - ``fuzzy_confirmed_tokens`` — sglang's fuzzy-admitted mass.

    ``backend_confirmed_tokens`` is explicitly NOT consulted: it is vLLM's
    ``cached_tokens``, which sums local prefix cache and external transfer,
    so with prefix caching on a repeated document reads as reuse there.

    None means no semantic signal was recorded at all — a pre-B12 row, where
    the engine was never asked for the split. None is not zero, and callers
    must not collapse it into one.

    A row the connector audit *did* join and that carries no
    ``runtime_materialized`` is the one case where an absent external count is
    a measurement: the audit looked and nothing was materialized. Those rows
    return 0 (a confirmed miss) rather than None, so they cannot fall through
    to the cached_tokens-based legacy hit.
    """
    external = row.external_confirmed_tokens
    fuzzy = row.fuzzy_confirmed_tokens or 0
    if external is None:
        if row.audit_materialized is False:
            return fuzzy
        return fuzzy or None
    return max(int(external), fuzzy)


def is_reuse_hit(row: RequestMetrics) -> bool | None:
    """Did this row clear the hit threshold on semantic reuse alone?

    None when :func:`semantic_reuse_tokens` is None, so an unmeasured arm
    reads as unmeasured rather than as a clean miss.
    """
    tokens = semantic_reuse_tokens(row)
    return None if tokens is None else tokens >= REUSE_HIT_THRESHOLD_TOKENS


def reuse_mechanism(row: RequestMetrics) -> str:
    """Classify HOW a warm arm reused: 'scatter' (fuzzy mass beyond the
    prefix path), 'head' (fuzzy folded into the prefix), 'external'
    (external-connector mass, the vLLM semantic path), 'exact' (prefix cache
    only — NOT a semantic win), or 'none'. Mechanisms damage quality
    differently (attention-sink positions vs mid-sequence) — never pool them.

    The 'exact' bucket is where a prefix-cache repeat lands, and it is kept
    out of :data:`SEMANTIC_MECHANISMS` on purpose: it is the bucket that
    exists so cached_tokens mass has somewhere honest to go.
    """
    fuzzy = row.fuzzy_confirmed_tokens or 0
    external = row.external_confirmed_tokens or 0
    prefix_path = row.backend_confirmed_tokens or 0  # local + external
    if fuzzy > prefix_path:
        return "scatter"
    if fuzzy > 0:
        return "head"
    if external >= REUSE_HIT_THRESHOLD_TOKENS:
        return "external"
    if prefix_path >= REUSE_HIT_THRESHOLD_TOKENS:
        return "exact"
    return "none"


def audit_was_read(row: RequestMetrics) -> bool:
    """Was a connector audit read for this row at all?

    ``audit_joined`` is None until :func:`sembench.connector_audit.join_requests`
    touches a row, so None is "no audit exists" and False is "the audit was
    read and held nothing about this request".
    """
    return row.audit_joined is not None


def audit_measured_row(row: RequestMetrics) -> bool:
    """Did the audit actually speak about THIS row?

    True only when the join found the connector's events for it. The
    distinction matters wherever an absent audit field is read as a
    measurement: on a joined row ``audit_advertised_tokens`` null means "the
    connector looked and advertised nothing", and on an unjoined row
    (``audit_joined`` False, every MATCH_MISSING request) it means nothing was
    measured at all. Collapsing the two prices the connector for requests the
    audit never saw.
    """
    return row.audit_joined is True


def audit_was_joined(rows: list[RequestMetrics]) -> bool:
    """Did a connector audit get joined onto any of these rows?

    This separates "the audit was read and said nothing about these requests"
    (False on the rows, True here) from "no audit exists" (None on the rows,
    False here). Every audit-derived rate below is null in the second case:
    without the audit, every row's ``external_confirmed_tokens`` is null for
    want of measurement, and a rate computed over that reads as a perfect
    score for a run that measured nothing.
    """
    return any(audit_was_read(row) for row in rows)


@dataclass(frozen=True)
class AuditableRows:
    """The rows an audit-derived metric may be computed over, and the rest.

    A row is auditable when the connector could have written events about it
    *and* the audit was actually read for it. Two kinds of row cannot be:

    - a **cold-arm** row, which ran against a server with no connector at all.
      A merged cold/warm document holds both arms in one list, and computing
      M1/M2/M7 over that list doubles every denominator with requests no
      connector ever saw — halving each rate for free.
    - a row the audit was never joined to (``audit_joined is None``), which is
      unmeasured rather than negative.
    """

    rows: tuple[RequestMetrics, ...]
    excluded_cold_arm: int
    excluded_not_joined: int


def auditable_rows(rows: list[RequestMetrics]) -> AuditableRows:
    """Split rows into the auditable ones and counts of what was excluded."""
    from sembench.pairing import COLD_ARM

    considered: list[RequestMetrics] = []
    cold = 0
    unjoined = 0
    for row in rows:
        if row.arm == COLD_ARM:
            cold += 1
            continue
        if not audit_was_read(row):
            unjoined += 1
            continue
        considered.append(row)
    return AuditableRows(tuple(considered), cold, unjoined)


def external_token_sources(rows: list[RequestMetrics]) -> list[str]:
    """Distinct provenance labels on ``external_confirmed_tokens``, sorted."""
    return sorted(
        {
            row.external_confirmed_tokens_source
            for row in rows
            if row.external_confirmed_tokens_source
        }
    )


def external_tokens_are_per_request(rows: list[RequestMetrics]) -> bool | None:
    """True only when every external value present is a per-request number.

    False means at least one row carries an ARM-LEVEL value (the per-arm delta
    of ``vllm:external_prefix_cache_hits``), which supports arm-level claims
    only. None means no row declared a source.
    """
    sources = external_token_sources(rows)
    if not sources:
        return None
    return all(source in PER_REQUEST_EXTERNAL_SOURCES for source in sources)
