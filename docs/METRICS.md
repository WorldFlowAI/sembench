# Metric Contract

This benchmark separates exact reuse, semantic discovery, backend eligibility,
and backend confirmation.

## Denominator

`total_blocks` counts full recipient prefill blocks only:

```text
total_blocks = floor(recipient_prompt_tokens / block_size)
```

Trailing partial blocks are excluded from block-rate denominators. Token-weighted
rates use full prompt token counts.

## Exact Baseline

`exact_block_hit_rate` uses content hash matches against all donor full blocks:

```text
exact_block_hit_rate = exact_hit_blocks / total_blocks
```

This is intentionally a strong baseline. It measures exact reusable blocks even
when they appear at different target positions.

## SemBlend Candidate Rate

`semantic_candidate_block_rate` counts full recipient blocks for which every
target token is marked `copy_from_donor` by SemBlend:

```text
semantic_candidate_block_rate = semantic_candidate_blocks / total_blocks
```

This is discovery/planning evidence. It is not backend-confirmed reuse.

## Backend Eligibility

`semantic_eligible_block_rate` counts candidate blocks that are contiguous from
one donor span and therefore plausible for block materialization:

```text
semantic_eligible_block_rate = semantic_eligible_blocks / total_blocks
semantic_eligible_lift = semantic_eligible_block_rate - exact_block_hit_rate
```

This is still an offline proxy. It does not mean a serving engine loaded the KV.

## Backend Confirmation

`backend_confirmed_block_rate` is populated only by live runners:

```text
backend_confirmed_block_rate = backend_confirmed_blocks / total_blocks
backend_confirmed_lift = backend_confirmed_block_rate - exact_block_hit_rate
```

For SGLang, the local runner derives this from backend-reported cached tokens.
If a server reports exact and semantic reuse through the same field, the result
should be treated as backend-confirmed reuse, not semantic-only reuse.

Engine log/audit summaries add a second confirmation layer:

```bash
sembench summarize-engine-events --engine vllm --input vllm.log
sembench summarize-engine-events --engine trtllm --input trtllm-audit.jsonl
sembench summarize-engine-events --engine sglang --input sglang.log
```

The summary field `materialized_semantic_kv_reuse` is true only when the parser
sees both a semantic hit and backend materialization/reuse evidence. For vLLM,
that means a SemBlend semantic lookup hit plus request-local load/materialized
events. For TensorRT-LLM, that means a `lookup` audit event with `found=true`
plus `materialized` audit events; `engine_blend_boundary` is tracked separately
because materialization alone is not enough to prove suffix-only engine
execution. For SGLang, current logs prove donor registration and semantic hits;
cached-token accounting from the response remains the primary
backend-confirmed reuse metric until SGLang emits explicit materialization
events.

## Counter Semantics: cached_tokens Is Not Semantic Reuse

The three confirmed-reuse fields on a request row do not mean the same thing,
and only two of them are evidence of semantic reuse:

```text
backend_confirmed_tokens   local prefix cache + external KV transfer (vLLM sums
                           them before the API sees them; vllm/v1/metrics/stats.py)
external_confirmed_tokens  external KV connector ONLY, local prefix cache excluded
fuzzy_confirmed_tokens     SGLang fuzzy-admitted mass; a local prefix hit never
                           lands here
```

`backend_confirmed_tokens` is populated from
`usage.prompt_tokens_details.cached_tokens`. With prefix caching on, replaying
the same document hits the local cache and inflates that field without any
connector involvement, so **it must never be gated on as semantic reuse**. The
hit rate, `reuse_mechanism`, and every speedup gate read
`external_confirmed_tokens` and `fuzzy_confirmed_tokens` instead; the
prefix-cache bucket (`exact`) is deliberately excluded from the semantic
mechanisms.

`external_confirmed_tokens is None` means the split was never measured. It does
**not** mean zero, and it is not counted as a miss: `hit_rate_external_confirmed`
is `None` — not `0.0` — when no pair carried the split, so a gate that refuses
`None` fails instead of passing on prefix-cache mass.

`external_confirmed_tokens_source` records where the number came from, because
the two sources support different claims:

| Source | Granularity | What may be said |
| --- | --- | --- |
| `connector_audit` | per request | "this request reused N external tokens" |
| `arm_prometheus_delta` | per arm | arm-level totals only — the counter (`vllm:external_prefix_cache_hits`) is process-wide and cannot be attributed to a request |

`external_confirmed_is_per_request` in the paired summary reports which of the
two the run actually carried.

## Engine-Side TTFT

With vLLM's `--enable-per-request-metrics` (phase 0 requires it; note it cannot
be combined with `--disable-log-stats`), each streamed response carries a
`metrics` chunk, and the runner splits it into two fields:

```text
engine_ttft_ms   time_to_first_token_ms, else first_token_time - first_scheduled_time
                 -- measured from scheduling, so queue wait is EXCLUDED
queue_time_ms    queue_time_ms, else first_scheduled_time - arrival_time
                 -- the wait itself, reported beside it rather than folded in
```

Seconds-valued keys are converted; a key without an `_ms` suffix is read as
seconds.

Client-side `ttft_ms` is the sum of these plus network. Under concurrency it is
dominated by queue wait, which is a function of offered load rather than of
cache reuse, so it is not comparable across arms at different widths. The
paired summary reports the engine-side comparison separately —
`engine_ttft_pairs`, `engine_ttft_speedup_median` (+ `_ci`),
`queue_time_cold_p50_ms`, `queue_time_warm_p50_ms` — and deliberately keeps it
out of the headline. Both fields are `None` on an engine that does not emit the
chunk; nothing is inferred.

## Engine Block (Prometheus window)

`engine-snapshot` / `engine-window` attach an `engine` block per arm:

```text
engine.flags                       the serve line and env this arm ran with
engine.phase0_flag_violations      flags phase 0 requires and this arm lacked
engine.phase0_flags_ok
engine.prometheus.delta            before/after counter deltas for the arm
  external_prefix_cache_queries_delta
  external_prefix_cache_hits_delta          advertised-and-ALLOCATED
  external_kv_transfer_prompt_tokens_delta  advertised-and-accepted only
  external_prefix_cache_hit_ratio
  counter_reset_detected                    a counter that went backwards mid-arm
  endpoints_paired / endpoints_failed       workers whose window actually closed
```

The deltas come from `vllm:external_prefix_cache_hits` /
`vllm:external_prefix_cache_queries` (recorded after the scheduler allocated
slots, so *advertised and allocated*) and
`vllm:prompt_tokens_by_source{source="external_kv_transfer"}` (set before
allocation, so *advertised and accepted* only — a worker that later declines
materialization never decrements it). They are reported under separate names so
the weaker one is never mistaken for the stronger.

Counters are read as a window, never as a total: a server reused across arms
carries the previous arm's mass. Neither counter is materialization —
materialization is observable only in the connector audit stream.

## Throughput Document

`run-live-gateway --concurrency N` (N > 1) writes a second document beside the
result — `--throughput-output`, defaulting to `--output` with its extension
replaced by `.throughput.json`:

```text
requests / items / errors / concurrency
wall_seconds                          dispatcher wall clock for the arm
settle_seconds_est                    post-donor settle the harness imposed
idle_seconds / settle_excluded_basis  "measured_idle" when idle was measured,
                                      "estimate" when it was modelled
requests_per_second                   every donor and recipient / wall_seconds
requests_per_second_excluding_settle  the same over served seconds only
output_tokens_per_second
donor_ttft_ms / recipient_ttft_ms     p50 / p90 / p99 / mean
recipient_latency_ms                  p50 / p90 / p99 / mean
```

Rate is reported twice on purpose. A settle that exists only to let the engine
index donors is idle time the server was never offered, and leaving it in the
denominator understates both arms by different amounts. TTFT percentiles come
from the per-request records, measured at the streamed first token; nothing
here re-derives TTFT from end-to-end latency.

## Gateway And Router Placement

`run-live-gateway` replays recipient requests through an OpenAI-compatible
gateway and records route metadata when the gateway exposes it:

```text
route_outcomes
semantic_placement_rate_by_request
route_endpoint_id
route_semantic_score
route_total_score
gateway_route_header
```

These are placement metrics, not KV materialization metrics. A semantic
placement outcome or route header means the control plane chose an affinity
route. It does not imply that SGLang, vLLM, or TensorRT-LLM materialized donor
KV for the request. Use a live backend runner or backend audit summary for
`backend_confirmed_block_rate`.

## Negative Controls

Negative controls pair unrelated donor and recipient contexts. The main safety
signal is:

```text
negative_control_semantic_eligible_rate
```

This should remain near zero before interpreting positive lift as useful.

## TTFT contract (paired runs)

The `paired` result block is the TTFT source of truth; single-arm
`mean_ttft_ms` is informational only.

- `blended_ttft_speedup_median` (+`_ci`) — headline, and the only speedup
  that is gateable (`--min-blended-ttft-speedup`). Every clean non-negative
  pair contributes its real cold/warm ratio; misses naturally contribute
  ~1.0x. Never report hit-only numbers as the headline. The median is the
  headline because speedups are ratios: one pair whose cold arm stalled
  carries a mean over a threshold the typical pair never reached. CIs come
  from a paired bootstrap that resamples `(cold[i], warm[i])` as a unit.
- `blended_ttft_speedup_mean` (+`ttft_speedup_mean`, `blended_ttft_speedup_ci`)
  — secondary, kept for continuity. Not gateable.
- `hit_only_ttft_speedup_median` (+`_ci`) — reported ONLY alongside `hit_rate`.
- `engine_ttft_speedup_median` (+`_ci`) — the engine-side companion, reported
  apart from the headline (see Engine-Side TTFT above).
- `ttft_{cold,warm}_p{50,95}_ms` — per-arm percentiles.
- `negative_control_ttft_speedup_median` / `negative_control_ttft_speedup_mean`
  — must sit at ~1.0; deviation means the cache acted on unrelated content
  (gate: `--max-negative-control-speedup-deviation`, which reads the mean).
- `hit_rate_external_confirmed` — the strict hit rate: external-connector
  mass only, prefix-cache repeats excluded. `None`, never `0.0`, when no pair
  carried the external split; `pairs_external_confirmed` /
  `pairs_external_unconfirmed` / `hits_unverified_external` say how many pairs
  could be judged. `hit_definition` names the exclusion in the document.
- `pairs_contaminated` — cold arms whose engine reported cached tokens;
  excluded from every aggregate and gateable via
  `--require-contamination-check`.
- Quality companion: `warm_vs_cold_output_rouge_l_mean` is judged against a
  protocol-matched noise-floor artifact (see CALIBRATION.md) — never as an
  absolute.
