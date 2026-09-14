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
connector involvement, so **it must never be gated on as semantic reuse**.
`reuse_mechanism` and every speedup gate read `external_confirmed_tokens` and
`fuzzy_confirmed_tokens` instead; the prefix-cache bucket (`exact`) is
deliberately excluded from the semantic mechanisms.

Two figures in the result do still read `cached_tokens`, and both are stated
here rather than disclaimed, because a disclaimer the code does not honour is
how prefix-cache mass gets published as semantic reuse:

| Figure | What it actually reads | Quotable as semantic reuse? |
| --- | --- | --- |
| `hit_rate_external_confirmed` | `semantic_reuse_tokens` only — `external_confirmed_tokens` or `fuzzy_confirmed_tokens`, over the pairs that carried one | **Yes.** This is the strict key. |
| `hit_rate` | the same, but a pair with no semantic signal at all falls back to `backend_confirmed_tokens >= 64` (`REUSE_HIT_THRESHOLD_TOKENS`) | **No.** Legacy figure. |
| `backend_confirmed_block_rate` (gated by `--min-backend-confirmed-block-rate`) | `cached_tokens` in block form | **No.** Engine coverage check. |
| `materialized_reuse_rate` | `runtime_materialized` events from the connector audit | **Yes**, and it is the only figure that proves materialization. |

**`hit_rate` must not be quoted for a vLLM run with prefix caching on.** On
such a run the legacy fallback is cleared by an ordinary repeated document, so
the number is a prefix-cache hit rate wearing a semantic name. Quote
`hit_rate_external_confirmed`, and report `pairs_external_confirmed` /
`pairs_external_unconfirmed` / `hits_unverified_external` beside it so the
reader can see how many pairs could be judged at all.

`external_confirmed_tokens is None` means the split was never measured. It does
**not** mean zero, and it is not counted as a miss: `hit_rate_external_confirmed`
is `None` — not `0.0` — when no pair carried the split, so a gate that refuses
`None` fails instead of passing on prefix-cache mass. The one exception is a
row the connector audit *did* join and that carried no `runtime_materialized`:
there the audit looked and found nothing, so the row counts as a confirmed
miss (0) rather than as unmeasured.

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
be combined with `--disable-log-stats`), a streamed chat completion carries a
top-level `metrics` object, and the runner splits it into two fields:

```text
engine_ttft_ms   time_to_first_token_ms, else first_token_time - first_scheduled_time
                 -- measured from scheduling, so queue wait is EXCLUDED
queue_time_ms    queue_time_ms, else first_scheduled_time - arrival_time
                 -- the wait itself, reported beside it rather than folded in
```

Seconds-valued keys are converted; a key without an `_ms` suffix is read as
seconds. The second name in each line is a fallback for other engines and for
vLLM's older timestamp-shaped record, not something 0.29 emits.

### Where the object actually is (vLLM 0.29.0, read from source)

The wire format was an assumption until it was checked, so the checks are
written down. On a streamed chat completion:

- `metrics` is a **top-level field of the chunk**, not nested under `usage`
  (`vllm/entrypoints/openai/chat_completion/protocol.py:180`).
- It rides on the **final usage chunk only** — the one whose `choices` is
  empty (`chat_completion/serving.py:817-870`). A parser that stops reading at
  the last content delta never sees it.
- That final chunk exists only when usage reporting is on, so the runner always
  sends `stream_options.include_usage: true`. Without it there is no usage
  chunk and therefore no metrics.
- The object is suppressed entirely when the request asks for more than one
  completion (`serving.py:841-842`). The runner never sets `n`.
- Its fields are `PerRequestMetrics`
  (`vllm/entrypoints/generate/base/protocol.py:55-62`):
  `time_to_first_token_ms`, `generation_time_ms`, `queue_time_ms`,
  `mean_itl_ms`, `tokens_per_second`, `speculative_decoding`. The chunk is
  serialized with `exclude_none=True`, so a field whose timestamps were
  unavailable is absent rather than null.
- `time_to_first_token_ms` is `(first_token_ts - scheduled_ts) * 1000` and
  `queue_time_ms` is `(scheduled_ts - queued_ts) * 1000`
  (`vllm/entrypoints/generate/base/serving.py:78-85`), so the two are disjoint
  and must never be summed into a "TTFT".

`tests/fixtures/vllm_0290_stream_chunk.json` holds that shape with the same
citations. It is **derived from source, not captured**, and it says so: replace
it with a chunk captured off the wire during the E5 GPU run. If the captured
chunk still parses, the derived shape was right; if it does not, that file was
the assumption that hid the difference.

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

## Connector Audit Join (M1 / M2 / M7)

### The join key

`run-live-gateway` and `merge-results` take `--connector-audit PATH`, the
connector's audit JSONL (`audit_path` / `SEMBLEND_VLLM_AUDIT_PATH`,
`schema_version` 2). The join is by request id and nothing else.

The id is **set, not captured**. For each request the runner derives one from
`(run_id, arm, item_id, role, stream_position)` and sends it two ways:

```text
X-Request-Id: sembench-warm-000012-recipient-1f0c7a4d9b2e6503   (header)
{"request_id": "sembench-warm-000012-recipient-1f0c7a4d9b2e6503", ...}  (body)
```

The header is what vLLM prefers (`_base_request_id`,
`vllm/entrypoints/serve/engine/serving.py:117-126`); the body field
(`ChatCompletionRequest.request_id`,
`chat_completion/protocol.py:389-396`) covers a front end that strips unknown
headers. vLLM then builds its own id as `chatcmpl-<sent id>`
(`chat_completion/serving.py:281-283`), and that is the string the connector
writes into every audit event.

Capturing `chunk["id"]` instead would not work: it gains a per-prompt suffix
when one HTTP request carries several prompts, which silently drops rows on an
equality join. Rows therefore carry the **sent** id (`engine_request_id` on
`RequestMetrics`), and `sembench.connector_audit` resolves an audited engine id
back to the header it came from — exact match first, then prefix/suffix
normalization, and an explicit *ambiguous* verdict rather than a guess when one
normalized key covers two audited requests.

Roles keep donors and recipients apart (`recipient`, `donor-0`, `donor-1`, …),
so a capture event is attributable to the request that seeded it. The arm and
the stream position are in the id's readable head; the digest tail covers the
run id and item id as well, so the two arms of a paired run and two runs
appending to one audit file cannot collide. Because nothing is random, the same
manifest replayed under the same `--run-id` re-derives the same ids.

A `--connector-audit` path that does not exist is refused before the arm issues
any traffic: a misspelled path and an arm that materialized nothing are
otherwise the same null.

### Did the engine keep the id? (`config.request_id_echo`)

The join is by construction only as long as the id survives the trip. A front
end that strips or rewrites `X-Request-Id` makes vLLM mint its own, and every
audit-derived metric then reads exactly like an arm that materialized nothing.
The engine echoes its id on every chunk, so each row keeps both — the id that
was sent (`engine_request_id`) and the id that came back
(`engine_response_id`) — and the result's config carries the comparison:

```text
request_id_echo.rows_checked
request_id_echo.rows_id_echoed                 chatcmpl-<sent> / cmpl-<sent>-0
request_id_echo.rows_id_mismatched             a front end minted its own id
request_id_echo.rows_without_engine_response_id
request_id_echo.mismatch_examples              up to three sent/returned pairs
```

`rows_id_mismatched > 0` is the difference between "no reuse happened" and
"the join key never arrived", and it is reported whether or not an audit was
joined.

### The manifest's half of the join

None of the inputs below can come from an engine, so `run-live-gateway`'s
row constructor stamps them from `WorkloadItem.metadata` (via
`manifest_expectations` in `sembench/schema.py`):

```text
expected_supplied_tokens       offline prediction: tokens the planner should supply
expected_span_target_start     offline prediction: where the span should start
traffic_class                  no_reuse | same_doc_new_instruction | revised_doc |
                               rope_delta_sweep | exact_repeat | propagation_probe |
                               reworded_doc
propagation_parent_item_id     the item a propagation probe repeats verbatim
rope_delta_bucket              0 | 128 | 512 | 2048 on the rope_delta_sweep class,
                               null elsewhere -- M6's quality split
stream_position                the item's place in the replay order -- M3 pairs
                               twins at the same position
wrapper_id                     the instruction wrapper the prompt was built with
                               (w1-terse ... w8-workflow)
wrapper_rank                   its rank in the workload's Zipf popularity order,
                               0 = most popular -- M1's shared-wrapper vs ad-hoc
                               strata; null is "no wrapper declared", not rank 0
```

Both live runners stamp them: `run-live-gateway` and `run-live-sglang`. A row
that carries none of them is outside every class-scoped metric (M1's
opportunity classes, M6's bucket split, M7's probe set) and outside the
stream-position check the pairing does.

An absent key stamps `null`, never `0`: "the manifest made no claim" and "the
manifest predicted nothing would be supplied" are different statements about
the same item. `expected_boundary_tokens` is deliberately **not** stamped — the
plan calls it an upper bound whose divergence is routine and legitimate,
because it depends on live GPU residency, eviction and preemption.

### The manifest's denominators

Section 4's class-scoped rates are taken over **manifest items**, not over the
rows a run produced:

```text
alignment_given_opportunity   / |{manifest items in same_doc_new_instruction u revised_doc}|
propagation_contamination     / |the 50 propagation_probe items|
```

The runner is the only stage that reads the manifest, so it writes the
per-class item counts into `config.manifest_class_counts` and they travel with
the result; `merge-results` carries them forward from the arms it joins. Every
class-scoped denominator is taken from them, and
`alignment_given_opportunity_denominator_source` /
`propagation_probe_set_source` report `manifest` or `rows_present` so the
substitution is never silent.

This matters in exactly one direction. A run that errored on half its
opportunity items, or was cut short by `--max-items`, divides by what survived
if the counts are missing — and therefore scores itself **better** for having
lost rows. `alignment_given_opportunity_rows_present` (+ its own denominator)
is published beside the headline so the gap between the two is visible.

### Payload fidelity: the field names are the connector's

Every field the fold reads was taken from the connector's own `_audit_event`
call. One of them is a trap worth naming, because getting it wrong produced a
null rather than an error:

```text
semantic_lookup_hit                             already_computed_tokens   (the boundary)
                                                reusable_tokens, similarity,
                                                materialization_kind, confidence_tier
semantic_span_load_advertised                   boundary, token_count, donor_start,
                                                target_start, snapped_spans
semantic_span_boundary_missed                   boundary, stored_donor_tokens,
                                                n_segments, n_raw_segments,
                                                segments_wrong_donor,
                                                segments_beyond_capture, raw_spans,
                                                snapped_spans
semantic_span_declined_unaligned_boundary       boundary, block_size
semantic_span_declined_below_min_after_clamp    boundary, token_count,
                                                min_semantic_span
semantic_span_supply_clamped                    boundary, requested_tokens,
                                                clamped_tokens   (a clamp, not a decline)
load_allocated / runtime_materialized           tokens
prefix_cache_blocks_evicted                     blocks_evicted
```

**`semantic_lookup_hit` has no `boundary` key.** It is written before any span
arithmetic runs and carries the boundary as `already_computed_tokens`; reading
`boundary` there folded every hit to a null boundary, dropped every hit out of
`alignment_given_match`'s denominator, and published the rate as null.

The three span **declines** are folded too, and land on the row as
`audit_span_decline_event` / `audit_span_decline_reason` /
`audit_span_declined_at`. A request whose lookup hit and whose span was then
declined — off an unaligned boundary, clamped below `min_semantic_span`, or
missed by the boundary — is precisely the misalignment M1 counts, and without
a boundary on its row it silently left the denominator.

### What the join produces

```text
connector_audit_present                   was any audit joined at all
connector_audit_rows_joined               rows the audit had something to say about
connector_audit_rows_considered           rows every rate below was computed over
connector_audit_rows_excluded_cold_arm    cold rows: no connector ran in that arm
connector_audit_rows_excluded_not_joined  rows no audit was joined to at all

manifest_class_counts                     the workload the denominators below count

alignment_given_match                     M1  + _numerator / _denominator
alignment_given_opportunity               M1  + _numerator / _denominator
alignment_given_opportunity_numerator_outside_classes
                                          advertises won outside the two classes
alignment_given_opportunity_denominator_source
                                          "manifest" or "rows_present"
alignment_given_opportunity_rows_present  same numerator over the rows held
alignment_given_opportunity_rows_present_denominator
boundary_alignment_rate                   alias of alignment_given_opportunity
alignment_by_wrapper_stratum              M1 per stratum: shared_wrapper / ad_hoc /
                                          unstratified, each with the three M1
                                          numbers and their numerators
wrapper_stratum_rule                      the stratum boundary, stated in the document
wrapper_stratum_head_share                what the PINNED boundary selected here
wrapper_stratum_head_is_majority          false when it no longer selects a majority
boundary_miss_breakdown                   M1, {reason: count}
span_decline_breakdown                    M1, {decline event: count}
expected_supplied_tokens_agreement_rate   M1 integrity check + _numerator / _denominator
expected_span_target_start_agreement_rate M1 integrity check + _numerator / _denominator

materialized_reuse_rate                   M2, token-weighted (the headline)
materialized_reuse_tokens                 its numerator
materialized_reuse_advertised_tokens      its denominator
materialized_reuse_superseded_tokens      mass written against a superseded promise
materialized_reuse_token_rate             alias of materialized_reuse_rate
materialized_reuse_request_rate           M2 by request + _numerator / _denominator

prefix_blocks_evicted                     M7's gating counter, + rows_with_prefix_blocks_evicted
propagation_cached_without_materialization_rate
                                          M7 supporting signal + _numerator / _denominator
```

And per row, from the audit fold:

```text
audit_semantic_lookup_hit                 did the provider find a donor at all
audit_lookup_hit_boundary                 the hit's own already_computed_tokens
audit_lookup_reusable_tokens              the hit's reusable_tokens (a claim, not reuse)
audit_observed_boundary                   last advertise, else last span event
audit_advertised_tokens                   the LAST advertise's token_count
audit_advertised_target_start
audit_boundary_at_span_start
audit_boundary_miss_reason                the partition above
audit_boundary_missed_at
audit_span_decline_event                  which of the three declines, and
audit_span_decline_reason                 why, and
audit_span_declined_at                    at which boundary
audit_load_allocated / audit_materialized
audit_superseded_materialized_tokens      mass written against a superseded promise
audit_declined_reasons
audit_prefix_blocks_evicted
```

**Which rows count.** Every rate above is computed over the *auditable* rows
only: an arm that ran a connector (never a `cold` row) and rows an audit was
actually joined to (`audit_joined is not None`). `merge-results` writes both
arms into one document, and computing M1/M2 over that list doubles every
denominator with requests no connector ever saw — halving each rate for free.
The two exclusion counters above say how many rows left, and why.

#### M1 — boundary alignment

Not one number. Section 4 of the phase-0 plan gives three, all conditioned on
`boundary > 0` and all deduped by `request_id`:

```text
alignment_given_match        = |{semantic_span_load_advertised, boundary>0, token_count>0}|
                             / |{semantic_lookup_hit, boundary>0}|

alignment_given_opportunity  = same numerator
                             / |{manifest items in same_doc_new_instruction ∪ revised_doc}|

boundary_miss_breakdown      = boundary_missed events partitioned by reason:
                               stored_donor_tokens == 0      -> donor_not_captured
                               stored_donor_tokens < span    -> donor_too_short
                               n_raw_segments > 0, snapped=0 -> below_min_semantic_span
                               otherwise                     -> true_misalignment
```

The two rates share one numerator and differ only in what they condition on.
`alignment_given_match` asks *when the provider found a donor, did the engine's
boundary land on a span?* — a property of the tokenizer and the template, and
null (never `1.0`) when the connector emits no `semantic_lookup_hit` events to
divide by. `alignment_given_opportunity` asks *of the traffic that should have
been reusable, how much was served?* — the product number, and therefore the
headline; `boundary_alignment_rate` is its alias and nothing else.

**The numerator is shared, exactly as section 4 writes it, so the opportunity
rate can exceed `1.0`.** Not every advertise comes from one of the two
classes: a `rope_delta_sweep` or `exact_repeat` item carries a donor and
advertises too. Restricting the numerator to the two classes would make the
number look like a fraction while answering a question section 4 did not ask,
so the excess is named instead —
`alignment_given_opportunity_numerator_outside_classes` is exactly how many of
the advertises came from outside the denominator's population, and a rate above
one is read against it.

`boundary_miss_breakdown`'s second line is read off the payload the connector
writes, not off the arithmetic above it. **A `stored_donor_tokens < span` test
is unreachable**: the connector trims every segment to the captured window
before it builds `raw_spans` —
`length = min(seg.token_count, stored_tokens - seg.donor_start)`, and a segment
whose length is `<= 0` is dropped and counted in `segments_beyond_capture` —
so no raw span can be longer than the stored donor and the comparison never
fires. A capture shortfall is `segments_beyond_capture > 0`, or, on a payload
predating those counters, `n_raw_segments == 0 and n_segments > 0` with no
`segments_wrong_donor` to account for it. Classifying on the unreachable rule
meant the bucket never fired and every short-donor miss was published as a
misalignment.

Beside it, `span_decline_breakdown` counts the three declines that follow a
lookup hit (`semantic_span_declined_unaligned_boundary`,
`semantic_span_declined_below_min_after_clamp`,
`semantic_span_boundary_missed`). Together the two breakdowns account for
every hit that was not served.

One deliberate deviation remains: a miss event carrying none of the
partition's fields is `unclassified` rather than `true_misalignment`. A pre-B9
connector's payload contains no diagnosis, and publishing one from it would
invent the finding.

The integrity check beside them compares the live planner with the offline
model: `expected_supplied_tokens_agreement_rate` over the rows that carried
both an expectation and an advertise, and the same for
`expected_span_target_start`. Divergence on the token count means the offline
model and the live engine disagree about the planner — investigate.

##### The wrapper strata: `alignment_by_wrapper_stratum`

Section 4 is explicit: **"Do not report a single blended alignment number.
Report it separately for the shared-wrapper stratum and the ad-hoc stratum."**
Alignment is a property of the token-level tail the donor and the recipient
wrapper share (caveat A of the plan), so a blended rate over a
popularity-skewed stream mostly reports which wrapper happened to be popular.

The manifest stamps `wrapper_id` (`w1-terse` … `w8-workflow`) and
`wrapper_rank` (0 = most popular) on every row, through the same
`manifest_expectations` path that carries `traffic_class`, `rope_delta_bucket`
and `stream_position`. The stratum rule, published in the document itself as
`wrapper_stratum_rule`:

```text
shared_wrapper   wrapper_rank <= 1   (w1-terse, w2-retrieval)
ad_hoc           wrapper_rank >= 2   (w3-extractive … w8-workflow)
unstratified     wrapper_rank null   (the manifest named no wrapper)
```

**`ad_hoc` is the plan's word for this stratum, not a description of the
traffic in it.** Stream B draws all eight wrappers from one fixed shared set
(`phase0-build-manifest.py`: `WRAPPERS`, sampled by
`zipf_weights(len(WRAPPERS), 1.1)`), so **no item in this workload carries a
one-off wrapper of its own**: ranks 2–7 are shared wrappers that are merely
unpopular — the tail of the popularity order — and their rank counts are
`{0: 479, 1: 222, 2: 161, 3: 101, 4: 104, 5: 75, 6: 55, 7: 53}`. Quote the
`ad_hoc` number as *tail-popularity* traffic; a reader told "the ad-hoc stratum
aligns at 1.03" will otherwise believe ad-hoc traffic was measured. The same
sentence travels in every document inside `wrapper_stratum_rule`. Nor is the
boundary a natural break in the distribution: the head is a thin 56.1% majority
and rank 1 (17.8% of rows) sits about five points above rank 2 (12.9%) — it is
the smallest prefix carrying a majority, which is a stated rule and not a gap
in the data.

**This boundary is an interpretation, and here is the one it rejected.** The
plan asks for the split (line 343) and defines neither "shared-wrapper" nor
"ad-hoc". The wording it uses elsewhere is pairwise — caveat A is about "the
donor and recipient wrappers" sharing a token-level tail, and the class table
asserts `donor.wrapper_id != recipient.wrapper_id` on `same_doc_new_instruction`
— so the closer literal reading is **donor/recipient wrapper identity**, which
the manifest's `donor_item_id` makes computable. It is rejected because it is
degenerate for M1: M1's opportunity population is `same_doc_new_instruction ∪
revised_doc`, and `same_doc_new_instruction` is wrapper-mismatched *by
construction*, so a donor-identity "shared" stratum would be near-empty and
explain nothing. The reading taken instead is **popularity**, which is what the
plan's own mechanism sentence (line 128, "the shared wrapper never enters the
prefix cache") is about: the wrapper the bulk of the stream sits on, which is
what makes `boundary > 0` at all.

**Why rank 1 is the boundary.** Stream B draws its eight wrappers from
Zipf(s=1.1) (`phase0-build-manifest.py`: `WRAPPERS`,
`zipf_weights(len(WRAPPERS), 1.1)`), which puts 39.8% of the stream on rank 0,
18.6% on rank 1, and under 9% on every rank below. Ranks 0–1 are therefore the
smallest prefix of the popularity order carrying a **majority** of the traffic
— 58.4% modelled, 56.1% of the 1,250 real rows in `phase0-stream-b.jsonl`
(479 + 222). That is what "shared wrapper" means operationally: the head the
bulk of the stream sits on, so the **recipient's own** wrapper is routinely
already resident in the prefix cache and its `boundary` is non-zero — lane 2's
precondition, and what the plan's line-128 mechanism is about. It is **not** a
claim that the donor was built with the same wrapper: in M1's opportunity
population the donor's wrapper differs from the recipient's by construction
(of the 450 `same_doc_new_instruction ∪ revised_doc` items in
`phase0-stream-b.jsonl`, the 225 that name a `donor_item_id` share the
recipient's `wrapper_id` in **zero** cases), which is precisely why the
identity reading was rejected two paragraphs up. Everything below the head is
traffic on a tail wrapper whose recipient wrapper is usually cold, so the
token-level tail alignment is a property of is a different one.

**The constant is pinned, so the document checks it.** `SHARED_WRAPPER_MAX_RANK`
is derived from that manifest and hardcoded, and on a manifest with a different
wrapper count or a different Zipf exponent the published rule ("the head the
majority of the stream sits on") would quietly stop being true. Every document
therefore publishes what the constant actually selected on its own rows:

```text
wrapper_stratum_head_share       shared_wrapper rows / rows that declared a rank
                                 (null when no row declared one)
wrapper_stratum_head_is_majority false when the pinned boundary no longer
                                 selects a majority of the ranked rows
```

A false majority flag does not invalidate the split — the strata still
partition the rows and still sum to the blended numerator — it says the
*name* has drifted from the manifest, and the boundary needs rederiving before
the stratum labels are quoted.

Each stratum publishes M1's three numbers over its own rows —
`alignment_given_match`, `alignment_given_opportunity`,
`boundary_miss_breakdown` (plus `span_decline_breakdown` and
`rows_considered`) — each with its numerator and denominator. The per-stratum
opportunity denominator is always `rows_present`: the manifest's class counts
are not split by wrapper, so there is no workload-level number to divide by and
mixing one in would compare a manifest count with a row count.

The three strata partition the rows the audit was joined to, so **the
per-stratum numerators sum to the blended numerator**: the split explains the
headline and cannot change it. `unstratified` exists so that identity holds on
a manifest predating `wrapper_id` — those rows are named rather than folded
into a stratum they never declared. `alignment_by_wrapper_stratum` is null when
no connector audit was joined, like every other rate in this block.

#### M2 — materialized reuse

```text
materialized_reuse_rate = Σ runtime_materialized.tokens
                        / Σ semantic_span_load_advertised.token_count
```

That token-weighted ratio is the headline and is what `materialized_reuse_rate`
carries (`materialized_reuse_token_rate` is an alias of the same number, kept
for continuity). The request-count form — how many advertising requests got
*any* of their promise — is a different question and has its own name,
`materialized_reuse_request_rate`. The two disagree whenever the served
requests are not the large ones, which is exactly when the difference matters.

**Both sums obey one rule: the last advertise, and the materializations that
followed it.** The denominator was already the last advertise — the connector
re-advertises only when the plan changed — while the numerator summed every
`runtime_materialized` event on the request. That is a different rule on each
half of one ratio: a request advertised at 256 tokens, materialized, then
re-advertised at 768 and materialized again reported `(256 + 768) / 768` —
133% of its promise, with no extra KV reused. "Followed" is decided by position
in the audit file, which is a real total order across both roles because both
append to one path. Mass written against a superseded promise is published as
`materialized_reuse_superseded_tokens` rather than dropped: KV was written,
just not against the promise the denominator holds.

An advertise is a promise and an allocation is a destination; only
`runtime_materialized` is evidence that KV was written.

#### M4 — miss tax (paired documents only)

```text
median ttft_ms(A4 | supplied == 0) - median ttft_ms(A1), over the same item_ids
```

"Supplied == 0" is read off the audit, not off the outcome: a pair is in the
population when its warm row **advertised nothing** —
`audit_advertised_tokens` is null (the connector said nothing about it) or `0`
(it looked and supplied nothing). A row that advertised and then failed to
materialize is *not* in it: it was served a promise, and its latency prices
keeping or breaking that promise rather than the cost of a miss. Negative
controls are excluded for the same reason the blended speedup excludes them,
and the count says so.

```text
miss_tax_ms                              median warm TTFT - median cold TTFT; positive is a tax
miss_tax_ms_median_of_differences        the paired form, + _ci
miss_tax_warm_ttft_p50_ms                its two terms
miss_tax_cold_ttft_p50_ms
miss_tax_pairs                           pairs in the population
miss_tax_pairs_without_ttft              in the population, but one arm never answered
miss_tax_pairs_advertising_excluded      pairs whose warm row advertised
miss_tax_pairs_not_audited_excluded      pairs whose warm row the audit never spoke
                                         about (audit_joined false or absent)
miss_tax_pairs_outside_capture_class_excluded
                                         non-advertising pairs the capture leg's
                                         no_reuse filter removed (0 on every other pair)
miss_tax_pairs_negative_control_excluded
miss_tax_definition                      the population, stated in the document
miss_tax_population                      the population RULE this document applied
miss_tax_source                          "connector_audit", or why the population
                                         could not be identified
miss_tax_lookup_latency_ms_sum           section 4's scheduler-thread leg, from the
miss_tax_lookups_total                   arm's engine counter window
miss_tax_lookup_ms_per_lookup
miss_tax_lookup_cost_source              "engine_window", or null when unmeasured
```

**The population is the pairs the audit measured, row by row.**
`audit_advertised_tokens` is null both when the connector looked and advertised
nothing *and* when the audit holds nothing about the request at all — every
`MATCH_MISSING` row, which the join stamps `audit_joined=false`. Reading the
second as the first prices the connector's miss path with requests no connector
event ever described, so a pair whose warm row the audit did not speak about is
dropped into `miss_tax_pairs_not_audited_excluded` **before** the "advertised
nothing" filter runs. That is the same row-level rule M1 and M2 apply through
`auditable_rows`: one definition of "the audit measured this row", shared by
all three.

**Without a joined audit the tax is unmeasured.** Above the row filter sits the
document-level guard the per-arm block applies (`audit_was_joined` over the
warm rows). With no audit anywhere, `miss_tax_ms`,
`miss_tax_ms_median_of_differences` (+ `_ci`), both p50s and the population
counts — `miss_tax_pairs`, `miss_tax_pairs_without_ttft`,
`miss_tax_pairs_advertising_excluded`, `miss_tax_pairs_not_audited_excluded`
and `miss_tax_pairs_outside_capture_class_excluded` — are null,
`miss_tax_source` says the population **could not be identified**, and
`miss_tax_population` says nothing was placed in it. Null here means
unmeasured, never "no tax". Two things are deliberately *not* in that list and
are always published: `miss_tax_pairs_negative_control_excluded`, which is a
property of the pair set rather than of the audited population, and the
`miss_tax_definition` / `_population` / `_source` strings, which state the rule
the document applied — including the rule that it could not be applied.

**The capture leg has its own population.** Section 4 defines the capture cost
as "A3 → A4 delta on `no_reuse` items", so on a document merged with
`--pair m4_capture` the population is the non-advertising pairs **that are
`no_reuse`**, and `miss_tax_population` names that rule. A
`same_doc_new_instruction` request that merely failed to advertise is a miss,
not a capture: pricing the capture path with it charges the connector for
traffic that had a donor. The pairs the filter removed are counted in
`miss_tax_pairs_outside_capture_class_excluded`; on every other document that
counter is `0` and the population is section 4's plain "supplied == 0".

Section 4 decomposes the tax into three legs. Only the first is a per-arm
counter ratio (`lookup_latency_ms_sum / lookups_total`), and it is available
only when the connector exports those counters onto the endpoint the run
scraped; stock vLLM exposes neither, so on a stock arm the leg is **null, not
zero** — a zero would subtract a cost nobody measured from a published tax.
The other two legs are cross-arm and each is a `merge-results --pair` away:
`m4_capture` (A3 → A4) and `m4_instrumentation` (A5 → A4). Each of those
documents' own `miss_tax_ms` is the leg it measured, and
`config.arm_pair` says which leg that is.

#### M7 — contamination / propagation (paired documents only)

M7 is a **cross-arm answer comparison**, so it lives in the `paired` block and
needs both arms:

```text
propagation_contamination_rate            null unless some run is IDENTIFIED as the
                                          cold (A1) arm and it scored a probe
propagation_contamination_numerator       propagated probes; null with the rate
propagation_contamination_denominator     the PROBE SET, not the scored probes
propagation_probe_set_source              "manifest" or "rows_present"
propagation_contamination_rate_scored_only
                                          + propagation_contamination_scored_denominator
propagation_cold_reference_arm            never null, and never provenance when
                                          _missing is true. Three shapes: an arm id
                                          ("A1") = that arm answered; "undeclared" =
                                          neither the reference document nor
                                          run.baseline_arm_declared named an arm;
                                          "required: A1" = the baseline is a known arm
                                          that is NOT the cold reference, so A1 is
                                          still needed
propagation_cold_reference_source         "reference_arm" | "reference_arm_undeclared"
                                          | "baseline_arm"; null whenever _missing
propagation_cold_reference_missing        true when no arm on this document answered as
                                          the cold reference — including when it names
                                          no arm at all
propagation_cold_reference_unusable       true when one was SUPPLIED and scored no
                                          probe at all (wrong manifest, rows stamped
                                          warm, every probe position-mismatched)
propagation_no_probe_scored               true when a non-empty probe set yielded no
                                          scored probe by ANY route; the rate, its
                                          numerator and the scored-only rate are null
                                          whatever supplied the cold answer
propagation_contamination_rate_vs_baseline_arm
                                          the same comparison against THIS document's
                                          baseline arm; a diagnostic, never section 4's
                                          metric
propagation_contamination_numerator_vs_baseline_arm
                                          its numerator; always published, because the
                                          diagnostic is always computed
propagation_definition                    what the comparison actually did
propagation_probe_pairs                   probe items present in both arms
propagation_probes_excluded_unclean_pair  no twin, contaminated cold arm, an error,
                                          or a stream-position mismatch
propagation_probes_unlinked               no parent_item_id on the row
propagation_probes_without_served_answer  parent absent, or answered nothing
propagation_probes_without_answers        probe missing an answer in an arm
propagation_probes_without_cold_reference the cold reference arm holds no usable
                                          answer for this probe
propagation_probes_reference_position_mismatched
                                          the reference row sat at another stream
                                          position, so it answered another stream
propagation_probes_absent_from_run        declared by the manifest, no row here
```

**The cold answer is A1's, not the merge baseline's.** Section 4 says the
fraction "whose answer in A6 matches the *served* output rather than the
*cold* (A1) output" — and the `m7_propagation` merge's baseline arm is **A4**,
the product arm, which can be contaminated on the very same probe. When both
merged arms drift towards the served answer they drift together, the strict
comparison finds no difference, and a fully contaminated workload can publish
`0.0`. So the reference is explicit and named in the document:

- `merge-results --cold-reference <A1 result.json>` supplies it. Reference
  rows are matched by `item_id` **and** stream position — a reference row at
  another position replayed a different stream and is not this probe's cold
  answer (counted in `propagation_probes_reference_position_mismatched`).
  `propagation_cold_reference_source` reads `reference_arm`, and
  `propagation_cold_reference_arm` is the arm **that document declares** for
  itself. `--cold-reference-arm` (default `A1`) is an operator assertion, so it
  is checked against the reference's own `--backend-id` and a contradiction is
  refused, exactly as `--pair` is checked against both source arms. A reference
  that declares no arm is accepted and recorded as `undeclared` under the
  source `reference_arm_undeclared`, rather than published as an asserted A1
  nothing verified.
- With no reference run, a document whose baseline **is** A1 — `--pair
  m3_ttft` / `m6_noise_floor`, or an unlabelled join whose
  `run.baseline_arm_declared` names A1 — answers from its own cold twin, and
  the source reads `baseline_arm`.
- With no reference run on a document whose baseline is not A1, there is no
  cold answer at all: `propagation_contamination_rate`, its numerator and the
  scored-only rate are **null**, `propagation_cold_reference_missing` is true,
  every probe is counted in `propagation_probes_without_cold_reference`, and
  `propagation_cold_reference_arm` reads `required: A1` — a requirement, not a
  claim that A1 was consulted. This is read off `--pair` **and** off
  `run.baseline_arm_declared`, which the merge stamps from the cold arm's
  `--backend-id` / `--baseline-id`: `--pair` is optional, so an A4 vs A6 merge
  that nobody labelled is still an m7-shaped merge and still gets no rate.
- **A run id is never an arm declaration.** `run.baseline_id` is run
  *identity* and falls back to the cold **run id** when the cold arm declared
  no label, and `arm_id_of` matches an arm name anywhere in a free-text id — so
  a run called `phase0-g5-a1-rack-cold` would resolve to A1. M7 therefore reads
  `run.baseline_arm_declared`, which carries the operator's declaration and has
  no such fallback (empty when none was made). The two fields are both written
  on every document: read `baseline_id` to learn which run was the baseline,
  `baseline_arm_declared` to learn which arm it said it was.
- **A document that identifies no arm at all gets no rate either**, and that
  is the default shape: `--backend-id` defaults to the empty string, so a plain
  `merge-results` declares no baseline arm at all, and a backend id that names
  a build rather than an arm (`vllm-0.29-span`,
  `sglang-fuzzy-pr31057`) resolves to no arm. Such a document is
  indistinguishable from an A4-vs-A6 merge that labelled nothing, so it is
  suppressed the same way, with `propagation_cold_reference_arm: "undeclared"`
  distinguishing the two. Publishing M7 takes an **affirmative** A1 signal —
  `--cold-reference`, `--pair m3_ttft` / `m6_noise_floor`, or a cold arm whose
  `--backend-id` names A1 — never the mere absence of a contradicting one.
  Through round 6's second pass this branch still published the number, which
  is the same 0.0-on-a-contaminated-workload the explicit reference exists to
  prevent, reachable with default flags and no optional argument at all.
- **A document that scored no probe at all publishes no rate**, whatever
  supplied the cold answer. A non-empty probe set none of whose members could
  be read — every probe unlinked, every parent absent, every pair excluded as
  unclean, the probe set never replayed, or a reference that answered none of
  them — can only produce `0 / |probe set|`, a clean contamination number
  manufactured entirely by failing to measure. The rate, its numerator and the
  scored-only rate are null there under `propagation_no_probe_scored: true`,
  and the exclusion counters below say which way it happened. This is
  source-independent on purpose: the identical condition used to publish `0.0`
  when the cold answer came from the document's own A1 baseline arm and null
  when it came from a supplied reference.
  `propagation_cold_reference_unusable: true` remains the narrower flag for the
  supplied-reference case (another manifest, rows stamped `arm='warm'`, every
  probe position-mismatched), so that case stays distinguishable from "no
  reference was given" and from "nothing was scored". `merge-results` refuses
  such a reference outright when it shares no item with the merged arms.
  An **empty** probe set is not this case: a workload with no probes had
  nothing to read, and its rate is null because 0/0 is.

`propagation_contamination_rate_vs_baseline_arm` is always published: it is the
same comparison taken against whatever this document calls its baseline. It is
a useful arm-vs-arm diagnostic and it is deliberately under a distinct name,
because on an `m7_propagation` document it is exactly the number that
under-reports.

**The denominator is the probe set.** Section 4 says "A4 vs A6 on the 50
`propagation_probe` items", and a probe that could not be scored is not
evidence of no contamination — it is a probe that was not read. Dividing by
the scored subset turns every failure to score into a better contamination
number, which is the one direction a contamination metric must never drift.
So the headline rate divides by the manifest's probe count and every exclusion
is published beside it; `propagation_contamination_rate_scored_only` keeps the
scored subset under its own name, to judge the headline by and never to
replace it.

A propagation probe is a verbatim repeat of an earlier request that was served
approximate KV, so three answers exist for one prompt: the **served** answer
(the parent item's answer in the same arm), the **cold** answer (this item's
answer in the cold reference arm above, which is what an uncontaminated engine
must return), and the treatment answer under test. A probe counts as propagated
when its treatment answer is strictly closer (ROUGE-L) to the served answer
than to the cold one — strictly, because a tie is not evidence.

Read it beside `prefix_blocks_evicted`, which is the counter section 4 makes
the gate on lane-2 quality: **until that counter reads non-zero on a
contaminated workload, treat every lane-2 quality number as unproven,
including a favourable one.**

`propagation_cached_without_materialization_rate` — probes that materialized
nothing of their own yet still reported cached tokens — is a per-row
supporting signal in the per-arm block, not M7.

Every rate is published beside its own numerator and denominator, so `0.0` over
three requests is never read as `0.0` over three hundred. Every audit-derived
key is `null` when no audit was joined — not `0.0`. A run that never looked
must not publish a clean score, and `config.connector_audit_join` records rows
matched exactly, matched after normalization, unmatched, ambiguous, and without
an id at all, so a shrunken denominator is never silent.

## Pairing (which twins are comparable)

Section 4 states M3 as "per `item_id`, at the same stream position". Both
twins carry the manifest's `stream_position`, so the pairing checks it: twins
at different positions did not replay the same stream — a re-ordered or edited
manifest between the arms — and their TTFT ratio measures the reorder, not the
cache. A row that declares no position makes no claim and is not excluded by
it.

```text
pairs_total                        items in the cold arm
pairs_used                         clean pairs with a TTFT on both sides
pairs_contaminated                 cold twin whose cache reset did not take
pairs_errored                      either arm returned an error
pairs_unpaired                     cold row with no warm twin
pairs_stream_position_mismatched   twins at different manifest positions
```

`pairs_unpaired` and `pairs_stream_position_mismatched` were previously
dropped without a count.

## Answer Quality By RoPE Delta (M6)

```text
quality_by_rope_delta_bucket   {"0": {...}, "128": {...}, "512": {...}, "2048": {...}}
                               per bucket: requests, mean_quality_f1,
                               mean_quality_rouge_l, quality_pass_rate, mean_ttft_ms
```

Section 4: "Report quality per RoPE-delta bucket from the `rope_delta_sweep`
class. A quality result gathered only at |delta| <= 11 does not transfer."
Re-rotating donor KV across a large positional delta is the specific quality
risk lane 2 carries, and a blended mean over a stream that is 93% |delta| ~ 0
cannot show it. The bucket comes from the manifest (`rope_delta_bucket`), keys
are strings because the result is JSON, and the whole block is `null` when no
row declares a bucket — the split was not measured, which is not the same as a
workload with no positional deltas.

## Latency

```text
ttft_ms      client-side, at the streamed first token (unchanged)
latency_ms   SERVICE latency: the server time this item cost, summed over its
             donor requests and its recipient request
```

`latency_ms` changed in round 2 and the change applies at **every**
concurrency, the serial path included. It is no longer the item's wall span.
A wall span also contains the dispatcher's own waits — the donor→recipient gap
and the post-donor settle — which are delays the harness imposed, not latency
the engine produced; leaving them in made an arm look slower in proportion to
how carefully it was isolated. Numbers from before that change are wall spans
and are not comparable with numbers after it.

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
- `negative_control_ttft_speedup_median` (+`_ci`) — must sit at ~1.0;
  deviation means the cache acted on unrelated content. This is what
  `--max-negative-control-speedup-deviation` gates, and the gate reports the
  CI beside the point estimate, for the same reason the headline gate reads a
  median: one control pair whose cold arm hit a slow prefill moves a mean far
  enough to fail a deviation gate on its own — reporting contamination that
  did not happen, and by the same arithmetic hiding one that did.
  `negative_control_ttft_speedup_mean` stays beside it as a secondary,
  tail-sensitive figure and is not gateable.
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
