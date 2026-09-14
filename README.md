# SemBench

SemBench is a benchmark suite for semantic KV cache reuse. It separates four
questions that are often conflated:

- Did an exact token/block cache already cover the request?
- Did a semantic planner find reusable donor spans?
- Did the serving backend actually materialize or reuse donor KV?
- Did the routed request preserve TTFT and answer quality while negative
  controls stayed cold?

The repository is standalone. Generated LongBench manifests and live result
artifacts are intentionally excluded from git; review the source dataset license
and your sharing policy before publishing generated data.

## Capabilities

- Deterministic fixture workloads for smoke tests.
- LongBench-derived replay manifests with enterprise-style transforms:
  instruction variants, same evidence with new tasks, RAG reorder, multi-donor
  composition, fuzzy edits, leading-evidence new-task, and negative controls.
- Offline exact-cache and semantic-candidate metrics.
- Live SGLang and generic OpenAI-compatible gateway replay.
- Backend log/audit parsers for vLLM, SGLang, and TensorRT-LLM.
- Result gates for quality, route placement, backend-confirmed reuse,
  materialized-token counts, and negative-control safety.
- Paired cold/warm arms with engine cache resets between them, joinable after
  the fact when the arms ran as separate server processes.

## Subcommands

| Subcommand | What it does |
| --- | --- |
| `build` | Build a manifest (fixture, synthetic-v1, longbench-v1/v2). |
| `checksum-manifest` | Print a manifest's SHA256. |
| `freeze` / `verify-frozen` | Build a frozen spec and check it still reproduces. |
| `audit-manifest` | Check a manifest for phantom cross-item block collisions. |
| `verify-endpoint` | Pre-flight a live endpoint (reachability, model identity). |
| `calibrate-noise-floor` | Measure cold/cold ROUGE-L self-agreement for quality gates. |
| `run-offline` | Offline exact-vs-SemBlend metrics; no server involved. |
| `run-live-sglang` | Replay a manifest against SGLang. |
| `run-live-gateway` | Replay through an OpenAI-compatible gateway/router; also the paired runner. |
| `run-load` | Drive donor->recipient streams concurrently; throughput and TTFT under load. |
| `merge-results` | Join a cold and a warm single-arm result into one paired result by `item_id`. |
| `engine-snapshot` | Read the engine's external-KV Prometheus counters; run once per arm boundary. |
| `engine-window` | Combine two snapshots plus the serve line into a result's `engine` block. |
| `summarize-engine-events` | Parse backend logs/audit streams for materialization evidence. |
| `collect-k8s-engine-events` | Pull pod logs and summarize them in one step. |
| `assert-result-gates` | Fail unless the artifacts meet quality, reuse, safety, and speedup gates. |

## Install

```bash
python -m pip install -e '.[dev,tokenizer]'
```

For LongBench ingestion:

```bash
python -m pip install -e '.[longbench,tokenizer,dev]'
```

For live HTTP replay:

```bash
python -m pip install -e '.[live,tokenizer,dev]'
```

## Fixture Smoke Test

```bash
python -m sembench build \
  --profile fixture \
  --output manifests/fixture.jsonl

python -m sembench run-offline \
  --manifest manifests/fixture.jsonl \
  --output results/fixture-offline.json \
  --block-size 16
```

## LongBench-Derived Replay

```bash
python -m sembench build \
  --profile longbench-v1 \
  --datasets qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news lcc repobench-p \
  --max-items-per-dataset 10 \
  --transforms instruction_variant same_evidence_new_task rag_reorder multi_donor_composite fuzzy_edit leading_evidence_new_task negative_control \
  --max-segments 4 \
  --min-segment-chars 400 \
  --output manifests/longbench-v1-enterprise-replay.jsonl

python -m sembench run-offline \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/longbench-v1-offline.json \
  --block-size 16 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct
```

The replay keeps LongBench as the source corpus while reshaping it into
semantic-KV reuse cases: same document with new tasks, reordered retrieval
chunks, multi-donor compositions, fuzzy formatting/edit changes, and unrelated
negative controls.

## Live Engine Replay

For SGLang:

```bash
python -m sembench run-live-sglang \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/longbench-v1-sglang.json \
  --base-url http://localhost:30000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-items 20
```

For any OpenAI-compatible gateway:

```bash
python -m sembench run-live-gateway \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/longbench-v1-gateway.json \
  --gateway-url http://localhost:8080 \
  --donor-url http://localhost:30000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-items 20
```

Gateway/router metrics answer whether traffic went through the expected serving
path. Backend log/audit summaries answer whether the engine reported semantic
KV materialization or reuse.

## Paired Cold/Warm Arms

A TTFT number is a measurement only when something cold was measured beside it.
`--paired` replays a cold and a warm twin per item, adjacent in the stream,
resetting the engine cache before each arm:

```bash
python -m sembench run-live-gateway \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/paired.json \
  --gateway-url http://router:8000 \
  --worker-url http://worker-0:8000,http://worker-1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --paired \
  --reset-url 'http://worker-0:8000/reset_prefix_cache?reset_external=true' \
  --reset-url 'http://worker-1:8000/reset_prefix_cache?reset_external=true' \
  --engine-serve-command-file serve-warm.txt
```

- `--paired` — cold and warm twin per item. Requires `--reset-url`, and
  requires a manifest that carries donor prompts; a manifest without them is
  refused before any traffic is issued rather than reported as a null pairing.
- `--reset-url URL` — engine cache-reset endpoint POSTed before each step;
  repeat per worker (vLLM:
  `http://host:8000/reset_prefix_cache?reset_external=true`). `--paired`
  without it is refused: the cold twin would be warmed by the arm before it.
  Each row records whether its reset actually landed (`cache_reset`), and a
  cold twin whose reset did not take — or that still reported cached tokens —
  is flagged `flush_contaminated` and excluded from every paired aggregate,
  counted in `pairs_contaminated` and gateable with
  `--require-contamination-check`.
- `--worker-url URL` — fleet worker endpoints donors are seeded on directly;
  repeat, or pass a comma-separated list. Recipients still go through
  `--gateway-url`, so what is measured is the router's placement decision.
  These endpoints are also the default `/metrics` scrape targets.
- `--concurrency N` — request streams in flight at once (default 1, serial).
  Above 1 the run also writes a throughput document (`--throughput-output`;
  default: `--output` with its extension replaced by `.throughput.json`).
  Donor->recipient separation, the post-donor settle and the stream order are
  enforced identically at every width, so `--min-donor-gap-requests` means the
  same thing serially and under load.
- `--connector-audit PATH` — the SemBlend vLLM connector's audit JSONL (its
  `audit_path` / `SEMBLEND_VLLM_AUDIT_PATH`). Joined onto the rows by request
  id before the result is written; this is what makes M1/M2/M7 obtainable.
  Also accepted by `merge-results`. A path that does not exist is refused
  before the arm issues any traffic, because "the audit file was misspelled"
  and "the arm materialized nothing" are otherwise the same null downstream.

`--concurrency > 1` is refused together with `--reset-url`, and therefore with
`--paired`: a cache reset firing mid-flight would flush the KV of requests
already in the air. Run the paired TTFT arms serially and the throughput arms
without resets:

```bash
python -m sembench run-live-gateway \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/load.json \
  --gateway-url http://router:8000 \
  --worker-url http://worker-0:8000,http://worker-1:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 8 \
  --min-donor-gap-requests 4
```

Under concurrency, client-side `ttft_ms` is dominated by queue wait. Compare
arms on `engine_ttft_ms` (see [docs/METRICS.md](docs/METRICS.md)).

When the two arms must run as separate server processes (stock baseline vs
connector), run them separately and join afterwards:

```bash
python -m sembench merge-results \
  --cold results/a1-stock-pc.json \
  --warm results/a4-conn-span.json \
  --output results/paired.json \
  --pair m3_ttft
```

The join is on `item_id` and the manifest SHA256, never a heuristic. It refuses
to merge when the arms do not join one-to-one, and when either result's own
`run.arm` label contradicts the flag it was passed under — handing the warm run
to `--cold` inverts every speedup downstream and nothing in the merged document
would say so.

`--pair` names which of the phase-0 plan's section-4 comparisons a merge is,
because section 4 asks for five different cold/warm joins over the same eight
arms and an unlabelled merged document cannot tell them apart:

```text
m3_ttft              A1 -> A4   M3 TTFT speedup and M6 answer quality: the headline
m6_noise_floor       A1 -> A2   the cold-vs-cold floor M6's margin is set from
m4_capture           A3 -> A4   M4's capture leg (miss_tax_ms on this document)
m4_instrumentation   A5 -> A4   M4's instrumentation leg; subtract it from any
                                published tax
m7_propagation       A4 -> A6   M7 contamination
```

The pair is recorded on the merged document as `config.arm_pair` and checked
against each arm's `--backend-id` / `--baseline-id`, so an A3-vs-A4 capture leg
cannot be published as the M3 headline. Arms that did not label themselves are
merged without complaint; a contradiction is refused.

The pair also decides two populations. On `m4_capture` the miss tax is taken
over `no_reuse` pairs only, because section 4 defines the capture cost as the
A3 → A4 delta *on `no_reuse` items*. And on `m7_propagation` the baseline arm
is A4 — the product arm — which is not section 4's cold reference, so M7 needs
a third arm:

```bash
python -m sembench merge-results \
  --cold results/a4-conn-span.json \
  --warm results/a6-conn-span-nomitigation.json \
  --output results/paired-m7.json \
  --pair m7_propagation \
  --cold-reference results/a1-stock-pc.json
```

`--cold-reference` takes an **A1** result and uses its answers as M7's cold
output, matched by `item_id` and stream position. `--cold-reference-arm`
(default `A1`) asserts which arm that document is, and the assertion is checked
against the reference's own `--backend-id`: a contradiction is refused, and a
reference that declares no arm is recorded as `undeclared` rather than as an
unverified A1. A reference that shares no item with the merged arms — another
manifest, or an already-merged document — is refused too, because it would
score no probe at all. What survives is recorded on the merged document as
`config.cold_reference_result` / `config.cold_reference_arm` and published as
`propagation_cold_reference_source` / `propagation_cold_reference_arm`.

Without a reference run, an A4-vs-A6 merge publishes
`propagation_contamination_rate: null` and
`propagation_cold_reference_missing: true` rather than comparing two arms that
can be contaminated together — and that holds whether or not the merge was
labelled `--pair m7_propagation`, because `run.baseline_arm_declared` carries
the cold arm's `--backend-id` and M7 reads it. Merges whose baseline already *is* A1
(`m3_ttft`, `m6_noise_floor`, or a join that declares A1) need no reference
run.

**A merge that identifies no arm gets no rate either.** `--backend-id` defaults
to empty, so a plain `merge-results` declares no baseline arm at all — and a
backend id naming a build (`vllm-0.29-span`) resolves to no arm — which is
indistinguishable from an unlabelled A4-vs-A6 merge. Such a document publishes
a null rate with `propagation_cold_reference_arm: "undeclared"`; pass
`--backend-id`, `--pair`, or `--cold-reference` to get M7. The cold **run id**
is never read as a declaration: `run.baseline_id` falls back to it so the
document says which run was the baseline, but M7 reads
`run.baseline_arm_declared`, which only an operator's `--backend-id` /
`--baseline-id` fills in. A run called `phase0-g5-a1-rack-cold` therefore
unlocks nothing.

## Engine Counters (the external-KV split)

`usage.prompt_tokens_details.cached_tokens` is the local prefix cache **plus**
external KV transfer, summed by the engine before it reaches the API. With
prefix caching on, a repeated document hits the local cache and lands in that
field, so it is not semantic-reuse evidence. vLLM exposes the split on
`/metrics`; read it as a before/after window per arm so the result carries a
delta rather than a process-lifetime total:

```bash
python -m sembench engine-snapshot \
  --metrics-url http://worker-0:8000 \
  --metrics-url http://worker-1:8000 \
  --output results/warm-before.json

# ... run the arm ...

python -m sembench engine-snapshot \
  --metrics-url http://worker-0:8000 \
  --metrics-url http://worker-1:8000 \
  --output results/warm-after.json

python -m sembench engine-window \
  --before results/warm-before.json \
  --after results/warm-after.json \
  --arm warm \
  --engine-serve-command-file serve-warm.txt \
  --result results/warm.json
```

`engine-window` writes the result's `engine` block: the serve line for the arm,
its `phase0_flag_violations` / `phase0_flags_ok`, and the counter deltas under
`prometheus.delta`. A counter that went backwards mid-arm is surfaced as
`counter_reset_detected` instead of being clamped to zero.

## Connector Audit Join (M1 / M2 / M7)

Neither `/metrics` nor `cached_tokens` can say a load was *materialized* rather
than advertised. Only the connector's audit stream can, so three of the phase-0
metrics exist only when a result has been joined against it:

```bash
python -m sembench run-live-gateway \
  --manifest manifests/longbench-v1-enterprise-replay.jsonl \
  --output results/warm.json \
  --gateway-url http://worker-0:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --run-id phase0-a4-warm \
  --connector-audit /var/log/semblend/audit.jsonl
```

The join is by request id and nothing else. The runner derives one id per
request from `(run_id, arm, item_id, role, stream_position)` — never a random
one — and sends it as `X-Request-Id` and as the request body's `request_id`.
vLLM adopts the header and builds its own id from it (`chatcmpl-<sent id>`),
which is the id the connector writes into every audit event, so the two sides
line up by construction rather than by inference. The id also lands on each row
as `engine_request_id`, and `sembench.connector_audit` resolves an audited
engine id back to the header it came from; two audited requests that normalize
to one client id are reported as ambiguous rather than attributed to a row.

Because the ids are derived, a re-run of the same manifest under the same
`--run-id` re-derives the same ids, and an audit written on a worker joins to a
result written anywhere else with neither side keeping a table.

The engine also echoes its own id on every chunk, so each row keeps both the
id that was sent (`engine_request_id`) and the id that came back
(`engine_response_id`), and the result's `config.request_id_echo` counts the
mismatches. A front end that strips `X-Request-Id` makes vLLM mint its own id,
which downstream is the same null as an arm that materialized nothing —
`rows_id_mismatched` is what separates the two.

The manifest supplies the other half of the join, stamped onto every row by
both live runners: `expected_supplied_tokens`, `expected_span_target_start`,
`traffic_class`, `rope_delta_bucket` (M6's quality split), `stream_position`
(M3 pairs twins at the same position), `wrapper_id` / `wrapper_rank` (M1's
shared-wrapper vs ad-hoc strata) and, for a probe, `parent_item_id`. An absent
key stamps `null`, never `0`.

The manifest also supplies the *denominators*. Section 4's class-scoped rates
count manifest items — 450 opportunity items, 50 propagation probes — not the
rows a run happened to produce, so the runner writes the per-class item counts
into `config.manifest_class_counts` and every such denominator is taken from
them. `merge-results` carries them through. Without them the metrics fall back
to the rows present and say so in
`alignment_given_opportunity_denominator_source` /
`propagation_probe_set_source`; a run that errored on half its opportunity
items would otherwise divide by what survived and score itself better for
having lost rows.

What the join adds to the result — always beside its own numerator and
denominator, so `0.0` over three requests is never read as `0.0` over three
hundred:

- **M1, three numbers, all conditioned on `boundary > 0`.**
  `alignment_given_match` divides the advertises by the lookup hits (when the
  provider found a donor, did the boundary land on a span?);
  `alignment_given_opportunity` divides the **same numerator** by the
  manifest's `same_doc_new_instruction ∪ revised_doc` item count (of the
  traffic that should have been reusable, how much was served?) and is the
  headline — `boundary_alignment_rate` is its alias and nothing else. The
  numerator is shared exactly as section 4 writes it, so the opportunity rate
  *can* exceed `1.0`: a `rope_delta_sweep` or `exact_repeat` item carries a
  donor and advertises too.
  `alignment_given_opportunity_numerator_outside_classes` says how many
  advertises came from outside the denominator's population, so a rate above
  one is readable instead of mysterious, and
  `alignment_given_opportunity_rows_present` is the same numerator over the
  opportunity rows this document actually holds.
  `boundary_miss_breakdown` partitions the misses into `donor_not_captured` /
  `donor_too_short` / `below_min_semantic_span` / `true_misalignment`, and
  `span_decline_breakdown` counts the three ways a span is declined *after* a
  lookup hit, so a low alignment rate comes with its diagnosis. Beside them,
  `expected_supplied_tokens_agreement_rate` checks the live planner against the
  offline model. Section 4 also forbids publishing the blended rate alone, so
  `alignment_by_wrapper_stratum` splits all three numbers into the
  **shared-wrapper** stratum (`wrapper_rank <= 1`: the Zipf head the majority
  of the stream sits on) and the **ad-hoc** stratum (every lower rank), plus an
  `unstratified` bucket for rows whose manifest named no wrapper.
  `wrapper_stratum_rule` states the boundary in the document; each stratum
  carries its own numerators and denominators, and they sum to the blended
  ones, so the split explains the headline and cannot change it.
- **M2, token-weighted.** `materialized_reuse_rate` is
  Σ `runtime_materialized` tokens / Σ advertised `token_count`
  (`materialized_reuse_token_rate` is an alias of it). Both sums obey one
  rule — the **last** advertise, and the materializations that followed it —
  so a re-advertised request cannot report more mass than it was promised;
  anything the worker wrote against a superseded promise is published as
  `materialized_reuse_superseded_tokens` rather than dropped. The
  request-count question — how many advertising requests got any of their
  promise — is `materialized_reuse_request_rate`, and neither substitutes for
  the other.
- **M4, the miss tax, lives in the `paired` block.** `miss_tax_ms` is
  median warm TTFT − median cold TTFT over the pairs whose warm row
  **advertised nothing** (`audit_advertised_tokens` null or `0`), which is
  section 4's `A4 | supplied == 0` population read off the audit rather than
  off the outcome; positive is a tax. `miss_tax_pairs`,
  `miss_tax_pairs_without_ttft`, `miss_tax_pairs_advertising_excluded` and
  `miss_tax_pairs_negative_control_excluded` say who was in it,
  `miss_tax_definition` and `miss_tax_population` state the population in the
  document itself (on an `m4_capture` merge that population is the `no_reuse`
  pairs, and `miss_tax_pairs_outside_capture_class_excluded` counts what the
  class filter removed), `miss_tax_source` says whether it could be identified
  at all — with no connector audit joined to the warm arm, "advertised
  nothing" is unreadable, so every number in this block is `null` rather than
  the whole cold-warm delta relabelled as a tax — and
  `miss_tax_ms_median_of_differences` (+ `_ci`) is the paired form of the same
  question. `miss_tax_lookup_ms_per_lookup` is section 4's scheduler-thread
  leg (`miss_tax_lookup_latency_ms_sum` / `miss_tax_lookups_total`) when the
  arm's engine window carries those counters — `null`, never `0.0`, when it
  does not (`miss_tax_lookup_cost_source` says which), because a zero would
  subtract a cost nobody measured. The other two legs are
  cross-arm: run `merge-results --pair m4_capture` (A3 vs A4) and
  `--pair m4_instrumentation` (A5 vs A4) and read each document's
  `miss_tax_ms`.
- **M7 is cross-arm and lives in the `paired` block.**
  `propagation_contamination_rate` is the share of the **`propagation_probe`
  set** — the manifest's 50 items, not the probes that happened to be
  scoreable — whose treatment answer is closer to the *served* answer (their
  parent item's answer in the same arm) than to the *cold* answer (their own
  answer in the **A1** reference arm, see `--cold-reference` above). A probe
  that could not be scored is not evidence of no contamination, so it stays in
  the denominator and is named: `propagation_probes_excluded_unclean_pair`,
  `propagation_probes_without_served_answer`, `propagation_probes_unlinked`,
  `propagation_probes_without_answers`,
  `propagation_probes_without_cold_reference`,
  `propagation_probes_reference_position_mismatched`,
  `propagation_probes_absent_from_run`.
  `propagation_contamination_rate_vs_baseline_arm` is the same comparison
  against whatever this document calls its baseline — an arm-vs-arm
  diagnostic, deliberately under its own name, because on an `m7_propagation`
  document that is the number that under-reports.
  `propagation_contamination_rate_scored_only` (+
  `propagation_contamination_scored_denominator`) is the scored subset under
  its own name — read it to judge the headline, never in place of it. Read
  both beside `prefix_blocks_evicted` — until that counter reads non-zero on a
  contaminated workload, every lane-2 quality number is unproven, including a
  favourable one. The per-row
  `propagation_cached_without_materialization_rate` is a supporting signal,
  not M7.
- `connector_audit_present` / `connector_audit_rows_joined`, the two exclusion
  counters (`connector_audit_rows_excluded_cold_arm` /
  `..._excluded_not_joined` — every rate above is computed over rows that
  could have been audited, never over a cold arm), and a
  `config.connector_audit_join` report counting rows matched exactly, matched
  after normalization, unmatched, ambiguous, and without an id at all.

Without `--connector-audit` every one of those keys is `null`, not `0.0`: a run
that never looked must not publish a clean score.

## Backend Audit Summaries

```bash
python -m sembench summarize-engine-events \
  --engine vllm \
  --input results/vllm-backend.log \
  --output results/vllm-engine-events.json

python -m sembench summarize-engine-events \
  --engine trtllm \
  --input results/trtllm-audit.jsonl \
  --output results/trtllm-engine-events.json

python -m sembench summarize-engine-events \
  --engine sglang \
  --input results/sglang-backend.log \
  --output results/sglang-engine-events.json
```

For Kubernetes-hosted engines:

```bash
python -m sembench collect-k8s-engine-events \
  --engine sglang \
  --namespace inference \
  --pod sglang-0 \
  --since-time 2026-06-22T18:00:00Z \
  --output-log results/sglang.log \
  --output-summary results/sglang-engine-events.json
```

## Acceptance Gates

```bash
python -m sembench assert-result-gates \
  --result results/longbench-v1-gateway.json \
  --engine-summary results/sglang-engine-events.json \
  --min-quality-pass-rate 0.80 \
  --min-backend-confirmed-block-rate 0.05 \
  --min-materialization-events 1 \
  --min-materialized-tokens 512 \
  --max-negative-control-confirmed-rate 0.0 \
  --min-blended-ttft-speedup 2.0 \
  --max-negative-control-speedup-deviation 0.15 \
  --require-contamination-check \
  --require-materialized-reuse \
  --require-no-engine-errors
```

A requested gate whose metric is absent from the result **fails**; skipping it
would read as a pass. `--allow-missing` opts back into the skip, on the record.
`--min-blended-ttft-speedup` gates `blended_ttft_speedup_median`, not the mean.

## Main Metrics

See [docs/METRICS.md](docs/METRICS.md) for the exact metric contract.

- `exact_block_hit_rate`: full recipient blocks found by exact token hash in
  the donor pool.
- `semantic_candidate_block_rate`: full recipient blocks proposed for semantic
  donor reuse.
- `semantic_eligible_block_rate`: candidate blocks that are aligned enough for
  backend materialization.
- `backend_confirmed_block_rate`: live-backend-confirmed block reuse.
- `semantic_eligible_lift`: semantic eligible rate minus exact rate.
- `backend_confirmed_lift`: live confirmed rate minus exact rate.
- `semantic_placement_rate_by_request`: fraction of replayed requests routed by
  semantic placement when route metadata is available.
- `negative_control_backend_confirmed_rate`: confirmed reuse on unrelated
  donor/recipient pairs.
- `external_confirmed_tokens`: reuse that came from the external KV connector
  only, with the local prefix cache excluded. This, not `cached_tokens`, is the
  semantic-reuse signal. `None` means the split was never measured; it does not
  mean zero. `external_confirmed_tokens_source` says whether the number is
  per-request (`connector_audit`) or an arm-level aggregate
  (`arm_prometheus_delta`), which supports arm-level statements only.
- `engine_ttft_ms` / `queue_time_ms`: engine-side TTFT measured from scheduling
  (queue wait excluded) and the queue wait itself, from vLLM's
  `--enable-per-request-metrics`. Under concurrency these are the comparable
  numbers; client TTFT is mostly queue.
- `latency_ms`: **service** latency — the server time the item cost, summed
  over its donor requests and its recipient request. Since round 2 it is no
  longer the item's wall span at any concurrency, serial included, because a
  wall span also contains the dispatcher's own donor-gap and settle waits,
  which are the harness's delay and not latency the engine produced. TTFT is
  unaffected; it is still measured at the streamed first token.
- `alignment_given_match` / `alignment_given_opportunity` (alias:
  `boundary_alignment_rate`) / `boundary_miss_breakdown` /
  `span_decline_breakdown`: M1. `null` until the result is joined against a
  connector audit (`--connector-audit`). The two rates share one numerator;
  `alignment_given_opportunity_numerator_outside_classes` is why it can exceed
  the two-class denominator, and
  `alignment_given_opportunity_denominator_source` says whether the
  denominator is the manifest's item count or the rows present.
- `materialized_reuse_rate` (token-weighted; alias
  `materialized_reuse_token_rate`), `materialized_reuse_request_rate` and
  `materialized_reuse_superseded_tokens`: M2 — the mass that arrived, the
  requests that got any of it, and the mass written against a promise the
  connector had already superseded.
- `miss_tax_ms` (+ `miss_tax_pairs`, `miss_tax_warm_ttft_p50_ms` /
  `miss_tax_cold_ttft_p50_ms`, `miss_tax_lookup_ms_per_lookup`) in the
  `paired` block: M4, the tax the connector charges on requests it could not
  serve.
- `propagation_contamination_rate` (over the manifest's probe set, +
  `propagation_probes_excluded_unclean_pair`,
  `propagation_probes_without_served_answer` and the other exclusion counts,
  with `propagation_contamination_rate_scored_only` beside it) in the `paired`
  block, and `prefix_blocks_evicted` beside them: M7 and the counter that
  gates lane-2 quality. `propagation_cached_without_materialization_rate` is
  the supporting per-row signal.
- Per-row audit fields: `audit_semantic_lookup_hit`,
  `audit_lookup_hit_boundary` (the hit's own `already_computed_tokens` — that
  event has no `boundary` key), `audit_lookup_reusable_tokens`,
  `audit_advertised_tokens`, `audit_span_decline_event` /
  `audit_span_decline_reason` / `audit_span_declined_at`,
  `audit_superseded_materialized_tokens`, `audit_prefix_blocks_evicted`.
- `alignment_given_opportunity_rows_present` (+
  `alignment_given_opportunity_rows_present_denominator`): M1's shared
  numerator over the opportunity rows this document holds, beside the
  manifest-denominated headline.
- `quality_by_rope_delta_bucket`: M6's quality split by the manifest's
  `rope_delta_bucket` (0 / 128 / 512 / 2048). A quality result gathered only
  at |delta| ~ 0 does not transfer, and a blended mean over a stream that is
  93% |delta| ~ 0 cannot show damage that appears at 2048. `null` when no row
  declares a bucket.
- `pairs_stream_position_mismatched` / `pairs_unpaired` in the `paired` block:
  twins the two arms replayed at different manifest stream positions (they did
  not replay the same stream, so their ratio measures the reorder) and cold
  rows with no warm twin. Both were previously dropped without a count.
- `blended_ttft_speedup_median` (+ `_ci`): the headline paired speedup and the
  number `--min-blended-ttft-speedup` gates. Speedups are ratios and ratios are
  heavy-tailed, so one stalled cold arm can carry a mean over a bar the typical
  pair never reached. `blended_ttft_speedup_mean` is retained as a secondary
  field and is not gateable.
- Throughput document (written whenever `--concurrency > 1`, or whenever
  `--throughput-output` is given):
  `requests_per_second`, `requests_per_second_excluding_settle`,
  `output_tokens_per_second`, and p50/p90/p99 donor and recipient TTFT.

Semantic discovery is not counted as confirmed KV reuse unless a live backend
or backend audit stream reports materialization or reuse.

`cached_tokens` is local prefix cache plus external transfer summed together.
Only `external_confirmed_tokens` (external connector) and
`fuzzy_confirmed_tokens` (SGLang fuzzy-admitted mass) are semantic reuse.

Two figures in the result still read `cached_tokens`. They are named here
rather than disclaimed away, because a disclaimer that does not match the code
is how a prefix-cache repeat gets published as semantic reuse:

- **`hit_rate` is the legacy figure and must not be quoted for a vLLM run with
  prefix caching on.** A pair that carried no semantic signal at all falls back
  to `backend_confirmed_tokens >= 64`, and on such a run that threshold is
  cleared by an ordinary repeated document. The strict key is
  `hit_rate_external_confirmed`: it counts only pairs whose external split was
  actually measured, and it is `null` — never `0.0` — until the connector audit
  populates that split, so a gate that refuses `null` fails instead of passing
  on prefix-cache mass. `pairs_external_confirmed` /
  `pairs_external_unconfirmed` / `hits_unverified_external` say how many pairs
  fell on each side.
- **`--min-backend-confirmed-block-rate` gates `backend_confirmed_block_rate`,
  which is `cached_tokens` in block form.** It is a coverage check on the
  engine, not a semantic-reuse gate. The semantic figure is
  `materialized_reuse_rate` from the connector audit join above, and nothing
  else in this suite can stand in for it.
