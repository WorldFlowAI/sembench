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
  --cold results/cold.json \
  --warm results/warm.json \
  --output results/paired.json
```

The join is on `item_id` and the manifest SHA256, never a heuristic. It refuses
to merge when the arms do not join one-to-one, and when either result's own
`run.arm` label contradicts the flag it was passed under — handing the warm run
to `--cold` inverts every speedup downstream and nothing in the merged document
would say so.

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
`fuzzy_confirmed_tokens` (SGLang fuzzy-admitted mass) are semantic reuse, and
neither the hit rate nor any gate is computed from `cached_tokens`.
