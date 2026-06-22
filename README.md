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
  --require-materialized-reuse \
  --require-no-engine-errors
```

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

Semantic discovery is not counted as confirmed KV reuse unless a live backend
or backend audit stream reports materialization or reuse.
