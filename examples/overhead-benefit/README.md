# Overhead and benefit on one workload

Manifests and runners for measuring what a semantic KV reuse layer costs
when it cannot help and what it returns when it can, on the same
hardware, engine, and prompts.

## Manifests

- `build_manifests.py`: ~3.5K-token items (`hit.jsonl`, `overhead.jsonl`,
  `mixed.jsonl`).
- `build_long_manifests.py`: 8K, 16K and 24K-token items. Per length:
  48 verified-paraphrase pairs (`amz-hit-<L>.jsonl`), 24 wrapper-shift
  pairs (`amz-shift-<L>.jsonl`, identical evidence under a different
  instruction wrapper), 24 unrelated recipients (`amz-overhead-<L>.jsonl`).
  Sentence counts are calibrated with the Qwen2.5 tokenizer.

Every item carries a donor prompt and a recipient prompt. Facts are
identical within an item and disjoint across items, so a served recipient
that names another item's service is a leak, and the summarizer checks
for it.

## Arms

For each length, TTFT is measured one request at a time and paired per
item against a cold run of the same recipient:

```
sembench run-live-gateway --manifest amz-hit-16000.jsonl --output baseline_hit_16000.jsonl \
  --gateway-url http://127.0.0.1:30001 --model Qwen/Qwen2.5-7B-Instruct --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --recipient-max-tokens 32 --arm single --post-donor-delay-ms 2000
```

Run once with the engine plain and once with the reuse layer on. The
zero-match manifest with the layer on is the miss tax: every lookup
misses and every donor is captured.

Throughput under load uses the same manifests with donor->recipient
streams in flight:

```
sembench run-load --manifest amz-hit-16000.jsonl --output tp_conn_hit_16000.json \
  --gateway-url http://127.0.0.1:30001 --model Qwen/Qwen2.5-7B-Instruct --concurrency 4
```

## Engine settings that matter at these lengths (A10G, 7B fp16)

- vLLM: one prefill chunk must cover the longest donor
  (`--max-num-batched-tokens 24576`), `--max-model-len 25600` so a 24K
  prompt plus its output fits, `--gpu-memory-utilization 0.90`.
- SGLang: `--chunked-prefill-size 8192 --context-length 25600
  --mem-fraction-static 0.80`, with `SEMBLEND_EMBED_DEVICE=cuda:0` and
  `SEMBLEND_EMBED_MAX_CHARS=200000` in the server environment so donors
  and queries are matched on their full text (the default truncation is
  tuned for CPU embedding and costs same-document hits on multi-passage
  prompts).
