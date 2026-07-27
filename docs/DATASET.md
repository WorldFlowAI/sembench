# SemBench Dataset v1 — Design

Status: DESIGN (no frozen build yet). This document specifies `sembench-corpus-v1`
before any data is generated, so the freeze is a deliberate act rather than an
accident of whatever the builder produced first.

## Goals

A versioned, reproducible workload set that measures **semantic (non-exact-prefix)
KV cache reuse** with the same rigor the metric contract already enforces:
discovery is never reuse, negative controls must stay cold, and every number is
traceable to a manifest checksum (`sembench checksum-manifest`, result schema v2).

## Composition

| Component | Items (target) | Source | License posture |
|---|---|---|---|
| LongBench-derived | ~1,200 | 10 LongBench v1 subsets × 8 transforms × ~15 sources | Builder-first: never redistributed; rebuilt from a pinned HF revision |
| Synthetic enterprise | ~240 | Original templates: support-ticket RAG, contract review, log triage | Ours (Apache-2.0), redistributable |
| Smoke split | 24 | Deterministic fixtures | Committed in-repo |

Context lengths 2k–16k tokens (tokenizer pinned: Qwen/Qwen2.5-7B-Instruct), with
the 8k+ band explicitly oversampled: fuzzy-match benefit is only realized past
~8k context, so the benchmark must resolve the benefit-vs-context curve rather
than average over it.

## Splits

`smoke` (24, committed) / `dev` (20%) / `test` (80%) / `live` (~240 stratified
subset sized for GPU replay). Splitting is **by `source_id`** so no document's
donor can leak into another split's recipients.

## Freezing model (builder-first)

The frozen artifact is not the data; it is the recipe plus checksums:

1. Pinned HF dataset revision (exact commit of THUDM/LongBench).
2. Deterministic transform code + committed seeds (build is byte-stable; CI
   enforces `cmp` equality of consecutive builds).
3. Committed `manifests/CHECKSUMS.v1.json`: SHA256 per manifest.

`sembench build --frozen v1` reproduces the corpus; `sembench verify-frozen`
(P1 deliverable) rebuilds and compares against the committed checksums. This
carries zero redistribution risk for gray-license subsets (multi_news, lcc,
repobench-p). A HuggingFace upload is drafted only for clean-license subsets
(CC-BY/MIT/US-gov + synthetic) and gated on license review.

## Collision audit (phantom-match guard)

Synthetic and transformed workloads can accidentally share token blocks across
unrelated items, producing phantom exact-prefix hits that masquerade as
semantic reuse. The frozen build must pass an audit: pairwise exact-block-hash
overlap between (a) any negative-control pair and (b) any cross-item
donor/recipient pair that is not deliberately related, must be zero. Any
violation fails the build.

## Per-transform coverage

Each of the 8 transforms (exact_repeat, instruction_variant,
same_evidence_new_task, leading_evidence_new_task, rag_reorder,
multi_donor_composite, fuzzy_edit, negative_control) targets ≥100 items in
`test` so per-transform metrics carry their own confidence intervals; the
`live` split preserves stratification.

## Positioning notes (honesty constraints)

- vCache publishes HF datasets named "SemBenchmark" — prompt-level semantic
  *response* caching, no KV tensors, no engines. Cite and differentiate.
- SCBench covers exact shared-context KV reuse; the chunk-level caching study
  (arXiv:2603.20218) compares reuse strategies empirically. Neither requires
  engine-reported materialization nor covers semantic (fuzzy) matching — that
  evidentiary bar is this benchmark's distinguishing contract.
- Candidate future transform family from arXiv:2601.08343 (multi-agent
  cross-candidate interaction failures): workloads where reuse SHOULD degrade
  quality, as gate-calibration cases.

## Open questions for review

1. Second tokenizer (Llama-3.1-8B) at v1, or defer to defensibility phase?
2. Synthetic-template count: 3 domains × 80 items vs 6 × 40?
3. Should `live` oversample 8k+ even harder (benefit threshold) at the cost of
   short-context coverage?
