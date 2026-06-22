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
