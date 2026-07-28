# Baseline Matrix

Every row is measured with the SAME paired protocol, frozen manifest (by
checksum), calibration-relative gates, and the same confirmation standard:
reuse counts only when the ENGINE reports it (client-side discovery never
counts). Baselines get the same rigor as the subject under test — a
comparison table is only as honest as its weakest row.

## Rows

| baseline_id | What it measures | Recipe | Status |
|---|---|---|---|
| `cold` | No reuse (prefix cache off / flushed) | any engine, flush per item | implicit in every paired run's cold arm |
| `sglang-stock-prefix-cache` | Engine exact-prefix cache | stock lmsysorg/sglang | **measured** (EXP-0001 paired smoke: warm-vs-cold ROUGE exactly 1.0 on full-prefix hits, 3.23x mean TTFT on exact_repeat; partial-prefix hits are NOT output-invariant — see CALIBRATION.md) |
| `sglang-fuzzy-pr31057-defaults` | Semantic single-run backend (PR #31057) | trunk overlay recipe (EXP-0002) | **measured**: ROUGE 0.573, TTFT 0.995x, ~1% realization |
| `sglang-multiseg-june` | Semantic multi-segment (June fork) | ECR semblend-fuzzy-multiseg (EXP-0004) | **measured**: ROUGE 0.645 @ ~95% realization; TTFT invalid (sync hot-path) |
| `lmcache-exact` | Exact chunk reuse via LMCache | vLLM + LMCache pod (below) | recipe drafted; run pending |
| `cacheblend-class` | Fuzzy blend via LMCache blend mode | version-pinned; WorldFlowAI LMCache forks (PRs #2803/#2804 era) as fallback — DISCLOSE fork use in any published row | pending |
| `semblend-multiseg-gated` | G5 recipe (trunk + gated emission) | EXP-0005 staging | staged |

## Confirmation standard per row

- SGLang rows: `sembench collect-k8s-engine-events --engine sglang` (fuzzy/segment log families).
- LMCache rows: `--engine lmcache` parser (retrieved/stored token families;
  version-pinned fixtures in tests). LMCache is an EXACT-reuse baseline:
  its `materialized_semantic_kv_reuse` is False by definition — retrieved
  tokens count as engine-confirmed reuse, not as semantic reuse.
- vLLM/TRT-LLM rows: existing parsers (SemBlend connector logs / audit JSONL).

## vLLM + LMCache pod recipe (draft)

Version pins (update at run time, record in RunMetadata.engine_version):
- vLLM: pin the exact version the LMCache release supports (check LMCache
  compatibility table at run time).
- LMCache: pin the release; if blend mode requires our fork, use the
  WorldFlowAI fork at the #2803/#2804-era commits and mark the row.

Pod shape (deploy/lmcache-baseline-pod.yaml, to be finalized at run time):
vLLM OpenAI server + `LMCACHE_CONFIG_FILE` mounting a config with local CPU
backend; `--kv-transfer-config` pointing at the LMCache connector. Replay via
`sembench run-live-gateway` (OpenAI-compatible) with `--backend-id
lmcache-exact`; collect pod logs with `--engine lmcache`.
Known operational hazard: LMCache cudaHostAlloc OOM cap on large host caches
(house skill) — size `max_local_cache_size` below host RAM headroom.

## Out-of-scope rows (documented honestly)

- SemShareKV: HF-transformers prototype, no serving engine → cannot produce
  engine-confirmed reuse under this standard; cited in docs, not tabled.
- C²KV: learned-sidecar approach; no public code at time of writing —
  positioning axis (zero-shot vs learned), not a reproducible row.
