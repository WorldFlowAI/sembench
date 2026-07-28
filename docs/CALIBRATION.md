# Noise-Floor Calibration

Quality gates in SemBench are **calibration-relative**: a warm-vs-cold quality
gate cannot run without a measured noise-floor artifact
(`sembench calibrate-noise-floor`), and results implausibly above the
self-agreement band fail rather than pass. This document records why, and the
first measured artifact.

## The floor is a property of the serving protocol, not the model

Measured 2026-07-28 (EXP-0001): Qwen/Qwen2.5-7B-Instruct on 1× A10G, stock
SGLang 0.5.16, frozen `synthetic-v1` manifest, n=149 cold/cold pairs
(1 contaminated pair excluded), `max_new_tokens=64`, temperature 0,
sequential requests, cache flushed between reruns:

| metric | value |
|---|---|
| ROUGE-L self-agreement mean | **1.0000** |
| p05 / p50 | 1.0000 / 1.0000 |
| per length band (2k/4k/8k/12k/16k) | all 1.0000 |

Outputs are **byte-identical** across reruns (verified independently of the
harness). Under this protocol the engine is deterministic, so **any**
warm-vs-cold divergence in a paired run is attributable to the cache-reuse
mechanism under test — attribution is total.

The frequently quoted "~0.845 identical-rerun ROUGE-L floor" for this model
class is a property of **concurrent/batched serving** (nondeterministic
reduction order under changing batch composition) and longer generations. It
is real, but it belongs to a different protocol.

## Rules

1. Calibrate under the SAME protocol your paired runs use (sequential
   harness runs → sequential calibration; concurrency experiments → measure a
   concurrent floor first).
2. Gates: `assert-result-gates --noise-floor-calibration <artifact>
   --max-warm-vs-cold-rouge-drop <eps>`. Absolute quality thresholds are
   refused by design.
3. A lossless baseline check comes free: a stock engine's exact-prefix cache
   must measure warm-vs-cold ROUGE-L = 1.0 under the sequential protocol
   (EXP-0001 paired smoke: 20/20 pairs, exactly 1.0). If your harness reports
   otherwise, the harness is broken.
4. Artifact provenance: calibration artifacts embed model, engine, engine
   version, n, contaminated/errored counts. Leaderboard rows must reference
   the artifact used.

## Follow-ups

- Measure and publish the CONCURRENT floor (production-realistic) as a second
  artifact class; benchmark reports should state which floor applies.
- Second model family (Llama-3.1-8B) floors in the defensibility phase.
