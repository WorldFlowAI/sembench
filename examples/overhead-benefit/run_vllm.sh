#!/bin/bash
# Amazon packet benchmark, vLLM engine (runs inside the GPU pod).
# Five runs over shared manifests, TTFT-streamed by sembench in-pod:
#   baseline_hit  no connector, paraphrase manifest   (cold reference)
#   baseline_ovh  no connector, unrelated manifest    (cold reference)
#   conn_ovh      connector on, unrelated manifest    (pure overhead: 0 hits)
#   conn_hit      connector on, paraphrase manifest   (benefit)
#   conn_mixed    connector on, 25% paraphrase mix    (fleet-shaped)
set -ex
MODEL=Qwen/Qwen2.5-7B-Instruct
SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
mkdir -p /work-vllm && tar xzf /work-vllm-src.tgz -C /work-vllm
cp -R /work-vllm/vllm/* "$SITE/vllm/"
python3 -c "import vllm.v1.core.sched.scheduler as s; assert 'supports_mid_request_matching' in open(s.__file__).read(); print('OVERLAY-OK')"
pip install --no-cache-dir --no-deps /connector.tar.gz /semblend.tar.gz /sembench.tar.gz \
  sentence-transformers rapidfuzz scikit-learn scipy joblib threadpoolctl narwhals aiohttp
python3 -c "
import numpy as np
from semblend_core.embedder import create_embedder
v = create_embedder('minilm').embed('liveness'); assert v is not None and len(np.asarray(v)) == 384
import sembench.gateway_live; print('CHAIN-OK embedder-live sembench-ok')"
# Pre-warm the tokenizer and model download once.
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL')"

make_cfg() {  # $1 = arm name
  cat > /tmp/kvcfg.json << JSON
{
  "kv_connector": "SemBlendVllmConnector",
  "kv_connector_module_path": "semblend_vllm_connector.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "mode": "semantic_span_experimental",
    "provider": "semblend",
    "model_id": "$MODEL",
    "min_prompt_tokens": 256,
    "min_similarity": 0.7,
    "min_reuse_ratio": 0.5,
    "min_semantic_span": 512,
    "embedder_type": "minilm",
    "lookup_top_k": 5,
    "register_donors": true,
    "enable_prompt_text": true,
    "log_decisions": true,
    "kv_storage_path": "/tmp/semblend-kv-$1",
    "audit_path": "/tmp/semblend-audit-$1.jsonl"
  }
}
JSON
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true; sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true; pkill -9 -f "EngineCore" 2>/dev/null || true
  for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${USED:-99999}" -lt 1500 ] && break; sleep 4
  done
}

serve() {  # $1 = arm name, $2 = "conn" | "plain"
  local extra=""
  [ "$2" = "conn" ] && { make_cfg "$1"; extra="--kv-transfer-config $(cat /tmp/kvcfg.json | tr -d '\n')"; }
  # shellcheck disable=SC2086
  nohup vllm serve $MODEL --port 30001 --gpu-memory-utilization 0.85 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 --max-model-len 8192 \
    --no-enable-prefix-caching $extra > "/tmp/vllm-serve-$1.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30001/health | grep -q 200 && break; sleep 10
  done
  curl -s -o /dev/null -w "healthcheck-$1:%{http_code}\n" http://127.0.0.1:30001/health
}

bench() {  # $1 = arm name, $2 = manifest
  python3 -m sembench run-live-gateway --manifest "/manifests/$2" --output "/results/$1.jsonl" \
    --gateway-url http://127.0.0.1:30001 --model "$MODEL" --tokenizer "$MODEL" \
    --recipient-max-tokens 32 --arm single --run-id "amz-vllm-$1" 2>&1 | tail -3
  python3 - "$1" << 'PY'
import json, statistics, sys
arm = sys.argv[1]
rows = []
for line in open(f"/results/{arm}.jsonl"):
    line = line.strip()
    if line.startswith("{") and '"item_id"' in line:
        try: rows.append(json.loads(line))
        except Exception: pass
tt = [r["ttft_ms"] for r in rows if r.get("ttft_ms")]
q = [r["quality_score"] for r in rows if r.get("quality_score") is not None]
print(f"ARM {arm} n={len(rows)} ttft_p50={statistics.median(tt) if tt else -1:.0f}ms "
      f"ttft_mean={statistics.mean(tt) if tt else -1:.0f}ms quality_mean={statistics.mean(q) if q else -1:.3f}")
PY
}

mkdir -p /results
export SEMBLEND_PARAPHRASE_SERVE=1 SEMBLEND_NLI_APPEAL=0 SEMBLEND_CHUNK_FAST_PATH=0

stop_server; serve baseline plain
bench baseline_hit amz-hit.jsonl
bench baseline_ovh amz-overhead.jsonl
stop_server

serve conn_ovh conn
bench conn_ovh amz-overhead.jsonl
ADV_OVH=$(grep -c semantic_span_load_advertised /tmp/semblend-audit-conn_ovh.jsonl || true)
stop_server

serve conn_hit conn
bench conn_hit amz-hit.jsonl
ADV_HIT=$(grep -c semantic_span_load_advertised /tmp/semblend-audit-conn_hit.jsonl || true)
MAT_HIT=$(grep -c '"event": "runtime_materialized"' /tmp/semblend-audit-conn_hit.jsonl || true)
stop_server

serve conn_mixed conn
bench conn_mixed amz-mixed.jsonl
MAT_MIX=$(grep -c '"event": "runtime_materialized"' /tmp/semblend-audit-conn_mixed.jsonl || true)
stop_server

cp /tmp/semblend-audit-*.jsonl /results/ 2>/dev/null || true
for f in /tmp/vllm-serve-*.log; do gzip -kf "$f"; cp "$f.gz" /results/; done
echo "AMZ-VLLM-VERDICT ovh_advertised=$ADV_OVH hit_advertised=$ADV_HIT hit_materialized=$MAT_HIT mixed_materialized=$MAT_MIX"
