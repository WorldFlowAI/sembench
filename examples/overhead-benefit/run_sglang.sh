#!/bin/bash
# Amazon packet benchmark, SGLang engine on the PR #31057 branch (in-pod).
# Same five runs as the vLLM script over the same manifests. The PR head's
# python tree is overlaid on the image (PYTHONPATH); if it cannot boot on
# today's image the exp0030-era PR tree (the mask-study commit) is used
# and reported as such.
set -ex
MODEL=Qwen/Qwen2.5-7B-Instruct
mkdir -p /work /results
tar xzf /pr-head.tgz -C /work
python3 -c "import importlib.metadata as m; print('image sglang', m.version('sglang'))"
pip install --no-cache-dir 'semblend==0.3.17' sentence-transformers aiohttp 2>&1 | tail -1
pip install --no-cache-dir --no-deps /sembench.tar.gz 2>&1 | tail -1
python3 -c "
import importlib.metadata as m, numpy as np
assert m.version('semblend') == '0.3.17'
from semblend_core.embedder import create_embedder
v = create_embedder('minilm').embed('liveness'); assert v is not None and len(np.asarray(v)) == 384
import sembench.gateway_live; print('CHAIN-OK semblend-0.3.17 embedder-live')"
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL')"

BASE_ENVS="SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 FLASHINFER_DISABLE_VERSION_CHECK=1 SEMBLEND_EMBEDDER=minilm SEMBLEND_MODEL_NAME=$MODEL"
TREE=pr-head

stop_server() {
  pkill -f "sglang[.]launch_server" 2>/dev/null || true; sleep 6
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null || true
  for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${USED:-99999}" -lt 1500 ] && break; sleep 4
  done
}

serve() {  # $1 = arm, $2 = "fuzzy" | "plain", $3 = extra envs
  local backend=""
  [ "$2" = "fuzzy" ] && backend="--radix-cache-backend fuzzy_match --fuzzy-model-arch qwen2.5-7b"
  # shellcheck disable=SC2086
  (cd /work && env $BASE_ENVS $3 PYTHONPATH=/work/python nohup python3 -m sglang.launch_server \
     --model-path $MODEL --host 0.0.0.0 --port 30000 --mem-fraction-static 0.75 \
     --chunked-prefill-size 4096 --context-length 8192 $backend > "/tmp/sglang-$1.log" 2>&1 &)
  UP=0
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health | grep -q 200 && { UP=1; break; }
    pgrep -f "sglang[.]launch_server" >/dev/null || break
    sleep 10
  done
  echo "boot-$1 up=$UP tree=$TREE"
  return $((1 - UP))
}

# Boot probe: PR head first, fall back to the exp0030-era PR tree.
stop_server
if ! serve probe plain ""; then
  tail -30 /tmp/sglang-probe.log || true
  stop_server
  rm -rf /work/python && tar xzf /pr-exp0030.tgz -C /work && TREE=pr-exp0030
  serve probe plain "" || { tail -40 /tmp/sglang-probe.log; echo "AMZ-SGLANG-VERDICT boot-failed"; exit 1; }
fi
stop_server

bench() {  # $1 = arm, $2 = manifest
  python3 -m sembench run-live-gateway --manifest "/manifests/$2" --output "/results/$1.jsonl" \
    --gateway-url http://127.0.0.1:30000 --model "$MODEL" --tokenizer "$MODEL" \
    --recipient-max-tokens 32 --arm single --run-id "amz-sglang-$1" 2>&1 | tail -3
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

serve baseline plain ""
bench baseline_hit amz-hit.jsonl
bench baseline_ovh amz-overhead.jsonl
stop_server

serve conn_ovh fuzzy "SEMBLEND_PARAPHRASE_SERVE=1"
bench conn_ovh amz-overhead.jsonl
HIT_OVH=$(grep -c "fuzzy match success" /tmp/sglang-conn_ovh.log || true)
stop_server

serve conn_hit fuzzy "SEMBLEND_PARAPHRASE_SERVE=1"
bench conn_hit amz-hit.jsonl
HIT_HIT=$(grep -c "fuzzy match success" /tmp/sglang-conn_hit.log || true)
REAL_HIT=$(grep -oE 'Realized [0-9]+ fuzzy tokens' /tmp/sglang-conn_hit.log | awk '{s+=$2} END {print s+0}')
stop_server

serve conn_mixed fuzzy "SEMBLEND_PARAPHRASE_SERVE=1"
bench conn_mixed amz-mixed.jsonl
HIT_MIX=$(grep -c "fuzzy match success" /tmp/sglang-conn_mixed.log || true)
stop_server

for f in /tmp/sglang-*.log; do gzip -kf "$f"; cp "$f.gz" /results/; done
echo "AMZ-SGLANG-VERDICT tree=$TREE ovh_hits=$HIT_OVH hit_hits=$HIT_HIT hit_realized_tokens=$REAL_HIT mixed_hits=$HIT_MIX"
