"""Long-context manifests for the Amazon packet (target band: 6-8K+).

For each context length L in {8K, 16K, 24K} tokens:
  hit-L.jsonl      48 verified-paraphrase pairs (facts identical per item,
                   disjoint across items, every sentence reworded/reordered)
  shift-L.jsonl    24 wrapper-shift pairs (identical evidence block under a
                   different instruction wrapper; interior-span lane)
  overhead-L.jsonl 24 unrelated recipients (zero-match traffic)
Sentence counts are calibrated with the Qwen2.5 tokenizer so prompts land
near the target length.
"""

import json
import sys

from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

SERVICES = [
    "billing-eu", "search-idx", "auth-gate", "media-enc", "geo-route", "ml-feat",
    "pay-clear", "doc-ocr", "cart-sync", "inv-ledger", "mail-relay", "cdn-edge",
    "fraud-score", "kyc-verify", "push-fanout", "log-ingest", "trace-agg", "quota-mgr",
    "sess-store", "img-resize", "audio-mix", "map-tiles", "rate-limit", "oauth-brk",
    "feed-rank", "ads-bid", "chat-relay", "vid-cdn", "pdf-render", "sms-gw",
    "etl-batch", "warehouse", "recsys-v2", "search-sug", "cache-tier", "dns-edge",
    "tls-term", "queue-mgr", "cron-sched", "backup-agt", "metrics-db", "alert-hub",
    "cost-meter", "tenant-api", "policy-eng", "audit-log", "kms-proxy", "vault-sync",
]
UNITS = [
    ("invoice pipeline", "ledger shard"), ("index rebuild", "crawler pool"),
    ("token refresh", "session vault"), ("transcode queue", "codec farm"),
    ("dispatch mesh", "region planner"), ("feature store", "embedding cache"),
    ("settlement batch", "clearing house"), ("scan pipeline", "layout parser"),
]
TOPICS = [
    "the migration of a coral reef nursery", "a bakery's sourdough schedule",
    "the tuning of a cathedral organ", "a vineyard's harvest calendar",
    "the restoration of a wooden sailboat", "a library's rare book audit",
    "the choreography of a marching band", "a glacier survey expedition",
]

DONOR_FORMS = [
    "service {svc} recorded {reqs} requests in window {i} while the {a} kept latency at {lat} ms across the {b}.",
    "in window {i} the {a} of service {svc} processed {reqs} requests and the {b} reported {lat} ms latency.",
    "window {i}: {svc} saw {reqs} requests, the {a} held steady, and latency on the {b} was {lat} ms.",
]
PARA_FORMS = [
    "during window {i} the {svc} service held latency to {lat} ms across the {b} as the {a} logged {reqs} requests.",
    "the {b} for {svc} measured {lat} ms of latency in window {i} while {reqs} requests moved through the {a}.",
    "for {svc}, window {i} brought {reqs} requests via the {a}, with the {b} sitting at {lat} ms latency.",
]
HEADER = "you are a triage assistant. review the following service report.\n"
SHIFT_HEADER = (
    "SESSION[reopened] operator=oncall-2 task=latency-review\n"
    "inspect the report below and list the windows whose latency exceeded the median.\n"
)


def sentences(svc, a, b, base, n, forms):
    out = []
    for i in range(n):
        reqs = 200 + base + (i % 40)
        lat = 90 + base + (i % 30)
        out.append(forms[i % len(forms)].format(svc=svc, a=a, b=b, i=i, reqs=reqs, lat=lat))
    return " ".join(out)


def unrelated(idx, n):
    topic = TOPICS[idx % len(TOPICS)]
    body = " ".join(
        f"note {i}: on day {idx * 900 + i} the team observed step {i} of {topic} and logged {7 * i + idx} remarks."
        for i in range(n)
    )
    return "you are a careful chronicler. read the following field notes.\n" + body + f"\ndescribe the progress of {topic}."


def calibrate(target_tokens, make):
    """Find the sentence count whose prompt lands within ~3% of target."""
    lo, hi = 10, 4000
    while lo < hi:
        mid = (lo + hi) // 2
        n = len(TOK.encode(make(mid)))
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return lo


def item(item_id, svc, donor, recipient, transform, negative, length):
    return {
        "item_id": item_id, "dataset": f"amz-long-{length}", "source_id": svc, "transform": transform,
        "donor_prompts": [{"donor_id": f"donor-{svc}-{length}", "text": donor, "label": transform, "metadata": {}}],
        "recipient_prompt": recipient, "input": "", "answers": [], "negative_control": negative,
        "metadata": {"lane": transform, "target_tokens": length}, "benchmark_version": "1",
    }


def main(out_dir):
    for length in (8000, 16000, 24000):
        svc0, (a0, b0) = SERVICES[0], UNITS[0]
        n_sent = calibrate(length, lambda n: HEADER + sentences(svc0, a0, b0, 1000, n, DONOR_FORMS) + "\nsummarize the health of x.")
        n_unrel = calibrate(length, lambda n: unrelated(0, n))
        hit, shift, ovh = [], [], []
        for idx, svc in enumerate(SERVICES):
            a, b = UNITS[idx % len(UNITS)]
            base = 1000 * (idx + 1)
            donor = HEADER + sentences(svc, a, b, base, n_sent, DONOR_FORMS) + f"\nsummarize the health of {svc}."
            para = HEADER + sentences(svc, a, b, base, n_sent, PARA_FORMS) + f"\nsummarize the health of {svc}."
            hit.append(item(f"hit-{length}-{idx:02d}", svc, donor, para, "verified_paraphrase", False, length))
            if idx < 24:
                evidence = sentences(svc, a, b, base, n_sent, DONOR_FORMS)
                shift.append(item(
                    f"shift-{length}-{idx:02d}", svc,
                    HEADER + evidence + f"\nsummarize the health of {svc}.",
                    SHIFT_HEADER + evidence + "\nwhich windows exceeded the median latency?",
                    "wrapper_shift", False, length,
                ))
                ovh.append(item(f"ovh-{length}-{idx:02d}", svc, donor, unrelated(idx, n_unrel), "unrelated", True, length))
        for name, items in (("hit", hit), ("shift", shift), ("overhead", ovh)):
            path = f"{out_dir}/amz-{name}-{length}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            sample = items[0]
            print(f"{path}: {len(items)} items, donor={len(TOK.encode(sample['donor_prompts'][0]['text']))} tok, recipient={len(TOK.encode(sample['recipient_prompt']))} tok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
