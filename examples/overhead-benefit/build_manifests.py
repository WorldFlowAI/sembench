"""Manifests for the vLLM overhead-vs-benefit benchmark (Amazon packet).

Three manifests over the same 24 donor/recipient topics:
  hit.jsonl      - recipients are verified paraphrases of their donors
  overhead.jsonl - recipients are unrelated to every donor (zero-match
                   traffic: measures lookup + capture cost, no benefit)
  mixed.jsonl    - 25% paraphrase / 75% unrelated, the shape of a real
                   fleet where most traffic never matches
Each item's fact set is numerically disjoint so the lexical gate can
never accept a cross-item donor.
"""

import json
import random
import sys

SERVICES = [
    "billing-eu", "search-idx", "auth-gate", "media-enc", "geo-route",
    "ml-feat", "pay-clear", "doc-ocr", "cart-sync", "inv-ledger",
    "mail-relay", "cdn-edge", "fraud-score", "kyc-verify", "push-fanout",
    "log-ingest", "trace-agg", "quota-mgr", "sess-store", "img-resize",
    "audio-mix", "map-tiles", "rate-limit", "oauth-brk",
]
UNITS = [
    ("invoice pipeline", "ledger shard"), ("index rebuild", "crawler pool"),
    ("token refresh", "session vault"), ("transcode queue", "codec farm"),
    ("dispatch mesh", "region planner"), ("feature store", "embedding cache"),
    ("settlement batch", "clearing house"), ("scan pipeline", "layout parser"),
]
UNRELATED_TOPICS = [
    "the migration of a coral reef nursery", "a bakery's sourdough schedule",
    "the tuning of a cathedral organ", "a vineyard's harvest calendar",
    "the restoration of a wooden sailboat", "a library's rare book audit",
    "the choreography of a marching band", "a glacier survey expedition",
]


def paraphrase_pair(svc, unit_a, unit_b, base, n=100):
    donor_sents, target_sents = [], []
    for i in range(n):
        reqs = 200 + base + (i % 40)
        lat = 90 + base + (i % 30)
        donor_sents.append(
            f"service {svc} recorded {reqs} requests in window {i} while the "
            f"{unit_a} kept latency at {lat} ms across the {unit_b}."
        )
        target_sents.append(
            f"during window {i} the {svc} service held latency to {lat} ms "
            f"across the {unit_b} as the {unit_a} logged {reqs} requests."
        )
    header = "you are a triage assistant. review the following service report.\n"
    footer = f"\nsummarize the health of {svc}."
    return header + " ".join(donor_sents) + footer, header + " ".join(target_sents) + footer


def unrelated_prompt(idx, n=100):
    topic = UNRELATED_TOPICS[idx % len(UNRELATED_TOPICS)]
    sents = [
        f"note {i}: on day {idx * 300 + i} the team observed step {i} of "
        f"{topic} and logged {7 * i + idx} remarks about it."
        for i in range(n)
    ]
    return (
        "you are a careful chronicler. read the following field notes.\n"
        + " ".join(sents)
        + f"\ndescribe the progress of {topic}."
    )


def item(item_id, svc, donor, recipient, transform, negative):
    return {
        "item_id": item_id,
        "dataset": "amz-vllm-overhead-benefit",
        "source_id": svc,
        "transform": transform,
        "donor_prompts": [
            {"donor_id": f"donor-{svc}", "text": donor, "label": transform, "metadata": {}}
        ],
        "recipient_prompt": recipient,
        "input": "",
        "answers": [],
        "negative_control": negative,
        "metadata": {"lane": transform},
        "benchmark_version": "1",
    }


def main(out_dir):
    random.seed(11)
    hit, overhead, mixed = [], [], []
    for idx, svc in enumerate(SERVICES):
        unit_a, unit_b = UNITS[idx % len(UNITS)]
        donor, target = paraphrase_pair(svc, unit_a, unit_b, base=1000 * (idx + 1))
        unrelated = unrelated_prompt(idx)
        hit.append(item(f"hit-{idx:02d}", svc, donor, target, "verified_paraphrase", False))
        overhead.append(item(f"ovh-{idx:02d}", svc, donor, unrelated, "unrelated", True))
        if idx % 4 == 0:
            mixed.append(item(f"mix-{idx:02d}", svc, donor, target, "verified_paraphrase", False))
        else:
            mixed.append(item(f"mix-{idx:02d}", svc, donor, unrelated, "unrelated", True))
    for name, items in (("hit", hit), ("overhead", overhead), ("mixed", mixed)):
        path = f"{out_dir}/amz-{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"wrote {len(items)} items to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
