"""Answer-quality scoring for live benchmark probes.

Three layered metrics, all dependency-free:
- ``quality_score``: legacy answer-token recall (kept for continuity).
- ``token_f1``: SQuAD-style normalized token F1 against the best reference.
- ``rouge_l``: LCS-based ROUGE-L F1 — the house warm-vs-cold quality metric
  (noise floor ~0.845 for identical reruns on Qwen2.5-7B; always interpret
  against a measured calibration, never as an absolute).
"""

from __future__ import annotations


def quality_score(output_text: str, answers: list[str]) -> float | None:
    """Return answer-token recall against the best reference answer."""
    if not answers:
        return None
    output_tokens = _quality_tokens(output_text)
    if not output_tokens:
        return 0.0
    return max(
        (_answer_recall(output_tokens, _quality_tokens(answer)) for answer in answers if answer),
        default=0.0,
    )


def token_f1(output_text: str, answers: list[str]) -> float | None:
    """SQuAD-style token F1 against the best reference answer."""
    if not answers:
        return None
    prediction = _normalize(output_text).split()
    best = 0.0
    for answer in answers:
        if not answer:
            continue
        best = max(best, _f1(prediction, _normalize(answer).split()))
    return best


def rouge_l(output_text: str, reference_text: str) -> float:
    """ROUGE-L F1 via longest common subsequence over normalized tokens."""
    hyp = _normalize(output_text).split()
    ref = _normalize(reference_text).split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l_best(output_text: str, answers: list[str]) -> float | None:
    """ROUGE-L F1 against the best reference answer."""
    if not answers:
        return None
    return max((rouge_l(output_text, answer) for answer in answers if answer), default=0.0)


def _f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction or not reference:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in reference:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in prediction:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    # Two-row DP keeps memory linear in the shorter side.
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _quality_tokens(text: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "as",
        "in",
        "of",
        "the",
        "to",
        "was",
        "were",
    }
    return [
        _stem_quality_token(token)
        for token in _normalize(text).replace(";", " ").replace(".", " ").split()
        if token not in stop
    ]


def _stem_quality_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) >= 3 and token[-1] == token[-2]:
        token = token[:-1]
    return token


def _answer_recall(output_tokens: list[str], answer_tokens: list[str]) -> float:
    if not answer_tokens:
        return 0.0
    output_counts: dict[str, int] = {}
    for token in output_tokens:
        output_counts[token] = output_counts.get(token, 0) + 1
    overlap = 0
    for token in answer_tokens:
        if output_counts.get(token, 0) > 0:
            overlap += 1
            output_counts[token] -= 1
    if overlap == 0:
        return 0.0
    return overlap / len(answer_tokens)


def _normalize(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text)
