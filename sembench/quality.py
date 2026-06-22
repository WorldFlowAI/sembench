"""Lightweight answer-quality scoring for live benchmark probes."""

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
