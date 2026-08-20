"""Deterministic fact-preserving paraphrase rewriting.

Rewrites prose densely enough that chunk-level token identity collapses,
while never touching the fact surface: digit tokens, their immediate
neighbors (magnitude words like "12 million" must survive as a unit), and
capitalized words anywhere but a swapped-in replacement's own casing.
Sentence-initial words never swap (they may host density fillers); swaps
apply to lowercase mid-sentence words only. Deterministic per
(text, seed_key) with no global RNG state, so frozen manifests stay
byte-stable.
"""

from __future__ import annotations

import hashlib
import re

_SWAPS = {
    "the": "that same",
    "a": "one",
    "is": "remains",
    "was": "had been",
    "said": "stated",
    "also": "additionally",
    "but": "yet",
    "and": "as well as",
    "report": "account",
    "government": "administration",
    "states": "notes",
    "provide": "supply",
    "however": "nevertheless",
    "because": "since",
    "during": "throughout",
    "before": "prior to",
    "after": "following",
    "many": "numerous",
    "some": "certain",
    "important": "significant",
    "large": "substantial",
    "small": "modest",
    "new": "recent",
    "use": "employ",
    "used": "employed",
    "make": "produce",
    "made": "produced",
    "show": "demonstrate",
    "shows": "demonstrates",
    "found": "identified",
    "help": "assist",
    "need": "require",
    "needs": "requires",
    "get": "obtain",
    "team": "group",
    "review": "assessment",
    "records": "entries",
    "requires": "mandates",
    "verified": "confirmed",
    "added": "introduced",
}

_FILLERS = ("indeed", "notably", "in fact")
_FORCE_WINDOW = 7


def derived_int(seed_key: str, lo: int, hi: int) -> int:
    """Stable integer in [lo, hi] from a string key (no RNG state)."""
    if hi < lo:
        raise ValueError(f"derived_int: empty range [{lo}, {hi}]")
    digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    return lo + int(digest[:8], 16) % (hi - lo + 1)


def _has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def _rewrite_sentence(sentence: str, filler_base: int) -> str:
    words = sentence.split(" ")
    out: list[str] = []
    since_change = 0
    filler_count = 0
    for idx, word in enumerate(words):
        prev_word = words[idx - 1] if idx > 0 else ""
        next_word = words[idx + 1] if idx + 1 < len(words) else ""
        fact_guard = (
            _has_digit(word) or _has_digit(prev_word) or _has_digit(next_word) or word[:1].isupper()
        )
        bare = re.sub(r"\W", "", word).lower()
        repl = _SWAPS.get(bare)
        since_change += 1
        if fact_guard:
            out.append(word)
            continue
        if repl is not None and since_change >= 2:
            lead, core, trail = re.match(r"^(\W*)(.*?)(\W*)$", word).groups()
            out.append(lead + repl + trail)
            since_change = 0
        elif since_change >= _FORCE_WINDOW:
            filler = _FILLERS[(filler_base + filler_count) % len(_FILLERS)]
            filler_count += 1
            out.append(f"{filler} {word}")
            since_change = 0
        else:
            out.append(word)
    return " ".join(out)


def split_sentences(text: str) -> list[str]:
    """Latin terminators split only when followed by whitespace (so "U.S."
    stays whole); CJK terminators split unconditionally (no space follows
    them in running text)."""
    normalized = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", normalized)
    return [s for s in parts if s]


# A merged sentence starts lowercase, so merging is only safe when the
# second sentence genuinely opens with a function word: the head must be
# lowercase after its capital (blocks "IT", "USA"), and the following
# word must not be capitalized (blocks "One Medical", "There Inc." —
# entity surfaces that must survive verbatim on both sides of the pair).
_SAFE_STARTERS = frozenset(
    "the a an this that these those it its they their we our he she there "
    "some many most one in on for and but after before during across".split()
)


def _merge_safe(sentence: str) -> bool:
    parts = sentence.split(" ")
    head = parts[0]
    nxt = parts[1] if len(parts) > 1 else ""
    return (
        head.lower() in _SAFE_STARTERS
        and (len(head) == 1 or head[1:].islower())
        and not nxt[:1].isupper()
    )


def _strip_terminator(sentence: str) -> str:
    """Remove exactly one sentence terminator, but never an abbreviation
    period ("U.S.", "Inc." keep their final dot)."""
    if sentence[-1:] in ".!?" and not (len(sentence) >= 2 and sentence[-2].isupper()):
        return sentence[:-1]
    return sentence


def rewrite_preserving_facts(text: str, seed_key: str) -> str:
    """Dense fact-preserving rewrite: per-word swaps plus pairwise sentence
    merges, so the output is token-divergent from the input at chunk
    granularity while every digit and entity survives verbatim."""
    filler_base = derived_int(f"filler:{seed_key}", 0, len(_FILLERS) - 1)
    sentences = split_sentences(text)
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        first = _rewrite_sentence(sentences[i], filler_base + i)
        if i + 1 < len(sentences) and _merge_safe(sentences[i + 1]):
            second = _rewrite_sentence(sentences[i + 1], filler_base + i + 1)
            merged.append(
                _strip_terminator(first) + "; in other words, " + second[:1].lower() + second[1:]
            )
            i += 2
        else:
            merged.append(first)
            i += 1
    return " ".join(merged)


def block_overlap_ratio(original: str, rewritten: str, block_words: int = 16) -> float:
    """Fraction of the rewritten text's word blocks that appear verbatim
    in the original. The paraphrase classes require this to stay low:
    residual verbatim runs would let identity-verified paths serve the
    span and contaminate the class."""

    def shingles(text: str) -> set[tuple[str, ...]]:
        words = text.split()
        if len(words) < block_words:
            return {tuple(words)} if words else set()
        return {tuple(words[i : i + block_words]) for i in range(len(words) - block_words + 1)}

    rewritten_shingles = shingles(rewritten)
    if not rewritten_shingles:
        return 1.0
    original_shingles = shingles(original)
    hits = sum(1 for s in rewritten_shingles if s in original_shingles)
    return hits / len(rewritten_shingles)


def inject_sentence(text: str, sentence: str) -> str:
    """Insert a sentence at the sentence-list midpoint (position 0 for a
    single-sentence text), keeping the probe away from the shared header
    and the request tail in ordinary prose."""
    sentences = split_sentences(text)
    mid = len(sentences) // 2
    return " ".join(sentences[:mid] + [sentence] + sentences[mid:])
