"""Tokenizer adapters used by the offline benchmark."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    """Small encode-only tokenizer protocol."""

    name: str

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""


@dataclass
class StableTokenizer:
    """Deterministic fallback tokenizer with no external dependencies.

    It is not model-equivalent. Use a Hugging Face tokenizer for publishable
    model-specific block counts.
    """

    name: str = "stable-regex-sha256"

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0).lower()
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            out.append(int.from_bytes(digest[:4], "little") & 0x7FFFFFFF)
        return out


class HuggingFaceTokenizer:
    """Thin wrapper around transformers.AutoTokenizer."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


def load_tokenizer(model_name: str | None) -> Tokenizer:
    """Load a requested tokenizer or fall back to the stable local tokenizer."""
    if model_name is None or model_name.strip() in ("", "stable", "stable-regex-sha256"):
        return StableTokenizer()
    return HuggingFaceTokenizer(model_name)
