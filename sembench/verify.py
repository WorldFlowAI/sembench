"""Endpoint pre-flight verification for live runs.

A live replay against the wrong pod, a stale build, or a different model
produces plausible-looking numbers that are wrong. Every live runner
verifies the endpoint before sending workload traffic: the server must be
reachable, its reported model must match the one the run claims to
measure, and whatever version identity the server exposes is captured
into the result's run metadata.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class EndpointReport:
    """Outcome of pre-flight verification for one endpoint."""

    base_url: str
    engine: str
    reachable: bool
    model_id: str = ""
    model_match: bool | None = None
    engine_version: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.reachable and self.model_match is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "engine": self.engine,
            "reachable": self.reachable,
            "model_id": self.model_id,
            "model_match": self.model_match,
            "engine_version": self.engine_version,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def fetch_json(url: str, timeout: float = _TIMEOUT_SECONDS) -> dict[str, Any]:
    """GET a JSON document; raises on transport or decode errors."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate_sglang_info(
    base_url: str,
    info: dict[str, Any],
    *,
    expect_model: str | None,
) -> EndpointReport:
    """Judge an SGLang /get_server_info (or /get_model_info) payload."""
    model_id = str(info.get("model_path") or info.get("model_id") or "")
    version = str(info.get("version") or info.get("sglang_version") or "")
    match = _model_matches(model_id, expect_model)
    errors = []
    if match is False:
        errors.append(f"endpoint serves {model_id!r}, run expects {expect_model!r}")
    return EndpointReport(
        base_url=base_url,
        engine="sglang",
        reachable=True,
        model_id=model_id,
        model_match=match,
        engine_version=version,
        errors=errors,
    )


def evaluate_openai_models(
    base_url: str,
    payload: dict[str, Any],
    *,
    expect_model: str | None,
) -> EndpointReport:
    """Judge an OpenAI-compatible /v1/models payload (gateway engines)."""
    model_ids = [str(entry.get("id", "")) for entry in payload.get("data", [])]
    match: bool | None
    if expect_model is None:
        match = None
    else:
        match = any(_model_matches(model_id, expect_model) for model_id in model_ids)
    errors = []
    if match is False:
        errors.append(f"gateway lists {model_ids!r}, run expects {expect_model!r}")
    return EndpointReport(
        base_url=base_url,
        engine="gateway",
        reachable=True,
        model_id=model_ids[0] if model_ids else "",
        model_match=match,
        errors=errors,
    )


def _model_matches(served: str, expected: str | None) -> bool | None:
    """None when no expectation was given.

    Served ids are often local paths (/models/Qwen2.5-7B-Instruct) while the
    run passes the HF id (Qwen/Qwen2.5-7B-Instruct) or vice versa, so accept
    a substring either way or equal final path components (case-insensitive).
    """
    if expected is None:
        return None
    if not served:
        return False
    if expected in served or served in expected:
        return True
    served_leaf = served.rstrip("/").rsplit("/", 1)[-1].lower()
    expected_leaf = expected.rstrip("/").rsplit("/", 1)[-1].lower()
    return served_leaf == expected_leaf


def verify_endpoint(
    *,
    engine: str,
    base_url: str,
    expect_model: str | None = None,
    fetcher=None,
) -> EndpointReport:
    """Fetch endpoint identity and judge it. Never raises: transport
    failures come back as reachable=False so callers can gate uniformly."""
    if fetcher is None:
        fetcher = fetch_json
    base = base_url.rstrip("/")
    try:
        if engine == "sglang":
            try:
                info = fetcher(f"{base}/get_server_info")
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
                info = fetcher(f"{base}/get_model_info")
            return evaluate_sglang_info(base, info, expect_model=expect_model)
        payload = fetcher(f"{base}/v1/models")
        return evaluate_openai_models(base, payload, expect_model=expect_model)
    except Exception as exc:  # noqa: BLE001 - uniform fail-closed report
        return EndpointReport(
            base_url=base,
            engine=engine,
            reachable=False,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
