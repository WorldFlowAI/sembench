import json

import pytest

from sembench.cli import main
from sembench.verify import verify_endpoint


def _fetcher(routes: dict[str, dict]):
    def fetch(url: str, timeout: float = 0.0):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected url {url}")

    return fetch


def test_sglang_pass_with_version_capture():
    report = verify_endpoint(
        engine="sglang",
        base_url="http://host:30000/",
        expect_model="Qwen/Qwen2.5-7B-Instruct",
        fetcher=_fetcher(
            {
                "/get_server_info": {
                    "model_path": "/models/Qwen2.5-7B-Instruct",
                    "version": "0.5.4",
                }
            }
        ),
    )
    assert report.passed
    assert report.engine_version == "0.5.4"
    assert report.model_match is True


def test_sglang_model_mismatch_fails():
    report = verify_endpoint(
        engine="sglang",
        base_url="http://host:30000",
        expect_model="Qwen/Qwen2.5-7B-Instruct",
        fetcher=_fetcher({"/get_server_info": {"model_path": "/models/Llama-3.1-8B"}}),
    )
    assert not report.passed
    assert report.model_match is False
    assert report.errors


def test_sglang_falls_back_to_get_model_info():
    calls = []

    def fetch(url: str, timeout: float = 0.0):
        calls.append(url)
        if url.endswith("/get_server_info"):
            raise json.JSONDecodeError("bad", "", 0)
        return {"model_path": "m", "version": "0.5.0"}

    report = verify_endpoint(engine="sglang", base_url="http://h", fetcher=fetch)
    assert report.reachable
    assert any(url.endswith("/get_model_info") for url in calls)


def test_gateway_membership_check():
    routes = {"/v1/models": {"data": [{"id": "qwen2.5-7b"}, {"id": "other"}]}}
    ok = verify_endpoint(
        engine="gateway",
        base_url="http://gw",
        expect_model="qwen2.5-7b",
        fetcher=_fetcher(routes),
    )
    bad = verify_endpoint(
        engine="gateway",
        base_url="http://gw",
        expect_model="mistral-7b",
        fetcher=_fetcher(routes),
    )
    assert ok.passed and ok.model_match is True
    assert not bad.passed and bad.model_match is False


def test_unreachable_is_fail_closed():
    report = verify_endpoint(
        engine="gateway",
        base_url="http://down",
        fetcher=_fetcher({"/v1/models": ConnectionError("refused")}),
    )
    assert not report.reachable
    assert not report.passed


def test_no_expectation_passes_without_match_claim():
    report = verify_endpoint(
        engine="sglang",
        base_url="http://h",
        fetcher=_fetcher({"/get_server_info": {"model_path": "anything"}}),
    )
    assert report.passed
    assert report.model_match is None


def test_cli_verify_endpoint_exit_codes(monkeypatch, capsys):
    import sembench.verify as verify_mod

    monkeypatch.setattr(
        verify_mod,
        "fetch_json",
        _fetcher({"/get_server_info": {"model_path": "/models/A", "version": "1"}}),
    )
    main(["verify-endpoint", "--engine", "sglang", "--base-url", "http://h", "--expect-model", "A"])
    assert json.loads(capsys.readouterr().out)["passed"] is True

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "verify-endpoint",
                "--engine",
                "sglang",
                "--base-url",
                "http://h",
                "--expect-model",
                "B-different",
            ]
        )
    assert excinfo.value.code == 3
