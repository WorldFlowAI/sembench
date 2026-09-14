"""Engine launch flags recorded per arm (B12).

Identical flags across arms is the premise of every paired number the
phase-0 report makes; only --kv-transfer-config is supposed to differ. Past
campaigns published arms whose serve lines were never captured, so a
disagreement between two results could not be attributed to anything.
These tests pin what gets recorded and what counts as an incomplete arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sembench.cli import main
from sembench.engine_config import (
    attach_engine_document,
    engine_document,
    load_engine_flags,
    parse_env_assignments,
    parse_serve_command,
    phase0_flag_violations,
    snapshot_document,
    snapshots_from_document,
)
from sembench.prometheus import (
    EXTERNAL_KV_TRANSFER_TOKENS_KEY,
    HITS_KEY,
    QUERIES_KEY,
    MetricsSnapshot,
    MetricsWindow,
)

_KV_TRANSFER = (
    '{"kv_connector":"SemBlendVllmConnector","kv_role":"kv_both",'
    '"kv_connector_module_path":"semblend_vllm_connector.connector"}'
)

_PHASE0_SERVE = (
    "VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-7B-Instruct "
    "--enable-prefix-caching --block-size 16 --max-model-len 8192 "
    "--max-num-batched-tokens 8192 --enable-chunked-prefill "
    "--enable-prompt-tokens-details --enable-per-request-metrics "
    f"--kv-transfer-config '{_KV_TRANSFER}'"
)


def _snapshot(url: str, queries: float, hits: float, external: float) -> MetricsSnapshot:
    return MetricsSnapshot(
        url=url,
        captured_at_utc="2026-01-01T00:00:00Z",
        counters={
            QUERIES_KEY: queries,
            HITS_KEY: hits,
            EXTERNAL_KV_TRANSFER_TOKENS_KEY: external,
        },
    )


def test_parse_serve_command_records_the_four_flags_that_move_the_number():
    flags = parse_serve_command(_PHASE0_SERVE)
    assert flags.prefix_caching is True
    assert flags.max_num_batched_tokens == 8192
    assert flags.block_size == 16
    assert flags.kv_connector == "SemBlendVllmConnector"
    assert flags.model == "Qwen/Qwen2.5-7B-Instruct"
    assert flags.serve_command == _PHASE0_SERVE


def test_prefix_caching_unpinned_is_not_recorded_as_off():
    """Unpinned means "whatever the engine defaults to", which is a different
    claim from "prefix caching was off" — the A0 vs A1 distinction."""
    flags = parse_serve_command("vllm serve m --block-size 16")
    assert flags.prefix_caching is None
    assert flags.prefix_caching_source == "engine_default"


def test_no_enable_prefix_caching_is_recorded_as_explicitly_off():
    flags = parse_serve_command("vllm serve m --no-enable-prefix-caching")
    assert flags.prefix_caching is False
    assert flags.prefix_caching_source == "flag"


def test_enforce_eager_is_recorded_as_cuda_graph_mode_none():
    flags = parse_serve_command("vllm serve m --enforce-eager")
    assert flags.enforce_eager is True
    assert flags.cuda_graph_mode == "NONE"
    assert flags.cuda_graph_mode_source == "enforce_eager"


def test_cuda_graph_mode_read_from_compilation_config_json():
    flags = parse_serve_command(
        'vllm serve m --compilation-config \'{"cudagraph_mode":"FULL_AND_PIECEWISE"}\''
    )
    assert flags.cuda_graph_mode == "FULL_AND_PIECEWISE"
    assert flags.cuda_graph_mode_source == "compilation_config"
    assert flags.compilation_config == {"cudagraph_mode": "FULL_AND_PIECEWISE"}


def test_cuda_graph_mode_is_unknown_when_never_pinned():
    flags = parse_serve_command("vllm serve m")
    assert flags.cuda_graph_mode is None
    assert flags.cuda_graph_mode_source == "engine_default"


def test_kv_transfer_config_is_parsed_into_connector_and_role():
    flags = parse_serve_command(f"vllm serve m --kv-transfer-config '{_KV_TRANSFER}'")
    assert flags.kv_role == "kv_both"
    assert flags.kv_transfer_config["kv_connector_module_path"].endswith("connector")


def test_stock_arm_records_a_null_kv_transfer_config():
    flags = parse_serve_command("vllm serve m --enable-prefix-caching")
    assert flags.kv_transfer_config is None
    assert flags.kv_connector is None


def test_env_prefix_on_the_serve_line_sets_server_dev_mode():
    flags = parse_serve_command("VLLM_SERVER_DEV_MODE=1 vllm serve m")
    assert flags.server_dev_mode is True
    assert flags.env["VLLM_SERVER_DEV_MODE"] == "1"


def test_separately_supplied_env_also_sets_server_dev_mode():
    flags = parse_serve_command("vllm serve m", {"VLLM_SERVER_DEV_MODE": "1"})
    assert flags.server_dev_mode is True


def test_argv_form_is_accepted_and_joined_back():
    flags = parse_serve_command(["vllm", "serve", "m", "--block-size", "16"])
    assert flags.block_size == 16
    assert "--block-size 16" in flags.serve_command


def test_equals_form_flags_are_parsed():
    flags = parse_serve_command("vllm serve m --block-size=16 --max-num-batched-tokens=4096")
    assert flags.block_size == 16
    assert flags.max_num_batched_tokens == 4096


def test_unbalanced_quoting_raises_rather_than_recording_an_empty_config():
    with pytest.raises(ValueError):
        parse_serve_command("vllm serve m --kv-transfer-config '{\"a\": 1}")


def test_phase0_serve_line_has_no_violations():
    assert phase0_flag_violations(parse_serve_command(_PHASE0_SERVE)) == []


def test_phase0_violations_name_every_missing_flag():
    violations = phase0_flag_violations(parse_serve_command("vllm serve m"))
    joined = " | ".join(violations)
    assert "--enable-prompt-tokens-details" in joined
    assert "--enable-per-request-metrics" in joined
    assert "--block-size" in joined
    assert "--max-num-batched-tokens" in joined
    assert "VLLM_SERVER_DEV_MODE" in joined
    assert "prefix caching state not pinned" in joined


def test_phase0_violation_when_block_size_is_not_sixteen():
    flags = parse_serve_command(_PHASE0_SERVE.replace("--block-size 16", "--block-size 32"))
    joined = " | ".join(phase0_flag_violations(flags))
    assert "does not match the phase-0 block size" in joined


def test_per_request_metrics_with_disable_log_stats_is_a_violation():
    flags = parse_serve_command(f"{_PHASE0_SERVE} --disable-log-stats")
    joined = " | ".join(phase0_flag_violations(flags))
    assert "vLLM refuses to start" in joined


def test_engine_document_marks_cached_tokens_as_local_plus_external():
    """With prefix caching on, usage.prompt_tokens_details.cached_tokens is a
    sum, so it must never be read back as semantic reuse."""
    document = engine_document(arm="warm", flags=parse_serve_command(_PHASE0_SERVE))
    assert document["cached_tokens_is_local_plus_external"] is True
    assert document["phase0_flags_ok"] is True
    assert document["arm"] == "warm"


def test_engine_document_without_a_serve_command_is_not_ok():
    document = engine_document(arm="cold", flags=None)
    assert document["flags"] is None
    assert document["phase0_flags_ok"] is False
    assert "no serve command recorded" in document["phase0_flag_violations"][0]


def test_engine_document_embeds_the_counter_window():
    window = MetricsWindow(
        before=(_snapshot("http://w0/metrics", 0.0, 0.0, 0.0),),
        after=(_snapshot("http://w0/metrics", 40960.0, 12288.0, 15360.0),),
    )
    document = engine_document(arm="warm", flags=parse_serve_command(_PHASE0_SERVE), window=window)
    assert document["prometheus"]["delta"][f"{HITS_KEY}_delta"] == 12288.0
    assert document["prometheus"]["delta"][f"{QUERIES_KEY}_delta"] == 40960.0


def test_parse_env_assignments_rejects_a_bare_key():
    with pytest.raises(ValueError):
        parse_env_assignments(["VLLM_SERVER_DEV_MODE"])


def test_load_engine_flags_reads_a_command_file(tmp_path: Path):
    command_file = tmp_path / "serve.txt"
    command_file.write_text(_PHASE0_SERVE + "\n", encoding="utf-8")
    flags = load_engine_flags(serve_command_file=str(command_file))
    assert flags is not None
    assert flags.max_num_batched_tokens == 8192


def test_load_engine_flags_returns_none_when_nothing_was_supplied():
    assert load_engine_flags() is None


def test_load_engine_flags_raises_on_a_missing_command_file(tmp_path: Path):
    with pytest.raises(ValueError):
        load_engine_flags(serve_command_file=str(tmp_path / "absent.txt"))


def test_snapshot_document_round_trips_through_json(tmp_path: Path):
    document = snapshot_document([_snapshot("http://w0/metrics", 1.0, 2.0, 3.0)])
    reloaded = snapshots_from_document(json.loads(json.dumps(document)))
    assert reloaded[0].counters[HITS_KEY] == 2.0
    assert reloaded[0].url == "http://w0/metrics"


def test_attach_engine_document_splices_into_an_existing_result(tmp_path: Path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"aggregate": {"request_count": 3}}), encoding="utf-8")
    attach_engine_document(result, engine_document(arm="cold", flags=None))
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["aggregate"]["request_count"] == 3
    assert payload["engine"]["arm"] == "cold"


def test_attach_engine_document_rejects_a_non_result_file(tmp_path: Path):
    target = tmp_path / "notjson.txt"
    target.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        attach_engine_document(target, {"arm": "cold"})


def test_write_result_always_emits_an_engine_key(tmp_path: Path):
    """Null means "no engine config recorded", which a reader must be able to
    tell apart from a recorded config."""
    from sembench.results import write_result

    out = tmp_path / "r.json"
    write_result(out, requests=[], config={"mode": "test"})
    assert json.loads(out.read_text(encoding="utf-8"))["engine"] is None


def _fake_scrape(batches: list[list[MetricsSnapshot]]):
    def scrape(urls, timeout: float = 15.0) -> list[MetricsSnapshot]:
        return batches.pop(0)

    return scrape


def test_run_live_gateway_records_flags_and_counter_deltas(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "fixture.jsonl"
    result = tmp_path / "gateway.json"
    main(["build", "--profile", "fixture", "--output", str(manifest)])

    monkeypatch.setattr("sembench.cli.run_live_gateway", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "sembench.cli.scrape_all",
        _fake_scrape(
            [
                [_snapshot("http://w0:8000/metrics", 100.0, 10.0, 10.0)],
                [_snapshot("http://w0:8000/metrics", 41060.0, 12298.0, 15370.0)],
            ]
        ),
    )

    main(
        [
            "run-live-gateway",
            "--manifest", str(manifest),
            "--output", str(result),
            "--gateway-url", "http://w0:8000",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--arm", "warm",
            "--skip-verify",
            "--engine-serve-command", _PHASE0_SERVE,
        ]
    )

    engine = json.loads(result.read_text(encoding="utf-8"))["engine"]
    assert engine["flags"]["prefix_caching"] is True
    assert engine["flags"]["max_num_batched_tokens"] == 8192
    assert engine["flags"]["kv_connector"] == "SemBlendVllmConnector"
    assert engine["phase0_flags_ok"] is True
    assert engine["prometheus"]["delta"][f"{HITS_KEY}_delta"] == 12288.0
    assert engine["prometheus"]["delta"][f"{QUERIES_KEY}_delta"] == 40960.0
    assert engine["prometheus"]["delta"][f"{EXTERNAL_KV_TRANSFER_TOKENS_KEY}_delta"] == 15360.0


def test_run_live_gateway_records_a_null_engine_block_when_no_serve_line_given(
    tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "fixture.jsonl"
    result = tmp_path / "gateway.json"
    main(["build", "--profile", "fixture", "--output", str(manifest)])
    monkeypatch.setattr("sembench.cli.run_live_gateway", lambda *args, **kwargs: [])
    monkeypatch.setattr("sembench.cli.scrape_all", _fake_scrape([[], []]))

    main(
        [
            "run-live-gateway",
            "--manifest", str(manifest),
            "--output", str(result),
            "--gateway-url", "http://w0:8000",
            "--model", "m",
            "--skip-verify",
        ]
    )

    engine = json.loads(result.read_text(encoding="utf-8"))["engine"]
    assert engine["flags"] is None
    assert engine["phase0_flags_ok"] is False


def test_run_live_gateway_refuses_an_incomplete_serve_line_when_required(
    tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "fixture.jsonl"
    main(["build", "--profile", "fixture", "--output", str(manifest)])
    monkeypatch.setattr("sembench.cli.run_live_gateway", lambda *args, **kwargs: [])
    monkeypatch.setattr("sembench.cli.scrape_all", _fake_scrape([[], []]))

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run-live-gateway",
                "--manifest", str(manifest),
                "--output", str(tmp_path / "gateway.json"),
                "--gateway-url", "http://w0:8000",
                "--model", "m",
                "--skip-verify",
                "--engine-serve-command", "vllm serve m",
                "--require-phase0-flags",
            ]
        )
    assert excinfo.value.code == 3


def test_engine_metrics_urls_default_to_the_donor_and_gateway():
    from sembench.cli import _engine_metrics_urls, build_parser

    args = build_parser().parse_args(
        [
            "run-live-gateway",
            "--manifest", "m.jsonl", "--output", "o.json",
            "--gateway-url", "http://gw:8000", "--donor-url", "http://w0:8000",
            "--model", "m",
        ]
    )
    assert _engine_metrics_urls(args) == ["http://w0:8000", "http://gw:8000"]


def test_engine_metrics_urls_prefer_explicit_workers():
    from sembench.cli import _engine_metrics_urls, build_parser

    args = build_parser().parse_args(
        [
            "run-live-gateway",
            "--manifest", "m.jsonl", "--output", "o.json",
            "--gateway-url", "http://gw:8000", "--model", "m",
            "--metrics-url", "http://w0:8000",
            "--metrics-url", "http://w1:8000",
        ]
    )
    assert _engine_metrics_urls(args) == ["http://w0:8000", "http://w1:8000"]


def test_engine_snapshot_and_window_commands_produce_the_engine_block(
    tmp_path: Path, monkeypatch, capsys
):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    window = tmp_path / "window.json"
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"aggregate": {"request_count": 1}}), encoding="utf-8")

    monkeypatch.setattr(
        "sembench.cli.scrape_all",
        _fake_scrape(
            [
                [_snapshot("http://w0:8000/metrics", 0.0, 0.0, 0.0)],
                [_snapshot("http://w0:8000/metrics", 2048.0, 512.0, 600.0)],
            ]
        ),
    )
    main(["engine-snapshot", "--metrics-url", "http://w0:8000", "--output", str(before)])
    main(["engine-snapshot", "--metrics-url", "http://w0:8000", "--output", str(after)])
    capsys.readouterr()

    main(
        [
            "engine-window",
            "--before", str(before),
            "--after", str(after),
            "--arm", "cold",
            "--output", str(window),
            "--result", str(result),
            "--engine-serve-command", _PHASE0_SERVE,
        ]
    )

    document = json.loads(window.read_text(encoding="utf-8"))
    assert document["prometheus"]["delta"][f"{HITS_KEY}_delta"] == 512.0
    assert json.loads(result.read_text(encoding="utf-8"))["engine"]["arm"] == "cold"


def test_engine_window_exits_three_on_an_incomplete_serve_line(tmp_path: Path, monkeypatch):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    monkeypatch.setattr(
        "sembench.cli.scrape_all",
        _fake_scrape([[_snapshot("http://w0/metrics", 0.0, 0.0, 0.0)]] * 2),
    )
    main(["engine-snapshot", "--metrics-url", "http://w0", "--output", str(before)])
    main(["engine-snapshot", "--metrics-url", "http://w0", "--output", str(after)])

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "engine-window",
                "--before", str(before),
                "--after", str(after),
                "--engine-serve-command", "vllm serve m",
                "--require-phase0-flags",
            ]
        )
    assert excinfo.value.code == 3
