"""Capture the exact engine launch configuration behind one benchmark arm.

A TTFT number is unreadable without the serve line that produced it. Prefix
caching on/off moves the baseline by more than the effect under test;
``--max-num-batched-tokens`` moves TTFT under chunked prefill; the CUDA
graph mode moves decode latency; and ``--kv-transfer-config`` is supposed
to be the *only* thing that differs between the baseline arm and the
product arm. Past campaigns published arms whose flags were never recorded,
so a disagreement between two results could not be attributed.

Every live result therefore carries: the verbatim serve command, the parsed
flags, the environment that changes engine behaviour, and the list of
phase-0 flags the protocol requires but the operator did not pass.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sembench.prometheus import COUNTER_SEMANTICS, MetricsSnapshot, MetricsWindow

PHASE0_BLOCK_SIZE = 16

# Flags that stand alone; everything else consumes the following token when
# that token is not itself a flag.
_BOOLEAN_FLAGS = frozenset(
    {
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--enable-prompt-caching",
        "--no-enable-prompt-caching",
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--disable-log-stats",
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
        "--enable-log-requests",
    }
)

_COMPILATION_FLAGS = ("--compilation-config", "-cc", "-O")
_CUDA_GRAPH_MODE_FLAGS = ("--cudagraph-mode", "--cuda-graph-mode")


@dataclass(frozen=True)
class EngineFlags:
    """Parsed vLLM serve line for one arm."""

    serve_command: str = ""
    argv: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    prefix_caching: bool | None = None
    prefix_caching_source: str = "engine_default"
    block_size: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    chunked_prefill: bool | None = None
    cuda_graph_mode: str | None = None
    cuda_graph_mode_source: str = "engine_default"
    enforce_eager: bool = False
    compilation_config: dict[str, Any] | None = None
    kv_transfer_config: dict[str, Any] | None = None
    kv_connector: str | None = None
    kv_role: str | None = None
    tensor_parallel_size: int | None = None
    enable_prompt_tokens_details: bool = False
    enable_per_request_metrics: bool = False
    disable_log_stats: bool = False
    server_dev_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "serve_command": self.serve_command,
            "argv": list(self.argv),
            "env": dict(self.env),
            "model": self.model,
            "prefix_caching": self.prefix_caching,
            "prefix_caching_source": self.prefix_caching_source,
            "block_size": self.block_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "chunked_prefill": self.chunked_prefill,
            "cuda_graph_mode": self.cuda_graph_mode,
            "cuda_graph_mode_source": self.cuda_graph_mode_source,
            "enforce_eager": self.enforce_eager,
            "compilation_config": self.compilation_config,
            "kv_transfer_config": self.kv_transfer_config,
            "kv_connector": self.kv_connector,
            "kv_role": self.kv_role,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enable_prompt_tokens_details": self.enable_prompt_tokens_details,
            "enable_per_request_metrics": self.enable_per_request_metrics,
            "disable_log_stats": self.disable_log_stats,
            "server_dev_mode": self.server_dev_mode,
        }


def _split_argv(command: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        try:
            return tuple(shlex.split(command))
        except ValueError as exc:
            raise ValueError(f"Unparseable serve command (unbalanced quoting): {exc}") from exc
    return tuple(str(token) for token in command)


def _split_env_prefix(argv: tuple[str, ...]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Peel leading ``KEY=VALUE`` tokens (and a bare ``env``) off a serve line."""
    env: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "env":
            index += 1
            continue
        if token.startswith("-") or "=" not in token:
            break
        key, _, value = token.partition("=")
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            break
        env[key] = value
        index += 1
    return env, argv[index:]


def _parse_tokens(argv: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    """Split argv into ``{flag: value|True}`` plus positionals."""
    flags: dict[str, Any] = {}
    positionals: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        index += 1
        if not token.startswith("-"):
            positionals.append(token)
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            flags[name] = value
            continue
        if token in _BOOLEAN_FLAGS:
            flags[token] = True
            continue
        if index < len(argv) and not argv[index].startswith("--"):
            flags[token] = argv[index]
            index += 1
            continue
        flags[token] = True
    return flags, positionals


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _first(flags: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in flags:
            return flags[name]
    return None


def _prefix_caching_state(flags: dict[str, Any]) -> tuple[bool | None, str]:
    """Prefix caching is tri-state: explicitly on, explicitly off, or unpinned.

    "Unpinned" is not the same as "off". vLLM 0.29 defaults it on, so an
    arm that never passed the flag is comparable only by assumption — which
    is exactly the gap this record closes.
    """
    for name in ("--no-enable-prefix-caching", "--no-enable-prompt-caching"):
        if flags.get(name):
            return False, "flag"
    for name in ("--enable-prefix-caching", "--enable-prompt-caching"):
        if name in flags:
            value = flags[name]
            if isinstance(value, str):
                return value.strip().lower() not in ("false", "0", "no"), "flag"
            return True, "flag"
    return None, "engine_default"


def _chunked_prefill_state(flags: dict[str, Any]) -> bool | None:
    if flags.get("--no-enable-chunked-prefill"):
        return False
    if "--enable-chunked-prefill" in flags:
        value = flags["--enable-chunked-prefill"]
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no")
        return True
    return None


def _cuda_graph_mode(
    flags: dict[str, Any], compilation: dict[str, Any] | None
) -> tuple[str | None, str]:
    explicit = _first(flags, _CUDA_GRAPH_MODE_FLAGS)
    if isinstance(explicit, str):
        return explicit.upper(), "flag"
    if compilation is not None and compilation.get("cudagraph_mode") is not None:
        return str(compilation["cudagraph_mode"]).upper(), "compilation_config"
    if flags.get("--enforce-eager"):
        return "NONE", "enforce_eager"
    return None, "engine_default"


def parse_serve_command(
    command: str | Sequence[str],
    env: dict[str, str] | None = None,
) -> EngineFlags:
    """Parse a vLLM serve line (optionally ``KEY=VALUE``-prefixed) into flags.

    Raises ValueError on a command string that cannot be tokenized, because
    silently recording an empty config is how an arm ends up published with
    no engine identity at all.
    """
    argv = _split_argv(command)
    inline_env, remainder = _split_env_prefix(argv)
    merged_env = {**inline_env, **(env or {})}
    flags, positionals = _parse_tokens(remainder)

    compilation = _as_json_object(_first(flags, _COMPILATION_FLAGS))
    kv_transfer = _as_json_object(flags.get("--kv-transfer-config"))
    prefix_caching, prefix_source = _prefix_caching_state(flags)
    graph_mode, graph_source = _cuda_graph_mode(flags, compilation)

    model = flags.get("--model")
    if not isinstance(model, str):
        # `vllm serve <model>` puts the model in the first positional after
        # the subcommand.
        tail = [token for token in positionals if token not in ("vllm", "serve", "python", "-m")]
        model = tail[0] if tail else None

    dev_mode = str(merged_env.get("VLLM_SERVER_DEV_MODE", "")).strip().lower() in (
        "1",
        "true",
        "on",
    )

    return EngineFlags(
        serve_command=command if isinstance(command, str) else shlex.join(argv),
        argv=argv,
        env=merged_env,
        model=model,
        prefix_caching=prefix_caching,
        prefix_caching_source=prefix_source,
        block_size=_as_int(flags.get("--block-size")),
        max_num_batched_tokens=_as_int(flags.get("--max-num-batched-tokens")),
        max_model_len=_as_int(flags.get("--max-model-len")),
        chunked_prefill=_chunked_prefill_state(flags),
        cuda_graph_mode=graph_mode,
        cuda_graph_mode_source=graph_source,
        enforce_eager=bool(flags.get("--enforce-eager")),
        compilation_config=compilation,
        kv_transfer_config=kv_transfer,
        kv_connector=(kv_transfer or {}).get("kv_connector"),
        kv_role=(kv_transfer or {}).get("kv_role"),
        tensor_parallel_size=_as_int(_first(flags, ("--tensor-parallel-size", "-tp"))),
        enable_prompt_tokens_details=bool(flags.get("--enable-prompt-tokens-details")),
        enable_per_request_metrics=bool(flags.get("--enable-per-request-metrics")),
        disable_log_stats=bool(flags.get("--disable-log-stats")),
        server_dev_mode=dev_mode,
    )


def phase0_flag_violations(flags: EngineFlags) -> list[str]:
    """Protocol flags the arm is missing, worst-first.

    These are not style preferences: without them the result is missing the
    fields the phase-0 metrics are computed from.
    """
    violations: list[str] = []
    if flags.prefix_caching is None:
        violations.append(
            "prefix caching state not pinned on the serve line "
            "(--enable-prefix-caching / --no-enable-prefix-caching); the arm is "
            "comparable only by assuming the engine default"
        )
    if flags.block_size is None:
        violations.append("--block-size not passed explicitly (phase 0 pins 16)")
    elif flags.block_size != PHASE0_BLOCK_SIZE:
        violations.append(
            f"--block-size {flags.block_size} does not match the phase-0 block size "
            f"{PHASE0_BLOCK_SIZE}; boundary alignment is computed against 16"
        )
    if flags.max_num_batched_tokens is None:
        violations.append(
            "--max-num-batched-tokens not pinned; chunked prefill batching moves TTFT "
            "and must be identical across arms"
        )
    if not flags.enable_prompt_tokens_details:
        violations.append(
            "--enable-prompt-tokens-details not passed; usage.prompt_tokens_details is "
            "absent and backend_confirmed_tokens will be null for every request"
        )
    if not flags.enable_per_request_metrics:
        violations.append(
            "--enable-per-request-metrics not passed; there is no engine-side TTFT "
            "excluding queue wait, so the concurrency arms are not evaluable"
        )
    if flags.enable_per_request_metrics and flags.disable_log_stats:
        violations.append(
            "--enable-per-request-metrics with --disable-log-stats; vLLM refuses to start"
        )
    if not flags.server_dev_mode:
        violations.append(
            "VLLM_SERVER_DEV_MODE=1 not set; POST /reset_prefix_cache?reset_external=true "
            "is not mounted and arms cannot be isolated from each other"
        )
    return violations


def engine_document(
    *,
    arm: str,
    flags: EngineFlags | None,
    window: MetricsWindow | None = None,
) -> dict[str, Any]:
    """The ``engine`` block of a result document for one arm."""
    violations = (
        phase0_flag_violations(flags)
        if flags is not None
        else ["no serve command recorded for this arm (--engine-serve-command)"]
    )
    document: dict[str, Any] = {
        "arm": arm,
        "flags": flags.to_dict() if flags is not None else None,
        "phase0_flag_violations": violations,
        "phase0_flags_ok": not violations,
        "prometheus": window.to_dict() if window is not None else None,
    }
    if flags is not None and flags.prefix_caching is not False:
        document["cached_tokens_is_local_plus_external"] = True
    return document


def parse_env_assignments(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` CLI arguments."""
    env: dict[str, str] = {}
    for raw in values:
        key, separator, value = str(raw).partition("=")
        if not separator or not key:
            raise ValueError(f"Engine env must be KEY=VALUE, got {raw!r}")
        env[key] = value
    return env


def load_engine_flags(
    *,
    serve_command: str | None = None,
    serve_command_file: str | None = None,
    env_assignments: Sequence[str] = (),
) -> EngineFlags | None:
    """Build EngineFlags from the CLI's three input forms, or None if unset."""
    command = serve_command
    if serve_command_file:
        try:
            command = Path(serve_command_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Cannot read serve command file {serve_command_file}: {exc}") from exc
    if not command:
        return None
    return parse_serve_command(command, parse_env_assignments(env_assignments))


def snapshot_document(snapshots: Sequence[MetricsSnapshot]) -> dict[str, Any]:
    """Serializable form of one before/after counter capture."""
    return {
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
        "semantics": dict(COUNTER_SEMANTICS),
    }


def snapshots_from_document(document: dict[str, Any]) -> tuple[MetricsSnapshot, ...]:
    """Rebuild snapshots written by ``snapshot_document``."""
    rows = document.get("snapshots")
    if not isinstance(rows, list):
        raise ValueError("Engine snapshot document has no 'snapshots' list")
    return tuple(
        MetricsSnapshot(
            url=str(row.get("url", "")),
            captured_at_utc=str(row.get("captured_at_utc", "")),
            counters=dict(row.get("counters") or {}),
            sample_count=int(row.get("sample_count") or 0),
            error=row.get("error"),
        )
        for row in rows
    )


def attach_engine_document(result_path: str | Path, document: dict[str, Any]) -> None:
    """Splice an engine block into an already-written result JSON.

    For arms driven by something other than a sembench runner: the result is
    still not readable without its engine config.
    """
    path = Path(result_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read result document {result_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Result document {result_path} is not a JSON object")
    updated = {**payload, "engine": document}
    path.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
