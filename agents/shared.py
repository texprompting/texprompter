from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI


_PACKAGE_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_PACKAGE_ENV_PATH)
load_dotenv()


def _normalize_openai_compat_stream_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Fold provider-specific delta fields into ``content`` for ChatOpenAI streaming.

    Some OpenAI-compat servers stream assistant text under delta keys LangChain's converter
    ignores (``reasoning_content``, ``thinking``, …); merge those into ``content`` so
    ``on_llm_new_token`` receives printable text.
    """
    if not isinstance(chunk, dict):
        return chunk
    out = copy.deepcopy(chunk)
    choices = out.get("choices")
    if choices is None and isinstance(out.get("chunk"), dict):
        choices = out["chunk"].get("choices")
    if not choices:
        return out
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return out
    base = delta.get("content")
    base_s = base if isinstance(base, str) else ""
    extra_parts: list[str] = []
    for key in ("reasoning_content", "thinking", "reasoning"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            extra_parts.append(val)
        elif isinstance(val, dict):
            nested = val.get("content") or val.get("text")
            if isinstance(nested, str) and nested:
                extra_parts.append(nested)
    if not extra_parts:
        return out
    delta["content"] = base_s + "".join(extra_parts)

    return out


class _OllamaCompatStreamChatOpenAI(ChatOpenAI):
    """Apply OpenAI-compat stream normalization before LangChain converts chunks."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        normalized = _normalize_openai_compat_stream_chunk(chunk)
        return super()._convert_chunk_to_generation_chunk(
            normalized, default_chunk_class, base_generation_info
        )


class OllamaLiveStreamHandler(BaseCallbackHandler):
    """Print assistant text tokens to stdout.

    Tool-call rounds stream with empty ``content``; emit deltas on stderr so the run
    still looks \"live\" (OpenAI-compatible servers behave this way).
    """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        chunk = kwargs.get("chunk")
        msg = getattr(chunk, "message", None) if chunk is not None else None

        effective = ""
        if token:
            effective = token
        elif chunk is not None:
            ct_text = getattr(chunk, "text", None)
            if isinstance(ct_text, str) and ct_text:
                effective = ct_text
            elif ct_text is not None:
                try:
                    effective = str(ct_text)
                except Exception:
                    effective = ""
        if not effective and msg is not None:
            try:
                tx = msg.text
                effective = str(tx) if tx is not None else ""
            except Exception:
                effective = ""
        if not effective and msg is not None:
            ak = getattr(msg, "additional_kwargs", None) or {}
            if isinstance(ak, dict):
                for key in ("reasoning_content", "thinking", "reasoning"):
                    v = ak.get(key)
                    if isinstance(v, str) and v:
                        effective = v
                        break

        if effective:
            sys.stdout.write(effective)
            sys.stdout.flush()
            return
        if msg is None:
            return
        tool_chunks = getattr(msg, "tool_call_chunks", None) or []
        if not tool_chunks:
            return
        parts: list[str] = []
        for tc in tool_chunks:
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "args", "") or ""
            if name or args:
                parts.append(f"{name}({args[:240]})")
        if parts:
            sys.stderr.write("[llm→tool] " + " · ".join(parts) + "\n")
            sys.stderr.flush()


def ollama_stream_to_stdout_enabled() -> bool:
    """When true, ChatOpenAI streams tokens and prints them live via runnable callbacks."""
    return os.getenv("OLLAMA_STREAM_STDOUT", "").strip().lower() in ("1", "true", "yes")


# Hard cap on the number of LangGraph steps a single ``create_agent.invoke`` call may take.
# At 12 we comfortably accommodate one round of parallel tool calls plus a few structured-
# output retries while still failing fast (instead of hanging) when the model gets stuck in
# a tool-call <-> validation-error ping-pong.
DEFAULT_AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))


# ---------------------------------------------------------------------------
# Stall classification helpers
# ---------------------------------------------------------------------------


class TokenOverflowError(RuntimeError):
    """Raised when the LLM returns ``finish_reason='length'``.

    This means the model hit its token limit mid-response, producing a
    truncated (and usually invalid) structured output.  Raising a hard error
    here prevents the agent from silently retrying on garbled JSON and instead
    surfaces the root cause immediately so it can be logged and diagnosed.
    """


def classify_exception(exc: Exception) -> "StallReason":  # noqa: F821 – resolved at runtime
    """Map an arbitrary exception to the nearest ``StallReason`` enum value.

    This is intentionally a best-effort heuristic: we inspect the exception
    type name and message text because many LangChain / LangGraph exceptions
    wrap the originating cause without re-raising it as a typed exception.
    """
    from schemas.basemodels import StallReason  # local import to avoid circular deps

    name = type(exc).__name__
    msg = str(exc).lower()

    if "graphrecursionerror" in name.lower() or "recursion limit" in msg:
        return StallReason.RECURSION_LIMIT
    if isinstance(exc, TokenOverflowError) or "tokenoverflow" in name.lower():
        return StallReason.TOKEN_OVERFLOW
    if "validationerror" in name.lower() or "pydantic" in msg:
        return StallReason.VALIDATION_ERROR
    if "timeout" in msg or "timed out" in msg or "readtimeout" in name.lower():
        return StallReason.NETWORK_TIMEOUT
    if name in ("ConnectionError", "RemoteDisconnected", "HTTPStatusError", "ConnectError"):
        return StallReason.NETWORK_ERROR
    if "connectionerror" in name.lower() or "remotedisconnected" in name.lower():
        return StallReason.NETWORK_ERROR
    if "structured_response" in msg or "parse" in name.lower():
        return StallReason.PARSE_FAILURE
    if "filenotfounderror" in name.lower() or "ioerror" in name.lower():
        return StallReason.FILE_IO_ERROR
    if "promptload" in name.lower() or "prompt" in name.lower() and "load" in msg:
        return StallReason.PROMPT_LOAD_FAILURE
    return StallReason.UNKNOWN


def _last_ai_content(messages: list[Any]) -> str | None:
    """Return the text content of the last AI message, or ``None`` if absent.

    Used as a last-ditch fallback when ``response['structured_response']`` is
    ``None`` but the model did emit text that may be valid JSON.
    """
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _prompt_metadata(prompt: Any) -> dict[str, str]:
    if prompt is None:
        return {}
    if hasattr(prompt, "as_dict"):
        raw = prompt.as_dict()
    elif isinstance(prompt, Mapping):
        raw = prompt
    else:
        raw = {
            "short_name": getattr(prompt, "short_name", None),
            "registry_name": getattr(prompt, "registry_name", None),
            "requested_uri": getattr(prompt, "requested_uri", None),
            "resolved_uri": getattr(prompt, "resolved_uri", None),
            "version": getattr(prompt, "version", None),
            "source": getattr(prompt, "source", None),
            "fallback_reason": getattr(prompt, "fallback_reason", None),
        }
    return {
        f"prompt.{key}": str(value)
        for key, value in raw.items()
        if value is not None and key != "template"
    }


def agent_invoke_config(
    *,
    stage: str | None = None,
    prompt: Any | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Return the ``RunnableConfig`` used for every ``create_agent.invoke`` call.

    Centralised so the recursion limit (and any future shared knobs like tags)
    stay consistent across all four agents and the evaluation harness.

    Note: ``langchain.agents.create_agent`` invokes the bound chat model as
    ``model.invoke(messages)`` without forwarding the outer agent config, so
    stdout streaming handlers must be attached on ``ChatOpenAI`` itself (see
    ``build_chat_model``), not here.
    """
    config: dict[str, Any] = {"recursion_limit": DEFAULT_AGENT_RECURSION_LIMIT}
    merged_metadata: dict[str, Any] = {}
    if stage:
        merged_metadata["agent.stage"] = stage
    merged_metadata.update(_prompt_metadata(prompt))
    if metadata:
        merged_metadata.update(dict(metadata))
    if merged_metadata:
        config["metadata"] = merged_metadata

    merged_tags = list(tags or [])
    if stage:
        merged_tags.append(f"stage:{stage}")
    if merged_tags:
        config["tags"] = merged_tags
    if run_name:
        config["run_name"] = run_name
    return config


def prompt_debug_payload(prompt: Any) -> dict[str, str]:
    """Return prompt metadata using stable keys for pipeline debug payloads."""
    if prompt is None:
        return {}
    if hasattr(prompt, "as_dict"):
        return prompt.as_dict()
    return {
        key.replace("prompt.", "", 1): value
        for key, value in _prompt_metadata(prompt).items()
    }


def _set_prompt_span_attributes(span: Any, *, stage: str, prompt: Any, user_message: str) -> None:
    attributes = {
        "agent.stage": stage,
        "prompt.user_message_chars": len(user_message),
        **_prompt_metadata(prompt),
    }
    for key, value in attributes.items():
        try:
            span.set_attribute(key, value)
        except Exception:
            continue

    try:
        from mlflow.tracing import set_span_chat_messages

        set_span_chat_messages(
            span,
            [
                {"role": "system", "content": getattr(prompt, "template", "")},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception:
        pass


def invoke_agent_with_prompt_trace(
    agent: Any,
    *,
    stage: str,
    prompt: Any,
    user_message: str,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a LangChain agent and add prompt lineage to the active MLflow trace.

    Raises
    ------
    TokenOverflowError
        If the last AI message in the response has ``finish_reason == 'length'``,
        indicating the model hit its context/token limit and the output is truncated.
        This is a hard failure so the pipeline surfaces the root cause instead of
        silently retrying on garbled JSON.
    """
    payload = {"messages": [{"role": "user", "content": user_message}]}
    config = agent_invoke_config(
        stage=stage,
        prompt=prompt,
        tags=tags,
        metadata=metadata,
        run_name=f"{stage}_agent",
    )

    def _invoke() -> dict[str, Any]:
        result = agent.invoke(payload, config=config)
        _check_finish_reason(result, stage=stage)
        return result

    try:
        import mlflow

        start_span = getattr(mlflow, "start_span")
        active_run = mlflow.active_run()
        get_current_active_span = getattr(mlflow, "get_current_active_span", None)
        active_span = get_current_active_span() if get_current_active_span else None
    except Exception:
        return _invoke()

    if active_run is None and active_span is None:
        return _invoke()

    try:
        span_cm = start_span(name=f"{stage}_prompt", span_type="CHAIN")
    except Exception:
        return _invoke()

    with span_cm as span:
        _set_prompt_span_attributes(
            span,
            stage=stage,
            prompt=prompt,
            user_message=user_message,
        )
        return _invoke()


def _check_finish_reason(response: dict[str, Any], *, stage: str) -> None:
    """Inspect the last AI message and hard-fail on ``finish_reason == 'length'``.

    When Ollama (or any OpenAI-compatible backend) truncates a response because
    the model hit its token limit, the OpenAI SDK sets ``finish_reason='length'``
    on the choice.  LangChain surfaces this on the AIMessage as
    ``response_metadata['finish_reason']``.  Truncated output is almost always
    invalid JSON for structured-output calls, so we raise immediately rather than
    letting the agent loop retry and hit the recursion cap.
    """
    messages = response.get("messages", []) if isinstance(response, dict) else []
    for message in reversed(messages or []):
        if getattr(message, "type", None) != "ai":
            continue
        meta = getattr(message, "response_metadata", None) or {}
        if not isinstance(meta, dict):
            break
        finish_reason = meta.get("finish_reason") or meta.get("stop_reason")
        if finish_reason and str(finish_reason).lower() == "length":
            raise TokenOverflowError(
                f"[{stage}] LLM hit token limit (finish_reason='length'). "
                "Increase OLLAMA_MAX_TOKENS or reduce the context size."
            )
        break  # only check the last AI message


def get_project_root() -> Path:
    """Return the repository root based on this package location."""
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_test_outputs_dir() -> Path:
    return get_project_root() / "TestOutputs"


def load_csv_input_schema(csv_file_path: str, preview_rows: int) -> dict[str, Any]:
    """Load the canonical CSV input schema payload used across agents and pipeline state."""
    module_path = get_data_dir() / "csv_to_input_scheme.py"
    spec = importlib.util.spec_from_file_location("csv_to_input_scheme", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load schema module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_input_data"):
        raise AttributeError("csv_to_input_scheme.py does not define get_input_data")

    get_input_data = getattr(module, "get_input_data")
    return get_input_data(csv_file_name=csv_file_path, preview_rows=preview_rows)


def canonical_openai_compat_base_url(host_or_url: str) -> str:
    """Normalize a host or base URL into an OpenAI SDK ``base_url`` ending with ``/v1``."""
    raw = host_or_url.strip()
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    base_url = raw.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def ollama_openai_compatible_base_url() -> str:
    """Return the OpenAI-compatible base URL for the configured Ollama server."""
    raw = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    return canonical_openai_compat_base_url(raw)


def ensure_mlflow_openai_base_url_for_ollama_judge() -> None:
    """If ``OPENAI_BASE_URL`` is unset, set it so MLflow genai judges hit Ollama like the pipeline.

    MLflow's LLM helpers default to ``https://api.openai.com`` when ``OPENAI_BASE_URL`` is empty.
    This project often only sets ``OLLAMA_BASE_URL`` + ``OPENAI_API_KEY=ollama`` for local models;
    without this, judges send the placeholder key to OpenAI Cloud and fail with 401.
    """
    if os.getenv("OPENAI_BASE_URL", "").strip():
        return

    if os.getenv("OLLAMA_BASE_URL", "").strip():
        os.environ["OPENAI_BASE_URL"] = canonical_openai_compat_base_url(os.getenv("OLLAMA_BASE_URL", ""))
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip().lower()
    if api_key == "ollama":
        os.environ["OPENAI_BASE_URL"] = canonical_openai_compat_base_url("http://localhost:11434")


def mlflow_guidelines_judge_model_uri() -> str | None:
    """Return the judge ``model=`` URI for ``mlflow.genai.scorers.Guidelines``, if any override applies.

    MLflow defaults to ``openai:/gpt-4.1-mini`` (see ``mlflow.genai.judges.utils.get_default_model``).
    When using local Ollama, use the same model tag as ``build_chat_model`` so the scorer does not
    request a nonexistent hosted model name from the local OpenAI-compat server.

    Overrides: ``MLFLOW_GENAI_GUIDELINES_MODEL``, then ``MLFLOW_JUDGE_MODEL``.
    """
    for env_name in ("MLFLOW_GENAI_GUIDELINES_MODEL", "MLFLOW_JUDGE_MODEL"):
        uri = os.getenv(env_name, "").strip()
        if uri:
            return uri

    if os.getenv("OLLAMA_BASE_URL", "").strip():
        tag = os.getenv("OLLAMA_MODEL", "qwen3.6:latest").strip()
        return f"openai:/{tag}" if tag else None

    api_key_judge = (os.getenv("OPENAI_API_KEY") or "").strip().lower()
    if api_key_judge == "ollama":
        tag = os.getenv("OLLAMA_MODEL", "qwen3.6:latest").strip()
        return f"openai:/{tag}" if tag else None

    return None


def build_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at the local Ollama OpenAI-compatible endpoint.

    Ollama exposes an OpenAI-compatible API at ``$OLLAMA_BASE_URL/v1``. By using
    ``langchain_openai.ChatOpenAI`` (which wraps the ``openai`` SDK), every LLM
    call is automatically captured by ``mlflow.langchain.autolog()`` (the
    pipeline's single source of trace truth).

    ``timeout`` and ``max_retries`` defend against an Ollama protocol corner
    case we hit in practice: when langgraph fans out parallel tool calls and
    the model later returns a long structured-output response, the openai HTTP
    client occasionally never observes the end of the stream and hangs on an
    idle keep-alive socket forever. Bounding both keeps the pipeline lively
    and surfaces the failure as a normal exception in the agent error path
    instead of an indefinite stall.

    Live token streaming to the terminal is enabled when ``OLLAMA_STREAM_STDOUT``
    is ``1``/``true``/``yes`` (or when the orchestrator sets it for
    ``--stream-pipeline-output``). Handlers are attached on this runnable because
    ``create_agent`` calls ``bound_model.invoke(messages)`` without propagating
    the agent graph's invoke callbacks.
    """
    base_url = ollama_openai_compatible_base_url()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"

    request_timeout_s = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "600"))
    max_retries = int(os.getenv("OLLAMA_REQUEST_MAX_RETRIES", "1"))
    max_tokens_raw = os.getenv("OLLAMA_MAX_TOKENS", "").strip()
    max_tokens = int(max_tokens_raw) if max_tokens_raw else None
    extra_body_raw = os.getenv("OLLAMA_EXTRA_BODY_JSON", "").strip()
    extra_body = json.loads(extra_body_raw) if extra_body_raw else None

    stream_stdout = ollama_stream_to_stdout_enabled()

    stream_callbacks = [OllamaLiveStreamHandler()] if stream_stdout else None

    return _OllamaCompatStreamChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=os.getenv("OLLAMA_MODEL", "qwen3.6:latest"),
        temperature=temperature,
        timeout=request_timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        extra_body=extra_body,
        streaming=stream_stdout,
        callbacks=stream_callbacks,
    )


def extract_tool_trace(messages: list[Any]) -> list[str]:
    """Walk a LangChain agent ``result["messages"]`` list and return tool call names in order.

    Used by the orchestrator's per-stage ``tool_trace`` field. Duplicates are removed
    while preserving first-seen order so the trace stays compact.
    """
    seen: list[str] = []
    for message in messages or []:
        if getattr(message, "type", None) != "ai":
            continue

        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            additional = getattr(message, "additional_kwargs", None) or {}
            raw_calls = additional.get("tool_calls", []) if isinstance(additional, dict) else []
            tool_calls = []
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    continue
                function_data = raw_call.get("function", {})
                if isinstance(function_data, dict):
                    name = function_data.get("name")
                    if isinstance(name, str):
                        tool_calls.append({"name": name})

        for tool_call in tool_calls:
            name = ""
            if isinstance(tool_call, dict):
                name = str(tool_call.get("name", "")).strip()
            else:
                name = str(getattr(tool_call, "name", "")).strip()
            if name and name not in seen:
                seen.append(name)
    return seen
