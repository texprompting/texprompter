from __future__ import annotations

import json
import os
import re
import importlib.util
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel


load_dotenv()


class StructuredAgentInvocationError(ValueError):
    """Raised when a structured agent call fails after retries."""

    def __init__(
        self,
        message: str,
        *,
        response_tool_name: str,
        seen_tool_names: list[str],
        debug_trace: dict[str, Any],
        last_validation_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.response_tool_name = response_tool_name
        self.seen_tool_names = seen_tool_names
        self.debug_trace = debug_trace
        self.last_validation_error = last_validation_error


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


def build_ollama_model(reasoning: bool = True) -> ChatOllama:
    """Create a configured ChatOllama model from environment variables."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_api_key = os.getenv("OLLAMA_API_KEY")

    headers: dict[str, str] = {}
    if ollama_api_key and ollama_api_key != "your_api_key_here":
        headers["Authorization"] = f"Bearer {ollama_api_key}"

    client_kwargs: dict[str, Any] = {"verify": False}
    if headers:
        client_kwargs["headers"] = headers

    return ChatOllama(
        base_url=ollama_base_url,
        model=os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q3_K_M"),
        client_kwargs=client_kwargs,
        reasoning=reasoning,
    )


def _normalize_tool_name(name: str) -> str:
    return name.strip().lower()


def _get_ai_messages(response: dict[str, Any]) -> list[Any]:
    return [
        message
        for message in response.get("messages", [])
        if getattr(message, "type", None) == "ai"
    ]


def _get_message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if hasattr(message, "tool_calls") and message.tool_calls:
        return list(message.tool_calls)

    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    raw_calls = additional_kwargs.get("tool_calls")
    if not raw_calls:
        return []

    normalized_calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue

        function_data = raw_call.get("function", {})
        if not isinstance(function_data, dict):
            continue

        name = function_data.get("name")
        raw_args = function_data.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (TypeError, ValueError):
            args = {}

        if isinstance(name, str) and name.strip():
            normalized_calls.append({"name": name, "args": args})

    return normalized_calls


def _extract_tool_args(response: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    target_name = _normalize_tool_name(tool_name)

    for message in reversed(_get_ai_messages(response)):
        for tool_call in _get_message_tool_calls(message):
            call_name = _normalize_tool_name(str(tool_call.get("name", "")))
            if call_name != target_name:
                continue

            args = tool_call.get("args", {})
            if isinstance(args, dict):
                return args

    return None


def _extract_validated_args_from_any_tool(
    response: dict[str, Any],
    response_model: type[BaseModel],
) -> dict[str, Any] | None:
    for message in reversed(_get_ai_messages(response)):
        for tool_call in _get_message_tool_calls(message):
            args = tool_call.get("args")
            if not isinstance(args, dict):
                continue

            try:
                validated = response_model.model_validate(args)
            except Exception:
                continue

            return validated.model_dump()

    return None


def _extract_validated_args_from_content(
    response: dict[str, Any],
    response_model: type[BaseModel],
) -> dict[str, Any] | None:
    ai_messages = _get_ai_messages(response)
    if not ai_messages:
        return None

    content = getattr(ai_messages[-1], "content", None)
    if content is None:
        return None

    text: str
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    chunks.append(maybe_text)
        text = "\n".join(chunks)
    else:
        try:
            text = json.dumps(content)
        except TypeError:
            return None

    candidate_strings: list[str] = [text.strip()]

    for match in re.finditer(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidate_strings.append(match.group(1).strip())

    for match in re.finditer(r"```\s*(.*?)```", text, flags=re.DOTALL):
        candidate_strings.append(match.group(1).strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidate_strings.append(text[first_brace : last_brace + 1].strip())

    seen_candidates: set[str] = set()
    for candidate in candidate_strings:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        try:
            validated = response_model.model_validate(parsed)
        except Exception:
            continue

        return validated.model_dump()

    return None


def _collect_seen_tool_names(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in _get_ai_messages(response):
        for tool_call in _get_message_tool_calls(message):
            name = str(tool_call.get("name", "")).strip()
            if name and name not in names:
                names.append(name)
    return names


def _preview_last_ai_message(response: dict[str, Any], max_chars: int = 1200) -> str:
    ai_messages = _get_ai_messages(response)
    if not ai_messages:
        return ""

    content = getattr(ai_messages[-1], "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    chunks.append(maybe_text)
        text = "\n".join(chunks)
    else:
        try:
            text = json.dumps(content, default=str)
        except Exception:
            text = str(content)

    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}...<truncated>"


def invoke_structured_agent(
    agent: Any,
    user_input: str,
    response_tool_name: str,
    retry_message: str,
    max_retries: int = 3,
    response_model: type[BaseModel] | None = None,
    tool_trace: list[str] | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a LangChain agent and enforce a structured response tool call."""
    messages: list[Any] = [{"role": "user", "content": user_input}]
    last_seen_tool_names: list[str] = []
    last_validation_error: str | None = None
    attempts: list[dict[str, Any]] = []

    def _commit_debug(success: bool, source: str) -> None:
        if debug_trace is None:
            return
        debug_trace.clear()
        debug_trace.update(
            {
                "response_tool_name": response_tool_name,
                "max_retries": max_retries,
                "attempt_count": len(attempts),
                "success": success,
                "success_source": source,
                "last_seen_tool_names": last_seen_tool_names,
                "last_validation_error": last_validation_error,
                "attempts": attempts,
            }
        )

    for attempt_index in range(1, max_retries + 1):
        response = agent.invoke({"messages": messages})
        seen_tool_names = _collect_seen_tool_names(response)
        last_seen_tool_names = seen_tool_names
        attempt_entry: dict[str, Any] = {
            "attempt": attempt_index,
            "seen_tool_names": seen_tool_names,
            "last_ai_message_preview": _preview_last_ai_message(response),
        }

        if tool_trace is not None:
            for seen_name in seen_tool_names:
                if seen_name not in tool_trace:
                    tool_trace.append(seen_name)

        args = _extract_tool_args(response, response_tool_name)
        if args is not None:
            attempt_entry["response_tool_detected"] = True
            if response_model is None:
                attempt_entry["result_source"] = "response_tool"
                attempts.append(attempt_entry)
                _commit_debug(True, "response_tool")
                return args
            try:
                validated = response_model.model_validate(args)
            except Exception as validation_error:
                last_validation_error = str(validation_error)
                attempt_entry["validation_error"] = last_validation_error
            else:
                attempt_entry["result_source"] = "response_tool"
                attempts.append(attempt_entry)
                _commit_debug(True, "response_tool")
                return validated.model_dump()
        else:
            attempt_entry["response_tool_detected"] = False

        if response_model is not None:
            fallback_args = _extract_validated_args_from_any_tool(response, response_model)
            if fallback_args is not None:
                attempt_entry["result_source"] = "fallback_any_tool"
                attempts.append(attempt_entry)
                _commit_debug(True, "fallback_any_tool")
                return fallback_args

            content_args = _extract_validated_args_from_content(response, response_model)
            if content_args is not None:
                attempt_entry["result_source"] = "fallback_content_json"
                attempts.append(attempt_entry)
                _commit_debug(True, "fallback_content_json")
                return content_args

        attempt_entry["result_source"] = "retry"
        attempts.append(attempt_entry)
        feedback = retry_message

        if last_validation_error:
            feedback = (
                f"{retry_message}\n"
                f"Validation failed for {response_tool_name}: {last_validation_error}\n"
                f"Return only a valid {response_tool_name} tool call with correct field types."
            )

        messages = response.get("messages", []) + [{"role": "user", "content": feedback}]

    seen_tools = ", ".join(last_seen_tool_names) if last_seen_tool_names else "none"
    validation_note = f" Last validation error: {last_validation_error}." if last_validation_error else ""
    message = (
        f"Failed to receive structured output via {response_tool_name} after {max_retries} attempts. "
        f"Seen tool calls: {seen_tools}.{validation_note}"
    )
    _commit_debug(False, "none")
    raise StructuredAgentInvocationError(
        message,
        response_tool_name=response_tool_name,
        seen_tool_names=last_seen_tool_names,
        debug_trace={
            "response_tool_name": response_tool_name,
            "max_retries": max_retries,
            "attempt_count": len(attempts),
            "success": False,
            "success_source": "none",
            "last_seen_tool_names": last_seen_tool_names,
            "last_validation_error": last_validation_error,
            "attempts": attempts,
        },
        last_validation_error=last_validation_error,
    )
