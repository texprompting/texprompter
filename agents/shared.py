from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel


load_dotenv()


def get_project_root() -> Path:
    """Return the repository root based on this package location."""
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_test_outputs_dir() -> Path:
    return get_project_root() / "TestOutputs"


def build_ollama_model(reasoning: bool = True) -> ChatOllama:
    """Create a configured ChatOllama model from environment variables."""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    ollama_api_key = os.getenv("OLLAMA_API_KEY")

    headers: dict[str, str] = {}
    if ollama_api_key and ollama_api_key != "your_api_key_here":
        headers["Authorization"] = f"Bearer {ollama_api_key}"

    return ChatOllama(
        base_url=ollama_base_url,
        model=os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q3_K_M"),
        client_kwargs={"headers": headers} if headers else {},
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


def invoke_structured_agent(
    agent: Any,
    user_input: str,
    response_tool_name: str,
    retry_message: str,
    max_retries: int = 3,
    response_model: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Invoke a LangChain agent and enforce a structured response tool call."""
    messages: list[Any] = [{"role": "user", "content": user_input}]
    last_seen_tool_names: list[str] = []
    last_validation_error: str | None = None

    for _attempt in range(max_retries):
        response = agent.invoke({"messages": messages})

        args = _extract_tool_args(response, response_tool_name)
        if args is not None:
            if response_model is None:
                return args
            try:
                validated = response_model.model_validate(args)
            except Exception as validation_error:
                last_validation_error = str(validation_error)
            else:
                return validated.model_dump()

        if response_model is not None:
            fallback_args = _extract_validated_args_from_any_tool(response, response_model)
            if fallback_args is not None:
                return fallback_args

            content_args = _extract_validated_args_from_content(response, response_model)
            if content_args is not None:
                return content_args

        last_seen_tool_names = _collect_seen_tool_names(response)
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
    raise ValueError(
        f"Failed to receive structured output via {response_tool_name} after {max_retries} attempts. "
        f"Seen tool calls: {seen_tools}.{validation_note}"
    )
