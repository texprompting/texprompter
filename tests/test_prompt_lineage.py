from __future__ import annotations

import sys
import types
from typing import Any

from agents.prompts import PromptLoadResult, load_system_prompt_result
from agents.shared import agent_invoke_config, prompt_debug_payload


def test_load_system_prompt_uses_latest_alias_and_falls_back(monkeypatch: Any) -> None:
    captured: dict[str, str] = {}

    def fake_load_prompt(uri: str) -> None:
        captured["uri"] = uri
        raise RuntimeError("registry unavailable")

    fake_mlflow = types.SimpleNamespace(genai=types.SimpleNamespace(load_prompt=fake_load_prompt))
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    result = load_system_prompt_result("use_case")

    assert captured["uri"] == "prompts:/texprompter.use_case@latest"
    assert result.source == "local_file"
    assert result.requested_uri == "prompts:/texprompter.use_case@latest"
    assert result.template
    assert "registry unavailable" in (result.fallback_reason or "")


def test_load_system_prompt_records_registry_version(monkeypatch: Any) -> None:
    def fake_load_prompt(uri: str) -> Any:
        assert uri == "prompts:/texprompter.modeling@production"
        return types.SimpleNamespace(template="registered prompt", version=7)

    fake_mlflow = types.SimpleNamespace(genai=types.SimpleNamespace(load_prompt=fake_load_prompt))
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    result = load_system_prompt_result("modeling", version="production")

    assert result.source == "registry"
    assert result.version == "7"
    assert result.resolved_uri == "prompts:/texprompter.modeling/7"
    assert result.template == "registered prompt"


def test_agent_invoke_config_includes_prompt_metadata() -> None:
    prompt = PromptLoadResult(
        short_name="scripting",
        registry_name="texprompter.scripting",
        requested_uri="prompts:/texprompter.scripting@latest",
        resolved_uri="prompts:/texprompter.scripting/3",
        version="3",
        template="system prompt",
        source="registry",
    )

    config = agent_invoke_config(stage="scripting", prompt=prompt)

    assert config["recursion_limit"] > 0
    assert config["metadata"]["agent.stage"] == "scripting"
    assert config["metadata"]["prompt.resolved_uri"] == "prompts:/texprompter.scripting/3"
    assert config["metadata"]["prompt.version"] == "3"
    assert "stage:scripting" in config["tags"]
    assert prompt_debug_payload(prompt)["resolved_uri"] == "prompts:/texprompter.scripting/3"
