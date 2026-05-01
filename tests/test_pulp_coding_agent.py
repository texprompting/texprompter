from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
)


def test_pulp_coding_agent_passes_static_context_without_tools(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from agents import Pulp_Coding_Agent as scripting_agent
    from agents.prompts import PromptLoadResult

    captured: dict[str, Any] = {}

    class FakeAgent:
        def invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [],
                "structured_response": ScriptingRecommendation(
                    code="print('ok')",
                    output_schema={"solution_status": "str"},
                    successful_implementation=True,
                    missing_info=[],
                    additional_info=[],
                ),
            }

    def fake_create_agent(**kwargs: Any) -> FakeAgent:
        captured["create_agent_kwargs"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(scripting_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(scripting_agent, "build_chat_model", lambda: object())
    monkeypatch.setattr(
        scripting_agent,
        "load_system_prompt_result",
        lambda _name: PromptLoadResult(
            short_name="scripting",
            registry_name="texprompter.scripting",
            requested_uri="prompts:/texprompter.scripting@latest",
            resolved_uri="prompts:/texprompter.scripting/1",
            version="1",
            template="system prompt",
            source="registry",
        ),
    )
    monkeypatch.setattr(scripting_agent, "get_test_outputs_dir", lambda: tmp_path)

    result = scripting_agent.run_pulp_coding_agent(
        csv_file_path="data/optimization_pipeline_test_easy.csv",
        modelling=ModellingRecommendation(
            col_names_used=["Product_ID"],
            parameters=[],
            variables=[],
            minimizing_problem=False,
            objective_function="max z",
            constraint_functions=["x <= 10"],
            explanation_of_ILP=["Test"],
            readable_documentation="# Model",
        ),
        preprocessing=PreprocessingRecommendation(
            input_schema_payload={"columns": ["Product_ID"]},
            mapper_script="def map_data(df):\n    return {}",
            mapping_notes=["Product_ID is the index"],
            assumptions=[],
        ),
        input_schema_payload={"columns": ["Product_ID"]},
        return_debug=True,
    )

    create_agent_kwargs = captured["create_agent_kwargs"]
    assert create_agent_kwargs["tools"] == []
    assert create_agent_kwargs["response_format"].handle_errors is False
    assert result["result"]["successful_implementation"] is True
    assert result["debug"]["milestones"][-1]["event"] == "model_request_complete"

    user_content = captured["payload"]["messages"][0]["content"]
    assert "get_mathematical_model" not in user_content
    assert '"mathematical_model"' in user_content
    assert '"input_schema_payload"' in user_content
    assert '"requested_output_schema"' in user_content
