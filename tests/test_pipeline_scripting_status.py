'''
Ensures scripting_node marks the pipeline
state as error and records a scripting:invalid trace
when the scripting agent reports an unsuccessful implementation.
'''


from __future__ import annotations

from typing import Any

from orchestrator.pipeline import scripting_node
from schemas.basemodels import ScriptingRecommendation


def test_scripting_node_sets_error_when_unsuccessful(monkeypatch: Any) -> None:
    def fake_run_scripting_agent(**_kwargs: Any) -> ScriptingRecommendation:
        return ScriptingRecommendation(
            code="print('broken')",
            output_schema={"solution_status": "str"},
            successful_implementation=False,
            missing_info=[],
            additional_info=["Generated code has syntax errors."],
        )

    monkeypatch.setattr("orchestrator.pipeline.run_scripting_agent", fake_run_scripting_agent)

    result = scripting_node(
        {
            "csv_file_path": "optimization_pipeline_test_easy.csv",
            "preview_rows": 5,
            "status": "ok",
            "errors": [],
            "traces": [],
        }
    )

    assert result["status"] == "error"
    assert result["traces"][-1] == "scripting:invalid"
    assert result["errors"][-1]["agent_name"] == "scripting_agent"
    assert "syntax errors" in result["errors"][-1]["message"]
