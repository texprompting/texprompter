from __future__ import annotations

from typing import Any

from orchestrator.pipeline import initialize_node, run_agent_node
from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
)


def test_initialize_node_populates_input_schema_payload() -> None:
    result = initialize_node(
        {
            "csv_file_path": "data/optimization_pipeline_test_easy.csv",
            "preview_rows": 3,
            "status": "ok",
            "errors": [],
            "traces": [],
        }
    )

    assert result["status"] == "ok"
    assert result["input_schema_payload"]
    assert "initialize:ok" in result["traces"]


def test_run_agent_node_use_case_contract(monkeypatch: Any) -> None:
    def fake_run_use_case_agent(**_kwargs: Any) -> UseCaseRecommendation:
        return UseCaseRecommendation(
            use_case_name="Production Planning",
            business_goal="Maximize profit",
            objective_direction="max",
            objective_variable="profit",
            decision_variables=["x_i"],
            required_columns=["Product_ID", "Profit_Per_Unit"],
            constraints_to_consider=["Machine_Time_per_Unit <= Available_Machine_Time"],
            assumptions=[],
            rationale="Synthetic test response",
        )

    monkeypatch.setattr("orchestrator.pipeline.run_use_case_agent", fake_run_use_case_agent)

    result_state = run_agent_node(
        "use_case",
        {
            "csv_file_path": "data/optimization_pipeline_test_easy.csv",
            "preview_rows": 5,
            "status": "ok",
            "errors": [],
            "traces": [],
        },
    )

    assert result_state.use_case is not None
    assert result_state.use_case.use_case_name == "Production Planning"
    assert result_state.traces[-1] == "use_case:ok"


def test_run_agent_node_modeling_receives_upstream_use_case(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_modeling_agent(**kwargs: Any) -> ModellingRecommendation:
        captured.update(kwargs)
        return ModellingRecommendation(
            col_names_used=["Product_ID"],
            parameters=[],
            variables=[],
            minimizing_problem=False,
            objective_function="max z",
            constraint_functions=["x <= 10"],
            explanation_of_ILP=["Constraint keeps production bounded."],
            readable_documentation="# Model",
        )

    monkeypatch.setattr("orchestrator.pipeline.run_modeling_agent", fake_run_modeling_agent)

    result_state = run_agent_node(
        "modeling",
        {
            "csv_file_path": "data/optimization_pipeline_test_easy.csv",
            "preview_rows": 5,
            "status": "ok",
            "errors": [],
            "traces": [],
            "use_case": {
                "use_case_name": "Production Planning",
                "business_goal": "Maximize profit",
                "objective_direction": "max",
                "objective_variable": "profit",
                "decision_variables": ["x_i"],
                "required_columns": ["Product_ID"],
                "constraints_to_consider": [],
                "assumptions": [],
                "rationale": "Test",
            },
        },
    )

    assert result_state.modelling is not None
    assert result_state.modelling.objective_function == "max z"
    assert captured.get("use_case") is not None
    assert result_state.traces[-1] == "modeling:ok"


def test_run_agent_node_preprocessing_consumes_state(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_preprocessing_agent(**kwargs: Any) -> PreprocessingRecommendation:
        captured.update(kwargs)
        return PreprocessingRecommendation(
            input_schema_payload={"preview": []},
            mapper_script="def map_data(df):\n    return {}",
            mapping_notes=["Mapped Product_ID to set I"],
            assumptions=[],
        )

    monkeypatch.setattr("orchestrator.pipeline.run_preprocessing_agent", fake_run_preprocessing_agent)

    result_state = run_agent_node(
        "preprocessing",
        {
            "csv_file_path": "data/optimization_pipeline_test_easy.csv",
            "preview_rows": 5,
            "status": "ok",
            "errors": [],
            "traces": [],
            "input_schema_payload": {"records": []},
            "use_case": {
                "use_case_name": "Production Planning",
                "business_goal": "Maximize profit",
                "objective_direction": "max",
                "objective_variable": "profit",
                "decision_variables": ["x_i"],
                "required_columns": ["Product_ID"],
                "constraints_to_consider": [],
                "assumptions": [],
                "rationale": "Test",
            },
            "modelling": {
                "col_names_used": ["Product_ID"],
                "parameters": [],
                "variables": [],
                "minimizing_problem": False,
                "objective_function": "max z",
                "constraint_functions": ["x <= 10"],
                "explanation_of_ILP": ["Test"],
                "readable_documentation": "# Model",
            },
        },
    )

    assert result_state.preprocessing is not None
    assert "map_data" in result_state.preprocessing.mapper_script
    assert captured.get("input_schema_payload") == {"records": []}
    assert result_state.traces[-1] == "preprocessing:ok"


def test_run_agent_node_scripting_consumes_state(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_run_scripting_agent(**kwargs: Any) -> ScriptingRecommendation:
        captured.update(kwargs)
        return ScriptingRecommendation(
            code="print('ok')",
            output_schema={"solution_status": "str"},
            successful_implementation=True,
            missing_info=[],
            additional_info=[],
        )

    monkeypatch.setattr("orchestrator.pipeline.run_scripting_agent", fake_run_scripting_agent)

    result_state = run_agent_node(
        "scripting",
        {
            "csv_file_path": "data/optimization_pipeline_test_easy.csv",
            "preview_rows": 5,
            "status": "ok",
            "errors": [],
            "traces": [],
            "input_schema_payload": {"records": []},
            "preprocessing": {
                "input_schema_payload": {"records": []},
                "mapper_script": "def map_data(df):\n    return {}",
                "mapping_notes": [],
                "assumptions": [],
            },
            "modelling": {
                "col_names_used": ["Product_ID"],
                "parameters": [],
                "variables": [],
                "minimizing_problem": False,
                "objective_function": "max z",
                "constraint_functions": ["x <= 10"],
                "explanation_of_ILP": ["Test"],
                "readable_documentation": "# Model",
            },
        },
    )

    assert result_state.scripting is not None
    assert result_state.scripting.successful_implementation is True
    assert captured.get("input_schema_payload") == {"records": []}
    assert result_state.traces[-1] == "scripting:ok"
