from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from agents.shared import (
    build_ollama_model,
    get_data_dir,
    get_test_outputs_dir,
    invoke_structured_agent,
    load_csv_input_schema,
)
from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
)


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _build_math_payload(modelling: ModellingRecommendation | dict[str, Any] | None) -> dict[str, Any]:
    if modelling is None:
        outputs_dir = get_test_outputs_dir()
        objective_path = outputs_dir / "llm_objective_function.md"
        constraints_path = outputs_dir / "llm_constraints.md"
        documentation_path = outputs_dir / "llm_output.md"
        constraints_raw = constraints_path.read_text(encoding="utf-8") if constraints_path.exists() else ""
        return {
            "mathematical_model": {
                "objective_function": objective_path.read_text(encoding="utf-8").strip()
                if objective_path.exists()
                else "",
                "constraint_functions": [
                    line.strip() for line in constraints_raw.splitlines() if line.strip()
                ],
                "readable_documentation": documentation_path.read_text(encoding="utf-8").strip()
                if documentation_path.exists()
                else "",
            }
        }

    if isinstance(modelling, ModellingRecommendation):
        model_dict = modelling.model_dump()
    else:
        model_dict = dict(modelling)

    return {
        "mathematical_model": {
            "objective_function": str(model_dict.get("objective_function", "")).strip(),
            "constraint_functions": [
                str(item).strip()
                for item in model_dict.get("constraint_functions", [])
                if str(item).strip()
            ],
            "readable_documentation": str(model_dict.get("readable_documentation", "")).strip(),
        }
    }


def run_pulp_coding_agent(
    csv_file_path: str | None = None,
    modelling: ModellingRecommendation | dict[str, Any] | None = None,
    preprocessing: PreprocessingRecommendation | dict[str, Any] | None = None,
    preview_rows: int = 5,
    input_schema_payload: dict[str, Any] | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate PuLP code from state-driven modelling and preprocessing context."""
    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "optimization_pipeline_test_easy.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    if input_schema_payload is not None:
        schema_payload = input_schema_payload
    elif preprocessing is not None and isinstance(preprocessing, PreprocessingRecommendation):
        schema_payload = preprocessing.input_schema_payload
    elif preprocessing is not None:
        schema_payload = dict(preprocessing).get("input_schema_payload", {})
    else:
        schema_payload = load_csv_input_schema(str(resolved_csv_path), preview_rows)

    @tool
    def get_input_schema_payload() -> dict[str, Any]:
        """Returns the input schema payload for solver generation."""
        return schema_payload

    @tool
    def get_mathematical_model() -> dict[str, Any]:
        """Returns the upstream mathematical model contract."""
        return _build_math_payload(modelling)

    @tool
    def get_requested_output_schema() -> dict[str, str]:
        """Returns required output payload fields for the generated solver."""
        return {
            "solution_status": "str",
            "objective_value": "float",
            "decision_variables": "dict[str, float]",
            "solver_message": "str",
        }

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[get_mathematical_model, get_input_schema_payload, get_requested_output_schema],
        system_prompt=(
            "You are a PuLP MILP coding agent. Convert the provided mathematical model to runnable PuLP code.\n\n"
            "Required workflow:\n"
            "1. Call get_mathematical_model\n"
            "2. Call get_input_schema_payload\n"
            "3. Call get_requested_output_schema\n"
            "4. Return ScriptingRecommendation only via tool call\n\n"
            "Rules:\n"
            "- Keep the model linear\n"
            "- Preserve all constraints from the mathematical model\n"
            "- Do not output plain text as final answer\n"
        ),
        response_format=ScriptingRecommendation,
    )

    tool_trace: list[str] = []
    debug_trace: dict[str, Any] = {}
    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Generate complete Python PuLP code, declared output schema, and execution notes from tool inputs."
        ),
        response_tool_name="ScriptingRecommendation",
        retry_message=(
            "You failed to output ScriptingRecommendation correctly. Return only a valid "
            "ScriptingRecommendation tool call with all required fields."
        ),
        response_model=ScriptingRecommendation,
        tool_trace=tool_trace,
        debug_trace=debug_trace,
    )

    recommendation = ScriptingRecommendation.model_validate(args)
    additional_info = list(recommendation.additional_info)
    try:
        compile(recommendation.code, "generated_pulp_model.py", "exec")
    except SyntaxError as syntax_error:
        additional_info.append(f"Generated code has syntax errors: {syntax_error}")
        recommendation = recommendation.model_copy(
            update={
                "successful_implementation": False,
                "additional_info": additional_info,
            }
        )

    output_path = get_test_outputs_dir() / "generated_pulp_model.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(recommendation.code, encoding="utf-8")

    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug_trace,
        }
    return recommendation.model_dump()


if __name__ == "__main__":
    result = run_pulp_coding_agent()
    print(json.dumps(result, indent=2))
