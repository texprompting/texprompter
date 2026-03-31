from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
)

from .shared import (
    build_ollama_model,
    get_data_dir,
    get_test_outputs_dir,
    invoke_structured_agent,
)


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _load_csv_schema_payload(csv_file_path: Path, preview_rows: int) -> dict[str, Any]:
    module_path = get_data_dir() / "csv_to_input_scheme.py"
    spec = importlib.util.spec_from_file_location("csv_to_input_scheme", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load schema module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_input_data"):
        raise AttributeError("csv_to_input_scheme.py does not define get_input_data")

    get_input_data = getattr(module, "get_input_data")
    return get_input_data(csv_file_name=str(csv_file_path), preview_rows=preview_rows)


def _build_math_payload(
    modelling: ModellingRecommendation | None,
) -> dict[str, Any]:
    if modelling is not None:
        return {
            "mathematical_model": {
                "objective_function": modelling.objective_function,
                "constraint_functions": modelling.constraint_functions,
                "readable_documentation": modelling.readable_documentation,
            }
        }

    outputs_dir = get_test_outputs_dir()
    objective = (outputs_dir / "llm_objective_function.md").read_text(
        encoding="utf-8"
    ) if (outputs_dir / "llm_objective_function.md").exists() else ""
    constraints = (outputs_dir / "llm_constraints.md").read_text(
        encoding="utf-8"
    ) if (outputs_dir / "llm_constraints.md").exists() else ""
    documentation = (outputs_dir / "llm_output.md").read_text(
        encoding="utf-8"
    ) if (outputs_dir / "llm_output.md").exists() else ""

    return {
        "mathematical_model": {
            "objective_function": objective.strip(),
            "constraint_functions": [
                line.strip() for line in constraints.splitlines() if line.strip()
            ],
            "readable_documentation": documentation.strip(),
        }
    }


def _persist_code(recommendation: ScriptingRecommendation) -> None:
    outputs_dir = get_test_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    generated_code_path = outputs_dir / "generated_pulp_model.py"
    generated_code_path.write_text(recommendation.code, encoding="utf-8")


def run_scripting_agent(
    csv_file_path: str,
    modelling: ModellingRecommendation | None,
    preprocessing: PreprocessingRecommendation | None,
    preview_rows: int = 5,
) -> ScriptingRecommendation:
    """Generate PuLP solver code and output schema based on modeling and schema payloads."""
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    @tool
    def get_input_schema_payload() -> dict[str, Any]:
        """Return solver input schema payload from preprocessing or shared utility."""
        if preprocessing is not None and preprocessing.input_schema_payload:
            return preprocessing.input_schema_payload

        return _load_csv_schema_payload(resolved_csv_path, preview_rows)

    @tool
    def get_mathematical_model() -> dict[str, Any]:
        """Return the mathematical model from previous node output."""
        return _build_math_payload(modelling)

    @tool
    def get_requested_output_schema() -> dict[str, str]:
        """Return the required top-level output schema of the solver function."""
        return {
            "solution_status": "str",
            "objective_value": "float",
            "decision_variables": "dict[str, float]",
            "solver_message": "str",
        }

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[
            get_mathematical_model,
            get_input_schema_payload,
            get_requested_output_schema,
        ],
        system_prompt=(
            "You are a PuLP MILP coding agent. Convert the provided mathematical model into runnable Python PuLP code. "
            "Use only tool outputs and do not invent unsupported assumptions.\n\n"
            "Required workflow:\n"
            "1. Call get_mathematical_model\n"
            "2. Call get_input_schema_payload\n"
            "3. Call get_requested_output_schema\n\n"
            "Final answer must be a ScriptingRecommendation tool call with fields: "
            "code, output_schema, successful_implementation, missing_info, additional_info."
        ),
        response_format=ScriptingRecommendation,
    )

    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Generate complete Python PuLP code and the declared output schema. "
            "Include objective, constraints, solve step, and result extraction."
        ),
        response_tool_name="ScriptingRecommendation",
        retry_message=(
            "Your previous response did not use the required ScriptingRecommendation tool call. "
            "Return only a valid ScriptingRecommendation with all required fields."
        ),
        response_model=ScriptingRecommendation,
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

    _persist_code(recommendation)
    return recommendation


if __name__ == "__main__":
    result = run_scripting_agent(
        csv_file_path="optimization_pipeline_test_easy.csv",
        modelling=None,
        preprocessing=None,
        preview_rows=5,
    )
    print(result.model_dump_json(indent=2))
