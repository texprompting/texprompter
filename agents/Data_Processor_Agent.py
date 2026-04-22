from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.shared import (
    build_ollama_model,
    get_data_dir,
    get_test_outputs_dir,
    invoke_structured_agent,
    load_csv_input_schema,
)
from schemas.basemodels import ModellingRecommendation, UseCaseRecommendation


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


class SetDefinition(BaseModel):
    name: str = Field(description="Name of the set (e.g., I, T)")
    description: str = Field(description="What the set represents")
    source_column: str = Field(description="CSV column used to derive the set")
    python_representation: str = Field(description="Python code snippet defining the set")


class ParameterDefinition(BaseModel):
    symbol: str = Field(description="Mathematical symbol (e.g., c_i)")
    description: str = Field(description="Meaning of the parameter")
    source_columns: list[str] = Field(description="CSV columns used to compute this parameter")
    python_representation: str = Field(description="Python dictionary or structure")


class DataPreparation(BaseModel):
    imports: str = Field(description="All required imports")
    data_loading: str = Field(description="Code to load CSV into pandas DataFrame")
    preprocessing_steps: list[str] = Field(
        description="Step-by-step explanation of transformations applied to the data"
    )
    sets: list[SetDefinition] = Field(description="All index sets used in the model")
    parameters: list[ParameterDefinition] = Field(
        description="All parameters derived from CSV data"
    )
    data_structures_ready: bool = Field(
        description="Final combined Python structures ready for optimization"
    )
    mapping_explanation: list[str] = Field(
        description="Explanation of how CSV columns map to mathematical symbols"
    )
    assumptions: list[str] = Field(description="Any assumptions made during data preparation")
    full_script: str = Field(description="Complete Python script that prepares the data")


def _resolve_csv_path(csv_file_path: str) -> Path:
    csv_path = Path(csv_file_path)
    if csv_path.is_absolute():
        return csv_path

    data_path = get_data_dir() / csv_file_path
    if data_path.exists():
        return data_path

    return csv_path.resolve()


def _build_model_payload(modelling: ModellingRecommendation | dict[str, Any] | None) -> dict[str, Any]:
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


def run_data_processor_agent(
    csv_file_path: str | None = None,
    use_case: UseCaseRecommendation | dict[str, Any] | None = None,
    modelling: ModellingRecommendation | dict[str, Any] | None = None,
    preview_rows: int = 5,
    input_schema_payload: dict[str, Any] | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate the legacy DataPreparation contract from state-driven context."""
    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "optimization_pipeline_easy.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    df = pd.read_csv(resolved_csv_path)
    schema_payload = input_schema_payload or load_csv_input_schema(
        csv_file_path=str(resolved_csv_path),
        preview_rows=preview_rows,
    )

    @tool
    def get_column_names() -> list[str]:
        """Returns the column names of the selected CSV."""
        return [str(column) for column in df.columns.tolist()]

    @tool
    def preview_csv() -> str:
        """Returns a compact preview of the selected CSV."""
        return df.head(preview_rows).to_string()

    @tool
    def get_input_schema_payload() -> dict[str, Any]:
        """Returns pipeline input schema payload for preprocessing."""
        return schema_payload

    @tool
    def get_mathematical_model() -> dict[str, Any]:
        """Returns upstream mathematical model contract."""
        return _build_model_payload(modelling)

    @tool
    def get_use_case_recommendation() -> dict[str, Any]:
        """Returns upstream use-case contract for preprocessing context."""
        if use_case is None:
            return {}
        if isinstance(use_case, UseCaseRecommendation):
            return use_case.model_dump()
        return dict(use_case)

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[
            get_mathematical_model,
            get_input_schema_payload,
            preview_csv,
            get_column_names,
            get_use_case_recommendation,
        ],
        system_prompt=(
            "You are an expert in preparing optimization data. Build sets and parameter mappings only.\n\n"
            "Do not build or solve a PuLP model in this stage.\n"
            "Required workflow:\n"
            "1. Call get_mathematical_model\n"
            "2. Call get_input_schema_payload\n"
            "3. Call preview_csv and get_column_names if needed\n"
            "4. Return DataPreparation only via tool call\n"
        ),
        response_format=DataPreparation,
    )

    tool_trace: list[str] = []
    debug_trace: dict[str, Any] = {}
    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Generate Python data preparation code and mapping documentation for the optimization problem."
        ),
        response_tool_name="DataPreparation",
        retry_message=(
            "You failed to output DataPreparation correctly. Return only a valid DataPreparation "
            "tool call with all required fields."
        ),
        response_model=DataPreparation,
        tool_trace=tool_trace,
        debug_trace=debug_trace,
    )

    payload = DataPreparation.model_validate(args).model_dump()
    output_path = get_test_outputs_dir() / "data_preparation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if return_debug:
        return {"result": payload, "tool_trace": tool_trace, "debug": debug_trace}
    return payload


if __name__ == "__main__":
    result = run_data_processor_agent()
    print(json.dumps(result, indent=2))