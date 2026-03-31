from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    UseCaseRecommendation,
)

from .shared import build_ollama_model, get_data_dir, get_test_outputs_dir, invoke_structured_agent


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


def _persist_mapper_script(script: str) -> None:
    outputs_dir = get_test_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "generated_csv_mapper.py").write_text(script, encoding="utf-8")


def run_preprocessing_agent(
    csv_file_path: str,
    use_case: UseCaseRecommendation | None,
    modelling: ModellingRecommendation | None,
    preview_rows: int = 5,
) -> PreprocessingRecommendation:
    """Generate csv->input-schema mapping script and preprocessing notes."""
    resolved_csv_path = _resolve_csv_path(csv_file_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

    schema_payload = _load_csv_schema_payload(resolved_csv_path, preview_rows)

    @tool
    def get_csv_input_schema_payload() -> dict[str, Any]:
        """Return the baseline payload produced by csv_to_input_scheme.py."""
        return schema_payload

    @tool
    def get_use_case_contract() -> dict[str, Any]:
        """Return use-case output from previous node."""
        return use_case.model_dump() if use_case is not None else {}

    @tool
    def get_model_contract() -> dict[str, Any]:
        """Return modeling output from previous node."""
        return modelling.model_dump() if modelling is not None else {}

    llm = build_ollama_model()
    agent = create_agent(
        model=llm,
        tools=[get_csv_input_schema_payload, get_use_case_contract, get_model_contract],
        system_prompt=(
            "You are a preprocessing agent. Build a compact solver input schema payload and a Python mapper script that transforms raw CSV into that schema. "
            "Use tool outputs only. Final answer must be a PreprocessingRecommendation tool call only."
        ),
        response_format=PreprocessingRecommendation,
    )

    args = invoke_structured_agent(
        agent=agent,
        user_input=(
            "Generate the preprocessing contract for this optimization flow, including a mapper_script and mapping_notes."
        ),
        response_tool_name="PreprocessingRecommendation",
        retry_message=(
            "Your previous response did not use the required PreprocessingRecommendation tool call. "
            "Return only a valid PreprocessingRecommendation with all required fields."
        ),
        response_model=PreprocessingRecommendation,
    )

    recommendation = PreprocessingRecommendation.model_validate(args)
    _persist_mapper_script(recommendation.mapper_script)
    return recommendation


if __name__ == "__main__":
    result = run_preprocessing_agent(
        csv_file_path="optimization_pipeline_test_easy.csv",
        use_case=None,
        modelling=None,
        preview_rows=5,
    )
    print(result.model_dump_json(indent=2))
