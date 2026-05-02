from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from agents.prompts import load_system_prompt_result
from agents.shared import (
    _last_ai_content,
    build_chat_model,
    extract_tool_trace,
    get_data_dir,
    get_test_outputs_dir,
    invoke_agent_with_prompt_trace,
    load_csv_input_schema,
    prompt_debug_payload,
)
from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
)


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


# Maximum number of characters allowed in the scripting agent's user message.
# When the assembled context exceeds this limit, sections are dropped in order:
# 1. input_schema_payload (largest, least essential to the solver logic)
# 2. preprocessing.mapper_script (large, but already summarised by mapping_notes)
# 3. mathematical_model.readable_documentation (verbose; objective + constraints remain)
# Set SCRIPTING_MAX_CONTEXT_CHARS=0 to disable truncation entirely.
_SCRIPTING_MAX_CONTEXT_CHARS = int(os.getenv("SCRIPTING_MAX_CONTEXT_CHARS", "24000"))


class ScriptingAgentError(RuntimeError):
    """Error raised with partial debug context from the scripting agent."""

    def __init__(
        self,
        message: str,
        *,
        debug: dict[str, Any],
        tool_trace: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.debug = debug
        self.tool_trace = tool_trace or []


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


def _coerce_recommendation(value: Any) -> ScriptingRecommendation:
    if isinstance(value, ScriptingRecommendation):
        return value
    if isinstance(value, BaseModel):
        return ScriptingRecommendation.model_validate(value.model_dump())
    if isinstance(value, dict):
        return ScriptingRecommendation.model_validate(value)
    raise TypeError(f"Unexpected structured_response type: {type(value)!r}")


def _requested_output_schema() -> dict[str, str]:
    return {
        "solution_status": "str",
        "objective_value": "float",
        "decision_variables": "dict[str, float]",
        "solver_message": "str",
    }


def _preprocessing_payload(
    preprocessing: PreprocessingRecommendation | dict[str, Any] | None,
) -> dict[str, Any]:
    if preprocessing is None:
        return {}
    if isinstance(preprocessing, PreprocessingRecommendation):
        return {
            "mapper_script": preprocessing.mapper_script,
            "mapping_notes": preprocessing.mapping_notes,
            "assumptions": preprocessing.assumptions,
        }
    preprocessing_dict = dict(preprocessing)
    return {
        "mapper_script": str(preprocessing_dict.get("mapper_script", "")),
        "mapping_notes": list(preprocessing_dict.get("mapping_notes", [])),
        "assumptions": list(preprocessing_dict.get("assumptions", [])),
    }


def _build_scripting_context(
    *,
    csv_path: Path,
    schema_payload: dict[str, Any],
    modelling: ModellingRecommendation | dict[str, Any] | None,
    preprocessing: PreprocessingRecommendation | dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "csv_file_path": str(csv_path),
        "mathematical_model": _build_math_payload(modelling)["mathematical_model"],
        "input_schema_payload": schema_payload,
        "preprocessing": _preprocessing_payload(preprocessing),
        "requested_output_schema": _requested_output_schema(),
    }


def _json_context(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=True, sort_keys=True)


def _truncate_scripting_context(
    context: dict[str, Any],
    *,
    max_chars: int,
    debug: dict[str, Any],
    add_milestone: Any,
) -> tuple[dict[str, Any], str]:
    """Trim the scripting context so that its JSON representation fits within ``max_chars``.

    Sections are dropped in priority order (cheapest to lose first):
    1. ``input_schema_payload`` – large, already encoded in the preprocessing script
    2. ``preprocessing.mapper_script`` – verbose; mapping_notes summarise intent
    3. ``mathematical_model.readable_documentation`` – objective + constraints remain

    Each truncation is recorded as a debug milestone and an MLflow tag so operators
    can identify which runs were affected and by how much.
    """
    import copy

    ctx = copy.deepcopy(context)
    truncations: list[str] = []

    def _fits() -> bool:
        return max_chars <= 0 or len(_json_context(ctx)) <= max_chars

    if not _fits():
        ctx["input_schema_payload"] = {"truncated": True, "reason": "context_size_limit"}
        truncations.append("input_schema_payload")

    if not _fits():
        preprocessing = ctx.get("preprocessing", {})
        if isinstance(preprocessing, dict) and preprocessing.get("mapper_script"):
            preprocessing["mapper_script"] = (
                "# truncated: see mapping_notes for intent"
            )
            ctx["preprocessing"] = preprocessing
            truncations.append("preprocessing.mapper_script")

    if not _fits():
        math_model = ctx.get("mathematical_model", {})
        if isinstance(math_model, dict) and math_model.get("readable_documentation"):
            math_model["readable_documentation"] = (
                "# truncated: see objective_function and constraint_functions"
            )
            ctx["mathematical_model"] = math_model
            truncations.append("mathematical_model.readable_documentation")

    if truncations:
        add_milestone(
            "context_truncated",
            truncated_sections=truncations,
            original_chars=len(_json_context(context)),
            truncated_chars=len(_json_context(ctx)),
        )
        debug["context_truncations"] = truncations
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.set_tag("scripting.context_truncated", ",".join(truncations))
                mlflow.log_metric("scripting.context_chars_truncated", len(_json_context(context)))
        except Exception:
            pass

    return ctx, _json_context(ctx)


def run_pulp_coding_agent(
    csv_file_path: str | None = None,
    modelling: ModellingRecommendation | dict[str, Any] | None = None,
    preprocessing: PreprocessingRecommendation | dict[str, Any] | None = None,
    preview_rows: int = 5,
    input_schema_payload: dict[str, Any] | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Generate PuLP code from state-driven modelling and preprocessing context."""
    started_at = time.time()
    debug: dict[str, Any] = {"milestones": []}
    tool_trace: list[str] = []

    def add_milestone(event: str, **details: Any) -> None:
        entry = {
            "event": event,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        entry.update(details)
        debug["milestones"].append(entry)

    resolved_csv_path = _resolve_csv_path(
        csv_file_path or os.getenv("PIPELINE_CSV_PATH", "optimization_pipeline_test_easy.csv")
    )
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")
    add_milestone("csv_resolved", csv_file_path=str(resolved_csv_path))

    if input_schema_payload is not None:
        schema_payload = input_schema_payload
    elif preprocessing is not None and isinstance(preprocessing, PreprocessingRecommendation):
        schema_payload = preprocessing.input_schema_payload
    elif preprocessing is not None:
        schema_payload = dict(preprocessing).get("input_schema_payload", {})
    else:
        schema_payload = load_csv_input_schema(str(resolved_csv_path), preview_rows)
    add_milestone(
        "context_built",
        schema_columns=len(schema_payload.get("columns", [])),
        schema_rows=schema_payload.get("shape", {}).get("rows"),
    )

    scripting_context = _build_scripting_context(
        csv_path=resolved_csv_path,
        schema_payload=schema_payload,
        modelling=modelling,
        preprocessing=preprocessing,
    )

    # Truncate context if it exceeds the configured limit (schema → mapper → math docs).
    scripting_context, context_json = _truncate_scripting_context(
        scripting_context,
        max_chars=_SCRIPTING_MAX_CONTEXT_CHARS,
        debug=debug,
        add_milestone=add_milestone,
    )

    user_message = (
        "Generate complete Python PuLP code, declared output schema, and execution notes "
        "from this JSON context. Use only the provided context; do not request tools.\n\n"
        f"{context_json}"
    )

    try:
        add_milestone("prompt_load_start")
        prompt = load_system_prompt_result("scripting")
        debug["prompt"] = prompt_debug_payload(prompt)
        add_milestone("prompt_loaded", prompt_chars=len(prompt.template))

        add_milestone("model_build_start")
        model = build_chat_model()
        debug["model"] = {
            "model": getattr(model, "model_name", None) or getattr(model, "model", None),
            "timeout": getattr(model, "request_timeout", None),
            "max_retries": getattr(model, "max_retries", None),
            "max_tokens": getattr(model, "max_tokens", None),
        }
        add_milestone("model_built")

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=prompt.template,
            response_format=ToolStrategy(ScriptingRecommendation, handle_errors=False),
        )

        add_milestone(
            "model_request_start",
            context_chars=len(user_message),
            response_strategy="tool_strategy_no_retry",
        )
        response = invoke_agent_with_prompt_trace(
            agent,
            stage="scripting",
            prompt=prompt,
            user_message=user_message,
            metadata={"response_strategy": "tool_strategy_no_retry"},
        )
        add_milestone("model_request_complete")
        tool_trace = extract_tool_trace(response.get("messages", []))

        structured = response.get("structured_response")
        if structured is None:
            # Fallback: attempt to parse the last AI message text as JSON.
            last_content = _last_ai_content(response.get("messages", []))
            if last_content:
                try:
                    structured = ScriptingRecommendation.model_validate_json(last_content)
                except Exception:
                    pass
        if structured is None:
            raise ValueError("scripting agent did not produce a structured_response.")
    except Exception as error:
        add_milestone("error", error_type=type(error).__name__, error=str(error))
        raise ScriptingAgentError(str(error), debug=debug, tool_trace=tool_trace) from error

    recommendation = _coerce_recommendation(structured)
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
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(recommendation.code, encoding="utf-8")
    except OSError as io_err:
        warnings.warn(
            f"generated_pulp_model.py write failed (non-fatal): {io_err}",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_debug:
        return {
            "result": recommendation.model_dump(),
            "tool_trace": tool_trace,
            "debug": debug,
        }
    return recommendation.model_dump()


if __name__ == "__main__":
    result = run_pulp_coding_agent()
    print(json.dumps(result, indent=2))
