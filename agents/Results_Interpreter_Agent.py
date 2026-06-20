from __future__ import annotations

import json
import warnings
from typing import Any

from langchain.agents import create_agent
from pydantic import BaseModel

from agents.prompts import load_system_prompt_result
from agents.shared import (
    _last_ai_content,
    build_chat_model,
    extract_tool_trace,
    invoke_agent_with_prompt_trace,
    prompt_debug_payload,
)
from schemas.basemodels import (
    ModellingRecommendation,
    ResultsInterpretationRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
)


warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


def _build_interpretation_context(
    use_case: UseCaseRecommendation | dict[str, Any] | None,
    modelling: ModellingRecommendation | dict[str, Any] | None,
    scripting: ScriptingRecommendation | dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the JSON context payload sent to the interpretation agent."""

    # --- use case ---
    uc_payload: dict[str, Any] = {}
    if use_case is not None:
        uc_dict = use_case.model_dump() if hasattr(use_case, "model_dump") else dict(use_case)
        uc_payload = {
            "use_case_name": uc_dict.get("use_case_name", ""),
            "business_goal": uc_dict.get("business_goal", ""),
            "objective_direction": uc_dict.get("objective_direction", ""),
            "objective_variable": uc_dict.get("objective_variable", ""),
        }

    # --- mathematical model ---
    model_payload: dict[str, Any] = {}
    if modelling is not None:
        md_dict = modelling.model_dump() if hasattr(modelling, "model_dump") else dict(modelling)
        model_payload = {
            "minimizing_problem": md_dict.get("minimizing_problem", True),
            "objective_function": md_dict.get("objective_function", ""),
            "constraint_functions": md_dict.get("constraint_functions", []),
            "explanation_of_ILP": md_dict.get("explanation_of_ILP", []),
        }

    # --- solver results ---
    solver_payload: dict[str, Any] = {}
    if scripting is not None:
        sc_dict = scripting.model_dump() if hasattr(scripting, "model_dump") else dict(scripting)
        solver_payload = {
            "solution_status": sc_dict.get("solution_status", ""),
            "objective_value": sc_dict.get("objective_value"),
            "decision_variables": sc_dict.get("decision_variables", {}),
            "solver_message": sc_dict.get("solver_message", ""),
            "successful_implementation": sc_dict.get("successful_implementation", False),
        }

    return {
        "use_case": uc_payload,
        "mathematical_model": model_payload,
        "solver_results": solver_payload,
    }


def run_results_interpreter_agent(
    use_case: UseCaseRecommendation | dict[str, Any] | None = None,
    modelling: ModellingRecommendation | dict[str, Any] | None = None,
    scripting: ScriptingRecommendation | dict[str, Any] | None = None,
    return_debug: bool = False,
) -> dict[str, Any]:
    """Interpret solver results and return a natural-language analysis."""

    context = _build_interpretation_context(use_case, modelling, scripting)
    context_json = json.dumps(context, indent=2, default=str)

    prompt = load_system_prompt_result("results_interpretation")
    agent = create_agent(
        model=build_chat_model(),
        tools=[],
        system_prompt=prompt.template,
        response_format=ResultsInterpretationRecommendation,
    )

    user_message = (
        "Interpret the following optimization results and produce a clear, "
        "actionable business analysis. Return ResultsInterpretationRecommendation "
        "only via tool call.\n\n"
        f"{context_json}"
    )

    response = invoke_agent_with_prompt_trace(
        agent,
        stage="results_interpretation",
        prompt=prompt,
        user_message=user_message,
    )

    structured = response.get("structured_response")
    if structured is None:
        # Fallback: attempt to parse the last AI message text as JSON.
        last_content = _last_ai_content(response.get("messages", []))
        if last_content:
            try:
                structured = ResultsInterpretationRecommendation.model_validate_json(last_content)
            except Exception:
                pass
    if structured is None:
        raise ValueError("results_interpreter_agent did not produce a structured_response.")

    if isinstance(structured, BaseModel):
        result = structured.model_dump()
    elif isinstance(structured, dict):
        result = ResultsInterpretationRecommendation.model_validate(structured).model_dump()
    else:
        result = ResultsInterpretationRecommendation.model_validate(
            structured.model_dump() if hasattr(structured, "model_dump") else structured
        ).model_dump()

    if return_debug:
        tool_trace = extract_tool_trace(response.get("messages", []))
        return {"result": result, "tool_trace": tool_trace, "debug": {"prompt": prompt_debug_payload(prompt)}}
    return result


if __name__ == "__main__":
    # Quick standalone test with mock data
    mock_scripting = {
        "code": "",
        "solution_status": "Optimal",
        "objective_value": 12345.67,
        "decision_variables": {"x_A": 100.0, "x_B": 50.0, "x_C": 0.0},
        "solver_message": "Optimal solution found",
        "successful_implementation": True,
    }
    mock_use_case = {
        "use_case_name": "production_optimization",
        "business_goal": "Maximize total production profit",
        "objective_direction": "max",
        "objective_variable": "profit",
    }
    mock_modelling = {
        "minimizing_problem": False,
        "objective_function": "max sum(profit_i * x_i)",
        "constraint_functions": ["sum(x_i) <= capacity", "x_i >= 0"],
        "explanation_of_ILP": ["Maximize profit subject to capacity"],
    }

    result = run_results_interpreter_agent(
        use_case=mock_use_case,
        modelling=mock_modelling,
        scripting=mock_scripting,
    )
    print(json.dumps(result, indent=2))
