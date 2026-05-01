"""Deterministic and LLM-judge scorers for the texprompter pipeline evaluation.

All deterministic scorers consume the dict produced by ``PipelineState.model_dump()``
(returned by ``predict_fn`` in ``run_eval.py``).
"""
from __future__ import annotations

from typing import Any

from mlflow.genai.scorers import Guidelines, scorer

from agents.shared import mlflow_guidelines_judge_model_uri
from schemas.basemodels import (
    ModellingRecommendation,
    PreprocessingRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
)


@scorer
def pipeline_ok(outputs: dict[str, Any]) -> bool:
    """The pipeline reached the end without setting ``status='error'``."""
    return isinstance(outputs, dict) and outputs.get("status") == "ok"


@scorer
def all_schemas_valid(outputs: dict[str, Any]) -> bool:
    """Every per-stage payload re-validates against its Pydantic schema."""
    if not isinstance(outputs, dict):
        return False

    pairs = (
        ("use_case", UseCaseRecommendation),
        ("modelling", ModellingRecommendation),
        ("preprocessing", PreprocessingRecommendation),
        ("scripting", ScriptingRecommendation),
    )
    for key, model in pairs:
        payload = outputs.get(key)
        if payload is None:
            return False
        try:
            model.model_validate(payload)
        except Exception:
            return False
    return True


@scorer
def scripting_code_compiles(outputs: dict[str, Any]) -> bool:
    """The scripting agent's generated PuLP code passes ``compile()``."""
    if not isinstance(outputs, dict):
        return False
    scripting = outputs.get("scripting") or {}
    code = scripting.get("code", "") if isinstance(scripting, dict) else ""
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        compile(code, "<eval>", "exec")
    except SyntaxError:
        return False
    return True


objective_aligned_judge = Guidelines(
    name="objective_aligned",
    guidelines=(
        "The pipeline output must contain a `use_case` whose `objective_variable` is consistent "
        "with the `modelling.objective_function` (same KPI / same min-vs-max direction). "
        "If either is missing or contradicts the other, the answer should fail."
    ),
    model=mlflow_guidelines_judge_model_uri(),
)


DETERMINISTIC_SCORERS = (pipeline_ok, all_schemas_valid, scripting_code_compiles)
