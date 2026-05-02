from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_json_collection(value: Any) -> Any:
    """Coerce a JSON-encoded ``list``/``dict`` string into the underlying Python object.

    Local LLMs (especially Qwen via the OpenAI-compatible Ollama endpoint) regularly
    emit ``list[...]`` / ``dict[...]`` fields as JSON-encoded strings instead of real
    arrays/objects. Without coercion, pydantic rejects them with ``list_type`` /
    ``dict_type`` errors, which sends ``langchain.agents.create_agent`` into a
    structured-output retry loop that can hang the pipeline. This validator is
    registered in ``mode='before'`` on every list/dict field of an agent output
    schema so the very first attempt parses cleanly.

    Behaviour:
    - Real ``list``/``dict``/``None`` values pass through unchanged.
    - A ``str`` that JSON-decodes into a ``list``/``dict`` is decoded.
    - Anything else is returned unchanged so pydantic can produce its normal error.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, (list, dict)):
        return decoded
    return value


def _coerce_bool(value: Any) -> Any:
    """Coerce bool-as-string emitted by local LLMs (``"True"`` / ``"False"``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return value


# ---------------------------------------------------------------------------
# Stall classification
# ---------------------------------------------------------------------------


class StallReason(str, Enum):
    """Enumeration of known root causes for agent pipeline stalls.

    Each value maps to an MLflow metric/tag key so stall distributions can be
    aggregated and optimised against across runs.
    """

    RECURSION_LIMIT = "recursion_limit"
    STRUCTURED_OUTPUT_RETRY = "structured_output_retry"
    VALIDATION_ERROR = "validation_error"
    PARSE_FAILURE = "parse_failure"
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_ERROR = "network_error"
    PROMPT_LOAD_FAILURE = "prompt_load_failure"
    TOKEN_OVERFLOW = "token_overflow"
    TOOL_CALL_FAILURE = "tool_call_failure"
    FILE_IO_ERROR = "file_io_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Output schemas (agent structured-output contracts)
# ---------------------------------------------------------------------------


class Parameter(BaseModel):
    """Symbolic model parameter used in the MILP formulation."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(description="LaTeX symbol, for example v_t or T.")
    description: str = Field(description="Meaning of the parameter in the production system.")


class VariableDefinition(BaseModel):
    """Decision variable definition used by the mathematical model."""

    model_config = ConfigDict(extra="ignore")

    variable: str = Field(description="LaTeX decision variable, for example x_{it}.")
    meaning: str = Field(description="Meaning of the variable in the system.")


def _coerce_nested_model_list(value: Any) -> Any:
    """Coerce a list whose items may be JSON-encoded strings into real dicts.

    Some local LLMs serialise a ``list[SomeModel]`` as a list where individual
    items are JSON strings rather than dicts. This runs *after* the outer
    list has been decoded by ``_coerce_json_collection``, so ``value`` is
    already a list at this point (or something else that Pydantic will reject).
    """
    if not isinstance(value, list):
        return _coerce_json_collection(value)
    result = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped and stripped[0] in "[{":
                try:
                    decoded = json.loads(stripped)
                    result.append(decoded)
                    continue
                except json.JSONDecodeError:
                    pass
        result.append(item)
    return result


class UseCaseRecommendation(BaseModel):
    """Output contract for the use-case discovery agent."""

    model_config = ConfigDict(extra="ignore")

    use_case_name: str = Field(description="Short name of the selected optimization use case.")
    business_goal: str = Field(description="Business goal in plain language.")
    objective_direction: Literal["min", "max"] = Field(
        description="Optimization direction, either min or max."
    )
    objective_variable: str = Field(description="Main variable or KPI to optimize.")
    decision_variables: list[str] = Field(
        default_factory=list,
        description="Variables the solver can adjust.",
    )
    required_columns: list[str] = Field(
        default_factory=list,
        description="CSV columns required to run this use case.",
    )
    constraints_to_consider: list[str] = Field(
        default_factory=list,
        description="Important operational constraints to include later.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions that compensate for missing data.",
    )
    rationale: str = Field(description="Why this is the best available optimization use case.")

    @field_validator("objective_direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> Any:
        """Map verbose direction strings emitted by local LLMs to the Literal values."""
        if isinstance(value, str):
            mapping = {
                "minimize": "min",
                "minimise": "min",
                "minimization": "min",
                "minimisation": "min",
                "maximum": "max",
                "maximize": "max",
                "maximise": "max",
                "maximization": "max",
                "maximisation": "max",
                "minimum": "min",
            }
            return mapping.get(value.strip().lower(), value)
        return value

    @field_validator(
        "decision_variables",
        "required_columns",
        "constraints_to_consider",
        "assumptions",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, value: Any) -> Any:
        return _coerce_json_collection(value)


class ModellingRecommendation(BaseModel):
    """Output contract for the MILP modeling agent."""

    model_config = ConfigDict(extra="ignore")

    col_names_used: list[str] = Field(
        default_factory=list,
        description="CSV columns needed by this mathematical model.",
    )
    parameters: list[Parameter] = Field(
        default_factory=list,
        description="Sets, indices, and fixed coefficients.",
    )
    variables: list[VariableDefinition] = Field(
        default_factory=list,
        description="All decision variables used in the model.",
    )
    minimizing_problem: bool = Field(
        description="True for minimization, False for maximization."
    )
    objective_function: str = Field(
        description="MILP objective function in pseudo-LaTeX."
    )
    constraint_functions: list[str] = Field(
        default_factory=list,
        description="List of constraints in pseudo-LaTeX.",
    )
    explanation_of_ILP: list[str] = Field(
        default_factory=list,
        description="Natural language explanations for objective and constraints.",
    )
    readable_documentation: str = Field(
        description="Single markdown block with complete model documentation."
    )

    @field_validator("minimizing_problem", mode="before")
    @classmethod
    def _coerce_minimizing_bool(cls, value: Any) -> Any:
        return _coerce_bool(value)

    @field_validator("col_names_used", "constraint_functions", "explanation_of_ILP", mode="before")
    @classmethod
    def _coerce_str_list_fields(cls, value: Any) -> Any:
        return _coerce_json_collection(value)

    @field_validator("parameters", "variables", mode="before")
    @classmethod
    def _coerce_nested_list_fields(cls, value: Any) -> Any:
        return _coerce_nested_model_list(value)


class PreprocessingRecommendation(BaseModel):
    """Output contract for the preprocessing and mapping agent."""

    model_config = ConfigDict(extra="ignore")

    input_schema_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Solver-ready input schema payload derived from CSV metadata.",
    )
    mapper_script: str = Field(
        description="Python script that maps raw CSV data into solver input format."
    )
    mapping_notes: list[str] = Field(
        default_factory=list,
        description="Notes about transformations and edge cases.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Fallback assumptions used in preprocessing.",
    )

    @field_validator(
        "input_schema_payload",
        "mapping_notes",
        "assumptions",
        mode="before",
    )
    @classmethod
    def _coerce_collection_fields(cls, value: Any) -> Any:
        return _coerce_json_collection(value)


class ScriptingRecommendation(BaseModel):
    """Output contract for the solver scripting agent."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Generated PuLP solver code.")
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Declared structure of the solver output payload.",
    )
    successful_implementation: bool = Field(
        description="True if a runnable implementation could be generated."
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Data or assumptions still needed for full fidelity.",
    )
    additional_info: list[str] = Field(
        default_factory=list,
        description="Extra diagnostics, warnings, or simplifications.",
    )

    @field_validator("successful_implementation", mode="before")
    @classmethod
    def _coerce_bool_field(cls, value: Any) -> Any:
        return _coerce_bool(value)

    @field_validator(
        "output_schema",
        "missing_info",
        "additional_info",
        mode="before",
    )
    @classmethod
    def _coerce_collection_fields(cls, value: Any) -> Any:
        return _coerce_json_collection(value)


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------


class AgentError(BaseModel):
    """Standardized error payload for a failed node execution."""

    agent_name: str
    message: str
    detail: str | None = None
    stall_reason: StallReason = StallReason.UNKNOWN
    """Classified root cause of the stall / failure (logged to MLflow)."""
    retry_steps_used: int | None = None
    """Number of LangGraph steps consumed before the failure (when available)."""
    validation_field: str | None = None
    """The Pydantic field that triggered a validation error, if applicable."""
    context_chars: int | None = None
    """Character count of the user message / context sent to the agent."""


class AgentExecutionMetadata(BaseModel):
    """Execution metadata for one agent stage, used for tracing and MLflow logging."""

    agent_name: str
    started_at: str
    completed_at: str | None = None
    duration_seconds: float | None = None
    status: Literal["ok", "error"]
    tool_calls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    steps_used: int | None = None
    """LangGraph steps consumed during this agent invocation."""
    context_chars: int | None = None
    """Character count of the user message sent to this agent."""
    prompt_source: str | None = None
    """Where the system prompt was loaded from: ``'registry'`` or ``'local_file'``."""


class PipelineState(BaseModel):
    """Shared LangGraph state passed between all pipeline nodes."""

    csv_file_path: str
    preview_rows: int = 5
    status: Literal["ok", "error"] = "ok"
    input_schema_payload: dict[str, Any] = Field(default_factory=dict)
    use_case: UseCaseRecommendation | None = None
    modelling: ModellingRecommendation | None = None
    preprocessing: PreprocessingRecommendation | None = None
    scripting: ScriptingRecommendation | None = None
    errors: list[AgentError] = Field(default_factory=list)
    traces: list[str] = Field(default_factory=list)
    llm_artifacts: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: list[AgentExecutionMetadata] = Field(default_factory=list)
    skip_stages: list[str] = Field(default_factory=list)
    retry_config: dict[str, int] = Field(default_factory=dict)
    llm_config: dict[str, str] = Field(default_factory=dict)
