from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """Symbolic model parameter used in the MILP formulation."""

    symbol: str = Field(description="LaTeX symbol, for example v_t or T.")
    description: str = Field(description="Meaning of the parameter in the production system.")


class VariableDefinition(BaseModel):
    """Decision variable definition used by the mathematical model."""

    variable: str = Field(description="LaTeX decision variable, for example x_{it}.")
    meaning: str = Field(description="Meaning of the variable in the system.")


class UseCaseRecommendation(BaseModel):
    """Output contract for the use-case discovery agent."""

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


class ModellingRecommendation(BaseModel):
    """Output contract for the MILP modeling agent."""

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


class PreprocessingRecommendation(BaseModel):
    """Output contract for the preprocessing and mapping agent."""

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


class ScriptingRecommendation(BaseModel):
    """Output contract for the solver scripting agent."""

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


class AgentError(BaseModel):
    """Standardized error payload for a failed node execution."""

    agent_name: str
    message: str
    detail: str | None = None


class PipelineState(BaseModel):
    """Shared LangGraph state passed between all pipeline nodes."""

    csv_file_path: str
    preview_rows: int = 5
    status: Literal["ok", "error"] = "ok"
    use_case: UseCaseRecommendation | None = None
    modelling: ModellingRecommendation | None = None
    preprocessing: PreprocessingRecommendation | None = None
    scripting: ScriptingRecommendation | None = None
    errors: list[AgentError] = Field(default_factory=list)
    traces: list[str] = Field(default_factory=list)
