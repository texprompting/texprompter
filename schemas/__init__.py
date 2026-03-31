"""Shared Pydantic contracts for cross-agent communication."""

from .basemodels import (
    AgentError,
    ModellingRecommendation,
    Parameter,
    PipelineState,
    PreprocessingRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
    VariableDefinition,
)

__all__ = [
    "AgentError",
    "ModellingRecommendation",
    "Parameter",
    "PipelineState",
    "PreprocessingRecommendation",
    "ScriptingRecommendation",
    "UseCaseRecommendation",
    "VariableDefinition",
]
