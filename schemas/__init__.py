"""Shared Pydantic contracts for cross-agent communication."""

from .basemodels import (
    AgentError,
    ModellingRecommendation,
    Parameter,
    ParameterEstimationRecommendation,
    ParameterRationale,
    ParameterValue,
    PipelineState,
    PreprocessingRecommendation,
    ResultsInterpretationRecommendation,
    ScriptingRecommendation,
    UseCaseRecommendation,
    VariableDefinition,
)

__all__ = [
    "AgentError",
    "ModellingRecommendation",
    "Parameter",
    "ParameterEstimationRecommendation",
    "ParameterRationale",
    "ParameterValue",
    "PipelineState",
    "PreprocessingRecommendation",
    "ResultsInterpretationRecommendation",
    "ScriptingRecommendation",
    "UseCaseRecommendation",
    "VariableDefinition",
]
