"""Orchestrator package for LangGraph pipeline execution.

The pipeline module is loaded lazily to avoid pre-import side effects when
executing ``python -m orchestrator.pipeline``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_pipeline_graph", "run_pipeline"]


def __getattr__(name: str) -> Any:
	if name in __all__:
		pipeline_module = import_module(".pipeline", __name__)
		return getattr(pipeline_module, name)
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
