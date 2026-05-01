from __future__ import annotations

import os
import time
from typing import cast

import pytest

from orchestrator.pipeline import PipelineStateDict, run_agent_node


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OLLAMA_TESTS", "0") != "1",
    reason="Set RUN_LIVE_OLLAMA_TESTS=1 to enable live Ollama integration tests.",
)
def test_live_nodes_can_run_individually() -> None:
    state: PipelineStateDict = {
        "csv_file_path": "data/optimization_pipeline_test_easy.csv",
        "preview_rows": 5,
        "status": "ok",
        "errors": [],
        "traces": [],
    }

    initialized = cast(PipelineStateDict, run_agent_node("initialize", state).model_dump())
    use_case = cast(PipelineStateDict, run_agent_node("use_case", initialized).model_dump())
    modelling = cast(PipelineStateDict, run_agent_node("modeling", use_case).model_dump())
    preprocessing = cast(PipelineStateDict, run_agent_node("preprocessing", modelling).model_dump())
    scripting = run_agent_node("scripting", preprocessing)

    assert scripting.status in {"ok", "error"}
    assert scripting.execution_metadata
    scripting_metadata = scripting.execution_metadata[-1]
    timeout_s = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_S", "180"))
    max_retries = int(os.getenv("OLLAMA_REQUEST_MAX_RETRIES", "1"))
    assert scripting_metadata.duration_seconds is not None
    assert scripting_metadata.duration_seconds <= (timeout_s * (max_retries + 1)) + 60
    if scripting.status == "error":
        assert any(note.startswith("debug_milestones=") for note in scripting_metadata.notes)


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OLLAMA_TESTS", "0") != "1",
    reason="Set RUN_LIVE_OLLAMA_TESTS=1 to enable live Ollama integration tests.",
)
def test_live_scripting_reports_diagnostics_with_debug_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_S", "15")
    monkeypatch.setenv("OLLAMA_REQUEST_MAX_RETRIES", "0")

    state: PipelineStateDict = {
        "csv_file_path": "data/optimization_pipeline_test_easy.csv",
        "preview_rows": 5,
        "status": "ok",
        "errors": [],
        "traces": [],
    }

    initialized = cast(PipelineStateDict, run_agent_node("initialize", state).model_dump())
    use_case = cast(PipelineStateDict, run_agent_node("use_case", initialized).model_dump())
    modelling = cast(PipelineStateDict, run_agent_node("modeling", use_case).model_dump())
    preprocessing = cast(PipelineStateDict, run_agent_node("preprocessing", modelling).model_dump())

    started_at = time.time()
    scripting = run_agent_node("scripting", preprocessing)
    elapsed = time.time() - started_at
    scripting_metadata = scripting.execution_metadata[-1]

    assert scripting.status in {"ok", "error"}
    assert elapsed <= 75
    assert scripting_metadata.duration_seconds is not None
    assert scripting_metadata.duration_seconds <= 75
    if scripting.status == "error":
        assert any(note.startswith("debug_milestones=") for note in scripting_metadata.notes)
