from __future__ import annotations

import os

import pytest

from orchestrator.pipeline import run_agent_node


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OLLAMA_TESTS", "0") != "1",
    reason="Set RUN_LIVE_OLLAMA_TESTS=1 to enable live Ollama integration tests.",
)
def test_live_nodes_can_run_individually() -> None:
    state = {
        "csv_file_path": "data/optimization_pipeline_test_easy.csv",
        "preview_rows": 5,
        "status": "ok",
        "errors": [],
        "traces": [],
    }

    initialized = run_agent_node("initialize", state).model_dump()
    use_case = run_agent_node("use_case", initialized).model_dump()
    modelling = run_agent_node("modeling", use_case).model_dump()
    preprocessing = run_agent_node("preprocessing", modelling).model_dump()
    scripting = run_agent_node("scripting", preprocessing)

    assert scripting.status in {"ok", "error"}
    assert scripting.execution_metadata
