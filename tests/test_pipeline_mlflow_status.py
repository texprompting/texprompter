from __future__ import annotations

from typing import Any

import pytest

from orchestrator import pipeline


class FakeActiveRun:
    class Info:
        run_id = "parent-run"
        experiment_id = "3"
        status = "RUNNING"

    class Data:
        tags = {"mlflow.runName": "parent"}

    info = Info()
    data = Data()


class FakeGraph:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.error is not None:
            raise self.error

        output = dict(state)
        if self.result is not None:
            output.update(self.result)
        return output


def _patch_mlflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "start_run": [],
        "end_run": [],
        "tags": {},
        "params": {},
        "metrics": {},
    }

    monkeypatch.setattr(pipeline, "_setup_mlflow", lambda: None)
    monkeypatch.setattr(
        pipeline.mlflow,
        "start_run",
        lambda **kwargs: calls["start_run"].append(kwargs),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "end_run",
        lambda **kwargs: calls["end_run"].append(kwargs),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_tags",
        lambda tags: calls["tags"].update(tags),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_tag",
        lambda key, value: calls["tags"].update({key: value}),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_params",
        lambda params: calls["params"].update(params),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_metric",
        lambda key, value: calls["metrics"].update({key: value}),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_text",
        lambda text, artifact_file: None,
    )
    monkeypatch.setattr(pipeline.mlflow, "active_run", lambda: None)
    return calls


def test_run_pipeline_ends_mlflow_run_as_finished_for_ok_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "build_pipeline_graph",
        lambda: FakeGraph({"status": "ok", "errors": [], "traces": ["initialize:ok"]}),
    )

    final_state = pipeline.run_pipeline(csv_file_path="data/example.csv", preview_rows=3)

    assert final_state.status == "ok"
    assert len(calls["start_run"]) == 1
    assert "nested" not in calls["start_run"][0]
    assert calls["tags"]["pipeline.status"] == "ok"
    assert calls["metrics"]["errors_count"] == 0.0
    assert calls["end_run"] == [{"status": "FINISHED"}]


def test_run_pipeline_ends_mlflow_run_as_failed_for_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "build_pipeline_graph",
        lambda: FakeGraph(
            {
                "status": "error",
                "errors": [
                    {
                        "agent_name": "scripting_agent",
                        "message": "Request timed out.",
                        "detail": "APITimeoutError('Request timed out.')",
                    }
                ],
                "traces": ["scripting:error"],
            }
        ),
    )

    final_state = pipeline.run_pipeline(csv_file_path="data/example.csv", preview_rows=3)

    assert final_state.status == "error"
    assert calls["tags"]["pipeline.status"] == "error"
    assert calls["metrics"]["errors_count"] == 1.0
    assert calls["end_run"] == [{"status": "FAILED"}]


def test_run_pipeline_starts_nested_run_when_parent_run_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_mlflow(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "build_pipeline_graph",
        lambda: FakeGraph({"status": "ok", "errors": [], "traces": ["initialize:ok"]}),
    )
    monkeypatch.setattr(pipeline.mlflow, "active_run", lambda: FakeActiveRun())

    final_state = pipeline.run_pipeline(csv_file_path="data/example.csv", preview_rows=3)

    assert final_state.status == "ok"
    assert calls["start_run"] == [
        {"run_name": calls["start_run"][0]["run_name"], "nested": True}
    ]
    assert calls["end_run"] == [{"status": "FINISHED"}]


def test_run_pipeline_ends_mlflow_run_as_failed_for_uncaught_graph_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_mlflow(monkeypatch)
    graph_error = RuntimeError("graph exploded")
    monkeypatch.setattr(
        pipeline,
        "build_pipeline_graph",
        lambda: FakeGraph(error=graph_error),
    )

    with pytest.raises(RuntimeError, match="graph exploded"):
        pipeline.run_pipeline(csv_file_path="data/example.csv", preview_rows=3)

    assert calls["end_run"] == [{"status": "FAILED"}]
