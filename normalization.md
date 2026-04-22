**Normalization Overview**

This project was refactored to a state-first, LangGraph-style pipeline. The goals were:
- expose each legacy agent as a simple callable `run_*` function
- pass a central `PipelineState` between nodes instead of relying on environment side-effects
- add optional MLflow-based observability with a sensible local default
- enable isolated per-agent testing and easy local invocation

**Major changes**
- State-first pipeline: an `initialize` node loads the CSV input schema into `state.input_schema_payload`. Nodes receive and return state.
- Normalized agents: legacy agents were converted to `run_context_agent`, `run_mathematical_modelling_agent`, `run_data_processor_agent`, `run_pulp_coding_agent` in `agents/`.
- Orchestrator: `orchestrator.pipeline` now provides `run_agent_node(node_name, state)` and a CLI to run the whole pipeline.
- MLflow: optional per-run tracking is available; the pipeline wraps runs so failures in tracking won't break tests.

**Quick: run the whole pipeline (CLI)**
- Example (PowerShell / cmd):

```
python -m orchestrator.pipeline data/optimization_pipeline_test_easy.csv --stream-pipeline-output
```

- Notes:
  - The first positional arg is the path to the input CSV.
  - `--stream-pipeline-output` enables live console streaming of node outputs (used during debugging).
  - Additional CLI flags may exist; you can also call the pipeline from Python for more control.

**Run a single node / agent (Python)**
- Use the orchestrator helper to run one node in-process (recommended for testing/fine-tuning):

```py
from orchestrator.pipeline import run_agent_node, PipelineState

state = PipelineState(input_csv_path='data/optimization_pipeline_test_easy.csv')
result_state = run_agent_node('use_case', state)
# result_state contains updated state and execution metadata
```

- Or call an agent directly:

```py
from agents.legacy_agents.context_agent import run_context_agent
out = run_context_agent('data/optimization_pipeline_test_easy.csv', preview_rows=5)
```

This direct call is useful when preparing datasets or extracting prompts for fine-tuning.

**MLflow configuration & usage**
- The pipeline uses MLflow for tracking when configured. Configure via environment variables before running:

- `MLFLOW_TRACKING_URI` — tracking server URI. Example local SQLite file:

```
set MLFLOW_TRACKING_URI=sqlite:///./mlflow.db    # Windows cmd
$env:MLFLOW_TRACKING_URI = 'sqlite:///./mlflow.db'  # PowerShell
```

- `MLFLOW_EXPERIMENT_NAME` — optional experiment name (default used if not set).

- Behavior:
  - If MLflow is not reachable or not configured, the pipeline will continue and tests will still run; MLflow errors are caught and logged.
  - For robust cross-platform local tracking prefer a SQLite URI (`sqlite:///./mlflow.db`) over MLflow's file-store.

- Start a local server/UI against the same backend used by the pipeline:

```
mlflow server --backend-store-uri "sqlite:///C:/Users/simon/6_Semester/kip/KIP_texprompter/texprompter/mlflow.db" --host 127.0.0.1 --port 5000
```

Then set:

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**What is now logged (thorough setup)**
- Per stage (`initialize`, `use_case`, `modeling`, `preprocessing`, `scripting`):
  - structured recommendation payloads
  - tool trace (`*_agent` tool calls in order)
  - `llm_diagnostics.json` with attempt-level details:
    - attempt count
    - seen tool names per attempt
    - preview of last model message per attempt
    - validation error (if schema validation failed)
  - stage metrics (`duration_seconds`, tool call count, llm attempt count)
- Pipeline-level artifacts:
  - `pipeline/final_state.json`
  - `pipeline/execution_metadata.json`
  - `pipeline/llm_artifacts.json`
  - `pipeline/errors.json`
- MLflow tracing:
  - spans are emitted via `mlflow.start_span(...)`
  - traces can appear in the **Traces** tab (not only Experiments/Runs)

**Where to inspect in UI**
- **Experiments** tab:
  - open a pipeline run and inspect the `Artifacts` panel for `*/llm_diagnostics.json`
  - inspect `Metrics` and `Tags` for stage-level counters and statuses
- **Traces** tab:
  - inspect pipeline/stage spans and error states
  - useful for timeline-style debugging when a stage fails to return structured output
- **Evaluation Runs** tab:
  - here you can see all the invoked agents
  - click on "go to run" (small rectangle with arrow that appears when hovering over agent name) you can information that may be interesting to you in Artifacts

**Troubleshooting MLflow visibility**
- If runs appear but traces are missing:
  - ensure pipeline and UI/server point to the same backend URI
  - verify trace tables in sqlite are non-zero (`trace_info`, `spans`)
- Avoid using legacy `mlruns` file store in this project; use sqlite backend instead.

**Best practices for MLflow**
- For CI or team runs, use a centralized MLflow server (HTTP URI) and set `MLFLOW_TRACKING_URI` accordingly.
- Use `MLFLOW_EXPERIMENT_NAME` to group runs from this pipeline.

**Finetuning / extracting training data from an agent**
- Call the agent directly to get the structured output which contains prompts, LLM traces, or generated code:

```py
from agents.mathematical_modelling import run_mathematical_modelling_agent
payload = run_mathematical_modelling_agent(csv_file_path='data/my.csv')
# Inspect payload for prompt examples or generated JSON to construct finetuning datasets
```

- The agent outputs are plain Python structures and often write compatibility artifacts to `TestOutputs/` for convenience.
state = PipelineState(csv_file_path='data/optimization_pipeline_test_easy.csv')
**Where to look next**
- Orchestrator entry: `orchestrator/pipeline.py` — pipeline CLI and helpers.
- Agent entrypoints: `agents/*` — normalized `run_*` functions.
- Shared helpers: `agents/shared.py` — CSV schema loader and invoke helpers.