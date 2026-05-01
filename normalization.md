**Normalization Overview**

This project was refactored to a state-first, LangGraph-style pipeline. The goals were:
- expose each legacy agent as a simple callable `run_*` function
- pass a central `PipelineState` between nodes instead of relying on environment side-effects
- use MLflow autologging for observability instead of hand-rolled instrumentation
- enable isolated per-agent testing and easy local invocation

**Major changes**
- State-first pipeline: an `initialize` node loads the CSV input schema into `state.input_schema_payload`. Nodes receive and return state.
- Normalized agents: legacy agents were converted to `run_context_agent`, `run_mathematical_modelling_agent`, `run_data_processor_agent`, `run_pulp_coding_agent` in `agents/`.
- Orchestrator: `orchestrator.pipeline` provides `run_agent_node(node_name, state)` and a CLI to run the whole pipeline.
- LLM client: agents call Ollama through `langchain_openai.ChatOpenAI` pointed at `$OLLAMA_BASE_URL/v1` (Ollama's OpenAI-compatible API). `mlflow.langchain.autolog()` captures every LangGraph node, ChatOpenAI call and tool invocation as a nested span (the trace UI then shows `use_case > LangGraph > model > tools > model` for every agent). We deliberately do **not** stack `mlflow.openai.autolog()` on top: the duplicate `Completions` span layer combined with langgraph's parallel tool fanout could deadlock the agent loop after the parallel tools returned.
- Resilience knobs:
  - `OLLAMA_REQUEST_TIMEOUT_S` (default `180`) and `OLLAMA_REQUEST_MAX_RETRIES` (default `1`) bound every Ollama call so a wedged HTTP read surfaces as `APITimeoutError` ("Request timed out.") in the agent's error path instead of an indefinite hang.
  - `AGENT_RECURSION_LIMIT` (default `12`) caps the number of LangGraph steps any single `agent.invoke` may take.
  - All output schemas (`UseCaseRecommendation`, `ModellingRecommendation`, `PreprocessingRecommendation`, `ScriptingRecommendation`, `ContextRecommendation`, `DataPreparation`) carry `mode='before'` field validators that auto-decode JSON-string returns into real lists/dicts. This is critical for local LLMs: Qwen via Ollama frequently emits e.g. `constraint_functions` as a stringified JSON array, which would otherwise fail pydantic validation and send `create_agent` into a structured-output retry loop.
- Prompt registry: agent system prompts live in `texprompter/prompts/*.txt` and are mirrored into the MLflow Prompt Registry by `python -m scripts.register_prompts`.

**Quick: run the whole pipeline (CLI)**
- Example:

```
python -m orchestrator.pipeline data/optimization_pipeline_test_easy.csv --stream-pipeline-output
```

- Notes:
  - The first positional arg is the path to the input CSV.
  - `--stream-pipeline-output` enables live console streaming of node outputs (used during debugging).

**Run a single node / agent (Python)**
- Use the orchestrator helper to run one node in-process (recommended for testing/fine-tuning):

```py
from orchestrator.pipeline import run_agent_node
from schemas.basemodels import PipelineState

state = PipelineState(csv_file_path='data/optimization_pipeline_test_easy.csv')
result_state = run_agent_node('use_case', state)
```

- Or call an agent directly:

```py
from agents.context_agent import run_context_agent
out = run_context_agent('data/optimization_pipeline_test_easy.csv', preview_rows=5)
```

**Environment**

Required `.env` variables (already templated in `texprompter/.env`):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.6:latest
OPENAI_API_KEY=ollama
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
MLFLOW_EXPERIMENT_NAME=texprompter_pipeline
# Optional: bound every Ollama call so a wedged read errors out cleanly.
OLLAMA_REQUEST_TIMEOUT_S=180
OLLAMA_REQUEST_MAX_RETRIES=1
# Optional: cap LangGraph steps per agent.invoke (default 12).
AGENT_RECURSION_LIMIT=12
```

Install the supplemental MLflow dependencies on top of the conda-lock environment:

```
pip install -r requirements-mlflow.txt
```

**MLflow tracking**

`run_pipeline()` calls `_setup_mlflow()` once per process which:
- sets `MLFLOW_TRACKING_URI` (defaults to `sqlite:///./mlflow.db`)
- sets the experiment to `MLFLOW_EXPERIMENT_NAME` (default `texprompter_pipeline`)
- enables `mlflow.langchain.autolog(run_tracer_inline=True)`

Every pipeline invocation runs inside a single `mlflow.start_run`. With autolog enabled, each LLM call, tool call, and LangGraph step is captured as a nested span — visible under the **Traces** tab in the MLflow UI without any per-node boilerplate.

Start the local MLflow UI against the same backend:

```
mlflow server --backend-store-uri "sqlite:///./mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

`--workers 1` is important when the backend store is sqlite. MLflow 3.x defaults to 4 uvicorn workers, and 4 concurrent writers on the same sqlite file deadlock and return 503 on every write endpoint (`experiments/create`, `runs/create`, `runs/log-metric`, etc.). For Postgres/MySQL backends the default is fine.

Then point the pipeline at it:

```
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**What gets logged automatically**
- Per LLM call: prompt, response, model, latency, token usage (via `mlflow.langchain.autolog`)
- Per agent run: tool calls, tool inputs/outputs, intermediate messages (via `mlflow.langchain.autolog`)
- Per pipeline run: parameters (`csv_file_path`, `preview_rows`), tags (`pipeline.status`), metrics (`errors_count`, `trace_count`)

The `PipelineState` itself still tracks `errors`, `traces`, and `execution_metadata` for downstream consumers (tests, eval scorers); the orchestrator no longer mirrors those into MLflow as artifacts because the auto-traces already cover them.

**Prompt Registry**

Edit a prompt:

1. Edit the relevant file in `texprompter/prompts/` (`use_case.txt`, `modeling.txt`, `preprocessing.txt`, or `scripting.txt`).
2. Run `python -m scripts.register_prompts` to push a new version into the registry. The script is idempotent — it only creates a new version when the local content differs from the registered one.
3. The next pipeline run picks up the new `latest` version automatically (`agents/prompts.py::load_system_prompt_result` calls `mlflow.genai.load_prompt(prompts:/texprompter.<name>@latest)`).

If the registry is unreachable, the loader falls back to the local `prompts/*.txt` file and records `source=local_file` in the run/trace metadata, so tests and offline runs still work. Set `MLFLOW_PROMPT_REGISTRY_REQUIRED=true` to fail instead.

**Evaluation**

Run the deterministic structural scorers over the seed dataset built from `data/*.csv`:

```
python -m evaluation.run_eval
```

Add the LLM-as-judge scorer (requires an OpenAI-compatible endpoint configured for the judge — Ollama via `OPENAI_BASE_URL=http://localhost:11434/v1` works):

```
python -m evaluation.run_eval --with-judge
```

Scorers live in `evaluation/scorers.py`:
- `pipeline_ok` — final state is not `error`
- `all_schemas_valid` — every per-stage payload re-validates against its Pydantic schema
- `scripting_code_compiles` — generated PuLP code passes `compile()`
- `objective_aligned` — `Guidelines` LLM-as-judge for `use_case.objective_variable` ↔ `modelling.objective_function` alignment

The dataset (`evaluation/datasets.py`) just supplies CSV paths; this is a regression / smoke harness, not a labelled benchmark. Results are visible in the MLflow UI under the **Runs → Evaluation** tab.

**Where to look next**
- Orchestrator entry: `orchestrator/pipeline.py`
- Agent entrypoints: `agents/*.py`
- Shared helpers: `agents/shared.py` (LLM client + tool-trace extraction)
- Prompt loader: `agents/prompts.py`
- Eval harness: `evaluation/`
