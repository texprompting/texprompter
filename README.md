# texprompter


## Installation Guide for Conda env:
🛠️ How to Set Up the Project Environment
To ensure everyone on the team has the exact same package versions regardless of whether you are on Windows, Linux, or macOS, we are using conda-lock. This prevents "it works on my machine" bugs.

Please follow these steps to set up your environment:

Step 1: Install conda-lock in your base environment
You only need to do this once on your machine. Open your terminal (or Anaconda Prompt on Windows) and run:

``` Bash
conda install -c conda-forge conda-lock
```
Step 2: Navigate to the project folder
Make sure you have pulled the latest code and use your terminal to navigate into the directory containing the conda-lock.yml file.

``` Bash
cd path/to/the/project/folder
```

Step 3: Install the locked environment
Run the following command. conda-lock will automatically detect your operating system and install the exact matching dependencies:

``` Bash
conda-lock install --name texprompting conda-lock.yml
``` 
(Note: If you want to use a different name for your local environment, you can change shared_project_env to whatever you prefer).

Step 4: Activate your new environment
Once the installation finishes, activate it just like a normal Conda environment:

``` Bash
conda activate texprompting
``` 


## LangGraph Scaffold

The current implementation uses a LangGraph state machine with four agent nodes:

1. Use Case Agent: analyzes CSV and selects the best OR use case.
2. Modeling Agent: builds MILP pseudo-LaTeX objective and constraints.
3. Preprocessing Agent: generates csv->input-schema mapping contract and script.
4. Scripting Agent: generates runnable PuLP code and output schema.

Shared contracts are defined in `schemas/basemodels.py` and the graph orchestrator is implemented in `orchestrator/pipeline.py`.


## Running The Pipeline

Run the LangGraph pipeline:

```bash
python -m orchestrator.pipeline 
```

Default input CSV is `data/optimization_pipeline_test_easy.csv`. Generated artifacts are written to `TestOutputs/`.


## Dev startup scripts (Unix / Windows)

Interactive launcher at the **`texprompter/`** package root:

- **Unix:** [`start_texprompter.sh`](start_texprompter.sh)
- **Windows (PowerShell):** [`start_texprompter.ps1`](start_texprompter.ps1)

Both run from `texprompter/` (repo root), prepend `PYTHONPATH`, and invoke [`scripts/texprompter_dev.py`](scripts/texprompter_dev.py).

The driver (**SSH key auth only** — no password variables or `sshpass`):

1. Loads **`texprompter/.env`** via `python-dotenv` (if installed).
2. Ensures **`RZ_KENNUNG`** is set (prompts once and saves). **`RZ_SSH_HOST`** defaults to `194.95.108.135` and is written to `.env` when missing. Optional: **`RZ_SSH_LOCAL_OLLAMA_PORT`** / **`RZ_SSH_REMOTE_OLLAMA_PORT`** (default `11434`).
3. Asks whether to run **`ssh-copy-id`** \[y/N\] (default **No**). If **No**, assumes your public key is already authorized on the jump host.
4. Opens an SSH tunnel **`localhost:<local> → localhost:<remote>` on the server** using `ssh -N` with `BatchMode=yes`.
5. Starts **MLflow** with `sqlite:///./mlflow.db` at **http://127.0.0.1:5000** (`--workers 1`).
6. Asks whether to enable **live LLM / pipeline streaming** (`OLLAMA_STREAM_STDOUT=1` and `--stream-pipeline-output` for pipeline runs).
7. Menu: **(1)** run the pipeline on a CSV chosen from the same list as evaluation seeds (paths under `data/`; industrial CSVs live in `data/versatile_producion_system` — that folder name matches the repo spelling), **(2)** run **`python -m evaluation.run_eval --with-judge`**.

On exit (or Ctrl+C), SSH and MLflow child processes started by the script are terminated.

```bash
cd texprompter
./start_texprompter.sh
```

```powershell
cd texprompter
.\start_texprompter.ps1
```

Direct Python (same cwd):

```bash
PYTHONPATH=. python scripts/texprompter_dev.py
```


## MLflow, Prompt Registry, Evaluation

This project uses MLflow autologging (no hand-rolled instrumentation). After completing the conda install above, also install the supplemental dependencies:

```bash
pip install -r requirements-mlflow.txt
```

The pipeline talks to a local Ollama instance through Ollama's OpenAI-compatible API (`$OLLAMA_BASE_URL/v1`). `mlflow.langchain.autolog(run_tracer_inline=True)` captures every LangGraph node, ChatOpenAI call and tool invocation as a nested span (`<agent> > LangGraph > model > tools > ...`). `run_tracer_inline=True` is **required** so the tracer context propagates back when LangGraph fans out parallel tool calls; otherwise the next model node never starts. We deliberately do **not** stack `mlflow.openai.autolog()` on top — the duplicate `Completions` layer combined with parallel tool fanout could deadlock the agent loop.

MLflow Prompt Registry entries are global, not scoped to an experiment. Runtime pipeline runs load prompts with `prompts:/texprompter.<stage>@latest`, then record the resolved prompt URI, version, and source on both the active run tags (`prompt.<stage>.*`) and the trace metadata. If the registry is unavailable, the loader falls back to `prompts/*.txt` and marks the source as `local_file`; set `MLFLOW_PROMPT_REGISTRY_REQUIRED=true` to fail instead. User messages are not separate registry prompts: they are logged as trace chat messages together with the system prompt so the MLflow trace view shows the actual request sent to each agent.

Resilience knobs (all optional, set in `.env`):

- `OLLAMA_REQUEST_TIMEOUT_S` (default `180`) and `OLLAMA_REQUEST_MAX_RETRIES` (default `1`) bound every Ollama HTTP call so a wedged response surfaces as `APITimeoutError` instead of an indefinite hang.
- `AGENT_RECURSION_LIMIT` (default `12`) caps the number of LangGraph steps any single `agent.invoke` may take, so a structured-output retry loop fails fast.

Useful commands:

```bash
mlflow server --backend-store-uri sqlite:///./mlflow.db --host 127.0.0.1 --port 5000 --workers 1   # UI (workers 1 required for sqlite)
python -m scripts.register_prompts                                                     # sync prompts/*.txt -> Prompt Registry
python -m evaluation.run_eval                                                          # deterministic scorers
python -m evaluation.run_eval --with-judge                                             # add LLM-as-judge
```

See `normalization.md` for full details (env vars, scorer descriptions, prompt-edit workflow).


## Scaffold Outline (apart from `agents/`)

- `orchestrator/pipeline.py`: graph wiring, state transitions, and node error handling.
- `schemas/basemodels.py`: all inter-agent message contracts.
- `prompts/`: agent system prompts (source of truth for the MLflow Prompt Registry).
- `scripts/register_prompts.py`: idempotent registration of `prompts/*.txt` into MLflow.
- `scripts/texprompter_dev.py`, `start_texprompter.sh`, `start_texprompter.ps1`: dev launcher (SSH tunnel, MLflow, menu).
- `evaluation/`: `mlflow.genai.evaluate` harness, scorers, and seed dataset.


## Contributing Guide

### Workflow for Contributors

Follow these steps when working on the project:

1. **Clone or Pull the Repository**
   - If you're new: `git clone <repo-url>` and follow the Installation Guide above
   - If you already have it: `git pull origin main` to get the latest changes

2. **Create a New Branch for Your Ticket**
   - Pull the latest main: `git pull origin main`
   - Create a new branch named after your ticket (e.g., `git checkout -b TICKET-ID-short-description`)
   - Use the format: `TICKET-ID-short-description` (kebab-case)

3. **Work on Your Changes**
   - Make your commits with clear, descriptive messages
   - Push your branch regularly: `git push origin TICKET-ID-short-description`

4. **Create a Pull Request**
   - Go to the repository on GitHub/GitLab and create a PR from your branch to `main`
   - Reference the JIRA ticket in the PR title or description (e.g., "MALOCHE-8: Add README")
   - Request code review from teammates
   - Ensure CI/tests pass before merging
   - There will be a random Reviewer assigned who needs to review and Merge the Code. 

5. **JIRA Integration**
   - Once your PR is linked in the title/description, JIRA should automatically detect the PR
   - A new Subtask is created for the Review. Assign the same Reviewer as in Github to the Subtask




# Versatile Production System (VPS) Dataset

## Overview

This repository contains process data from the **Versatile Production System (VPS)**, a modular smart factory demonstrator from the SmartFactory OWL environment. The system is used in industrial research for applications such as **machine learning, anomaly detection, process monitoring, and alarm management**.

The VPS simulates an end-to-end production workflow for popcorn processing and packaging using interconnected industrial modules.

---

## System Description

The VPS consists of five main modules:

### Delivery Module
Raw corn is delivered into the system via conveyor belt and transported into a stainless steel funnel. A pressure conveyor moves the material into storage.

### Storage Module
Acts as a buffer for raw material. When sufficient corn is available and downstream capacity allows, material is pneumatically transferred to dosing.

### Dosing Module
A controlled amount of corn is measured using a load cell and dosing screw mechanism. The measured portion is then transferred to the filling module.

### Filling Module
Bottles are processed on a rotary table through multiple stages:

- Cleaning using compressed air  
- Filling with corn  
- Lid placement using a pneumatic gripper  
- Screwing on lids  
- Quality inspection using a camera system  

### Production Module
Corn is heated and expanded into popcorn. The product is collected into cups until a weight threshold is reached. Excess popcorn is redirected to an overflow container.

---

## Dataset Content

The dataset contains time-series process data collected from sensors and actuators across the VPS modules.

Typical data includes:
- Sensor readings (e.g., weight, pressure, levels)
- Actuator states (e.g., motors, valves, conveyors)
- Process states and cycle information
- Control signals from the automation system

> Note: The exact signals and file names may depend on the specific experiment export.

---

## Acknowledgements

This dataset originates from:

- inIT – Institute Industrial IT  
- Ostwestfalen-Lippe University of Applied Sciences  

The VPS is part of the **SmartFactory OWL** research infrastructure.

It has been used in research projects such as **IMPROVE**, funded by the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 678867).

---

## License / Usage

This dataset is publicly available under [these terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Users are responsible for complying with the original data provider’s terms. For commercial use, please consult the original dataset owners.
