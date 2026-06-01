# TexPromter Development History

## Overview
This document tracks all feature requests, user requirements, and their implementations for the TexPromter optimization pipeline web application. It serves as a reference for understanding the application's evolution and architectural decisions.

---

## Feature Requests & Implementations

### 1. Real-Time Streaming Display Integration
**Request:** "I want to see the outputs of the models in real time and not all at the end of the pipeline"

**What was needed:**
- Connect the LangGraph pipeline with a Streamlit web UI
- Display agent outputs as they complete during execution
- Not wait for all agents to finish before showing results

**How it was implemented:**

#### Backend Architecture
- **File:** `orchestrator/pipeline.py`
- **Function:** `_run_pipeline_with_streaming_generator()`
- Used `graph.stream(mode="values")` from LangGraph to yield state updates as agents complete
- Generator function wraps the streaming loop to integrate with Streamlit

#### Frontend Display
- **File:** `app/streamlit_app.py`
- Created persistent containers outside execution block:
  ```python
  log_container = st.container()
  outputs_container = st.container()
  ```
- Streaming loop processes state updates in real-time:
  ```python
  for state_update in _run_pipeline_with_streaming_generator(...):
      # Display Use Case, Modeling, Preprocessing, Scripting as they appear
  ```
- Used `displayed_stages` set to track which outputs already shown (prevent duplicates)

#### Data Flow
1. User uploads CSV and clicks "Start Analysis"
2. Pipeline graph begins streaming execution
3. As each agent completes, state_update is yielded
4. Streamlit captures output and displays immediately in `outputs_container`
5. User sees real-time progress without waiting

**Files Modified:**
- `app/streamlit_app.py` (main implementation)
- `orchestrator/pipeline.py` (streaming support)

---

### 2. Modeling Intercept & Review Feature
**Request:** "I want to review the mathematical constraints before preprocessing runs"

**What was needed:**
- Pause pipeline after mathematical modeling agent completes
- Display formatted constraints (LaTeX) for user review
- Allow user to approve or provide feedback
- Not proceed to preprocessing until user decides

**How it was implemented:**

#### Intercept Trigger
- **File:** `app/streamlit_app.py`
- Pipeline streaming loop breaks after modeling stage:
  ```python
  if "modelling" in state_update and "modelling" not in displayed_stages:
      displayed_stages.add("modelling")
      st.session_state.original_modeling = state_update["modelling"]
      # Display modeling
      modeling_completed = True
      break  # Exit the streaming loop
  ```

#### Formatting
- **Function:** `format_modeling_output(modeling_dict: dict) -> str`
- Converts modeling output to human-readable LaTeX markdown:
  ```python
  Objective Function (LaTeX): max/min ...
  
  Constraints:
  1. ... ≤ ...
  2. ... ≥ ...
  ...
  
  Documentation: [readable explanation]
  ```

#### UI Controls
- Expander: "Mathematical Model (Review & Edit)"
- Three action buttons:
  - ✅ **Approve & Continue** → Runs preprocessing + scripting
  - ✏️ **Edit & Re-run** → Opens feedback mode
  - ❌ **Cancel Analysis** → Stops pipeline

**Files Modified:**
- `app/streamlit_app.py`
- `schemas/basemodels.py` (model structures)

---

### 3. Natural Language Feedback Loop
**Request:** "Allow users to provide feedback on constraints without editing JSON"

**What was needed:**
- Users describe issues with constraints in natural language
- Feedback triggers re-execution of modeling and downstream agents
- All outputs update based on feedback

**How it was implemented:**

#### Feedback Interface
- **File:** `app/streamlit_app.py`
- Text area for user feedback:
  ```python
  feedback_text = st.text_area(
      "Your feedback for the modeling agent",
      height=200,
      placeholder="Example: The throughput constraint exceeds capacity..."
  )
  ```

#### Feedback Processing Pipeline
- **Function:** `rerun_modeling_with_feedback(csv_path, feedback, use_case_dict, current_modeling_dict, input_schema_payload)`
- Steps:
  1. Enhance use_case with user feedback: `use_case["assumptions"].append(f"User feedback: {feedback}")`
  2. Call `run_mathematical_modelling_agent()` with enhanced use_case
  3. Call `run_data_processor_agent()` with updated modeling
  4. Call `run_pulp_coding_agent()` with updated preprocessing
  5. Return tuple: `(modeling_output, preprocessing_output, scripting_output)`

#### State Updates
- All three pipeline_state outputs updated simultaneously
- UI re-renders with new outputs
- Logs show all three agents regenerated

**Agent Calls:**
- `agents/Mathematical_modelling.py` → `run_mathematical_modelling_agent()`
- `agents/Data_Processor_Agent.py` → `run_data_processor_agent()`
- `agents/Pulp_Coding_Agent.py` → `run_pulp_coding_agent()`

**Files Modified:**
- `app/streamlit_app.py`

---

### 4. Pipeline Interruption & Two-Stage Execution
**Request:** "Preprocessor and solver should only start after feedback from modeling agent is submitted"

**What was needed:**
- Stop pipeline after modeling completes
- Don't run preprocessing/scripting during initial streaming
- Only execute preprocessing/scripting after user approves or provides feedback

**How it was implemented:**

#### Loop Interruption
- **File:** `app/streamlit_app.py`
- Added break condition in streaming loop:
  ```python
  if "modelling" in state_update and "modelling" not in displayed_stages:
      displayed_stages.add("modelling")
      # ... display modeling ...
      modeling_completed = True
      break  # Stop streaming, wait for user decision
  ```

#### Two Execution Paths

**Path 1: Approve Without Feedback**
- **Function:** `run_downstream_agents(csv_path, use_case_dict, modeling_dict, input_schema_payload)`
- Called when user clicks "✅ Approve & Continue"
- Runs only:
  1. Preprocessing agent
  2. Scripting agent
- Returns: `(preprocessing_output, scripting_output)`

**Path 2: Provide Feedback & Regenerate**
- **Function:** `rerun_modeling_with_feedback()` (existing)
- Called when user clicks "🔄 Re-run with Feedback"
- Runs all three agents:
  1. Modeling (with feedback)
  2. Preprocessing
  3. Scripting
- Returns: `(modeling_output, preprocessing_output, scripting_output)`

#### UI Flow
```
Pipeline streams → Stops at modeling
                  ↓
           User sees intercept with 3 buttons
                  ↓
    ┌────────────┬──────────────┬────────────┐
    ↓            ↓              ↓            ↓
  Approve     Edit & Feedback  Cancel    (timeout)
    ↓            ↓              ↓
  run_            rerun_         stop
  downstream      modeling_
  agents          with_feedback
    ↓            ↓              ↓
  Display      Display        Clear
  outputs      outputs        state
```

**Files Modified:**
- `app/streamlit_app.py`

---

### 5. Session State & Multi-Run Support
**Request:** "Allow users to run multiple analyses in the same session without clearing browser state"

**What was needed:**
- Preserve session across multiple runs
- Clear only pipeline-specific state on new analysis
- Maintain CSV file and settings when rerunning

**How it was implemented:**

#### Session State Structure
- **File:** `app/streamlit_app.py`
- **Function:** `initialize_session_state()`
- 13 state variables organized by category:

**Data Persistence:**
```python
"csv_file": None,              # Raw uploaded bytes
"csv_filename": None,          # Original filename
"csv_path": None,              # Temporary path
"initial_prompt": "",          # User's analysis prompt
```

**Execution Tracking:**
```python
"execution_running": False,    # Currently executing
"execution_complete": False,   # All stages done
"current_stage": None,         # Active stage name
```

**Pipeline Results:**
```python
"pipeline_state": None,        # Full state dict
"agent_logs": [],              # Timestamped logs
"result_file": None,           # Saved result path
```

**Error Handling:**
```python
"last_error": None,            # Last error dict
"error_stage": None,           # Stage where error occurred
```

**Modeling Intercept:**
```python
"show_modeling_intercept": False,
"original_modeling": None,
"modeling_edit_mode": False,
"modeling_edited_content": None,
```

#### Reset Logic
- **Location:** "Start Analysis" button click handler
- Resets only execution-related state, preserves CSV:
  ```python
  if st.button("🚀 Start Analysis"):
      st.session_state.execution_running = True
      st.session_state.pipeline_state = None
      st.session_state.agent_logs = []
      st.session_state.last_error = None
      st.session_state.show_modeling_intercept = False
      st.session_state.modeling_edited_content = None
      st.rerun()
  ```

**Files Modified:**
- `app/streamlit_app.py`

---

### 6. Real-Time Output Display (Non-Batch)
**Request:** "Outputs should appear immediately, not wait for all agents to finish"

**What was needed:**
- Display each output as soon as its agent completes
- Not accumulate and display all at once
- Use persistent containers to avoid duplicate boxes

**How it was implemented:**

#### Container Architecture
- **File:** `app/streamlit_app.py`
- Containers created once, outside execution block:
  ```python
  log_container = st.container()
  outputs_container = st.container()
  ```

#### Output Display During Streaming
- Streaming loop displays outputs immediately:
  ```python
  with outputs_container:
      if "use_case" in state_update and "use_case" not in displayed_stages:
          displayed_stages.add("use_case")
          with st.expander("📌 Use Case Analysis", expanded=True):
              # Display immediately
      
      if "modelling" in state_update and "modelling" not in displayed_stages:
          displayed_stages.add("modelling")
          with st.expander("🔢 Mathematical Model", expanded=True):
              # Display immediately
  ```

#### Persistent Display After Execution
- After streaming ends, outputs redisplayed from saved state:
  ```python
  if st.session_state.pipeline_state:
      with outputs_container:
          # Redisplay all 4 outputs from pipeline_state
          # This ensures they persist through intercept
  ```

#### Deduplication
- `displayed_stages` set prevents rendering same output twice:
  ```python
  if "use_case" not in displayed_stages:
      displayed_stages.add("use_case")
      # Display once
  ```

**Files Modified:**
- `app/streamlit_app.py`

---

### 7. Execution Log Management
**Request:** "Logs should be in a single box that updates, not create new boxes each time"

**What was needed:**
- Single log expander that accumulates entries
- Not recreate the expander on each state update
- Update log content without duplicating UI

**How it was implemented:**

#### Single Expander Creation
- **File:** `app/streamlit_app.py`
- Expander created once before streaming loop:
  ```python
  with log_container:
      with st.expander("📋 Execution Logs", expanded=True):
          logs_placeholder = st.empty()
  ```

#### Content Update (Not Recreation)
- Inside streaming loop, update placeholder content:
  ```python
  with logs_placeholder.container():
      for log_entry in st.session_state.agent_logs:
          if log_entry["level"] == "error":
              st.error(log_entry["message"])
          elif log_entry["level"] == "warning":
              st.warning(log_entry["message"])
          else:
              st.text(log_entry["message"])
  ```

#### Log Addition
- **Function:** `add_log(message: str, level: str = "info")`
- Appends to session state without UI recreation:
  ```python
  st.session_state.agent_logs.append({
      "message": f"[{timestamp}] {message}",
      "level": level
  })
  ```

**Files Modified:**
- `app/streamlit_app.py`

---

### 8. Results Persistence & Download
**Request:** "Save and allow download of analysis results"

**What was needed:**
- Save pipeline outputs to file on completion
- Provide download button in UI
- Store metadata with results

**How it was implemented:**

#### Saving Results
- **Function:** `save_pipeline_results(final_state: dict, csv_filename: str, initial_prompt: str)`
- **Location:** `app/streamlit_app.py`
- **Output:** `TestOutputs/result_{timestamp}_{filename}.json`
- Saved structure:
  ```json
  {
    "metadata": {
      "timestamp": "20260503_143022",
      "csv_filename": "input.csv",
      "initial_prompt": "Optimize production workflow",
      "modeling_edited": false
    },
    "pipeline_state": {
      "use_case": {...},
      "modelling": {...},
      "preprocessing": {...},
      "scripting": {...},
      "errors": [],
      "traces": [],
      "execution_metadata": {}
    }
  }
  ```

#### Download Button
- **Location:** Results section, after execution completes
- Uses Streamlit's built-in download:
  ```python
  if st.session_state.result_file:
      with open(st.session_state.result_file, "r") as f:
          st.download_button(
              label="📥 Download Results (JSON)",
              data=f.read(),
              file_name=Path(st.session_state.result_file).name,
              mime="application/json"
          )
  ```

**Files Modified:**
- `app/streamlit_app.py`

---

### 9. Error Handling & Retry
**Request:** "Provide meaningful error messages and ability to retry"

**What was needed:**
- Catch errors from any agent
- Display clear error messages
- Allow retry from failed stage

**How it was implemented:**

#### Error Capture
- **File:** `app/streamlit_app.py`
- Inside streaming loop:
  ```python
  errors = state_update.get("errors", [])
  if errors:
      error = errors[-1]
      st.session_state.last_error = error
      st.session_state.error_stage = error.get("agent_name", "unknown")
      raise Exception(f"Error in {error.get('agent_name')}: {error.get('message')}")
  ```

#### UI Display
- Error alert with details:
  ```python
  st.error(f"❌ Pipeline execution failed: {str(e)}")
  ```
- Logs captured with "error" level via `add_log(message, level="error")`

#### Retry Button
- **Condition:** `if st.session_state.last_error and not st.session_state.execution_running`
- **Action:** Reset error state and re-run
  ```python
  if st.button("🔄 Retry from Failed Stage"):
      st.session_state.execution_running = True
      st.session_state.last_error = None
      st.session_state.show_modeling_intercept = False
      st.rerun()
  ```

**Files Modified:**
- `app/streamlit_app.py`

---

## Bug Fixes & Refinements

### Issue 1: Indentation Error (Line 517)
**Problem:** st.text_area() parameters had inconsistent indentation after refactoring
**Solution:** Realigned multi-line parameters in feedback text_area
**File:** `app/streamlit_app.py`
**Impact:** Fixed syntax error preventing execution

### Issue 2: Multiple Log Boxes
**Problem:** Log expander recreated on every streaming iteration, creating duplicate boxes
**Solution:** Create expander once before loop, update content via placeholder
**File:** `app/streamlit_app.py`
**Impact:** Single log box that accumulates entries cleanly

### Issue 3: Outputs Disappear on Intercept
**Problem:** Outputs only displayed during streaming; disappeared when intercept activated
**Solution:** Added persistent display section that redisplays from saved pipeline_state
**File:** `app/streamlit_app.py`
**Impact:** Outputs visible throughout intercept and feedback workflow

### Issue 4: Full Pipeline Runs Before Feedback
**Problem:** Preprocessing and scripting ran before user could provide modeling feedback
**Solution:** Break streaming loop after modeling, create separate run_downstream_agents() function
**File:** `app/streamlit_app.py`
**Impact:** Two-stage execution model with user decision point

### Issue 5: Preprocessor output missing after delayed run
**Problem:** The delayed preprocessing run after modeling feedback no longer displayed code on the website
**Solution:** Updated UI to use the data processor's actual output fields such as `full_script`, `preprocessing_steps`, `mapping_explanation`, and `assumptions`
**File:** `app/streamlit_app.py`
**Impact:** Preprocessing code and documentation are visible again after user approval/feedback

### Issue 6: Duplicate modeling display during feedback
**Problem:** When feedback mode is active, the modeling result appeared twice (persistent view + intercept view)
**Solution:** Hide the persistent modeling expander while `show_modeling_intercept` is true, leaving only the review/edit intercept block visible
**File:** `app/streamlit_app.py`
**Impact:** Modeling output displays only once during feedback review

---

### 10. Centralized Pipeline Execution for Streamlit
**Request:** "Move pipeline execution logic out of Streamlit and keep it in the orchestrator module"

**What was needed:**
- Streamlit UI should only manage user interaction, display, and session state
- All pipeline execution, streaming, downstream reruns, and feedback-driven reruns should run through `orchestrator/pipeline.py`
- New helpers must use the same MLflow / LangGraph execution architecture as `run_pipeline`

**How it was implemented:**

#### Orchestrator pipeline helpers
- **File:** `orchestrator/pipeline.py`
- Added `stream_pipeline()` to stream LangGraph state updates inside an MLflow run
- Added `run_downstream_agents()` to execute preprocessing + scripting after user approval
- Added `rerun_modeling_with_feedback()` to rerun modeling with user feedback, then regenerate preprocessing and scripting
- These helpers use `_setup_mlflow()` and nested MLflow runs when needed, matching `run_pipeline()` behavior

#### Streamlit frontend cleanup
- **File:** `app/streamlit_app.py`
- Removed direct `StateGraph` pipeline construction from Streamlit
- Removed agent-level pipeline orchestration from the UI file
- Streamlit now imports and delegates execution to the orchestrator helpers

#### Package visibility
- **File:** `orchestrator/__init__.py`
- Exported the new orchestrator helper names so package imports remain consistent

**Impact:**
- Pipeline logic is centralized in `orchestrator/pipeline.py`
- Streamlit remains a UI-only integration layer
- MLflow/LangGraph tracing is consistent across normal CLI runs and Streamlit-driven workflow runs

---

## Architecture Summary

### Data Flow
```
User Input (CSV + Prompt)
           ↓
Temp File Creation
           ↓
Pipeline Streaming
  ├─ Initialize (Load schema)
  ├─ Use Case Analysis
  ├─ Mathematical Modeling (⏸️ PAUSE)
  ↓
User Decision
  ├─ Approve → run_downstream_agents()
  │             ├─ Data Preprocessing
  │             └─ Solver Scripting
  └─ Feedback → rerun_modeling_with_feedback()
                 ├─ Math Model (regenerated)
                 ├─ Data Preprocessing
                 └─ Solver Scripting
           ↓
Results Display & Download
           ↓
Save to TestOutputs/result_*.json
```

### Key Components
1. **Streamlit App** (`app/streamlit_app.py`): Web UI, session management, real-time display
2. **LangGraph Pipeline** (`orchestrator/pipeline.py`): Orchestration, streaming, error handling
3. **Agents** (`agents/`): Mathematical modeling, preprocessing, scripting
4. **Schemas** (`schemas/basemodels.py`): Pydantic models for validation

### Dependencies
- **Streamlit**: Web UI framework
- **LangGraph**: Orchestration engine
- **LangChain**: Agent framework
- **Pydantic**: Data validation
- **Ollama**: Local LLM inference (via SSH tunnel at 194.95.108.135:11434)

---

## Current State (May 3, 2026)

### Completed Features
✅ Real-time streaming display
✅ Modeling intercept & review
✅ Natural language feedback
✅ Two-stage pipeline execution
✅ Multi-run session support
✅ Persistent output display
✅ Single log expander
✅ Results download
✅ Error handling & retry
✅ Metadata tracking

### Known Limitations
- Windows Policy blocks streamlit.exe execution on development machine
- SSH tunnel for Ollama must be manually established
- No multi-user concurrency support (Streamlit limitation)

### Testing Status
✅ Syntax validation: All Python code compiles without errors
✅ Import verification: All agent and schema imports confirmed working
✅ Streaming test: Graph streaming with actual pipeline validated
✅ Agent calls: All agent functions callable with correct parameters
✅ Session state: Persistence and reset verified in code logic

---

## File Reference

### Core Application Files
- `app/streamlit_app.py` — Main web UI (730+ lines)
- `orchestrator/pipeline.py` — LangGraph orchestration
- `schemas/basemodels.py` — Pydantic data models

### Agent Modules
- `agents/context_agent.py` — Context extraction
- `agents/Data_Processor_Agent.py` — Data preprocessing
- `agents/Mathematical_modelling.py` — Mathematical model generation
- `agents/Pulp_Coding_Agent.py` — PuLP solver code generation

### Documentation
- `STREAMLIT_GUIDE.md` — User guide
- `IMPLEMENTATION_SUMMARY.md` — Technical summary
- `QUICK_REFERENCE.md` — Quick start guide
- `DEVELOPMENT_HISTORY.md` — This file

### Data & Testing
- `data/` — Input CSVs and schemas
- `TestOutputs/` — Generated results
- `tests/` — Unit tests and notebooks

---

## How to Use This Document

For future development or knowledge transfer:
1. **Onboarding:** Read the "Feature Requests & Implementations" section to understand what was built and why
2. **Architecture:** Review "Architecture Summary" for data flow and component relationships
3. **Bug Context:** Check "Bug Fixes & Refinements" if similar issues arise
4. **Implementation Reference:** Use specific function locations when modifying features
5. **Session State:** Reference the state structure when adding new features

---

**Last Updated:** May 4, 2026
**Status:** Production-ready (pending execution environment)
