import streamlit as st
import pandas as pd
import json
import tempfile
from datetime import datetime
from pathlib import Path
from io import StringIO
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.pipeline import run_pipeline
from schemas.basemodels import PipelineState, ModellingRecommendation

# Configure Streamlit
st.set_page_config(page_title="TexPrompter - Workflow Optimizer", layout="wide")
st.title("TexPrompter - Workflow Optimizer")

# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables on first load."""
    defaults = {
        "csv_file": None,
        "csv_filename": None,
        "csv_path": None,
        "initial_prompt": "",
        "pipeline_state": None,
        "current_stage": None,
        "agent_logs": [],
        "execution_running": False,
        "execution_complete": False,
        "last_error": None,
        "error_stage": None,
        "modeling_edit_mode": False,
        "modeling_edited_content": None,
        "result_file": None,
        "show_modeling_intercept": False,
        "original_modeling": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


initialize_session_state()

# ============================================================================
# Utility Functions
# ============================================================================

def add_log(message: str, level: str = "info"):
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    st.session_state.agent_logs.append({"message": formatted_msg, "level": level})


def save_pipeline_results(final_state: dict, csv_filename: str, initial_prompt: str):
    """Save pipeline results to JSON file."""
    output_dir = Path(__file__).parent.parent / "TestOutputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"result_{timestamp}_{csv_filename.replace('.csv', '')}.json"
    
    # Prepare results with metadata
    results = {
        "metadata": {
            "timestamp": timestamp,
            "csv_filename": csv_filename,
            "initial_prompt": initial_prompt,
            "modeling_edited": st.session_state.modeling_edited_content is not None,
        },
        "pipeline_state": final_state,
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    return str(result_file)


def extract_stage_from_traces(traces: list) -> str:
    """Extract the current stage from pipeline traces."""
    if not traces:
        return "unknown"
    
    last_trace = traces[-1] if traces else ""
    if ":" in last_trace:
        return last_trace.split(":")[0]
    return last_trace


def rerun_modeling_with_feedback(csv_path: str, feedback: str, use_case_dict: dict, current_modeling_dict: dict, input_schema_payload: dict) -> tuple[dict, dict, dict]:
    """
    Rerun the modeling agent with user feedback AND rerun downstream agents.
    
    Args:
        csv_path: Path to the CSV file
        feedback: User feedback about the constraints
        use_case_dict: The use case output from previous agent
        current_modeling_dict: The current modeling output to improve upon
        input_schema_payload: The CSV schema payload from initialize stage
    
    Returns:
        Tuple of (modeling_output, preprocessing_output, scripting_output)
    """
    try:
        from agents.Mathematical_modelling import run_mathematical_modelling_agent
        from agents.Data_Processor_Agent import run_data_processor_agent
        from agents.Pulp_Coding_Agent import run_pulp_coding_agent
        
        add_log("🔄 Step 1: Regenerating mathematical model with feedback...")
        
        # Step 1: Rerun the modeling agent with the use case + feedback
        enhanced_use_case = use_case_dict.copy() if isinstance(use_case_dict, dict) else use_case_dict
        if isinstance(enhanced_use_case, dict):
            # Add feedback to assumptions/notes
            if "assumptions" not in enhanced_use_case:
                enhanced_use_case["assumptions"] = []
            enhanced_use_case["assumptions"].append(f"User feedback: {feedback}")
        
        # Rerun the modeling agent
        modeling_result = run_mathematical_modelling_agent(
            csv_file_path=csv_path,
            use_case=enhanced_use_case,
            preview_rows=5,
            return_debug=False
        )
        
        # Extract the modeling recommendation
        if isinstance(modeling_result, dict):
            modeling_output = modeling_result.get("result") if "result" in modeling_result else modeling_result
        else:
            modeling_output = modeling_result
        
        add_log("✓ Mathematical model regenerated")
        
        # Step 2: Rerun the preprocessing agent with updated modeling
        add_log("🔄 Step 2: Regenerating data preprocessing with updated model...")
        
        preprocessing_result = run_data_processor_agent(
            csv_file_path=csv_path,
            use_case=enhanced_use_case,
            modelling=modeling_output,
            input_schema_payload=input_schema_payload,
            preview_rows=5,
            return_debug=False
        )
        
        if isinstance(preprocessing_result, dict):
            preprocessing_output = preprocessing_result.get("result") if "result" in preprocessing_result else preprocessing_result
        else:
            preprocessing_output = preprocessing_result
        
        add_log("✓ Data preprocessing regenerated")
        
        # Step 3: Rerun the scripting agent with updated preprocessing
        add_log("🔄 Step 3: Regenerating solver code with updated model...")
        
        scripting_result = run_pulp_coding_agent(
            csv_file_path=csv_path,
            modelling=modeling_output,
            preprocessing=preprocessing_output,
            preview_rows=5,
            input_schema_payload=input_schema_payload,
            return_debug=False
        )
        
        if isinstance(scripting_result, dict):
            scripting_output = scripting_result.get("result") if "result" in scripting_result else scripting_result
        else:
            scripting_output = scripting_result
        
        add_log("✓ Solver code regenerated")
        add_log("✅ All downstream agents updated with feedback")
        
        return modeling_output, preprocessing_output, scripting_output
    
    except Exception as e:
        add_log(f"❌ Error in feedback pipeline: {str(e)}", level="error")
        raise


def run_downstream_agents(csv_path: str, use_case_dict: dict, modeling_dict: dict, input_schema_payload: dict) -> tuple[dict, dict]:
    """
    Run only the preprocessing and scripting agents (used when user approves without feedback).
    
    Args:
        csv_path: Path to the CSV file
        use_case_dict: The use case output
        modeling_dict: The modeling output
        input_schema_payload: The CSV schema payload from initialize stage
    
    Returns:
        Tuple of (preprocessing_output, scripting_output)
    """
    try:
        from agents.Data_Processor_Agent import run_data_processor_agent
        from agents.Pulp_Coding_Agent import run_pulp_coding_agent
        
        add_log("🔄 Running data preprocessing agent...")
        
        preprocessing_result = run_data_processor_agent(
            csv_file_path=csv_path,
            use_case=use_case_dict,
            modelling=modeling_dict,
            input_schema_payload=input_schema_payload,
            preview_rows=5,
            return_debug=False
        )
        
        if isinstance(preprocessing_result, dict):
            preprocessing_output = preprocessing_result.get("result") if "result" in preprocessing_result else preprocessing_result
        else:
            preprocessing_output = preprocessing_result
        
        add_log("✓ Data preprocessing completed")
        
        add_log("🔄 Running solver code generation agent...")
        
        scripting_result = run_pulp_coding_agent(
            csv_file_path=csv_path,
            modelling=modeling_dict,
            preprocessing=preprocessing_output,
            preview_rows=5,
            input_schema_payload=input_schema_payload,
            return_debug=False
        )
        
        if isinstance(scripting_result, dict):
            scripting_output = scripting_result.get("result") if "result" in scripting_result else scripting_result
        else:
            scripting_output = scripting_result
        
        add_log("✓ Solver code generation completed")
        add_log("✅ All downstream agents completed")
        
        return preprocessing_output, scripting_output
    
    except Exception as e:
        add_log(f"❌ Error in downstream pipeline: {str(e)}", level="error")
        raise


# ============================================================================
# Utility Functions (continued)
# ============================================================================

def format_modeling_output(modeling_dict: dict) -> str:
    """Format modeling output for display."""
    if not modeling_dict:
        return "No modeling output available"
    
    output = []
    
    # Objective Function
    if "objective_function" in modeling_dict:
        output.append("## Objective Function")
        output.append(f"$$\n{modeling_dict['objective_function']}\n$$")
    
    # Constraints
    if "constraint_functions" in modeling_dict and modeling_dict["constraint_functions"]:
        output.append("\n## Constraints")
        for i, constraint in enumerate(modeling_dict["constraint_functions"], 1):
            output.append(f"\n**Constraint {i}:**")
            output.append(f"$$\n{constraint}\n$$")
    
    # Documentation
    if "readable_documentation" in modeling_dict:
        output.append("\n## Documentation")
        output.append(modeling_dict["readable_documentation"])
    
    return "\n".join(output)


# ============================================================================
# Pipeline Streaming Generator
# ============================================================================

def _run_pipeline_with_streaming_generator(csv_path: str, initial_prompt: str = ""):
    """
    Generator that yields state updates from the pipeline.
    Wraps the pipeline's streaming functionality.
    """
    try:
        from langgraph.graph import StateGraph
        from orchestrator.pipeline import build_pipeline_graph
        from schemas.basemodels import PipelineState
        
        graph = build_pipeline_graph()
        initial_state = PipelineState(
            csv_file_path=csv_path,
            preview_rows=5,
        )
        
        # Stream pipeline execution
        for state_update in graph.stream(initial_state.model_dump(), stream_mode="values"):
            if isinstance(state_update, dict):
                yield state_update
    
    except Exception as e:
        st.error(f"Failed to run pipeline: {str(e)}")
        raise


# ============================================================================
# Main UI Layout
# ============================================================================

# Sidebar with input controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"], key="csv_uploader")
    
    if uploaded_file:
        st.session_state.csv_file = uploaded_file.getvalue()
        st.session_state.csv_filename = uploaded_file.name
    
    # Optional initial prompt
    st.session_state.initial_prompt = st.text_area(
        "Optional: Initial Prompt for Context Agent",
        value=st.session_state.initial_prompt,
        help="Provide context or specific instructions for the analysis",
        height=100
    )
    
    # Start button
    if st.session_state.csv_file and not st.session_state.execution_running:
        if st.button("🚀 Start Analysis", use_container_width=True):
            st.session_state.execution_running = True
            st.session_state.execution_complete = False
            st.session_state.pipeline_state = None
            st.session_state.agent_logs = []
            st.session_state.last_error = None
            st.session_state.error_stage = None
            st.session_state.show_modeling_intercept = False
            st.session_state.modeling_edit_mode = False
            st.session_state.modeling_edited_content = None
            st.session_state.result_file = None
            st.session_state.original_modeling = None
            add_log("Analysis started...")
            st.rerun()
    
    # Retry button (shown on error)
    if st.session_state.last_error and not st.session_state.execution_running:
        if st.button("🔄 Retry from Failed Stage", use_container_width=True):
            st.session_state.execution_running = True
            st.session_state.last_error = None
            st.session_state.show_modeling_intercept = False
            add_log(f"Retrying from stage: {st.session_state.error_stage}...")
            st.rerun()

# Main content area
if not st.session_state.csv_file:
    st.info("👈 Please upload a CSV file to get started")
else:
    # Display CSV preview
    with st.expander("📊 CSV Preview", expanded=False):
        try:
            df = pd.read_csv(StringIO(st.session_state.csv_file.decode()))
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    # ========================================================================
    # Create persistent containers for logs and outputs
    # ========================================================================
    log_container = st.container()
    outputs_container = st.container()
    
    # ========================================================================
    # Pipeline Execution
    # ========================================================================
    
    if st.session_state.execution_running and st.session_state.pipeline_state is None:
        # Run pipeline with streaming
        try:
            # Save CSV temporarily
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
                tmp.write(st.session_state.csv_file)
                csv_path = tmp.name
                st.session_state.csv_path = csv_path
            
            st.divider()
            st.header("📈 Real-Time Execution Progress")
            
            # Create placeholder for logs (inside persistent container, created once before loop)
            with log_container:
                with st.expander("📋 Execution Logs", expanded=True):
                    logs_placeholder = st.empty()
            
            # Track which stages have been displayed
            displayed_stages = set()
            seen_traces = 0
            modeling_completed = False
            
            # Stream pipeline execution
            with st.status("Executing pipeline...", expanded=True) as status:
                for state_update in _run_pipeline_with_streaming_generator(csv_path, st.session_state.initial_prompt):
                    if isinstance(state_update, dict):
                        st.session_state.pipeline_state = state_update
                        
                        # Extract and display traces
                        traces = state_update.get("traces", [])
                        if isinstance(traces, list):
                            for trace in traces[seen_traces:]:
                                add_log(trace)
                                status.write(f"✓ {trace}")
                            seen_traces = len(traces)
                        
                        # Update logs in placeholder (not in loop - create expander once)
                        with logs_placeholder.container():
                            for log_entry in st.session_state.agent_logs:
                                if log_entry["level"] == "error":
                                    st.error(log_entry["message"])
                                elif log_entry["level"] == "warning":
                                    st.warning(log_entry["message"])
                                else:
                                    st.text(log_entry["message"])
                        
                        # Check for errors
                        errors = state_update.get("errors", [])
                        if errors:
                            error = errors[-1]
                            st.session_state.last_error = error
                            st.session_state.error_stage = error.get("agent_name", "unknown")
                            raise Exception(f"Error in {error.get('agent_name')}: {error.get('message')}")
                        
                        # Display outputs in real-time as they appear
                        with outputs_container:
                            # Use Case output
                            if "use_case" in state_update and "use_case" not in displayed_stages:
                                if state_update.get("use_case"):
                                    displayed_stages.add("use_case")
                                    with st.expander("📌 Use Case Analysis", expanded=True):
                                        use_case = state_update["use_case"]
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.subheader("Business Goal")
                                            st.write(use_case.get("business_goal", "N/A"))
                                        with col2:
                                            st.subheader("Objective Direction")
                                            st.write(use_case.get("objective_direction", "N/A"))
                                        
                                        if "decision_variables" in use_case:
                                            st.subheader("Decision Variables")
                                            for var in use_case.get("decision_variables", []):
                                                st.write(f"• {var}")
                                        
                                        if "constraints_to_consider" in use_case:
                                            st.subheader("Constraints to Consider")
                                            for constraint in use_case.get("constraints_to_consider", []):
                                                st.write(f"• {constraint}")
                            
                            # Modeling output - INTERRUPT PIPELINE HERE
                            if "modelling" in state_update and "modelling" not in displayed_stages:
                                if state_update.get("modelling"):
                                    displayed_stages.add("modelling")
                                    st.session_state.original_modeling = state_update["modelling"]
                                    with st.expander("🔢 Mathematical Model", expanded=True):
                                        modeling_dict = state_update["modelling"]
                                        st.markdown(format_modeling_output(modeling_dict))
                                    
                                    # Break out of the loop - preprocessing and scripting will only run after feedback
                                    add_log("⏸️ Waiting for user feedback on mathematical model...")
                                    modeling_completed = True
                                    break
            
            st.session_state.execution_running = False
            st.session_state.execution_complete = False  # Still waiting for feedback
            
            # Check if we should show modeling intercept
            if (st.session_state.pipeline_state and 
                "modelling" in st.session_state.pipeline_state and
                st.session_state.pipeline_state["modelling"]):
                st.session_state.show_modeling_intercept = True
                st.session_state.original_modeling = st.session_state.pipeline_state["modelling"]
            
            st.rerun()
        
        except Exception as e:
            st.session_state.execution_running = False
            st.session_state.last_error = str(e)
            add_log(f"Error: {str(e)}", level="error")
            st.error(f"❌ Pipeline execution failed: {str(e)}")
            st.rerun()
    
    # ========================================================================
    # Display Agent Outputs
    # ========================================================================
    
    if st.session_state.pipeline_state:
        # Display all outputs from pipeline state (real-time during execution, persistent after)
        with outputs_container:
            # Use Case output
            if st.session_state.pipeline_state.get("use_case"):
                use_case = st.session_state.pipeline_state["use_case"]
                with st.expander("📌 Use Case Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Business Goal")
                        st.write(use_case.get("business_goal", "N/A"))
                    with col2:
                        st.subheader("Objective Direction")
                        st.write(use_case.get("objective_direction", "N/A"))
                    
                    if "decision_variables" in use_case:
                        st.subheader("Decision Variables")
                        for var in use_case.get("decision_variables", []):
                            st.write(f"• {var}")
                    
                    if "constraints_to_consider" in use_case:
                        st.subheader("Constraints to Consider")
                        for constraint in use_case.get("constraints_to_consider", []):
                            st.write(f"• {constraint}")
            
            # Modeling output
            if st.session_state.pipeline_state.get("modelling") and not st.session_state.show_modeling_intercept:
                with st.expander("🔢 Mathematical Model", expanded=True):
                    modeling_dict = st.session_state.pipeline_state["modelling"]
                    st.markdown(format_modeling_output(modeling_dict))
            
            # Preprocessing output
            if st.session_state.pipeline_state.get("preprocessing"):
                preprocessing = st.session_state.pipeline_state["preprocessing"]
                with st.expander("🔄 Data Preprocessing", expanded=True):
                    st.subheader("Generated Data Preparation Code")
                    preprocessing_code = preprocessing.get("full_script") or preprocessing.get("mapper_script")
                    st.code(preprocessing_code or "No preprocessing script generated", language="python")
                    
                    if preprocessing.get("preprocessing_steps"):
                        st.subheader("Preprocessing Steps")
                        for step in preprocessing.get("preprocessing_steps", []):
                            st.write(f"• {step}")
                    
                    if preprocessing.get("mapping_explanation"):
                        st.subheader("Mapping Explanation")
                        for explanation in preprocessing.get("mapping_explanation", []):
                            st.write(f"• {explanation}")
                    
                    if preprocessing.get("assumptions"):
                        st.subheader("Assumptions")
                        for assumption in preprocessing.get("assumptions", []):
                            st.write(f"• {assumption}")
            
            # Scripting output
            if st.session_state.pipeline_state.get("scripting"):
                scripting = st.session_state.pipeline_state["scripting"]
                with st.expander("💻 Generated Solver Code", expanded=True):
                    st.subheader("PuLP Solver Code")
                    code = scripting.get("code", "No code generated")
                    st.code(code, language="python")
                    
                    if scripting.get("output_schema"):
                        st.subheader("Output Schema")
                        st.json(scripting.get("output_schema"))
                    
                    if scripting.get("successful_implementation"):
                        st.success("✅ Implementation successful!")
                    else:
                        st.warning("⚠️ There may be issues with the implementation")
        
        # Only show modeling intercept after execution completes
        if st.session_state.show_modeling_intercept:
            st.divider()
            st.warning("⚠️ Please review and approve the mathematical model before continuing")
            
            with st.expander("🔢 Mathematical Model (Review & Edit)", expanded=True):
                # Display formatted modeling
                modeling_dict = st.session_state.original_modeling or st.session_state.pipeline_state.get("modelling")
                st.markdown(format_modeling_output(modeling_dict))
                
                st.divider()
                
                # Edit controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Approve & Continue", use_container_width=True):
                        try:
                            # Run the downstream agents without feedback
                            with st.spinner("⏳ Running preprocessing and solver agents..."):
                                preprocessing_output, scripting_output = run_downstream_agents(
                                    csv_path=st.session_state.csv_path,
                                    use_case_dict=st.session_state.pipeline_state.get("use_case", {}),
                                    modeling_dict=st.session_state.original_modeling or st.session_state.pipeline_state.get("modelling", {}),
                                    input_schema_payload=st.session_state.pipeline_state.get("input_schema_payload", {})
                                )
                                
                                # Update pipeline state with outputs
                                st.session_state.pipeline_state["preprocessing"] = preprocessing_output
                                st.session_state.pipeline_state["scripting"] = scripting_output
                                st.session_state.show_modeling_intercept = False
                                st.session_state.modeling_edited_content = None
                                st.session_state.execution_complete = True
                                add_log("✅ Pipeline completed")
                                st.success("✅ All agents completed! Review the results below.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to run downstream agents: {str(e)}")
                            add_log(f"Error running downstream agents: {str(e)}", level="error")
                
                with col2:
                    if st.button("✏️ Edit & Re-run", use_container_width=True):
                        st.session_state.modeling_edit_mode = True
                
                with col3:
                    if st.button("❌ Cancel Analysis", use_container_width=True):
                        st.session_state.execution_running = False
                        st.session_state.execution_complete = False
                        st.session_state.show_modeling_intercept = False
                        add_log("Analysis cancelled by user")
                        st.rerun()
                
                # Edit mode - User provides feedback
                if st.session_state.modeling_edit_mode:
                    st.subheader("Provide Feedback on Mathematical Model")
                    st.write("If some constraints are not feasible or need adjustment, describe the issues below:")
                    
                    feedback_text = st.text_area(
                        "Your feedback for the modeling agent",
                        value="",
                        height=200,
                        key="modeling_feedback_area",
                        placeholder="Example: The throughput constraint of 500 units/hour exceeds our current capacity of 400. Please adjust the objective to maximize profit instead of throughput, with a throughput constraint ≤ 400."
                    )
                    
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("🔄 Re-run with Feedback", use_container_width=True):
                            if not feedback_text.strip():
                                st.error("Please provide feedback before re-running")
                            else:
                                try:
                                    # Rerun all downstream agents with feedback
                                    with st.spinner("⏳ Processing feedback and regenerating all models..."):
                                        updated_modeling, updated_preprocessing, updated_scripting = rerun_modeling_with_feedback(
                                            csv_path=st.session_state.csv_path,
                                            feedback=feedback_text,
                                            use_case_dict=st.session_state.pipeline_state.get("use_case", {}),
                                            current_modeling_dict=modeling_dict,
                                            input_schema_payload=st.session_state.pipeline_state.get("input_schema_payload", {})
                                        )
                                        
                                        # Update pipeline state with ALL new outputs
                                        st.session_state.pipeline_state["modelling"] = updated_modeling
                                        st.session_state.pipeline_state["preprocessing"] = updated_preprocessing
                                        st.session_state.pipeline_state["scripting"] = updated_scripting
                                        st.session_state.modeling_edited_content = updated_modeling
                                        st.session_state.modeling_edit_mode = False
                                        st.session_state.show_modeling_intercept = False
                                        st.session_state.execution_complete = True
                                        
                                        add_log(f"All agents regenerated with user feedback: {feedback_text[:50]}...")
                                        st.success("✅ All models updated! Review the changes below.")
                                        st.rerun()
                                
                                except Exception as e:
                                    st.error(f"Failed to process feedback: {str(e)}")
                                    add_log(f"Error processing feedback: {str(e)}", level="error")
                    
                    with col_cancel:
                        if st.button("Cancel Feedback", use_container_width=True):
                            st.session_state.modeling_edit_mode = False
                            st.rerun()
    
    # ========================================================================
    # Final Results & Download
    # ========================================================================
    
    if st.session_state.execution_complete and st.session_state.pipeline_state:
        st.divider()
        st.header("🎉 Analysis Complete!")
        
        # Save results
        if not st.session_state.result_file:
            result_file = save_pipeline_results(
                st.session_state.pipeline_state,
                st.session_state.csv_filename,
                st.session_state.initial_prompt
            )
            st.session_state.result_file = result_file
            add_log(f"Results saved to {result_file}")
        
        # Download results
        results_json = json.dumps(st.session_state.pipeline_state, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Summary
        st.success("✅ All stages completed successfully!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", "Complete")
        with col2:
            st.metric("Total Logs", len(st.session_state.agent_logs))
        with col3:
            if st.session_state.result_file:
                st.metric("Results Saved", "✓")
        with col4:
            if st.session_state.modeling_edited_content:
                st.metric("Model Edited", "Yes")
    
    elif st.session_state.last_error:
        st.divider()
        st.error("❌ Analysis Failed")
        st.error(f"Error in stage: {st.session_state.error_stage}")
        st.error(f"Details: {st.session_state.last_error}")




