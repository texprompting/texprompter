import streamlit as st
import pandas as pd
import json
import tempfile
from datetime import datetime
from pathlib import Path
from io import StringIO
import importlib
import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Reload changed modules when Streamlit runs in a long-lived process.
for module_name in ["services.pipeline_service"]:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from services.pipeline_service import PipelineService
pipeline_service = PipelineService()

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
        "pipeline_state": {},  
        "current_stage": None,
        "agent_logs": [],
        "execution_running": False,
        "execution_complete": False,
        "last_error": None,
        "error_stage": None,
        "modeling_edit_mode": False,
        "result_file": None,
        "show_modeling_intercept": False,
        "original_modeling": None,
        "use_intercept": True,  
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
def display_modeling_output(modeling_dict: dict):
    """Format modeling output supporting dynamic variable/parameter keys and math symbols."""
    if not modeling_dict:
        st.write("No modeling output available")
        return
    is_minimizing = modeling_dict.get("minimizing_problem", True)
    problem_type = "Minimization" if is_minimizing else "Maximization"
    st.header(f"MILP Model Documentation ({problem_type})")
    
    # ------------------------------------------------------------------------
    # 1. Objective Function Display (Using st.latex for Block Math)
    # ------------------------------------------------------------------------
    st.subheader("Objective Function")
    st.markdown(f"**{'Minimize' if is_minimizing else 'Maximize'}**")
    obj_fn = modeling_dict.get("objective_function", "")
    if obj_fn:
        # Strip off markdown math block wraps ($$) if the file accidentally saved them
        clean_obj = str(obj_fn).strip().strip("$")
        # Ensure backslashes survive internal string escaping
        try:
            clean_obj = clean_obj.encode('utf-8').decode('unicode_escape')
        except Exception:
            pass
        st.latex("\\min" if is_minimizing else "\\max" + clean_obj)
    else:
        st.write("No objective function defined.")
    col1, col2 = st.columns(2)
    
    # ------------------------------------------------------------------------
    # 2. Decision Variables (Using Inline Markdown Math $...$)
    # ------------------------------------------------------------------------
    with col1:
        with st.expander("Decision Variables", expanded=True):
            variables = modeling_dict.get("variables", [])
            if variables:
                for var in variables:
                    if isinstance(var, dict):
                        name = var.get("variable") or var.get("name") or ""
                        desc = var.get("meaning") or var.get("description") or ""
                    else:
                        name = getattr(var, "variable", None) or getattr(var, "name", "")
                        desc = getattr(var, "meaning", None) or getattr(var, "description", "")
                    
                    clean_name = str(name).strip().strip("$")
                    try:
                        clean_name = clean_name.encode('utf-8').decode('unicode_escape')
                    except Exception:
                        pass
                    st.markdown(f"${clean_name}$ : {desc}")
            else:
                st.write("No variables defined.")
    # ------------------------------------------------------------------------
    # 3. Parameters & Coefficients (Using Inline Markdown Math $...$)
    # ------------------------------------------------------------------------
    with col2:
        with st.expander("Parameters & Coefficients", expanded=True):
            parameters = modeling_dict.get("parameters", [])
            if parameters:
                for param in parameters:
                    if isinstance(param, dict):
                        symbol = param.get("symbol") or param.get("name") or ""
                        desc = param.get("description") or param.get("meaning") or ""
                    else:
                        symbol = getattr(param, "symbol", None) or getattr(var, "name", "")
                        desc = getattr(param, "meaning", None) or getattr(var, "description", "")
                    
                    clean_symbol = str(symbol).strip().strip("$")
                    try:
                        clean_symbol = clean_symbol.encode('utf-8').decode('unicode_escape')
                    except Exception:
                        pass
                    st.markdown(f"${clean_symbol}$ : {desc}")
            else:
                st.write("No parameters defined.")
    # ------------------------------------------------------------------------
    # 4. Constraints Display Loop (Using st.latex for Block Math Layouts)
    # ------------------------------------------------------------------------
    st.subheader("Constraints")
    constraints = modeling_dict.get("constraint_functions", [])
    if constraints:
        for constraint in constraints:
            if isinstance(constraint, str):
                # Clean mathematical markdown syntax wrappers out
                clean_constraint = constraint.strip().strip("$")
                
                # Force Python to correctly preserve single raw backslashes (e.g., \sum, \cdot) 
                # instead of misinterpreting them as standard Python string escape behaviors
                try:
                    clean_constraint = clean_constraint.encode('utf-8').decode('unicode_escape')
                except Exception:
                    pass
                
                st.latex(clean_constraint)
    else:
        st.write("No constraints defined.")
# ============================================================================
# Sidebar Configuration Interface
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"], key="csv_uploader")
    if uploaded_file:
        # Detect new uploaded file and clear previous states/cache
        if st.session_state.csv_filename != uploaded_file.name:
            st.session_state.csv_file = uploaded_file.getvalue()
            st.session_state.csv_filename = uploaded_file.name
            st.session_state.csv_path = None
            st.session_state.pipeline_state = {}
            st.session_state.execution_running = False
            st.session_state.execution_complete = False
            st.session_state.last_error = None
            st.session_state.error_stage = None
            st.session_state.show_modeling_intercept = False
            st.session_state.modeling_edit_mode = False
            st.session_state.result_file = None
            st.session_state.original_modeling = None
            st.session_state.agent_logs = []
        else:
            st.session_state.csv_file = uploaded_file.getvalue()
    
    st.session_state.initial_prompt = st.text_area(
        "Optional: Initial Prompt for Context Agent",
        value=st.session_state.initial_prompt,
        height=100
    )
    
    st.session_state.use_intercept = st.checkbox(
        "Enable Human Intercept Loop", 
        value=st.session_state.use_intercept,
        help="Check this to review everything up to Parameter Estimation before running downstream script generators."
    )
    
    if st.session_state.csv_file and not st.session_state.execution_running:
        if st.button("🚀 Start Analysis", use_container_width=True):
            st.session_state.execution_running = True
            st.session_state.execution_complete = False
            st.session_state.pipeline_state = {}  
            st.session_state.agent_logs = []
            st.session_state.last_error = None
            st.session_state.error_stage = None
            st.session_state.show_modeling_intercept = False
            st.session_state.modeling_edit_mode = False
            st.session_state.result_file = None
            st.session_state.original_modeling = None
            st.session_state.csv_path = None  # Clear temp path to force rebuilding on new run
            add_log("Analysis started...")
            st.rerun()
if not st.session_state.csv_file:
    st.info("👈 Please upload a CSV file to get started")
else:
    with st.expander("📊 CSV Preview", expanded=False):
        try:
            df = pd.read_csv(StringIO(st.session_state.csv_file.decode()))
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    log_container = st.container()
    outputs_container = st.container()
    
    # ========================================================================
    # Primary Pipeline Core Stream Engine
    # ========================================================================
    if st.session_state.execution_running:
        try:
            if not st.session_state.csv_path:
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
                    tmp.write(st.session_state.csv_file)
                    st.session_state.csv_path = tmp.name
            
            st.divider()
            st.header("📈 Real-Time Execution Progress")
            
            with log_container:
                with st.expander("📋 Execution Logs", expanded=True):
                    logs_placeholder = st.empty()
            
            displayed_stages = set()
            seen_traces = 0
            
            with st.status("Executing pipeline...", expanded=True) as status:
                for state_update in pipeline_service.start_pipeline(
                    csv_file_path=st.session_state.csv_path,
                    preview_rows=5,
                    initial_prompt=st.session_state.initial_prompt,
                ):
                    if isinstance(state_update, dict):
                        st.session_state.pipeline_state.update(state_update)
                        
                        traces = state_update.get("traces", [])
                        if isinstance(traces, list):
                            for trace in traces[seen_traces:]:
                                add_log(trace)
                                status.write(f"✓ {trace}")
                            seen_traces = len(traces)
                        
                        with logs_placeholder.container():
                            for log_entry in st.session_state.agent_logs:
                                st.text(log_entry["message"])
                        
                        if clean_update.get("errors"):
                            error = clean_update["errors"][-1]
                            st.session_state.last_error = error
                            st.session_state.error_stage = error.get("agent_name", "unknown")
                            raise Exception(f"Error in {error.get('agent_name')}: {error.get('message')}")
                        
                        # HALT CONDITION: Capture state right after parameter estimation settles
                        if "parameter_estimation" in clean_update and clean_update.get("parameter_estimation"):
                            if "parameter_estimation" not in displayed_stages:
                                displayed_stages.add("parameter_estimation")
                                if st.session_state.use_intercept:
                                    st.session_state.show_modeling_intercept = True
                                    st.session_state.execution_running = False
                                    add_log("⏸️ Human intercept triggered after Parameter Estimation.")
                                    st.rerun()
            # Automatic continuous execution path when intercept loop option is unchecked
            if not st.session_state.use_intercept:
                with st.spinner("⏳ Automatically running downstream code blocks..."):
                    raw_uc = st.session_state.pipeline_state.get("use_case", {})
                    raw_md = st.session_state.pipeline_state.get("modelling", {})
                    raw_sc = st.session_state.pipeline_state.get("input_schema_payload", {})
                    
                    downstream = pipeline_service.run_downstream(
                        csv_file_path=st.session_state.csv_path,
                        use_case=raw_uc,
                        modelling=raw_md,
                        input_schema_payload=raw_sc,
                        preview_rows=5,
                    )
                    st.session_state.pipeline_state["preprocessing"] = downstream.get("preprocessing")
                    st.session_state.pipeline_state["scripting"] = downstream.get("scripting")
            
            st.session_state.execution_running = False
            st.session_state.execution_complete = True
            st.rerun()
        
        except Exception as e:
            st.session_state.execution_running = False
            st.session_state.last_error = str(e)
            add_log(f"Error: {str(e)}", level="error")
            st.error(f"❌ Pipeline execution failed: {str(e)}")
            st.rerun()
    
    # ========================================================================
    # Modular UI Stage Output Components
    # ========================================================================
    if st.session_state.pipeline_state:
        with outputs_container:
            # 1. Use Case Details View
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
            
            # 2. Complete Equation Configuration Render Frame
            if st.session_state.pipeline_state.get("modelling"):
                if st.session_state.execution_complete or not st.session_state.show_modeling_intercept:
                    with st.expander("🔢 Mathematical Model & Parameters", expanded=True):
                        display_modeling_output(st.session_state.pipeline_state["modelling"])
        
        # ========================================================================
        # Human-in-the-loop Intercept UI Window Block
        # ========================================================================
        if st.session_state.show_modeling_intercept and not st.session_state.execution_complete:
            st.divider()
            st.warning("⚠️ Review Phase: Validate the generated mathematical configuration before proceeding to script writing.")
            
            with st.expander("Review Active Mathematical Model & Parameters", expanded=True):
                display_modeling_output(st.session_state.pipeline_state.get("modelling", {}))
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Approve Everything & Continue", use_container_width=True):
                    try:
                        with st.spinner("⏳ Compiling data preprocessing mappers and runtime scripts..."):
                            c_use_case = st.session_state.pipeline_state.get("use_case", {})
                            c_modeling = st.session_state.pipeline_state.get("modelling", {})
                            c_schema = st.session_state.pipeline_state.get("input_schema_payload", {})
                            
                            downstream = pipeline_service.run_downstream(
                                csv_file_path=st.session_state.csv_path,
                                use_case=c_use_case,
                                modelling=c_modeling,
                                input_schema_payload=c_schema,
                                preview_rows=5,
                            )
                            st.session_state.pipeline_state["preprocessing"] = downstream.get("preprocessing")
                            st.session_state.pipeline_state["scripting"] = downstream.get("scripting")
                            
                            st.session_state.show_modeling_intercept = False
                            st.session_state.execution_complete = True
                            add_log("✅ Downstream run completed successfully.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Downstream execution failed: {str(e)}")
            
            with col2:
                if st.button("✏️ Edit & Re-run Both Steps", use_container_width=True):
                    st.session_state.modeling_edit_mode = True
                    st.rerun()
            
            with col3:
                if st.button("❌ Cancel Run", use_container_width=True):
                    # Fully clear state and abort current run
                    st.session_state.pipeline_state = {}
                    st.session_state.execution_running = False
                    st.session_state.execution_complete = False
                    st.session_state.show_modeling_intercept = False
                    st.session_state.modeling_edit_mode = False
                    st.session_state.csv_path = None
                    st.session_state.agent_logs = []
                    st.session_state.result_file = None
                    st.session_state.original_modeling = None
                    st.session_state.last_error = None
                    st.session_state.error_stage = None
                    add_log("Run cancelled and execution state reset.")
                    st.rerun()
            
            if st.session_state.modeling_edit_mode:
                st.subheader("Provide Corrective Feedback")
                feedback_text = st.text_area("Detail your adjustments for the Modeling and Parameter estimation agents:", height=150)
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("🔄 Regenerate Model & Estimates", use_container_width=True):
                        if not feedback_text.strip():
                            st.error("Please supply explicit adjustment directions.")
                        else:
                            try:
                                with st.spinner("⏳ Regenerating configuration structures via feedback loop..."):
                                    c_use_case = st.session_state.pipeline_state.get("use_case", {})
                                    c_schema = st.session_state.pipeline_state.get("input_schema_payload", {})
                                    
                                    feedback_result = pipeline_service.regenerate_from_feedback(
                                        csv_file_path=st.session_state.csv_path,
                                        use_case=c_use_case,
                                        input_schema_payload=c_schema,
                                        feedback=feedback_text,
                                        preview_rows=5,
                                    )
                                    st.session_state.pipeline_state.update(feedback_result)
                                    
                                    st.session_state.modeling_edit_mode = False
                                    st.session_state.show_modeling_intercept = False
                                    st.session_state.execution_complete = True
                                    add_log("🔄 Pipeline updated via feedback modifications.")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error executing feedback adjustment: {str(e)}")
                
                with col_cancel:
                    if st.button("Cancel Feedback", use_container_width=True):
                        st.session_state.modeling_edit_mode = False
                        st.rerun()
    
    # ========================================================================
    # Export File Storage Handler
    # ========================================================================
    if st.session_state.execution_complete and st.session_state.pipeline_state:
        st.divider()
        st.header("🎉 Run Complete!")
        
        if not st.session_state.result_file:
            result_file = pipeline_service.save_results(
                st.session_state.pipeline_state,
                st.session_state.csv_filename,
                st.session_state.initial_prompt
            )
            st.session_state.result_file = result_file
        
        results_json = json.dumps(st.session_state.pipeline_state, indent=2, default=str)
        st.download_button(
            label="📥 Export Complete Results Payload (JSON)",
            data=results_json,
            file_name=f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    elif st.session_state.last_error:
        st.divider()
        st.error(f"❌ Execution stopped due to error: {st.session_state.last_error}")
# 5. Operational Script Layout & Real Optimization Results View
if st.session_state.pipeline_state.get("scripting"):
    scripting = st.session_state.pipeline_state["scripting"]
    with st.expander("💻 Generated Solver Code & Optimization Results", expanded=True):
        
        tab_metrics, tab_code = st.tabs(["📊 Optimization Results", "📄 Generated Python Code"])
        
        with tab_code:
            st.code(scripting.get("code", "# No code generated"), language="python")
            if scripting.get("successful_implementation"):
                st.success("✅ Code compiled and executed in the sandbox successfully!")
                
        with tab_metrics:
            st.header("🎯 Sandbox Optimization Outputs")
            
            status_val = scripting.get("solution_status") or ""
            obj_val = scripting.get("objective_value")
            dec_vars = scripting.get("decision_variables")
            
            if not status_val:
                status_val = "Executed Cleanly" if scripting.get("successful_implementation") else "Execution Failed"
            if not isinstance(dec_vars, dict):
                dec_vars = {}
            
            col1, col2 = st.columns(2)
            with col1:
                if "optimal" in str(status_val).lower():
                    st.metric(label="Solver Status", value=f"🟢 {status_val}")
                elif "infeasible" in str(status_val).lower():
                    st.metric(label="Solver Status", value=f"🔴 {status_val}")
                else:
                    st.metric(label="Solver Status", value=f"ℹ️ {status_val}")
                    
            with col2:
                if obj_val is not None:
                    try:
                        st.metric(label="Objective Value (Optimal Target)", value=f"{float(obj_val):,.2f}")
                    except (ValueError, TypeError):
                        st.metric(label="Objective Value (Optimal Target)", value=str(obj_val))
                else:
                    st.metric(label="Objective Value (Optimal Target)", value="No Single Objective Target Numeric Return")
                    
            st.divider()
            
            if dec_vars:
                st.subheader("💡 Optimal Decision Allocations")
                
                df_vars = pd.DataFrame(list(dec_vars.items()), columns=["Variable / Resource Allocation", "Calculated Optimal Value"])
                
                hide_zeros = st.checkbox("Hide variables with a 0 value allocation", value=False)
                if hide_zeros:
                    df_vars = df_vars[df_vars["Calculated Optimal Value"] > 0]
                    
                col_table, col_chart = st.columns([1, 1.2])
                with col_table:
                    st.markdown("**Allocation Data Frame**")
                    st.dataframe(df_vars, use_container_width=True, hide_index=True)
                    
                with col_chart:
                    st.markdown("**Visual Allocation Chart**")
                    if not df_vars.empty:
                        st.bar_chart(data=df_vars, x="Variable / Resource Allocation", y="Calculated Optimal Value", use_container_width=True)
                    else:
                        st.info("No rows match active filter bounds.")
            else:
                st.info("ℹ️ Optimization finished perfectly, but no decision variable allocation arrays were found in the return data frame.")