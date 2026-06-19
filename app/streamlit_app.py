import streamlit as st
import pandas as pd
import json
import importlib
import sys
from io import StringIO
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Reload changed modules when Streamlit runs in a long-lived process.
for module_name in ["app.state", "app.ui", "app.actions"]:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from app.state import initialize_session_state, add_log
from app.ui import display_modeling_output
from app.actions import (
    execute_pipeline,
    approve_and_continue,
    regenerate_feedback,
    reset_run_state,
    save_results,
)

# Configure Streamlit
st.set_page_config(page_title="TexPrompter - Workflow Optimizer", layout="wide")
st.title("TexPrompter - Workflow Optimizer")

initialize_session_state()

# ============================================================================
# Sidebar Configuration Interface
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"], key="csv_uploader")
    if uploaded_file:
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
        height=100,
    )

    st.session_state.use_intercept = st.checkbox(
        "Enable Human Intercept Loop",
        value=st.session_state.use_intercept,
        help="Check this to review everything up to Parameter Estimation before running downstream script generators.",
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
            st.session_state.csv_path = None
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

    with log_container:
        with st.expander("📋 Execution Logs", expanded=True):
            for log_entry in st.session_state.agent_logs:
                st.text(log_entry["message"])

    if st.session_state.execution_running:
        try:
            execute_pipeline()
        except Exception as e:
            st.session_state.execution_running = False
            st.session_state.last_error = str(e)
            add_log(f"Error: {str(e)}", level="error")
            st.error(f"❌ Pipeline execution failed: {str(e)}")
            st.rerun()

    if st.session_state.pipeline_state:
        with outputs_container:
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

            if st.session_state.pipeline_state.get("modelling"):
                if st.session_state.execution_complete or not st.session_state.show_modeling_intercept:
                    with st.expander("🔢 Mathematical Model & Parameters", expanded=True):
                        display_modeling_output(st.session_state.pipeline_state["modelling"])

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
                            approve_and_continue()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Downstream execution failed: {str(e)}")

            with col2:
                if st.button("✏️ Edit & Re-run Both Steps", use_container_width=True):
                    st.session_state.modeling_edit_mode = True
                    st.rerun()

            with col3:
                if st.button("❌ Cancel Run", use_container_width=True):
                    reset_run_state()
                    add_log("Run cancelled and execution state reset.")
                    st.rerun()

            if st.session_state.modeling_edit_mode:
                st.subheader("Provide Corrective Feedback")
                feedback_text = st.text_area(
                    "Detail your adjustments for the Modeling and Parameter estimation agents:",
                    height=150,
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("🔄 Regenerate Model & Estimates", use_container_width=True):
                        if not feedback_text.strip():
                            st.error("Please supply explicit adjustment directions.")
                        else:
                            try:
                                with st.spinner("⏳ Regenerating configuration structures via feedback loop..."):
                                    regenerate_feedback(feedback_text)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error executing feedback adjustment: {str(e)}")

                with col_cancel:
                    if st.button("Cancel Feedback", use_container_width=True):
                        st.session_state.modeling_edit_mode = False
                        st.rerun()

    if st.session_state.execution_complete and st.session_state.pipeline_state:
        st.divider()
        st.header("🎉 Run Complete!")

        if not st.session_state.result_file:
            result_file = save_results()
            st.session_state.result_file = result_file

        results_json = json.dumps(st.session_state.pipeline_state, indent=2, default=str)
        st.download_button(
            label="📥 Export Complete Results Payload (JSON)",
            data=results_json,
            file_name=f"pipeline_results_{st.session_state.csv_filename.replace('.csv', '')}_{st.session_state.initial_prompt[:20]}.json",
            mime="application/json",
            use_container_width=True,
        )

    elif st.session_state.last_error:
        st.divider()
        st.error(f"❌ Execution stopped due to error: {st.session_state.last_error}")

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