import streamlit as st
from datetime import datetime


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


def add_log(message: str, level: str = "info"):
    """Add a log message to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    st.session_state.agent_logs.append({"message": formatted_msg, "level": level})
