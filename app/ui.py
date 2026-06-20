import re

import streamlit as st


def normalize_latex(s: str) -> str:
    """Normalize LaTeX strings from LLM output for Streamlit rendering.

    LLMs produce varying backslash-escaping levels depending on how the
    JSON structured output is serialised:

    * Double-escaped: ``\\\\sum`` (2 actual backslashes) → ``\\sum``
    * Already correct: ``\\sum`` (1 backslash) → ``\\sum`` (unchanged)

    ``st.latex()`` and Streamlit's inline ``$…$`` markdown both require
    exactly one backslash per LaTeX command.  This function collapses
    runs of 2+ consecutive backslashes into a single backslash so the
    result is always correct regardless of the upstream escaping level.
    """
    if not s:
        return s
    # Strip surrounding whitespace and dollar-sign delimiters some models emit
    s = str(s).strip().strip("$")
    # Collapse 2+ consecutive backslashes → 1 backslash
    return re.sub(r"\\{2,}", lambda _: "\\", s)


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
    obj_fn = modeling_dict.get("objective_function", "")
    if obj_fn:
        clean_obj = normalize_latex(obj_fn)
        prefix = r"\min \;" if is_minimizing else r"\max \;"
        st.latex(prefix + clean_obj)
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

                    clean_name = normalize_latex(name)
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
                        symbol = getattr(param, "symbol", None) or getattr(param, "name", "")
                        desc = getattr(param, "meaning", None) or getattr(param, "description", "")

                    clean_symbol = normalize_latex(symbol)
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
                clean_constraint = normalize_latex(constraint)
                st.latex(clean_constraint)
    else:
        st.write("No constraints defined.")
