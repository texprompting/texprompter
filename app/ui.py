import streamlit as st


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
        clean_obj = str(obj_fn).strip().strip("$")
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
                        symbol = getattr(param, "symbol", None) or getattr(param, "name", "")
                        desc = getattr(param, "meaning", None) or getattr(param, "description", "")

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
                clean_constraint = constraint.strip().strip("$")
                try:
                    clean_constraint = clean_constraint.encode('utf-8').decode('unicode_escape')
                except Exception:
                    pass
                st.latex(clean_constraint)
    else:
        st.write("No constraints defined.")
