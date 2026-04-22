import pandas as pd
from pulp import (
    LpProblem,
    LpVariable,
    lpSum,
    LpMaximize,
    PULP_CBC_CMD,
    LpStatus,
)


def solve_product_mix(
    csv_path: str = "optimization_pipeline_test_easy.csv",
    capacity_machine_a: float = 100.0,
    capacity_machine_b: float = 100.0,
    capacity_labor: float = 500.0,
    capacity_raw_material: float = 500.0,
) -> dict:
    """
    Solve the product mix optimization MILP.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file with product data.
    capacity_machine_a : float
        Total available Machine A hours (C^A).
    capacity_machine_b : float
        Total available Machine B hours (C^B).
    capacity_labor : float
        Total available Labor hours (C^L).
    capacity_raw_material : float
        Total available Raw Material units (C^R).

    Returns
    -------
    dict
        Dictionary containing solution_status, objective_value,
        decision_variables, and solver_message.
    """
    # ── 1. Load input data ──────────────────────────────────────────
    df = pd.read_csv(csv_path)

    products = df["Product_ID"].tolist()

    profit = df["Profit_Per_Unit"].to_dict()
    machine_a_req = df["Machine_A_Hours_Req"].to_dict()
    machine_b_req = df["Machine_B_Hours_Req"].to_dict()
    labor_req = df["Labor_Hours_Req"].to_dict()
    raw_mat_req = df["Raw_Material_Units_Req"].to_dict()
    min_prod = df["Min_Production_Requirement"].to_dict()
    max_demand = df["Max_Market_Demand"].to_dict()

    # ── 2. Build the LP problem ─────────────────────────────────────
    prob = LpProblem("ProductMix", LpMaximize)

    # Decision variables: L_i ≤ x_i ≤ D_i
    x = {
        i: LpVariable(
            f"x_{i}",
            lowBound=min_prod.get(i, 0),
            upBound=max_demand.get(i, 0),
            cat="Continuous",
        )
        for i in products
    }

    # Objective: max Σ c_i * x_i
    prob += lpSum([profit[i] * x[i] for i in products])

    # Resource capacity constraints
    prob += lpSum([machine_a_req[i] * x[i] for i in products]) <= capacity_machine_a, "MachineA"
    prob += lpSum([machine_b_req[i] * x[i] for i in products]) <= capacity_machine_b, "MachineB"
    prob += lpSum([labor_req[i] * x[i] for i in products]) <= capacity_labor, "Labor"
    prob += lpSum([raw_mat_req[i] * x[i] for i in products]) <= capacity_raw_material, "RawMaterial"

    # ── 3. Solve ────────────────────────────────────────────────────
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # ── 4. Extract results ──────────────────────────────────────────
    status = LpStatus[prob.status]
    obj_value = prob.objective.value() if prob.objective else 0.0
    dec_vars = {i: x[i].varValue if x[i].varValue is not None else 0.0 for i in products}
    solver_msg = str(prob.message)

    return {
        "solution_status": status,
        "objective_value": float(obj_value),
        "decision_variables": dec_vars,
        "solver_message": solver_msg,
    }


# ── Quick CLI wrapper (optional) ────────────────────────────────────
if __name__ == "__main__":
    result = solve_product_mix()
    print("Status :", result["solution_status"])
    print("Obj.   :", result["objective_value"])
    print("Vars.  :", result["decision_variables"])
    print("Message:", result["solver_message"])