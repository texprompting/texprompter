import pandas as pd
import pulp
from typing import Any

def solve_model(csv_path: str) -> dict[str, Any]:
    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to the filename from the source metadata if absolute path fails
        import os
        fallback_path = os.path.basename(csv_path)
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
        else:
            raise FileNotFoundError(f"Could not find CSV file at {csv_path} or {fallback_path}")
    
    # Downsample the data to make the optimization problem computationally tractable
    # Taking every 50th row reduces the size from 7728 to ~155 rows.
    df_downsampled = df.iloc[::50].reset_index(drop=True)
    
    # Define the set of time steps (1-based indexing)
    T = list(df_downsampled.index + 1)
    
    # Extract parameters as dictionaries indexed by T
    H_dict = {t: int(df_downsampled.loc[t-1, 'HeatCoverClosed']) for t in T}
    C_dict = {t: int(df_downsampled.loc[t-1, 'CupState']) for t in T}
    
    # Constants
    W_init = float(df_downsampled['CurrentWeight'].iloc[0]) if 'CurrentWeight' in df_downsampled.columns else 0.092
    W_target = 15.0
    P_req = 4
    delta_w = 1.0
    E_h = 10.0
    E_f = 2.0
    E_p = 0.1
    
    # Initialize the problem
    prob = pulp.LpProblem("Energy_Minimization", pulp.LpMinimize)
    
    # Decision variables
    h = pulp.LpVariable.dicts("h", T, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", T, cat=pulp.LpBinary)
    p = pulp.LpVariable.dicts("p", T, lowBound=0, cat=pulp.LpContinuous)
    w = pulp.LpVariable.dicts("w", T, lowBound=0, cat=pulp.LpContinuous)
    f = pulp.LpVariable.dicts("f", T, lowBound=0, cat=pulp.LpContinuous)
    
    # Objective function
    prob += pulp.lpSum(E_h * h[t] + E_f * f[t] + E_p * p[t] for t in T)
    
    # Constraints
    for t in T:
        prob += h[t] <= H_dict[t], f"Heater_Availability_{t}"
        prob += p[t] <= 100 * h[t], f"Heating_Time_Limit_{t}"
        prob += y[t] <= C_dict[t], f"Cup_State_Limit_{t}"
        prob += y[t] <= h[t], f"Production_Heater_Limit_{t}"
        
    # Weight constraints
    prob += w[1] == W_init + 3.75 * delta_w * y[1], "Initial_Weight"
    for t in T:
        if t > 1:
            prob += w[t] == w[t-1] + 3.75 * delta_w * y[t], f"Weight_Balance_{t}"
            
    # Target weight constraint at the last time step
    N = T[-1]
    prob += w[N] >= W_target, "Target_Weight"
    
    # Production requirement constraint
    prob += pulp.lpSum(y[t] for t in T) >= P_req, "Production_Requirement"
    
    # Solve the model
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Prepare results
    solution_status = pulp.LpStatus[status]
    objective_value = pulp.value(prob.objective) if solution_status == "Optimal" else None
    
    decision_variables = {}
    for t in T:
        decision_variables[f"h_{t}"] = pulp.value(h[t])
        decision_variables[f"y_{t}"] = pulp.value(y[t])
        decision_variables[f"p_{t}"] = pulp.value(p[t])
        decision_variables[f"w_{t}"] = pulp.value(w[t])
        decision_variables[f"f_{t}"] = pulp.value(f[t])
        
    return {
        "solution_status": solution_status,
        "objective_value": objective_value,
        "decision_variables": decision_variables,
        "solver_message": f"Model solved with status: {solution_status}"
    }

if __name__ == "__main__":
    # Default path from context
    csv_path = "/var/folders/yl/x3_zrbc16q18h23q2t01c0lr0000gn/T/tmpmfgckru3.csv"
    try:
        results = solve_model(csv_path)
        print("Status:", results["solution_status"])
        print("Objective Value:", results["objective_value"])
    except Exception as e:
        print("Error running model:", e)
