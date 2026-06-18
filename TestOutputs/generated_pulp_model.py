import pandas as pd
import pulp
from typing import Any

def solve_model(csv_path: str) -> dict[str, Any]:
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Preprocessing: Ensure correct data types and handle any missing values
    df['HeatCoverClosed'] = df['HeatCoverClosed'].fillna(0).astype(int)
    df['Produce'] = df['Produce'].fillna(0).astype(int)
    
    # Define the set of time steps
    T = df.index.tolist()
    
    # Define parameters as dictionaries indexed by time step t
    H = df['HeatCoverClosed'].to_dict()
    R = df['Produce'].to_dict()
    
    # Scalar parameters
    alpha = 15.0
    beta = 3.0
    gamma = 0.2
    P_min = 10.0
    
    # Define the problem (minimizing cost/energy)
    prob = pulp.LpProblem("Heater_Fan_Optimization", pulp.LpMinimize)
    
    # Decision variables
    # x_t: HeaterOn (binary)
    # y_t: FanOn (binary)
    # u_t: HeatingTimePercent (continuous, between 0 and 100)
    x = pulp.LpVariable.dicts("x", T, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", T, cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", T, lowBound=0, upBound=100, cat=pulp.LpContinuous)
    
    # Constraints
    for t in T:
        # x_t <= H_t
        prob += x[t] <= H[t], f"HeatCoverClosed_constraint_{t}"
        # x_t >= R_t
        prob += x[t] >= R[t], f"Produce_constraint_{t}"
        # u_t <= 100 * x_t
        prob += u[t] <= 100 * x[t], f"Max_HeatingTimePercent_{t}"
        # u_t >= P_min * x_t
        prob += u[t] >= P_min * x[t], f"Min_HeatingTimePercent_{t}"
        # y_t >= x_t
        prob += y[t] >= x[t], f"Fan_Heater_relation_{t}"
        
    # Objective function
    prob += pulp.lpSum(alpha * x[t] + beta * y[t] + gamma * u[t] for t in T)
    
    # Solve the problem
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract decision variables
    decision_variables = {}
    for t in T:
        decision_variables[f"x_{t}"] = float(x[t].varValue) if x[t].varValue is not None else 0.0
        decision_variables[f"y_{t}"] = float(y[t].varValue) if y[t].varValue is not None else 0.0
        decision_variables[f"u_{t}"] = float(u[t].varValue) if u[t].varValue is not None else 0.0
        
    objective_value = pulp.value(prob.objective)
    solution_status = pulp.LpStatus[status]
    
    return {
        "solution_status": solution_status,
        "objective_value": objective_value,
        "decision_variables": decision_variables,
        "solver_message": f"Optimization completed with status: {solution_status}"
    }

if __name__ == "__main__":
    # Example execution with default path
    import os
    csv_path = "/var/folders/yl/x3_zrbc16q18h23q2t01c0lr0000gn/T/tmpsmp8c87w.csv"
    if os.path.exists(csv_path):
        results = solve_model(csv_path)
        print("Status:", results["solution_status"])
        print("Objective Value:", results["objective_value"])
    else:
        print(f"File not found: {csv_path}")
