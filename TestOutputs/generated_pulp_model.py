import pandas as pd
import pulp
import json

def solve_optimization(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract sets and parameters
    P = df['Product_ID'].tolist()
    Profit = df.set_index('Product_ID')['Profit_Per_Unit'].to_dict()
    MachA = df.set_index('Product_ID')['Machine_A_Hours_Req'].to_dict()
    MachB = df.set_index('Product_ID')['Machine_B_Hours_Req'].to_dict()
    Labor = df.set_index('Product_ID')['Labor_Hours_Req'].to_dict()
    Mat = df.set_index('Product_ID')['Raw_Material_Units_Req'].to_dict()
    MinProd = df.set_index('Product_ID')['Min_Production_Requirement'].to_dict()
    MaxDem = df.set_index('Product_ID')['Max_Market_Demand'].to_dict()
    
    # Capacities
    Cap_A = 11000
    Cap_B = 13000
    Cap_L = 21600
    Cap_M = 65000
    
    # Define problem
    prob = pulp.LpProblem("Product_Mix_Optimization", pulp.LpMaximize)
    
    # Decision variables with bounds
    x = {p: pulp.LpVariable(f"x_{p}", lowBound=MinProd[p], upBound=MaxDem[p], cat='Continuous') for p in P}
    
    # Objective function
    prob += pulp.lpSum(Profit[p] * x[p] for p in P), "Total_Profit"
    
    # Constraints
    prob += pulp.lpSum(MachA[p] * x[p] for p in P) <= Cap_A, "Machine_A_Capacity"
    prob += pulp.lpSum(MachB[p] * x[p] for p in P) <= Cap_B, "Machine_B_Capacity"
    prob += pulp.lpSum(Labor[p] * x[p] for p in P) <= Cap_L, "Labor_Capacity"
    prob += pulp.lpSum(Mat[p] * x[p] for p in P) <= Cap_M, "Material_Capacity"
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Prepare output
    decision_variables = {p: pulp.value(x[p]) for p in P}
    objective_value = pulp.value(prob.objective)
    solution_status = pulp.LpStatus[status]
    solver_message = f"Optimization completed with status: {solution_status}"
    
    output = {
        "decision_variables": decision_variables,
        "objective_value": objective_value,
        "solution_status": solution_status,
        "solver_message": solver_message
    }
    
    return output

if __name__ == "__main__":
    csv_path = "/var/folders/yl/x3_zrbc16q18h23q2t01c0lr0000gn/T/tmph8kdqqit.csv"
    result = solve_optimization(csv_path)
    print(json.dumps(result, indent=4))
