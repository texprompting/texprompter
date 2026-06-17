import pandas as pd
import pulp

def solve_model():
    # Load the CSV file
    file_path = '/tmp/tmp4s30cp_l.csv'
    df = pd.read_csv(file_path)
    
    # Aggregate by Product_SKU to ensure uniqueness and handle any potential duplicates
    df_clean = df.groupby('Product_SKU', as_index=False).agg({
        'Sale_Price_Per_Unit': 'mean',
        'Material_Cost_Per_Unit': 'mean',
        'Molding_Mins_Req': 'mean',
        'Assembly_Mins_Req': 'mean',
        'Max_Market_Capacity': 'mean'
    })
    
    # Define the set of products
    P = df_clean['Product_SKU'].tolist()
    
    # Define parameters as dictionaries mapping Product_SKU to values
    S = df_clean.set_index('Product_SKU')['Sale_Price_Per_Unit'].to_dict()
    C = df_clean.set_index('Product_SKU')['Material_Cost_Per_Unit'].to_dict()
    M = df_clean.set_index('Product_SKU')['Molding_Mins_Req'].to_dict()
    A = df_clean.set_index('Product_SKU')['Assembly_Mins_Req'].to_dict()
    D = df_clean.set_index('Product_SKU')['Max_Market_Capacity'].to_dict()
    
    # Initialize the problem
    prob = pulp.LpProblem("Product_Mix_Optimization", pulp.LpMaximize)
    
    # Decision variables
    x = {p: pulp.LpVariable(f"x_{p}", lowBound=0, upBound=D[p], cat='Continuous') for p in P}
    
    # Objective function
    prob += pulp.lpSum((S[p] - C[p]) * x[p] for p in P), "Total_Profit"
    
    # Constraints
    prob += pulp.lpSum(M[p] * x[p] for p in P) <= 450000.0, "Molding_Capacity"
    prob += pulp.lpSum(A[p] * x[p] for p in P) <= 450000.0, "Assembly_Capacity"
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Prepare output
    decision_variables = {p: float(pulp.value(x[p])) for p in P}
    objective_value = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else 0.0
    solution_status = pulp.LpStatus[status]
    solver_message = f"Optimization status: {solution_status}"
    
    output = {
        "decision_variables": decision_variables,
        "objective_value": objective_value,
        "solution_status": solution_status,
        "solver_message": solver_message
    }
    
    return output

if __name__ == '__main__':
    res = solve_model()
    print(res)
