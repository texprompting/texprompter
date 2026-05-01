import pandas as pd
import pulp

# Load data
csv_file_path = "/home/bene/Documents/Coding/Uni/KI-Projekt/texprompter/data/optimization_pipeline_test_easy.csv"
df = pd.read_csv(csv_file_path)

# Define index set P
P = df['Product_ID'].tolist()

# Define parameters as dictionaries keyed by product ID
c = dict(zip(df['Product_ID'], df['Profit_Per_Unit']))
M_A = dict(zip(df['Product_ID'], df['Machine_A_Hours_Req']))
M_B = dict(zip(df['Product_ID'], df['Machine_B_Hours_Req']))
L = dict(zip(df['Product_ID'], df['Labor_Hours_Req']))
R = dict(zip(df['Product_ID'], df['Raw_Material_Units_Req']))
D = dict(zip(df['Product_ID'], df['Max_Market_Demand']))
m = dict(zip(df['Product_ID'], df['Min_Production_Requirement']))

# Global resource capacities (external constants not in CSV)
C_A = 1000  # Total available Machine A hours
C_B = 800   # Total available Machine B hours
C_L = 500   # Total available Labor hours
C_R = 2000  # Total available Raw material units

# Create the problem (maximization)
prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

# Create decision variables (non-negative integers)
x = pulp.LpVariable.dicts("Production", P, lowBound=0, cat='Integer')

# Objective function: Maximize total profit
prob += pulp.lpSum([c[p] * x[p] for p in P]), "Total_Profit"

# Constraints
# Machine A capacity
prob += pulp.lpSum([M_A[p] * x[p] for p in P]) <= C_A, "Machine_A_Capacity"

# Machine B capacity
prob += pulp.lpSum([M_B[p] * x[p] for p in P]) <= C_B, "Machine_B_Capacity"

# Labor capacity
prob += pulp.lpSum([L[p] * x[p] for p in P]) <= C_L, "Labor_Capacity"

# Raw material capacity
prob += pulp.lpSum([R[p] * x[p] for p in P]) <= C_R, "Raw_Material_Capacity"

# Demand limits (upper bound)
for p in P:
    prob += x[p] <= D[p], f"Demand_Limit_{p}"

# Minimum production requirements (lower bound)
for p in P:
    prob += x[p] >= m[p], f"Min_Production_{p}"

# Solve the problem
prob.solve()

# Extract results
decision_variables = {p: pulp.value(x[p]) for p in P}
objective_value = pulp.value(prob.objective)
solution_status = pulp.LpStatus[prob.status]

# Generate solver message based on status
if solution_status == 'Optimal':
    solver_message = "Optimal solution found."
elif solution_status == 'Infeasible':
    solver_message = "No feasible solution exists."
elif solution_status == 'Unbounded':
    solver_message = "The problem is unbounded."
else:
    solver_message = f"Solver status: {solution_status}"

# Return the results in the requested schema
results = {
    "decision_variables": decision_variables,
    "objective_value": objective_value,
    "solution_status": solution_status,
    "solver_message": solver_message
}