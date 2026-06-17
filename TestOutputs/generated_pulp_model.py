import pandas as pd
import pulp

# 1. Load Data
csv_path = "/var/folders/yl/x3_zrbc16q18h23q2t01c0lr0000gn/T/tmpwhpprvlw.csv"
df = pd.read_csv(csv_path)

# 2. Define Parameters
P = df['Product_ID'].tolist()
pi_p = dict(zip(df['Product_ID'], df['Profit_Per_Unit']))
a_p = dict(zip(df['Product_ID'], df['Machine_A_Hours_Req']))
b_p = dict(zip(df['Product_ID'], df['Machine_B_Hours_Req']))
l_p = dict(zip(df['Product_ID'], df['Labor_Hours_Req']))
r_p = dict(zip(df['Product_ID'], df['Raw_Material_Units_Req']))
min_p = dict(zip(df['Product_ID'], df['Min_Production_Requirement']))
max_p = dict(zip(df['Product_ID'], df['Max_Market_Demand']))

# 3. Define Capacity Constants
CAPACITY_A = 16600.0
CAPACITY_B = 19500.0
CAPACITY_L = 32400.0
CAPACITY_R = 98000.0

# 4. Create Problem
prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

# 5. Create Decision Variables (Integer)
x = pulp.LpVariable.dicts("x", P, lowBound=0, cat=pulp.LpInteger)

# 6. Add Objective Function
prob += pulp.lpSum([pi_p[p] * x[p] for p in P]), "Total_Profit"

# 7. Add Constraints
prob += pulp.lpSum([a_p[p] * x[p] for p in P]) <= CAPACITY_A, "Machine_A_Capacity"
prob += pulp.lpSum([b_p[p] * x[p] for p in P]) <= CAPACITY_B, "Machine_B_Capacity"
prob += pulp.lpSum([l_p[p] * x[p] for p in P]) <= CAPACITY_L, "Labor_Capacity"
prob += pulp.lpSum([r_p[p] * x[p] for p in P]) <= CAPACITY_R, "Raw_Material_Capacity"

# 8. Apply Variable Bounds
for p in P:
    x[p].lowBound = min_p[p]
    x[p].upBound = max_p[p]

# 9. Solve
prob.solve()

# 10. Extract Results
result = {
    "decision_variables": {p: x[p].varValue for p in P},
    "objective_value": pulp.value(prob.objective),
    "solution_status": pulp.LpStatus[prob.status],
    "solver_message": str(pulp.LpStatus[prob.status])
}

print(result)