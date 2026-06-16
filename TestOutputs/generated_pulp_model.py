import pandas as pd
import pulp

# Load data
df = pd.read_csv('/var/folders/yl/x3_zrbc16q18h23q2t01c0lr0000gn/T/tmpmy48smww.csv')

# Define set I
I = df['Product_ID'].tolist()

# Define parameters
p_i = dict(zip(df['Product_ID'], df['Profit_Per_Unit']))
a_i = dict(zip(df['Product_ID'], df['Machine_A_Hours_Req']))
b_i = dict(zip(df['Product_ID'], df['Machine_B_Hours_Req']))
l_i = dict(zip(df['Product_ID'], df['Labor_Hours_Req']))
r_i = dict(zip(df['Product_ID'], df['Raw_Material_Units_Req']))
L_i = dict(zip(df['Product_ID'], df['Min_Production_Requirement'].astype(float)))
U_i = dict(zip(df['Product_ID'], df['Max_Market_Demand'].astype(float)))

# Resource capacities (RHS constants)
CAPACITY_A = 100.0
CAPACITY_B = 120.0
CAPACITY_L = 200.0
CAPACITY_R = 600.0

# Create the problem
prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable.dicts("x", I, lowBound=0, upBound=None, cat='Continuous')

# Apply lower and upper bounds to variables
for i in I:
    x[i].lowBound = L_i[i]
    x[i].upBound = U_i[i]

# Objective function
prob += pulp.lpSum([p_i[i] * x[i] for i in I]), "Total_Profit"

# Constraints
prob += pulp.lpSum([a_i[i] * x[i] for i in I]) <= CAPACITY_A, "Machine_A_Capacity"
prob += pulp.lpSum([b_i[i] * x[i] for i in I]) <= CAPACITY_B, "Machine_B_Capacity"
prob += pulp.lpSum([l_i[i] * x[i] for i in I]) <= CAPACITY_L, "Labor_Capacity"
prob += pulp.lpSum([r_i[i] * x[i] for i in I]) <= CAPACITY_R, "Raw_Material_Capacity"

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Extract results
decision_variables = {i: pulp.value(x[i]) for i in I}
objective_value = pulp.value(prob.objective)
solution_status = pulp.LpStatus[prob.status]
solver_message = f"Solver status: {solution_status}"

# Output schema
output = {
    "decision_variables": decision_variables,
    "objective_value": objective_value,
    "solution_status": solution_status,
    "solver_message": solver_message
}

print(output)