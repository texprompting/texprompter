import pandas as pd
from pulp import LpMaximize, LpProblem, lpSum, LpVariable, value

df = pd.read_csv('texprompter/data/optimization_pipeline_test_easy.csv')
model = LpProblem(name="production-optimization", sense=LpMaximize)
product_ids = df['Product_ID']
x = LpVariable.dicts("Quantity", product_ids, lowBound=0, upBound=None, cat='Integer')
model += lpSum([df.loc[i, 'Profit_Per_Unit'] * x[product_ids[i]] for i in range(len(product_ids))])
machine_a_hours = df['Machine_A_Hours_Req']
machine_b_hours = df['Machine_B_Hours_Req']
labor_hours = df['Labor_Hours_Req']
raw_material_units = df['Raw_Material_Units_Req']
model += lpSum([machine_a_hours[i] * x[product_ids[i]] for i in range(len(product_ids))]) <= 1000.0
model += lpSum([machine_b_hours[i] * x[product_ids[i]] for i in range(len(product_ids))]) <= 1000.0
model += lpSum([labor_hours[i] * x[product_ids[i]] for i in range(len(product_ids))]) <= 1500.0
model += lpSum([raw_material_units[i] * x[product_ids[i]] for i in range(len(product_ids))]) <= 5000.0
min_production_requirements = df['Min_Production_Requirement']
max_market_demands = df['Max_Market_Demand']
for i in range(len(product_ids)):
    model += x[product_ids[i]] >= min_production_requirements[i]
    model += x[product_ids[i]] <= max_market_demands[i]
status = model.solve()
print(f"Status: {model.status}")
if status == 1:
    print("Optimal solution found.")
    print(f"Objective value: {value(model.objective)}")
    for v in x.values():
        print(f"{v.name}: {v.value()}")
else:
    print("No optimal solution found.")