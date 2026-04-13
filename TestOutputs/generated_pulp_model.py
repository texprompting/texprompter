from pulp import LpMaximize, LpProblem, lpSum, LpVariable

def get_input_schema_payload():
    return {
        "n": 5,
        "m": 3,
        "profits": [10, 20, 30, 40, 50],
        "weights": [1, 2, 3, 4, 5],
        "capacity": 15
    }


def get_requested_output_schema():
    return {
        "objective": float,
        "selected_items": list,
        "total_profit": float,
        "total_weight": float
    }


def get_mathematical_model(input_payload):
    model = LpProblem(name="knapsack-problem", sense=LpMaximize)
    
    # Initialize the problem variables
    item_vars = LpVariable.dicts("Item", range(input_payload["n"]), lowBound=0, upBound=1, cat='Integer')
    
    # Add the objective
    model += lpSum([input_payload["profits"][i] * item_vars[i] for i in range(input_payload["n"])])
    
    # Add constraint
    model += lpSum([input_payload["weights"][i] * item_vars[i] for i in range(input_payload["n"])]) <= input_payload["capacity"]
    
    # Solve the problem
    status = model.solve()
    
    # Get the selected items
    selected_items = [i for i in range(input_payload["n"]) if item_vars[i].varValue]
    
    return {
        "objective": model.objective.value(),
        "selected_items": selected_items,
        "total_profit": sum([input_payload["profits"][i] for i in selected_items]),
        "total_weight": sum([input_payload["weights"][i] for i in selected_items])
    }


def main():
    input_schema = get_input_schema_payload()
    result = get_mathematical_model(input_schema)
    output_schema = get_requested_output_schema()
    
    print(f"Result: {json.dumps(result, indent=4)}")
    print(f"Output Schema: {output_schema}")

if __name__ == "__main__":
    main()