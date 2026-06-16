## Production Planning MILP Formulation

### Parameters
- $P$: Set of products (50 products in the dataset)
- $\\pi_p = 47.03$: Mean Profit per unit of product $p$
- $a_p = 1.71$: Mean Machine A hours required per unit of product $p$
- $b_p = 2.01$: Mean Machine B hours required per unit of product $p$
- $l_p = 3.34$: Mean Labor hours required per unit of product $p$
- $r_p = 10.08$: Mean Raw material units required per unit of product $p$
- $m_p = 5.8$: Mean Minimum production requirement for product $p$
- $d_p = 258.78$: Mean Maximum market demand for product $p$
- $C_A = 171.0$: Available capacity for Machine A (hours)
- $C_B = 201.0$: Available capacity for Machine B (hours)
- $C_L = 334.0$: Available capacity for Labor (hours)
- $C_R = 1008.0$: Available capacity for Raw Material (units)

### Decision Variables
- $x_p$: Production quantity of product $p$

### Objective Function
Maximize total profit:
$$\\max \\sum_{p \\in P} 47.03 x_p$$

### Constraints
1. **Machine A Capacity**: $\\sum_{p \\in P} 1.71 x_p \\le 171.0$
2. **Machine B Capacity**: $\\sum_{p \\in P} 2.01 x_p \\le 201.0$
3. **Labor Capacity**: $\\sum_{p \\in P} 3.34 x_p \\le 334.0$
4. **Raw Material Capacity**: $\\sum_{p \\in P} 10.08 x_p \\le 1008.0$
5. **Market Demand**: $x_p \\le 258.78 \\quad \\forall p \\in P$
6. **Minimum Production**: $x_p \\ge 5.8 \\quad \\forall p \\in P$
7. **Non-negativity**: $x_p \\ge 0 \\quad \\forall p \\in P$

### Explanation
This model determines the optimal production mix to maximize total profit while respecting resource limitations (machines, labor, raw materials) and market constraints (minimum production requirements and maximum demand). The parameters $\\pi_p, a_p, b_p, l_p, r_p, m_p, d_p$ are represented by their mean values from the dataset statistics to provide a representative aggregate model. The aggregate capacities ($C_A, C_B, C_L, C_R$) were estimated based on a representative production batch size of 100 units using these mean resource requirements. The formulation is a linear program that can be solved efficiently using standard MILP solvers.