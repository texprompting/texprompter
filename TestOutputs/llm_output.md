## Production Planning MILP Formulation

### Parameters
- $I$: Set of products
- $c_i$: Profit per unit of product $i$
- $a_i$: Machine A hours per unit of product $i$
- $b_i$: Machine B hours per unit of product $i$
- $l_i$: Labor hours per unit of product $i$
- $r_i$: Raw material units per unit of product $i$
- $dmin_i$: Minimum production requirement for product $i$
- $dmax_i$: Maximum market demand for product $i$
- $C_A, C_B, C_L, C_R$: Available capacities for Machine A, Machine B, Labor, and Raw Material
- $D_{min}, D_{max}$: Global minimum and maximum total production requirements

### Decision Variables
- $x_i$: Production quantity for product $i$

### Objective
Maximize total profit: $\max \sum_{i \in I} c_i x_i$

### Constraints
1. **Machine A Capacity**: Total hours used on Machine A cannot exceed $C_A$.
2. **Machine B Capacity**: Total hours used on Machine B cannot exceed $C_B$.
3. **Labor Capacity**: Total labor hours used cannot exceed $C_L$.
4. **Raw Material Capacity**: Total raw material units used cannot exceed $C_R$.
5. **Product Demand Bounds**: Each product's production must stay within its specific min/max bounds.
6. **Global Production Bounds**: Total production across all products must stay within global min/max limits.
7. **Integrality**: Production quantities must be non-negative integers.

### Explanation
This MILP model determines the optimal production mix to maximize total profit while respecting resource limitations (machines, labor, materials) and market constraints (per-product and global demand bounds). The formulation ensures feasible production levels that align with business goals and operational capacities.