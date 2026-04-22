## Product Mix Optimization MILP Formulation

**Sets and Indices:**
- $P$: Set of products, indexed by $i$

**Parameters:**
- $c_i$: Profit per unit of product $i$
- $m^A_i$: Machine A hours required per unit of product $i$
- $m^B_i$: Machine B hours required per unit of product $i$
- $l_i$: Labor hours required per unit of product $i$
- $r_i$: Raw material units required per unit of product $i$
- $M_i$: Maximum market demand for product $i$
- $L_i$: Minimum production requirement for product $i$
- $C^A$: Total available Machine A hours
- $C^B$: Total available Machine B hours
- $C^L$: Total available Labor hours
- $C^R$: Total available Raw Material units

**Decision Variables:**
- $x_i$: Quantity of product $i$ to produce ($\ge 0$)

**Objective Function:**
Maximize Total Profit:
$$\max \sum_{i \in P} c_i x_i$$

**Constraints:**
1. **Machine A Capacity:** $\sum_{i \in P} m^A_i x_i \le C^A$
2. **Machine B Capacity:** $\sum_{i \in P} m^B_i x_i \le C^B$
3. **Labor Capacity:** $\sum_{i \in P} l_i x_i \le C^L$
4. **Raw Material Capacity:** $\sum_{i \in P} r_i x_i \le C^R$
5. **Demand Bounds:** $L_i \le x_i \le M_i \quad \forall i \in P$
6. **Non-negativity:** $x_i \ge 0 \quad \forall i \in P$

**Explanation:**
This model optimizes the product mix to maximize profit while respecting resource limitations and market constraints. The resource capacity parameters ($C^A, C^B, C^L, C^R$) are typically external constraints not present in the provided CSV but necessary for a complete formulation.