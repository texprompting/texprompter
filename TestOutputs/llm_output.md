# Product Production Optimization Model

This Mixed-Integer Linear Programming (MILP) model optimizes the production quantities for a set of products to maximize total profit, given resource constraints and market requirements.

## Parameters
- $I$: Set of all products.
- $p_i$: Profit per unit of product $i$.
- $a_i$: Machine A hours required per unit of product $i$.
- $b_i$: Machine B hours required per unit of product $i$.
- $l_i$: Labor hours required per unit of product $i$.
- $r_i$: Raw material units required per unit of product $i$.
- $min\_q_i$: Minimum production requirement for product $i$.
- $max\_q_i$: Maximum market demand for product $i$.
- $C_A, C_B, C_L, C_R$: Total available capacities for Machine A, Machine B, Labor, and Raw Materials, respectively.

## Decision Variables
- $x_i$: Quantity of product $i$ to produce (Integer).

## Mathematical Formulation

**Objective:**
Maximize the total profit:
$$\max \sum_{i \in I} p_i x_i$$

**Constraints:**
1. **Machine A Capacity:** $\sum_{i \in I} a_i x_i \le C_A$
2. **Machine B Capacity:** $\sum_{i \in I} b_i x_i \le C_B$
3. **Labor Capacity:** $\sum_{i \in I} l_i x_i \le C_L$
4. **Raw Material Capacity:** $\sum_{i \in I} r_i x_i \le C_R$
5. **Minimum Requirement:** $x_i \ge min\_q_i, \quad \forall i \in I$
6. **Maximum Demand:** $x_i \le max\_q_i, \quad \forall i \in I$
7. **Integrality:** $x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I$

## Explanation
The model seeks to allocate limited resources (Machine hours, Labor, Raw Materials) among various products to maximize overall profit. It respects physical resource limits and business constraints such as minimum production quotas and maximum market saturation levels.