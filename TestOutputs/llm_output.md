## Problem Description
The goal is to optimize the quantity to produce for each Product_ID i.
## Model Formulation
The model is formulated as a Mixed-Integer Linear Programming (MILP) problem.
## Objective Function
The objective function is to maximize the total profit: $\max \sum_{i=1}^{n} P_i x_i$
## Constraints
The constraints of the model are:
* Machine A hours: $\sum_{i=1}^{n} M_{Ai} x_i \leq T_A$
* Machine B hours: $\sum_{i=1}^{n} M_{Bi} x_i \leq T_B$
* Labor hours: $\sum_{i=1}^{n} L_i x_i \leq T_L$
* Raw material units: $\sum_{i=1}^{n} R_i x_i \leq S$
* Market demand: $0 \leq x_i \leq D_i$ for all $i$
## Variables
The variable of the model is:
* $x_i$: Quantity of Product_ID i to be produced