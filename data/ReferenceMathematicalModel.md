## SALBP-1 to ILP (integer Linear Programming) Formulation

**Parameters**

| Symbol                 | Description                                                                                                                  |
|------------------------|------------------------------------------------------------------------------------------------------------------------------|
| $T$                    | Set of $n$ tasks, $t \in T$                                                                                                  |
| $v_t \in \mathbb{N}^+$ | Processing time required for task $t$                                                                                        |
| $S$                    | Set of $\bar{m}$ stations $s \in S = \{0..\bar{m}-1\}$, <br> ($\bar{m}$ is an upper bound for the number of stations needed) |
| $P = (T,E)$            | Precedence graph (cycle-free) containing an edge $(t_1, t_2) \in E \subseteq T^2$ iff $t_1$ has to be executed before $t_2$  |
| $c \in \mathbb{N}^+$   | Cycle time (max. amount of time a station has to execute all assigned tasks)                                                 |

**Variables**

| Variable             | Meaning                                              |
|----------------------|------------------------------------------------------|
| $x_{ts} \in \{0,1\}$ | $1$ if task $t$ is assigned to station $s$, $0$ else |
| $y_s \in \{0,1\}$    | $1$ if station $s$ is used, $0$ else                 |

**Integer Linear Programming Formulation**

$$
\begin{aligned}
\mathrm{min}\quad & \sum_{s} s \cdot y_s && \quad(1.1) \\
\mathrm{s.t.}\quad & \sum_{s \in S} x_{ts} = 1 && \forall t \in T, \quad(1.2) \\
& \sum_{t \in T} v_t \cdot x_{ts} \le c \cdot y_s && \forall s \in S, \quad(1.3) \\
& \sum_{s \in S} s \cdot x_{ts} \le \sum_{s \in S} s \cdot x_{t's} && \forall (t,t') \in E, \quad(1.4) \\
& y_s \ge y_{s+1} && \forall s \in S \setminus \{\bar{m}-1\}, \quad(1.5) \\
& x_{ts},\ y_s \in \{0,1\} && \forall t \in T,\ s \in S. \quad(1.6)
\end{aligned}
$$

**Explanation of the ILP Formulation**

1. (1.1) Objective: Minimize the total number of stations used and break symmetries (using earlier stations costs less, cf. (1.5)).
2. (1.2) Every task is assigned to exactly one station.
3. (1.3) Every station adheres to the cycle time.
4. (1.4) Every precedence relation is respected. (Every preceding task is assigned to an earlier or the same station as each of its succeeding tasks.)
5. (1.5) The stations used are consecutive, i.e., no intermediate station is empty. To minimize the objective, this constraint is not necessary. However, it is symmetry breaking and can thus help the ILP solver to finish faster. If the unweighted objective is used (w/o $s$ as coefficient), it ensures that the stations used in the solution are numbered consecutively.
6. (1.6) All variables are binary.