This folder contains a number of simple linear and quadratic optimization tests.


# Linear Problems

Linear optimization tests are stored in the following format:

```
FEASIBLE
NV NE NI
ELEMENTS_OF_F
ELEMENTS_OF_A
ELEMENTS_OF_B
ELEMENTS_OF_C
ELEMENTS_OF_D
SOLUTION
```

- The first line, `FEASIBLE` contains a single boolean - either `True` or `False` - telling if the problem has a solution.
- The second line contains three integers, respectively the number of variables `NV`, the number of equality constraints `NE` and the number of inequality constraints `NI`.
- The third line contains `NV` values, representing the coefficients of the decision variables in the objective function.
- If `NE>0`, the following line contains `NE*NV` values, corresponding to the equality constraint matrix `A` in row-major format - the first `NV` elements are thus the first row of `A`.
- If `NE>0`, the following line contains `NE` values, corresponding to the elements of the equality constraint vector `b`.
- If `NI>0`, the following line contains `NI*NV` values, corresponding to the inequality constraint matrix `C` in row-major format - the first `NV` elements are thus the first row of `C`.
- If `NI>0`, the following line contains `NI` values, corresponding to the elements of the inequality constraint vector `d`.
- The last line contains the solution vector - if the problem is infeasible, this line is still present but the content can be disregarded.


# Quadratic Problems

Quadratic optimization tests are stored in the following format:

```
FEASIBLE
NV NO NE NI
ELEMENTS_OF_Q
ELEMENTS_OF_R
ELEMENTS_OF_A
ELEMENTS_OF_B
ELEMENTS_OF_C
ELEMENTS_OF_D
SOLUTION
```

The format is almost identical to that of linear problems, the difference being the presence of an extra variable `NO` telling how many "tasks" are in the objective. The objective itself is defined by the matrix `Q` and the vector `r` - rather than a single vector `f` as in the linear case.
