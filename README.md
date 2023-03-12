# Tinylp
A simple and embeddable linear programming solver with no external dependencies. 
It solves problems of type $$\min_x c^Tx\; Ax=b, x\ge 0,$$ where A is assumed to have full row rank.
To keep the Tinylp code simple the caller must convert more general problems to this form.
Problems with 50 variables and 20 equality constraints can be run in about 1 ms (Javascript version using Node on a M1 Mac). A testscase with 1000 variables and 400 constraints finishes in about 7s on the same machine.

Error checking is simplistic, and the library cannot diagnose all infeasible or ill-conditioned
programs. It is recommended to prototype linear programs using a more robust solver such as the ones available through scipy.optimize.linprog(). 

The implementation is based on the Primal-dual affine scaling solver described by John Wu in his course IMSE881 at Kansas state university. 
The interior point direction is computed using dense Cholesky factorization, which reasonably limits the solver to problems with less than 1000 variables. 

It is easy and encouraged to port this code to other languages, i.e. C, C++ etc. 
