# Tinylp
A simple and embeddable linear programming solver with no external dependencies. 
It solves problems of type min c'x s.t. Ax=b, x>=0. To keep the Tinylp code simple the caller must convert more general problems to this form.

The implementation is based on the Primal-dual affine scaling solver described by John Wu in his course IMSE881 at Kansas state university. 
The interior point direction is computed using dense Cholesky factorization, which reasonably limits the solver to problems with less than 1000 variables. 

