//This primal-dual affine scaling method was implemented from the lecture notes of 
//John Wu for the class IMSE881 at Kansas State University.

// Attempt to solve the linear program
// min_x c'x
// st. Ax = b, x >= 0.

// Return x, s, status, where s is the dual solution.
// sigma is the expected reduction in mu per iteration.


function inner(x,y) {
    let s = 0;
    for (let i=0; i<x.length; i++) {
	s += x[i]*y[i];
    }
    return s;
}

function norm(x) {
    let s = 0;
    for (let i=0; i<x.length; i++) {
	s += x[i]*x[i];
    }
    return Math.sqrt(s);
}

function nrneg(x) {
    let n = 0;
    for (let i=0; i<x.length; i++)
	if (x[i]<=0)
	    n+= 1;
    return n;
}

// Solve Ax=b by Cholesky factorization.
// A is replaced by its Cholesky factor and b by the solution x.
// The A matrix is given in lower triangular form
// A = [A11 A21 A22 A31 A32 A33 ...]
function cholsolve(A,b) {
    // Replace A by its Cholesky factorization
    let ii = 0;
    let i0 = 0;
    for (let i = 0;; i++) {
        let j0 = 0;
        for (let j = 0; j < i; j++) {
            let s = 0;
            for (let k = 0; k < j; k++) {
                s += A[i0+k]*A[j0+k];
	    }
            A[ii] = (A[ii]-s)/A[j+j0];
            ii += 1;
            j0 += j + 1;
	}
        let s = 0;
        for (let k = 0; k < i; k++) 
            s += A[i0+k]*A[j0+k];
        A[ii] = Math.sqrt(A[ii] - s);
        ii += 1;
        if (ii >= A.length)
            break;
        i0 += i + 1;
    }
    // Forward substitute
    let k = 0;
    for (let r = 0; r < b.length; r++) {
        let s = 0;
        for (let c = 0; c < r; c++) {
            s += A[k]*b[c];
            k += 1;
	}
        b[r] = (b[r] - s)/A[k];
	k += 1;
    }
    // Back-substitute
    k -= 1;
    for (let r = b.length-1; r >= 0; r--) {
        let s = 0;
        let l = k;
        for (let c = r+1; c < b.length; c++) {
            l += c;
            s += A[l]*b[c];
	}
        b[r] = (b[r] - s)/A[k];
        k -= r + 1;
    }
}

// Identify the active (nonzero) set of and re-solve the problem. x
// is replaced by the new solution. If this produces a feasible
// solution it is probably of higher accuracy than the original x.
function polish_solution(res, c, A, b) {
    const x = res.x;
    const w = res.w;
    const stride = x.length;
    const nact = b.length;
    if (res.status != 'OK')
	return res;
    
    let idx = new Int32Array(x.length);
    for (let i = 0; i<idx.length; i++)
	idx[i] = i;
    // Find the largest x values and assume they are the active ones
    idx.sort((i,j) => x[i] < x[j] ? 1 : x[i] > x[j] ? -1 : 0);
    
    // Re-solve primal problem with the active set indicated by the first elements of idx
    let T = new Float64Array((nact*(nact+1))/2);
    // use w as right hand side storage
    w.fill(0);
    let ii = 0;
    for (let i=0;i<nact;i++){
	for (let j=0;j<b.length;j++)
	    w[i] += A[idx[i] + stride*j]*b[j];	
        for (let j = 0; j <= i; j++) {
	    for (let kk=0;kk<nact;kk++)
                T[ii] +=  A[idx[i]+ kk*stride]*A[idx[j] + kk*stride];
            ii += 1;
	}
    }
    cholsolve(T, w);
    x.fill(0);
    for (let i=0;i<nact;i++)
	x[idx[i]] = w[i];
    // Re-solve the Dual for w and then s
    ii = 0;
    T.fill(0);
    w.fill(0);
    for (let i=0;i<nact;i++){
	for (let j=0;j<nact;j++)
	    w[i] += A[i*stride + idx[j]]*c[idx[j]];
        for (let j = 0; j <= i; j++) {
	    for (let kk=0;kk<nact;kk++)
                T[ii] +=  A[idx[kk] + stride*i]*A[idx[kk] + stride*j];
            ii += 1;
	}
    }
    cholsolve(T, w);
    // s = c - A'w
    const s = res.s;
    s.fill(0);
    for (let i=nact;i<s.length;i++) {	 
	s[idx[i]] = c[idx[i]];
	for (let j=0;j<w.length;j++)
	    s[idx[i]] -= A[stride*j + idx[i]]*w[j];
    }
    // Check if new solutions are feasible.
    for (let i=0;i<x.length;i++)
	if (x[i] < 0) {
	    res.status = "Polish x failed"
	}
    for (let i=0;i<s.length;i++)
	if (s[i] < 0) {
	    res.status = "Polish s failed"
	}
    return res;
}

// Solve the linear program
// min c'x s.t. Ax=b, x >= 0.
// where A must have full row rank.
//
// The dual program is
// max b'w s.t. A'w + s = c, s >= 0
//
// If the solver reaches maximum iterations the problem is
// most likely infeasible but the solver does not detect this.
// A is row-packed, i.e. A = [A11 A12 A13 .. A21 A22 .. ANM]
function solve_lp(c, A, b, sigma=0.1, max_iter=20, eps=1e-8, polish=true) {
    const x = new Float64Array(c.length);
    const dx = new Float64Array(c.length);
    const s = new Float64Array(c.length);
    const ds = new Float64Array(c.length);
    const w = new Float64Array(b.length);
    const dw = new Float64Array(b.length);
    const t = new Float64Array(b.length);
    const u = new Float64Array(c.length);
    const v = new Float64Array(c.length);
    const p = new Float64Array(c.length);
    const T = new Float64Array((b.length*(b.length+1))/2);
    const D2_diag = new Float64Array(c.length);
    const stride = c.length; // distance between rows in the linearized A array
    const eps1 = eps;
    const eps2 = eps;
    const eps3 = eps;
    const alpha = 0.99;
    const feasible_eps = eps;
    let n = x.length;
    // Initial guesses
    x.fill(1.0);
    s.fill(1.0);
    for (k = 0; k < max_iter; k++) {
        let mu = sigma*inner(x,s)/n;
        //t = b - np.dot(A,x);
	for (let i=0; i<t.length;i++) {
	    t[i] = b[i];
	    for (let j=0; j<x.length; j++)
		t[i] -= A[i*stride + j]*x[j];
	}
        // u = c - np.dot(A.T,w) - s;
	for (let i=0; i<u.length; i++) {
	    u[i] = c[i] - s[i];
	    for (let j=0; j<w.length; j++)
		u[i] -= A[j*stride + i]*w[j];
	}	
        // v = mu-x*s;
	for (let i=0; i<v.length; i++)
	    v[i] = mu - x[i]*s[i];
        // p = v/x;
	for (let i=0; i<p.length;i++)
	    p[i] = v[i]/x[i];

        if (mu < eps1 && norm(t)/(1.0 + norm(b)) < eps2) {
	    let nn = 0;
	    for (let i=0;i<u.length;i++)
		if (u[i] < 0)
		    nn += 1;
            if (nn>0) {
                if (norm(u)/(1 + norm(c)) < eps3) {
		    let res = {x:x, s:s, w:w, status:"OK"};
		    if (polish)
			return polish_solution(res,c,A,b);
		    else
			return res;
		}
	    } else {
		let res = {x:x, s:s, w:w, status:"OK"};
		if (polish)
		    return polish_solution(res,c,A,b);
		else
		    return res;
	    }
	}
	for (let i=0; i<D2_diag.length; i++)
            D2_diag[i] = x[i]/s[i];
	// rhs = np.dot(A,D2_diag*(u-p)) + t
	for (let i=0;i<dw.length;i++) { // dw stores the right hand side of the equation here
            dw[i] = t[i];
	    for (let j=0;j<p.length;j++)
		dw[i] += A[i*stride + j]*D2_diag[j]*(u[j] - p[j]);
	}
	// Form A*D2*A.T
	let ii = 0;
	for (let i=0;i<b.length;i++) { 
            for (let j = 0; j <= i; j++) {
		T[ii] = 0;
		for (let kk=0;kk<x.length;kk++)
                    T[ii] +=  A[i*stride + kk]*D2_diag[kk]*A[j*stride + kk]; //np.inner(A[i,:],D2_diag*A[j,:]) 
                ii += 1;
	    }
	}
	cholsolve(T, dw);
	for (let i=0;i<ds.length;i++) {
	    ds[i] = u[i];
	    for (let j=0;j<dw.length;j++)
		ds[i] -= A[j*stride + i]*dw[j]; //np.dot(A.T,dw)
	}
	for (let i=0;i<dx.length;i++)
            dx[i] = D2_diag[i]*(p[i]-ds[i]);
        if (norm(t) < feasible_eps && nrneg(dx) == 0 && inner(c,dx)<0)
            return {x:x, s:s, w:w, status:"Primal Unbounded"};
        if (norm(u) < feasible_eps && nrneg(ds) == 0 && inner(b,dw)>0)
            return {x:x, s:s, w:w, status:"Dual Unbounded"};
	let mP = 1;
	for (let i=0;i<dx.length;i++)
	    mP = Math.max(mP, -dx[i]/(alpha*x[i]))
	// betaP = 1./max(1., np.max(-dx/(alpha*x)))
	let betaP = 1.0/mP;
	let mD = 1;
	for (let i=0;i<ds.length;i++)
	    mD = Math.max(mD, -ds[i]/(alpha*s[i]))
	// betaD = 1./max(1., np.max(-ds/(alpha*s)))
        let betaD = 1./mD;
	for (let i=0; i<dx.length; i++)
            x[i] += betaP*dx[i]
	for (let i=0; i<dw.length; i++)
	    w[i] += betaD*dw[i]
	for (let i=0; i<ds.length; i++)
            s[i] += betaD*ds[i]
    }
    return {x:x, s:s, w:w, status:"Too many iterations"};
}



// Return the duality gap (difference between primal and dual solutions).
// Should be small but positive.
function gap(res, c, b) {
    return inner(c,res.x) - inner(b,res.w)
}

function test_7_3() {
    let c = new Float64Array([0.04075823, 0.11821758, 0.45436518, 2.18293562, 1.71302413,
			      1.02178906, 1.10531284]);
    let A = new Float64Array([-1.19981773e+01, -4.22388691e-01,  1.07448018e+00, -3.80682659e-04,
			      1.85058368e+00,  3.34701783e-03,  4.25726284e-02,  3.44525475e+00,
			      1.89881128e-02, -6.35029764e+00, -6.85323162e+00, -6.22657891e-01,
			      9.25567499e+00, -5.79039199e-02, -1.00740761e-01, -1.86229980e-02,
			      -5.16821656e-01,  1.05094994e+00,  6.43709621e-02,  3.17506881e-02,
			      8.20537163e-03]);
    let b = new Float64Array([0.37161125, 0.72462025, 1.10243664]);
    let xref = [0.        , 0.        , 0.        , 1.01137669, 0.19949536,
		0.84056931, 0.        ];
    let res = solve_lp(c,A,b);
    console.log(res);
    console.log(xref);
    console.log({gap:gap(res, c, b)});
}

