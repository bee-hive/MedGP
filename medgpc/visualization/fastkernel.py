import numpy as np

def compute_A_matrix(Q, D, R, hyp):
    A_matrix_list = []
    for q in range(Q):
        A = np.zeros((D, R))
        for d in range(D):
            for r in range(R):
                A[d, r] = hyp[D + q*(D*R) + d*R + r]
        A_matrix_list.append(A)
    return A_matrix_list

def compute_B_matrix(Q, D, R, hyp):
    A_matrix_list = compute_A_matrix(Q, D, R, hyp)
    lambda_vector_list = compute_kappa_vector(Q, D, R, hyp)
    B_matrix_list = []
    for q in range(Q):
        A = A_matrix_list[q]
        lam = lambda_vector_list[q]
        B = np.dot(A, A.T) + np.diag(lam.flatten())
        B_matrix_list.append(B)
    return B_matrix_list

def compute_kappa_vector(Q, D, R, hyp):
    lambda_vector_list = []
    for q in range(Q):
        lam = np.zeros((D, ))
        for d in range(D):
            lam[d] = np.exp(hyp[D + Q*(D*R+2) + q*D + d])
        lambda_vector_list.append(lam)
    return lambda_vector_list

def compute_sm_1d(mu, v, x):
    rsq = compute_squared_dist(x, np.atleast_2d([0]).T)
    r = np.sqrt(rsq)
    response = compute_k(rsq*v, r*mu)
    return response

def compute_squared_dist(x, x2):
    xsq = np.sum(np.square(x),1)
    x2sq = np.sum(np.square(x2),1)
    dist_matrix = -2.*np.dot(x, x2.T) + xsq[:,None] + x2sq[None,:]
    dist_matrix = np.clip(dist_matrix, 0, np.inf)
    return dist_matrix

def compute_k(d2v, dm):
    k = np.exp(-2*(np.pi**2)*d2v)*np.cos(2*np.pi*dm);
    return k

def compute_se_1d(s2, lc, x):
    rsq = compute_squared_dist(x, np.atleast_2d([0]).T)
    rsq = rsq / (lc ** 2.)
    response = (s2 ** 2.) * np.exp(-0.5*rsq)
    return response

