"""Shape basis models.
"""
import numpy as np
import model
from util import norm, normed, vech, from_vech,  vc
from model import Scene
import scipy.linalg
import rigid

def to_S_sharp(S):
    F,N = S.shape
    F /= 3
    S_sharp = np.zeros((F,3*N))
    S_sharp[:,:N] = S[::3]
    S_sharp[:,N:2*N] = S[1::3]
    S_sharp[:,2*N:3*N] = S[2::3]
    return S_sharp

def to_S(S_sharp):
    F,N = S_sharp.shape
    N /= 3
    S = np.zeros((3*F,N))
    S[::3] = S_sharp[:,:N]
    S[1::3] = S_sharp[:,N:2*N]
    S[2::3] = S_sharp[:,2*N:3*N]
    return S

def shrink_vec(x, v):
    return np.maximum(x - v, 0.0)
    
def shrink(X, v):
    U, s, Vt = np.linalg.svd(X, 0)
    s = shrink_vec(s,v)
    return np.dot(U, np.dot(np.diag(s), Vt))

#@profile
def S_from_Rs(W, Rs, use_method_1 = False, rank = None):
    
    # Create block diagonal matrix R
    R = scipy.linalg.block_diag(*Rs[:,:2,:].tolist())
    
    # Initialize from method 1.
    S = np.dot(np.linalg.pinv(R), W)
    
    if use_method_1:
        return S
    
    # Use the paramaterization in Dai CVPR 2012 and
    # algorithm from 
    # S. Ma, D. Goldfarb, and L. Chen. "Fixed point and bregman iterative 
    # methods for matrix rank minimization.", Mathematical Programming, 
    # Series A. 2011
    
    # These values seem to work and be reasonably fast, but were definitely not
    # set in any principled way.
    tau = 1
    S_sharp = to_S_sharp(S)
    mu_bar = 0.0001
    eta = .5
    mu = 100000
    xtol = 0.0001/tau
    
    while mu > mu_bar:
        while True:
            old_P_sharp = S_sharp
            S = to_S(S_sharp)
            g_S_sharp = to_S_sharp(np.dot(R.T, np.dot(R, S) - W))
            Y = S_sharp - tau * g_S_sharp
            S_sharp = shrink(Y, tau * mu)

            # Check for convergence.            
            err = np.linalg.norm(old_P_sharp - S_sharp, 'fro')/ max(1.0, np.linalg.norm(S_sharp, 'fro'))
            if err < xtol:
                break
        # Update mu.
        mu = max(eta * mu, mu_bar)
    
    # Possibly project to a low rank matrix.
    if rank != None:
        U, s, Vh = np.linalg.svd(S_sharp)
        U = U[:,:rank]
        s = np.diag(s[:rank])
        Vh = Vh[:rank,:]
        
        S_sharp = np.dot(U, np.dot(s, Vh))
        
    # Convert back to S
    S = to_S(S_sharp)
    
    return S


def find_linear_basis_for_vech_Q_k(PI_hat):
    K = PI_hat.shape[1]/3
    F = PI_hat.shape[0]/2
    
    # Find the 2F constraints on vech(Qk)
    A = np.zeros((2*F, (9*K*K + 3*K)/2))
    
    for f in range(F):
        
        # Extract the two rows.
        x_f, y_f = PI_hat[f*2:f*2+2]
        
        # Constraints on vech(Qk) best detailed in equation 2 (rigid case)
        # and the discussion about how this is the same for the nonrigid case
        A[f*2] = vc(x_f, y_f)
        A[f*2+1] = vc(x_f - y_f, x_f + y_f)

    # Find SVD of the equations.
    U, s, Vh = np.linalg.svd(A)
    
    # And extract approximate 2*K*K+K dimensional nullspace (See Dai 2021).
    B = Vh[-(2*K*K-K):].T
    
    return B


def find_G_k(M_hat):
    
    K = M_hat.shape[1]/3
    F = M_hat.shape[0]/2

    # Get the linear basis that vech(Q_k) should reside in.
    U = find_linear_basis_for_vech_Q_k(M_hat)

    # 2D subspace of vech style.
    B = np.asarray([from_vech(U[:,i], 3*K, 3*K, sym=True).flatten() for i in range(U.shape[1])]).T
    
    # Coefficients correspond to taking trace.
    from cvxopt import matrix, solvers
    
    # Calculate the cost vector.
    non_diag = np.where(np.eye(3*K).flatten()!=1)[0]
    c = B.copy()
    c[non_diag,:] = 0
    c = c.sum(axis=0)

    # This matrix should be positive semi-definite.    
    G = [matrix(-B)]
    h = [matrix(np.zeros((3*K,3*K)))]

    # We fix the first row of the recovered M_hat to have unit length
    # which avoids the trivial solution.
    A = np.dot(vc(M_hat[0], M_hat[0]), U)[np.newaxis,:].copy()
    A = matrix(A)
    b = matrix([[1.0]])
    
    # Disable progress reporting in solver.
    solvers.options['show_progress'] = False
    
    # Solve the sdp using cvxopt
    sol = solvers.sdp(matrix(c), Gs = G, hs = h, A = A, b = b)
    
    # Extract the solution.
    x = np.asarray(sol['x'])
    
    # Upgrade to vec(Q_k)
    y = np.dot(B, x)

    # And reshape to Q_k
    Q_k = y.reshape((3*K, 3*K))
    
    # Factor into G_k
    Q_k_U, Q_k_s, Q_k_V = np.linalg.svd(Q_k)
    G_k = Q_k_U[:,:3] * np.sqrt(Q_k_s)[np.newaxis,:3]
    
    # Squared residual objective function as defined in Dai2012.
    def f(g_k):
        G_k = g_k.reshape(3*K, 3)
        M_k = np.dot(M_hat, G_k)
        M_kx = M_k[0::2]
        M_ky = M_k[1::2]
        term1 = np.square(1 - (M_kx * M_kx).sum(axis=-1)/(M_ky*M_ky).sum(axis=-1 )) 
        term2 = np.square(2*(M_ky*M_kx).sum(axis=-1)/(M_ky*M_ky).sum(axis=-1))
        err = np.sum(term1 + term2)
        return err

    # Minimize this function with LBFGS
    from scipy.optimize import fmin_l_bfgs_b
    rc = fmin_l_bfgs_b(f, G_k.flatten(), approx_grad=True)        
    G_k = rc[0].reshape(3*K,3)
    
    return G_k

def Rs_from_M_k(M_k):
    """Estimates rotations from a column triple of M."""
    
    n_frames = M_k.shape[0]/2
    
    # Estimate scales of each rotation matrix by averaging norms of two rows.
    scales = .5 * (norm(M_k[0::2], axis=1) + norm(M_k[1::2], axis=1))
    
    # Rescale to be close to rotation matrices.
    R = M_k.reshape(n_frames, 2, 3) / scales[:,np.newaxis,np.newaxis]    
    Rs = []
    
    # Upgrade to real rotations.
    for f in range(n_frames):
        Rx = R[f,0]
        Ry = R[f,1]
        Rz = normed(np.cross(Rx,Ry))
        U,s,Vh = np.linalg.svd(np.asarray([Rx,Ry,Rz]))
        Rs.append(np.dot(U, Vh))
        
    return np.asarray(Rs)

def factor(W, n_basis=2, use_method_1 = False):
    """Factor's W using
    
    Dai, Y., Li, H. and He, M. "A Simple Prior-free Method for Non-Rigid
    Structure-from-Motion Factorization". CVPR, 2012
    
    Note that we use the letter M in place of PI from this paper.
    """

    n_frames = W.shape[0]/2

    # Estimate 2D translation.
    T = W.mean(axis=-1)
    Ts = np.zeros((n_frames,3))
    Ts[:,:2] = T.reshape(n_frames,2)

    # Center observation matrix
    W_cent = W.copy() - T[:,np.newaxis]

    # Factor
    M_hat, B_hat = rigid.factor_matrix(W_cent, n_basis*3)
    
    # Recover G_k
    G_k = find_G_k(M_hat)
    
    # Recover M_k
    M_k = np.dot(M_hat, G_k)
    
    # Recover R
    Rs = Rs_from_M_k(M_k)
    
    # Recover S
    S = S_from_Rs(W_cent, Rs, use_method_1 = use_method_1, rank = n_basis)
    
    # Check residual    
    scene = Scene(S = S, Rs = Rs, Ts = Ts)

    return scene

if __name__ == '__main__':
    
    # Set the seed.
    np.random.seed(0)
    
    # Generate some synthetic data.
    n_frames = 200
    gt_model = model.simple_model(n_frames)
    W = gt_model.W

    # Use the Dai algorithm.
    inf_model = factor(W, use_method_1 = 0)
    
    # Register to ground truth
    inf_model.register(gt_model)
    
    model.compare(inf_model, gt_model, visualize=False)
