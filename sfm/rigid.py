"""
This module implements rigid factorization.
"""

import numpy as np
from numpy.linalg import svd, solve
import numpy.linalg
import model
import util
from model import BasisShapeModel

def factor_matrix(W, J=3):
    """Obtain the best rank J factorization MS to W

    Computes a factorization of W into M and B such that ||W-MB||_F
    is minimized.

    Input:
    W is a MxN matrix.

    Returns:
    M -- A MxR matrix
    B -- A RxN matrix
    """

    # Decompose using SVD.
    U, s, Vt = svd(W, full_matrices=0)

    # Compute the factorization.
    sqrt_sigma = np.diag(np.sqrt(s[:J]))
    M = np.dot(U[:,:J], sqrt_sigma)
    B = np.dot(sqrt_sigma, Vt[:J])

    return M, B

def factor(W):
    """
    This implements rigid factorization as described in
    
    Tomasi, C. & Kanade, T. "Shape and motion from image streams under
    orthography: a factorization method International Journal of Computer
    Vision, 1992
    """
    
    F = W.shape[0]/2
    N = W.shape[1]
    
    # Center W
    T = W.mean(axis=1)
    W = W - T[:, np.newaxis]

    # Factor W
    M_hat, B_hat = factor_matrix(W, J=3)

    # Where we will build the linear system.
    A = np.zeros((3*F, 6))
    b = np.zeros((3*F,))

    for f in range(F):
        
        # Extract the two rows.
        x_f, y_f = M_hat[f*2:f*2+2]
        
        # Both rows must have unit length.
        A[f*3] = util.vc(x_f, x_f)
        b[f*3] = 1.0
        A[f*3+1] = util.vc(y_f,y_f)
        b[f*3+1] = 1.0
        
        # And be orthogonal.
        A[f*3+2] = util.vc(x_f - y_f, x_f + y_f)

    # Recovec vech(Q) and Q
    vech_Q = np.linalg.lstsq(A, b)[0]
    Q = util.from_vech(vech_Q, 3, 3, sym=True)

    # Factor out G recovery matrix
    G, Gt = factor_matrix(Q)
    
    # Upgrade M and B.
    M = np.dot(M_hat, G)
    B = np.linalg.solve(G, B_hat)
    
    # Find actual rotations matrices.
    Rs = np.zeros((F, 3, 3))
    Rs[:,:2] = M.reshape(F,2,3)
    Rs[:,2] = util.normed(np.cross(Rs[:,0], Rs[:,1], axis=-1))

    # And 3D translations.
    Ts = np.zeros((F,3))
    Ts[:,:2] = T.reshape(F,2)
    
    model = BasisShapeModel(Rs, Bs = B[np.newaxis,:,:], Ts = Ts)

    return model

def main():
    
    gt_model = model.rigid_model(100)
    
    W = gt_model.W

    inf_model = factor(W)
    inf_model.register(gt_model)

    model.compare(inf_model, gt_model)

if __name__ == '__main__':
    main()
    