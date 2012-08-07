"""Utility functions."""

import numpy as np

def vec(X, order='F'):
    """Returns the vectorization of X.  Columns of X are stacked.  (The opposite of X.flatten()).""" 
    assert X.ndim == 2, 'vec operator requires a matrix.'
    
    return X.flatten(order=order)

def from_vec(x, m, n, order='F'):
    return x.reshape(m, n, order=order)

def vech(X, order='F'):
    """Returns vectorization of lower triangle of X in column major order by default."""
    assert X.ndim == 2, 'vech operator requires a matrix.'
    
    m, n = X.shape
    if order == 'F':
        idx = np.where(1 - np.tri(n,m, -1, dtype=int))
        return X.T[idx]
    elif order == 'C':
        i,j = np.where(np.tri(m,n, dtype=int))
    else:
        raise Exception("Only order C and F are allowed")
    
    return X[i,j]

def from_vech(x, m, n, order='F', sym = False):
    
    X = np.zeros((m,n), dtype = x.dtype)
    if order == 'F':
        idx = np.where(1 - np.tri(n,m,-1, dtype=int))
        X.T[idx] = x
    elif order == 'C':
        idx = np.where(np.tri(m,n, dtype=int))
        X[idx] = x
    else:
        raise Exception("Only C and F ordering allowed.")
    
    if sym:
        if m != n:
            raise Exception("Can only do this for square matrices.")
        X = X + X.T - np.diag(np.diag(X))
    return X

def vc(a, b):
    """vc operator from Brand 2005
    """
    assert a.ndim == 1 and b.ndim == 1 and a.size == b.size
    a = a[:,np.newaxis]
    b = b[:,np.newaxis]
    return vech(np.dot(a, b.T) + np.dot(b, a.T) - np.diag((a*b).flat))

def hat_operator(omega):
    F = omega.shape[0]
    
    # Create the hat matrix.
    OH = np.zeros((F,3,3))
    
    o1, o2, o3 = omega.T
    
    OH[:,0,1] = -o3
    OH[:,0,2] = o2
    OH[:,1,0] = o3
    OH[:,1,2] = -o1
    OH[:,2,0] = -o2
    OH[:,2,1] = o1

    return OH

def axis_angle_to_Rs(omega, theta):
    
    F = theta.shape[0]
    
    # Calculate omega hat.
    omega_hat = hat_operator(omega) 
    
    # Use Rodriguez' formula
    Rs = np.zeros((F,3,3))
    
    # Term 1.
    Rs += np.eye(3)[np.newaxis,...]
    
    # Term 2.
    Rs += np.sin(theta)[:,np.newaxis,np.newaxis] * omega_hat
    
    # Term 3.
    tmp = omega[:,:,np.newaxis] * omega[:,np.newaxis,:] - np.eye(3)[np.newaxis,...]
    Rs += (1 - np.cos(theta))[:,np.newaxis,np.newaxis] * tmp
    
    return Rs
    
def rotvecs_to_Rs(omega):
    """
    omega - Fx3 rotation vectors.
    
    Returns a Fx3x3 Rs tensor.
    """
    
    # Make sure, that we don't modify the original vector.
    omega = omega.copy()
    
    # Calculate the norm of the vectors.
    theta = norm(omega, axis=1)
    
    # When theta is zero, the rotation will be the identity
    # and 2pi is also the identity.
    theta[theta==0] = np.pi*2
    
    # This allows us to normalize without getting NaNs
    omega /= theta[:,np.newaxis]
    
    # Now use Rodrigues' formula to calculate
    Rs = axis_angle_to_Rs(omega, theta)

    return Rs

def norm(x, axis= -1):
    return np.sqrt((x * x).sum(axis=axis))

def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))

def normed(x, axis = -1):
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("axis(=%d) out of bounds" % axis)
    if axis < 0:
        axis += x.ndim
    shape = list(x.shape)
    shape[axis] = 1
    shape = tuple(shape)
    return x / norm(x,axis=axis).reshape(shape)