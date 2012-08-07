import numpy as np
import nose
import numpy.testing as npt
from nose.exc import SkipTest
from scipy.linalg import expm
import model
import shape
import util
import rigid

def test_rot_code():
    
    # It would be very unlikely for any of
    F = 10
    np.random.seed(0) 
    
    # Create a bunch of random rotation vectors.
    omega = np.random.randn(F,3) * 5
    
    # Make sure one has zero norm.
    omega[3,:] = 0.0
    
    # Use the sfm implemenation which relies on Rodriguez' formula.
    Rs_hat = util.rotvecs_to_Rs(omega)
    
    # Calculate it the slow way using the matrix exponential.
    omega_hats = util.hat_operator(omega)
    Rs_gt = np.asarray([expm(omega_hat) for omega_hat in omega_hats])
    
    npt.assert_allclose(Rs_hat, Rs_gt)
    
def test_recover_Gk_no_noise():
    
    # Generate some synthetic data.    
    n_frames = 20
    n_basis = 3
    n_points = 200
    gt_model = model.generate_synthetic_model(n_frames, n_basis, n_points)

    # Factor the matrix.    
    from rigid import factor_matrix
    W = gt_model.W.copy()
    W -= W.mean(axis=-1)[:,np.newaxis]
    M_hat, B_hat = factor_matrix(W, J=n_basis*3)

    # Try to find a G_k
    G_k = shape.find_G_k(M_hat)
    
    # Get the k'th column triple of M_k
    M_k = np.dot(M_hat, G_k)
    M_kx = M_k[0::2]
    M_ky = M_k[1::2]
    
    # Ensure they are orthogonal.
    npt.assert_allclose((M_kx*M_ky).sum(), 0, atol=1e-3)
    
    # And that the length of x and y are equal.
    npt.assert_allclose((M_kx*M_kx).sum() - (M_ky*M_ky).sum(), 0, atol=1e-3)


def test_shape_no_noise():
    
    # Generate some synthetic data.    
    n_frames = 40
    n_basis = 2
    n_points = 20
    gt_model = model.generate_synthetic_model(n_frames, n_basis, n_points)
    
    W = gt_model.W
    
    # Recover the model
    inf_model = shape.factor(W, n_basis=n_basis)
    
    # Register with ground truth.
    inf_model.register(gt_model)
    
    delta = util.norm(inf_model.Ps - gt_model.Ps, axis=1)
    
    # Ensure the average error is low.
    npt.assert_allclose(delta.mean(), 0, atol=0.1)
    
def test_rigid_no_noise():
    
    # Generate some synthetic data.    
    n_frames = 40
    n_basis = 1
    n_points = 20
    
    # Take a randomly generated model.
    gt_model = model.generate_synthetic_model(n_frames, n_basis, n_points)
    
    # Force scale factor to 1.
    gt_model.C = np.ones((n_frames, 1))
    
    W = gt_model.W
    
    # Recover the model
    inf_model = rigid.factor(W)
    
    # Register with ground truth.
    inf_model.register(gt_model)
    
    delta = util.norm(inf_model.Ps - gt_model.Ps, axis=1)
    
    # Ensure the average error is low.
    npt.assert_allclose(delta.mean(), 0, atol=1e-10)
