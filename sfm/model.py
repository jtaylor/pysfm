import numpy as np
from util import norm, normed, rotvecs_to_Rs

def point_set_error(Ps, Qs):
    return norm(Ps - Qs, axis=1).mean(axis=-1)

class Scene(object):
    
    def __init__(self, Ss = None, S = None, Rs = None, Ts = None):
        
        # Try to use S if Ss is not defined.
        if Ss == None:
            assert S.ndim == 2
            assert S.shape[0] % 3 == 0
            Ss = S.reshape(S.shape[0]/3, 3, S.shape[1])
        
        assert Ss.ndim == 3 and Ss.shape[1] == 3
                
        self.n_frames = Ss.shape[0]
        self.n_points = Ss.shape[2]

        if Rs == None:
            Rs = np.asarray([np.eye(3) for i in range(self.n_frames)])
        
        if Ts == None:
            Ts = np.zeros((self.n_frames, 3))   
        
        assert Rs.shape[0] == self.n_frames
        assert Rs.shape[1] == Rs.shape[2] == 3

        self.Rs = Rs        
        self.Ss = Ss
        self.Ts = Ts

    @property
    def S(self):
        return self.Ss.reshape(self.n_frames*3, self.n_points)
    
    @property
    def Ps(self):
        return np.einsum('fij,fjk->fik', self.Rs, self.Ss) + self.Ts[:,:,np.newaxis]
    
    @property
    def P(self):
        return self.Ps.reshape(self.n_frames*3, self.n_points)
    
    @property
    def Ws(self):
        return self.Ps[:,:2]
    
    @property
    def W(self):
        return self.Ps[:,:2].reshape(self.n_frames*2, self.n_points)
    
    @property
    def T(self):
        return self.Ts.reshape(self.n_frames*3)
    
    def register(self, gt):

        # Ground truth points.        
        Ps_gt = gt.Ps.copy()
        zbar_gt = Ps_gt[:,2].mean(axis=-1)
        
        # Find optimal translation
        P1s = self.Ps.copy()
        delta1 = zbar_gt - P1s[:,2].mean(axis=-1)
        P1s[:,2] += delta1[:,np.newaxis]
        err1 = point_set_error(P1s, Ps_gt)
        
        # Reflected optimal translation.
        P2s = self.Ps.copy()
        P2s[:,2] *= -1
        delta2 = zbar_gt - P2s[:,2].mean(axis=-1)
        P2s[:,2] += delta2[:,np.newaxis]
        err2 = point_set_error(P2s, Ps_gt)
        
        # Figure out which errors are better.
        which = np.argmin(np.asarray([err1, err2]), axis=0)
        
        # Update z components for when no depth flip is needed.
        idx = which == 0
        self.Ts[idx,2] += delta1[idx]
        
        # Update z component for when a depth flip is needed.
        # FIXME: This should be thought about, right now these
        # rotations are becoming reflections, which is probably not
        # optimal.
        idx = which == 1
        self.Ts[idx, 2] *= -1
        self.Ts[idx, 2] += delta2[idx]
        self.Rs[idx,2] *= -1
        
    def visualize(self):
        visualize_models([self])
    
    def __str__(self):
        return 'Scene with {n_points} points across {n_frames} frames'.format(**self.__dict__)

class BasisShapeModel(Scene):
    
    def __init__(self, Rs, Bs, C = None, Ts = None):
        self.n_frames = Rs.shape[0]
        self.n_basis = Bs.shape[0]
        self.n_points = Bs.shape[2]
        
        if Ts == None:
            Ts = np.zeros((self.n_frames, 3))   
        
        if C == None:
            C = np.ones((self.n_frames, self.n_basis))

        assert Ts.ndim == 2 and Ts.shape[0] == self.n_frames, 'Ts matrix the wrong size %s'.format((str(Ts.shape)))
        assert C.ndim == 2 and C.shape == (self.n_frames, self.n_basis), 'C matrix the wrong size'

        self.Rs = Rs
        self.Bs = Bs
        self.Ts = Ts
        self.C = C
        
    @property
    def Ss(self):
        return np.einsum('fk,kij->fij', self.C, self.Bs)
    
    def __str__(self):
        return 'LinearModel with {n_basis} basis shapes of {n_points} points across {n_frames} frames'.format(**self.__dict__)

def generate_rotation_from_optical_axis(D, safety_checks = False):
    """
    D -- a Fx3 vector of directions.
    
    returns R so that
    Rd = FxN vectors of [0 0 1]
    """
    
    assert D.ndim == 2 and D.shape[1] == 3
    
    F = D.shape[0]
    
    # Space for the rotations.
    R = np.zeros((F,3,3))
    
    # Put the D vectors along the Z axis.
    R[:,2,:] = normed(D)
    
    # Make x direction orthogonal in the y=0 plane.
    R[:,0,0] = R[:,2,2]
    R[:,0,2] = -R[:,2,0]
    R[:,0,:] = normed(R[:,0,:], axis=-1)

    # Make the y direction the cross product.
    # Negation makes this a rotation instead of a reflection.
    R[:,1,:] = -np.cross(R[:,0,:], R[:,2,:], axis=-1)

    if safety_checks:
        for f in range(F):
            if np.linalg.matrix_rank(R[f]) != 3:
                raise Exception("Low rank.")
            if np.linalg.det(R[f]) < 0:
                # Flip one of the rows signs to change this reflection to a rotation matrix.
                R[f,0,:]*= -1

    return R   

    
def generate_smooth_rotations(n_frames):

    # Create directions that circle in a halo around the origin.
    x = np.cos(np.linspace(0, np.pi*2, n_frames))
    y = np.ones((n_frames,))*.5
    z = np.sin(np.linspace(0, np.pi*2, n_frames))

    # Generate rotations from these directions.    
    R = generate_rotation_from_optical_axis(np.asarray([x,y,z]).T)
    
    return R
    
def generate_cube(interval = 0.2):
    
    start = -1 + interval
    stop = 1 - interval
    
    B = []
    for i_dim1 in range(3):
        for i_dim2 in range(i_dim1+1, 3):
            for i_face in range(2):
                for x in np.arange(start, stop, interval):
                    for y in np.arange(start, stop, interval):
                        p = np.ones(3)
                        if i_face:
                            p *= -1
                        p[i_dim1] = x
                        p[i_dim2] = y
                        B += [p]

    B = np.asarray(B).T
    return B

def generate_synthetic_model(n_frames, n_basis, n_points):
    
    Bs = np.random.randn(n_basis, 3, n_points)*10
    R = rotvecs_to_Rs(np.random.randn(n_frames,3))
    C = np.random.randn(n_frames, n_basis)

    return BasisShapeModel(R, Bs, C=C)

def generate_synthetic_rigid_model(n_frames, n_points):
    
    m = generate_synthetic_model(n_frames, 1, n_points)
    m.C[:] = 1
    return m

def simple_model(n_frames):
    Bs = np.asarray([generate_cube(), generate_cube()]) 
    Bs[0] = np.random.randn(*Bs[0].shape)*.2
    R = generate_smooth_rotations(n_frames)
    C = np.asarray([np.linspace(0,.2,n_frames), np.linspace(1,.8,n_frames)]).T
    model = BasisShapeModel(R, Bs, C = C)
    return model

def rigid_model(n_frames):
    Bs = np.asarray([generate_cube()]) 
    R = generate_smooth_rotations(n_frames)
    model = BasisShapeModel(R, Bs)
    return model

def visualize_models(models):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    plt.ion()

    n_frames = models[0].n_frames

    lines = []
    Pss = [model.Ps for model in models]
      
    for f in range(n_frames):
        if f == 0:
            for Ps in Pss:
                lines.append(ax.plot(Ps[f,0], Ps[f,1], Ps[f,2], 'x')[0])
        else:
            for (line, Ps) in zip(lines, Pss):
                line.set_data([Ps[f,0], Ps[f,1]])
                line.set_3d_properties(Ps[f,2])
            
        plt.draw()
        plt.pause(.3)


def compare(inf, gt, visualize = True):
    print 'Comparing two models:'
    print 'Model 1:', str(inf)
    print 'Model 2:', str(gt)
    
    assert inf.n_frames == gt.n_frames
    assert inf.n_points == inf.n_points
    
    P_gt = gt.Ps
    P_inf = inf.Ps
    
    # Get scale of data as defined in Dai2012.
    sigma = np.std(P_gt, axis=-1).mean()
    
    # This is the average 3D error rescaled by the scale of the data.
    # I think this is close to what is being used in Dai2012.
    print 'Scaled 3D error', norm(P_gt - P_inf, axis=1).mean()/sigma
    
    if visualize:
        visualize_models([gt, inf])

    









    
    