"""
This shows how to run the Dai et al. 2012 algorithm on a 
matlab .mat file with a single variable P_gt
a 3FxN matrix where the first F rows are X coordinates
the next F are Y and the last F are Z coordinates.

This has currently only been tested with walking.mat
in the zip file at:
http://www2.ece.ohio-state.edu/~gotardop/files/Gotardo_Martinez_CSF_PAMI.zip

Run it via:
python scripts/run_prior_free_on_mat.py path/to/walking.mat
"""

from sfm import model
from sfm import shape
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('matfile')
parser.add_argument('-k', '--n-basis', default=2, type=int)
parser.add_argument('--use-method-1', default=False, action='store_true')
parser.add_argument('--visualize', default=False, action='store_true')
args = parser.parse_args()

# Load the walking dataset, you need to make sure
import scipy.io
S = scipy.io.loadmat(args.matfile)['P3_gt']

# The rows are in an incompatible order so make a copy and fix.
Sin = S.copy()
F = Sin.shape[0]/3
S[0::3] = Sin[:F]
S[1::3] = Sin[F:F*2]
S[2::3] = Sin[F*2:F*3]

# Makes visualization better and easier to add rotations if desired.
S -= S.mean(axis=-1)[:,np.newaxis]

# Create a scene and get observation matrix W.
gt = model.Scene(S = S)
print gt
W = gt.W

# Factor into a shape model.
inf = shape.factor(W, n_basis = args.n_basis, use_method_1 = args.use_method_1)

# Register to the ground truth scene.
inf.register(gt)

# And compare.
model.compare(inf, gt, visualize = args.visualize)