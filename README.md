pysfm
=====

Structure from Motion Algorithms in Python.

Eventually, this is intended to be a collection of factorization based structure from motion algorithms.
Currently, it only contains a standard rigid factorization algorithm and a state of the art non-rigid shape basis factorization method [(Dai et al. 2012)][Dai2012] that won the best paper at CVPR2012.

**NOTE: This is extremely beta, and no claim is made about the correctness of these implementations.  There are likely bugs, and thus contributions and/or friendly comments are welcome.  See below for contact information.**

## Requirements

* setuptools (you likely have this)
* numpy
* scipy
* [CVXOPT](http://abel.ee.ucla.edu/cvxopt/) (for the shape-basis method)
* matplotlib (for viewing results)
* nose (if you want to run the test suite).

## Instructions

To run Dai et al. 2012 on an observation matrix W

    import sfm

    # Run Dai2012 with 3 basis shapes.
    inferred_model = sfm.factor(W, n_basis = 3)
    
    # Get the Fx3xN tensor of points in the
    # cameras reference frame.
    Ps = inferred_model.Ps

    # To view the recovery (using matplotlib)
    inferred_model.visualize()

## Contact

To contact the author email jtaylor**FOO**cs.toronto.edu where **FOO** is replaced with the at symbol. 


[Dai2012]: http://users.cecs.anu.edu.au/~hongdong/CVPR12_Nonrigid_CRC_12_preprint.pdf "Dai, Y., Li, H. and He, M. A Simple Prior-free Method for Non-Rigid Structure-from-Motion Factorization. CVPR, 2012"
