import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence , lobpcg
from scipy.sparse import isspmatrix
import scipy as sp
from time import time

def compute_rtb_modes(K, P, n_modes=20, tol=1e-6):
    """
    Compute lowest-frequency RTB normal modes.
    
    Args:
        K: sparse all-atom Hessian [3N x 3N] (scipy.sparse)
        P: sparse projection matrix [3N x 6n] (scipy.sparse)
        n_modes: number of non-rigid-body modes to compute
        tol: eigenvalue convergence tolerance

    Returns:
        L_full: [3N x n_modes] projected full-atom modes (numpy array)
        eigvals: [n_modes] eigenvalues
    """
    if not isspmatrix(K) or not isspmatrix(P):
        raise ValueError("K and P must be scipy sparse matrices")

    # RTB-reduced Hessian


    
    assert (P.T @ P).diagonal().min()>10**-6,  (" P has columns with zero norm — i.e., some of the rigid-body basis vectors are either degenerate or completely missing.")
    
    
    K_rtb = P.T @ K @ P  # [6n x 6n], still sparse
    print(f"[INFO] Projecting full Hessian to RTB space... (size: {K_rtb.shape})")
    # Compute eigenvalues/eigenvectors of RTB Hessian
    # We skip first 6 zero modes (rigid-body)
    print(f"[INFO] Diagonalizing projected Hessian to find {n_modes} low-frequency modes...")
    start = time()
    try:
        #eigvals, eigvecs = eigsh(K_rtb, k=n_modes, which='SM', tol=tol)
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        eigvals, eigvecs = eigsh(K_rtb, k=n_modes+6, sigma=0, which='LM', tol=tol) # if K_rtb is positive semi-defined
        # Initial guess: random block of shape (N, n_modes+6)
        #print(eigvals)
        #print(eigvecs[:,0])
        #X = np.random.rand(K_rtb.shape[0], n_modes + 6)
        # Use diagonal as preconditioner
        #diag = K_rtb.diagonal()
        #diag[K_rtb.diagonal()==0]=1e-8
        #M = sp.sparse.diags(1.0 / diag)
        #eigvals, eigvecs = lobpcg(K_rtb, X, M=M, tol=tol, largest=False)
        
    except ArpackNoConvergence as e:
        print(f"[WARNING] Only {e.eigenvalues.shape[0]} modes converged out of {n_modes}")
        eigvals = e.eigenvalues
        eigvecs = e.eigenvectors
    print(f"[INFO] eigsh completed in {time() - start:.2f} sec")
    # Remove near-zero modes (rigid body)
    #nonzero_mask = eigvals > 1e-8
    #eigvals = eigvals[nonzero_mask]
    #eigvecs = eigvecs[:, nonzero_mask]



    # Project modes back to atom space
    L_full = P @ eigvecs  # [3N x n_modes]
    return L_full, eigvals[6:] , eigvecs[:, 6:]
