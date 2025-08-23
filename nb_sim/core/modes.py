import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence , lobpcg
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import spilu
import scipy as sp
from scipy.sparse import eye
from time import time

def compute_rtb_modes(K, P, n_modes=20, tol=1e-8):
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
    
    #assert (P.T @ P).diagonal().min()>10**-6,  (" P has columns with zero norm — i.e., some of the rigid-body basis vectors are either degenerate or completely missing.")
    column_mask = np.ones(P.shape[1], dtype=bool)
    #PP = (P.T @ P).toarray()
    #print(PP[:12,:24])
    #import matplotlib.pyplot as plt
    #plt.imshow(PP)
    #plt.colorbar()
    #plt.show()
    K_rtb = P.T @ K @ P  # [6n x 6n], still sparse
    
    # 1) Atom-space (ANM) 3x3-block interactions: count distinct (i_atom, j_atom)
    #    blocks that have ≥1 nonzero entry in K.
    K_coo = K.tocoo()
    anm_block_pairs = set(zip(K_coo.row // 3, K_coo.col // 3))
    ANM_interactions_3x3 = len(anm_block_pairs)

    # 2) RTB-space block–block interactions from K_rtb (= P^T K P)
    #    We need a mapping from each RTB column to its parent block.
    #    Columns belonging to the same rigid block touch exactly the same atom rows in P.
    Pc = P.tocsc()
    col2blk = np.empty(P.shape[1], dtype=np.int32)
    sig2bid = {}
    bid = 0
    for c in range(Pc.shape[1]):
        rows = Pc.indices[Pc.indptr[c]:Pc.indptr[c+1]]
        atoms = tuple(np.unique(rows // 3))  # set of atoms this column (DOF) touches
        blk_id = sig2bid.setdefault(atoms, bid)
        if blk_id == bid:
            bid += 1
        col2blk[c] = blk_id

    Kr = K_rtb.tocoo()
    rtb_block_pairs = {(col2blk[r], col2blk[c]) for r, c in zip(Kr.row, Kr.col)}
    RTB_interactions_blocks = len(rtb_block_pairs)

    print(f"[INFO] ANM 3x3 blocks with ≥1 nz: {ANM_interactions_3x3}")
    print(f"[INFO] RTB block–block couplings: {RTB_interactions_blocks}")
    
    print(f"[INFO] Projecting full Hessian to RTB space... (size: {K_rtb.shape})")
    # Compute eigenvalues/eigenvectors of RTB Hessian
    # We skip first 6 zero modes (rigid-body)
    print(f"[INFO] Diagonalizing projected Hessian to find {n_modes} low-frequency modes...")
    start = time()
    try:
        #eigvals, eigvecs = eigsh(K_rtb.toarray(), k=n_modes+6, which='SM', tol=tol)
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        epsilon = 1e-10
        eigvals, eigvecs = eigsh(K_rtb, k=n_modes+6, sigma=epsilon, which='LM', tol=tol) # if K_rtb is positive semi-defined

        # Initial guess: random block of shape (N, n_modes+6)
        #print(eigvecs[:,0])
        #X = np.random.rand(K_rtb.shape[0], n_modes + 6)
        # Use diagonal as preconditioner
        #diag = K_rtb.diagonal()
        #diag[K_rtb.diagonal()==0]=1e-8
        #ilu = spilu(K_rtb.tocsc(), drop_tol=1e-3)
        #M = sp.linalg.LinearOperator(K_rtb.shape, ilu.solve)
        #M = sp.sparse.diags(1.0 / diag)
        #eigvals, eigvecs = lobpcg(K_rtb, X, M=M, tol=tol, largest=False)
        #print(eigvecs[:10,0])
        #print(eigvecs[:10,4])
        #print(eigvecs[:10,6])
        print("Mode Frequencies : ")
        #import math
        #print(np.sqrt(eigvals))
        print(eigvals)
        #print("Diagnostics : ")
        #expct = [0.00483575,0.0107064,0.0113522,0.0143678]
        #print(eigvals[6:]/[x**2 for x in expct])
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

    return L_full, eigvals , eigvecs , column_mask


