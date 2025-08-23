import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import scipy.sparse as sp

def build_anm_hessian(mol,coords, gamma=0.001, cutoff=5, dtype=np.float64): #6.55183
    """
    Build a memory-efficient sparse ANM Hessian using KD-tree.

    Args:
        coords: torch.Tensor of shape [N, 3]
        gamma: spring constant
        cutoff: cutoff radius in Ångstroms
        dtype: np.float32 (default) or np.float64

    Returns:
        scipy.sparse.coo_matrix: (3N x 3N) stiffness matrix
        
    Fully Vectorized
    """
    device = coords.device
    try:
        coords_np = coords.detach().to("cpu").numpy().astype(dtype)
    except Exception:
        coords_np = np.array(coords.tolist(), dtype=dtype)
    VDW = dict(C=1.70, N=1.55, O=1.52, S=1.80, P=1.80, H=1.20, MG=1.73, NA=2.27, CL=1.75)

    radii = np.array([VDW.get(elem, 1.70) for elem, *_ in mol.atoms])
    
    N = coords.shape[0]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')  # Mx2 (i<j)
    i = pairs[:,0]; j = pairs[:,1]

    rij = coords[i] - coords[j]           # (M,3)
    dist2 = np.einsum('ij,ij->i', rij, rij)
    mask = dist2 > 1e-18
    i, j, rij, dist2 = i[mask], j[mask], rij[mask], dist2[mask]

    u_ij = rij / np.sqrt(dist2)[:,None]   # (M,3)
    # k_ij blocks: -gamma * u u^T  (3x3 each)
    k_ij = -gamma * (u_ij[:,:,None] * u_ij[:,None,:])  # (M,3,3)
    # Build indices for 3x3 blocks
    blocks = np.arange(3)
    a,b = np.meshgrid(blocks, blocks, indexing='ij')
    a = a.ravel(); b = b.ravel()

    # Off-diagonals (i,j) and (j,i)
    row_ij = 3*i[:,None] + a; col_ij = 3*j[:,None] + b
    row_ji = 3*j[:,None] + a; col_ji = 3*i[:,None] + b

    # Diagonals (i,i) and (j,j)
    row_ii = 3*i[:,None] + a; col_ii = 3*i[:,None] + b
    row_jj = 3*j[:,None] + a; col_jj = 3*j[:,None] + b

    data_ij = k_ij[:,a,b]                 # (M,9)
    data_ji = k_ij[:,b,a]                 # symmetric
    data_ii = -k_ij[:,a,b]                # minus row sum
    data_jj = -k_ij[:,b,a]

    row = np.concatenate([row_ij, row_ji, row_ii, row_jj], axis=None)
    col = np.concatenate([col_ij, col_ji, col_ii, col_jj], axis=None)
    dat = np.concatenate([data_ij, data_ji, data_ii, data_jj], axis=None)

    K = coo_matrix((dat, (row, col)), shape=(3*N, 3*N))
    #print(K.toarray())
    print(f"[INFO] Hessian built with {len(i)} spring pairs (batched), dtype={dtype.__name__}")
    
    return K


def mass_weight_hessian(K, masses):
    mass_diag = np.repeat(masses, 3)  # shape [3N]
    M_inv_sqrt = 1.0 / np.sqrt(mass_diag)
    D = sp.diags(M_inv_sqrt)
    return D @ K @ D




#old version


def build_anm_hessian1(coords, gamma=1, cutoff=5.0, dtype=np.float32):
    """
    Build a memory-efficient sparse ANM Hessian using KD-tree.

    Args:
        coords: torch.Tensor of shape [N, 3]
        gamma: spring constant
        cutoff: cutoff radius in Ångstroms
        dtype: np.float32 (default) or np.float64

    Returns:
        scipy.sparse.coo_matrix: (3N x 3N) stiffness matrix
        
    Not Vectorized
    """
    device = coords.device
    try:
        coords_np = coords.detach().to("cpu").numpy().astype(dtype)
    except Exception:
        coords_np = np.array(coords.tolist(), dtype=dtype)
    N = coords_np.shape[0]

    tree = cKDTree(coords_np)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')  # [M, 2]

    row_idx = []
    col_idx = []
    data = []

    for i, j in pairs:
        rij = coords_np[i] - coords_np[j]
        dist2 = np.dot(rij, rij)
        if dist2 < 1e-8:
            continue

        outer = np.outer(rij, rij)
        k_ij = -gamma * outer / dist2  # [3x3]

        for a in range(3):
            for b in range(3):
                # i-j and j-i (off-diagonal)
                row_idx += [3*i+a, 3*j+a]
                col_idx += [3*j+b, 3*i+b]
                data    += [k_ij[a, b], k_ij[b, a]]

                # i-i and j-j (diagonal corrections)
                row_idx += [3*i+a, 3*j+a]
                col_idx += [3*i+b, 3*j+b]
                data    += [-k_ij[a, b], -k_ij[b, a]]

    K = coo_matrix((data, (row_idx, col_idx)), shape=(3*N, 3*N), dtype=dtype)
    print(f"[INFO] Hessian built with {len(pairs)} spring pairs, dtype={dtype.__name__}")
    return K


def build_anm_hessian2(coords, gamma=1.0, cutoff=5.0, dtype=np.float32):
    """
    Build a memory-efficient sparse ANM Hessian using KD-tree and vectorized operations.

    Args:
        coords: torch.Tensor of shape [N, 3] or np.ndarray
        gamma: spring constant
        cutoff: cutoff radius in Ångstroms
        dtype: np.float32 (default) or np.float64

    Returns:
        scipy.sparse.coo_matrix: (3N x 3N) stiffness matrix
    
    Partially Vectorized
    """
    device = coords.device
    try:
        coords_np = coords.detach().to("cpu").numpy().astype(dtype)
    except Exception:
        coords_np = np.array(coords.tolist(), dtype=dtype)

    N = coords_np.shape[0]
    tree = cKDTree(coords_np)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')  # shape [M, 2]
    i = pairs[:, 0]
    j = pairs[:, 1]

    # Vectorized differences
    rij = coords_np[i] - coords_np[j]     # [M, 3]
    dist2 = np.sum(rij**2, axis=1)        # [M]
    valid = dist2 > 1e-8                  # Avoid degenerate bonds

    i = i[valid]
    j = j[valid]
    rij = rij[valid]
    dist2 = dist2[valid]

    M = len(i)
    # Compute outer products vectorized
    rij_reshaped = rij[:, :, None]              # [M, 3, 1]
    rij_outer = rij_reshaped @ rij_reshaped.transpose(0, 2, 1)  # [M, 3, 3]
    k_ij = -gamma * rij_outer / dist2[:, None, None]            # [M, 3, 3]

    # Now build sparse indices for COO format
    row_idx = []
    col_idx = []
    data = []

    for a in range(3):
        for b in range(3):
            # Off-diagonal: i-j and j-i
            row_idx.extend(3 * i + a)
            col_idx.extend(3 * j + b)
            data.extend(k_ij[:, a, b])

            row_idx.extend(3 * j + a)
            col_idx.extend(3 * i + b)
            data.extend(k_ij[:, b, a])  # symmetric

            # Diagonal corrections: i-i and j-j
            row_idx.extend(3 * i + a)
            col_idx.extend(3 * i + b)
            data.extend(-k_ij[:, a, b])

            row_idx.extend(3 * j + a)
            col_idx.extend(3 * j + b)
            data.extend(-k_ij[:, b, a])

    K = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N, 3 * N), dtype=dtype)
    print(f"[INFO] Hessian built with {len(i)} spring pairs (vectorized), dtype={dtype.__name__}")
    return K
