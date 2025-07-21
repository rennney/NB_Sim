import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

def build_anm_hessian1(coords, gamma=1.0, cutoff=10.0, dtype=np.float32):
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


def build_anm_hessian2(coords, gamma=1.0, cutoff=10.0, dtype=np.float32):
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

def build_anm_hessian(coords, gamma=1.0, cutoff=10.0, dtype=np.float32):
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

    N = coords_np.shape[0]
    tree = cKDTree(coords_np)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]

    rij = coords_np[i] - coords_np[j]
    dist2 = np.sum(rij ** 2, axis=1)
    valid = dist2 > 1e-8
    i = i[valid]
    j = j[valid]
    rij = rij[valid]
    dist2 = dist2[valid]

    M = len(i)
    rij_outer = rij[:, :, None] * rij[:, None, :]  # [M, 3, 3]
    k_ij = -gamma * rij_outer / dist2[:, None, None]  # [M, 3, 3]

    # Batch indexing for 3x3 blocks
    blocks = np.arange(3)
    a, b = np.meshgrid(blocks, blocks, indexing='ij')  # (3,3)
    a = a.flatten()  # (9,)
    b = b.flatten()  # (9,)

    idx_repeat = np.repeat(np.arange(M), 9)
    a_rep = np.tile(a, M)
    b_rep = np.tile(b, M)
    
    row_offdiag_ij = 3 * np.repeat(i, 9) + np.tile(a, M)
    col_offdiag_ij = 3 * np.repeat(j, 9) + np.tile(b, M)
    val_offdiag_ij = k_ij[idx_repeat, a_rep, b_rep]

    row_offdiag_ji = 3 * np.repeat(j, 9) + np.tile(a, M)
    col_offdiag_ji = 3 * np.repeat(i, 9) + np.tile(b, M)
    val_offdiag_ji = k_ij[idx_repeat, b_rep, a_rep]

    row_diag_ii = 3 * np.repeat(i, 9) + np.tile(a, M)
    col_diag_ii = 3 * np.repeat(i, 9) + np.tile(b, M)
    val_diag_ii = -k_ij[idx_repeat, a_rep, b_rep]

    row_diag_jj = 3 * np.repeat(j, 9) + np.tile(a, M)
    col_diag_jj = 3 * np.repeat(j, 9) + np.tile(b, M)
    val_diag_jj = -k_ij[idx_repeat, b_rep, a_rep]

    # Stack all triplets
    row_idx = np.concatenate([row_offdiag_ij, row_offdiag_ji, row_diag_ii, row_diag_jj])
    col_idx = np.concatenate([col_offdiag_ij, col_offdiag_ji, col_diag_ii, col_diag_jj])
    data =    np.concatenate([val_offdiag_ij, val_offdiag_ji, val_diag_ii, val_diag_jj])

    # Now coo_matrix
    K = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N, 3 * N), dtype=dtype)
    print(f"[INFO] Hessian built with {M} spring pairs (batched), dtype={dtype.__name__}")
    return K
