import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import scipy.sparse as sp

def build_anm_hessian(mol, coords, gamma=0.001, cutoff=5.0, dtype=np.float64):
    """
    Build sparse ANM Hessian (3N x 3N) using KD-tree neighbor search.
    Accepts Torch tensor on any device; computes on CPU/NumPy.
    Returns: scipy.sparse.coo_matrix with dtype=np.float64 (default).
    """
    # ---- always move to CPU NumPy for SciPy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    coords_np = np.ascontiguousarray(coords_np, dtype=dtype)
    N = coords_np.shape[0]

    # KD-tree on CPU
    tree = cKDTree(coords_np)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')   # shape [M,2], i<j
    if pairs.size == 0:
        # no springs within cutoff: return empty matrix
        return coo_matrix((3*N, 3*N), dtype=dtype)

    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)

    # pairwise unit vectors
    rij   = coords_np[i] - coords_np[j]                            # [M,3]
    dist2 = np.einsum('ij,ij->i', rij, rij)                        # [M]
    mask  = dist2 > 1e-18
    i, j, rij, dist2 = i[mask], j[mask], rij[mask], dist2[mask]
    if i.size == 0:
        return coo_matrix((3*N, 3*N), dtype=dtype)

    inv_r = 1.0 / np.sqrt(dist2)
    u_ij  = rij * inv_r[:, None]                                   # [M,3]

    # k_ij blocks: -gamma * (u u^T), one 3x3 per edge
    k_ij = -gamma * (u_ij[:, :, None] * u_ij[:, None, :])          # [M,3,3]

    # indices for 3x3 blocks
    a, b = np.meshgrid(np.arange(3, dtype=np.int64),
                       np.arange(3, dtype=np.int64), indexing='ij')
    a = a.ravel(); b = b.ravel()

    row_ij = 3*i[:, None] + a; col_ij = 3*j[:, None] + b
    row_ji = 3*j[:, None] + a; col_ji = 3*i[:, None] + b
    row_ii = 3*i[:, None] + a; col_ii = 3*i[:, None] + b
    row_jj = 3*j[:, None] + a; col_jj = 3*j[:, None] + b

    data_ij = k_ij[:, a, b]
    data_ji = k_ij[:, b, a]           # symmetric
    data_ii = -k_ij[:, a, b]          # minus row sum
    data_jj = -k_ij[:, b, a]

    row = np.concatenate([row_ij, row_ji, row_ii, row_jj], axis=None)
    col = np.concatenate([col_ij, col_ji, col_ii, col_jj], axis=None)
    dat = np.concatenate([data_ij, data_ji, data_ii, data_jj], axis=None).astype(dtype, copy=False)

    K = coo_matrix((dat, (row, col)), shape=(3*N, 3*N), dtype=dtype)
    print(f"[INFO] Hessian built with {i.size} spring pairs, dtype={K.dtype}")
    return K

def mass_weight_hessian(K, masses):
    mass_diag = np.repeat(np.asarray(masses, dtype=np.float64), 3)  # [3N]
    D = sp.diags(1.0 / np.sqrt(mass_diag))
    return D @ K @ D
