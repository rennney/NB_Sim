import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

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
