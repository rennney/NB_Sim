import numpy as np
from scipy.sparse import coo_matrix

def build_rtb_projection(blocks, N_atoms):
    """
    Construct the sparse RTB projection matrix P ∈ R^{3N × 6n}.
    
    Each block contributes 6 columns:
        - 3 for translation (mass-weighted identity)
        - 3 for rotation: cross(r_i - r_COM)

    Args:
        blocks: list of RigidBlock instances
        N_atoms: total number of atoms

    Returns:
        P: scipy.sparse.coo_matrix of shape (3N_atoms, 6 * n_blocks)
    """
    row_idx = []
    col_idx = []
    data = []

    for b, block in enumerate(blocks):
        atom_ids = block.atom_indices
        try:
            coords = block.atom_coords.cpu().numpy()
            masses = block.atom_masses.cpu().numpy()
            com = block.com.cpu().numpy()
        except Exception:
            coords = np.array(block.atom_coords.tolist())
            masses = np.array(block.atom_masses.tolist())
            com = np.array(block.com.tolist())
        Mb = block.mass.item()

        for i, atom_id in enumerate(atom_ids):
            m = masses[i]
            r = coords[i]
            rel = r - com

            for j in range(3):  # x, y, z
                # Translation block: mass-weighted identity
                row = 3 * atom_id + j
                col = 6 * b + j
                row_idx.append(row)
                col_idx.append(col)
                data.append(np.sqrt(m / Mb))

                # Rotation block: cross product
                rel_vec = cross_unit(j, rel)  # e_j × (r_i - COM)
                for k in range(3):
                    row = 3 * atom_id + k
                    col = 6 * b + 3 + j
                    row_idx.append(row)
                    col_idx.append(col)
                    data.append(np.sqrt(m) * rel_vec[k])

    P = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N_atoms, 6 * len(blocks)))
    return P.tocsr()

def cross_unit(j, vec):
    """Cross product of unit basis e_j with vector vec"""
    if j == 0:
        return np.array([0, -vec[2], vec[1]])
    elif j == 1:
        return np.array([vec[2], 0, -vec[0]])
    elif j == 2:
        return np.array([-vec[1], vec[0], 0])
