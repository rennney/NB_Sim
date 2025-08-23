import numpy as np
from scipy.sparse import coo_matrix

def build_rtb_projection(blocks, N_atoms):
    import numpy as np
    from scipy.sparse import coo_matrix

    row_idx = []
    col_idx = []
    data = []
    block_dof = []
    col_offset = 0

    for b, block in enumerate(blocks):
        atom_ids = block.atom_indices
        use_rotation = True
        try:
            coords = block.atom_coords.cpu().numpy()
            masses = block.atom_masses.cpu().numpy()
            com = block.com.cpu().numpy()
        except Exception:
            coords = np.array(block.atom_coords.tolist())
            masses = np.array(block.atom_masses.tolist())
            com = np.array(block.com.tolist())
        Mb = block.mass.item()
        try:
            I_b = block.compute_inertia_tensor().cpu().numpy()
        except Exception:
            I_b = np.array(block.compute_inertia_tensor().tolist())

        try:
            evals, evecs = np.linalg.eigh(I_b)
            evals_clamped = np.clip(evals, 1e-8, None)
            I_b_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals_clamped)) @ evecs.T
        except np.linalg.LinAlgError:
            print(f"[WARN] Block {b} has singular inertia tensor, use translation only")
            use_rotation = False
        if use_rotation:
            block_dof.append(6)
        else:
            block_dof.append(3)

        # === Translation ===
        for i, atom_id in enumerate(atom_ids):
            m = masses[i]
            for j in range(3):  # x, y, z
                row = 3 * atom_id + j
                col = col_offset + j
                row_idx.append(row)
                col_idx.append(col)
                data.append(np.sqrt(m / Mb))

        # === Rotation (Fixed) ===
        if use_rotation:
            # Build full 3N_block x 3 matrix
            B_rot = np.zeros((3 * len(atom_ids), 3))

            E = np.eye(3)
            for j in range(3):
                u = I_b_inv_sqrt @ E[:, j]           # u_j = I^{-1/2} e_j
                col_entries = []
                for i in range(len(atom_ids)):
                    rel = coords[i] - com            # r_i - c
                    rot_vec = np.cross(rel, u)       # rel × (I^{-1/2} e_j)   <-- RIGHT
                    col_entries.extend(np.sqrt(masses[i]) * rot_vec)
                B_rot[:, j] = col_entries

            # Orthonormalize rotation columns
            #Q_rot, _ = np.linalg.qr(B_rot)  # shape [3N_block x 3]
            Q_rot = B_rot
            for j in range(3):  # rotation axis index
                for i, atom_id in enumerate(atom_ids):
                    for k in range(3):  # vector component
                        row = 3 * atom_id + k
                        col = col_offset + 3 + j
                        val = Q_rot[3 * i + k, j]
                        row_idx.append(row)
                        col_idx.append(col)
                        data.append(val)

        col_offset += 6 if use_rotation else 3
    P = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N_atoms, col_offset))
    
    return P.tocsr(), block_dof





def build_rtb_projection_old(blocks, N_atoms):
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
    block_dof = []
    col_offset = 0
    for b, block in enumerate(blocks):
        atom_ids = block.atom_indices
        use_rotation = True
        try:
            coords = block.atom_coords.cpu().numpy()
            masses = block.atom_masses.cpu().numpy()
            com = block.com.cpu().numpy()
        except Exception:
            coords = np.array(block.atom_coords.tolist())
            masses = np.array(block.atom_masses.tolist())
            com = np.array(block.com.tolist())
        Mb = block.mass.item()
        try:
            I_b = block.compute_inertia_tensor().cpu().numpy()
        except Exception:
            I_b = np.array(block.compute_inertia_tensor().tolist())

        try:
                #I_b = block.compute_inertia_tensor().cpu().numpy()
            evals, evecs = np.linalg.eigh(I_b)
            evals_clamped = np.clip(evals, 1e-8, None)  # avoid div by zero
            I_b_inv_sqrt = (evecs @ np.diag(1.0 / np.sqrt(evals_clamped)) @ evecs.T)  # [3x3]
        except np.linalg.LinAlgError:
            print(f"[WARN] Block {b} has singular inertia tensor, use translation only")
            use_rotation=False
        if use_rotation:
            block_dof.append(6)
        else:
            block_dof.append(3)
            continue
        for i, atom_id in enumerate(atom_ids):
            m = masses[i]
            r = coords[i]
            rel = r - com

            for j in range(3):  # x, y, z
                # Translation block
                row = 3 * atom_id + j
                col = col_offset + j
                row_idx.append(row)
                col_idx.append(col)
                data.append(np.sqrt(m / Mb))#np.sqrt(m / Mb))
                if not use_rotation:
                    continue
                # Rotation block
                rel_vec = -cross_unit(j, rel)  # e_j × (r_i - COM)
                rot_vec = (I_b_inv_sqrt @ rel_vec)  # shape (3,)
            #print("rel : ",rel)
            #print("rel_vec : ",rel_vec)
            #print("rot_vec : ",np.sqrt(m) *rot_vec)
                for k in range(3):
                    row = 3 * atom_id + k
                    col = col_offset + 3 + j
                    row_idx.append(row)
                    col_idx.append(col)
                    data.append(np.sqrt(m) * rot_vec[k])
                    #data.append(rel_vec[k])
        col_offset += 6 if use_rotation else 3
    P = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N_atoms, col_offset))

    return P.tocsr() , block_dof

def cross_unit(j, vec):
    """Cross product of unit basis e_j with vector vec"""
    if j == 0:
        return np.array([0, -vec[2], vec[1]])
    elif j == 1:
        return np.array([vec[2], 0, -vec[0]])
    elif j == 2:
        return np.array([-vec[1], vec[0], 0])



def build_rtb_projection_old(blocks, N_atoms):
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

        try:
            try:
                I_b = block.compute_inertia_tensor().cpu().numpy()
            except Exception:
                I_b = np.array(block.compute_inertia_tensor().tolist())
                #I_b = block.compute_inertia_tensor().cpu().numpy()
            evals, evecs = np.linalg.eigh(I_b)
            evals_clamped = np.clip(evals, 1e-8, None)  # avoid div by zero
            I_b_inv_sqrt = (evecs @ np.diag(1.0 / np.sqrt(evals_clamped)) @ evecs.T)  # [3x3]
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping block {b} due to singular inertia tensor")
            continue
        for i, atom_id in enumerate(atom_ids):
            m = masses[i]
            r = coords[i]
            rel = r - com

            for j in range(3):  # x, y, z
                # Translation block
                row = 3 * atom_id + j
                col = 6 * b + j
                row_idx.append(row)
                col_idx.append(col)
                data.append(np.sqrt(m / Mb))#np.sqrt(m / Mb))

                # Rotation block
                rel_vec = -cross_unit(j, rel)  # e_j × (r_i - COM)
                rot_vec = (I_b_inv_sqrt @ rel_vec)  # shape (3,)
                print("rel : ",rel)
                print("rel_vec : ",rel_vec)
                print("rot_vec : ",rot_vec)
                for k in range(3):
                    row = 3 * atom_id + k
                    col = 6 * b + 3 + j
                    row_idx.append(row)
                    col_idx.append(col)
                    data.append(np.sqrt(m) * rot_vec[k])
                    #data.append(rel_vec[k])

    P = coo_matrix((data, (row_idx, col_idx)), shape=(3 * N_atoms, 6 * len(blocks)))
    return P.tocsr()
