import torch

def kabsch_align(moving_xyz, target_xyz, weights=None):
    """
    moving_xyz -> rotate/translate to target_xyz
    moving_xyz, target_xyz: [N,3] tensors (same N, device can be CUDA)
    weights: [N] or None. If None, uniform.
    Returns R [3,3], t [3], and aligned coords.
    """
    device = moving_xyz.device
    moving = moving_xyz
    target = target_xyz

    if weights is None:
        w = torch.ones(moving.shape[0], device=device, dtype=moving.dtype)
    else:
        w = weights.to(device=device, dtype=moving.dtype)

    w = w / (w.sum() + 1e-12)
    # Weighted centroids
    mu_m = (w[:, None] * moving).sum(dim=0)
    mu_t = (w[:, None] * target).sum(dim=0)

    Xm = moving - mu_m
    Xt = target - mu_t

    # Weighted covariance
    H = Xm.T @ (w[:, None] * Xt)  # [3,3]
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection fix
    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_t - R @ mu_m
    aligned = (R @ moving.T).T + t
    return R, t, aligned

def align_final_to_initial(mol_init, mol_final, mode="atoms", mass_weighted=True):
    """
    mode: "atoms" (use matched atoms) or "blocks" (use block COMs)
    mass_weighted: if True, weight by atomic masses (atoms mode) or block masses (blocks mode)
    """
    # Build coordinates and weights
    if mode == "atoms":
        X = mol_final.coords          # [N,3] torch
        Y = mol_init.coords           # [N,3] torch
        if mass_weighted:
            masses = torch.tensor([a[2] for a in mol_final.atoms], device=X.device, dtype=X.dtype)
            w = masses
        else:
            w = None
    elif mode == "blocks":
        # Align by block centers of mass (robust to side-chain differences)
        def block_COMs(mol):
            coms = []
            weights = []
            for b in mol.blocks:
                idx = b.atom_indices
                coords = mol.coords[idx]
                m = torch.tensor([mol.atoms[i][2] for i in idx], device=coords.device, dtype=coords.dtype)
                M = m.sum()
                com = (coords * m[:, None]).sum(dim=0) / (M + 1e-12)
                coms.append(com)
                weights.append(M)
            return torch.stack(coms, dim=0), torch.tensor(weights, device=coords.device, dtype=coords.dtype)
        X, wX = block_COMs(mol_final)
        Y, wY = block_COMs(mol_init)
        # Sanity: blocks correspond 1:1 because you built them from matching residues
        if mass_weighted:
            w = wX  # same indexing/order
        else:
            w = None
    else:
        raise ValueError("mode must be 'atoms' or 'blocks'")

    R, t, _ = kabsch_align(X, Y, weights=w)

    # Apply to all atoms of mol_final (even if 'blocks' mode used for fitting)
    mol_final.coords = (R @ mol_final.coords.T).T + t
    return R, t
