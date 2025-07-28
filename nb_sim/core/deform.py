import torch
from nb_sim.utils.geometry import rotation_matrix, apply_nonlinear_deform


def deform_structure(mol,blocks, eigvecs, amplitude, mode_index=0, device=None, block_dofs=None):
    """
    Apply nonlinear motion to rigid blocks based on eigenmode deformation.
    
    Args:
        blocks: List[RigidBlock]
        amplitude: scalar amplitude multiplier
        device: torch.device

    Returns:
        coords_deformed: [N, 3] tensor of deformed atom coordinates
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #coords_out = mol.coords.new_zeros((mol.coords.shape[0], 3)).double()
    coords_out = mol.coords.clone()
    # Map from atom index to final coord
    atom_idx_map = {}

    if block_dofs is None:
        block_dofs = [6] * len(blocks)
    offset = 0
    for b, block in enumerate(blocks):
        atom_ids = block.atom_indices
        atom_coords = block.atom_coords.to(device)
        masses = block.atom_masses.to(device)
        com = block.com.to(device)

        Mb = masses.sum().item()
        sqrt_m = torch.sqrt(masses)

        # Estimate v and omega: first 3 and last 3 of mode
        I_b = block.compute_inertia_tensor().to(device)
        evals, evecs = torch.linalg.eigh(I_b)
        evals_clamped = torch.clamp(evals, min=1e-8)
        I_inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals_clamped)) @ evecs.T
        vec6 = eigvecs[:, mode_index] if isinstance(eigvecs, torch.Tensor) else eigvecs[:, mode_index]
        
        dof = block_dofs[b]
        vec_b = vec6[offset : offset + dof]
        offset += dof


        v = torch.as_tensor(vec_b[:3], dtype=torch.float64, device=device)
        if dof == 6 and b<len(blocks)-1:
            omega = torch.as_tensor(vec_b[3:], dtype=torch.float64, device=device)
            coords_def = apply_nonlinear_deform(atom_coords, v / Mb**0.5,  I_inv_sqrt@omega, com, amplitude)
        else:
            continue
            #coords_def = apply_nonlinear_deform(atom_coords, v / Mb**0.5, torch.zeros(3, dtype=torch.float64, device=device), com, amplitude)

        for i, aid in enumerate(atom_ids):
            coords_out[aid] = coords_def[i]
#        v = torch.as_tensor(vec6[:3], dtype=torch.float64, device=device) # translational part
#        omega = torch.as_tensor(vec6[3:], dtype=torch.float64, device=device)  # angular part
        # Apply extrapolation per block
        #print(f"[DEBUG] block {b}, v = {v}, ||v|| = {v.norm():.4e}, omega = {omega}, ||omega|| = {omega.norm():.4e}")
#        coords_def = apply_nonlinear_deform(atom_coords, v/Mb**0.5, I_inv_sqrt @ omega, com, amplitude)

#        for i, aid in enumerate(atom_ids):
#            atom_idx_map[aid] = coords_def[i]

    # Final assembly of reordered output
#    for aid, coord in atom_idx_map.items():
#        coords_out[aid] = coord
    return coords_out
