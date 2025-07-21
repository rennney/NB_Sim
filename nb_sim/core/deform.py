import torch
from nb_sim.utils.geometry import rotation_matrix, apply_nonlinear_deform

def deform_structure(mol,blocks, eigvecs, amplitude, mode_index=0, device=None):
    """
    Apply nonlinear motion to rigid blocks based on eigenmode deformation.
    
    Args:
        blocks: List[RigidBlock]
        L_full: [3N, M] all-atom normal modes (torch tensor)
        amplitude: scalar amplitude multiplier
        device: torch.device

    Returns:
        coords_deformed: [N, 3] tensor of deformed atom coordinates
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #L_full = L_full.to(device).double()
    
    # Flatten all atom coords in [3N] format
    #coords_out = torch.zeros_like(blocks[0].atom_coords.new_zeros((sum(len(b.atom_indices) for b in blocks), 3)))
    coords_out = mol.coords.new_zeros((mol.coords.shape[0], 3)).double()
    # Map from atom index to final coord
    atom_idx_map = {}

    for b, block in enumerate(blocks):
        atom_ids = block.atom_indices
        atom_coords = block.atom_coords.to(device)
        masses = block.atom_masses.to(device)
        com = block.com.to(device)

        Mb = masses.sum()
        sqrt_m = torch.sqrt(masses)

        # Extract 6-vector for this block: [v_x, v_y, v_z, w_x, w_y, w_z]
        #indices = []
        #for aid in atom_ids:
        #    indices += [3 * aid + i for i in range(3)]
        #vec_block = L_full[indices]  # [3N_b, M]

        # For now just use the first mode (can generalize later)
        #displacements = vec_block[:, mode_index].reshape(-1, 3)  # [N_b, 3]

        # Estimate v and omega: first 3 and last 3 of mode

        vec6 = eigvecs[:, mode_index] if isinstance(eigvecs, torch.Tensor) else eigvecs[:, mode_index]

        vec6 = vec6[6 * b : 6 * b + 6]

        
        v = torch.as_tensor(vec6[:3], dtype=torch.float64, device=device) # translational part
        omega = torch.as_tensor(vec6[3:], dtype=torch.float64, device=device)  # angular part
        # Apply extrapolation per block
        #print(f"[DEBUG] block {b}, v = {v}, ||v|| = {v.norm():.4e}, omega = {omega}, ||omega|| = {omega.norm():.4e}")
        coords_def = apply_nonlinear_deform(atom_coords, v, omega, com, amplitude)
        for i, aid in enumerate(atom_ids):
            atom_idx_map[aid] = coords_def[i]

    # Final assembly of reordered output
    for aid, coord in atom_idx_map.items():
        coords_out[aid] = coord
    return coords_out
