import torch

class RigidBlock:
    def __init__(self, atom_indices, atom_coords, atom_masses, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.atom_indices = atom_indices
        self.atom_coords = atom_coords.to(self.device)
        self.atom_masses = atom_masses.to(self.device)

        self.mass = self.atom_masses.sum()
        self.com = self.compute_center_of_mass()
        self.inertia = self.compute_inertia_tensor()

    def compute_center_of_mass(self):
        return (self.atom_coords * self.atom_masses[:, None]).sum(dim=0) / self.mass


    #intersting method that effectively computes the inertia tensor I = sum_i m_i [ (r_i dot r_i) I_3x3 - r_i cross r_i ], r_i is a vctor
    def compute_inertia_tensor(self):
        r = self.atom_coords - self.com  # [N, 3]
        m = self.atom_masses             # [N]

        # Precompute terms
        r_sq = (r ** 2).sum(dim=1)       # [N]
        outer = r[:, :, None] * r[:, None, :]  # [N, 3, 3]

        # Identity matrix broadcasted to [N, 3, 3]
        I_3 = torch.eye(3, dtype=torch.float64, device=self.device).expand(r.shape[0], 3, 3)

        # Mass-weighted terms
        term1 = m[:, None, None] * r_sq[:, None, None] * I_3   # [N, 3, 3]
        term2 = m[:, None, None] * outer                      # [N, 3, 3]

        I = (term1 - term2).sum(dim=0)  # [3, 3]
        return I

    def build_projection_vectors(self):
        """Return the 6 rigid-body basis vectors (3 trans + 3 rot) for this block as [3N, 6] tensor."""
        coords = self.atom_coords
        masses = self.atom_masses
        com = self.com
        N = coords.shape[0]

        rel = coords - com  # [N, 3]
        sqrt_m = torch.sqrt(masses)  # [N]

        # Create 3 translation vectors (broadcasted sqrt_m along each axis)
        trans = torch.eye(3, device=self.device, dtype=torch.float64).repeat(N, 1)  # [3N, 3]
        trans *= sqrt_m.repeat_interleave(3).unsqueeze(1)  # broadcast mass weights

        # Compute cross products for rotation basis: cross(e_k, r)
        # Cross products for each basis direction: x, y, z
        zeros = torch.zeros_like(rel[:, 0])
        rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]

        rot = torch.stack([
            torch.stack([ zeros, -rz,  ry], dim=1),
            torch.stack([ rz,  zeros, -rx], dim=1),
            torch.stack([-ry,  rx, zeros], dim=1),
        ], dim=2)  # [N, 3, 3]

        # Apply sqrt_m weighting
        rot = rot * sqrt_m[:, None, None]  # [N, 3, 3]

        # Reshape to [3N, 3] rotation block
        rot = rot.transpose(0, 2).reshape(3*N, 3)  # [3N, 3]

        # Concatenate [trans | rot]
        B = torch.cat([trans, rot], dim=1)  # [3N, 6]
        return B
