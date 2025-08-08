import torch
from torch.nn.functional import softplus
from nb_sim.core.deform import deform_structure
from nb_sim.utils.validation import filter_valid_blocks
from nb_sim.core.rtb import build_rtb_projection
from nb_sim.core.modes import compute_rtb_modes
import numpy as np


def fitter(mol_init, mol_final, eigvec_sel, blocks, block_dofs,
                         max_iter=200, positive_only=False, verbose=True):
    """
    Fit RTB coefficients with LBFGS for NOLB-style nonlinear deformation
    applied to initial geometry.

    Args:
        mol_init: Molecule object (initial)
        mol_final: Molecule object (target)
        eigvec_sel: torch.Tensor [sum(block_dofs) x m], RTB eigenvectors
        blocks: list of RigidBlock objects
        block_dofs: list of DOFs per block
        max_iter: max LBFGS iterations
        positive_only: constrain coefficients >= 0
        verbose: print convergence info

    Returns:
        coeffs_final: torch.Tensor [m]
    """
    device = mol_init.coords.device
    target_coords = mol_final.coords.to(device, dtype=torch.float64)

    # Initial guess: ones to avoid zero omega_norm
    params = torch.ones(eigvec_sel.shape[1], dtype=torch.float64, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([params], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        coeffs = softplus(params) if positive_only else params
        combined_rtb = eigvec_sel @ coeffs
        coords_pred = deform_structure(mol_init, blocks, combined_rtb,
                                       amplitude=1.0, mode_index=-1,
                                       block_dofs=block_dofs)
        loss = torch.nn.functional.mse_loss(coords_pred, target_coords)
        if verbose:
            with torch.no_grad():
                rmsd_val = torch.sqrt(torch.mean((coords_pred - target_coords) ** 2)).item()
                print(f"[LBFGS] Loss = {loss.item():.6e}, RMSD = {rmsd_val:.6f}")
        loss.backward()
        return loss

    optimizer.step(closure)

    coeffs_final = softplus(params).detach() if positive_only else params.detach()
    return coeffs_final
