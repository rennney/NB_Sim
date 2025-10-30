import sys
import json
import click
from time import time
from pathlib import Path
from nb_sim.io.viewer import save_pdb, save_pdb_like_original, save_pdb_trajectory, launch_pymol
from nb_sim.io.pdb_parser import Molecule, resolve_pdb_input
from nb_sim.utils.validation import filter_valid_blocks
from nb_sim.core.anm import build_anm_hessian, mass_weight_hessian
from nb_sim.core.rtb import build_rtb_projection
from nb_sim.core.modes import compute_rtb_modes
from nb_sim.core.deform import deform_structure
from nb_sim.core.fitter import fitter
from nb_sim.utils.align import align_final_to_initial
import torch
import math
import numpy as np


@click.group()
@click.option("-s","--store",type=click.Path(),
              envvar="NMSIM_STORE",
              help="File for primary data storage (input/output)")
@click.option("-o","--outstore",type=click.Path(),
              help="File for output (primary only input)")
@click.pass_context
def cli(ctx, store, outstore):
    '''
    NMSim command line interface
    '''
    pass

@cli.command()
@click.option("-i", "--input", required=True, type=str,
              help="Input PDB file or PDB ID (e.g., 4bij)")
@click.option("-o", "--output", required=False, type=click.Path(),
              help="Output PDB file (deformed)")
@click.option("-n", "--n_modes", default=6, show_default=True,
              help="Number of normal modes to compute")
@click.option("-a", "--amplitude", default=5.0, show_default=True,
              help="Deformation amplitude")
@click.option("--view/--no-view", default=True,
              help="Launch PyMOL for visualizing original and deformed structures")
@click.option("-m","--mode", default=0, show_default=True,
              help="Which vibrational mode to apply (0-indexed)")
@click.option("-f", "--frames", default=11)
def run_simulator(input, output, n_modes, amplitude, view,mode,frames):
    """
    Run Simulation
    """
    start = time()
    pdb_path = resolve_pdb_input(input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = Molecule(str(pdb_path), device=device)
    print(f"[INFO] Loaded {len(mol.atoms)} atoms into {len(mol.blocks)} blocks.")

    print("[INFO] Building ANM Hessian...")
    K = build_anm_hessian(mol,mol.coords)
    
    masses = np.array([atom[2] for atom in mol.atoms], dtype=np.float64)
    K_w = mass_weight_hessian(K,masses)
    #print(K_w.shape)
    print("[INFO] Building RTB projection matrix...")
    blocks = filter_valid_blocks(mol.blocks)
    P, block_dof= build_rtb_projection(blocks, N_atoms=len(mol.atoms))
    print(f"[INFO] Computing {n_modes} RTB normal modes...")
    L_full, eigvals, eigenvec , column_mask = compute_rtb_modes(K_w, P, n_modes=n_modes)
    def filter_block_dofs(block_dofs, column_mask):
        new_block_dofs = []
        offset = 0
        for dof in block_dofs:
            if np.all(column_mask[offset : offset + dof]):
                new_block_dofs.append(dof)
            offset += dof
        return new_block_dofs
    #block_dofs_filtered = filter_block_dofs(block_dof, column_mask)
    print(f"[INFO] Applying deformation with amplitude {amplitude}...")
    try:
        modes_tensor = torch.from_numpy(L_full)
    except Exception:
        modes_tensor = torch.tensor(L_full.tolist(), dtype=torch.float64)
    #coords_def = deform_structure(mol,blocks, eigenvec, amplitude,mode_index=mode)
    out_path = output or Path(pdb_path).with_suffix(".deformed.pdb")
    #save_pdb_like_original(pdb_path,out_path, coords_def)
    #print(f"[INFO] Deformed structure saved to {out_path}")
    #alpha_vals = torch.cat([
    #    torch.linspace(0, -amplitude, frames // 4 + 1),                      # 0 → -amp
    #    torch.linspace(-amplitude, amplitude, frames // 2 + 1)[1:],                 # -amp → +amp
    #    torch.linspace(amplitude, 0, frames // 4 + 1)[1:]                     # +amp → 0
    #])
    #alpha_vals = torch.tensor([0,-amplitude/2,-amplitude,-amplitude,-amplitude/2,0,amplitude/2,amplitude,amplitude,amplitude/2])
    #alpha_vals = torch.tensor([0,amplitude/2,amplitude,amplitude,amplitude/2,0,-amplitude/2,-amplitude,-amplitude,-amplitude/2])
    k = torch.arange(frames,  device=device)
    alpha_vals = amplitude * torch.sin(2*math.pi * k / frames)
    #print("frames: ",alpha_vals)
    #print(eigenvec.shape,sum(block_dofs_filtered))
    #check_rotation_magnitudes(eigenvec,block_dof,mode_index=mode)
    coord_list = [deform_structure(mol,blocks, eigenvec, a,mode_index=mode,block_dofs=block_dof) for a in alpha_vals]
    save_pdb_trajectory(pdb_path, out_path, coord_list,mol)
    print(f"[INFO] Full simulation complited in {time() - start:.2f} sec")
    if view:
        launch_pymol(pdb_path, out_path,only_deformed=False)




@cli.command()
@click.option("-i", "--input", required=True, type=str,
              help="Input PDB file or PDB ID (e.g., 4bij)")
@click.option("-o", "--output", required=False, type=click.Path(),
              help="Output PDB file (deformed)")
@click.option("-n", "--n_modes", default=6, show_default=True,
              help="Number of normal modes to compute")
@click.option("-a", "--amplitude", default=5.0, show_default=True,
              help="Deformation amplitude")
@click.option("--view/--no-view", default=True,
              help="Launch PyMOL for visualizing original and deformed structures")
@click.option("-f", "--frames", default=11, show_default=True,
              help="Number of frames in oscillation")
@click.option("-m", "--modes", required=True, type=str,
              help="Comma-separated list of mode indices (e.g., 0,1,4)")
@click.option("-w","--weights", type=str, default=None,
              help="Comma-separated weights for each mode (e.g., 0.2,0.5,0.3). Must match number of modes.")

def run_simulator_multimode(input, output, n_modes, amplitude, view, frames, modes, weights):
    """
    Run Simulation With Combined Modes
    """
    from time import time
    import numpy as np
    import torch
    from pathlib import Path

    start = time()
    pdb_path = resolve_pdb_input(input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol = Molecule(str(pdb_path), device=device)
    print(f"[INFO] Loaded {len(mol.atoms)} atoms into {len(mol.blocks)} blocks.")

    print("[INFO] Building ANM Hessian...")
    K = build_anm_hessian(mol,mol.coords)

    masses = np.array([atom[2] for atom in mol.atoms], dtype=np.float64)
    K_w = mass_weight_hessian(K, masses)
    print(K_w.shape)

    print("[INFO] Building RTB projection matrix...")
    blocks = filter_valid_blocks(mol.blocks)
    P, block_dof = build_rtb_projection(blocks, N_atoms=len(mol.atoms))

    print(f"[INFO] Computing {n_modes} RTB normal modes...")
    L_full, eigvals, eigvec, column_mask = compute_rtb_modes(K_w, P, n_modes=n_modes)

    if not modes:
        raise ValueError("You must specify at least one mode index using --modes")

    mode_idxs = np.array([int(m.strip()) for m in modes.split(",")], dtype=int)

    if weights:
        weight_vals = np.array([float(w.strip()) for w in weights.split(",")], dtype=np.float64)
        if len(weight_vals) != len(mode_idxs):
            raise ValueError("Number of weights must match number of modes")
    else:
        weight_vals = np.ones_like(mode_idxs, dtype=np.float64)

    weight_vals /= np.linalg.norm(weight_vals)

    # Construct combined eigenvector
    combined_vec = sum(w * eigvec[:, idx] for w, idx in zip(weight_vals, mode_idxs))
    #combined_vec = combined_vec / np.linalg.norm(combined_vec)  # normalize for amplitude scaling

    print(f"[INFO] Applying deformation with amplitude {amplitude} from modes {mode_idxs}...")


    alpha_vals = torch.cat([
        torch.linspace(0, amplitude, frames // 4 + 1),                      # 0 → -amp
        torch.linspace(amplitude, -amplitude, frames // 2 + 1)[1:],                 # -amp → +amp
        torch.linspace(-amplitude, 0, frames // 4 + 1)[1:]                     # +amp → 0
    ])

    coord_list = [deform_structure(mol, blocks, combined_vec, a, mode_index=-1,block_dofs=block_dof) for a in alpha_vals]

    out_path = output or Path(pdb_path).with_suffix(".combined.deformed.pdb")
    save_pdb_trajectory(pdb_path, out_path, coord_list, mol)

    print(f"[INFO] Full simulation completed in {time() - start:.2f} sec")
    if view:
        launch_pymol(pdb_path, out_path, only_deformed=False)





def check_rotation_magnitudes(eigvecs, block_dofs, mode_index=0):
    print(f"Analyzing RTB mode {mode_index}")
    offset = 0
    rotation_norms = []
    translation_norms = []

    for b, dof in enumerate(block_dofs):
        vec = eigvecs[offset:offset + dof, mode_index]
        offset += dof

        if dof == 6:
            v = vec[:3]
            w = vec[3:]
            vnorm = np.linalg.norm(v)
            wnorm = np.linalg.norm(w)
            print(f"Block {b:3d}: ||v|| = {vnorm:.3e}, ||ω|| = {wnorm:.3e}, ratio ω/v = {wnorm/vnorm:.2f}")
            rotation_norms.append(wnorm)
            translation_norms.append(vnorm)
        else:
            print(f"Block {b:3d}: translation-only")

    rotation_norms = np.array(rotation_norms)
    translation_norms = np.array(translation_norms)

    print("\n=== Summary ===")
    print(f"Average ||v||: {translation_norms.mean():.3e}")
    print(f"Average ||ω||: {rotation_norms.mean():.3e}")
    print(f"Max ||ω||:     {rotation_norms.max():.3e}")
    print(f"Min ||ω||:     {rotation_norms.min():.3e}")

    return rotation_norms, translation_norms


@cli.command()
@click.option("-i", "--initial", required=True, type=str)
@click.option("-g", "--goal", required=True, type=str)
@click.option("-n", "--n_modes", default=20, show_default=True)
@click.option("--skip", default=0, show_default=True)
@click.option("--max-iter", default=200, show_default=True)
@click.option("--positive-only", is_flag=True, default=False)
@click.option("--frames", default=6, show_default=True)
@click.option("-o", "--output", required=False, type=click.Path())
@click.option("--view/--no-view", default=True)

def fit_modes(initial, goal, n_modes, skip,
                             max_iter, positive_only, frames, output, view):
    """
    Fit coefficients for initial RTB modes with LBFGS (NOLB-style)
    and generate nonlinear rigid-block morph from initial structure.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pdb_init = resolve_pdb_input(initial)
    pdb_final = resolve_pdb_input(goal)
    mol_init_full = Molecule(str(pdb_init), device=device)
    mol_final_full = Molecule(str(pdb_final), device=device)

    # Get matching atom indices
    idx_self, idx_other = mol_init_full.match_atoms_by_residue(mol_final_full)

    # Reduce mol_final to only matching atoms
    mol_final = Molecule.__new__(Molecule)  # bypass __init__
    mol_final.device = device
    mol_final.atoms = [mol_final_full.atoms[j] for j in idx_other]
    mol_final.coords = mol_final_full.coords[idx_other]
    mol_final.residue_map = {rid: [] for rid in mol_init_full.residue_map.keys() if rid in [a[1] for a in mol_final.atoms]}
    for k, (element, res_id, mass, resname, serial) in enumerate(mol_final.atoms):
        mol_final.residue_map[res_id].append(k)
    mol_final.blocks = []
    mol_final._build_blocks()
    
    idx_self, idx_other = mol_final.match_atoms_by_residue(mol_init_full)

    # Reduce mol_final to only matching atoms
    mol_init = Molecule.__new__(Molecule)  # bypass __init__
    mol_init.device = device
    mol_init.atoms = [mol_init_full.atoms[j] for j in idx_other]
    mol_init.coords = mol_init_full.coords[idx_other]
    mol_init.residue_map = {rid: [] for rid in mol_final_full.residue_map.keys() if rid in [a[1] for a in mol_init.atoms]}
    for k, (element, res_id, mass, resname, serial) in enumerate(mol_init.atoms):
        mol_init.residue_map[res_id].append(k)
    mol_init.blocks = []
    mol_init._build_blocks()
    
    R, t = align_final_to_initial(mol_init, mol_final, mode="atoms", mass_weighted=True)

    print("Check Mol Residues : ", len(mol_init.blocks),len(mol_final.blocks))
    # Build Hessian & modes from initial structure
    K = build_anm_hessian(mol_init,mol_init.coords)
    masses_np = np.array([atom[2] for atom in mol_init.atoms], dtype=np.float64)
    K_w = mass_weight_hessian(K, masses_np)
    blocks = filter_valid_blocks(mol_init.blocks)
    P, block_dofs = build_rtb_projection(blocks, N_atoms=len(mol_init.atoms))
    _, eigvals, eigvec, _ = compute_rtb_modes(K_w, P, n_modes=n_modes + skip)
    eigvec_sel = torch.from_numpy(eigvec[:, skip:]).to(device, dtype=torch.float64)

    # Fit coefficients
    coeffs_final = fitter(mol_init, mol_final, eigvec_sel,
                                        blocks, block_dofs,
                                        max_iter=max_iter,
                                        positive_only=positive_only)

    print("[INFO] Final coefficients:", coeffs_final.cpu().numpy())

    # Build combined RTB vector from initial modes
    combined_rtb_final = eigvec_sel @ coeffs_final

    # Generate trajectory from initial structure

    #alpha_vals = torch.cat([
    #    torch.linspace(0, -1, frames // 4 + 1),                      # 0 → -amp
    #    torch.linspace(-1, 1, frames // 2 + 1)[1:],                 # -amp → +amp
    #    torch.linspace(1, 0, frames // 4 + 1)[1:]                     # +amp → 0
    #])
    alpha_vals = torch.cat([
        torch.linspace(0, 1, frames // 2 + 1),                      # 0 → -amp
        torch.linspace(1, 0, frames // 2 + 1)[1:]                     # +amp → 0
    ])
    #k = torch.arange(frames,  device=device)
    #alpha_vals = 1 * torch.sin(2*math.pi * k / frames)
    coord_list = [
        deform_structure(mol_init, blocks, combined_rtb_final,
                         amplitude=a.item(), mode_index=-1,
                         block_dofs=block_dofs)
        for a in alpha_vals
    ]

    out_path = output or Path(pdb_init).with_suffix(".nolb_lbfgs_fit.pdb")
    save_pdb_trajectory(pdb_init, out_path, coord_list, mol_init)
    print(f"[INFO] NOLB-LBFGS trajectory saved to {out_path}")

    if view:
        launch_pymol(pdb_init, out_path,pdb_final, only_deformed=False)






@cli.command()
@click.option("-i", "--input", required=False, type=str,
              help="Input PDB file or PDB ID (e.g., 4bij)")
def run_test(input):
    """
    Run Small Test
    """
    coords = torch.tensor([
        # Block 0
        [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.2],
        [1.75, 0.4, 0.3],

        # Block 1 (translated)
        [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.2],
        [1.75, 0.4, 0.3],

    ], dtype=torch.float64)
    masses = torch.ones(coords.shape[0], dtype=torch.float64)
    blocks = []
    from nb_sim.core.rigid_block import RigidBlock
    for i in range(2):
        atom_ids = list(range(3 * i, 3 * (i + 1)))
        block_coords = coords[atom_ids]
        block_masses = masses[atom_ids]
        block = RigidBlock(atom_ids, block_coords, block_masses)
        blocks.append(block)
    for b in blocks:
        print(b.atom_coords)
        I = b.compute_inertia_tensor()
        print("estimate")
        com = np.average(np.array(b.atom_coords.tolist()),axis=0,weights=np.array(b.atom_masses.tolist()))
        rc = np.array(b.atom_coords.tolist()) - com
        I_np = np.zeros((3, 3))
        for r, m in zip(rc, np.array(masses.tolist())):
            I_np += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        print(I_np)
        print("from the code")
        print(I)
        print(b.com)
        evals = torch.linalg.eigvalsh(I)
        print(f"Block {i} inertia eigenvalues:", evals)
    K = build_anm_hessian(coords)
    K_w = mass_weight_hessian(K,np.array(masses.tolist()))
    print(K.toarray())
    print(K_w.toarray())
    print("[INFO] Building RTB projection matrix...")
    #blocks = filter_valid_blocks(blocks)
    P = build_rtb_projection(blocks, N_atoms=len(masses))
    print(P)
    print(P.toarray())
    print(f"[INFO] Computing {10} RTB normal modes...")
    L_full, eigvals, eigenvec = compute_rtb_modes(K_w, P, n_modes=1)
    
    
def main():
    cli(obj=None)


if '__main__' == __name__:
    main()
