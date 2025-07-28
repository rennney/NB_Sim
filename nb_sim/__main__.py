import sys
import json
import click
from time import time
from pathlib import Path
from nb_sim.io.viewer import save_pdb, save_pdb_like_original, save_pdb_trajectory, launch_pymol
from nb_sim.io.pdb_parser import Molecule, resolve_pdb_input
from nb_sim.utils.validation import filter_valid_blocks
from nb_sim.core.anm import build_anm_hessian, mass_weight_hessian ,build_anm_hessian1,build_anm_hessian2
from nb_sim.core.rtb import build_rtb_projection
from nb_sim.core.modes import compute_rtb_modes
from nb_sim.core.deform import deform_structure
import torch
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
    K = build_anm_hessian(mol.coords)
    
    masses = np.array([atom[2] for atom in mol.atoms], dtype=np.float64)
    K_w = mass_weight_hessian(K,masses)
    print(K_w.shape)
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
    alpha_vals = torch.cat([torch.linspace(-amplitude, amplitude, frames // 2 + 1),torch.linspace(amplitude, -amplitude, frames // 2 + 1)[1:]])
    #print(eigenvec.shape,sum(block_dofs_filtered))
    #check_rotation_magnitudes(eigenvec,block_dof,mode_index=mode)
    coord_list = [deform_structure(mol,blocks, eigenvec, a,mode_index=mode,block_dofs=block_dof) for a in alpha_vals]
    save_pdb_trajectory(pdb_path, out_path, coord_list,mol)
    print(f"[INFO] Full simulation complited in {time() - start:.2f} sec")
    if view:
        launch_pymol(pdb_path, out_path,only_deformed=False)




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
