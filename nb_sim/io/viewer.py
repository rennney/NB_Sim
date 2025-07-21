from pathlib import Path
import subprocess

def save_pdb(path, mol, coords):
    """
    Save deformed coordinates to a PDB file.

    Args:
        path: output PDB file path (str or Path)
        mol: Molecule object
        coords: [N, 3] torch tensor of deformed positions
    """
    path = Path(path)
    with open(path, "w") as f:
        for i, ((element, res_id, _,resname), pos) in enumerate(zip(mol.atoms, coords)):
            chain_id, resnum = res_id
            x, y, z = pos.tolist()
            line = (
                f"ATOM  {i+1:5d} {element:>2s}  {resname:3s} {chain_id:1s}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}\n"
            )
            f.write(line)
        f.write("END\n")

def save_pdb_like_original(pdb_in, pdb_out, coords):
    """
    Save deformed coordinates to a PDB file by modifying only x,y,z fields
    in the original PDB text lines.

    Args:
        pdb_in: original PDB file path
        pdb_out: output PDB file path
        coords: torch tensor of deformed positions, shape [N, 3]
    """
    with open(pdb_in, "r") as fin:
        lines = fin.readlines()

    with open(pdb_out, "w") as fout:
        atom_idx = 0
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if atom_idx >= len(coords):
                    raise ValueError("More ATOM lines than coordinates given")

                x, y, z = coords[atom_idx].tolist()
                # Format: columns 31–54 (inclusive): %8.3f each
                new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                # Replace in fixed-width format
                new_line = line[:30] + new_xyz + line[54:]
                fout.write(new_line)
                atom_idx += 1
            else:
                fout.write(line)

        fout.write("END\n")

def save_pdb_trajectory(pdb_in, pdb_out, coord_list):
    """
    Save a multi-model PDB trajectory by modifying only x, y, z fields
    in the original PDB lines (preserves full formatting).

    Args:
        pdb_in: path to original PDB file
        pdb_out: path to output trajectory (multi-model PDB)
        coord_list: list of [N, 3] torch tensors of coordinates
    """
    with open(pdb_in, "r") as fin:
        ref_lines = [line for line in fin if line.startswith(("ATOM", "HETATM"))]

    n_atoms = len(ref_lines)

    with open(pdb_out, "w") as fout:
        for model_idx, coords in enumerate(coord_list):
            if len(coords) != n_atoms:
                raise ValueError(f"[Frame {model_idx}] Coordinate count {len(coords)} ≠ {n_atoms} atoms")

            fout.write(f"MODEL     {model_idx+1:4d}\n")
            for i, line in enumerate(ref_lines):
                x, y, z = coords[i].tolist()
                new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                fout.write(line[:30] + new_xyz + line[54:])
            fout.write("ENDMDL\n")

        fout.write("END\n")

def launch_pymol(pdb_ref=None, pdb_def=None, only_deformed=False, headless=False):
    """
    Launch PyMOL to visualize original and/or deformed structures.

    Args:
        pdb_ref: path to reference/original PDB
        pdb_def: path to deformed PDB
        only_deformed: if True, only load and show deformed structure
        headless: if True, suppress GUI
    """
    print("[INFO] Launching PyMOL...")

    script = []

    if only_deformed:
        script.append(f"load {pdb_def}, deformed")
        script.append("hide everything")
        script.append("show cartoon, deformed")
        script.append("color orange, deformed")
        script.append("zoom deformed")
    else:
        script.append(f"load {pdb_ref}, original")
        script.append(f"load {pdb_def}, deformed")
        script.append("hide everything")
        script.append("show cartoon, original")
        script.append("show cartoon, deformed")
        script.append("color gray80, original")
        script.append("color orange, deformed")
        script.append("align deformed, original")
        script.append("zoom")

    flags = ["pymol"]
    if headless:
        flags += ["-cq"]
    flags += ["-d", "\n".join(script)]

    subprocess.run(flags)
