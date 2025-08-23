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


def save_pdb_trajectory(pdb_in, pdb_out, coord_list, mol):
    """
    Save a multi-model PDB trajectory by updating only x, y, z coordinates
    for atoms retained in `mol`, matching by atom serial number.
    - If pdb_in already has MODEL records, only the FIRST model's body is used as template.
    - Original MODEL/ENDMDL/END records are stripped from the template.
    """
    with open(pdb_in, "r") as fin:
        orig = fin.readlines()

    # --- carve out a template body (no MODEL/ENDMDL/END) ---
    model_starts = [i for i, l in enumerate(orig) if l.startswith("MODEL")]
    if model_starts:
        s = model_starts[0]
        try:
            e = next(i for i in range(s + 1, len(orig)) if orig[i].startswith("ENDMDL"))
        except StopIteration:
            e = len(orig) - 1
        template_lines = [l for l in orig[s+1:e] if not l.startswith(("MODEL", "ENDMDL", "END"))]
    else:
        template_lines = [l for l in orig if not l.startswith(("MODEL", "ENDMDL", "END"))]

    # --- serial mapping from mol order -> coord index ---
    atom_serials = []
    for i, atom in enumerate(mol.atoms):
        if len(atom) >= 5:
            atom_serials.append(int(atom[4]))
        else:
            raise ValueError(f"Missing serial number in atom {i}: {atom}")
    serial_to_coord_idx = {s: i for i, s in enumerate(atom_serials)}

    # Sanity
    for model_idx, coords in enumerate(coord_list):
        if len(coords) != len(atom_serials):
            raise ValueError(
                f"[Frame {model_idx}] Coordinate count {len(coords)} "
                f"≠ number of retained atoms {len(atom_serials)}"
            )

    def rewrite_atom_line(line, xyz):
        x, y, z = xyz
        new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
        return line[:30] + new_xyz + "  "+line[54:]

    # --- write trajectory ---
    with open(pdb_out, "w") as fout:
        for model_idx, coords in enumerate(coord_list):
            fout.write(f"MODEL     {model_idx+1:4d}\n")
            for line in template_lines:
                if line.startswith(("ATOM", "HETATM")):
                    try:
                        s = int(line[6:11])
                    except ValueError:
                        # Nonstandard serial; just pass through unchanged
                        fout.write(line)
                        continue
                    if s in serial_to_coord_idx:
                        idx = serial_to_coord_idx[s]
                        fout.write(rewrite_atom_line(line, coords[idx].tolist()))
                    else:
                        # "skipped" atom: preserve as-is
                        fout.write(line)
                else:
                    # metadata/TER/etc: preserve
                    fout.write(line)
            fout.write("ENDMDL\n")
        fout.write("END\n")


def save_pdb_trajectory_old(pdb_in, pdb_out, coord_list, mol):
    """
    Save a multi-model PDB trajectory by updating only x, y, z coordinates
    for atoms retained in `mol`, matching them by atom serial number.
    All other lines (TER, skipped atoms, metadata) are preserved.

    Args:
        pdb_in: original PDB file path
        pdb_out: output multi-model PDB file path
        coord_list: list of [N_filtered, 3] torch tensors (only for retained atoms)
        mol: Molecule object (with filtered atoms, each having a serial number)
    """
    with open(pdb_in, "r") as fin:
        orig_lines = fin.readlines()

    # Build mapping from atom serial number → original line index
    serial_to_line_idx = {
        int(line[6:11]): i
        for i, line in enumerate(orig_lines)
        if line.startswith(("ATOM", "HETATM"))
    }

    # Extract serials from mol.atoms (expecting atom tuples with serial at index 4)
    atom_serials = []
    for i, atom in enumerate(mol.atoms):
        if len(atom) >= 5:
            atom_serials.append(atom[4])
        else:
            raise ValueError(f"Missing serial number in atom {i}: {atom}")

    # Get matching line indices in original file
    retained_line_indices = []
    for s in atom_serials:
        if s not in serial_to_line_idx:
            raise KeyError(f"[ERROR] Atom serial {s} not found in original PDB file.")
        retained_line_indices.append(serial_to_line_idx[s])

    # Sanity check
    for model_idx, coords in enumerate(coord_list):
        if len(coords) != len(retained_line_indices):
            raise ValueError(
                f"[Frame {model_idx}] Coordinate count {len(coords)} "
                f"≠ number of retained atoms {len(retained_line_indices)}"
            )

    # Write the new PDB trajectory
    with open(pdb_out, "w") as fout:
        for model_idx, coords in enumerate(coord_list):
            fout.write(f"MODEL     {model_idx+1:4d}\n")

            coord_iter = iter(coords.tolist())
            for i, line in enumerate(orig_lines):
                if i in retained_line_indices:
                    x, y, z = next(coord_iter)
                    new_xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    new_line = line[:30] + new_xyz + line[54:]
                    fout.write(new_line)
                else:
                    fout.write(line)

            fout.write("ENDMDL\n")

        fout.write("END\n")


def launch_pymol(pdb_ref=None, pdb_def=None, pdb_final=None, only_deformed=False, headless=False):
    """
    Launch PyMOL to visualize original, deformed, and optionally final-aligned structures.

    Args:
        pdb_ref: path to reference/original PDB
        pdb_def: path to deformed PDB
        pdb_final: optional path to final/aligned PDB
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
        if pdb_ref:
            script.append(f"load {pdb_ref}, original")
            script.append("show cartoon, original")
            script.append("color gray80, original")
        if pdb_def:
            script.append(f"load {pdb_def}, deformed")
            script.append("show cartoon, deformed")
            script.append("color orange, deformed")
        if pdb_final:
            script.append(f"load {pdb_final}, final")
            script.append("show cartoon, final")
            script.append("color cyan, final")

        # Align all to original if available
        if pdb_ref:
            if pdb_def:
                script.append("align deformed, original")
            if pdb_final:
                script.append("align final, original")
        script.append("zoom")

    flags = ["pymol"]
    if headless:
        flags += ["-cq"]
    flags += ["-d", "\n".join(script)]

    subprocess.run(flags)
