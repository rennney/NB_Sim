import torch
from Bio.PDB import PDBParser
from Bio.PDB import PDBList
from nb_sim.core.rigid_block import RigidBlock
from nb_sim.utils.masses import atomic_masses

class Molecule:
    def __init__(self, pdb_file, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.atoms = []         # (element, res_id, mass,resname)
        self.coords = []        # raw list of [x, y, z]
        self.residue_map = {}   # res_id -> list of atom indices
        self.blocks = []        # List[RigidBlock]

        self._load_pdb(pdb_file)
        self._build_blocks()

    def _load_pdb(self, filename):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('mol', filename)

        for model in structure:
            for chain in model:
                for residue in chain:
                    hetfield, resseq, icode = residue.id

                    # Skip water molecules
                    if residue.resname.strip() in {"HOH", "WAT", "H2O"}:
                        continue

                    # Skip hetero residues (HETATM) if desired
                    if hetfield.strip() != "":
                        continue

                    # Use full ID to avoid clashing residues after TER
                    res_id = (chain.id, resseq, icode)
                    print(res_id)
                    for atom in residue:
                        print(atom)
                        element = atom.element.strip().capitalize()
                        if not element:
                            continue

                        mass = atomic_masses.get(element, 12.0)
                        serial = atom.serial_number
                        self.atoms.append((element, res_id, mass, residue.resname.strip(),serial))
                        self.coords.append(atom.coord)
                        self.residue_map.setdefault(res_id, []).append(len(self.atoms) - 1)
        print(self.residue_map)
        self.coords = torch.tensor(self.coords, dtype=torch.float64, device=self.device)

    def _build_blocks(self):
        for res_id, atom_ids in self.residue_map.items():
            coords = self.coords[atom_ids]
            masses = torch.tensor([self.atoms[i][2] for i in atom_ids], dtype=torch.float64, device=self.device)
            block = RigidBlock(atom_ids, coords, masses, device=self.device)
            self.blocks.append(block)


def resolve_pdb_input(pdb_input):
    """
    Resolves input: either returns local path or downloads from RCSB
    """
    from pathlib import Path
    
    if Path(pdb_input).exists():
        return Path(pdb_input)
    
    pdb_id = pdb_input.lower()
    pdb_id = pdb_id.split(".")[0]
    print(f"[INFO] Downloading PDB ID '{pdb_id}' from RCSB...")
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Failed to download PDB ID '{pdb_id}'")
    return pdb_path
