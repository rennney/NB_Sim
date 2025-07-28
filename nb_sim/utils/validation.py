import numpy as np

def filter_valid_blocks(blocks):
    # Currently no filter
    valid = []
    for i, block in enumerate(blocks):
        #if len(block.atom_indices) < 3:
            #print(block.__dict__)
            #print(f"[SKIP] Block {i} too small")
            #continue

        try:
            coords = block.atom_coords.cpu().numpy()
            com = block.com.cpu().numpy()
        except Exception:
            coords = np.array(block.atom_coords.tolist())
            com = np.array(block.com.tolist())

        rel = coords - com
        #if np.linalg.matrix_rank(np.dot(rel.T, rel), tol=1e-8) < 3:
            #print(f"[SKIP] Block {i} has singular inertia")
            #continue

        valid.append(block)

    print(f"[INFO] {len(valid)} / {len(blocks)} blocks retained")
    return valid

