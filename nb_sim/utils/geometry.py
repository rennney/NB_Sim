import torch

def rotation_matrix(axis, angle):
    """
    Rodrigues rotation matrix
    axis: [3], unit vector
    angle: scalar
    """
    #axis = axis / (axis.norm() + 1e-8)
    K = skew(axis)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return I + torch.sin(angle) * K + (2*torch.sin(angle/2)**2) * (K @ K)
    #return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

def skew(v):
    zero = torch.tensor(0.0, device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([zero, -v[2], v[1]]),
        torch.stack([v[2], zero, -v[0]]),
        torch.stack([-v[1], v[0], zero])
    ])

def apply_nonlinear_deform(coords, v, omega, com, amplitude):
    """
    Apply nonlinear rigid-body motion based on v, omega.
    coords: [N, 3]
    """
    #print("Call Deformation")
    #print("v=",v)
    #print("w=",omega)
    #print("coords=",coords)
    #print("com=",com)
    #print("amp=",amplitude)
    omega_norm = omega.norm()
    if omega_norm < 1e-18:
        return coords + amplitude * v  # pure translation
    n = omega / omega_norm
    dphi = amplitude * omega_norm
    # Split v into parallel and perpendicular components
    v_parallel =  (v @ n) * n
    v_perp = v - v_parallel
    #print("n=",n)
    #print("omega_norm=",omega_norm)
    #print("dphi=",dphi)
    #print("v_par=",v_parallel)
    #print("v_perp=",v_perp)
    # Rotation center
    r = com +  torch.cross(n, v_perp) / (omega_norm)
    import math
    R = rotation_matrix(n, dphi)
    #print("check shapes rotation : ",R.shape,(coords - r).shape)
    rotated = (coords - r)@R.T  + r
    #print(r)
    #print((coords - r))
    #print(rotated)
    #print(v_parallel * amplitude)
    return rotated + v_parallel * amplitude


