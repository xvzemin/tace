################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import itertools
from itertools import combinations
from typing import Dict, Tuple, Optional, NamedTuple


import torch
from torch import Tensor


from cartnn.util import scatter_sum

STR = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
EINSUM_STR = list("defghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

RADIAL_BASIS = {
    "radial_basis": "j0",
    "num_radial_basis": 8,
    "distance_transform": None,
    "polynomial_cutoff": 5,
    "order": 0,
    "trainable": False,
    "apply_cutoff": True,
}


ANGULAR_BASIS = {
    "type": "ictd",
    "norm": True,
}


RADIAL_MLP = {
    "hidden": [
        [64, 64, 64],
        [64, 64, 64],
    ],
    "act": "silu",
    "bias": False,
    "enable_layer_norm": False,
}


INTER = {
    "max_paths": 1,
    "restriction": [None, None],
    "allow_nosym": True,
    "kernel": False,
    "residual": False,
}


PROD = {
    "restriction": None,
    "correlation": 3,
    "allow_nosym": True,
    "element": True,
    "coupled": True,
    "kernel": False,
    "add_source_target_embedding": False,
    "normalizer": {
      "type": "fixed",
      "hidden": [64],
      "act_1": 'silu',
      "act_2": 'tanh',
      "bias": False,
      "scale_shift_trainable": True,
    }

}


ICTD = {"weight": "max"}


READOUT_MLP = {"hidden": [16], "act": "silu", "bias": False}


SCALE_SHIFT = {
    "scale_type": "rms_forces",
    "shift_type": "mean_delta_energy_per_atom",
    "scale_trainable": False,
    "shift_trainable": False,
    "scale_dict": "auto",
    "shift_dict": "auto",
}
SHORT_RANGE = {
    'enable_zbl': False
}

LONG_RANGE = {
    'les': 
        {
            'enable_les': False,
            'les_arguments': None,
        },
}


def expand_dims_to(T: Tensor, n_dim: int, dim: int = -1) -> Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


def nonsym_tensor_to_sym(T: Tensor) -> Tensor:

    rank = T.ndim - 2
    perm_indices = list(range(-rank, 0))
    perms = list(itertools.permutations(perm_indices))

    sym_Ts = []
    for perm in perms:
        full_perm = list(range(T.ndim - rank)) + [T.ndim + i for i in perm]
        permuted_T = T.permute(full_perm)
        sym_Ts.append(permuted_T)

    sym_T = torch.stack(sym_Ts, dim=0).mean(dim=0)
    return sym_T


def delta_tensor(i: int, j: int, ndim: int, device=None, dtype=None) -> Tensor:

    delta = torch.eye(3, device=device, dtype=dtype)
    for _ in range(ndim - 2):
        delta = delta.unsqueeze(0)
    perm = list(range(ndim))
    perm[i], perm[-2] = perm[-2], perm[i]
    perm[j], perm[-1] = perm[-1], perm[j]
    delta = delta.permute(*perm)
    return delta


def sym_tensor_to_traceless(T: Tensor) -> Tensor:
    """
    For r >= 4, numerical errors can be significant; full tracelessness requires using float64

    Compute the first-order, second-order, ..., up to floor(n/2) traces step by step,
    and subtract their corresponding contributions to obtain a fully symmetric traceless tensor
    """

    B, C = T.shape[:2]
    spatial_shape = T.shape[2:]

    T = T.view(B * C, *spatial_shape)

    ndim = T.ndim
    n = ndim - 1

    result = T.clone()
    base_combs = list(combinations(range(-n, 0), 2))
    for k in range(1, n // 2 + 1):
        denom = 1.0
        for j in range(2, k + 2):
            denom *= 3 + 2 * (n - j)
        coeff = ((-1) ** k) / denom
        corr = torch.zeros_like(T)
        for pairs in combinations(base_combs, k):
            idxs = [idx for pair in pairs for idx in pair]
            if len(set(idxs)) < 2 * k:
                continue
            delta = torch.ones_like(T)
            for i, j in pairs:
                delta = delta * delta_tensor(i, j, ndim, device=T.device, dtype=T.dtype)
            trace = torch.sum(T * delta, dim=tuple(idxs), keepdim=True)
            corr += delta * trace
        result = result + coeff * corr
    return result.view(B, C, *spatial_shape)


def add_to_left(
    T1: Dict[int, Tensor], T2: Dict[int, Tensor]
) -> Dict[int, torch.Tensor]:

    for k in T2:
        if k in T1:
            T1[k] = T1[k] + T2[k]
        else:
            T1[k] = T2[k]
    return T1


def add_to_right(T1: Dict[int, Tensor], T2: Dict[int, Tensor]) -> Dict[int, Tensor]:

    for k in T1:
        if k in T2:
            T2[k] = T2[k] + T1[k]
        else:
            T2[k] = T1[k]
    return T2


def satisfy(r_1:int, r_2: int, restriction, r_o: Optional[int] = None, tensor_product_only: bool = False):
    r_1_r_2 = None; bool_1 = True
    r_o_r_1 = None; bool_2 = True

    if isinstance(restriction, str):
        r_1_r_2 = restriction
    elif isinstance(restriction, Dict):
        r_1_r_2 = restriction.get('r_1_r_2', None)
        r_o_r_1 = restriction.get('r_o_r_1', None)
    else:
        return True
    
    if r_1_r_2 == "<=":
        bool_1 = (r_1 <= r_2)
    elif r_1_r_2 == "==":
        bool_1 = (r_1 == r_2)
    else:
        bool_1 = True
    
    if r_o is not None:
        if r_o_r_1 == "<=":
            bool_2 = (r_o <= r_1)
        elif r_o_r_1 == "==":
            bool_2 = (r_o == r_1)
        else:
            bool_2 = True
    return bool_1 and bool_2


def compute_fixed_charge_dipole(
    charges: Tensor,
    positions: Tensor,
    batch: Tensor,
    num_graphs: int,
) -> Tensor:
    mu = positions * charges.unsqueeze(-1) * 4.8032047  # e·Å to Debye
    return scatter_sum(src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)


def torch_full_3x3_to_voigt_6_stress(stress_tensor: Tensor) -> Tensor:
    """
    Convert stress tensor [batch, 3, 3] -> [batch, 6] in Voigt notation,
    matching ASE's full_3x3_to_voigt_6_stress.
    """
    s = stress_tensor
    s_voigt = torch.stack(
        [
            s[..., 0, 0],  # σ_xx
            s[..., 1, 1],  # σ_yy
            s[..., 2, 2],  # σ_zz
            0.5 * (s[..., 1, 2] + s[..., 2, 1]),  # σ_yz
            0.5 * (s[..., 0, 2] + s[..., 2, 0]),  # σ_xz
            0.5 * (s[..., 0, 1] + s[..., 1, 0]),  # σ_xy
        ],
        dim=-1,
    )
    return s_voigt


def vec_to_skew(v: Tensor) -> Tensor:
    """ TODO, maybe not (1,1,1) to (2,1,1), should use basis change
    v: (B, 3) tensor
    return: (B, 3, 3) skew-symmetric matrix
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    zero = torch.zeros_like(x)

    row1 = torch.stack([zero, -z, y], dim=1)
    row2 = torch.stack([z, zero, -x], dim=1)
    row3 = torch.stack([-y, x, zero], dim=1)

    skew = torch.stack([row1, row2, row3], dim=1)
    return skew


def select_corresponding_level_for_scalar(x: Tensor, node_level: Tensor, num_levels: int) -> Tensor:
    '''
    For rank-0 tensor, 
    '''
    B = x.size(0)
    C_LEVELS = x.size(1)
    mask = torch.zeros(B, num_levels, C_LEVELS // num_levels, device=x.device, dtype=x.dtype)
    idx = torch.arange(B, device=x.device, dtype=torch.int64)
    mask[idx, node_level, :] = 1
    mask = mask.reshape((B, C_LEVELS))
    return x * mask

def select_corresponding_level_for_tensor(x: Tensor, node_level: Tensor, num_levels: int) -> Tensor:
    '''
    For rank>0 tensor, 
    '''
    B = x.size(0)
    C_LEVELS = x.size(1)

    mask = torch.zeros(B, num_levels, C_LEVELS // num_levels, device=x.device, dtype=x.dtype)
    idx = torch.arange(B, device=x.device, dtype=torch.int64)
    mask[idx, node_level, :] = 1
    mask = mask.reshape((B,C_LEVELS))
    return x * expand_dims_to(mask, x.ndim, -1)

class Graph(NamedTuple):
    lmp: bool
    lmp_data: Optional[Tensor]
    lmp_natoms: Tuple[int, int]
    num_graphs: int
    displacement: Optional[Tensor]
    positions: Tensor
    edge_vector: Tensor
    edge_length: Tensor
    lattice: Tensor
    node_level: Tensor
    num_atoms_arange: Tensor


class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None
    

def dict2flatten(max_r: int, t: Dict[int, Tensor]):
    tmp = []
    B, C = t[0].shape[:2]
    for k in sorted(t.keys()):
        flat = t[k].reshape(B, C, -1) 
        tmp.append(flat)
    return torch.cat(tmp, dim=-1).reshape(B, -1)


def flatten2dict(max_r: int, t: Tensor, C: int) -> Dict[int, Tensor]:
    B = t.size(0)
    ndim = (3 ** (max_r + 1) - 1) // 2
    t = t.reshape(B, C, ndim)  
    outs = {}
    start_idx = 0
    for r in range(max_r+1):
        shape = (B, C,) + (3,) * r
        delta = 3**r
        outs[r] = t[:, :, start_idx:start_idx+delta].reshape(shape)
        start_idx += delta
    return outs


