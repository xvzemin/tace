################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from itertools import combinations


import torch
from torch import Tensor
from e3nn.o3._spherical_harmonics import SphericalHarmonics


################################################################################
# Authors: Zemin Xu
# License: Academic / Non-Commercial Use Only
################################################################################
# Using ICT_decomposition Code from paper,
# High-Rank Irreducible Cartesian Tensor Decomposition and Bases of Equivariant Spaces

from typing import Tuple, List


import torch
from e3nn import o3


def ICT_decomposition(n_total: int) -> Tuple[List[List[int]], List[torch.Tensor]]:
    n_now = 0
    j_now = 0
    path_list = []
    this_path = []
    this_pathmatrix = o3.wigner_3j(0, 0, 0)
    pathmatrices_list = []

    # generate paths and path matrices
    def paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total):
        if n_now <= n_total:
            this_path.append(j_now)
            for j in [j_now + 1, j_now, j_now - 1]:
                if not (j_now == 0 and (j != 1)) and n_now + 1 <= n_total:
                    cgmatrix = o3.wigner_3j(1, j_now, j)
                    this_pathmatrix_ = torch.einsum(
                        "abc,dce->dabe", this_pathmatrix, cgmatrix
                    )
                    this_pathmatrix_ = this_pathmatrix_.reshape(
                        cgmatrix.shape[0], -1, cgmatrix.shape[-1]
                    )
                    paths_generate(
                        n_now + 1, j, this_path.copy(), this_pathmatrix_, n_total
                    )
            if n_now == n_total:
                this_pathmatrix = this_pathmatrix.reshape(-1, this_pathmatrix.shape[-1])
                this_pathmatrix = this_pathmatrix * (
                    1.0 / (this_pathmatrix**2).sum(0)[0] ** (0.5)
                )  # normalize
                pathmatrices_list.append(this_pathmatrix)
                path_list.append(this_path)
        return

    paths_generate(n_now, j_now, this_path, this_pathmatrix, n_total)
    decomp_list = []
    for path_matrix in pathmatrices_list:
        decomp_list.append(path_matrix @ path_matrix.T)
    return path_list, decomp_list

def expand_dims_to(T: Tensor, n_dim: int, dim: int = -1) -> Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


SphericalHarmonics = SphericalHarmonics


def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


def _norm(n: int) -> float:
    """(2n - 1)!! / n!"""
    num = double_factorial(2 * n - 1)
    den = factorial(n)
    return num / den


def delta_tensor(i: int, j: int, ndim: int, device=None, dtype=None) -> Tensor:
    delta = torch.eye(3, device=device, dtype=dtype)
    for _ in range(ndim - 2):
        delta = delta.unsqueeze(0)
    perm = list(range(ndim))
    perm[i], perm[-2] = perm[-2], perm[i]
    perm[j], perm[-1] = perm[-1], perm[j]
    delta = delta.permute(*perm)
    return delta


def symmetric_outer_product(v: Tensor, n: int, norm: bool = True) -> Tensor:
    out = torch.ones_like(v[..., 0])
    for _ in range(n):
        out = out[..., None] * expand_dims_to(v, out.ndim + 1, dim=v.ndim - 1)
    if norm:
        out = out * _norm(n)
    return out


def subtract_traces(T: Tensor, n: int) -> Tensor:
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
                delta = delta * delta_tensor(
                    i, j, n + 1, device=T.device, dtype=T.dtype
                )
            trace = torch.sum(T * delta, dim=tuple(idxs), keepdim=True)
            corr += delta * trace
        result = result + coeff * corr
    return result


def symmetric_traceless_outer_product(v: Tensor, n: int, norm: bool = True) -> Tensor:
    T = symmetric_outer_product(v, n, norm)
    return subtract_traces(T, n)


class AngularICTD(torch.nn.Module):
    def __init__(self, r: int, norm: bool = True) -> None:
        super().__init__()
        Ps, Qs = ICT_decomposition(r)
        self.register_buffer(f"Q", Qs[0].to(torch.get_default_dtype()))
        self.r = r
        self.norm = norm
        del Ps, Qs

    def forward(self, v: Tensor) -> Tensor:
        T = symmetric_outer_product(v, self.r, self.norm)
        B = T.size(0)
        if B == 0:
            return T # isolated atoms TODO check whether exists BUG
        else:
            REST = T.size()[1:]
            T = T.reshape(B, -1)
            T = T @ self.Q
            T = T.reshape((B,) + REST)
            return T

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.r}, norm={self.norm})"


ANGULAR_EMBEDDING_GTO = symmetric_outer_product
ANGULAR_EMBEDDING_ICT = symmetric_traceless_outer_product


def check_all_traces_zero(T: Tensor, atol=1e-6) -> bool:
    n = T.ndim - 1
    for i, j in combinations(range(-n, 0), 2):
        trace = torch.einsum("...ii->...", T.movedim((i, j), (-2, -1)))
        if not torch.allclose(trace, torch.zeros_like(trace), atol=atol):
            return False
    return True


def check_ctr_to_unity(T: Tensor, v: Tensor) -> Tensor:
    "[batch, 3,3...] ✖️ [batch, 3]"
    n = T.ndim - 1
    while n >= 2:
        y = expand_dims_to(v, n + 1, 1)
        T = torch.sum(T * y, dim=-1)
        n = T.ndim - 1
    assert torch.allclose(T, v)


def check_recursively_detrace():
    torch.set_default_dtype(torch.float64)
    v = torch.randn(4, 3)
    v = v / torch.norm(v, keepdim=True, dim=-1)
    for n in range(1, 10):
        T = ANGULAR_EMBEDDING_ICT(v, n)
        _bool = check_all_traces_zero(T)
        print(f"n={n}: All traces zero? {_bool}; shape: {T.shape}")
        assert _bool
        check_ctr_to_unity(T, v)


def check_ict_decomposition():
    torch.set_default_dtype(torch.float64)
    v = torch.randn(4, 3)
    v = v / torch.norm(v, keepdim=True, dim=-1)
    for n in range(1, 6):
        T = symmetric_outer_product(v, n)
        T = AngularICTD(n)(v)
        _bool = check_all_traces_zero(T)
        print(f"n={n}: All traces zero? {_bool}; shape: {T.shape}")
        assert _bool
        check_ctr_to_unity(T, v)


def check_equivalent():
    torch.set_default_dtype(torch.float64)
    v = torch.randn(4, 3)
    v = v / torch.norm(v, keepdim=True, dim=-1)
    for n in range(1, 6):
        T1 = ANGULAR_EMBEDDING_ICT(v, n)
        T2 = symmetric_outer_product(v, n)
        T2 = AngularICTD(n)(v)
        assert torch.allclose(T1, T2), f"v={n} failed"

class AngularICTD(torch.nn.Module):
    def __init__(self, r: int, norm: bool = True) -> None:
        super().__init__()
        Ps, Qs = ICT_decomposition(r)
        self.register_buffer(f"Q", Qs[0].to(torch.get_default_dtype()))
        self.r = r
        self.norm = norm
        del Ps, Qs

    def forward(self, v: Tensor) -> Tensor:
        T = symmetric_outer_product(v, self.r, self.norm)
        return T
        # B = T.size(0)
        # if B == 0:
        #     return T # isolated atoms TODO check whether exists BUG
        # else:
        #     REST = T.size()[1:]
        #     T = T.reshape(B, -1)
        #     T = T @ self.Q
        #     T = T.reshape((B,) + REST)
        #     return T

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.r}, norm={self.norm})"

ictd_2 = AngularICTD(2, True)
ictd_3 = AngularICTD(3, True)
x = torch.tensor([[0, 0, 1]], dtype=torch.float32)
z = torch.tensor([[1, 2, 3]], dtype=torch.float32)
x = ictd_2(x)
print(x)
out = torch.einsum('bi, bjk -> bijk', z, x)
print(out)
