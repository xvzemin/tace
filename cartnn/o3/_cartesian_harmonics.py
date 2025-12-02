################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import List, Union, Any, Optional
from itertools import combinations


import torch
from torch import Tensor


from cartnn.util.jit import compile_mode
from ._irreps import Irreps
from ._ictd import ICTD


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


def expand_dims_to(T: Tensor, n_dim: int, dim: int = -1) -> Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


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

# numerical
def symmetric_traceless_outer_product(v: Tensor, n: int, norm: bool = True) -> Tensor:
    T = symmetric_outer_product(v, n, norm)
    return subtract_traces(T, n)


@compile_mode("script")
class CartesianHarmonics(torch.nn.Module):
    norm: bool
    traceless: bool 
    normalize: bool
    normalization: Optional[str]
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str
    _slice_start_list: List[int]
    _slice_stop_list: List[int]
    eps: float

    def __init__(
        self,
        irreps_out: Union[int, List[int], str, Irreps],
        normalize: bool,
        normalization: Optional[str] = None, # no practical use, just for compatibility with e3nn.
        irreps_in: Any = None,
        norm: bool = True,
        traceless: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps("1e")
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps("1x1o"), Irreps("1x1e")):
            raise ValueError(
                f"irreps_in for SphericalHarmonics must be either a vector (`1x1o`) or a pseudovector (`1x1e`), "
                f"not `{irreps_in}`"
            )
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p**l:
                    raise ValueError(
                        f"irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {l} with parity "
                        f"p = {p}, which is inconsistent with the input parity {input_p} — the output parity should have been "
                        f"p = {input_p**l}"
                    )
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        _slice_start_list = []
        _slice_stop_list = []
        start = 0
        for l in ls:
            stop = start + 3**l
            _slice_start_list.append(start)
            _slice_stop_list.append(stop)
            start = stop

        irreps_out = Irreps([(1, (l, input_p**l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f"cartesian_harmonics({ls})"
        self.normalize = normalize
        self.normalization = normalization
        self.norm = norm
        self.traceless = traceless
        self._slice_start_list = _slice_start_list
        self._slice_stop_list = _slice_stop_list
        self.eps = eps
    
        for l in range(self._lmax+1):
            PS, DS, CS, SS = ICTD(l, l)
            self.register_buffer(f"D{l}", DS[0].to(torch.get_default_dtype()))
            del PS, DS, CS, SS
        

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=-1, eps=self.eps)
        T = torch.ones_like(v[..., 0])
        B = T.size(0)
        edge_attrs: List[Tensor] = []
        edge_attrs.append(T.view(B, -1))

        for l in range(1, self._lmax+1):
            T = T[..., None] * expand_dims_to(v, T.ndim + 1, dim=v.ndim - 1)
            edge_attrs.append(T.view(B, -1))

        for l in range(1, self._lmax+1):
            T = edge_attrs[l]
            if self.norm:
                T = T * _norm(l)
            if self.traceless:
                if B != 0:
                    T = T @ self.D(l).to(v.dtype)
            edge_attrs[l] = T

        ch = torch.cat(edge_attrs, dim=-1)
        if not self._is_range_lmax:
            ch = torch.cat(
                [
                    ch[..., start:stop] 
                    for start, stop in zip(self._slice_start_list, self._slice_stop_list)
                ], 
                dim=-1
            )
        return ch
    
    def D(self, l: int):
        return dict(self.named_buffers())[f"D{l}"]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(irreps_out={self.irreps_out}, normalize={self.normalize}, norm={self.norm}, traceless={self.traceless})"
