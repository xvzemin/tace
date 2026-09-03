################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from itertools import combinations
from typing import Any, Dict, List, Union

import torch
from e3nn import get_optimization_defaults
from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import _spherical_harmonics
from e3nn.util.jit import compile_mode

from .time_reversal import make_irrep
from .ictd import ICTD
from .utils import expand_dims_to


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


def delta_tensor(i: int, j: int, ndim: int, device=None, dtype=None) -> torch.Tensor:
    delta = torch.eye(3, device=device, dtype=dtype)
    for _ in range(ndim - 2):
        delta = delta.unsqueeze(0)
    perm = list(range(ndim))
    perm[i], perm[-2] = perm[-2], perm[i]
    perm[j], perm[-1] = perm[-1], perm[j]
    delta = delta.permute(*perm)
    return delta


def symmetric_outer_product(v: torch.Tensor, n: int, norm: bool = True) -> torch.Tensor:
    out = torch.ones_like(v[..., 0])
    for _ in range(n):
        out = out[..., None] * expand_dims_to(v, out.ndim + 1, dim=v.ndim - 1)
    if norm:
        out = out * _norm(n)
    return out


def subtract_traces(T: torch.Tensor, n: int) -> torch.Tensor:
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
def symmetric_traceless_outer_product(
    v: torch.Tensor, n: int, norm: bool = True
) -> torch.Tensor:
    T = symmetric_outer_product(v, n, norm)
    return subtract_traces(T, n)



class CartesianHarmonics(torch.nn.Module):
    def __init__(
        self,
        lmax: int,
        norm: bool = True,
        traceless: bool = True,
        *,
        normalize: bool = False,
        return_dict: bool = True,
        eps: float = 1e-12,  # TODO
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.norm = norm
        self.traceless = traceless
        self.normalize = normalize
        self.return_dict = return_dict
        self.eps = eps
        for l in range(self.lmax + 1):
            _, _, cart2sph, sph2cart = ICTD(l, l, decomposition=False)
            self.register_buffer(
                f"C{l}",
                cart2sph[0].to(torch.get_default_dtype()),
                persistent=False,
            )
            self.register_buffer(
                f"CT{l}",
                sph2cart[0].to(torch.get_default_dtype()),
                persistent=False,
            )

    def forward(self, v: torch.Tensor) -> Union[Dict[int, torch.Tensor], torch.Tensor]:
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=-1, eps=self.eps)

        T = torch.ones_like(v[..., 0])
        batch = T.shape[0]
        edge_attrs: Dict[int, torch.Tensor] = {0: T.unsqueeze(1)}
        flat_attrs: List[torch.Tensor] = [T.reshape(batch, -1)]

        for l in range(1, self.lmax + 1):
            T = T[..., None] * expand_dims_to(v, T.ndim + 1, dim=v.ndim - 1)
            field = T.reshape(batch, -1)
            if self.norm:
                field = field * _norm(l)
            if self.traceless and batch != 0:
                field = torch.matmul(torch.matmul(field, self.C(l)), self.CT(l))
            flat_attrs.append(field)
            if self.return_dict:
                edge_attrs[l] = field.reshape((batch,) + T.shape[1:]).unsqueeze(-1)

        if self.return_dict:
            return edge_attrs
        return torch.cat(flat_attrs, dim=-1)

    def C(self, l: int) -> torch.Tensor:
        return getattr(self, f"C{l}")

    def CT(self, l: int) -> torch.Tensor:
        return getattr(self, f"CT{l}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r={self.lmax}, norm={self.norm}, "
            f"traceless={self.traceless}, normalize={self.normalize}, "
            f"return_dict={self.return_dict})"
        )


class SphericalHarmonics(torch.nn.Module):
    """
    Copy from e3nn for torch.save
    """

    normalize: bool
    normalization: str
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str

    def __init__(
        self,
        irreps_out: Union[int, List[int], str, Irreps],
        normalize: bool,
        normalization: str = "integral",
        irreps_in: Any = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.normalization = normalization
        assert normalization in ["integral", "component", "norm"]

        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for _, ir in irreps_out:
                if ir.l % 2 == 1:
                    irreps_in = Irreps(
                        [(1, make_irrep(1, ir.p, getattr(ir, "t", 1)))]
                    )
                    break
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if (
            len(irreps_in) != 1
            or irreps_in[0].mul != 1
            or irreps_in[0].ir.l != 1
        ):
            raise ValueError(
                "irreps_in for SphericalHarmonics must contain exactly one "
                f"l=1 irrep, not `{irreps_in}`"
            )
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p  # pylint: disable=no-member
        input_t = getattr(irreps_in[0].ir, "t", 1)

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, ir in irreps_out:
                if ir.p != input_p**ir.l:
                    raise ValueError(
                        f"irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {ir.l} with parity "
                        f"p = {ir.p}, which is inconsistent with the input parity {input_p} — the output parity should have been "
                        f"p = {input_p**ir.l}"
                    )
                if getattr(ir, "t", 1) != input_t**ir.l:
                    raise ValueError(
                        f"irreps_out `{irreps_out}` passed to SphericalHarmonics "
                        f"asked for an output of l = {ir.l} with time-reversal "
                        f"parity t = {ir.t}, which is inconsistent with the "
                        f"input time-reversal parity {input_t}"
                    )
                ls.extend([ir.l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        irreps_out = Irreps(
            [(1, make_irrep(l, input_p**l, input_t**l)) for l in ls]
        ).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f"spherical_harmonics({ls})"

        _lmax = 11
        if self._lmax > _lmax:
            raise NotImplementedError(
                f"spherical_harmonics maximum l implemented is {_lmax}, send us an email to ask for more"
            )

        # if get_optimization_defaults()["jit_mode"] == "script":
        #     self.sph_func = torch.jit.script(_spherical_harmonics)
        # elif get_optimization_defaults()["jit_mode"] == "compile":
        #     self.sph_func = torch.compile(_spherical_harmonics, fullgraph=True)
        # else:
        #     self.sph_func = _spherical_harmonics
        self.sph_func = _spherical_harmonics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # - PROFILER - with torch.autograd.profiler.record_function(self._prof_str):
        if self.normalize:
            x = torch.nn.functional.normalize(
                x, dim=-1
            )  # forward 0's instead of nan for zero-radius

        sh = self.sph_func(self._lmax, x[..., 0], x[..., 1], x[..., 2])

        if not self._is_range_lmax:
            sh = torch.cat(
                [sh[..., l * l : (l + 1) * (l + 1)] for l in self._ls_list], dim=-1
            )

        if self.normalization == "integral":
            sh.div_(math.sqrt(4 * math.pi))
        elif self.normalization == "norm":
            sh.div_(
                torch.cat(
                    [
                        math.sqrt(2 * l + 1)
                        * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
                        for l in self._ls_list
                    ]
                )
            )

        return sh


class SolidHarmonics(SphericalHarmonics):
    def __init__(
        self,
        irreps_out: Union[int, List[int], str, Irreps],
        normalization: str = "integral",
        irreps_in: Any = None,
    ) -> None:
        super().__init__(
            irreps_out,
            normalize=False,
            normalization=normalization,
            irreps_in=irreps_in,
        )
