################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Any, Dict, List, Union

import torch
from e3nn.o3._irreps import Irreps
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from equivariantx.co3 import path_matrix

from .time_reversal import make_irrep
from .utils import expand_dims_to


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
            self.register_buffer(f"C{l}", path_matrix(l).clone(), persistent=False)

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
                field = field * (math.comb(2 * l, l) / 2**l)
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
        return self.C(l).T

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r={self.lmax}, norm={self.norm}, "
            f"traceless={self.traceless}, normalize={self.normalize}, "
            f"return_dict={self.return_dict})"
        )


class SphericalHarmonics(torch.nn.Module):
    """Real spherical harmonics of vectors with spatial and time parity.

    Parameters
    ----------
    irreps_out : int, sequence of int or Irreps
        Output degrees or irreducible representations.
    normalize : bool
        Normalize input vectors to unit length.
    normalization : {"integral", "component", "norm"}
        Output normalization.
    irreps_in : Irreps, optional
        Representation of the input vector.

    Examples
    --------
    >>> SphericalHarmonics([0, 1], normalize=True)(torch.ones(2, 3)).shape
    torch.Size([2, 4])
    """

    normalize: bool
    normalization: str
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool

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
                    irreps_in = Irreps([(1, make_irrep(1, ir.p, getattr(ir, "t", 1)))])
                    break
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if len(irreps_in) != 1 or irreps_in[0].mul != 1 or irreps_in[0].ir.l != 1:
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

        _lmax = 11
        if self._lmax > _lmax:
            raise NotImplementedError(
                f"Spherical harmonics support degrees up to {_lmax}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = torch.nn.functional.normalize(
                x, dim=-1
            )  # forward 0's instead of nan for zero-radius

        sh = _spherical_harmonics(self._lmax, x[..., 0], x[..., 1], x[..., 2])

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
