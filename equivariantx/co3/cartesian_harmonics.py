"""Cartesian harmonics from symmetric traceless powers of vectors."""

import math

import torch
from e3nn import o3

from .basis import path_matrix
from .irreps import Irrep, Irreps


class CartesianHarmonics(torch.nn.Module):
    """Evaluate real Cartesian harmonics.

    Parameters
    ----------
    irreps_out : int, sequence of int or Irreps
        Requested angular degrees or representations. An integer selects
        that degree, not all preceding degrees.
    normalize : bool
        Normalize input vectors before evaluating the polynomials.
    normalization : {"integral", "component", "norm"}, optional
        Normalization in orthonormal irreducible coordinates.
    irreps_in : Irreps or str, optional
        A single vector or axial-vector representation. Defaults to 1o.
    """

    def __init__(self, irreps_out, normalize, normalization="integral", irreps_in=None):
        super().__init__()
        if normalization not in ("integral", "component", "norm"):
            raise ValueError("normalization must be 'integral', 'component' or 'norm'.")
        labels = (
            Irreps(irreps_out)
            if isinstance(irreps_out, (str, Irreps, o3.Irreps))
            else None
        )
        if irreps_in is None and labels is not None:
            irreps_in = next(
                ([(1, tuple([1, *ir[1:]]))] for _, ir in labels if ir.l % 2), "1o"
            )
        self.irreps_in = Irreps("1o" if irreps_in is None else irreps_in)
        if (
            len(self.irreps_in) != 1
            or self.irreps_in[0].mul != 1
            or self.irreps_in[0].ir.l != 1
        ):
            raise ValueError("irreps_in must contain exactly one degree-one irrep.")
        ir = self.irreps_in[0].ir
        degrees = (
            ([irreps_out] if isinstance(irreps_out, int) else list(irreps_out))
            if labels is None
            else labels.ls
        )
        entries = []
        for l in degrees:
            label = (l, *(parity**l for parity in ir[1:]))
            entries.append((1, Irrep(label)))
        self.irreps_out = Irreps(entries).simplify()
        if labels is not None and labels != self.irreps_out:
            if labels.simplify() != self.irreps_out:
                raise ValueError(
                    "Output parities must be the corresponding powers of input parities."
                )
            self.irreps_out = labels
        self.normalize = normalize
        self.normalization = normalization
        self.lmax = max(degrees, default=0)
        for l in sorted(set(degrees)):
            matrix = path_matrix(l)
            # ||STF(n**l)||^2 = l! / (2l-1)!! on the unit sphere.
            scale = math.sqrt(math.comb(2 * l, l) / 2**l)
            if normalization == "component":
                scale *= math.sqrt(2 * l + 1)
            elif normalization == "integral":
                scale *= math.sqrt((2 * l + 1) / (4 * math.pi))
            self.register_buffer(
                f"basis_{l}",
                matrix.to(torch.get_default_dtype()).clone(),
                persistent=False,
            )
            self.register_buffer(
                f"scaled_basis_{l}",
                (scale * matrix.T).to(torch.get_default_dtype()),
                persistent=False,
            )

    def forward(self, vectors):
        """Return (..., irreps_out.dim) harmonics from (..., 3) vectors."""
        if vectors.shape[-1] != 3:
            raise ValueError("Input vectors must have three Cartesian entries.")
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        powers = [torch.ones_like(vectors[..., :1])]
        for _ in range(self.lmax):
            powers.append(
                (powers[-1].unsqueeze(-1) * vectors.unsqueeze(-2)).flatten(-2)
            )
        values = []
        for mul, ir in self.irreps_out:
            x = (powers[ir.l] @ getattr(self, f"basis_{ir.l}")) @ getattr(
                self, f"scaled_basis_{ir.l}"
            )
            values.extend([x] * mul)
        return torch.cat(values, dim=-1) if values else vectors[..., :0]

    def extra_repr(self):
        return f"{self.irreps_out}, normalize={self.normalize}, normalization={self.normalization}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for l in {ir.l for _, ir in self.irreps_out}:
            matrix = path_matrix(l)
            scale = math.sqrt(math.comb(2 * l, l) / 2**l)
            if self.normalization == "component":
                scale *= math.sqrt(2 * l + 1)
            elif self.normalization == "integral":
                scale *= math.sqrt((2 * l + 1) / (4 * math.pi))
            for name, value in (
                (f"basis_{l}", matrix),
                (f"scaled_basis_{l}", scale * matrix.T),
            ):
                self._buffers[name] = value.to(self._buffers[name]).clone()
        return self
