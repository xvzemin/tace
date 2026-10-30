"""Orthonormal Cartesian bases for real circular harmonics."""

from collections import defaultdict
from functools import lru_cache

import torch

from .irreps import Irreps


@lru_cache(maxsize=None)
def path_matrix(m):
    """Return the orthonormal circular-to-Cartesian path matrix.

    Parameters
    ----------
    m : int
        Non-negative tensor order.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 matrix of shape ``(2**m, 1 if m == 0 else 2)``
        with orthonormal columns. Do not modify it in place.

    Notes
    -----
    The columns are the real and imaginary parts of the m-fold tensor
    product of ``(1, 1j)``, normalized to unit norm.

    Examples
    --------
    >>> from eqx import co2
    >>> matrix = co2.path_matrix(3)
    >>> matrix.shape
    torch.Size([8, 2])
    >>> x = torch.randn(4, 2, dtype=matrix.dtype, device=matrix.device)
    >>> torch.testing.assert_close((x @ matrix.T) @ matrix, x)
    """
    if not isinstance(m, int) or m < 0:
        raise ValueError("The order must be a non-negative integer.")
    if m == 0:
        return torch.ones(1, 1, dtype=torch.float64, device="cpu")
    matrix = torch.eye(2, dtype=torch.float64, device="cpu")
    for _ in range(1, m):
        real, imag = matrix.unbind(-1)
        matrix = torch.stack(
            (torch.cat((real, -imag)), torch.cat((imag, real))), dim=-1
        )
    return matrix * 2.0 ** ((1 - m) / 2)


@lru_cache(maxsize=None)
def projector_matrix(m):
    """Return the analytic planar STF projector of shape (2**m, 2**m).

    Parameters
    ----------
    m : int
        Non-negative Cartesian rank.

    Returns
    -------
    torch.Tensor
        Cached CPU float64 orthogonal projector. Do not modify it in place.

    Examples
    --------
    >>> from eqx import co2
    >>> matrix = co2.projector_matrix(2)
    >>> matrix.shape
    torch.Size([4, 4])
    >>> torch.testing.assert_close(matrix @ matrix, matrix)
    """
    if not isinstance(m, int) or m < 0:
        raise ValueError("The order must be a non-negative integer.")
    if m == 0:
        return torch.ones(1, 1, dtype=torch.float64, device="cpu")
    # Each second-axis index contributes a factor i to (1, i) tensor powers.
    counts = torch.zeros(1, dtype=torch.long, device="cpu")
    for _ in range(m):
        counts = torch.cat((counts, counts + 1))
    cosine = torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=torch.float64, device="cpu")
    return cosine[(counts[:, None] - counts[None, :]) % 4] * 2.0 ** (1 - m)


class ChangeOfBasis(torch.nn.Module):
    """Convert spherical O(2) features and Cartesian tensors.

    Parameters
    ----------
    irreps : Irreps or str
        Representation labels and channel multiplicities.
    inverse : bool, optional
        False embeds compact O(2) features. True extracts compact features
        and discards components outside the symmetric traceless subspace.

    Notes
    -----
    Spherical O(2) features use ``ir_mul`` layout; Cartesian tensors use
    ``mul_ir`` layout.

    Examples
    --------
    >>> from eqx import co2
    >>> to_cartesian = co2.ChangeOfBasis("2x2m")
    >>> to_spherical = co2.ChangeOfBasis("2x2m", inverse=True)
    >>> x = torch.randn(5, 4)
    >>> tensor = to_cartesian(x)
    >>> tensor.shape
    torch.Size([5, 8])
    >>> torch.testing.assert_close(to_spherical(tensor), x)
    """

    def __init__(self, irreps, inverse=False):
        super().__init__()
        cartesian = Irreps(irreps)
        self.inverse = inverse
        self.irreps_in = cartesian if inverse else cartesian.circular()
        self.irreps_out = cartesian.circular() if inverse else cartesian
        self.cartesian = cartesian
        self.slices_in = self.irreps_in.slices()
        self.input_dim = self.irreps_in.dim
        self.paths = tuple(
            (mul, ir.m, ir.dim, ir.circular_dim, section)
            for (mul, ir), section in zip(cartesian, self.slices_in)
        )
        for m in sorted({ir.m for _, ir in cartesian}):
            self.register_buffer(
                f"basis_{m}",
                path_matrix(m).to(torch.get_default_dtype()).clone(),
                persistent=False,
            )

    def forward(self, features):
        """Transform features with trailing size irreps_in.dim."""
        if features.shape[-1] != self.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        values = []
        for mul, m, dim, circular_dim, section in self.paths:
            matrix = getattr(self, f"basis_{m}")
            x = features[..., section]
            if self.inverse:
                x = x.reshape(*features.shape[:-1], mul, dim) @ matrix
                x = x.transpose(-1, -2)
            else:
                x = x.reshape(*features.shape[:-1], circular_dim, mul)
                x = x.transpose(-1, -2) @ matrix.T
            values.append(x.flatten(-2))
        return torch.cat(values, -1) if values else features[..., :0]

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, inverse={self.inverse}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for m in {ir.m for _, ir in self.cartesian}:
            name = f"basis_{m}"
            self._buffers[name] = path_matrix(m).to(getattr(self, name)).clone()
        return self


class Projector(torch.nn.Module):
    """Project Cartesian tensors onto their symmetric traceless subspaces.

    Parameters
    ----------
    irreps : Irreps or str
        Cartesian representation layout.
    method : {"matrix", "factorized"}, optional
        Analytic Cartesian projection or factored path-matrix evaluation.
        Defaults to "matrix".

    Examples
    --------
    >>> from eqx import co2
    >>> project = co2.Projector("2m")
    >>> x = torch.arange(4.0).reshape(2, 2)
    >>> result = project(x.flatten()).reshape(2, 2)
    >>> expected = (x + x.T) / 2 - torch.eye(2) * x.trace() / 2
    >>> torch.testing.assert_close(result, expected)
    """

    def __init__(self, irreps, *, method="matrix"):
        super().__init__()
        if method not in ("matrix", "factorized"):
            raise ValueError("method must be 'matrix' or 'factorized'.")
        self.method = method
        self.irreps_in = self.irreps_out = Irreps(irreps)
        self.input_dim = self.irreps_in.dim
        groups = defaultdict(list)
        for i, (mul, ir) in enumerate(self.irreps_in):
            if ir.m > 1:
                groups[ir.m].append((i, mul))
        self.groups = tuple(
            (m, tuple(entries)) for m, entries in sorted(groups.items())
        )
        for m, _ in self.groups:
            self.register_buffer(
                f"matrix_{m}",
                self.projection_matrix(m).to(torch.get_default_dtype()).clone(),
                persistent=False,
            )

    def projection_matrix(self, m):
        """Return fixed coefficients for the selected projection method."""
        return projector_matrix(m) if self.method == "matrix" else path_matrix(m)

    def forward(self, features):
        """Project equal-order Cartesian blocks together."""
        if features.shape[-1] != self.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        values = [
            features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
            for (mul, ir), section in zip(self.irreps_in, self.irreps_in.slices())
        ]
        for m, entries in self.groups:
            value = (
                values[entries[0][0]]
                if len(entries) == 1
                else torch.cat([values[i] for i, _ in entries], dim=-2)
            )
            matrix = getattr(self, f"matrix_{m}")
            value = value @ matrix
            if self.method == "factorized":
                value = value @ matrix.T
            for (i, _), part in zip(
                entries, value.split([mul for _, mul in entries], -2)
            ):
                values[i] = part
        return (
            torch.cat([x.flatten(-2) for x in values], -1)
            if values
            else features[..., :0]
        )

    def extra_repr(self):
        return f"{self.irreps_in}, method={self.method}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for m, _ in self.groups:
            name = f"matrix_{m}"
            self._buffers[name] = (
                self.projection_matrix(m).to(getattr(self, name)).clone()
            )
        return self
