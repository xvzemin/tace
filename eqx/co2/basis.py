"""Orthonormal Cartesian bases for real circular harmonics."""

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
        Cached CPU float64 matrix of shape (2**m, 1 if m == 0 else 2).
        Columns are orthonormal.

    Notes
    -----
    The columns are the real and imaginary parts of the m-fold tensor
    product of ``(1, 1j)``. Each step prepends a Cartesian index.
    Normalization is applied once after constructing the integer entries.
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


class ChangeOfBasis(torch.nn.Module):
    """Convert circular ir_mul features and Cartesian mul_ir tensors.

    Parameters
    ----------
    irreps : Irreps or str
        Representation labels and channel multiplicities.
    inverse : bool, optional
        False embeds compact O(2) features. True extracts compact features
        and discards components outside the symmetric traceless subspace.
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
    """

    def __init__(self, irreps):
        super().__init__()
        self.irreps_in = self.irreps_out = Irreps(irreps)
        self.basis = ChangeOfBasis(irreps, inverse=True)

    def forward(self, features):
        """Apply the rectangular path matrix and its transpose."""
        if features.shape[-1] != self.basis.input_dim:
            raise ValueError("The feature dimension does not match irreps_in.")
        values = []
        for mul, m, dim, _, section in self.basis.paths:
            matrix = getattr(self.basis, f"basis_{m}")
            x = features[..., section].reshape(*features.shape[:-1], mul, dim)
            values.append(((x @ matrix) @ matrix.T).flatten(-2))
        return torch.cat(values, -1) if values else features[..., :0]

    def extra_repr(self):
        return str(self.irreps_in)
