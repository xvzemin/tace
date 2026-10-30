"""Restriction of three-dimensional STF tensors to a transverse plane."""

import torch

from eqx import co3, o2

from .basis import path_matrix
from .irreps import Irrep, Irreps
from .restriction import Restriction, restriction_matrix, restriction_scale
from .tensor_product import quarter_turn


def transform_tensor(tensor, matrix, rank):
    """Apply a rectangular matrix to every index of a flattened tensor."""
    a, b = matrix.shape[-2:]
    for k in range(rank):
        tensor = tensor.reshape(*tensor.shape[:-1], b ** (rank - k - 1), b, a**k)
        tensor = torch.einsum("...uiv,...ji->...ujv", tensor, matrix).flatten(-3)
    return tensor


class LocalFrame(torch.nn.Module):
    """Restrict Cartesian O(3) features to Cartesian O(2) irreps.

    Parameters
    ----------
    irreps : co3.Irreps or str
        Input representations in flattened ``mul_ir`` storage.
    mmax : int, optional
        Largest retained planar tensor order. Defaults to the largest rank.
    basis_change : bool, optional
        Unify the reflection action of natural and unnatural parity at
        positive orders. Defaults to True.
    method : {"direct", "rotation", "projection"}, optional
        Direct rectangular contractions, full Cartesian rotation, or
        three-dimensional transverse projection. Defaults to "direct".

    Notes
    -----
    Rotation matrices map the reference direction onto the positive y axis.
    Their z and x rows define the two planar axes. All tensor contractions
    use Cartesian indices; no Wigner-D matrices are constructed. The output
    uses flattened ``mul_ir`` storage with ``2**m`` entries per channel.
    Keeping every order makes the map invertible on the STF subspace.
    ``projection`` first constructs ``3**m`` transverse tensors. ``rotation``
    first rotates the full ``3**l`` tensor. ``direct`` avoids both intermediates.

    Examples
    --------
    >>> from eqx import co2, co3
    >>> frame = co2.LocalFrame("2x1o + 2x2e")
    >>> x = co3.Irreps("2x1o + 2x2e").randn(4, -1)
    >>> rotation = torch.eye(3).expand(4, 3, 3)
    >>> torch.testing.assert_close(frame.to_global(frame(x, rotation), rotation), x)
    """

    def __init__(self, irreps, mmax=None, basis_change=True, *, method="direct"):
        super().__init__()
        if method not in ("direct", "rotation", "projection"):
            raise ValueError("method must be 'direct', 'rotation', or 'projection'.")
        self.method = method
        self.irreps_in = co3.Irreps(irreps)
        self.lmax = max((ir.l for _, ir in self.irreps_in), default=0)
        if mmax is not None and (not isinstance(mmax, int) or mmax < 0):
            raise ValueError("mmax must be a non-negative integer.")
        self.mmax = self.lmax if mmax is None else min(mmax, self.lmax)
        self.basis_change = basis_change
        self.irreps_out = Irreps(
            o2.LocalFrame.restrict(self.irreps_in.spherical(), self.mmax)
        )
        self.input_dim, self.output_dim = self.irreps_in.dim, self.irreps_out.dim
        self.restrictions = torch.nn.ModuleDict(
            {
                str(ir.l): Restriction(ir.l)
                for _, ir in self.irreps_in
                if method == "projection"
            }
        )
        entries, offsets = [], [0] * len(self.irreps_out)
        for (mul, ir), section in zip(self.irreps_in, self.irreps_in.slices()):
            paths = []
            for m in range(min(ir.l, self.mmax) + 1):
                local_ir = Irrep(m, ir.p * (-1) ** ir.l if m == 0 else 0, ir.t)
                index = self.irreps_out.index(local_ir)
                paths.append((index, slice(offsets[index], offsets[index] + mul)))
                offsets[index] += mul
                name = f"basis_{ir.l}_{m}"
                if not hasattr(self, name):
                    self.register_buffer(
                        name,
                        self.basis_matrix(ir.l, m).to(torch.get_default_dtype()),
                        persistent=False,
                    )
                    if method == "rotation":
                        self.register_buffer(
                            f"indices_{ir.l}_{m}",
                            self.planar_indices(ir.l, m),
                            persistent=False,
                        )
            entries.append((section, mul, ir.l, ir.p != (-1) ** ir.l, tuple(paths)))
        self.entries = tuple(entries)
        if method != "projection":
            for l in sorted({ir.l for _, ir in self.irreps_in}):
                self.register_buffer(
                    f"synthesis_{l}",
                    self.synthesis_matrix(l, min(l, self.mmax)).to(
                        torch.get_default_dtype()
                    ),
                    persistent=False,
                )

    @staticmethod
    def basis_matrix(l, m):
        """Return the fixed planar convention matching spherical O(2) features."""
        rows = [l] if m == 0 else [l + m, l - m]
        change = restriction_matrix(l, m)[:, rows].T
        basis = path_matrix(m)
        return basis @ change @ basis.T

    @staticmethod
    def planar_indices(l, m):
        """Select m planar indices followed by l-m longitudinal indices."""
        indices = torch.arange(2**m, dtype=torch.long, device="cpu")
        planar = torch.zeros_like(indices)
        for k in range(m):
            planar = planar + 2 * (1 - ((indices // 2**k) % 2)) * 3**k
        return planar * 3 ** (l - m) + (3 ** (l - m) - 1) // 2

    @classmethod
    def synthesis_matrix(cls, l, mmax):
        """Return the fixed Cartesian reconstruction matrix at the y axis."""
        basis = co3.path_matrix(l)
        blocks = [
            restriction_scale(l, m)
            * cls.basis_matrix(l, m)
            @ basis.index_select(0, cls.planar_indices(l, m))
            for m in range(mmax + 1)
        ]
        return basis @ torch.cat(blocks, dim=0).T

    def rotate(self, features, rotation):
        """Rotate every Cartesian index, retaining the full input layout."""
        if features.shape[-1] != self.input_dim or rotation.shape[-2:] != (3, 3):
            raise ValueError("Incorrect feature or rotation shape.")
        shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
        values = [
            transform_tensor(
                features[..., section]
                .reshape(*features.shape[:-1], mul, 3**l)
                .expand(*shape, mul, 3**l),
                rotation.unsqueeze(-3),
                l,
            ).flatten(-2)
            for section, mul, l, _, _ in self.entries
        ]
        return torch.cat(values, -1) if values else features[..., :0]

    def to_local(self, features, rotation):
        """Extract each planar STF order in two-dimensional Cartesian storage.

        Parameters
        ----------
        features : torch.Tensor
            STF features of shape ``(..., irreps_in.dim)``.
        rotation : torch.Tensor
            Proper alignment matrices of shape ``(..., 3, 3)``.
        """
        if features.shape[-1] != self.input_dim or rotation.shape[-2:] != (3, 3):
            raise ValueError("Incorrect feature or rotation shape.")
        direction = rotation[..., 1, :].unsqueeze(-2)
        axes = rotation[..., [2, 0], :].unsqueeze(-3)
        shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
        outputs = [[] for _ in self.irreps_out]
        for section, mul, l, odd, paths in self.entries:
            tensor = (
                features[..., section]
                .reshape(*features.shape[:-1], mul, 3**l)
                .expand(*shape, mul, 3**l)
            )
            if self.method == "projection":
                values = self.restrictions[str(l)](tensor, direction)
            elif self.method == "rotation":
                tensor = transform_tensor(tensor, rotation.unsqueeze(-3), l)
            else:
                values = [tensor]
                for rank in range(l, 0, -1):
                    value = values[-1].unflatten(-1, (3 ** (rank - 1), 3))
                    values.append((value @ direction.unsqueeze(-1)).squeeze(-1))
                values.reverse()
            for m, (index, _) in enumerate(paths):
                if self.method == "rotation":
                    value = tensor.index_select(-1, getattr(self, f"indices_{l}_{m}"))
                else:
                    value = transform_tensor(values[m], axes, m)
                if self.method != "projection":
                    value = value * restriction_scale(l, m)
                value = value @ getattr(self, f"basis_{l}_{m}").T
                if odd and m and self.basis_change:
                    value = quarter_turn(value)
                outputs[index].append(value)
        return (
            torch.cat([torch.cat(parts, -2).flatten(-2) for parts in outputs], -1)
            if outputs
            else features[..., :0]
        )

    def to_global(self, features, rotation):
        """Reconstruct global STF features from the retained planar orders."""
        if features.shape[-1] != self.output_dim or rotation.shape[-2:] != (3, 3):
            raise ValueError("Incorrect feature or rotation shape.")
        direction = rotation[..., 1, :].unsqueeze(-2)
        axes = rotation[..., [2, 0], :].transpose(-1, -2).unsqueeze(-3)
        local = [
            features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
            for (mul, ir), section in zip(self.irreps_out, self.irreps_out.slices())
        ]
        outputs = []
        shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
        for _, mul, l, odd, paths in self.entries:
            values = []
            orders = l + 1 if self.method == "projection" else len(paths)
            for m in range(orders):
                if m >= len(paths):
                    values.append(features.new_zeros((*shape, mul, 3**m)))
                    continue
                index, section = paths[m]
                value = local[index][..., section, :]
                if odd and m and self.basis_change:
                    value = -quarter_turn(value)
                if self.method == "projection":
                    value = value @ getattr(self, f"basis_{l}_{m}")
                    value = transform_tensor(value, axes, m)
                values.append(value)
            if self.method == "projection":
                value = self.restrictions[str(l)].inverse(values, direction)
            else:
                value = torch.cat(values, -1) @ getattr(self, f"synthesis_{l}").T
                value = value.expand(*shape, mul, 3**l)
                value = transform_tensor(
                    value, rotation.transpose(-1, -2).unsqueeze(-3), l
                )
            outputs.append(value.flatten(-2))
        return torch.cat(outputs, -1) if outputs else features[..., :0]

    def forward(self, features, rotation):
        return self.to_local(features, rotation)

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, method={self.method}"

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for _, _, l, _, paths in self.entries:
            for m in range(len(paths)):
                name = f"basis_{l}_{m}"
                self._buffers[name] = self.basis_matrix(l, m).to(getattr(self, name))
        if self.method != "projection":
            for l in {ir.l for _, ir in self.irreps_in}:
                name = f"synthesis_{l}"
                self._buffers[name] = self.synthesis_matrix(l, min(l, self.mmax)).to(
                    getattr(self, name)
                )
        return self
