################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import NamedTuple, Optional

import torch
from e3nn import o3

from ._layout import _Permute, wigner_indices, wigner_orders
from .irreps import Irrep, Irreps


class _FrameEntry(NamedTuple):
    global_slice: slice
    local_indices: tuple[int, ...]
    local_slices: tuple[slice, ...]
    mul: int
    degree: int
    odd: bool


class LocalFrame(torch.nn.Module):
    """Rotate global O(3) features into a local O(2) frame.

    Parameters
    ----------
    irreps : o3.Irreps or str
        Global input representation, including time parity when present.
    mmax : int, optional
        Largest local O(2) order to retain. Defaults to the largest degree in
        ``irreps``; larger values have no effect.
    reverse : bool, optional
        Display the local-to-global mapping in ``repr``. Defaults to ``False``;
        does not change the computation.
    basis_change : bool, optional
        Apply the fixed basis change to positive orders of unnatural-parity
        entries. Defaults to ``True``. If ``False``, retain the spherical
        harmonic basis; subsequent operators must account for the different
        reflection matrices of natural- and unnatural-parity inputs.
    """

    @staticmethod
    def restrict(irreps: o3.Irreps, mmax: Optional[int] = None) -> Irreps:
        """Restrict global O(3) entries to local O(2) entries.

        Parameters
        ----------
        irreps : o3.Irreps or str
            Global representation to restrict.
        mmax : int, optional
            Largest positive local order to retain. If omitted, all orders up
            to the largest global degree are retained.

        Returns
        -------
        Irreps
            Regrouped local representation. For a global entry ``(l, p, t)``,
            the order-zero reflection parity is ``p * (-1)**l`` and every
            restricted entry retains ``t``.
        """
        irreps = o3.Irreps(irreps)
        if mmax is None:
            mmax = irreps.lmax if irreps else 0
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        irrep_list = []
        for entry in irreps:
            ir, mul = entry.ir, entry.mul
            time_parity = getattr(ir, "t", 1)
            irrep_list.append((Irrep(0, ir.p * ((-1) ** ir.l), time_parity), mul))
            irrep_list.extend(
                (Irrep(order, 0, time_parity), mul)
                for order in range(1, min(ir.l, mmax) + 1)
            )
        return Irreps(irrep_list).regroup()

    def __init__(
        self,
        irreps: o3.Irreps,
        mmax: Optional[int] = None,
        reverse: bool = False,
        basis_change: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps)
        self.lmax = self.irreps_in.lmax if self.irreps_in else 0
        if mmax is None:
            mmax = self.lmax
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        if not isinstance(reverse, bool):
            raise TypeError("reverse must be a boolean.")
        self.mmax = min(mmax, self.lmax)
        self.reverse = reverse
        self.basis_change = basis_change
        self.irreps_out = self.restrict(self.irreps_in, self.mmax)
        self.global_irreps = self.irreps_in
        self.local_irreps = self.irreps_out
        self.input_dim = self.irreps_in.dim
        self.output_dim = self.irreps_out.dim

        global_slices = self.irreps_in.slices()
        local_indices = {ir: index for index, (ir, _) in enumerate(self.irreps_out)}
        local_offsets = [0] * len(self.irreps_out)
        entries = []
        for global_slice, global_entry in zip(global_slices, self.irreps_in):
            ir, mul = global_entry.ir, global_entry.mul
            retained_mmax = min(ir.l, self.mmax)
            time_parity = getattr(ir, "t", 1)
            local_irrep_list = [Irrep(0, ir.p * ((-1) ** ir.l), time_parity)]
            local_irrep_list.extend(
                Irrep(order, 0, time_parity) for order in range(1, retained_mmax + 1)
            )
            entry_local_indices = tuple(
                local_indices[local_ir] for local_ir in local_irrep_list
            )
            entry_local_slices = []
            for index in entry_local_indices:
                start = local_offsets[index]
                entry_local_slices.append(slice(start, start + mul))
                local_offsets[index] += mul
            entries.append(
                _FrameEntry(
                    global_slice,
                    entry_local_indices,
                    tuple(entry_local_slices),
                    mul,
                    ir.l,
                    ir.p * ((-1) ** ir.l) == -1,
                )
            )
        self._entries = tuple(entries)
        self._degrees = tuple(sorted({entry.degree for entry in entries}))
        self._multiplicities = tuple(
            sum(entry.mul for entry in entries if entry.degree == degree)
            for degree in self._degrees
        )
        self._global_sizes = tuple(
            (2 * degree + 1) * mul
            for degree, mul in zip(self._degrees, self._multiplicities)
        )
        self._local_sizes = tuple(
            (2 * min(degree, self.mmax) + 1) * mul
            for degree, mul in zip(self._degrees, self._multiplicities)
        )
        self._matrix_sizes = tuple(
            (2 * degree + 1) * (2 * min(degree, self.mmax) + 1)
            for degree in self._degrees
        )
        global_index = []
        local_index = [0] * self.output_dim
        local_sign = [1.0] * self.output_dim
        dense_rows, dense_columns, row_strides, packed_index = [], [], [], []
        local_slices = self.irreps_out.slices()
        offset = 0
        for degree, mul in zip(self._degrees, self._multiplicities):
            selected = [entry for entry in entries if entry.degree == degree]
            dim = 2 * degree + 1
            orders = [0] + [
                s * m for m in range(1, min(degree, self.mmax) + 1) for s in (1, -1)
            ]
            for m in range(dim):
                for entry in selected:
                    global_index.extend(
                        range(
                            entry.global_slice.start + m * entry.mul,
                            entry.global_slice.start + (m + 1) * entry.mul,
                        )
                    )
            channel = 0
            for entry in selected:
                for m, (index, local_slice) in enumerate(
                    zip(entry.local_indices, entry.local_slices)
                ):
                    for component in range(1 if m == 0 else 2):
                        swapped = self.basis_change and entry.odd and m > 0
                        row = (
                            0
                            if m == 0
                            else 2 * m - 1 + (1 - component if swapped else component)
                        )
                        start = (
                            local_slices[index].start
                            + component * self.irreps_out[index].mul
                            + local_slice.start
                        )
                        for u in range(entry.mul):
                            local_index[start + u] = offset + row * mul + channel + u
                            local_sign[start + u] = (
                                -1.0 if swapped and component == 0 else 1.0
                            )
                channel += entry.mul
            rows = wigner_indices(degree, min(degree, self.mmax), self.lmax)
            for row, (m, dense_row) in enumerate(zip(orders, rows)):
                for column in range(dim):
                    dense_rows.append(dense_row)
                    dense_columns.append(degree**2 + column)
                    row_strides.append(row)
                    packed_index.append(
                        degree * (4 * degree**2 - 1) // 3 + (degree + m) * dim + column
                    )
            offset += len(orders) * mul
        self._global_identity = global_index == list(range(self.input_dim))
        self._local_identity = local_index == list(range(self.output_dim))
        self._has_signs = -1.0 in local_sign
        global_dim = (self.lmax + 1) ** 2
        local_dim = global_dim - (self.lmax - self.mmax) * (self.lmax - self.mmax + 1)
        for name, indices in (
            ("_global_index", global_index),
            (
                "_global_inverse",
                sorted(range(self.input_dim), key=global_index.__getitem__),
            ),
            ("_local_index", local_index),
            (
                "_local_inverse",
                sorted(range(self.output_dim), key=local_index.__getitem__),
            ),
            ("_dense_rows", dense_rows),
            ("_dense_columns", dense_columns),
            ("_row_strides", row_strides),
            ("_packed_index", packed_index),
            (
                "_dense_index",
                [row * global_dim + col for row, col in zip(dense_rows, dense_columns)],
            ),
            (
                "_dense_inverse_index",
                [col * local_dim + row for row, col in zip(dense_rows, dense_columns)],
            ),
        ):
            self.register_buffer(
                name, torch.tensor(indices, dtype=torch.long), persistent=False
            )
        self.register_buffer("_local_sign", torch.tensor(local_sign), persistent=False)

    def _rotation_matrices(self, wigner: torch.Tensor, inverse: bool = False):
        """Extract degree matrices from dense or packed Wigner storage."""
        if wigner.ndim == 2:
            degree = self.lmax + 1
            if wigner.size(-1) < degree * (4 * degree**2 - 1) // 3:
                raise ValueError(
                    "Wigner packed dimension must cover every O(3) degree."
                )
            matrices = wigner.index_select(1, self._packed_index)
            wigner_mmax = self.lmax
        else:
            global_dim, local_dim = (
                wigner.shape[-2:] if inverse else wigner.shape[-2:][::-1]
            )
            lmax, wigner_mmax = wigner_orders(
                global_dim, local_dim, lmax=self.lmax, mmax=self.mmax
            )
            if lmax == self.lmax and wigner_mmax == self.mmax:
                indices = self._dense_inverse_index if inverse else self._dense_index
            else:
                rows = self._dense_rows + (lmax - self.lmax) * self._row_strides
                indices = (
                    self._dense_columns * local_dim + rows
                    if inverse
                    else rows * global_dim + self._dense_columns
                )
            matrices = wigner.flatten(1).index_select(1, indices)
        outputs = []
        for degree, matrix in zip(
            self._degrees, matrices.split(self._matrix_sizes, dim=1)
        ):
            retained = 2 * min(degree, self.mmax) + 1
            matrix = matrix.view(wigner.size(0), retained, 2 * degree + 1)
            if inverse:
                matrix = matrix.transpose(1, 2)
                source = 2 * min(degree, wigner_mmax) + 1
                if source != retained:
                    matrix = matrix * math.sqrt(source / retained)
            outputs.append(matrix)
        return outputs

    def __repr__(self) -> str:
        irreps_in, irreps_out = (
            (self.local_irreps, self.global_irreps)
            if self.reverse
            else (self.global_irreps, self.local_irreps)
        )
        options = f"mmax={self.mmax}"
        if not self.basis_change:
            options += ", basis_change=False"
        return f"{self.__class__.__name__}({irreps_in} -> {irreps_out})({options})"

    def to_local(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate global features into their local frame.

        Parameters
        ----------
        features : torch.Tensor
            Global features with shape ``(batch, ..., irreps_in.dim)`` in
            flattened ``ir_mul`` order.
        wigner : torch.Tensor
            Global-to-local matrices with shape
            ``(batch, local_wigner_dim, (L + 1)**2)``. The global degree ``L``
            and retained local orders are inferred from these dimensions and
            must cover the representation used by this module. Alternatively,
            full degree matrices packed as ``(batch, sum((2*l+1)**2))``.

        Returns
        -------
        torch.Tensor
            Local features with shape ``(batch, ..., irreps_out.dim)``.
        """
        if features.ndim < 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                "LocalFrame input trailing dimension must be "
                f"{self.input_dim}, got {tuple(features.shape)}."
            )
        if wigner.ndim not in (2, 3) or features.size(0) != wigner.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        matrices = self._rotation_matrices(wigner)
        if not self._global_identity:
            features = _Permute.apply(
                features, self._global_index, self._global_inverse
            )
        outputs = []
        for degree, mul, matrix, values in zip(
            self._degrees,
            self._multiplicities,
            matrices,
            features.split(self._global_sizes, dim=-1),
        ):
            values = values.view(*features.shape[:-1], 2 * degree + 1, mul)
            values = (
                torch.bmm(matrix, values)
                if features.ndim == 2
                else torch.einsum("bij,b...jk->b...ik", matrix, values)
            )
            outputs.append(values.flatten(-2))
        if not outputs:
            return features[..., :0]
        features = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if not self._local_identity:
            features = _Permute.apply(features, self._local_index, self._local_inverse)
        return features * self._local_sign if self._has_signs else features

    def forward(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`to_local`."""
        return self.to_local(features, wigner)

    def to_global(
        self,
        features: torch.Tensor,
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate local features back into the global frame.

        Parameters
        ----------
        features : torch.Tensor
            Local features with shape ``(batch, ..., irreps_out.dim)``.
        wigner_inv : torch.Tensor
            Local-to-global matrices with shape
            ``(batch, (L + 1)**2, local_wigner_dim)``. The layout is inferred
            from these dimensions. Additional degrees are ignored; additional
            local orders are accepted with the corresponding inverse rescaling.
            Alternatively, pass the same packed degree matrices as to
            :meth:`to_local`; their transpose and inverse scale are applied here.

        Returns
        -------
        torch.Tensor
            Global features with shape ``(batch, ..., irreps_in.dim)`` in
            flattened ``ir_mul`` order.
        """
        if features.ndim < 2 or features.size(-1) != self.output_dim:
            raise ValueError(
                "LocalFrame input trailing dimension must be "
                f"{self.output_dim}, got {tuple(features.shape)}."
            )
        if wigner_inv.ndim not in (2, 3) or features.size(0) != wigner_inv.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        matrices = self._rotation_matrices(wigner_inv, inverse=True)
        if self._has_signs:
            features = features * self._local_sign
        if not self._local_identity:
            features = _Permute.apply(features, self._local_inverse, self._local_index)
        outputs = []
        for degree, mul, matrix, values in zip(
            self._degrees,
            self._multiplicities,
            matrices,
            features.split(self._local_sizes, dim=-1),
        ):
            retained = 2 * min(degree, self.mmax) + 1
            values = values.view(*features.shape[:-1], retained, mul)
            values = (
                torch.bmm(matrix, values)
                if features.ndim == 2
                else torch.einsum("bij,b...jk->b...ik", matrix, values)
            )
            outputs.append(values.flatten(-2))
        if not outputs:
            return features[..., :0]
        features = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return (
            features
            if self._global_identity
            else _Permute.apply(features, self._global_inverse, self._global_index)
        )
