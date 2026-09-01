################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import NamedTuple, Optional

import torch
from e3nn import o3

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
    irreps : O(3) irreps-like
        Global input representation. Every entry is stored in flattened
        ``ir_mul`` order.
    lmax : int
        Maximum global degree covered by the supplied Wigner matrices. It must
        be at least the largest degree in ``irreps``.
    mmax : int, optional
        Largest local O(2) order to retain. Defaults to ``lmax``.

    Notes
    -----
    The first tensor dimension is the rotation batch. Additional leading
    dimensions, such as a source/target axis, are preserved. The local output
    representation is available as :attr:`irreps_out`.
    """

    @staticmethod
    def restrict(irreps: o3.Irreps, mmax: Optional[int] = None) -> Irreps:
        """Restrict global O(3) entries to local O(2) entries.

        Parameters
        ----------
        irreps : O(3) irreps-like
            Global representation to restrict.
        mmax : int, optional
            Largest positive local order to retain. If omitted, all orders up
            to the largest global degree are retained.

        Returns
        -------
        Irreps
            Regrouped local representation. The order-zero parity for a
            global entry ``(l, p)`` is ``p * (-1)**l``.
        """
        irreps = o3.Irreps(irreps)
        if mmax is None:
            mmax = irreps.lmax
        if not isinstance(mmax, int) or isinstance(mmax, bool):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        irrep_list = []
        for entry in irreps:
            ir, mul = entry.ir, entry.mul
            irrep_list.append((Irrep(0, ir.p * ((-1) ** ir.l)), mul))
            irrep_list.extend(
                (Irrep(order, 0), mul) for order in range(1, min(ir.l, mmax) + 1)
            )
        return Irreps(irrep_list).regroup()

    def __init__(
        self,
        irreps: o3.Irreps,
        lmax: int,
        mmax: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps)
        if not isinstance(lmax, int) or isinstance(lmax, bool):
            raise TypeError("lmax must be an integer.")
        if self.irreps_in.lmax > lmax:
            raise ValueError("lmax must cover every O(3) irrep.")
        if mmax is None:
            mmax = lmax
        if not isinstance(mmax, int) or isinstance(mmax, bool):
            raise TypeError("mmax must be an integer.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")
        self.lmax = lmax
        self.mmax = mmax
        self.irreps_out = self.restrict(self.irreps_in, mmax)
        self.global_irreps = self.irreps_in
        self.local_irreps = self.irreps_out

        global_slices = self.irreps_in.slices()
        local_indices = {ir: index for index, (ir, _) in enumerate(self.irreps_out)}
        local_offsets = [0] * len(self.irreps_out)
        entries = []
        wigner_rows = []
        wigner_columns = []
        for global_slice, global_entry in zip(global_slices, self.irreps_in):
            ir, mul = global_entry.ir, global_entry.mul
            retained_mmax = min(ir.l, mmax)
            local_irrep_list = [Irrep(0, ir.p * ((-1) ** ir.l))]
            local_irrep_list.extend(
                Irrep(order, 0) for order in range(1, retained_mmax + 1)
            )
            entry_local_indices = tuple(
                local_indices[local_ir] for local_ir in local_irrep_list
            )
            entry_local_slices = []
            for index in entry_local_indices:
                start = local_offsets[index]
                entry_local_slices.append(slice(start, start + mul))
                local_offsets[index] += mul
            rows = [ir.l]
            for order in range(1, retained_mmax + 1):
                offset = (
                    lmax
                    + 1
                    + sum(
                        2 * (lmax + 1 - lower_order) for lower_order in range(1, order)
                    )
                )
                degree_offset = ir.l - order
                rows.extend(
                    (
                        offset + degree_offset,
                        offset + (lmax + 1 - order) + degree_offset,
                    )
                )
            wigner_rows.append(torch.tensor(rows, dtype=torch.long))
            wigner_columns.append(torch.arange(ir.l**2, (ir.l + 1) ** 2))
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
        rotation_groups = []
        for index, entry in enumerate(entries):
            for indices in rotation_groups:
                if entries[indices[0]].mul == entry.mul and all(
                    entries[grouped_index].degree != entry.degree
                    for grouped_index in indices
                ):
                    indices.append(index)
                    break
            else:
                rotation_groups.append([index])
        self._rotation_groups = tuple(tuple(indices) for indices in rotation_groups)
        for group_index, indices in enumerate(self._rotation_groups):
            self.register_buffer(
                f"wigner_rows_{group_index}",
                torch.cat([wigner_rows[index] for index in indices]),
                persistent=False,
            )
            self.register_buffer(
                f"wigner_columns_{group_index}",
                torch.cat([wigner_columns[index] for index in indices]),
                persistent=False,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps_in} -> "
            f"{self.irreps_out})(mmax={self.mmax})"
        )

    @staticmethod
    def _apply_rotation(
        rotation: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bij,b...jk->b...ik", rotation, features)

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
            ``(batch, local_wigner_dim, (lmax + 1)**2)``.

        Returns
        -------
        torch.Tensor
            Local features with shape ``(batch, ..., irreps_out.dim)``.
        """
        if features.ndim < 2 or features.size(-1) != self.irreps_in.dim:
            raise ValueError(
                "LocalFrame input trailing dimension must be "
                f"{self.irreps_in.dim}, got {tuple(features.shape)}."
            )
        if features.size(0) != wigner.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        outputs = [[] for _ in self.irreps_out]
        for group_index, indices in enumerate(self._rotation_groups):
            values = torch.cat(
                [
                    features[..., self._entries[index].global_slice].reshape(
                        *features.shape[:-1],
                        2 * self._entries[index].degree + 1,
                        self._entries[index].mul,
                    )
                    for index in indices
                ],
                dim=-2,
            )
            rows = getattr(self, f"wigner_rows_{group_index}")
            columns = getattr(self, f"wigner_columns_{group_index}")
            rotation = wigner.index_select(1, rows).index_select(2, columns)
            values = self._apply_rotation(rotation, values)
            offset = 0
            for entry_index in indices:
                entry = self._entries[entry_index]
                outputs[entry.local_indices[0]].append(
                    (
                        entry.local_slices[0].start,
                        values[..., offset : offset + 1, :],
                    )
                )
                offset += 1
                for local_position, local_index in enumerate(
                    entry.local_indices[1:], start=1
                ):
                    pair = values[..., offset : offset + 2, :]
                    if entry.odd:
                        pair = torch.cat((-pair[..., 1:2, :], pair[..., :1, :]), dim=-2)
                    outputs[local_index].append(
                        (entry.local_slices[local_position].start, pair)
                    )
                    offset += 2
        if outputs:
            flattened = []
            for (ir, mul), parts in zip(self.irreps_out, outputs):
                parts = [part for _, part in sorted(parts, key=lambda item: item[0])]
                values = (
                    parts[0].contiguous()
                    if len(parts) == 1
                    else torch.cat(parts, dim=-1)
                )
                flattened.append(values.view(*features.shape[:-1], ir.dim * mul))
            return torch.cat(flattened, dim=-1)
        return features.new_empty(*features.shape[:-1], 0)

    def forward(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`to_local`."""
        return self.to_local(features, wigner)

    def _wigner_mmax(self, local_dim: int) -> int:
        for candidate in range(self.mmax, self.lmax + 1):
            expected = (
                self.lmax
                + 1
                + sum(2 * (self.lmax + 1 - order) for order in range(1, candidate + 1))
            )
            if local_dim == expected:
                return candidate
        raise ValueError("Wigner inverse has an incompatible local dimension.")

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
            ``(batch, (lmax + 1)**2, local_wigner_dim)``. A matrix retaining
            more local orders than this module is accepted and rescaled.

        Returns
        -------
        torch.Tensor
            Global features with shape ``(batch, ..., irreps_in.dim)`` in
            flattened ``ir_mul`` order.
        """
        if features.ndim < 2 or features.size(-1) != self.irreps_out.dim:
            raise ValueError(
                "LocalFrame input trailing dimension must be "
                f"{self.irreps_out.dim}, got {tuple(features.shape)}."
            )
        if features.size(0) != wigner_inv.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        if wigner_inv.size(-2) != (self.lmax + 1) ** 2:
            raise ValueError("Wigner inverse has an incompatible global dimension.")
        wigner_mmax = self._wigner_mmax(wigner_inv.size(-1))
        local_values = [
            features[..., ir_slice].reshape(*features.shape[:-1], ir.dim, mul)
            for (ir, mul), ir_slice in zip(
                self.irreps_out,
                self.irreps_out.slices(),
            )
        ]

        outputs = [None] * len(self._entries)
        for group_index, indices in enumerate(self._rotation_groups):
            group_values = []
            for entry_index in indices:
                entry = self._entries[entry_index]
                entry_values = [
                    local_values[entry.local_indices[0]][..., entry.local_slices[0]]
                ]
                for local_index, local_slice in zip(
                    entry.local_indices[1:],
                    entry.local_slices[1:],
                ):
                    pair = local_values[local_index][..., local_slice]
                    if entry.odd:
                        pair = torch.cat((pair[..., 1:2, :], -pair[..., :1, :]), dim=-2)
                    entry_values.append(pair)
                values = torch.cat(entry_values, dim=-2)
                retained = 2 * min(entry.degree, self.mmax) + 1
                source = 2 * min(entry.degree, wigner_mmax) + 1
                group_values.append(values * math.sqrt(source / retained))
            values = torch.cat(group_values, dim=-2)
            rows = getattr(self, f"wigner_rows_{group_index}")
            columns = getattr(self, f"wigner_columns_{group_index}")
            rotation = wigner_inv.index_select(1, columns).index_select(2, rows)
            values = self._apply_rotation(rotation, values)
            offset = 0
            for entry_index in indices:
                entry = self._entries[entry_index]
                width = 2 * entry.degree + 1
                outputs[entry_index] = values[..., offset : offset + width, :].reshape(
                    *features.shape[:-1],
                    entry.mul * width,
                )
                offset += width
        if outputs:
            return torch.cat(outputs, dim=-1)
        return features.new_empty(*features.shape[:-1], 0)
