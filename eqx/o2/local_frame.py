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
        Global input representation, including time parity when present.
        Every entry is stored in flattened ``ir_mul`` order.
    mmax : int, optional
        Largest local O(2) order to retain. Defaults to the largest degree in
        ``irreps``; larger values have no effect.
    reverse : bool, optional
        Reverse the global/local order in the module representation. This only
        changes how the module is displayed.

    Notes
    -----
    The first tensor dimension is the rotation batch. Additional leading
    dimensions, such as a source/target axis, are preserved. The local output
    representation is available as :attr:`irreps_out`.
    Wigner layout is inferred from the matrix dimensions. Matrices shared
    across representations may contain additional degrees or local orders.
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
            irrep_list.append(
                (Irrep(0, ir.p * ((-1) ** ir.l), time_parity), mul)
            )
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
        self.irreps_out = self.restrict(self.irreps_in, self.mmax)
        self.global_irreps = self.irreps_in
        self.local_irreps = self.irreps_out
        self.input_dim = self.irreps_in.dim
        self.output_dim = self.irreps_out.dim

        global_slices = self.irreps_in.slices()
        local_indices = {ir: index for index, (ir, _) in enumerate(self.irreps_out)}
        local_offsets = [0] * len(self.irreps_out)
        entries = []
        wigner_rows = []
        wigner_row_strides = []
        wigner_columns = []
        for global_slice, global_entry in zip(global_slices, self.irreps_in):
            ir, mul = global_entry.ir, global_entry.mul
            retained_mmax = min(ir.l, self.mmax)
            time_parity = getattr(ir, "t", 1)
            local_irrep_list = [
                Irrep(0, ir.p * ((-1) ** ir.l), time_parity)
            ]
            local_irrep_list.extend(
                Irrep(order, 0, time_parity)
                for order in range(1, retained_mmax + 1)
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
            row_strides = [0]
            for order in range(1, retained_mmax + 1):
                rows.extend(
                    (
                        ir.l + (2 * order - 1) * (self.lmax + 1) - order**2,
                        ir.l + 2 * order * (self.lmax + 1) - order * (order + 1),
                    )
                )
                row_strides.extend((2 * order - 1, 2 * order))
            wigner_rows.append(torch.tensor(rows, dtype=torch.long))
            wigner_row_strides.append(torch.tensor(row_strides, dtype=torch.long))
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
                f"wigner_row_strides_{group_index}",
                torch.cat([wigner_row_strides[index] for index in indices]),
                persistent=False,
            )
            self.register_buffer(
                f"wigner_columns_{group_index}",
                torch.cat([wigner_columns[index] for index in indices]),
                persistent=False,
            )

    def __repr__(self) -> str:
        irreps_in, irreps_out = (
            (self.local_irreps, self.global_irreps)
            if self.reverse
            else (self.global_irreps, self.local_irreps)
        )
        return (
            f"{self.__class__.__name__}({irreps_in} -> "
            f"{irreps_out})(mmax={self.mmax})"
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
            ``(batch, local_wigner_dim, (L + 1)**2)``. The global degree ``L``
            and retained local orders are inferred from these dimensions and
            must cover the representation used by this module.

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
        if wigner.ndim != 3 or features.size(0) != wigner.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        lmax, _ = self._wigner_orders(wigner.size(-1), wigner.size(-2))
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
            if lmax != self.lmax and self.mmax:
                rows = rows + (lmax - self.lmax) * getattr(
                    self, f"wigner_row_strides_{group_index}"
                )
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
        return features.new_empty((*features.shape[:-1], 0))

    def forward(
        self,
        features: torch.Tensor,
        wigner: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`to_local`."""
        return self.to_local(features, wigner)

    def _wigner_orders(self, global_dim: int, local_dim: int) -> tuple[int, int]:
        lmax = int(math.sqrt(global_dim)) - 1
        if (lmax + 1) ** 2 != global_dim or lmax < self.lmax:
            raise ValueError("Wigner global dimension must cover every O(3) degree.")
        missing = global_dim - local_dim
        if missing < 0:
            raise ValueError("Wigner has an incompatible local dimension.")
        # Removing the highest local orders removes n * (n + 1) rows.
        omitted = (int(math.sqrt(4 * missing + 1)) - 1) // 2
        mmax = lmax - omitted
        if omitted * (omitted + 1) != missing or not self.mmax <= mmax <= lmax:
            raise ValueError("Wigner local dimension must cover all required orders.")
        return lmax, mmax

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
        if wigner_inv.ndim != 3 or features.size(0) != wigner_inv.size(0):
            raise ValueError("Feature and Wigner batch dimensions must match.")
        lmax, wigner_mmax = self._wigner_orders(
            wigner_inv.size(-2), wigner_inv.size(-1)
        )
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
            if lmax != self.lmax and self.mmax:
                rows = rows + (lmax - self.lmax) * getattr(
                    self, f"wigner_row_strides_{group_index}"
                )
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
        return features.new_empty((*features.shape[:-1], 0))
