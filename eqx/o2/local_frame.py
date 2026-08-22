################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Optional

import torch
from e3nn import o3

from .irreps import Irrep, Irreps


class LocalFrame(torch.nn.Module):
    """Convert global O(3) irreps to and from local O(2) irreps.

    This is the specialized bridge between global O(3) features and the
    otherwise standalone O(2) layers. :meth:`to_local` maps global features to
    grouped local blocks; :meth:`to_global` maps those blocks back to the
    configured global layout. ``mmax`` retains only local orders
    ``0 <= m <= mmax``. Both ``layout="mul_ir"`` and ``layout="ir_mul"`` are
    flattened layouts. The same layout is used by the input and output.
    """

    @staticmethod
    def restrict(irreps: o3.Irreps, mmax: Optional[int] = None) -> Irreps:
        """Return the local O(2) block metadata for an O(3) layout."""
        irreps = o3.Irreps(irreps)
        if mmax is None:
            mmax = irreps.lmax
        if not isinstance(mmax, int) or isinstance(mmax, bool):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        groups = []
        for _, irrep in irreps:
            zero_parity = irrep.p * ((-1) ** irrep.l)
            groups.append((1, Irrep(0, zero_parity)))
            groups.extend(
                (1, Irrep(order, 0))
                for order in range(1, min(irrep.l, mmax) + 1)
            )
        return Irreps(groups).regroup()

    def __init__(
        self,
        irreps: o3.Irreps,
        lmax: int,
        mmax: Optional[int] = None,
        layout: str = "mul_ir",
    ) -> None:
        super().__init__()

        self.global_irreps = o3.Irreps(irreps)
        self.channels = Irreps.common_multiplicity(self.global_irreps)
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError("layout must be 'mul_ir' or 'ir_mul'.")
        self.layout = layout
        if not isinstance(lmax, int) or isinstance(lmax, bool):
            raise TypeError("lmax must be an integer.")
        if self.global_irreps.lmax > lmax:
            raise ValueError("lmax must cover every O(3) irrep.")
        if mmax is None:
            mmax = lmax
        if not isinstance(mmax, int) or isinstance(mmax, bool):
            raise TypeError("mmax must be an integer.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")
        self.lmax = lmax
        self.mmax = mmax
        self.local_irreps = self.restrict(self.global_irreps, mmax)
        self.global_dimensions = tuple(
            irrep.dim for _, irrep in self.global_irreps
        )
        self.global_component_dim = sum(self.global_dimensions)

        group_slices = []
        offset = 0
        for _, irrep in self.global_irreps:
            group_slices.append(slice(offset, offset + irrep.dim))
            offset += irrep.dim
        self.group_slices = tuple(group_slices)

        sources_by_irrep = {irrep: [] for _, irrep in self.local_irreps}
        for group_index, (_, irrep) in enumerate(self.global_irreps):
            zero_irrep = Irrep(0, irrep.p * ((-1) ** irrep.l))
            sources_by_irrep[zero_irrep].append(group_index)
            for order in range(1, min(irrep.l, mmax) + 1):
                sources_by_irrep[Irrep(order, 0)].append(group_index)
        self.block_sources = tuple(
            tuple(sources_by_irrep[irrep]) for _, irrep in self.local_irreps
        )

        block_lookup = {
            (irrep, group_index): (block_index, source_position)
            for block_index, ((_, irrep), sources) in enumerate(
                zip(self.local_irreps, self.block_sources)
            )
            for source_position, group_index in enumerate(sources)
        }

        rotation_groups = []
        degree_counts = {}
        for group_index, (_, irrep) in enumerate(self.global_irreps):
            rotation_index = degree_counts.get(irrep.l, 0)
            degree_counts[irrep.l] = rotation_index + 1
            while len(rotation_groups) <= rotation_index:
                rotation_groups.append([])
            rotation_groups[rotation_index].append(group_index)
        rotation_groups = [
            tuple(
                sorted(groups, key=lambda index: self.global_irreps[index][1].l)
            )
            for groups in rotation_groups
        ]
        self.rotation_groups = tuple(rotation_groups)

        full_size = (lmax + 1) ** 2
        rotation_is_full = []
        rotation_uses_input = []
        source_locations = {}
        rotation_inverse_specs = []
        for rotation_index, groups in enumerate(self.rotation_groups):
            columns = []
            for group_index in groups:
                degree = self.global_irreps[group_index][1].l
                columns.extend(range(degree**2, (degree + 1) ** 2))

            rows = []
            row_locations = {}
            for order in range(mmax + 1):
                active = [
                    group_index
                    for group_index in groups
                    if self.global_irreps[group_index][1].l >= order
                ]
                if order == 0:
                    for group_index in active:
                        row_locations[(group_index, order)] = (len(rows),)
                        rows.append(self.global_irreps[group_index][1].l)
                    continue

                count = lmax + 1 - order
                row_offset = (lmax + 1) + sum(
                    2 * (lmax + 1 - lower_order) for lower_order in range(1, order)
                )
                real_positions = {}
                for group_index in active:
                    degree = self.global_irreps[group_index][1].l
                    real_positions[group_index] = len(rows)
                    rows.append(row_offset + degree - order)
                for group_index in active:
                    degree = self.global_irreps[group_index][1].l
                    imaginary_position = len(rows)
                    rows.append(row_offset + count + degree - order)
                    row_locations[(group_index, order)] = (
                        real_positions[group_index],
                        imaginary_position,
                    )

            row_tensor = torch.tensor(rows, dtype=torch.int64)
            column_tensor = torch.tensor(columns, dtype=torch.int64)
            self.register_buffer(
                f"rotation_rows_{rotation_index}",
                row_tensor,
                persistent=False,
            )
            self.register_buffer(
                f"rotation_columns_{rotation_index}",
                column_tensor,
                persistent=False,
            )
            rotation_is_full.append(
                rows == list(range(full_size)) and columns == list(range(full_size))
            )
            rotation_uses_input.append(
                groups == tuple(range(len(self.global_irreps)))
            )

            inverse_specs = []
            for order in range(mmax + 1):
                active = [
                    group_index
                    for group_index in groups
                    if self.global_irreps[group_index][1].l >= order
                ]
                if order == 0:
                    for group_index in active:
                        irrep = self.global_irreps[group_index][1]
                        local_irrep = Irrep(0, irrep.p * ((-1) ** irrep.l))
                        block_index, source_position = block_lookup[
                            (local_irrep, group_index)
                        ]
                        inverse_specs.append((block_index, source_position, 0, 1))
                    continue
                for component in range(2):
                    for group_index in active:
                        irrep = self.global_irreps[group_index][1]
                        block_index, source_position = block_lookup[
                            (Irrep(order, 0), group_index)
                        ]
                        odd = irrep.p * ((-1) ** irrep.l) == -1
                        if not odd:
                            local_component, sign = component, 1
                        elif component == 0:
                            local_component, sign = 1, 1
                        else:
                            local_component, sign = 0, -1
                        inverse_specs.append(
                            (
                                block_index,
                                source_position,
                                local_component,
                                sign,
                            )
                        )
            rotation_inverse_specs.append(tuple(inverse_specs))

            for group_index in groups:
                irrep = self.global_irreps[group_index][1]
                odd = irrep.p * ((-1) ** irrep.l) == -1
                for order in range(min(irrep.l, mmax) + 1):
                    source_locations[(group_index, order)] = (
                        rotation_index,
                        *row_locations[(group_index, order)],
                        odd,
                    )

        self.rotation_is_full = tuple(rotation_is_full)
        self.rotation_uses_input = tuple(rotation_uses_input)
        self.rotation_inverse_specs = tuple(rotation_inverse_specs)
        self.source_locations = tuple(
            tuple(source_locations[(group_index, irrep.m)] for group_index in sources)
            for (_, irrep), sources in zip(self.local_irreps, self.block_sources)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.global_irreps} -> "
            f"{self.channels * self.local_irreps})"
            f"(mmax={self.mmax}, layout={self.layout!r})"
        )

    def _wigner_mmax(self, wigner_inv: torch.Tensor) -> int:
        if wigner_inv.size(-2) != (self.lmax + 1) ** 2:
            raise ValueError("Wigner inverse has an incompatible global dimension.")
        local_dim = wigner_inv.size(-1)
        for wigner_mmax in range(self.mmax, self.lmax + 1):
            expected = self.lmax + 1 + sum(
                2 * (self.lmax + 1 - order)
                for order in range(1, wigner_mmax + 1)
            )
            if local_dim == expected:
                return wigner_mmax
        raise ValueError("Wigner inverse has an incompatible local dimension.")

    def _inverse_scale(self, degree: int, wigner_mmax: int) -> float:
        source_components = 2 * min(degree, wigner_mmax) + 1
        retained_components = 2 * min(degree, self.mmax) + 1
        return math.sqrt(source_components / retained_components)

    def _unpack_global(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2 or input.size(-1) != self.global_irreps.dim:
            raise ValueError(
                f"{self.layout} input must have shape "
                f"(batch, {self.global_irreps.dim}), got {tuple(input.shape)}."
            )

        offset = 0
        blocks = []
        batch = input.size(0)
        for dimension in self.global_dimensions:
            width = self.channels * dimension
            block = input[:, offset : offset + width]
            if self.layout == "mul_ir":
                block = block.reshape(batch, self.channels, dimension).transpose(
                    1,
                    2,
                )
            else:
                block = block.reshape(batch, dimension, self.channels)
            blocks.append(block)
            offset += width
        return torch.cat(blocks, dim=1).contiguous()

    def _pack_global(self, input: torch.Tensor) -> torch.Tensor:
        expected = (self.global_component_dim, self.channels)
        if input.ndim != 3 or tuple(input.shape[-2:]) != expected:
            raise ValueError(
                "Internal global tensor must have trailing shape "
                f"{expected}, got {tuple(input.shape)}."
            )

        offset = 0
        blocks = []
        batch = input.size(0)
        for dimension in self.global_dimensions:
            block = input[:, offset : offset + dimension]
            if self.layout == "mul_ir":
                block = block.transpose(1, 2).contiguous()
            blocks.append(block.reshape(batch, -1))
            offset += dimension
        return torch.cat(blocks, dim=-1)

    def _rotate_global_irreps(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        input = self._unpack_global(input)
        rotated_groups = []
        for rotation_index, groups in enumerate(self.rotation_groups):
            if self.rotation_uses_input[rotation_index]:
                group_input = input
            else:
                group_input = torch.cat(
                    [
                        input[:, self.group_slices[group_index]]
                        for group_index in groups
                    ],
                    dim=1,
                )
            if self.rotation_is_full[rotation_index]:
                rotation = wigner
            else:
                rows = getattr(self, f"rotation_rows_{rotation_index}")
                columns = getattr(self, f"rotation_columns_{rotation_index}")
                rotation = wigner.index_select(1, rows).index_select(2, columns)
            rotated_groups.append(torch.bmm(rotation, group_input))
        return tuple(rotated_groups)

    def to_local(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        rotated_groups = self._rotate_global_irreps(input, wigner)

        output_blocks = []
        for (_, irrep), locations in zip(self.local_irreps, self.source_locations):
            parts = []
            for location in locations:
                rotation_index, first_position, *remainder = location
                rotated = rotated_groups[rotation_index]
                if irrep.m == 0:
                    parts.append(rotated[:, first_position : first_position + 1])
                    continue
                second_position, odd = remainder
                first = rotated[:, first_position : first_position + 1]
                second = rotated[:, second_position : second_position + 1]
                parts.append(
                    torch.cat((-second, first), dim=1)
                    if odd
                    else torch.cat((first, second), dim=1)
                )
            output_blocks.append(torch.cat(parts, dim=-1))
        return tuple(output_blocks)

    def forward(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.to_local(input, wigner)

    def to_local_channel_major(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return ``(edge, channel, irrep_dim, multiplicity)`` blocks."""
        rotated_groups = self._rotate_global_irreps(input, wigner)
        output_blocks = []
        for (_, irrep), locations in zip(self.local_irreps, self.source_locations):
            parts = []
            for location in locations:
                rotation_index, first_position, *remainder = location
                rotated = rotated_groups[rotation_index]
                if irrep.m == 0:
                    part = rotated[:, first_position : first_position + 1]
                else:
                    second_position, odd = remainder
                    first = rotated[:, first_position : first_position + 1]
                    second = rotated[:, second_position : second_position + 1]
                    part = (
                        torch.cat((-second, first), dim=1)
                        if odd
                        else torch.cat((first, second), dim=1)
                    )
                parts.append(part.transpose(1, 2))
            output_blocks.append(torch.stack(parts, dim=-1))
        return tuple(output_blocks)

    def to_global(
        self,
        input_blocks: tuple[torch.Tensor, ...],
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        if len(input_blocks) != len(self.local_irreps):
            raise ValueError("Expected one input block per local O(2) irrep group.")
        block_channels = []
        for input_block, sources in zip(input_blocks, self.block_sources):
            if input_block.size(-1) % len(sources) != 0:
                raise ValueError("Invalid grouped O(2) block width.")
            block_channels.append(input_block.size(-1) // len(sources))
        if len(set(block_channels)) != 1:
            raise ValueError("Grouped O(2) blocks must share channel counts.")

        wigner_mmax = self._wigner_mmax(wigner_inv)
        output_by_group = [None] * len(self.global_irreps)
        for rotation_index, (groups, inverse_specs) in enumerate(
            zip(self.rotation_groups, self.rotation_inverse_specs)
        ):
            rows = []
            for block_index, source_position, component, sign in inverse_specs:
                channels = block_channels[block_index]
                input_block = input_blocks[block_index]
                row = input_block[
                    :,
                    component : component + 1,
                    source_position * channels : (source_position + 1) * channels,
                ]
                rows.append(row if sign == 1 else -row)
            group_input = torch.cat(rows, dim=1)
            if self.rotation_is_full[rotation_index]:
                rotation = wigner_inv
            else:
                row_indices = getattr(
                    self,
                    f"rotation_rows_{rotation_index}",
                )
                column_indices = getattr(
                    self,
                    f"rotation_columns_{rotation_index}",
                )
                rotation = wigner_inv.index_select(1, column_indices).index_select(
                    2, row_indices
                )
            rotated = torch.bmm(rotation, group_input)

            offset = 0
            for group_index in groups:
                irrep = self.global_irreps[group_index][1]
                width = irrep.dim
                output_by_group[group_index] = rotated[
                    :, offset : offset + width
                ] * self._inverse_scale(irrep.l, wigner_mmax)
                offset += width
        return self._pack_global(torch.cat(output_by_group, dim=1))

    def to_global_channel_major(
        self,
        input_blocks: tuple[torch.Tensor, ...],
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ``(edge, channel, irrep_dim, multiplicity)`` blocks."""
        if len(input_blocks) != len(self.local_irreps):
            raise ValueError("Expected one input block per local O(2) irrep group.")
        channels = input_blocks[0].size(1)
        for input_block, ((multiplicity, irrep), sources) in zip(
            input_blocks,
            zip(self.local_irreps, self.block_sources),
        ):
            expected_shape = (channels, irrep.dim, multiplicity)
            if tuple(input_block.shape[-3:]) != expected_shape:
                raise ValueError(
                    "Channel-major O(2) block trailing shape must be "
                    f"{expected_shape}, got {tuple(input_block.shape)}."
                )
            if multiplicity != len(sources):
                raise RuntimeError("Local O(2) multiplicity resolution failed.")

        wigner_mmax = self._wigner_mmax(wigner_inv)
        output_by_group = [None] * len(self.global_irreps)
        for rotation_index, (groups, inverse_specs) in enumerate(
            zip(self.rotation_groups, self.rotation_inverse_specs)
        ):
            rows = []
            for block_index, source_position, component, sign in inverse_specs:
                row = input_blocks[block_index][
                    :,
                    :,
                    component,
                    source_position,
                ].unsqueeze(1)
                rows.append(row if sign == 1 else -row)
            group_input = torch.cat(rows, dim=1)
            if self.rotation_is_full[rotation_index]:
                rotation = wigner_inv
            else:
                row_indices = getattr(
                    self,
                    f"rotation_rows_{rotation_index}",
                )
                column_indices = getattr(
                    self,
                    f"rotation_columns_{rotation_index}",
                )
                rotation = wigner_inv.index_select(1, column_indices).index_select(
                    2, row_indices
                )
            rotated = torch.bmm(rotation, group_input)

            offset = 0
            for group_index in groups:
                irrep = self.global_irreps[group_index][1]
                width = irrep.dim
                output_by_group[group_index] = rotated[
                    :, offset : offset + width
                ] * self._inverse_scale(irrep.l, wigner_mmax)
                offset += width
        return self._pack_global(torch.cat(output_by_group, dim=1))
