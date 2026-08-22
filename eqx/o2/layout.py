################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Optional

import torch
from e3nn import o3

from .irreps import Irrep, Irreps


class _O3LayoutTransform(torch.nn.Module):
    """Convert MulIr to IrMul Layout."""

    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.multiplicities = tuple(multiplicity for multiplicity, _ in self.irreps)
        self.dimensions = tuple(irrep.dim for _, irrep in self.irreps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        offset = 0
        blocks = []
        batch = input.size(0)
        for multiplicity, dimension in zip(self.multiplicities, self.dimensions):
            width = multiplicity * dimension
            block = input[:, offset : offset + width]
            blocks.append(block.reshape(batch, multiplicity, dimension))
            offset += width
        return torch.cat(blocks, dim=-1).transpose(-1, -2).contiguous()

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        input = input.transpose(-1, -2).contiguous()
        offset = 0
        blocks = []
        batch = input.size(0)
        for dimension in self.dimensions:
            block = input[:, :, offset : offset + dimension]
            blocks.append(block.reshape(batch, -1))
            offset += dimension
        return torch.cat(blocks, dim=-1)


class O3O2Layout(torch.nn.Module):
    """Rotate O(3) towers directly into contiguous local O(2) blocks.

    This is the specialized bridge between global O(3) features and the
    otherwise standalone O(2) layers. Calling the module maps global features
    to grouped local blocks; :meth:`inverse` maps those blocks back to the
    global layout. ``mmax`` optionally retains only local orders
    ``0 <= m <= mmax`` while leaving the global O(3) layout unchanged.
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
    ) -> None:
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.channels = Irreps.common_multiplicity(self.irreps)
        if not isinstance(lmax, int) or isinstance(lmax, bool):
            raise TypeError("lmax must be an integer.")
        if self.irreps.lmax > lmax:
            raise ValueError("lmax must cover every O(3) irrep.")
        if mmax is None:
            mmax = lmax
        if not isinstance(mmax, int) or isinstance(mmax, bool):
            raise TypeError("mmax must be an integer.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")
        self.lmax = lmax
        self.mmax = mmax
        self.local_irreps = self.restrict(self.irreps, mmax)
        self.layout = _O3LayoutTransform(self.irreps)

        group_slices = []
        offset = 0
        for _, irrep in self.irreps:
            group_slices.append(slice(offset, offset + irrep.dim))
            offset += irrep.dim
        self.group_slices = tuple(group_slices)

        sources_by_irrep = {irrep: [] for _, irrep in self.local_irreps}
        for group_index, (_, irrep) in enumerate(self.irreps):
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

        towers = []
        degree_counts = {}
        for group_index, (_, irrep) in enumerate(self.irreps):
            tower_index = degree_counts.get(irrep.l, 0)
            degree_counts[irrep.l] = tower_index + 1
            while len(towers) <= tower_index:
                towers.append([])
            towers[tower_index].append(group_index)
        towers = [
            tuple(sorted(groups, key=lambda index: self.irreps[index][1].l))
            for groups in towers
        ]
        self.tower_groups = tuple(towers)

        full_size = (lmax + 1) ** 2
        tower_is_full = []
        tower_uses_input = []
        source_locations = {}
        tower_inverse_specs = []
        for tower_index, groups in enumerate(self.tower_groups):
            columns = []
            for group_index in groups:
                degree = self.irreps[group_index][1].l
                columns.extend(range(degree**2, (degree + 1) ** 2))

            rows = []
            row_locations = {}
            for order in range(mmax + 1):
                active = [
                    group_index
                    for group_index in groups
                    if self.irreps[group_index][1].l >= order
                ]
                if order == 0:
                    for group_index in active:
                        row_locations[(group_index, order)] = (len(rows),)
                        rows.append(self.irreps[group_index][1].l)
                    continue

                count = lmax + 1 - order
                row_offset = (lmax + 1) + sum(
                    2 * (lmax + 1 - lower_order) for lower_order in range(1, order)
                )
                real_positions = {}
                for group_index in active:
                    degree = self.irreps[group_index][1].l
                    real_positions[group_index] = len(rows)
                    rows.append(row_offset + degree - order)
                for group_index in active:
                    degree = self.irreps[group_index][1].l
                    imaginary_position = len(rows)
                    rows.append(row_offset + count + degree - order)
                    row_locations[(group_index, order)] = (
                        real_positions[group_index],
                        imaginary_position,
                    )

            row_tensor = torch.tensor(rows, dtype=torch.int64)
            column_tensor = torch.tensor(columns, dtype=torch.int64)
            self.register_buffer(
                f"tower_rows_{tower_index}",
                row_tensor,
                persistent=False,
            )
            self.register_buffer(
                f"tower_columns_{tower_index}",
                column_tensor,
                persistent=False,
            )
            tower_is_full.append(
                rows == list(range(full_size)) and columns == list(range(full_size))
            )
            tower_uses_input.append(groups == tuple(range(len(self.irreps))))

            inverse_specs = []
            for order in range(mmax + 1):
                active = [
                    group_index
                    for group_index in groups
                    if self.irreps[group_index][1].l >= order
                ]
                if order == 0:
                    for group_index in active:
                        irrep = self.irreps[group_index][1]
                        local_irrep = Irrep(0, irrep.p * ((-1) ** irrep.l))
                        block_index, source_position = block_lookup[
                            (local_irrep, group_index)
                        ]
                        inverse_specs.append((block_index, source_position, 0, 1))
                    continue
                for component in range(2):
                    for group_index in active:
                        irrep = self.irreps[group_index][1]
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
            tower_inverse_specs.append(tuple(inverse_specs))

            for group_index in groups:
                irrep = self.irreps[group_index][1]
                odd = irrep.p * ((-1) ** irrep.l) == -1
                for order in range(min(irrep.l, mmax) + 1):
                    source_locations[(group_index, order)] = (
                        tower_index,
                        *row_locations[(group_index, order)],
                        odd,
                    )

        self.tower_is_full = tuple(tower_is_full)
        self.tower_uses_input = tuple(tower_uses_input)
        self.tower_inverse_specs = tuple(tower_inverse_specs)
        self.source_locations = tuple(
            tuple(source_locations[(group_index, irrep.m)] for group_index in sources)
            for (_, irrep), sources in zip(self.local_irreps, self.block_sources)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.irreps} -> "
            f"{self.channels * self.local_irreps})(mmax={self.mmax})"
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

    def _rotate_towers(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        input = self.layout(input)
        tower_outputs = []
        for tower_index, groups in enumerate(self.tower_groups):
            if self.tower_uses_input[tower_index]:
                tower_input = input
            else:
                tower_input = torch.cat(
                    [
                        input[:, self.group_slices[group_index]]
                        for group_index in groups
                    ],
                    dim=1,
                )
            if self.tower_is_full[tower_index]:
                rotation = wigner
            else:
                rows = getattr(self, f"tower_rows_{tower_index}")
                columns = getattr(self, f"tower_columns_{tower_index}")
                rotation = wigner.index_select(1, rows).index_select(2, columns)
            tower_outputs.append(torch.bmm(rotation, tower_input))
        return tuple(tower_outputs)

    def forward(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        tower_outputs = self._rotate_towers(input, wigner)

        output_blocks = []
        for (_, irrep), locations in zip(self.local_irreps, self.source_locations):
            parts = []
            for location in locations:
                tower_index, first_position, *remainder = location
                tower_output = tower_outputs[tower_index]
                if irrep.m == 0:
                    parts.append(tower_output[:, first_position : first_position + 1])
                    continue
                second_position, odd = remainder
                first = tower_output[:, first_position : first_position + 1]
                second = tower_output[:, second_position : second_position + 1]
                parts.append(
                    torch.cat((-second, first), dim=1)
                    if odd
                    else torch.cat((first, second), dim=1)
                )
            output_blocks.append(torch.cat(parts, dim=-1))
        return tuple(output_blocks)

    def forward_channel_major(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return ``(edge, channel, irrep_dim, multiplicity)`` blocks."""
        tower_outputs = self._rotate_towers(input, wigner)
        output_blocks = []
        for (_, irrep), locations in zip(self.local_irreps, self.source_locations):
            parts = []
            for location in locations:
                tower_index, first_position, *remainder = location
                tower_output = tower_outputs[tower_index]
                if irrep.m == 0:
                    part = tower_output[:, first_position : first_position + 1]
                else:
                    second_position, odd = remainder
                    first = tower_output[:, first_position : first_position + 1]
                    second = tower_output[:, second_position : second_position + 1]
                    part = (
                        torch.cat((-second, first), dim=1)
                        if odd
                        else torch.cat((first, second), dim=1)
                    )
                parts.append(part.transpose(1, 2))
            output_blocks.append(torch.stack(parts, dim=-1))
        return tuple(output_blocks)

    def inverse(
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
        output_by_group = [None] * len(self.irreps)
        for tower_index, (groups, inverse_specs) in enumerate(
            zip(self.tower_groups, self.tower_inverse_specs)
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
            tower_input = torch.cat(rows, dim=1)
            if self.tower_is_full[tower_index]:
                rotation = wigner_inv
            else:
                row_indices = getattr(self, f"tower_rows_{tower_index}")
                column_indices = getattr(self, f"tower_columns_{tower_index}")
                rotation = wigner_inv.index_select(1, column_indices).index_select(
                    2, row_indices
                )
            tower_output = torch.bmm(rotation, tower_input)

            offset = 0
            for group_index in groups:
                irrep = self.irreps[group_index][1]
                width = irrep.dim
                output_by_group[group_index] = tower_output[
                    :, offset : offset + width
                ] * self._inverse_scale(irrep.l, wigner_mmax)
                offset += width
        return self.layout.inverse(torch.cat(output_by_group, dim=1))

    def inverse_channel_major(
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
        output_by_group = [None] * len(self.irreps)
        for tower_index, (groups, inverse_specs) in enumerate(
            zip(self.tower_groups, self.tower_inverse_specs)
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
            tower_input = torch.cat(rows, dim=1)
            if self.tower_is_full[tower_index]:
                rotation = wigner_inv
            else:
                row_indices = getattr(self, f"tower_rows_{tower_index}")
                column_indices = getattr(self, f"tower_columns_{tower_index}")
                rotation = wigner_inv.index_select(1, column_indices).index_select(
                    2, row_indices
                )
            tower_output = torch.bmm(rotation, tower_input)

            offset = 0
            for group_index in groups:
                irrep = self.irreps[group_index][1]
                width = irrep.dim
                output_by_group[group_index] = tower_output[
                    :, offset : offset + width
                ] * self._inverse_scale(irrep.l, wigner_mmax)
                offset += width
        return self.layout.inverse(torch.cat(output_by_group, dim=1))
