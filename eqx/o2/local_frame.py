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
    """Convert between global O(3) and edge-local O(2) representations.

    Inputs and outputs use ``(batch, component, channel)`` tensors. Conversion
    to and from the default e3nn layout is owned by the calling model.
    ``local_irreps`` contains the complete O(2) multiplicities, including the
    common channel multiplicity of the input O(3) irreps.
    """

    @staticmethod
    def restrict(irreps: o3.Irreps, mmax: Optional[int] = None) -> Irreps:
        """Restrict complete O(3) irreps to complete O(2) irreps."""
        irreps = o3.Irreps(irreps)
        if mmax is None:
            mmax = irreps.lmax
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")

        irrep_list = []
        for mul, ir in irreps:
            zero_parity = ir.p * ((-1) ** ir.l)
            irrep_list.append((mul, Irrep(0, zero_parity)))
            irrep_list.extend(
                (mul, Irrep(order, 0))
                for order in range(1, min(ir.l, mmax) + 1)
            )
        return Irreps(irrep_list).regroup()

    def __init__(
        self,
        irreps: o3.Irreps,
        lmax: int,
        mmax: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.global_irreps = o3.Irreps(irreps)
        self.channels = Irreps.common_multiplicity(self.global_irreps)
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer.")
        if self.global_irreps.lmax > lmax:
            raise ValueError("lmax must cover every O(3) irrep.")
        if mmax is None:
            mmax = lmax
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax.")
        self.lmax = lmax
        self.mmax = mmax
        self.local_irreps = self.restrict(self.global_irreps, mmax)

        irrep_slices = []
        offset = 0
        for mul, ir in self.global_irreps:
            irrep_slices.append(slice(offset, offset + ir.dim))
            offset += ir.dim
        self.irrep_slices = tuple(irrep_slices)
        self.global_component_dim = offset

        sources_by_ir = {ir: [] for mul, ir in self.local_irreps}
        for ir_index, (mul, ir) in enumerate(self.global_irreps):
            zero_ir = Irrep(0, ir.p * ((-1) ** ir.l))
            sources_by_ir[zero_ir].append(ir_index)
            for order in range(1, min(ir.l, mmax) + 1):
                sources_by_ir[Irrep(order, 0)].append(ir_index)
        self.block_sources = tuple(
            tuple(sources_by_ir[ir]) for mul, ir in self.local_irreps
        )

        block_lookup = {
            (ir, ir_index): (block_index, source_position)
            for block_index, ((mul, ir), sources) in enumerate(
                zip(self.local_irreps, self.block_sources)
            )
            for source_position, ir_index in enumerate(sources)
        }

        rotation_irrep_indices = []
        degree_counts = {}
        for ir_index, (mul, ir) in enumerate(self.global_irreps):
            rotation_index = degree_counts.get(ir.l, 0)
            degree_counts[ir.l] = rotation_index + 1
            while len(rotation_irrep_indices) <= rotation_index:
                rotation_irrep_indices.append([])
            rotation_irrep_indices[rotation_index].append(ir_index)
        self.rotation_irrep_indices = tuple(
            tuple(
                sorted(
                    irrep_indices,
                    key=lambda index: self.global_irreps[index][1].l,
                )
            )
            for irrep_indices in rotation_irrep_indices
        )

        full_size = (lmax + 1) ** 2
        wigner_is_full = []
        rotation_uses_input = []
        source_locations = {}
        inverse_specs_list = []
        for rotation_index, irrep_indices in enumerate(
            self.rotation_irrep_indices
        ):
            columns = []
            for ir_index in irrep_indices:
                mul, ir = self.global_irreps[ir_index]
                columns.extend(range(ir.l**2, (ir.l + 1) ** 2))

            rows = []
            row_locations = {}
            for order in range(mmax + 1):
                active_indices = [
                    ir_index
                    for ir_index in irrep_indices
                    if self.global_irreps[ir_index][1].l >= order
                ]
                if order == 0:
                    for ir_index in active_indices:
                        mul, ir = self.global_irreps[ir_index]
                        row_locations[(ir_index, order)] = (len(rows),)
                        rows.append(ir.l)
                    continue

                count = lmax + 1 - order
                row_offset = (lmax + 1) + sum(
                    2 * (lmax + 1 - lower_order)
                    for lower_order in range(1, order)
                )
                real_positions = {}
                for ir_index in active_indices:
                    mul, ir = self.global_irreps[ir_index]
                    real_positions[ir_index] = len(rows)
                    rows.append(row_offset + ir.l - order)
                for ir_index in active_indices:
                    mul, ir = self.global_irreps[ir_index]
                    imaginary_position = len(rows)
                    rows.append(row_offset + count + ir.l - order)
                    row_locations[(ir_index, order)] = (
                        real_positions[ir_index],
                        imaginary_position,
                    )

            self.register_buffer(
                f"wigner_rows_{rotation_index}",
                torch.tensor(rows, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                f"wigner_columns_{rotation_index}",
                torch.tensor(columns, dtype=torch.int64),
                persistent=False,
            )
            wigner_is_full.append(
                rows == list(range(full_size))
                and columns == list(range(full_size))
            )
            rotation_uses_input.append(
                irrep_indices == tuple(range(len(self.global_irreps)))
            )

            inverse_specs = []
            for order in range(mmax + 1):
                active_indices = [
                    ir_index
                    for ir_index in irrep_indices
                    if self.global_irreps[ir_index][1].l >= order
                ]
                if order == 0:
                    for ir_index in active_indices:
                        mul, ir = self.global_irreps[ir_index]
                        local_ir = Irrep(0, ir.p * ((-1) ** ir.l))
                        block_index, source_position = block_lookup[
                            (local_ir, ir_index)
                        ]
                        inverse_specs.append(
                            (block_index, source_position, 0, 1)
                        )
                    continue

                for component in range(2):
                    for ir_index in active_indices:
                        mul, ir = self.global_irreps[ir_index]
                        block_index, source_position = block_lookup[
                            (Irrep(order, 0), ir_index)
                        ]
                        odd = ir.p * ((-1) ** ir.l) == -1
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
            inverse_specs_list.append(tuple(inverse_specs))

            for ir_index in irrep_indices:
                mul, ir = self.global_irreps[ir_index]
                odd = ir.p * ((-1) ** ir.l) == -1
                for order in range(min(ir.l, mmax) + 1):
                    source_locations[(ir_index, order)] = (
                        rotation_index,
                        *row_locations[(ir_index, order)],
                        odd,
                    )

        self.wigner_is_full = tuple(wigner_is_full)
        self.rotation_uses_input = tuple(rotation_uses_input)
        self.inverse_specs = tuple(inverse_specs_list)
        self.source_locations = tuple(
            tuple(
                source_locations[(ir_index, ir.m)] for ir_index in sources
            )
            for (mul, ir), sources in zip(
                self.local_irreps,
                self.block_sources,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.global_irreps} -> "
            f"{self.local_irreps})(mmax={self.mmax})"
        )

    def to_local(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if input.ndim != 3 or input.size(1) != self.global_component_dim:
            raise ValueError(
                "Local-frame input must have shape "
                f"(batch, {self.global_component_dim}, channels), "
                f"got {tuple(input.shape)}."
            )
        if input.size(0) != wigner.size(0):
            raise ValueError("Input and Wigner batch dimensions must match.")

        rotated_list = []
        for rotation_index, irrep_indices in enumerate(
            self.rotation_irrep_indices
        ):
            if self.rotation_uses_input[rotation_index]:
                irrep_input = input
            else:
                irrep_input = torch.cat(
                    [input[:, self.irrep_slices[index]] for index in irrep_indices],
                    dim=1,
                )
            if self.wigner_is_full[rotation_index]:
                rotation = wigner
            else:
                rows = getattr(self, f"wigner_rows_{rotation_index}")
                columns = getattr(self, f"wigner_columns_{rotation_index}")
                rotation = wigner.index_select(1, rows).index_select(2, columns)
            rotated_list.append(torch.bmm(rotation, irrep_input))

        output_blocks = []
        for (mul, ir), locations in zip(
            self.local_irreps,
            self.source_locations,
        ):
            irrep_list = []
            for location in locations:
                rotation_index, first_position, *remainder = location
                rotated = rotated_list[rotation_index]
                if ir.m == 0:
                    irrep_list.append(
                        rotated[:, first_position : first_position + 1]
                    )
                    continue
                second_position, odd = remainder
                first = rotated[:, first_position : first_position + 1]
                second = rotated[:, second_position : second_position + 1]
                irrep_list.append(
                    torch.cat((-second, first), dim=1)
                    if odd
                    else torch.cat((first, second), dim=1)
                )
            output_blocks.append(torch.cat(irrep_list, dim=-1))
        return tuple(output_blocks)

    def forward(
        self,
        input: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.to_local(input, wigner)

    def to_global(
        self,
        input_blocks: tuple[torch.Tensor, ...],
        wigner_inv: torch.Tensor,
    ) -> torch.Tensor:
        if len(input_blocks) != len(self.local_irreps):
            raise ValueError("Expected one input block per local O(2) irrep.")

        channel_list = []
        for input_block, sources in zip(input_blocks, self.block_sources):
            if input_block.size(-1) % len(sources) != 0:
                raise ValueError("Invalid O(2) block width.")
            channel_list.append(input_block.size(-1) // len(sources))
        if len(set(channel_list)) != 1 or channel_list[0] != self.channels:
            raise ValueError("O(2) blocks must use the configured channel count.")

        if wigner_inv.size(-2) != (self.lmax + 1) ** 2:
            raise ValueError("Wigner inverse has an incompatible global dimension.")
        local_dim = wigner_inv.size(-1)
        wigner_mmax = None
        for candidate in range(self.mmax, self.lmax + 1):
            expected = self.lmax + 1 + sum(
                2 * (self.lmax + 1 - order)
                for order in range(1, candidate + 1)
            )
            if local_dim == expected:
                wigner_mmax = candidate
                break
        if wigner_mmax is None:
            raise ValueError("Wigner inverse has an incompatible local dimension.")

        output_by_irrep = [None] * len(self.global_irreps)
        for rotation_index, (irrep_indices, inverse_specs) in enumerate(
            zip(self.rotation_irrep_indices, self.inverse_specs)
        ):
            row_list = []
            for block_index, source_position, component, sign in inverse_specs:
                input_block = input_blocks[block_index]
                row = input_block[
                    :,
                    component : component + 1,
                    source_position
                    * self.channels : (source_position + 1)
                    * self.channels,
                ]
                row_list.append(row if sign == 1 else -row)
            irrep_input = torch.cat(row_list, dim=1)
            if self.wigner_is_full[rotation_index]:
                rotation = wigner_inv
            else:
                rows = getattr(self, f"wigner_rows_{rotation_index}")
                columns = getattr(self, f"wigner_columns_{rotation_index}")
                rotation = wigner_inv.index_select(1, columns).index_select(
                    2,
                    rows,
                )
            rotated = torch.bmm(rotation, irrep_input)

            offset = 0
            for ir_index in irrep_indices:
                mul, ir = self.global_irreps[ir_index]
                retained = 2 * min(ir.l, self.mmax) + 1
                source = 2 * min(ir.l, wigner_mmax) + 1
                output_by_irrep[ir_index] = rotated[
                    :, offset : offset + ir.dim
                ] * math.sqrt(source / retained)
                offset += ir.dim
        return torch.cat(output_by_irrep, dim=1)
