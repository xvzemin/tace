################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Complete local O(2) magnetic interactions for the e3nn model."""

from typing import Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .. import o2
from ..layout import LayoutTransform


def _common_multiplicity(irreps: o3.Irreps, name: str) -> int:
    multiplicities = {multiplicity for multiplicity, _ in irreps}
    if len(multiplicities) != 1:
        raise ValueError(f"{name} must use one common multiplicity per irrep.")
    return next(iter(multiplicities))


def _restrict_irreps(irreps: o3.Irreps) -> o2.Irreps:
    groups = []
    for _, irrep in irreps:
        groups.extend(o2.restrict_o3_irrep(irrep.l, irrep.p).groups)
    return o2.Irreps(groups).regroup()


class _O3O2Layout(torch.nn.Module):
    """Rotate O(3) towers directly into contiguous local O(2) blocks."""

    def __init__(self, irreps: o3.Irreps, lmax: int) -> None:
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        if self.irreps.lmax > lmax:
            raise ValueError("lmax must cover every O(3) irrep.")
        self.local_irreps = _restrict_irreps(self.irreps)
        self.layout = LayoutTransform(self.irreps)

        group_slices = []
        offset = 0
        for _, irrep in self.irreps:
            group_slices.append(slice(offset, offset + irrep.dim))
            offset += irrep.dim
        self.group_slices = tuple(group_slices)

        sources_by_irrep = {irrep: [] for _, irrep in self.local_irreps}
        for group_index, (_, irrep) in enumerate(self.irreps):
            zero_irrep = o2.Irrep(0, irrep.p * ((-1) ** irrep.l))
            sources_by_irrep[zero_irrep].append(group_index)
            for order in range(1, irrep.l + 1):
                sources_by_irrep[o2.Irrep(order, 0)].append(group_index)
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
            for order in range(lmax + 1):
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
            for order in range(lmax + 1):
                active = [
                    group_index
                    for group_index in groups
                    if self.irreps[group_index][1].l >= order
                ]
                if order == 0:
                    for group_index in active:
                        irrep = self.irreps[group_index][1]
                        local_irrep = o2.Irrep(0, irrep.p * ((-1) ** irrep.l))
                        block_index, source_position = block_lookup[
                            (local_irrep, group_index)
                        ]
                        inverse_specs.append((block_index, source_position, 0, 1))
                    continue
                for component in range(2):
                    for group_index in active:
                        irrep = self.irreps[group_index][1]
                        block_index, source_position = block_lookup[
                            (o2.Irrep(order, 0), group_index)
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
                for order in range(irrep.l + 1):
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
                width = self.irreps[group_index][1].dim
                output_by_group[group_index] = tower_output[:, offset : offset + width]
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
                width = self.irreps[group_index][1].dim
                output_by_group[group_index] = tower_output[:, offset : offset + width]
                offset += width
        return self.layout.inverse(torch.cat(output_by_group, dim=1))


class O2MagneticScatterLinear(torch.nn.Module):
    """Edge-aligned complete-O2 magnetic linear convolution.

    The source node and source magnetic solid harmonics are concatenated in
    the local representation. In ``uv`` mode, edge weights first apply a
    diagonal channel-wise radial map and an independent internal ``uv`` linear
    mixes channels. In ``uu`` mode, edge weights parameterize the external
    channel-wise linear directly.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        path_mode: str,
    ) -> None:
        super().__init__()

        if path_mode not in {"uv", "uu"}:
            raise ValueError("path_mode must be 'uv' or 'uu'.")
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_irreps = o3.Irreps(magnetic_irreps)
        self.num_channel = num_channel
        self.path_mode = path_mode
        if _common_multiplicity(self.irreps_node, "irreps_node") != num_channel:
            raise ValueError("irreps_node multiplicity must equal num_channel.")
        if _common_multiplicity(self.irreps_out, "irreps_out") != num_channel:
            raise ValueError("irreps_out multiplicity must equal num_channel.")

        self.node_layout = _O3O2Layout(self.irreps_node, lmax)
        self.magnetic_layout = _O3O2Layout(self.magnetic_irreps, lmax)
        self.output_layout = _O3O2Layout(self.irreps_out, lmax)
        self.irreps_in_local = o2.Irreps(
            self.node_layout.local_irreps.groups
            + self.magnetic_layout.local_irreps.groups
        ).regroup()
        self.irreps_out_local = self.output_layout.local_irreps

        self.input_block_irreps = tuple(irrep for _, irrep in self.irreps_in_local)
        self.node_block_indices = {
            irrep: index
            for index, (_, irrep) in enumerate(self.node_layout.local_irreps)
        }
        self.magnetic_block_indices = {
            irrep: index
            for index, (_, irrep) in enumerate(self.magnetic_layout.local_irreps)
        }

        if path_mode == "uv":
            self.linear = o2.Linear(
                self.irreps_in_local,
                self.irreps_out_local,
                num_channel,
                path_mode="uv",
                bias=False,
            )
            self.weight_numel = self.irreps_in_local.num_irreps * num_channel
        else:
            input_groups = {
                irrep: (index, multiplicity)
                for index, (multiplicity, irrep) in enumerate(self.irreps_in_local)
            }
            self.uu_group_specs = tuple(
                (
                    *input_groups.get(output_irrep, (-1, 0)),
                    output_multiplicity,
                )
                for output_multiplicity, output_irrep in self.irreps_out_local
            )
            self.weight_numel = sum(
                num_channel * input_multiplicity * output_multiplicity
                for _, input_multiplicity, output_multiplicity in self.uu_group_specs
            )

    def _uu_linear(
        self,
        input_blocks: tuple[torch.Tensor, ...],
        radial_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        radial_weights = radial_weights.reshape(radial_weights.size(0), -1)
        output_blocks = []
        offset = 0
        zero_dependency = input_blocks[0].sum() * 0 + radial_weights.sum() * 0
        for (
            (output_multiplicity, output_irrep),
            (input_index, input_multiplicity, _),
        ) in zip(self.irreps_out_local, self.uu_group_specs):
            if input_index < 0:
                output_blocks.append(
                    radial_weights.new_zeros(
                        radial_weights.size(0),
                        self.num_channel,
                        output_irrep.dim,
                        output_multiplicity,
                    )
                    + zero_dependency
                )
                continue
            weight_numel = self.num_channel * output_multiplicity * input_multiplicity
            weight = radial_weights[:, offset : offset + weight_numel].reshape(
                radial_weights.size(0),
                self.num_channel,
                output_multiplicity,
                input_multiplicity,
            )
            output_block = torch.matmul(
                input_blocks[input_index],
                weight.transpose(-1, -2),
            )
            output_blocks.append(output_block * input_multiplicity**-0.5)
            offset += weight_numel
        if offset != radial_weights.size(-1):
            raise ValueError("Invalid grouped UU radial weight size.")
        return tuple(output_blocks)

    def forward(
        self,
        node_feats: torch.Tensor,
        magnetic_node_attrs: torch.Tensor,
        radial_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if wigner is None or wigner_inv is None:
            raise ValueError("O2 magnetic convolution requires edge Wigner matrices.")

        source = edge_index[0]
        if self.path_mode == "uv":
            node_blocks = self.node_layout(node_feats[source], wigner)
            magnetic_blocks = self.magnetic_layout(
                magnetic_node_attrs[source].unsqueeze(-1),
                wigner,
            )
            magnetic_blocks = tuple(
                block.reshape(*block.shape, 1)
                .expand(*block.shape, self.num_channel)
                .reshape(*block.shape[:-1], block.size(-1) * self.num_channel)
                for block in magnetic_blocks
            )
        else:
            node_blocks = self.node_layout.forward_channel_major(
                node_feats[source],
                wigner,
            )
            magnetic_blocks = self.magnetic_layout.forward_channel_major(
                magnetic_node_attrs[source].unsqueeze(-1),
                wigner,
            )
            magnetic_blocks = tuple(
                block.expand(-1, self.num_channel, -1, -1) for block in magnetic_blocks
            )

        input_blocks = []
        for irrep in self.input_block_irreps:
            parts = []
            node_index = self.node_block_indices.get(irrep)
            if node_index is not None:
                parts.append(node_blocks[node_index])
            magnetic_index = self.magnetic_block_indices.get(irrep)
            if magnetic_index is not None:
                parts.append(magnetic_blocks[magnetic_index])
            input_blocks.append(torch.cat(parts, dim=-1))
        input_blocks = tuple(input_blocks)

        if self.path_mode == "uv":
            radial_weights = radial_weights.reshape(radial_weights.size(0), -1)
            weighted_blocks = []
            offset = 0
            for (multiplicity, _), input_block in zip(
                self.irreps_in_local,
                input_blocks,
            ):
                width = multiplicity * self.num_channel
                weight = radial_weights[:, offset : offset + width].unsqueeze(-2)
                weighted_blocks.append(input_block * weight)
                offset += width
            output_blocks = self.linear.forward_grouped(tuple(weighted_blocks))
        else:
            output_blocks = self._uu_linear(input_blocks, radial_weights)

        if self.path_mode == "uv":
            messages = self.output_layout.inverse(output_blocks, wigner_inv)
        else:
            messages = self.output_layout.inverse_channel_major(
                output_blocks,
                wigner_inv,
            )
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
