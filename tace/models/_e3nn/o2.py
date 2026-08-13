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


def _common_multiplicity(irreps: o3.Irreps, name: str) -> int:
    multiplicities = {multiplicity for multiplicity, _ in irreps}
    if len(multiplicities) != 1:
        raise ValueError(f"{name} must use one common multiplicity per irrep.")
    return next(iter(multiplicities))


class O2MagneticScatterLinear(torch.nn.Module):
    """Edge-aligned O2 magnetic gated convolution.

    The source node and source magnetic solid harmonics are concatenated in
    the local representation. Edge weights first apply a diagonal
    channel-wise radial map. Two internal ``uv`` linears surround an
    :class:`O2Gate`, giving the local ``uv -> gate -> uv`` pattern.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps,
        *,
        num_channel: int,
        lmax: int,
        act_0e: torch.nn.Module,
        act_0o: Union[torch.nn.Module, None],
        act_lm: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_irreps = o3.Irreps(magnetic_irreps)
        self.num_channel = num_channel
        if _common_multiplicity(self.irreps_node, "irreps_node") != num_channel:
            raise ValueError("irreps_node multiplicity must equal num_channel.")
        if _common_multiplicity(self.irreps_out, "irreps_out") != num_channel:
            raise ValueError("irreps_out multiplicity must equal num_channel.")

        self.node_layout = o2.O3O2Layout(self.irreps_node, lmax)
        self.magnetic_layout = o2.O3O2Layout(self.magnetic_irreps, lmax)
        self.output_layout = o2.O3O2Layout(self.irreps_out, lmax)
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

        self.gate = o2.O2Gate(
            self.irreps_out_local,
            act_0e=act_0e,
            act_0o=act_0o,
            act_lm=act_lm,
        )
        self.linear_in = o2.Linear(
            self.irreps_in_local,
            self.gate.irreps_in,
            num_channel,
            path_mode="uv",
            bias=False,
        )
        self.linear_out = o2.Linear(
            self.gate.irreps_out,
            self.irreps_out_local,
            num_channel,
            path_mode="uv",
            bias=False,
        )
        self.weight_numel = self.irreps_in_local.num_irreps * num_channel

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
        if offset != radial_weights.size(-1):
            raise ValueError("Invalid O2 magnetic radial weight size.")

        hidden_blocks = self.linear_in.forward_grouped(tuple(weighted_blocks))
        hidden_blocks = self.gate.forward_grouped(hidden_blocks)
        output_blocks = self.linear_out.forward_grouped(hidden_blocks)
        messages = self.output_layout.inverse(output_blocks, wigner_inv)
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
