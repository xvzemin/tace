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
    return o2.Irreps(groups)


class _O3O2Layout(torch.nn.Module):
    """Rotate one channel-separated O(3) layout to and from local O(2)."""

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

        local_indices = [[degree] for degree in range(lmax + 1)]
        offset = lmax + 1
        for order in range(1, lmax + 1):
            count = lmax + 1 - order
            for degree in range(order, lmax + 1):
                index = degree - order
                local_indices[degree].extend((offset + index, offset + count + index))
            offset += 2 * count

        for degree, indices in enumerate(local_indices):
            self.register_buffer(
                f"local_indices_{degree}",
                torch.tensor(indices, dtype=torch.int64),
                persistent=False,
            )
            basis = torch.eye(2 * degree + 1)
            quarter_turn = basis.new_tensor([[0.0, -1.0], [1.0, 0.0]])
            for order in range(1, degree + 1):
                start = 1 + 2 * (order - 1)
                basis[start : start + 2, start : start + 2] = quarter_turn
            self.register_buffer(f"odd_basis_{degree}", basis, persistent=False)

    def forward(self, input: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor:
        input = self.layout(input)
        blocks = []
        for (_, irrep), component_slice in zip(self.irreps, self.group_slices):
            indices = getattr(self, f"local_indices_{irrep.l}")
            start = irrep.l**2
            stop = (irrep.l + 1) ** 2
            rotation = wigner.index_select(1, indices)[:, :, start:stop]
            block = torch.bmm(rotation, input[:, component_slice])
            if irrep.p * ((-1) ** irrep.l) == -1:
                basis = getattr(self, f"odd_basis_{irrep.l}").to(block)
                block = torch.matmul(basis, block)
            blocks.append(block)
        return torch.cat(blocks, dim=1)

    def inverse(self, input: torch.Tensor, wigner_inv: torch.Tensor) -> torch.Tensor:
        blocks = []
        for (_, irrep), component_slice in zip(self.irreps, self.group_slices):
            indices = getattr(self, f"local_indices_{irrep.l}")
            start = irrep.l**2
            stop = (irrep.l + 1) ** 2
            rotation = wigner_inv[:, start:stop].index_select(2, indices)
            block = input[:, component_slice]
            if irrep.p * ((-1) ** irrep.l) == -1:
                basis = getattr(self, f"odd_basis_{irrep.l}").to(block)
                block = torch.matmul(basis.transpose(0, 1), block)
            blocks.append(torch.bmm(rotation, block))
        return self.layout.inverse(torch.cat(blocks, dim=1))


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
        )
        self.irreps_out_local = self.output_layout.local_irreps

        if path_mode == "uv":
            identity_paths = tuple(
                (index, index) for index in range(len(self.irreps_in_local.expanded()))
            )
            self.radial_linear = o2.Linear(
                self.irreps_in_local,
                self.irreps_in_local,
                num_channel,
                path_mode="uu",
                internal_weights=False,
                bias=False,
                path_norm=False,
                path=identity_paths,
            )
            self.linear = o2.Linear(
                self.irreps_in_local,
                self.irreps_out_local,
                num_channel,
                path_mode="uv",
                bias=False,
            )
            self.weight_numel = self.radial_linear.weight_numel
        else:
            self.linear = o2.Linear(
                self.irreps_in_local,
                self.irreps_out_local,
                num_channel,
                path_mode="uu",
                internal_weights=False,
                bias=False,
            )
            self.weight_numel = self.linear.weight_numel

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
        node_local = self.node_layout(node_feats[source], wigner)
        magnetic_local = self.magnetic_layout(
            magnetic_node_attrs[source].unsqueeze(-1),
            wigner,
        ).expand(-1, -1, self.num_channel)
        input_local = torch.cat((node_local, magnetic_local), dim=1)

        if self.path_mode == "uv":
            radial_weights = radial_weights.reshape(
                radial_weights.size(0),
                *self.radial_linear.weight_shape,
            )
            input_local = self.radial_linear(input_local, radial_weights)
            output_local = self.linear(input_local)
        else:
            radial_weights = radial_weights.reshape(
                radial_weights.size(0),
                *self.linear.weight_shape,
            )
            output_local = self.linear(input_local, radial_weights)

        messages = self.output_layout.inverse(output_local, wigner_inv)
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
