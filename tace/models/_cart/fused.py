################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from eqx import co3
from tace.utils.torch_scatter import scatter_sum

from .base import possible_irreps
from .paths import tensor_product_instructions


class uuuTensorProduct(torch.nn.Module):
    """Channel-wise Cartesian tensor product used by the product basis."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        l1l2=None,
        trainable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.irreps_in1 = co3.Irreps(irreps_in1)
        self.irreps_in2 = co3.Irreps(irreps_in2)
        self.irreps_out = co3.Irreps(irreps_out)
        instructions = tensor_product_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            mode="uuu",
            trainable=trainable,
            l1l2=l1l2,
        )
        self.tensor_product = co3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.tensor_product.weight_numel

    def forward(self, input1, input2, weight=None):
        return self.tensor_product(input1, input2, weight)


class O3ScatterTensorProduct(torch.nn.Module):
    """Edge Cartesian tensor product followed by target-node scatter."""

    def __init__(self, irreps_in1, irreps_in2, irreps_out, *, l1l2=None) -> None:
        super().__init__()
        self.irreps_in1 = co3.Irreps(irreps_in1)
        self.irreps_in2 = co3.Irreps(irreps_in2)
        self.irreps_out = co3.Irreps(irreps_out)
        instructions = tensor_product_instructions(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            mode="u1u",
            trainable=True,
            l1l2=l1l2,
        )
        self.tensor_product = co3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.tensor_product.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        messages = self.tensor_product(
            node_feats[source],
            edge_attrs,
            conv_weights,
        )
        return scatter_sum(
            messages,
            target,
            dim=0,
            dim_size=node_feats.size(0),
        )


class O3MagneticScatterTensorProduct(torch.nn.Module):
    """Couple node, edge, and magnetic fields before target-node scatter."""

    def __init__(
        self,
        irreps_node,
        irreps_edge,
        irreps_magnetic,
        irreps_out,
        *,
        l1l2=None,
    ) -> None:
        super().__init__()
        self.irreps_node = co3.Irreps(irreps_node)
        self.irreps_edge = co3.Irreps(irreps_edge)
        self.irreps_magnetic = co3.Irreps(irreps_magnetic)
        self.irreps_out = co3.Irreps(irreps_out)
        max_degree = self.irreps_out.lmax + self.irreps_magnetic.lmax
        intermediate_types = possible_irreps(
            self.irreps_node,
            self.irreps_edge,
            parity=True,
            lmax=max_degree,
        )
        multiplicity = self.irreps_node[0].mul
        if any(mul != multiplicity for _, mul in self.irreps_node):
            raise ValueError("Magnetic scatter requires equal node multiplicities.")
        self.irreps_intermediate = co3.Irreps(
            [(ir, multiplicity) for ir, _ in intermediate_types]
        )
        edge_instructions = tensor_product_instructions(
            self.irreps_node,
            self.irreps_edge,
            self.irreps_intermediate,
            mode="u1u",
            trainable=True,
            l1l2=l1l2,
        )
        magnetic_instructions = tensor_product_instructions(
            self.irreps_intermediate,
            self.irreps_magnetic,
            self.irreps_out,
            mode="u1u",
            trainable=True,
        )
        self.edge_product = co3.TensorProduct(
            self.irreps_node,
            self.irreps_edge,
            self.irreps_intermediate,
            edge_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.magnetic_product = co3.TensorProduct(
            self.irreps_intermediate,
            self.irreps_magnetic,
            self.irreps_out,
            magnetic_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.edge_product.weight_numel
        self.extra_weight_numel = self.magnetic_product.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        extra_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        intermediate = self.edge_product(node_feats[source], edge_attrs, conv_weights)
        messages = self.magnetic_product(
            intermediate,
            magnetic_attrs[source],
            extra_weights,
        )
        return scatter_sum(
            messages,
            target,
            dim=0,
            dim_size=node_feats.size(0),
        )


__all__ = [
    "O3MagneticScatterTensorProduct",
    "O3ScatterTensorProduct",
    "uuuTensorProduct",
]
