################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from eqx import co3
from tace.utils.torch_scatter import scatter_sum

from .paths import generate_paths


class uuuTensorProduct(torch.nn.Module):
    """Channel-wise tensor product used by the product basis."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        *,
        l1l2=None,
        trainable: bool = False,
        identical_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.path_irreps_in1 = co3.Irreps(irreps_in1)
        self.path_irreps_in2 = co3.Irreps(irreps_in2)
        self.irreps_in1 = self.path_irreps_in1.simplify()
        self.irreps_in2 = self.path_irreps_in2.simplify()
        self.irreps_out = co3.Irreps(irreps_out)
        instructions, actual_irreps_out = generate_paths(
            self.path_irreps_in1,
            self.path_irreps_in2,
            self.irreps_out,
            mode="uuu",
            trainable=trainable,
            l1l2=l1l2,
            identical_inputs=identical_inputs,
        )
        self.path_irreps_out = actual_irreps_out
        self.tp = co3.TensorProduct(
            self.path_irreps_in1,
            self.path_irreps_in2,
            self.path_irreps_out,
            instructions,
            project=True,
            simplify=True,
            internal_weights=False,
            shared_weights=False,
        )
        self.irreps_out = self.tp.irreps_out
        self.instructions = self.tp.instructions
        self.weight_numel = self.tp.weight_numel

    def forward(self, input1, input2, weight=None):
        return self.tp(input1, input2, weight)


class O3ScatterTensorProduct(torch.nn.Module):
    """Edge tensor product followed by target-node scatter."""

    def __init__(self, irreps_in1, irreps_in2, irreps_out, *, l1l2=None) -> None:
        super().__init__()
        self.irreps_in1 = co3.Irreps(irreps_in1)
        self.irreps_in2 = co3.Irreps(irreps_in2)
        requested_irreps_out = co3.Irreps(irreps_out)
        instructions, path_irreps_out = generate_paths(
            self.irreps_in1,
            self.irreps_in2,
            requested_irreps_out,
            mode="u1u",
            trainable=True,
            l1l2=l1l2,
        )
        self.tp = co3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            path_irreps_out,
            instructions,
            project=False,
            simplify=True,
            internal_weights=False,
            shared_weights=False,
        )
        self.irreps_out = self.tp.irreps_out
        self.instructions = self.tp.instructions
        self.weight_numel = self.tp.weight_numel

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        messages = self.tp(
            node_feats[source],
            edge_attrs,
            conv_weights,
        )
        messages = scatter_sum(
            messages,
            target,
            dim=0,
            dim_size=node_feats.size(0),
        )
        return messages


__all__ = ["O3ScatterTensorProduct", "uuuTensorProduct"]
