################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Tuple

import torch
from e3nn import o3

try:
    import openequivariance as oeq
except Exception:
    pass


def _to_oeq_irreps(irreps: o3.Irreps):
    return oeq.Irreps([(mul, (ir.l, ir.p)) for mul, ir in irreps])


class e3nnOeqTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Tuple,
        shared_weights: bool = False,
    ):
        super().__init__()

        dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
        tpp = oeq.TPProblem(
            _to_oeq_irreps(irreps_in1),
            _to_oeq_irreps(irreps_in2),
            _to_oeq_irreps(irreps_out),
            instructions,
            shared_weights=shared_weights,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        self.oeq_tp = oeq.TensorProduct(
            tpp,
            torch_op=True,
            use_opaque=False,
        )
        self.weight_numel = self.oeq_tp.weight_numel
        self.irreps_out = irreps_out

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return self.oeq_tp(x, y, w)


class e3nnOeqScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Tuple,
    ):
        super().__init__()

        dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
        tpp = oeq.TPProblem(
            _to_oeq_irreps(irreps_in1),
            _to_oeq_irreps(irreps_in2),
            _to_oeq_irreps(irreps_out),
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        self.oeq_tp = oeq.TensorProductConv(
            tpp,
            deterministic=False,
            kahan=False,
            torch_op=True,
            use_opaque=False,
        )
        self.weight_numel = self.oeq_tp.weight_numel
        self.irreps_out = irreps_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.oeq_tp(
            node_feats, edge_attrs, conv_weights, edge_index[1], edge_index[0]
        )  # target, source
