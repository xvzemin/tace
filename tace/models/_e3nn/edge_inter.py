################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

from eqx import o2

from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP
from .base import EdgeInteraction
from .fused import UvuTensorProduct
from .paths import generate_paths


class O3CgtpEdgeInteraction(EdgeInteraction):
    """Couple edge features with supplied spherical harmonics using CGTP."""

    def _setup(self) -> None:
        self.irreps_hidden = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_in.regroup()]
        )
        self.linear_up = e3nnLinear(self.irreps_in, self.irreps_hidden, bias=False)
        self._setup_convolution()

        radial_basis = self.radial_basis
        radial_norm = radial_basis["mlp_norm"] or "layer_norm"
        self.edge_info = MLP[radial_basis["mlp_type"]](
            [radial_basis["num_radial_basis"] + 2 * self.num_elements]
            + list(radial_basis["hidden"])
            + [self.weight_numel],
            bias=radial_basis["bias"],
            layer_norm=radial_norm == "layer_norm",
            rms_norm=radial_norm == "rms_norm",
            act="silu",
        )

    def _setup_convolution(self) -> None:
        instructions, self.irreps_out = generate_paths(
            self.irreps_out, self.irreps_hidden, self.irreps_sh
        )
        self.rejector = UvuTensorProduct(
            self.irreps_hidden,
            self.irreps_sh,
            self.irreps_out,
            instructions,
            shared_weights=False,
        )
        self.weight_numel = self.rejector.weight_numel

    def convolve(
        self,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        weights: torch.Tensor,
        edge_wigner: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.rejector(edge_feats, edge_attrs, weights)

    def forward(
        self,
        edge_feats: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        edge_cutoff: torch.Tensor,
        edge_wigner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_feats = self.linear_up(edge_feats)
        weights = self.edge_info(edge_radial_basis)
        return self.convolve(edge_feats, edge_attrs, weights * edge_cutoff, edge_wigner)


class O2EdgeInteraction(O3CgtpEdgeInteraction):
    """Apply radially weighted O(2) Linear--Gate--Linear in an edge-aligned frame."""

    def _setup_convolution(self) -> None:
        self.irreps_out = o3.Irreps(
            [(self.num_channel, ir) for _, ir in self.irreps_out]
        )
        self.local_frame_in = o2.LocalFrame(self.irreps_hidden)
        self.local_frame_out = o2.LocalFrame(self.irreps_out, reverse=True)
        self.reshape_in = LayoutTransform(
            self.irreps_hidden, layout_out="flatten_ir_mul"
        )
        self.reshape_out = LayoutTransform(self.irreps_out, layout_out="flatten_ir_mul")
        local_irreps_in = self.local_frame_in.irreps_out
        local_irreps_out = self.local_frame_out.irreps_out
        hidden_irreps = local_irreps_out.filter(
            keep=lambda entry: local_irreps_in.count(entry.ir) > 0
        )
        irreps_scalars = hidden_irreps.filter(keep=lambda entry: entry.ir.m == 0)
        irreps_gated = hidden_irreps.filter(keep=lambda entry: entry.ir.m > 0)
        irreps_gates = (
            o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
            if irreps_gated.num_irreps
            else o2.Irreps()
        )
        gate = o2.Gate(
            irreps_scalars,
            [
                torch.nn.SiLU() if ir.is_invariant_scalar() else torch.nn.Tanh()
                for ir, _ in irreps_scalars
            ],
            irreps_gates,
            [torch.nn.Sigmoid()] if len(irreps_gates) else [],
            irreps_gated,
        )
        self.rejector = torch.nn.Sequential(
            o2.Linear(local_irreps_in, gate.irreps_in),
            gate,
            o2.Linear(gate.irreps_out, local_irreps_out),
        )
        indices = []
        offset = 0
        for ir, mul in self.local_frame_in.irreps_out:
            indices.append(torch.arange(offset, offset + mul).repeat(ir.dim))
            offset += mul
        self.weight_numel = offset
        self.register_buffer("weight_index", torch.cat(indices), persistent=False)

    def convolve(
        self,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor | None,
        weights: torch.Tensor,
        edge_wigner: torch.Tensor | None,
    ) -> torch.Tensor:
        local = self.local_frame_in.to_local(self.reshape_in(edge_feats), edge_wigner)
        local = self.rejector(local * weights.index_select(-1, self.weight_index))
        return self.reshape_out.inverse(
            self.local_frame_out.to_global(local, edge_wigner)
        )


EDGE_INTERACTION: dict[str, type[EdgeInteraction]] = {
    "cgtp": O3CgtpEdgeInteraction,
    "o2": O2EdgeInteraction,
}
