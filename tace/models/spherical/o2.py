################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional

import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx import o2
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..utils import repr_without


def uv_convolution(
    module,
    node_feats,
    magnetic_edge_attrs,
    conv_weights,
    edge_index,
    wigner,
    wigner_inv,
    edge_cutoff,
    radial_network=None,
):
    """Evaluate radial weighting and the fused local O(2) MLP."""
    linears = [module.linear_up, module.linear_down]
    message = module.eqx_tp(
        module.reshape_in(node_feats),
        module.reshape_magnetic(magnetic_edge_attrs)
        if magnetic_edge_attrs is not None
        else None,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff,
        tuple(value for linear in linears for value in (linear.weight, linear.bias)),
        radial_network=radial_network,
    )
    return module.reshape_out.inverse(message)


def uv_kernel(module):
    """Describe the native paths; unsupported activations keep their Torch path."""
    try:
        return eqx_conv.UvO2TensorProductConv(
            module.local_frame_in,
            [module.linear_up, module.linear_down],
            [module.nonlinearity],
            module.local_frame_out,
            backend="cuda",
            radial_mode="full",
            frame_edge=getattr(module, "magnetic_frame", None),
        )
    except NotImplementedError:
        return None


class UvO2TensorProductConv(torch.nn.Module):
    """Local O(2) Linear → Gate → Linear convolution.

    Parameters
    ----------
    irreps_in, irreps_out : e3nn.o3.Irreps
        Input and output representations.
    num_channel : int
        Multiplicity of each representation.
    mmax : int
        Maximum local order.
    even_scalar_act, odd_scalar_act, tensor_act : torch.nn.Module
        Even-scalar, odd-scalar and gate activations. ``None`` gates odd scalars.

    Examples
    --------
    >>> conv = UvO2TensorProductConv(
    ...     "2x0e", "2x0e", num_channel=2, mmax=0,
    ...     even_scalar_act=torch.nn.SiLU(), odd_scalar_act=torch.nn.Tanh(),
    ...     tensor_act=torch.nn.Sigmoid(),
    ... )
    >>> conv.weight_numel
    4
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        num_channel: int,
        mmax: int,
        even_scalar_act: torch.nn.Module,
        odd_scalar_act: Optional[torch.nn.Module],
        tensor_act: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channel = num_channel
        self.mmax = min(self.irreps_in.lmax, mmax)
        if any(entry.mul != num_channel for entry in self.irreps_in + self.irreps_out):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")
        self.local_frame_in = o2.LocalFrame(self.irreps_in, mmax=self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            mmax=self.mmax,
            reverse=True,
        )
        self.reshape_in = LayoutTransform(
            self.irreps_in,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.reshape_out = LayoutTransform(
            self.irreps_out,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_irreps_in = 2 * self.node_irreps
        self.local_irreps_out = self.local_frame_out.irreps_out
        hidden_irreps = self.local_irreps_out.filter(
            keep=lambda ir_mul: self.local_irreps_in.count(ir_mul.ir) > 0
        )
        scalar_entries = []
        scalar_acts = []
        gated_entries = []
        for ir, mul in hidden_irreps:
            if ir.is_invariant_scalar():
                scalar_entries.append((ir, mul))
                scalar_acts.append(even_scalar_act)
            elif ir.m == 0 and odd_scalar_act is not None:
                scalar_entries.append((ir, mul))
                scalar_acts.append(odd_scalar_act)
            else:
                gated_entries.append((ir, mul))
        irreps_gated = o2.Irreps(gated_entries)
        irreps_gates = (
            o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
            if irreps_gated.num_irreps
            else o2.Irreps()
        )
        self.nonlinearity = o2.Gate(
            o2.Irreps(scalar_entries),
            scalar_acts,
            irreps_gates,
            [tensor_act] if len(irreps_gates) else [],
            irreps_gated,
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
        )
        self.linear_up = o2.Linear(
            self.local_irreps_in,
            self.nonlinearity.irreps_in,
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
        )
        self.linear_down = o2.Linear(
            self.nonlinearity.irreps_out,
            self.local_irreps_out,
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
        )
        self.weight_numel = self.local_irreps_in.num_irreps

        self.eqx_tp = uv_kernel(self)

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def _to_local(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        paired = self.local_frame_in.to_local(
            node_features[edge_index.T],
            wigner,
        )
        return node_features, paired[:, 0], paired[:, 1]

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        inputs = []
        offset = 0
        for (ir, mul), ir_slice in zip(
            self.node_irreps,
            self.node_irreps.slices(),
        ):
            values = torch.cat(
                (
                    target_features[..., ir_slice].view(
                        target_features.size(0), ir.dim, mul
                    ),
                    source_features[..., ir_slice].view(
                        source_features.size(0), ir.dim, mul
                    ),
                ),
                dim=-1,
            )
            width = 2 * mul
            weight = conv_weights[..., offset : offset + width].unsqueeze(-2)
            values = values * weight
            inputs.append(values.reshape(values.size(0), ir.dim * width))
            offset += width
        features = torch.cat(inputs, dim=-1)
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")
        projected = self.linear_up(features)
        hidden = self.nonlinearity(projected)
        message = self.linear_down(hidden)
        return message

    def _to_global(
        self,
        message: torch.Tensor,
        edge_index: torch.Tensor,
        wigner_inv: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        message = self.local_frame_out.to_global(message, wigner_inv)
        message = message * edge_cutoff
        message = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return self.reshape_out.inverse(message)

    def forward(
        self,
        node_feats: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Optional[torch.Tensor],
        wigner_inv: Optional[torch.Tensor],
        edge_cutoff: Optional[torch.Tensor] = None,
        radial_network=None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if (
            getattr(self, "eqx_tp", None) is not None
            and node_feats.is_cuda
            and node_feats.dtype in (torch.float32, torch.float64)
            and acceleration_enabled("eqx", kernel="conv")
        ):
            return uv_convolution(
                self,
                node_feats,
                None,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff,
                radial_network=radial_network,
            )
        node_features, source_features, target_features = self._to_local(
            node_feats, edge_index, wigner
        )
        message = self._convolution(
            source_features,
            target_features,
            conv_weights,
        )
        return self._to_global(
            message,
            edge_index,
            wigner_inv,
            edge_cutoff,
            node_features.size(0),
        )
