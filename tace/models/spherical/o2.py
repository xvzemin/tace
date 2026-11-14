################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Optional

import torch
from e3nn import o3

from equivariantx import conv as eqx_conv
from equivariantx import o2
from tace.utils.env import acceleration_enabled
from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import torchLinear
from ..softmax import GraphSoftmax
from ..utils import repr_without


class RadialRotaryComplexAttention(torch.nn.Module):
    def __init__(
        self,
        irreps: o2.Irreps,
        message_irreps: o2.Irreps,
        num_head: int,
        num_radial_basis: int,
    ) -> None:
        super().__init__()
        self.irreps = o2.Irreps(irreps)
        self.message_irreps = o2.Irreps(message_irreps)
        self.num_head = num_head
        for _, mul in self.irreps:
            if num_head < 1 or mul % num_head != 0:
                raise ValueError("num_head must divide every O2 multiplicity.")
        for _, mul in self.message_irreps:
            if mul % num_head != 0:
                raise ValueError("num_head must divide every message multiplicity.")
        self.q_proj = o2.Linear(
            self.irreps,
            self.irreps,
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
        )
        self.k_proj = o2.Linear(
            self.irreps,
            self.irreps,
            backend="cuda" if acceleration_enabled("eqx", kernel="linear") else "torch",
        )
        self.radial_proj = torchLinear(num_radial_basis, 2 * num_head)
        torch.nn.init.zeros_(self.radial_proj.weight)
        torch.nn.init.zeros_(self.radial_proj.bias)
        self.scale = math.sqrt(num_head / self.irreps.dim)
        self.graph_softmax = GraphSoftmax()

    def forward(
        self,
        message: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_radial_basis: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        query = self.q_proj(target_features)
        key = self.k_proj(source_features)
        score = edge_radial_basis.new_zeros(
            edge_radial_basis.size(0),
            self.num_head,
        )
        for (ir, mul), ir_slice in zip(self.irreps, self.irreps.slices()):
            shape = (query.size(0), ir.dim, self.num_head, mul // self.num_head)
            score = score + (
                query[..., ir_slice].reshape(shape) * key[..., ir_slice].reshape(shape)
            ).sum(dim=(1, 3))
        radial_scale, radial_shift = self.radial_proj(edge_radial_basis).chunk(
            2, dim=-1
        )
        score = score * self.scale * (2.0 * torch.sigmoid(radial_scale)) + radial_shift
        attention = (
            self.graph_softmax(
                score,
                edge_index[1],
                num_nodes=num_nodes,
                exp_rescale=edge_cutoff,
            )
            * edge_cutoff
        )

        outputs = []
        for (ir, mul), ir_slice in zip(
            self.message_irreps,
            self.message_irreps.slices(),
        ):
            shape = (
                message.size(0),
                ir.dim,
                self.num_head,
                mul // self.num_head,
            )
            output = message[..., ir_slice].reshape(shape)
            output = output * attention[:, None, :, None]
            outputs.append(output.reshape(message.size(0), ir.dim * mul))
        return torch.cat(outputs, dim=-1)


def uv_convolution(
    module,
    node_feats,
    magnetic_edge_attrs,
    conv_weights,
    edge_index,
    wigner,
    wigner_inv,
    edge_radial_basis,
    edge_cutoff,
    radial_network=None,
):
    """Call the shared UV kernel without changing checkpoint parameter ownership."""
    linears = [module.linear_up, module.linear_down]
    radial_attention = None
    if module.attention is not None:
        if edge_radial_basis is None:
            raise ValueError("O2 radial rotary attention requires edge_radial_basis.")
        linears.extend((module.attention.q_proj, module.attention.k_proj))
        radial_attention = module.attention.radial_proj(edge_radial_basis)
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
        radial_attention,
        radial_network=radial_network,
    )
    return module.reshape_out.inverse(message)


def uv_kernel(module):
    """Describe the native paths; unsupported activations keep their Torch path."""
    attention = module.attention
    try:
        return eqx_conv.UvO2TensorProductConv(
            module.local_frame_in,
            module.linear_up,
            module.nonlinearity,
            module.linear_down,
            module.local_frame_out,
            backend="cuda",
            frame_edge=getattr(module, "magnetic_frame", None),
            query=attention.q_proj if attention is not None else None,
            key=attention.k_proj if attention is not None else None,
            num_heads=module.num_head if attention is not None else 1,
            attention_scale=attention.scale if attention is not None else 1.0,
            eps=attention.graph_softmax.eps if attention is not None else 1e-16,
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
    num_head, num_radial_basis : int
        Attention heads and radial width used by :class:`UvRraO2TensorProductConv`.

    Examples
    --------
    >>> conv = UvO2TensorProductConv(
    ...     "2x0e", "2x0e", num_channel=2, mmax=0,
    ...     even_scalar_act=torch.nn.SiLU(), odd_scalar_act=torch.nn.Tanh(),
    ...     tensor_act=torch.nn.Sigmoid(), num_head=1, num_radial_basis=4,
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
        num_head: int,
        num_radial_basis: int,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_channel = num_channel
        self.mmax = min(self.irreps_in.lmax, mmax)
        self.num_head = num_head
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

        self.attention = self._build_attention(num_head, num_radial_basis)
        self.eqx_tp = uv_kernel(self)

    def _build_attention(self, num_head, num_radial_basis):
        return None

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
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
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
        if self.attention is not None:
            if edge_radial_basis is None:
                raise ValueError(
                    "O2 radial rotary attention requires edge_radial_basis."
                )
            message = self.attention(
                message,
                source_features,
                target_features,
                edge_radial_basis,
                edge_index,
                edge_cutoff,
                num_nodes,
            )
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
        if self.attention is None:
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
        edge_radial_basis: Optional[torch.Tensor] = None,
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
                edge_radial_basis,
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
            edge_index,
            edge_radial_basis,
            edge_cutoff,
            node_features.size(0),
        )
        return self._to_global(
            message,
            edge_index,
            wigner_inv,
            edge_cutoff,
            node_features.size(0),
        )


class UvRraO2TensorProductConv(UvO2TensorProductConv):
    """Local O(2) UV convolution with radial rotary attention.

    Parameters are those of :class:`UvO2TensorProductConv`. Attention uses
    ``num_head`` heads and ``num_radial_basis`` radial features. Scalar-only
    inputs retain the ordinary UV convolution.

    Examples
    --------
    >>> conv = UvRraO2TensorProductConv(
    ...     "2x0e+2x1o", "2x0e+2x1o", num_channel=2, mmax=1,
    ...     even_scalar_act=torch.nn.SiLU(), odd_scalar_act=torch.nn.Tanh(),
    ...     tensor_act=torch.nn.Sigmoid(), num_head=1, num_radial_basis=4,
    ... )
    >>> isinstance(conv.attention, RadialRotaryComplexAttention)
    True
    """

    def _build_attention(self, num_head, num_radial_basis):
        if self.node_irreps.mmax == 0:
            return None
        return RadialRotaryComplexAttention(
            self.node_irreps, self.local_irreps_out, num_head, num_radial_basis
        )
