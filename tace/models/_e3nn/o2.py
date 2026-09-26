################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import math
from typing import Optional

import torch
from e3nn import o3

from eqx import o2
from eqx.conv import UuO2TensorProductConv, UvO2TensorProductConv
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
        self.q_proj = o2.Linear(self.irreps, self.irreps)
        self.k_proj = o2.Linear(self.irreps, self.irreps)
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
    )
    return module.reshape_out.inverse(message)


def uv_kernel(module):
    """Describe the native paths; unsupported activations keep their Torch path."""
    attention = module.attention
    try:
        return UvO2TensorProductConv(
            module.local_frame_in,
            module.linear_up,
            module.nonlinearity,
            module.linear_down,
            module.local_frame_out,
            frame_edge=getattr(module, "magnetic_frame", None),
            query=attention.q_proj if attention is not None else None,
            key=attention.k_proj if attention is not None else None,
            num_heads=module.num_head if attention is not None else 1,
            attention_scale=attention.scale if attention is not None else 1.0,
            eps=attention.graph_softmax.eps if attention is not None else 1e-16,
        )
    except NotImplementedError:
        return None


class O2ScatterTensorProduct(torch.nn.Module):
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
        use_radial_rotary_attention: bool,
        linear_type: str = "uv",
    ) -> None:
        super().__init__()
        if linear_type not in ("uv", "uu"):
            raise ValueError("linear_type must be 'uv' or 'uu'.")
        if linear_type == "uu" and use_radial_rotary_attention:
            raise ValueError("uu_o2 does not support radial rotary attention.")
        self.linear_type = linear_type
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
        self.local_irreps_in = (
            self.node_irreps if linear_type == "uu" else 2 * self.node_irreps
        )
        self.local_irreps_out = self.local_frame_out.irreps_out
        if linear_type == "uu":
            self.linear = o2.UuLinear(
                self.local_irreps_in,
                self.local_irreps_out,
                num_channel,
            )
            self.weight_numel = self.linear.weight_numel
            self.eqx_tp = UuO2TensorProductConv(
                self.local_frame_in, self.linear, self.local_frame_out
            )
        else:
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
            )
            self.linear_up = o2.Linear(
                self.local_irreps_in,
                self.nonlinearity.irreps_in,
            )
            self.linear_down = o2.Linear(
                self.nonlinearity.irreps_out,
                self.local_irreps_out,
            )
            self.weight_numel = self.local_irreps_in.num_irreps

        self.use_radial_rotary_attention = (
            use_radial_rotary_attention and self.node_irreps.mmax > 0
        )
        self.attention = (
            RadialRotaryComplexAttention(
                self.node_irreps,
                self.local_irreps_out,
                num_head,
                num_radial_basis,
            )
            if self.use_radial_rotary_attention
            else None
        )
        if self.linear_type == "uv":
            self.eqx_tp = uv_kernel(self)

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def forward_stream(
        self, node_feats, radial, projection, edge_index, wigner, edge_cutoff, graph
    ):
        """Fuse the channelwise paths without retaining edge features or weights."""
        vectors = graph.edge_vector if graph is not None else None
        if wigner.ndim == 3:
            # Mixed interactions may still use order-major, truncated frames.
            # Keep their matrix derivatives, including the zero-padded rows.
            lmax = math.isqrt(wigner.size(-1)) - 1
            blocks = []
            for l in range(
                max(self.local_frame_in.lmax, self.local_frame_out.lmax) + 1
            ):
                retained = min(l, self.mmax)
                rows = [
                    l
                    if m == 0
                    else l + (2 * m - 1) * (lmax + 1) - m * m
                    if m > 0
                    else l + 2 * (-m) * (lmax + 1) - (-m) * (-m + 1)
                    for m in range(-retained, retained + 1)
                ]
                block = wigner[:, rows, l * l : (l + 1) ** 2]
                blocks.append(
                    torch.nn.functional.pad(
                        block, (0, 0, l - retained, l - retained)
                    ).flatten(1)
                )
            wigner = torch.cat(blocks, dim=1)
            vectors = None
        message = self.eqx_tp(
            self.reshape_in(node_feats),
            radial,
            projection,
            wigner,
            edge_cutoff,
            edge_index,
            node_feats.size(0),
            vectors=vectors,
        )
        return self.reshape_out.inverse(message)

    def _to_local(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        node_features = self.reshape_in(node_features)
        if self.linear_type == "uu":
            source_features = self.local_frame_in.to_local(
                node_features[edge_index[0]],
                wigner,
            )
            return node_features, source_features, None
        paired = self.local_frame_in.to_local(
            node_features[edge_index.T],
            wigner,
        )
        return node_features, paired[:, 0], paired[:, 1]

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: Optional[torch.Tensor],
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if self.linear_type == "uu":
            return self.linear(source_features, conv_weights)
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
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if (
            self.linear_type == "uv"
            and getattr(self, "eqx_tp", None) is not None
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


class O2ScatterMagneticTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_edge_irreps: o3.Irreps,
        *,
        num_channel: int,
        mmax: int,
        even_scalar_act: torch.nn.Module,
        odd_scalar_act: Optional[torch.nn.Module],
        tensor_act: torch.nn.Module,
        num_head: int,
        num_radial_basis: int,
        use_radial_rotary_attention: bool,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.magnetic_edge_irreps = o3.Irreps(magnetic_edge_irreps)
        self.num_channel = num_channel
        self.mmax = min(
            max(self.irreps_in.lmax, self.magnetic_edge_irreps.lmax),
            mmax,
        )
        self.num_head = num_head
        if any(entry.mul != num_channel for entry in self.irreps_in + self.irreps_out):
            raise ValueError("irreps_in/out multiplicity must equal num_channel.")
        if any(entry.mul != num_channel for entry in self.magnetic_edge_irreps):
            raise ValueError(
                "magnetic_edge_irreps multiplicity must equal num_channel."
            )

        self.local_frame_in = o2.LocalFrame(self.irreps_in, mmax=self.mmax)
        self.local_frame_out = o2.LocalFrame(
            self.irreps_out,
            mmax=self.mmax,
            reverse=True,
        )
        self.magnetic_frame = o2.LocalFrame(
            self.magnetic_edge_irreps,
            mmax=self.mmax,
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
        self.reshape_magnetic = LayoutTransform(
            self.magnetic_edge_irreps,
            layout_in="flatten_mul_ir",
            layout_out="flatten_ir_mul",
        )
        self.node_irreps = self.local_frame_in.irreps_out
        self.local_magnetic_irreps = self.magnetic_frame.irreps_out
        self.local_irreps_in = (
            self.node_irreps + self.node_irreps + self.local_magnetic_irreps
        ).regroup()
        self.local_irreps_out = self.local_frame_out.irreps_out
        self.use_time_reversal = any(ir.t == -1 for ir, _ in self.local_irreps_in)
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
        if self.use_time_reversal:
            gate_entries = [(o2.Irrep("0ee"), mul) for _, mul in gated_entries]
            gate_acts = [tensor_act] * len(gate_entries)
            time_odd_scalars = [
                ir for ir, _ in self.local_irreps_in if ir.m == 0 and ir.t == -1
            ]
            time_odd_irreps = [ir for ir, _ in self.local_irreps_in if ir.t == -1]
            for ir_out, mul in self.local_irreps_out:
                path = next(
                    (
                        (ir_gate, ir_gated)
                        for ir_gate in time_odd_scalars
                        for ir_gated in time_odd_irreps
                        if ir_gated * ir_gate == (ir_out,)
                    ),
                    None,
                )
                if path is not None:
                    ir_gate, ir_gated = path
                    gate_entries.append((ir_gate, mul))
                    gate_acts.append(odd_scalar_act)
                    gated_entries.append((ir_gated, mul))
            irreps_gated = o2.Irreps(gated_entries)
            irreps_gates = o2.Irreps(gate_entries)
        else:
            irreps_gates = (
                o2.Irreps([(o2.Irrep("0ee"), irreps_gated.num_irreps)])
                if irreps_gated.num_irreps
                else o2.Irreps()
            )
            gate_acts = [tensor_act] if len(irreps_gates) else []
        self.nonlinearity = o2.Gate(
            o2.Irreps(scalar_entries),
            scalar_acts,
            irreps_gates,
            gate_acts,
            irreps_gated,
        )
        self.linear_up = o2.Linear(
            self.local_irreps_in,
            self.nonlinearity.irreps_in,
            biases=True,
        )
        self.linear_down = o2.Linear(
            self.nonlinearity.irreps_out,
            self.local_irreps_out,
            biases=True,
        )
        self.weight_numel = self.local_irreps_in.num_irreps
        self.use_radial_rotary_attention = (
            use_radial_rotary_attention and self.node_irreps.mmax > 0
        )
        self.attention = (
            RadialRotaryComplexAttention(
                self.node_irreps,
                self.local_irreps_out,
                num_head,
                num_radial_basis,
            )
            if self.use_radial_rotary_attention
            else None
        )
        self.eqx_tp = uv_kernel(self)

    def __repr__(self) -> str:
        return repr_without(self, "reshape_in", "reshape_out")

    def _to_local(
        self,
        node_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self.reshape_in(node_features)
        magnetic_edge_attrs = self.reshape_magnetic(magnetic_edge_attrs)
        paired_node = self.local_frame_in.to_local(node_features[edge_index.T], wigner)
        magnetic_edge_attrs = self.magnetic_frame.to_local(
            magnetic_edge_attrs,
            wigner,
        )
        return (
            node_features,
            paired_node[:, 0],
            paired_node[:, 1],
            magnetic_edge_attrs,
        )

    def _convolution(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        edge_radial_basis: Optional[torch.Tensor],
        edge_cutoff: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weighted = []
        offset = 0
        feature_sets = (
            (target_features, self.node_irreps),
            (source_features, self.node_irreps),
            (magnetic_edge_attrs, self.local_magnetic_irreps),
        )
        for ir, mul in self.local_irreps_in:
            inputs = []
            for features, irreps in feature_sets:
                for (input_ir, input_mul), ir_slice in zip(
                    irreps,
                    irreps.slices(),
                ):
                    if input_ir == ir:
                        inputs.append(
                            features[..., ir_slice].view(
                                features.size(0), ir.dim, input_mul
                            )
                        )
            values = torch.cat(inputs, dim=-1)
            weight = conv_weights[..., offset : offset + mul].unsqueeze(-2)
            weighted.append((values * weight).reshape(values.size(0), ir.dim * mul))
            offset += mul
        if offset != conv_weights.size(-1):
            raise ValueError("Invalid O2 convolution weight size.")
        projected = self.linear_up(torch.cat(weighted, dim=-1))
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
        magnetic_edge_attrs: torch.Tensor,
        conv_weights: torch.Tensor,
        edge_index: torch.Tensor,
        wigner: Optional[torch.Tensor],
        wigner_inv: Optional[torch.Tensor],
        edge_radial_basis: Optional[torch.Tensor] = None,
        edge_cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_cutoff is None:
            raise ValueError("O2 convolution requires edge_cutoff.")
        if wigner is None or wigner_inv is None:
            raise ValueError("O2 convolution requires Wigner matrices.")
        if (
            getattr(self, "eqx_tp", None) is not None
            and node_feats.is_cuda
            and node_feats.dtype in (torch.float32, torch.float64)
            and acceleration_enabled("eqx", kernel="conv")
        ):
            return uv_convolution(
                self,
                node_feats,
                magnetic_edge_attrs,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_radial_basis,
                edge_cutoff,
            )
        (
            node_features,
            source_features,
            target_features,
            magnetic_edge_attrs,
        ) = self._to_local(node_feats, magnetic_edge_attrs, edge_index, wigner)
        message = self._convolution(
            source_features,
            target_features,
            magnetic_edge_attrs,
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
