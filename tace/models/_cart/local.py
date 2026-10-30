"""Cartesian local-frame interactions."""

import math

import torch

from eqx import co2, co3, o2
from eqx.o2.rotation_matrix import rotation_matrix_to_y_axis
from eqx.ops.radial import materialize
from tace.models._e3nn.tece_oam_rra import Convolution
from tace.models.layout import LayoutTransform
from tace.utils.torch_scatter import scatter_sum


class RotationMatrices(torch.nn.Module):
    """Return proper alignment matrices for Cartesian tensor contractions."""

    def forward(self, vectors):
        rotation = rotation_matrix_to_y_axis(vectors)
        return rotation, rotation

    def forward_packed(self, vectors):
        return rotation_matrix_to_y_axis(vectors)


def linear(module):
    """Copy a circular channel map into Cartesian storage."""
    result = co2.Linear(
        module.irreps_in,
        module.irreps_out,
        internal_weights=module.internal_weights,
        shared_weights=module.shared_weights,
        instructions=[(p.i_in, p.i_out) for p in module.instructions if p.i_in >= 0],
        biases=[
            any(p.i_in < 0 and p.i_out == j for p in module.instructions)
            for j in range(len(module.irreps_out))
        ],
        path_normalization=module.path_normalization,
    )
    result.weight, result.bias = module.weight, module.bias
    return result


def gate(module):
    """Preserve normalized scalar functions in a Cartesian gate."""
    result = co2.Gate(
        module.irreps_scalars,
        [None] * len(module.act_scalars.acts),
        module.irreps_gates,
        [None] * len(module.act_gates.acts),
        module.irreps_gated,
    )
    result.act_scalars, result.act_gates = module.act_scalars, module.act_gates
    result.mul = co2.ElementwiseTensorProduct(
        result.irreps_gated, co2.Irreps(module.act_gates.irreps_out)
    )
    result.irreps_out = (
        co2.Irreps(module.act_scalars.irreps_out) + result.mul.irreps_out
    )
    return result


def inverse_scale(frame):
    """Return the retained-order variance normalization in Cartesian storage."""
    return torch.cat(
        [
            torch.full(
                (mul * ir.dim,),
                math.sqrt((2 * ir.l + 1) / (2 * min(ir.l, frame.mmax) + 1)),
                dtype=torch.float64,
            )
            for mul, ir in frame.irreps_in
        ]
    )


class RadialAttention(torch.nn.Module):
    """Apply invariant radial attention to Cartesian planar features."""

    def __init__(self, reference):
        super().__init__()
        self.irreps = co2.Irreps(reference.irreps)
        self.message_irreps = co2.Irreps(reference.message_irreps)
        self.q_proj, self.k_proj = linear(reference.q_proj), linear(reference.k_proj)
        self.radial_proj = reference.radial_proj
        self.graph_softmax = reference.graph_softmax
        self.num_head, self.scale = reference.num_head, reference.scale

    def forward(self, message, source, target, radial, edge_index, cutoff, num_nodes):
        query, key = self.q_proj(target), self.k_proj(source)
        score = radial.new_zeros(radial.shape[0], self.num_head)
        for (mul, ir), section in zip(self.irreps, self.irreps.slices()):
            shape = (query.shape[0], self.num_head, mul // self.num_head, ir.dim)
            score = score + (
                query[:, section].reshape(shape) * key[:, section].reshape(shape)
            ).sum(dim=(-2, -1))
        scale, shift = self.radial_proj(radial).chunk(2, dim=-1)
        score = score * self.scale * (2 * torch.sigmoid(scale)) + shift
        attention = (
            self.graph_softmax(
                score, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff
            )
            * cutoff
        )
        values = []
        for (mul, ir), section in zip(
            self.message_irreps, self.message_irreps.slices()
        ):
            value = message[:, section].reshape(
                message.shape[0], self.num_head, mul // self.num_head, ir.dim
            )
            values.append((value * attention[:, :, None, None]).flatten(1))
        return torch.cat(values, dim=-1)


class UuConvolution(torch.nn.Module):
    """Apply a channel-wise local map to Cartesian source features."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(self.irreps_in, reference.mmax)
        self.local_frame_out = co2.LocalFrame(self.irreps_out, reference.mmax)
        self.linear = co2.UuLinear(
            self.local_frame_in.irreps_out,
            self.local_frame_out.irreps_out,
            reference.num_channel,
        )
        self.weight_numel = reference.weight_numel
        self.eqx_tp = None
        self.register_buffer(
            "scale", inverse_scale(self.local_frame_out), persistent=False
        )

    def forward(self, features, weights, edge_index, rotation, rotation_inv, cutoff):
        local = self.local_frame_in(features[edge_index[0]], rotation)
        local = self.linear(local, weights)
        message = (
            self.local_frame_out.to_global(local, rotation_inv) * self.scale * cutoff
        )
        return scatter_sum(message, edge_index[1], dim=0, dim_size=features.shape[0])

    def forward_stream(
        self,
        features,
        radial,
        projection,
        edge_index,
        rotation,
        cutoff,
        graph,
        *,
        radial_network=None,
    ):
        """Evaluate radial coefficients without changing Cartesian storage."""
        radial = materialize(radial, edge_index)
        if radial_network is not None:
            radial = radial_network(radial)
        if radial.shape[-1] + 1 == projection.shape[0]:
            radial = torch.cat((radial, torch.ones_like(radial[..., :1])), dim=-1)
        return self(
            features, radial @ projection, edge_index, rotation, rotation, cutoff
        )


class UvConvolution(torch.nn.Module):
    """Apply radial weighting, channel mixing and gates in Cartesian O(2)."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(self.irreps_in, reference.mmax)
        self.local_frame_out = co2.LocalFrame(self.irreps_out, reference.mmax)
        self.node_irreps = self.local_frame_in.irreps_out
        self.linear_up, self.linear_down = (
            linear(reference.linear_up),
            linear(reference.linear_down),
        )
        self.nonlinearity = gate(reference.nonlinearity)
        self.weight_numel = reference.weight_numel
        self.eqx_tp = None
        self.local_irreps_in = self.linear_up.irreps_in
        self.magnetic_frame = None
        if hasattr(reference, "magnetic_edge_irreps"):
            self.magnetic_edge_irreps = co3.Irreps(reference.magnetic_edge_irreps)
            self.magnetic_frame = co2.LocalFrame(
                self.magnetic_edge_irreps, reference.mmax
            )
        self.attention = (
            RadialAttention(reference.attention)
            if reference.attention is not None
            else None
        )
        self.register_buffer(
            "scale", inverse_scale(self.local_frame_out), persistent=False
        )

    def forward(
        self,
        features,
        weights,
        edge_index,
        rotation,
        rotation_inv,
        edge_radial_basis=None,
        edge_cutoff=None,
        radial_network=None,
    ):
        return self.convolution(
            features,
            weights,
            edge_index,
            rotation,
            rotation_inv,
            edge_radial_basis,
            edge_cutoff,
            radial_network,
        )

    def convolution(
        self,
        features,
        weights,
        edge_index,
        rotation,
        rotation_inv,
        edge_radial_basis,
        edge_cutoff,
        radial_network=None,
        magnetic=None,
    ):
        if radial_network is not None:
            weights = radial_network(materialize(weights, edge_index))
        source = self.local_frame_in(features[edge_index[0]], rotation)
        target = self.local_frame_in(features[edge_index[1]], rotation)
        inputs = [(target, self.node_irreps), (source, self.node_irreps)]
        if magnetic is not None:
            inputs.append(
                (
                    self.magnetic_frame(magnetic, rotation),
                    self.magnetic_frame.irreps_out,
                )
            )
        values, offset = [], 0
        for mul, ir in self.local_irreps_in:
            value = torch.cat(
                [
                    value[:, section].reshape(-1, size, ir.dim)
                    for value, irreps in inputs
                    for (size, input_ir), section in zip(irreps, irreps.slices())
                    if input_ir == ir
                ],
                -2,
            )
            values.append((value * weights[:, offset : offset + mul, None]).flatten(-2))
            offset += mul
        local = self.linear_down(
            self.nonlinearity(self.linear_up(torch.cat(values, -1)))
        )
        if self.attention is not None:
            local = self.attention(
                local,
                source,
                target,
                edge_radial_basis,
                edge_index,
                edge_cutoff,
                features.shape[0],
            )
        message = self.local_frame_out.to_global(local, rotation_inv) * self.scale
        if self.attention is None:
            message = message * edge_cutoff
        return scatter_sum(message, edge_index[1], dim=0, dim_size=features.shape[0])


class MagneticUvConvolution(UvConvolution):
    """Include Cartesian magnetic edge tensors in the local channel map."""

    def forward(
        self,
        features,
        magnetic,
        weights,
        edge_index,
        rotation,
        rotation_inv,
        edge_radial_basis=None,
        edge_cutoff=None,
        radial_network=None,
    ):
        return self.convolution(
            features,
            weights,
            edge_index,
            rotation,
            rotation_inv,
            edge_radial_basis,
            edge_cutoff,
            radial_network,
            magnetic,
        )


class UvTensorProduct(torch.nn.Module):
    """Apply a local Linear--Gate--Linear map without node aggregation."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(self.irreps_in)
        self.local_frame_out = co2.LocalFrame(self.irreps_out)
        self.linear_up, self.linear_down = (
            linear(reference.linear_up),
            linear(reference.linear_down),
        )
        self.gate = gate(reference.gate)
        self.weight_numel = reference.weight_numel
        indices, offset = [], 0
        for mul, ir in self.local_frame_in.irreps_out:
            indices.extend(
                u for u in range(offset, offset + mul) for _ in range(ir.dim)
            )
            offset += mul
        self.register_buffer(
            "weight_index", torch.tensor(indices, dtype=torch.long), persistent=False
        )

    def forward(self, features, rotation, weights):
        local = self.local_frame_in(features, rotation) * weights.index_select(
            -1, self.weight_index
        )
        return self.local_frame_out.to_global(
            self.linear_down(self.gate(self.linear_up(local))), rotation
        )


class RadialRotaryConvolution(Convolution):
    """Use Cartesian plane restriction with the existing compact RRA operator."""

    def setup_cartesian(self):
        self.local_frame_in = co2.LocalFrame(
            [(2 * mul, ir) for mul, ir in self.reshape_in.irreps], self.mmax
        )
        self.local_frame_out = co2.LocalFrame(self.reshape_out.irreps, self.mmax)
        self.to_circular = co2.ChangeOfBasis(
            self.local_frame_in.irreps_out, inverse=True
        )
        self.from_circular = co2.ChangeOfBasis(self.local_frame_out.irreps_out)
        self.register_buffer(
            "inverse_scale", inverse_scale(self.local_frame_out), persistent=False
        )
        self.to_spherical = co3.ChangeOfBasis(self.reshape_in.irreps, inverse=True)
        self.from_spherical = co3.ChangeOfBasis(self.reshape_out.irreps)
        self.stream_reshape_in = LayoutTransform(self.reshape_in.irreps.spherical())
        self.stream_reshape_out = LayoutTransform(self.reshape_out.irreps.spherical())
        self.wigner = o2.WignerD(self.lmax, self.lmax, method="recursive")

    def prepare_stream(self, features, rotation, rotation_inv):
        """Convert node tensors and alignment matrices to the compact kernel layout."""
        wigner = torch.cat(
            [block.flatten(1) for block in self.wigner.from_matrix(rotation)], 1
        )
        return self.stream_reshape_in(self.to_spherical(features)), wigner, wigner

    def finish_stream(self, features):
        """Restore Cartesian node storage after the fused graph reduction."""
        return self.from_spherical(self.stream_reshape_out.inverse(features))

    def align(self, features, rotation, fused):
        blocks = features.split(self.reshape_in.dims, dim=-2)
        features = torch.cat(
            [block.transpose(-1, -2).flatten(-2) for block in blocks], -1
        )
        local = self.to_circular(self.local_frame_in(features, rotation))
        return local.reshape(local.shape[0], -1, 2 * self.num_channel)

    def unalign(self, features, rotation, fused):
        local = self.from_circular(features.flatten(1))
        value = self.local_frame_out.to_global(local, rotation) * self.inverse_scale
        return torch.cat(
            [
                value[:, section].reshape(-1, mul, ir.dim).transpose(-1, -2)
                for (mul, ir), section in zip(
                    self.local_frame_out.irreps_in,
                    self.local_frame_out.irreps_in.slices(),
                )
            ],
            -2,
        )
