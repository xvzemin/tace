################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx import co2, co3, o2
from equivariantx.o2.rotation_matrix import rotation_matrix_to_y_axis
from equivariantx.o3 import TransposeIrrepsLayout
from equivariantx.ops.common.gate import gate_program
from equivariantx.ops.common.radial import materialize
from tace.models.spherical.tece_oam_rra import Convolution
from tace.utils.env import acceleration_enabled, eqx_backend, memory_first
from tace.utils.torch_scatter import scatter_sum


class RotationMatrices(torch.nn.Module):
    """Return proper alignment matrices for Cartesian tensor contractions."""

    def __init__(self):
        super().__init__()
        self.backend = eqx_backend("conv")

    def forward(self, vectors):
        backend = self.backend if vectors.is_cuda else "torch"
        rotation = rotation_matrix_to_y_axis(vectors, backend=backend)
        return rotation, rotation

    def forward_packed(self, vectors):
        backend = self.backend if vectors.is_cuda else "torch"
        return rotation_matrix_to_y_axis(vectors, backend=backend)


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
        backend=eqx_backend("linear"),
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
        backend=eqx_backend("linear"),
    )
    result.act_scalars, result.act_gates = module.act_scalars, module.act_gates
    result._paths = module._paths
    result.irreps_out = co2.Irreps(module.irreps_out)
    if result.backend == "cuda":
        result.cuda_program = gate_program(result, cartesian=True)
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


def use_streamed_convolution(plan, features, edge_index):
    """Select the streamed schedule for large FP32 CUDA graphs."""
    return (
        plan is not None
        and features.is_cuda
        and features.dtype == torch.float32
        and statically_known_true(edge_index.shape[1] >= 8192)
    )


class UuConvolution(torch.nn.Module):
    """Apply a channel-wise local map to Cartesian source features."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(
            self.irreps_in,
            reference.mmax,
            backend=eqx_backend("conv"),
        )
        self.local_frame_out = co2.LocalFrame(
            self.irreps_out,
            reference.mmax,
            backend=eqx_backend("conv"),
        )
        self.linear = co2.UuLinear(
            self.local_frame_in.irreps_out,
            self.local_frame_out.irreps_out,
            reference.num_channel,
            backend=eqx_backend("linear"),
        )
        self.weight_numel = reference.weight_numel
        self.eqx_tp = None
        if (
            acceleration_enabled("eqx", kernel="conv")
            and reference.num_channel == 64
            and max(self.local_frame_in.lmax, self.local_frame_out.lmax) <= 2
        ):
            self.eqx_tp = co2.UuTensorProductConv(
                self.local_frame_in,
                self.linear,
                self.local_frame_out,
                radial_mode="full" if memory_first() else "last",
            ).plan
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
        edge_pairs=None,
    ):
        """Evaluate radial coefficients without changing Cartesian storage."""
        if use_streamed_convolution(self.eqx_tp, features, edge_index):
            return (
                self.eqx_tp(
                    features,
                    radial,
                    edge_index,
                    rotation,
                    cutoff,
                    projection=projection,
                    radial_network=radial_network,
                    edge_pairs=edge_pairs,
                )
                * self.scale
            )
        radial = materialize(
            radial,
            edge_index
            if edge_pairs is None
            else edge_index[:, edge_pairs.representatives],
        )
        if radial_network is not None:
            from equivariantx.ops.common.radial import radial_reference

            radial = radial_reference(radial_network, radial)
        if radial.shape[-1] + 1 == projection.shape[0]:
            radial = torch.cat((radial, torch.ones_like(radial[..., :1])), dim=-1)
        weights = radial @ projection if projection.shape[0] else radial
        return self(
            features,
            weights if edge_pairs is None else edge_pairs.expand(weights),
            edge_index,
            rotation,
            rotation,
            cutoff,
        )


class UvConvolution(torch.nn.Module):
    """Apply radial weighting, channel mixing and gates in Cartesian O(2)."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(
            self.irreps_in,
            reference.mmax,
            backend=eqx_backend("conv"),
        )
        self.local_frame_out = co2.LocalFrame(
            self.irreps_out,
            reference.mmax,
            backend=eqx_backend("conv"),
        )
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
        if (
            not hasattr(reference, "magnetic_edge_irreps")
            and acceleration_enabled("eqx", kernel="conv")
            and reference.num_channel == 64
            and max(self.local_frame_in.lmax, self.local_frame_out.lmax) <= 2
        ):
            try:
                self.eqx_tp = co2.UvTensorProductConv(
                    self.local_frame_in,
                    self.linear_up,
                    self.nonlinearity,
                    self.linear_down,
                    self.local_frame_out,
                    radial_mode="full" if memory_first() else "last",
                ).plan
            except NotImplementedError:
                # User-supplied scalar activations can lack an edge expression.
                pass
        if hasattr(reference, "magnetic_edge_irreps"):
            self.magnetic_edge_irreps = co3.Irreps(reference.magnetic_edge_irreps)
            self.magnetic_frame = co2.LocalFrame(
                self.magnetic_edge_irreps,
                reference.mmax,
                backend=eqx_backend("conv"),
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
        edge_pairs=None,
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
            edge_pairs=edge_pairs,
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
        edge_pairs=None,
    ):
        if (
            use_streamed_convolution(self.eqx_tp, features, edge_index)
            and rotation_inv is rotation
            and magnetic is None
        ):
            return (
                self.eqx_tp(
                    features,
                    weights,
                    edge_index,
                    rotation,
                    edge_cutoff,
                    (
                        self.linear_up.weight,
                        self.linear_up.bias,
                        self.linear_down.weight,
                        self.linear_down.bias,
                    ),
                    radial_network=radial_network,
                    edge_pairs=edge_pairs,
                )
                * self.scale
            )
        if isinstance(weights, tuple):
            radial_index = (
                edge_index
                if edge_pairs is None
                else edge_index[:, edge_pairs.representatives]
            )
            weights = materialize(weights, radial_index)
        if radial_network is not None:
            weights = radial_network(weights)
        if edge_pairs is not None:
            weights = edge_pairs.expand(weights)
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
        message = self.local_frame_out.to_global(local, rotation_inv) * self.scale
        message = message * edge_cutoff
        return scatter_sum(message, edge_index[1], dim=0, dim_size=features.shape[0])


class UvTensorProduct(torch.nn.Module):
    """Apply a local Linear--Gate--Linear map without node aggregation."""

    def __init__(self, reference):
        super().__init__()
        self.irreps_in, self.irreps_out = (
            co3.Irreps(reference.irreps_in),
            co3.Irreps(reference.irreps_out),
        )
        self.local_frame_in = co2.LocalFrame(
            self.irreps_in,
            backend=eqx_backend("conv"),
        )
        self.local_frame_out = co2.LocalFrame(
            self.irreps_out,
            backend=eqx_backend("conv"),
        )
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
            [(2 * mul, ir) for mul, ir in self.reshape_in.irreps],
            self.mmax,
            backend=eqx_backend("conv"),
        )
        self.local_frame_out = co2.LocalFrame(
            self.reshape_out.irreps,
            self.mmax,
            backend=eqx_backend("conv"),
        )
        self.alignment_layout = co3.TransposeIrrepsLayout(
            self.local_frame_in.irreps_in,
            source="ir_mul",
            target="mul_ir",
            flat_in=False,
            backend=eqx_backend("conv"),
        )
        self.to_circular = co2.ChangeOfBasis(
            self.local_frame_in.irreps_out,
            inverse=True,
            backend=eqx_backend("conv"),
        )
        self.from_circular = co2.ChangeOfBasis(
            self.local_frame_out.irreps_out,
            backend=eqx_backend("conv"),
        )
        self.register_buffer(
            "inverse_scale", inverse_scale(self.local_frame_out), persistent=False
        )
        self.to_spherical = co3.ChangeOfBasis(
            self.reshape_in.irreps,
            inverse=True,
            backend=eqx_backend("conv"),
        )
        self.from_spherical = co3.ChangeOfBasis(
            self.reshape_out.irreps,
            backend=eqx_backend("conv"),
        )
        self.stream_reshape_in = TransposeIrrepsLayout(
            self.reshape_in.irreps.spherical(),
            source="mul_ir",
            target="ir_mul",
            flat_out=False,
            backend=eqx_backend("conv"),
        )
        self.stream_reshape_out = TransposeIrrepsLayout(
            self.reshape_out.irreps.spherical(),
            source="mul_ir",
            target="ir_mul",
            flat_out=False,
            backend=eqx_backend("conv"),
        )
        self.wigner = o2.WignerD(
            self.lmax,
            self.lmax,
            method="recursive",
            backend=eqx_backend("conv"),
        )

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
        features = self.alignment_layout(features)
        local = self.to_circular(self.local_frame_in(features, rotation))
        return local.reshape(local.shape[0], -1, 2 * self.num_channel)

    def unalign(self, features, rotation, fused):
        local = self.from_circular(features.flatten(1))
        value = self.local_frame_out.to_global(local, rotation) * self.inverse_scale
        return self.reshape_out(value)
