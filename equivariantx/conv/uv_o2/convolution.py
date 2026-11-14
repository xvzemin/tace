"""Fused local frame, radial, gate and aggregation operations for UV convolutions."""

from functools import lru_cache

import torch

from ...o2.layout import irrep_layout
from ...o2.wigner import WignerD
from ...ops import config, resolve_backend
from ...ops.common.gate import gate_program
from ...ops.common.linear import linear_program
from ...ops.common.torch.linear import (
    linear_flat as linear,  # noqa: F401 - compatibility alias
)
from ...ops.common.torch.linear import linear_layout
from ...ops.precision import validate_module_precision
from ...ops.torch.conv.uv_o2 import convolution as torch_kernel
from ...ops.torch.o2 import wigner as torch_wigner
from ...utils.frame_program import frame_description, rotate
from ...utils.metadata import parse_metadata
from ...utils.program import Program
from .transverse import TransverseFrame


def linear_description(linear, transverse=False):
    return linear_layout(
        [(dim, mul) for dim, mul, _ in irrep_layout(linear.irreps_in, transverse)],
        [(dim, mul) for dim, mul, _ in irrep_layout(linear.irreps_out, transverse)],
        linear.instructions,
    )


class UvO2TensorProductConv(torch.nn.Module):
    """Mix and aggregate local O(2) features in flattened ``ir_mul`` layout.

    Parameters
    ----------
    frame_in, frame_out : LocalFrame
        Node input and output representations, in flattened ir_mul layout.
    linear_up, linear_down : equivariantx.o2.Linear
        Channel maps before and after the gate. Supply their parameters to
        :meth:`forward`.
    gate : equivariantx.o2.Gate
        Normalized scalar activations and gates, including odd scalar gates.
    frame_edge : LocalFrame, optional
        Additional edge representation, e.g. magnetic tensor-product features.
    query, key : equivariantx.o2.Linear, optional
        Attention channel maps. Supply both or neither.
    num_heads : int, optional
        Attention heads, partitioning each local multiplicity.
    attention_scale, eps : float, optional
        Score normalization and shifted-softmax denominator regularization.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CUDA streams the complete non-attention convolution
        in edge tiles and recomputes intermediates for higher derivatives.
        Attention uses separate projections and a graph softmax. ``"torch"``
        uses PyTorch operations and automatic differentiation on either device.
    radial_mode : {"last", "full"}, optional
        ``"last"`` evaluates hidden layers with Torch and fuses the final
        projection and its pointwise tail into the edge program.
        ``"full"`` composes all radial layers into that program and recomputes
        activations during differentiation. CUDA keeps this mode tiled during
        compilation; ``"last"`` permits automatic polynomial expansion.

    Examples
    --------
    >>> import torch
    >>> import equivariantx as eqx
    >>> frame = eqx.o2.LocalFrame("2x0e + 2x1o")
    >>> gate = eqx.o2.Gate("4x0e", [torch.nn.functional.silu],
    ...                "2x0e", [torch.sigmoid], "2x1m")
    >>> inputs = (frame.irreps_out + frame.irreps_out).regroup()
    >>> up = eqx.o2.Linear(inputs, gate.irreps_in)
    >>> down = eqx.o2.Linear(gate.irreps_out, frame.irreps_out)
    >>> layer = eqx.conv.UvO2TensorProductConv(
    ...     frame, up, gate, down, frame, backend="torch",
    ... )
    >>> features = torch.randn(3, frame.input_dim)
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> vectors = torch.randn(4, 3)
    >>> weights = torch.randn(4, layer.weight_numel)
    >>> parameters = [up.weight, up.bias, down.weight, down.bias]
    >>> layer(features, None, weights, edge_index, None, None,
    ...       torch.ones(4, 1), parameters, vectors=vectors).shape
    torch.Size([3, 8])
    """

    def __init__(
        self,
        frame_in,
        linear_up,
        gate,
        linear_down,
        frame_out,
        *,
        frame_edge=None,
        query=None,
        key=None,
        num_heads=1,
        attention_scale=1.0,
        eps=1e-16,
        backend=None,
        radial_mode="last",
    ):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be last or full.")
        self.radial_mode = radial_mode
        if (query is None) != (key is None):
            raise ValueError("Supply both query and key maps, or neither.")
        local_inputs = frame_in.irreps_out + frame_in.irreps_out
        if frame_edge is not None:
            local_inputs += frame_edge.irreps_out
        if linear_up.irreps_in != local_inputs.regroup():
            raise ValueError(
                "Linear input must regroup target, source and edge irreps."
            )
        if (
            linear_up.irreps_out != gate.irreps_in
            or linear_down.irreps_in != gate.irreps_out
            or linear_down.irreps_out != frame_out.irreps_out
        ):
            raise ValueError("Linear, gate and frame representations do not match.")
        if query is not None:
            for module in (query, key):
                if (
                    module.irreps_in != frame_in.irreps_out
                    or module.irreps_out != frame_in.irreps_out
                ):
                    raise ValueError(
                        "Query and key must preserve the node local irreps."
                    )
            if num_heads < 1 or any(
                mul % num_heads for _, mul in frame_in.irreps_out + frame_out.irreps_out
            ):
                raise ValueError(
                    "Attention heads must divide every local multiplicity."
                )
        self.frames = tuple(
            frame_description(f) if f is not None else None
            for f in (frame_in, frame_edge, frame_out)
        )
        degree = max(f.lmax for f in (frame_in, frame_edge, frame_out) if f is not None)
        self.degree = degree
        self.wigner = WignerD(degree, degree, backend=backend)
        self.transverse_frames = torch.nn.ModuleList(
            TransverseFrame(f, backend) if f is not None and f.basis_change else None
            for f in (frame_in, frame_edge, frame_out)
        )
        self.input_dim, self.edge_dim = (
            frame_in.input_dim,
            frame_edge.input_dim if frame_edge is not None else 0,
        )
        self.output_dim = frame_out.input_dim
        self.irreps_in = linear_up.irreps_in
        self.node_irreps = frame_in.irreps_out
        self.edge_irreps = frame_edge.irreps_out if frame_edge is not None else ()
        self.irreps_out = frame_out.irreps_out
        self.linears = tuple(
            linear_description(l)
            for l in (linear_up, linear_down, query, key)
            if l is not None
        )
        self.transverse_linears = tuple(
            linear_description(l, True)
            for l in (linear_up, linear_down, query, key)
            if l is not None
        )
        self.gate_metadata = gate_program(gate, blocks=True)
        self.transverse_gate_metadata = gate_program(gate, True, blocks=True)
        self.attention = query is not None
        self.num_heads, self.attention_scale, self.eps = num_heads, attention_scale, eps
        self.weight_numel = self.irreps_in.num_irreps
        self.score_metadata = self.score_program() if self.attention else ""
        self.transverse_score_metadata = (
            self.score_program(True) if self.attention else ""
        )

    @lru_cache(maxsize=64)
    @torch.compiler.assume_constant_result
    def prepare_program(self, local_dim, global_dim, packed=False, transverse=False):
        program = Program()
        if transverse:
            width = sum(
                dim * mul for dim, mul, _ in irrep_layout(self.node_irreps, True)
            )
            source = program.input(0, "edge", width)
            target = program.input(1, "edge", width)
            magnetic = program.input(
                2,
                "edge",
                sum(dim * mul for dim, mul, _ in irrep_layout(self.edge_irreps, True)),
            )
            weight = program.input(3, "edge", self.weight_numel)
        else:
            node = program.input(0, "source", self.input_dim)
            target = program.input(0, "target", self.input_dim)
            edge = program.input(1, "edge", self.edge_dim)
            weight = program.input(2, "edge", self.weight_numel)
            wigner = program.input(3, "edge", local_dim * global_dim)
            source = rotate(
                program,
                self.frames[0],
                node,
                wigner,
                local_dim,
                global_dim,
                packed=packed,
                native=self.backend == "cuda",
            )
            target = rotate(
                program,
                self.frames[0],
                target,
                wigner,
                local_dim,
                global_dim,
                packed=packed,
                native=self.backend == "cuda",
            )
            magnetic = (
                rotate(
                    program,
                    self.frames[1],
                    edge,
                    wigner,
                    local_dim,
                    global_dim,
                    packed=packed,
                    native=self.backend == "cuda",
                )
                if self.frames[1]
                else None
            )
        outputs, offset = [], 0
        for ir, mul in self.irreps_in:
            values = []
            dim = 2 * ir.m + 1 if transverse else ir.dim
            for a in range(dim):
                for feature, irreps in (
                    (target, self.node_irreps),
                    (source, self.node_irreps),
                    (magnetic, self.edge_irreps),
                ):
                    if feature is None:
                        continue
                    for (ir_in, width), (_, _, start) in zip(
                        irreps, irrep_layout(irreps, transverse)
                    ):
                        if ir_in == ir:
                            values.append(
                                program.slice(feature, start + a * width, width)
                            )
            value = program.concatenate(values)
            scale = program.gather(weight, tuple(range(offset, offset + mul)) * dim)
            outputs.append(program.binary("mul", value, scale))
            offset += mul
        result = [program.concatenate(outputs)]
        if self.attention:
            result.extend((source, target))
        return repr(
            (tuple(program.nodes), tuple((x, i, "edge") for i, x in enumerate(result)))
        )

    def score_program(self, transverse=False):
        program = Program()
        width = sum(
            dim * mul for dim, mul, _ in irrep_layout(self.node_irreps, transverse)
        )
        query, key = [program.input(i, "edge", width) for i in (0, 1)]
        radial = program.input(2, "edge", 2 * self.num_heads)
        heads = tuple(
            c // (mul // self.num_heads)
            for dim, mul, _ in irrep_layout(self.node_irreps, transverse)
            for _ in range(dim)
            for c in range(mul)
        )
        score = program.scatter(
            program.binary("mul", query, key), heads, self.num_heads
        )
        scale = program.scale(
            program.unary("sigmoid", program.slice(radial, 0, self.num_heads)),
            2 * self.attention_scale,
        )
        score = program.binary(
            "add",
            program.binary("mul", score, scale),
            program.slice(radial, self.num_heads, self.num_heads),
        )
        return repr((tuple(program.nodes), ((score, 0, "edge"),)))

    @lru_cache(maxsize=64)
    @torch.compiler.assume_constant_result
    def output_program(self, local_dim, global_dim, packed=False, transverse=False):
        program = Program()
        width = sum(
            dim * mul for dim, mul, _ in irrep_layout(self.irreps_out, transverse)
        )
        message = program.input(0, "edge", width)
        wigner = program.input(1, "edge", local_dim * global_dim)
        scale = program.input(2, "edge", self.num_heads if self.attention else 1)
        heads = tuple(
            c // (mul // self.num_heads) if self.attention else 0
            for dim, mul, _ in irrep_layout(self.irreps_out, transverse)
            for _ in range(dim)
            for c in range(mul)
        )
        message = program.binary("mul", message, program.gather(scale, heads))
        if transverse:
            return repr((tuple(program.nodes), ((message, 0, "edge"),)))
        output = rotate(
            program,
            self.frames[2],
            message,
            wigner,
            local_dim,
            global_dim,
            inverse=True,
            packed=packed,
            native=self.backend == "cuda",
        )
        return repr((tuple(program.nodes), ((output, 0, "target"),)))

    @lru_cache(maxsize=64)
    @torch.compiler.assume_constant_result
    def convolution_program(
        self, local_dim, global_dim, packed, transverse, native=True
    ):
        """Encode the complete convolution without an attention reduction."""
        if transverse:
            program = Program()
            features = program.input(0, "source", self.input_dim)
            target = program.input(0, "target", self.input_dim)
            edge = program.input(1, "edge", self.edge_dim)
            radial = program.input(2, "edge", self.weight_numel)
            geometry = program.input(3, "edge", 3)
            node_frame, edge_frame, _ = self.transverse_frames
            local_inputs = [
                node_frame.program(program, features, geometry, native=native),
                node_frame.program(program, target, geometry, native=native),
                edge_frame.program(program, edge, geometry, native=native)
                if edge_frame
                else edge,
                radial,
            ]
            linears = self.transverse_linears
            local = program.compose(
                self.prepare_program(local_dim, global_dim, packed, True),
                local_inputs,
            )[0]
        else:
            nodes, outputs = parse_metadata(
                self.prepare_program(local_dim, global_dim, packed, False)
            )
            program, local = Program(nodes), outputs[0][0]
            linears = self.linears
        inverse = program.input(4, "edge", 0 if transverse else local_dim * global_dim)
        cutoff = program.input(5, "edge", 1)
        values, offset = [], 0
        for dim, mul in linears[0][0]:
            values.append(program.slice(local, offset, dim * mul))
            offset += dim * mul
        for layer, description in enumerate(linears[:2]):
            ins, outs, paths, biases, _ = description
            weight = program.input(
                6 + layer * 2,
                "shared",
                sum(ins[i][1] * outs[j][1] for i, j, _, _ in paths),
            )
            bias = program.input(7 + layer * 2, "shared", sum(n for _, _, n in biases))
            values = linear_program(program, values, weight, bias, description)
            if layer == 0:
                values = program.compose(
                    self.transverse_gate_metadata if transverse else self.gate_metadata,
                    [*values, geometry] if transverse else values,
                )
        message = program.concatenate(values)
        output = program.compose(
            self.output_program(local_dim, global_dim, packed, transverse),
            [message, inverse, cutoff],
        )[0]
        if transverse:
            output = self.transverse_frames[2].program(
                program,
                output,
                geometry,
                inverse=True,
                native=native,
            )
        return repr((tuple(program.nodes), ((output, 0, "target"),)))

    def forward(
        self,
        features,
        edge_features,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        cutoff,
        parameters,
        radial_attention=None,
        *,
        vectors=None,
        radial_network=None,
    ):
        """Evaluate the convolution with explicit weights and optional attention.

        Parameters
        ----------
        features : torch.Tensor
            Node features in flattened ``ir_mul`` layout.
        edge_features : torch.Tensor or None
            Optional edge features in flattened ``ir_mul`` layout.
        conv_weights : torch.Tensor or tuple
            Radial coefficients with shape ``(edges, weight_numel)``.
            With ``radial_network``, its inputs may instead be ``(tensor, kind)``
            partitions with ``kind="edge"``, ``"source"`` or ``"target"``.
        edge_index : torch.Tensor
            Source and target indices with shape ``(2, edges)``.
        wigner, wigner_inv : torch.Tensor or None
            Order-major alignment matrices and their scaled inverses, or
            full degree matrices packed into one row per edge. For packed
            storage, pass wigner_inv=None; the same matrices are transposed
            during the output contraction without a separate allocation.
        cutoff : torch.Tensor
            Edge envelope with shape ``(edges, 1)``.
        parameters : sequence of torch.Tensor
            Flat weight and bias tensors for up, down, then optional query/key.
        radial_attention : torch.Tensor, optional
            Projected attention scale and shift with shape
            ``(edges, 2 * num_heads)``.
        vectors : torch.Tensor, optional
            Nonzero directions, shape ``(edges, 3)``. Select transverse
            restriction without alignment; pass None for both Wigner tensors.
            Positive orders use spherical subspaces of width ``2*m+1``.
        radial_network : torch.nn.Module, optional
            Network mapping ``conv_weights`` inputs to radial coefficients.
            ``radial_mode`` selects final-projection or full-network fusion.

        Returns
        -------
        torch.Tensor
            Aggregated node features in flattened ``ir_mul`` layout with
            the global output representation declared by ``frame_out``.
        """
        validate_module_precision(
            self,
            (
                features,
                edge_features,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                cutoff,
                parameters,
                radial_attention,
                vectors,
                radial_network,
            ),
        )
        from ...ops.common.radial import cast_radial

        conv_weights = cast_radial(conv_weights, features.dtype)
        cutoff = cutoff.to(features)
        if edge_features is not None:
            edge_features = edge_features.to(features)
        if radial_attention is not None:
            radial_attention = radial_attention.to(features)
        if vectors is not None and vectors.dtype != features.dtype:
            wigner = (
                torch_wigner.packed(self.wigner, vectors, "recursive")
                if self.backend == "torch" and config.TORCH_REFERENCE
                else self.wigner.forward_packed(
                    vectors, method="auto" if self.backend == "cuda" else "recursive"
                )
            ).to(features)
            wigner_inv = None
            vectors = None
        if wigner is not None:
            wigner = wigner.to(features)
        if wigner_inv is not None:
            wigner_inv = wigner_inv.to(features)
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.conv.uv_o2 import convolution as cuda_kernel

            return cuda_kernel.evaluate(
                self,
                features,
                edge_features,
                conv_weights,
                wigner,
                wigner_inv,
                cutoff,
                parameters,
                radial_attention,
                edge_index,
                vectors,
                radial_network,
            )
        return torch_kernel.evaluate(
            self,
            features,
            edge_features,
            conv_weights,
            wigner,
            wigner_inv,
            cutoff,
            parameters,
            radial_attention,
            edge_index,
            vectors,
            radial_network,
        )
