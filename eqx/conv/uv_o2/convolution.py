"""Fused local frame, radial, gate and aggregation operations for UV convolutions."""

from functools import lru_cache

import torch

from ...backends.torch.edge import evaluate_torch
from ...backends.torch.linear import linear as linear_blocks
from ...backends.torch.linear import linear_layout
from ...o2.layout import irrep_layout
from ...ops.edge import evaluate
from ...ops.frame_program import frame_description, rotate
from ...ops.gate import gate_program
from ...ops.program import Program
from ..attention import graph_softmax
from .transverse import TransverseFrame


def linear_description(linear, transverse=False):
    return linear_layout(
        [(dim, mul) for dim, mul, _ in irrep_layout(linear.irreps_in, transverse)],
        [(dim, mul) for dim, mul, _ in irrep_layout(linear.irreps_out, transverse)],
        linear.instructions,
    )


def linear(features, weight, bias, description):
    """Keep channel contractions in batched GEMM, including parameter adjoints."""
    inputs = description[0]
    values = [
        value.reshape(features.shape[0], dim, mul)
        for value, (dim, mul) in zip(
            features.split([dim * mul for dim, mul in inputs], -1), inputs
        )
    ]
    result = linear_blocks(values, weight, bias, description, features.shape[:-1])
    return (
        torch.cat([value.flatten(-2) for value in result], -1)
        if result
        else features[..., :0]
    )


class UvO2TensorProductConv(torch.nn.Module):
    """Apply a channel-mixing O(2) convolution with optional edge representations.

    Parameters
    ----------
    frame_in, frame_out : LocalFrame
        Node input and output representations, in flattened ir_mul layout.
    linear_up, linear_down : eqx.o2.Linear
        Channel maps before and after the gate. Supply their parameters to
        :meth:`forward`.
    gate : eqx.o2.Gate
        Normalized scalar activations and gates, including odd scalar gates.
    frame_edge : LocalFrame, optional
        Additional edge representation, e.g. magnetic tensor-product features.
    query, key : eqx.o2.Linear, optional
        Attention channel maps. Supply both or neither.
    num_heads : int, optional
        Attention heads, partitioning each local multiplicity.
    attention_scale, eps : float, optional
        Score normalization and shifted-softmax denominator regularization.
    backend : {"cuda", "torch"}, optional
        Execution backend. ``"torch"`` evaluates aligned and transverse forms
        with PyTorch operations and automatic differentiation on either device.

    Notes
    -----
    Features use flattened ``ir_mul`` storage. CUDA fuses frame operations,
    gates and aggregation; channel mixing uses PyTorch matrix products.
    Both backends support higher derivatives.

    Examples
    --------
    >>> from eqx import conv, o2
    >>> frame = o2.LocalFrame("2x0e + 2x1o")
    >>> gate = o2.Gate("4x0e", [torch.nn.functional.silu],
    ...                "2x0e", [torch.sigmoid], "2x1m")
    >>> inputs = (frame.irreps_out + frame.irreps_out).regroup()
    >>> up = o2.Linear(inputs, gate.irreps_in)
    >>> down = o2.Linear(gate.irreps_out, frame.irreps_out)
    >>> layer = conv.UvO2TensorProductConv(
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
        backend="cuda",
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be cuda or torch.")
        self.backend = backend
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
            )
            target = rotate(
                program,
                self.frames[0],
                target,
                wigner,
                local_dim,
                global_dim,
                packed=packed,
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
            CUDA evaluates nonlinear layers in bounded tiles without retaining
            activations or full-edge coefficients.

        Returns
        -------
        torch.Tensor
            Aggregated node features in flattened ``ir_mul`` layout with
            the global output representation declared by ``frame_out``.
        """
        source, target = edge_index[0], edge_index[1]
        transverse = vectors is not None
        fused = self.backend == "cuda" and features.is_cuda
        execute = evaluate if fused else evaluate_torch
        if radial_network is not None and not fused:
            from ...ops.radial import materialize

            conv_weights = radial_network(materialize(conv_weights, edge_index))
            radial_network = None
        if edge_features is None:
            edge_features = features.new_empty((source.shape[0], 0))
        if transverse:
            if any(
                f is None and description is not None
                for f, description in zip(self.transverse_frames, self.frames)
            ):
                raise ValueError(
                    "Transverse evaluation requires canonical O(2) frames."
                )
            if fused:
                from ...ops.wigner import alignment_cuda

                direction = alignment_cuda(repr("normalize"), [vectors])[0]
            else:
                direction = vectors / vectors.norm(dim=-1, keepdim=True)
            edges = source.numel()
            edge_ids = torch.arange(edges, device=source.device)
            node_frame, edge_frame, _ = self.transverse_frames
            projected_source = node_frame(features, direction, source, edge_ids, edges)
            projected_target = node_frame(features, direction, target, edge_ids, edges)
            projected_edge = (
                edge_frame(edge_features, direction, edge_ids, edge_ids, edges)
                if edge_frame is not None
                else edge_features
            )
            local_inputs = [
                projected_source,
                projected_target,
                projected_edge,
                conv_weights,
            ]
            local_dim = global_dim = 0
            packed = False
            linears = self.transverse_linears
        else:
            if wigner is None:
                raise ValueError("Supply Wigner matrices or vectors.")
            packed = wigner.ndim == 2
            local_dim, global_dim = (wigner.shape[1], 1) if packed else wigner.shape[1:]
            if packed:
                wigner_inv = wigner
            local_inputs = [features, edge_features, conv_weights, wigner]
            linears = self.linears
        metadata = self.prepare_program(local_dim, global_dim, packed, transverse)
        if radial_network is not None:
            from ...ops.radial import compose_radial

            metadata, local_inputs = compose_radial(
                metadata, local_inputs, 3 if transverse else 2, radial_network
            )
        local = execute(
            metadata,
            local_inputs,
            source,
            target,
            features.shape[0],
        )
        values = [
            value.reshape(source.shape[0], dim, mul)
            for value, (dim, mul) in zip(
                local[0].split([dim * mul for dim, mul in linears[0][0]], -1),
                linears[0][0],
            )
        ]
        message = linear_blocks(values, *parameters[:2], linears[0], (source.shape[0],))
        message = execute(
            self.transverse_gate_metadata if transverse else self.gate_metadata,
            [value.flatten(-2) for value in message]
            + ([direction] if transverse else []),
            source,
            target,
            features.shape[0],
        )
        message = linear_blocks(
            [
                value.reshape(source.shape[0], dim, mul)
                for value, (dim, mul) in zip(message, linears[1][0])
            ],
            *parameters[2:4],
            linears[1],
            (source.shape[0],),
        )
        message = torch.cat([value.flatten(-2) for value in message], -1)
        scale = cutoff
        if self.attention:
            if radial_attention is None:
                raise ValueError("Attention requires radial scale and shift.")
            query = linear(local[2], *parameters[4:6], linears[2])
            key = linear(local[1], *parameters[6:8], linears[3])
            score = execute(
                self.transverse_score_metadata if transverse else self.score_metadata,
                [query, key, radial_attention],
                source,
                target,
                features.shape[0],
            )[0]
            scale = (
                graph_softmax(
                    score, target, features.shape[0], cutoff, self.eps, fused=fused
                )
                * cutoff
            )
        output = execute(
            self.output_program(local_dim, global_dim, packed, transverse),
            [
                message,
                message.new_empty((source.numel(), 0)) if transverse else wigner_inv,
                scale,
            ],
            source,
            target,
            features.shape[0],
        )[0]
        if transverse:
            output = self.transverse_frames[2](
                output, direction, edge_ids, target, features.shape[0], inverse=True
            )
        return output
