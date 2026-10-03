"""Fused local frame, radial, gate and aggregation operations for UV convolutions."""

import math
from functools import lru_cache

import torch

from ...o2._layout import wigner_indices, wigner_orders
from ..attention import graph_softmax
from ..edge import evaluate, evaluate_torch
from ..program import Program, activate
from .transverse import TransverseFrame, layout


def frame_description(frame):
    return (
        frame.mmax,
        frame.basis_change,
        tuple((ir.dim, mul) for ir, mul in frame.irreps_out),
        tuple(
            (
                entry.global_slice.start,
                entry.mul,
                entry.degree,
                entry.odd,
                entry.local_indices,
                tuple(s.start for s in entry.local_slices),
            )
            for entry in frame._entries
        ),
    )


def rotate(
    program,
    description,
    features,
    wigner,
    local_dim,
    global_dim,
    inverse=False,
    packed=False,
):
    """Encode degree-wise rotation, regrouping and signed basis permutations."""
    mmax, basis_change, irreps, entries = description
    if packed:
        lmax = wigner_mmax = max(x[2] for x in entries) if entries else 0
    else:
        lmax, wigner_mmax = wigner_orders(
            global_dim,
            local_dim,
            lmax=max(x[2] for x in entries) if entries else 0,
            mmax=mmax,
        )
    slices, offset = [], 0
    for dim, mul in irreps:
        slices.append(offset)
        offset += dim * mul
    parts = [[] for _ in irreps]
    outputs = []
    for start, mul, ell, odd, indices, starts in entries:
        retained = min(ell, mmax)
        dim = 2 * ell + 1
        rows = (
            (ell, *(ell + s * m for m in range(1, retained + 1) for s in (1, -1)))
            if packed
            else wigner_indices(ell, retained, lmax)
        )
        offset = ell * (4 * ell**2 - 1) // 3
        if inverse:
            values = []
            for m, (index, begin) in enumerate(zip(indices, starts)):
                width = irreps[index][1]
                for real in range(1 if m == 0 else 2):
                    swapped = basis_change and odd and m > 0
                    position = (1 - real) if swapped else real
                    value = program.slice(
                        features, slices[index] + position * width + begin, mul
                    )
                    values.append(
                        program.scale(value, -1 if swapped and real == 1 else 1)
                    )
            value = program.concatenate(values)
            value = program.scale(
                value, math.sqrt((2 * min(ell, wigner_mmax) + 1) / len(rows))
            )
            matrix = program.gather(
                wigner,
                (offset + b * dim + a for a in range(dim) for b in rows)
                if packed
                else (
                    a * local_dim + b
                    for a in range(ell * ell, (ell + 1) ** 2)
                    for b in rows
                ),
            )
            outputs.append(program.matmul(matrix, value, dim, len(rows), mul))
        else:
            matrix = program.gather(
                wigner,
                (offset + a * dim + b for a in rows for b in range(dim))
                if packed
                else (
                    a * global_dim + b
                    for a in rows
                    for b in range(ell * ell, (ell + 1) ** 2)
                ),
            )
            value = program.matmul(
                matrix, program.slice(features, start, dim * mul), len(rows), dim, mul
            )
            for m, (index, begin) in enumerate(zip(indices, starts)):
                values = []
                for real in range(1 if m == 0 else 2):
                    swapped = basis_change and odd and m > 0
                    position = (1 - real) if swapped else real
                    a = 0 if m == 0 else 2 * m - 1 + position
                    values.append(
                        program.scale(
                            program.slice(value, a * mul, mul),
                            -1 if swapped and real == 0 else 1,
                        )
                    )
                parts[index].append((begin, mul, program.concatenate(values)))
    if inverse:
        return program.concatenate(outputs)
    outputs = []
    for (dim, _), parts_ir in zip(irreps, parts):
        parts_ir.sort()
        outputs.extend(
            program.slice(value, a * mul, mul)
            for a in range(dim)
            for _, mul, value in parts_ir
        )
    return program.concatenate(outputs)


def linear_description(linear, transverse=False):
    paths = tuple(
        (ins.i_in, ins.i_out, offset, ins.path_weight)
        for ins, (offset, _) in zip(linear._weight_instructions, linear._weight_offsets)
    )
    return (
        layout(linear.irreps_in, transverse),
        tuple((dim, mul) for dim, mul, _ in layout(linear.irreps_out, transverse)),
        paths,
        tuple(linear._bias_offsets.items()),
    )


def linear(features, weight, bias, description):
    """Keep channel contractions in batched GEMM, including parameter adjoints."""
    inputs, outputs, paths, biases = description
    values = [
        features[:, start : start + dim * mul].reshape(features.shape[0], dim, mul)
        for dim, mul, start in inputs
    ]
    result = []
    for j, (dim, mul) in enumerate(outputs):
        terms = [
            (
                values[i].reshape(features.shape[0] * dim, inputs[i][1])
                @ (
                    weight[offset : offset + inputs[i][1] * mul].view(inputs[i][1], mul)
                    * scale
                )
            ).reshape(features.shape[0], dim, mul)
            for i, k, offset, scale in paths
            if k == j
        ]
        value = (
            sum(terms[1:], terms[0])
            if terms
            else features.new_zeros((features.shape[0], dim, mul))
        )
        for k, (start, size) in biases:
            if k == j:
                value = value + bias[start : start + size]
        result.append(value.reshape(features.shape[0], dim * mul))
    return torch.cat(result, dim=-1)


def gate_program(gate, transverse=False):
    program = Program()
    inputs = layout(gate.irreps_in, transverse)
    features = program.input(0, "edge", sum(dim * mul for dim, mul, _ in inputs))
    direction = program.input(1, "edge", 3) if transverse else None
    scalars, gates = [], []
    for locations, activation, outputs in (
        (gate._scalar_locations, gate.act_scalars, scalars),
        (gate._gate_locations, gate.act_gates, gates),
    ):
        for i, act in zip(locations, activation.acts):
            dim, mul, start = inputs[i]
            outputs.append(
                activate(program, program.slice(features, start, dim * mul), act)
            )
    outputs = list(scalars)
    for path in gate._paths:
        dim, _, start = inputs[gate._gated_locations[path.i_gated]]
        width = gate.irreps_gated[path.i_gated].mul
        scalar = program.slice(gates[path.i_gate], path.gate_start, path.mul)
        if transverse and path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0:
            from ...o3 import so3_generators

            value = program.concatenate(
                [
                    program.slice(
                        features, start + a * width + path.gated_start, path.mul
                    )
                    for a in range(dim)
                ]
            )
            vector = program.gather(
                direction, tuple(a for a in range(3) for _ in range(path.mul))
            )
            generator = so3_generators(path.ir_gated.m) / path.ir_gated.m
            paths = tuple(
                ((i, a, j), float(generator[a, j, i]))
                for a, j, i in generator.nonzero().tolist()
            )
            value = program.product(
                (value, vector, program.constant(dim * path.mul, 0)),
                (path.mul, (dim, 3, dim), paths),
                2,
            )
            outputs.append(
                program.binary(
                    "mul", value, program.gather(scalar, tuple(range(path.mul)) * dim)
                )
            )
            continue
        for a in range(dim):
            odd = path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0
            b = 1 - a if odd else a
            value = program.slice(
                features, start + b * width + path.gated_start, path.mul
            )
            value = program.scale(value, -1 if odd and a == 0 else 1)
            outputs.append(program.binary("mul", value, scalar))
    output = program.concatenate(outputs)
    return repr((tuple(program.nodes), ((output, 0, "edge"),)))


class UvO2TensorProductConv(torch.nn.Module):
    """Apply a channel-mixing O(2) convolution with optional edge representations.

    Parameters
    ----------
    frame_in, frame_out : LocalFrame
        Node input and output representations, in flattened ir_mul layout.
    linear_up, linear_down : eqx.o2.Linear
        Channel maps before and after the gate. Parameters remain owned by
        the caller and are supplied to forward, preserving checkpoint layouts.
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
        self.gate_metadata = gate_program(gate)
        self.transverse_gate_metadata = gate_program(gate, True)
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
            width = sum(dim * mul for dim, mul, _ in layout(self.node_irreps, True))
            source = program.input(0, "edge", width)
            target = program.input(1, "edge", width)
            magnetic = program.input(
                2,
                "edge",
                sum(dim * mul for dim, mul, _ in layout(self.edge_irreps, True)),
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
                        irreps, layout(irreps, transverse)
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
        width = sum(dim * mul for dim, mul, _ in layout(self.node_irreps, transverse))
        query, key = [program.input(i, "edge", width) for i in (0, 1)]
        radial = program.input(2, "edge", 2 * self.num_heads)
        heads = tuple(
            c // (mul // self.num_heads)
            for dim, mul, _ in layout(self.node_irreps, transverse)
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
        width = sum(dim * mul for dim, mul, _ in layout(self.irreps_out, transverse))
        message = program.input(0, "edge", width)
        wigner = program.input(1, "edge", local_dim * global_dim)
        scale = program.input(2, "edge", self.num_heads if self.attention else 1)
        heads = tuple(
            c // (mul // self.num_heads) if self.attention else 0
            for dim, mul, _ in layout(self.irreps_out, transverse)
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
    ):
        """Evaluate the convolution with explicit weights and optional attention.

        Parameters
        ----------
        features : torch.Tensor
            Node features in flattened ``ir_mul`` layout.
        edge_features : torch.Tensor or None
            Optional edge features in flattened ``ir_mul`` layout.
        conv_weights : torch.Tensor
            Radial coefficients with shape ``(edges, weight_numel)``.
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
                from ...kernels.wigner import alignment_cuda

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
        local = execute(
            self.prepare_program(local_dim, global_dim, packed, transverse),
            local_inputs,
            source,
            target,
            features.shape[0],
        )
        message = linear(local[0], *parameters[:2], linears[0])
        message = execute(
            self.transverse_gate_metadata if transverse else self.gate_metadata,
            [message, direction] if transverse else [message],
            source,
            target,
            features.shape[0],
        )[0]
        message = linear(message, *parameters[2:4], linears[1])
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
