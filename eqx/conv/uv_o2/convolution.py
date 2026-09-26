"""Fused local frame, radial, gate and aggregation operations for UV convolutions."""

import math
from functools import lru_cache

import torch

from ..attention import graph_softmax
from ..edge import evaluate
from ..program import Program


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
    program, description, features, wigner, local_dim, global_dim, inverse=False
):
    """Encode degree-wise rotation, regrouping and signed basis permutations."""
    mmax, basis_change, irreps, entries = description
    lmax = math.isqrt(global_dim) - 1
    omitted = (math.isqrt(4 * (global_dim - local_dim) + 1) - 1) // 2
    wigner_mmax = lmax - omitted
    if (lmax + 1) ** 2 != global_dim or omitted * (
        omitted + 1
    ) != global_dim - local_dim:
        raise ValueError("Invalid Wigner matrix dimensions.")
    if entries and (lmax < max(x[2] for x in entries) or wigner_mmax < mmax):
        raise ValueError("Wigner matrices do not cover the local frame.")
    slices, offset = [], 0
    for dim, mul in irreps:
        slices.append(offset)
        offset += dim * mul
    parts = [[] for _ in irreps]
    outputs = []
    for start, mul, ell, odd, indices, starts in entries:
        retained = min(ell, mmax)
        rows = [ell]
        for m in range(1, retained + 1):
            rows.extend(
                (
                    ell + (2 * m - 1) * (lmax + 1) - m * m,
                    ell + 2 * m * (lmax + 1) - m * (m + 1),
                )
            )
        dim = 2 * ell + 1
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
                (
                    a * local_dim + b
                    for a in range(ell * ell, (ell + 1) ** 2)
                    for b in rows
                ),
            )
            outputs.append(program.matmul(matrix, value, dim, len(rows), mul))
        else:
            matrix = program.gather(
                wigner,
                (
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


def linear_description(linear):
    paths = tuple(
        (ins.i_in, ins.i_out, offset, ins.path_weight)
        for ins, (offset, _) in zip(linear._weight_instructions, linear._weight_offsets)
    )
    return (
        tuple(
            (ir.dim, mul, s.start)
            for (ir, mul), s in zip(linear.irreps_in, linear.irreps_in.slices())
        ),
        tuple((ir.dim, mul) for ir, mul in linear.irreps_out),
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


def activate(program, value, module):
    if module is None:
        return value
    name = type(module).__name__
    if name == "normalize2mom":
        return program.scale(
            activate(program, value, module.f), 1 if module._is_id else module.cst
        )
    if name == "ScaledActivation":
        return program.scale(
            activate(program, value, module.activation), module.scale_factor
        )
    operations = {
        "SiLU": "silu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Identity": "identity",
    }
    if name not in operations:
        raise NotImplementedError(f"Fused O(2) gate does not support {name}.")
    return value if name == "Identity" else program.unary(operations[name], value)


def gate_program(gate):
    program = Program()
    features = program.input(0, "edge", gate.irreps_in.dim)
    scalars, gates = [], []
    for locations, activation, outputs in (
        (gate._scalar_locations, gate.act_scalars, scalars),
        (gate._gate_locations, gate.act_gates, gates),
    ):
        for i, act in zip(locations, activation.acts):
            s = gate._input_slices[i]
            outputs.append(
                activate(
                    program, program.slice(features, s.start, s.stop - s.start), act
                )
            )
    outputs = list(scalars)
    for path in gate._paths:
        start = gate._input_slices[gate._gated_locations[path.i_gated]].start
        width = gate.irreps_gated[path.i_gated].mul
        scalar = program.slice(gates[path.i_gate], path.gate_start, path.mul)
        for a in range(path.ir_gated.dim):
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
    """Fuse a UV local convolution, optionally including magnetic edge features.

    Parameters
    ----------
    frame_in, frame_out : LocalFrame
        Node input and output representations, in flattened ir_mul layout.
    linear_up, linear_down : Linear
        Channel maps before and after the gate. Parameters remain owned by
        the caller and are supplied to forward, preserving checkpoint layouts.
    gate : Gate
        Normalized scalar activations and gates, including odd scalar gates.
    frame_edge : LocalFrame, optional
        Additional edge representation, e.g. magnetic tensor-product features.
    query, key : Linear, optional
        Attention channel maps. Supply both or neither.
    num_heads : int, optional
        Attention heads, partitioning each local multiplicity.
    attention_scale, eps : float, optional
        Score normalization and shifted-softmax denominator regularization.

    Notes
    -----
    CUDA fuses gather, degree-wise rotations, basis changes and radial scaling;
    normalized gates; attention scores; and inverse rotation with aggregation.
    Channel mixing uses batched GEMM rather than per-edge matrix-vector loops.
    Local GEMM operands are materialized, but global edge messages are not.
    Analytic expression adjoints support higher derivatives and torch.compile.
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
    ):
        super().__init__()
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
        self.gate_metadata = gate_program(gate)
        self.attention = query is not None
        self.num_heads, self.attention_scale, self.eps = num_heads, attention_scale, eps
        self.weight_numel = self.irreps_in.num_irreps
        self.score_metadata = self.score_program() if self.attention else ""

    @torch.compiler.assume_constant_result
    @lru_cache(maxsize=64)
    def prepare_program(self, local_dim, global_dim):
        program = Program()
        node = program.input(0, "source", self.input_dim)
        target = program.input(0, "target", self.input_dim)
        edge = program.input(1, "edge", self.edge_dim)
        weight = program.input(2, "edge", self.weight_numel)
        wigner = program.input(3, "edge", local_dim * global_dim)
        source = rotate(program, self.frames[0], node, wigner, local_dim, global_dim)
        target = rotate(program, self.frames[0], target, wigner, local_dim, global_dim)
        magnetic = (
            rotate(program, self.frames[1], edge, wigner, local_dim, global_dim)
            if self.frames[1]
            else None
        )
        outputs, offset = [], 0
        for ir, mul in self.irreps_in:
            values = []
            for a in range(ir.dim):
                for feature, irreps in (
                    (target, self.node_irreps),
                    (source, self.node_irreps),
                    (magnetic, self.edge_irreps),
                ):
                    if feature is None:
                        continue
                    for (ir_in, width), s in zip(irreps, irreps.slices()):
                        if ir_in == ir:
                            values.append(
                                program.slice(feature, s.start + a * width, width)
                            )
            value = program.concatenate(values)
            scale = program.gather(weight, tuple(range(offset, offset + mul)) * ir.dim)
            outputs.append(program.binary("mul", value, scale))
            offset += mul
        result = [program.concatenate(outputs)]
        if self.attention:
            result.extend((source, target))
        return repr(
            (tuple(program.nodes), tuple((x, i, "edge") for i, x in enumerate(result)))
        )

    def score_program(self):
        program = Program()
        query, key = [program.input(i, "edge", self.node_irreps.dim) for i in (0, 1)]
        radial = program.input(2, "edge", 2 * self.num_heads)
        heads = tuple(
            c // (mul // self.num_heads)
            for ir, mul in self.node_irreps
            for _ in range(ir.dim)
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

    @torch.compiler.assume_constant_result
    @lru_cache(maxsize=64)
    def output_program(self, local_dim, global_dim):
        program = Program()
        message = program.input(0, "edge", self.irreps_out.dim)
        wigner = program.input(1, "edge", local_dim * global_dim)
        scale = program.input(2, "edge", self.num_heads if self.attention else 1)
        heads = tuple(
            c // (mul // self.num_heads) if self.attention else 0
            for ir, mul in self.irreps_out
            for _ in range(ir.dim)
            for c in range(mul)
        )
        message = program.binary("mul", message, program.gather(scale, heads))
        output = rotate(
            program,
            self.frames[2],
            message,
            wigner,
            local_dim,
            global_dim,
            inverse=True,
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
    ):
        """Evaluate CUDA convolution with explicit weights and optional RRA.

        Parameters
        ----------
        features, edge_features : torch.Tensor
            Node and optional edge features in flattened ir_mul layout.
        conv_weights : torch.Tensor
            Radial coefficients, shape (edges, weight_numel).
        edge_index : torch.Tensor
            Source and target indices, shape (2, edges).
        wigner, wigner_inv : torch.Tensor
            Order-major alignment matrices and their scaled inverses.
        cutoff : torch.Tensor
            Edge envelope, shape (edges, 1).
        parameters : sequence of torch.Tensor
            Flat weight and bias tensors for up, down, then optional query/key.
        radial_attention : torch.Tensor, optional
            Projected attention scale and shift, shape (edges, 2 * num_heads).
        """
        source, target = edge_index[0], edge_index[1]
        if edge_features is None:
            edge_features = features.new_empty((source.shape[0], 0))
        local = evaluate(
            self.prepare_program(wigner.shape[1], wigner.shape[2]),
            [features, edge_features, conv_weights, wigner],
            source,
            target,
            features.shape[0],
        )
        message = linear(local[0], *parameters[:2], self.linears[0])
        message = evaluate(
            self.gate_metadata, [message], source, target, features.shape[0]
        )[0]
        message = linear(message, *parameters[2:4], self.linears[1])
        scale = cutoff
        if self.attention:
            if radial_attention is None:
                raise ValueError("Attention requires radial scale and shift.")
            query = linear(local[2], *parameters[4:6], self.linears[2])
            key = linear(local[1], *parameters[6:8], self.linears[3])
            score = evaluate(
                self.score_metadata,
                [query, key, radial_attention],
                source,
                target,
                features.shape[0],
            )[0]
            scale = (
                graph_softmax(score, target, features.shape[0], cutoff, self.eps)
                * cutoff
            )
        return evaluate(
            self.output_program(wigner_inv.shape[2], wigner_inv.shape[1]),
            [message, wigner_inv, scale],
            source,
            target,
            features.shape[0],
        )[0]
