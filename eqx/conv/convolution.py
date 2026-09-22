################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Indexed aligned-frame contractions and their transposes."""

import math

import torch
from e3nn import o3


class _Contraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, program, source, target, *operands):
        ctx.plan = plan
        ctx.program = program
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(source, target, *operands)
        results = [None] * (
            1 + max(slot for _, _, pairs in program for _, slot in pairs)
        )
        for mapping, _, pairs in program:
            for role, slot in pairs:
                if results[slot] is None:
                    results[slot] = torch.zeros_like(
                        operands[mapping[role]], memory_format=torch.contiguous_format
                    )
        if source.numel() and plan.path_data:
            calls = [
                (
                    tuple(role for role, _ in pairs),
                    tuple(operands[i] for i in mapping),
                    tuple(results[slot] for _, slot in pairs),
                    weighted_only,
                )
                for mapping, weighted_only, pairs in program
            ]
            if plan.backend == "triton" and operands[0].is_cuda:
                from .triton import contract_many

                contract_many(plan, source, target, calls)
            else:
                for outputs, values, destinations, weighted_only in calls:
                    for output, result in zip(outputs, destinations):
                        if result.numel():
                            plan.reference(
                                output, source, target, values, result, weighted_only
                            )
        return tuple(results)

    @staticmethod
    def backward(ctx, *grad_outputs):
        source, target, *operands = ctx.saved_tensors
        values = list(operands)
        cotangents = {}
        for slot, value in enumerate(grad_outputs):
            if value is not None:
                cotangents[slot] = len(values)
                values.append(value)
        terms = {}
        destinations = {}
        for mapping, weighted_only, pairs in ctx.program:
            for output, slot in pairs:
                if slot not in cotangents:
                    continue
                if output == 2 and not operands[mapping[2]].numel():
                    continue
                replacement = list(mapping)
                replacement[output] = cotangents[slot]
                # Weight differentiation permanently excludes unweighted paths.
                key = (
                    tuple(replacement),
                    ctx.plan.has_unweighted and (weighted_only or output in (1, 2)),
                )
                for role, index in enumerate(mapping):
                    if role != output and ctx.needs_input_grad[index + 4]:
                        destination = destinations.setdefault(index, len(destinations))
                        terms.setdefault(key, []).append((role, destination))
        gradients = [None] * len(operands)
        if terms:
            program = tuple(
                (mapping, weighted, tuple(pairs))
                for (mapping, weighted), pairs in terms.items()
            )
            results = _Contraction.apply(ctx.plan, program, source, target, *values)
            for index, slot in destinations.items():
                gradients[index] = results[slot]
        return None, None, None, None, *gradients


class Convolution(torch.nn.Module):
    """Schedule a tensor product with indexed input and output nodes.

    Parameters
    ----------
    tensor_product : O3TensorProduct
        Tensor product defining paths, normalization and feature layouts.
    backend : {"torch", "triton"}, optional
        Execution backend. Defaults to ``"torch"`` on every device.
        ``"triton"`` uses fused CUDA contractions and falls back to PyTorch
        on CPU. Triton is imported only when this backend executes on CUDA.

    Notes
    -----
    The scalar contraction has seven operands: input node features, radial
    features, radial projection, input rotations, output rotations, harmonic
    amplitudes, and output node features. Computing any one operand's adjoint
    uses the same contraction with that operand designated as the output.
    This rule also applies to higher derivatives. Triton contractions fuse
    gathers, both feature rotations, sparse coupling and reductions without
    retaining edge messages. Large radial projections use matrix products in
    bounded edge chunks; small projections are evaluated inside the kernel.
    Rotation matrices are supplied as packed degree blocks, not zero-padded
    block-diagonal matrices. Paths sharing an input block reuse its rotation
    within each angular/channel tile. Input adjoints are accumulated locally
    before the inverse rotation. Workspaces for projected weights are reused
    across chunks and across mixed derivative terms, and are not saved for
    backward. Path dependencies are retained through each transpose, including
    mixtures of weighted and unweighted instructions. Small angular blocks
    and derivative programs use register-resident channelwise contractions
    with fused mixed adjoints. Common rotations and local contractions are
    reused, and gradients are combined before inverse rotations. Larger
    programs use tiled CUDA contractions to bound compilation and register
    usage. Graph reductions use atomic additions, so summation order is not
    deterministic.
    """

    def __init__(self, tensor_product, *, backend="torch"):
        super().__init__()
        if backend not in ("torch", "triton"):
            raise ValueError("backend must be torch or triton.")
        self.backend = backend
        self.irreps_in = tensor_product.irreps_in1
        self.irreps_out = tensor_product.irreps_out
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        self.num_harmonics = tensor_product.num_harmonics
        self.degree_width = 1 << (2 * tensor_product.lmax).bit_length()
        self.degree_offsets = tuple(
            sum((2 * k + 1) ** 2 for k in range(l))
            for l in range(tensor_product.lmax + 1)
        )
        input_slices = self.irreps_in.slices()
        output_slices = self.irreps_out.slices()
        paths = []
        indices = []
        coefficients = []
        offset = 0
        self.path_data = []
        self.sparse_paths = []
        for ins in self.instructions:
            mul, ir = self.irreps_in[ins.i_in1]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            ir_sh = tensor_product.irreps_in2[ins.i_in2].ir
            weight_offset = offset if ins.has_weight else -1
            if ins.has_weight:
                offset += math.prod(ins.path_shape)
            if not mul or not mul_out or not ins.path_weight:
                continue
            pole = (
                1.0
                if tensor_product.normalization == "norm"
                else math.sqrt(2 * ir_sh.l + 1)
            )
            if tensor_product.normalization == "integral":
                pole /= math.sqrt(4 * math.pi)
            cg = o3.wigner_3j(ir.l, ir_sh.l, ir_out.l, dtype=torch.float64)[
                :, ir_sh.l, :
            ]
            cg = cg * (pole * ins.path_weight)
            # Each row and column of the real m2=0 slice has at most one
            # nonzero. Both orientations are retained for transposed calls.
            if (cg.count_nonzero(0) > 1).any() or (cg.count_nonzero(1) > 1).any():
                raise ValueError("Expected a signed-order-diagonal CG slice.")
            map_forward = torch.zeros(self.degree_width, dtype=torch.int32)
            map_reverse = torch.zeros_like(map_forward)
            coefficient_forward = torch.zeros(self.degree_width, dtype=torch.float64)
            coefficient_reverse = torch.zeros_like(coefficient_forward)
            for m, n in cg.nonzero().tolist():
                map_forward[n] = m
                map_reverse[m] = n
                coefficient_forward[n] = coefficient_reverse[m] = cg[m, n]
            indices.append(torch.stack((map_forward, map_reverse)))
            coefficients.append(torch.stack((coefficient_forward, coefficient_reverse)))
            paths.append(
                (
                    input_slices[ins.i_in1].start,
                    output_slices[ins.i_out].start,
                    mul,
                    mul_out,
                    ir.dim,
                    ir_out.dim,
                    self.degree_offsets[ir.l],
                    self.degree_offsets[ir_out.l],
                    weight_offset,
                    ins.i_in2,
                )
            )
            self.path_data.append((ins.connection_mode, paths[-1]))
            self.sparse_paths.append(
                tuple((m, n, float(cg[m, n])) for m, n in cg.nonzero().tolist())
            )

        # Small compile-time groups bound code size and register lifetimes.
        groups = {}
        for index, (mode, path) in enumerate(self.path_data):
            if mode == "uvu":
                groups.setdefault((path[0], path[2], path[4]), []).append(index)
        self.static_groups = tuple(
            tuple(
                (self.path_data[i][1], self.sparse_paths[i])
                for i in indices[start : start + 4]
            )
            for indices in groups.values()
            for start in range(0, len(indices), 4)
        )
        self.channelwise = all(mode == "uvu" for mode, _ in self.path_data)
        self.has_unweighted = any(path[8] < 0 for _, path in self.path_data)

        self.register_buffer(
            "paths",
            torch.tensor(paths, dtype=torch.int32).reshape(-1, 10),
            persistent=False,
        )
        self.register_buffer(
            "indices",
            torch.stack(indices)
            if indices
            else torch.empty(0, 2, self.degree_width, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "coefficients",
            (
                torch.stack(coefficients)
                if coefficients
                else torch.empty(0, 2, self.degree_width)
            ).to(torch.get_default_dtype()),
            persistent=False,
        )
        tiles = {}
        for index, (mode, path) in enumerate(self.path_data):
            degree = 1 << (max(path[4], path[5]) - 1).bit_length()
            channels = 1 << (max(path[2], path[3]) - 1).bit_length()
            if mode == "uvu" and channels >= 16:
                degree = max(8, degree)
            width = min(channels, 32 if mode == "uvu" else 16, max(4, 512 // degree))
            entries = tiles.setdefault((mode, degree, width), {})
            for u in range(0, path[2], width):
                shared = entries.setdefault((path[0], u), [])
                for v in [u] if mode == "uvu" else range(0, path[3], width):
                    shared.append((index, u, v))
        self.tile_groups = tuple(sorted(tiles))
        for index, key in enumerate(self.tile_groups):
            entries = []
            offsets = [0]
            for shared in tiles[key].values():
                entries.extend(shared)
                offsets.append(len(entries))
            self.register_buffer(
                f"tiles_{index}",
                torch.tensor(entries, dtype=torch.int32).reshape(-1, 3),
                persistent=False,
            )
            self.register_buffer(
                f"tile_offsets_{index}",
                torch.tensor(offsets, dtype=torch.int32),
                persistent=False,
            )

    def forward(
        self, features, radial, projection, wigner, amplitudes, edge_index, num_nodes
    ):
        """Gather, couple and sum features at target nodes.

        Parameters
        ----------
        features : torch.Tensor
            Node features of shape ``(nodes, irreps_in.dim)`` in flattened
            ``ir_mul`` order.
        radial : torch.Tensor
            Radial features of shape ``(edges, channels)`` or
            ``(1, channels)``. With an empty projection, these are the
            tensor-product path weights in instruction order.
        projection : torch.Tensor
            Radial projection of shape ``(channels, weight_numel)``. Use
            shape ``(0, weight_numel)`` for directly supplied path weights.
        wigner : torch.Tensor
            Packed degree-wise Wigner matrices, with shape
            ``(edges, sum((2*l+1)**2))`` or a shared leading dimension of one.
        amplitudes : torch.Tensor
            Invariant harmonic amplitudes of shape ``(edges, num_harmonics)``
            or ``(1, num_harmonics)``.
        edge_index : torch.Tensor
            Source and target indices of shape ``(2, edges)``.
        num_nodes : int
            Number of target nodes.

        Returns
        -------
        torch.Tensor
            Target features of shape ``(num_nodes, irreps_out.dim)`` in the
            declared, unsimplified ``ir_mul`` layout.
        """
        # A zero-stride placeholder supplies the output shape without allocating
        # a second node output. It is never read by the forward contraction.
        output = features.new_empty(1).expand(num_nodes, self.irreps_out.dim)
        return _Contraction.apply(
            self,
            ((tuple(range(7)), False, ((6, 0),)),),
            edge_index[0],
            edge_index[1],
            features,
            radial,
            projection,
            wigner,
            wigner,
            amplitudes,
            output,
        )[0]

    def reference(self, output, source, target, operands, result, weighted_only=False):
        """Evaluate the same contractions using ordinary tensor operations."""
        x, radial, projection, din, dout, amplitudes, y = operands
        projected = projection.numel() != 0
        edges = source.numel()
        for index, (mode, path) in enumerate(self.path_data):
            start, end, mul, mul_out, dim, dim_out, dstart, dend, weight, harmonic = (
                path
            )
            if ((weighted_only or output in (1, 2)) and weight < 0) or (
                output == 2 and not projected
            ):
                continue
            cg = x.new_zeros(dim, dim_out).scatter(
                0,
                self.indices[index, 0, :dim_out].long().unsqueeze(0),
                self.coefficients[index, 0, :dim_out].to(x).unsqueeze(0),
            )
            tensors = [
                None
                if output == 0
                else x[source, start : start + mul * dim].reshape(edges, dim, mul),
                None,
                None,
                None
                if output == 3
                else din[:, dstart : dstart + dim * dim]
                .reshape(-1, dim, dim)
                .expand(edges, -1, -1),
                None
                if output == 4
                else dout[:, dend : dend + dim_out * dim_out]
                .reshape(-1, dim_out, dim_out)
                .expand(edges, -1, -1),
                None if output == 5 else amplitudes[:, harmonic].expand(edges),
                None
                if output == 6
                else y[target, end : end + mul_out * dim_out].reshape(
                    edges, dim_out, mul_out
                ),
            ]
            tokens = ["eau", "ek", "kuv", "ema", "enb", "e", "ebv"]
            width = mul if mode == "uvu" else mul * mul_out
            weight_shape = (mul,) if mode == "uvu" else (mul, mul_out)
            if mode == "uvu":
                tokens[2], tokens[6] = "ku", "ebu"
            if weight >= 0:
                if projected:
                    if output != 1:
                        tensors[1] = radial.expand(edges, -1)
                    if output != 2:
                        tensors[2] = projection[:, weight : weight + width].reshape(
                            projection.size(0), *weight_shape
                        )
                else:
                    tokens[1] = "eu" if mode == "uvu" else "euv"
                    if output != 1:
                        tensors[1] = (
                            radial[:, weight : weight + width]
                            .reshape(-1, *weight_shape)
                            .expand(edges, *weight_shape)
                        )
            active = [
                (token, value)
                for token, value in zip(tokens, tensors)
                if value is not None
            ]
            value = torch.einsum(
                ",".join([token for token, _ in active] + ["mn"])
                + "->"
                + tokens[output],
                *[value for _, value in active],
                cg,
            )
            if output in (0, 6):
                node_indices = source if output == 0 else target
                begin, size = (
                    (start, mul * dim) if output == 0 else (end, mul_out * dim_out)
                )
                result[:, begin : begin + size].index_add_(
                    0, node_indices, value.reshape(edges, size)
                )
            elif output in (3, 4):
                begin, size = (
                    (dstart, dim * dim) if output == 3 else (dend, dim_out * dim_out)
                )
                value = value.reshape(edges, size)
                result[:, begin : begin + size].add_(
                    value if result.size(0) != 1 else value.sum(0, keepdim=True)
                )
            elif output == 5:
                result[:, harmonic].add_(
                    value if result.size(0) != 1 else value.sum().reshape(1)
                )
            elif output == 2:
                result[:, weight : weight + width].add_(
                    value.reshape(projection.size(0), width)
                )
            elif projected:
                result.add_(
                    value if result.size(0) != 1 else value.sum(0, keepdim=True)
                )
            else:
                value = value.reshape(edges, width)
                result[:, weight : weight + width].add_(
                    value if result.size(0) != 1 else value.sum(0, keepdim=True)
                )
