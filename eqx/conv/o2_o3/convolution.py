################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Indexed aligned-frame contractions and their transposes."""

import math
from ast import literal_eval
from dataclasses import dataclass
from functools import lru_cache

import torch
from e3nn import o3

from ..contraction import adjoint_program, parse_program


@dataclass(eq=False)
class _KernelPlan:
    """Tensor-free scheduling metadata shared by compiled calls."""

    path_data: tuple
    weight_numel: int
    has_unweighted: bool
    sparse_paths: tuple
    scalar_paths: tuple = ()


@lru_cache(maxsize=256)
def kernel_plan(metadata):
    return _KernelPlan(*literal_eval(metadata))


@torch.library.custom_op("eqx::contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Keep runtime scheduling and kernel compilation outside tensor tracing."""
    plan = kernel_plan(metadata)
    results = contraction_fake(metadata, program, source, target, operands)
    for result in results:
        result.zero_()
    if source.numel() and plan.path_data:
        from .cuda import contract_many

        calls = [
            (
                tuple(role for role, _ in pairs),
                tuple(operands[i] for i in mapping),
                tuple(results[slot] for _, slot in pairs),
                weighted_only,
            )
            for mapping, weighted_only, pairs in parse_program(program)
        ]
        contract_many(plan, source, target, calls)
    return results


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands):
    program = parse_program(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def contraction_setup_context(ctx, inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse_program(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        ctx.needs_input_grad[4],
        kernel_plan(ctx.kernel_metadata).has_unweighted,
    )
    gradients = [None] * len(operands)
    if program:
        results = contraction(
            ctx.kernel_metadata,
            repr(program),
            source,
            target,
            values,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


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
        program, values, destinations = adjoint_program(
            ctx.program,
            operands,
            grad_outputs,
            ctx.needs_input_grad[4:],
            ctx.plan.has_unweighted,
        )
        gradients = [None] * len(operands)
        if program:
            results = _Contraction.apply(ctx.plan, program, source, target, *values)
            for index, slot in destinations.items():
                gradients[index] = results[slot]
        return None, None, None, None, *gradients


class O2O3TensorProductConv(torch.nn.Module):
    """Evaluate an O(3) tensor-product convolution in aligned O(2) frames.

    Parameters
    ----------
    tensor_product : O3TensorProduct
        Tensor product defining paths, normalization and feature layouts.
    backend : {"torch", "cuda"}, optional
        Execution backend. Defaults to generated ``"cuda"`` kernels on CUDA
        and ordinary tensor operations on CPU.
        The CUDA backend supports ``"uvu"`` instructions only. Use ``"torch"``
        for channel-mixing ``"uvw"`` instructions.

    Notes
    -----
    The scalar contraction has seven operands: input node features, radial
    features, radial projection, input rotations, output rotations, harmonic
    amplitudes, and output node features. Computing any one operand's adjoint
    uses the same contraction with that operand designated as the output.
    This rule also applies to higher derivatives. Accelerated contractions fuse
    gathers, both feature rotations, sparse coupling and reductions without
    retaining edge messages. Radial projections use matrix products
    in bounded edge chunks.

    With ``vectors`` supplied, the rotation matrices are cached values.
    Direction derivatives instead contract sparse angular tensors obtained by
    applying rotation generators to every angular index, including indices
    introduced by earlier derivatives. This retains recursive higher
    derivatives without allocating rotation-matrix adjoints. Radial amplitudes
    remain independent differentiable operands. Paths with degree-zero
    harmonics bypass both rotations, without merging paths or their weights.

    Rotation matrices are supplied as packed degree blocks, not zero-padded
    block-diagonal matrices. Paths sharing an input block reuse its rotation
    within each angular/channel tile. Compatible output paths of the same
    degree rotate jointly in bounded path/channel tiles. Their independent
    output slots are preserved, and output-matrix adjoints are summed across
    paths before the channel reduction. Sparse CG coefficients are combined
    with output rotation entries and reused across channels, without storing
    edge-wise coupling matrices. The transposed contractions use the same
    combined entries; rotation-matrix adjoints retain the explicit coefficients.
    Instructions writing the same output entry are summed in the aligned
    frame before their shared inverse rotation. Independent output entries
    remain separate, even when they have the same degree.
    Input adjoints are accumulated locally before the inverse rotation.
    Workspaces for projected weights are reused
    across chunks and across mixed derivative terms, and are not saved for
    backward. Shared radial inputs are projected once per call; their projected
    adjoints are reduced across chunks before the projection transpose.
    Per-edge weight adjoints with a single contribution overwrite their
    workspaces directly, avoiding a separate clear and read-modify-write.
    Shared and mixed adjoints retain additive reduction.
    Path dependencies are retained through each transpose, including
    mixtures of weighted and unweighted instructions. Channelwise contractions
    use the same register-resident kernels at every angular degree. Mixed
    adjoints share local rotations and accumulate into their destinations
    before inverse rotations. Direction-vector cotangents are reduced within
    each channel warp before their inverse rotation. Channel-independent vector
    operands are rotated once per edge tile. Compilation partitions derivative
    programs using shared dependencies, reduction axes and register usage, rather
    than an angular-degree threshold. Wide channel tiles share rotation matrices across
    warps, while narrow tiles process independent edges per warp. Shared input
    and output rotations accumulate into one adjoint. Their path and channel
    contributions are combined in bounded shared memory before global reduction.
    Compiled occupancy provides initial block sizes. Outside CUDA Graph capture,
    sufficiently large calls measure candidate sizes using bounded private
    outputs; the selected launch configurations are cached. Cached phases launch
    together.
    Source- or receiver-owned tiles accumulate node contributions in registers;
    split rows and shared gradients use atomic additions. Node boundaries are
    identified inside sorted edge tiles without degree counts or task buffers.
    Mixed direction derivatives reuse owned node inputs across incident edges.
    Their ownership is selected per compiled tile, retaining edge execution when
    persistent accumulators would reduce occupancy, require spilling, or need
    further path splitting.
    Plans are reused for unchanged index storage, including detached views,
    with capture-safe sorting for CUDA Graph replay. CUDA kernels
    are generated and compiled with NVRTC on first use; their binaries are
    cached independently of graph sizes and learned parameters. Warm up the
    required derivatives before CUDA Graph capture. Atomic reductions mean
    that summation order is not generally deterministic.
    """

    def __init__(self, tensor_product, *, backend="cuda"):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        if backend == "cuda" and any(
            ins.connection_mode != "uvu" for ins in tensor_product.instructions
        ):
            raise NotImplementedError(
                "The CUDA convolution supports only 'uvu' instructions. "
                "Use backend='torch' for other connection modes."
            )
        self.backend = backend
        self.irreps_in = tensor_product.irreps_in1
        self.irreps_out = tensor_product.irreps_out
        self.output_dim = self.irreps_out.dim
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        self.num_harmonics = tensor_product.num_harmonics
        self.degree_offsets = tuple(
            sum((2 * k + 1) ** 2 for k in range(l))
            for l in range(tensor_product.lmax + 1)
        )
        input_slices = self.irreps_in.slices()
        output_slices = self.irreps_out.slices()
        offset = 0
        self.path_data = []
        self.sparse_paths = []
        scalar_paths = []
        for ins in self.instructions:
            mul, ir = self.irreps_in[ins.i_in1]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            ir_sh = tensor_product.irreps_in2[ins.i_in2].ir
            weight_offset = offset if ins.has_weight else -1
            if ins.has_weight:
                offset += math.prod(ins.path_shape)
            if not mul or not mul_out or not ins.path_weight:
                continue
            if ir_sh.l == 0:
                scalar_paths.append(len(self.path_data))
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
            # Static coefficients and tensor buffers use the same construction
            # precision, including when the module is later promoted to float64.
            cg = (cg * (pole * ins.path_weight)).to(torch.get_default_dtype())
            # Each row and column of the real m2=0 slice has at most one nonzero.
            if (cg.count_nonzero(0) > 1).any() or (cg.count_nonzero(1) > 1).any():
                raise ValueError("Expected a signed-order-diagonal CG slice.")
            self.register_buffer(f"cg_{len(self.path_data)}", cg, persistent=False)
            path = (
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
            self.path_data.append((ins.connection_mode, path))
            self.sparse_paths.append(
                tuple((m, n, float(cg[m, n])) for m, n in cg.nonzero().tolist())
            )

        self.has_unweighted = any(path[8] < 0 for _, path in self.path_data)
        self.kernel_metadata = repr(
            (
                tuple(self.path_data),
                self.weight_numel,
                self.has_unweighted,
                tuple(self.sparse_paths),
            )
        )
        self.direction_metadata = repr(
            (*literal_eval(self.kernel_metadata), tuple(scalar_paths))
        )

    def forward(
        self,
        features,
        radial,
        projection,
        wigner,
        amplitudes,
        edge_index,
        num_nodes,
        *,
        vectors=None,
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
        vectors : torch.Tensor, optional
            Nonzero frame directions, with shape ``(edges, 3)`` or ``(1, 3)``.
            When supplied, ``wigner`` must contain their alignment matrices.
            Direction derivatives use sparse rotation generators rather than
            matrix adjoints. The matrices are treated as cached values, and
            zero-degree harmonic paths bypass both rotations. Without vectors,
            the matrices remain independent differentiable inputs.

        Returns
        -------
        torch.Tensor
            Target features of shape ``(num_nodes, irreps_out.dim)`` in the
            declared, unsimplified ``ir_mul`` layout.
        """
        # A zero-stride placeholder supplies the output shape without allocating
        # a second node output. It is never read by the forward contraction.
        output = features.new_empty(1).expand(num_nodes, self.output_dim)
        # Both rotations depend on the same tensor. Keep one operand so their
        # adjoints accumulate together, including in recursively transposed calls.
        program = (((0, 1, 2, 3, 3, 4, 5), False, ((6, 0),)),)
        operands = [
            features,
            radial,
            projection,
            wigner,
            amplitudes,
            output,
        ]
        if vectors is not None:
            from .geometry import direction_contraction

            operands[3] = wigner.detach()
            return direction_contraction(
                self.direction_metadata,
                repr(tuple((0, *term) for term in program)),
                vectors,
                edge_index[0],
                edge_index[1],
                operands,
                self.backend == "cuda",
            )[0]
        if self.backend == "cuda" and features.is_cuda:
            return contraction(
                self.kernel_metadata,
                repr(program),
                edge_index[0],
                edge_index[1],
                operands,
            )[0]
        return _Contraction.apply(
            self,
            program,
            edge_index[0],
            edge_index[1],
            *operands,
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
            cg = getattr(self, f"cg_{index}").to(x)
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
