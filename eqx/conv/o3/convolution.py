################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""O(3) convolutions with external radial projections."""

import math

import torch
from e3nn import o3

from ..._metadata import parse_metadata
from ..contraction import adjoint_program


@torch.library.custom_op("eqx::o3_contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Run sparse contractions outside the tensor tracing boundary."""
    from ...kernels.cuda_graph import convolution

    results = contraction_fake(metadata, program, source, target, operands)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        source, target, *values = inputs
        for result in outputs:
            result.zero_()
        if source.numel() and parse_metadata(metadata)[0]:
            from .cuda import contract

            contract(metadata, terms, source, target, values, outputs)

    convolution(
        "o3", (metadata, program), run, source, target, operands, results, terms, 4
    )
    return results


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands):
    program = parse_metadata(program)
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
    ctx.program = parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    metadata = parse_metadata(ctx.kernel_metadata)
    geometric = len(metadata) > 3
    needs_grad = list(ctx.needs_input_grad[4])
    if geometric:
        for mapping, _, _ in ctx.program:
            needs_grad[mapping[5]] = False
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        needs_grad,
        metadata[2],
    )
    if geometric:
        terms = {}
        cotangents = {id(value): index for index, value in enumerate(values)}
        for mapping, weighted, pairs in ctx.program:
            index = mapping[5]
            if not ctx.needs_input_grad[4][index]:
                continue
            for output, slot in pairs:
                if grad_outputs[slot] is None or (
                    output == 2 and not operands[mapping[2]].numel()
                ):
                    continue
                destination = destinations.setdefault(index, len(destinations))
                replacement = list(mapping)
                replacement[output] = cotangents[id(grad_outputs[slot])]
                replacement.append(index)
                key = tuple(replacement), metadata[2] and (weighted or output in (1, 2))
                terms.setdefault(key, []).append((len(mapping), destination))
        program += tuple(
            (mapping, weighted, tuple(pairs))
            for (mapping, weighted), pairs in terms.items()
        )
    gradients = [None] * len(operands)
    if program:
        results = contraction(
            ctx.kernel_metadata, repr(program), source, target, values
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


class O3TensorProductConv(torch.nn.Module):
    """Gather, couple and sum O(3) features with a fused radial projection.

    Parameters
    ----------
    tensor_product : e3nn.o3.TensorProduct
        Defines the irreps, instructions and normalization. Only ``uvu``
        instructions are supported. Weights are supplied to :meth:`forward`.
    backend : {"cuda", "torch"}, optional
        CUDA generates sparse contractions on first use and caches the
        compiled binaries. CPU inputs use ordinary PyTorch operations.
    normalization : {"component", "integral", "norm"}, optional
        Spherical-harmonic normalization when vectors are supplied.
    normalize : bool, optional
        Normalize vector inputs before evaluating harmonics. If false,
        evaluate regular solid harmonics at the supplied vectors.

    Notes
    -----
    Features use flattened ``ir_mul`` order within each irrep. Instruction
    order, weight order and output multiplicities are preserved, including
    repeated output irreps. The CUDA contraction fuses source gathering,
    sparse Clebsch--Gordan products and target reduction.
    Edge messages are not materialized. Radial projections use bounded
    temporary workspaces that are recomputed rather than saved for backward.

    Forward and transposed contractions share a multilinear program. Each
    backward transposes that program again, supporting force training and
    higher derivatives. Shared angular factors and partial gradients are
    reused across paths and derivative terms.
    Vector inputs use fixed Cartesian harmonic polynomials. Their derivatives
    are contracted in registers before channel reduction, without storing
    spherical-harmonic cotangents. Integer polynomial coefficients are
    collected before normalization; there is no angular grid or fitted basis.
    Compatible path tiles share a CUDA grid and are interleaved over the same
    edge ranges. Compiled register usage limits fusion; larger programs retain
    separate launches. Concurrent reductions preserve independent path outputs.
    Reductions use atomic additions and are not generally deterministic.
    Warm up required derivatives before CUDA Graph capture.
    """

    def __init__(
        self,
        tensor_product,
        *,
        backend="cuda",
        normalization="component",
        normalize=True,
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be torch or cuda.")
        if any(ins.connection_mode != "uvu" for ins in tensor_product.instructions):
            raise NotImplementedError(
                "O3TensorProductConv supports only 'uvu' instructions."
            )
        self.backend = backend
        if normalization not in ("component", "integral", "norm"):
            raise ValueError("normalization must be integral, component or norm.")
        self.normalization = normalization
        self.normalize = normalize
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.input_dim = self.irreps_in1.dim
        self.edge_dim = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim
        self.instructions = tuple(tensor_product.instructions)
        self.weight_numel = tensor_product.weight_numel
        slices = [
            ir.slices() for ir in (self.irreps_in1, self.irreps_in2, self.irreps_out)
        ]
        paths, offset = [], 0
        for ins in self.instructions:
            mul1, ir1 = self.irreps_in1[ins.i_in1]
            mul2, ir2 = self.irreps_in2[ins.i_in2]
            _, ir_out = self.irreps_out[ins.i_out]
            weight = offset if ins.has_weight else -1
            if ins.has_weight:
                offset += math.prod(ins.path_shape)
            if not mul1 or not mul2 or not ins.path_weight:
                continue
            cg = o3.wigner_3j(ir1.l, ir2.l, ir_out.l)
            self.register_buffer(f"cg_{len(paths)}", cg, persistent=False)
            paths.append(
                (
                    slices[0][ins.i_in1].start,
                    slices[1][ins.i_in2].start,
                    slices[2][ins.i_out].start,
                    mul1,
                    mul2,
                    ir1.dim,
                    ir2.dim,
                    ir_out.dim,
                    weight,
                    ins.path_weight,
                    tuple(
                        (a, b, c, float(cg[a, b, c]))
                        for a, b, c in cg.nonzero().tolist()
                    ),
                )
            )
        self.paths = tuple(paths)
        self.kernel_metadata = repr(
            (self.paths, self.weight_numel, any(path[8] < 0 for path in paths))
        )
        amplitude_offsets, offset = {}, 0
        for (mul, _), section in zip(self.irreps_in2, slices[1]):
            amplitude_offsets[section.start] = offset
            offset += mul
        self.amplitude_dim = offset
        self.harmonic_metadata = repr(
            (
                tuple((p[0], amplitude_offsets[p[1]], *p[2:]) for p in self.paths),
                self.weight_numel,
                any(path[8] < 0 for path in paths),
                normalization,
            )
        )

    def forward(
        self,
        features,
        edge_attrs,
        radial,
        projection,
        edge_index,
        num_nodes=None,
        *,
        vectors=None,
        amplitudes=None,
    ):
        """Evaluate the indexed convolution.

        Parameters
        ----------
        features : torch.Tensor
            Source features, shape ``(nodes, irreps_in1.dim)``.
        edge_attrs : torch.Tensor
            Edge features, shape ``(edges, irreps_in2.dim)`` or a shared row.
        radial : torch.Tensor
            Radial features, shape ``(edges, radial_dim)`` or a shared row.
            With an empty projection, these are path weights directly.
        projection : torch.Tensor
            Shared weights, shape ``(radial_dim, weight_numel)``. Shape
            ``(0, weight_numel)`` selects directly supplied path weights.
        edge_index : torch.Tensor
            Source and target indices, shape ``(2, edges)``, dtype int64.
        num_nodes : int, optional
            Number of target nodes. Defaults to the number of source nodes.
        vectors : torch.Tensor, optional
            Edge vectors, shape ``(edges, 3)``. When supplied, replace
            ``edge_attrs`` with spherical harmonics, and contract
            Cartesian derivatives directly without harmonic cotangents.
            The ``normalize`` and ``normalization`` settings apply here only.
        amplitudes : torch.Tensor, optional
            Harmonic amplitudes in irrep multiplicity order. Shape is
            ``(edges, irreps_in2.num_irreps)``; either dimension may be one.
            Used only with ``vectors``; defaults to one.

        Returns
        -------
        torch.Tensor
            Target features, shape ``(num_nodes, irreps_out.dim)``.
        """
        if vectors is not None:
            if vectors.ndim != 2 or vectors.shape != (edge_index.size(1), 3):
                raise ValueError("vectors must have shape (edges, 3).")
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            amplitudes = (
                features.new_ones((1, self.amplitude_dim))
                if amplitudes is None
                else amplitudes.expand(amplitudes.size(0), self.amplitude_dim)
            )
            edge_attrs = amplitudes
        if any(value.ndim != 2 for value in (features, edge_attrs, radial, projection)):
            raise ValueError("Features, radial inputs and projection must be matrices.")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        if features.size(1) != self.input_dim or edge_attrs.size(1) != (
            self.amplitude_dim if vectors is not None else self.edge_dim
        ):
            raise ValueError(
                "Feature dimensions do not match the tensor-product irreps."
            )
        if projection.size(1) != self.weight_numel:
            raise ValueError("Projection width must equal weight_numel.")
        if radial.size(1) != (
            projection.size(0) if projection.numel() else self.weight_numel
        ):
            raise ValueError(
                "Radial width does not match the projection or path weights."
            )
        if any(
            value.size(0) not in (1, edge_index.size(1))
            for value in (edge_attrs, radial)
        ):
            raise ValueError(
                "Edge attributes and radial features need one row per edge or one shared row."
            )
        num_nodes = features.size(0) if num_nodes is None else num_nodes
        operands = [
            features,
            radial,
            projection,
            edge_attrs,
            features.new_empty(1).expand(num_nodes, self.output_dim),
        ]
        if self.backend == "cuda" and features.is_cuda:
            if vectors is not None:
                operands.append(vectors)
            return contraction(
                self.harmonic_metadata if vectors is not None else self.kernel_metadata,
                repr(((tuple(range(len(operands))), False, ((4, 0),)),)),
                edge_index[0],
                edge_index[1],
                operands,
            )[0]
        if vectors is not None:
            sections, offset = [], 0
            for mul, ir in self.irreps_in2:
                harmonic = o3.spherical_harmonics(
                    ir.l, vectors, normalize=False, normalization=self.normalization
                )
                sections.append(
                    (
                        harmonic.unsqueeze(-1)
                        * amplitudes[:, None, offset : offset + mul]
                    ).flatten(1)
                )
                offset += mul
            edge_attrs = torch.cat(sections, dim=-1)
        return self.reference(
            features, edge_attrs, radial, projection, edge_index, num_nodes
        )

    def reference(
        self, features, edge_attrs, radial, projection, edge_index, num_nodes
    ):
        """Evaluate the same paths with ordinary tensor operations."""
        source, target = edge_index
        edges = source.numel()
        weights = radial @ projection if projection.numel() else radial
        weights = weights.expand(edges, -1)
        attrs = edge_attrs.expand(edges, -1)
        # Keep zero derivatives defined for empty graphs and unused operands.
        zero = sum(
            value.sum() * 0 for value in (features, edge_attrs, radial, projection)
        )
        result = features.new_zeros((num_nodes, self.output_dim)) + zero
        for index, path in enumerate(self.paths):
            start, attr, end, mul1, mul2, dim1, dim2, dim_out, weight, factor, _ = path
            x = features[source, start : start + dim1 * mul1].reshape(edges, dim1, mul1)
            y = attrs[:, attr : attr + dim2 * mul2].reshape(edges, dim2, mul2)
            cg = getattr(self, f"cg_{index}").to(features)
            if weight >= 0:
                w = weights[:, weight : weight + mul1 * mul2].reshape(edges, mul1, mul2)
                message = torch.einsum("abc,eau,ebv,euv->ecu", cg, x, y, w)
            else:
                message = torch.einsum("abc,eau,ebv->ecu", cg, x, y)
            message = (factor * message).flatten(1)
            result[:, end : end + dim_out * mul1] += message.new_zeros(
                (num_nodes, dim_out * mul1)
            ).index_add(0, target, message)
        return result

    def extra_repr(self):
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"{self.weight_numel} weights, backend={self.backend!r}"
        )
