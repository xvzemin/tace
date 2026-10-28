"""Spherical O(3) tensor-product convolutions."""

import math

import torch
from e3nn import o3

from ...o3 import GauntTensorProduct, MatrixTensorProduct
from ...utils.autograd import gradient_mask
from ...utils.metadata import parse_metadata
from ..contraction import adjoint_program
from ..network import materialize


@torch.library.custom_op("eqx::o3_contraction", mutates_args=(), device_types="cuda")
def contraction(
    metadata: str,
    program: str,
    source: torch.Tensor | None,
    target: torch.Tensor | None,
    operands: list[torch.Tensor],
    *,
    batch_paths: bool = False,
) -> list[torch.Tensor]:
    """Run sparse contractions outside the tensor tracing boundary."""
    from ...kernels.cuda_graph import convolution

    results = contraction_fake(metadata, program, source, target, operands)
    terms = parse_metadata(program)

    def run(inputs, outputs):
        source, target, *values = inputs
        for result in outputs:
            result.zero_()
        count = values[0].shape[0] if source is None else source.numel()
        if count and parse_metadata(metadata)[0]:
            from .cuda import contract

            contract(metadata, terms, source, target, values, outputs, batch_paths)

    if source is None:
        run((source, target, *operands), results)
    else:
        convolution(
            "o3",
            (metadata, program, batch_paths),
            run,
            source,
            target,
            operands,
            results,
            terms,
            4,
        )
    return results


@contraction.register_fake
def contraction_fake(metadata, program, source, target, operands, *, batch_paths=False):
    program = parse_metadata(program)
    results = [None] * (1 + max(slot for _, _, pairs in program for _, slot in pairs))
    for mapping, _, pairs in program:
        for role, slot in pairs:
            if results[slot] is None:
                results[slot] = torch.empty_like(
                    operands[mapping[role]], memory_format=torch.contiguous_format
                )
    return results


def contraction_setup_context(ctx, inputs, keyword_only_inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata = metadata
    ctx.program = parse_metadata(program)
    ctx.batch_paths = keyword_only_inputs["batch_paths"]
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def contraction_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    metadata = parse_metadata(ctx.kernel_metadata)
    geometric = len(metadata) > 3
    required = gradient_mask(operands, ctx.needs_input_grad[4])
    needs_grad = list(required)
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
            if not required[index]:
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
            ctx.kernel_metadata,
            repr(program),
            source,
            target,
            values,
            batch_paths=ctx.batch_paths,
        )
        for index, slot in destinations.items():
            gradients[index] = results[slot]
    return None, None, None, None, gradients


contraction.register_autograd(
    contraction_backward, setup_context=contraction_setup_context
)


class O3TensorProductConv(torch.nn.Module):
    """Gather, couple and sum O(3) features with radial path weights.

    Parameters
    ----------
    tensor_product : e3nn.o3.TensorProduct
        Defines the irreps, instructions and normalization. Only ``uvu``
        instructions are supported. Weights are supplied to :meth:`forward`.
    backend : {"cuda", "torch"}, optional
        CUDA generates sparse contractions on first use and caches the
        compiled binaries. ``"torch"`` evaluates the CG contractions with
        PyTorch operations and automatic differentiation on either device.
    normalization : {"component", "integral", "norm"}, optional
        Spherical-harmonic normalization when vectors are supplied.
    normalize : bool, optional
        Normalize vector inputs before evaluating harmonics. If false,
        evaluate regular solid harmonics at the supplied vectors.
    batch_paths : bool, optional
        Batch compatible CUDA path groups when weights are supplied directly.
        Defaults to false.

    Notes
    -----
    Features use flattened ``ir_mul`` storage. Instructions, weights and
    output multiplicities are preserved. CUDA supports float32 and float64,
    including higher derivatives. Atomic reductions are not bitwise deterministic.
    """

    batch_paths = False

    def __init__(
        self,
        tensor_product,
        *,
        backend="cuda",
        normalization="component",
        normalize=True,
        batch_paths=False,
    ):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be torch or cuda.")
        if any(ins.connection_mode != "uvu" for ins in tensor_product.instructions):
            raise NotImplementedError(
                "O3TensorProductConv supports only 'uvu' instructions."
            )
        self.backend = backend
        self.batch_paths = batch_paths
        if normalization not in ("component", "integral", "norm"):
            raise ValueError("normalization must be integral, component or norm.")
        self.normalization = normalization
        self.normalize = normalize
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.input_dim = self.irreps_in1.dim
        self.harmonic_input_dim = self.input_dim
        self.edge_dim = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim
        self.harmonic_output_dim = self.output_dim
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
        radial_network=None,
    ):
        """Evaluate the indexed convolution.

        Parameters
        ----------
        features : torch.Tensor
            Source features, shape ``(nodes, irreps_in1.dim)``.
        edge_attrs : torch.Tensor
            Edge features, shape ``(edges, irreps_in2.dim)`` or a shared row.
        radial : torch.Tensor or tuple
            Radial features, shape ``(edges, radial_dim)`` or a shared row.
            With an empty projection, these are path weights directly.
            With ``radial_network``, a tuple of ``(tensor, kind)`` partitions
            may use ``kind="edge"``, ``"source"`` or ``"target"``.
        projection : torch.Tensor
            Shared weights, shape ``(radial_dim, weight_numel)``. Shape
            ``(0, weight_numel)`` selects directly supplied path weights.
        edge_index : torch.Tensor or None
            Source and target indices, shape ``(2, edges)``, dtype int64.
            None evaluates a row-wise tensor product without graph reduction.
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
        radial_network : torch.nn.Module, optional
            Sequential network preceding ``projection``. CUDA evaluates its
            nonlinear layers in bounded edge tiles and recomputes activations
            during backward. A final affine bias may be appended as the last
            row of ``projection``. PyTorch evaluates the network normally.

        Returns
        -------
        torch.Tensor
            Target features, shape ``(num_nodes, irreps_out.dim)``.
        """
        indexed = edge_index is not None
        if not indexed and (
            vectors is not None or radial_network is not None or projection.numel()
        ):
            raise ValueError(
                "Row-wise products require explicit attributes and weights."
            )
        count = edge_index.size(1) if indexed else features.size(0)
        if radial_network is not None and (
            self.backend != "cuda" or not features.is_cuda
        ):
            radial = radial_network(materialize(radial, edge_index))
            if radial.shape[-1] + 1 == projection.shape[0]:
                radial = torch.cat((radial, torch.ones_like(radial[:, :1])), -1)
            radial_network = None
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
        radial_values = radial[0][0] if isinstance(radial, tuple) else radial
        if any(
            value.ndim != 2
            for value in (features, edge_attrs, radial_values, projection)
        ):
            raise ValueError("Features, radial inputs and projection must be matrices.")
        if indexed and (edge_index.ndim != 2 or edge_index.size(0) != 2):
            raise ValueError("edge_index must have shape (2, edges).")
        if features.size(1) != (
            self.harmonic_input_dim if vectors is not None else self.input_dim
        ) or edge_attrs.size(1) != (
            self.amplitude_dim if vectors is not None else self.edge_dim
        ):
            raise ValueError(
                "Feature dimensions do not match the tensor-product irreps."
            )
        if projection.size(1) != self.weight_numel:
            raise ValueError("Projection width must equal weight_numel.")
        if radial_network is None and radial.size(1) != (
            projection.size(0) if projection.numel() else self.weight_numel
        ):
            raise ValueError(
                "Radial width does not match the projection or path weights."
            )
        if any(
            value.size(0) not in (1, count)
            for value in (
                (edge_attrs,) if isinstance(radial, tuple) else (edge_attrs, radial)
            )
        ):
            raise ValueError(
                "Edge attributes and radial features need one row per edge or one shared row."
            )
        num_nodes = features.size(0) if num_nodes is None else num_nodes
        if not indexed and num_nodes != features.size(0):
            raise ValueError("Row-wise products require one output row per input row.")
        operands = [
            features,
            radial,
            projection,
            edge_attrs,
            features.new_empty(1).expand(
                num_nodes,
                self.harmonic_output_dim if vectors is not None else self.output_dim,
            ),
        ]
        if self.backend == "cuda" and features.is_cuda:
            if vectors is not None:
                operands.append(vectors)
            if radial_network is not None:
                from ..network import convolve

                return convolve(
                    "o3",
                    self.harmonic_metadata
                    if vectors is not None
                    else self.kernel_metadata,
                    operands,
                    edge_index,
                    radial_network,
                )
            return contraction(
                self.harmonic_metadata if vectors is not None else self.kernel_metadata,
                repr(((tuple(range(len(operands))), False, ((4, 0),)),)),
                edge_index[0] if indexed else None,
                edge_index[1] if indexed else None,
                operands,
                batch_paths=self.batch_paths,
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
        result = self.reference(
            features, edge_attrs, radial, projection, edge_index, num_nodes
        )
        return result if vectors is None else result + vectors.sum() * 0

    def reference(
        self, features, edge_attrs, radial, projection, edge_index, num_nodes
    ):
        """Evaluate the same paths with ordinary tensor operations."""
        indexed = edge_index is not None
        edges = edge_index.size(1) if indexed else features.size(0)
        if indexed:
            source, target = edge_index
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
            x = features[:, start : start + dim1 * mul1]
            x = (x[source] if indexed else x).reshape(edges, dim1, mul1)
            y = attrs[:, attr : attr + dim2 * mul2].reshape(edges, dim2, mul2)
            cg = getattr(self, f"cg_{index}").to(features)
            if weight >= 0:
                w = weights[:, weight : weight + mul1 * mul2].reshape(edges, mul1, mul2)
                message = torch.einsum("abc,eau,ebv,euv->ecu", cg, x, y, w)
            else:
                message = torch.einsum("abc,eau,ebv->ecu", cg, x, y)
            message = (factor * message).flatten(1)
            result[:, end : end + dim_out * mul1] += (
                message.new_zeros((num_nodes, dim_out * mul1)).index_add(
                    0, target, message
                )
                if indexed
                else message
            )
        return result

    def extra_repr(self):
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"{self.weight_numel} weights, backend={self.backend!r}"
        )


class TensorProductConv(torch.nn.Module):
    """Gather source features, couple edge attributes and sum at destinations."""

    def __init__(self, tensor_product):
        super().__init__()
        self.tensor_product = tensor_product
        self.irreps_in1 = tensor_product.irreps_in1
        self.irreps_in2 = tensor_product.irreps_in2
        self.irreps_out = tensor_product.irreps_out
        self.weight_numel = tensor_product.weight_numel
        self.input_dim = self.irreps_in1.dim
        self.edge_dim = self.irreps_in2.dim
        self.output_dim = self.irreps_out.dim

    def forward(
        self,
        features,
        edge_attrs,
        edge_index,
        weight=None,
        num_nodes=None,
        *,
        radial=None,
        radial_network=None,
    ):
        """Evaluate messages and sum them at destination nodes.

        Parameters
        ----------
        features : torch.Tensor
            Source-node features of shape ``(nodes, irreps_in1.dim)``.
        edge_attrs : torch.Tensor
            Edge features of shape ``(edges, irreps_in2.dim)``.
        edge_index : torch.Tensor
            Int64 source and destination indices of shape ``(2, edges)``.
        weight : torch.Tensor, optional
            Shared factors of shape ``(weight_numel,)`` or edge factors of shape
            ``(edges, weight_numel)``, according to the tensor product.
        num_nodes : int, optional
            Number of destination nodes. Defaults to the source-node count.
        radial : torch.Tensor or tuple, optional
            Inputs to radial_network. A tuple of ``(tensor, storage)`` pairs
            may use ``"edge"``, ``"source"`` and ``"target"`` storage.
        radial_network : torch.nn.Module, optional
            Network producing edge factors from radial inputs. Requires an
            externally weighted tensor product with shared_weights=False.
            Cannot be combined with weight.

        Returns
        -------
        torch.Tensor
            Destination features of shape ``(num_nodes, irreps_out.dim)``.
        """
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("features must have shape (nodes, irreps_in1.dim).")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError("edge_index must be an int64 tensor of shape (2, edges).")
        if edge_attrs.shape != (edge_index.shape[1], self.edge_dim):
            raise ValueError("edge_attrs must have shape (edges, irreps_in2.dim).")
        if radial_network is not None:
            if weight is not None or radial is None:
                raise ValueError(
                    "Provide radial inputs instead of weight when using radial_network."
                )
            if (
                self.tensor_product.shared_weights
                or self.tensor_product.internal_weights
            ):
                raise ValueError("radial_network requires external, unshared weights.")
            weight = radial_network(materialize(radial, edge_index))
        elif radial is not None:
            raise ValueError("radial inputs require radial_network.")
        if (
            weight is not None
            and weight.ndim > 1
            and weight.shape != (edge_index.shape[1], self.weight_numel)
        ):
            raise ValueError("Edge weights must have shape (edges, weight_numel).")
        messages = self.tensor_product(features[edge_index[0]], edge_attrs, weight)
        num_nodes = features.shape[0] if num_nodes is None else num_nodes
        return messages.new_zeros((num_nodes, self.output_dim)).index_add(
            0, edge_index[1], messages
        )

    def extra_repr(self):
        return f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}"


class GauntTensorProductConv(TensorProductConv):
    """Gather, multiply spherical signals and sum at destination nodes.

    Parameters
    ----------
    tensor_product : eqx.o3.GauntTensorProduct
        Defines the irreps, factors and normalization.

    Notes
    -----
    Features use ``mul_ir`` layout. PyTorch operations support CPU, CUDA and
    automatic differentiation. Edge factors and messages are materialized.
    """

    def __init__(self, tensor_product):
        if not isinstance(tensor_product, GauntTensorProduct):
            raise TypeError("tensor_product must be a GauntTensorProduct.")
        super().__init__(tensor_product)


class MatrixTensorProductConv(TensorProductConv):
    """Gather, multiply matrix representations and sum at destination nodes.

    Parameters
    ----------
    tensor_product : eqx.o3.MatrixTensorProduct
        Defines the irreps, factors, matrix degree and normalization.

    Notes
    -----
    Features use ``mul_ir`` layout. PyTorch operations support CPU, CUDA and
    automatic differentiation. Edge factors and messages are materialized.
    """

    def __init__(self, tensor_product):
        if not isinstance(tensor_product, MatrixTensorProduct):
            raise TypeError("tensor_product must be a MatrixTensorProduct.")
        super().__init__(tensor_product)
