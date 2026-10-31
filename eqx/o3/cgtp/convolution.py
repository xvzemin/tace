"""Spherical O(3) tensor-product convolutions."""

import math

import torch
from e3nn import o3

from ...backends.torch.tensor_product import TensorProduct as GroupedTensorProduct
from ...backends.torch.tensor_product import instruction_groups
from ...ops.o3 import contraction
from ...ops.radial import materialize


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

    Examples
    --------
    >>> from e3nn import o3
    >>> from eqx.conv import O3TensorProductConv
    >>> tp = o3.TensorProduct(
    ...     "2x0e", "1x1o", "2x1o", [(0, 0, 0, "uvu", True)],
    ...     internal_weights=False, shared_weights=False,
    ... )
    >>> conv = O3TensorProductConv(tp, backend="torch")
    >>> x = torch.randn(3, 2)
    >>> edge_attrs = torch.randn(4, 3)
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> radial = torch.randn(4, 8)
    >>> projection = torch.randn(8, tp.weight_numel)
    >>> conv(x, edge_attrs, radial, projection, edge_index).shape
    torch.Size([3, 6])
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
        self.output_sections = tuple(
            (section.start, section.stop - section.start) for section in slices[2]
        )
        self.harmonic_entries = tuple((mul, ir.l) for mul, ir in self.irreps_in2)
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
        couplings = [
            (
                self.irreps_in1[p.i_in1].ir.l,
                self.irreps_in2[p.i_in2].ir.l,
                self.irreps_out[p.i_out].ir.l,
            )
            for p in self.instructions
        ]
        self.reference_order, groups = instruction_groups(self.instructions, couplings)
        self.grouped_reference = len(groups) * 2 < len(self.instructions)
        if self.grouped_reference:
            for degrees in dict.fromkeys(couplings):
                self.register_buffer(
                    "reference_cg_" + "_".join(map(str, degrees)),
                    o3.wigner_3j(*degrees),
                    persistent=False,
                )
            self.register_buffer(
                "reference_scales",
                torch.tensor(
                    [self.instructions[i].path_weight for i in self.reference_order],
                    dtype=torch.float64,
                ),
                persistent=False,
            )
            self.reference_contraction = GroupedTensorProduct(
                [
                    tuple((entry.mul, entry.ir.dim) for entry in irreps)
                    for irreps in (self.irreps_in1, self.irreps_in2, self.irreps_out)
                ],
                groups,
                self.couple,
                layout="ir_mul",
            )
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
                from ...ops.radial import convolve

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
            for mul, degree in self.harmonic_entries:
                harmonic = o3.spherical_harmonics(
                    degree, vectors, normalize=False, normalization=self.normalization
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
        if self.grouped_reference:
            message = self.reference_contraction(
                features[source] if indexed else features,
                attrs,
                weights,
                self.reference_scales.to(features),
            )
        else:
            inputs, angular, outputs = {}, {}, {}
            for index, path in enumerate(self.paths):
                start, attr, end, mul1, mul2, dim1, dim2, dim_out, weight, factor, _ = (
                    path
                )
                key = start, mul1, dim1
                if key not in inputs:
                    x = features[:, start : start + dim1 * mul1]
                    inputs[key] = (x[source] if indexed else x).reshape(
                        edges, dim1, mul1
                    )
                other = attr, mul2, dim2
                if other not in angular:
                    angular[other] = attrs[:, attr : attr + dim2 * mul2].reshape(
                        edges, dim2, mul2
                    )
                x, y = inputs[key], angular[other]
                cg = getattr(self, f"cg_{index}").to(features)
                if weight >= 0:
                    w = weights[:, weight : weight + mul1 * mul2].reshape(
                        edges, mul1, mul2
                    )
                    value = torch.einsum("abc,eau,ebv,euv->ecu", cg, x, y, w)
                else:
                    value = torch.einsum("abc,eau,ebv->ecu", cg, x, y)
                value = (factor * value).flatten(1)
                outputs[end] = value if end not in outputs else outputs[end] + value
            blocks = [
                outputs[start]
                if start in outputs
                else features.new_zeros((edges, width))
                for start, width in self.output_sections
            ]
            message = (
                torch.cat(blocks, -1) if blocks else features.new_zeros((edges, 0))
            )
        return (
            features.new_zeros((num_nodes, self.output_dim)).index_add(
                0, target, message
            )
            if indexed
            else message
        ) + zero

    def couple(self, x, y, l1, l2, l3):
        """Contract angular indices while retaining independent channels."""
        cg = getattr(self, f"reference_cg_{l1}_{l2}_{l3}").to(x)
        return torch.einsum("...a,...b,abc->...c", x, y, cg)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if getattr(self, "grouped_reference", False):
            self.reference_scales = torch.tensor(
                [self.instructions[i].path_weight for i in self.reference_order],
                dtype=torch.float64,
                device="cpu",
            ).to(self.reference_scales)
        return self

    def extra_repr(self):
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"{self.weight_numel} weights, backend={self.backend!r}"
        )
