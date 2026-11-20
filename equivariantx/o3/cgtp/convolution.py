"""Spherical O(3) tensor-product convolutions."""

import math

import torch
from e3nn import o3

from ...ops import config, resolve_backend
from ...ops.common.convolution import edge_indices
from ...ops.common.radial import cast_radial, radial_features
from ...ops.common.torch.tensor_product import TensorProduct as GroupedTensorProduct
from ...ops.common.torch.tensor_product import instruction_groups
from ...ops.common.weights import IndexedProjection, prepare_indexed, radial_policy
from ...ops.precision import validate_module_precision
from ...ops.torch.o3.cgtp import convolution as torch_kernel


class O3TensorProductConv(torch.nn.Module):
    """Gather, couple and sum O(3) features in flattened ``ir_mul`` layout.

    Parameters
    ----------
    tensor_product : e3nn.o3.TensorProduct
        Defines the irreps, instructions and normalization. Only ``uvu``
        instructions are supported. Weights are supplied to :meth:`forward`.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        CUDA contractions in float32 or float64, or PyTorch operations.
        CPU tensors use PyTorch. Both backends support higher derivatives.
    normalization : {"component", "integral", "norm"}, optional
        Spherical-harmonic normalization when vectors are supplied.
    normalize : bool, optional
        Normalize vector inputs before evaluating harmonics. If false,
        evaluate regular solid harmonics at the supplied vectors.
    batch_paths : bool, optional
        Batch compatible CUDA path groups when weights are supplied directly.
        Defaults to false.
    radial_mode : {"last", "full"}, optional
        ``"last"`` keeps hidden MLP activations and fuses the final projection.
        ``"full"`` includes the complete radial network in the convolution
        program and recomputes its activations during differentiation.

    Examples
    --------
    >>> import torch
    >>> import equivariantx as eqx
    >>> from e3nn import o3
    >>> tp = o3.TensorProduct(
    ...     "2x0e", "1x1o", "2x1o", [(0, 0, 0, "uvu", True)],
    ...     internal_weights=False, shared_weights=False,
    ... )
    >>> conv = eqx.conv.O3TensorProductConv(tp, backend="torch")
    >>> x = torch.randn(3, 2)
    >>> edge_attrs = torch.randn(4, 3)
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> radial = torch.randn(4, 8)
    >>> projection = torch.randn(8, tp.weight_numel)
    >>> conv(x, edge_attrs, radial, projection, edge_index).shape
    torch.Size([3, 6])
    """

    batch_paths = False
    angular_groups = ()

    def __init__(
        self,
        tensor_product,
        *,
        backend=None,
        normalization="component",
        normalize=True,
        batch_paths=False,
        radial_mode="last",
    ):
        backend = resolve_backend(backend)
        super().__init__()
        if any(ins.connection_mode != "uvu" for ins in tensor_product.instructions):
            raise NotImplementedError(
                "O3TensorProductConv supports only 'uvu' instructions."
            )
        self.backend = backend
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be last or full.")
        self.radial_mode = radial_mode
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
        self.unit_harmonic_copies = all(mul == 1 for mul, _ in self.harmonic_entries)
        degrees = [degree for _, degree in self.harmonic_entries]
        self.harmonics = None
        if degrees:
            try:
                self.harmonics = o3.SphericalHarmonics(
                    degrees, normalize=False, normalization=normalization
                )
            except NotImplementedError:
                # Precomputed attributes also support degrees not implemented by e3nn.
                pass
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
        self.angular_groups = (
            torch_kernel.angular_groups(self) if backend == "torch" else ()
        )
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
        edge_pairs=None,
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
        edge_index : torch.Tensor, tuple of torch.Tensor or None
            Source and target int64 indices, stacked as ``(2, edges)`` or
            supplied as two vectors.
            None evaluates a row-wise tensor product without graph reduction.
        num_nodes : int, optional
            Number of target nodes. Defaults to the number of source nodes.
        vectors : torch.Tensor, optional
            Edge vectors, shape ``(edges, 3)``. When supplied, replace
            ``edge_attrs`` with spherical harmonics, and contract
            Cartesian derivatives directly without harmonic cotangents.
            The ``normalize`` and ``normalization`` settings apply here only.
            Higher-precision vectors are encoded before casting the angular
            features to the node-feature dtype.
        amplitudes : torch.Tensor, optional
            Harmonic amplitudes in irrep multiplicity order. Shape is
            ``(edges, irreps_in2.num_irreps)``; either dimension may be one.
            Used only with ``vectors``; defaults to one.
        radial_network : torch.nn.Module, optional
            Network preceding ``projection``. Omit it to supply hidden features
            and fuse only the final projection. ``radial_mode`` controls hidden
            layer evaluation on either backend. Append an affine bias as the last projection row
            and a constant input channel when no network is supplied.
        edge_pairs : equivariantx.conv.EdgePairs, optional
            Supply invariant radial inputs per undirected pair. Geometry and messages remain directed.

        Returns
        -------
        torch.Tensor
            Target features, shape ``(num_nodes, irreps_out.dim)``.
        """
        pair_index = None
        if edge_pairs is not None:
            if edge_index is None:
                raise ValueError("Pair weights require a graph convolution.")
            from ...ops.common.edge_pairs import reciprocal_weights

            compact = self.backend == "cuda" and features.is_cuda
            radial, projection, radial_network = reciprocal_weights(
                self,
                edge_pairs,
                radial,
                projection,
                radial_network,
                edge_index,
                compact=compact,
                dtype=features.dtype,
            )
            if compact and radial_network is None and not projection.shape[0]:
                pair_index = edge_pairs.inverse
        validate_module_precision(
            self,
            (
                features,
                edge_attrs,
                radial,
                projection,
                edge_index,
                num_nodes,
                vectors,
                amplitudes,
                radial_network,
            ),
        )
        indexed = edge_index is not None
        if not indexed and (
            vectors is not None or radial_network is not None or projection.numel()
        ):
            raise ValueError(
                "Row-wise products require explicit attributes and weights."
            )
        if indexed:
            edge_index = edge_indices(edge_index)
        count = edge_index[0].numel() if indexed else features.size(0)
        radial = cast_radial(radial, features.dtype)
        projection = projection.to(features)
        if isinstance(radial_network, IndexedProjection):
            compact = (
                indexed
                and self.backend == "cuda"
                and features.is_cuda
                and not radial_network.scales
            )
            if (
                compact
                and 0 < len(radial_network.terms) <= 3
                and radial_network.packable_projection()
            ):
                radial, pair_index = radial_network.packed_terms(project=True)
                radial_network = None
            else:
                radial, radial_network = radial_network.bind_pairs(radial, self)
                if (
                    compact
                    and radial_network.lookup_only
                    and 0 < len(radial_network.terms) <= 4
                ):
                    radial, pair_index = radial_network.packed_terms()
                    radial_network = None
                else:
                    radial, radial_network = prepare_indexed(
                        self, radial, radial_network, edge_index, projection, bound=True
                    )
        elif radial_network is not None and (
            radial_policy(self, radial_network) == "last"
            or (
                config.TORCH_REFERENCE
                and (self.backend == "torch" or not features.is_cuda)
            )
        ):
            radial = radial_features(
                radial_network,
                radial,
                edge_index,
                projection,
                reference=config.TORCH_REFERENCE
                and (self.backend == "torch" or not features.is_cuda),
            )
            radial_network = None
        if vectors is not None:
            if vectors.ndim != 2 or vectors.shape != (count, 3):
                raise ValueError("vectors must have shape (edges, 3).")
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            if amplitudes is not None:
                amplitudes = amplitudes.expand(amplitudes.size(0), self.amplitude_dim)
            if (
                vectors.dtype != features.dtype
                or self.backend == "torch"
                or not features.is_cuda
            ):
                edge_attrs = torch_kernel.harmonic_attributes(self, vectors, amplitudes)
                # The constant harmonic has a zero (not absent) vector gradient.
                if self.edge_dim == self.amplitude_dim:
                    edge_attrs = edge_attrs + vectors[..., :0].sum()
                vectors = None
            else:
                if amplitudes is None:
                    amplitudes = features.new_ones((1, self.amplitude_dim))
                edge_attrs = amplitudes
        edge_attrs = edge_attrs.to(features)
        radial_values = radial[0][0] if isinstance(radial, tuple) else radial
        if any(
            value.ndim != 2
            for value in (features, edge_attrs, radial_values, projection)
        ):
            raise ValueError("Features, radial inputs and projection must be matrices.")
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
                (edge_attrs,)
                if isinstance(radial, tuple) or pair_index is not None
                else (edge_attrs, radial)
            )
        ):
            raise ValueError(
                "Edge attributes and radial features need one row per edge or one shared row."
            )
        num_nodes = features.size(0) if num_nodes is None else num_nodes
        if not indexed and num_nodes != features.size(0):
            raise ValueError("Row-wise products require one output row per input row.")
        if self.backend == "cuda" and radial_network is None:
            from ...ops.common.radial import align_projection

            radial, projection = align_projection(radial, projection)
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.o3.cgtp import convolution as cuda_kernel

            return cuda_kernel.evaluate(
                self,
                features,
                edge_attrs,
                radial,
                projection,
                edge_index,
                num_nodes,
                vectors,
                amplitudes,
                radial_network,
                pair_index,
            )
        return torch_kernel.evaluate(
            self,
            features,
            edge_attrs,
            radial,
            projection,
            edge_index,
            num_nodes,
            radial_network,
        )

    def reference(
        self, features, edge_attrs, radial, projection, edge_index, num_nodes
    ):
        """Evaluate the same paths with ordinary tensor operations."""
        return torch_kernel.reference(
            self, features, edge_attrs, radial, projection, edge_index, num_nodes
        )

    def couple(self, x, y, l1, l2, l3):
        """Contract angular indices while retaining independent channels."""
        cg = getattr(self, f"reference_cg_{l1}_{l2}_{l3}").to(x)
        return torch.einsum("...ab,abc->...c", x.unsqueeze(-1) * y.unsqueeze(-2), cg)

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
